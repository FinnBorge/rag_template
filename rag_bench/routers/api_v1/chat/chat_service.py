from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
from uuid import uuid4
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

from injector import inject, singleton
from pydantic import BaseModel

from rag_bench.core.engine import RAGEngine, SourcedAnswer
from rag_bench.core.types import Message, Conversation
from rag_bench.routers.api_v1.chat.types import ChatMetadata, ChatResponse, ChatDocument


logger = logging.getLogger(__name__)


class StreamingChunk(BaseModel):
    content: str
    done: bool = False


class ConversationEntry:
    """Wrapper for conversation with TTL tracking."""

    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        self.last_accessed = datetime.utcnow()

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.utcnow()

    def is_expired(self, ttl: timedelta) -> bool:
        """Check if this entry has expired."""
        return datetime.utcnow() - self.last_accessed > ttl


class ConversationStore:
    """
    Bounded conversation store with TTL and max size limits.

    Prevents unbounded memory growth by:
    - Limiting max number of conversations
    - Expiring conversations after TTL
    - Using LRU eviction when at capacity
    """

    DEFAULT_MAX_SIZE = 1000
    DEFAULT_TTL = timedelta(hours=1)

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, ConversationEntry] = OrderedDict()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID, returns None if not found or expired."""
        entry = self._store.get(conversation_id)
        if entry is None:
            return None

        if entry.is_expired(self._ttl):
            del self._store[conversation_id]
            logger.debug(f"Conversation {conversation_id} expired")
            return None

        # Move to end (most recently used) and update access time
        self._store.move_to_end(conversation_id)
        entry.touch()
        return entry.conversation

    def set(self, conversation: Conversation) -> None:
        """Store a conversation, evicting oldest if at capacity."""
        # Clean expired entries periodically
        self._cleanup_expired()

        # Evict oldest if at capacity
        while len(self._store) >= self._max_size:
            oldest_id, _ = self._store.popitem(last=False)
            logger.debug(f"Evicted conversation {oldest_id} (LRU)")

        self._store[conversation.id] = ConversationEntry(conversation)

    def _cleanup_expired(self) -> None:
        """Remove expired entries (called periodically)."""
        expired_ids = [
            cid for cid, entry in self._store.items()
            if entry.is_expired(self._ttl)
        ]
        for cid in expired_ids:
            del self._store[cid]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired conversations")

    def __len__(self) -> int:
        return len(self._store)


@singleton
class ChatService:
    """Service for handling chat interactions with the RAG engine."""

    @inject
    def __init__(
        self,
        rag_engine: RAGEngine,
    ) -> None:
        self.rag_engine = rag_engine
        self._conversations = ConversationStore()

    def _get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get an existing conversation or create a new one."""
        if conversation_id:
            existing = self._conversations.get(conversation_id)
            if existing is not None:
                return existing

        # Create new conversation
        conversation = Conversation()
        self._conversations.set(conversation)

        return conversation
    
    async def generate_response(
        self,
        query: str,
        metadata: ChatMetadata,
        include_document_text: bool = False,
    ) -> ChatResponse:
        """Generate a response for a given query."""
        # Get or create conversation
        conversation_id = metadata.conversation_id
        conversation = self._get_or_create_conversation(conversation_id)
        
        # Add user message to conversation
        conversation.add_message("user", query)
        
        # Generate answer
        result = await self.rag_engine.generate_answer(query, conversation_id)
        
        # Add assistant message to conversation
        conversation.add_message("assistant", result.answer)
        
        # Create response
        return self._create_response(query, result, metadata, include_document_text)
    
    async def stream_response(
        self,
        query: str,
        metadata: ChatMetadata,
        include_document_text: bool = False,
    ) -> Generator[str, None, None]:
        """Stream a response for a given query."""
        # Generate the full response first (in a real implementation this would be streaming)
        response = await self.generate_response(query, metadata, include_document_text)
        
        # Split the answer into chunks for demonstration
        chunks = [response.answer[i:i+20] for i in range(0, len(response.answer), 20)]
        
        # Return the streaming response
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            
            # Create a partial response with the current chunk
            partial_response = ChatResponse(
                id=response.id,
                answer=chunk if i == 0 else response.answer[:i*20+len(chunk)],
                question=query,
                metadata=metadata,
                documents=response.documents if is_last else [],
                status="completed" if is_last else "pending",
                model=response.model
            )
            
            yield partial_response.model_dump_json() + "\n\n"
    
    def _create_response(
        self,
        query: str,
        result: SourcedAnswer,
        metadata: ChatMetadata,
        include_document_text: bool = False,
    ) -> ChatResponse:
        """Create a chat response from the RAG engine result."""
        documents = []
        
        # Create document objects for response
        for source in result.sources:
            doc = ChatDocument(
                title=source.get("title", ""),
                url=source.get("url", ""),
                snippet=source.get("content", "") if include_document_text else None,
                source=source.get("source", ""),
                source_id=str(source.get("id", "")),
                publish_date=source.get("date", ""),
                relevance_score=source.get("relevance_score")
            )
            documents.append(doc)
        
        # Create and return the response
        return ChatResponse(
            id=str(uuid4()),
            answer=result.answer,
            question=query,
            metadata=metadata,
            documents=documents,
            status="completed"
        )