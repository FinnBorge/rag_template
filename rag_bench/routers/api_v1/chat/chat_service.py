from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
import logging
import threading

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
        self.last_accessed = datetime.now(timezone.utc)

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now(timezone.utc)

    def is_expired(self, ttl: timedelta) -> bool:
        """Check if this entry has expired."""
        return datetime.now(timezone.utc) - self.last_accessed > ttl


class ConversationStore:
    """
    Thread-safe bounded conversation store with TTL and max size limits.

    Prevents unbounded memory growth by:
    - Limiting max number of conversations
    - Expiring conversations after TTL
    - Using LRU eviction when at capacity

    Thread-safety is ensured via a reentrant lock.
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
        self._lock = threading.RLock()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID, returns None if not found or expired."""
        with self._lock:
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
        with self._lock:
            # Clean expired entries periodically
            self._cleanup_expired()

            # Evict oldest if at capacity
            while len(self._store) >= self._max_size:
                oldest_id, _ = self._store.popitem(last=False)
                logger.debug(f"Evicted conversation {oldest_id} (LRU)")

            self._store[conversation.id] = ConversationEntry(conversation)

    def _cleanup_expired(self) -> None:
        """Remove expired entries. Must be called with lock held."""
        expired_ids = [
            cid for cid, entry in self._store.items()
            if entry.is_expired(self._ttl)
        ]
        for cid in expired_ids:
            del self._store[cid]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired conversations")

    def __len__(self) -> int:
        with self._lock:
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
        
        # Add user message to conversation (immutable update)
        conversation, _ = conversation.with_message("user", query)

        # Generate answer
        result = await self.rag_engine.generate_answer(query, conversation_id)

        # Add assistant message to conversation (immutable update)
        conversation, _ = conversation.with_message("assistant", result.answer)

        # Store updated conversation
        self._conversations.set(conversation)
        
        # Create response
        return self._create_response(query, result, metadata, include_document_text)
    
    async def stream_response(
        self,
        query: str,
        metadata: ChatMetadata,
        include_document_text: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for a given query.

        NOTE: This is simulated streaming - the full response is generated first,
        then chunked for delivery. True token-by-token streaming requires LLM
        provider support (e.g., OpenAI streaming API).

        For benchmarking purposes, use generate_response() instead to get
        accurate timing metrics.

        Args:
            query: The user's question
            metadata: Request metadata
            include_document_text: Whether to include full document text

        Yields:
            JSON-encoded ChatResponse objects with incremental answer text
        """
        # Generate complete response first (simulated streaming)
        response = await self.generate_response(query, metadata, include_document_text)

        # Yield chunks to simulate streaming
        chunk_size = 20
        answer = response.answer
        total_chunks = (len(answer) + chunk_size - 1) // chunk_size

        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(answer))
            is_last = (i == total_chunks - 1)

            partial_response = ChatResponse(
                id=response.id,
                answer=answer[:end],  # Accumulate answer progressively
                question=query,
                metadata=metadata,
                documents=response.documents if is_last else [],
                status="completed" if is_last else "streaming",
                model=response.model,
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