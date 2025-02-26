from typing import List, Optional, Dict, Any, Generator
from uuid import uuid4

from injector import inject, singleton
from pydantic import BaseModel

from rag_bench.core.engine import RAGEngine, SourcedAnswer
from rag_bench.core.types import Message, Conversation
from rag_bench.routers.api_v1.chat.types import ChatMetadata, ChatResponse, ChatDocument


class StreamingChunk(BaseModel):
    content: str
    done: bool = False


@singleton
class ChatService:
    """Service for handling chat interactions with the RAG engine."""
    
    @inject
    def __init__(
        self,
        rag_engine: RAGEngine,
    ) -> None:
        self.rag_engine = rag_engine
        self._conversations: Dict[str, Conversation] = {}
    
    def _get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get an existing conversation or create a new one."""
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        
        # Create new conversation
        conversation = Conversation()
        self._conversations[conversation.id] = conversation
        
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