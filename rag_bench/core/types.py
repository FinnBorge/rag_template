from typing import List, Protocol, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

from langchain.schema import Document as LangchainDocument
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Simple document representation."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryEnhancer(ABC):
    """Base class for query enhancing components."""
    
    @abstractmethod
    async def enhance(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Enhance a query with additional context or modifications."""
        pass


class DocumentWithScore:
    """Document with a relevance score."""
    def __init__(self, document: LangchainDocument, score: float):
        self.document = document
        self.score = float(score)  # Ensure score is a float


class DocumentPostProcessor(ABC):
    """Base class for document post-processing components."""
    
    @abstractmethod
    async def process(
        self, 
        documents: List[DocumentWithScore], 
        query: str
    ) -> List[DocumentWithScore]:
        """Process a list of documents and return a filtered or modified list."""
        pass


class EmbeddingComponent(Protocol):
    """Protocol defining the embedding component interface."""
    
    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a query string."""
        pass
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        pass


class LLMComponent(Protocol):
    """Protocol defining the LLM component interface."""
    
    async def agenerate(self, template: str, **kwargs) -> str:
        """Generate text using the LLM."""
        pass


class VectorStoreComponent(Protocol):
    """Protocol defining the vector store component interface."""
    
    async def asimilarity_search_with_scores(
        self, 
        query: str, 
        k: int = 4
    ) -> List[tuple[LangchainDocument, float]]:
        """Search for documents similar to the query."""
        pass
    
    async def aadd_documents(self, documents: List[LangchainDocument]) -> None:
        """Add documents to the vector store."""
        pass


class Message(BaseModel):
    """A message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Conversation(BaseModel):
    """A conversation with message history."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get the conversation history, optionally limited to the last N messages."""
        if limit is not None:
            return self.messages[-limit:]
        return self.messages
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }