from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatMetadata(BaseModel):
    user_id: Optional[int] = Field(default=None)
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))


class ChatDocument(BaseModel):
    """A document with relevance information."""
    
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    source: str
    source_id: Optional[str] = None
    publish_date: Optional[str] = None
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response from RAG Bench."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    question: str
    metadata: ChatMetadata
    model: str = "rag-bench-model"
    documents: list[ChatDocument] = []
    status: Literal["pending", "completed"] = "completed"
    
    @staticmethod
    def from_document_with_score(doc_with_score, include_document_text: bool = False) -> "ChatDocument":
        doc = doc_with_score.document
        metadata = doc.metadata
        
        return ChatDocument(
            title=metadata.get("title", ""),
            url=metadata.get("url", ""),
            snippet=doc.page_content if include_document_text else None,
            source=metadata.get("source", ""),
            source_id=str(metadata.get("id", "")),
            publish_date=metadata.get("date"),
            relevance_score=doc_with_score.score
        )