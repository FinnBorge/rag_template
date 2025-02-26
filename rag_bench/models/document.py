import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, Index, JSON, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from pgvector.sqlalchemy import Vector

from rag_bench.db.base import Base


class Document(Base):
    """Document model for storing source documents."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=False, default=dict)
    embedding = Column(Vector(1536), nullable=True)
    source = Column(String, nullable=False)
    url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Index on the embedding vector for similarity search
    __table_args__ = (
        Index("ix_documents_embedding_cosine_ops", embedding, postgresql_using="ivfflat", postgresql_with={"lists": 100}),
    )
    
    def __repr__(self):
        return f"<Document id={self.id} source={self.source}>"