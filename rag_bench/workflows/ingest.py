"""
Document ingestion workflow for the RAG benchmarking system.
"""
import logging
from typing import List, Optional, Dict, Any

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from rag_bench.components.embedding_component import EmbeddingComponent
from rag_bench.components.vector_store_component import VectorStoreComponent
from rag_bench.core.types import Document
from rag_bench.models.document import DocumentModel

logger = logging.getLogger(__name__)

class IngestConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    add_source_metadata: bool = True
    embedding_batch_size: int = 32


class DocumentIngester:
    """Handles document ingestion into the RAG system."""
    
    def __init__(
        self,
        embedding_component: EmbeddingComponent,
        vector_store_component: VectorStoreComponent,
        config: Optional[IngestConfig] = None
    ):
        """Initialize the document ingester with components and configuration."""
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.config = config or IngestConfig()
        
        # Download nltk data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a text splitter based on the configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    
    def _process_chunks(
        self, 
        chunks: List[str], 
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Process text chunks into Documents."""
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            doc = Document(
                text=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
            
        return documents
    
    def ingest_text(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Ingest a text document into the system.
        
        Args:
            text: The document text
            metadata: Document metadata
            
        Returns:
            List of ingested documents
        """
        # Split text into chunks
        splitter = self._create_text_splitter()
        chunks = splitter.split_text(text)
        
        # Create document objects from chunks
        documents = self._process_chunks(chunks, metadata)
        
        # Generate embeddings and store in vector store
        embeddings = self.embedding_component.get_embeddings_batch(
            [doc.text for doc in documents]
        )
        
        # Store documents with embeddings
        for doc, embedding in zip(documents, embeddings):
            document_model = DocumentModel(
                text=doc.text,
                metadata=doc.metadata,
                embedding=embedding
            )
            self.vector_store_component.add_document(document_model)
            
        logger.info(f"Ingested {len(documents)} document chunks")
        return documents
    
    def ingest_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Ingest multiple documents into the system.
        
        Args:
            texts: List of document texts
            metadatas: List of document metadata
            
        Returns:
            List of ingested documents
        """
        all_documents = []
        
        for text, metadata in zip(texts, metadatas):
            documents = self.ingest_text(text, metadata)
            all_documents.extend(documents)
            
        return all_documents


def ingest_document_from_file(
    ingester: DocumentIngester,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Ingest a document from a file.
    
    Args:
        ingester: DocumentIngester instance
        file_path: Path to the document file
        metadata: Optional metadata to add
        
    Returns:
        List of ingested documents
    """
    if metadata is None:
        metadata = {}
    
    # Add file path to metadata
    metadata["source"] = file_path
    
    # Read file content
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return ingester.ingest_text(text, metadata)