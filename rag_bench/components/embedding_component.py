import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np

from langchain_openai import OpenAIEmbeddings
from injector import inject

from rag_bench.settings.settings import Settings
from rag_bench.core.types import EmbeddingComponent


logger = logging.getLogger(__name__)


class OpenAIEmbeddingComponent(EmbeddingComponent):
    """Component for generating embeddings using OpenAI APIs."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_settings = settings.openai
        
        if not self.openai_settings:
            raise ValueError("OpenAI settings are required for OpenAIEmbeddingComponent")
            
        # Initialize embedding model
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.openai_settings.embedding_model,
                openai_api_key=self.openai_settings.api_key,
                openai_api_base=self.openai_settings.api_base
            )
            logger.info(f"Initialized OpenAI embeddings with model {self.openai_settings.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a query string."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            return self.embeddings.embed_documents(documents)
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in range(len(documents))]
            
    # For backward compatibility with existing code
    def get_embedding(self, text: str) -> List[float]:
        """Alias for embed_query."""
        return self.embed_query(text)
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents."""
        return self.embed_documents(texts)


class MockEmbeddingComponent(EmbeddingComponent):
    """Mock component for testing without API calls."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_dim = 1536
    
    def embed_query(self, text: str) -> List[float]:
        """Generate a random embedding for a query string."""
        # Use hash of text as seed for reproducibility
        np.random.seed(hash(text) % 2**32)
        return list(np.random.normal(0, 1, self.embedding_dim))
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate random embeddings for a list of documents."""
        return [self.embed_query(doc) for doc in documents]
        
    # For backward compatibility with existing code
    def get_embedding(self, text: str) -> List[float]:
        """Alias for embed_query."""
        return self.embed_query(text)
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents."""
        return self.embed_documents(texts)


class HuggingFaceEmbeddingComponent(EmbeddingComponent):
    """Component for generating embeddings using HuggingFace models."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.hf_settings = settings.huggingface
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.hf_settings.embedding_model
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized HuggingFace embeddings with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a query string."""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            # Return a zero vector as fallback
            return [0.0] * 384  # Default dimension for many sentence-transformers models
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            embeddings = self.model.encode(documents)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 384 for _ in range(len(documents))]
    
    # For backward compatibility with existing code
    def get_embedding(self, text: str) -> List[float]:
        """Alias for embed_query."""
        return self.embed_query(text)
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents."""
        return self.embed_documents(texts)


def get_embedding_component(settings: Settings) -> Union[OpenAIEmbeddingComponent, HuggingFaceEmbeddingComponent, MockEmbeddingComponent]:
    """Factory function to create the appropriate embedding component based on settings."""
    embedding_mode = settings.embedding.mode
    
    if embedding_mode == "openai":
        return OpenAIEmbeddingComponent(settings)
    elif embedding_mode == "huggingface":
        return HuggingFaceEmbeddingComponent(settings)
    elif embedding_mode == "mock":
        return MockEmbeddingComponent(settings)
    else:
        raise ValueError(f"Unknown embedding mode: {embedding_mode}")