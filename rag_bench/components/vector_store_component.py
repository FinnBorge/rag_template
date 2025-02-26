import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores.pgvector import PGVector
from injector import inject
import sqlalchemy
import pgvector

from rag_bench.settings.settings import Settings
from rag_bench.core.types import VectorStoreComponent


logger = logging.getLogger(__name__)


class PGVectorStoreComponent(VectorStoreComponent):
    """Component for interacting with a PostgreSQL pgvector store."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pgvector_settings = settings.pgvector or settings.postgres
        
        connection_string = self.pgvector_settings.to_uri()
        self.collection_name = "documents"
        
        # Initialize connection
        try:
            self.engine = sqlalchemy.create_engine(connection_string)
            
            # Verify pgvector extension
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
                if result.fetchone() is None:
                    logger.warning("Vector extension not found in PostgreSQL, attempting to create...")
                    conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                
            logger.info("Successfully connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise
        
        # Initialize PGVector with a dummy embedding function that will be replaced when needed
        class DummyEmbedding:
            def embed_documents(self, texts):
                return [[0.0] * 1536 for _ in texts]
            
            def embed_query(self, text):
                return [0.0] * 1536
                
        # Initialize PGVector
        self.vector_store = PGVector(
            connection_string=connection_string,
            collection_name=self.collection_name,
            embedding_function=DummyEmbedding()  # Dummy function to initialize
        )
    
    async def asimilarity_search_with_scores(
        self, 
        query: str,
        k: int = 4,
        embedding_function = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query."""
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
            
        # Create a LangChain compatible embedding function
        class LangChainEmbeddingAdapter:
            def __init__(self, embedding_component):
                self.embedding_component = embedding_component
                
            def embed_documents(self, texts):
                return self.embedding_component.embed_documents(texts)
                
            def embed_query(self, text):
                return self.embedding_component.embed_query(text)
        
        # Create the adapter
        adapter = LangChainEmbeddingAdapter(embedding_function)
            
        # Get embeddings for the query
        try:
            # Update vector store's embedding function
            self.vector_store.embedding_function = adapter
            
            # Call the similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    async def aadd_documents(
        self, 
        documents: List[LangchainDocument],
        embedding_function = None
    ) -> None:
        """Add documents to the vector store."""
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
            
        try:
            # Create a LangChain compatible embedding function
            class LangChainEmbeddingAdapter:
                def __init__(self, embedding_component):
                    self.embedding_component = embedding_component
                    
                def embed_documents(self, texts):
                    return self.embedding_component.embed_documents(texts)
                    
                def embed_query(self, text):
                    return self.embedding_component.embed_query(text)
            
            # Create the adapter
            adapter = LangChainEmbeddingAdapter(embedding_function)
            
            # Update vector store's embedding function
            self.vector_store.embedding_function = adapter
            
            # Add documents
            self.vector_store.add_documents(
                documents=documents
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise


class QdrantVectorStoreComponent(VectorStoreComponent):
    """Component for interacting with a Qdrant vector store."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.qdrant_settings = settings.qdrant
        
        if not self.qdrant_settings:
            raise ValueError("Qdrant settings are required for QdrantVectorStoreComponent")
        
        # Import here to avoid forcing dependency if not used
        from langchain_community.vectorstores import Qdrant
        
        # Initialize Qdrant client
        try:
            if self.qdrant_settings.url:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(
                    url=self.qdrant_settings.url,
                    api_key=self.qdrant_settings.api_key,
                    prefer_grpc=self.qdrant_settings.prefer_grpc
                )
            else:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(
                    host=self.qdrant_settings.host,
                    port=self.qdrant_settings.port,
                    prefer_grpc=self.qdrant_settings.prefer_grpc
                )
                
            self.collection_name = self.qdrant_settings.collection_name
            logger.info(f"Successfully connected to Qdrant at collection={self.collection_name}")
            
            # Initialize vector store
            self.vector_store = None  # Will be initialized when needed
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def _get_vector_store(self, embedding_function):
        """Get or initialize the vector store with an embedding function."""
        from langchain_community.vectorstores import Qdrant
        
        if self.vector_store is None:
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=embedding_function
            )
        return self.vector_store
    
    async def asimilarity_search_with_scores(
        self, 
        query: str,
        k: int = 4,
        embedding_function = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query."""
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
            
        vector_store = self._get_vector_store(embedding_function)
        
        try:
            docs_with_scores = vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    async def aadd_documents(
        self, 
        documents: List[LangchainDocument],
        embedding_function = None
    ) -> None:
        """Add documents to the vector store."""
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
            
        vector_store = self._get_vector_store(embedding_function)
        
        try:
            vector_store.add_documents(documents=documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise


class MockVectorStoreComponent(VectorStoreComponent):
    """Mock component for in-memory vector storage for testing."""
    
    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.documents = []  # In-memory storage for documents
        self.embedding_dim = 1536  # Default embedding dimension
        logger.info("Initialized MockVectorStoreComponent")
        
    def _compute_similarity(self, query_embedding, doc_embedding):
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(query_embedding, doc_embedding)
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(doc_embedding)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
            
        return dot_product / (query_norm * doc_norm)
    
    async def asimilarity_search_with_scores(
        self, 
        query: str,
        k: int = 4,
        embedding_function = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query in memory."""
        if not self.documents:
            # Return empty list if no documents have been added
            logger.warning("No documents in mock vector store")
            return []
            
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
        
        # Get embedding for the query
        query_embedding = embedding_function.embed_query(query)
        
        # Calculate similarities with each document
        similarities = []
        for doc, doc_embedding in self.documents:
            similarity = self._compute_similarity(query_embedding, doc_embedding)
            similarities.append((doc, similarity))
        
        # Sort by similarity (highest first) and limit to k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def aadd_documents(
        self, 
        documents: List[LangchainDocument],
        embedding_function = None
    ) -> None:
        """Add documents to the in-memory store."""
        if not documents:
            return
            
        if embedding_function is None:
            # Import here to avoid circular imports
            from rag_bench.components.embedding_component import get_embedding_component
            embedding_function = get_embedding_component(self.settings)
        
        # Get embeddings for documents
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_function.embed_documents(texts)
        
        # Store documents with their embeddings
        for doc, embedding in zip(documents, embeddings):
            self.documents.append((doc, embedding))
            
        logger.info(f"Added {len(documents)} documents to mock vector store (total: {len(self.documents)})")