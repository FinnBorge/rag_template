"""
Vector store components for document storage and retrieval.
"""
import asyncio
import logging
from typing import List, Tuple, Optional

import numpy as np
from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores.pgvector import PGVector
from injector import inject
import sqlalchemy

from rag_bench.settings.settings import Settings
from rag_bench.core.types import VectorStoreComponent


logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter:
    """Adapter to make our embedding component compatible with LangChain."""

    def __init__(self, embedding_component):
        self.embedding_component = embedding_component

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_component.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_component.embed_query(text)


class DummyEmbedding:
    """Placeholder embedding for initialization."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self.dim


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
                result = conn.execute(
                    sqlalchemy.text(
                        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                    )
                )
                if result.fetchone() is None:
                    logger.warning(
                        "Vector extension not found in PostgreSQL, attempting to create..."
                    )
                    conn.execute(
                        sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                    )
                    conn.commit()

            logger.info("Successfully connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

        # Initialize PGVector with dummy embedding (will be replaced per-query)
        self.vector_store = PGVector(
            connection_string=connection_string,
            collection_name=self.collection_name,
            embedding_function=DummyEmbedding(),
        )

    def _get_embedding_component(self):
        """Get the embedding component, avoiding circular imports."""
        from rag_bench.components.embedding_component import get_embedding_component
        return get_embedding_component(self.settings)

    async def asimilarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        embedding_function=None,
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query."""
        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        adapter = LangChainEmbeddingAdapter(embedding_function)

        try:
            # Update vector store's embedding function
            self.vector_store.embedding_function = adapter

            # Run synchronous DB call in thread pool to avoid blocking
            docs_with_scores = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query=query,
                k=k,
            )
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    async def aadd_documents(
        self,
        documents: List[LangchainDocument],
        embedding_function=None,
    ) -> None:
        """Add documents to the vector store."""
        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        adapter = LangChainEmbeddingAdapter(embedding_function)

        try:
            self.vector_store.embedding_function = adapter

            # Run synchronous DB call in thread pool to avoid blocking
            await asyncio.to_thread(
                self.vector_store.add_documents,
                documents=documents,
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
        from qdrant_client import QdrantClient

        # Initialize Qdrant client
        try:
            if self.qdrant_settings.url:
                self.client = QdrantClient(
                    url=self.qdrant_settings.url,
                    api_key=self.qdrant_settings.api_key,
                    prefer_grpc=self.qdrant_settings.prefer_grpc,
                )
            else:
                self.client = QdrantClient(
                    host=self.qdrant_settings.host,
                    port=self.qdrant_settings.port,
                    prefer_grpc=self.qdrant_settings.prefer_grpc,
                )

            self.collection_name = self.qdrant_settings.collection_name
            logger.info(
                f"Successfully connected to Qdrant at collection={self.collection_name}"
            )

            self.vector_store = None  # Lazy initialization

        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise

    def _get_embedding_component(self):
        """Get the embedding component, avoiding circular imports."""
        from rag_bench.components.embedding_component import get_embedding_component
        return get_embedding_component(self.settings)

    def _get_vector_store(self, embedding_function):
        """Get or initialize the vector store with an embedding function."""
        from langchain_community.vectorstores import Qdrant

        if self.vector_store is None:
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=embedding_function,
            )
        return self.vector_store

    async def asimilarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        embedding_function=None,
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query."""
        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        adapter = LangChainEmbeddingAdapter(embedding_function)
        vector_store = self._get_vector_store(adapter)

        try:
            # Run synchronous call in thread pool to avoid blocking
            docs_with_scores = await asyncio.to_thread(
                vector_store.similarity_search_with_score,
                query=query,
                k=k,
            )
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    async def aadd_documents(
        self,
        documents: List[LangchainDocument],
        embedding_function=None,
    ) -> None:
        """Add documents to the vector store."""
        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        adapter = LangChainEmbeddingAdapter(embedding_function)
        vector_store = self._get_vector_store(adapter)

        try:
            # Run synchronous call in thread pool to avoid blocking
            await asyncio.to_thread(
                vector_store.add_documents,
                documents=documents,
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise


class MockVectorStoreComponent(VectorStoreComponent):
    """Mock component for in-memory vector storage for testing."""

    @inject
    def __init__(self, settings: Settings):
        self.settings = settings
        self.documents: List[Tuple[LangchainDocument, List[float]]] = []
        self.embedding_dim = 1536
        logger.info("Initialized MockVectorStoreComponent")

    def _compute_similarity(
        self,
        query_embedding: List[float],
        doc_embedding: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        query_arr = np.array(query_embedding)
        doc_arr = np.array(doc_embedding)

        dot_product = np.dot(query_arr, doc_arr)
        query_norm = np.linalg.norm(query_arr)
        doc_norm = np.linalg.norm(doc_arr)

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return float(dot_product / (query_norm * doc_norm))

    def _get_embedding_component(self):
        """Get the embedding component, avoiding circular imports."""
        from rag_bench.components.embedding_component import get_embedding_component
        return get_embedding_component(self.settings)

    async def asimilarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        embedding_function=None,
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for documents similar to the query in memory."""
        if not self.documents:
            logger.warning("No documents in mock vector store")
            return []

        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        # Get embedding for the query
        query_embedding = embedding_function.embed_query(query)

        # Calculate similarities with each document
        similarities = [
            (doc, self._compute_similarity(query_embedding, doc_embedding))
            for doc, doc_embedding in self.documents
        ]

        # Sort by similarity (highest first) and limit to k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    async def aadd_documents(
        self,
        documents: List[LangchainDocument],
        embedding_function=None,
    ) -> None:
        """Add documents to the in-memory store."""
        if not documents:
            return

        if embedding_function is None:
            embedding_function = self._get_embedding_component()

        # Get embeddings for documents
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_function.embed_documents(texts)

        # Store documents with their embeddings
        for doc, embedding in zip(documents, embeddings):
            self.documents.append((doc, embedding))

        logger.info(
            f"Added {len(documents)} documents to mock vector store "
            f"(total: {len(self.documents)})"
        )
