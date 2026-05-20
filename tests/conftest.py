"""
Shared pytest fixtures for RAG Bench tests.

Fixtures follow a layered approach:
- Mock components for unit tests (fast, isolated)
- Real components with test databases for integration tests
- Full app client for E2E tests
"""
import pytest
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from langchain.schema import Document as LangchainDocument

from rag_bench.settings.settings import (
    Settings,
    ServerSettings,
    LLMSettings,
    EmbeddingSettings,
    VectorStoreSettings,
    RagSettings,
    PostgresSettings,
)
from rag_bench.core.types import DocumentWithScore


# =============================================================================
# Settings Fixtures
# =============================================================================

@pytest.fixture
def mock_settings() -> Settings:
    """Minimal settings configured for mock components."""
    return Settings(
        server=ServerSettings(env_name="test", port=8000),
        llm=LLMSettings(mode="mock"),
        embedding=EmbeddingSettings(mode="mock"),
        vectorstore=VectorStoreSettings(mode="mock"),
        rag=RagSettings(similarity_top_k=3, similarity_threshold=0.7),
        postgres=PostgresSettings(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            database="test_rag_bench",
        ),
    )


# =============================================================================
# Mock Component Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_component():
    """Mock LLM that returns predictable responses."""
    mock = AsyncMock()
    mock.agenerate = AsyncMock(return_value="This is a test response based on the documents.")
    return mock


@pytest.fixture
def mock_embedding_component():
    """Mock embedding component with deterministic vectors."""
    mock = MagicMock()

    def mock_embed_query(text: str) -> List[float]:
        # Return a deterministic vector based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val >> i) % 100 / 100.0 for i in range(384)]

    def mock_embed_documents(texts: List[str]) -> List[List[float]]:
        return [mock_embed_query(t) for t in texts]

    mock.embed_query = mock_embed_query
    mock.embed_documents = mock_embed_documents
    return mock


@pytest.fixture
def mock_vector_store_component():
    """Mock vector store with in-memory document storage."""
    mock = AsyncMock()
    mock._documents = []

    async def mock_search(query: str, k: int = 4) -> List[tuple]:
        # Return stored documents with fake scores
        results = []
        for i, doc in enumerate(mock._documents[:k]):
            score = 0.9 - (i * 0.1)  # Decreasing scores
            results.append((doc, score))
        return results

    async def mock_add(documents: List[LangchainDocument]) -> None:
        mock._documents.extend(documents)

    mock.asimilarity_search_with_scores = mock_search
    mock.aadd_documents = mock_add
    return mock


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_documents() -> List[LangchainDocument]:
    """Sample documents for testing retrieval."""
    return [
        LangchainDocument(
            page_content="RAG combines retrieval with generation for better answers.",
            metadata={"source": "rag_intro.txt", "id": "doc1"}
        ),
        LangchainDocument(
            page_content="Vector embeddings capture semantic meaning of text.",
            metadata={"source": "embeddings.txt", "id": "doc2"}
        ),
        LangchainDocument(
            page_content="LLMs generate text based on patterns learned during training.",
            metadata={"source": "llm_basics.txt", "id": "doc3"}
        ),
    ]


@pytest.fixture
def sample_documents_with_scores(sample_documents) -> List[DocumentWithScore]:
    """Sample documents with relevance scores."""
    return [
        DocumentWithScore(document=sample_documents[0], score=0.95),
        DocumentWithScore(document=sample_documents[1], score=0.82),
        DocumentWithScore(document=sample_documents[2], score=0.71),
    ]


# =============================================================================
# RAG Engine Fixture
# =============================================================================

@pytest.fixture
def rag_engine(mock_settings, mock_llm_component, mock_vector_store_component):
    """RAG engine configured with mock components."""
    from rag_bench.core.engine import RAGEngine

    return RAGEngine(
        settings=mock_settings,
        llm_component=mock_llm_component,
        vector_store_component=mock_vector_store_component,
        query_enhancers=[],
        document_post_processors=[],
    )


# =============================================================================
# FastAPI Test Client Fixture
# =============================================================================

@pytest.fixture
def test_client(mock_settings):
    """FastAPI test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    # Patch settings before importing app
    with patch('rag_bench.settings.settings.settings', return_value=mock_settings):
        with patch('rag_bench.settings.settings.unsafe_typed_settings', mock_settings):
            # Import must happen inside patch context
            from rag_bench.main import app
            yield TestClient(app)
