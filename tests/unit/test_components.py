"""
Unit tests for LLM, Embedding, and VectorStore components.

These tests verify the component interfaces and mock implementations.
Real provider tests (OpenAI, etc.) are integration tests that require credentials.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from rag_bench.settings.settings import Settings, LLMSettings, EmbeddingSettings
from rag_bench.components.llm_component import MockLLMComponent
from rag_bench.components.embedding_component import (
    MockEmbeddingComponent,
    HuggingFaceEmbeddingComponent,
    get_embedding_component,
)
from rag_bench.components.vector_store_component import MockVectorStoreComponent
from langchain.schema import Document as LangchainDocument


# =============================================================================
# MockLLMComponent Tests
# =============================================================================

class TestMockLLMComponent:
    """Tests for MockLLMComponent."""

    @pytest.fixture
    def component(self, mock_settings):
        return MockLLMComponent(mock_settings)

    @pytest.mark.asyncio
    async def test_returns_mock_response(self, component):
        # Arrange
        template = "Answer this: {question}"

        # Act
        result = await component.agenerate(template, question="What is 2+2?")

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_handles_any_template_variables(self, component):
        # Arrange
        template = "Context: {context}\nQuery: {query}\nAnswer:"

        # Act
        result = await component.agenerate(
            template,
            context="Some context",
            query="Some query"
        )

        # Assert
        assert isinstance(result, str)


# =============================================================================
# MockEmbeddingComponent Tests
# =============================================================================

class TestMockEmbeddingComponent:
    """Tests for MockEmbeddingComponent."""

    @pytest.fixture
    def component(self, mock_settings):
        return MockEmbeddingComponent(mock_settings)

    def test_embed_query_returns_correct_dimension(self, component):
        # Arrange
        text = "test query"

        # Act
        embedding = component.embed_query(text)

        # Assert
        assert len(embedding) == 1536  # Default dimension

    def test_embed_query_is_deterministic(self, component):
        # Arrange
        text = "same text"

        # Act
        embedding1 = component.embed_query(text)
        embedding2 = component.embed_query(text)

        # Assert - same input should give same output
        assert embedding1 == embedding2

    def test_embed_query_differs_for_different_text(self, component):
        # Arrange & Act
        embedding1 = component.embed_query("text one")
        embedding2 = component.embed_query("text two")

        # Assert
        assert embedding1 != embedding2

    def test_embed_documents_returns_list(self, component):
        # Arrange
        texts = ["doc 1", "doc 2", "doc 3"]

        # Act
        embeddings = component.embed_documents(texts)

        # Assert
        assert len(embeddings) == 3
        assert all(len(e) == 1536 for e in embeddings)

    def test_embed_documents_handles_empty_list(self, component):
        # Act
        embeddings = component.embed_documents([])

        # Assert
        assert embeddings == []

    def test_get_embedding_alias(self, component):
        # Arrange
        text = "test"

        # Act
        result1 = component.embed_query(text)
        result2 = component.get_embedding(text)

        # Assert - alias should return same result
        assert result1 == result2


# =============================================================================
# MockVectorStoreComponent Tests
# =============================================================================

class TestMockVectorStoreComponent:
    """Tests for MockVectorStoreComponent."""

    @pytest.fixture
    def component(self, mock_settings):
        return MockVectorStoreComponent(mock_settings)

    @pytest.fixture
    def sample_docs(self):
        return [
            LangchainDocument(page_content="First document about RAG", metadata={"id": "1"}),
            LangchainDocument(page_content="Second document about embeddings", metadata={"id": "2"}),
        ]

    @pytest.mark.asyncio
    async def test_add_and_search_documents(self, component, sample_docs, mock_embedding_component):
        # Arrange
        await component.aadd_documents(sample_docs, mock_embedding_component)

        # Act
        results = await component.asimilarity_search_with_scores(
            "RAG query",
            k=2,
            embedding_function=mock_embedding_component
        )

        # Assert
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], LangchainDocument) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_store(self, component, mock_embedding_component):
        # Act
        results = await component.asimilarity_search_with_scores(
            "query",
            k=5,
            embedding_function=mock_embedding_component
        )

        # Assert
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_k_limit(self, component, mock_embedding_component):
        # Arrange
        docs = [
            LangchainDocument(page_content=f"Document {i}", metadata={"id": str(i)})
            for i in range(10)
        ]
        await component.aadd_documents(docs, mock_embedding_component)

        # Act
        results = await component.asimilarity_search_with_scores(
            "query",
            k=3,
            embedding_function=mock_embedding_component
        )

        # Assert
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_similarity_scores_are_valid(self, component, sample_docs, mock_embedding_component):
        # Arrange
        await component.aadd_documents(sample_docs, mock_embedding_component)

        # Act
        results = await component.asimilarity_search_with_scores(
            "test query",
            k=2,
            embedding_function=mock_embedding_component
        )

        # Assert - cosine similarity should be between -1 and 1
        for doc, score in results:
            assert -1.0 <= score <= 1.0


# =============================================================================
# Component Factory Tests
# =============================================================================

class TestGetEmbeddingComponent:
    """Tests for the embedding component factory function."""

    def test_returns_mock_for_mock_mode(self, mock_settings):
        # Arrange
        mock_settings.embedding.mode = "mock"

        # Act
        component = get_embedding_component(mock_settings)

        # Assert
        assert isinstance(component, MockEmbeddingComponent)

    def test_raises_for_unknown_mode(self, mock_settings):
        # Arrange
        mock_settings.embedding.mode = "unknown"

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            get_embedding_component(mock_settings)
