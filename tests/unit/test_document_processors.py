"""
Unit tests for document post-processors.

These tests verify filtering, reranking, and pipeline behavior
using mock components to isolate the processor logic.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain.schema import Document as LangchainDocument

from rag_bench.core.types import DocumentWithScore
from rag_bench.core.document_processors import (
    ThresholdFilter,
    SemanticReranker,
    LLMReranker,
    DiversityReranker,
    ProcessingPipeline,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def documents_with_scores():
    """Create test documents with varying scores."""
    docs = [
        LangchainDocument(page_content="High relevance document", metadata={"id": "1"}),
        LangchainDocument(page_content="Medium relevance document", metadata={"id": "2"}),
        LangchainDocument(page_content="Low relevance document", metadata={"id": "3"}),
        LangchainDocument(page_content="Very low relevance document", metadata={"id": "4"}),
    ]
    return [
        DocumentWithScore(document=docs[0], score=0.95),
        DocumentWithScore(document=docs[1], score=0.75),
        DocumentWithScore(document=docs[2], score=0.65),
        DocumentWithScore(document=docs[3], score=0.40),
    ]


# =============================================================================
# ThresholdFilter Tests
# =============================================================================

class TestThresholdFilter:
    """Tests for ThresholdFilter."""

    @pytest.mark.asyncio
    async def test_filters_below_threshold(self, documents_with_scores):
        # Arrange
        filter = ThresholdFilter(threshold=0.7)

        # Act
        result = await filter.process(documents_with_scores, "test query")

        # Assert
        assert len(result) == 2
        assert all(doc.score >= 0.7 for doc in result)

    @pytest.mark.asyncio
    async def test_keeps_all_above_threshold(self, documents_with_scores):
        # Arrange
        filter = ThresholdFilter(threshold=0.3)

        # Act
        result = await filter.process(documents_with_scores, "test query")

        # Assert
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_filters_all_when_threshold_high(self, documents_with_scores):
        # Arrange
        filter = ThresholdFilter(threshold=0.99)

        # Act
        result = await filter.process(documents_with_scores, "test query")

        # Assert
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_handles_empty_input(self):
        # Arrange
        filter = ThresholdFilter(threshold=0.5)

        # Act
        result = await filter.process([], "test query")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_exact_threshold_boundary(self):
        # Arrange
        doc = LangchainDocument(page_content="test", metadata={})
        docs = [DocumentWithScore(document=doc, score=0.7)]
        filter = ThresholdFilter(threshold=0.7)

        # Act
        result = await filter.process(docs, "test query")

        # Assert - score equal to threshold should be kept
        assert len(result) == 1


# =============================================================================
# SemanticReranker Tests
# =============================================================================

class TestSemanticReranker:
    """Tests for SemanticReranker."""

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding that returns predictable vectors."""
        mock = MagicMock()

        # Query embedding - unit vector in one direction
        mock.embed_query.return_value = [1.0, 0.0, 0.0]

        # Document embeddings - varying similarity to query
        mock.embed_documents.return_value = [
            [0.5, 0.5, 0.5],   # Medium similarity
            [0.9, 0.1, 0.1],   # High similarity
            [0.1, 0.9, 0.1],   # Low similarity
        ]
        return mock

    @pytest.mark.asyncio
    async def test_reranks_by_semantic_similarity(self, mock_embedding):
        # Arrange
        docs = [
            LangchainDocument(page_content="doc1", metadata={}),
            LangchainDocument(page_content="doc2", metadata={}),
            LangchainDocument(page_content="doc3", metadata={}),
        ]
        docs_with_scores = [
            DocumentWithScore(document=docs[0], score=0.9),  # Originally highest
            DocumentWithScore(document=docs[1], score=0.8),
            DocumentWithScore(document=docs[2], score=0.7),
        ]
        reranker = SemanticReranker(mock_embedding)

        # Act
        result = await reranker.process(docs_with_scores, "test query")

        # Assert - doc2 should now be first (highest semantic similarity)
        assert result[0].document.page_content == "doc2"

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, mock_embedding):
        # Arrange
        reranker = SemanticReranker(mock_embedding)

        # Act
        result = await reranker.process([], "test query")

        # Assert
        assert result == []


# =============================================================================
# LLMReranker Tests
# =============================================================================

class TestLLMReranker:
    """Tests for LLMReranker."""

    @pytest.fixture
    def mock_llm(self):
        mock = AsyncMock()
        # Return scores in reverse order to test reranking
        mock.agenerate = AsyncMock(return_value="3,8,5")
        return mock

    @pytest.fixture
    def test_docs(self):
        docs = [
            LangchainDocument(page_content="First doc", metadata={}),
            LangchainDocument(page_content="Second doc", metadata={}),
            LangchainDocument(page_content="Third doc", metadata={}),
        ]
        return [
            DocumentWithScore(document=docs[0], score=0.9),
            DocumentWithScore(document=docs[1], score=0.8),
            DocumentWithScore(document=docs[2], score=0.7),
        ]

    @pytest.mark.asyncio
    async def test_reranks_based_on_llm_scores(self, mock_llm, test_docs):
        # Arrange
        reranker = LLMReranker(mock_llm)

        # Act
        result = await reranker.process(test_docs, "test query")

        # Assert - second doc should be first (score 8)
        assert result[0].document.page_content == "Second doc"
        assert result[1].document.page_content == "Third doc"
        assert result[2].document.page_content == "First doc"

    @pytest.mark.asyncio
    async def test_normalizes_scores_to_0_1(self, mock_llm, test_docs):
        # Arrange
        reranker = LLMReranker(mock_llm)

        # Act
        result = await reranker.process(test_docs, "test query")

        # Assert - scores are min-max normalized to [0, 1]
        # LLM returns "3,8,5" -> normalized: 8 becomes 1.0 (max)
        assert result[0].score == 1.0  # Highest score normalized to 1.0
        assert 0.0 <= result[1].score <= 1.0
        assert 0.0 <= result[2].score <= 1.0

    @pytest.mark.asyncio
    async def test_returns_original_on_parse_error(self, test_docs):
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.agenerate = AsyncMock(return_value="invalid response")
        reranker = LLMReranker(mock_llm)

        # Act
        result = await reranker.process(test_docs, "test query")

        # Assert - should return original order
        assert result[0].document.page_content == "First doc"

    @pytest.mark.asyncio
    async def test_returns_original_on_wrong_count(self, test_docs):
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.agenerate = AsyncMock(return_value="8,5")  # Only 2 scores for 3 docs
        reranker = LLMReranker(mock_llm)

        # Act
        result = await reranker.process(test_docs, "test query")

        # Assert - should return original
        assert len(result) == 3
        assert result[0].document.page_content == "First doc"

    @pytest.mark.asyncio
    async def test_returns_original_on_llm_error(self, test_docs):
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.agenerate = AsyncMock(side_effect=Exception("LLM error"))
        reranker = LLMReranker(mock_llm)

        # Act
        result = await reranker.process(test_docs, "test query")

        # Assert - graceful fallback
        assert result == test_docs


# =============================================================================
# ProcessingPipeline Tests
# =============================================================================

class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    @pytest.mark.asyncio
    async def test_applies_processors_in_order(self, documents_with_scores):
        # Arrange
        filter1 = ThresholdFilter(threshold=0.5)  # Removes score < 0.5
        filter2 = ThresholdFilter(threshold=0.7)  # Removes score < 0.7
        pipeline = ProcessingPipeline([filter1, filter2])

        # Act
        result = await pipeline.process(documents_with_scores, "test query")

        # Assert - both filters applied
        assert len(result) == 2
        assert all(doc.score >= 0.7 for doc in result)

    @pytest.mark.asyncio
    async def test_stops_on_empty_result(self):
        # Arrange
        strict_filter = ThresholdFilter(threshold=0.99)
        never_called = AsyncMock()
        pipeline = ProcessingPipeline([strict_filter, never_called])

        docs = [
            DocumentWithScore(
                document=LangchainDocument(page_content="test", metadata={}),
                score=0.5
            )
        ]

        # Act
        result = await pipeline.process(docs, "test query")

        # Assert
        assert result == []
        never_called.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, documents_with_scores):
        # Arrange
        pipeline = ProcessingPipeline([])

        # Act
        result = await pipeline.process(documents_with_scores, "test query")

        # Assert - unchanged
        assert result == documents_with_scores
