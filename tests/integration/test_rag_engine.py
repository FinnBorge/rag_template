"""
Integration tests for RAGEngine.

These tests verify the full RAG pipeline works correctly with
mock components, testing the interaction between:
- Query enhancement
- Document retrieval
- Document post-processing
- Answer generation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain.schema import Document as LangchainDocument

from rag_bench.core.engine import RAGEngine, SourcedAnswer
from rag_bench.core.types import DocumentWithScore
from rag_bench.core.query_enhancers import StopWordRemovalEnhancer
from rag_bench.core.document_processors import ThresholdFilter


class TestRAGEnginePipeline:
    """Integration tests for the full RAG pipeline."""

    @pytest.mark.asyncio
    async def test_generate_answer_returns_sourced_answer(
        self, rag_engine, mock_vector_store_component, sample_documents
    ):
        # Arrange - add documents to mock store
        await mock_vector_store_component.aadd_documents(sample_documents)

        # Act
        result = await rag_engine.generate_answer("What is RAG?")

        # Assert
        assert isinstance(result, SourcedAnswer)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_includes_sources_in_response(
        self, rag_engine, mock_vector_store_component, sample_documents
    ):
        # Arrange
        await mock_vector_store_component.aadd_documents(sample_documents)

        # Act
        result = await rag_engine.generate_answer("embeddings question")

        # Assert
        assert isinstance(result.sources, list)
        # Sources should contain metadata from retrieved docs

    @pytest.mark.asyncio
    async def test_handles_no_documents(self, rag_engine):
        # Arrange - empty vector store

        # Act
        result = await rag_engine.generate_answer("query with no matching docs")

        # Assert - should still return a response
        assert isinstance(result, SourcedAnswer)
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_applies_query_enhancers(
        self, mock_settings, mock_llm_component, mock_vector_store_component
    ):
        # Arrange
        enhancer = StopWordRemovalEnhancer()
        engine = RAGEngine(
            settings=mock_settings,
            llm_component=mock_llm_component,
            vector_store_component=mock_vector_store_component,
            query_enhancers=[enhancer],
            document_post_processors=[],
        )

        # Act
        enhanced = await engine._enhance_query("what is the meaning of life")

        # Assert - stop words should be removed
        assert "the" not in enhanced.split()
        assert "meaning" in enhanced
        assert "life" in enhanced

    @pytest.mark.asyncio
    async def test_applies_document_processors(
        self, mock_settings, mock_llm_component, mock_vector_store_component, sample_documents
    ):
        # Arrange
        await mock_vector_store_component.aadd_documents(sample_documents)

        # Use a strict threshold filter
        processor = ThresholdFilter(threshold=0.99)
        engine = RAGEngine(
            settings=mock_settings,
            llm_component=mock_llm_component,
            vector_store_component=mock_vector_store_component,
            query_enhancers=[],
            document_post_processors=[processor],
        )

        # Act
        result = await engine.generate_answer("test query")

        # Assert - high threshold should filter all docs
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_conversation_id_passed_to_enhancers(
        self, mock_settings, mock_llm_component, mock_vector_store_component
    ):
        # Arrange
        mock_enhancer = AsyncMock()
        mock_enhancer.enhance = AsyncMock(return_value="enhanced query")

        engine = RAGEngine(
            settings=mock_settings,
            llm_component=mock_llm_component,
            vector_store_component=mock_vector_store_component,
            query_enhancers=[mock_enhancer],
            document_post_processors=[],
        )

        # Act
        await engine.generate_answer("test query", conversation_id="conv-123")

        # Assert
        mock_enhancer.enhance.assert_called_once_with("test query", "conv-123")


class TestRAGEngineDocumentFormatting:
    """Tests for document formatting in the RAG engine."""

    @pytest.fixture
    def docs_with_metadata(self):
        return [
            DocumentWithScore(
                document=LangchainDocument(
                    page_content="Document content here",
                    metadata={"source": "test.txt", "url": "http://example.com"}
                ),
                score=0.9
            ),
        ]

    def test_format_documents_includes_source(self, rag_engine, docs_with_metadata):
        # Act
        formatted = rag_engine._format_documents_for_prompt(docs_with_metadata)

        # Assert
        assert "test.txt" in formatted
        assert "Document content here" in formatted

    def test_prepare_sources_extracts_metadata(self, rag_engine, docs_with_metadata):
        # Act
        sources = rag_engine._prepare_sources_for_response(docs_with_metadata)

        # Assert
        assert len(sources) == 1
        assert sources[0]["source"] == "test.txt"
        assert sources[0]["url"] == "http://example.com"
        assert sources[0]["relevance_score"] == 0.9


class TestRAGEngineErrorHandling:
    """Tests for error handling in the RAG engine."""

    @pytest.mark.asyncio
    async def test_continues_on_enhancer_error(
        self, mock_settings, mock_llm_component, mock_vector_store_component
    ):
        # Arrange
        failing_enhancer = AsyncMock()
        failing_enhancer.enhance = AsyncMock(side_effect=Exception("Enhancer failed"))

        engine = RAGEngine(
            settings=mock_settings,
            llm_component=mock_llm_component,
            vector_store_component=mock_vector_store_component,
            query_enhancers=[failing_enhancer],
            document_post_processors=[],
        )

        # Act - should not raise
        result = await engine.generate_answer("test query")

        # Assert - should still return result
        assert isinstance(result, SourcedAnswer)

    @pytest.mark.asyncio
    async def test_continues_on_processor_error(
        self, mock_settings, mock_llm_component, mock_vector_store_component, sample_documents
    ):
        # Arrange
        await mock_vector_store_component.aadd_documents(sample_documents)

        failing_processor = AsyncMock()
        failing_processor.process = AsyncMock(side_effect=Exception("Processor failed"))

        engine = RAGEngine(
            settings=mock_settings,
            llm_component=mock_llm_component,
            vector_store_component=mock_vector_store_component,
            query_enhancers=[],
            document_post_processors=[failing_processor],
        )

        # Act - should not raise
        result = await engine.generate_answer("test query")

        # Assert
        assert isinstance(result, SourcedAnswer)
