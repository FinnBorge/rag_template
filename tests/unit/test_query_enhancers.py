"""
Unit tests for query enhancers.

These tests verify that each enhancer correctly transforms queries
without testing the full RAG pipeline.
"""
import pytest
from unittest.mock import AsyncMock

from rag_bench.core.query_enhancers import (
    HyponymExpansionEnhancer,
    LLMQueryExpansionEnhancer,
    StopWordRemovalEnhancer,
    HybridQueryEnhancer,
)


class TestStopWordRemovalEnhancer:
    """Tests for StopWordRemovalEnhancer."""

    @pytest.fixture
    def enhancer(self):
        return StopWordRemovalEnhancer()

    @pytest.mark.asyncio
    async def test_removes_common_stop_words(self, enhancer):
        # Arrange
        query = "what is the meaning of life"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert "the" not in result.split()
        assert "of" not in result.split()
        assert "meaning" in result
        assert "life" in result

    @pytest.mark.asyncio
    async def test_preserves_query_when_all_stop_words(self, enhancer):
        # Arrange
        query = "the a an"

        # Act
        result = await enhancer.enhance(query)

        # Assert - should return original when nothing would remain
        assert result == query

    @pytest.mark.asyncio
    async def test_handles_empty_query(self, enhancer):
        # Arrange
        query = ""

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_custom_stop_words(self):
        # Arrange
        custom_stops = {"foo", "bar"}
        enhancer = StopWordRemovalEnhancer(stop_words=custom_stops)
        query = "foo baz bar qux"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert result == "baz qux"


class TestHyponymExpansionEnhancer:
    """Tests for HyponymExpansionEnhancer."""

    @pytest.fixture
    def enhancer(self):
        hyponym_map = {
            "medication": ["drug", "pill", "tablet"],
            "doctor": ["physician", "specialist"],
        }
        return HyponymExpansionEnhancer(hyponym_map)

    @pytest.mark.asyncio
    async def test_expands_known_terms(self, enhancer):
        # Arrange
        query = "what medication should I take"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert "medication" in result
        assert "drug" in result or "pill" in result or "tablet" in result
        assert "OR" in result

    @pytest.mark.asyncio
    async def test_no_expansion_for_unknown_terms(self, enhancer):
        # Arrange
        query = "what is the weather today"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert result == query

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, enhancer):
        # Arrange
        query = "ask the DOCTOR about this"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert "physician" in result or "specialist" in result

    @pytest.mark.asyncio
    async def test_does_not_duplicate_existing_terms(self, enhancer):
        # Arrange - query already contains a hyponym
        query = "what drug or medication should I take"

        # Act
        result = await enhancer.enhance(query)

        # Assert - should not add "drug" again since it's already present
        assert result.count("drug") == 1


class TestLLMQueryExpansionEnhancer:
    """Tests for LLMQueryExpansionEnhancer."""

    @pytest.fixture
    def mock_llm(self):
        mock = AsyncMock()
        mock.agenerate = AsyncMock(return_value="expanded query with synonyms and related terms")
        return mock

    @pytest.fixture
    def enhancer(self, mock_llm):
        return LLMQueryExpansionEnhancer(mock_llm)

    @pytest.mark.asyncio
    async def test_calls_llm_with_query(self, enhancer, mock_llm):
        # Arrange
        query = "diabetes treatment options"

        # Act
        await enhancer.enhance(query)

        # Assert
        mock_llm.agenerate.assert_called_once()
        call_kwargs = mock_llm.agenerate.call_args.kwargs
        assert call_kwargs["query"] == query

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, enhancer):
        # Arrange
        query = "diabetes treatment"

        # Act
        result = await enhancer.enhance(query)

        # Assert
        assert result == "expanded query with synonyms and related terms"

    @pytest.mark.asyncio
    async def test_returns_original_on_llm_error(self, mock_llm):
        # Arrange
        mock_llm.agenerate = AsyncMock(side_effect=Exception("LLM error"))
        enhancer = LLMQueryExpansionEnhancer(mock_llm)
        query = "original query"

        # Act
        result = await enhancer.enhance(query)

        # Assert - should gracefully fall back to original
        assert result == query


class TestHybridQueryEnhancer:
    """Tests for HybridQueryEnhancer (pipeline of enhancers)."""

    @pytest.mark.asyncio
    async def test_applies_enhancers_in_order(self):
        # Arrange - create enhancers that leave a trace
        class TracingEnhancer:
            def __init__(self, suffix):
                self.suffix = suffix

            async def enhance(self, query, conversation_id=None):
                return f"{query}_{self.suffix}"

        enhancer1 = TracingEnhancer("first")
        enhancer2 = TracingEnhancer("second")
        hybrid = HybridQueryEnhancer([enhancer1, enhancer2])

        # Act
        result = await hybrid.enhance("query")

        # Assert - order matters
        assert result == "query_first_second"

    @pytest.mark.asyncio
    async def test_empty_enhancer_list(self):
        # Arrange
        hybrid = HybridQueryEnhancer([])

        # Act
        result = await hybrid.enhance("unchanged query")

        # Assert
        assert result == "unchanged query"
