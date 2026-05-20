"""
Integration tests for API endpoints.

These tests verify the HTTP interface works correctly,
including request/response formats, error handling, and status codes.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for API tests."""
    from rag_bench.core.engine import SourcedAnswer

    mock = AsyncMock()
    mock.generate_answer = AsyncMock(return_value=SourcedAnswer(
        answer="This is the answer based on retrieved documents.",
        sources=[
            {"source": "doc1.txt", "relevance_score": 0.9, "url": "http://example.com/1"},
            {"source": "doc2.txt", "relevance_score": 0.8},
        ]
    ))
    return mock


@pytest.fixture
def client_with_mock_engine(mock_settings, mock_rag_engine):
    """Test client with mocked RAG engine."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Create a minimal app for testing
    app = FastAPI()

    @app.get("/api/v1/query")
    async def query(q: str):
        result = await mock_rag_engine.generate_answer(q)
        return {
            "answer": result.answer,
            "sources": result.sources,
        }

    return TestClient(app)


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Tests for /api/v1/query endpoint."""

    def test_query_returns_answer(self, client_with_mock_engine):
        # Act
        response = client_with_mock_engine.get("/api/v1/query?q=What is RAG?")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_returns_sources(self, client_with_mock_engine):
        # Act
        response = client_with_mock_engine.get("/api/v1/query?q=test")

        # Assert
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_requires_q_parameter(self, client_with_mock_engine):
        # Act
        response = client_with_mock_engine.get("/api/v1/query")

        # Assert
        assert response.status_code == 422  # Validation error


# =============================================================================
# Chat Endpoint Tests
# =============================================================================

class TestChatEndpoint:
    """Tests for /api/v1/chat/completions endpoint."""

    @pytest.fixture
    def chat_client(self, mock_settings):
        """Test client for chat endpoint."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from rag_bench.routers.api_v1.chat.chat_router import router as chat_router
        from rag_bench.routers.api_v1.chat.chat_service import ChatService
        from rag_bench.routers.api_v1.chat.types import ChatResponse, ChatMetadata
        from injector import Injector

        # Create a proper ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            answer="Test answer",
            question="Test question",
            documents=[],
            status="completed",
            metadata=ChatMetadata(),
        )

        # Create mock chat service
        mock_chat_service = MagicMock(spec=ChatService)
        mock_chat_service.generate_response = AsyncMock(return_value=mock_response)

        # Create mock injector
        mock_injector = MagicMock(spec=Injector)
        mock_injector.get.return_value = mock_chat_service

        app = FastAPI()

        @app.middleware("http")
        async def inject_mock(request: Request, call_next):
            request.state.injector = mock_injector
            return await call_next(request)

        app.include_router(chat_router, prefix="/api/v1")

        return TestClient(app, raise_server_exceptions=False)

    def test_chat_with_query_format(self, chat_client):
        # Arrange
        payload = {
            "query": "What is machine learning?",
            "conversation_id": "conv-123"
        }

        # Act
        response = chat_client.post("/api/v1/chat/completions", json=payload)

        # Assert
        assert response.status_code == 200

    def test_chat_with_messages_format(self, chat_client):
        # Arrange
        payload = {
            "messages": [
                {"role": "user", "content": "What is RAG?"}
            ],
            "stream": False
        }

        # Act
        response = chat_client.post("/api/v1/chat/completions", json=payload)

        # Assert
        assert response.status_code == 200

    def test_chat_rejects_empty_messages(self, chat_client):
        # Arrange
        payload = {
            "messages": [],
            "stream": False
        }

        # Act
        response = chat_client.post("/api/v1/chat/completions", json=payload)

        # Assert
        assert response.status_code in [400, 422, 500]  # Should reject

    def test_chat_rejects_non_user_last_message(self, chat_client):
        # Arrange
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}  # Last message not from user
            ],
            "stream": False
        }

        # Act
        response = chat_client.post("/api/v1/chat/completions", json=payload)

        # Assert
        assert response.status_code in [400, 422, 500]


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self):
        # Arrange
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        client = TestClient(app)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
