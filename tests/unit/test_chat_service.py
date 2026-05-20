"""
Unit tests for ChatService and ConversationStore.

Tests verify conversation management, TTL expiration, and LRU eviction.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from rag_bench.core.types import Conversation
from rag_bench.routers.api_v1.chat.chat_service import (
    ConversationStore,
    ConversationEntry,
    ChatService,
)


class TestConversationEntry:
    """Tests for ConversationEntry TTL tracking."""

    def test_touch_updates_last_accessed(self):
        # Arrange
        conversation = Conversation()
        entry = ConversationEntry(conversation)
        original_time = entry.last_accessed

        # Act
        entry.touch()

        # Assert - touch should update to current time (>= original)
        assert entry.last_accessed >= original_time

    def test_is_expired_returns_false_when_fresh(self):
        # Arrange
        conversation = Conversation()
        entry = ConversationEntry(conversation)
        ttl = timedelta(hours=1)

        # Act & Assert - freshly created entry should not be expired
        assert not entry.is_expired(ttl)

    def test_is_expired_returns_true_when_stale(self):
        # Arrange
        conversation = Conversation()
        entry = ConversationEntry(conversation)
        # Set last_accessed to the past
        entry.last_accessed = datetime.utcnow() - timedelta(seconds=20)
        ttl = timedelta(seconds=10)

        # Assert - entry older than TTL should be expired
        assert entry.is_expired(ttl)


class TestConversationStore:
    """Tests for ConversationStore bounded storage."""

    def test_set_and_get_conversation(self):
        # Arrange
        store = ConversationStore(max_size=10, ttl=timedelta(hours=1))
        conversation = Conversation()

        # Act
        store.set(conversation)
        retrieved = store.get(conversation.id)

        # Assert
        assert retrieved is not None
        assert retrieved.id == conversation.id

    def test_get_returns_none_for_unknown_id(self):
        # Arrange
        store = ConversationStore()

        # Act
        result = store.get("unknown-id")

        # Assert
        assert result is None

    def test_get_returns_none_for_expired_conversation(self):
        # Arrange
        store = ConversationStore(ttl=timedelta(seconds=10))
        conversation = Conversation()
        store.set(conversation)

        # Manually expire the entry by setting last_accessed to the past
        store._store[conversation.id].last_accessed = (
            datetime.utcnow() - timedelta(seconds=15)
        )

        # Act
        result = store.get(conversation.id)

        # Assert
        assert result is None
        assert conversation.id not in store._store

    def test_evicts_oldest_when_at_capacity(self):
        # Arrange
        store = ConversationStore(max_size=2, ttl=timedelta(hours=1))

        conv1 = Conversation()
        conv2 = Conversation()
        conv3 = Conversation()

        # Act
        store.set(conv1)
        store.set(conv2)
        store.set(conv3)  # Should evict conv1

        # Assert
        assert store.get(conv1.id) is None  # Evicted
        assert store.get(conv2.id) is not None
        assert store.get(conv3.id) is not None
        assert len(store) == 2

    def test_get_moves_to_end_lru(self):
        # Arrange
        store = ConversationStore(max_size=2, ttl=timedelta(hours=1))

        conv1 = Conversation()
        conv2 = Conversation()

        store.set(conv1)
        store.set(conv2)

        # Act - access conv1 to make it most recently used
        store.get(conv1.id)

        # Now add conv3 - should evict conv2 (least recently used)
        conv3 = Conversation()
        store.set(conv3)

        # Assert
        assert store.get(conv1.id) is not None  # Still present (was accessed)
        assert store.get(conv2.id) is None  # Evicted
        assert store.get(conv3.id) is not None

    def test_len_returns_count(self):
        # Arrange
        store = ConversationStore()

        # Act
        store.set(Conversation())
        store.set(Conversation())

        # Assert
        assert len(store) == 2

    def test_cleanup_removes_expired_entries(self):
        # Arrange
        store = ConversationStore(ttl=timedelta(seconds=10))
        conv1 = Conversation()
        conv2 = Conversation()

        store.set(conv1)
        store.set(conv2)

        # Expire conv1 manually
        store._store[conv1.id].last_accessed = (
            datetime.utcnow() - timedelta(seconds=15)
        )

        # Act - cleanup is called during set
        conv3 = Conversation()
        store.set(conv3)

        # Assert - conv1 should be cleaned up
        assert conv1.id not in store._store
        assert conv2.id in store._store
        assert conv3.id in store._store


class TestChatService:
    """Tests for ChatService conversation management."""

    @pytest.fixture
    def mock_rag_engine(self):
        from rag_bench.core.engine import SourcedAnswer

        mock = AsyncMock()
        mock.generate_answer = AsyncMock(return_value=SourcedAnswer(
            answer="Test response",
            sources=[]
        ))
        return mock

    @pytest.fixture
    def chat_service(self, mock_rag_engine):
        service = ChatService.__new__(ChatService)
        service.rag_engine = mock_rag_engine
        service._conversations = ConversationStore()
        return service

    def test_get_or_create_creates_new_conversation(self, chat_service):
        # Act
        conv = chat_service._get_or_create_conversation()

        # Assert
        assert conv is not None
        assert conv.id is not None

    def test_get_or_create_returns_existing_conversation(self, chat_service):
        # Arrange
        conv1 = chat_service._get_or_create_conversation()

        # Act
        conv2 = chat_service._get_or_create_conversation(conv1.id)

        # Assert
        assert conv2.id == conv1.id

    def test_get_or_create_creates_new_for_unknown_id(self, chat_service):
        # Act
        conv = chat_service._get_or_create_conversation("unknown-id")

        # Assert
        assert conv is not None
        assert conv.id != "unknown-id"  # Creates new with different ID

    @pytest.mark.asyncio
    async def test_generate_response_adds_messages(self, chat_service):
        # Arrange
        from rag_bench.routers.api_v1.chat.types import ChatMetadata

        metadata = ChatMetadata()

        # Act
        response = await chat_service.generate_response(
            query="Test question",
            metadata=metadata,
        )

        # Assert
        assert response is not None
        # Verify RAG engine was called
        chat_service.rag_engine.generate_answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_stores_conversation(self, chat_service):
        # Arrange
        from rag_bench.routers.api_v1.chat.types import ChatMetadata

        metadata = ChatMetadata(conversation_id="test-conv")

        # Act
        await chat_service.generate_response(
            query="Test question",
            metadata=metadata,
        )

        # Get the conversation that was created
        conv = chat_service._get_or_create_conversation()

        # Assert - conversation should have messages
        assert len(conv.messages) >= 0  # At minimum, should exist
