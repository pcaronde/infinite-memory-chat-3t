"""
Tests for InfiniteMemoryChat class
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from infinite_memory_chat import InfiniteMemoryChat, MAX_MESSAGES, ARCHIVE_COUNT, MAX_ARCHIVE_FILES, CONSOLIDATION_COUNT
from vector_backends.openai_backend import OpenAIBackend
from vector_backends.mongodb_backend import MongoDBBackend
from config import BackendConfig, BackendType


class TestInfiniteMemoryChatInitialization:
    """Test initialization of InfiniteMemoryChat"""

    def test_init_with_provided_backend(self):
        """Test initialization with a provided backend"""
        mock_backend = Mock(spec=OpenAIBackend)

        with patch('infinite_memory_chat.OpenAI') as mock_openai:
            chat = InfiniteMemoryChat(backend=mock_backend)

            assert chat.backend is mock_backend
            assert chat.conversation_history == []
            assert chat.archive_count == 0
            assert chat.store_id is None
            assert chat.session_id is not None
            mock_openai.assert_called_once()

    def test_init_with_config_openai(self):
        """Test initialization with OpenAI config"""
        config = BackendConfig(
            backend_type=BackendType.OPENAI,
            openai_api_key="test-key"
        )

        with patch('infinite_memory_chat.OpenAI') as mock_openai, \
             patch('infinite_memory_chat.create_backend') as mock_create_backend:

            mock_backend = Mock(spec=OpenAIBackend)
            mock_create_backend.return_value = mock_backend

            chat = InfiniteMemoryChat(config=config)

            assert chat.backend is mock_backend
            assert chat.config is config
            mock_create_backend.assert_called_once_with(config)

    def test_init_with_config_mongodb(self):
        """Test initialization with MongoDB config"""
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            openai_api_key="test-key",
            mongodb_connection_string="mongodb://localhost:27017",
            mongodb_database="test_db"
        )

        with patch('infinite_memory_chat.OpenAI') as mock_openai, \
             patch('infinite_memory_chat.create_backend') as mock_create_backend:

            mock_backend = Mock(spec=MongoDBBackend)
            mock_create_backend.return_value = mock_backend

            chat = InfiniteMemoryChat(config=config)

            assert chat.backend is mock_backend
            assert chat.config is config

    def test_init_fallback_to_openai_on_error(self):
        """Test fallback to OpenAI backend when preferred backend fails"""
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            openai_api_key="test-key",
            mongodb_connection_string="invalid-connection",
            mongodb_database="test_db"
        )

        with patch('infinite_memory_chat.OpenAI') as mock_openai, \
             patch('infinite_memory_chat.create_backend') as mock_create_backend:

            # First call raises error, second call returns fallback backend
            mock_create_backend.side_effect = [
                Exception("MongoDB connection failed"),
                Mock(spec=OpenAIBackend)
            ]

            chat = InfiniteMemoryChat(config=config)

            # Should have called create_backend twice (once for MongoDB, once for fallback)
            assert mock_create_backend.call_count == 2
            # Second call should be with OpenAI config
            fallback_call = mock_create_backend.call_args_list[1]
            fallback_config = fallback_call[0][0]
            assert fallback_config.backend_type == BackendType.OPENAI

    def test_init_session_id_format(self):
        """Test that session_id is generated with correct format"""
        mock_backend = Mock()

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # Session ID should be in format YYYYMMDD_HHMMSS
            assert len(chat.session_id) == 15
            assert chat.session_id[8] == '_'
            # Should be parseable as datetime
            datetime.strptime(chat.session_id, "%Y%m%d_%H%M%S")


class TestVectorStoreSetup:
    """Test vector store setup"""

    def test_setup_vector_store(self):
        """Test setting up a vector store"""
        mock_backend = Mock()
        mock_backend.setup_store.return_value = "store-123"

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)
            store_id = chat.setup_vector_store()

            assert store_id == "store-123"
            assert chat.store_id == "store-123"
            mock_backend.setup_store.assert_called_once_with(chat.session_id)


class TestArchiveMessages:
    """Test message archival functionality"""

    def test_archive_messages_below_threshold(self):
        """Test that archival doesn't happen below MAX_MESSAGES"""
        mock_backend = Mock()

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # Add messages below threshold
            for i in range(MAX_MESSAGES - 1):
                chat.conversation_history.append({
                    "role": "user",
                    "content": f"Message {i}"
                })

            chat.archive_messages()

            # Should not archive
            assert len(chat.conversation_history) == MAX_MESSAGES - 1
            assert chat.archive_count == 0
            mock_backend.archive_messages.assert_not_called()

    def test_archive_messages_at_threshold(self):
        """Test archival when MAX_MESSAGES is reached"""
        mock_backend = Mock()
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = False

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # Add exactly MAX_MESSAGES
            for i in range(MAX_MESSAGES):
                chat.conversation_history.append({
                    "role": "user",
                    "content": f"Message {i}"
                })

            chat.archive_messages()

            # Should archive ARCHIVE_COUNT messages
            assert len(chat.conversation_history) == MAX_MESSAGES - ARCHIVE_COUNT
            assert chat.archive_count == 1

            # Verify archived messages are the oldest ones
            mock_backend.archive_messages.assert_called_once()
            archived_messages = mock_backend.archive_messages.call_args[0][0]
            assert len(archived_messages) == ARCHIVE_COUNT
            assert archived_messages[0]["content"] == "Message 0"

    def test_archive_messages_triggers_consolidation(self):
        """Test that consolidation is triggered when MAX_ARCHIVE_FILES is reached"""
        mock_backend = Mock()
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = True
        mock_backend.get_archive_count.return_value = 50
        mock_backend.consolidate_archives.return_value = {"status": "success"}

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # Add messages to trigger archival
            for i in range(MAX_MESSAGES):
                chat.conversation_history.append({
                    "role": "user",
                    "content": f"Message {i}"
                })

            chat.archive_messages()

            # Should trigger consolidation
            mock_backend.should_consolidate.assert_called_once_with(MAX_ARCHIVE_FILES)
            mock_backend.consolidate_archives.assert_called_once()

    def test_archive_messages_multiple_times(self):
        """Test archiving messages multiple times"""
        mock_backend = Mock()
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = False

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # First archival
            for i in range(MAX_MESSAGES):
                chat.conversation_history.append({"role": "user", "content": f"Msg {i}"})
            chat.archive_messages()

            assert chat.archive_count == 1

            # Second archival
            for i in range(ARCHIVE_COUNT):
                chat.conversation_history.append({"role": "user", "content": f"Msg {i+MAX_MESSAGES}"})
            chat.archive_messages()

            assert chat.archive_count == 2


class TestConsolidation:
    """Test consolidation triggering"""

    def test_trigger_consolidation_with_archives(self):
        """Test consolidation with existing archives"""
        mock_backend = Mock()
        mock_backend.get_archive_count.return_value = CONSOLIDATION_COUNT
        mock_backend.consolidate_archives.return_value = {"status": "success", "files_merged": 50}

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat._trigger_consolidation()

            # Should consolidate CONSOLIDATION_COUNT archives
            mock_backend.consolidate_archives.assert_called_once()
            archive_ids = mock_backend.consolidate_archives.call_args[0][0]
            assert len(archive_ids) == CONSOLIDATION_COUNT

    def test_trigger_consolidation_with_fewer_archives(self):
        """Test consolidation when there are fewer archives than CONSOLIDATION_COUNT"""
        mock_backend = Mock()
        mock_backend.get_archive_count.return_value = 10
        mock_backend.consolidate_archives.return_value = {"status": "success"}

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat._trigger_consolidation()

            # Should only consolidate available archives
            archive_ids = mock_backend.consolidate_archives.call_args[0][0]
            assert len(archive_ids) == 10

    def test_trigger_consolidation_failure(self):
        """Test handling of consolidation failure"""
        mock_backend = Mock()
        mock_backend.get_archive_count.return_value = 50
        mock_backend.consolidate_archives.return_value = None

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)
            # Should not raise exception
            chat._trigger_consolidation()

            mock_backend.consolidate_archives.assert_called_once()


class TestChat:
    """Test chat functionality"""

    def test_chat_basic_message(self):
        """Test basic chat message"""
        mock_backend = Mock(spec=MongoDBBackend)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Hello! How can I help you?"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            response = chat.chat("Hello")

            assert response == "Hello! How can I help you?"
            assert len(chat.conversation_history) == 2  # User + Assistant
            assert chat.conversation_history[0]["role"] == "user"
            assert chat.conversation_history[0]["content"] == "Hello"
            assert chat.conversation_history[1]["role"] == "assistant"

    def test_chat_with_openai_backend_with_archives(self):
        """Test chat using OpenAI backend with archives"""
        mock_backend = Mock(spec=OpenAIBackend)
        mock_backend.vector_store_id = "vs-123"
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response with context"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.store_id = "store-123"
            chat.archive_count = 1

            response = chat.chat("What did we discuss?")

            # Should use file_search tool
            call_args = mock_client.responses.create.call_args
            assert call_args[1]["tools"][0]["type"] == "file_search"
            assert call_args[1]["tools"][0]["vector_store_ids"] == ["vs-123"]

    def test_chat_with_openai_backend_no_archives(self):
        """Test chat using OpenAI backend without archives"""
        mock_backend = Mock(spec=OpenAIBackend)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.archive_count = 0

            response = chat.chat("Hello")

            # Should NOT use file_search tool
            call_args = mock_client.responses.create.call_args
            assert "tools" not in call_args[1]

    def test_chat_with_mongodb_backend_with_archives(self):
        """Test chat using MongoDB backend with archive search"""
        mock_backend = Mock(spec=MongoDBBackend)
        mock_backend.search_archives.return_value = [
            {
                "archive_id": 1,
                "score": 0.95,
                "messages": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"}
                ]
            }
        ]

        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response with MongoDB context"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.archive_count = 1

            response = chat.chat("Follow-up question")

            # Should search archives
            mock_backend.search_archives.assert_called_once_with("Follow-up question", limit=3)

            # Should include archive context in system prompt
            call_args = mock_client.responses.create.call_args
            system_message = call_args[1]["input"][0]
            assert "Relevant conversation history from your memory" in system_message["content"]
            assert "Archive #1" in system_message["content"]

    def test_chat_with_mongodb_backend_no_results(self):
        """Test chat using MongoDB backend when search returns no results"""
        mock_backend = Mock(spec=MongoDBBackend)
        mock_backend.search_archives.return_value = []

        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.archive_count = 1

            response = chat.chat("Question")

            # Should still work without archive context
            assert response == "Response"

    def test_chat_with_mongodb_backend_search_error(self):
        """Test chat when MongoDB archive search fails"""
        mock_backend = Mock(spec=MongoDBBackend)
        mock_backend.search_archives.side_effect = Exception("Search failed")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response despite error"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.archive_count = 1

            # Should not raise exception
            response = chat.chat("Question")
            assert response == "Response despite error"

    def test_chat_triggers_archival(self):
        """Test that chat triggers archival when threshold is reached"""
        mock_backend = Mock()
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = False

        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)

            # Add messages up to threshold
            for i in range(MAX_MESSAGES - 1):
                chat.conversation_history.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Message {i}"
                })

            # This chat should trigger archival
            chat.chat("Last message")

            # Should have archived messages
            mock_backend.archive_messages.assert_called_once()


class TestGetStatus:
    """Test status reporting"""

    def test_get_status_basic(self):
        """Test getting status information"""
        mock_backend = Mock()
        mock_backend.get_status.return_value = {
            "archived_files": 5,
            "consolidations": 1,
            "consolidated_size_mb": 10.5,
            "backend_type": "mongodb"
        }

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.conversation_history = [{"role": "user", "content": "msg"}] * 10
            chat.archive_count = 2
            chat.store_id = "store-123"

            status = chat.get_status()

            assert status["active_messages"] == 10
            assert status["archived_files"] == 5
            assert status["consolidations"] == 1
            assert status["total_messages"] == 10 + (2 * ARCHIVE_COUNT)
            assert status["consolidated_size_mb"] == 10.5
            assert status["backend_type"] == "mongodb"
            assert status["store_id"] == "store-123"

    def test_get_status_no_archives(self):
        """Test status with no archives"""
        mock_backend = Mock()
        mock_backend.get_status.return_value = {
            "archived_files": 0,
            "consolidations": 0,
            "consolidated_size_mb": 0,
            "backend_type": "openai"
        }

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            status = chat.get_status()

            assert status["active_messages"] == 0
            assert status["archived_files"] == 0
            assert status["total_messages"] == 0

    def test_get_status_with_missing_backend_fields(self):
        """Test status when backend returns incomplete data"""
        mock_backend = Mock()
        mock_backend.get_status.return_value = {}

        with patch('infinite_memory_chat.OpenAI'):
            chat = InfiniteMemoryChat(backend=mock_backend)

            status = chat.get_status()

            # Should have default values
            assert status["archived_files"] == 0
            assert status["consolidations"] == 0
            assert status["consolidated_size_mb"] == 0
            assert status["backend_type"] == "unknown"


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_conversation_flow(self):
        """Test a complete conversation flow with archival"""
        mock_backend = Mock()
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = False
        mock_backend.setup_store.return_value = "store-123"
        mock_backend.get_status.return_value = {
            "archived_files": 3,
            "consolidations": 0,
            "consolidated_size_mb": 1.5,
            "backend_type": "test"
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        with patch('infinite_memory_chat.OpenAI', return_value=mock_client):
            chat = InfiniteMemoryChat(backend=mock_backend)
            chat.setup_vector_store()

            # Have a conversation that triggers archival
            # Each chat() adds 2 messages (user + assistant)
            # After 10 chats: 20 messages → archival → 10 remain
            # After 15 chats: 20 messages → archival → 10 remain
            # After 20 chats: 20 messages → archival → 10 remain
            # Total: 3 archival operations
            for i in range(MAX_MESSAGES):
                response = chat.chat(f"Message {i}")
                assert response == "Response"

            # Should have triggered archival 3 times
            assert chat.archive_count == 3
            assert len(chat.conversation_history) == ARCHIVE_COUNT

            # Get status
            status = chat.get_status()
            assert status["archived_files"] == 3
            assert status["active_messages"] == ARCHIVE_COUNT
