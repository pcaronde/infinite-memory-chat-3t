"""
Tests for vector backend implementations.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from vector_backends.base import VectorBackend
from vector_backends.openai_backend import OpenAIBackend


class TestVectorBackendInterface:
    """Test that backend implementations follow the interface."""
    
    @patch('vector_backends.openai_backend.OpenAI')
    def test_openai_backend_implements_interface(self, mock_openai):
        """Test that OpenAI backend implements all required methods."""
        backend = OpenAIBackend()
        
        # Check that all abstract methods are implemented
        assert hasattr(backend, 'setup_store')
        assert hasattr(backend, 'archive_messages')
        assert hasattr(backend, 'search_archives')
        assert hasattr(backend, 'consolidate_archives')
        assert hasattr(backend, 'get_archive_count')
        assert hasattr(backend, 'cleanup_archives')
        assert hasattr(backend, 'get_status')
        assert hasattr(backend, 'should_consolidate')


class TestOpenAIBackend:
    """Test OpenAI backend implementation."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        # Mock vector store creation
        client.vector_stores.create.return_value = Mock(id="test_vector_store_id")
        # Mock file creation
        client.files.create.return_value = Mock(id="test_file_id")
        # Mock vector store file operations
        client.vector_stores.files.create.return_value = None
        client.vector_stores.files.retrieve.return_value = Mock(status="completed")
        client.vector_stores.files.delete.return_value = None
        client.files.delete.return_value = None
        return client
    
    @pytest.fixture
    def backend(self, mock_openai_client):
        """Create OpenAI backend with mocked client."""
        return OpenAIBackend(client=mock_openai_client)
    
    def test_setup_store(self, backend, mock_openai_client):
        """Test vector store setup."""
        session_id = "test_session_123"
        
        store_id = backend.setup_store(session_id)
        
        assert store_id == "test_vector_store_id"
        assert backend.vector_store_id == "test_vector_store_id"
        mock_openai_client.vector_stores.create.assert_called_once_with(
            name=f"chat_memory_{session_id}"
        )
    
    def test_get_archive_count_initially_zero(self, backend):
        """Test that archive count starts at zero."""
        assert backend.get_archive_count() == 0
    
    def test_should_consolidate(self, backend):
        """Test consolidation trigger logic."""
        assert not backend.should_consolidate(100)  # No archives yet
        
        # Add some mock archived files
        backend.archived_files = [{"archive_id": i} for i in range(100)]
        assert backend.should_consolidate(100)
        assert not backend.should_consolidate(101)
    
    def test_get_status(self, backend):
        """Test status reporting."""
        status = backend.get_status()
        
        expected_keys = [
            "backend_type", "vector_store_id", "archived_files", 
            "consolidations", "consolidated_size_mb", "files"
        ]
        
        for key in expected_keys:
            assert key in status
        
        assert status["backend_type"] == "openai"
        assert status["archived_files"] == 0
        assert status["consolidations"] == 0
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_archive_messages_success(self, mock_json_dump, mock_open, backend, mock_openai_client):
        """Test successful message archival."""
        backend.vector_store_id = "test_store_id"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = backend.archive_messages(messages, 1, "test_session")
        
        assert result is True
        assert len(backend.archived_files) == 1
        assert backend.archived_files[0]["archive_id"] == 1
    
    def test_archive_messages_no_store(self, backend):
        """Test archival fails without vector store."""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = backend.archive_messages(messages, 1, "test_session")
        
        assert result is False
    
    def test_search_archives_returns_empty(self, backend):
        """Test search returns empty list (OpenAI handles search via file_search tool)."""
        results = backend.search_archives("test query")
        assert results == []


class TestBackendIntegration:
    """Integration tests for backend usage in main application."""
    
    @patch('infinite_memory_chat.OpenAI')  # Mock the OpenAI client in main module
    def test_backend_can_be_injected(self, mock_openai):
        """Test that a backend can be injected into InfiniteMemoryChat."""
        from infinite_memory_chat import InfiniteMemoryChat
        from vector_backends.openai_backend import OpenAIBackend
        
        mock_client = Mock()
        backend = OpenAIBackend(client=mock_client)
        
        chat = InfiniteMemoryChat(backend=backend)
        
        assert chat.backend is backend
        assert isinstance(chat.backend, OpenAIBackend)
    
    @patch('os.environ.get')
    @patch('infinite_memory_chat.OpenAI')  # Mock the OpenAI client in main module  
    @patch('vector_backends.openai_backend.OpenAI')  # Mock the OpenAI client in backend
    def test_default_backend_is_openai(self, mock_backend_openai, mock_main_openai, mock_env_get):
        """Test that default backend is OpenAI."""
        from infinite_memory_chat import InfiniteMemoryChat
        from vector_backends.openai_backend import OpenAIBackend
        
        # Mock environment variable for API key
        def mock_env_side_effect(key, default=None):
            if key == 'OPENAI_API_KEY':
                return 'test-api-key'
            elif key == 'VECTOR_BACKEND':
                return 'openai'  # Default to OpenAI
            return default
        
        mock_env_get.side_effect = mock_env_side_effect
        
        chat = InfiniteMemoryChat()
        assert isinstance(chat.backend, OpenAIBackend)
