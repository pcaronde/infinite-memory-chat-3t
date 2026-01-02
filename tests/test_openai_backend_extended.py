"""
Extended tests for OpenAI backend implementation.
Covers edge cases and error scenarios.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from vector_backends.openai_backend import OpenAIBackend


class TestOpenAIBackendInitialization:
    """Test initialization of OpenAI backend"""

    @patch('vector_backends.openai_backend.OpenAI')
    def test_init_without_client(self, mock_openai_class):
        """Test initialization creates a new OpenAI client"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        backend = OpenAIBackend()

        assert backend.client is mock_client
        assert backend.vector_store_id is None
        assert backend.archived_files == []
        assert backend.consolidation_count == 0
        mock_openai_class.assert_called_once()

    def test_init_with_client(self):
        """Test initialization with provided client"""
        mock_client = Mock()
        backend = OpenAIBackend(client=mock_client)

        assert backend.client is mock_client
        assert backend.vector_store_id is None
        assert backend.archived_files == []
        assert backend.consolidation_count == 0


class TestArchiveMessagesExtended:
    """Extended tests for message archival"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        mock_client = Mock()
        mock_client.vector_stores.files.retrieve.return_value = Mock(status="completed")
        return OpenAIBackend(client=mock_client)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_file_creation(self, mock_json_dump, mock_file, backend):
        """Test that archive creates proper file structure"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="file-123")

        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]

        result = backend.archive_messages(messages, 1, "session-123")

        # Verify file was opened for writing
        mock_file.assert_any_call("/tmp/archive_session-123_1.json", "w", encoding="utf-8")

        # Verify JSON structure
        assert mock_json_dump.called
        saved_data = mock_json_dump.call_args[0][0]
        assert "summary" in saved_data
        assert "data" in saved_data
        assert saved_data["data"]["archive_id"] == 1
        assert saved_data["data"]["session_id"] == "session-123"
        assert "User: Question" in saved_data["summary"]
        assert "Assistant: Answer" in saved_data["summary"]

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_uploads_to_openai(self, mock_json_dump, mock_file, backend):
        """Test that archive uploads file to OpenAI"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="file-123")

        messages = [{"role": "user", "content": "Test"}]

        result = backend.archive_messages(messages, 1, "session-123")

        # Verify file was uploaded
        backend.client.files.create.assert_called_once()
        assert backend.client.files.create.call_args[1]["purpose"] == "assistants"

        # Verify file was added to vector store
        backend.client.vector_stores.files.create.assert_called_once_with(
            vector_store_id="vs-123",
            file_id="file-123"
        )

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_tracks_file(self, mock_json_dump, mock_file, backend):
        """Test that archived files are tracked"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="file-123")

        messages = [{"role": "user", "content": "Test"}]

        backend.archive_messages(messages, 5, "session-123")

        assert len(backend.archived_files) == 1
        assert backend.archived_files[0]["file_id"] == "file-123"
        assert backend.archived_files[0]["archive_id"] == 5
        assert backend.archived_files[0]["filename"] == "/tmp/archive_session-123_5.json"

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_upload_indexing_failure(self, mock_json_dump, mock_file, backend):
        """Test archival when indexing fails"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="file-123")
        # Simulate indexing failure
        backend.client.vector_stores.files.retrieve.return_value = Mock(status="failed")

        messages = [{"role": "user", "content": "Test"}]

        with patch('time.time', side_effect=[0, 0, 31]):  # Simulate timeout
            result = backend.archive_messages(messages, 1, "session-123")

        assert result is False

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_openai_upload_error(self, mock_json_dump, mock_file, backend):
        """Test archival handles OpenAI upload errors"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.side_effect = Exception("Upload failed")

        messages = [{"role": "user", "content": "Test"}]

        result = backend.archive_messages(messages, 1, "session-123")

        assert result is False

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_archive_messages_multiple_messages(self, mock_json_dump, mock_file, backend):
        """Test archiving multiple messages"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="file-123")

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        result = backend.archive_messages(messages, 1, "session-123")

        assert result is True
        # Verify all messages are in the summary
        saved_data = mock_json_dump.call_args[0][0]
        summary = saved_data["summary"]
        assert "Message 0" in summary
        assert "Message 9" in summary


class TestWaitForIndexing:
    """Test indexing wait functionality"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        return OpenAIBackend(client=Mock())

    def test_wait_for_indexing_immediate_success(self, backend):
        """Test when file is indexed immediately"""
        backend.vector_store_id = "vs-123"
        backend.client.vector_stores.files.retrieve.return_value = Mock(status="completed")

        result = backend._wait_for_indexing("file-123")

        assert result is True
        backend.client.vector_stores.files.retrieve.assert_called_once()

    @patch('time.sleep')
    def test_wait_for_indexing_eventual_success(self, mock_sleep, backend):
        """Test when file takes time to index"""
        backend.vector_store_id = "vs-123"

        # First two calls return in_progress, third returns completed
        backend.client.vector_stores.files.retrieve.side_effect = [
            Mock(status="in_progress"),
            Mock(status="in_progress"),
            Mock(status="completed")
        ]

        result = backend._wait_for_indexing("file-123", timeout=10)

        assert result is True
        assert backend.client.vector_stores.files.retrieve.call_count == 3

    @patch('time.sleep')
    @patch('time.time')
    def test_wait_for_indexing_timeout(self, mock_time, mock_sleep, backend):
        """Test timeout during indexing"""
        backend.vector_store_id = "vs-123"

        # Simulate timeout by making time progress
        mock_time.side_effect = [0, 0, 31]  # Start, check, timeout
        backend.client.vector_stores.files.retrieve.return_value = Mock(status="in_progress")

        result = backend._wait_for_indexing("file-123", timeout=30)

        assert result is False

    def test_wait_for_indexing_error(self, backend):
        """Test error during status check"""
        backend.vector_store_id = "vs-123"
        backend.client.vector_stores.files.retrieve.side_effect = Exception("API error")

        result = backend._wait_for_indexing("file-123")

        assert result is False


class TestConsolidateArchives:
    """Test archive consolidation"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        mock_client = Mock()
        mock_client.vector_stores.files.retrieve.return_value = Mock(status="completed")
        return OpenAIBackend(client=mock_client)

    def test_consolidate_no_archives(self, backend):
        """Test consolidation with no archives"""
        result = backend.consolidate_archives([])

        assert result is None

    def test_consolidate_archives_not_found(self, backend):
        """Test consolidation when archive IDs don't match"""
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        result = backend.consolidate_archives(["999"])

        assert result is None

    @patch('builtins.open', new_callable=mock_open, read_data='{"data": {"messages": []}}')
    @patch('os.path.getsize', return_value=1024)
    @patch('json.dump')
    @patch('json.load')
    def test_consolidate_archives_success(self, mock_json_load, mock_json_dump, mock_getsize, mock_file, backend):
        """Test successful consolidation"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"},
            {"archive_id": 2, "file_id": "f2", "filename": "/tmp/test2.json"}
        ]

        mock_json_load.return_value = {
            "summary": "Test",
            "data": {"messages": [{"role": "user", "content": "Test"}]}
        }

        backend.client.files.create.return_value = Mock(id="consolidated-file")

        result = backend.consolidate_archives(["1", "2"])

        assert result == "consolidated_1"
        assert backend.consolidation_count == 1
        assert len(backend.archived_files) == 1
        assert backend.archived_files[0]["is_consolidated"] is True

    @patch('builtins.open', new_callable=mock_open, read_data='{"archives": [{"data": "test"}]}')
    @patch('os.path.getsize', return_value=1024)
    @patch('json.dump')
    @patch('json.load')
    def test_consolidate_already_consolidated_files(self, mock_json_load, mock_json_dump, mock_getsize, mock_file, backend):
        """Test consolidating files that are already consolidated"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {
                "archive_id": "consolidated_1",
                "file_id": "f1",
                "filename": "/tmp/consolidated_1.json",
                "is_consolidated": True
            }
        ]

        mock_json_load.return_value = {
            "type": "consolidated_archive",
            "archives": [
                {"data": {"messages": [{"role": "user", "content": "Test1"}]}},
                {"data": {"messages": [{"role": "user", "content": "Test2"}]}}
            ]
        }

        backend.client.files.create.return_value = Mock(id="new-consolidated")

        result = backend.consolidate_archives(["consolidated_1"])

        # Should extract archives from consolidated file
        assert result is not None
        assert backend.consolidation_count == 1

    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('os.path.getsize', return_value=600 * 1024 * 1024)  # 600MB - over limit
    @patch('json.dump')
    @patch('json.load')
    @patch('os.remove')
    def test_consolidate_file_too_large(self, mock_remove, mock_json_load, mock_json_dump, mock_getsize, mock_file, backend):
        """Test consolidation aborts if result file is too large"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        mock_json_load.return_value = {"data": {"messages": []}}

        result = backend.consolidate_archives(["1"])

        assert result is None
        # Should have attempted to remove the oversized file
        mock_remove.assert_called_once()

    @patch('builtins.open')
    @patch('json.load', side_effect=FileNotFoundError)
    @patch('json.dump')
    @patch('os.path.getsize', return_value=1024)
    def test_consolidate_source_file_read_error(self, mock_getsize, mock_json_dump, mock_json_load, mock_file, backend):
        """Test consolidation when source file cannot be read"""
        backend.vector_store_id = "vs-123"
        backend.client.files.create.return_value = Mock(id="consolidated-file")

        # Setup file that will fail to read
        mock_open_handler = mock_open()
        mock_file.side_effect = [
            mock_open_handler.return_value,  # Reading source file
            mock_open_handler.return_value   # Writing consolidated file
        ]

        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test.json"}
        ]

        # Even with read error, consolidation attempts but may fail
        result = backend.consolidate_archives(["1"])

        # Consolidation fails when source files can't be read properly
        assert result is None

    @patch('builtins.open', new_callable=mock_open, read_data='{"data": {}}')
    @patch('os.path.getsize', return_value=1024)
    @patch('json.dump')
    @patch('json.load')
    def test_consolidate_upload_error(self, mock_json_load, mock_json_dump, mock_getsize, mock_file, backend):
        """Test consolidation handles upload errors"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        mock_json_load.return_value = {"data": {"messages": []}}
        backend.client.files.create.side_effect = Exception("Upload failed")

        result = backend.consolidate_archives(["1"])

        assert result is None

    @patch('builtins.open', new_callable=mock_open, read_data='{"data": {}}')
    @patch('os.path.getsize')
    @patch('json.dump')
    @patch('json.load')
    def test_consolidate_skips_large_consolidated_files(self, mock_json_load, mock_json_dump, mock_getsize, mock_file, backend):
        """Test consolidation skips files near size limit"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {
                "archive_id": "consolidated_1",
                "file_id": "f1",
                "filename": "/tmp/large.json",
                "is_consolidated": True
            }
        ]

        # File is 450MB (90% of 500MB limit)
        mock_getsize.return_value = 450 * 1024 * 1024

        result = backend.consolidate_archives(["consolidated_1"])

        assert result is None


class TestCleanupArchives:
    """Test archive cleanup"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        return OpenAIBackend(client=Mock())

    def test_cleanup_single_archive(self, backend):
        """Test cleaning up a single archive"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"},
            {"archive_id": 2, "file_id": "f2", "filename": "/tmp/test2.json"}
        ]

        result = backend.cleanup_archives(["1"])

        assert result is True
        assert len(backend.archived_files) == 1
        assert backend.archived_files[0]["archive_id"] == 2

        # Verify API calls
        backend.client.vector_stores.files.delete.assert_called_once_with(
            vector_store_id="vs-123",
            file_id="f1"
        )
        backend.client.files.delete.assert_called_once_with(file_id="f1")

    def test_cleanup_multiple_archives(self, backend):
        """Test cleaning up multiple archives"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"},
            {"archive_id": 2, "file_id": "f2", "filename": "/tmp/test2.json"},
            {"archive_id": 3, "file_id": "f3", "filename": "/tmp/test3.json"}
        ]

        result = backend.cleanup_archives(["1", "3"])

        assert result is True
        assert len(backend.archived_files) == 1
        assert backend.archived_files[0]["archive_id"] == 2

    def test_cleanup_nonexistent_archive(self, backend):
        """Test cleanup of archive that doesn't exist"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        result = backend.cleanup_archives(["999"])

        assert result is False
        assert len(backend.archived_files) == 1

    def test_cleanup_with_deletion_error(self, backend):
        """Test cleanup when deletion fails"""
        backend.vector_store_id = "vs-123"
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        backend.client.vector_stores.files.delete.side_effect = Exception("Delete failed")

        result = backend.cleanup_archives(["1"])

        assert result is False
        # File should still be removed from tracking despite error
        assert len(backend.archived_files) == 1

    def test_cleanup_empty_list(self, backend):
        """Test cleanup with empty list"""
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/test1.json"}
        ]

        result = backend.cleanup_archives([])

        assert result is True
        assert len(backend.archived_files) == 1


class TestGetStatus:
    """Test status reporting"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        return OpenAIBackend(client=Mock())

    def test_get_status_empty(self, backend):
        """Test status with no archives"""
        status = backend.get_status()

        assert status["backend_type"] == "openai"
        assert status["vector_store_id"] is None
        assert status["archived_files"] == 0
        assert status["consolidations"] == 0
        assert status["consolidated_size_mb"] == 0
        assert status["files"] == []

    @patch('os.path.getsize', return_value=2048)
    def test_get_status_with_archives(self, mock_getsize, backend):
        """Test status with archived files"""
        backend.vector_store_id = "vs-123"
        backend.consolidation_count = 2
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1"},
            {
                "archive_id": "consolidated_1",
                "file_id": "f2",
                "filename": "/tmp/consolidated_1.json",
                "is_consolidated": True
            }
        ]

        status = backend.get_status()

        assert status["vector_store_id"] == "vs-123"
        assert status["archived_files"] == 2
        assert status["consolidations"] == 2
        assert status["consolidated_size_mb"] == 2048 / 1024 / 1024
        assert len(status["files"]) == 2

    @patch('os.path.getsize', side_effect=FileNotFoundError)
    def test_get_status_missing_file(self, mock_getsize, backend):
        """Test status when consolidated file is missing"""
        backend.archived_files = [
            {
                "archive_id": "consolidated_1",
                "file_id": "f1",
                "filename": "/tmp/missing.json",
                "is_consolidated": True
            }
        ]

        status = backend.get_status()

        # Should not crash, just skip missing file
        assert status["consolidated_size_mb"] == 0

    def test_get_status_file_metadata(self, backend):
        """Test status includes file metadata"""
        backend.archived_files = [
            {"archive_id": 1, "file_id": "f1", "filename": "/tmp/archive_1.json"},
            {
                "archive_id": "consolidated_1",
                "file_id": "f2",
                "filename": "/tmp/consolidated_1.json",
                "is_consolidated": True
            }
        ]

        with patch('os.path.getsize', return_value=1024):
            status = backend.get_status()

        assert len(status["files"]) == 2
        assert status["files"][0]["archive_id"] == 1
        assert status["files"][0]["is_consolidated"] is False
        assert status["files"][1]["archive_id"] == "consolidated_1"
        assert status["files"][1]["is_consolidated"] is True


class TestShouldConsolidate:
    """Test consolidation trigger logic"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        return OpenAIBackend(client=Mock())

    def test_should_consolidate_below_threshold(self, backend):
        """Test consolidation not triggered below threshold"""
        backend.archived_files = [{"archive_id": i} for i in range(50)]

        assert not backend.should_consolidate(100)

    def test_should_consolidate_at_threshold(self, backend):
        """Test consolidation triggered at threshold"""
        backend.archived_files = [{"archive_id": i} for i in range(100)]

        assert backend.should_consolidate(100)

    def test_should_consolidate_above_threshold(self, backend):
        """Test consolidation triggered above threshold"""
        backend.archived_files = [{"archive_id": i} for i in range(150)]

        assert backend.should_consolidate(100)

    def test_should_consolidate_empty(self, backend):
        """Test consolidation not triggered with no archives"""
        assert not backend.should_consolidate(100)


class TestGetArchiveCount:
    """Test archive count"""

    @pytest.fixture
    def backend(self):
        """Create backend with mocked client"""
        return OpenAIBackend(client=Mock())

    def test_get_archive_count_empty(self, backend):
        """Test count with no archives"""
        assert backend.get_archive_count() == 0

    def test_get_archive_count_with_files(self, backend):
        """Test count with archived files"""
        backend.archived_files = [
            {"archive_id": 1},
            {"archive_id": 2},
            {"archive_id": "consolidated_1"}
        ]

        assert backend.get_archive_count() == 3
