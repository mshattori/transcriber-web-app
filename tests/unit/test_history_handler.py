"""
Unit tests for HistoryHandler with test data.

Tests both real and mock implementations of job history management.
"""

import os
import json
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from handlers.history_handler import HistoryHandler, MockHistoryHandler
from config.test_config import TestConfig


class TestHistoryHandler:
    """Test suite for real HistoryHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = HistoryHandler()
        self.test_config = TestConfig()
        
        # Load test transcript
        transcript_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_transcript.txt'
        )
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                self.test_transcript = f.read()
        else:
            self.test_transcript = "Test transcript content."
        
        # Create temporary data directory for testing
        self.temp_data_dir = tempfile.mkdtemp()
        self.original_data_dir = self.handler.data_dir
        self.handler.data_dir = self.temp_data_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original data directory
        self.handler.data_dir = self.original_data_dir
        
        # Clean up temporary directory
        if os.path.exists(self.temp_data_dir):
            shutil.rmtree(self.temp_data_dir)
    
    def create_test_job(self, job_id: str, date: str = "2024-08-10", 
                       filename: str = "test_audio.mp3", 
                       include_translation: bool = False):
        """Helper to create a test job in the temp data directory."""
        job_dir = os.path.join(self.temp_data_dir, date, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Create transcript file
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(self.test_transcript)
        
        # Create translation file if requested
        if include_translation:
            translation_path = os.path.join(job_dir, "transcript.ja.txt")
            translation_text = """# 00:00:00 --> 00:02:30
大規模言語モデルのテスト転写です。
これは翻訳機能をテストするためのサンプルコンテンツです。"""
            with open(translation_path, 'w', encoding='utf-8') as f:
                f.write(translation_text)
            
            # Create integrated display file
            from integrated_display import format_integrated_display
            # Use a transcript with timestamps for proper integrated display
            timestamped_transcript = """# 00:00:00 --> 00:02:30
Large language models test transcript.
This is sample content for testing the translation functionality."""
            integrated_display = format_integrated_display(timestamped_transcript, translation_text)
            integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
            with open(integrated_path, 'w', encoding='utf-8') as f:
                f.write(integrated_display)
        
        # Create metadata file
        metadata = {
            "job_id": job_id,
            "timestamp": "2024-08-10T10:30:00",
            "original_filename": filename,
            "file_info": {
                "size_mb": 2.5,
                "duration_seconds": 180.0
            },
            "settings": {
                "default_language": "en",
                "audio_model": "whisper-1"
            },
            "translation_enabled": include_translation,
            "translation_available": include_translation
        }
        
        if include_translation:
            metadata["settings"]["default_translation_language"] = "Japanese"
        
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return job_dir
    
    def test_get_job_history_empty(self):
        """Test getting job history when no jobs exist."""
        jobs = self.handler.get_job_history()
        
        assert isinstance(jobs, list)
        assert len(jobs) == 0
    
    @patch('file_manager.get_display_content_from_job')
    @patch('file_manager.load_job_files')
    def test_load_job_content_with_integrated_display(self, mock_load_files, mock_get_display):
        """Test loading job content with integrated display."""
        # Create a test job first
        job_id = "test-job-123"
        self.create_test_job(job_id, include_translation=True)
        
        # Mock the file manager functions to return integrated display
        mock_get_display.return_value = "# 00:00:00 --> 00:02:30\nTest transcript\n\n──────────── Translation ────────────\n\nテスト転写"
        mock_load_files.return_value = ("Test transcript", "テスト転写", "")
        
        display_content, translation, metadata = self.handler.load_job_content(job_id)
        
        assert "Test transcript" in display_content
        assert "テスト転写" in display_content
        assert "Translation" in display_content
        assert translation == "テスト転写"
        
        # Verify calls were made
        mock_get_display.assert_called_once()
        mock_load_files.assert_called_once()
    
    @patch('file_manager.get_display_content_from_job')
    @patch('file_manager.load_job_files')
    def test_load_job_transcript_with_integrated_display(self, mock_load_files, mock_get_display):
        """Test loading job transcript with integrated display."""
        # Create a test job first
        job_id = "test-job-123"
        self.create_test_job(job_id, include_translation=True)
        
        # Mock the file manager functions to return integrated display
        mock_get_display.return_value = "# 00:00:00 --> 00:02:30\nTest transcript\n\n──────────── Translation ────────────\n\nテスト転写"
        mock_load_files.return_value = ("Test transcript", "テスト転写", "")
        
        display_content, translation = self.handler.load_job_transcript(job_id)
        
        assert "Test transcript" in display_content
        assert "テスト転写" in display_content
        assert "Translation" in display_content
        assert translation == "テスト転写"
    
    def test_get_job_history_with_translation_info(self):
        """Test getting job history with translation information."""
        # Create test jobs with and without translation
        self.create_test_job("job-001", include_translation=True)
        self.create_test_job("job-002", include_translation=False)
        
        jobs = self.handler.get_job_history_with_translation_info()
        
        assert len(jobs) >= 2
        # Check that jobs are returned (exact format may vary based on metadata)
    
    def test_get_job_history_single_job(self):
        """Test getting job history with single job."""
        job_id = "test-001"
        self.create_test_job(job_id)
        
        jobs = self.handler.get_job_history()
        
        assert len(jobs) == 1
        job = jobs[0]
        assert job[0] == job_id  # job_id
        assert job[2] == "test_audio.mp3"  # filename
        assert job[3] == "180.0s"  # duration
        assert job[4] == "en"  # language
        assert job[5] == "Completed"  # status
    
    def test_get_job_history_multiple_jobs(self):
        """Test getting job history with multiple jobs."""
        # Create jobs in different dates
        self.create_test_job("job-001", "2024-08-10", "audio1.mp3")
        self.create_test_job("job-002", "2024-08-09", "audio2.mp3")
        self.create_test_job("job-003", "2024-08-10", "audio3.mp3")
        
        jobs = self.handler.get_job_history()
        
        assert len(jobs) == 3
        
        # Should be sorted by date (newest first)
        job_ids = [job[0] for job in jobs]
        assert "job-001" in job_ids
        assert "job-002" in job_ids 
        assert "job-003" in job_ids
    
    def test_get_job_history_invalid_metadata(self):
        """Test getting job history with invalid metadata files."""
        job_dir = os.path.join(self.temp_data_dir, "2024-08-10", "invalid-job")
        os.makedirs(job_dir, exist_ok=True)
        
        # Create invalid metadata file
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            f.write("{invalid json}")
        
        # Should not crash and return empty list
        jobs = self.handler.get_job_history()
        assert isinstance(jobs, list)
        assert len(jobs) == 0
    
    def test_load_job_transcript_nonexistent_job(self):
        """Test loading transcript for non-existent job."""
        transcript, translation = self.handler.load_job_transcript("nonexistent")
        
        assert transcript == ""
        assert translation == ""
    
    def test_load_job_transcript_existing_job(self):
        """Test loading transcript for existing job."""
        job_id = "test-transcript"
        self.create_test_job(job_id)
        
        display_content, translation = self.handler.load_job_transcript(job_id)
        
        # Should return display content (transcript only since no translation)
        assert self.test_transcript in display_content or display_content == self.test_transcript
        assert translation == ""
    
    def test_load_job_transcript_with_translation(self):
        """Test loading transcript and translation for job with translation."""
        job_id = "test-with-translation"
        self.create_test_job(job_id, include_translation=True)
        
        display_content, translation = self.handler.load_job_transcript(job_id)
        
        # Should return integrated display content when translation exists
        assert len(display_content) > 0
        assert "大規模言語モデル" in translation
        # Display content should contain both transcript and translation
        assert "Large language" in display_content or "test transcript" in display_content
        assert "大規模言語モデル" in display_content or "Translation" in display_content
    
    def test_load_job_transcript_multiple_dates(self):
        """Test loading transcript from job across multiple date folders."""
        # Create jobs in different dates with same ID prefix
        self.create_test_job("multi-001", "2024-08-09")
        self.create_test_job("multi-002", "2024-08-10")
        
        # Should find the job regardless of date folder
        display_content1, _ = self.handler.load_job_transcript("multi-001")
        display_content2, _ = self.handler.load_job_transcript("multi-002")
        
        # Should return display content containing the transcript
        assert self.test_transcript in display_content1 or display_content1 == self.test_transcript
        assert self.test_transcript in display_content2 or display_content2 == self.test_transcript
    
    def test_load_job_transcript_missing_files(self):
        """Test loading transcript when transcript file is missing."""
        job_dir = os.path.join(self.temp_data_dir, "2024-08-10", "missing-transcript")
        os.makedirs(job_dir, exist_ok=True)
        
        # Only create metadata, no transcript file
        metadata = {"job_id": "missing-transcript"}
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        transcript, translation = self.handler.load_job_transcript("missing-transcript")
        
        assert transcript == ""  # Should handle missing file gracefully
        assert translation == ""
    
    def test_has_translation_available_real_jobs(self):
        """Test translation availability check for real jobs."""
        # Create job without translation
        job_id_no_translation = "test-no-translation"
        self.create_test_job(job_id_no_translation, include_translation=False)
        
        # Create job with translation
        job_id_with_translation = "test-with-translation"
        self.create_test_job(job_id_with_translation, include_translation=True)
        
        # Test translation availability
        assert self.handler.has_translation_available(job_id_no_translation) == False
        assert self.handler.has_translation_available(job_id_with_translation) == True
        
        # Test non-existent job
        assert self.handler.has_translation_available("nonexistent") == False
        
        # Test empty job ID
        assert self.handler.has_translation_available("") == False
    
    def test_integrated_display_loading(self):
        """Test that integrated display is loaded when available."""
        job_id = "test-integrated-display"
        self.create_test_job(job_id, include_translation=True)
        
        display_content, translation = self.handler.load_job_transcript(job_id)
        
        # Should return integrated display content
        assert len(display_content) > 0
        assert len(translation) > 0
        
        # Display content should contain both transcript and translation elements
        assert "Large language" in display_content or "test transcript" in display_content
        assert "大規模言語モデル" in display_content or "Translation" in display_content


class TestMockHistoryHandler:
    """Test suite for MockHistoryHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = MockHistoryHandler()
        self.test_config = TestConfig()
    
    def test_get_job_history_returns_mock_data(self):
        """Test that mock returns predefined job history."""
        jobs = self.handler.get_job_history()
        
        assert isinstance(jobs, list)
        assert len(jobs) == 3  # Should have 3 mock jobs
        
        # Check structure of first job
        job = jobs[0]
        assert len(job) == 6  # job_id, timestamp, filename, duration, language, status
        assert job[0].startswith("mock-")  # job_id starts with mock-
        assert "Completed" in job[5]  # status
    
    def test_get_job_history_consistent_results(self):
        """Test that mock returns consistent results."""
        jobs1 = self.handler.get_job_history()
        jobs2 = self.handler.get_job_history()
        
        assert jobs1 == jobs2  # Should be identical
    
    def test_load_job_transcript_known_jobs(self):
        """Test loading transcripts for known mock job IDs."""
        # Load transcript for known mock job
        transcript, translation = self.handler.load_job_transcript("mock-001")
        
        assert len(transcript) > 0
        assert "mock transcript" in transcript.lower()
        assert len(translation) > 0
        assert "モック" in translation or "テスト" in translation
    
    def test_load_job_transcript_unknown_job(self):
        """Test loading transcript for unknown job ID."""
        transcript, translation = self.handler.load_job_transcript("unknown-job")
        
        assert len(transcript) > 0
        assert "unknown-job" in transcript  # Should include the job ID
        assert "mock transcript" in transcript.lower()
        assert len(translation) > 0
    
    def test_load_job_transcript_integrated_display(self):
        """Test that mock handler returns integrated display format."""
        display_content, translation = self.handler.load_job_transcript("mock-001")
        
        # Should contain both transcript and translation content
        assert "mock transcript" in display_content.lower()
        assert "モック" in display_content or "テスト" in display_content
        assert "Translation" in display_content  # Should have translation separator
        
        # Timestamp should appear only once in integrated display
        timestamp_count = display_content.count("# 00:00:00 --> 00:01:00")
        assert timestamp_count == 1
    
    def test_load_job_transcript_no_translation(self):
        """Test loading job with no translation (mock-002)."""
        display_content, translation = self.handler.load_job_transcript("mock-002")
        
        # Should contain transcript but no translation separator
        assert "mock transcript" in display_content.lower()
        assert translation == ""
        assert "Translation" not in display_content
    
    def test_load_job_content_with_metadata(self):
        """Test loading job content with metadata."""
        display_content, translation, metadata = self.handler.load_job_content("mock-001")
        
        assert len(display_content) > 0
        assert len(translation) > 0
        assert isinstance(metadata, dict)
        assert "translation_enabled" in metadata
        assert "translation_available" in metadata
        assert metadata["translation_enabled"] == True
        assert metadata["translation_available"] == True
    
    def test_get_job_history_with_translation_info(self):
        """Test getting job history with translation information."""
        jobs = self.handler.get_job_history_with_translation_info()
        
        assert len(jobs) == 3
        
        # Check that translation info is included in language field
        job_languages = [job[4] for job in jobs]  # language field
        assert "auto+ja" in job_languages  # Job with translation
        assert "en" in job_languages       # Job without translation
        assert "ja+en" in job_languages    # Job with translation
    
    def test_load_job_transcript_empty_job_id(self):
        """Test loading transcript with empty job ID."""
        transcript, translation = self.handler.load_job_transcript("")
        
        assert transcript == ""
        assert translation == ""
    
    def test_load_job_transcript_different_jobs(self):
        """Test that different job IDs return different content."""
        transcript1, translation1 = self.handler.load_job_transcript("mock-001")
        transcript2, translation2 = self.handler.load_job_transcript("mock-002")
        transcript3, translation3 = self.handler.load_job_transcript("mock-003")
        
        # All should be non-empty but different
        assert len(transcript1) > 0
        assert len(transcript2) > 0
        assert len(transcript3) > 0
        
        # Content should be different for different job IDs
        assert transcript1 != transcript2
        assert transcript2 != transcript3
    
    def test_mock_responses_are_instant(self):
        """Test that mock responses are returned instantly."""
        import time
        
        start_time = time.time()
        
        jobs = self.handler.get_job_history()
        transcript, translation = self.handler.load_job_transcript("mock-001")
        
        elapsed_time = time.time() - start_time
        
        # Mock should be very fast (< 10ms)
        assert elapsed_time < 0.01
        assert len(jobs) > 0
        assert len(transcript) > 0
    
    def test_has_translation_available(self):
        """Test translation availability check for mock jobs."""
        # Jobs with translation
        assert self.handler.has_translation_available("mock-001") == True
        assert self.handler.has_translation_available("mock-003") == True
        
        # Job without translation
        assert self.handler.has_translation_available("mock-002") == False
        
        # Unknown job
        assert self.handler.has_translation_available("unknown-job") == False
        
        # Empty job ID
        assert self.handler.has_translation_available("") == False


class TestHistoryHandlerIntegration:
    """Integration tests comparing real and mock handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.real_handler = HistoryHandler()
        self.mock_handler = MockHistoryHandler()
        self.test_config = TestConfig()
    
    def test_handler_interface_compatibility(self):
        """Test that real and mock handlers have compatible interfaces."""
        # Both should have the same methods (excluding attributes)
        real_methods = [method for method in dir(self.real_handler) 
                       if not method.startswith('_') and callable(getattr(self.real_handler, method))]
        mock_methods = [method for method in dir(self.mock_handler) 
                       if not method.startswith('_') and callable(getattr(self.mock_handler, method))]
        
        assert set(real_methods) == set(mock_methods)
    
    def test_both_handlers_return_same_types(self):
        """Test that both handlers return the same data types."""
        # Test get_job_history
        real_jobs = self.real_handler.get_job_history()
        mock_jobs = self.mock_handler.get_job_history()
        
        assert type(real_jobs) == type(mock_jobs)  # Both should return lists
        assert isinstance(real_jobs, list)
        assert isinstance(mock_jobs, list)
        
        # Test load_job_transcript
        real_transcript, real_translation = self.real_handler.load_job_transcript("test")
        mock_transcript, mock_translation = self.mock_handler.load_job_transcript("test")
        
        assert type(real_transcript) == type(mock_transcript)  # Both strings
        assert type(real_translation) == type(mock_translation)  # Both strings
        assert isinstance(real_transcript, str)
        assert isinstance(real_translation, str)
        assert isinstance(mock_transcript, str)
        assert isinstance(mock_translation, str)
    
    def test_job_history_structure_compatibility(self):
        """Test that job history has compatible structure."""
        mock_jobs = self.mock_handler.get_job_history()
        
        # Mock should have jobs
        assert len(mock_jobs) > 0
        
        # Each job should have the expected structure
        for job in mock_jobs:
            assert len(job) == 6  # job_id, timestamp, filename, duration, language, status
            assert isinstance(job[0], str)  # job_id
            assert isinstance(job[1], str)  # timestamp
            assert isinstance(job[2], str)  # filename
            assert isinstance(job[3], str)  # duration
            assert isinstance(job[4], str)  # language
            assert isinstance(job[5], str)  # status
    
    def test_transcript_loading_behavior(self):
        """Test that transcript loading behaves consistently."""
        # Both handlers should handle empty job ID the same way
        real_empty = self.real_handler.load_job_transcript("")
        mock_empty = self.mock_handler.load_job_transcript("")
        
        assert real_empty == ("", "")
        assert mock_empty == ("", "")
        
        # Both should return strings for valid job IDs
        real_result = self.real_handler.load_job_transcript("nonexistent")
        mock_result = self.mock_handler.load_job_transcript("nonexistent")
        
        assert isinstance(real_result[0], str)
        assert isinstance(real_result[1], str)
        assert isinstance(mock_result[0], str)
        assert isinstance(mock_result[1], str)

class TestHistoryHandlerThreeFormatFileLoading:
    """Test suite for 3-format file loading functionality in HistoryHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = HistoryHandler()
        self.mock_handler = MockHistoryHandler()
        self.test_config = TestConfig()
        
        # Create temporary data directory for testing
        self.temp_data_dir = tempfile.mkdtemp()
        self.original_data_dir = self.handler.data_dir
        self.handler.data_dir = self.temp_data_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original data directory
        self.handler.data_dir = self.original_data_dir
        
        # Clean up temporary directory
        if os.path.exists(self.temp_data_dir):
            shutil.rmtree(self.temp_data_dir)
    
    def create_three_format_job(self, job_id: str, date: str = "2024-08-10"):
        """Helper to create a test job with all 3 file formats."""
        job_dir = os.path.join(self.temp_data_dir, date, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Create transcript.txt (original)
        transcript_content = """# 00:00:00 --> 00:02:30
This is the original transcript content.
It contains the raw transcription from the audio file.

# 00:02:30 --> 00:05:00
Second section of the original transcript.
This demonstrates multiple timestamp sections."""
        
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        # Create transcript.ja.txt (translation)
        translation_content = """# 00:00:00 --> 00:02:30
これは元の転写コンテンツです。
音声ファイルからの生の転写が含まれています。

# 00:02:30 --> 00:05:00
元の転写の2番目のセクション。
これは複数のタイムスタンプセクションを示しています。"""
        
        translation_path = os.path.join(job_dir, "transcript.ja.txt")
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(translation_content)
        
        # Create transcript_integrated.txt (integrated display)
        from integrated_display import format_integrated_display
        integrated_content = format_integrated_display(transcript_content, translation_content)
        
        integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
        with open(integrated_path, 'w', encoding='utf-8') as f:
            f.write(integrated_content)
        
        # Create metadata.json
        metadata = {
            "job_id": job_id,
            "timestamp": "2024-08-10T10:30:00",
            "original_filename": "test_audio.mp3",
            "file_info": {
                "size_mb": 2.5,
                "duration_seconds": 300.0
            },
            "settings": {
                "default_language": "en",
                "audio_model": "whisper-1",
                "translation_enabled": True,
                "default_translation_language": "Japanese"
            },
            "translation_enabled": True,
            "translation_available": True,
            "files": {
                "transcript": "transcript.txt",
                "translation": "transcript.ja.txt",
                "integrated": "transcript_integrated.txt"
            }
        }
        
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return job_dir
    
    def test_load_job_content_with_all_three_formats(self):
        """Test loading job content when all 3 file formats are available."""
        job_id = "test-three-formats"
        self.create_three_format_job(job_id)
        
        display_content, translation, metadata = self.handler.load_job_content(job_id)
        
        # Should return integrated display content
        assert len(display_content) > 0
        assert "This is the original transcript content." in display_content
        assert "これは元の転写コンテンツです。" in display_content
        assert "Translation" in display_content
        
        # Translation should be the raw translation content
        assert "これは元の転写コンテンツです。" in translation
        assert "音声ファイルからの生の転写が含まれています。" in translation
        
        # Metadata should indicate translation availability
        assert metadata.get("translation_enabled") == True
        assert metadata.get("translation_available") == True
        assert "files" in metadata
        assert metadata["files"]["integrated"] == "transcript_integrated.txt"
    
    def test_load_job_transcript_prefers_integrated_display(self):
        """Test that load_job_transcript prefers integrated display file when available."""
        job_id = "test-integrated-preference"
        self.create_three_format_job(job_id)
        
        display_content, translation = self.handler.load_job_transcript(job_id)
        
        # Should return integrated display content (not just transcript)
        assert "This is the original transcript content." in display_content
        assert "これは元の転写コンテンツです。" in display_content
        assert "Translation" in display_content
        
        # Timestamp should appear only once (integrated display behavior)
        assert display_content.count("# 00:00:00 --> 00:02:30") == 1
        assert display_content.count("# 00:02:30 --> 00:05:00") == 1
        
        # Translation should be available separately
        assert len(translation) > 0
        assert "これは元の転写コンテンツです。" in translation
    
    def test_load_job_content_missing_integrated_file(self):
        """Test loading job content when integrated file is missing."""
        job_id = "test-missing-integrated"
        job_dir = self.create_three_format_job(job_id)
        
        # Remove integrated display file
        integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
        if os.path.exists(integrated_path):
            os.remove(integrated_path)
        
        display_content, translation, metadata = self.handler.load_job_content(job_id)
        
        # Should still return content (fallback to transcript + translation)
        assert len(display_content) > 0
        assert len(translation) > 0
        
        # Should still indicate translation availability
        assert metadata.get("translation_available") == True
    
    def test_load_job_content_only_transcript_file(self):
        """Test loading job content when only transcript file exists."""
        job_id = "test-transcript-only"
        job_dir = os.path.join(self.temp_data_dir, "2024-08-10", job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Create only transcript file
        transcript_content = "# 00:00:00 --> 00:02:30\nTranscript only content."
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        # Create metadata without translation
        metadata = {
            "job_id": job_id,
            "translation_enabled": False,
            "translation_available": False
        }
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        display_content, translation, metadata_result = self.handler.load_job_content(job_id)
        
        # Should return transcript content only
        assert display_content == transcript_content
        assert translation == ""
        assert metadata_result.get("translation_available") == False
    
    def test_get_job_history_with_three_format_info(self):
        """Test that job history correctly identifies jobs with 3-format files."""
        # Create jobs with different file configurations
        self.create_three_format_job("job-with-all-formats")
        
        # Create job with only transcript
        job_dir = os.path.join(self.temp_data_dir, "2024-08-10", "job-transcript-only")
        os.makedirs(job_dir, exist_ok=True)
        
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("Transcript only content.")
        
        metadata = {
            "job_id": "job-transcript-only",
            "timestamp": "2024-08-10T11:00:00",
            "original_filename": "audio_only.mp3",
            "file_info": {"duration_seconds": 120.0},
            "settings": {"default_language": "en"},
            "translation_enabled": False,
            "translation_available": False
        }
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        jobs = self.handler.get_job_history_with_translation_info()
        
        assert len(jobs) >= 2
        
        # Find our test jobs
        job_languages = {}
        for job in jobs:
            job_languages[job[0]] = job[4]  # job_id -> language
        
        # Job with translation should show combined language
        if "job-with-all-formats" in job_languages:
            assert "+" in job_languages["job-with-all-formats"]  # Should be "en+ja"
        
        # Job without translation should show single language
        if "job-transcript-only" in job_languages:
            assert "+" not in job_languages["job-transcript-only"]  # Should be just "en"
    
    def test_has_translation_available_with_three_formats(self):
        """Test translation availability detection with 3-format files."""
        # Create job with all formats
        job_id_with_translation = "job-with-translation"
        self.create_three_format_job(job_id_with_translation)
        
        # Create job without translation
        job_id_without_translation = "job-without-translation"
        job_dir = os.path.join(self.temp_data_dir, "2024-08-10", job_id_without_translation)
        os.makedirs(job_dir, exist_ok=True)
        
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("Transcript content.")
        
        metadata = {
            "job_id": job_id_without_translation,
            "translation_enabled": False,
            "translation_available": False
        }
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        # Test translation availability
        assert self.handler.has_translation_available(job_id_with_translation) == True
        assert self.handler.has_translation_available(job_id_without_translation) == False
        assert self.handler.has_translation_available("nonexistent-job") == False
    
    def test_mock_handler_three_format_compatibility(self):
        """Test that MockHistoryHandler is compatible with 3-format file loading."""
        # Test load_job_content
        display_content, translation, metadata = self.mock_handler.load_job_content("mock-001")
        
        assert len(display_content) > 0
        assert len(translation) > 0
        assert isinstance(metadata, dict)
        assert "translation_available" in metadata
        
        # Should return integrated display format
        assert "mock transcript" in display_content.lower()
        assert "Translation" in display_content
        
        # Test load_job_transcript
        display_content2, translation2 = self.mock_handler.load_job_transcript("mock-001")
        
        # Should be consistent with load_job_content
        assert display_content == display_content2
        assert translation == translation2
    
    def test_file_format_priority_order(self):
        """Test that files are loaded in the correct priority order."""
        job_id = "test-priority-order"
        job_dir = self.create_three_format_job(job_id)
        
        # Modify integrated file to have different content
        integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
        with open(integrated_path, 'w', encoding='utf-8') as f:
            f.write("INTEGRATED DISPLAY CONTENT - should be preferred")
        
        display_content, translation, metadata = self.handler.load_job_content(job_id)
        
        # Should prefer integrated display file
        assert "INTEGRATED DISPLAY CONTENT" in display_content
        
        # Remove integrated file and test fallback
        os.remove(integrated_path)
        
        display_content2, translation2, metadata2 = self.handler.load_job_content(job_id)
        
        # Should fallback to transcript only (current behavior of get_display_content_from_job)
        assert "This is the original transcript content." in display_content2
        # Translation should still be available separately
        assert len(translation2) > 0
        assert "これは元の転写コンテンツです。" in translation2