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
            with open(translation_path, 'w', encoding='utf-8') as f:
                f.write("大規模言語モデルのテスト転写です。")
        
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
            }
        }
        
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return job_dir
    
    def test_get_job_history_empty(self):
        """Test getting job history when no jobs exist."""
        jobs = self.handler.get_job_history()
        
        assert isinstance(jobs, list)
        assert len(jobs) == 0
    
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
        
        transcript, translation = self.handler.load_job_transcript(job_id)
        
        assert transcript == self.test_transcript
        assert translation == ""
    
    def test_load_job_transcript_with_translation(self):
        """Test loading transcript and translation for job with translation."""
        job_id = "test-with-translation"
        self.create_test_job(job_id, include_translation=True)
        
        transcript, translation = self.handler.load_job_transcript(job_id)
        
        assert transcript == self.test_transcript
        assert translation == "大規模言語モデルのテスト転写です。"
    
    def test_load_job_transcript_multiple_dates(self):
        """Test loading transcript from job across multiple date folders."""
        # Create jobs in different dates with same ID prefix
        self.create_test_job("multi-001", "2024-08-09")
        self.create_test_job("multi-002", "2024-08-10")
        
        # Should find the job regardless of date folder
        transcript1, _ = self.handler.load_job_transcript("multi-001")
        transcript2, _ = self.handler.load_job_transcript("multi-002")
        
        assert transcript1 == self.test_transcript
        assert transcript2 == self.test_transcript
    
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


class TestHistoryHandlerIntegration:
    """Integration tests comparing real and mock handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.real_handler = HistoryHandler()
        self.mock_handler = MockHistoryHandler()
        self.test_config = TestConfig()
    
    def test_handler_interface_compatibility(self):
        """Test that real and mock handlers have compatible interfaces."""
        # Both should have the same methods
        real_methods = [method for method in dir(self.real_handler) if not method.startswith('_')]
        mock_methods = [method for method in dir(self.mock_handler) if not method.startswith('_')]
        
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