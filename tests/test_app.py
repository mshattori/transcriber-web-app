"""
Unit tests for app.py module.

Tests main application functionality, settings validation, and utility functions.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import (
    AppState,
    load_default_settings,
    validate_settings,
    format_transcript_for_display,
    create_download_files,
    get_job_history,
    load_job_transcript
)


class TestAppState:
    """Test application state management."""
    
    def test_app_state_initialization(self):
        """Test AppState initialization."""
        state = AppState()
        
        assert state.current_job_id is None
        assert state.current_transcript is None
        assert state.current_translation is None
        assert state.chat_history == []
        assert state.processing_progress == 0.0
        assert state.processing_message == ""


class TestDefaultSettings:
    """Test default settings functionality."""
    
    @patch('src.app.load_config')
    def test_load_default_settings_success(self, mock_load_config):
        """Test successful loading of default settings."""
        mock_config = {
            'audio_models': ['whisper-1', 'gpt-4o-mini-transcribe'],
            'language_models': ['gpt-4o-mini', 'gpt-4o'],
            'system_message': 'Test system message',
            'default_language': 'auto',
            'default_translation_language': 'Japanese',
            'default_chunk_minutes': 5
        }
        mock_load_config.return_value = mock_config
        
        settings = load_default_settings()
        
        assert settings['audio_model'] == 'whisper-1'
        assert settings['language_model'] == 'gpt-4o-mini'
        assert settings['system_message'] == 'Test system message'
        assert settings['default_language'] == 'auto'
        assert settings['chunk_minutes'] == 5
        assert settings['translation_enabled'] is False
    
    @patch('src.app.load_config')
    def test_load_default_settings_fallback(self, mock_load_config):
        """Test fallback when config loading fails."""
        mock_load_config.side_effect = Exception("Config error")
        
        settings = load_default_settings()
        
        # Should return fallback settings
        assert settings['audio_model'] == 'whisper-1'
        assert settings['language_model'] == 'gpt-4o-mini'
        assert 'system_message' in settings
        assert settings['api_key'] == ''


class TestSettingsValidation:
    """Test settings validation functionality."""
    
    def test_validate_settings_valid(self):
        """Test validation of valid settings."""
        valid_settings = {
            'api_key': 'sk-1234567890abcdef1234567890abcdef',
            'audio_model': 'whisper-1',
            'language_model': 'gpt-4o-mini',
            'chunk_minutes': 5
        }
        
        is_valid, error_msg = validate_settings(valid_settings)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_settings_missing_api_key(self):
        """Test validation with missing API key."""
        invalid_settings = {
            'api_key': '',
            'audio_model': 'whisper-1',
            'language_model': 'gpt-4o-mini',
            'chunk_minutes': 5
        }
        
        is_valid, error_msg = validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "API key" in error_msg
    
    def test_validate_settings_invalid_chunk_minutes(self):
        """Test validation with invalid chunk minutes."""
        invalid_settings = {
            'api_key': 'sk-1234567890abcdef1234567890abcdef',
            'audio_model': 'whisper-1',
            'language_model': 'gpt-4o-mini',
            'chunk_minutes': 15  # Out of range (1-10)
        }
        
        is_valid, error_msg = validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Chunk duration" in error_msg
    
    def test_validate_settings_missing_model(self):
        """Test validation with missing model selection."""
        invalid_settings = {
            'api_key': 'sk-1234567890abcdef1234567890abcdef',
            'audio_model': '',
            'language_model': 'gpt-4o-mini',
            'chunk_minutes': 5
        }
        
        is_valid, error_msg = validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Audio model" in error_msg


class TestTranscriptFormatting:
    """Test transcript formatting functionality."""
    
    def test_format_transcript_for_display_basic(self):
        """Test basic transcript formatting."""
        transcript = "# 00:00:00 --> 00:01:00\nFirst segment\n\n# 00:01:00 --> 00:02:00\nSecond segment"
        
        formatted = format_transcript_for_display(transcript)
        
        assert '<span class="timestamp">' in formatted
        assert 'First segment' in formatted
        assert 'Second segment' in formatted
        assert '<br>' in formatted  # Newlines converted to breaks
    
    def test_format_transcript_for_display_empty(self):
        """Test formatting of empty transcript."""
        formatted = format_transcript_for_display("")
        
        assert formatted == ""
    
    def test_format_transcript_for_display_no_timestamps(self):
        """Test formatting transcript without timestamps."""
        transcript = "Plain text without timestamps"
        
        formatted = format_transcript_for_display(transcript)
        
        assert '<span class="timestamp">' not in formatted
        assert transcript in formatted


class TestDownloadFiles:
    """Test download file creation functionality."""
    
    @patch('src.app.datetime')
    @patch('src.app.os.path.exists')
    @patch('src.app.zipfile.ZipFile')
    def test_create_download_files_with_translation(self, mock_zipfile, mock_exists, mock_datetime):
        """Test creating download files with translation enabled."""
        mock_datetime.now().strftime.return_value = "2024-01-15"
        mock_exists.return_value = True
        
        settings = {'translation_enabled': True, 'default_translation_language': 'Japanese'}
        
        # Mock ZipFile context manager
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__ = MagicMock(return_value=mock_zip_instance)
        mock_zipfile.return_value.__exit__ = MagicMock(return_value=None)
        
        result = create_download_files("test_job_123", settings)
        
        expected_path = os.path.join("data", "2024-01-15", "test_job_123", "transcript.zip")
        assert result == expected_path
        
        # Verify files were added to zip
        assert mock_zip_instance.write.call_count >= 1
    
    @patch('src.app.datetime')
    @patch('src.app.os.path.exists')
    def test_create_download_files_without_translation(self, mock_exists, mock_datetime):
        """Test creating download files without translation."""
        mock_datetime.now().strftime.return_value = "2024-01-15"
        mock_exists.return_value = True
        
        settings = {'translation_enabled': False}
        
        result = create_download_files("test_job_123", settings)
        
        expected_path = os.path.join("data", "2024-01-15", "test_job_123", "transcript.txt")
        assert result == expected_path
    
    def test_create_download_files_no_job_id(self):
        """Test creating download files with no job ID."""
        with pytest.raises(Exception):  # Should raise gr.Error in actual app
            create_download_files("", {})


class TestJobHistory:
    """Test job history functionality."""
    
    @patch('src.app.os.path.exists')
    @patch('src.app.os.listdir')
    def test_get_job_history_empty(self, mock_listdir, mock_exists):
        """Test getting job history when data directory doesn't exist."""
        mock_exists.return_value = False
        
        jobs = get_job_history()
        
        assert jobs == []
    
    @patch('src.app.os.path.exists')
    @patch('src.app.os.listdir')
    @patch('src.app.os.path.isdir')
    @patch('builtins.open')
    def test_get_job_history_with_jobs(self, mock_open, mock_isdir, mock_listdir, mock_exists):
        """Test getting job history with existing jobs."""
        # Mock directory structure
        mock_exists.return_value = True
        mock_listdir.side_effect = [
            ["2024-01-15"],  # Date folders
            ["job_123"]      # Job folders
        ]
        mock_isdir.return_value = True
        
        # Mock metadata file
        mock_metadata = {
            "job_id": "job_123",
            "timestamp": "2024-01-15T10:30:00",
            "original_filename": "test.mp3",
            "file_info": {"duration_seconds": 300},
            "settings": {"default_language": "en"}
        }
        
        mock_open.return_value.__enter__ = MagicMock()
        mock_open.return_value.__exit__ = MagicMock()
        
        with patch('json.load', return_value=mock_metadata):
            jobs = get_job_history()
        
        assert len(jobs) == 1
        assert jobs[0][0] == "job_123"  # job_id
        assert jobs[0][2] == "test.mp3"  # filename
        assert jobs[0][5] == "Completed"  # status


class TestLoadJobTranscript:
    """Test job transcript loading functionality."""
    
    @patch('src.app.os.path.exists')
    @patch('src.app.os.listdir')
    @patch('src.app.os.path.isdir')
    @patch('builtins.open')
    def test_load_job_transcript_success(self, mock_open, mock_isdir, mock_listdir, mock_exists):
        """Test successful job transcript loading."""
        # Mock directory structure
        def side_effect_exists(path):
            return "job_123" in path
        
        mock_exists.side_effect = side_effect_exists
        mock_listdir.return_value = ["2024-01-15"]
        mock_isdir.return_value = True
        
        # Mock file reading
        def side_effect_open(path, mode='r', encoding=None):
            mock_file = MagicMock()
            if "transcript.txt" in path:
                mock_file.read.return_value = "Original transcript text"
            elif "transcript.ja.txt" in path:
                mock_file.read.return_value = "Translated transcript text"
            return mock_file.__enter__.return_value
        
        mock_open.side_effect = side_effect_open
        
        # Mock listdir for job directory
        with patch('src.app.os.listdir') as mock_job_listdir:
            mock_job_listdir.return_value = ["transcript.txt", "transcript.ja.txt", "metadata.json"]
            
            transcript, translation = load_job_transcript("job_123")
        
        assert transcript == "Original transcript text"
        assert translation == "Translated transcript text"
    
    def test_load_job_transcript_empty_job_id(self):
        """Test loading job transcript with empty job ID."""
        transcript, translation = load_job_transcript("")
        
        assert transcript == ""
        assert translation == ""
    
    @patch('src.app.os.path.exists')
    def test_load_job_transcript_nonexistent_job(self, mock_exists):
        """Test loading transcript for nonexistent job."""
        mock_exists.return_value = False
        
        transcript, translation = load_job_transcript("nonexistent_job")
        
        assert transcript == ""
        assert translation == ""


if __name__ == "__main__":
    pytest.main([__file__])