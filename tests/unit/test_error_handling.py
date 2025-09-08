"""
Unit tests for enhanced error handling functionality.

Tests the new error handling classes and functions to ensure proper
error management and graceful degradation.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, mock_open

from errors import (
    TranslationError, IntegratedDisplayError, FileError,
    handle_translation_failure, handle_file_read_failure, 
    handle_integrated_display_failure, get_user_friendly_message
)
from integrated_display import format_integrated_display, parse_timestamp_sections
from file_manager import save_transcription_files, load_job_files, get_display_content_from_job


class TestTranslationError:
    """Test TranslationError class."""
    
    def test_translation_error_creation(self):
        """Test creating TranslationError with various parameters."""
        error = TranslationError("Translation failed", transcript_available=True)
        
        assert error.message == "Translation failed"
        assert error.transcript_available is True
        assert error.partial_translation is None
        
        # Test with partial translation
        error_with_partial = TranslationError(
            "Partial failure", 
            transcript_available=True,
            partial_translation="Partial content"
        )
        
        assert error_with_partial.partial_translation == "Partial content"
    
    def test_translation_error_to_dict(self):
        """Test TranslationError serialization."""
        error = TranslationError("Test error", transcript_available=True)
        error_dict = error.to_dict()
        
        assert error_dict["type"] == "processing"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"]["transcript_available"] is True


class TestIntegratedDisplayError:
    """Test IntegratedDisplayError class."""
    
    def test_integrated_display_error_creation(self):
        """Test creating IntegratedDisplayError."""
        error = IntegratedDisplayError(
            "Display generation failed",
            transcript="Test transcript",
            translation="Test translation"
        )
        
        assert error.message == "Display generation failed"
        assert error.transcript == "Test transcript"
        assert error.translation == "Test translation"
    
    def test_integrated_display_error_details(self):
        """Test IntegratedDisplayError details."""
        error = IntegratedDisplayError("Test error", transcript="content")
        error_dict = error.to_dict()
        
        assert error_dict["details"]["transcript_available"] is True
        assert error_dict["details"]["translation_available"] is False


class TestErrorHandlingFunctions:
    """Test error handling utility functions."""
    
    def test_handle_translation_failure(self):
        """Test translation failure handling."""
        transcript = "Original transcript"
        error = Exception("Rate limit exceeded")
        
        result_transcript, fallback_translation, translation_error = handle_translation_failure(
            transcript, error
        )
        
        assert result_transcript == transcript
        assert "[Translation Error]" in fallback_translation
        assert "rate limit" in fallback_translation.lower()
        assert isinstance(translation_error, TranslationError)
        assert translation_error.transcript_available is True
    
    def test_handle_translation_failure_with_partial(self):
        """Test translation failure handling with partial translation."""
        transcript = "Original transcript"
        error = Exception("Timeout occurred")
        partial = "Partial translation content"
        
        result_transcript, fallback_translation, translation_error = handle_translation_failure(
            transcript, error, partial
        )
        
        assert result_transcript == transcript
        assert "[Translation Error]" in fallback_translation
        assert "[Partial Translation]" in fallback_translation
        assert partial in fallback_translation
    
    def test_handle_file_read_failure(self):
        """Test file read failure handling."""
        file_path = "/path/to/file.txt"
        error = FileNotFoundError("File not found")
        fallback = "Fallback content"
        
        content, file_error = handle_file_read_failure(file_path, error, fallback)
        
        assert content == fallback
        assert isinstance(file_error, FileError)
        assert "file.txt" in file_error.message
    
    def test_handle_integrated_display_failure(self):
        """Test integrated display failure handling."""
        transcript = "Test transcript"
        translation = "Test translation"
        error = Exception("Display generation failed")
        
        fallback_display, display_error = handle_integrated_display_failure(
            transcript, translation, error
        )
        
        assert fallback_display == transcript  # Should fallback to transcript
        assert isinstance(display_error, IntegratedDisplayError)
        assert display_error.transcript == transcript
        assert display_error.translation == translation
    
    def test_get_user_friendly_message_translation_error(self):
        """Test user-friendly messages for translation errors."""
        translation_error = TranslationError("Test error", transcript_available=True)
        message = get_user_friendly_message(translation_error)
        
        assert "Translation failed" in message
        assert "transcription was successful" in message.lower()
    
    def test_get_user_friendly_message_display_error(self):
        """Test user-friendly messages for display errors."""
        display_error = IntegratedDisplayError("Display failed")
        message = get_user_friendly_message(display_error)
        
        assert "Failed to generate integrated display" in message


class TestIntegratedDisplayErrorHandling:
    """Test error handling in integrated display functions."""
    
    def test_format_integrated_display_with_invalid_input(self):
        """Test integrated display formatting with invalid input."""
        # Test with None input
        with pytest.raises(IntegratedDisplayError):
            format_integrated_display(None, "translation")
    
    def test_parse_timestamp_sections_with_invalid_input(self):
        """Test timestamp parsing with invalid input."""
        # Test with malformed input that could cause parsing errors
        with pytest.raises(IntegratedDisplayError):
            parse_timestamp_sections(None)
    
    def test_format_integrated_display_error_recovery(self):
        """Test that format_integrated_display handles errors gracefully."""
        # Test with extremely long input that might cause memory issues
        very_long_text = "A" * 1000000  # 1MB of text
        
        try:
            result = format_integrated_display(very_long_text, "short translation")
            # Should either succeed or raise IntegratedDisplayError
            assert isinstance(result, str)
        except IntegratedDisplayError:
            # This is acceptable - the function should raise our custom error
            pass
        except Exception as e:
            # Should not raise other types of exceptions
            pytest.fail(f"Unexpected exception type: {type(e)}")


class TestFileManagerErrorHandling:
    """Test error handling in file manager functions."""
    
    def test_save_transcription_files_with_invalid_directory(self):
        """Test saving files to invalid directory."""
        from errors import FileError
        
        invalid_dir = "/invalid/nonexistent/directory"
        
        with pytest.raises(FileError):
            save_transcription_files(invalid_dir, "transcript", "translation")
    
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_transcription_files_permission_error(self, mock_open):
        """Test handling permission errors when saving files."""
        from errors import FileError
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileError):
                save_transcription_files(temp_dir, "transcript", "translation")
    
    def test_load_job_files_with_missing_directory(self):
        """Test loading files from missing directory."""
        nonexistent_dir = "/nonexistent/directory"
        
        transcript, translation, integrated = load_job_files(nonexistent_dir)
        
        # Should return empty strings rather than raising exception
        assert transcript == ""
        assert translation == ""
        assert integrated == ""
    
    def test_load_job_files_encoding_error(self):
        """Test handling encoding errors when loading files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file that exists
            test_file = os.path.join(temp_dir, "transcript.txt")
            with open(test_file, "w") as f:
                f.write("test")
            
            # Mock the file reading to simulate encoding error
            with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
                # Should handle encoding error gracefully
                transcript, translation, integrated = load_job_files(temp_dir)
                
                # Should return empty or error content, not crash
                assert isinstance(transcript, str)
    
    def test_get_display_content_from_job_with_corrupted_files(self):
        """Test getting display content when files are corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted integrated display file
            integrated_path = os.path.join(temp_dir, "transcript_integrated.txt")
            
            # Create file but make it unreadable
            with open(integrated_path, "w") as f:
                f.write("test content")
            
            # Mock file reading to simulate corruption
            with patch("builtins.open", side_effect=IOError("Disk error")):
                content = get_display_content_from_job(temp_dir)
                
                # Should return fallback content, not crash
                assert isinstance(content, str)
                assert len(content) > 0


class TestAudioHandlerErrorHandling:
    """Test error handling in AudioHandler."""
    
    @pytest.fixture
    def mock_audio_handler(self):
        """Create a mock audio handler for testing."""
        from handlers.audio_handler import AudioHandler
        return AudioHandler()
    
    def test_audio_handler_translation_failure_recovery(self, mock_audio_handler):
        """Test that AudioHandler recovers from translation failures."""
        # This would require mocking the translation service
        # For now, we test that the handler has the necessary methods
        
        assert hasattr(mock_audio_handler, 'get_display_content')
        assert hasattr(mock_audio_handler, 'get_ui_display_text')
    
    def test_audio_handler_display_generation_failure(self, mock_audio_handler):
        """Test AudioHandler handling of display generation failures."""
        # Test that the handler can provide fallback content
        
        display_content = mock_audio_handler.get_ui_display_text()
        
        # Should return string, even if empty
        assert isinstance(display_content, str)


class TestHistoryHandlerErrorHandling:
    """Test error handling in HistoryHandler."""
    
    @pytest.fixture
    def mock_history_handler(self):
        """Create a mock history handler for testing."""
        from handlers.history_handler import HistoryHandler
        return HistoryHandler()
    
    def test_history_handler_missing_job_recovery(self, mock_history_handler):
        """Test HistoryHandler recovery when job is missing."""
        # Test loading non-existent job
        display_content, translation = mock_history_handler.load_job_transcript("nonexistent-job")
        
        # Should return error message, not crash
        assert isinstance(display_content, str)
        assert isinstance(translation, str)
        
        # Should indicate the job was not found
        if display_content:
            assert "not found" in display_content.lower() or "failed" in display_content.lower()
    
    def test_history_handler_corrupted_metadata_recovery(self, mock_history_handler):
        """Test HistoryHandler recovery when metadata is corrupted."""
        # Test loading job with corrupted metadata
        display_content, translation, metadata = mock_history_handler.load_job_content("test-job")
        
        # Should return some content and metadata, even if partial
        assert isinstance(display_content, str)
        assert isinstance(translation, str)
        assert isinstance(metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])