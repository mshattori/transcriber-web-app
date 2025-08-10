"""
Unit tests for AudioHandler with test_audio.mp3.

Tests both real and mock implementations of audio processing logic.
"""

import os
import tempfile
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from handlers.audio_handler import AudioHandler, MockAudioHandler, ProcessingResult
from config.test_config import TestConfig


class TestAudioHandler:
    """Test suite for real AudioHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = AudioHandler()
        self.test_config = TestConfig()
        self.test_audio_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_audio.mp3'
        )
        self.test_settings = self.test_config.get_test_settings()
    
    def test_validate_audio_with_test_file(self):
        """Test audio validation with actual test file."""
        if not os.path.exists(self.test_audio_path):
            pytest.skip("test_audio.mp3 not found")
        
        is_valid, error_msg, file_info = self.handler.validate_audio(self.test_audio_path)
        
        assert is_valid is True
        assert error_msg is None
        assert isinstance(file_info, dict)
        assert 'size_mb' in file_info
        assert 'duration_seconds' in file_info
        assert file_info['size_mb'] > 0
        assert file_info['duration_seconds'] > 0
    
    def test_validate_audio_nonexistent_file(self):
        """Test audio validation with non-existent file."""
        is_valid, error_msg, file_info = self.handler.validate_audio("/nonexistent/file.mp3")
        
        assert is_valid is False
        assert error_msg is not None
        assert isinstance(file_info, dict)
    
    def test_validate_settings_valid(self):
        """Test settings validation with valid settings."""
        is_valid, error_msg = self.handler.validate_settings(self.test_settings)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_settings_missing_api_key(self):
        """Test settings validation with missing API key."""
        invalid_settings = self.test_settings.copy()
        invalid_settings["api_key"] = ""
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "API key" in error_msg
    
    def test_validate_settings_invalid_chunk_duration(self):
        """Test settings validation with invalid chunk duration."""
        invalid_settings = self.test_settings.copy()
        invalid_settings["chunk_minutes"] = 15  # Too large
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Chunk duration" in error_msg
    
    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        estimates = self.handler.estimate_processing_time(5.0, 3)
        
        assert isinstance(estimates, dict)
        assert 'estimated_chunks' in estimates
        assert 'total_time_seconds' in estimates
        assert estimates['estimated_chunks'] > 0
        assert estimates['total_time_seconds'] > 0
    
    @patch('handlers.audio_handler.transcribe_chunked')
    @patch('handlers.audio_handler.create_job_directory')
    @patch('handlers.audio_handler.safe_execute')
    async def test_process_audio_transcription_only(self, mock_safe_execute, mock_create_dir, mock_transcribe):
        """Test audio processing with transcription only (mocked APIs)."""
        if not os.path.exists(self.test_audio_path):
            pytest.skip("test_audio.mp3 not found")
        
        # Mock returns
        mock_create_dir.return_value = "/tmp/test_job"
        mock_safe_execute.return_value = None
        
        # Mock transcription result
        mock_transcript_result = Mock()
        mock_transcript_result.text = "Test transcription text"
        mock_transcript_result.word_count = 100
        mock_transcript_result.total_duration = 60.0
        mock_transcript_result.processing_time = 5.0
        mock_transcribe.return_value = mock_transcript_result
        
        # Settings without translation
        settings = self.test_settings.copy()
        settings["translation_enabled"] = False
        
        # Mock progress callback
        progress_callback = Mock()
        
        result = await self.handler.process_audio(
            self.test_audio_path,
            settings,
            progress_callback
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.transcript == "Test transcription text"
        assert result.translation == ""
        assert len(result.job_id) == 8
        assert result.settings_used == settings
        
        # Verify calls
        mock_transcribe.assert_called_once()
        progress_callback.assert_called()
    
    @patch('handlers.audio_handler.transcribe_chunked')
    @patch('handlers.audio_handler.translate_transcript_full')
    @patch('handlers.audio_handler.create_job_directory')
    @patch('handlers.audio_handler.safe_execute')
    async def test_process_audio_with_translation(self, mock_safe_execute, mock_create_dir, 
                                                mock_translate, mock_transcribe):
        """Test audio processing with transcription and translation (mocked APIs)."""
        if not os.path.exists(self.test_audio_path):
            pytest.skip("test_audio.mp3 not found")
        
        # Mock returns
        mock_create_dir.return_value = "/tmp/test_job"
        mock_safe_execute.return_value = None
        
        # Mock transcription result
        mock_transcript_result = Mock()
        mock_transcript_result.text = "Test transcription text"
        mock_transcript_result.word_count = 100
        mock_transcript_result.total_duration = 60.0
        mock_transcript_result.processing_time = 5.0
        mock_transcribe.return_value = mock_transcript_result
        
        # Mock translation result
        mock_translation_result = Mock()
        mock_translation_result.translated_text = "テスト転写テキスト"
        mock_translate.return_value = mock_translation_result
        
        # Settings with translation
        settings = self.test_settings.copy()
        settings["translation_enabled"] = True
        settings["default_translation_language"] = "Japanese"
        
        result = await self.handler.process_audio(
            self.test_audio_path,
            settings,
            None
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.transcript == "Test transcription text"
        assert result.translation == "テスト転写テキスト"
        assert len(result.job_id) == 8
        
        # Verify both transcription and translation were called
        mock_transcribe.assert_called_once()
        mock_translate.assert_called_once()


class TestMockAudioHandler:
    """Test suite for MockAudioHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = MockAudioHandler()
        self.test_config = TestConfig()
        self.test_audio_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_audio.mp3'
        )
        self.test_settings = self.test_config.get_test_settings()
    
    def test_validate_audio_always_valid(self):
        """Test that mock validation always returns valid."""
        is_valid, error_msg, file_info = self.handler.validate_audio(self.test_audio_path)
        
        assert is_valid is True
        assert error_msg is None
        assert isinstance(file_info, dict)
        assert file_info['size_mb'] > 0
        assert file_info['duration_seconds'] > 0
    
    def test_validate_audio_nonexistent_file(self):
        """Test that mock validation returns valid even for non-existent files."""
        is_valid, error_msg, file_info = self.handler.validate_audio("/nonexistent/file.mp3")
        
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_settings_always_valid(self):
        """Test that mock settings validation always returns valid."""
        is_valid, error_msg = self.handler.validate_settings(self.test_settings)
        
        assert is_valid is True
        assert error_msg == ""
        
        # Even with invalid settings
        invalid_settings = {"api_key": "", "chunk_minutes": 100}
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_estimate_processing_time(self):
        """Test mock processing time estimation."""
        estimates = self.handler.estimate_processing_time(5.0, 3)
        
        assert isinstance(estimates, dict)
        assert estimates['estimated_chunks'] == 2
        assert estimates['total_time_seconds'] == 6.0
    
    async def test_process_audio_transcription_only(self):
        """Test mock audio processing with transcription only."""
        settings = self.test_settings.copy()
        settings["translation_enabled"] = False
        
        progress_callback = Mock()
        
        result = await self.handler.process_audio(
            self.test_audio_path,
            settings,
            progress_callback
        )
        
        assert isinstance(result, ProcessingResult)
        assert "mock transcript" in result.transcript.lower()
        assert result.translation == ""
        assert result.job_id == "mock-job-123"
        assert result.settings_used == settings
        
        # Verify progress callbacks were made
        progress_callback.assert_called()
    
    async def test_process_audio_with_translation(self):
        """Test mock audio processing with translation."""
        settings = self.test_settings.copy()
        settings["translation_enabled"] = True
        settings["default_translation_language"] = "Japanese"
        
        result = await self.handler.process_audio(
            self.test_audio_path,
            settings,
            None
        )
        
        assert isinstance(result, ProcessingResult)
        assert "mock transcript" in result.transcript.lower()
        assert len(result.translation) > 0
        assert "モック" in result.translation or "テスト" in result.translation
        assert result.job_id == "mock-job-123"
    
    async def test_process_audio_realistic_timing(self):
        """Test that mock processing has realistic timing (not instant)."""
        import time
        
        start_time = time.time()
        
        result = await self.handler.process_audio(
            self.test_audio_path,
            self.test_settings,
            None
        )
        
        elapsed_time = time.time() - start_time
        
        # Should take at least some time due to asyncio.sleep calls
        assert elapsed_time >= 0.1  # At least 100ms
        assert isinstance(result, ProcessingResult)


class TestAudioHandlerIntegration:
    """Integration tests comparing real and mock handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.real_handler = AudioHandler()
        self.mock_handler = MockAudioHandler()
        self.test_config = TestConfig()
        self.test_audio_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_audio.mp3'
        )
    
    def test_handler_interface_compatibility(self):
        """Test that real and mock handlers have compatible interfaces."""
        # Both should have the same methods
        real_methods = [method for method in dir(self.real_handler) if not method.startswith('_')]
        mock_methods = [method for method in dir(self.mock_handler) if not method.startswith('_')]
        
        assert set(real_methods) == set(mock_methods)
    
    async def test_both_handlers_return_same_result_type(self):
        """Test that both handlers return the same result type."""
        settings = self.test_config.get_test_settings()
        
        # Mock handler (always works)
        mock_result = await self.mock_handler.process_audio(
            self.test_audio_path,
            settings,
            None
        )
        
        assert isinstance(mock_result, ProcessingResult)
        
        # Real handler interface test (mocked to avoid API calls)
        with patch('handlers.audio_handler.transcribe_chunked') as mock_transcribe, \
             patch('handlers.audio_handler.create_job_directory') as mock_create_dir, \
             patch('handlers.audio_handler.safe_execute'):
            
            mock_create_dir.return_value = "/tmp/test_job"
            mock_transcript_result = Mock()
            mock_transcript_result.text = "Test"
            mock_transcript_result.word_count = 1
            mock_transcript_result.total_duration = 1.0
            mock_transcript_result.processing_time = 1.0
            mock_transcribe.return_value = mock_transcript_result
            
            if os.path.exists(self.test_audio_path):
                real_result = await self.real_handler.process_audio(
                    self.test_audio_path,
                    settings,
                    None
                )
                
                assert isinstance(real_result, ProcessingResult)
                assert type(real_result) == type(mock_result)