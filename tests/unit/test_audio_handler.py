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


class TestAudioHandlerIntegratedDisplay:
    """Test suite for integrated display functionality in AudioHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = AudioHandler()
        self.mock_handler = MockAudioHandler()
        self.test_config = TestConfig()
    
    def test_get_display_content_empty(self):
        """Test getting display content when no processing has occurred."""
        transcript, translation, available = self.handler.get_display_content()
        
        assert transcript == ""
        assert translation == ""
        assert available is False
    
    def test_get_display_content_with_data(self):
        """Test getting display content after setting data."""
        self.handler.current_transcript = "Test transcript"
        self.handler.current_translation = "Test translation"
        
        transcript, translation, available = self.handler.get_display_content()
        
        assert transcript == "Test transcript"
        assert translation == "Test translation"
        assert available is True
    
    def test_get_ui_display_text_transcript_only(self):
        """Test getting UI display text with transcript only."""
        self.handler.current_transcript = "# 00:00:00 --> 00:02:30\nTest transcript"
        self.handler.current_translation = ""
        
        display_text = self.handler.get_ui_display_text()
        
        assert display_text == "# 00:00:00 --> 00:02:30\nTest transcript"
    
    def test_get_ui_display_text_with_translation(self):
        """Test getting UI display text with translation."""
        self.handler.current_transcript = "# 00:00:00 --> 00:02:30\nTest transcript"
        self.handler.current_translation = "# 00:00:00 --> 00:02:30\nテスト転写"
        
        display_text = self.handler.get_ui_display_text()
        
        # Should contain both transcript and translation
        assert "Test transcript" in display_text
        assert "テスト転写" in display_text
        assert "Translation" in display_text
        # Timestamp should appear only once
        assert display_text.count("# 00:00:00 --> 00:02:30") == 1
    
    @patch('handlers.audio_handler.validate_audio_file')
    @patch('handlers.audio_handler.transcribe_chunked')
    @patch('handlers.audio_handler.create_job_directory')
    @patch('handlers.audio_handler.save_transcription_files')
    @patch('handlers.audio_handler.save_job_metadata')
    @pytest.mark.asyncio
    async def test_process_audio_returns_display_text(self, mock_save_metadata, mock_save_files, 
                                                    mock_create_dir, mock_transcribe, mock_validate_audio):
        """Test that process_audio returns display_text in result."""
        # Mock setup
        mock_validate_audio.return_value = (True, None, {"size_mb": 2.5, "duration_seconds": 150.0})
        mock_create_dir.return_value = "/tmp/test_job"
        mock_save_files.return_value = {"transcript": "transcript.txt"}
        mock_save_metadata.return_value = "metadata.json"
        
        mock_transcript_result = Mock()
        mock_transcript_result.text = "# 00:00:00 --> 00:02:30\nTest transcript"
        mock_transcript_result.word_count = 100
        mock_transcript_result.total_duration = 60.0
        mock_transcript_result.processing_time = 5.0
        mock_transcribe.return_value = mock_transcript_result
        
        settings = self.test_config.get_test_settings()
        settings["translation_enabled"] = False
        
        result = await self.handler.process_audio(
            "/test/audio.mp3",
            settings,
            None
        )
        
        assert hasattr(result, 'display_text')
        assert result.display_text == "# 00:00:00 --> 00:02:30\nTest transcript"
    
    @patch('handlers.audio_handler.validate_audio_file')
    @patch('handlers.audio_handler.transcribe_chunked')
    @patch('handlers.audio_handler.translate_transcript_full')
    @patch('handlers.audio_handler.create_job_directory')
    @patch('handlers.audio_handler.save_transcription_files')
    @patch('handlers.audio_handler.save_job_metadata')
    @pytest.mark.asyncio
    async def test_process_audio_with_translation_display_text(self, mock_save_metadata, mock_save_files,
                                                             mock_create_dir, mock_translate, mock_transcribe, mock_validate_audio):
        """Test that process_audio returns integrated display text with translation."""
        # Mock setup
        mock_validate_audio.return_value = (True, None, {"size_mb": 2.5, "duration_seconds": 150.0})
        mock_create_dir.return_value = "/tmp/test_job"
        mock_save_files.return_value = {"transcript": "transcript.txt", "translation": "transcript.ja.txt"}
        mock_save_metadata.return_value = "metadata.json"
        
        mock_transcript_result = Mock()
        mock_transcript_result.text = "# 00:00:00 --> 00:02:30\nTest transcript"
        mock_transcript_result.word_count = 100
        mock_transcript_result.total_duration = 60.0
        mock_transcript_result.processing_time = 5.0
        mock_transcribe.return_value = mock_transcript_result
        
        mock_translation_result = Mock()
        mock_translation_result.translated_text = "# 00:00:00 --> 00:02:30\nテスト転写"
        mock_translate.return_value = mock_translation_result
        
        settings = self.test_config.get_test_settings()
        settings["translation_enabled"] = True
        settings["default_translation_language"] = "Japanese"
        
        result = await self.handler.process_audio(
            "/test/audio.mp3",
            settings,
            None
        )
        
        assert hasattr(result, 'display_text')
        assert "Test transcript" in result.display_text
        assert "テスト転写" in result.display_text
        assert "Translation" in result.display_text
        # Timestamp should appear only once in integrated display
        assert result.display_text.count("# 00:00:00 --> 00:02:30") == 1
    
    def test_mock_handler_display_methods(self):
        """Test that MockAudioHandler has the same display methods."""
        # Test get_display_content
        transcript, translation, available = self.mock_handler.get_display_content()
        
        assert transcript == "# 00:00:00 --> 00:02:30\nMock transcript content"
        assert translation == "# 00:00:00 --> 00:02:30\nMock translation content"
        assert available is True
        
        # Test get_ui_display_text
        display_text = self.mock_handler.get_ui_display_text()
        
        assert "Mock transcript content" in display_text
        assert "Mock translation content" in display_text
        assert "Translation" in display_text
    
    @pytest.mark.asyncio
    async def test_mock_handler_returns_display_text(self):
        """Test that MockAudioHandler returns display_text in result."""
        settings = self.test_config.get_test_settings()
        settings["translation_enabled"] = True
        
        result = await self.mock_handler.process_audio(
            "/test/audio.mp3",
            settings,
            None
        )
        
        assert hasattr(result, 'display_text')
        assert len(result.display_text) > 0
        assert "mock transcript" in result.display_text.lower()
        assert "Translation" in result.display_text


class TestAudioHandlerIntegratedDisplayGeneration:
    """Test suite for integrated display generation functionality in AudioHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = AudioHandler()
        self.mock_handler = MockAudioHandler()
        self.test_config = TestConfig()
    
    def test_format_for_display_transcript_only(self):
        """Test formatting transcript for display without translation."""
        test_text = "# 00:00:00 --> 00:02:30\nTest transcript content"
        
        # Set current transcript
        self.handler.current_transcript = test_text
        self.handler.current_translation = ""
        
        display_text = self.handler.get_ui_display_text()
        
        assert display_text == test_text
        assert "Translation" not in display_text
    
    def test_format_for_display_with_translation(self):
        """Test formatting transcript and translation for integrated display."""
        transcript = "# 00:00:00 --> 00:02:30\nTest transcript content"
        translation = "# 00:00:00 --> 00:02:30\nテスト転写コンテンツ"
        
        # Set current content
        self.handler.current_transcript = transcript
        self.handler.current_translation = translation
        
        display_text = self.handler.get_ui_display_text()
        
        # Should contain both transcript and translation
        assert "Test transcript content" in display_text
        assert "テスト転写コンテンツ" in display_text
        assert "Translation" in display_text
        # Timestamp should appear only once
        assert display_text.count("# 00:00:00 --> 00:02:30") == 1
    
    def test_format_for_display_multiple_sections(self):
        """Test formatting multiple timestamp sections for integrated display."""
        transcript = """# 00:00:00 --> 00:02:30
First section content.

# 00:02:30 --> 00:05:00
Second section content."""
        
        translation = """# 00:00:00 --> 00:02:30
最初のセクションコンテンツ。

# 00:02:30 --> 00:05:00
2番目のセクションコンテンツ。"""
        
        # Set current content
        self.handler.current_transcript = transcript
        self.handler.current_translation = translation
        
        display_text = self.handler.get_ui_display_text()
        
        # Should contain all content
        assert "First section content." in display_text
        assert "Second section content." in display_text
        assert "最初のセクションコンテンツ。" in display_text
        assert "2番目のセクションコンテンツ。" in display_text
        
        # Should have translation separators for each section
        assert display_text.count("Translation") == 2
        
        # Timestamps should appear only once each
        assert display_text.count("# 00:00:00 --> 00:02:30") == 1
        assert display_text.count("# 00:02:30 --> 00:05:00") == 1
    
    def test_get_display_content_state_management(self):
        """Test that display content reflects current handler state."""
        # Initially empty
        transcript, translation, available = self.handler.get_display_content()
        assert transcript == ""
        assert translation == ""
        assert available is False
        
        # After setting transcript only
        self.handler.current_transcript = "Test transcript"
        transcript, translation, available = self.handler.get_display_content()
        assert transcript == "Test transcript"
        assert translation == ""
        assert available is False
        
        # After setting both transcript and translation
        self.handler.current_translation = "Test translation"
        transcript, translation, available = self.handler.get_display_content()
        assert transcript == "Test transcript"
        assert translation == "Test translation"
        assert available is True
    
    def test_mock_handler_integrated_display_generation(self):
        """Test that MockAudioHandler generates proper integrated display."""
        # Test get_display_content
        transcript, translation, available = self.mock_handler.get_display_content()
        
        assert "Mock transcript content" in transcript
        assert "Mock translation content" in translation
        assert available is True
        
        # Test get_ui_display_text
        display_text = self.mock_handler.get_ui_display_text()
        
        assert "Mock transcript content" in display_text
        assert "Mock translation content" in display_text
        assert "Translation" in display_text
        # Timestamp should appear only once in integrated display
        assert display_text.count("# 00:00:00 --> 00:02:30") == 1
    
    @patch('handlers.audio_handler.validate_audio_file')
    @patch('handlers.audio_handler.transcribe_chunked')
    @patch('handlers.audio_handler.translate_transcript_full')
    @patch('handlers.audio_handler.create_job_directory')
    @patch('handlers.audio_handler.save_transcription_files')
    @patch('handlers.audio_handler.save_job_metadata')
    @pytest.mark.asyncio
    async def test_process_audio_generates_integrated_display(self, mock_save_metadata, mock_save_files,
                                                            mock_create_dir, mock_translate, mock_transcribe, mock_validate_audio):
        """Test that process_audio generates proper integrated display text."""
        # Mock setup
        mock_validate_audio.return_value = (True, None, {"size_mb": 2.5, "duration_seconds": 150.0})
        mock_create_dir.return_value = "/tmp/test_job"
        mock_save_files.return_value = {"transcript": "transcript.txt", "translation": "transcript.ja.txt"}
        mock_save_metadata.return_value = "metadata.json"
        
        # Mock transcription with multiple sections
        mock_transcript_result = Mock()
        mock_transcript_result.text = """# 00:00:00 --> 00:02:30
First section of transcript.

# 00:02:30 --> 00:05:00
Second section of transcript."""
        mock_transcript_result.word_count = 100
        mock_transcript_result.total_duration = 60.0
        mock_transcript_result.processing_time = 5.0
        mock_transcribe.return_value = mock_transcript_result
        
        # Mock translation with multiple sections
        mock_translation_result = Mock()
        mock_translation_result.translated_text = """# 00:00:00 --> 00:02:30
転写の最初のセクション。

# 00:02:30 --> 00:05:00
転写の2番目のセクション。"""
        mock_translate.return_value = mock_translation_result
        
        settings = self.test_config.get_test_settings()
        settings["translation_enabled"] = True
        settings["default_translation_language"] = "Japanese"
        
        result = await self.handler.process_audio(
            "/test/audio.mp3",
            settings,
            None
        )
        
        # Verify integrated display is generated correctly
        assert hasattr(result, 'display_text')
        assert "First section of transcript." in result.display_text
        assert "Second section of transcript." in result.display_text
        assert "転写の最初のセクション。" in result.display_text
        assert "転写の2番目のセクション。" in result.display_text
        
        # Verify timestamps appear only once each
        assert result.display_text.count("# 00:00:00 --> 00:02:30") == 1
        assert result.display_text.count("# 00:02:30 --> 00:05:00") == 1
        
        # Verify translation separators
        assert result.display_text.count("Translation") == 2
    
    @pytest.mark.asyncio
    async def test_mock_process_audio_integrated_display_consistency(self):
        """Test that MockAudioHandler process_audio generates consistent integrated display."""
        settings = self.test_config.get_test_settings()
        settings["translation_enabled"] = True
        settings["default_translation_language"] = "Japanese"
        
        result = await self.mock_handler.process_audio(
            "/test/audio.mp3",
            settings,
            None
        )
        
        # Verify display_text matches what get_ui_display_text would return
        expected_display = self.mock_handler.get_ui_display_text()
        
        # Both should contain the same integrated display elements
        assert "mock transcript" in result.display_text.lower()
        assert "Translation" in result.display_text
        assert result.display_text.count("# 00:00:00 --> 00:02:30") == 1