"""
Test chat functionality compatibility with integrated display feature.

Ensures that chat functionality continues to use the original transcript
as context, not the integrated display content.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from src.app import load_job_transcript, app_state
from src.handlers.chat_handler import ChatHandler, MockChatHandler
from src.file_manager import save_transcription_files, save_job_metadata


class TestChatIntegrationCompatibility:
    """Test chat functionality compatibility with integrated display."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.original_transcript = """# 00:00:00 --> 00:02:30
This is the original transcript content.
It contains the actual spoken words from the audio.

# 00:02:30 --> 00:05:00
More original content continues here.
This should be used as chat context."""
        
        self.translation = """# 00:00:00 --> 00:02:30
これは元の転写コンテンツです。
音声から実際に話された言葉が含まれています。

# 00:02:30 --> 00:05:00
より多くの元のコンテンツがここに続きます。
これはチャットコンテキストとして使用されるべきです。"""
        
        self.integrated_display = """# 00:00:00 --> 00:02:30

This is the original transcript content.
It contains the actual spoken words from the audio.

──────────────── Translation ────────────────

これは元の転写コンテンツです。
音声から実際に話された言葉が含まれています。


# 00:02:30 --> 00:05:00

More original content continues here.
This should be used as chat context.

──────────────── Translation ────────────────

より多くの元のコンテンツがここに続きます。
これはチャットコンテキストとして使用されるべきです。"""
        
        self.test_settings = {
            "api_key": "test-api-key",
            "language_model": "gpt-4o-mini",
            "system_message": "You are a helpful assistant."
        }
    
    def test_load_job_transcript_returns_separate_content(self):
        """Test that load_job_transcript returns display content and original transcript separately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock job directory with files
            job_id = "test-job-123"
            
            # Save files using the file manager
            save_transcription_files(
                job_dir=temp_dir,
                transcript=self.original_transcript,
                translation=self.translation,
                settings={"translation_enabled": True}
            )
            save_job_metadata(
                job_dir=temp_dir,
                job_id=job_id,
                original_filename="test_audio.mp3",
                file_info={"duration_seconds": 300.0, "size_bytes": 1024000},
                settings={"translation_enabled": True}
            )
            
            # Mock find_job_directory to return our temp directory
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Verify that display content is the integrated display
                assert "Translation" in display_content
                assert "これは元の転写コンテンツです" in display_content
                assert "This is the original transcript content" in display_content
                
                # Verify that original transcript is separate and clean
                assert original_transcript == self.original_transcript
                assert "Translation" not in original_transcript
                assert "これは元の転写コンテンツです" not in original_transcript
                
                # Verify translation is returned correctly
                assert translation == self.translation
    
    def test_app_state_uses_original_transcript_for_chat(self):
        """Test that app_state.current_transcript is set to original transcript, not display content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-job-456"
            
            # Save files
            save_transcription_files(
                job_dir=temp_dir,
                transcript=self.original_transcript,
                translation=self.translation,
                settings={"translation_enabled": True}
            )
            save_job_metadata(
                job_dir=temp_dir,
                job_id=job_id,
                original_filename="test_audio.mp3",
                file_info={"duration_seconds": 300.0, "size_bytes": 1024000},
                settings={"translation_enabled": True}
            )
            
            # Mock find_job_directory
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Simulate what happens in the app when loading a job
                app_state.current_job_id = job_id
                app_state.current_transcript = original_transcript  # This should be original, not display
                
                # Verify that current_transcript is the original transcript
                assert app_state.current_transcript == self.original_transcript
                assert "Translation" not in app_state.current_transcript
                assert "これは元の転写コンテンツです" not in app_state.current_transcript
    
    @patch('src.handlers.chat_handler.chat_with_context')
    def test_chat_handler_uses_original_transcript_context(self, mock_chat_with_context):
        """Test that ChatHandler uses original transcript as context, not integrated display."""
        mock_chat_with_context.return_value = "Response based on original transcript"
        
        # Set up chat handler
        chat_handler = ChatHandler()
        
        # Set context to original transcript (this is what should happen)
        chat_handler.set_context(self.original_transcript)
        
        # Handle a message
        message = "What is this transcript about?"
        history = []
        
        new_history, empty_input = chat_handler.handle_message(message, history, self.test_settings)
        
        # Verify that chat_with_context was called with original transcript
        mock_chat_with_context.assert_called_once_with(
            api_key=self.test_settings["api_key"],
            model=self.test_settings["language_model"],
            question=message,
            context_text=self.original_transcript,  # Should be original, not integrated
            system_message=self.test_settings.get("system_message", ""),
            temperature=0.7
        )
        
        # Verify response
        assert len(new_history) == 2
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == message
        assert new_history[1]["role"] == "assistant"
        assert new_history[1]["content"] == "Response based on original transcript"
    
    def test_mock_chat_handler_compatibility(self):
        """Test that MockChatHandler works correctly with original transcript context."""
        mock_handler = MockChatHandler()
        
        # Set context to original transcript
        mock_handler.set_context(self.original_transcript)
        
        # Verify context is set correctly
        assert mock_handler.current_context == self.original_transcript
        
        # Handle a message
        message = "summary"
        history = []
        
        new_history, empty_input = mock_handler.handle_message(message, history, self.test_settings)
        
        # Verify mock response includes context information
        assert len(new_history) == 1
        assert new_history[0][0] == message
        response = new_history[0][1]
        assert "mock summary" in response.lower()
        assert "(This response was generated using the transcript as context)" in response
    
    def test_chat_context_not_contaminated_by_translation(self):
        """Test that chat context doesn't include translation content."""
        chat_handler = ChatHandler()
        
        # Set context to original transcript
        chat_handler.set_context(self.original_transcript)
        
        # Verify context doesn't contain translation markers or Japanese text
        assert "Translation" not in chat_handler.current_context
        assert "──────────────" not in chat_handler.current_context
        assert "これは" not in chat_handler.current_context
        assert "です。" not in chat_handler.current_context
        
        # Verify it contains the original English content
        assert "This is the original transcript content" in chat_handler.current_context
        assert "actual spoken words from the audio" in chat_handler.current_context
    
    def test_integrated_display_vs_chat_context_separation(self):
        """Test that integrated display and chat context are properly separated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-separation-789"
            
            # Save files
            save_transcription_files(
                job_dir=temp_dir,
                transcript=self.original_transcript,
                translation=self.translation,
                settings={"translation_enabled": True}
            )
            save_job_metadata(
                job_dir=temp_dir,
                job_id=job_id,
                original_filename="test_audio.mp3",
                file_info={"duration_seconds": 300.0, "size_bytes": 1024000},
                settings={"translation_enabled": True}
            )
            
            # Mock find_job_directory
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Verify display content has both languages
                assert "This is the original transcript content" in display_content
                assert "これは元の転写コンテンツです" in display_content
                assert "Translation" in display_content
                
                # Verify original transcript has only English
                assert "This is the original transcript content" in original_transcript
                assert "これは元の転写コンテンツです" not in original_transcript
                assert "Translation" not in original_transcript
                
                # Verify they are different
                assert display_content != original_transcript
                assert len(display_content) > len(original_transcript)