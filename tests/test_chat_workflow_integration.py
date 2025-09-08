"""
Integration test for chat workflow with integrated display feature.

Tests the complete workflow: load job -> set chat context -> handle message
"""

import os
import tempfile
from unittest.mock import patch
import pytest

from src.app import load_job_transcript, app_state
from src.handlers.chat_handler import ChatHandler, MockChatHandler
from src.file_manager import save_transcription_files, save_job_metadata


class TestChatWorkflowIntegration:
    """Test complete chat workflow integration."""
    
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
        
        self.test_settings = {
            "api_key": "test-api-key",
            "language_model": "gpt-4o-mini",
            "system_message": "You are a helpful assistant."
        }
    
    def test_complete_workflow_with_mock_handler(self):
        """Test complete workflow: load job -> set context -> chat with MockChatHandler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-workflow-123"
            
            # Step 1: Save files (simulating completed transcription job)
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
            
            # Step 2: Load job (simulating history tab selection)
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Step 3: Update app state (simulating what happens in the app)
                app_state.current_job_id = job_id
                app_state.current_transcript = original_transcript  # Should be original, not display
                
                # Step 4: Initialize chat handler and set context
                chat_handler = MockChatHandler()
                chat_handler.set_context(app_state.current_transcript)
                
                # Step 5: Verify context is set correctly
                assert chat_handler.current_context == self.original_transcript
                assert "Translation" not in chat_handler.current_context
                assert "これは" not in chat_handler.current_context
                
                # Step 6: Handle chat message
                message = "What is this transcript about?"
                history = []
                
                new_history, empty_input = chat_handler.handle_message(message, history, self.test_settings)
                
                # Step 7: Verify chat response
                assert len(new_history) == 1
                assert new_history[0][0] == message
                response = new_history[0][1]
                assert "mock" in response.lower()  # Mock response indicator
                assert "(This response was generated using the transcript as context)" in response
                assert empty_input == ""
    
    def test_workflow_preserves_original_transcript_for_chat(self):
        """Test that the workflow preserves original transcript for chat context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-preserve-456"
            
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
            
            # Load job and update app state
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Simulate app state update
                app_state.current_transcript = original_transcript
                
                # Verify that display content has both languages
                assert "This is the original transcript content" in display_content
                assert "これは元の転写コンテンツです" in display_content
                assert "Translation" in display_content
                
                # Verify that app_state.current_transcript has only original
                assert "This is the original transcript content" in app_state.current_transcript
                assert "これは元の転写コンテンツです" not in app_state.current_transcript
                assert "Translation" not in app_state.current_transcript
                
                # Verify they are different
                assert display_content != app_state.current_transcript
                assert len(display_content) > len(app_state.current_transcript)
    
    def test_chat_context_isolation_from_integrated_display(self):
        """Test that chat context is isolated from integrated display content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-isolation-789"
            
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
            
            # Load job
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # Set up chat handler with original transcript
                chat_handler = MockChatHandler()
                chat_handler.set_context(original_transcript)
                
                # Verify context doesn't contain integrated display elements
                context = chat_handler.current_context
                
                # Should contain original content
                assert "This is the original transcript content" in context
                assert "actual spoken words from the audio" in context
                
                # Should NOT contain translation elements
                assert "これは元の転写コンテンツです" not in context
                assert "Translation" not in context
                assert "──────────────" not in context
                
                # Should be exactly the original transcript
                assert context == self.original_transcript
    
    def test_workflow_without_translation(self):
        """Test workflow when no translation is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "test-no-translation-101"
            
            # Save files without translation
            save_transcription_files(
                job_dir=temp_dir,
                transcript=self.original_transcript,
                translation="",  # No translation
                settings={"translation_enabled": False}
            )
            save_job_metadata(
                job_dir=temp_dir,
                job_id=job_id,
                original_filename="test_audio.mp3",
                file_info={"duration_seconds": 300.0, "size_bytes": 1024000},
                settings={"translation_enabled": False}
            )
            
            # Load job
            with patch('util.find_job_directory', return_value=temp_dir):
                display_content, original_transcript, translation = load_job_transcript(job_id)
                
                # When no translation, display content should be same as original transcript
                assert display_content == original_transcript
                assert translation == ""
                
                # Set up chat handler
                chat_handler = MockChatHandler()
                chat_handler.set_context(original_transcript)
                
                # Verify context is correct
                assert chat_handler.current_context == self.original_transcript
                
                # Handle message
                message = "Summarize this transcript"
                history = []
                new_history, empty_input = chat_handler.handle_message(message, history, self.test_settings)
                
                # Verify response
                assert len(new_history) == 1
                response = new_history[0][1]
                assert "mock response" in response.lower()  # Mock response indicator
                assert "(This response was generated using the transcript as context)" in response