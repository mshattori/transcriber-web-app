"""
Unit tests for ChatHandler with test transcript context.

Tests both real and mock implementations of chat functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from handlers.chat_handler import ChatHandler, MockChatHandler
from config.test_config import TestConfig


class TestChatHandler:
    """Test suite for real ChatHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = ChatHandler()
        self.test_config = TestConfig()
        self.test_settings = self.test_config.get_test_settings()
        
        # Load test transcript for context
        transcript_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_transcript.txt'
        )
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                self.test_transcript = f.read()
        else:
            self.test_transcript = "Test transcript content for chat context."
    
    def test_set_context(self):
        """Test setting chat context."""
        self.handler.set_context(self.test_transcript)
        
        assert self.handler.current_context == self.test_transcript
    
    def test_clear_history(self):
        """Test clearing chat history."""
        # Add some mock history
        self.handler.chat_history = [{"user": "test", "assistant": "response"}]
        
        result = self.handler.clear_history()
        
        assert result == []
        assert self.handler.chat_history == []
    
    def test_handle_empty_message(self):
        """Test handling empty message."""
        history = []
        
        new_history, empty_input = self.handler.handle_message("", history, self.test_settings)
        
        assert new_history == []
        assert empty_input == ""
    
    def test_handle_message_missing_api_key(self):
        """Test handling message with missing API key."""
        settings = self.test_settings.copy()
        settings["api_key"] = ""
        
        with pytest.raises(ValueError, match="API key"):
            self.handler.handle_message("Hello", [], settings)
    
    @patch('handlers.chat_handler.chat_with_context')
    def test_handle_message_with_context(self, mock_chat_with_context):
        """Test handling message with context (mocked API)."""
        mock_chat_with_context.return_value = "This is a response based on the transcript context."
        
        # Set context
        self.handler.set_context(self.test_transcript)
        
        history = []
        message = "What is this text about?"
        
        new_history, empty_input = self.handler.handle_message(message, history, self.test_settings)
        
        assert len(new_history) == 1
        assert new_history[0][0] == message
        assert new_history[0][1] == "This is a response based on the transcript context."
        assert empty_input == ""
        
        # Verify chat_with_context was called with correct parameters
        mock_chat_with_context.assert_called_once_with(
            api_key=self.test_settings["api_key"],
            model=self.test_settings["language_model"],
            question=message,
            context_text=self.test_transcript,
            system_message=self.test_settings.get("system_message", ""),
            temperature=0.7
        )
    
    @patch('handlers.chat_handler.chat_completion')
    def test_handle_message_without_context(self, mock_chat_completion):
        """Test handling message without context (mocked API)."""
        mock_chat_completion.return_value = ("This is a general response.", None)
        
        # No context set
        history = []
        message = "Hello, how are you?"
        
        new_history, empty_input = self.handler.handle_message(message, history, self.test_settings)
        
        assert len(new_history) == 1
        assert new_history[0][0] == message
        assert new_history[0][1] == "This is a general response."
        assert empty_input == ""
        
        # Verify chat_completion was called
        mock_chat_completion.assert_called_once_with(
            api_key=self.test_settings["api_key"],
            model=self.test_settings["language_model"],
            message=message,
            system_message=self.test_settings.get("system_message", ""),
            temperature=0.7
        )
    
    @patch('handlers.chat_handler.chat_with_context')
    def test_handle_multiple_messages(self, mock_chat_with_context):
        """Test handling multiple messages in conversation."""
        mock_chat_with_context.side_effect = [
            "First response",
            "Second response",
            "Third response"
        ]
        
        self.handler.set_context(self.test_transcript)
        
        history = []
        
        # First message
        history, _ = self.handler.handle_message("First question", history, self.test_settings)
        assert len(history) == 1
        
        # Second message
        history, _ = self.handler.handle_message("Second question", history, self.test_settings)
        assert len(history) == 2
        
        # Third message
        history, _ = self.handler.handle_message("Third question", history, self.test_settings)
        assert len(history) == 3
        
        # Verify conversation structure
        assert history[0] == ["First question", "First response"]
        assert history[1] == ["Second question", "Second response"]
        assert history[2] == ["Third question", "Third response"]


class TestMockChatHandler:
    """Test suite for MockChatHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = MockChatHandler()
        self.test_config = TestConfig()
        self.test_settings = self.test_config.get_test_settings()
        
        # Load test transcript for context
        transcript_path = os.path.join(
            os.path.dirname(__file__), 
            '../data/test_transcript.txt'
        )
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                self.test_transcript = f.read()
        else:
            self.test_transcript = "Test transcript content for chat context."
    
    def test_set_context(self):
        """Test setting chat context in mock."""
        self.handler.set_context(self.test_transcript)
        
        assert self.handler.current_context == self.test_transcript
    
    def test_clear_history(self):
        """Test clearing chat history in mock."""
        self.handler.chat_history = [{"user": "test", "assistant": "response"}]
        
        result = self.handler.clear_history()
        
        assert result == []
        assert self.handler.chat_history == []
    
    def test_handle_empty_message(self):
        """Test handling empty message in mock."""
        history = []
        
        new_history, empty_input = self.handler.handle_message("", history, self.test_settings)
        
        assert new_history == []
        assert empty_input == ""
    
    def test_handle_message_with_keywords(self):
        """Test mock responses to specific keywords."""
        history = []
        
        # Test "hello" keyword
        new_history, _ = self.handler.handle_message("Hello there!", history, self.test_settings)
        assert len(new_history) == 1
        assert "mock chat assistant" in new_history[0][1].lower()
        
        # Test "summary" keyword
        history = []
        new_history, _ = self.handler.handle_message("Give me a summary", history, self.test_settings)
        assert "mock summary" in new_history[0][1].lower()
        
        # Test "translate" keyword
        history = []
        new_history, _ = self.handler.handle_message("Can you translate this?", history, self.test_settings)
        assert "mock translation" in new_history[0][1].lower()
        
        # Test "key points" keyword
        history = []
        new_history, _ = self.handler.handle_message("What are the key points?", history, self.test_settings)
        assert "mock key points" in new_history[0][1].lower()
    
    def test_handle_message_default_response(self):
        """Test default mock response for unknown keywords."""
        history = []
        message = "This is a random question without keywords"
        
        new_history, empty_input = self.handler.handle_message(message, history, self.test_settings)
        
        assert len(new_history) == 1
        assert new_history[0][0] == message
        assert "mock response" in new_history[0][1].lower()
        assert "ui testing" in new_history[0][1].lower()
        assert empty_input == ""
    
    def test_handle_message_with_context(self):
        """Test mock response includes context information."""
        self.handler.set_context(self.test_transcript)
        
        history = []
        message = "Tell me about this content"
        
        new_history, _ = self.handler.handle_message(message, history, self.test_settings)
        
        assert len(new_history) == 1
        response = new_history[0][1]
        
        # Should mention context in response when context is available
        assert "context" in response.lower()
    
    def test_handle_message_without_context(self):
        """Test mock response when no context is set."""
        # Ensure no context is set
        self.handler.current_context = ""
        
        history = []
        message = "Tell me something"
        
        new_history, _ = self.handler.handle_message(message, history, self.test_settings)
        
        assert len(new_history) == 1
        # Should not mention context when none is available
        response = new_history[0][1]
        assert "context" not in response.lower()
    
    def test_mock_responses_are_instant(self):
        """Test that mock responses are returned instantly."""
        import time
        
        start_time = time.time()
        
        history = []
        self.handler.handle_message("Test message", history, self.test_settings)
        
        elapsed_time = time.time() - start_time
        
        # Mock should be very fast (< 10ms)
        assert elapsed_time < 0.01
    
    def test_multiple_conversations(self):
        """Test multiple conversation turns with mock."""
        history = []
        
        # First turn
        history, _ = self.handler.handle_message("Hello", history, self.test_settings)
        assert len(history) == 1
        assert "hello" in history[0][1].lower()
        
        # Second turn with different keyword
        history, _ = self.handler.handle_message("Give me a summary", history, self.test_settings)
        assert len(history) == 2
        assert "summary" in history[1][1].lower()
        
        # Third turn with default response
        history, _ = self.handler.handle_message("Random question", history, self.test_settings)
        assert len(history) == 3
        assert "mock response" in history[2][1].lower()


class TestChatHandlerIntegration:
    """Integration tests comparing real and mock handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.real_handler = ChatHandler()
        self.mock_handler = MockChatHandler()
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
    
    def test_handler_interface_compatibility(self):
        """Test that real and mock handlers have compatible interfaces."""
        # Both should have the same methods
        real_methods = [method for method in dir(self.real_handler) if not method.startswith('_')]
        mock_methods = [method for method in dir(self.mock_handler) if not method.startswith('_')]
        
        assert set(real_methods) == set(mock_methods)
    
    def test_both_handlers_accept_same_parameters(self):
        """Test that both handlers accept the same parameters."""
        settings = self.test_config.get_test_settings()
        history = []
        message = "Test message"
        
        # Mock handler (always works)
        mock_history, mock_input = self.mock_handler.handle_message(message, history, settings)
        
        assert isinstance(mock_history, list)
        assert isinstance(mock_input, str)
        
        # Real handler with mocked API
        with patch('handlers.chat_handler.chat_completion') as mock_chat:
            mock_chat.return_value = ("Test response", None)
            
            real_history, real_input = self.real_handler.handle_message(message, history, settings)
            
            # Should return same types
            assert isinstance(real_history, list)
            assert isinstance(real_input, str)
            assert type(real_history) == type(mock_history)
            assert type(real_input) == type(mock_input)
    
    def test_both_handlers_context_behavior(self):
        """Test that both handlers handle context similarly."""
        # Set context on both handlers
        self.real_handler.set_context(self.test_transcript)
        self.mock_handler.set_context(self.test_transcript)
        
        assert self.real_handler.current_context == self.test_transcript
        assert self.mock_handler.current_context == self.test_transcript
        
        # Clear context on both
        self.real_handler.current_context = ""
        self.mock_handler.current_context = ""
        
        assert self.real_handler.current_context == ""
        assert self.mock_handler.current_context == ""