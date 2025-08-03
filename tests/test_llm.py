"""
Unit tests for llm.py module.

Tests LLM functionality, translation, chat context injection, and CLI interface.
"""

import os
import json
import asyncio
from unittest.mock import patch, MagicMock
import pytest

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import (
    TranslationSegment,
    TranslationResult,
    ChatContext,
    chat_completion,
    chat_completion_async,
    chat_with_context,
    structured_completion,
    parse_transcript_to_json,
    reconstruct_transcript_from_json,
    translate_transcript_json,
    translate_transcript_full,
    get_language_code
)


class TestDataModels:
    """Test Pydantic data models for LLM functionality."""
    
    def test_translation_segment_creation(self):
        """Test TranslationSegment model creation."""
        segment = TranslationSegment(
            ts="00:00:00 --> 00:01:00",
            text="This is a test segment."
        )
        
        assert segment.ts == "00:00:00 --> 00:01:00"
        assert segment.text == "This is a test segment."
    
    def test_translation_result_creation(self):
        """Test TranslationResult model creation."""
        result = TranslationResult(
            original_text="Original text",
            translated_text="Translated text",
            target_language="Japanese",
            processing_time=5.2,
            segment_count=3
        )
        
        assert result.original_text == "Original text"
        assert result.translated_text == "Translated text"
        assert result.target_language == "Japanese"
        assert result.processing_time == 5.2
        assert result.segment_count == 3
    
    def test_chat_context_creation(self):
        """Test ChatContext model creation."""
        context = ChatContext(
            context_text="Transcript context",
            question="What is the main topic?",
            system_message="You are a helpful assistant."
        )
        
        assert context.context_text == "Transcript context"
        assert context.question == "What is the main topic?"
        assert context.system_message == "You are a helpful assistant."


class TestBasicChatFunctionality:
    """Test basic chat completion functionality."""
    
    @patch('src.llm.openai')
    def test_chat_completion_first_call(self, mock_openai):
        """Test chat completion on first call."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Assistant response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        response, history = chat_completion(
            api_key="test-key",
            model="gpt-4o-mini",
            message="Hello",
            system_message="You are helpful"
        )
        
        assert response == "Assistant response"
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Assistant response"
    
    @patch('src.llm.openai')
    def test_chat_completion_with_history(self, mock_openai):
        """Test chat completion with existing history."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Follow-up response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        existing_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        response, history = chat_completion(
            api_key="test-key",
            model="gpt-4o-mini",
            message="Follow-up question",
            history=existing_history
        )
        
        assert response == "Follow-up response"
        assert len(history) == 4  # previous 2 + new 2
    
    @patch('src.llm.chat_completion')
    async def test_chat_completion_async_success(self, mock_chat_completion):
        """Test async chat completion success."""
        mock_chat_completion.return_value = ("Response", [])
        
        response, history = await chat_completion_async(
            api_key="test-key",
            model="gpt-4o-mini",
            message="Test message"
        )
        
        assert response == "Response"
        assert history == []
    
    @patch('src.llm.chat_completion')
    async def test_chat_completion_async_with_retries(self, mock_chat_completion):
        """Test async chat completion with retries."""
        # First two calls fail, third succeeds
        mock_chat_completion.side_effect = [
            Exception("Temporary error"),
            Exception("Another error"),
            ("Success", [])
        ]
        
        response, history = await chat_completion_async(
            api_key="test-key",
            model="gpt-4o-mini",
            message="Test message",
            max_retries=3
        )
        
        assert response == "Success"
        assert mock_chat_completion.call_count == 3


class TestChatWithContext:
    """Test chat with context injection functionality."""
    
    @patch('src.llm.openai')
    @patch('src.llm.validate_api_key')
    def test_chat_with_context_success(self, mock_validate, mock_openai):
        """Test successful chat with context injection."""
        mock_validate.return_value = None
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Context-aware response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        response = chat_with_context(
            api_key="test-key",
            model="gpt-4o-mini",
            question="What is discussed?",
            context_text="This is the transcript context.",
            system_message="You are helpful."
        )
        
        assert response == "Context-aware response"
        
        # Verify that openai was called with proper message structure
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        # Should have 3 messages: system, context, question
        assert len(messages) == 3
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'  # context
        assert messages[2]['role'] == 'user'  # question
        assert "transcript" in messages[1]['content'].lower()
        assert "What is discussed?" in messages[2]['content']
    
    @patch('src.llm.validate_api_key')
    def test_chat_with_context_invalid_api_key(self, mock_validate):
        """Test chat with context with invalid API key."""
        from errors import ValidationError
        mock_validate.side_effect = ValidationError("Invalid API key")
        
        with pytest.raises(ValidationError):
            chat_with_context(
                api_key="invalid-key",
                model="gpt-4o-mini",
                question="Test question",
                context_text="Test context"
            )
    
    @patch('src.llm.validate_api_key')
    def test_chat_with_context_empty_question(self, mock_validate):
        """Test chat with context with empty question."""
        from errors import ValidationError
        mock_validate.return_value = None
        
        with pytest.raises(ValidationError) as exc_info:
            chat_with_context(
                api_key="test-key",
                model="gpt-4o-mini",
                question="",
                context_text="Test context"
            )
        
        assert "Question cannot be empty" in str(exc_info.value)


class TestTranscriptParsing:
    """Test transcript parsing and reconstruction functionality."""
    
    def test_parse_transcript_to_json(self):
        """Test parsing transcript to JSON format."""
        transcript = """# 00:00:00 --> 00:01:00
First minute content

# 00:01:00 --> 00:02:00
Second minute content"""
        
        segments = parse_transcript_to_json(transcript)
        
        assert len(segments) == 2
        assert segments[0]["ts"] == "00:00:00 --> 00:01:00"
        assert segments[0]["text"] == "First minute content"
        assert segments[1]["ts"] == "00:01:00 --> 00:02:00"
        assert segments[1]["text"] == "Second minute content"
    
    def test_parse_transcript_to_json_empty(self):
        """Test parsing empty transcript."""
        segments = parse_transcript_to_json("")
        
        assert segments == []
    
    def test_parse_transcript_to_json_multiline_content(self):
        """Test parsing transcript with multiline content."""
        transcript = """# 00:00:00 --> 00:01:00
First line of content
Second line of content

# 00:01:00 --> 00:02:00
Single line"""
        
        segments = parse_transcript_to_json(transcript)
        
        assert len(segments) == 2
        assert "First line of content\nSecond line of content" in segments[0]["text"]
        assert segments[1]["text"] == "Single line"
    
    def test_reconstruct_transcript_from_json(self):
        """Test reconstructing transcript from JSON format."""
        segments = [
            {"ts": "00:00:00 --> 00:01:00", "text": "First minute"},
            {"ts": "00:01:00 --> 00:02:00", "text": "Second minute"}
        ]
        
        transcript = reconstruct_transcript_from_json(segments)
        
        assert "# 00:00:00 --> 00:01:00" in transcript
        assert "First minute" in transcript
        assert "# 00:01:00 --> 00:02:00" in transcript
        assert "Second minute" in transcript
    
    def test_reconstruct_transcript_from_json_empty(self):
        """Test reconstructing transcript from empty JSON."""
        transcript = reconstruct_transcript_from_json([])
        
        assert transcript == ""


class TestTranslationFunctionality:
    """Test translation functionality."""
    
    @patch('src.llm.structured_completion')
    @patch('src.llm.validate_api_key')
    async def test_translate_transcript_json_success(self, mock_validate, mock_structured):
        """Test successful JSON transcript translation."""
        mock_validate.return_value = None
        
        # Mock structured completion response
        mock_response = {
            "segments": [
                {"ts": "00:00:00 --> 00:01:00", "text": "翻訳されたテキスト"}
            ]
        }
        mock_structured.return_value = mock_response
        
        input_segments = [
            {"ts": "00:00:00 --> 00:01:00", "text": "Original text"}
        ]
        
        result = await translate_transcript_json(
            api_key="test-key",
            model="gpt-4o-mini",
            transcript_json=input_segments,
            target_language="Japanese"
        )
        
        assert len(result) == 1
        assert result[0]["ts"] == "00:00:00 --> 00:01:00"
        assert result[0]["text"] == "翻訳されたテキスト"
    
    @patch('src.llm.validate_api_key')
    async def test_translate_transcript_json_invalid_api_key(self, mock_validate):
        """Test translation with invalid API key."""
        from errors import ValidationError
        mock_validate.side_effect = ValidationError("Invalid API key")
        
        with pytest.raises(ValidationError):
            await translate_transcript_json(
                api_key="invalid-key",
                model="gpt-4o-mini",
                transcript_json=[{"ts": "00:00:00 --> 00:01:00", "text": "Test"}],
                target_language="Japanese"
            )
    
    @patch('src.llm.validate_api_key')
    async def test_translate_transcript_json_empty_target_language(self, mock_validate):
        """Test translation with empty target language."""
        from errors import ValidationError
        mock_validate.return_value = None
        
        with pytest.raises(ValidationError) as exc_info:
            await translate_transcript_json(
                api_key="test-key",
                model="gpt-4o-mini",
                transcript_json=[{"ts": "00:00:00 --> 00:01:00", "text": "Test"}],
                target_language=""
            )
        
        assert "Target language cannot be empty" in str(exc_info.value)
    
    @patch('src.llm.parse_transcript_to_json')
    @patch('src.llm.translate_transcript_chunked')
    @patch('src.llm.reconstruct_transcript_from_json')
    async def test_translate_transcript_full_workflow(self, mock_reconstruct, mock_translate_chunked, mock_parse):
        """Test full translation workflow."""
        # Mock the workflow steps
        mock_parse.return_value = [{"ts": "00:00:00 --> 00:01:00", "text": "Original"}]
        mock_translate_chunked.return_value = [{"ts": "00:00:00 --> 00:01:00", "text": "Translated"}]
        mock_reconstruct.return_value = "# 00:00:00 --> 00:01:00\nTranslated"
        
        result = await translate_transcript_full(
            api_key="test-key",
            model="gpt-4o-mini",
            transcript_text="# 00:00:00 --> 00:01:00\nOriginal",
            target_language="Japanese"
        )
        
        assert isinstance(result, TranslationResult)
        assert result.translated_text == "# 00:00:00 --> 00:01:00\nTranslated"
        assert result.target_language == "Japanese"
        assert result.segment_count == 1


class TestLanguageCodeUtility:
    """Test language code utility functionality."""
    
    @patch('src.llm.load_config')
    def test_get_language_code_success(self, mock_load_config):
        """Test successful language code retrieval."""
        mock_config = {
            'translation_languages': {
                'Japanese': 'ja',
                'English': 'en'
            }
        }
        mock_load_config.return_value = mock_config
        
        code = get_language_code("Japanese")
        
        assert code == "ja"
    
    @patch('src.llm.load_config')
    def test_get_language_code_fallback(self, mock_load_config):
        """Test language code fallback functionality."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")
        
        code = get_language_code("Japanese")
        
        assert code == "ja"  # Should use fallback mapping
    
    @patch('src.llm.load_config')
    def test_get_language_code_unknown_language(self, mock_load_config):
        """Test language code for unknown language."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")
        
        code = get_language_code("Unknown")
        
        assert code == "unknown"  # Should return lowercase


class TestStructuredCompletion:
    """Test structured completion functionality."""
    
    @patch('src.llm.openai')
    def test_structured_completion_success(self, mock_openai):
        """Test successful structured completion."""
        mock_response = MagicMock()
        mock_response.choices[0].message.model_dump.return_value = {
            "content": {"key": "value"}
        }
        mock_openai.chat.completions.create.return_value = mock_response
        
        schema = {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        }
        
        result = structured_completion(
            api_key="test-key",
            model="gpt-4o-mini",
            system_prompt="System prompt",
            user_prompt="User prompt",
            json_schema=schema
        )
        
        assert result == {"key": "value"}
        
        # Verify structured output was requested
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args[1]['response_format']['type'] == 'json_object'
        assert call_args[1]['response_format']['schema'] == schema


if __name__ == "__main__":
    pytest.main([__file__])