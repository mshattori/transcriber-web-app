"""
Unit tests for transcribe.py module.

Tests transcription functionality, chunked processing, and CLI interface.
"""

import os
import tempfile
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transcribe import (
    TranscriptionChunk,
    TranscriptionResult,
    transcribe,
    transcribe_single_chunk,
    transcribe_chunked,
    merge_transcription_results,
    format_transcript_for_display
)


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_transcription_chunk_creation(self):
        """Test TranscriptionChunk model creation."""
        chunk = TranscriptionChunk(
            chunk_id="chunk_01",
            start_time=0.0,
            end_time=300.0,
            text="This is a test transcription.",
            confidence=0.95
        )
        
        assert chunk.chunk_id == "chunk_01"
        assert chunk.start_time == 0.0
        assert chunk.end_time == 300.0
        assert chunk.text == "This is a test transcription."
        assert chunk.confidence == 0.95
    
    def test_transcription_result_creation(self):
        """Test TranscriptionResult model creation."""
        chunks = [
            TranscriptionChunk(
                chunk_id="chunk_01",
                start_time=0.0,
                end_time=300.0,
                text="First chunk",
                confidence=None
            )
        ]
        
        result = TranscriptionResult(
            text="Full transcription text",
            chunks=chunks,
            total_duration=300.0,
            word_count=50,
            processing_time=15.5
        )
        
        assert result.text == "Full transcription text"
        assert len(result.chunks) == 1
        assert result.total_duration == 300.0
        assert result.word_count == 50
        assert result.processing_time == 15.5


class TestBasicTranscription:
    """Test basic transcription functionality."""
    
    @patch('src.transcribe.openai')
    @patch('src.transcribe.Path')
    def test_transcribe_success(self, mock_path, mock_openai):
        """Test successful basic transcription."""
        # Mock file existence and openai response
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.open.return_value.__enter__ = MagicMock()
        mock_path_instance.open.return_value.__exit__ = MagicMock()
        mock_path.return_value = mock_path_instance
        
        mock_openai.audio.transcriptions.create.return_value = "Transcribed text"
        
        result = transcribe("test.mp3", "test-api-key", "whisper-1", "en")
        
        assert result == "Transcribed text"
        mock_openai.audio.transcriptions.create.assert_called_once()
    
    @patch('src.transcribe.Path')
    def test_transcribe_file_not_found(self, mock_path):
        """Test transcription with missing file."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with pytest.raises(FileNotFoundError):
            transcribe("nonexistent.mp3", "test-api-key")


class TestChunkedTranscription:
    """Test chunked transcription functionality."""
    
    @patch('src.transcribe.split_audio')
    @patch('src.transcribe.transcribe_single_chunk')
    @patch('src.transcribe.merge_transcription_results')
    @patch('src.transcribe.cleanup_chunks')
    async def test_transcribe_chunked_success(self, mock_cleanup, mock_merge, mock_single_chunk, mock_split):
        """Test successful chunked transcription."""
        # Mock split_audio
        mock_split.return_value = ["chunk_01.mp3", "chunk_02.mp3"]
        
        # Mock transcribe_single_chunk
        mock_result_1 = {
            'text': 'First chunk text',
            'segments': [],
            'duration': 300,
            'language': 'en'
        }
        mock_result_2 = {
            'text': 'Second chunk text',
            'segments': [],
            'duration': 300,
            'language': 'en'
        }
        mock_single_chunk.side_effect = [mock_result_1, mock_result_2]
        
        # Mock merge_transcription_results
        mock_merge.return_value = "First chunk text\n\nSecond chunk text"
        
        result = await transcribe_chunked(
            audio_path="test.mp3",
            api_key="test-api-key",
            model="whisper-1",
            language="en",
            chunk_minutes=5
        )
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "First chunk text\n\nSecond chunk text"
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "chunk_01"
        assert result.chunks[1].chunk_id == "chunk_02"
        
        # Verify cleanup was called
        mock_cleanup.assert_called_once()
    
    @patch('src.transcribe.validate_api_key')
    @patch('src.transcribe.validate_file_path')
    @patch('src.transcribe.openai')
    async def test_transcribe_single_chunk_success(self, mock_openai, mock_validate_file, mock_validate_api):
        """Test successful single chunk transcription."""
        # Mock validation
        mock_validate_api.return_value = None
        mock_validate_file.return_value = None
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.text = "Transcribed chunk text"
        mock_response.segments = []
        mock_openai.audio.transcriptions.create.return_value = mock_response
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_path = temp_file.name
        
        try:
            result = await transcribe_single_chunk(
                chunk_path=temp_path,
                api_key="test-api-key",
                model="whisper-1",
                language="en"
            )
            
            assert result['text'] == "Transcribed chunk text"
            assert 'segments' in result
            assert 'duration' in result
            assert 'language' in result
        finally:
            os.unlink(temp_path)
    
    @patch('src.transcribe.validate_api_key')
    async def test_transcribe_single_chunk_invalid_api_key(self, mock_validate_api):
        """Test single chunk transcription with invalid API key."""
        from errors import ValidationError
        mock_validate_api.side_effect = ValidationError("Invalid API key")
        
        with pytest.raises(ValidationError):
            await transcribe_single_chunk(
                chunk_path="test.mp3",
                api_key="invalid-key",
                model="whisper-1",
                language="en"
            )


class TestMergeTranscriptionResults:
    """Test transcription result merging functionality."""
    
    def test_merge_transcription_results_basic(self):
        """Test basic transcription result merging."""
        results = [
            {'text': 'First chunk', 'segments': [], 'duration': 300},
            {'text': 'Second chunk', 'segments': [], 'duration': 300}
        ]
        chunks = ['chunk_01.mp3', 'chunk_02.mp3']
        
        merged = merge_transcription_results(results, chunks, include_timestamps=False)
        
        assert 'First chunk' in merged
        assert 'Second chunk' in merged
    
    def test_merge_transcription_results_with_timestamps(self):
        """Test transcription result merging with timestamps."""
        results = [
            {'text': 'First chunk', 'segments': [], 'duration': 300},
            {'text': 'Second chunk', 'segments': [], 'duration': 300}
        ]
        chunks = ['chunk_01.mp3', 'chunk_02.mp3']
        
        merged = merge_transcription_results(results, chunks, include_timestamps=True)
        
        assert '# 00:00:00 --> 00:05:00' in merged
        assert '# 00:05:00 --> 00:10:00' in merged
        assert 'First chunk' in merged
        assert 'Second chunk' in merged
    
    def test_merge_transcription_results_empty(self):
        """Test transcription result merging with empty results."""
        merged = merge_transcription_results([], [], include_timestamps=False)
        
        assert merged == ""


class TestFormatTranscriptForDisplay:
    """Test transcript formatting functionality."""
    
    def test_format_transcript_for_display_basic(self):
        """Test basic transcript formatting."""
        transcript = "# 00:00:00 --> 00:01:00\nFirst minute of audio\n\n# 00:01:00 --> 00:02:00\nSecond minute of audio"
        
        formatted = format_transcript_for_display(transcript)
        
        assert '<span class="timestamp">' in formatted
        assert 'First minute of audio' in formatted
        assert 'Second minute of audio' in formatted
    
    def test_format_transcript_for_display_empty(self):
        """Test transcript formatting with empty input."""
        formatted = format_transcript_for_display("")
        
        assert formatted == ""
    
    def test_format_transcript_for_display_no_timestamps(self):
        """Test transcript formatting without timestamps."""
        transcript = "Just plain text without timestamps"
        
        formatted = format_transcript_for_display(transcript)
        
        assert transcript in formatted
        assert '<span class="timestamp">' not in formatted


class TestProgressCallback:
    """Test progress callback functionality."""
    
    @patch('src.transcribe.split_audio')
    @patch('src.transcribe.transcribe_single_chunk')
    @patch('src.transcribe.merge_transcription_results')
    @patch('src.transcribe.cleanup_chunks')
    async def test_transcribe_chunked_with_progress_callback(self, mock_cleanup, mock_merge, mock_single_chunk, mock_split):
        """Test chunked transcription with progress callback."""
        # Mock dependencies
        mock_split.return_value = ["chunk_01.mp3"]
        mock_single_chunk.return_value = {
            'text': 'Test text',
            'segments': [],
            'duration': 300,
            'language': 'en'
        }
        mock_merge.return_value = "Test text"
        
        # Track progress callback calls
        progress_calls = []
        
        def mock_progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        await transcribe_chunked(
            audio_path="test.mp3",
            api_key="test-api-key",
            progress_callback=mock_progress_callback
        )
        
        # Verify progress callback was called
        assert len(progress_calls) > 0
        
        # Check that progress goes from 0 to 1
        first_progress = progress_calls[0][0]
        last_progress = progress_calls[-1][0]
        
        assert first_progress >= 0.0
        assert last_progress == 1.0
        
        # Check that messages are informative
        messages = [call[1] for call in progress_calls]
        assert any("Splitting audio" in msg for msg in messages)
        assert any("completed" in msg.lower() for msg in messages)


if __name__ == "__main__":
    pytest.main([__file__])