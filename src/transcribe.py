"""
Enhanced transcription module with chunked processing and progress reporting.

Supports large file transcription with OpenAI Whisper API, including
real-time progress updates and retry logic with exponential backoff.
"""

import argparse
import asyncio
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import openai
from pydantic import BaseModel

from .util import format_duration, split_audio


class TranscriptionChunk(BaseModel):
    """Data model for transcription chunk."""
    chunk_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float | None = None


class TranscriptionResult(BaseModel):
    """Data model for complete transcription result."""
    text: str
    chunks: list[TranscriptionChunk]
    total_duration: float
    word_count: int
    processing_time: float


def transcribe(
    audio_path: str,
    api_key: str,
    model: str = "whisper-1",
    language: str = "auto"
) -> str:
    """
    Single audio file transcription using OpenAI Speech-to-Text API.
    
    Args:
        audio_path: Path to audio file
        api_key: OpenAI API key
        model: Model to use for transcription
        language: Language code or "auto" for automatic detection
        
    Returns:
        Transcribed text
    """
    openai.api_key = api_key

    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    with p.open("rb") as f:
        resp = openai.audio.transcriptions.create(
            model=model,
            file=f,
            language=None if language == "auto" else language,
            response_format="text"
        )
    return resp


async def transcribe_single_chunk(
    chunk_path: str,
    api_key: str,
    model: str,
    language: str,
    temperature: float = 0.0,
    include_timestamps: bool = True
) -> dict:
    """
    Transcribe a single audio chunk with retry logic.
    
    Args:
        chunk_path: Path to chunk file
        api_key: OpenAI API key
        model: Model to use
        language: Language code
        temperature: Temperature for transcription
        include_timestamps: Whether to include timestamps
        
    Returns:
        Dictionary with transcription result and metadata
    """
    from errors import (
        APIError,
        NetworkError,
        handle_openai_error,
        safe_execute,
        validate_api_key,
        validate_file_path,
    )

    # Validate inputs
    validate_api_key(api_key)
    validate_file_path(chunk_path, must_exist=True)

    openai.api_key = api_key

    # Retry logic with exponential backoff
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            def _transcribe_chunk():
                with open(chunk_path, "rb") as f:
                    # Use tempfile.SpooledTemporaryFile for memory efficiency
                    with tempfile.SpooledTemporaryFile() as temp_file:
                        temp_file.write(f.read())
                        temp_file.seek(0)

                        response_format = "verbose_json" if include_timestamps else "text"

                        resp = openai.audio.transcriptions.create(
                            model=model,
                            file=(Path(chunk_path).name, temp_file),
                            language=None if language == "auto" else language,
                            temperature=temperature,
                            response_format=response_format
                        )

                        if include_timestamps and hasattr(resp, 'segments'):
                            return {
                                'text': resp.text,
                                'segments': resp.segments,
                                'duration': getattr(resp, 'duration', 0),
                                'language': getattr(resp, 'language', language)
                            }
                        else:
                            return {
                                'text': resp if isinstance(resp, str) else resp.text,
                                'segments': [],
                                'duration': 0,
                                'language': language
                            }

            return safe_execute(_transcribe_chunk, error_context=f"transcribing chunk {chunk_path}")

        except (APIError, NetworkError) as e:
            last_error = e

            # Don't retry on authentication or quota errors
            if hasattr(e, 'status_code') and e.status_code in [401, 402, 403]:
                raise e

            if attempt == max_retries - 1:
                raise e

            # Exponential backoff: wait 2^attempt seconds
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)

        except Exception as e:
            # Convert to appropriate error type
            api_error = handle_openai_error(e)
            last_error = api_error

            # Don't retry on authentication or quota errors
            if hasattr(api_error, 'status_code') and api_error.status_code in [401, 402, 403]:
                raise api_error

            if attempt == max_retries - 1:
                raise api_error

            # Exponential backoff: wait 2^attempt seconds
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)

    # Should not reach here, but just in case
    raise last_error or APIError("Max retries exceeded", api_name="OpenAI")


async def transcribe_chunked(
    audio_path: str,
    api_key: str,
    model: str = "whisper-1",
    language: str = "auto",
    chunk_minutes: int = 5,
    temperature: float = 0.0,
    include_timestamps: bool = True,
    progress_callback: Callable | None = None,
    job_dir: str | None = None
) -> TranscriptionResult:
    """
    Transcribe large audio file using chunked processing with progress reporting.
    
    Implements the 4-step processing sequence from INITIAL.md:
    1. Split audio into chunks with pydub
    2. Transcribe chunks with yield-based progress
    3. Real-time updates during processing  
    4. Merge results intelligently
    
    Args:
        audio_path: Path to input audio file
        api_key: OpenAI API key
        model: Model to use for transcription
        language: Language code or "auto"
        chunk_minutes: Duration of each chunk in minutes
        temperature: Temperature for transcription
        include_timestamps: Whether to include timestamps
        progress_callback: Optional callback for progress updates
        job_dir: Optional job directory to save split audio files
        
    Returns:
        TranscriptionResult with complete transcription
    """
    start_time = time.time()

    # Step 1: Split audio into chunks
    if progress_callback:
        progress_callback(0.1, "Splitting audio into chunks...")

    chunks, temp_dir = split_audio(audio_path, chunk_minutes, overlap_seconds=2)
    total_chunks = len(chunks)

    # Step 2: Transcribe chunks with progress
    results = []
    chunk_objects = []

    for i, chunk_path in enumerate(chunks):
        print(f"[DEBUG] Processing chunk: {chunk_path}")
        if progress_callback:
            progress_percent = 0.1 + (i / total_chunks) * 0.8  # 10% to 90%
            progress_callback(progress_percent, f"Processing chunk {i+1}/{total_chunks}")

        try:
            result = await transcribe_single_chunk(
                chunk_path, api_key, model, language, temperature, include_timestamps
            )

            # Create chunk object
            chunk_obj = TranscriptionChunk(
                chunk_id=f"chunk_{i+1:02d}",
                start_time=i * chunk_minutes * 60,  # Approximate start time
                end_time=(i + 1) * chunk_minutes * 60,  # Approximate end time
                text=result['text'],
                confidence=None  # OpenAI doesn't provide confidence scores
            )

            chunk_objects.append(chunk_obj)
            results.append(result)

        except Exception as e:
            # Clean up partial chunks on error
            from util import cleanup_chunks
            cleanup_chunks(chunks, temp_dir)

            from errors import APIError, TranscriberError

            if isinstance(e, TranscriberError):
                raise e
            else:
                raise APIError(f"Failed to transcribe chunk {i+1}: {str(e)}", api_name="OpenAI")

    # Step 3: Merge overlapping chunks intelligently
    if progress_callback:
        progress_callback(0.9, "Merging transcription results...")

    merged_text = merge_transcription_results(results, include_timestamps, chunk_minutes)

    # Calculate statistics
    word_count = len(merged_text.split())
    processing_time = time.time() - start_time

    # Copy chunks to job directory if specified
    if job_dir:
        import shutil
        for chunk_path in chunks:
            try:
                chunk_filename = os.path.basename(chunk_path)
                dest_path = os.path.join(job_dir, chunk_filename)
                shutil.copy2(chunk_path, dest_path)
                print(f"[DEBUG] Saved chunk to job directory: {dest_path}")
            except Exception as e:
                print(f"Warning: Failed to copy chunk {chunk_path} to job directory: {e}")

    # Cleanup temporary files
    from util import cleanup_chunks
    cleanup_chunks(chunks, temp_dir)

    if progress_callback:
        progress_callback(1.0, "Transcription completed!")

    return TranscriptionResult(
        text=merged_text,
        chunks=chunk_objects,
        total_duration=len(chunk_objects) * chunk_minutes * 60,  # Approximate
        word_count=word_count,
        processing_time=processing_time
    )


def merge_transcription_results(
    results: list[dict],
    include_timestamps: bool = True,
    chunk_minutes: int = 5
) -> str:
    """
    Merge overlapping chunk transcription results intelligently.
    
    Args:
        results: List of transcription results from chunks
        include_timestamps: Whether to include timestamps in output
        chunk_minutes: Duration of each chunk in minutes
        
    Returns:
        Merged transcription text with proper formatting
    """
    if not results:
        return ""

    merged_segments = []
    current_time = 0

    for i, result in enumerate(results):
        text = result.get('text', '').strip()
        if not text:
            continue

        if include_timestamps:
            # Format with timestamp as per INITIAL.md: # HH:MM:SS --> HH:MM:SS
            chunk_duration_seconds = chunk_minutes * 60
            start_time = format_duration(current_time)
            end_time = format_duration(current_time + chunk_duration_seconds)

            segment = f"# {start_time} --> {end_time}\n{text}"
            merged_segments.append(segment)
            current_time += chunk_duration_seconds  # Move forward by actual chunk duration
        else:
            merged_segments.append(text)

    return "\n\n".join(merged_segments)


def format_transcript_for_display(transcript: str) -> str:
    """
    Format transcript for display in Gradio with timestamp styling.
    
    Converts timestamps to HTML spans for CSS styling as specified in INITIAL.md.
    
    Args:
        transcript: Raw transcript with timestamps
        
    Returns:
        HTML-formatted transcript
    """
    import re

    # Convert timestamp format # HH:MM:SS --> HH:MM:SS to HTML spans
    timestamp_pattern = r'# (\d{2}:\d{2}:\d{2}) --> (\d{2}:\d{2}:\d{2})'

    def replace_timestamp(match):
        start_time = match.group(1)
        end_time = match.group(2)
        return f'<span class="timestamp"># {start_time} --> {end_time}</span>'

    formatted = re.sub(timestamp_pattern, replace_timestamp, transcript)
    return formatted


# CLI Interface Implementation (INITIAL.md requirement)
def main():
    """Command-line interface for transcription functionality."""
    parser = argparse.ArgumentParser(description="Audio transcription CLI")
    parser.add_argument("--file", required=True, help="Audio file path")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--model", default="whisper-1", help="Model to use")
    parser.add_argument("--language", default="auto", help="Language code")
    parser.add_argument("--chunk-minutes", type=int, default=5, help="Chunk duration in minutes")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    try:
        # Run async transcription
        result = asyncio.run(transcribe_chunked(
            audio_path=args.file,
            api_key=args.api_key,
            model=args.model,
            language=args.language,
            chunk_minutes=args.chunk_minutes,
            temperature=args.temperature,
            include_timestamps=args.timestamps,
            progress_callback=lambda p, m: print(f"Progress: {p*100:.1f}% - {m}")
        ))

        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Transcription saved to: {args.output}")
        else:
            print("Transcription Result:")
            print("=" * 50)
            print(result.text)
            print("=" * 50)
            print(f"Word count: {result.word_count}")
            print(f"Processing time: {result.processing_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
