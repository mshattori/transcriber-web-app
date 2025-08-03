"""
Enhanced LLM module for transcriber web app.

Supports translation with structured JSON outputs, chat context injection,
and CLI interface for testing as specified in INITIAL.md.
"""

import asyncio
import argparse
import json
import time
from typing import List, Dict, Tuple, Any, Optional, Callable

import openai
from pydantic import BaseModel

from util import load_config


class TranslationSegment(BaseModel):
    """Data model for translation segment with timestamp."""
    ts: str  # Timestamp in format "HH:MM:SS --> HH:MM:SS"
    text: str  # Text content to be translated


class TranslationResult(BaseModel):
    """Data model for complete translation result."""
    original_text: str
    translated_text: str
    target_language: str
    processing_time: float
    segment_count: int


class ChatContext(BaseModel):
    """Data model for chat context injection."""
    context_text: str
    question: str
    system_message: str


def chat_completion(
    api_key: str,
    model: str,
    message: str,
    system_message: Optional[str] = None,
    history: List[Dict[str, str]] | None = None,
    temperature: float = 0.7,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Basic chat completion with history support.
    
    Args:
        api_key: OpenAI API key
        model: Model to use for chat
        message: User message
        system_message: System message (optional)
        history: Previous conversation history
        temperature: Temperature for response generation
        
    Returns:
        Tuple of (assistant_response, updated_history)
    """
    openai.api_key = api_key

    if history is None or len(history) == 0:
        # First call: system + user
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})
        new_history: List[Dict[str, str]] = []
    else:
        # Subsequent calls
        messages = history + [{"role": "user", "content": message}]
        new_history = history.copy()

    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    assistant_reply = resp.choices[0].message.content.strip()

    # Add latest user/assistant to history
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, new_history


async def chat_completion_async(
    api_key: str,
    model: str,
    message: str,
    system_message: Optional[str] = None,
    history: List[Dict[str, str]] | None = None,
    temperature: float = 0.7,
    max_retries: int = 3
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Async chat completion with retry logic.
    
    Args:
        api_key: OpenAI API key
        model: Model to use for chat
        message: User message
        system_message: System message (optional)
        history: Previous conversation history
        temperature: Temperature for response generation
        max_retries: Maximum number of retries on failure
        
    Returns:
        Tuple of (assistant_response, updated_history)
    """
    for attempt in range(max_retries):
        try:
            return chat_completion(api_key, model, message, system_message, history, temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)
    
    raise RuntimeError("Max retries exceeded")


def chat_with_context(
    api_key: str,
    model: str,
    question: str,
    context_text: str,
    system_message: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Chat with context injection using 3-message pattern from INITIAL.md.
    
    Implements the exact pattern:
    1. system: config.yaml system_message
    2. user: context text (transcription)
    3. user: actual question
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        question: User's question
        context_text: Transcription text to inject as context
        system_message: System message from config
        temperature: Temperature for response generation
        
    Returns:
        Assistant's response
    """
    from errors import (
        validate_api_key, ValidationError, handle_openai_error, safe_execute
    )
    
    # Validate inputs
    validate_api_key(api_key)
    
    if not question.strip():
        raise ValidationError("Question cannot be empty", field="question")
    
    if not model.strip():
        raise ValidationError("Model must be specified", field="model")
    
    openai.api_key = api_key
    
    def _chat_with_context():
        messages = []
        
        # Step 1: System message
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Step 2: Context injection
        if context_text and context_text.strip():
            context_message = f"以下は文字起こしされたテキストです。この内容を参考にして質問に答えてください。\n\n{context_text}"
            messages.append({"role": "user", "content": context_message})
        
        # Step 3: Actual question
        messages.append({"role": "user", "content": question})
        
        resp = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        
        return resp.choices[0].message.content.strip()
    
    try:
        return safe_execute(_chat_with_context, error_context="chat with context")
    except Exception as e:
        raise handle_openai_error(e)


def structured_completion(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Get structured JSON output using OpenAI structured outputs.
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        system_prompt: System prompt
        user_prompt: User prompt
        json_schema: JSON schema for structured output
        temperature: Temperature for response generation
        
    Returns:
        Parsed Python dictionary
    """
    openai.api_key = api_key

    resp = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object", "schema": json_schema},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.model_dump()["content"]


def parse_transcript_to_json(transcript_text: str) -> List[Dict[str, str]]:
    """
    Parse transcript text to JSON format for translation.
    
    Converts format from:
        # 00:00:00 --> 00:01:00
        Text content here
    
    To JSON array:
        [{"ts": "00:00:00 --> 00:01:00", "text": "Text content here"}, ...]
    
    Args:
        transcript_text: Original transcript with timestamps
        
    Returns:
        List of dictionaries with "ts" and "text" fields
    """
    segments = []
    
    # Split by timestamp lines and process pairs
    lines = transcript_text.strip().split('\n')
    current_timestamp = None
    current_text_lines: List[str] = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a timestamp (starts with #)
        if line.startswith('# '):
            # Save previous segment if exists
            if current_timestamp and current_text_lines:
                text_content = '\n'.join(current_text_lines).strip()
                if text_content:
                    segments.append({
                        "ts": current_timestamp.replace('# ', ''),
                        "text": text_content
                    })
            
            # Start new segment
            current_timestamp = line
            current_text_lines = []
        else:
            # Accumulate text lines
            current_text_lines.append(line)
    
    # Handle final segment
    if current_timestamp and current_text_lines:
        text_content = '\n'.join(current_text_lines).strip()
        if text_content:
            segments.append({
                "ts": current_timestamp.replace('# ', ''),
                "text": text_content
            })
    
    return segments


def reconstruct_transcript_from_json(segments: List[Dict[str, str]]) -> str:
    """
    Reconstruct transcript text from JSON format after translation.
    
    Converts JSON array back to original format:
        [{"ts": "00:00:00 --> 00:01:00", "text": "Translated text"}, ...]
    
    Back to:
        # 00:00:00 --> 00:01:00
        Translated text
    
    Args:
        segments: List of dictionaries with "ts" and "text" fields
        
    Returns:
        Reconstructed transcript text with timestamps
    """
    result_lines = []
    
    for segment in segments:
        # Add timestamp line
        result_lines.append(f"# {segment['ts']}")
        
        # Add text content
        text = segment.get('text', '').strip()
        if text:
            result_lines.append(text)
        
        # Add blank line between segments
        result_lines.append('')
    
    # Remove trailing empty line
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)


async def translate_transcript_json(
    api_key: str,
    model: str,
    transcript_json: List[Dict[str, str]],
    target_language: str,
    source_language: str = "auto",
    temperature: float = 0.3,
    progress_callback: Optional[Callable] = None
) -> List[Dict[str, str]]:
    """
    Translate transcript using JSON structured approach from INITIAL.md.
    
    Uses OpenAI structured outputs to ensure JSON format compliance.
    Only translates the "text" field, keeps "ts" unchanged.
    
    Args:
        api_key: OpenAI API key
        model: Language model to use
        transcript_json: List of segments with "ts" and "text" fields
        target_language: Target language for translation
        source_language: Source language (default: "auto")
        temperature: Temperature for translation
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of translated segments with same structure
    """
    openai.api_key = api_key
    
    # Create JSON schema for structured output
    json_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ts": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["ts", "text"]
                }
            }
        },
        "required": ["segments"]
    }
    
    # System prompt for translation
    system_prompt = f"""You are a professional translator. Translate only the "text" field of each segment to {target_language}. 
Keep the "ts" field exactly unchanged. Maintain natural flow and consistency across segments.
Output must be valid JSON following the provided schema."""
    
    # User prompt with JSON data
    user_prompt = f"""Translate the following transcript segments. Only translate the "text" fields to {target_language}, keep "ts" fields unchanged:

{json.dumps({"segments": transcript_json}, ensure_ascii=False, indent=2)}"""
    
    if progress_callback:
        progress_callback(0.1, "Starting translation...")
    
    try:
        from errors import validate_api_key, ValidationError, handle_openai_error
        
        # Validate inputs
        validate_api_key(api_key)
        
        if not target_language.strip():
            raise ValidationError("Target language cannot be empty", field="target_language")
        
        if not transcript_json:
            raise ValidationError("No transcript segments to translate", field="transcript_json")
        
        # Use structured completion for guaranteed JSON output
        result = structured_completion(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            temperature=temperature
        )
        
        if progress_callback:
            progress_callback(0.9, "Processing translation result...")
        
        # Parse result and return segments
        if isinstance(result, str):
            result = json.loads(result)
        
        translated_segments = result.get("segments", [])
        
        # Validate result has same number of segments
        if len(translated_segments) != len(transcript_json):
            raise ValidationError(
                f"Translation returned {len(translated_segments)} segments but expected {len(transcript_json)}",
                field="translation_result"
            )
        
        if progress_callback:
            progress_callback(1.0, "Translation completed!")
        
        return translated_segments
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        else:
            api_error = handle_openai_error(e)
            raise api_error


async def translate_transcript_chunked(
    api_key: str,
    model: str,
    transcript_json: List[Dict[str, str]],
    target_language: str,
    source_language: str = "auto",
    temperature: float = 0.3,
    max_tokens_per_chunk: int = 100000,
    progress_callback: Optional[Callable] = None
) -> List[Dict[str, str]]:
    """
    Translate large transcripts by splitting into chunks to handle token limits.
    
    Args:
        api_key: OpenAI API key
        model: Language model to use
        transcript_json: List of segments to translate
        target_language: Target language
        source_language: Source language
        temperature: Temperature for translation
        max_tokens_per_chunk: Maximum tokens per chunk (rough estimate)
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of translated segments
    """
    if len(transcript_json) == 0:
        return []
    
    # Estimate tokens (rough: 4 chars per token)
    total_chars = sum(len(json.dumps(segment)) for segment in transcript_json)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens <= max_tokens_per_chunk:
        # Single chunk translation
        return await translate_transcript_json(
            api_key, model, transcript_json, target_language, 
            source_language, temperature, progress_callback
        )
    
    # Split into chunks
    chunks = []
    current_chunk: List[Dict[str, str]] = []
    current_chars = 0
    
    for segment in transcript_json:
        segment_chars = len(json.dumps(segment))
        
        if current_chars + segment_chars > max_tokens_per_chunk * 4 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [segment]
            current_chars = segment_chars
        else:
            current_chunk.append(segment)
            current_chars += segment_chars
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Translate chunks
    translated_segments = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress = (i / total_chunks) * 0.9
            progress_callback(progress, f"Translating chunk {i+1}/{total_chunks}")
        
        chunk_result = await translate_transcript_json(
            api_key, model, chunk, target_language,
            source_language, temperature, None
        )
        
        translated_segments.extend(chunk_result)
    
    if progress_callback:
        progress_callback(1.0, "All chunks translated!")
    
    return translated_segments


async def translate_transcript_full(
    api_key: str,
    model: str,
    transcript_text: str,
    target_language: str,
    source_language: str = "auto",
    temperature: float = 0.3,
    progress_callback: Optional[Callable] = None
) -> TranslationResult:
    """
    Full transcript translation workflow implementing INITIAL.md strategy.
    
    1. Parse transcript to JSON format
    2. Translate using structured outputs (chunked if needed)
    3. Reconstruct to original format
    4. Return complete result
    
    Args:
        api_key: OpenAI API key
        model: Language model to use
        transcript_text: Original transcript with timestamps
        target_language: Target language for translation
        source_language: Source language (default: "auto")
        temperature: Temperature for translation
        progress_callback: Optional callback for progress updates
        
    Returns:
        TranslationResult with original and translated text
    """
    start_time = time.time()
    
    if progress_callback:
        progress_callback(0.1, "Parsing transcript to JSON...")
    
    # Step 1: Parse to JSON
    transcript_json = parse_transcript_to_json(transcript_text)
    
    if not transcript_json:
        raise ValueError("No valid transcript segments found")
    
    if progress_callback:
        progress_callback(0.2, f"Found {len(transcript_json)} segments to translate...")
    
    # Step 2: Translate using chunked approach
    translated_json = await translate_transcript_chunked(
        api_key=api_key,
        model=model,
        transcript_json=transcript_json,
        target_language=target_language,
        source_language=source_language,
        temperature=temperature,
        progress_callback=lambda p, m: progress_callback(0.2 + p * 0.7, m) if progress_callback else None
    )
    
    if progress_callback:
        progress_callback(0.9, "Reconstructing translated transcript...")
    
    # Step 3: Reconstruct to original format
    translated_text = reconstruct_transcript_from_json(translated_json)
    
    processing_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(1.0, "Translation completed!")
    
    return TranslationResult(
        original_text=transcript_text,
        translated_text=translated_text,
        target_language=target_language,
        processing_time=processing_time,
        segment_count=len(transcript_json)
    )


def get_language_code(language_name: str) -> str:
    """
    Get language code from language name using config.yaml mappings.
    
    Args:
        language_name: Language name (e.g., "Japanese", "English")
        
    Returns:
        Language code (e.g., "ja", "en")
    """
    try:
        config = load_config()
        translation_languages = config.get('translation_languages', {})
        return translation_languages.get(language_name, language_name.lower())
    except (FileNotFoundError, ValueError, KeyError):
        # Fallback to common language codes
        language_map = {
            'japanese': 'ja',
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'chinese': 'zh'
        }
        return language_map.get(language_name.lower(), language_name.lower())


# CLI Interface Implementation (INITIAL.md requirement)
def main():
    """Command-line interface for LLM functionality."""
    parser = argparse.ArgumentParser(description="LLM utilities CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with context injection")
    chat_parser.add_argument("--api-key", required=True, help="OpenAI API key")
    chat_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    chat_parser.add_argument("--question", required=True, help="Question to ask")
    chat_parser.add_argument("--context", help="Context text file path")
    chat_parser.add_argument("--context-text", help="Context text directly")
    chat_parser.add_argument("--system-message", help="Custom system message")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    
    # Translation command
    translate_parser = subparsers.add_parser("translate", help="Translate transcript")
    translate_parser.add_argument("--api-key", required=True, help="OpenAI API key")
    translate_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    translate_parser.add_argument("--input", required=True, help="Input transcript file")
    translate_parser.add_argument("--output", help="Output file path")
    translate_parser.add_argument("--target-language", required=True, help="Target language")
    translate_parser.add_argument("--source-language", default="auto", help="Source language")
    translate_parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    
    # Parse JSON command
    parse_parser = subparsers.add_parser("parse", help="Parse transcript to JSON")
    parse_parser.add_argument("--input", required=True, help="Input transcript file")
    parse_parser.add_argument("--output", help="Output JSON file")
    
    # Reconstruct command
    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct from JSON")
    reconstruct_parser.add_argument("--input", required=True, help="Input JSON file")
    reconstruct_parser.add_argument("--output", help="Output transcript file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "chat":
            # Load context
            context_text = ""
            if args.context:
                with open(args.context, 'r', encoding='utf-8') as f:
                    context_text = f.read()
            elif args.context_text:
                context_text = args.context_text
            
            # Load system message from config if not provided
            system_message = args.system_message
            if not system_message:
                try:
                    config = load_config()
                    system_message = config.get('system_message', '')
                except (FileNotFoundError, ValueError, KeyError):
                    system_message = "You are a helpful assistant."
            
            # Chat with context
            if context_text:
                response = chat_with_context(
                    api_key=args.api_key,
                    model=args.model,
                    question=args.question,
                    context_text=context_text,
                    system_message=system_message,
                    temperature=args.temperature
                )
            else:
                response, _ = chat_completion(
                    api_key=args.api_key,
                    model=args.model,
                    message=args.question,
                    system_message=system_message,
                    temperature=args.temperature
                )
            
            print("Assistant Response:")
            print("=" * 50)
            print(response)
            
        elif args.command == "translate":
            # Load transcript
            with open(args.input, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            # Translate
            print(f"Translating to {args.target_language}...")
            result = asyncio.run(translate_transcript_full(
                api_key=args.api_key,
                model=args.model,
                transcript_text=transcript_text,
                target_language=args.target_language,
                source_language=args.source_language,
                temperature=args.temperature,
                progress_callback=lambda p, m: print(f"Progress: {p*100:.1f}% - {m}")
            ))
            
            # Output results
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result.translated_text)
                print(f"Translation saved to: {args.output}")
            else:
                print("Translated Text:")
                print("=" * 50)
                print(result.translated_text)
            
            print("=" * 50)
            print(f"Segments processed: {result.segment_count}")
            print(f"Processing time: {result.processing_time:.2f} seconds")
            
        elif args.command == "parse":
            # Parse transcript to JSON
            with open(args.input, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            segments = parse_transcript_to_json(transcript_text)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
                print(f"JSON saved to: {args.output}")
            else:
                print("Parsed JSON:")
                print("=" * 50)
                print(json.dumps({"segments": segments}, ensure_ascii=False, indent=2))
            
            print(f"Found {len(segments)} segments")
            
        elif args.command == "reconstruct":
            # Reconstruct from JSON
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            transcript_text = reconstruct_transcript_from_json(segments)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                print(f"Transcript saved to: {args.output}")
            else:
                print("Reconstructed Transcript:")
                print("=" * 50)
                print(transcript_text)
            
            print(f"Reconstructed {len(segments)} segments")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())