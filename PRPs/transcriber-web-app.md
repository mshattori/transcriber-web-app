name: "Transcriber Web App - Complete Implementation PRP"
description: |

## Purpose
Comprehensive implementation plan for a browser-based audio transcription web application using Gradio, OpenAI APIs, and structured audio processing with translation capabilities.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Build a complete browser-based transcription web application that allows users to upload large audio files (up to 1 hour), automatically chunk them to bypass OpenAI's 25MB API limit, transcribe using OpenAI Whisper, optionally translate the results, and provide an interactive chat interface for discussing the transcribed content.

## Why
- **Business value**: Provides accessible transcription service for long-form audio content
- **User impact**: Enables processing of hour-long audio files that exceed API limits
- **Integration**: Modular design allows for future extensions and API integrations
- **Problems solved**: Handles large file transcription, provides translation capabilities, offers interactive analysis of transcribed content

## What
A Gradio-based web application with real-time progress tracking, drag-and-drop file upload, intelligent audio chunking, OpenAI API integration for transcription and translation, chat functionality, and comprehensive history management.

### Success Criteria
- [ ] Successfully upload and process audio files up to 1 hour in length
- [ ] Automatic chunking with 2-second overlap to handle 25MB API limits
- [ ] Real-time progress updates during transcription
- [ ] Optional translation with consistent terminology preservation
- [ ] Interactive chat functionality using transcribed content as context
- [ ] Comprehensive history management with job persistence
- [ ] Downloadable results in multiple formats (.txt, .zip with original/translated)
- [ ] Configurable settings persistence in browser localStorage

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://www.gradio.app/guides/blocks-and-event-listeners/
  why: Core Gradio Blocks patterns for event handling and component interaction
  
- url: https://www.gradio.app/guides/state-in-blocks#browser-state
  why: Browser localStorage integration for API keys and settings persistence
  
- url: https://platform.openai.com/docs/guides/speech-to-text
  why: OpenAI Whisper API documentation including parameters and limits
  
- url: https://platform.openai.com/docs/guides/structured-outputs
  why: Structured JSON outputs for translation workflow
  
- file: examples/transcribe.py
  why: Basic transcription pattern to extend and enhance
  
- file: examples/llm.py
  why: Chat completion and structured output patterns to follow

- docfile: CLAUDE.md
  why: Project conventions, file organization, testing requirements
```

### Current Codebase Tree
```bash
transcriber-web-app/
├── CLAUDE.md
├── INITIAL.md
├── examples/
│   ├── transcribe.py        # Basic transcription example
│   └── llm.py               # LLM chat and structured output example
└── PRPs/
    └── templates/
        └── prp_base.md
```

### Desired Codebase Tree with Files to be Added
```bash
transcriber-web-app/
├── src/
│   ├── app.py              # Main Gradio application (UI/routing only)
│   ├── transcribe.py       # Whisper/OpenAI transcription logic
│   ├── llm.py              # OpenAI Chat/translation functionality
│   ├── util.py             # Audio splitting utilities (pydub)
│   └── config.yaml         # Model configuration and system messages
├── tests/
│   ├── test_transcribe.py  # Unit tests for transcription
│   ├── test_llm.py         # Unit tests for LLM functionality
│   └── test_util.py        # Unit tests for utilities
├── data/                   # Job history storage (auto-created)
├── examples/               # Existing examples
├── venv_linux/            # Virtual environment
└── requirements.txt        # Dependencies
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: OpenAI API has 25MB file size limit for transcriptions
# Solution: Use pydub to split into N-minute chunks with 2-second overlap

# CRITICAL: Gradio file upload progress is only available AFTER upload completes
# Solution: Use gr.Progress() for processing steps, not upload itself

# CRITICAL: OpenAI structured outputs require specific JSON schema format
# Example: {"type": "object", "properties": {...}, "required": [...]}

# CRITICAL: Browser localStorage requires JSON serializable data
# Use gr.BrowserState for API keys and settings persistence

# CRITICAL: pydub requires ffmpeg for MP3 support
# Ensure ffmpeg is available in deployment environment

# CRITICAL: Large audio files can cause memory issues
# Use streaming/chunked processing approach

# CRITICAL: Gradio State objects must be deepcopy-able
# Simple data types work, complex objects need special handling
```

## Implementation Blueprint

### Data Models and Structure

Create core data models ensuring type safety and consistency:

```python
# Pydantic models for data validation
class TranscriptionJob(BaseModel):
    job_id: str
    filename: str
    file_size: int
    language: str
    model: str
    chunk_duration: int
    translation_enabled: bool
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime]
    total_chunks: int
    processed_chunks: int

class TranscriptionChunk(BaseModel):
    chunk_id: str
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float]

class TranslationResult(BaseModel):
    original_text: str
    translated_text: str
    target_language: str
    timestamp: str
```

### List of Tasks to be Completed (In Order)

```yaml
Task 1 - Setup Project Structure:
CREATE src/ directory and required files:
  - COPY examples/transcribe.py to src/transcribe.py
  - COPY examples/llm.py to src/llm.py
  - CREATE src/app.py (main Gradio application)
  - CREATE src/util.py (audio utilities)
  - CREATE src/config.yaml (configuration)
  - CREATE tests/ directory with test files
  - CREATE requirements.txt with dependencies

Task 2 - Create Configuration System:
MODIFY src/config.yaml:
  - DEFINE audio_models list (whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe)
  - DEFINE language_models list (gpt-4o-mini, gpt-4o, gpt-4o-speed)
  - DEFINE default system_message for chat functionality
  - INCLUDE language options and defaults

Task 3 - Implement Audio Utilities:
CREATE src/util.py:
  - FUNCTION split_audio(file_path, chunk_minutes, overlap_seconds) using pydub
  - FUNCTION validate_audio_file(file_path) for format/size checking
  - FUNCTION estimate_processing_time(file_size, chunk_duration)
  - HANDLE memory-efficient audio processing for large files

Task 4 - Enhance Transcription Module:
MODIFY src/transcribe.py:
  - EXTEND transcribe function to handle chunked processing
  - ADD yield-based progress reporting for gr.Progress integration
  - IMPLEMENT retry logic with exponential backoff for API errors
  - ADD timestamp preservation and formatting
  - INCLUDE confidence scores and metadata handling

Task 5 - Enhance LLM Module:
MODIFY src/llm.py:
  - ADD translation function using structured outputs
  - IMPLEMENT chat_with_context for transcription-aware conversations
  - ADD JSON schema definitions for translation workflow
  - INCLUDE error handling for translation edge cases

Task 6 - Create Main Gradio Application:
CREATE src/app.py:
  - DESIGN UI layout with gr.Blocks following INITIAL.md specifications
  - IMPLEMENT drag-and-drop file upload with gr.File
  - CREATE settings modal with gr.Modal for API keys and configuration
  - ADD progress tracking with gr.Progress for real-time updates
  - IMPLEMENT chat interface with gr.Chatbot
  - CREATE history management with gr.Dataframe
  - ADD download functionality for results

Task 7 - Implement Job Management:
ADD to src/app.py:
  - CREATE job persistence in data/{YYYY-MM-DD}/{job_id}/ structure
  - IMPLEMENT job status tracking and resume capability
  - ADD job history display and management
  - INCLUDE job cleanup and deletion functionality

Task 8 - Add Browser State Management:
ENHANCE src/app.py:
  - IMPLEMENT gr.BrowserState for API key persistence
  - ADD settings persistence (models, languages, preferences)
  - INCLUDE localStorage integration for user preferences
  - CREATE settings load/save functionality

Task 9 - Create Translation Workflow:
INTEGRATE translation features:
  - IMPLEMENT full-text translation using JSON structured approach
  - ADD translation toggle and target language selection
  - CREATE dual-tab display for original/translated content
  - IMPLEMENT ZIP download for both versions

Task 10 - Add Error Handling and Validation:
ENHANCE all modules:
  - ADD comprehensive error handling with user-friendly messages
  - IMPLEMENT input validation for all user inputs
  - ADD rate limiting awareness and retry logic
  - CREATE graceful degradation for API failures

Task 11 - Implement Testing Suite:
CREATE comprehensive tests:
  - UNIT tests for transcribe.py functions
  - UNIT tests for llm.py functions  
  - UNIT tests for util.py functions
  - INTEGRATION tests for full workflows
  - MOCK API responses for testing
```

### Per Task Pseudocode

```python
# Task 3: Audio Utilities Implementation
def split_audio(file_path: str, chunk_minutes: int, overlap_seconds: int = 2) -> List[str]:
    """
    Split audio file into overlapping chunks to handle API limits.
    
    Args:
        file_path: Path to input audio file
        chunk_minutes: Duration of each chunk in minutes
        overlap_seconds: Overlap between chunks in seconds
    
    Returns:
        List of chunk file paths
    """
    # PATTERN: Use pydub AudioSegment for cross-format support
    audio = AudioSegment.from_file(file_path)
    
    chunk_length_ms = chunk_minutes * 60 * 1000
    overlap_ms = overlap_seconds * 1000
    
    chunks = []
    start = 0
    chunk_num = 1
    
    while start < len(audio):
        # CRITICAL: Ensure overlap with previous chunk (except first)
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        
        # PATTERN: Save chunks with numbered filenames
        chunk_path = f"temp_chunk_{chunk_num:02d}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
        
        # CRITICAL: Move start position with overlap consideration
        start = end - overlap_ms if end < len(audio) else end
        chunk_num += 1
    
    return chunks

# Task 4: Enhanced Transcription with Progress
async def transcribe_chunked(
    chunks: List[str], 
    api_key: str, 
    model: str, 
    language: str,
    progress: callable = None
) -> str:
    """
    Transcribe audio chunks with progress reporting.
    """
    # PATTERN: Use yield for real-time progress updates
    results = []
    total_chunks = len(chunks)
    
    for i, chunk_path in enumerate(chunks):
        if progress:
            progress(i / total_chunks, f"Processing chunk {i+1}/{total_chunks}")
        
        # RETRY PATTERN: Exponential backoff for API errors
        for attempt in range(3):
            try:
                result = await transcribe_single_chunk(chunk_path, api_key, model, language)
                results.append(result)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                await asyncio.sleep(2 ** attempt)
    
    # CRITICAL: Merge overlapping chunks intelligently
    return merge_transcription_results(results)

# Task 5: Translation with Structured Output
def translate_full_text(
    transcript: str, 
    target_language: str, 
    api_key: str, 
    model: str
) -> str:
    """
    Translate complete transcript using structured JSON approach.
    """
    # PATTERN: Parse timestamps and text into JSON structure
    segments = parse_transcript_segments(transcript)
    
    # CRITICAL: Use OpenAI structured outputs for reliable JSON
    json_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["timestamp", "text"]
                }
            }
        },
        "required": ["segments"]
    }
    
    # CRITICAL: Translate only text fields, preserve timestamps
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Translate only the 'text' fields to {target_language}. Keep timestamps unchanged."},
            {"role": "user", "content": json.dumps({"segments": segments})}
        ],
        response_format={"type": "json_object", "schema": json_schema}
    )
    
    return reconstruct_transcript(response.choices[0].message.content)
```

### Integration Points
```yaml
GRADIO COMPONENTS:
  - gr.File: Drag-and-drop upload with file validation
  - gr.Progress: Real-time transcription progress
  - gr.BrowserState: API key and settings persistence
  - gr.Modal: Settings and history management
  - gr.Chatbot: Interactive chat with transcript context

OPENAI API:
  - /v1/audio/transcriptions: Whisper transcription
  - /v1/chat/completions: LLM chat and translation
  - structured outputs: JSON-formatted translation

PYDUB INTEGRATION:
  - AudioSegment: Cross-format audio loading
  - Audio splitting: Fixed-duration chunking
  - Export functionality: MP3 output format

FILESYSTEM:
  - data/{YYYY-MM-DD}/{job_id}/: Job persistence
  - Temporary files: Chunked audio storage
  - Config loading: YAML configuration
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
# Use venv_linux as specified in CLAUDE.md
source venv_linux/bin/activate

# Style and syntax checking
ruff check src/ --fix
mypy src/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE comprehensive test suite with pytest
# tests/test_transcribe.py
def test_audio_splitting():
    """Test audio file splitting functionality"""
    # Test with sample audio file
    chunks = split_audio("test_audio.mp3", chunk_minutes=2)
    assert len(chunks) > 0
    assert all(os.path.exists(chunk) for chunk in chunks)

def test_transcription_with_mock():
    """Test transcription with mocked OpenAI API"""
    with mock.patch('openai.audio.transcriptions.create') as mock_transcribe:
        mock_transcribe.return_value = "Test transcription"
        result = transcribe("test_audio.mp3", "test_key")
        assert result == "Test transcription"

def test_translation_structured_output():
    """Test translation using structured outputs"""
    # Test JSON schema validation and translation workflow
    pass

def test_chat_with_context():
    """Test chat functionality with transcript context"""
    # Test context injection and conversation flow
    pass
```

```bash
# Run and iterate until passing:
pytest tests/ -v

# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Tests
```bash
# Test the complete application
cd src/
python app.py

# Manual testing checklist:
# 1. File upload works with drag-and-drop
# 2. Settings modal saves API key to localStorage
# 3. Transcription processes with progress updates
# 4. Translation works with JSON structured output
# 5. Chat functionality uses transcript context
# 6. History management persists jobs
# 7. Download functionality generates correct files
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] Manual UI testing successful
- [ ] File upload and processing works end-to-end
- [ ] API error handling graceful
- [ ] Settings persistence functional
- [ ] Translation workflow complete
- [ ] Chat context integration working
- [ ] Job history management operational

---

## Anti-Patterns to Avoid
- ❌ Don't split audio without overlap (creates context loss)
- ❌ Don't ignore OpenAI API rate limits and 25MB file size limits
- ❌ Don't store API keys on server-side (use browser localStorage only)
- ❌ Don't use sync functions for API calls (use async/await)
- ❌ Don't hardcode file paths (use proper temp file handling)
- ❌ Don't skip error handling for API failures
- ❌ Don't create overly large files without memory considerations
- ❌ Don't ignore CLAUDE.md conventions for testing and file organization

## Critical Implementation Notes

### Audio Processing Memory Management
Use streaming approaches for large files. Process chunks individually rather than loading entire audio file into memory.

### API Key Security
Never store API keys server-side. Use gr.BrowserState with localStorage for client-side persistence only.

### Error Recovery
Implement comprehensive retry logic with exponential backoff for OpenAI API calls. Handle rate limits gracefully.

### Translation Consistency
Use full-text translation rather than chunk-by-chunk to maintain terminology consistency and context preservation.

### Progress Tracking
Provide detailed progress updates for long-running operations. Use gr.Progress with descriptive messages.

### File Management
Implement proper cleanup of temporary files. Use organized directory structure for job persistence.

**Confidence Score: 9/10** - This PRP provides comprehensive context, detailed implementation guidance, and addresses all major technical challenges. The modular approach and extensive documentation references should enable successful one-pass implementation.