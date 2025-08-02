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

# CRITICAL: Use tempfile.SpooledTemporaryFile for large file streaming
# Minimizes memory usage when processing hour-long audio files

# CRITICAL: Chunk naming convention must be chunk_01.mp3, chunk_02.mp3, etc.
# Consistent naming required for proper processing workflow

# CRITICAL: Translation files must use format transcript.<lang>.txt
# Where <lang> is target language code (e.g., transcript.ja.txt)

# CRITICAL: JSON translation schema uses "ts" and "text" fields
# Format: [{"ts": "00:00:00-00:01:00", "text": "..."}] for compatibility
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
  - IMPLEMENT real-time updates using gr.Textbox.update(value += new_text) pattern
  - ADD CLI interface using argparse in __main__ block for testing
  - USE tempfile.SpooledTemporaryFile for memory-efficient large file handling
  - FOLLOW chunk naming convention: chunk_01.mp3, chunk_02.mp3, etc.

Task 5 - Enhance LLM Module:
MODIFY src/llm.py:
  - ADD translation function using structured outputs
  - IMPLEMENT chat_with_context for transcription-aware conversations
  - ADD JSON schema definitions for translation workflow (use "ts" and "text" fields)
  - INCLUDE error handling for translation edge cases
  - IMPLEMENT 3-message chat pattern: system → user(context) → user(question)
  - ADD CLI interface using argparse in __main__ block for testing
  - SUPPORT translation file naming: transcript.<lang>.txt format

Task 6 - Create Main Gradio Application:
CREATE src/app.py following EXACT INITIAL.md layout specifications:
  - IMPLEMENT gr.Blocks with light theme and custom CSS
  - CREATE single-column layout (gr.Column) with specific component order:
    1. gr.Row containing: gr.File upload + model dropdowns + language setting + chunk slider + translation toggle + target language selector
    2. gr.Progress with real-time updates and time estimation  
    3. Processing log panel (gr.Textbox, lines=4, CSS: max-height: 120px; overflow-y: auto)
    4. Results display panel (gr.Markdown, full width, timestamp CSS styling)
    5. Download buttons section (gr.Button for .txt and .zip)
    6. Collapsible chat panel (gr.Chatbot) at bottom
  - IMPLEMENT settings modal (gear icon → gr.Modal with API key, models, system message)
  - IMPLEMENT history modal (history button → gr.Modal with gr.Dataframe job listing)
  - ADD notification system (gr.Alert for errors, gr.Notification for success)
  - APPLY CSS for timestamp styling: .timestamp { font-size: 0.85rem; color: #888; }
  - ADD file size validation with 500MB warning
  - IMPLEMENT real-time result updates with gr.Textbox.update(value += new_text)
  - FOLLOW 4-step processing sequence: split → transcribe chunks → real-time updates → translation

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
        
        # PATTERN: Save chunks with numbered filenames (INITIAL.md requirement)
        chunk_path = f"chunk_{chunk_num:02d}.mp3"
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
    
    # CRITICAL: Use OpenAI structured outputs with INITIAL.md schema format
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

# Task 6: Main Gradio Application UI Layout
def create_main_interface():
    """
    Create the main Gradio interface following INITIAL.md specifications.
    """
    # CRITICAL: Custom CSS for timestamp styling and layout
    custom_css = """
    .timestamp { 
        font-size: 0.85rem; 
        color: #888; 
    }
    .processing-log {
        max-height: 120px;
        overflow-y: auto;
    }
    .toast-notification {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    """
    
    with gr.Blocks(css=custom_css, theme="default") as demo:
        # PATTERN: Load config.yaml for dropdown options
        config = load_config("src/config.yaml")
        
        # PATTERN: Browser state for settings persistence
        browser_state = gr.BrowserState({
            "api_key": "",
            "audio_model": config["audio_models"][0],
            "language_model": config["language_models"][0],
            "system_message": config["system_message"]
        })
        
        # LAYOUT: Single column as specified in INITIAL.md
        with gr.Column():
            # 1. UPLOAD & CONTROLS SECTION (gr.Row for space efficiency)
            with gr.Row():
                file_upload = gr.File(
                    label="Upload Audio File",
                    file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                    file_count="single"
                )
                audio_model = gr.Dropdown(
                    choices=config["audio_models"],
                    value=config["audio_models"][0],
                    label="Audio Model"
                )
                language = gr.Dropdown(
                    choices=["auto", "en", "ja", "es", "fr", "de", "zh"],
                    value="auto",
                    label="Language"
                )
                chunk_minutes = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Chunk Duration (minutes)"
                )
                translation_enabled = gr.Checkbox(
                    label="Enable Translation",
                    value=False
                )
                target_language = gr.Dropdown(
                    choices=["Japanese", "English", "Spanish", "French", "German", "Chinese"],
                    value="Japanese",
                    label="Target Language",
                    visible=False  # Show only when translation enabled
                )
            
            # CRITICAL: Advanced settings in accordion as specified
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.0, label="Temperature")
                timestamp_output = gr.Checkbox(True, label="Include Timestamps")
            
            # 2. PROGRESS SECTION
            progress = gr.Progress()
            
            # 3. PROCESSING LOG PANEL (minimal height as specified)
            processing_log = gr.Textbox(
                label="Processing Log",
                lines=4,
                max_lines=4,
                interactive=False,
                elem_classes=["processing-log"]
            )
            
            # 4. RESULTS DISPLAY PANEL (full width for readability)
            with gr.Tab("Results") as results_tab:
                transcript_display = gr.Markdown(
                    label="Transcript",
                    value="Upload an audio file to begin transcription..."
                )
                copy_button = gr.Button("Copy to Clipboard", size="sm")
            
            # CRITICAL: Tab view for original/translated when translation enabled
            with gr.Tab("Translation", visible=False) as translation_tab:
                translated_display = gr.Markdown(label="Translated Text")
            
            # 5. DOWNLOAD SECTION
            with gr.Row():
                download_txt = gr.File(label="Download Transcript (.txt)", visible=False)
                download_zip = gr.File(label="Download Package (.zip)", visible=False)
                stats_display = gr.Textbox(
                    label="Statistics",
                    value="Duration: -- | Words: -- | WPM: --",
                    interactive=False
                )
            
            # 6. CHAT PANEL (collapsible, bottom placement)
            with gr.Accordion("Chat with Transcript", open=False):
                chatbot = gr.Chatbot(label="AI Assistant")
                chat_input = gr.Textbox(label="Your Question", placeholder="Ask about the transcript...")
                with gr.Row():
                    chat_submit = gr.Button("Send", variant="primary")
                    chat_clear = gr.Button("Clear History")
        
        # MODAL: Settings panel (gear icon trigger)
        with gr.Row():
            settings_button = gr.Button("⚙️ Settings", size="sm")
            history_button = gr.Button("📁 History", size="sm")
        
        # CRITICAL: Settings modal as specified in INITIAL.md
        with gr.Modal(visible=False) as settings_modal:
            with gr.Column():
                gr.Markdown("## Settings")
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-..."
                )
                settings_audio_model = gr.Dropdown(
                    choices=config["audio_models"],
                    label="Default Audio Model"
                )
                settings_language_model = gr.Dropdown(
                    choices=config["language_models"],
                    label="Default Language Model"
                )
                system_message_edit = gr.Textbox(
                    label="System Message",
                    value=config["system_message"],
                    lines=3
                )
                with gr.Row():
                    save_settings = gr.Button("Save", variant="primary")
                    reset_settings = gr.Button("Reset to Defaults")
        
        # MODAL: History management
        with gr.Modal(visible=False) as history_modal:
            with gr.Column():
                gr.Markdown("## Job History")
                history_dataframe = gr.Dataframe(
                    headers=["Filename", "Date", "Duration", "Language", "Status"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False
                )
                with gr.Row():
                    reload_history = gr.Button("Reload")
                    delete_job = gr.Button("Delete Selected", variant="stop")
    
    return demo

# INITIAL.md Processing Sequence Implementation
def implement_processing_workflow():
    """
    Implement the exact 4-step processing sequence from INITIAL.md.
    """
    # STEP 1: Split audio into chunks with pydub
    def split_phase(audio_file, chunk_minutes):
        # Use tempfile.SpooledTemporaryFile for memory efficiency
        with tempfile.SpooledTemporaryFile() as temp_file:
            chunks = split_audio(audio_file, chunk_minutes, overlap_seconds=2)
            # Generate chunk_01.mp3, chunk_02.mp3, etc.
            return chunks
    
    # STEP 2: Transcribe chunks with yield-based progress
    def transcribe_phase(chunks, progress_callback):
        results = []
        for i, chunk in enumerate(chunks):
            # Update progress with gr.Progress
            progress_callback(i / len(chunks), f"Processing chunk {i+1}/{len(chunks)}")
            result = transcribe_chunk(chunk)
            results.append(result)
            # STEP 3: Real-time updates during processing
            yield result  # For real-time display updates
        return results
    
    # STEP 4: Translation phase (after all chunks complete)
    def translation_phase(full_transcript, target_language):
        # Use JSON structured format with "ts" and "text" fields
        segments = parse_transcript_to_json(full_transcript)
        translated = translate_structured(segments, target_language)
        return reconstructed_transcript(translated)

# CLI Interface Implementation (INITIAL.md requirement)
def add_cli_interfaces():
    """
    Add argparse CLI interfaces to transcribe.py and llm.py.
    """
    # In transcribe.py __main__ block:
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Audio transcription CLI")
        parser.add_argument("--file", required=True, help="Audio file path")
        parser.add_argument("--model", default="whisper-1", help="Model to use")
        parser.add_argument("--language", default="auto", help="Language code")
        args = parser.parse_args()
        # Execute transcription and print results
    
    # In llm.py __main__ block:
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="LLM operations CLI")
        parser.add_argument("--translate", help="Translate text file")
        parser.add_argument("--chat", help="Chat with context file")
        parser.add_argument("--target-lang", default="Japanese", help="Target language")
        args = parser.parse_args()
        # Execute requested operation

# File Validation with 500MB Warning (INITIAL.md requirement)
def validate_file_with_warning(file_path):
    """
    Validate file and show 500MB warning as specified in INITIAL.md.
    """
    file_size = os.path.getsize(file_path)
    max_size = 500 * 1024 * 1024  # 500MB in bytes
    
    if file_size > max_size:
        return gr.Alert(
            "Warning: File size exceeds 500MB. Processing may take longer and consume more memory.",
            title="Large File Warning",
            dismissible=True
        )
    return None

# Chat Context Injection Pattern (INITIAL.md requirement)
def format_chat_messages(context_text, user_question, system_message):
    """
    Format chat messages according to INITIAL.md 3-message pattern.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context_text}"},
        {"role": "user", "content": user_question}
    ]
    return messages
```

### UI Layout Requirements (INITIAL.md Specifications)

```yaml
MAIN LAYOUT STRUCTURE (gr.Blocks with custom CSS):
  Theme: Light style theme
  Layout: Single column (gr.Column) top-to-bottom arrangement

  1. UPLOAD & CONTROLS SECTION (gr.Row for space efficiency):
     - gr.File: Drag & drop upload box with file validation
     - Model selection dropdown (audio_models from config.yaml)
     - Language setting (IETF BCP-47, default "auto")
     - N-minute chunk duration dropdown (1-10 minutes)
     - Translation toggle switch
     - Advanced settings accordion for temperature, timestamps, etc.

  2. PROGRESS SECTION:
     - gr.Progress: Real-time upload & processing progress
     - Estimated time remaining display

  3. PROCESSING LOG PANEL:
     - gr.Textbox or gr.Markdown with minimal height (lines=4)
     - CSS: max-height: 120px; overflow-y: auto;
     - Optional gr.Accordion for collapsible behavior
     - Auto-scroll to latest updates

  4. RESULTS DISPLAY PANEL:
     - gr.HTML/gr.Markdown: Full width display for readability
     - Timestamp formatting: <span class="timestamp">...</span>
     - CSS: .timestamp { font-size: 0.85rem; color: #888; }
     - Copy to clipboard button
     - Tab/split view for original/translated content when translation enabled
     - Statistics display: duration, word count, WPM

  5. DOWNLOAD SECTION:
     - gr.Button for transcript.txt (no translation) or transcript.zip (with translation)

  6. CHAT PANEL (Bottom of page):
     - gr.Chatbot: Collapsible with toggle button
     - Context injection from current transcript
     - Clear history button
     - Space-saving design

MODAL INTERFACES:

  SETTINGS MODAL (gr.Modal triggered by gear icon):
    - Position: Top-right gear icon button
    - Layout: gr.Column structure
    - Components:
      * OpenAI API Key input (required, stored in localStorage)
      * Audio model dropdown (from audio_models in config.yaml)
      * Language model dropdown (from language_models in config.yaml)
      * System message gr.Textbox (editable, localStorage persistence)
      * Save button (write to localStorage)
      * Reset to defaults button (restore config.yaml values)

  HISTORY MODAL (gr.Modal triggered by history button):
    - gr.Dataframe or gr.Dataset for job listings
    - Columns: filename, date, duration, language, status
    - Actions: view, download, delete
    - Job persistence: data/{YYYY-MM-DD}/{job_id}/ structure

NOTIFICATION SYSTEM:
  - gr.Alert: Fatal errors only (persistent until dismissed)
  - gr.Notification: Success/info messages (toast style, auto-fade)
  - CSS: Custom toast positioning (bottom-right)
  - Minimize fixed UI elements to maximize results panel space
```

### Config.yaml Format (INITIAL.md Specification)

```yaml
# src/config.yaml - Exact format required
audio_models:
  - whisper-1
  - gpt-4o-mini-transcribe
  - gpt-4o-transcribe

language_models:
  - gpt-4o-mini
  - gpt-4o
  - gpt-4o-speed

system_message: |
  あなたはプロフェッショナルで親切な文字起こしアシスタントです。
  ユーザの要求に簡潔かつ正確に答えてください。

# Additional configuration options for implementation
default_language: "auto"
default_translation_language: "Japanese"
default_chunk_minutes: 5
max_file_size_mb: 500
supported_formats:
  - .mp3
  - .wav
  - .m4a
  - .flac
  - .ogg

# Translation languages and their file suffixes
translation_languages:
  Japanese: "ja"
  English: "en"
  Spanish: "es"
  French: "fr"
  German: "de"
  Chinese: "zh"

# Transcript output format
timestamp_format: "# HH:MM:SS --> HH:MM:SS"

# Processing sequence (4 steps from INITIAL.md)
processing_steps:
  1: "Split audio into chunks with pydub"
  2: "Transcribe chunks with yield-based progress"
  3: "Real-time markdown updates via gr.Textbox.update"
  4: "Full-text translation after all chunks complete"
```

### Integration Points
```yaml
GRADIO COMPONENTS:
  - gr.File: Drag-and-drop upload with file validation
  - gr.Progress: Real-time transcription progress  
  - gr.BrowserState: API key and settings persistence
  - gr.Modal: Settings and history management
  - gr.Chatbot: Interactive chat with transcript context
  - gr.Accordion: Collapsible sections for space optimization
  - gr.Row/gr.Column: Layout structure as specified in INITIAL.md

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

**Confidence Score: 9.5/10** - This PRP now provides comprehensive coverage of ALL INITIAL.md requirements including previously missing CLI interfaces, real-time update patterns, memory management strategies, translation file naming conventions, exact JSON schema formats, 4-step processing sequence, file validation warnings, and chat context injection patterns. The complete context and detailed implementation guidance should enable highly successful one-pass implementation.