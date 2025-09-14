"""
Main Gradio application for transcriber web app.

Implements the exact UI layout and functionality specified in INITIAL.md.
Supports environment-based configuration for production, testing, and mock UI modes.
"""

import json
import logging
import os
import traceback
from typing import Any

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Handler imports for separation of UI and business logic
from .config import AppConfig
from .errors import (
    APIError,
    FileError,
    IntegratedDisplayError,
    TranscriberError,
    TranslationError,
    ValidationError,
    get_user_friendly_message,
    validate_api_key,
)
from .handlers import (
    AudioHandler,
    ChatHandler,
    HistoryHandler,
    MockAudioHandler,
    MockChatHandler,
    MockHistoryHandler,
    MockSettingsHandler,
    SettingsHandler,
)
from .llm import chat_completion, chat_with_context, get_language_code

# Legacy imports for backward compatibility (will be removed gradually)
from .file_manager import create_download_package, get_display_content_from_job, load_job_files
from .util import (
    load_config,
    find_job_directory,
)

# Custom CSS for modern redesigned UI
CUSTOM_CSS = """
.status-container {
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #f8f9fa;
    margin: 10px 0;
}

.progress-dots {
    font-size: 18px;
    letter-spacing: 2px;
    margin-bottom: 8px;
    color: #007bff;
}

.progress-bar-container {
    background: #e9ecef;
    border-radius: 10px;
    height: 8px;
    margin-bottom: 8px;
}

.progress-bar {
    background: linear-gradient(90deg, #007bff, #0056b3);
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s ease;
}

.status-message {
    font-weight: 500;
    color: #495057;
}

.tab-content {
    padding: 20px 0;
}

.upload-area {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 40px;
    text-align: center;
    background: #f8f9fa;
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #007bff;
    background: #e3f2fd;
}

.config-section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    margin-bottom: 20px;
}

.results-container {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
}

.chat-container {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
}

.job-selector {
    background: white;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #e0e0e0;
}

.job-selector input[type="radio"] {
    margin-right: 10px;
}

.job-selector label {
    display: block !important;
    margin-bottom: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    transition: background-color 0.2s ease;
}

.job-selector label:hover {
    background-color: #f8f9fa;
}

.job-selector input[type="radio"]:checked + span {
    background-color: #e3f2fd;
    font-weight: 500;
}

.job-details {
    margin-top: 15px;
}

/* Timestamp styling for display */
.timestamp {
    font-size: 0.85rem;
    color: #888;
    font-weight: 500;
}
"""

# Global state management
class AppState:
    def __init__(self):
        self.current_job_id: str | None = None
        self.current_transcript: str | None = None
        self.current_translation: str | None = None
        self.chat_history: list[dict[str, str]] = []
        self.processing_progress: float = 0.0
        self.processing_message: str = ""

app_state = AppState()

# Chunk duration options as dropdown choices
CHUNK_DURATIONS = ["1 min", "2 min", "3 min", "5 min", "10 min"]

def create_status_html(current_chunk: int, total_chunks: int, message: str) -> str:
    """Create HTML for status indicator with progress dots and bar."""
    if total_chunks == 0:
        return """
        <div class="status-container">
            <div class="status-message">Ready to process audio file</div>
        </div>
        """

    progress_pct = (current_chunk / total_chunks) * 100
    dots = "●" * current_chunk + "○" * (total_chunks - current_chunk)

    return f"""
    <div class="status-container">
        <div class="progress-dots">{dots}</div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: {progress_pct}%"></div>
        </div>
        <div class="status-message">{message}</div>
    </div>
    """

def load_default_settings() -> dict[str, Any]:
    """Load default settings from config.yaml."""
    try:
        config = load_config()
        # Load API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY", "")
        return {
            "api_key": api_key,
            "audio_model": config["audio_models"][0] if config["audio_models"] else "whisper-1",
            "language_model": config["language_models"][0] if config["language_models"] else "gpt-4o-mini",
            "system_message": config.get("system_message", ""),
            "default_language": config.get("default_language", "auto"),
            "default_translation_language": config.get("default_translation_language", "Japanese"),
            "chunk_minutes": config.get("default_chunk_minutes", 1),
            "translation_enabled": False
        }
    except Exception:
        # Load API key from environment variable even in fallback
        api_key = os.getenv("OPENAI_API_KEY", "")
        return {
            "api_key": api_key,
            "audio_model": "whisper-1",
            "language_model": "gpt-4o-mini",
            "system_message": "あなたはプロフェッショナルで親切な文字起こしアシスタントです。",
            "default_language": "auto",
            "default_translation_language": "Japanese",
            "chunk_minutes": 1,
            "translation_enabled": False
        }

def ensure_settings_structure(browser_state_value: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure browser state has proper settings structure with defaults."""
    if not isinstance(browser_state_value, dict):
        print("No settings found in browser state, using defaults.")
        return load_default_settings()

    # Fill in any missing keys with defaults
    defaults = load_default_settings()
    for key, default_value in defaults.items():
        if key not in browser_state_value:
            browser_state_value[key] = default_value

    return browser_state_value

def validate_settings(settings: dict[str, Any]) -> tuple[bool, str]:
    """Validate user settings."""
    try:
        # Validate API key
        api_key = settings.get("api_key", "").strip()
        validate_api_key(api_key)

        # Validate model selections
        if not settings.get("audio_model", "").strip():
            raise ValidationError("Audio model selection is required", field="audio_model")

        if not settings.get("language_model", "").strip():
            raise ValidationError("Language model selection is required", field="language_model")

        # Validate chunk duration
        chunk_minutes = settings.get("chunk_minutes", 5)
        if not isinstance(chunk_minutes, (int, float)) or chunk_minutes < 1 or chunk_minutes > 10:
            raise ValidationError(
                "Chunk duration must be between 1-10 minutes",
                field="chunk_minutes",
                value=chunk_minutes
            )

        return True, ""

    except ValidationError as e:
        return False, get_user_friendly_message(e)

def progress_callback(progress: float, message: str):
    """Update global progress state for UI updates."""
    app_state.processing_progress = progress
    app_state.processing_message = message

async def process_audio_file(
    audio_file: str,
    browser_state_value: dict[str, Any],
    ui_settings: dict[str, Any],
    progress: gr.Progress = None
) -> tuple[str, str, str, dict[str, Any]]:
    """
    Process audio file with transcription and optional translation using AudioHandler.
    
    Returns:
        Tuple of (display_text, translation_text, job_id, settings_used)
    """
    # Load persistent settings and merge UI settings
    settings = load_settings_from_browser_state(browser_state_value)
    settings.update(ui_settings)

    try:
        print(f"[DEBUG] Received audio file: {audio_file}")

        # Get the appropriate audio handler based on environment
        try:
            config = AppConfig()
            print(f"[DEBUG] Config environment: {config.env}")
        except Exception as e:
            logging.error(f"Failed to initialize AppConfig: {str(e)}", exc_info=True)
            raise gr.Error(get_user_friendly_message(e))

        try:
            if config.env == "mock-ui":
                audio_handler = MockAudioHandler()
                print(f"[DEBUG] Using MockAudioHandler")
            else:
                audio_handler = AudioHandler()
                print(f"[DEBUG] Using AudioHandler")
        except ImportError as e:
            logging.error(f"Import error creating audio handler: {str(e)}", exc_info=True)
            print(f"[DEBUG] Import error details: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise gr.Error(get_user_friendly_message(e))
        except Exception as e:
            logging.error(f"Unexpected error creating audio handler: {str(e)}", exc_info=True)
            print(f"[DEBUG] Handler creation error: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise gr.Error(get_user_friendly_message(e))

        print(f"[DEBUG] Audio handler created successfully")

        # Process audio using the handler
        try:
            result = await audio_handler.process_audio(
                audio_file,
                settings,
                progress_callback=lambda p, m: progress(p, m) if progress else None
            )
            print(f"[DEBUG] Audio processing completed successfully")
        except Exception as e:
            logging.error(f"Error in audio processing: {str(e)}", exc_info=True)
            print(f"[DEBUG] Processing error: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise gr.Error(get_user_friendly_message(e))

        # Update app state
        app_state.current_job_id = result.job_id
        app_state.current_transcript = result.transcript
        app_state.current_translation = result.translation

        if progress:
            progress(1.0, "Processing completed!")

        # Return display_text instead of transcript for UI
        return result.display_text, result.translation, result.job_id, result.settings_used

    except ValidationError as e:
        logging.error(f"Validation error: {str(e)}", exc_info=True)
        raise gr.Error(get_user_friendly_message(e))
    except APIError as e:
        logging.error(f"API error: {str(e)}", exc_info=True)
        raise gr.Error(get_user_friendly_message(e))
    except FileError as e:
        logging.error(f"File error: {str(e)}", exc_info=True)
        raise gr.Error(get_user_friendly_message(e))
    except TranslationError as e:
        # For translation errors, show warning but don't fail completely
        logging.warning(f"Translation failed: {str(e)}")
        # Return the result with error message in translation field
        return result.display_text, result.translation, result.job_id, result.settings_used
    except IntegratedDisplayError as e:
        # For display errors, show warning but don't fail completely
        logging.warning(f"Integrated display generation failed: {str(e)}")
        # Return transcript only
        return result.transcript, result.translation, result.job_id, result.settings_used
    except TranscriberError as e:
        logging.error(f"Transcriber error: {str(e)}", exc_info=True)
        raise gr.Error(get_user_friendly_message(e))
    except Exception as e:
        # Log unexpected errors
        logging.error(f"Unexpected error in process_audio_file: {str(e)}", exc_info=True)
        print(f"[DEBUG] Unexpected error: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        raise gr.Error(f"An unexpected error occurred: {str(e)}")

def format_transcript_for_display(text: str) -> str:
    """Format transcript for HTML display with timestamp styling."""
    if not text:
        return ""

    # Convert markdown timestamps to HTML with CSS classes
    import re

    # Pattern for # HH:MM:SS --> HH:MM:SS
    pattern = r'# (\d{2}:\d{2}:\d{2} --> \d{2}:\d{2}:\d{2})'

    def replace_timestamp(match):
        timestamp = match.group(1)
        return f'<span class="timestamp"># {timestamp}</span>'

    formatted = re.sub(pattern, replace_timestamp, text)

    # Convert newlines to HTML breaks for proper display
    formatted = formatted.replace('\n', '<br>')

    return formatted

def create_download_files(job_id: str, settings: dict[str, Any]) -> str:
    """Create download files using the new file management system."""
    if not job_id:
        raise gr.Error("No transcript available for download")

    try:
        # Find job directory
        job_dir = find_job_directory(job_id)
        if not job_dir:
            raise gr.Error("Transcript files not found for the given job ID")

        # Use the new file management system
        return create_download_package(job_dir, job_id)

    except Exception as e:
        raise gr.Error(f"Failed to create download package: {str(e)}")

def get_job_history() -> list[list[str]]:
    """Get list of previous jobs for history view."""
    jobs = []
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        return []

    try:
        for date_folder in sorted(os.listdir(data_dir), reverse=True):
            date_path = os.path.join(data_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            for job_folder in os.listdir(date_path):
                job_path = os.path.join(date_path, job_folder)
                metadata_path = os.path.join(job_path, "metadata.json")

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, encoding='utf-8') as f:
                            metadata = json.load(f)

                        jobs.append([
                            metadata.get("job_id", job_folder),
                            metadata.get("timestamp", ""),
                            metadata.get("original_filename", ""),
                            f"{metadata.get('file_info', {}).get('duration_seconds', 0):.1f}s",
                            metadata.get("settings", {}).get("default_language", ""),
                            "Completed"
                        ])
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
    except OSError:
        pass

    return jobs

def load_job_transcript(job_id: str) -> tuple[str, str, str]:
    """Load display content, original transcript, and translation for a specific job using new file management."""
    if not job_id:
        return "", "", ""

    try:
        # Find job directory
        job_dir = find_job_directory(job_id)
        if not job_dir:
            return "", "", ""

        # Get display content (integrated display if available, otherwise transcript)
        display_content = get_display_content_from_job(job_dir)

        # Get individual files for reference
        transcript, translation, _ = load_job_files(job_dir)

        return display_content, transcript, translation

    except Exception as e:
        print(f"Error loading job transcript: {e}")
        return "", "", ""

def handle_chat_message(
    message: str,
    history: list[dict[str, str]],
    settings: dict[str, Any]
) -> tuple[list[dict[str, str]], str]:
    """Handle chat message with context injection."""
    if not message.strip():
        return history, ""

    if not settings.get("api_key"):
        gr.Warning("Please set your OpenAI API key in settings")
        return history, ""

    try:
        # Use transcript as context if available
        context_text = app_state.current_transcript or ""

        if context_text:
            response = chat_with_context(
                api_key=settings["api_key"],
                model=settings["language_model"],
                question=message,
                context_text=context_text,
                system_message=settings.get("system_message", ""),
                temperature=0.7
            )
        else:
            response, _ = chat_completion(
                api_key=settings["api_key"],
                model=settings["language_model"],
                message=message,
                system_message=settings.get("system_message", ""),
                temperature=0.7
            )

        # Update history in Gradio messages format
        new_history = history.copy() if history else []
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": response})

        return new_history, ""

    except Exception as e:
        gr.Error(f"Chat error: {str(e)}")
        return history, ""

def clear_chat_history() -> list[dict[str, str]]:
    """Clear chat history."""
    app_state.chat_history = []
    return []

def toggle_translation_target(enabled):
    """Toggle translation target dropdown visibility."""
    return gr.update(visible=enabled)

def create_main_tab(config: dict[str, Any], audio_handler, chat_handler, settings_handler, env: str):
    """Create the main tab with audio processing interface."""
    with gr.TabItem("Main", elem_classes=["tab-content"]):
        # Row 1: Configuration Panel
        with gr.Group(elem_classes=["config-section"]):
            gr.Markdown("### Configuration")
            with gr.Row():
                audio_model = gr.Dropdown(
                    choices=config["audio_models"],
                    value=config["audio_models"][0] if config["audio_models"] else "whisper-1",
                    label="Audio Model",
                    scale=1,
                    interactive=True
                )
                language_select = gr.Dropdown(
                    choices=["auto"] + list(config.get("translation_languages", {}).keys()),
                    value="auto",
                    label="Language",
                    scale=1,
                    interactive=True
                )
                chunk_duration = gr.Dropdown(
                    choices=CHUNK_DURATIONS,
                    value="1 min",
                    label="Chunk Duration",
                    scale=1,
                    interactive=True
                )

            with gr.Row():
                translation_enabled = gr.Checkbox(
                    label="Enable Translation",
                    value=False,
                    scale=1
                )
                translation_target = gr.Dropdown(
                    choices=list(config.get("translation_languages", {}).keys()),
                    value="Japanese",
                    label="Translation Target",
                    visible=False,
                    scale=2,
                    interactive=True
                )

        # Row 2: File Upload
        with gr.Group():
            audio_input = gr.File(
                label="Upload Audio File",
                file_types=["audio"],
                elem_classes=["upload-area"]
            )

        # Row 3: Processing Control
        with gr.Row():
            process_btn = gr.Button(
                "🎯 Start Processing",
                variant="primary",
                size="lg",
                scale=1
            )

        # Row 4: Status Indicator (separate from results)
        with gr.Group():
            status_display = gr.HTML(
                value=create_status_html(0, 0, "Ready to process audio file")
            )

        # Row 5: Results Display
        with gr.Group(elem_classes=["results-container"]):
            gr.Markdown("### Transcription Results")
            results_display = gr.Textbox(
                placeholder="Transcription results will appear here as processing completes...",
                lines=10,
                max_lines=15,
                show_copy_button=True,
                container=False
            )

            with gr.Row():
                download_btn = gr.DownloadButton(
                    "📥 Download Results",
                    size="sm",
                    visible=False
                )

        # Chat Interface
        with gr.Group(elem_classes=["chat-container"]):
            gr.Markdown("### Chat with Transcript")
            with gr.Accordion("💬 Ask questions about your transcript", open=False):
                chat_interface = gr.Chatbot(
                    label="",
                    height=250,
                    type="messages"
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask a question about the transcript...",
                        label="",
                        scale=4
                    )
                    chat_send_btn = gr.Button("Send", scale=1)
                    chat_clear_btn = gr.Button("Clear", scale=1)

        return {
            "audio_input": audio_input,
            "audio_model": audio_model,
            "language_select": language_select,
            "chunk_duration": chunk_duration,
            "translation_enabled": translation_enabled,
            "translation_target": translation_target,
            "process_btn": process_btn,
            "status_display": status_display,
            "results_display": results_display,
            "download_btn": download_btn,
            "chat_interface": chat_interface,
            "chat_input": chat_input,
            "chat_send_btn": chat_send_btn,
            "chat_clear_btn": chat_clear_btn
        }

def create_settings_tab(config: dict[str, Any], settings_handler):
    """Create the settings tab."""
    with gr.TabItem("Settings", elem_classes=["tab-content"]):
        gr.Markdown("### API Configuration")

        with gr.Group():
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                info="Your OpenAI API key for transcription and chat services"
            )

            with gr.Row():
                settings_audio_model = gr.Dropdown(
                    choices=config["audio_models"],
                    value=config["audio_models"][0] if config["audio_models"] else "whisper-1",
                    label="Default Audio Model",
                    interactive=True
                )
                settings_language_model = gr.Dropdown(
                    choices=config["language_models"],
                    value=config["language_models"][0] if config["language_models"] else "gpt-4o-mini",
                    label="Default Language Model",
                    interactive=True
                )

            system_message_input = gr.Textbox(
                label="System Message",
                value=config.get("system_message", "You are a helpful transcription assistant."),
                lines=3,
                info="Customize the AI assistant's behavior"
            )

        # Settings controls
        with gr.Row():
            save_settings_btn = gr.Button(
                "💾 Save Settings",
                variant="primary"
            )
            reset_settings_btn = gr.Button(
                "🔄 Reset to Default"
            )

        return {
            "api_key_input": api_key_input,
            "settings_audio_model": settings_audio_model,
            "settings_language_model": settings_language_model,
            "system_message_input": system_message_input,
            "save_settings_btn": save_settings_btn,
            "reset_settings_btn": reset_settings_btn
        }

def create_history_tab(history_handler):
    """Create the history tab with job management using Radio buttons."""
    with gr.TabItem("History", elem_classes=["tab-content"]):
        gr.Markdown("### Job History")

        with gr.Group():
            # Create radio button options from job data with vertical layout
            job_history = history_handler.get_job_history_with_translation_info()
            job_options = []
            for job in job_history:
                job_id, timestamp, filename, duration, language, status = job
                # Format as compact readable option for vertical layout
                status_icon = "✅" if status == "Completed" else "❌"
                if timestamp:
                    date_part = timestamp.split()[0] if " " in timestamp else timestamp[:10]
                    time_part = timestamp.split()[1] if " " in timestamp else timestamp[11:19]
                else:
                    date_part = "N/A"
                    time_part = "N/A"
                label = f"{job_id} • {filename} • {date_part} {time_part} • {duration} • {language} {status_icon}"
                job_options.append((label, job_id))

            job_selector = gr.Radio(
                choices=job_options,
                label="Select Job",
                value=None,
                elem_classes=["job-selector"]
            )

            # Job details display
            with gr.Group():
                gr.Markdown("### Selected Job Details")
                job_details_display = gr.HTML(
                    value="<p style='color: #888; text-align: center;'>Select a job above to view details</p>",
                    elem_classes=["job-details"]
                )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                delete_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")
                load_btn = gr.Button("📄 Load Transcript", size="sm", variant="primary")
                history_download_btn = gr.DownloadButton("📥 Download", size="sm", visible=False)

        return {
            "job_selector": job_selector,
            "job_details_display": job_details_display,
            "refresh_btn": refresh_btn,
            "delete_btn": delete_btn,
            "load_btn": load_btn,
            "history_download_btn": history_download_btn
        }

# Create the Gradio interface
def create_app(env: str = "prod"):
    """
    Create Gradio app with environment-specific handlers.
    
    Args:
        env: Environment name (prod, test, mock-ui)
        
    Returns:
        Gradio app instance
    """
    # Initialize configuration
    app_config = AppConfig(env)
    config = app_config.get_all()

    # Initialize handlers based on environment
    if env == "mock-ui":
        # Use mock handlers for UI testing
        audio_handler = MockAudioHandler()
        chat_handler = MockChatHandler()
        history_handler = MockHistoryHandler()
        settings_handler = MockSettingsHandler()
    else:
        # Use real handlers for production and testing
        audio_handler = AudioHandler()
        chat_handler = ChatHandler()
        history_handler = HistoryHandler()
        settings_handler = SettingsHandler()

    with gr.Blocks(css=CUSTOM_CSS, title="Audio Transcription App") as app:
        # Browser state for settings persistence
        browser_state = gr.BrowserState(
            load_default_settings(),
            storage_key="transcriber_app_settings"
        )

        # Main tab navigation
        with gr.Tabs():
            main_components = create_main_tab(config, audio_handler, chat_handler, settings_handler, env)
            settings_components = create_settings_tab(config, settings_handler)
            history_components = create_history_tab(history_handler)

        # Extract components for event handling
        audio_input = main_components["audio_input"]
        audio_model = main_components["audio_model"]
        language_select = main_components["language_select"]
        chunk_duration = main_components["chunk_duration"]
        translation_enabled = main_components["translation_enabled"]
        translation_target = main_components["translation_target"]
        process_btn = main_components["process_btn"]
        status_display = main_components["status_display"]
        results_display = main_components["results_display"]
        download_btn = main_components["download_btn"]
        chat_interface = main_components["chat_interface"]
        chat_input = main_components["chat_input"]
        chat_send_btn = main_components["chat_send_btn"]
        chat_clear_btn = main_components["chat_clear_btn"]

        api_key_input = settings_components["api_key_input"]
        settings_audio_model = settings_components["settings_audio_model"]
        settings_language_model = settings_components["settings_language_model"]
        system_message_input = settings_components["system_message_input"]
        save_settings_btn = settings_components["save_settings_btn"]
        reset_settings_btn = settings_components["reset_settings_btn"]

        job_selector = history_components["job_selector"]
        job_details_display = history_components["job_details_display"]
        refresh_btn = history_components["refresh_btn"]
        delete_btn = history_components["delete_btn"]
        load_btn = history_components["load_btn"]
        history_download_btn = history_components["history_download_btn"]

        # Event handlers

        # Show/hide translation controls
        translation_enabled.change(
            toggle_translation_target,
            inputs=[translation_enabled],
            outputs=[translation_target]
        )

        # This section was already replaced with the new process_audio function above

        # Settings functions

        def save_settings(api_key, audio_model, language_model, system_message, browser_state_value):
            # Ensure proper structure and merge with existing settings
            current_settings = ensure_settings_structure(browser_state_value)

            # Update with new settings
            current_settings.update({
                "api_key": api_key,
                "audio_model": audio_model,
                "language_model": language_model,
                "system_message": system_message
            })

            gr.Info("Settings saved successfully!")
            return current_settings

        def reset_settings():
            default_settings = load_default_settings()
            return (
                default_settings.get("api_key", ""),
                default_settings.get("audio_model", ""),
                default_settings.get("language_model", ""),
                default_settings.get("system_message", "")
            )

        # Processing functions for audio
        async def process_audio_wrapper(
            audio_file,
            browser_state_value,
            audio_model_val,
            language_select_val,
            chunk_duration_val,
            translation_enabled_val,
            translation_target_val
        ):
            """Process audio file with new UI structure."""
            # Load settings from browser state
            base_settings = ensure_settings_structure(browser_state_value)

            # Convert chunk duration from dropdown to minutes
            chunk_minutes = int(chunk_duration_val.split()[0])

            # Convert language name to language code if not "auto"
            language_code = language_select_val
            if language_select_val != "auto":
                language_code = get_language_code(language_select_val)

            ui_settings = {
                "audio_model": audio_model_val,
                "default_language": language_code,
                "chunk_minutes": chunk_minutes,
                "translation_enabled": translation_enabled_val,
                "default_translation_language": translation_target_val if translation_enabled_val else ""
            }

            # Merge settings using handler
            settings = settings_handler.merge_settings(base_settings, ui_settings)

            # Use audio handler for processing
            try:
                result = await audio_handler.process_audio(
                    audio_file,
                    settings
                )

                transcript = result.transcript
                translation = result.translation
                job_id = result.job_id
                settings_used = result.settings_used

                # Store current transcript and job_id for chat and download
                app_state.current_transcript = transcript
                app_state.current_job_id = job_id

                # Show download button and update status
                return (
                    result.display_text,  # Use display_text instead of transcript for integrated display
                    create_status_html(0, 0, "Processing completed successfully!"),
                    gr.update(visible=True)
                )

            except Exception as e:
                import traceback
                logging.error(f"Error in process_audio_wrapper: {str(e)}", exc_info=True)
                error_msg = get_user_friendly_message(e) if hasattr(e, '__class__') else str(e)
                error_msg += '\n' + traceback.format_exc()
                return (
                    f"Error: {error_msg}",
                    create_status_html(0, 0, f"Processing failed: {error_msg}"),
                    gr.update(visible=False)
                )

        # Download button click handler
        def handle_download():
            """Handle download button click for current job."""
            if not app_state.current_job_id:
                raise gr.Error("No transcript available for download")

            try:
                return create_download_files(app_state.current_job_id, {})
            except Exception as e:
                raise gr.Error(f"Failed to create download package: {str(e)}")

        download_btn.click(
            handle_download,
            outputs=[download_btn]
        )

        # History tab event handlers
        selected_job_id = gr.State(value=None)

        def handle_job_selection(job_id):
            """Handle job selection in history tab."""
            if not job_id:
                return (
                    "<p style='color: #888; text-align: center;'>Select a job above to view details</p>",
                    gr.update(visible=False),
                    job_id
                )

            try:
                # Load job details using history handler
                job_details = history_handler.get_job_details(job_id)

                # Format job details for display
                # Format timestamp for display
                timestamp = job_details.get('timestamp', 'N/A')
                if timestamp and timestamp != 'N/A':
                    # Format timestamp nicely
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('T', ' '))
                        formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_timestamp = timestamp
                else:
                    formatted_timestamp = 'N/A'

                details_html = f"""
                <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <h4>Job Details</h4>
                    <p><strong>Timestamp:</strong> {formatted_timestamp}</p>
                    <p><strong>File:</strong> {job_details.get('original_filename', 'N/A')}</p>
                    <p><strong>Duration:</strong> {job_details.get('file_info', {}).get('duration_seconds', 0):.1f}s</p>
                    <p><strong>Translation:</strong> {'Enabled' if job_details.get('translation_enabled', False) else 'Disabled'}</p>
                    <p><strong>Job ID:</strong> {job_details.get('job_id', 'N/A')}</p>
                    <p><strong>Status:</strong> Completed</p>
                </div>
                """

                return (
                    details_html,
                    gr.update(visible=True),
                    job_id
                )

            except Exception as e:
                return (
                    f"<p style='color: #dc3545;'>Error loading job details: {str(e)}</p>",
                    gr.update(visible=False),
                    None
                )

        def handle_history_download(job_id):
            """Handle download for selected history job."""
            if not job_id:
                raise gr.Error("No job selected for download")

            try:
                return create_download_files(job_id, {})
            except Exception as e:
                raise gr.Error(f"Failed to create download package: {str(e)}")

        def handle_load_transcript(job_id):
            """Load transcript from history job into main display."""
            if not job_id:
                raise gr.Error("No job selected")

            try:
                display_content, original_transcript, translation = load_job_transcript(job_id)

                # Update app state
                app_state.current_job_id = job_id
                app_state.current_transcript = original_transcript  # For chat context - use original transcript, not display content

                return (
                    display_content,
                    gr.update(visible=True)  # Show main download button
                )

            except Exception as e:
                raise gr.Error(f"Failed to load transcript: {str(e)}")

        def refresh_job_history():
            """Refresh the job history list."""
            try:
                job_history = history_handler.get_job_history_with_translation_info()
                job_options = []
                for job in job_history:
                    job_id, timestamp, filename, duration, language, status = job
                    status_icon = "✅" if status == "Completed" else "❌"
                    if timestamp:
                        date_part = timestamp.split()[0] if " " in timestamp else timestamp[:10]
                        time_part = timestamp.split()[1] if " " in timestamp else timestamp[11:19]
                    else:
                        date_part = "N/A"
                        time_part = "N/A"
                    label = f"{date_part} {time_part} • {filename} • {job_id} • {duration} • {language} {status_icon}"
                    job_options.append((label, job_id))

                return gr.update(choices=job_options, value=None)

            except Exception as e:
                gr.Error(f"Failed to refresh job history: {str(e)}")
                return gr.update()

        def handle_delete_job(job_id):
            """Delete selected job and refresh history."""
            if not job_id:
                raise gr.Error("No job selected for deletion")
            
            try:
                success, message = history_handler.delete_job(job_id)
                
                if success:
                    gr.Info(message)
                    # Refresh the job history list after successful deletion
                    job_history = history_handler.get_job_history_with_translation_info()
                    job_options = []
                    for job in job_history:
                        job_id_item, timestamp, filename, duration, language, status = job
                        status_icon = "✅" if status == "Completed" else "❌"
                        if timestamp:
                            date_part = timestamp.split()[0] if " " in timestamp else timestamp[:10]
                            time_part = timestamp.split()[1] if " " in timestamp else timestamp[11:19]
                        else:
                            date_part = "N/A"
                            time_part = "N/A"
                        label = f"{date_part} {time_part} • {filename} • {job_id_item} • {duration} • {language} {status_icon}"
                        job_options.append((label, job_id_item))
                    
                    return (
                        gr.update(choices=job_options, value=None),  # Reset job selector
                        "<p style='color: #888; text-align: center;'>Select a job above to view details</p>",  # Reset details display
                        gr.update(visible=False),  # Hide download button
                        None  # Reset selected job ID
                    )
                else:
                    raise gr.Error(message)
                    
            except Exception as e:
                raise gr.Error(f"Failed to delete job: {str(e)}")

        # Connect history tab events
        job_selector.change(
            handle_job_selection,
            inputs=[job_selector],
            outputs=[job_details_display, history_download_btn, selected_job_id]
        )

        history_download_btn.click(
            handle_history_download,
            inputs=[selected_job_id],
            outputs=[history_download_btn]
        )

        load_btn.click(
            handle_load_transcript,
            inputs=[selected_job_id],
            outputs=[results_display, download_btn]
        )

        refresh_btn.click(
            refresh_job_history,
            outputs=[job_selector]
        )

        delete_btn.click(
            handle_delete_job,
            inputs=[selected_job_id],
            outputs=[job_selector, job_details_display, history_download_btn, selected_job_id]
        )

        # Process button click handler
        process_btn.click(
            process_audio_wrapper,
            inputs=[
                audio_input,
                browser_state,
                audio_model,
                language_select,
                chunk_duration,
                translation_enabled,
                translation_target
            ],
            outputs=[
                results_display,
                status_display,
                download_btn
            ]
        )

        # Translation target visibility toggle
        translation_enabled.change(
            toggle_translation_target,
            inputs=[translation_enabled],
            outputs=[translation_target]
        )

        # Settings button removed - using accordion instead

        save_settings_btn.click(
            save_settings,
            inputs=[api_key_input, settings_audio_model, settings_language_model, system_message_input, browser_state],
            outputs=[browser_state]
        )

        reset_settings_btn.click(
            reset_settings,
            outputs=[api_key_input, settings_audio_model, settings_language_model, system_message_input]
        )

        # History functions are now handled in the history tab event handlers above

        # Chat functions with handler
        def handle_chat_wrapper(message, history, browser_state_value):
            settings = ensure_settings_structure(browser_state_value)

            # Set context for chat handler
            context_text = app_state.current_transcript or ""
            chat_handler.set_context(context_text)

            return chat_handler.handle_message(message, history, settings)

        chat_send_btn.click(
            handle_chat_wrapper,
            inputs=[chat_input, chat_interface, browser_state],
            outputs=[chat_interface, chat_input]
        )

        chat_input.submit(
            handle_chat_wrapper,
            inputs=[chat_input, chat_interface, browser_state],
            outputs=[chat_interface, chat_input]
        )

        chat_clear_btn.click(
            lambda: chat_handler.clear_history(),
            outputs=[chat_interface]
        )


        # Page load initialization - load settings from browser state
        def initialize_components(browser_state_value):
            settings = ensure_settings_structure(browser_state_value)
            return (
                settings.get("audio_model", config["audio_models"][0] if config["audio_models"] else "whisper-1"),
                settings.get("default_language", "auto"),
                "1 min",  # Default chunk duration
                settings.get("translation_enabled", False),
                settings.get("default_translation_language", "Japanese"),
                # Settings panel fields
                settings.get("api_key", ""),
                settings.get("audio_model", config["audio_models"][0] if config["audio_models"] else "whisper-1"),
                settings.get("language_model", config["language_models"][0] if config["language_models"] else "gpt-4o-mini"),
                settings.get("system_message", config.get("system_message", ""))
            )

        app.load(
            initialize_components,
            inputs=[browser_state],
            outputs=[
                audio_model, language_select, chunk_duration, translation_enabled, translation_target,
                # Settings panel fields
                api_key_input, settings_audio_model, settings_language_model, system_message_input
            ]
        )

    return app

def main():
    """Main application function."""
    # Get environment from environment variable or default to prod
    env = os.getenv("APP_ENV", "prod")

    print(f"Starting transcriber web app in {env} mode...")
    if env == "mock-ui":
        print("Using mock handlers for UI testing")

    demo = create_app(env=env)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
