"""
Comprehensive error handling and validation for transcriber web app.

Custom exceptions and error utilities for better user experience and debugging.
"""

import logging
import traceback
import os
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import openai


class ErrorType(Enum):
    """Error type categories for better error handling."""
    VALIDATION = "validation"
    FILE_IO = "file_io"
    API = "api"
    NETWORK = "network"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    UI = "ui"


class TranscriberError(Exception):
    """Base exception for transcriber app errors."""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PROCESSING, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/debugging."""
        return {
            "type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "traceback": traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else None
        }


class ValidationError(TranscriberError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, ErrorType.VALIDATION, {"field": field, "value": value})
        self.field = field
        self.value = value


class FileError(TranscriberError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message, ErrorType.FILE_IO, {"file_path": file_path, "operation": operation})
        self.file_path = file_path
        self.operation = operation


class APIError(TranscriberError):
    """Raised when API calls fail."""
    
    def __init__(self, message: str, api_name: str = "OpenAI", status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message, ErrorType.API, {
            "api_name": api_name, 
            "status_code": status_code, 
            "retry_after": retry_after
        })
        self.api_name = api_name
        self.status_code = status_code
        self.retry_after = retry_after


class NetworkError(TranscriberError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, timeout: Optional[int] = None):
        super().__init__(message, ErrorType.NETWORK, {"timeout": timeout})
        self.timeout = timeout


class ConfigurationError(TranscriberError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_file: Optional[str] = None, missing_keys: Optional[list] = None):
        super().__init__(message, ErrorType.CONFIGURATION, {
            "config_file": config_file,
            "missing_keys": missing_keys
        })
        self.config_file = config_file
        self.missing_keys = missing_keys


class MemoryError(TranscriberError):
    """Raised when memory constraints are exceeded."""
    
    def __init__(self, message: str, file_size_mb: Optional[float] = None, memory_limit_mb: Optional[float] = None):
        super().__init__(message, ErrorType.MEMORY, {
            "file_size_mb": file_size_mb,
            "memory_limit_mb": memory_limit_mb
        })
        self.file_size_mb = file_size_mb
        self.memory_limit_mb = memory_limit_mb


class UIError(TranscriberError):
    """Raised when UI operations fail."""
    
    def __init__(self, message: str, component: Optional[str] = None, action: Optional[str] = None):
        super().__init__(message, ErrorType.UI, {"component": component, "action": action})
        self.component = component
        self.action = action


class TranslationError(TranscriberError):
    """Raised when translation operations fail."""
    
    def __init__(self, message: str, transcript_available: bool = True, partial_translation: Optional[str] = None):
        super().__init__(message, ErrorType.PROCESSING, {
            "transcript_available": transcript_available,
            "partial_translation": partial_translation
        })
        self.transcript_available = transcript_available
        self.partial_translation = partial_translation


class IntegratedDisplayError(TranscriberError):
    """Raised when integrated display generation fails."""
    
    def __init__(self, message: str, transcript: Optional[str] = None, translation: Optional[str] = None):
        super().__init__(message, ErrorType.PROCESSING, {
            "transcript_available": bool(transcript),
            "translation_available": bool(translation)
        })
        self.transcript = transcript
        self.translation = translation


def validate_api_key(api_key: str) -> None:
    """
    Validate OpenAI API key format.
    
    Args:
        api_key: API key to validate
        
    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key:
        raise ValidationError("API key is required", field="api_key")
    
    if not isinstance(api_key, str):
        raise ValidationError("API key must be a string", field="api_key", value=type(api_key))
    
    if not api_key.startswith("sk-"):
        raise ValidationError("API key must start with 'sk-'", field="api_key")
    
    if len(api_key) < 20:
        raise ValidationError("API key is too short", field="api_key", value=len(api_key))


def validate_file_path(file_path: str, must_exist: bool = True) -> None:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Raises:
        ValidationError: If file path is invalid
        FileError: If file doesn't exist when required
    """
    if not file_path:
        raise ValidationError("File path is required", field="file_path")
    
    if not isinstance(file_path, str):
        raise ValidationError("File path must be a string", field="file_path", value=type(file_path))
    
    from pathlib import Path
    
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise FileError(f"File not found: {file_path}", file_path=file_path, operation="validation")
        
        if must_exist and not path.is_file():
            raise FileError(f"Path is not a file: {file_path}", file_path=file_path, operation="validation")
            
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid file path: {str(e)}", field="file_path", value=file_path)


def validate_audio_file_extended(file_path: str, max_size_mb: float = 500) -> Dict[str, Any]:
    """
    Extended audio file validation with detailed error reporting.
    
    Args:
        file_path: Path to audio file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        Dictionary with validation results and file info
        
    Raises:
        ValidationError: If file validation fails
        FileError: If file operations fail
    """
    validate_file_path(file_path, must_exist=True)
    
    from pathlib import Path
    
    try:
        path = Path(file_path)
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Check file size
        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed ({max_size_mb}MB)",
                field="file_size",
                value=file_size_mb
            )
        
        # Check file extension
        supported_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.webm']
        file_ext = path.suffix.lower()
        
        if file_ext not in supported_extensions:
            raise ValidationError(
                f"Unsupported file format: {file_ext}. Supported formats: {', '.join(supported_extensions)}",
                field="file_extension",
                value=file_ext
            )
        
        # Try to get audio info
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000.0
            
            # Check minimum duration
            if duration_seconds < 1:
                raise ValidationError(
                    "Audio file is too short (minimum 1 second required)",
                    field="duration",
                    value=duration_seconds
                )
            
            # Check maximum duration (e.g., 2 hours)
            if duration_seconds > 7200:
                raise ValidationError(
                    f"Audio file is too long ({duration_seconds/3600:.1f} hours). Maximum 2 hours allowed.",
                    field="duration",
                    value=duration_seconds
                )
            
            file_info = {
                "size_mb": file_size_mb,
                "duration_seconds": duration_seconds,
                "format": file_ext,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "needs_warning": file_size_mb > 100  # Warning for files > 100MB
            }
            
            return file_info
            
        except Exception as e:
            raise FileError(
                f"Failed to process audio file: {str(e)}",
                file_path=file_path,
                operation="audio_analysis"
            )
    
    except OSError as e:
        raise FileError(
            f"Failed to access file: {str(e)}",
            file_path=file_path,
            operation="file_access"
        )


def handle_openai_error(error: Exception) -> APIError:
    """
    Convert OpenAI errors to our custom APIError with better messaging.
    
    Args:
        error: Original OpenAI error
        
    Returns:
        APIError with user-friendly message
    """
    error_str = str(error)
    
    # Rate limit errors
    if "rate limit" in error_str.lower() or "429" in error_str:
        return APIError(
            "Rate limit exceeded. Please wait a moment before trying again.",
            api_name="OpenAI",
            status_code=429,
            retry_after=60
        )
    
    # Authentication errors
    if "authentication" in error_str.lower() or "401" in error_str or "invalid api key" in error_str.lower():
        return APIError(
            "Invalid API key. Please check your OpenAI API key in settings.",
            api_name="OpenAI", 
            status_code=401
        )
    
    # Quota/billing errors
    if "quota" in error_str.lower() or "billing" in error_str.lower() or "insufficient" in error_str.lower():
        return APIError(
            "OpenAI quota exceeded or billing issue. Please check your OpenAI account.",
            api_name="OpenAI",
            status_code=402
        )
    
    # Model not found
    if "model" in error_str.lower() and "not found" in error_str.lower():
        return APIError(
            "Selected model is not available. Please choose a different model.",
            api_name="OpenAI",
            status_code=404
        )
    
    # Request too large
    if "request too large" in error_str.lower() or "413" in error_str:
        return APIError(
            "Request too large. Try reducing the chunk size or file size.",
            api_name="OpenAI",
            status_code=413
        )
    
    # Network/timeout errors
    if "timeout" in error_str.lower() or "connection" in error_str.lower():
        return APIError(
            "Network timeout. Please check your internet connection and try again.",
            api_name="OpenAI",
            status_code=None
        )
    
    # Generic server errors
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return APIError(
            "OpenAI service temporarily unavailable. Please try again later.",
            api_name="OpenAI",
            status_code=500
        )
    
    # Generic API error
    return APIError(
        f"OpenAI API error: {error_str}",
        api_name="OpenAI"
    )


def handle_gradio_error(error: Exception, component: str = "unknown", action: str = "unknown") -> UIError:
    """
    Convert Gradio errors to our custom UIError.
    
    Args:
        error: Original Gradio error
        component: UI component where error occurred
        action: Action that caused the error
        
    Returns:
        UIError with user-friendly message
    """
    error_str = str(error)
    
    if "file upload" in error_str.lower():
        return UIError(
            "File upload failed. Please try uploading the file again.",
            component=component,
            action="file_upload"
        )
    
    if "download" in error_str.lower():
        return UIError(
            "Download failed. Please try again or contact support.",
            component=component,
            action="download"
        )
    
    return UIError(
        f"UI error in {component}: {error_str}",
        component=component,
        action=action
    )


def safe_execute(func, *args, error_context: str = "", **kwargs):
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_context: Context description for errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or raises appropriate TranscriberError
    """
    try:
        return func(*args, **kwargs)
    except TranscriberError:
        # Re-raise our custom errors
        raise
    except openai.OpenAIError as e:
        raise handle_openai_error(e)
    except (OSError, IOError) as e:
        raise FileError(f"File operation failed: {str(e)}", operation=error_context)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid input: {str(e)}")
    except MemoryError as e:
        raise MemoryError(f"Out of memory: {str(e)}")
    except Exception as e:
        # Log unexpected errors for debugging
        logging.error(f"Unexpected error in {error_context}: {str(e)}", exc_info=True)
        raise TranscriberError(
            f"Unexpected error occurred: {str(e)}",
            ErrorType.PROCESSING,
            {"context": error_context, "original_error": str(e)}
        )


def setup_error_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup comprehensive error logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def handle_translation_failure(
    transcript: str, 
    error: Exception, 
    partial_translation: Optional[str] = None
) -> Tuple[str, str, TranslationError]:
    """
    Handle translation failure gracefully while preserving transcript.
    
    Args:
        transcript: Successfully transcribed text
        error: Translation error that occurred
        partial_translation: Any partial translation that was completed
        
    Returns:
        Tuple of (transcript, fallback_translation, translation_error)
    """
    # Create user-friendly error message for translation failure
    if "rate limit" in str(error).lower():
        error_msg = "Translation rate limit exceeded. Please try again later."
    elif "quota" in str(error).lower() or "billing" in str(error).lower():
        error_msg = "Translation quota exceeded. Please check your OpenAI account."
    elif "timeout" in str(error).lower() or "connection" in str(error).lower():
        error_msg = "Translation service timeout. Please try again."
    else:
        error_msg = f"Translation service error: {str(error)}"
    
    # Create fallback translation text with error information
    fallback_translation = f"[Translation Error]\n{error_msg}\n\nTranscription completed successfully. You can download the transcript and try translation again later."
    
    # If we have partial translation, include it
    if partial_translation and partial_translation.strip():
        fallback_translation += f"\n\n[Partial Translation]\n{partial_translation}"
    
    translation_error = TranslationError(
        error_msg,
        transcript_available=True,
        partial_translation=partial_translation
    )
    
    return transcript, fallback_translation, translation_error


def handle_file_read_failure(
    file_path: str, 
    error: Exception, 
    fallback_content: str = ""
) -> Tuple[str, FileError]:
    """
    Handle file reading failure with fallback content.
    
    Args:
        file_path: Path to file that failed to read
        error: File reading error that occurred
        fallback_content: Fallback content to use
        
    Returns:
        Tuple of (content, file_error)
    """
    error_msg = f"Failed to read file {os.path.basename(file_path)}: {str(error)}"
    
    file_error = FileError(
        error_msg,
        file_path=file_path,
        operation="read"
    )
    
    return fallback_content, file_error


def handle_integrated_display_failure(
    transcript: str, 
    translation: str, 
    error: Exception
) -> Tuple[str, IntegratedDisplayError]:
    """
    Handle integrated display generation failure with fallback.
    
    Args:
        transcript: Original transcript text
        translation: Translation text
        error: Display generation error that occurred
        
    Returns:
        Tuple of (fallback_display_text, integrated_display_error)
    """
    error_msg = f"Failed to generate integrated display: {str(error)}"
    
    # Fallback to transcript only
    fallback_display = transcript
    
    integrated_error = IntegratedDisplayError(
        error_msg,
        transcript=transcript,
        translation=translation
    )
    
    return fallback_display, integrated_error


def create_error_report(error: TranscriberError) -> Dict[str, Any]:
    """
    Create detailed error report for debugging.
    
    Args:
        error: TranscriberError instance
        
    Returns:
        Detailed error report dictionary
    """
    import sys
    import platform
    from datetime import datetime
    
    return {
        "timestamp": datetime.now().isoformat(),
        "error": error.to_dict(),
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable
        }
    }


# Error messages for different scenarios
ERROR_MESSAGES = {
    "api_key_missing": "Please enter your OpenAI API key in the settings panel (⚙️ button).",
    "file_too_large": "File is too large. Maximum size allowed is {max_size}MB.",
    "unsupported_format": "Unsupported audio format. Please use MP3, WAV, M4A, FLAC, or OGG files.",
    "network_timeout": "Network timeout. Please check your internet connection and try again.",
    "processing_failed": "Processing failed. Please try again or check your settings.",
    "translation_failed": "Translation failed. The transcription was successful, but translation encountered an error.",
    "download_failed": "Download failed. Please try again.",
    "settings_invalid": "Invalid settings. Please check your configuration.",
    "job_not_found": "Job not found. It may have been deleted or moved.",
    "insufficient_quota": "OpenAI API quota exceeded. Please check your OpenAI account billing.",
    "model_unavailable": "Selected model is not available. Please choose a different model.",
    "translation_partial_failure": "Translation partially failed. Transcription completed successfully, but some translation segments may be missing.",
    "integrated_display_failed": "Failed to generate integrated display. Showing transcription only.",
    "file_read_failed": "Failed to read file. Using fallback content.",
    "translation_service_unavailable": "Translation service is temporarily unavailable. Transcription completed successfully."
}


def get_user_friendly_message(error: TranscriberError) -> str:
    """
    Get user-friendly error message for UI display.
    
    Args:
        error: TranscriberError instance
        
    Returns:
        User-friendly error message
    """
    # Handle translation-specific errors
    if isinstance(error, TranslationError):
        if error.transcript_available:
            return ERROR_MESSAGES["translation_failed"]
        else:
            return f"Translation failed: {error.message}"
    
    # Handle integrated display errors
    if isinstance(error, IntegratedDisplayError):
        return ERROR_MESSAGES["integrated_display_failed"]
    
    # Map specific errors to user-friendly messages
    if error.error_type == ErrorType.API and hasattr(error, 'status_code') and error.status_code == 401:
        return ERROR_MESSAGES["api_key_missing"]
    elif error.error_type == ErrorType.API and hasattr(error, 'status_code') and error.status_code == 429:
        retry_after = getattr(error, 'retry_after', 60)
        return f"Rate limit exceeded. Please wait {retry_after} seconds before trying again."
    elif error.error_type == ErrorType.API and hasattr(error, 'status_code') and error.status_code == 402:
        return ERROR_MESSAGES["insufficient_quota"]
    elif error.error_type == ErrorType.VALIDATION and hasattr(error, 'field') and "file_size" in str(error.field):
        return ERROR_MESSAGES["file_too_large"].format(max_size=500)
    elif error.error_type == ErrorType.VALIDATION and hasattr(error, 'field') and "file_extension" in str(error.field):
        return ERROR_MESSAGES["unsupported_format"]
    elif error.error_type == ErrorType.NETWORK:
        return ERROR_MESSAGES["network_timeout"]
    elif error.error_type == ErrorType.FILE_IO:
        return f"File error: {error.message}"
    elif error.error_type == ErrorType.CONFIGURATION:
        return ERROR_MESSAGES["settings_invalid"]
    else:
        return error.message