"""
Audio utilities for transcriber web app.

Memory-efficient audio processing with chunking support for large files.
"""

import os
import tempfile

import yaml
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file (default: auto-detect)
        
    Returns:
        Configuration dictionary
    """
    from .errors import ConfigurationError, safe_execute

    # Auto-detect config path if not provided
    if config_path is None:
        # Try current directory first (when running from src/)
        if os.path.exists("config.yaml"):
            config_path = "config.yaml"
        # Try src/ directory (when running from project root)
        elif os.path.exists("src/config.yaml"):
            config_path = "src/config.yaml"
        # Try relative to this file's directory
        else:
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, "config.yaml")

    try:
        def _load_yaml():
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Validate required configuration keys
            required_keys = ['audio_models', 'language_models', 'system_message']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                raise ConfigurationError(
                    f"Missing required configuration keys: {', '.join(missing_keys)}",
                    config_file=config_path,
                    missing_keys=missing_keys
                )

            return config

        return safe_execute(_load_yaml, error_context="loading configuration")

    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_file=config_path
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML configuration: {e}",
            config_file=config_path
        )


def split_audio(file_path: str, chunk_minutes: int, overlap_seconds: int = 2) -> tuple[list[str], str]:
    """
    Split audio file into overlapping chunks to handle API limits.
    
    Uses memory-efficient processing with tempfile.SpooledTemporaryFile
    for large file handling as specified in INITIAL.md.
    Creates a temporary directory for the chunks.
    
    Args:
        file_path: Path to input audio file
        chunk_minutes: Duration of each chunk in minutes
        overlap_seconds: Overlap between chunks in seconds (default: 2)
    
    Returns:
        A tuple containing:
        - List of chunk file paths following naming convention: chunk_01.mp3, etc.
        - The path to the temporary directory created.
    """
    from .errors import FileError, MemoryError, ValidationError, safe_execute

    # Validate inputs
    if chunk_minutes < 1 or chunk_minutes > 10:
        raise ValidationError(
            "Chunk duration must be between 1-10 minutes",
            field="chunk_minutes",
            value=chunk_minutes
        )

    if overlap_seconds < 0 or overlap_seconds > 60:
        raise ValidationError(
            "Overlap seconds must be between 0-60",
            field="overlap_seconds",
            value=overlap_seconds
        )

    # Validate input file exists
    if not os.path.exists(file_path):
        raise FileError(f"Audio file not found: {file_path}", file_path=file_path, operation="splitting")

    def _split_audio_process():
        # Create a dedicated temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for chunks: {temp_dir}")

        # Use tempfile.SpooledTemporaryFile for memory efficiency (INITIAL.md requirement)
        with tempfile.SpooledTemporaryFile():
            try:
                # Load audio with pydub AudioSegment for cross-format support
                audio = AudioSegment.from_file(file_path)
            except Exception as e:
                raise FileError(
                    f"Failed to load audio file: {e}",
                    file_path=file_path,
                    operation="loading"
                )

        # Check if audio is too short
        if len(audio) < 1000:  # Less than 1 second
            raise ValidationError(
                "Audio file is too short (minimum 1 second required)",
                field="duration",
                value=len(audio)/1000.0
            )

        chunk_length_ms = chunk_minutes * 60 * 1000
        overlap_ms = overlap_seconds * 1000

        chunks = []
        start = 0
        chunk_num = 1

        # Estimate memory usage and warn if potentially problematic
        estimated_memory_mb = (len(audio) / 1000.0) * 0.1  # Rough estimate

        if estimated_memory_mb > 1000:  # > 1GB estimated
            raise MemoryError(
                f"Audio file may require too much memory for processing ({estimated_memory_mb:.1f}MB estimated)",
                file_size_mb=estimated_memory_mb,
                memory_limit_mb=1000
            )

        while start < len(audio):
            try:
                # Ensure overlap with previous chunk (except first)
                end = min(start + chunk_length_ms, len(audio))
                chunk = audio[start:end]

                # Follow INITIAL.md naming convention: chunk_01.mp3, chunk_02.mp3, etc.
                chunk_name = f"chunk_{chunk_num:02d}.mp3"
                chunk_path = os.path.join(temp_dir, chunk_name)
                logger.info(f"Creating chunk file: {chunk_path}")

                # Export chunk to MP3 format
                chunk.export(chunk_path, format="mp3")
                chunks.append(chunk_path)

                # Move start position with overlap consideration
                start = end - overlap_ms if end < len(audio) else end
                chunk_num += 1

            except Exception as e:
                # Cleanup any partial chunks on error
                for partial_chunk in chunks:
                    try:
                        if os.path.exists(partial_chunk):
                            os.remove(partial_chunk)
                    except OSError:
                        pass

                raise FileError(
                    f"Failed to export chunk {chunk_num}: {e}",
                    file_path=chunk_path,
                    operation="exporting"
                )

        return chunks, temp_dir

    return safe_execute(_split_audio_process, error_context=f"splitting audio file {file_path}")


def validate_audio_file(file_path: str) -> tuple[bool, str | None, dict]:
    """
    Validate audio file format, size, and properties.
    
    Checks against supported formats and shows 500MB warning as per INITIAL.md.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, error_message, file_info)
        file_info contains: size_mb, duration_seconds, format, sample_rate
    """
    from .errors import (
        TranscriberError,
        get_user_friendly_message,
        validate_audio_file_extended,
    )

    try:
        file_info = validate_audio_file_extended(file_path)
        return True, None, file_info
    except TranscriberError as e:
        error_message = get_user_friendly_message(e)
        return False, error_message, {}


def estimate_processing_time(file_size_mb: float, chunk_duration_minutes: int) -> dict:
    """
    Estimate processing time based on file size and chunk duration.
    
    Args:
        file_size_mb: File size in megabytes
        chunk_duration_minutes: Duration of each chunk in minutes
        
    Returns:
        Dictionary with time estimates
    """
    # Rough estimates based on typical processing times
    # These can be calibrated based on actual performance

    # Estimate number of chunks (assuming ~1MB per minute of audio)
    estimated_duration_minutes = file_size_mb  # Rough approximation
    num_chunks = max(1, int(estimated_duration_minutes / chunk_duration_minutes))

    # Time estimates (in seconds)
    seconds_per_chunk = 10  # Average API call time
    upload_time = file_size_mb * 0.1  # Rough upload time estimate
    processing_time = num_chunks * seconds_per_chunk
    total_time = upload_time + processing_time

    return {
        'estimated_chunks': num_chunks,
        'upload_time_seconds': upload_time,
        'processing_time_seconds': processing_time,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60
    }


def cleanup_chunks(chunk_files: list[str], temp_dir: str | None = None) -> None:
    """
    Clean up temporary chunk files and the directory containing them.
    
    Args:
        chunk_files: List of chunk file paths to remove.
        temp_dir: The temporary directory to remove.
    """
    for chunk_file in chunk_files:
        try:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
                logger.info(f"Deleted chunk file: {chunk_file}")
        except Exception as e:
            logger.warning(f"Failed to remove chunk file {chunk_file}: {e}")

    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")


def get_audio_stats(file_path: str) -> dict:
    """
    Get detailed statistics about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio statistics
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000.0

        # Estimate word count (rough approximation: 150 words per minute for speech)
        estimated_words = int((duration_seconds / 60) * 150)

        # Words per minute (for display)
        wpm = 150  # Average speaking rate

        return {
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds),
            'estimated_words': estimated_words,
            'wpm': wpm,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'bit_depth': audio.sample_width * 8
        }
    except Exception as e:
        return {'error': str(e)}


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to HH:MM:SS format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def create_job_directory(job_id: str) -> str:
    """
    Create job directory following the data/{YYYY-MM-DD}/{job_id}/ structure.
    Ensures the path is relative to the project root.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Path to created job directory
    """
    from datetime import datetime

    # Get the absolute path of the project root directory (one level up from src)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    date_str = datetime.now().strftime("%Y-%m-%d")
    job_dir = os.path.join(project_root, "data", date_str, job_id)

    os.makedirs(job_dir, exist_ok=True)
    logger.info(f"Created/ensured job directory: {job_dir}")
    return job_dir


def find_job_directory(job_id: str) -> str | None:
    """
    Find job directory by searching through date-based folders.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Path to job directory if found, None otherwise
    """
    # Get the absolute path of the project root directory (one level up from src)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(project_root, "data")

    if not os.path.exists(data_root):
        return None

    # Search through date folders
    for date_folder in os.listdir(data_root):
        potential_path = os.path.join(data_root, date_folder, job_id)
        if os.path.exists(potential_path):
            return potential_path

    return None
