"""
Pytest configuration and shared fixtures for transcriber web app tests.

This file contains pytest configuration and fixtures that are shared
across multiple test files.
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def mock_api_key():
    """Provide a mock OpenAI API key for testing."""
    return "sk-1234567890abcdef1234567890abcdef1234567890abcdef"


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        'audio_models': ['whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'],
        'language_models': ['gpt-4o-mini', 'gpt-4o', 'gpt-4o-speed'],
        'system_message': 'あなたはプロフェッショナルで親切な文字起こしアシスタントです。',
        'default_language': 'auto',
        'default_translation_language': 'Japanese',
        'default_chunk_minutes': 5,
        'max_file_size_mb': 500,
        'supported_formats': ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
        'translation_languages': {
            'Japanese': 'ja',
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Chinese': 'zh'
        },
        'timestamp_format': '# HH:MM:SS --> HH:MM:SS',
        'processing_steps': {
            1: 'Split audio into chunks with pydub',
            2: 'Transcribe chunks with yield-based progress',
            3: 'Real-time markdown updates via gr.Textbox.update',
            4: 'Full-text translation after all chunks complete'
        }
    }


@pytest.fixture
def mock_openai_response():
    """Provide a mock OpenAI API response."""
    response = MagicMock()
    response.text = "Mock transcription result"
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Mock chat response"
    response.choices[0].message.model_dump.return_value = {"content": {"result": "success"}}
    return response


@pytest.fixture
def sample_transcript():
    """Provide a sample transcript for testing."""
    return """# 00:00:00 --> 00:01:00
This is the first minute of the transcript.

# 00:01:00 --> 00:02:00
This is the second minute of the transcript.

# 00:02:00 --> 00:03:00
This is the third minute of the transcript."""


@pytest.fixture
def sample_transcript_json():
    """Provide a sample transcript in JSON format for testing."""
    return [
        {"ts": "00:00:00 --> 00:01:00", "text": "This is the first minute of the transcript."},
        {"ts": "00:01:00 --> 00:02:00", "text": "This is the second minute of the transcript."},
        {"ts": "00:02:00 --> 00:03:00", "text": "This is the third minute of the transcript."}
    ]


@pytest.fixture
def temporary_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_file.write(b"fake audio data for testing")
        temp_path = temp_file.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def mock_audio_segment():
    """Provide a mock AudioSegment for testing."""
    with patch('src.util.AudioSegment') as mock_audio:
        mock_instance = MagicMock()
        mock_instance.__len__ = MagicMock(return_value=60000)  # 60 seconds
        mock_instance.frame_rate = 44100
        mock_instance.channels = 2
        mock_instance.sample_width = 2
        mock_audio.from_file.return_value = mock_instance
        yield mock_audio


@pytest.fixture
def mock_job_directory(tmp_path):
    """Create a temporary job directory structure for testing."""
    # Create job directory structure
    job_id = "test_job_123"
    date_str = "2024-01-15"
    job_dir = tmp_path / "data" / date_str / job_id
    job_dir.mkdir(parents=True)
    
    # Create sample files
    transcript_file = job_dir / "transcript.txt"
    transcript_file.write_text("Sample transcript content")
    
    translation_file = job_dir / "transcript.ja.txt"
    translation_file.write_text("Sample translation content")
    
    metadata_file = job_dir / "metadata.json"
    metadata = {
        "job_id": job_id,
        "timestamp": "2024-01-15T10:30:00",
        "original_filename": "test.mp3",
        "file_info": {"duration_seconds": 300, "size_mb": 15.5},
        "settings": {"default_language": "en"},
        "translation_enabled": True
    }
    metadata_file.write_text(str(metadata).replace("'", '"'))  # JSON format
    
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "data_dir": str(tmp_path / "data"),
        "transcript_content": "Sample transcript content",
        "translation_content": "Sample translation content"
    }


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset module imports between tests to avoid state pollution."""
    # Store original modules
    original_modules = dict(sys.modules)
    
    yield
    
    # Remove any modules that were imported during the test
    modules_to_remove = []
    for module_name in sys.modules:
        if module_name not in original_modules and (
            module_name.startswith('src.') or 
            module_name in ['util', 'errors', 'transcribe', 'llm', 'app']
        ):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]


# Configure pytest options
def pytest_configure(config):
    """Configure pytest with custom options."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Configure async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()