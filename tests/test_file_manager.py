"""
Tests for file management functionality.
"""

import os
import json
import tempfile
import shutil
import zipfile
import pytest
from src.file_manager import (
    save_transcription_files,
    save_job_metadata,
    load_job_files,
    load_job_metadata,
    get_display_content_from_job,
    create_download_package
)


@pytest.fixture
def temp_job_dir():
    """Create a temporary job directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_transcript():
    """Sample transcript text."""
    return """# 00:00:00 --> 00:02:30
This is a test transcript.
It has multiple lines.

# 00:02:30 --> 00:05:00
This is the second section."""


@pytest.fixture
def sample_translation():
    """Sample translation text."""
    return """# 00:00:00 --> 00:02:30
これはテスト転写です。
複数行があります。

# 00:02:30 --> 00:05:00
これは2番目のセクションです。"""


@pytest.fixture
def sample_settings():
    """Sample settings dictionary."""
    return {
        "translation_enabled": True,
        "default_translation_language": "Japanese",
        "audio_model": "whisper-1",
        "language_model": "gpt-4o-mini"
    }


class TestSaveTranscriptionFiles:
    """Test transcription file saving functionality."""
    
    def test_save_transcript_only(self, temp_job_dir, sample_transcript):
        """Test saving transcript only (no translation)."""
        settings = {"translation_enabled": False}
        
        saved_files = save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            "", 
            settings
        )
        
        # Check that only transcript file was saved
        assert 'transcript' in saved_files
        assert 'translation' not in saved_files
        assert 'integrated' not in saved_files
        
        # Check file content
        with open(saved_files['transcript'], 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == sample_transcript
    
    def test_save_transcript_and_translation(
        self, 
        temp_job_dir, 
        sample_transcript, 
        sample_translation, 
        sample_settings
    ):
        """Test saving transcript and translation."""
        saved_files = save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            sample_translation, 
            sample_settings
        )
        
        # Check that all files were saved
        assert 'transcript' in saved_files
        assert 'translation' in saved_files
        assert 'integrated' in saved_files
        
        # Check transcript file
        with open(saved_files['transcript'], 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == sample_transcript
        
        # Check translation file
        with open(saved_files['translation'], 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == sample_translation
        
        # Check integrated file exists and has content
        with open(saved_files['integrated'], 'r', encoding='utf-8') as f:
            content = f.read()
        assert "This is a test transcript." in content
        assert "これはテスト転写です。" in content
        assert "Translation" in content
    
    def test_save_with_default_language_code(self, temp_job_dir, sample_transcript, sample_translation):
        """Test saving with default language code when not specified."""
        settings = {"translation_enabled": True}
        
        saved_files = save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            sample_translation, 
            settings
        )
        
        # Check that translation file uses default 'ja' extension
        translation_file = saved_files['translation']
        assert translation_file.endswith('transcript.ja.txt')


class TestSaveJobMetadata:
    """Test job metadata saving functionality."""
    
    def test_save_metadata(self, temp_job_dir, sample_settings):
        """Test saving job metadata."""
        job_id = "test-job-123"
        original_filename = "test_audio.mp3"
        file_info = {"size_mb": 2.5, "duration_seconds": 150.0}
        transcript_stats = {"word_count": 100, "duration": 150.0}
        saved_files = {"transcript": "transcript.txt", "translation": "transcript.ja.txt"}
        
        metadata_path = save_job_metadata(
            temp_job_dir,
            job_id,
            original_filename,
            file_info,
            sample_settings,
            transcript_stats,
            saved_files
        )
        
        # Check that metadata file was created
        assert os.path.exists(metadata_path)
        
        # Check metadata content
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata['job_id'] == job_id
        assert metadata['original_filename'] == original_filename
        assert metadata['translation_enabled'] == True
        assert metadata['translation_available'] == True
        assert 'timestamp' in metadata
        assert metadata['files']['transcript'] == 'transcript.txt'
        assert metadata['files']['translation'] == 'transcript.ja.txt'


class TestLoadJobFiles:
    """Test job file loading functionality."""
    
    def test_load_files(
        self, 
        temp_job_dir, 
        sample_transcript, 
        sample_translation, 
        sample_settings
    ):
        """Test loading job files."""
        # First save files
        save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            sample_translation, 
            sample_settings
        )
        
        # Then load them
        transcript, translation, integrated = load_job_files(temp_job_dir)
        
        assert transcript == sample_transcript
        assert translation == sample_translation
        assert "This is a test transcript." in integrated
        assert "これはテスト転写です。" in integrated
        assert "Translation" in integrated
    
    def test_load_transcript_only(self, temp_job_dir, sample_transcript):
        """Test loading transcript only."""
        # Save only transcript
        settings = {"translation_enabled": False}
        save_transcription_files(temp_job_dir, sample_transcript, "", settings)
        
        # Load files
        transcript, translation, integrated = load_job_files(temp_job_dir)
        
        assert transcript == sample_transcript
        assert translation == ""
        assert integrated == ""


class TestGetDisplayContentFromJob:
    """Test display content retrieval functionality."""
    
    def test_get_integrated_display(
        self, 
        temp_job_dir, 
        sample_transcript, 
        sample_translation, 
        sample_settings
    ):
        """Test getting integrated display content."""
        # Save files with translation
        save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            sample_translation, 
            sample_settings
        )
        
        # Get display content
        content = get_display_content_from_job(temp_job_dir)
        
        # Should return integrated display
        assert "This is a test transcript." in content
        assert "これはテスト転写です。" in content
        assert "Translation" in content
    
    def test_get_transcript_only(self, temp_job_dir, sample_transcript):
        """Test getting transcript only when no translation."""
        # Save only transcript
        settings = {"translation_enabled": False}
        save_transcription_files(temp_job_dir, sample_transcript, "", settings)
        
        # Get display content
        content = get_display_content_from_job(temp_job_dir)
        
        # Should return transcript only
        assert content == sample_transcript


class TestCreateDownloadPackage:
    """Test download package creation functionality."""
    
    def test_create_single_file_download(self, temp_job_dir, sample_transcript):
        """Test creating download package with single file."""
        # Save only transcript
        settings = {"translation_enabled": False}
        save_transcription_files(temp_job_dir, sample_transcript, "", settings)
        
        # Create download package
        download_path = create_download_package(temp_job_dir, "test-job")
        
        # Should return path to single file
        assert download_path.endswith("transcript.txt")
        assert os.path.exists(download_path)
    
    def test_create_zip_download(
        self, 
        temp_job_dir, 
        sample_transcript, 
        sample_translation, 
        sample_settings
    ):
        """Test creating download package with multiple files (ZIP)."""
        # Save files with translation
        save_transcription_files(
            temp_job_dir, 
            sample_transcript, 
            sample_translation, 
            sample_settings
        )
        
        # Create download package
        download_path = create_download_package(temp_job_dir, "test-job")
        
        # Should return path to ZIP file
        assert download_path.endswith(".zip")
        assert os.path.exists(download_path)
        
        # Check ZIP contents
        with zipfile.ZipFile(download_path, 'r') as zipf:
            files = zipf.namelist()
            assert "transcript.txt" in files
            assert "transcript.ja.txt" in files
            assert "transcript_integrated.txt" in files