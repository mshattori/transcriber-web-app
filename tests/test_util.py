"""
Unit tests for util.py module.

Tests audio utilities, configuration loading, and validation functions.
"""

import os
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
import pytest
import yaml
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Handle relative imports by adding the src directory as a package
import importlib.util
import types

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import modules with proper handling
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
errors_module = import_module_from_path('errors', os.path.join(src_dir, 'errors.py'))
util_module = import_module_from_path('util', os.path.join(src_dir, 'util.py'))

# Import specific functions/classes
from errors import ConfigurationError, ValidationError
from util import (
    load_config,
    validate_audio_file, 
    estimate_processing_time,
    get_audio_stats,
    format_duration,
    create_job_directory
)


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_config_success(self):
        """Test successful config loading."""
        mock_config = {
            'audio_models': ['whisper-1'],
            'language_models': ['gpt-4o-mini'],
            'system_message': 'Test message'
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
            config = load_config("test_config.yaml")
            
        assert config == mock_config
        assert 'audio_models' in config
        assert 'language_models' in config
        assert 'system_message' in config
    
    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config("nonexistent_config.yaml")
        
        assert "Configuration file not found" in str(exc_info.value)
    
    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with pytest.raises(ConfigurationError) as exc_info:
                load_config("invalid_config.yaml")
        
        assert "Invalid YAML configuration" in str(exc_info.value)
    
    def test_load_config_missing_required_keys(self):
        """Test config loading with missing required keys."""
        incomplete_config = {'audio_models': ['whisper-1']}
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(incomplete_config))):
            with pytest.raises(ConfigurationError) as exc_info:
                load_config("incomplete_config.yaml")
        
        assert "Missing required configuration keys" in str(exc_info.value)


class TestValidateAudioFile:
    """Test audio file validation functionality."""
    
    @patch('src.util.validate_audio_file_extended')
    def test_validate_audio_file_success(self, mock_validate_extended):
        """Test successful audio file validation."""
        mock_file_info = {
            'size_mb': 10.5,
            'duration_seconds': 300,
            'format': '.mp3',
            'sample_rate': 44100,
            'channels': 2,
            'needs_warning': False
        }
        mock_validate_extended.return_value = mock_file_info
        
        is_valid, error_msg, file_info = validate_audio_file("test.mp3")
        
        assert is_valid is True
        assert error_msg is None
        assert file_info == mock_file_info
    
    @patch('src.util.validate_audio_file_extended')
    def test_validate_audio_file_failure(self, mock_validate_extended):
        """Test audio file validation failure."""
        mock_validate_extended.side_effect = ValidationError("File too large", field="file_size")
        
        is_valid, error_msg, file_info = validate_audio_file("large_file.mp3")
        
        assert is_valid is False
        assert "File too large" in error_msg
        assert file_info == {}


class TestEstimateProcessingTime:
    """Test processing time estimation functionality."""
    
    def test_estimate_processing_time_small_file(self):
        """Test processing time estimation for small file."""
        estimates = estimate_processing_time(file_size_mb=5, chunk_duration_minutes=5)
        
        assert 'estimated_chunks' in estimates
        assert 'upload_time_seconds' in estimates
        assert 'processing_time_seconds' in estimates
        assert 'total_time_seconds' in estimates
        assert 'total_time_minutes' in estimates
        
        assert estimates['estimated_chunks'] == 1
        assert estimates['total_time_seconds'] > 0
    
    def test_estimate_processing_time_large_file(self):
        """Test processing time estimation for large file."""
        estimates = estimate_processing_time(file_size_mb=50, chunk_duration_minutes=5)
        
        assert estimates['estimated_chunks'] == 10
        assert estimates['total_time_seconds'] > 50  # Should be higher for larger file


class TestGetAudioStats:
    """Test audio statistics functionality."""
    
    @patch('src.util.AudioSegment')
    def test_get_audio_stats_success(self, mock_audio_segment):
        """Test successful audio stats extraction."""
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=60000)  # 60 seconds
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.sample_width = 2
        
        mock_audio_segment.from_file.return_value = mock_audio
        
        stats = get_audio_stats("test.mp3")
        
        assert 'duration_seconds' in stats
        assert 'duration_formatted' in stats
        assert 'estimated_words' in stats
        assert 'sample_rate' in stats
        assert 'channels' in stats
        assert 'bit_depth' in stats
        
        assert stats['duration_seconds'] == 60.0
        assert stats['sample_rate'] == 44100
        assert stats['channels'] == 2
        assert stats['bit_depth'] == 16
    
    @patch('src.util.AudioSegment')
    def test_get_audio_stats_error(self, mock_audio_segment):
        """Test audio stats extraction with error."""
        mock_audio_segment.from_file.side_effect = Exception("Failed to load audio")
        
        stats = get_audio_stats("invalid.mp3")
        
        assert 'error' in stats
        assert "Failed to load audio" in stats['error']


class TestFormatDuration:
    """Test duration formatting functionality."""
    
    def test_format_duration_seconds_only(self):
        """Test formatting duration with only seconds."""
        assert format_duration(45) == "00:45"
    
    def test_format_duration_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        assert format_duration(125) == "02:05"
    
    def test_format_duration_hours_minutes_seconds(self):
        """Test formatting duration with hours, minutes, and seconds."""
        assert format_duration(3665) == "01:01:05"
    
    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        assert format_duration(0) == "00:00"


class TestCreateJobDirectory:
    """Test job directory creation functionality."""
    
    @patch('src.util.os.makedirs')
    @patch('src.util.datetime')
    def test_create_job_directory(self, mock_datetime, mock_makedirs):
        """Test job directory creation."""
        mock_datetime.now().strftime.return_value = "2024-01-15"
        
        job_dir = create_job_directory("test_job_123")
        
        expected_dir = os.path.join("data", "2024-01-15", "test_job_123")
        assert job_dir == expected_dir
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])