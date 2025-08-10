"""
Unit tests for SettingsHandler with test configuration.

Tests both real and mock implementations of settings management.
"""

import os
import pytest
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from handlers.settings_handler import SettingsHandler, MockSettingsHandler
from config.test_config import TestConfig


class TestSettingsHandler:
    """Test suite for real SettingsHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = SettingsHandler()
        self.test_config = TestConfig()
    
    def test_load_default_settings(self):
        """Test loading default settings."""
        settings = self.handler.load_default_settings()
        
        assert isinstance(settings, dict)
        
        # Check required keys
        required_keys = [
            "api_key", "audio_model", "language_model", "system_message",
            "default_language", "default_translation_language", 
            "chunk_minutes", "translation_enabled"
        ]
        
        for key in required_keys:
            assert key in settings
        
        # Check default values
        assert settings["api_key"] == ""  # Should be empty by default
        assert settings["translation_enabled"] is False
        assert isinstance(settings["chunk_minutes"], int)
        assert 1 <= settings["chunk_minutes"] <= 10
    
    def test_save_settings_to_browser_state(self):
        """Test saving settings to browser state."""
        settings = {"api_key": "test-key", "audio_model": "whisper-1"}
        browser_state = {}
        
        updated_state = self.handler.save_settings_to_browser_state(settings, browser_state)
        
        assert "settings" in updated_state
        assert updated_state["settings"] == settings
    
    def test_save_settings_to_browser_state_none_input(self):
        """Test saving settings with None browser state."""
        settings = {"api_key": "test-key"}
        
        updated_state = self.handler.save_settings_to_browser_state(settings, None)
        
        assert isinstance(updated_state, dict)
        assert updated_state["settings"] == settings
    
    def test_load_settings_from_browser_state(self):
        """Test loading settings from browser state."""
        saved_settings = {"api_key": "saved-key", "audio_model": "whisper-large"}
        browser_state = {"settings": saved_settings}
        
        loaded_settings = self.handler.load_settings_from_browser_state(browser_state)
        
        assert loaded_settings == saved_settings
    
    def test_load_settings_from_browser_state_missing(self):
        """Test loading settings when not present in browser state."""
        browser_state = {"other_data": "value"}
        
        loaded_settings = self.handler.load_settings_from_browser_state(browser_state)
        
        # Should return defaults
        default_settings = self.handler.load_default_settings()
        assert loaded_settings == default_settings
    
    def test_load_settings_from_browser_state_none(self):
        """Test loading settings from None browser state."""
        loaded_settings = self.handler.load_settings_from_browser_state(None)
        
        # Should return defaults
        default_settings = self.handler.load_default_settings()
        assert loaded_settings == default_settings
    
    def test_validate_settings_valid(self):
        """Test settings validation with valid settings."""
        test_settings = self.test_config.get_test_settings()
        
        is_valid, error_msg = self.handler.validate_settings(test_settings)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_settings_missing_api_key(self):
        """Test settings validation with missing API key."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["api_key"] = ""
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "API key" in error_msg
    
    def test_validate_settings_missing_audio_model(self):
        """Test settings validation with missing audio model."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["audio_model"] = ""
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Audio model" in error_msg
    
    def test_validate_settings_missing_language_model(self):
        """Test settings validation with missing language model."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["language_model"] = ""
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Language model" in error_msg
    
    def test_validate_settings_invalid_chunk_minutes_too_small(self):
        """Test settings validation with chunk duration too small."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["chunk_minutes"] = 0
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Chunk duration" in error_msg
        assert "1-10 minutes" in error_msg
    
    def test_validate_settings_invalid_chunk_minutes_too_large(self):
        """Test settings validation with chunk duration too large."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["chunk_minutes"] = 15
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Chunk duration" in error_msg
    
    def test_validate_settings_invalid_chunk_minutes_non_numeric(self):
        """Test settings validation with non-numeric chunk duration."""
        invalid_settings = self.test_config.get_test_settings()
        invalid_settings["chunk_minutes"] = "invalid"
        
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        
        assert is_valid is False
        assert "Chunk duration" in error_msg
    
    def test_merge_settings(self):
        """Test merging base settings with UI settings."""
        base_settings = {
            "api_key": "base-key",
            "audio_model": "whisper-1",
            "language_model": "gpt-4",
            "existing_setting": "keep_this"
        }
        
        ui_settings = {
            "audio_model": "whisper-large",  # Override
            "new_setting": "add_this"  # Add
        }
        
        merged = self.handler.merge_settings(base_settings, ui_settings)
        
        assert merged["api_key"] == "base-key"  # Preserved
        assert merged["audio_model"] == "whisper-large"  # Overridden
        assert merged["language_model"] == "gpt-4"  # Preserved
        assert merged["existing_setting"] == "keep_this"  # Preserved
        assert merged["new_setting"] == "add_this"  # Added
    
    def test_get_config_choices(self):
        """Test getting configuration choices."""
        choices = self.handler.get_config_choices()
        
        assert isinstance(choices, dict)
        
        # Check required keys
        assert "audio_models" in choices
        assert "language_models" in choices
        assert "translation_languages" in choices
        assert "languages" in choices
        
        # Check data types
        assert isinstance(choices["audio_models"], list)
        assert isinstance(choices["language_models"], list)
        assert isinstance(choices["translation_languages"], list)
        assert isinstance(choices["languages"], list)
        
        # Check content
        assert len(choices["audio_models"]) > 0
        assert len(choices["language_models"]) > 0
        assert "auto" in choices["languages"]  # Should include auto option


class TestMockSettingsHandler:
    """Test suite for MockSettingsHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.handler = MockSettingsHandler()
        self.test_config = TestConfig()
    
    def test_load_default_settings(self):
        """Test loading default settings from mock."""
        settings = self.handler.load_default_settings()
        
        assert isinstance(settings, dict)
        
        # Should have same structure as real handler
        required_keys = [
            "api_key", "audio_model", "language_model", "system_message",
            "default_language", "default_translation_language", 
            "chunk_minutes", "translation_enabled"
        ]
        
        for key in required_keys:
            assert key in settings
        
        # Mock should have test API key
        assert "mock" in settings["api_key"].lower() or "test" in settings["api_key"].lower()
    
    def test_save_settings_to_browser_state(self):
        """Test saving settings to browser state in mock."""
        settings = {"api_key": "mock-key", "audio_model": "whisper-1"}
        browser_state = {}
        
        updated_state = self.handler.save_settings_to_browser_state(settings, browser_state)
        
        assert "settings" in updated_state
        assert updated_state["settings"] == settings
    
    def test_load_settings_from_browser_state(self):
        """Test loading settings from browser state in mock."""
        saved_settings = {"api_key": "saved-mock-key"}
        browser_state = {"settings": saved_settings}
        
        loaded_settings = self.handler.load_settings_from_browser_state(browser_state)
        
        assert loaded_settings == saved_settings
    
    def test_load_settings_from_browser_state_fallback(self):
        """Test loading settings fallback to defaults in mock."""
        loaded_settings = self.handler.load_settings_from_browser_state({})
        
        default_settings = self.handler.load_default_settings()
        assert loaded_settings == default_settings
    
    def test_validate_settings_always_valid(self):
        """Test that mock validation always returns valid."""
        # Valid settings
        valid_settings = self.test_config.get_test_settings()
        is_valid, error_msg = self.handler.validate_settings(valid_settings)
        assert is_valid is True
        assert error_msg == ""
        
        # Invalid settings (should still pass in mock)
        invalid_settings = {
            "api_key": "",  # Empty
            "audio_model": "",  # Empty
            "chunk_minutes": 100  # Too large
        }
        is_valid, error_msg = self.handler.validate_settings(invalid_settings)
        assert is_valid is True
        assert error_msg == ""
    
    def test_merge_settings(self):
        """Test merging settings in mock."""
        base_settings = {"base": "value", "override": "old"}
        ui_settings = {"override": "new", "added": "value"}
        
        merged = self.handler.merge_settings(base_settings, ui_settings)
        
        assert merged["base"] == "value"
        assert merged["override"] == "new"
        assert merged["added"] == "value"
    
    def test_get_config_choices(self):
        """Test getting configuration choices from mock."""
        choices = self.handler.get_config_choices()
        
        assert isinstance(choices, dict)
        assert "audio_models" in choices
        assert "language_models" in choices
        assert "translation_languages" in choices
        assert "languages" in choices
        
        # Mock should have reasonable test data
        assert "whisper-1" in choices["audio_models"]
        assert "gpt-4o-mini" in choices["language_models"]
        assert "Japanese" in choices["translation_languages"]
        assert "auto" in choices["languages"]
    
    def test_mock_responses_are_instant(self):
        """Test that mock responses are returned instantly."""
        import time
        
        start_time = time.time()
        
        # Perform multiple operations
        self.handler.load_default_settings()
        self.handler.get_config_choices()
        self.handler.validate_settings({})
        
        elapsed_time = time.time() - start_time
        
        # Mock should be very fast (< 10ms)
        assert elapsed_time < 0.01


class TestSettingsHandlerIntegration:
    """Integration tests comparing real and mock handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.real_handler = SettingsHandler()
        self.mock_handler = MockSettingsHandler()
        self.test_config = TestConfig()
    
    def test_handler_interface_compatibility(self):
        """Test that real and mock handlers have compatible interfaces."""
        # Both should have the same methods
        real_methods = [method for method in dir(self.real_handler) if not method.startswith('_')]
        mock_methods = [method for method in dir(self.mock_handler) if not method.startswith('_')]
        
        assert set(real_methods) == set(mock_methods)
    
    def test_both_handlers_return_same_types(self):
        """Test that both handlers return the same data types."""
        # Test load_default_settings
        real_defaults = self.real_handler.load_default_settings()
        mock_defaults = self.mock_handler.load_default_settings()
        
        assert type(real_defaults) == type(mock_defaults)
        assert isinstance(real_defaults, dict)
        assert isinstance(mock_defaults, dict)
        
        # Should have same keys
        assert set(real_defaults.keys()) == set(mock_defaults.keys())
    
    def test_settings_structure_compatibility(self):
        """Test that settings have compatible structure."""
        real_settings = self.real_handler.load_default_settings()
        mock_settings = self.mock_handler.load_default_settings()
        
        # Both should have the same structure
        for key in real_settings:
            assert key in mock_settings
            assert type(real_settings[key]) == type(mock_settings[key])
    
    def test_validation_return_types(self):
        """Test that validation returns compatible types."""
        test_settings = self.test_config.get_test_settings()
        
        real_result = self.real_handler.validate_settings(test_settings)
        mock_result = self.mock_handler.validate_settings(test_settings)
        
        assert type(real_result) == type(mock_result)
        assert isinstance(real_result, tuple)
        assert isinstance(mock_result, tuple)
        assert len(real_result) == 2
        assert len(mock_result) == 2
        
        # Both should return (bool, str)
        assert isinstance(real_result[0], bool)
        assert isinstance(real_result[1], str)
        assert isinstance(mock_result[0], bool)
        assert isinstance(mock_result[1], str)
    
    def test_config_choices_structure(self):
        """Test that config choices have compatible structure."""
        real_choices = self.real_handler.get_config_choices()
        mock_choices = self.mock_handler.get_config_choices()
        
        # Both should return dicts with same keys
        assert type(real_choices) == type(mock_choices)
        assert set(real_choices.keys()) == set(mock_choices.keys())
        
        # All values should be lists
        for key in real_choices:
            assert isinstance(real_choices[key], list)
            assert isinstance(mock_choices[key], list)
    
    def test_browser_state_operations_compatibility(self):
        """Test that browser state operations work the same way."""
        settings = {"test": "value"}
        
        # Save operations
        real_saved = self.real_handler.save_settings_to_browser_state(settings, {})
        mock_saved = self.mock_handler.save_settings_to_browser_state(settings, {})
        
        assert type(real_saved) == type(mock_saved)
        assert real_saved["settings"] == mock_saved["settings"]
        
        # Load operations
        browser_state = {"settings": settings}
        
        real_loaded = self.real_handler.load_settings_from_browser_state(browser_state)
        mock_loaded = self.mock_handler.load_settings_from_browser_state(browser_state)
        
        assert type(real_loaded) == type(mock_loaded)
        assert real_loaded == mock_loaded