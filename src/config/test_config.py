"""
Test configuration management with .env override support.

Provides test-specific configuration for business logic testing.
"""

import os
from typing import Any

from .app_config import AppConfig


class TestConfig(AppConfig):
    """Test configuration manager with enhanced test settings."""

    def __init__(self):
        """Initialize test configuration."""
        super().__init__(env="test")
        self._setup_test_data()

    def _setup_test_data(self):
        """Setup test-specific data and paths."""
        # Test audio files
        self.test_audio_files = {
            "small": self.get("test_audio_file", "tests/data/test_audio_small.wav"),
            "medium": "tests/data/test_audio_medium.wav",
            "large": "tests/data/test_audio_large.wav",
            "japanese": "tests/data/test_audio_japanese.wav",
            "english": "tests/data/test_audio_english.wav"
        }

        # Test output directory
        self.test_output_dir = self.get("test_output_dir", "tests/output")

        # Test settings configurations
        self.test_settings = {
            "basic": {
                "api_key": self.get("openai_api_key", "test-api-key"),
                "audio_model": "whisper-1",
                "language_model": "gpt-4o-mini",
                "default_language": "auto",
                "chunk_minutes": 5,
                "translation_enabled": False,
                "system_message": "Test system message"
            },
            "with_translation": {
                "api_key": self.get("openai_api_key", "test-api-key"),
                "audio_model": "whisper-1",
                "language_model": "gpt-4o-mini",
                "default_language": "auto",
                "default_translation_language": "Japanese",
                "chunk_minutes": 5,
                "translation_enabled": True,
                "system_message": "Test system message with translation"
            },
            "japanese": {
                "api_key": self.get("openai_api_key", "test-api-key"),
                "audio_model": "whisper-1",
                "language_model": "gpt-4o-mini",
                "default_language": "ja",
                "default_translation_language": "English",
                "chunk_minutes": 3,
                "translation_enabled": True,
                "system_message": "Japanese test system message"
            }
        }

    def get_test_audio_file(self, file_type: str = "small") -> str:
        """
        Get test audio file path.
        
        Args:
            file_type: Type of test audio file (small, medium, large, japanese, english)
            
        Returns:
            Path to test audio file
        """
        return self.test_audio_files.get(file_type, self.test_audio_files["small"])

    def get_test_settings(self, settings_type: str = "basic") -> dict[str, Any]:
        """
        Get test settings configuration.
        
        Args:
            settings_type: Type of settings (basic, with_translation, japanese)
            
        Returns:
            Test settings dictionary
        """
        return self.test_settings.get(settings_type, self.test_settings["basic"]).copy()

    def get_output_dir(self) -> str:
        """
        Get test output directory.
        
        Returns:
            Test output directory path
        """
        return self.test_output_dir

    def setup_test_env(self):
        """Setup test environment (create directories, etc.)."""
        # Create test output directory
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir, exist_ok=True)

        # Create test data directory if it doesn't exist
        test_data_dir = "tests/data"
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir, exist_ok=True)

    def cleanup_test_env(self):
        """Cleanup test environment (remove temporary files, etc.)."""
        import shutil

        # Remove test output directory if it exists
        if os.path.exists(self.test_output_dir):
            try:
                shutil.rmtree(self.test_output_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup test output directory: {e}")

    def has_real_api_key(self) -> bool:
        """
        Check if real API key is available for integration tests.
        
        Returns:
            True if real API key is configured
        """
        return self.has_api_key()

    def get_mock_responses(self) -> dict[str, Any]:
        """
        Get mock API responses for testing.
        
        Returns:
            Dictionary of mock responses
        """
        return {
            "transcription": {
                "text": "This is a mock transcription result for testing purposes.",
                "word_count": 10,
                "total_duration": 30.0,
                "processing_time": 2.5
            },
            "translation": {
                "translated_text": "これはテスト用のモック翻訳結果です。",
                "source_language": "en",
                "target_language": "ja"
            },
            "chat": {
                "response": "This is a mock chat response for testing.",
                "usage": {"total_tokens": 50}
            }
        }
