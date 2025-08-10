"""
Application configuration management with environment variable overrides.

Handles loading configuration from YAML and overriding with environment variables.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from util import load_config


class AppConfig:
    """Application configuration manager."""
    
    def __init__(self, env: str = "prod"):
        """
        Initialize application configuration.
        
        Args:
            env: Environment name (prod, test, mock-ui)
        """
        self.env = env
        self.config: Dict[str, Any] = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from YAML and environment variables."""
        # Load .env file if it exists
        env_path = self._find_env_file()
        if env_path:
            load_dotenv(env_path)
        
        # Load base configuration from YAML
        try:
            self.config = load_config()
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
            self.config = self._get_default_config()
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _find_env_file(self) -> Optional[str]:
        """Find .env file in project structure."""
        # Try different locations for .env file
        possible_paths = [
            ".env",  # Current directory
            "../.env",  # Parent directory (from src/)
            "../../.env",  # Project root (from src/config/)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when YAML loading fails."""
        return {
            "audio_models": ["whisper-1", "whisper-large"],
            "language_models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "translation_languages": {
                "Japanese": "ja",
                "English": "en", 
                "Spanish": "es",
                "French": "fr",
                "German": "de"
            },
            "system_message": "あなたはプロフェッショナルで親切な文字起こしアシスタントです。",
            "default_language": "auto",
            "default_translation_language": "Japanese",
            "default_chunk_minutes": 5
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_overrides = {
            # API Configuration
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            
            # Model Configuration
            "default_audio_model": os.getenv("DEFAULT_AUDIO_MODEL"),
            "default_language_model": os.getenv("DEFAULT_LANGUAGE_MODEL"),
            
            # Default Settings
            "default_language": os.getenv("DEFAULT_LANGUAGE"),
            "default_translation_language": os.getenv("DEFAULT_TRANSLATION_LANGUAGE"),
            "default_chunk_minutes": self._parse_int_env("DEFAULT_CHUNK_MINUTES"),
            
            # Test Configuration
            "test_audio_file": os.getenv("TEST_AUDIO_FILE"),
            "test_output_dir": os.getenv("TEST_OUTPUT_DIR"),
        }
        
        # Apply non-None overrides
        for key, value in env_overrides.items():
            if value is not None:
                self.config[key] = value
    
    def _parse_int_env(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Parse integer environment variable."""
        value = os.getenv(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                print(f"Warning: Invalid integer value for {key}: {value}")
        return default
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def is_test_env(self) -> bool:
        """Check if running in test environment."""
        return self.env in ["test", "mock-ui"]
    
    def is_mock_env(self) -> bool:
        """Check if running in mock environment."""
        return self.env == "mock-ui"
    
    def get_app_env(self) -> str:
        """Get current application environment."""
        return self.env
    
    def has_api_key(self) -> bool:
        """Check if API key is configured."""
        api_key = self.get("openai_api_key")
        return bool(api_key and api_key.strip() and not api_key.startswith("your_"))
    
    def get_test_audio_file(self) -> Optional[str]:
        """Get test audio file path."""
        return self.get("test_audio_file")
    
    def get_test_output_dir(self) -> Optional[str]:
        """Get test output directory."""
        return self.get("test_output_dir")