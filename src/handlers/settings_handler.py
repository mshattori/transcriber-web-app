"""
Settings handler - manages application settings and persistence.

Separates settings business logic from UI event handlers.
"""

from typing import Dict, Any, Optional

from util import load_config
from errors import validate_api_key, ValidationError, get_user_friendly_message


class SettingsHandler:
    """Real settings handler."""
    
    def __init__(self):
        self.config = load_config()
    
    def load_default_settings(self) -> Dict[str, Any]:
        """
        Load default settings from config.yaml.
        
        Returns:
            Default settings dictionary
        """
        try:
            return {
                "api_key": "",
                "audio_model": self.config["audio_models"][0] if self.config["audio_models"] else "whisper-1",
                "language_model": self.config["language_models"][0] if self.config["language_models"] else "gpt-4o-mini",
                "system_message": self.config.get("system_message", ""),
                "default_language": self.config.get("default_language", "auto"),
                "default_translation_language": self.config.get("default_translation_language", "Japanese"),
                "chunk_minutes": self.config.get("default_chunk_minutes", 5),
                "translation_enabled": False
            }
        except Exception:
            return {
                "api_key": "",
                "audio_model": "whisper-1",
                "language_model": "gpt-4o-mini", 
                "system_message": "あなたはプロフェッショナルで親切な文字起こしアシスタントです。",
                "default_language": "auto",
                "default_translation_language": "Japanese",
                "chunk_minutes": 5,
                "translation_enabled": False
            }
    
    def save_settings_to_browser_state(
        self,
        settings: Dict[str, Any],
        browser_state_value: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save settings to browser localStorage.
        
        Args:
            settings: Settings to save
            browser_state_value: Current browser state
            
        Returns:
            Updated browser state
        """
        if not isinstance(browser_state_value, dict):
            browser_state_value = {}
        browser_state_value["settings"] = settings
        return browser_state_value
    
    def load_settings_from_browser_state(
        self,
        browser_state_value: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load settings from browser localStorage with fallback to defaults.
        
        Args:
            browser_state_value: Browser state value
            
        Returns:
            Settings dictionary
        """
        if isinstance(browser_state_value, dict) and "settings" in browser_state_value:
            return browser_state_value["settings"]
        
        print("No settings found in browser state, using defaults.")
        return self.load_default_settings()
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate user settings.
        
        Args:
            settings: Settings to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
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
    
    def merge_settings(
        self,
        base_settings: Dict[str, Any],
        ui_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge UI settings with base settings.
        
        Args:
            base_settings: Base settings from browser state
            ui_settings: Settings from UI components
            
        Returns:
            Merged settings dictionary
        """
        merged = base_settings.copy()
        merged.update(ui_settings)
        return merged
    
    def get_config_choices(self) -> Dict[str, list]:
        """
        Get configuration choices for UI dropdowns.
        
        Returns:
            Dictionary with model and language choices
        """
        return {
            "audio_models": self.config.get("audio_models", ["whisper-1"]),
            "language_models": self.config.get("language_models", ["gpt-4o-mini"]),
            "translation_languages": list(self.config.get("translation_languages", {}).keys()),
            "languages": ["auto"] + list(self.config.get("translation_languages", {}).keys())
        }


class MockSettingsHandler:
    """Mock settings handler for UI testing."""
    
    def __init__(self):
        self.mock_config = {
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
    
    def load_default_settings(self) -> Dict[str, Any]:
        """Mock default settings loading."""
        return {
            "api_key": "mock-api-key-for-testing",
            "audio_model": "whisper-1",
            "language_model": "gpt-4o-mini",
            "system_message": "Mock system message for testing",
            "default_language": "auto",
            "default_translation_language": "Japanese",
            "chunk_minutes": 5,
            "translation_enabled": False
        }
    
    def save_settings_to_browser_state(
        self,
        settings: Dict[str, Any],
        browser_state_value: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Mock settings saving - always succeeds."""
        if not isinstance(browser_state_value, dict):
            browser_state_value = {}
        browser_state_value["settings"] = settings
        return browser_state_value
    
    def load_settings_from_browser_state(
        self,
        browser_state_value: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Mock settings loading from browser state."""
        if isinstance(browser_state_value, dict) and "settings" in browser_state_value:
            return browser_state_value["settings"]
        return self.load_default_settings()
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Mock settings validation - always returns valid."""
        return True, ""
    
    def merge_settings(
        self,
        base_settings: Dict[str, Any],
        ui_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock settings merging."""
        merged = base_settings.copy()
        merged.update(ui_settings)
        return merged
    
    def get_config_choices(self) -> Dict[str, list]:
        """Mock configuration choices."""
        return {
            "audio_models": self.mock_config["audio_models"],
            "language_models": self.mock_config["language_models"], 
            "translation_languages": list(self.mock_config["translation_languages"].keys()),
            "languages": ["auto"] + list(self.mock_config["translation_languages"].keys())
        }