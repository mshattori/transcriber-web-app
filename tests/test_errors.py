"""
Unit tests for errors.py module.

Tests custom exceptions, error handling utilities, and validation functions.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from errors import (
    ErrorType,
    TranscriberError,
    ValidationError,
    FileError,
    APIError,
    NetworkError,
    ConfigurationError,
    MemoryError,
    UIError,
    validate_api_key,
    validate_file_path,
    validate_audio_file_extended,
    handle_openai_error,
    get_user_friendly_message,
    safe_execute
)


class TestErrorTypes:
    """Test custom error types and their functionality."""
    
    def test_transcriber_error_basic(self):
        """Test basic TranscriberError functionality."""
        error = TranscriberError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_type == ErrorType.PROCESSING
        assert error.details == {}
    
    def test_transcriber_error_with_details(self):
        """Test TranscriberError with details."""
        details = {"key": "value", "number": 42}
        error = TranscriberError("Test error", ErrorType.VALIDATION, details)
        
        assert error.error_type == ErrorType.VALIDATION
        assert error.details == details
        
        error_dict = error.to_dict()
        assert error_dict["type"] == "validation"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == details
    
    def test_validation_error(self):
        """Test ValidationError specific functionality."""
        error = ValidationError("Invalid field", field="test_field", value="invalid_value")
        
        assert error.field == "test_field"
        assert error.value == "invalid_value"
        assert error.error_type == ErrorType.VALIDATION
    
    def test_file_error(self):
        """Test FileError specific functionality."""
        error = FileError("File not found", file_path="/path/to/file", operation="reading")
        
        assert error.file_path == "/path/to/file"
        assert error.operation == "reading"
        assert error.error_type == ErrorType.FILE_IO
    
    def test_api_error(self):
        """Test APIError specific functionality."""
        error = APIError("API failed", api_name="OpenAI", status_code=429, retry_after=60)
        
        assert error.api_name == "OpenAI"
        assert error.status_code == 429
        assert error.retry_after == 60
        assert error.error_type == ErrorType.API


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key."""
        valid_key = "sk-1234567890abcdef1234567890abcdef"
        
        # Should not raise an exception
        validate_api_key(valid_key)
    
    def test_validate_api_key_empty(self):
        """Test API key validation with empty key."""
        with pytest.raises(ValidationError) as exc_info:
            validate_api_key("")
        
        assert "API key is required" in str(exc_info.value)
        assert exc_info.value.field == "api_key"
    
    def test_validate_api_key_wrong_format(self):
        """Test API key validation with wrong format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_api_key("invalid-key-format")
        
        assert "API key must start with 'sk-'" in str(exc_info.value)
    
    def test_validate_api_key_too_short(self):
        """Test API key validation with too short key."""
        with pytest.raises(ValidationError) as exc_info:
            validate_api_key("sk-123")
        
        assert "API key is too short" in str(exc_info.value)
    
    def test_validate_file_path_valid_existing(self):
        """Test file path validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Should not raise an exception
            validate_file_path(temp_path, must_exist=True)
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_path_missing_required(self):
        """Test file path validation with missing required file."""
        with pytest.raises(FileError) as exc_info:
            validate_file_path("/nonexistent/file/path", must_exist=True)
        
        assert "File not found" in str(exc_info.value)
    
    def test_validate_file_path_empty(self):
        """Test file path validation with empty path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path("", must_exist=False)
        
        assert "File path is required" in str(exc_info.value)


class TestOpenAIErrorHandling:
    """Test OpenAI error handling functionality."""
    
    def test_handle_rate_limit_error(self):
        """Test handling of rate limit errors."""
        rate_limit_error = Exception("Rate limit exceeded (429)")
        
        api_error = handle_openai_error(rate_limit_error)
        
        assert isinstance(api_error, APIError)
        assert api_error.status_code == 429
        assert "Rate limit exceeded" in api_error.message
        assert api_error.retry_after == 60
    
    def test_handle_auth_error(self):
        """Test handling of authentication errors."""
        auth_error = Exception("Invalid API key (401)")
        
        api_error = handle_openai_error(auth_error)
        
        assert isinstance(api_error, APIError)
        assert api_error.status_code == 401
        assert "Invalid API key" in api_error.message
    
    def test_handle_quota_error(self):
        """Test handling of quota exceeded errors."""
        quota_error = Exception("Quota exceeded - insufficient funds")
        
        api_error = handle_openai_error(quota_error)
        
        assert isinstance(api_error, APIError)
        assert api_error.status_code == 402
        assert "quota exceeded" in api_error.message.lower()
    
    def test_handle_generic_error(self):
        """Test handling of generic errors."""
        generic_error = Exception("Something went wrong")
        
        api_error = handle_openai_error(generic_error)
        
        assert isinstance(api_error, APIError)
        assert "Something went wrong" in api_error.message


class TestUserFriendlyMessages:
    """Test user-friendly error message generation."""
    
    def test_get_user_friendly_message_api_key(self):
        """Test user-friendly message for API key error."""
        error = APIError("Invalid API key", status_code=401)
        
        message = get_user_friendly_message(error)
        
        assert "Please enter your OpenAI API key" in message
    
    def test_get_user_friendly_message_rate_limit(self):
        """Test user-friendly message for rate limit error."""
        error = APIError("Rate limit exceeded", status_code=429, retry_after=120)
        
        message = get_user_friendly_message(error)
        
        assert "Rate limit exceeded" in message
        assert "120 seconds" in message
    
    def test_get_user_friendly_message_file_size(self):
        """Test user-friendly message for file size error."""
        error = ValidationError("File too large", field="file_size")
        
        message = get_user_friendly_message(error)
        
        assert "Maximum size allowed" in message
    
    def test_get_user_friendly_message_unsupported_format(self):
        """Test user-friendly message for unsupported format error."""
        error = ValidationError("Unsupported format", field="file_extension")
        
        message = get_user_friendly_message(error)
        
        assert "Unsupported audio format" in message
    
    def test_get_user_friendly_message_generic(self):
        """Test user-friendly message for generic error."""
        error = TranscriberError("Generic processing error")
        
        message = get_user_friendly_message(error)
        
        assert message == "Generic processing error"


class TestSafeExecute:
    """Test safe execution utility."""
    
    def test_safe_execute_success(self):
        """Test safe execution with successful function."""
        def successful_function():
            return "success"
        
        result = safe_execute(successful_function, error_context="test")
        
        assert result == "success"
    
    def test_safe_execute_transcriber_error(self):
        """Test safe execution with TranscriberError."""
        def failing_function():
            raise ValidationError("Test validation error")
        
        with pytest.raises(ValidationError):
            safe_execute(failing_function, error_context="test")
    
    def test_safe_execute_generic_error(self):
        """Test safe execution with generic error."""
        def failing_function():
            raise ValueError("Generic error")
        
        with pytest.raises(ValidationError) as exc_info:
            safe_execute(failing_function, error_context="test")
        
        assert "Invalid input" in str(exc_info.value)
    
    def test_safe_execute_with_args(self):
        """Test safe execution with function arguments."""
        def function_with_args(x, y, z=None):
            return x + y + (z or 0)
        
        result = safe_execute(function_with_args, 1, 2, z=3, error_context="test")
        
        assert result == 6


if __name__ == "__main__":
    pytest.main([__file__])