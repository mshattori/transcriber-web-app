"""
Configuration management for transcriber web app.

Provides environment-based configuration with .env override support.
"""

from .app_config import AppConfig
from .test_config import TestConfig

__all__ = ["AppConfig", "TestConfig"]