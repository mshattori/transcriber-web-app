"""
Handler modules for transcriber web app.

Separates UI logic from business logic using the handler pattern.
"""

from .audio_handler import AudioHandler, MockAudioHandler
from .chat_handler import ChatHandler, MockChatHandler
from .history_handler import HistoryHandler, MockHistoryHandler
from .settings_handler import MockSettingsHandler, SettingsHandler

__all__ = [
    "AudioHandler",
    "MockAudioHandler",
    "ChatHandler",
    "MockChatHandler",
    "HistoryHandler",
    "MockHistoryHandler",
    "SettingsHandler",
    "MockSettingsHandler",
]
