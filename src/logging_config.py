"""
Centralized logging configuration for the transcriber web app.

- Outputs to console and a per-environment log file under logs/.
- Supports a single environment variable LOG_LEVEL to override level.
- No log rotation; files are opened in append mode.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


DEFAULT_LEVELS = {
    "prod": logging.INFO,
    "test": logging.DEBUG,
    "mock-ui": logging.DEBUG,
}


def _parse_level(level_str: str | None, default: int) -> int:
    if not level_str:
        return default
    try:
        return getattr(logging, level_str.upper())
    except AttributeError:
        return default


def init_logging(app_env: str | None = None) -> logging.Logger:
    """
    Initialize root logging handlers.

    - Determines environment from APP_ENV if not provided.
    - Creates logs/app-{env}.log (append) without rotation.
    - Respects LOG_LEVEL for global override.
    - Reduces noisy third-party loggers to WARNING.
    """
    env = (app_env or os.getenv("APP_ENV") or "prod").strip() or "prod"

    # Determine level
    default_level = DEFAULT_LEVELS.get(env, logging.INFO)
    level = _parse_level(os.getenv("LOG_LEVEL"), default_level)

    root = logging.getLogger()

    # If already configured (handlers exist), just set level and return
    if root.handlers:
        root.setLevel(level)
        return logging.getLogger(__name__)

    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / f"app-{env}.log"
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as e:
        # Fall back to console-only if file logging cannot be initialized
        logging.getLogger(__name__).warning(
            f"File logging disabled (failed to init logs dir/file): {e}"
        )

    # Reduce noise from third-party libraries
    for noisy in ("urllib3", "openai", "gradio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(__name__)

