"""
utils/logger.py — Loguru-based logging with rotating file + stderr output.
Singleton logger accessible via get_logger().
"""

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from utils.config import get_config

_logger_configured: bool = False


def setup_logger() -> None:
    """Configure the loguru logger with rotating file and stderr output.

    - Creates LOG_DIR if it doesn't exist.
    - Rotates every day at midnight.
    - Retains logs for 30 days.
    - Format includes timestamp, level, module, function, line, and message.
    - Also outputs colorized logs to stderr.
    """
    global _logger_configured
    if _logger_configured:
        return

    config = get_config()
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers to avoid duplicates
    _loguru_logger.remove()

    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )

    # File handler with rotation
    _loguru_logger.add(
        log_dir / "trading_machine_{time}.log",
        rotation="00:00",
        retention="30 days",
        format=log_format,
        level="DEBUG",
        encoding="utf-8",
        enqueue=True,
    )

    # Stderr handler with colors
    _loguru_logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG",
        colorize=True,
    )

    _logger_configured = True


def get_logger():
    """Return the configured loguru logger singleton.

    If the logger hasn't been set up yet, it will be configured automatically.
    """
    if not _logger_configured:
        setup_logger()
    return _loguru_logger
