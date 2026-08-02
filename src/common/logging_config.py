"""Central logging configuration for Auto Maple."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_LOGGER_NAME = "auto_maple"
_CONFIGURED = False


def project_root() -> Path:
    """Return the project root for source and frozen builds."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def configure_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure console and rotating file logging exactly once."""
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if _CONFIGURED:
        return logger

    directory = Path(log_dir) if log_dir is not None else project_root() / "logs"
    directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        directory / "auto-maple.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    _CONFIGURED = True
    logger.debug("Logging initialized at %s", directory)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the Auto Maple namespace."""
    configure_logging()
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
