"""
logger.py — Centralised logging configuration for CricVision.

Provides a single ``get_logger(name)`` factory that returns child loggers
rooted under ``CricVision``.  Two handlers are attached at first call:

* **Console** (``StreamHandler``) — INFO level, concise format.
* **File** (``RotatingFileHandler``) — DEBUG level, appends to
  ``logs/cricvision.log`` with 5 MB rotation and 5 backups.

Usage::

    from ball_tracking.logger import get_logger
    logger = get_logger(__name__)
    logger.info("processing started")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from ball_tracking.config import (
    ROOT_LOGGER_NAME, LOG_DIR, LOG_FILE,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    LOG_FILE_FMT, LOG_CONSOLE_FMT, LOG_DATE_FMT,
)

# ── Internal state ───────────────────────────────────────────────────────────
_configured = False


def _configure_root() -> None:
    """Attach console + rotating-file handlers to the root CricVision logger."""
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)            # allow everything through

    # ── Console handler (INFO) ───────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_CONSOLE_FMT, datefmt=LOG_DATE_FMT))
    root.addHandler(console)

    # ── File handler (DEBUG, rotating) ───────────────────────────────────
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FILE_FMT, datefmt=LOG_DATE_FMT))
    root.addHandler(file_handler)

    _configured = True


# ── Public API ───────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the CricVision root.

    Calling ``get_logger("ball_tracking.orchestrator")`` yields a logger named
    ``CricVision.ball_tracking.orchestrator``.
    """
    _configure_root()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
