import logging
import os
from typing import Optional


_ROOT_LOGGER_NAME = "diffcg"


def _get_root_logger() -> logging.Logger:
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    # Avoid duplicate handlers if configure() is called multiple times
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers) and not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the diffcg namespace.

    Examples
    --------
    >>> from diffcg.util.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.debug("message")
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return _get_root_logger()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure(level: Optional[str] = None,
              fmt: Optional[str] = None,
              datefmt: Optional[str] = None) -> None:
    """Configure the package logger.

    If not provided, the level and format are read from environment variables:
      - DIFFCG_LOG_LEVEL (default: INFO)
      - DIFFCG_LOG_FORMAT (default: "%(asctime)s | %(levelname)s | %(name)s: %(message)s")

    Notes
    -----
    This function is safe to call multiple times. It updates the existing handler
    on the root diffcg logger instead of adding duplicates.
    """
    level_str = level or os.getenv("DIFFCG_LOG_LEVEL", "INFO")
    level_value = getattr(logging, level_str.upper(), logging.INFO)

    fmt_str = fmt or os.getenv(
        "DIFFCG_LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )

    datefmt_str = datefmt or os.getenv("DIFFCG_LOG_DATEFMT", "%H:%M:%S")

    root = _get_root_logger()
    root.setLevel(level_value)

    # Replace existing non-null handlers with a single StreamHandler
    handler: Optional[logging.Handler] = None
    for h in root.handlers:
        if not isinstance(h, logging.NullHandler):
            handler = h
            break

    if handler is None:
        # Remove NullHandler and attach a StreamHandler
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]
        handler = logging.StreamHandler()
        root.addHandler(handler)

    handler.setLevel(level_value)
    handler.setFormatter(logging.Formatter(fmt=fmt_str, datefmt=datefmt_str))


def set_level(level: str) -> None:
    """Set the logging level of the root diffcg logger."""
    root = _get_root_logger()
    level_value = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level_value)
    for h in root.handlers:
        if not isinstance(h, logging.NullHandler):
            h.setLevel(level_value)


def enable_debug() -> None:
    """Convenience helper to set DEBUG level."""
    set_level("DEBUG")


