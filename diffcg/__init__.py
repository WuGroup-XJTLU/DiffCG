import os
from logging import NullHandler, getLogger

# Attach a NullHandler by default so importing users don't get noisy logs unless
# they configure logging explicitly.
_logger = getLogger("diffcg")
if not _logger.handlers:
    _logger.addHandler(NullHandler())

# Re-export logging configuration helpers for convenience and allow optional
# auto-configuration for terminal output via env vars.
try:
    from diffcg.util.logger import configure as configure_logging, get_logger

    # If DIFFCG_LOG_AUTO is set truthy, or DIFFCG_LOG_LEVEL is provided,
    # auto-configure a StreamHandler that prints to the terminal.
    _auto = os.getenv("DIFFCG_LOG_AUTO")
    _level_present = os.getenv("DIFFCG_LOG_LEVEL") is not None
    if (_auto and _auto.lower() not in ("0", "false", "no")) or _level_present:
        configure_logging()
except Exception:
    # Soft-fail if optional deps are missing during partial installs
    pass

#from diffcg import configure_logging

configure_logging(level="DEBUG")