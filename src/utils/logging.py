import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s", DEFAULT_FORMAT
)


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Create and configure a logger.

    If a logger with handlers already exists it will not add duplicate handlers
    (useful when this is called multiple times). Returns the configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid adding handlers multiple times in environments that import repeatedly
    if logger.handlers:
        return logger

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(sh)

    # Optional rotating file handler
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.setFormatter(DEFAULT_FORMATTER)
        logger.addHandler(fh)

    return logger


__all__ = ["setup_logger"]
