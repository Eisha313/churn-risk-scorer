"""Logging configuration for the churn risk scorer."""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import settings, OUTPUT_DIR


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure and return the application logger.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to settings.log_level.
        log_file: Optional path to write logs to file.
        format_string: Custom format string for log messages.
    
    Returns:
        Configured logger instance.
    """
    level = level or settings.log_level
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("churn_risk_scorer")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional name suffix for the logger.
    
    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"churn_risk_scorer.{name}")
    return logging.getLogger("churn_risk_scorer")


# Initialize default logger
logger = setup_logging()
