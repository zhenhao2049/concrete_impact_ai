"""Logging helpers for command-line tools.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create a standard project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

