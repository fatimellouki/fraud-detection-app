"""Timing utilities for measuring model performance."""

import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            logger.info(f"{self.name}: {self.elapsed:.4f}s")

    @property
    def elapsed_ms(self):
        return self.elapsed * 1000


def timed(func):
    """Decorator that logs execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper
