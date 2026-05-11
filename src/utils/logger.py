"""Logging configuration for the fraud detection project."""

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure project-wide logging.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Root logger.
    """
    log_format = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Reduce noise from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)

    return logging.getLogger()
