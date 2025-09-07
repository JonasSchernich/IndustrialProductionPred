# src/utils/logging.py
from __future__ import annotations
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Setzt ein schlankes Logging-Format auf Root-Logger.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
