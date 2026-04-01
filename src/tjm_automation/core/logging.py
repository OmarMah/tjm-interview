from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=_coerce_level(level),
        format=LOG_FORMAT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _coerce_level(level: str) -> int:
    normalized = level.upper()
    resolved = getattr(logging, normalized, None)
    if not isinstance(resolved, int):
        return logging.INFO
    return resolved

