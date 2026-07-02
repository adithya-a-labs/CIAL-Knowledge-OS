"""Configurable structured logging for pipeline and run diagnostics."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__)


class StructuredJsonFormatter(logging.Formatter):
    """Render one machine-readable JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_RECORD_KEYS
                and key not in {"message", "asctime"}
                and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


class _PerRecordFileHandler(logging.Handler):
    """Append one record per open so interrupted Windows runs release the file."""

    def __init__(self, path: Path, *, encoding: str = "utf-8") -> None:
        super().__init__()
        self.path = path
        self.encoding = encoding

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self.path.parent.is_dir():
                return
            message = self.format(record)
            with self.path.open("a", encoding=self.encoding) as handle:
                handle.write(message + "\n")
        except Exception:
            self.handleError(record)


def configure_logging(
    *,
    level: str = "INFO",
    structured: bool = True,
    log_path: str | Path | None = None,
) -> tuple[logging.Handler, ...]:
    """Configure package logging and return the handlers created.

    Existing application/root handlers are preserved. Prior handlers created by
    this function are replaced to prevent duplicate lines across notebook runs.
    """

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level!r}.")
    package_logger = logging.getLogger("cial_knowledge_os")
    package_logger.setLevel(numeric_level)
    package_logger.propagate = False
    for handler in list(package_logger.handlers):
        if getattr(handler, "_cial_managed", False):
            package_logger.removeHandler(handler)
            handler.close()

    formatter: logging.Formatter = (
        StructuredJsonFormatter()
        if structured
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    )
    handlers: list[logging.Handler] = []
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    stream_handler._cial_managed = True  # type: ignore[attr-defined]
    package_logger.addHandler(stream_handler)
    handlers.append(stream_handler)

    if log_path is not None:
        resolved = Path(log_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _PerRecordFileHandler(resolved)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler._cial_managed = True  # type: ignore[attr-defined]
        package_logger.addHandler(file_handler)
        handlers.append(file_handler)
    return tuple(handlers)


def close_logging(handlers: tuple[logging.Handler, ...]) -> None:
    """Flush and release handlers created for a completed run."""

    package_logger = logging.getLogger("cial_knowledge_os")
    for handler in handlers:
        handler.flush()
        package_logger.removeHandler(handler)
        handler.close()
