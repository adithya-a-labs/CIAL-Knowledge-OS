"""Durable JSONL traces and atomic progress snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from .events import ExecutionEvent


def _json_default(value: Any) -> str:
    return str(value)


class JSONTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def __call__(self, event: ExecutionEvent) -> None:
        line = json.dumps(
            event.to_dict(), ensure_ascii=False, default=_json_default
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class ProgressSnapshotWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                snapshot, ensure_ascii=False, indent=2, default=_json_default
            ),
            encoding="utf-8",
        )
        try:
            os.replace(temporary, self.path)
        except PermissionError:
            # Some Windows scanners briefly lock the destination. A direct
            # rewrite preserves visibility while observer failures remain
            # isolated from the pipeline.
            self.path.write_text(
                temporary.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            temporary.unlink(missing_ok=True)
