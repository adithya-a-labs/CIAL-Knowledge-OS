"""Abstract OCR contract used by ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class OCRExtractionResult:
    text: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseOCREngine(Protocol):
    name: str

    def extract(self, image_path: str | Path) -> OCRExtractionResult:
        """Extract text and OCR metadata from one image path."""
