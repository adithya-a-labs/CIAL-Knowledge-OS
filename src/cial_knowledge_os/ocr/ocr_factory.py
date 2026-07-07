"""OCR engine factory."""

from __future__ import annotations

from typing import Any

from .base_ocr import BaseOCREngine
from .tesseract_ocr import TesseractOCREngine


def create_ocr_engine(config: Any) -> BaseOCREngine:
    engine = str(getattr(config, "ocr_engine", "tesseract") or "tesseract").casefold()
    if engine != "tesseract":
        raise ValueError(f"Unsupported OCR engine: {engine}")
    return TesseractOCREngine(
        language=str(getattr(config, "ocr_language", "eng") or "eng"),
        preprocessing_enabled=bool(getattr(config, "ocr_preprocessing", True)),
    )
