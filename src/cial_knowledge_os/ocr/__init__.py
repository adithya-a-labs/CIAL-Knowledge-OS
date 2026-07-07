"""OCR engines for image-backed ingestion."""

from .base_ocr import BaseOCREngine, OCRExtractionResult
from .ocr_factory import create_ocr_engine
from .tesseract_ocr import TesseractOCREngine

__all__ = [
    "BaseOCREngine",
    "OCRExtractionResult",
    "TesseractOCREngine",
    "create_ocr_engine",
]
