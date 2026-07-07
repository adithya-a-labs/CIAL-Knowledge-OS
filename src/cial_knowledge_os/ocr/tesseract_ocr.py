"""Tesseract-backed OCR implementation with local preprocessing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base_ocr import OCRExtractionResult


class TesseractOCREngine:
    name = "tesseract"

    def __init__(
        self,
        *,
        language: str = "eng",
        preprocessing_enabled: bool = True,
    ) -> None:
        self.language = language
        self.preprocessing_enabled = preprocessing_enabled

    def _preprocess(self, image: Any) -> tuple[Any, list[str]]:
        from PIL import ImageOps, ImageEnhance, ImageFilter

        steps: list[str] = []
        image = ImageOps.exif_transpose(image)
        steps.append("exif_orientation")
        image = image.convert("L")
        steps.append("grayscale")
        image = ImageEnhance.Contrast(image).enhance(1.5)
        steps.append("contrast_enhancement")
        if min(image.size) < 900:
            scale = max(1, int(900 / max(1, min(image.size))))
            if scale > 1:
                image = image.resize(
                    (image.width * scale, image.height * scale)
                )
                steps.append("small_image_resize")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        steps.append("median_denoise")
        return image, steps

    def extract(self, image_path: str | Path) -> OCRExtractionResult:
        path = Path(image_path).expanduser().resolve()
        started = time.perf_counter()
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            return OCRExtractionResult(
                text="",
                status="OCR_FAILED",
                metadata={
                    "ocr_engine": self.name,
                    "ocr_language": self.language,
                    "source_format": path.suffix.lstrip(".").lower(),
                },
                error=(
                    "OCR dependencies are unavailable. Install pytesseract and "
                    "Pillow from requirements.txt."
                ),
            )

        try:
            with Image.open(path) as image:
                original_width, original_height = image.size
                preprocessing: list[str] = []
                working = image
                if self.preprocessing_enabled:
                    working, preprocessing = self._preprocess(image)
                text = pytesseract.image_to_string(
                    working,
                    lang=self.language,
                ).strip()
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                version = str(pytesseract.get_tesseract_version())
                words = text.split()
                return OCRExtractionResult(
                    text=text,
                    status="OCR_SUCCESS" if text else "OCR_SKIPPED",
                    metadata={
                        "ocr_engine": self.name,
                        "ocr_engine_version": version,
                        "ocr_language": self.language,
                        "preprocessing_applied": preprocessing,
                        "extraction_time_ms": elapsed_ms,
                        "extracted_character_count": len(text),
                        "extracted_word_count": len(words),
                        "image_width": original_width,
                        "image_height": original_height,
                        "source_format": path.suffix.lstrip(".").lower(),
                        "ocr_status": "OCR_SUCCESS" if text else "OCR_SKIPPED",
                    },
                )
        except Exception as exc:
            return OCRExtractionResult(
                text="",
                status="OCR_FAILED",
                metadata={
                    "ocr_engine": self.name,
                    "ocr_language": self.language,
                    "preprocessing_applied": [],
                    "extraction_time_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "extracted_character_count": 0,
                    "extracted_word_count": 0,
                    "source_format": path.suffix.lstrip(".").lower(),
                    "ocr_status": "OCR_FAILED",
                },
                error=str(exc),
            )
