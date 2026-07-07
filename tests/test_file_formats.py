from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cial_knowledge_os.file_formats import (
    SupportStatus,
    get_file_extension,
    get_file_format_info,
    is_recognized_file,
    is_supported_file,
    scan_file_format_readiness,
    validate_ingestion_file,
)


class FileFormatRegistryTests(unittest.TestCase):
    def assert_status(self, filename: str, status: SupportStatus) -> None:
        info = get_file_format_info(filename)
        self.assertEqual(info["support_status"], status.value)

    def test_currently_processed_formats_are_supported(self) -> None:
        for filename in (
            "manual.PDF",
            "procedure.docx",
            "legacy.doc",
            "table.xlsx",
            "legacy.xls",
            "records.csv",
            "deck.pptx",
            "legacy.ppt",
            "notes.txt",
            "readme.md",
            "guide.markdown",
            "page.html",
            "page.htm",
            "data.json",
            "data.xml",
            "config.yaml",
            "config.yml",
        ):
            with self.subTest(filename=filename):
                self.assert_status(filename, SupportStatus.SUPPORTED_NOW)
                self.assertTrue(is_supported_file(filename))
                self.assertTrue(is_recognized_file(filename))

    def test_ocr_formats_require_ocr(self) -> None:
        for filename in ("scan.png", "photo.jpg", "photo.jpeg", "scan.tiff", "scan.tif"):
            with self.subTest(filename=filename):
                info = get_file_format_info(filename)
                self.assertEqual(info["support_status"], SupportStatus.OCR_SUPPORTED.value)
                self.assertTrue(info["requires_ocr"])
                validation = validate_ingestion_file(filename)
                self.assertTrue(validation["valid_for_ingestion"])
                self.assertEqual(validation["action"], "ocr_then_process")

    def test_recognized_future_formats_are_not_ingested(self) -> None:
        for filename in (
            "message.msg",
            "archive.zip",
            "script.py",
            "drawing.dwg",
            "Dockerfile",
            "package.json",
            "requirements.txt",
            "docker-compose.yml",
            ".env",
        ):
            with self.subTest(filename=filename):
                info = get_file_format_info(filename)
                self.assertEqual(
                    info["support_status"],
                    SupportStatus.RECOGNIZED_FUTURE_SUPPORT.value,
                )
                self.assertFalse(info["ingestion_enabled"])
                self.assertFalse(is_supported_file(filename))
                self.assertTrue(is_recognized_file(filename))
                self.assertEqual(
                    validate_ingestion_file(filename)["action"],
                    "skip_with_warning",
                )

    def test_unknown_and_extension_normalization(self) -> None:
        self.assertEqual(get_file_extension("archive.tar.gz"), "gz")
        self.assertEqual(get_file_extension("Dockerfile"), "dockerfile")
        self.assert_status("MANUAL.PDF", SupportStatus.SUPPORTED_NOW)
        self.assert_status("unknown.weird", SupportStatus.UNSUPPORTED)
        self.assertFalse(is_recognized_file("unknown.weird"))
        self.assertEqual(validate_ingestion_file("unknown.weird")["action"], "reject")

    def test_scan_file_format_readiness_summarizes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "ops/manual.pdf",
                "ops/scan.png",
                "future/mail.msg",
                "future/code.py",
                "unknown/blob.zzz",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"content")

            readiness = scan_file_format_readiness(root)

        self.assertEqual(readiness["total_files"], 5)
        self.assertEqual(readiness["processable_files"], 2)
        self.assertEqual(readiness["ocr_files"], 1)
        self.assertEqual(readiness["recognized_future_files"], 2)
        self.assertEqual(readiness["unsupported_files"], 1)
        self.assertEqual(readiness["extension_distribution"]["pdf"], 1)
        self.assertEqual(readiness["support_status_distribution"]["SUPPORTED_NOW"], 1)
        self.assertEqual(readiness["support_status_distribution"]["OCR_SUPPORTED"], 1)
        self.assertEqual(len(readiness["skipped_files"]), 3)


if __name__ == "__main__":
    unittest.main()
