from __future__ import annotations

import copy
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from cial_knowledge_os.phase4_reporting import write_phase4_html


class _StandaloneHTMLParser(HTMLParser):
    """Exercise the standard-library HTML parser over a generated report."""


class Phase4CitationReportingTests(unittest.TestCase):
    def _citations(self, root: Path) -> list[dict[str, object]]:
        first = root / "CISG-2026-01.pdf"
        second = root / "airport-controls.pdf"
        first.write_bytes(b"%PDF-test")
        second.write_bytes(b"%PDF-test")
        return [
            {
                "reference_id": 1,
                "source": first.name,
                "source_file": first.name,
                "page_number": 47,
                "chunk_id": "151",
                "score": 0.81234,
                "pdf_link": first.as_uri() + "#page=47",
            },
            {
                "reference_id": 2,
                "source": second.name,
                "source_file": second.name,
                "page_number": 8,
                "chunk_id": "ops:8:2",
                "score": 0.64,
                "pdf_link": second.as_uri() + "#page=8",
            },
        ]

    def _report(
        self,
        root: Path,
        answer: str,
        citations: list[dict[str, object]],
    ) -> str:
        path = root / "report.html"
        rows = [{"question": "What controls apply?", "answer": answer}]
        traces = [
            {
                "question": "What controls apply?",
                "answer": answer,
                "citations": citations,
            }
        ]
        write_phase4_html(
            path,
            rows=rows,
            traces=traces,
            summary={"question_count": 1},
            metrics={},
        )
        return path.read_text(encoding="utf-8")

    def test_numeric_markers_render_as_inline_pdf_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            report = self._report(
                root,
                "Apply the primary control [1] and verify operations [2].",
                citations,
            )

            self.assertEqual(report.count('class="inline-citation"'), 2)
            self.assertIn(">[" + "1]</a>", report)
            self.assertIn("#page=47", report)
            self.assertIn(
                'title="CISG-2026-01.pdf | Page 47 | Chunk 151 | '
                'Score 0.8123"',
                report,
            )
            self.assertNotIn('class="citation-chips"', report)
            self.assertIn(
                '<details class="citation-details">',
                report,
            )

    def test_source_page_chunk_marker_resolves_inline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            marker = "[CISG-2026-01 | Page 47 | Chunk 151]"
            report = self._report(
                root,
                f"The requirement is documented in {marker}.",
                citations,
            )

            self.assertIn(
                f'class="inline-citation" href="{citations[0]["pdf_link"]}"',
                report,
            )
            self.assertIn(
                "[CISG-2026-01 | Page 47 | Chunk 151]</a>",
                report,
            )

    def test_missing_inline_markers_get_compact_citation_chips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            report = self._report(
                root,
                "The evidence supports a phased control rollout.",
                citations,
            )

            self.assertIn('class="citation-chips"', report)
            self.assertEqual(report.count('class="citation-chip"'), 2)
            self.assertIn("Sources:</span>", report)

    def test_report_is_standalone_safe_and_does_not_mutate_export_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            citations[0]["pdf_link"] = "javascript:alert(1)"
            original = copy.deepcopy(citations)
            report = self._report(
                root,
                "**Control** [1]\n\n<script>alert('unsafe')</script>\n\n"
                "References:\n[1] CISG-2026-01.pdf | page 47 | chunk 151",
                citations,
            )

            parser = _StandaloneHTMLParser()
            parser.feed(report)
            parser.close()
            self.assertTrue(report.casefold().startswith("<!doctype html>"))
            self.assertIn("<strong>Control</strong>", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            self.assertNotIn("javascript:alert", report)
            self.assertNotIn("References:", report)
            self.assertNotIn("https://cdn", report)
            self.assertEqual(citations, original)


if __name__ == "__main__":
    unittest.main()
