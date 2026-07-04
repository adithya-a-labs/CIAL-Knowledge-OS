from __future__ import annotations

import copy
import json
import re
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
        *,
        trace_updates: dict[str, object] | None = None,
    ) -> str:
        path = root / "report.html"
        rows = [{"question": "What controls apply?", "answer": answer}]
        selected = []
        quality = []
        bm25_results = []
        for index, citation in enumerate(citations, start=1):
            text = (
                f"Original selected evidence {index}: software vendor controls "
                "must be independently verified before approval. "
                + ("Additional exact evidence text. " * 12)
            )
            selected.append(
                {
                    "source": citation["source"],
                    "page_number": citation["page_number"],
                    "chunk_id": citation["chunk_id"],
                    "reranker_score": 0.91 - index / 10,
                    "rrf_score": 0.02,
                    "retrieval_sources": (
                        ["dense", "bm25"] if index == 1 else ["bm25"]
                    ),
                    "matched_terms": ["software", "vendor", "approval"],
                    "text": text,
                }
            )
            quality.append(
                {
                    "source": citation["source"],
                    "page_number": citation["page_number"],
                    "chunk_id": citation["chunk_id"],
                    "reranker_score": 0.91 - index / 10,
                    "retrieval_source": (
                        "both" if index == 1 else "bm25"
                    ),
                    "evidence_strength": (
                        "strong" if index == 1 else "medium"
                    ),
                }
            )
            bm25_results.append(
                {
                    "source": citation["source"],
                    "page_number": citation["page_number"],
                    "chunk_id": citation["chunk_id"],
                    "matched_terms": ["software", "vendor", "approval"],
                }
            )
        trace = {
            "question": "What controls apply?",
            "answer": answer,
            "citations": citations,
            "selected_chunks": selected,
            "final_context_chunks": selected,
            "bm25_results": bm25_results,
            "evidence_quality": {"chunks": quality},
        }
        trace.update(trace_updates or {})
        traces = [
            trace
        ]
        write_phase4_html(
            path,
            rows=rows,
            traces=traces,
            summary={"question_count": 1},
            metrics={},
        )
        return path.read_text(encoding="utf-8")

    def _preview_data(self, report: str) -> dict[str, dict[str, object]]:
        match = re.search(
            r'<script type="application/json" '
            r'id="citation-preview-data">(.*?)</script>',
            report,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

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
                'data-citation-title="CISG-2026-01.pdf | Page 47 | Chunk 151 | '
                'Score 0.8123"',
                report,
            )
            preview = self._preview_data(report)["q1-c1"]
            self.assertEqual(preview["source"], "CISG-2026-01.pdf")
            self.assertEqual(preview["page"], "47")
            self.assertEqual(preview["chunk"], "151")
            self.assertEqual(preview["reranker_score"], "0.8100")
            self.assertEqual(preview["evidence_strength"], "Strong")
            self.assertEqual(preview["retrieval_source"], "Dense + BM25 + RRF")
            self.assertLessEqual(len(preview["snippet"]), 260)
            self.assertTrue(
                preview["snippet"].startswith("Original selected evidence 1:")
            )
            self.assertEqual(report.count(preview["snippet"]), 1)
            self.assertEqual(
                preview["matched_terms"],
                ["software", "vendor", "approval"],
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
                f'href="{citations[0]["pdf_link"]}"',
                report,
            )
            self.assertIn(
                "[CISG-2026-01 | Page 47 | Chunk 151]</a>",
                report,
            )

    def test_adaptive_headings_render_fully_with_inline_citations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._report(
                root,
                "## Top Risks\n\nThe primary risk is documented [1].\n\n"
                "## Next Actions\n\nValidate the control owner [2].",
                self._citations(root),
            )

            self.assertIn("<h2>Top Risks</h2>", report)
            self.assertIn("<h2>Next Actions</h2>", report)
            self.assertIn("Validate the control owner", report)
            self.assertEqual(report.count('class="inline-citation"'), 2)

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

    def test_missing_pdf_and_missing_snippet_have_popover_fallbacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            citations[0]["pdf_link"] = None
            report = self._report(
                root,
                "Apply the unavailable source [1].",
                citations,
            )
            preview = self._preview_data(report)["q1-c1"]

            self.assertFalse(preview["pdf_available"])
            self.assertTrue(
                preview["snippet"].startswith("Original selected evidence 1:")
            )
            self.assertIn(
                "Source available but PDF link unavailable.",
                report,
            )
            self.assertIn(
                'class="inline-citation no-citation-link"',
                report,
            )

            missing_snippet_report = self._report(
                root,
                "Apply the unavailable source [1].",
                citations,
                trace_updates={
                    "final_context_chunks": [],
                    "selected_chunks": [],
                },
            )
            missing_preview = self._preview_data(
                missing_snippet_report
            )["q1-c1"]

            self.assertEqual(missing_preview["snippet"], "")
            self.assertIn(
                "Evidence preview unavailable.",
                missing_snippet_report,
            )

    def test_popover_is_standalone_viewport_aware_and_theme_ready(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._report(
                root,
                "Use both controls [1] [2].",
                self._citations(root),
            )

            self.assertIn('id="citation-popover"', report)
            self.assertIn('role="tooltip"', report)
            self.assertIn("pointerenter", report)
            self.assertIn("pointerleave", report)
            self.assertIn("window.innerWidth", report)
            self.assertIn("window.innerHeight", report)
            self.assertIn("--popover-background:", report)
            self.assertIn("background:var(--popover-background)", report)
            self.assertIn('document.createElement("mark")', report)
            self.assertEqual(
                len(self._preview_data(report)),
                2,
            )
            self.assertNotIn("https://cdn", report)
            self.assertNotIn("<link rel=", report)

    def test_theme_controls_support_light_dark_system_and_persistence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._report(
                root,
                "Use both controls [1] [2].",
                self._citations(root),
            )

            required_variables = (
                "--background:",
                "--card-background:",
                "--text:",
                "--muted-text:",
                "--border:",
                "--accent:",
                "--success:",
                "--warning:",
                "--error:",
                "--code-block:",
                "--citation-badge:",
                "--table-header:",
                "--chart-container:",
            )
            for variable in required_variables:
                self.assertIn(variable, report)
            self.assertIn('[data-theme="dark"]', report)
            self.assertIn('fill="var(--chart-bar)"', report)
            self.assertIn("fill:var(--text)", report)
            for theme in ("light", "dark", "system"):
                self.assertIn(
                    f'data-theme-choice="{theme}"',
                    report,
                )
                self.assertIn(
                    f'aria-label="Use {theme} theme"',
                    report,
                )
            self.assertIn('role="group"', report)
            self.assertIn('aria-label="Report color theme"', report)
            self.assertIn('window.matchMedia("(prefers-color-scheme: dark)")', report)
            self.assertIn(
                'window.localStorage.getItem(storageKey)',
                report,
            )
            self.assertIn(
                'window.localStorage.setItem(storageKey, preference)',
                report,
            )
            self.assertIn(
                '"cial-phase4-report-theme"',
                report,
            )
            self.assertIn(
                "document.documentElement.dataset.theme",
                report,
            )
            self.assertIn(
                'systemTheme.addEventListener("change", followSystemTheme)',
                report,
            )
            self.assertNotIn("https://cdn", report)
            self.assertNotIn("<link rel=", report)

    def test_snippet_html_is_json_escaped_and_rendered_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citations = self._citations(root)
            exact_text = (
                "Evidence <img src=x onerror=alert(1)> requires vendor review."
            )
            evidence = {
                "source": citations[0]["source"],
                "page_number": citations[0]["page_number"],
                "chunk_id": citations[0]["chunk_id"],
                "reranker_score": 0.8,
                "retrieval_sources": ["dense"],
                "text": exact_text,
            }
            report = self._report(
                root,
                "Apply the control [1].",
                citations,
                trace_updates={
                    "selected_chunks": [evidence],
                    "final_context_chunks": [evidence],
                },
            )

            self.assertEqual(
                self._preview_data(report)["q1-c1"]["snippet"],
                exact_text,
            )
            self.assertNotIn("<img src=x", report)
            self.assertIn("\\u003cimg src=x", report)

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
            self.assertNotIn("<script>alert", report)
            self.assertNotIn("javascript:alert", report)
            self.assertNotIn("References:", report)
            self.assertNotIn("https://cdn", report)
            self.assertEqual(citations, original)


if __name__ == "__main__":
    unittest.main()
