from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from cial_knowledge_os.chunking import chunk_documents
from cial_knowledge_os.config import KnowledgeOSConfig, Phase4Config
from cial_knowledge_os.loaders import (
    _base_metadata,
    discover_knowledge_documents,
    load_pdf_documents,
)
from cial_knowledge_os.phase4_runner import Phase4Runner


class CorpusArchitectureTests(unittest.TestCase):
    @staticmethod
    def _fake_pdf_page(path: Path, corpus_root: Path) -> list[Document]:
        return [
            Document(
                page_content=f"Content from {path.name}",
                metadata=_base_metadata(
                    path,
                    "pymupdf",
                    1,
                    corpus_root=corpus_root,
                ),
            )
        ]

    def test_config_standardizes_canonical_and_legacy_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))

            self.assertEqual(
                config.knowledge_root,
                (Path(directory) / "data" / "files").resolve(),
            )
            self.assertEqual(
                config.legacy_pdf_root,
                (Path(directory) / "data" / "pdf").resolve(),
            )
            self.assertEqual(config.pdf_data_dir, config.legacy_pdf_root)

    def test_recursive_discovery_and_taxonomy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            source = (
                config.knowledge_root
                / "cybersecurity"
                / "nist"
                / "NIST_AI_RMF.PDF"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-test")

            corpus_root, paths = discover_knowledge_documents(config)
            with patch.dict(sys.modules, {"docling": None}), patch(
                "cial_knowledge_os.loaders._load_pdf_with_pymupdf",
                side_effect=self._fake_pdf_page,
            ):
                documents = load_pdf_documents(config)
            chunks = chunk_documents(documents, config)

            self.assertEqual(corpus_root, config.knowledge_root)
            self.assertEqual(paths, [source])
            metadata = documents[0].metadata
            self.assertEqual(metadata["file_name"], "NIST_AI_RMF.PDF")
            self.assertEqual(metadata["source_filename"], "NIST_AI_RMF.PDF")
            self.assertEqual(metadata["absolute_path"], str(source.resolve()))
            self.assertEqual(
                metadata["relative_path"],
                "cybersecurity/nist/NIST_AI_RMF.PDF",
            )
            self.assertEqual(metadata["category"], "cybersecurity")
            self.assertEqual(metadata["collection"], "nist")
            self.assertEqual(metadata["document_type"], "pdf")
            self.assertEqual(metadata["page_number"], 1)
            self.assertEqual(metadata["source"], str(source.resolve()))
            self.assertTrue(chunks)
            self.assertEqual(chunks[0].metadata["category"], "cybersecurity")
            self.assertEqual(chunks[0].metadata["collection"], "nist")
            self.assertEqual(
                chunks[0].metadata["relative_path"],
                "cybersecurity/nist/NIST_AI_RMF.PDF",
            )

    def test_canonical_documents_take_precedence_over_legacy_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            canonical = config.knowledge_root / "legal" / "policy.pdf"
            legacy = config.legacy_pdf_root / "legacy.pdf"
            canonical.parent.mkdir(parents=True)
            legacy.parent.mkdir(parents=True)
            canonical.write_bytes(b"%PDF-canonical")
            legacy.write_bytes(b"%PDF-legacy")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                corpus_root, paths = discover_knowledge_documents(config)

            self.assertEqual(corpus_root, config.knowledge_root)
            self.assertEqual(paths, [canonical])
            self.assertEqual(caught, [])

    def test_missing_or_empty_canonical_root_does_not_use_legacy_pdfs(self) -> None:
        for create_empty_root in (False, True):
            with self.subTest(create_empty_root=create_empty_root):
                with tempfile.TemporaryDirectory() as directory:
                    config = KnowledgeOSConfig(project_root=Path(directory))
                    if create_empty_root:
                        config.knowledge_root.mkdir(parents=True)
                    legacy = config.legacy_pdf_root / "legacy.pdf"
                    legacy.parent.mkdir(parents=True)
                    legacy.write_bytes(b"%PDF-test")

                    corpus_root, paths = discover_knowledge_documents(config)

                    self.assertEqual(corpus_root, config.knowledge_root)
                    self.assertEqual(paths, [])

    def test_recognized_future_types_are_skipped_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            future_document = config.knowledge_root / "legal" / "policy.msg"
            future_document.parent.mkdir(parents=True)
            future_document.write_bytes(b"not loaded yet")

            with self.assertLogs(
                "cial_knowledge_os.loaders",
                level="WARNING",
            ) as captured:
                corpus_root, paths = discover_knowledge_documents(config)

            self.assertEqual(corpus_root, config.knowledge_root)
            self.assertEqual(paths, [])
            self.assertTrue(
                any(
                    "document_type_not_ingested" in message
                    for message in captured.output
                )
            )


class _CorpusAwarePhase4Pipeline:
    def __init__(self, config: Phase4Config) -> None:
        self.config = config
        self.metrics: dict[str, float] = {}
        self.is_ready_for_answering = True

    def answer(self, question: str) -> dict[str, object]:
        _, paths = discover_knowledge_documents(self.config)
        source = paths[0]
        metadata = _base_metadata(
            source,
            "test",
            1,
            corpus_root=self.config.knowledge_root,
        )
        retrieved = {
            "text": "Approved local control.",
            "source": source.name,
            "page_number": 1,
            "chunk_id": f"{source.name}:p1:c0",
            "score": 1.0,
            "metadata": metadata,
        }
        return {
            "question": question,
            "answer": "Approved local control. [1]",
            "raw_answer": "Approved local control. [1]",
            "answer_status": "answered",
            "retrieved": [retrieved],
            "selected_evidence": [retrieved],
            "context": "Approved local control.",
            "context_stages": {"retrieved": [retrieved], "final": [retrieved]},
            "citations": [
                {
                    "source": source.name,
                    "source_path": str(source.resolve()),
                    "page_number": 1,
                    "chunk_id": f"{source.name}:p1:c0",
                    "pdf_link": source.resolve().as_uri() + "#page=1",
                }
            ],
            "retrieval_trace": {},
            "phase4_trace": {},
            "evidence_quality": {"summary": {}},
        }


class Phase4KnowledgeRootTests(unittest.TestCase):
    def test_phase4_runner_exports_from_canonical_knowledge_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase4Config(
                project_root=Path(directory),
                phase4_trace_mode="compact",
            )
            source = config.knowledge_root / "aviation" / "icao" / "annex.pdf"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-test")

            result = Phase4Runner(
                pipeline=_CorpusAwarePhase4Pipeline(config),
                config=config,
            ).run(
                questions=["What control applies?"],
                run_mode="smoke",
            )

            self.assertTrue(result.paths.results_csv.is_file())
            self.assertTrue(result.paths.results_xlsx.is_file())
            self.assertTrue(result.paths.report_html.is_file())
            self.assertTrue(result.paths.retrieval_json.is_file())
            retrieval = json.loads(
                result.paths.retrieval_json.read_text(encoding="utf-8")
            )
            self.assertIn("aviation/icao/annex.pdf", json.dumps(retrieval))


class MigrationScriptTests(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_pdf_to_files.py"

    def _run(self, root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.SCRIPT),
                "--project-root",
                str(root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_dry_run_does_not_copy_and_default_run_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "pdf" / "manual.pdf"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-test")
            destination = root / "data" / "files" / "legacy_pdf" / source.name

            dry_run = self._run(root, "--dry-run")
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertIn("Source count: 1", dry_run.stdout)
            self.assertIn("Copied count: 1", dry_run.stdout)
            self.assertFalse(destination.exists())

            copied = self._run(root)
            self.assertEqual(copied.returncode, 0, copied.stderr)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertTrue(source.exists())

            repeated = self._run(root)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Copied count: 0", repeated.stdout)
            self.assertIn("Skipped existing count: 1", repeated.stdout)

    def test_move_removes_source_only_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "pdf" / "manual.pdf"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-test")
            destination = root / "data" / "files" / "legacy_pdf" / source.name

            moved = self._run(root, "--move")

            self.assertEqual(moved.returncode, 0, moved.stderr)
            self.assertFalse(source.exists())
            self.assertEqual(destination.read_bytes(), b"%PDF-test")


if __name__ == "__main__":
    unittest.main()
