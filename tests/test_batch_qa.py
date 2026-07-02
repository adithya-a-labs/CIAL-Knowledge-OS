from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from cial_knowledge_os.batch_qa import (
    CSV_COLUMNS,
    PHASE2_CSV_COLUMNS,
    export_batch_answers,
)
from cial_knowledge_os.rag_pipeline import BasicRAGPipeline
from cial_knowledge_os.token_budget import create_token_manager


class _ReadyPipeline:
    def __init__(self, project_root: Path) -> None:
        self.config = SimpleNamespace(
            project_root=project_root,
            top_k=3,
            ollama_model_name="local-test-model",
            embedding_model_name="local-test-embeddings",
        )
        self.metrics: dict[str, float] = {}
        self.answer_calls = 0

    @property
    def is_ready_for_answering(self) -> bool:
        return True

    def answer(self, question: str) -> dict[str, object]:
        self.answer_calls += 1
        return {"answer": f"Answer: {question}", "retrieved": []}


class _ReadyPhase2Pipeline:
    def __init__(self, project_root: Path) -> None:
        self.config = SimpleNamespace(
            project_root=project_root,
            top_k=3,
            retrieval_top_k=10,
            ollama_model_name="local-test-model",
            embedding_model_name="local-test-embeddings",
        )
        self.metrics = {
            "generation_latency": 0.2,
            "retrieval_latency": 0.1,
        }
        self.retrieval_depths: list[int] = []

    @property
    def is_ready_for_answering(self) -> bool:
        return True

    def answer(self, question: str) -> dict[str, object]:
        self.retrieval_depths.append(self.config.retrieval_top_k)
        metadata = {
            "source": "C:/corpus/manual.pdf",
            "file_name": "manual.pdf",
            "page_number": 7,
            "chunk_id": "manual:p7:c1",
            "chunk_index": 1,
        }
        evidence = {
            "text": "Grounded evidence.",
            "score": 0.75,
            "source": "manual.pdf",
            "page_number": 7,
            "chunk_id": "manual:p7:c1",
            "metadata": metadata,
        }
        insufficient = "unsupported" in question.casefold()
        answer = (
            "The retrieved documents do not contain sufficient evidence to "
            "answer this question. Based only on the indexed corpus, no reliable "
            "answer could be generated."
            if insufficient
            else "Grounded answer [1]."
        )
        return {
            "answer": answer,
            "raw_answer": answer,
            "answer_status": (
                "insufficient_evidence" if insufficient else "answered"
            ),
            "retrieved": [evidence, evidence],
            "query_variants": [
                {"technique": "original", "query": question},
                {"technique": "rewritten", "query": f"Rewritten {question}"},
                {
                    "technique": "keyword_expanded",
                    "query": f"Expanded {question}",
                },
                {
                    "technique": "domain_reformulation",
                    "query": f"CIAL {question}",
                },
            ],
            "context_stages": {
                "retrieved": [evidence] * 34,
                "deduplicated": [evidence] * 19,
                "expanded": [evidence] * 28,
                "merged": [evidence] * 8,
                "compressed": [evidence] * 7,
            },
            "stage_counts": {
                "retrieved": 34,
                "deduplicated": 19,
                "expanded": 28,
                "merged": 8,
                "compressed": 7,
            },
            "context": "Final grounded context.",
        }


class ExportBatchAnswersTests(unittest.TestCase):
    def test_uninitialized_pipeline_fails_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            config = SimpleNamespace(project_root=project_root, top_k=3)
            pipeline = BasicRAGPipeline(config=config)
            pipeline.answer = Mock(
                side_effect=AssertionError(
                    "answer() must not run before readiness validation"
                )
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"Call pipeline\.load\(\), pipeline\.chunk\(\), "
                r"pipeline\.embed\(\), and pipeline\.index\(\)",
            ):
                export_batch_answers(pipeline=pipeline, questions=["Question?"])

            pipeline.answer.assert_not_called()
            self.assertFalse((project_root / "outputs").exists())

    def test_ready_pipeline_exports_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = _ReadyPipeline(Path(temporary_directory))

            output_path = export_batch_answers(
                pipeline=pipeline,
                questions=["Question?"],
                run_name="readiness-test",
            )

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(pipeline.answer_calls, 1)
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["answer"], "Answer: Question?")
            self.assertEqual(list(rows[0]), CSV_COLUMNS)

    def test_phase2_export_appends_trace_columns_and_uses_phase2_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = _ReadyPhase2Pipeline(Path(temporary_directory))

            output_path = export_batch_answers(
                pipeline=pipeline,
                questions=["Operational question?", "Unsupported question?"],
                run_name="phase-2-test",
                top_k=12,
            )

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = reader.fieldnames

            self.assertEqual(columns, [*CSV_COLUMNS, *PHASE2_CSV_COLUMNS])
            required_phase2_columns = {
                "query_variants",
                "chunks_before_deduplication",
                "chunks_after_deduplication",
                "chunks_after_neighbor_expansion",
                "merged_context_sections",
                "final_context_sections",
                "final_context_characters",
                "final_context_tokens_estimate",
                "answer_status",
                "retrieval_trace",
            }
            self.assertTrue(required_phase2_columns.issubset(columns or []))
            self.assertEqual(pipeline.retrieval_depths, [12, 12])
            self.assertEqual(pipeline.config.retrieval_top_k, 10)
            self.assertEqual(rows[0]["top_k"], "12")
            self.assertEqual(rows[0]["chunks_before_deduplication"], "34")
            self.assertEqual(rows[0]["chunks_after_deduplication"], "19")
            self.assertEqual(rows[0]["chunks_after_neighbor_expansion"], "28")
            self.assertEqual(rows[0]["merged_context_sections"], "8")
            self.assertEqual(rows[0]["final_context_sections"], "7")
            self.assertEqual(
                rows[0]["final_context_characters"],
                str(len("Final grounded context.")),
            )
            self.assertEqual(
                int(rows[0]["final_context_tokens_estimate"]),
                create_token_manager().count("Final grounded context."),
            )
            self.assertEqual(len(json.loads(rows[0]["query_variants"])), 4)
            self.assertEqual(rows[0]["answer_status"], "Answered")
            self.assertEqual(
                rows[1]["answer_status"],
                "Insufficient Evidence",
            )
            self.assertIn("Original Query:", rows[0]["retrieval_trace"])
            self.assertIn("Retrieved 34 chunks", rows[0]["retrieval_trace"])
            self.assertIn("Final Context: 7 merged sections", rows[0]["retrieval_trace"])


if __name__ == "__main__":
    unittest.main()
