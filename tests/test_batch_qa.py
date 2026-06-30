from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from cial_knowledge_os.batch_qa import export_batch_answers
from cial_knowledge_os.rag_pipeline import BasicRAGPipeline


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


if __name__ == "__main__":
    unittest.main()
