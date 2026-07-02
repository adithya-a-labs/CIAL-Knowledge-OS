from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cial_knowledge_os.benchmark_loader import load_benchmark
from cial_knowledge_os.experiment_config import ExperimentGrid
from cial_knowledge_os.experiment_runner import ExperimentRunner
from cial_knowledge_os.experiment_runner import ReconfiguringPipelineFactory
from cial_knowledge_os.token_budget import create_token_manager


class _OfflinePipeline:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.config = SimpleNamespace(retrieval_top_k=top_k)
        self.metrics: dict[str, float] = {}

    def answer(self, question: str) -> dict[str, object]:
        self.metrics = {
            "retrieval_latency": self.top_k / 1000,
            "context_construction_latency": 0.002,
            "generation_latency": 0.01,
            "total_pipeline_latency": 0.012 + self.top_k / 1000,
        }
        unsupported = "unsupported" in question.casefold()
        answer = (
            "The retrieved documents do not contain sufficient evidence."
            if unsupported
            else "CERT-In provides the grounded answer [1]."
        )
        evidence = {
            "text": "CERT-In evidence",
            "score": 0.9,
            "metadata": {
                "file_name": "policy.pdf",
                "page_number": 1,
                "chunk_id": "policy:1",
            },
        }
        return {
            "answer": answer,
            "answer_status": (
                "insufficient_evidence" if unsupported else "answered"
            ),
            "citations": [{"source": "policy.pdf", "page_number": 1}],
            "retrieved": [evidence] * self.top_k,
            "context": "CERT-In evidence",
            "stage_counts": {
                "retrieved": self.top_k,
                "deduplicated": self.top_k - 1,
                "expanded": self.top_k + 1,
                "merged": 2,
                "compressed": 1,
            },
        }


class EvaluationFrameworkTests(unittest.TestCase):
    def _benchmark(self, root: Path):
        path = root / "benchmark.csv"
        path.write_text(
            "question_id,question,answer,expected_keywords,category,difficulty,"
            "should_answer\n"
            "1,Who issued it?,CERT-In issued it,CERT-In,factual,easy,true\n"
            "2,Unsupported question,No answer,,unsupported,hard,false\n",
            encoding="utf-8",
        )
        return load_benchmark(path)

    def test_grid_is_deterministic_cartesian_product(self) -> None:
        configs = ExperimentGrid(
            {"retrieval_top_k": [3, 5], "neighbor_window": [0, 1]}
        ).expand()
        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0].experiment_id, "experiment_001")
        self.assertEqual(
            configs[-1].parameters,
            {"retrieval_top_k": 5, "neighbor_window": 1},
        )

    def test_sweep_writes_csv_summary_recommendation_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            output = root / "outputs" / "phase"
            runner = ExperimentRunner(
                pipeline_factory=lambda config: _OfflinePipeline(
                    int(config.parameters["retrieval_top_k"])
                ),
                benchmark=benchmark,
                output_root=output,
            )

            result = runner.run(
                ExperimentGrid({"retrieval_top_k": [3, 5]})
            )

            self.assertEqual(len(result.experiment_files), 2)
            self.assertTrue(result.summary_file.is_file())
            self.assertTrue(result.recommendation_file.is_file())
            self.assertTrue(result.dashboard_file.is_file())
            with result.experiment_files[0].open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["passed_answer_test"], "True")
            self.assertEqual(rows[1]["safe_failure"], "True")
            self.assertEqual(
                int(rows[0]["estimated_tokens"]),
                create_token_manager().count("CERT-In evidence"),
            )
            dashboard = result.dashboard_file.read_text(encoding="utf-8")
            self.assertIn("Configuration Leaderboard", dashboard)
            self.assertIn("Experiment Explorer", dashboard)
            self.assertIn("experiment_001", dashboard)
            self.assertNotIn("https://", dashboard)
            self.assertNotIn("<script src=", dashboard)

    def test_reused_pipeline_configuration_is_restored(self) -> None:
        pipeline = _OfflinePipeline(10)
        factory = ReconfiguringPipelineFactory(pipeline)
        config = ExperimentGrid({"retrieval_top_k": [3]}).expand()[0]

        configured = factory(config)
        self.assertIs(configured, pipeline)
        self.assertEqual(pipeline.config.retrieval_top_k, 3)

        factory.close()
        self.assertEqual(pipeline.config.retrieval_top_k, 10)


if __name__ == "__main__":
    unittest.main()
