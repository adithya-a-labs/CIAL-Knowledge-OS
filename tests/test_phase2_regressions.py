from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cial_knowledge_os.citations import build_citations, render_citations
from cial_knowledge_os.config import KnowledgeOSConfig, Phase2Config
from cial_knowledge_os.context_builder import (
    ContextBuilder,
    compress_context,
    merge_overlapping_chunks,
)
from cial_knowledge_os.phase2_pipeline import Phase2RAGPipeline
from cial_knowledge_os.query_transformations import QueryTransformer
from cial_knowledge_os.retrieval_postprocessing import (
    deduplicate_results,
    expand_neighbor_chunks,
)


def _result(
    index: int,
    *,
    source: str | None = "manual.pdf",
    page: int | None = 1,
    chunk_id: str | None = None,
    score: float | None = None,
    text: str | None = None,
    matched_queries: list[str] | None = None,
) -> dict[str, Any]:
    identifier = chunk_id if chunk_id is not None else f"chunk-{index}"
    metadata = {
        "source": f"C:/corpus/{source}" if source else None,
        "file_name": source,
        "page_number": page,
        "chunk_id": identifier,
        "chunk_index": index,
    }
    return {
        "id": f"point-{index}",
        "text": text or f"Evidence {index} " + ("x" * 180),
        "score": score if score is not None else 1.0 - index / 1_000,
        "source": source,
        "page_number": page,
        "chunk_id": identifier,
        "metadata": metadata,
        "matched_queries": matched_queries or ["original"],
    }


def _corpus(source: str, count: int) -> list[Document]:
    return [
        Document(
            page_content=f"Corpus chunk {index}.",
            metadata={
                "source": f"C:/corpus/{source}",
                "file_name": source,
                "page_number": 1,
                "chunk_id": f"chunk-{index}",
                "chunk_index": index,
            },
        )
        for index in range(count)
    ]


class _CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        return "One grounded answer [1]."


class _TracePipeline(Phase2RAGPipeline):
    def __init__(self, *args: Any, search_results: list[dict[str, Any]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.search_results = search_results

    def _search(self, query: str) -> list[dict[str, Any]]:
        return [dict(result) for result in self.search_results]


class ConfigurationRegressionTests(unittest.TestCase):
    def test_phase_defaults_remain_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phase1 = KnowledgeOSConfig(project_root=Path(directory))
            phase2 = Phase2Config(project_root=Path(directory))

        self.assertEqual(phase1.qdrant_collection_name, "cial_basic_rag")
        self.assertEqual(phase2.qdrant_collection_name, "cial_phase2")
        self.assertEqual(phase1.top_k, 3)
        self.assertEqual(phase2.top_k, 3)
        self.assertEqual(phase2.retrieval_top_k, 10)
        self.assertEqual(phase1.max_context_chars, 3_000)
        self.assertEqual(phase2.max_context_chars, 20_000)
        self.assertGreater(phase2.max_context_chars, phase1.max_context_chars)
        self.assertEqual(phase1.qdrant_dir.name, "cial_basic_rag")
        self.assertEqual(phase2.qdrant_dir.name, "cial_phase2")


class LargeRetrievalAndOrderingTests(unittest.TestCase):
    def test_large_deduplication_keeps_best_scores_queries_and_order(self) -> None:
        raw: list[dict[str, Any]] = []
        for index in range(50):
            raw.extend(
                [
                    _result(
                        index,
                        score=0.4 + index / 1_000,
                        matched_queries=["original"],
                    ),
                    _result(
                        index,
                        score=0.8 + index / 1_000,
                        matched_queries=["rewritten"],
                    ),
                ]
            )

        deduplicated = deduplicate_results(raw)
        scores = [result["score"] for result in deduplicated]

        self.assertEqual(len(deduplicated), 50)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score >= 0.8 for score in scores))
        self.assertTrue(
            all(
                result["matched_queries"] == ["original", "rewritten"]
                for result in deduplicated
            )
        )

    def test_large_context_build_is_consistent_and_bounded(self) -> None:
        results = [
            _result(
                index,
                source=f"document-{index}.pdf",
                page=index + 1,
                text=f"Section {index} " + ("x" * 350),
            )
            for index in range(75)
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = Phase2Config(
                project_root=Path(directory),
                enable_neighbor_expansion=False,
                enable_overlap_merging=False,
                max_context_chars=6_000,
            )
            built = ContextBuilder(config).build(results)

        counts = built.stage_counts()
        self.assertEqual(counts["retrieved"], 75)
        self.assertEqual(counts["deduplicated"], 75)
        self.assertEqual(counts["expanded"], 75)
        self.assertEqual(counts["merged"], 75)
        self.assertEqual(counts["compressed"], len(built.compressed))
        self.assertLessEqual(len(built.context), 6_000)
        self.assertTrue(all(value >= 0 for value in counts.values()))

    def test_large_fake_pipeline_does_not_crash(self) -> None:
        results = [
            _result(
                index,
                source=f"document-{index % 10}.pdf",
                page=index // 10 + 1,
            )
            for index in range(75)
        ]
        with tempfile.TemporaryDirectory() as directory:
            llm = _CountingLLM()
            pipeline = _TracePipeline(
                Phase2Config(
                    project_root=Path(directory),
                    enable_neighbor_expansion=False,
                    max_context_chars=3_000,
                ),
                llm=llm,
                search_results=results,
            )
            response = pipeline.answer("Summarize the controls.")

        self.assertTrue(response["answer"])
        self.assertLessEqual(len(response["context"]), 3_000)
        self.assertEqual(llm.calls, 1)


class DuplicateIdentityTests(unittest.TestCase):
    def test_same_text_with_different_identity_is_not_deduplicated(self) -> None:
        common_text = "Identical text."
        results = [
            _result(1, page=1, chunk_id="chunk-a", text=common_text),
            _result(2, page=2, chunk_id="chunk-b", text=common_text),
        ]
        self.assertEqual(len(deduplicate_results(results)), 2)

    def test_same_complete_identity_is_deduplicated(self) -> None:
        results = [
            _result(1, source="a.pdf", page=3, chunk_id="same"),
            _result(2, source="a.pdf", page=3, chunk_id="same"),
        ]
        self.assertEqual(len(deduplicate_results(results)), 1)

    def test_same_chunk_id_from_different_sources_is_not_deduplicated(self) -> None:
        results = [
            _result(1, source="a.pdf", page=3, chunk_id="same"),
            _result(2, source="b.pdf", page=3, chunk_id="same"),
        ]
        self.assertEqual(len(deduplicate_results(results)), 2)

    def test_incomplete_identity_does_not_collapse_unrelated_chunks(self) -> None:
        results = [
            _result(
                1,
                source=None,
                page=None,
                chunk_id=None,
                text="First unknown chunk.",
            ),
            _result(
                2,
                source=None,
                page=None,
                chunk_id=None,
                text="Second unknown chunk.",
            ),
        ]
        for result in results:
            result["chunk_id"] = None
            result["metadata"] = None

        self.assertEqual(len(deduplicate_results(results)), 2)


class NeighborBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = "manual.pdf"
        self.corpus = _corpus(self.source, 4)

    def _seed(self, index: int) -> dict[str, Any]:
        seed = _result(
            index,
            source=self.source,
            page=1,
            chunk_id=f"chunk-{index}",
        )
        seed["metadata"]["source"] = f"C:/corpus/{self.source}"
        return seed

    def test_first_chunk_expands_to_self_and_next(self) -> None:
        expanded = expand_neighbor_chunks(
            [self._seed(0)],
            self.corpus,
            window=1,
        )
        self.assertEqual(
            {result["chunk_id"] for result in expanded},
            {"chunk-0", "chunk-1"},
        )

    def test_last_chunk_expands_to_previous_and_self(self) -> None:
        expanded = expand_neighbor_chunks(
            [self._seed(3)],
            self.corpus,
            window=1,
        )
        self.assertEqual(
            {result["chunk_id"] for result in expanded},
            {"chunk-2", "chunk-3"},
        )

    def test_zero_window_returns_only_original_chunks(self) -> None:
        expanded = expand_neighbor_chunks(
            [self._seed(2)],
            self.corpus,
            window=0,
        )
        self.assertEqual(
            [result["chunk_id"] for result in expanded],
            ["chunk-2"],
        )

    def test_missing_corpus_neighbor_keeps_seed_without_crashing(self) -> None:
        expanded = expand_neighbor_chunks(
            [self._seed(99)],
            self.corpus,
            window=1,
        )
        self.assertEqual(
            [result["chunk_id"] for result in expanded],
            ["chunk-99"],
        )


class QueryTransformationEdgeCaseTests(unittest.TestCase):
    def _config(self, root: Path, **values: Any) -> Phase2Config:
        return Phase2Config(project_root=root, **values)

    def test_max_query_variants_limits_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = QueryTransformer(
                self._config(Path(directory), max_query_variants=2)
            )
            variants = transformer.generate("Please explain runway safety.")
        self.assertEqual(len(variants), 2)

    def test_duplicate_variants_are_removed_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = QueryTransformer(self._config(Path(directory)))
            variants = transformer.generate("runway inspection")
        queries = [variant.query.casefold() for variant in variants]
        self.assertEqual(len(queries), len(set(queries)))

    def test_very_long_query_does_not_crash(self) -> None:
        query = ("runway maintenance safety " * 2_000).strip()
        with tempfile.TemporaryDirectory() as directory:
            variants = QueryTransformer(
                self._config(Path(directory))
            ).generate(query)
        self.assertTrue(variants)
        self.assertEqual(variants[0].query, query)

    def test_disabled_transformations_leave_only_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory),
                enable_query_rewrite=False,
                enable_keyword_expansion=False,
                enable_domain_reformulation=False,
            )
            variants = QueryTransformer(config).generate("Question?")
        self.assertEqual([variant.technique for variant in variants], ["original"])

    def test_disabled_multi_query_leaves_only_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            variants = QueryTransformer(
                self._config(Path(directory), enable_multi_query=False)
            ).generate("Please explain runway safety.")
        self.assertEqual([variant.technique for variant in variants], ["original"])


class ContextBudgetAndMetadataTests(unittest.TestCase):
    def test_context_budget_scaling_is_monotonic(self) -> None:
        results = [
            _result(
                index,
                source=f"document-{index}.pdf",
                page=index + 1,
                text=f"Evidence {index} " + ("x" * 700),
            )
            for index in range(60)
        ]
        retained: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for budget in (3_000, 6_000, 20_000, 25_000):
                with self.subTest(budget=budget):
                    config = Phase2Config(
                        project_root=root,
                        enable_neighbor_expansion=False,
                        enable_overlap_merging=False,
                        max_context_chars=budget,
                    )
                    built = ContextBuilder(config).build(results)
                    counts = built.stage_counts()
                    self.assertLessEqual(len(built.context), budget)
                    self.assertTrue(all(value >= 0 for value in counts.values()))
                    self.assertEqual(counts["retrieved"], len(results))
                    self.assertEqual(
                        counts["compressed"],
                        len(built.compressed),
                    )
                    retained.append(counts["compressed"])

        self.assertEqual(retained, sorted(retained))

    def test_missing_metadata_is_rendered_safely(self) -> None:
        results = [
            {
                "text": "Evidence without nested metadata.",
                "score": None,
                "metadata": None,
            },
            {
                "text": "Evidence with known source.",
                "score": 0.61,
                "source": "known.pdf",
                "page_number": None,
                "chunk_id": None,
                "metadata": {
                    "source": "C:/corpus/known.pdf",
                    "file_name": "known.pdf",
                    "page_number": None,
                    "chunk_id": None,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            built = ContextBuilder(
                Phase2Config(
                    project_root=Path(directory),
                    enable_neighbor_expansion=False,
                    enable_overlap_merging=False,
                )
            ).build(results)
        rendered = render_citations(build_citations(built.compressed))

        self.assertTrue(built.context)
        self.assertNotIn("N/A", built.context.upper())
        self.assertNotIn("None", rendered)
        self.assertNotIn("N/A", rendered.upper())
        self.assertIn("known.pdf", built.context)
        self.assertIn("Similarity Score: 0.610", built.context)


class EndToEndTraceConsistencyTests(unittest.TestCase):
    def test_offline_pipeline_trace_is_internally_consistent(self) -> None:
        results = [_result(1, score=0.8), _result(2, score=0.7)]
        with tempfile.TemporaryDirectory() as directory:
            llm = _CountingLLM()
            pipeline = _TracePipeline(
                Phase2Config(
                    project_root=Path(directory),
                    enable_neighbor_expansion=False,
                ),
                llm=llm,
                search_results=results,
            )
            response = pipeline.answer("Please explain the indexed controls.")

        self.assertTrue(response["query_variants"])
        self.assertTrue(response["retrieved_by_query"])
        self.assertTrue(response["retrieved"])
        self.assertTrue(response["context_stages"])
        self.assertTrue(response["stage_counts"])
        self.assertTrue(response["citations"])
        for stage, values in response["context_stages"].items():
            self.assertEqual(response["stage_counts"][stage], len(values))
        self.assertEqual(llm.calls, 1)
        self.assertEqual(response["raw_answer"], "One grounded answer [1].")


class LightweightPerformanceRegressionTests(unittest.TestCase):
    def test_hundred_chunk_postprocessing_completes_quickly(self) -> None:
        results = [
            _result(
                index,
                source=f"document-{index % 10}.pdf",
                page=index // 10 + 1,
            )
            for index in range(100)
        ]
        corpus = [
            Document(
                page_content=str(result["text"]),
                metadata=dict(result["metadata"]),
            )
            for result in results
        ]

        started = time.perf_counter()
        deduplicated = deduplicate_results(results)
        expanded = expand_neighbor_chunks(
            deduplicated,
            corpus,
            window=1,
        )
        merged = merge_overlapping_chunks(expanded)
        compressed, context = compress_context(
            merged,
            max_chars=12_000,
        )
        elapsed = time.perf_counter() - started

        self.assertTrue(deduplicated)
        self.assertTrue(expanded)
        self.assertTrue(merged)
        self.assertTrue(compressed)
        self.assertLessEqual(len(context), 20_000)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
