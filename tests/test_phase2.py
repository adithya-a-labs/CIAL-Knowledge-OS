from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from cial_knowledge_os.citations import build_citations, render_answer_with_citations
from cial_knowledge_os.config import KnowledgeOSConfig, Phase2Config
from cial_knowledge_os.context_builder import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    ContextBuilder,
    merge_overlapping_chunks,
)
from cial_knowledge_os.llm import generate_answer
from cial_knowledge_os.phase2_pipeline import Phase2RAGPipeline
from cial_knowledge_os.query_transformations import QueryTransformer
from cial_knowledge_os.retrieval_postprocessing import (
    deduplicate_results,
    expand_neighbor_chunks,
    retrieve_multiple_queries,
)


def _result(
    chunk_index: int,
    *,
    score: float = 0.5,
    text: str | None = None,
    source: str = "CISG-2026-01.pdf",
    page: int = 47,
) -> dict[str, object]:
    chunk = f"chunk-{chunk_index}"
    return {
        "id": chunk,
        "text": text or f"Evidence from {chunk}.",
        "score": score,
        "source": source,
        "page_number": page,
        "chunk_id": chunk,
        "metadata": {
            "source": f"C:/corpus/{source}",
            "file_name": source,
            "page_number": page,
            "chunk_id": chunk,
            "chunk_index": chunk_index,
        },
    }


class _NeverCalledLLM:
    def invoke(self, prompt: str) -> str:
        raise AssertionError("LLM must not be called without context")


class _CitingLLM:
    def invoke(self, prompt: str) -> str:
        self.prompt = prompt
        return "The indexed procedure requires a control step [1]."


class _OfflinePhase2Pipeline(Phase2RAGPipeline):
    def _search(self, query: str) -> list[dict[str, object]]:
        return [_result(1, score=0.7)]


class _EmptyOfflinePhase2Pipeline(Phase2RAGPipeline):
    def _search(self, query: str) -> list[dict[str, object]]:
        return []


class Phase2ConfigurationTests(unittest.TestCase):
    def test_phase1_defaults_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phase1 = KnowledgeOSConfig(project_root=Path(directory))
            phase2 = Phase2Config(project_root=Path(directory))

        self.assertEqual(phase1.top_k, 3)
        self.assertEqual(phase2.top_k, 3)
        self.assertEqual(phase2.retrieval_top_k, 10)

    def test_phase2_validates_new_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "retrieval_top_k"):
                Phase2Config(
                    project_root=Path(directory),
                    retrieval_top_k=0,
                )


class QueryTransformationTests(unittest.TestCase):
    def test_all_transformations_are_exposed_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = QueryTransformer(
                Phase2Config(project_root=Path(directory))
            )
            query = "Could you please explain runway maintenance safety?"
            variants = transformer.generate(query)

        self.assertEqual(
            [variant.technique for variant in variants],
            [
                "original",
                "rewritten",
                "keyword_expanded",
                "domain_reformulation",
            ],
        )
        self.assertIn("ATC clearance", variants[2].query)
        self.assertIn("CIAL airport operations", variants[3].query)

    def test_custom_strategy_is_an_extension_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = QueryTransformer(
                Phase2Config(project_root=Path(directory))
            )
            transformer.register("test_strategy", lambda query: f"local::{query}")
            result = transformer.transform("question", "test_strategy")

        self.assertEqual(result.query, "local::question")


class RetrievalPostprocessingTests(unittest.TestCase):
    def test_deduplication_uses_source_page_and_chunk_id(self) -> None:
        first = _result(151, score=0.665)
        duplicate = _result(151, score=0.700)
        duplicate["matched_queries"] = ["rewritten"]
        first["matched_queries"] = ["original"]

        results = deduplicate_results([first, duplicate])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["score"], 0.700)
        self.assertEqual(
            results[0]["matched_queries"],
            ["original", "rewritten"],
        )

    def test_multi_query_merges_evidence_not_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = QueryTransformer(
                Phase2Config(project_root=Path(directory), max_query_variants=2)
            )
            variants = transformer.generate("Could you explain runway inspection?")

        def search(query: str) -> list[dict[str, object]]:
            return [_result(1, score=0.6), _result(len(query), score=0.4)]

        merged, by_query = retrieve_multiple_queries(variants, search)

        self.assertEqual(set(by_query), {"original", "rewritten"})
        self.assertTrue(all("text" in result for result in merged))
        self.assertNotIn("answer", merged[0])
        self.assertIn("original", merged[0]["matched_queries"])

    def test_neighbor_expansion_adds_adjacent_chunks(self) -> None:
        documents = [
            Document(
                page_content=f"Text {index}",
                metadata={
                    "source": "C:/corpus/manual.pdf",
                    "file_name": "manual.pdf",
                    "page_number": 10,
                    "chunk_id": f"chunk-{index}",
                    "chunk_index": index,
                },
            )
            for index in range(5)
        ]
        seed = _result(2, source="manual.pdf", page=10)
        seed["metadata"]["source"] = "C:/corpus/manual.pdf"  # type: ignore[index]

        expanded = expand_neighbor_chunks([seed], documents, window=1)

        self.assertEqual(
            {result["chunk_id"] for result in expanded},
            {"chunk-1", "chunk-2", "chunk-3"},
        )
        self.assertEqual(
            sum(not result["is_neighbor"] for result in expanded),
            1,
        )


class ContextConstructionTests(unittest.TestCase):
    def test_merging_removes_text_overlap(self) -> None:
        left = _result(1, text="Alpha shared")
        right = _result(2, text="shared omega")

        merged = merge_overlapping_chunks([left, right])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "Alpha shared omega")
        self.assertEqual(merged[0]["merged_chunk_count"], 2)

    def test_builder_exposes_stages_and_rich_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase2Config(
                project_root=Path(directory),
                enable_neighbor_expansion=False,
                max_context_chars=1_000,
            )
            built = ContextBuilder(config).build([_result(151, score=0.665)])

        self.assertEqual(
            built.stage_counts(),
            {
                "retrieved": 1,
                "deduplicated": 1,
                "expanded": 1,
                "merged": 1,
                "compressed": 1,
            },
        )
        self.assertIn("Document: CISG-2026-01.pdf", built.context)
        self.assertIn("Page: 47", built.context)
        self.assertIn("Chunk ID: chunk-151", built.context)
        self.assertIn("Similarity Score: 0.665", built.context)
        self.assertNotIn("N/A", built.context)

    def test_compression_respects_character_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase2Config(
                project_root=Path(directory),
                enable_neighbor_expansion=False,
                max_context_chars=180,
            )
            built = ContextBuilder(config).build(
                [_result(1, text="x" * 500)]
            )

        self.assertLessEqual(len(built.context), 180)
        self.assertTrue(built.compressed[0]["context_truncated"])
        self.assertLess(len(built.compressed[0]["text"]), 500)


class SafeFailureAndCitationTests(unittest.TestCase):
    def test_phase2_empty_context_uses_explicit_safe_failure(self) -> None:
        answer = generate_answer(
            _NeverCalledLLM(),
            "Unknown?",
            "",
            no_evidence_response=INSUFFICIENT_EVIDENCE_RESPONSE,
        )
        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_RESPONSE)

    def test_safe_failure_does_not_append_references(self) -> None:
        citations = build_citations([_result(151)])
        rendered = render_answer_with_citations(
            INSUFFICIENT_EVIDENCE_RESPONSE,
            citations,
        )
        self.assertEqual(rendered, INSUFFICIENT_EVIDENCE_RESPONSE)

    def test_phase2_pipeline_returns_inspectable_trace_and_one_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            llm = _CitingLLM()
            pipeline = _OfflinePhase2Pipeline(
                Phase2Config(
                    project_root=Path(directory),
                    enable_neighbor_expansion=False,
                    max_query_variants=2,
                ),
                llm=llm,
            )
            response = pipeline.answer("Could you explain the control step?")

        self.assertIn("query_variants", response)
        self.assertIn("context_stages", response)
        self.assertIn("Document: CISG-2026-01.pdf", llm.prompt)
        self.assertIn("References:", response["answer"])

    def test_empty_retrieval_does_not_require_a_local_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _EmptyOfflinePhase2Pipeline(
                Phase2Config(project_root=Path(directory))
            )
            response = pipeline.answer("Question absent from the corpus?")

        self.assertEqual(response["answer"], INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(response["citations"], [])


if __name__ == "__main__":
    unittest.main()
