from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cial_knowledge_os.config import KnowledgeOSConfig, Phase2Config, Phase3Config
from cial_knowledge_os.fusion import ReciprocalRankFusion
from cial_knowledge_os.phase3_pipeline import Phase3RAGPipeline
from cial_knowledge_os.retrievers import BM25Retriever
from cial_knowledge_os.token_budget import (
    TiktokenTokenizer,
    TokenBudgetManager,
    create_token_manager,
)


def _result(index: int, text: str, score: float = 0.5) -> dict[str, Any]:
    return {
        "id": index,
        "text": text,
        "score": score,
        "source": "manual.pdf",
        "page_number": 2,
        "chunk_id": f"chunk-{index}",
        "metadata": {
            "source": "C:/corpus/manual.pdf",
            "file_name": "manual.pdf",
            "page_number": 2,
            "chunk_id": f"chunk-{index}",
            "chunk_index": index,
        },
    }


class _CharacterTokenizer:
    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, values: list[int], **_: Any) -> str:
        return "".join(chr(value) for value in values)


class _StaticRetriever:
    def __init__(self, name: str, results: list[dict[str, Any]]) -> None:
        self.name = name
        self.results = results

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return [dict(value) for value in self.results[:top_k]]


class _CitingLLM:
    def invoke(self, prompt: str) -> str:
        self.prompt = prompt
        return "The exact identifier is required [1]."


class Phase3ConfigurationTests(unittest.TestCase):
    def test_previous_phase_defaults_remain_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = KnowledgeOSConfig(project_root=root)
            phase2 = Phase2Config(project_root=root)
            phase3 = Phase3Config(project_root=root)

        self.assertEqual(phase1.qdrant_collection_name, "cial_basic_rag")
        self.assertEqual(phase2.qdrant_collection_name, "cial_phase2")
        self.assertEqual(phase1.top_k, 3)
        self.assertEqual(phase2.retrieval_top_k, 10)
        self.assertEqual(phase3.retrieval_mode, "hybrid")
        self.assertEqual(phase3.max_context_tokens, 4096)
        self.assertEqual(phase3.tokenizer_encoding_name, "cl100k_base")

    def test_invalid_phase3_values_fail_actionably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "retrieval_mode"):
                Phase3Config(
                    project_root=Path(directory),
                    retrieval_mode="",
                )
            with self.assertRaisesRegex(ValueError, "citation_base_url"):
                Phase3Config(
                    project_root=Path(directory),
                    citation_link_mode="localhost",
                )


class BM25AndFusionTests(unittest.TestCase):
    def test_bm25_finds_exact_rare_identifier_and_reuses_index(self) -> None:
        chunks = [
            Document(
                page_content="General runway inspection procedure",
                metadata={
                    "source": "C:/corpus/manual.pdf",
                    "file_name": "manual.pdf",
                    "page_number": 1,
                    "chunk_id": "general",
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="Control identifier AGL-47 requires verification",
                metadata={
                    "source": "C:/corpus/manual.pdf",
                    "file_name": "manual.pdf",
                    "page_number": 2,
                    "chunk_id": "agl-47",
                    "chunk_index": 1,
                },
            ),
            Document(
                page_content="Passenger terminal queue monitoring",
                metadata={
                    "source": "C:/corpus/terminal.pdf",
                    "file_name": "terminal.pdf",
                    "page_number": 1,
                    "chunk_id": "terminal",
                    "chunk_index": 0,
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            retriever = BM25Retriever(cache_path=Path(directory) / "index.pkl")
            self.assertTrue(retriever.index(chunks))
            self.assertFalse(retriever.index(chunks))
            results = retriever.retrieve("AGL-47", top_k=2)

        self.assertEqual(results[0]["chunk_id"], "agl-47")
        self.assertEqual(results[0]["retrieval_sources"], ["bm25"])

    def test_corrupted_bm25_cache_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "index.pkl"
            cache.write_bytes(b"not-a-pickle")
            retriever = BM25Retriever(cache_path=cache)
            rebuilt = retriever.index(
                [
                    Document(
                        page_content="usable lexical evidence",
                        metadata={"source": "a.txt", "chunk_id": "a"},
                    )
                ]
            )
        self.assertTrue(rebuilt)

    def test_rrf_uses_rank_positions_and_preserves_scores(self) -> None:
        first = _result(1, "first", score=0.91)
        second = _result(2, "second", score=0.62)
        fused = ReciprocalRankFusion(rank_constant=10).fuse(
            {
                "dense": [first, second],
                "bm25": [second, first],
            }
        )

        self.assertEqual([item["chunk_id"] for item in fused], ["chunk-1", "chunk-2"])
        self.assertEqual(set(fused[0]["retrieval_sources"]), {"dense", "bm25"})
        self.assertEqual(fused[0]["retrieval_scores"]["dense"], 0.91)
        self.assertNotEqual(fused[0]["score"], 0.91)


class TokenBudgetAndPipelineTests(unittest.TestCase):
    def test_tiktoken_is_the_primary_exact_counter(self) -> None:
        manager = create_token_manager(encoding_name="cl100k_base")

        self.assertIsInstance(manager.tokenizer, TiktokenTokenizer)
        self.assertEqual(manager.count("hello world"), 2)
        self.assertEqual(manager.truncate("hello world again", 2), "hello world")
        self.assertEqual(
            manager.remaining(used_tokens=2, max_tokens=5),
            3,
        )

    def test_token_manager_counts_and_truncates_exactly(self) -> None:
        manager = TokenBudgetManager(_CharacterTokenizer(), max_tokens=5)
        self.assertEqual(manager.count("abcdef"), 6)
        self.assertEqual(manager.truncate("abcdef", 5), "abcde")
        usage = manager.record_usage(
            used=5,
            truncated_sections=1,
            omitted_sections=0,
        )
        self.assertEqual(usage.remaining, 0)

    def test_phase3_pipeline_preserves_phase2_trace_and_adds_hybrid_fields(
        self,
    ) -> None:
        evidence = _result(1, "Exact AGL-47 evidence.", score=0.8)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Phase3RAGPipeline(
                Phase3Config(
                    project_root=Path(directory),
                    max_context_tokens=500,
                    max_query_variants=1,
                    enable_neighbor_expansion=False,
                ),
                llm=_CitingLLM(),
                tokenizer=_CharacterTokenizer(),
                retrievers={
                    "dense": _StaticRetriever("dense", [evidence]),
                    "bm25": _StaticRetriever("bm25", [evidence]),
                },
            )
            response = pipeline.answer("What is AGL-47?")

        self.assertEqual(response["retrieval_mode"], "hybrid")
        self.assertIn("context_stages", response)
        self.assertIn("token_usage", response)
        self.assertLessEqual(
            response["token_usage"]["used"],
            response["token_usage"]["budget"],
        )
        self.assertTrue(response["citations"][0]["pdf_link"].startswith("file:///"))
        self.assertIn("References:", response["answer"])
        trace = response["question_trace"]
        self.assertEqual(trace["question"], "What is AGL-47?")
        self.assertEqual(len(trace["dense_results"]), 1)
        self.assertEqual(len(trace["bm25_results"]), 1)
        self.assertEqual(len(trace["fused_results"]), 1)
        self.assertEqual(trace["overlap"]["both_count"], 1)
        self.assertEqual(
            trace["deduplication"]["key"],
            "source + page + chunk_id",
        )
        self.assertEqual(trace["token_usage"]["chunks_included"], 1)
        self.assertTrue(trace["decision_summary"])
        self.assertEqual(
            set(trace["citations"][0]["retrieval_sources"]),
            {"dense", "bm25"},
        )

    def test_phase3_context_budget_uses_tiktoken_by_default(self) -> None:
        evidence = _result(
            2,
            "Exact evidence repeated for token fitting. " * 20,
            score=0.8,
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Phase3RAGPipeline(
                Phase3Config(
                    project_root=Path(directory),
                    max_context_tokens=80,
                    max_query_variants=1,
                    enable_neighbor_expansion=False,
                ),
                llm=_CitingLLM(),
                retrievers={
                    "dense": _StaticRetriever("dense", [evidence]),
                    "bm25": _StaticRetriever("bm25", [evidence]),
                },
            )
            response = pipeline.answer("What is documented?")

        usage = response["token_usage"]
        self.assertEqual(usage["encoding_name"], "cl100k_base")
        self.assertEqual(
            usage["used"],
            pipeline.token_manager.count(response["context"]),
        )
        self.assertLessEqual(usage["used"], 80)
        self.assertGreaterEqual(usage["remaining"], 0)

    def test_custom_retrieval_mode_is_added_by_injection_only(self) -> None:
        evidence = _result(3, "Graph-backed evidence.", score=0.9)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Phase3RAGPipeline(
                Phase3Config(
                    project_root=Path(directory),
                    retrieval_mode="knowledge_graph",
                    max_context_tokens=500,
                    max_query_variants=1,
                    enable_neighbor_expansion=False,
                ),
                llm=_CitingLLM(),
                tokenizer=_CharacterTokenizer(),
                retrievers={
                    "knowledge_graph": _StaticRetriever(
                        "knowledge_graph",
                        [evidence],
                    )
                },
            )
            response = pipeline.answer("What is documented?")

        self.assertEqual(response["retrieval_mode"], "knowledge_graph")
        self.assertTrue(response["retrieved"])

    def test_character_budget_fallback_still_reports_exact_tiktoken_usage(
        self,
    ) -> None:
        evidence = _result(4, "Legacy character-bounded evidence.", score=0.9)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Phase3RAGPipeline(
                Phase3Config(
                    project_root=Path(directory),
                    max_context_tokens=None,
                    max_context_chars=500,
                    max_query_variants=1,
                    enable_neighbor_expansion=False,
                ),
                llm=_CitingLLM(),
                retrievers={
                    "dense": _StaticRetriever("dense", [evidence]),
                    "bm25": _StaticRetriever("bm25", [evidence]),
                },
            )
            response = pipeline.answer("What is documented?")

        usage = response["token_usage"]
        self.assertEqual(usage["budget_type"], "characters_legacy")
        self.assertEqual(usage["encoding_name"], "cl100k_base")
        self.assertEqual(
            usage["context_tokens"],
            pipeline.token_manager.count(response["context"]),
        )
        self.assertEqual(usage["character_budget"], 500)


if __name__ == "__main__":
    unittest.main()
