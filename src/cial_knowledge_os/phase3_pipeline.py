"""Phase 3 hybrid retrieval with backward-compatible Phase 2 contracts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from .benchmarking import Timer
from .citation_links import CitationLinkBuilder
from .citations import build_citations, render_answer_with_citations
from .config import Phase3Config
from .context_builder import ContextBuilder, INSUFFICIENT_EVIDENCE_RESPONSE
from .fusion import ReciprocalRankFusion
from .llm import LocalLLM, build_grounded_prompt
from .phase2_pipeline import Phase2RAGPipeline
from .query_transformations import QueryTransformer
from .retrieval import search_similar_chunks
from .retrieval_trace import build_question_trace
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from .token_budget import (
    TokenBudgetManager,
    TokenManager,
    Tokenizer,
    create_token_budget_manager,
    create_token_manager,
)

logger = logging.getLogger(__name__)


class Phase3RAGPipeline(Phase2RAGPipeline):
    """Add lexical/hybrid retrieval while preserving Phase 2 response keys."""

    config: Phase3Config

    def __init__(
        self,
        config: Phase3Config | None = None,
        *,
        embedding_model: SentenceTransformer | None = None,
        llm: LocalLLM | None = None,
        query_transformer: QueryTransformer | None = None,
        tokenizer: Tokenizer | None = None,
        retrievers: Mapping[str, Retriever] | None = None,
    ) -> None:
        phase3_config = config or Phase3Config()
        self._provided_tokenizer = tokenizer
        self._injected_retrievers = dict(retrievers or {})
        self._retrievers: dict[str, Retriever] = {}
        self.bm25_retriever: BM25Retriever | None = None
        self.hybrid_retriever: HybridRetriever | None = None
        self.last_modality_results_by_query: dict[
            str,
            dict[str, list[dict[str, Any]]],
        ] = {}
        self.token_manager: TokenManager
        self.token_budget_manager: TokenBudgetManager | None = None
        self._token_config_key: tuple[int | None, str, int | None] | None = None
        super().__init__(
            config=phase3_config,
            embedding_model=embedding_model,
            llm=llm,
            query_transformer=query_transformer,
        )
        self.citation_link_builder = CitationLinkBuilder(
            mode=phase3_config.citation_link_mode,
            base_url=phase3_config.citation_base_url,
        )
        self._configure_token_management()

    def _configure_token_management(self) -> None:
        key = (
            id(self._provided_tokenizer)
            if self._provided_tokenizer is not None
            else None,
            self.config.tokenizer_encoding_name,
            self.config.max_context_tokens,
        )
        if self._token_config_key == key:
            return
        if self.config.max_context_tokens is None:
            self.token_manager = create_token_manager(
                encoding_name=self.config.tokenizer_encoding_name,
                tokenizer=self._provided_tokenizer,
            )
            self.context_builder = ContextBuilder(self.config)
            self.token_budget_manager = None
            self._token_config_key = key
            return
        self.token_budget_manager = create_token_budget_manager(
            max_tokens=self.config.max_context_tokens,
            encoding_name=self.config.tokenizer_encoding_name,
            tokenizer=self._provided_tokenizer,
        )
        self.token_manager = self.token_budget_manager
        self.context_builder = ContextBuilder(
            self.config,
            token_budget_manager=self.token_budget_manager,
        )
        self._token_config_key = key

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError(
                "Dense retrieval is not initialized. Call embed() and index() "
                "before using dense or hybrid retrieval."
            )
        return search_similar_chunks(
            self.client,
            query,
            self.embedding_model,
            self.config,
            top_k=top_k,
        )

    def _ensure_retrievers(self) -> None:
        if self._retrievers:
            return
        dense = self._injected_retrievers.get("dense") or DenseRetriever(
            self._dense_search
        )
        lexical = self._injected_retrievers.get("bm25")
        if lexical is None:
            cache_path = (
                Path(self.config.bm25_cache_dir)
                / self.config.bm25_cache_filename
            )
            lexical = BM25Retriever(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
                cache_path=cache_path,
            )
        self._retrievers = {
            **self._injected_retrievers,
            "dense": dense,
            "bm25": lexical,
        }
        if isinstance(lexical, BM25Retriever):
            self.bm25_retriever = lexical
        self.hybrid_retriever = HybridRetriever(
            [dense, lexical],
            fuser=ReciprocalRankFusion(
                rank_constant=self.config.rrf_k,
                weights={
                    "dense": self.config.dense_weight,
                    "bm25": self.config.bm25_weight,
                },
            ),
            candidate_limits={
                "dense": self.config.dense_top_k,
                "bm25": self.config.bm25_top_k,
            },
            parallel=self.config.parallel_retrieval,
        )

    def on_config_changed(self) -> None:
        """Refresh cheap configuration-bound components for experiment sweeps."""

        self._configure_token_management()
        if self.hybrid_retriever is not None:
            self.hybrid_retriever.fuser = ReciprocalRankFusion(
                rank_constant=self.config.rrf_k,
                weights={
                    "dense": self.config.dense_weight,
                    "bm25": self.config.bm25_weight,
                },
            )
            self.hybrid_retriever.candidate_limits = {
                "dense": self.config.dense_top_k,
                "bm25": self.config.bm25_top_k,
            }
            self.hybrid_retriever.parallel = self.config.parallel_retrieval

    def build_lexical_index(self) -> bool:
        """Build or reuse BM25 from the already-created Phase 2 chunks."""

        if not self.chunks and self.indexing_plan is None:
            raise RuntimeError(
                "Call load() and chunk() before building the BM25 index."
            )
        self._ensure_retrievers()
        lexical = self._retrievers["bm25"]
        index = getattr(lexical, "index", None)
        if not callable(index):
            return False
        with Timer(self.metrics, "bm25_indexing_time"):
            return bool(index(self.chunks))

    def index(self):
        """Build dense and lexical indexes once from the shared chunk corpus."""

        plan = self.indexing_plan
        if (
            plan is not None
            and plan.unchanged
            and not plan.corpus_changed
            and self.config.retrieval_mode in {"bm25", "hybrid"}
        ):
            cache_path = (
                Path(self.config.bm25_cache_dir)
                / self.config.bm25_cache_filename
            )
            self.indexing_summary["bm25_cache_validated"] = cache_path.is_file()
            if not cache_path.is_file():
                logger.warning(
                    "bm25_cache_missing_rebuild_required",
                    extra={
                        "event": "indexing_consistency",
                        "bm25_cache_path": str(cache_path),
                    },
                )
        client = super().index()
        if self.config.retrieval_mode in {"bm25", "hybrid"}:
            self.build_lexical_index()
            plan = self.indexing_plan
            self.indexing_summary["bm25_rebuilt"] = bool(
                plan is None
                or plan.corpus_changed
                or not self.config.incremental_indexing_enabled
            )
            self.execution_manager.emit(
                "bm25_health_checked",
                stage="indexing",
                status="completed",
                payload={
                    "indexed": bool(
                        getattr(self.bm25_retriever, "is_indexed", True)
                    ),
                    "rebuilt": self.indexing_summary["bm25_rebuilt"],
                    "chunk_count": len(self.chunks),
                },
                source="phase3_pipeline.index",
            )
            self.execution_manager.complete_stage(
                "indexing",
                event_type="indexing_completed",
                metrics={
                    "indexing_latency_seconds": self.metrics.get(
                        "indexing_time", 0.0
                    ),
                    "bm25_indexing_latency_seconds": self.metrics.get(
                        "bm25_indexing_time", 0.0
                    ),
                },
                points_upserted=int(
                    self.indexing_summary.get("chunks_added", 0)
                ),
                **self.indexing_summary,
            )
        logger.info(
            "phase3_index_ready",
            extra={
                "event": "indexing",
                "retrieval_mode": self.config.retrieval_mode,
                "chunk_count": len(self.chunks),
                "indexing_summary": self.indexing_summary,
            },
        )
        return client

    def _search(self, query: str) -> list[dict[str, Any]]:
        self._ensure_retrievers()
        mode = self.config.retrieval_mode
        if mode in {"bm25", "hybrid"}:
            lexical = self._retrievers["bm25"]
            if not bool(getattr(lexical, "is_indexed", True)):
                self.build_lexical_index()
        if mode == "hybrid":
            assert self.hybrid_retriever is not None
            results = self.hybrid_retriever.retrieve(
                query,
                top_k=self.config.retrieval_top_k,
            )
            self.last_modality_results_by_query[query] = {
                **{
                    name: [dict(item) for item in values]
                    for name, values in self.hybrid_retriever.last_rankings.items()
                },
                "fused": [dict(item) for item in results],
            }
            return results
        if mode not in self._retrievers:
            available = ", ".join(sorted([*self._retrievers, "hybrid"]))
            raise ValueError(
                f"Retrieval mode '{mode}' has no registered retriever. "
                f"Available modes: {available}."
            )
        results = self._retrievers[mode].retrieve(
            query,
            top_k=self.config.retrieval_top_k,
        )
        self.last_modality_results_by_query[query] = {
            mode: [dict(item) for item in results],
            "fused": [dict(item) for item in results],
        }
        return results

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Reset and capture modality-level results for one question."""

        self.last_modality_results_by_query = {}
        return super().retrieve(question)

    def answer(self, question: str) -> dict[str, Any]:
        """Run Phase 2 orchestration with Phase 3 retrieval and enrichment."""

        self._configure_token_management()
        response = super().answer(question)
        compressed = response.get("context_stages", {}).get("compressed", [])
        if response.get("answer_status") == "answered":
            citations = build_citations(
                compressed,
                link_resolver=self.citation_link_builder,
            )
            response["citations"] = citations
            response["answer"] = render_answer_with_citations(
                str(response.get("raw_answer") or ""),
                citations,
            )
        context = str(response.get("context") or "")
        response["prompt"] = (
            build_grounded_prompt(
                question,
                context,
                no_evidence_response=INSUFFICIENT_EVIDENCE_RESPONSE,
            )
            if context
            else ""
        )
        usage = (
            self.token_budget_manager.last_usage
            if self.token_budget_manager is not None
            else None
        )
        context_token_count = self.token_manager.count(context)
        response["token_usage"] = (
            {
                "budget": usage.budget,
                "used": usage.used,
                "remaining": usage.remaining,
                "context_tokens": usage.used,
                "encoding_name": usage.encoding_name,
                "truncated_sections": usage.truncated_sections,
                "omitted_sections": usage.omitted_sections,
                "budget_type": "tokens",
            }
            if usage is not None
            else {
                "budget": None,
                "used": context_token_count,
                "remaining": None,
                "context_tokens": context_token_count,
                "encoding_name": self.token_manager.encoding_name,
                "truncated_sections": sum(
                    bool(item.get("context_truncated"))
                    for item in compressed
                ),
                "omitted_sections": 0,
                "budget_type": "characters_legacy",
                "character_budget": self.config.max_context_chars,
                "characters_used": len(context),
                "characters_remaining": max(
                    0,
                    self.config.max_context_chars - len(context),
                ),
            }
        )
        response["retrieval_mode"] = self.config.retrieval_mode
        response["retrieval_trace"] = {
            "mode": self.config.retrieval_mode,
            "query_variants": response.get("query_variants") or [],
            "retrieved_by_query": response.get("retrieved_by_query") or {},
            "stage_counts": response.get("stage_counts") or {},
            "token_usage": response["token_usage"],
        }
        response["question_trace"] = build_question_trace(
            question=question,
            response=response,
            modality_results_by_query=self.last_modality_results_by_query,
            token_manager=self.token_manager,
            config=self.config,
            metrics=self.metrics,
            link_resolver=self.citation_link_builder,
        )
        self.metrics["context_tokens"] = float(
            response["token_usage"]["context_tokens"]
        )
        logger.info(
            "phase3_answer_complete",
            extra={
                "event": "answer",
                "retrieval_mode": self.config.retrieval_mode,
                "answer_status": response.get("answer_status"),
                "citation_count": len(response.get("citations") or []),
            },
        )
        return response
