"""Phase 2 orchestration layered on the completed Basic RAG pipeline."""

from __future__ import annotations

import time
from typing import Any

from sentence_transformers import SentenceTransformer

from .benchmarking import Timer
from .citations import build_citations, render_answer_with_citations
from .config import Phase2Config
from .context_builder import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    ContextBuilder,
)
from .llm import LocalLLM, create_local_llm, generate_answer
from .query_transformations import QueryTransformer, QueryVariant
from .rag_pipeline import BasicRAGPipeline
from .retrieval import search_similar_chunks
from .retrieval_postprocessing import deduplicate_results, retrieve_multiple_queries


class Phase2RAGPipeline(BasicRAGPipeline):
    """Add query variants and inspectable context construction to Phase 1."""

    config: Phase2Config

    def __init__(
        self,
        config: Phase2Config | None = None,
        *,
        embedding_model: SentenceTransformer | None = None,
        llm: LocalLLM | None = None,
        query_transformer: QueryTransformer | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        phase2_config = config or Phase2Config()
        super().__init__(
            config=phase2_config,
            embedding_model=embedding_model,
            llm=llm,
        )
        self.query_transformer = query_transformer or QueryTransformer(phase2_config)
        self.context_builder = context_builder or ContextBuilder(phase2_config)
        self.last_query_variants: list[QueryVariant] = []
        self.last_retrieval_by_query: dict[str, list[dict[str, Any]]] = {}
        self.last_merged_retrieval: list[dict[str, Any]] = []

    def transform_query(self, question: str) -> list[QueryVariant]:
        """Expose the configured query variants before retrieval."""

        self.last_query_variants = self.query_transformer.generate(question)
        return self.last_query_variants

    def _search(self, query: str) -> list[dict[str, Any]]:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("Call index() before retrieve().")
        return search_similar_chunks(
            self.client,
            query,
            self.embedding_model,
            self.config,
            top_k=self.config.retrieval_top_k,
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve multiple query variants and merge evidence, not answers."""

        variants = self.transform_query(question)
        with Timer(self.metrics, "retrieval_latency"):
            merged, by_query = retrieve_multiple_queries(
                variants,
                self._search,
                deduplicate=False,
            )
            retrieved = deduplicate_results(merged)
        self.last_retrieval_by_query = by_query
        self.last_merged_retrieval = merged
        return retrieved

    def _generate_grounded_answer(self, question: str, context: str) -> str:
        """Generate with the established Phase 2/3 concise grounded prompt.

        Inputs are the user question and final fitted context; output is the raw
        local-model answer. This private hook preserves the exact Phase 1--3
        generation behavior while allowing later phases to specialize answer
        presentation without changing retrieval, context, or citation contracts.
        """

        if self.llm is None:
            raise RuntimeError("Initialize the local LLM before generation.")
        return generate_answer(
            self.llm,
            question,
            context,
            no_evidence_response=INSUFFICIENT_EVIDENCE_RESPONSE,
        )

    def answer(self, question: str) -> dict[str, Any]:
        """Construct compressed context and generate one grounded answer."""

        started_at = time.perf_counter()
        results = self.retrieve(question)
        with Timer(self.metrics, "context_construction_latency"):
            context_result = self.context_builder.build(
                self.last_merged_retrieval,
                corpus_chunks=self.chunks,
            )
        with Timer(self.metrics, "generation_latency"):
            if not context_result.context.strip():
                raw_answer = INSUFFICIENT_EVIDENCE_RESPONSE
            else:
                if self.llm is None:
                    self.llm = create_local_llm(self.config)
                raw_answer = self._generate_grounded_answer(
                    question,
                    context_result.context,
                )
        self.metrics["total_pipeline_latency"] = time.perf_counter() - started_at
        
        normalized_answer = raw_answer.lower()

        insufficient_markers = (
            "insufficient evidence",
            "not available in the retrieved documents",
            "retrieved documents do not contain",
            "does not contain sufficient",
            "cannot be answered",
            "based only on the indexed corpus, no reliable answer",
        )

        answer_status = (
            "insufficient_evidence"
            if raw_answer == INSUFFICIENT_EVIDENCE_RESPONSE
            or any(marker in normalized_answer for marker in insufficient_markers)
            else "answered"
        )

        if answer_status == "insufficient_evidence":
            citations = []
            answer = raw_answer
        else:
            citations = build_citations(context_result.compressed)
            answer = render_answer_with_citations(raw_answer, citations)
        return {
            "question": question,
            "query_variants": [
                variant.as_dict() for variant in self.last_query_variants
            ],
            "retrieved_by_query": self.last_retrieval_by_query,
            "retrieved": results,
            "context_stages": {
                "retrieved": context_result.retrieved,
                "deduplicated": context_result.deduplicated,
                "expanded": context_result.expanded,
                "merged": context_result.merged,
                "compressed": context_result.compressed,
            },
            "stage_counts": context_result.stage_counts(),
            "context": context_result.context,
            "raw_answer": raw_answer,
            "answer": answer,
            "answer_status": answer_status,
            "citations": citations,
        }
