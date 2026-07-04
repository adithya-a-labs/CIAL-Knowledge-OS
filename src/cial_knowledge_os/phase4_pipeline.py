"""Phase 4 reranking and evidence selection over Phase 3 hybrid retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from sentence_transformers import SentenceTransformer

from .citations import build_citations, render_answer_with_citations
from .config import Phase4Config
from .context_builder import INSUFFICIENT_EVIDENCE_RESPONSE, compress_context
from .evidence_quality import EvidenceQualityScorer
from .evidence_selector import EvidenceSelectionResult, EvidenceSelector
from .llm import LocalLLM
from .phase3_pipeline import Phase3RAGPipeline
from .phase4_trace import build_phase4_trace
from .query_transformations import QueryTransformer
from .reranker import CrossEncoderReranker, RerankResult, Reranker
from .retrievers import Retriever
from .token_budget import Tokenizer


class Phase4RAGPipeline(Phase3RAGPipeline):
    """Improve Phase 3 precision before generation while preserving its API.

    Inputs are a :class:`Phase4Config` and optional injected embedding, LLM,
    query transformer, tokenizer, retrievers, reranker, selector, and quality
    scorer. The output of :meth:`answer` retains all Phase 2/3 response keys and
    adds candidate, reranking, selection, quality, token-reduction, and trace
    data.

    Reranking happens after RRF because dense similarity and BM25 scores occupy
    incompatible scales; RRF first combines rank evidence without averaging raw
    scores, then a cross-encoder evaluates question/chunk pairs on one scoring
    surface. Selected evidence is passed into the existing token-aware context
    builder, citation engine, and grounded LLM. This additive extension keeps
    Phase 1--3 classes, notebooks, configuration fields, and exports valid.
    """

    config: Phase4Config

    def __init__(
        self,
        config: Phase4Config | None = None,
        *,
        embedding_model: SentenceTransformer | None = None,
        llm: LocalLLM | None = None,
        query_transformer: QueryTransformer | None = None,
        tokenizer: Tokenizer | None = None,
        retrievers: Mapping[str, Retriever] | None = None,
        reranker: Reranker | None = None,
        evidence_selector: EvidenceSelector | None = None,
        evidence_quality_scorer: EvidenceQualityScorer | None = None,
    ) -> None:
        phase4_config = config or Phase4Config()
        self.reranker = reranker or CrossEncoderReranker(
            phase4_config.reranker_model_name,
            device=phase4_config.reranker_device,
            batch_size=phase4_config.reranker_batch_size,
            local_files_only=phase4_config.reranker_local_files_only,
        )
        self._injected_evidence_selector = evidence_selector
        self._injected_quality_scorer = evidence_quality_scorer
        self.evidence_selector: EvidenceSelector
        self.evidence_quality_scorer: EvidenceQualityScorer
        self.last_candidate_pool: list[dict[str, Any]] = []
        self.last_reranked_candidates: list[dict[str, Any]] = []
        self.last_selected_chunks: list[dict[str, Any]] = []
        self.last_discarded_chunks: list[dict[str, Any]] = []
        self.last_selection_result: EvidenceSelectionResult | None = None
        self._phase4_component_key: tuple[Any, ...] | None = None
        super().__init__(
            config=phase4_config,
            embedding_model=embedding_model,
            llm=llm,
            query_transformer=query_transformer,
            tokenizer=tokenizer,
            retrievers=retrievers,
        )
        self._configure_phase4_components()

    def _configure_phase4_components(self) -> None:
        key = (
            id(self.token_manager),
            self.config.evidence_selection_strategies,
            self.config.min_selected_evidence,
            self.config.max_selected_evidence,
            self.config.reranker_score_threshold,
            self.config.fallback_to_top_n_if_empty,
            self.config.fallback_top_n,
            self.config.evidence_token_budget,
            self.config.selected_evidence_target_min_tokens,
            self.config.selected_evidence_target_max_tokens,
            self.config.evidence_max_chunks_per_source,
            self.config.evidence_redundancy_threshold,
            self.config.evidence_strong_threshold,
            self.config.evidence_medium_threshold,
        )
        if self._phase4_component_key == key:
            return
        self.evidence_selector = (
            self._injected_evidence_selector
            or EvidenceSelector(
                self.token_manager,
                strategies=self.config.evidence_selection_strategies,
                min_selected_evidence=self.config.min_selected_evidence,
                max_selected_evidence=self.config.max_selected_evidence,
                score_threshold=self.config.reranker_score_threshold,
                token_budget=self.config.evidence_token_budget,
                max_chunks_per_source=self.config.evidence_max_chunks_per_source,
                redundancy_threshold=self.config.evidence_redundancy_threshold,
                fallback_to_top_n_if_empty=(
                    self.config.fallback_to_top_n_if_empty
                ),
                fallback_top_n=self.config.fallback_top_n,
                target_min_tokens=(
                    self.config.selected_evidence_target_min_tokens
                ),
                target_max_tokens=(
                    self.config.selected_evidence_target_max_tokens
                ),
            )
        )
        self.evidence_quality_scorer = (
            self._injected_quality_scorer
            or EvidenceQualityScorer(
                strong_threshold=self.config.evidence_strong_threshold,
                medium_threshold=self.config.evidence_medium_threshold,
                link_resolver=self.citation_link_builder,
            )
        )
        self._phase4_component_key = key

    def on_config_changed(self) -> None:
        """Refresh Phase 3 and Phase 4 components after safe config sweeps.

        The method accepts the same no-argument contract used by the existing
        evaluation runner. Injected test doubles remain in place; default
        selectors and quality scorers are rebuilt only when relevant settings
        change.
        """

        super().on_config_changed()
        self._phase4_component_key = None
        self._configure_phase4_components()

    def _passthrough_rerank(
        self,
        candidates: list[dict[str, Any]],
    ) -> RerankResult:
        enriched = []
        for rank, value in enumerate(candidates, start=1):
            candidate = dict(value)
            candidate["original_rrf_rank"] = rank
            candidate["reranked_rank"] = rank
            candidate["reranker_score"] = float(
                candidate.get("rrf_score")
                or candidate.get("score")
                or 0.0
            )
            enriched.append(candidate)
        return RerankResult(tuple(enriched), 0.0, "disabled")

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve, rerank, and select evidence for one question.

        Phase 3 performs query transformation, dense/BM25 retrieval, RRF, and
        cross-query deduplication first. The candidate pool is then capped,
        reranked (or passed through when disabled), and filtered by the
        evidence selector. The returned list and ``last_merged_retrieval`` are
        the selected chunks consumed by the inherited context builder.
        """

        self._configure_phase4_components()
        phase3_candidates = super().retrieve(question)
        self.last_candidate_pool = [
            dict(item)
            for item in phase3_candidates[: self.config.reranker_candidate_top_k]
        ]

        # RRF is rank-based by design. Applying the cross-encoder here avoids
        # pretending cosine, BM25, and RRF values can be directly averaged.
        rerank_result = (
            self.reranker.rerank(question, self.last_candidate_pool)
            if self.config.reranker_enabled
            else self._passthrough_rerank(self.last_candidate_pool)
        )
        self.metrics["reranker_latency"] = rerank_result.latency_seconds
        self.last_reranked_candidates = [
            dict(item) for item in rerank_result.candidates
        ]

        selection = self.evidence_selector.select(
            self.last_reranked_candidates
        )
        self.last_selection_result = selection
        self.metrics["evidence_selection_latency"] = selection.latency_seconds
        self.last_selected_chunks = [dict(item) for item in selection.selected]
        self.last_discarded_chunks = [dict(item) for item in selection.discarded]

        # Phase 2/3 context construction stays unchanged; replacing this
        # internal hand-off is what preserves their public response contract.
        self.last_merged_retrieval = [
            dict(item) for item in self.last_selected_chunks
        ]
        return [dict(item) for item in self.last_selected_chunks]

    def answer(self, question: str) -> dict[str, Any]:
        """Run Phase 4 end to end and return a Phase 3-compatible response.

        The inherited pipeline performs local generation, citations, exact token
        fitting, and the Phase 3 trace. This method adds evidence quality,
        candidate-to-context token reduction, discard reasons, latency stages,
        and a serializable Phase 4 trace. No cloud service is called.
        """

        response = super().answer(question)
        selection_result = self.last_selection_result
        weak_evidence = bool(
            selection_result and selection_result.weak_evidence
        )
        mixed_confidence = bool(
            self.last_selected_chunks
            and any(item.get("weak_evidence") for item in self.last_selected_chunks)
            and not weak_evidence
        )
        evidence_confidence = (
            "none"
            if not self.last_selected_chunks
            else "weak"
            if weak_evidence
            else "mixed"
            if mixed_confidence
            else "strong"
        )
        if (
            weak_evidence
            and self.config.weak_evidence_answer_allowed
            and response.get("answer_status") == "answered"
        ):
            response["answer"] = (
                "**Caution — low-confidence evidence:** The reranker found "
                "usable context, but all selected chunks were below the "
                "configured score threshold. Verify the cited sources before "
                "acting.\n\n"
                + str(response.get("answer") or "")
            )
        elif (
            weak_evidence
            and not self.config.weak_evidence_answer_allowed
        ):
            response["raw_answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
            response["answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
            response["answer_status"] = "insufficient_evidence"
            response["citations"] = []
        phase3_trace_value = response.get("question_trace")
        phase3_trace = (
            dict(phase3_trace_value)
            if isinstance(phase3_trace_value, Mapping)
            else {}
        )
        final_chunks = (
            response.get("context_stages", {}).get("compressed", [])
            if isinstance(response.get("context_stages"), Mapping)
            else []
        )
        if (
            final_chunks
            and (
                not weak_evidence
                or self.config.weak_evidence_answer_allowed
            )
            and response.get("answer_status") != "answered"
        ):
            # A reranker score is a confidence signal, not proof that the
            # retrieved text is unusable. Preserve answerability with a clearly
            # labeled extractive fallback instead of converting non-empty
            # evidence into "no evidence."
            cautious_citations = build_citations(
                final_chunks,
                link_resolver=self.citation_link_builder,
            )
            excerpts = []
            for index, item in enumerate(final_chunks[:3], start=1):
                text = " ".join(str(item.get("text") or "").split())
                excerpts.append(f"- [{index}] {text[:500]}")
            cautious_raw_answer = (
                "**Caution — evidence review required:** The selected passages "
                "are usable, but the local generator did not produce a "
                "confident synthesis. Review these grounded excerpts before "
                "acting.\n\n"
                + "\n".join(excerpts)
            )
            response["raw_answer"] = cautious_raw_answer
            response["answer"] = render_answer_with_citations(
                cautious_raw_answer,
                cautious_citations,
            )
            response["answer_status"] = "answered"
            response["citations"] = cautious_citations
        quality = self.evidence_quality_scorer.score(self.last_selected_chunks)
        # Compare serialized context blocks rather than raw text alone. Phase 3
        # context includes citation headers, and omitting that overhead would
        # understate the tokens Phase 4 avoids.
        _, candidate_context = compress_context(
            self.last_candidate_pool,
            max_chars=1,
            enabled=False,
        )
        candidate_tokens = self.token_manager.count(candidate_context)
        selected_tokens = sum(
            int(
                item.get("evidence_token_count")
                or self.token_manager.count(str(item.get("text") or ""))
            )
            for item in self.last_selected_chunks
        )
        final_context_tokens = self.token_manager.count(
            str(response.get("context") or "")
        )
        token_reduction_percent = (
            round(
                100.0
                * max(0, candidate_tokens - final_context_tokens)
                / candidate_tokens,
                2,
            )
            if candidate_tokens
            else 0.0
        )
        discard_reasons = Counter(
            str(item.get("discard_reason") or "unspecified")
            for item in self.last_discarded_chunks
        )
        token_efficiency = {
            "candidate_tokens": candidate_tokens,
            "selected_evidence_tokens": selected_tokens,
            "final_context_tokens": final_context_tokens,
            "token_reduction_percent": token_reduction_percent,
            "candidate_chunk_count": len(self.last_candidate_pool),
            "selected_chunk_count": len(self.last_selected_chunks),
            "discarded_chunk_count": len(self.last_discarded_chunks),
            "chunks_discarded": len(self.last_discarded_chunks),
            "discard_reason_distribution": dict(sorted(discard_reasons.items())),
            "usable_candidate_count": (
                selection_result.usable_candidate_count
                if selection_result is not None
                else len(self.last_candidate_pool)
            ),
            "threshold_pass_count": (
                selection_result.threshold_pass_count
                if selection_result is not None
                else 0
            ),
            "fallback_used": bool(
                selection_result and selection_result.fallback_used
            ),
            "weak_evidence": weak_evidence,
            "evidence_confidence": evidence_confidence,
        }
        latency = {
            "retrieval_seconds": float(
                self.metrics.get("retrieval_latency") or 0.0
            ),
            "reranking_seconds": float(
                self.metrics.get("reranker_latency") or 0.0
            ),
            "evidence_selection_seconds": float(
                self.metrics.get("evidence_selection_latency") or 0.0
            ),
            "context_construction_seconds": float(
                self.metrics.get("context_construction_latency") or 0.0
            ),
            "generation_seconds": float(
                self.metrics.get("generation_latency") or 0.0
            ),
            "total_pipeline_seconds": float(
                self.metrics.get("total_pipeline_latency") or 0.0
            ),
            "artifact_export_seconds": None,
        }
        evidence_quality = {
            "chunks": list(quality.chunks),
            "summary": quality.summary,
        }
        trace = build_phase4_trace(
            question=question,
            phase3_trace=phase3_trace,
            candidate_pool=self.last_candidate_pool,
            reranked_candidates=self.last_reranked_candidates,
            selected_chunks=self.last_selected_chunks,
            discarded_chunks=self.last_discarded_chunks,
            final_context_chunks=final_chunks,
            evidence_quality=evidence_quality,
            token_usage=token_efficiency,
            latency=latency,
            citations=response.get("citations") or [],
            answer=str(response.get("answer") or response.get("raw_answer") or ""),
            answer_status=str(response.get("answer_status") or ""),
            trace_mode=self.config.phase4_trace_mode,
            medium_score_threshold=self.config.evidence_medium_threshold,
        )
        trace_payload = trace.to_dict()
        reranker_load_source = str(
            getattr(self.reranker, "load_source", None) or "unknown"
        )
        trace_payload["reranker"] = {
            "model_name": getattr(
                self.reranker,
                "model_name",
                self.config.reranker_model_name,
            ),
            "load_source": reranker_load_source,
            "local_files_only": bool(
                getattr(
                    self.reranker,
                    "local_files_only",
                    self.config.reranker_local_files_only,
                )
            ),
        }
        response.update(
            {
                "candidate_pool": [dict(item) for item in self.last_candidate_pool],
                "reranked_candidates": [
                    dict(item) for item in self.last_reranked_candidates
                ],
                "selected_evidence": [
                    dict(item) for item in self.last_selected_chunks
                ],
                "discarded_evidence": [
                    dict(item) for item in self.last_discarded_chunks
                ],
                "evidence_quality": evidence_quality,
                "token_efficiency": token_efficiency,
                "evidence_confidence": evidence_confidence,
                "weak_evidence": weak_evidence,
                "reranker_load_source": reranker_load_source,
                "phase3_question_trace": phase3_trace,
                "question_trace": trace_payload,
            }
        )
        response["retrieval_trace"] = {
            **dict(response.get("retrieval_trace") or {}),
            "candidate_count": len(self.last_candidate_pool),
            "reranked_count": len(self.last_reranked_candidates),
            "selected_count": len(self.last_selected_chunks),
            "discarded_count": len(self.last_discarded_chunks),
            "token_reduction_percent": token_reduction_percent,
        }
        self.metrics.update(
            {
                "candidate_tokens": float(candidate_tokens),
                "selected_evidence_tokens": float(selected_tokens),
                "context_tokens": float(final_context_tokens),
                "token_reduction_percent": token_reduction_percent,
                "selected_chunk_count": float(len(self.last_selected_chunks)),
                "discarded_chunk_count": float(len(self.last_discarded_chunks)),
            }
        )
        return response
