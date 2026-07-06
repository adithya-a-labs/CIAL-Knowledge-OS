"""Phase 4 reranking and evidence selection over Phase 3 hybrid retrieval."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Mapping
from statistics import fmean
from typing import Any

from sentence_transformers import SentenceTransformer

from .citations import build_citations, render_answer_with_citations
from .config import Phase4Config
from .context_builder import INSUFFICIENT_EVIDENCE_RESPONSE, compress_context
from .evidence_quality import EvidenceQualityScorer
from .evidence_selector import EvidenceSelectionResult, EvidenceSelector
from .execution import ExecutionManager
from .llm import GenerationFailedError, LocalLLM
from .phase3_pipeline import Phase3RAGPipeline
from .phase4_trace import build_phase4_trace
from .query_transformations import QueryTransformer
from .reranker import CrossEncoderReranker, RerankResult, Reranker
from .retrievers import Retriever
from .token_budget import Tokenizer

logger = logging.getLogger(__name__)

UNSUPPORTED_QUERY_RESPONSE = (
    "The indexed documents do not contain enough evidence to answer this "
    "question. This appears to require live/current/external data."
)

_CURRENT_DATA_MARKERS = re.compile(
    r"\b(?:now|latest|current(?:ly)?|today(?:'s)?|tomorrow(?:'s)?|live|"
    r"real[- ]?time)\b",
    re.IGNORECASE,
)
_CURRENT_DATA_DOMAINS = (
    re.compile(r"\b(?:weather|forecast)\b", re.IGNORECASE),
    re.compile(r"\b(?:share|stock)\s+price\b", re.IGNORECASE),
    re.compile(r"\bipl\b.*\b(?:score|match|fixture|result)\b", re.IGNORECASE),
    re.compile(r"\b(?:score|match|fixture|result)\b.*\bipl\b", re.IGNORECASE),
    re.compile(r"\bcafeteria\b.*\bmenu\b", re.IGNORECASE),
    re.compile(r"\bmenu\b.*\bcafeteria\b", re.IGNORECASE),
    re.compile(r"\bnetwork\s+topology\b", re.IGNORECASE),
)
_SUPPORT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "now",
    "latest",
    "current",
    "currently",
    "today",
    "tomorrow",
    "live",
}


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
        execution_manager: ExecutionManager | None = None,
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
        if execution_manager is not None:
            self.execution_manager = execution_manager
        self._configure_phase4_components()

    def _build_phase4_prompt(
        self,
        question: str,
        context: str,
        *,
        weak_evidence: bool,
    ) -> str:
        """Build detailed Phase 4 instructions over selected evidence only.

        Inputs are the question, the already-fitted selected-evidence context,
        and whether every selected chunk is below the configured reranker
        threshold. The output is a strict local-LLM prompt. Detail and structure
        are presentation requirements, not permission to add facts: the prompt
        preserves Phase 3 grounding and reference-ID rules, gives weak evidence
        an explicit caveat, and reserves safe failure for truly empty context.
        """

        effective_minimum = self.config.min_answer_words
        if (
            effective_minimum is not None
            and self.config.max_answer_words is not None
        ):
            effective_minimum = min(
                effective_minimum,
                self.config.max_answer_words,
            )
        minimum_words = (
            f"Aim for at least {effective_minimum} words when the "
            "evidence supports that depth. "
            if effective_minimum is not None
            else ""
        )
        maximum_words = (
            f"Do not exceed {self.config.max_answer_words} words. "
            "Prefer concise prioritization over dropping citations or adding "
            "unsupported compression.\n"
            if self.config.max_answer_words is not None
            else ""
        )
        if not self.config.prefer_structured_answers:
            structure = "Use a coherent, decision-oriented narrative.\n"
            content_requirements = """- Produce a comprehensive, decision-oriented synthesis rather than a simple summary of retrieved evidence.
- Explain only implications, actions, risks, gaps, dependencies, procedures, and recommendations that directly answer the question and are supported by the selected evidence.
- Expand important concepts when the retrieved evidence provides sufficient detail.
- Explain why recommendations matter, not only what should be done.
- Where multiple evidence sources support the same conclusion, synthesize them into one coherent explanation rather than listing them independently.
- Discuss operational, security, governance, implementation, or compliance implications only when directly supported by the evidence.
- Highlight dependencies, assumptions, and evidence gaps where appropriate.
- Prioritize information density over answer length.
- Avoid filler, repetition, unsupported background, speculation, and artificial padding.
"""
        elif self.config.adaptive_answer_sections:
            decision_notes_family = (
                "\n- Decision Notes"
                if self.config.include_decision_notes
                else ""
            )
            structure = f"""Choose the answer structure that best fits the question. Do not use every section by default. Use only sections that improve clarity.

Allowed section families (select only the relevant ones):
- Direct Answer
- Overview
- Key Concepts
- Key Findings
- Evidence
- Top Risks
- Business Impact
- Recommended Controls
- Recommended Actions
- Step-by-Step Procedure
- Comparison
- Key Differences
- Recommendation
- Priority Matrix
- Implementation Checklist
- Owners and Dependencies
- Operational Implications
- Evidence Gaps
- Caveats
- Next Actions
- Immediate / Next / Later Actions{decision_notes_family}

Question-shape guidance:
- If the question asks "what is", "define", or "explain", prefer Overview + Key Concepts + Evidence.
- If it asks about risks or biggest threats, prefer Top Risks + Business Impact + Recommended Controls.
- If it asks "how should we" or "what should we do", prefer Recommended Actions + Implementation Checklist + Caveats.
- If it asks to compare or distinguish differences, prefer Comparison + Key Differences + Recommendation.
- If it asks for a process, procedure, or steps, prefer Step-by-Step Procedure + Owners and Dependencies + Evidence.
- If it asks to prioritize, prefer Priority Matrix + Immediate / Next / Later Actions.
- If support is incomplete, explain Evidence Gaps instead of forcing a full report.

Use clear Markdown headings and bullets where they improve readability. Do not create tables or diagrams.
"""
            if self.config.include_decision_notes:
                structure += (
                    "Use Decision Notes only when they materially clarify a "
                    "decision, follow-up validation, or unresolved evidence gap.\n"
                )
            content_requirements = """- Produce a comprehensive, enterprise-grade synthesis rather than a collection of extracted facts.
- Cover only the findings, implications, controls, risks, procedures, comparisons, dependencies, trade-offs, and next actions that directly answer the question.
- Expand important concepts when supported by the retrieved evidence.
- Explain not only what the evidence says, but why it matters for enterprise decision-making.
- When multiple sources contribute to the same conclusion, synthesize them into a single coherent explanation instead of listing them separately.
- For every recommendation, explain:
  - what should be done
  - why it matters
  - expected operational or security benefit
  - prerequisites or dependencies when supported by the evidence
- Discuss implementation complexity, governance considerations, operational impact, compliance implications, and long-term maintenance only when supported by the evidence.
- Prioritize recommendations only when the evidence supports an ordering.
- Clearly distinguish confirmed evidence from evidence gaps.
- Do not force unrelated analysis merely to fill a section.
- Prioritize information density over answer length.
- Avoid filler, repetition, unsupported background, speculation, and artificial padding.
"""
        else:
            structure = """Use clear Markdown headings and, where supported, organize the answer as:
- Executive answer
- Evidence-backed findings
- Operational implications
- Recommended controls or actions
- Risks, gaps, and caveats
"""
            if self.config.include_decision_notes:
                structure += (
                    "- Include a short Decision notes section that distinguishes "
                    "immediate actions, follow-up validation, and unresolved "
                    "evidence gaps.\n"
                )
            content_requirements = """- Produce a comprehensive, enterprise-grade synthesis rather than a simple summary of the retrieved evidence.
- Explain operational implications and supported recommended controls or actions.
- Explain why each recommendation matters and the expected operational or security benefit when supported by the evidence.
- Identify supported risks, evidence gaps, dependencies, implementation considerations, and caveats.
- Expand important concepts when the selected evidence provides sufficient detail.
- Where multiple evidence sources support the same conclusion, synthesize them into a single coherent explanation instead of listing them independently.
- Prioritize recommendations only when the evidence supports an ordering.
- Clearly distinguish confirmed evidence from unresolved evidence gaps.
- Prioritize information density over answer length.
- Avoid filler, repetition, unsupported background, speculation, and artificial padding.
"""
        weak_rule = (
            "- All selected evidence is below the reranker threshold. State this limitation prominently, use cautious language, and recommend source verification before action.\n"
            if weak_evidence
            else ""
        )
        return f"""You are a strict grounded-answering system producing a {self.config.answer_detail_level} decision-support answer.

Answer the QUESTION using only the provided SELECTED EVIDENCE.

Grounding rules:
1. Use only facts directly supported by SELECTED EVIDENCE.
2. Do not use outside knowledge, invent controls, or infer unsupported organization-specific details.
3. Cite every key factual claim and recommendation inline using the exact reference IDs shown in the evidence, such as [1].
4. Do not invent, alter, or renumber reference IDs.
5. If evidence supports only part of the question, answer that part and identify the remaining gap.
6. Reply exactly "{INSUFFICIENT_EVIDENCE_RESPONSE}" only when SELECTED EVIDENCE is empty or contains no usable information.

Answer requirements:
- Produce a {self.config.answer_detail_level} synthesis from the selected evidence.
- Produce a comprehensive, enterprise-grade synthesis that balances depth with clarity.
- Think like an experienced enterprise consultant preparing advice for a technical decision-maker.
- Do not merely summarize retrieved passages; interpret, connect, and synthesize them into a coherent explanation.
- Expand important concepts only when supported by the retrieved evidence.
- Explain relationships between findings instead of presenting isolated facts.
- Prioritize information density over answer length.
- Every recommendation must remain fully grounded in the selected evidence.
{content_requirements}{weak_rule}{minimum_words}{maximum_words}
{structure}
SELECTED EVIDENCE
{context}

QUESTION
{question}

ANSWER
"""

    def _generate_grounded_answer(self, question: str, context: str) -> str:
        """Generate a detailed Phase 4 synthesis without expanding context.

        The inputs and raw string output match the inherited private generation
        hook. Only the prompt style changes: the local model sees the same final
        selected evidence that Phase 3 context construction fitted. Phase 1--3
        continue using their established concise prompt.
        """

        if self.llm is None:
            raise RuntimeError("Initialize the local LLM before generation.")
        selection = self.last_selection_result
        prompt = self._build_phase4_prompt(
            question,
            context,
            weak_evidence=bool(selection and selection.weak_evidence),
        )
        total_attempts = self.config.generation_retries + 1
        last_error: Exception | None = None
        generation_started = time.perf_counter()
        self.execution_manager.start_stage(
            "generation",
            event_type="generation_started",
            model=self.config.ollama_model_name,
        )
        for attempt in range(1, total_attempts + 1):
            try:
                answer = str(self.llm.invoke(prompt)).strip()
                self.metrics["generation_attempts"] = float(attempt)
                self.metrics["generation_retry_count"] = float(attempt - 1)
                self.execution_manager.complete_stage(
                    "generation",
                    event_type="generation_completed",
                    metrics={"retry_count": attempt - 1},
                    model=self.config.ollama_model_name,
                )
                return answer
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable_generation_error(exc)
                logger.exception(
                    "phase4_generation_attempt_failed",
                    extra={
                        "event": "generation_retry",
                        "attempt": attempt,
                        "total_attempts": total_attempts,
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                    },
                )
                if not retryable or attempt >= total_attempts:
                    self.metrics["generation_attempts"] = float(attempt)
                    self.metrics["generation_retry_count"] = float(attempt - 1)
                    self.execution_manager.emit(
                        "generation_failed",
                        stage="generation",
                        status="failed",
                        error=str(exc),
                        elapsed_seconds=time.perf_counter() - generation_started,
                        metrics={"retry_count": attempt - 1},
                        payload={"model": self.config.ollama_model_name},
                        source="phase4_pipeline",
                    )
                    raise GenerationFailedError(
                        exc,
                        attempts=attempt,
                    ) from exc
                next_attempt = attempt + 1
                message = (
                    "Generation failed; retrying attempt "
                    f"{next_attempt}/{total_attempts} after cooldown."
                )
                print(message)
                logger.warning(
                    message,
                    extra={
                        "event": "generation_retry",
                        "next_attempt": next_attempt,
                        "total_attempts": total_attempts,
                        "cooldown_seconds": (
                            self.config.retry_cooldown_seconds
                        ),
                    },
                )
                if self.config.retry_cooldown_seconds:
                    time.sleep(self.config.retry_cooldown_seconds)
        assert last_error is not None
        self.execution_manager.emit(
            "generation_failed",
            stage="generation",
            status="failed",
            error=str(last_error),
            elapsed_seconds=time.perf_counter() - generation_started,
            metrics={"retry_count": total_attempts - 1},
            source="phase4_pipeline",
        )
        raise GenerationFailedError(last_error, attempts=total_attempts)

    @staticmethod
    def _is_retryable_generation_error(error: Exception) -> bool:
        """Return whether a local generation failure is safe to retry.

        The input is the exception raised by the LLM adapter. The boolean output
        recognizes Ollama runner/stream/server failures plus common local
        transport exceptions. Retrieval and reranking are outside this method,
        so retrying never repeats those expensive stages.
        """

        message = f"{type(error).__name__}: {error}".casefold()
        retryable_markers = (
            "model runner has unexpectedly stopped",
            "model runner stopped",
            "no data received from ollama stream",
            "status code: 500",
            "status code 500",
            "responseerror",
            "std::bad_alloc",
            "connection reset",
            "connection refused",
            "connection aborted",
            "timed out",
            "timeout",
            "stream",
        )
        return isinstance(
            error,
            (ConnectionError, OSError, TimeoutError),
        ) or any(marker in message for marker in retryable_markers)

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

    def _fallback_evidence_is_sufficient(
        self,
        *,
        evidence_confidence: str,
        final_chunks: list[dict[str, Any]],
    ) -> bool:
        """Return whether selected evidence may support an extractive fallback."""

        if not final_chunks:
            return False
        selected = self.last_selected_chunks
        if not selected:
            return False
        scores = [
            float(item.get("reranker_score") or 0.0)
            for item in selected
        ]
        minimum = float(self.config.min_fallback_reranker_score)
        score_passed = max(scores) >= minimum and fmean(scores) >= minimum
        all_selection_fallback = all(
            str(item.get("selection_reason") or "") == "adaptive_fallback"
            for item in selected
        )
        all_weak = all(bool(item.get("weak_evidence")) for item in selected)
        weak_override = (
            self.config.allow_extractive_fallback_for_weak_evidence
            and evidence_confidence == "weak"
        )
        confidence_passed = (
            evidence_confidence in {"strong", "mixed"} or weak_override
        )
        selection_passed = (
            not all_selection_fallback and not all_weak
        ) or weak_override
        return score_passed and confidence_passed and selection_passed

    @staticmethod
    def _requires_current_external_data(question: str) -> bool:
        normalized = " ".join(str(question).split())
        return bool(_CURRENT_DATA_MARKERS.search(normalized)) or any(
            pattern.search(normalized) for pattern in _CURRENT_DATA_DOMAINS
        )

    @staticmethod
    def _current_question_has_direct_support(
        question: str,
        selected_chunks: list[dict[str, Any]],
        *,
        evidence_sufficient: bool,
    ) -> bool:
        """Conservatively identify direct indexed support for a current query."""

        if not evidence_sufficient or not selected_chunks:
            return False
        evidence = " ".join(
            str(item.get("text") or item.get("page_content") or "")
            for item in selected_chunks
        ).casefold()
        question_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", question.casefold())
            if len(token) > 1 and token not in _SUPPORT_STOP_WORDS
        }
        if not question_terms:
            return False
        overlap = sum(term in evidence for term in question_terms)
        required_overlap = min(2, len(question_terms))
        temporal_or_domain_support = bool(
            _CURRENT_DATA_MARKERS.search(evidence)
        ) or any(pattern.search(evidence) for pattern in _CURRENT_DATA_DOMAINS)
        return overlap >= required_overlap and temporal_or_domain_support

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve, rerank, and select evidence for one question.

        Phase 3 performs query transformation, dense/BM25 retrieval, RRF, and
        cross-query deduplication first. The candidate pool is then capped,
        reranked (or passed through when disabled), and filtered by the
        evidence selector. The returned list and ``last_merged_retrieval`` are
        the selected chunks consumed by the inherited context builder.
        """

        self._configure_phase4_components()
        self.execution_manager.start_stage(
            "retrieval", event_type="retrieval_started"
        )
        phase3_candidates = super().retrieve(question)
        self.execution_manager.complete_stage(
            "retrieval",
            event_type="retrieval_completed",
            metrics={
                "retrieval_latency_seconds": self.metrics.get(
                    "retrieval_latency", 0.0
                )
            },
            candidate_count=len(phase3_candidates),
        )
        self.last_candidate_pool = [
            dict(item)
            for item in phase3_candidates[: self.config.reranker_candidate_top_k]
        ]

        # RRF is rank-based by design. Applying the cross-encoder here avoids
        # pretending cosine, BM25, and RRF values can be directly averaged.
        self.execution_manager.start_stage(
            "reranking", event_type="reranking_started"
        )
        rerank_result = (
            self.reranker.rerank(question, self.last_candidate_pool)
            if self.config.reranker_enabled
            else self._passthrough_rerank(self.last_candidate_pool)
        )
        self.metrics["reranker_latency"] = rerank_result.latency_seconds
        self.last_reranked_candidates = [
            dict(item) for item in rerank_result.candidates
        ]
        self.execution_manager.complete_stage(
            "reranking",
            event_type="reranking_completed",
            metrics={
                "reranking_latency_seconds": rerank_result.latency_seconds
            },
            candidate_count=len(self.last_reranked_candidates),
        )

        self.execution_manager.start_stage(
            "evidence_selection",
            event_type="evidence_selection_started",
        )
        selection = self.evidence_selector.select(
            self.last_reranked_candidates
        )
        self.last_selection_result = selection
        self.metrics["evidence_selection_latency"] = selection.latency_seconds
        self.last_selected_chunks = [dict(item) for item in selection.selected]
        self.last_discarded_chunks = [dict(item) for item in selection.discarded]
        self.execution_manager.complete_stage(
            "evidence_selection",
            event_type="evidence_selection_completed",
            metrics={
                "evidence_selection_latency_seconds": selection.latency_seconds
            },
            selected_count=len(self.last_selected_chunks),
            discarded_count=len(self.last_discarded_chunks),
        )

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
        quality = self.evidence_quality_scorer.score(self.last_selected_chunks)
        context = str(response.get("context") or "")
        response["prompt"] = (
            self._build_phase4_prompt(
                question,
                context,
                weak_evidence=weak_evidence,
            )
            if context
            else ""
        )
        final_chunks = (
            response.get("context_stages", {}).get("compressed", [])
            if isinstance(response.get("context_stages"), Mapping)
            else []
        )
        final_chunks = [
            dict(item) for item in final_chunks if isinstance(item, Mapping)
        ]
        fallback_evidence_sufficient = self._fallback_evidence_is_sufficient(
            evidence_confidence=evidence_confidence,
            final_chunks=final_chunks,
        )
        current_data_query = (
            self.config.unsupported_query_detection_enabled
            and self._requires_current_external_data(question)
        )
        unsupported_query = current_data_query and not (
            self._current_question_has_direct_support(
                question,
                self.last_selected_chunks,
                evidence_sufficient=fallback_evidence_sufficient,
            )
        )
        fallback_candidate = bool(
            final_chunks and response.get("answer_status") != "answered"
        )
        extractive_fallback_used = False
        fallback_blocked = False

        if unsupported_query:
            fallback_blocked = fallback_candidate
            response["raw_answer"] = UNSUPPORTED_QUERY_RESPONSE
            response["answer"] = UNSUPPORTED_QUERY_RESPONSE
            response["answer_status"] = "unsupported_query"
            response["citations"] = []
        elif (
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
        if (
            not unsupported_query
            and fallback_candidate
            and fallback_evidence_sufficient
            and response.get("answer_status") != "answered"
        ):
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
            extractive_fallback_used = True
        elif (
            not unsupported_query
            and fallback_candidate
            and response.get("answer_status") != "answered"
        ):
            fallback_blocked = True
            response["raw_answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
            response["answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
            response["answer_status"] = "insufficient_evidence"
            response["citations"] = []
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
            "extractive_fallback_used": extractive_fallback_used,
            "fallback_blocked": fallback_blocked,
            "unsupported_query_detected": unsupported_query,
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
        # Phase 3 builds its trace before this subclass restores the Phase 4
        # prompt. Refresh generation-only diagnostics so prompt/answer token
        # counts describe what the model actually received and returned.
        generation = phase3_trace.get("generation")
        generation = dict(generation) if isinstance(generation, Mapping) else {}
        generation.update(
            {
                "prompt_tokens": self.token_manager.count(
                    str(response.get("prompt") or "")
                ),
                "answer_tokens": self.token_manager.count(
                    str(response.get("raw_answer") or response.get("answer") or "")
                ),
                "status": response.get("answer_status"),
            }
        )
        phase3_trace["generation"] = generation
        phase3_trace["answer"] = str(response.get("answer") or "")
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
                "extractive_fallback_used": extractive_fallback_used,
                "fallback_blocked": fallback_blocked,
                "unsupported_query_detected": unsupported_query,
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
                "unsupported_query_count": float(
                    self.metrics.get("unsupported_query_count") or 0.0
                )
                + float(response.get("answer_status") == "unsupported_query"),
                "insufficient_evidence_count": float(
                    self.metrics.get("insufficient_evidence_count") or 0.0
                )
                + float(
                    response.get("answer_status") == "insufficient_evidence"
                ),
                "extractive_fallback_count": float(
                    self.metrics.get("extractive_fallback_count") or 0.0
                )
                + float(extractive_fallback_used),
                "fallback_blocked_count": float(
                    self.metrics.get("fallback_blocked_count") or 0.0
                )
                + float(fallback_blocked),
            }
        )
        return response
