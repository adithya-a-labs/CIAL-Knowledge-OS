"""Explainable keep/discard decisions over reranked Phase 4 evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .metadata import chunk_identity
from .token_budget import TokenManager

_WORD_PATTERN = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)


def _source(candidate: Mapping[str, Any]) -> str:
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        metadata.get("source")
        or candidate.get("source")
        or metadata.get("file_name")
        or "unknown"
    )


def _terms(text: str) -> set[str]:
    return {value.casefold() for value in _WORD_PATTERN.findall(text)}


def _jaccard(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 1.0


@dataclass(frozen=True, slots=True)
class EvidenceSelectionResult:
    """Capture selected and discarded chunks plus measurable decision costs.

    ``selected`` preserves reranker order and records whether each chunk passed
    the threshold or entered through adaptive fallback. ``discarded`` records
    one normalized primary ``discard_reason`` per rejected candidate. Token
    counts use the injected Phase 3 token manager, so selection and final
    context accounting share one tokenizer.
    """

    selected: tuple[dict[str, Any], ...]
    discarded: tuple[dict[str, Any], ...]
    selected_tokens: int
    latency_seconds: float
    fallback_used: bool = False
    weak_evidence: bool = False
    usable_candidate_count: int = 0
    threshold_pass_count: int = 0
    discard_reason_counts: dict[str, int] | None = None


class EvidenceSelector:
    """Choose the smallest strong evidence set before context construction.

    Inputs are reranked candidate dictionaries and a shared ``TokenManager``.
    Enabled strategies can enforce score threshold, exact deduplication,
    lexical redundancy reduction, source concentration limits, a total evidence
    token budget, and maximum evidence count. The output explains every keep or
    discard decision.

    Selection occurs after RRF and reranking: RRF combines ranks from
    incomparable dense/BM25 score spaces, while the cross-encoder evaluates each
    question/chunk pair. Raw retriever scores are intentionally never averaged.
    Thresholding is advisory when it would starve a non-empty candidate pool;
    the configured evidence floor and token target preserve answerability.
    Phase 3 context construction remains reusable and receives only the
    selected chunks.
    """

    def __init__(
        self,
        token_manager: TokenManager,
        *,
        strategies: Sequence[str],
        max_chunks: int | None = None,
        min_selected_evidence: int = 0,
        max_selected_evidence: int | None = None,
        score_threshold: float,
        token_budget: int,
        max_chunks_per_source: int,
        redundancy_threshold: float,
        fallback_to_top_n_if_empty: bool = True,
        fallback_top_n: int = 3,
        target_min_tokens: int = 0,
        target_max_tokens: int | None = None,
    ) -> None:
        effective_max = (
            max_selected_evidence
            if max_selected_evidence is not None
            else max_chunks
        )
        if effective_max is None or effective_max <= 0:
            raise ValueError(
                "max_selected_evidence or max_chunks must be greater than zero."
            )
        if min_selected_evidence < 0:
            raise ValueError("min_selected_evidence must be non-negative.")
        if min_selected_evidence > effective_max:
            raise ValueError(
                "min_selected_evidence must not exceed max_selected_evidence."
            )
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero.")
        if max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be greater than zero.")
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between zero and one.")
        if fallback_top_n <= 0:
            raise ValueError("fallback_top_n must be greater than zero.")
        if target_min_tokens < 0:
            raise ValueError("target_min_tokens must be non-negative.")
        if target_max_tokens is not None and target_max_tokens <= 0:
            raise ValueError("target_max_tokens must be greater than zero.")
        if (
            target_max_tokens is not None
            and target_min_tokens > target_max_tokens
        ):
            raise ValueError(
                "target_min_tokens must not exceed target_max_tokens."
            )
        if target_max_tokens is not None and target_max_tokens > token_budget:
            raise ValueError(
                "target_max_tokens must not exceed the evidence token budget."
            )
        self.token_manager = token_manager
        self.strategies = frozenset(strategies)
        self.min_selected_evidence = min_selected_evidence
        self.max_selected_evidence = effective_max
        # Keep the old attribute for callers that inspect it directly.
        self.max_chunks = effective_max
        self.score_threshold = float(score_threshold)
        self.token_budget = token_budget
        self.max_chunks_per_source = max_chunks_per_source
        self.redundancy_threshold = redundancy_threshold
        self.fallback_to_top_n_if_empty = fallback_to_top_n_if_empty
        self.fallback_top_n = fallback_top_n
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens or token_budget

    def select(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> EvidenceSelectionResult:
        """Return explainable keep/discard decisions for ranked candidates.

        Each input is copied and annotated with ``selected``,
        ``discard_reason``, ``selection_reason``, confidence, and exact token
        count. Threshold-qualified chunks are preferred; adaptive fallback then
        satisfies the evidence floor and normal-QA token target when usable
        candidates remain. Zero selection is reserved for an empty or entirely
        invalid candidate pool, preserving Phase 3 safe-failure behavior.
        """

        started = perf_counter()
        selected: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        usable: list[dict[str, Any]] = []
        identities: set[tuple[Any, ...]] = set()
        source_counts: Counter[str] = Counter()
        used_tokens = 0

        for value in candidates:
            candidate = dict(value)
            text = str(
                candidate.get("text")
                or candidate.get("page_content")
                or ""
            ).strip()
            token_count = self.token_manager.count(text) if text else 0
            candidate["text"] = text
            candidate["evidence_token_count"] = token_count
            try:
                candidate["reranker_score"] = float(
                    candidate.get("reranker_score")
                )
            except (TypeError, ValueError):
                candidate["reranker_score"] = 0.0
            if not text or token_count <= 0:
                candidate["selected"] = False
                candidate["discard_reason"] = "empty_text"
                invalid.append(candidate)
            else:
                usable.append(candidate)

        threshold_enabled = "reranker_score_threshold" in self.strategies
        threshold_pass_count = sum(
            not threshold_enabled
            or float(candidate["reranker_score"]) >= self.score_threshold
            for candidate in usable
        )
        required_floor = min(
            self.min_selected_evidence,
            len(usable),
            self.max_selected_evidence,
        )
        if (
            threshold_pass_count == 0
            and usable
            and self.fallback_to_top_n_if_empty
        ):
            required_floor = min(
                max(required_floor, self.fallback_top_n),
                len(usable),
                self.max_selected_evidence,
            )

        def is_redundant(candidate: Mapping[str, Any]) -> bool:
            if chunk_identity(candidate) in identities:
                return True
            if "redundancy_reduction" not in self.strategies:
                return False
            text = str(candidate.get("text") or "")
            return any(
                _jaccard(text, str(existing.get("text") or ""))
                >= self.redundancy_threshold
                for existing in selected
            )

        def add_candidate(
            candidate: dict[str, Any],
            *,
            selection_reason: str,
        ) -> None:
            nonlocal used_tokens
            candidate["selected"] = True
            candidate["discard_reason"] = None
            candidate["selection_reason"] = selection_reason
            candidate["weak_evidence"] = (
                threshold_enabled
                and float(candidate["reranker_score"]) < self.score_threshold
            )
            selected.append(candidate)
            identities.add(chunk_identity(candidate))
            source_counts[_source(candidate)] += 1
            used_tokens += int(candidate["evidence_token_count"])

        # Pass 1 keeps threshold-qualified evidence while aiming for a useful
        # token range. The threshold ranks confidence; it is not allowed to
        # erase every otherwise usable candidate.
        for candidate in usable:
            if len(selected) >= self.max_selected_evidence:
                break
            if (
                threshold_enabled
                and float(candidate["reranker_score"]) < self.score_threshold
            ):
                continue
            if is_redundant(candidate):
                continue
            if (
                "source_diversity" in self.strategies
                and source_counts[_source(candidate)]
                >= self.max_chunks_per_source
            ):
                continue
            token_count = int(candidate["evidence_token_count"])
            if (
                "token_budget" in self.strategies
                and used_tokens + token_count > self.token_budget
            ):
                continue
            if (
                self.target_min_tokens > 0
                and
                len(selected) >= required_floor
                and used_tokens >= self.target_min_tokens
            ):
                break
            if (
                len(selected) >= required_floor
                and used_tokens + token_count > self.target_max_tokens
            ):
                continue
            add_candidate(candidate, selection_reason="threshold_pass")

        fallback_used = False

        def fill_from_remaining(*, relax_structure: bool) -> None:
            nonlocal fallback_used
            for candidate in usable:
                if any(candidate is item for item in selected):
                    continue
                if len(selected) >= self.max_selected_evidence:
                    return
                need_floor = len(selected) < required_floor
                need_tokens = used_tokens < self.target_min_tokens
                if not need_floor and not need_tokens:
                    return
                if not relax_structure:
                    if is_redundant(candidate):
                        continue
                    if (
                        "source_diversity" in self.strategies
                        and source_counts[_source(candidate)]
                        >= self.max_chunks_per_source
                    ):
                        continue
                token_count = int(candidate["evidence_token_count"])
                if (
                    "token_budget" in self.strategies
                    and used_tokens + token_count > self.token_budget
                    and not need_floor
                ):
                    continue
                if (
                    not need_floor
                    and used_tokens + token_count > self.target_max_tokens
                ):
                    continue
                add_candidate(candidate, selection_reason="adaptive_fallback")
                fallback_used = True

        # Prefer diverse, non-redundant fallback evidence. If that still cannot
        # satisfy the minimum count, relax structural constraints rather than
        # returning an empty context from a non-empty candidate pool.
        fill_from_remaining(relax_structure=False)
        if len(selected) < required_floor:
            fill_from_remaining(relax_structure=True)

        discarded: list[dict[str, Any]] = [*invalid]
        for candidate in usable:
            if any(candidate is item for item in selected):
                continue
            score = float(candidate["reranker_score"])
            token_count = int(candidate["evidence_token_count"])
            if threshold_enabled and score < self.score_threshold:
                reason = "threshold_failed"
            elif is_redundant(candidate):
                reason = "redundancy"
            elif (
                "source_diversity" in self.strategies
                and source_counts[_source(candidate)]
                >= self.max_chunks_per_source
            ):
                reason = "source_diversity_limit"
            elif (
                "token_budget" in self.strategies
                and (
                    used_tokens + token_count > self.token_budget
                    or used_tokens + token_count > self.target_max_tokens
                )
            ):
                reason = "token_budget"
            else:
                reason = "lower_rank_fallback"
            candidate["selected"] = False
            candidate["discard_reason"] = reason
            discarded.append(candidate)

        discard_counts = Counter(
            str(candidate["discard_reason"]) for candidate in discarded
        )
        weak_evidence = bool(selected) and not any(
            not bool(candidate.get("weak_evidence")) for candidate in selected
        )
        return EvidenceSelectionResult(
            selected=tuple(selected),
            discarded=tuple(discarded),
            selected_tokens=used_tokens,
            latency_seconds=perf_counter() - started,
            fallback_used=fallback_used,
            weak_evidence=weak_evidence,
            usable_candidate_count=len(usable),
            threshold_pass_count=threshold_pass_count,
            discard_reason_counts=dict(sorted(discard_counts.items())),
        )
