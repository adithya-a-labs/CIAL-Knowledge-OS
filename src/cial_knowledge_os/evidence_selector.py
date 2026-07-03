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

    ``selected`` preserves reranker order and contains only chunks satisfying
    every enabled strategy. ``discarded`` records one primary ``discard_reason``
    per rejected candidate. Token counts use the injected Phase 3 token manager,
    so selection and final context accounting share one tokenizer.
    """

    selected: tuple[dict[str, Any], ...]
    discarded: tuple[dict[str, Any], ...]
    selected_tokens: int
    latency_seconds: float


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
    Phase 3 context construction remains reusable and receives only the selected
    chunks.
    """

    def __init__(
        self,
        token_manager: TokenManager,
        *,
        strategies: Sequence[str],
        max_chunks: int,
        score_threshold: float,
        token_budget: int,
        max_chunks_per_source: int,
        redundancy_threshold: float,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero.")
        if max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be greater than zero.")
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between zero and one.")
        self.token_manager = token_manager
        self.strategies = frozenset(strategies)
        self.max_chunks = max_chunks
        self.score_threshold = float(score_threshold)
        self.token_budget = token_budget
        self.max_chunks_per_source = max_chunks_per_source
        self.redundancy_threshold = redundancy_threshold

    def select(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> EvidenceSelectionResult:
        """Return explainable keep/discard decisions for ranked candidates.

        Each input is copied and annotated with ``selected``,
        ``discard_reason``, and ``evidence_token_count``. Strong chunks are not
        retained merely to fill the context window: a chunk is discarded when
        it is weak, duplicate, redundant, source-concentrated, over budget, or
        beyond the configured evidence count. Empty input returns an empty
        result and remains compatible with Phase 3 safe-failure behavior.
        """

        started = perf_counter()
        selected: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        identities: set[tuple[Any, ...]] = set()
        source_counts: Counter[str] = Counter()
        used_tokens = 0

        for value in candidates:
            candidate = dict(value)
            text = str(candidate.get("text") or candidate.get("page_content") or "")
            token_count = self.token_manager.count(text)
            candidate["evidence_token_count"] = token_count
            reason: str | None = None

            if "top_k" in self.strategies and len(selected) >= self.max_chunks:
                reason = "max_evidence_count"
            elif (
                "reranker_score_threshold" in self.strategies
                and float(candidate.get("reranker_score") or 0.0)
                < self.score_threshold
            ):
                reason = "low_score"
            elif chunk_identity(candidate) in identities:
                reason = "duplicate"
            elif (
                "redundancy_reduction" in self.strategies
                and any(
                    _jaccard(
                        text,
                        str(existing.get("text") or existing.get("page_content") or ""),
                    )
                    >= self.redundancy_threshold
                    for existing in selected
                )
            ):
                reason = "redundant"
            elif (
                "source_diversity" in self.strategies
                and source_counts[_source(candidate)]
                >= self.max_chunks_per_source
            ):
                # A concentration cap prevents one long document from crowding
                # out independent evidence and creating misleading confidence.
                reason = "source_diversity"
            elif (
                "token_budget" in self.strategies
                and used_tokens + token_count > self.token_budget
            ):
                # Token reduction is a quality and latency control, not just a
                # hard model limit; oversized marginal chunks are excluded.
                reason = "token_budget"

            if reason is not None:
                candidate["selected"] = False
                candidate["discard_reason"] = reason
                discarded.append(candidate)
                continue

            candidate["selected"] = True
            candidate["discard_reason"] = None
            selected.append(candidate)
            identities.add(chunk_identity(candidate))
            source_counts[_source(candidate)] += 1
            used_tokens += token_count

        return EvidenceSelectionResult(
            selected=tuple(selected),
            discarded=tuple(discarded),
            selected_tokens=used_tokens,
            latency_seconds=perf_counter() - started,
        )
