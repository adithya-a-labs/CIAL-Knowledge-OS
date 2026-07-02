"""Rank-based fusion strategies for independently scored retrievers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .metadata import chunk_identity, normalize_result

logger = logging.getLogger(__name__)


def _fusion_identity(result: Mapping[str, Any], position: int) -> tuple[Any, ...]:
    identity = chunk_identity(result)
    if identity[0] and identity[2]:
        return identity
    point_id = result.get("id")
    if point_id is not None:
        return ("point", str(point_id))
    return ("occurrence", position, str(result.get("text") or ""))


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusion:
    """Fuse rankings without comparing incompatible raw score scales.

    Each document receives ``weight / (rank_constant + rank)`` from every
    ranking in which it appears. Retriever-specific scores and ranks remain in
    the result for inspection, while the public ``score`` becomes the RRF score.
    """

    rank_constant: int = 60
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank_constant <= 0:
            raise ValueError("rank_constant must be greater than zero.")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("RRF weights must be greater than zero.")

    def fuse(
        self,
        rankings: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a deterministic fused ranking with modality provenance."""

        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero.")
        accumulated: dict[tuple[Any, ...], dict[str, Any]] = {}
        first_seen = 0
        for retriever_name, ranking in rankings.items():
            weight = float(self.weights.get(retriever_name, 1.0))
            seen_in_ranking: set[tuple[Any, ...]] = set()
            for rank, raw_result in enumerate(ranking, start=1):
                result = normalize_result(raw_result)
                identity = _fusion_identity(result, rank)
                if identity in seen_in_ranking:
                    continue
                seen_in_ranking.add(identity)
                entry = accumulated.get(identity)
                if entry is None:
                    first_seen += 1
                    entry = {
                        "result": result,
                        "rrf_score": 0.0,
                        "retrieval_ranks": {},
                        "retrieval_scores": {},
                        "first_seen": first_seen,
                    }
                    accumulated[identity] = entry
                entry["rrf_score"] += weight / (self.rank_constant + rank)
                entry["retrieval_ranks"][retriever_name] = rank
                entry["retrieval_scores"][retriever_name] = result.get("score")

        ordered = sorted(
            accumulated.values(),
            key=lambda entry: (-entry["rrf_score"], entry["first_seen"]),
        )
        fused: list[dict[str, Any]] = []
        for entry in ordered[:limit]:
            result = dict(entry["result"])
            result.update(
                {
                    "score": float(entry["rrf_score"]),
                    "rrf_score": float(entry["rrf_score"]),
                    "retrieval_ranks": dict(entry["retrieval_ranks"]),
                    "retrieval_scores": dict(entry["retrieval_scores"]),
                    "retrieval_sources": list(entry["retrieval_ranks"]),
                }
            )
            fused.append(result)
        logger.info(
            "rrf_complete",
            extra={
                "event": "reciprocal_rank_fusion",
                "ranking_counts": {
                    name: len(ranking) for name, ranking in rankings.items()
                },
                "result_count": len(fused),
                "rank_constant": self.rank_constant,
            },
        )
        return fused
