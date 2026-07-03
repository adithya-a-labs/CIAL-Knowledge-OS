"""Evidence quality diagnostics for selected Phase 4 chunks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .metadata import chunk_id, page_number, source_label, source_path


def _retrieval_source(candidate: Mapping[str, Any]) -> str:
    values = candidate.get("retrieval_sources")
    values = values if isinstance(values, (list, tuple, set)) else []
    normalized = {str(value).casefold() for value in values}
    if {"dense", "bm25"} <= normalized:
        return "both"
    if "dense" in normalized:
        return "dense"
    if "bm25" in normalized:
        return "bm25"
    return str(candidate.get("retrieval_source") or "unknown")


@dataclass(frozen=True, slots=True)
class EvidenceQualityReport:
    """Hold per-chunk quality records and aggregate evidence diagnostics.

    ``chunks`` reports reranker strength, retriever provenance, citation
    availability, and metadata completeness for every selected chunk.
    ``summary`` contains distributions and source-diversity measures suitable
    for JSON traces, CSV metrics, notebooks, and standalone reports.
    """

    chunks: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class EvidenceQualityScorer:
    """Classify selected evidence without making unsupported quality claims.

    Inputs are selected chunk mappings plus configurable strong/medium
    reranker-score thresholds and an optional citation-link resolver. Outputs
    are deterministic diagnostics rather than semantic correctness judgments.
    Source diversity is reported because independent documents/pages reduce
    concentration risk, but it is not treated as proof that the answer is true.

    The scorer reads additive Phase 4 fields while preserving Phase 3 metadata
    and citation contracts. It does not modify input chunks.
    """

    def __init__(
        self,
        *,
        strong_threshold: float,
        medium_threshold: float,
        link_resolver: Any | None = None,
    ) -> None:
        if medium_threshold > strong_threshold:
            raise ValueError(
                "medium_threshold must not exceed strong_threshold."
            )
        self.strong_threshold = float(strong_threshold)
        self.medium_threshold = float(medium_threshold)
        self.link_resolver = link_resolver

    def _strength(self, score: float) -> str:
        if score >= self.strong_threshold:
            return "strong"
        if score >= self.medium_threshold:
            return "medium"
        return "weak"

    def _citation_link(self, candidate: Mapping[str, Any]) -> str | None:
        if self.link_resolver is None:
            return None
        build = getattr(self.link_resolver, "build", None)
        if callable(build):
            return build(candidate)
        if callable(self.link_resolver):
            return self.link_resolver(candidate)
        return None

    def score(
        self,
        selected: Sequence[Mapping[str, Any]],
    ) -> EvidenceQualityReport:
        """Return per-chunk evidence strength and aggregate quality signals.

        Reranker scores are classified with configured thresholds. Metadata is
        complete only when source, page, and chunk identifiers are available.
        Citation availability requires traceable source/chunk metadata; a
        clickable link is reported separately because non-PDF evidence can
        still support a structured citation.
        """

        records: list[dict[str, Any]] = []
        sources: set[str] = set()
        pages: set[tuple[str, str]] = set()
        strengths: Counter[str] = Counter()
        provenance: Counter[str] = Counter()

        for position, candidate in enumerate(selected, start=1):
            score = float(candidate.get("reranker_score") or 0.0)
            source = source_path(candidate)
            page = page_number(candidate)
            identifier = chunk_id(candidate)
            retrieval_source = _retrieval_source(candidate)
            strength = self._strength(score)
            link = self._citation_link(candidate)
            source_present = bool(source)
            page_present = page not in {None, ""}
            chunk_present = bool(identifier)
            metadata_complete = source_present and page_present and chunk_present
            citation_available = source_present and chunk_present
            records.append(
                {
                    "rank": position,
                    "source": source_label(candidate),
                    "source_path": source,
                    "page_number": page,
                    "chunk_id": identifier,
                    "reranker_score": score,
                    "retrieval_source": retrieval_source,
                    "source_diversity_contribution": (
                        1 if source and source not in sources else 0
                    ),
                    "citation_available": citation_available,
                    "citation_link": link,
                    "metadata_complete": metadata_complete,
                    "metadata_fields": {
                        "source": source_present,
                        "page": page_present,
                        "chunk_id": chunk_present,
                    },
                    "evidence_strength": strength,
                    "token_count": int(
                        candidate.get("evidence_token_count")
                        or candidate.get("token_count")
                        or 0
                    ),
                }
            )
            if source:
                sources.add(source)
                if page_present:
                    pages.add((source, str(page)))
            strengths[strength] += 1
            provenance[retrieval_source] += 1

        scores = [float(record["reranker_score"]) for record in records]
        summary = {
            "selected_chunk_count": len(records),
            "average_reranker_score": round(fmean(scores), 6) if scores else 0.0,
            "minimum_reranker_score": min(scores) if scores else None,
            "maximum_reranker_score": max(scores) if scores else None,
            "unique_source_count": len(sources),
            "unique_page_count": len(pages),
            "source_diversity_ratio": (
                round(len(sources) / len(records), 6) if records else 0.0
            ),
            "citation_available_count": sum(
                bool(record["citation_available"]) for record in records
            ),
            "metadata_complete_count": sum(
                bool(record["metadata_complete"]) for record in records
            ),
            "strength_distribution": {
                name: strengths.get(name, 0)
                for name in ("strong", "medium", "weak")
            },
            "retrieval_source_distribution": dict(sorted(provenance.items())),
        }
        return EvidenceQualityReport(tuple(records), summary)
