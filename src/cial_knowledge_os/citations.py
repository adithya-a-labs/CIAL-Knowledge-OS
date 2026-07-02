"""Deterministic citation construction and rendering for grounded answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class CitationLinkResolver(Protocol):
    """Optional link resolver accepted by :func:`build_citations`."""

    def build(self, result: Mapping[str, Any]) -> str | None: ...

_REFERENCE_ID_PATTERN = re.compile(r"\[(\d+)\]")
_GENERIC_REFERENCE_LINE_PATTERN = re.compile(
    r"(?im)^\s*references?\s*:\s*(?:\[\d+\][,\s]*)+\s*$"
)
_NO_EVIDENCE_ANSWERS = {
    "It is not available in the retrieved documents.",
    (
        "The retrieved documents do not contain sufficient evidence to answer "
        "this question. Based only on the indexed corpus, no reliable answer "
        "could be generated."
    ),
}


def build_citations(
    results: Iterable[Mapping[str, Any]],
    *,
    link_resolver: CitationLinkResolver | None = None,
) -> list[dict[str, Any]]:
    """Build ranked citations that align one-to-one with retrieval results.

    Link enrichment is opt-in so Phase 1 and Phase 2 output remains unchanged.
    """

    citations: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        metadata_value = result.get("metadata")
        metadata = (
            metadata_value if isinstance(metadata_value, Mapping) else {}
        )
        source_path = metadata.get("source")
        source_value = (
            metadata.get("file_name")
            or result.get("source")
            or source_path
        )
        source_file = Path(str(source_value)).name if source_value else None
        citations.append(
            {
                "reference_id": rank,
                # Preserve the existing public citation keys.
                "source": source_file,
                "page_number": result.get(
                    "page_number",
                    metadata.get("page_number"),
                ),
                "chunk_id": result.get("chunk_id", metadata.get("chunk_id")),
                "score": result.get("score"),
                # Retain both human-readable and traceable source forms.
                "source_file": source_file,
                "source_path": source_path,
                **(
                    {"pdf_link": link_resolver.build(result)}
                    if link_resolver is not None
                    else {}
                ),
            }
        )
    return citations


def _format_score(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_citations(
    citations: Iterable[Mapping[str, Any]],
    *,
    reference_ids: Sequence[int] | None = None,
) -> str:
    """Render citation metadata as a readable reference list."""

    requested = set(reference_ids) if reference_ids is not None else None
    lines: list[str] = []
    for citation in citations:
        reference_id = int(citation["reference_id"])
        if requested is not None and reference_id not in requested:
            continue
        source = (
            citation.get("source_file")
            or citation.get("source")
            or "unknown source"
        )
        details = [str(source)]
        page = citation.get("page_number")
        if page is not None and page != "":
            details.append(f"page {page}")
        chunk_id = citation.get("chunk_id")
        if chunk_id is not None and chunk_id != "":
            details.append(f"chunk {chunk_id}")
        score = _format_score(citation.get("score"))
        if score is not None:
            details.append(f"score {score}")
        pdf_link = citation.get("pdf_link")
        if pdf_link:
            details.append(f"PDF {pdf_link}")
        lines.append(f"[{reference_id}] " + " | ".join(details))
    return "\n".join(lines)


def render_answer_with_citations(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
) -> str:
    """Resolve numeric answer markers into a metadata-rich reference section."""

    cleaned_answer = _GENERIC_REFERENCE_LINE_PATTERN.sub("", answer).strip()
    if not cleaned_answer or cleaned_answer in _NO_EVIDENCE_ANSWERS or not citations:
        return cleaned_answer

    valid_ids = {
        int(citation["reference_id"])
        for citation in citations
    }
    referenced_ids = list(
        dict.fromkeys(
            reference_id
            for match in _REFERENCE_ID_PATTERN.finditer(answer)
            if (reference_id := int(match.group(1))) in valid_ids
        )
    )
    # If a model omitted inline markers, retain traceability by resolving all
    # retrieved evidence rather than returning an uncited enterprise answer.
    references = render_citations(
        citations,
        reference_ids=referenced_ids or None,
    )
    if not references:
        return cleaned_answer
    return f"{cleaned_answer}\n\nReferences:\n{references}"
