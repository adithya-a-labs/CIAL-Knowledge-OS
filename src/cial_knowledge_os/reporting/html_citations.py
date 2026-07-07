"""Safe Markdown and citation rendering shared by standalone HTML reports."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..phase3_reporting import render_safe_markdown

_REFERENCE_TAIL = re.compile(r"(?im)^\s*references?\s*:\s*$")
_BRACKETED_MARKER = re.compile(r"\[([^\[\]\r\n]+)\]")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(record: Mapping[str, Any], *keys: str) -> Any:
    metadata = _mapping(record.get("metadata"))
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return None


def _source_name(record: Mapping[str, Any]) -> str:
    source = _value(
        record,
        "source_file",
        "source",
        "file_name",
        "relative_path",
    )
    return Path(str(source)).name if source else "Unknown source"


def _score(record: Mapping[str, Any]) -> Any:
    return _value(record, "reranker_score", "score", "rrf_score")


def _format_score(value: Any) -> str:
    if value in {None, ""}:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _safe_href(value: Any) -> str | None:
    link = str(value or "").strip()
    if not link or any(ord(character) < 32 for character in link):
        return None
    try:
        scheme = urlsplit(link).scheme.casefold()
    except ValueError:
        return None
    return link if scheme in {"file", "http", "https"} else None


def _matches(
    citation: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    citation_chunk = str(_value(citation, "chunk_id") or "").casefold()
    evidence_chunk = str(_value(evidence, "chunk_id") or "").casefold()
    if citation_chunk and evidence_chunk:
        return citation_chunk == evidence_chunk
    citation_source = _source_name(citation).casefold()
    evidence_source = _source_name(evidence).casefold()
    citation_page = str(_value(citation, "page_number", "page") or "")
    evidence_page = str(_value(evidence, "page_number", "page") or "")
    return (
        citation_source == evidence_source
        and (not citation_page or not evidence_page or citation_page == evidence_page)
    )


def _citation_records(
    citations: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_values = [dict(item) for item in citations]
    if not source_values:
        source_values = [
            {
                "reference_id": index,
                "source": _source_name(item),
                "page_number": _value(item, "page_number", "page"),
                "chunk_id": _value(item, "chunk_id"),
                "score": _score(item),
            }
            for index, item in enumerate(evidence, start=1)
        ]
    records: list[dict[str, Any]] = []
    for position, citation in enumerate(source_values, start=1):
        matching = next(
            (item for item in evidence if _matches(citation, item)),
            {},
        )
        reference_id = citation.get("reference_id", position)
        preview = _value(
            matching,
            "content",
            "text",
            "text_preview",
            "caption",
            "ocr_text",
        )
        records.append(
            {
                **citation,
                "reference_id": reference_id,
                "source_file": _source_name(citation)
                if _source_name(citation) != "Unknown source"
                else _source_name(matching),
                "page_number": _value(citation, "page_number", "page")
                or _value(matching, "page_number", "page"),
                "chunk_id": _value(citation, "chunk_id")
                or _value(matching, "chunk_id"),
                "score": _score(matching)
                if _score(matching) not in {None, ""}
                else _score(citation),
                "modality": _value(matching, "modality") or "text",
                "preview": str(preview or "").strip()[:420],
                "pdf_link": citation.get("pdf_link"),
                "evidence_id": _value(matching, "evidence_id"),
            }
        )
    return records


def _citation_title(citation: Mapping[str, Any]) -> str:
    details = [_source_name(citation)]
    page = _value(citation, "page_number", "page")
    chunk = _value(citation, "chunk_id")
    if page not in {None, ""}:
        details.append(f"Page {page}")
    if chunk not in {None, ""}:
        details.append(f"Chunk {chunk}")
    details.append(f"Score {_format_score(_score(citation))}")
    details.append(f"Modality {_value(citation, 'modality') or 'text'}")
    return " | ".join(details)


def _resolve_marker(
    marker: str,
    citations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    cleaned = marker.strip()
    numeric = re.fullmatch(r"(?i)(?:source\s*)?(\d+)", cleaned)
    if numeric:
        reference_id = numeric.group(1)
        return next(
            (
                citation
                for citation in citations
                if str(citation.get("reference_id")) == reference_id
            ),
            None,
        )
    normalized = cleaned.casefold()
    matches = []
    for citation in citations:
        identifiers = {
            str(citation.get("evidence_id") or "").casefold(),
            str(citation.get("chunk_id") or "").casefold(),
        }
        if normalized in identifiers:
            matches.append(citation)
            continue
        source = Path(_source_name(citation)).stem.casefold()
        page = str(citation.get("page_number") or "").casefold()
        chunk = str(citation.get("chunk_id") or "").casefold()
        if (
            source
            and source in normalized
            and (not re.search(r"(?i)\bpage\b", cleaned) or page in normalized)
            and (not re.search(r"(?i)\bchunk\b", cleaned) or chunk in normalized)
        ):
            matches.append(citation)
    return matches[0] if len(matches) == 1 else None


def _looks_like_citation(marker: str) -> bool:
    return bool(
        re.fullmatch(r"(?i)(?:source\s*)?\d+", marker.strip())
        or re.search(r"(?i)\b(?:source|page|chunk|citation)\b", marker)
    )


def _citation_chip(
    label: str,
    citation: Mapping[str, Any],
    *,
    question_index: int,
) -> str:
    reference_id = str(citation.get("reference_id") or "?")
    target = f"q{question_index}-source-{reference_id}"
    return (
        f'<a class="citation-chip" href="#{html.escape(target, quote=True)}" '
        f'title="{html.escape(_citation_title(citation), quote=True)}" '
        f'aria-label="{html.escape(_citation_title(citation), quote=True)}">'
        f"{html.escape(label)}</a>"
    )


def render_markdown_with_citations(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    *,
    question_index: int,
) -> tuple[str, list[dict[str, Any]], set[str]]:
    """Render safe Markdown and replace citation markers with local chips."""

    records = _citation_records(citations, evidence)
    cleaned = str(answer or "")
    tail = _REFERENCE_TAIL.search(cleaned)
    if tail and records:
        cleaned = cleaned[: tail.start()].rstrip()
    token_prefix = "CIALPHASEFIVECITATIONTOKEN"
    while token_prefix in cleaned:
        token_prefix += "X"
    replacements: dict[str, str] = {}
    resolved: set[str] = set()
    pieces: list[str] = []
    cursor = 0
    for match in _BRACKETED_MARKER.finditer(cleaned):
        marker = match.group(1)
        citation = _resolve_marker(marker, records)
        if citation is None and not _looks_like_citation(marker):
            continue
        token = f"{token_prefix}{len(replacements)}END"
        pieces.extend((cleaned[cursor : match.start()], token))
        cursor = match.end()
        if citation is None:
            replacements[token] = (
                '<span class="citation-chip unresolved-citation" '
                'title="Unresolved citation reference">'
                f"{html.escape(match.group(0))}</span>"
            )
        else:
            reference_id = str(citation.get("reference_id") or "?")
            resolved.add(reference_id)
            replacements[token] = _citation_chip(
                match.group(0),
                citation,
                question_index=question_index,
            )
    pieces.append(cleaned[cursor:])
    rendered = render_safe_markdown("".join(pieces))
    for token, replacement in replacements.items():
        rendered = rendered.replace(token, replacement)
    return rendered, records, resolved


def render_source_cards(
    citations: Sequence[Mapping[str, Any]],
    *,
    question_index: int,
    resolved_ids: set[str],
) -> str:
    """Render compact source cards aligned with inline citation IDs."""

    if not citations:
        return '<p class="muted">No structured citation sources were produced.</p>'
    cards = []
    for citation in citations:
        reference_id = str(citation.get("reference_id") or "?")
        source_id = f"q{question_index}-source-{reference_id}"
        metadata = [
            f"Page {_value(citation, 'page_number', 'page') or 'N/A'}",
            f"Chunk {_value(citation, 'chunk_id') or 'N/A'}",
            f"Score {_format_score(_score(citation))}",
            f"Modality {_value(citation, 'modality') or 'text'}",
        ]
        link = _safe_href(citation.get("pdf_link"))
        action = (
            f'<a class="source-action" href="{html.escape(link, quote=True)}">'
            "Open document</a>"
            if link
            else '<span class="muted">Document link unavailable</span>'
        )
        cards.append(
            f'<article class="source-card" id="{html.escape(source_id, quote=True)}">'
            '<div class="source-card-head">'
            f'<span class="citation-reference">Source {html.escape(reference_id)}</span>'
            + (
                '<span class="source-used">Cited inline</span>'
                if reference_id in resolved_ids
                else '<span class="source-unused">Available source</span>'
            )
            + "</div>"
            f"<h4>{html.escape(_source_name(citation))}</h4>"
            f'<p class="source-meta">{html.escape(" | ".join(metadata))}</p>'
            f'<p class="source-preview">{html.escape(str(citation.get("preview") or "Preview unavailable."))}</p>'
            f"{action}</article>"
        )
    return '<div class="source-card-grid">' + "".join(cards) + "</div>"


def render_fallback_citation_chips(
    citations: Sequence[Mapping[str, Any]],
    *,
    question_index: int,
) -> str:
    """Preserve Phase 4 behavior when generated text omits inline markers."""

    if not citations:
        return ""
    chips = "".join(
        _citation_chip(
            f"[{citation.get('reference_id') or '?'}]",
            citation,
            question_index=question_index,
        )
        for citation in citations
    )
    return (
        '<div class="citation-chips" aria-label="Supporting answer sources">'
        '<span class="citation-chips-label">Supporting sources:</span>'
        f"{chips}</div>"
    )
