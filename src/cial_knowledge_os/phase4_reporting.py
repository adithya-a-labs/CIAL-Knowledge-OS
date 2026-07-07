"""Standalone, offline Phase 4 reports and decision visualizations."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .phase3_reporting import render_safe_markdown


_REFERENCE_TAIL_PATTERN = re.compile(r"(?im)^\s*references?\s*:\s*$")
_BRACKETED_CITATION_PATTERN = re.compile(r"\[([^\[\]\r\n]+)\]")
_POPOVER_SCRIPT = r"""
(() => {
  const dataNode = document.getElementById("citation-preview-data");
  const popover = document.getElementById("citation-popover");
  if (!dataNode || !popover) return;
  const previews = JSON.parse(dataNode.textContent || "{}");
  const fields = Object.fromEntries(
    Array.from(popover.querySelectorAll("[data-field]")).map(
      (node) => [node.dataset.field, node]
    )
  );
  let activeBadge = null;

  const setText = (name, value) => {
    fields[name].textContent = value || "Unavailable";
  };

  const renderSnippet = (snippet, terms) => {
    const target = fields.snippet;
    target.replaceChildren();
    if (!snippet) {
      target.textContent = "Evidence preview unavailable.";
      target.classList.add("missing");
      return;
    }
    target.classList.remove("missing");
    const usableTerms = Array.from(
      new Set((terms || []).map((term) => String(term)).filter(Boolean))
    ).sort((left, right) => right.length - left.length);
    if (!usableTerms.length) {
      target.textContent = snippet;
      return;
    }
    const lowerSnippet = snippet.toLocaleLowerCase();
    let cursor = 0;
    while (cursor < snippet.length) {
      let nextIndex = -1;
      let nextTerm = "";
      for (const term of usableTerms) {
        const found = lowerSnippet.indexOf(term.toLocaleLowerCase(), cursor);
        if (
          found >= 0 &&
          (nextIndex < 0 || found < nextIndex ||
            (found === nextIndex && term.length > nextTerm.length))
        ) {
          nextIndex = found;
          nextTerm = term;
        }
      }
      if (nextIndex < 0) {
        target.append(document.createTextNode(snippet.slice(cursor)));
        break;
      }
      if (nextIndex > cursor) {
        target.append(
          document.createTextNode(snippet.slice(cursor, nextIndex))
        );
      }
      const mark = document.createElement("mark");
      mark.textContent = snippet.slice(nextIndex, nextIndex + nextTerm.length);
      target.append(mark);
      cursor = nextIndex + nextTerm.length;
    }
  };

  const position = (clientX, clientY) => {
    const margin = 8;
    const gap = 14;
    const rect = popover.getBoundingClientRect();
    let left = clientX + gap;
    let top = clientY + gap;
    if (left + rect.width + margin > window.innerWidth) {
      left = clientX - rect.width - gap;
    }
    if (top + rect.height + margin > window.innerHeight) {
      top = clientY - rect.height - gap;
    }
    popover.style.left = `${Math.max(margin, Math.min(
      left,
      window.innerWidth - rect.width - margin
    ))}px`;
    popover.style.top = `${Math.max(margin, Math.min(
      top,
      window.innerHeight - rect.height - margin
    ))}px`;
  };

  const show = (badge, clientX, clientY) => {
    const preview = previews[badge.dataset.citationPreview];
    if (!preview) return;
    activeBadge = badge;
    setText("source", preview.source);
    setText("page", preview.page);
    setText("chunk", preview.chunk);
    setText("score", preview.reranker_score);
    setText("strength", preview.evidence_strength);
    setText("retriever", preview.retrieval_source);
    fields.action.textContent = preview.pdf_available
      ? "Click to open PDF"
      : "Source available but PDF link unavailable.";
    renderSnippet(preview.snippet, preview.matched_terms);
    popover.hidden = false;
    popover.setAttribute("aria-hidden", "false");
    badge.setAttribute("aria-describedby", "citation-popover");
    position(clientX, clientY);
  };

  const hide = (badge) => {
    if (badge && badge !== activeBadge) return;
    if (activeBadge) activeBadge.removeAttribute("aria-describedby");
    activeBadge = null;
    popover.hidden = true;
    popover.setAttribute("aria-hidden", "true");
  };

  document.querySelectorAll("[data-citation-preview]").forEach((badge) => {
    badge.addEventListener("pointerenter", (event) => {
      show(badge, event.clientX, event.clientY);
    });
    badge.addEventListener("pointermove", (event) => {
      if (activeBadge === badge) position(event.clientX, event.clientY);
    });
    badge.addEventListener("pointerleave", () => hide(badge));
    badge.addEventListener("focus", () => {
      const rect = badge.getBoundingClientRect();
      show(badge, rect.left + rect.width / 2, rect.bottom);
    });
    badge.addEventListener("blur", () => hide(badge));
  });
  window.addEventListener("resize", () => hide(activeBadge));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hide(activeBadge);
  });
})();
"""

_THEME_SCRIPT = r"""
(() => {
  const storageKey = "cial-phase4-report-theme";
  const allowedThemes = new Set(["light", "dark", "system"]);
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  let preference = "system";
  try {
    const saved = window.localStorage.getItem(storageKey);
    if (allowedThemes.has(saved)) preference = saved;
  } catch (_error) {
    preference = "system";
  }

  const resolvedTheme = (choice) => (
    choice === "system"
      ? (systemTheme.matches ? "dark" : "light")
      : choice
  );

  const updateButtons = () => {
    document.querySelectorAll("[data-theme-choice]").forEach((button) => {
      const selected = button.dataset.themeChoice === preference;
      button.setAttribute("aria-pressed", String(selected));
      button.classList.toggle("active", selected);
    });
  };

  const applyTheme = (choice, persist = false) => {
    preference = allowedThemes.has(choice) ? choice : "system";
    document.documentElement.dataset.theme = resolvedTheme(preference);
    document.documentElement.dataset.themePreference = preference;
    updateButtons();
    if (persist) {
      try {
        window.localStorage.setItem(storageKey, preference);
      } catch (_error) {
        // The report remains fully usable when file-browser storage is blocked.
      }
    }
  };

  applyTheme(preference);
  const bindThemeControls = () => {
    document.querySelectorAll("[data-theme-choice]").forEach((button) => {
      button.addEventListener("click", () => {
        applyTheme(button.dataset.themeChoice, true);
      });
    });
    updateButtons();
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindThemeControls, {
      once: true,
    });
  } else {
    bindThemeControls();
  }
  const followSystemTheme = () => {
    if (preference === "system") applyTheme("system");
  };
  if (typeof systemTheme.addEventListener === "function") {
    systemTheme.addEventListener("change", followSystemTheme);
  } else if (typeof systemTheme.addListener === "function") {
    systemTheme.addListener(followSystemTheme);
  }
})();
"""


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _bar_svg(
    values: Sequence[tuple[str, float]],
    *,
    title: str,
    color: str = "var(--chart-bar)",
    text_color: str = "var(--text)",
    muted_text_color: str = "var(--muted-text)",
    width: int = 720,
) -> str:
    safe_values = [(str(label), max(0.0, float(value))) for label, value in values]
    height = max(150, 70 + 34 * len(safe_values))
    maximum = max((value for _, value in safe_values), default=1.0) or 1.0
    rows = []
    for index, (label, value) in enumerate(safe_values):
        y = 48 + index * 34
        bar_width = int((width - 290) * value / maximum)
        rows.append(
            f'<text x="12" y="{y + 15}" class="label">{html.escape(label)}</text>'
            f'<rect x="210" y="{y}" width="{bar_width}" height="20" rx="4" '
            f'fill="{color}"></rect>'
            f'<text x="{220 + bar_width}" y="{y + 15}" class="value">'
            f'{html.escape(_number(value))}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title, quote=True)}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<style>.title{{font:600 16px system-ui;fill:{text_color}}}'
        f'.label,.value{{font:12px system-ui;fill:{muted_text_color}}}'
        "</style>"
        f'<text x="12" y="24" class="title">{html.escape(title)}</text>'
        + "".join(rows)
        + "</svg>"
    )


def _aggregate_chart_values(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    candidate = selected = final = discarded = 0.0
    latency: Counter[str] = Counter()
    strengths: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    candidate_sources: set[str] = set()
    selected_sources: set[str] = set()
    for trace in traces:
        token = trace.get("token_usage")
        token = token if isinstance(token, Mapping) else {}
        candidate += float(token.get("candidate_tokens") or 0.0)
        selected += float(token.get("selected_evidence_tokens") or 0.0)
        final += float(token.get("final_context_tokens") or 0.0)
        discarded += float(token.get("discarded_chunk_count") or 0.0)
        for key, value in (trace.get("latency") or {}).items():
            if key.endswith("_seconds") and value is not None:
                latency[key.removesuffix("_seconds").replace("_", " ")] += float(value)
        quality = trace.get("evidence_quality")
        quality = quality if isinstance(quality, Mapping) else {}
        summary = quality.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        strengths.update(summary.get("strength_distribution") or {})
        for item in trace.get("discarded_chunks") or []:
            reasons[str(item.get("discard_reason") or "unspecified")] += 1
        for item in trace.get("candidate_pool") or []:
            metadata = item.get("metadata") or {}
            source = metadata.get("source") or item.get("source")
            if source:
                candidate_sources.add(str(source))
        for item in trace.get("selected_chunks") or []:
            metadata = item.get("metadata") or {}
            source = metadata.get("source") or item.get("source")
            if source:
                selected_sources.add(str(source))
    return {
        "funnel": [
            ("Hybrid candidates", sum(len(t.get("candidate_pool") or []) for t in traces)),
            ("Reranked candidates", sum(len(t.get("reranked_candidates") or []) for t in traces)),
            ("Selected evidence", sum(len(t.get("selected_chunks") or []) for t in traces)),
            ("Final context chunks", sum(len(t.get("final_context_chunks") or []) for t in traces)),
        ],
        "tokens": [
            ("Candidate tokens", candidate),
            ("Selected evidence tokens", selected),
            ("Final context tokens", final),
        ],
        "latency": sorted(latency.items()),
        "strengths": [(name.title(), strengths.get(name, 0)) for name in ("strong", "medium", "weak")],
        "diversity": [
            ("Candidate unique sources", len(candidate_sources)),
            ("Selected unique sources", len(selected_sources)),
        ],
        "selection": [
            ("Selected", sum(len(t.get("selected_chunks") or []) for t in traces)),
            ("Discarded", discarded),
        ],
        "discard_reasons": sorted(reasons.items()),
    }


def write_phase4_figures(
    figures_dir: str | Path,
    traces: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    """Write reusable inline-SVG-style charts for one Phase 4 run.

    Inputs are the configured figures directory and serialized question traces.
    Outputs are local SVG files covering the candidate funnel, tokens, latency,
    evidence strength, source diversity, selection, and discard reasons. The
    files use no scripts, fonts, CDNs, or network resources and add to the Phase
    3 artifact bundle without changing existing filenames.
    """

    target = Path(figures_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    charts = _aggregate_chart_values(traces)
    specifications = {
        "candidate_funnel.svg": ("Candidate pool funnel", charts["funnel"], "#2563eb"),
        "token_reduction.svg": ("Token reduction", charts["tokens"], "#0f766e"),
        "phase4_latency.svg": ("Latency breakdown", charts["latency"], "#7c3aed"),
        "evidence_strength.svg": ("Evidence strength", charts["strengths"], "#b45309"),
        "source_diversity.svg": ("Source diversity", charts["diversity"], "#0369a1"),
        "selected_vs_discarded.svg": ("Selected vs discarded", charts["selection"], "#15803d"),
        "discard_reasons.svg": ("Discard reasons", charts["discard_reasons"], "#be123c"),
    }
    written = []
    for filename, (title, values, color) in specifications.items():
        path = target / filename
        path.write_text(
            _bar_svg(
                values,
                title=title,
                color=color,
                text_color="#172033",
                muted_text_color="#344054",
            ),
            encoding="utf-8",
        )
        written.append(path)
    return tuple(written)


def _cards(values: Sequence[tuple[str, Any]]) -> str:
    return "".join(
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(str(value))
        + "</strong></div>"
        for label, value in values
    )


def _readiness_section(readiness: Mapping[str, Any]) -> str:
    if not readiness:
        return (
            "<section><h2>Enterprise File Format Readiness</h2>"
            '<p class="muted">No file-format readiness scan was attached to this run.</p>'
            "</section>"
        )
    extension_values = [
        (str(item.get("extension") or ""), float(item.get("count") or 0))
        for item in readiness.get("extensions") or []
        if isinstance(item, Mapping)
    ][:12]
    status_values = [
        (str(label), float(value or 0))
        for label, value in (readiness.get("support_status_distribution") or {}).items()
    ]
    category_values = [
        (str(label), float(value or 0))
        for label, value in (readiness.get("category_distribution") or {}).items()
    ]
    extension_rows = [
        {
            **dict(item),
            "sample_filenames": ", ".join(
                str(value) for value in item.get("sample_filenames", [])
            ),
        }
        for item in readiness.get("extensions") or []
        if isinstance(item, Mapping)
    ]
    skipped = [
        dict(item)
        for item in readiness.get("skipped_files") or []
        if isinstance(item, Mapping)
    ][:20]
    warnings_html = (
        _table(
            skipped,
            (
                ("path", "File"),
                ("support_status", "Status"),
                ("category", "Category"),
                ("action", "Action"),
                ("reason", "Reason"),
            ),
        )
        if skipped
        else '<p class="muted">No future-support or unsupported files were detected.</p>'
    )
    charts = (
        '<div class="grid">'
        + _bar_svg(extension_values, title="Top file extensions")
        + _bar_svg(status_values, title="Support status distribution", color="#0f766e")
        + _bar_svg(category_values, title="Category coverage", color="#7c3aed")
        + "</div>"
    )
    return f"""<section><h2>Enterprise File Format Readiness</h2>
<div class="metrics">{_cards([
("Total files scanned", readiness.get("total_files", 0)),
("Processable files", readiness.get("processable_files", 0)),
("OCR-supported files", readiness.get("ocr_files", 0)),
("Recognized future-support files", readiness.get("recognized_future_files", 0)),
("Unsupported files", readiness.get("unsupported_files", 0)),
])}</div>
{charts}
<h3>Extension Readiness</h3>{_table(
    extension_rows,
    (
        ("extension", "Extension"),
        ("count", "Count"),
        ("category", "Category"),
        ("format_label", "Format"),
        ("support_status", "Support status"),
        ("ingestion_enabled", "Ingestion enabled"),
        ("requires_ocr", "Requires OCR"),
        ("sample_filenames", "Samples"),
    ),
)}
<h3>Skipped and Warning Files</h3>{warnings_html}</section>"""


def _ocr_section(ocr_summary: Mapping[str, Any]) -> str:
    if not ocr_summary:
        return (
            "<section><h2>OCR Processing Summary</h2>"
            '<p class="muted">No OCR processing metrics were attached to this run.</p>'
            "</section>"
        )
    failures = [
        dict(item)
        for item in ocr_summary.get("failures") or []
        if isinstance(item, Mapping)
    ]
    return f"""<section><h2>OCR Processing Summary</h2>
<div class="metrics">{_cards([
("OCR files processed", ocr_summary.get("total_ocr_files_processed", 0)),
("OCR successes", ocr_summary.get("ocr_success_count", 0)),
("OCR failures", ocr_summary.get("ocr_failure_count", 0)),
("OCR success rate", str(round(float(ocr_summary.get("ocr_success_rate") or 0.0) * 100, 2)) + "%"),
("Average OCR time", str(ocr_summary.get("average_ocr_processing_time_ms", 0.0)) + " ms"),
("Extracted characters", ocr_summary.get("total_extracted_characters", 0)),
("Extracted words", ocr_summary.get("total_extracted_words", 0)),
("OCR engine", ocr_summary.get("ocr_engine_used", "tesseract")),
])}</div>
{_table(
    failures,
    (
        ("filename", "File"),
        ("failure_reason", "Failure reason"),
        ("action", "Action taken"),
    ),
) if failures else '<p class="muted">No OCR failures were recorded.</p>'}
</section>"""


def _table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> str:
    if not rows:
        return '<p class="muted">No records.</p>'
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _safe_citation_href(value: Any) -> str | None:
    """Return a safe local/HTTP PDF link suitable for an HTML ``href``."""

    if value is None or value == "":
        return None
    link = str(value).strip()
    if not link or any(ord(character) < 32 for character in link):
        return None
    try:
        scheme = urlsplit(link).scheme.casefold()
    except ValueError:
        return None
    # Phase 3/4 produces file:// links for offline use and localhost HTTP(S)
    # links when explicitly configured. Reject active or unknown schemes so
    # citation metadata cannot introduce executable links into the report.
    if scheme not in {"file", "http", "https"}:
        return None
    return link


def _citation_title(citation: Mapping[str, Any]) -> str:
    """Build hover text containing all available citation provenance."""

    source = (
        citation.get("source_file")
        or citation.get("source")
        or "Unknown source"
    )
    parts = [str(source)]
    page = citation.get("page_number")
    if page is not None and page != "":
        parts.append(f"Page {page}")
    chunk = citation.get("chunk_id")
    if chunk is not None and chunk != "":
        parts.append(f"Chunk {chunk}")
    score = citation.get("score")
    if score is not None and score != "":
        try:
            score = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score = str(score)
        parts.append(f"Score {score}")
    return " | ".join(parts)


def _record_value(record: Mapping[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is not None and value != "":
        return value
    metadata = record.get("metadata")
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _source_key(value: Any) -> str:
    source = Path(str(value or "")).name.casefold()
    return source.removesuffix(".pdf")


def _citation_matches_record(
    citation: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    citation_source = _source_key(
        citation.get("source_file") or citation.get("source")
    )
    record_source = _source_key(
        _record_value(record, "source")
        or _record_value(record, "file_name")
        or record.get("source_path")
    )
    if citation_source and record_source and citation_source != record_source:
        return False

    citation_chunk = str(citation.get("chunk_id") or "").strip().casefold()
    record_chunk = str(_record_value(record, "chunk_id") or "").strip().casefold()
    if citation_chunk and record_chunk:
        return (
            citation_chunk == record_chunk
            or citation_chunk in record_chunk
            or record_chunk in citation_chunk
        )

    citation_page = str(citation.get("page_number") or "").strip().casefold()
    record_page = str(
        _record_value(record, "page_number")
        or _record_value(record, "page")
        or ""
    ).strip().casefold()
    return bool(
        (citation_source or citation_page)
        and (not citation_page or not record_page or citation_page == record_page)
    )


def _matching_record(
    citation: Mapping[str, Any],
    records: Any,
) -> Mapping[str, Any] | None:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return None
    return next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and _citation_matches_record(citation, record)
        ),
        None,
    )


def _retrieval_source_label(
    evidence: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    raw_sources = evidence.get("retrieval_sources")
    sources = (
        [str(value).casefold() for value in raw_sources]
        if isinstance(raw_sources, Sequence)
        and not isinstance(raw_sources, (str, bytes))
        else []
    )
    fallback = str(
        evidence.get("retrieval_source")
        or quality.get("retrieval_source")
        or ""
    ).casefold()
    if fallback == "both":
        sources.extend(("dense", "bm25"))
    elif fallback:
        sources.append(fallback)
    if any(
        evidence.get(key) not in {None, ""}
        for key in ("rrf_score", "rrf_rank", "original_rrf_rank")
    ):
        sources.append("rrf")

    labels = []
    for key, label in (("dense", "Dense"), ("bm25", "BM25"), ("rrf", "RRF")):
        if key in sources:
            labels.append(label)
    return " + ".join(labels) or "Unavailable"


def _matched_terms(
    citation: Mapping[str, Any],
    trace: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    values: list[Any] = []
    direct = evidence.get("matched_terms")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        values.extend(direct)
    for key in (
        "bm25_results",
        "dense_results",
        "rrf_fused_candidates",
        "candidate_pool",
    ):
        records = trace.get(key)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if not _citation_matches_record(citation, record):
                continue
            terms = record.get("matched_terms")
            if isinstance(terms, Sequence) and not isinstance(
                terms,
                (str, bytes),
            ):
                values.extend(terms)

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = str(value).strip()
        normalized = term.casefold()
        if len(term) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(term)
        if len(unique) == 16:
            break
    return unique


def _citation_previews(
    trace: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    *,
    question_index: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Attach bounded, exact evidence previews without duplicating chunk text."""

    quality_value = trace.get("evidence_quality")
    quality_mapping = (
        quality_value if isinstance(quality_value, Mapping) else {}
    )
    quality_records = quality_mapping.get("chunks")
    enriched: list[dict[str, Any]] = []
    previews: dict[str, dict[str, Any]] = {}
    for position, citation in enumerate(citations, start=1):
        preview_id = f"q{question_index}-c{position}"
        evidence = (
            _matching_record(citation, trace.get("final_context_chunks"))
            or _matching_record(citation, trace.get("selected_chunks"))
            or {}
        )
        quality = _matching_record(citation, quality_records) or {}
        text = str(evidence.get("text") or evidence.get("text_preview") or "")
        score = evidence.get("reranker_score")
        if score in {None, ""}:
            score = quality.get("reranker_score")
        try:
            formatted_score = f"{float(score):.4f}"
        except (TypeError, ValueError):
            formatted_score = "Unavailable"

        previews[preview_id] = {
            "source": str(
                citation.get("source_file")
                or citation.get("source")
                or _record_value(evidence, "source")
                or "Unknown source"
            ),
            "page": str(citation.get("page_number") or "N/A"),
            "chunk": str(citation.get("chunk_id") or "N/A"),
            "reranker_score": formatted_score,
            "evidence_strength": str(
                quality.get("evidence_strength") or "Unavailable"
            ).title(),
            "retrieval_source": _retrieval_source_label(evidence, quality),
            # This is a direct substring of the selected evidence, never a
            # generated summary. Compact traces already cap text_preview.
            "snippet": text[:260],
            "matched_terms": _matched_terms(citation, trace, evidence),
            "pdf_available": bool(
                _safe_citation_href(citation.get("pdf_link"))
            ),
        }
        enriched_citation = dict(citation)
        enriched_citation["_preview_id"] = preview_id
        enriched.append(enriched_citation)
    return enriched, previews


def _json_for_html(value: Any) -> str:
    """Serialize JSON safely inside a standalone HTML script element."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _structured_citation(
    marker: str,
    citations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Resolve a source/page/chunk marker to one structured citation."""

    parts = [part.strip() for part in marker.split("|") if part.strip()]
    if len(parts) < 2:
        return None
    source_label = re.sub(
        r"(?i)^source\s*:?\s*",
        "",
        parts[0],
    ).strip()
    page_match = re.search(r"(?i)\bpage\s*:?\s*(.+)", marker)
    chunk_match = re.search(r"(?i)\bchunk\s*:?\s*(.+)", marker)
    page = (
        page_match.group(1).split("|", maxsplit=1)[0].strip()
        if page_match
        else ""
    )
    chunk = (
        chunk_match.group(1).split("|", maxsplit=1)[0].strip()
        if chunk_match
        else ""
    )
    if not page and not chunk:
        return None

    source_key = _source_key(source_label)
    matches: list[Mapping[str, Any]] = []
    for citation in citations:
        citation_source = _source_key(
            citation.get("source_file") or citation.get("source")
        )
        source_matches = (
            not source_key
            or (
                bool(citation_source)
                and (
                    source_key == citation_source
                    or source_key in citation_source
                    or citation_source in source_key
                )
            )
        )
        citation_page = str(citation.get("page_number") or "").strip()
        citation_chunk = str(citation.get("chunk_id") or "").strip()
        page_matches = not page or page.casefold() == citation_page.casefold()
        chunk_matches = (
            not chunk
            or chunk.casefold() == citation_chunk.casefold()
            or citation_chunk.casefold().endswith(chunk.casefold())
        )
        if source_matches and page_matches and chunk_matches:
            matches.append(citation)
    return matches[0] if len(matches) == 1 else None


def _marker_citation(
    marker: str,
    citations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    cleaned = marker.strip()
    if cleaned.isdigit():
        for citation in citations:
            if str(citation.get("reference_id") or "") == cleaned:
                return citation
        return None
    return _structured_citation(cleaned, citations)


def _citation_badge(
    citation: Mapping[str, Any],
    *,
    label: str,
    css_class: str = "inline-citation",
) -> str:
    title = html.escape(_citation_title(citation), quote=True)
    escaped_label = html.escape(label)
    href = _safe_citation_href(citation.get("pdf_link"))
    preview_id = html.escape(
        str(citation.get("_preview_id") or ""),
        quote=True,
    )
    preview_attribute = (
        f' data-citation-preview="{preview_id}"'
        if preview_id
        else ""
    )
    aria_label = html.escape(
        f"{label}: {_citation_title(citation)}. "
        "Hover for evidence preview.",
        quote=True,
    )
    if href:
        return (
            f'<a class="{css_class}"{preview_attribute} '
            f'href="{html.escape(href, quote=True)}" '
            f'aria-label="{aria_label}" data-citation-title="{title}">'
            f"{escaped_label}</a>"
        )
    return (
        f'<span class="{css_class} no-citation-link"{preview_attribute} '
        f'tabindex="0" aria-label="{aria_label}" '
        f'data-citation-title="{title}">'
        f"{escaped_label}</span>"
    )


def _render_answer_with_inline_citations(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
) -> tuple[str, int]:
    """Safely render an answer and replace recognized markers with links.

    Inputs are generated Markdown and structured Phase 4 citations. The output
    contains escaped, dependency-free HTML plus the number of inline markers
    resolved. Trusted placeholder tokens are substituted only after the shared
    safe Markdown renderer has escaped model output, preserving Phase 3
    grounding and HTML-safety behavior.
    """

    cleaned = str(answer)
    if citations:
        tail = _REFERENCE_TAIL_PATTERN.search(cleaned)
        if tail:
            cleaned = cleaned[: tail.start()].rstrip()

    prefix = "CIALINLINECITATIONTOKEN"
    while prefix in cleaned:
        prefix += "X"
    replacements: dict[str, str] = {}
    pieces: list[str] = []
    cursor = 0
    for match in _BRACKETED_CITATION_PATTERN.finditer(cleaned):
        citation = _marker_citation(match.group(1), citations)
        if citation is None:
            continue
        token = f"{prefix}{len(replacements)}END"
        pieces.append(cleaned[cursor : match.start()])
        pieces.append(token)
        cursor = match.end()
        replacements[token] = _citation_badge(
            citation,
            label=match.group(0),
        )
    pieces.append(cleaned[cursor:])
    rendered = render_safe_markdown("".join(pieces))
    for token, badge in replacements.items():
        rendered = rendered.replace(token, badge)
    return rendered, len(replacements)


def _fallback_citation_chips(
    citations: Sequence[Mapping[str, Any]],
) -> str:
    if not citations:
        return ""
    chips = "".join(
        _citation_badge(
            citation,
            label=f"[{citation.get('reference_id', '?')}]",
            css_class="citation-chip",
        )
        for citation in citations
    )
    return (
        '<div class="citation-chips" aria-label="Answer citations">'
        f'<span class="citation-chips-label">Sources:</span>{chips}</div>'
    )


def _citations(citations: Sequence[Mapping[str, Any]]) -> str:
    if not citations:
        return '<p class="muted">No citations were produced.</p>'
    cards = []
    for citation in citations:
        reference_id = citation.get("reference_id", "?")
        label = (
            f"{citation.get('source_file') or citation.get('source') or 'Unknown'}"
            f" · Page {citation.get('page_number') or 'N/A'}"
            f" · Chunk {citation.get('chunk_id') or 'N/A'}"
        )
        link = _safe_citation_href(citation.get("pdf_link"))
        action = (
            f'<a href="{html.escape(link, quote=True)}">Open PDF</a>'
            if link
            else '<span class="muted">No PDF link</span>'
        )
        cards.append(
            '<div class="citation-card">'
            f'<span class="citation-reference">[{html.escape(str(reference_id))}]</span>'
            f'<strong title="{html.escape(_citation_title(citation), quote=True)}">'
            f"{html.escape(label)}</strong>{action}</div>"
        )
    return '<div class="citation-list">' + "".join(cards) + "</div>"


def write_phase4_html(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write a polished standalone Phase 4 engineering control-room report.

    Inputs are compact result rows, full/compact traces, and aggregate summaries.
    The output is one double-click-openable HTML file with safe Markdown answers,
    clickable citations, inline SVG decision charts, reranking/selection tables,
    diagnostics, and collapsible context/debug data. It reuses Phase 3 artifact
    paths and adds no external CSS, JavaScript, CDN, or cloud dependency.
    """

    target = Path(path).expanduser().resolve()
    charts = _aggregate_chart_values(traces)
    chart_html = {
        key: _bar_svg(values, title=key.replace("_", " ").title())
        for key, values in charts.items()
    }
    indexing_value = summary.get("indexing_summary")
    indexing = (
        indexing_value if isinstance(indexing_value, Mapping) else {}
    )
    indexing_cards = (
        [
            ("Index new files", indexing.get("new_files", 0)),
            ("Index changed files", indexing.get("changed_files", 0)),
            ("Index unchanged files", indexing.get("unchanged_files", 0)),
            ("Index deleted files", indexing.get("deleted_files", 0)),
            ("Chunks added", indexing.get("chunks_added", 0)),
            ("Chunks removed", indexing.get("chunks_removed", 0)),
        ]
        if indexing
        else []
    )
    answer_sections = []
    reranking_sections = []
    selection_sections = []
    quality_sections = []
    debug_sections = []
    diagnostics = []
    citation_preview_data: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        trace = traces[index - 1] if index <= len(traces) else {}
        question = str(row.get("question") or trace.get("question") or "")
        citation_value = trace.get("citations")
        citations = (
            [
                citation
                for citation in citation_value
                if isinstance(citation, Mapping)
            ]
            if isinstance(citation_value, Sequence)
            and not isinstance(citation_value, (str, bytes))
            else []
        )
        citations, question_previews = _citation_previews(
            trace,
            citations,
            question_index=index,
        )
        citation_preview_data.update(question_previews)
        answer_html, inline_citation_count = (
            _render_answer_with_inline_citations(
                str(row.get("answer") or trace.get("answer") or ""),
                citations,
            )
        )
        fallback_chips = (
            _fallback_citation_chips(citations)
            if inline_citation_count == 0
            else ""
        )
        answer_status = str(
            row.get("answer_status")
            or trace.get("answer_status")
            or "Unknown"
        )
        status_key = re.sub(
            r"[^a-z0-9]+",
            "-",
            answer_status.casefold(),
        ).strip("-")
        citation_details = (
            '<details class="citation-details">'
            f"<summary>Citation details ({len(citations)})</summary>"
            f"{_citations(citations)}</details>"
            if citations
            else _citations(citations)
        )
        answer_sections.append(
            f'<article class="answer-card"><p class="eyebrow">Question {index}</p>'
            f"<h3>{html.escape(question)}</h3>"
            f'<p class="answer-status status-{html.escape(status_key, quote=True)}">'
            f"{html.escape(answer_status)}</p>"
            f'<div class="answer-content">{answer_html}</div>'
            f"{fallback_chips}{citation_details}</article>"
        )
        reranking_sections.append(
            f"<h3>{index}. {html.escape(question)}</h3>"
            + _table(
                trace.get("reranked_candidates") or [],
                (
                    ("original_rrf_rank", "RRF rank"),
                    ("reranked_rank", "Reranked rank"),
                    ("reranker_score", "Reranker score"),
                    ("source", "Source"),
                    ("page_number", "Page"),
                    ("chunk_id", "Chunk"),
                ),
            )
        )
        selection_rows = [
            *[dict(item) | {"decision": "selected"} for item in trace.get("selected_chunks") or []],
            *[
                dict(item) | {"decision": f"discarded: {item.get('discard_reason')}"}
                for item in trace.get("discarded_chunks") or []
            ],
        ]
        selection_sections.append(
            f"<h3>{index}. {html.escape(question)}</h3>"
            + _table(
                selection_rows,
                (
                    ("reranked_rank", "Rank"),
                    ("decision", "Decision"),
                    ("selection_reason", "Selection reason"),
                    ("weak_evidence", "Weak evidence"),
                    ("reranker_score", "Score"),
                    ("evidence_token_count", "Tokens"),
                    ("source", "Source"),
                ),
            )
        )
        quality = trace.get("evidence_quality") or {}
        quality_sections.append(
            f"<h3>{index}. {html.escape(question)}</h3>"
            + _table(
                quality.get("chunks") or [],
                (
                    ("rank", "Rank"),
                    ("evidence_strength", "Strength"),
                    ("reranker_score", "Score"),
                    ("retrieval_source", "Retriever"),
                    ("metadata_complete", "Metadata complete"),
                    ("citation_available", "Citation available"),
                ),
            )
        )
        for item in trace.get("decision_summary") or []:
            diagnostics.append(
                f"<li><strong>{html.escape(str(item.get('signal') or 'diagnostic'))}:</strong> "
                f"{html.escape(str(item.get('recommendation') or ''))}</li>"
            )
        debug_sections.append(
            f'<details><summary>{index}. {html.escape(question)} — context and trace</summary>'
            f"<h4>Final context</h4><pre>{html.escape(str(trace.get('phase3_trace', {}).get('final_context') or ''))}</pre>"
            f"<h4>Pipeline flow</h4><pre>{html.escape(' → '.join(trace.get('pipeline_flow') or []))}</pre>"
            "</details>"
        )
    comparison_note = (
        "Phase 4 changes evidence precision, not the intended answer depth. It "
        "retains Phase 3 grounding and citation rules while requesting detailed, "
        "decision-useful synthesis from selected evidence only. No "
        "benchmark-qualified Phase 3 versus Phase 4 quality comparison is "
        "attached to this run; full qualification remains pending."
    )
    readiness_value = summary.get("file_format_readiness") or metrics.get(
        "file_format_readiness"
    )
    readiness = readiness_value if isinstance(readiness_value, Mapping) else {}
    ocr_value = summary.get("ocr_summary") or metrics.get("ocr_summary")
    ocr_summary = ocr_value if isinstance(ocr_value, Mapping) else {}
    readiness_html = _readiness_section(readiness)
    ocr_html = _ocr_section(ocr_summary)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 4 Reranking and Evidence Selection</title>
<script>{_THEME_SCRIPT}</script>
<style>
:root{{
  color-scheme:light;
  --background:#f3f6fb;--card-background:#ffffff;--answer-background:#ffffff;
  --text:#172033;--muted-text:#667085;--border:#dfe5ef;--accent:#1d4ed8;
  --accent-contrast:#ffffff;--link:#1d4ed8;--success:#166534;
  --success-background:#dcfce7;--warning:#92400e;--warning-background:#fef3c7;
  --error:#991b1b;--error-background:#fee2e2;--neutral:#374151;
  --neutral-background:#e5e7eb;--failure:#6b21a8;--failure-background:#f3e8ff;
  --code-block:#0f172a;--code-text:#dbeafe;--inline-code:#eef2ff;
  --citation-badge:#eaf2ff;--citation-text:#174ea6;--citation-border:#93b4e8;
  --citation-hover:#dbeafe;--citation-card:#f8fafc;--citation-disabled:#475467;
  --table-header:#eef3fb;--table-hover:#f8fafc;--chart-container:#ffffff;
  --chart-bar:#2563eb;--metric-background:#f8fafc;--details-background:#ffffff;
  --popover-background:#ffffff;--popover-text:#172033;--popover-muted:#667085;
  --popover-border:#cbd5e1;--popover-snippet:#f1f5f9;--mark-background:#fde68a;
  --mark-text:#713f12;--control-background:#ffffff;--control-hover:#eef3fb;
  --control-active:#dbeafe;--focus:#2563eb;--header-start:#102a56;
  --header-end:#1d4ed8;--header-text:#ffffff;--header-muted:#bfdbfe;
  --shadow:#102a560d;--popover-shadow:#0f172a3d;
}}
[data-theme="dark"]{{
  color-scheme:dark;
  --background:#080f1d;--card-background:#111827;--answer-background:#131d2d;
  --text:#e8eef8;--muted-text:#a8b3c5;--border:#334155;--accent:#78b3ff;
  --accent-contrast:#08111f;--link:#93c5fd;--success:#86efac;
  --success-background:#14532d;--warning:#fde68a;--warning-background:#713f12;
  --error:#fecaca;--error-background:#7f1d1d;--neutral:#d1d5db;
  --neutral-background:#374151;--failure:#e9d5ff;--failure-background:#581c87;
  --code-block:#030712;--code-text:#dbeafe;--inline-code:#1e293b;
  --citation-badge:#172554;--citation-text:#bfdbfe;--citation-border:#3b82f6;
  --citation-hover:#1e3a8a;--citation-card:#182235;--citation-disabled:#cbd5e1;
  --table-header:#1e293b;--table-hover:#182235;--chart-container:#0f172a;
  --chart-bar:#60a5fa;--metric-background:#182235;--details-background:#0f172a;
  --popover-background:#111827;--popover-text:#f8fafc;--popover-muted:#a8b3c5;
  --popover-border:#475569;--popover-snippet:#1e293b;--mark-background:#854d0e;
  --mark-text:#fef3c7;--control-background:#111827;--control-hover:#1e293b;
  --control-active:#1e3a8a;--focus:#93c5fd;--header-start:#111c36;
  --header-end:#1e3a8a;--header-text:#f8fafc;--header-muted:#bfdbfe;
  --shadow:#00000040;--popover-shadow:#00000099;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--background);color:var(--text);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}}
header{{padding:30px 34px;border:1px solid var(--border);border-radius:18px;background:linear-gradient(135deg,var(--header-start),var(--header-end));color:var(--header-text);box-shadow:0 8px 24px var(--shadow)}}
.header-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:24px}}.header-copy{{min-width:0}}
h1{{margin:0 0 8px;font-size:32px}}h2{{margin-top:0}}
.theme-switcher{{display:inline-flex;flex:0 0 auto;gap:3px;padding:4px;border:1px solid var(--header-muted);border-radius:10px;background:var(--control-background)}}
.theme-switcher button{{border:0;border-radius:7px;padding:7px 10px;background:transparent;color:var(--text);font:600 12px system-ui,-apple-system,Segoe UI,sans-serif;cursor:pointer}}
.theme-switcher button:hover{{background:var(--control-hover)}}.theme-switcher button.active{{background:var(--control-active);color:var(--citation-text)}}
.theme-switcher button:focus-visible,.inline-citation:focus-visible,.citation-chip:focus-visible,summary:focus-visible{{outline:3px solid var(--focus);outline-offset:2px}}
.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
section{{background:var(--card-background);border:1px solid var(--border);border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 6px 18px var(--shadow)}}
.answer-card{{background:var(--answer-background);border:1px solid var(--border);border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 4px 14px var(--shadow)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:18px}}
.metric{{background:var(--metric-background);border:1px solid var(--border);border-radius:10px;padding:14px}}.metric span{{display:block;color:var(--muted-text);font-size:12px}}.metric strong{{font-size:22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
svg{{width:100%;height:auto;border:1px solid var(--border);border-radius:10px;background:var(--chart-container)}}
.table-wrap{{overflow:auto;border:1px solid var(--border);border-radius:9px}}table{{width:100%;border-collapse:collapse;background:var(--card-background)}}th,td{{padding:9px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}th{{background:var(--table-header);position:sticky;top:0}}tbody tr:hover{{background:var(--table-hover)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.08em;color:var(--accent);font-weight:700}}header .eyebrow{{color:var(--header-muted)}}.muted{{color:var(--muted-text)}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--code-block);color:var(--code-text);padding:14px;border:1px solid var(--border);border-radius:9px}}code{{background:var(--inline-code);color:var(--text);padding:2px 5px;border-radius:4px}}
.answer-status{{display:inline-flex;margin:0 0 12px;padding:4px 9px;border-radius:999px;background:var(--neutral-background);color:var(--neutral);font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.04em}}
.status-answered{{background:var(--success-background);color:var(--success)}}.status-insufficient-evidence{{background:var(--warning-background);color:var(--warning)}}.status-unsupported-query{{background:var(--error-background);color:var(--error)}}.status-generation-failed{{background:var(--failure-background);color:var(--failure)}}
.inline-citation,.citation-chip{{display:inline-flex;align-items:center;border:1px solid var(--citation-border);border-radius:999px;background:var(--citation-badge);color:var(--citation-text);font-size:.78em;font-weight:700;line-height:1.35;padding:1px 6px;text-decoration:none;vertical-align:.08em;white-space:nowrap}}
.inline-citation:hover,.citation-chip:hover{{background:var(--citation-hover);border-color:var(--accent)}}.no-citation-link{{color:var(--citation-disabled);border-color:var(--border);background:var(--metric-background);cursor:help}}
.citation-chips{{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:8px 0 12px}}.citation-chips-label{{color:var(--muted-text);font-size:12px;font-weight:650}}.citation-list{{display:grid;gap:8px}}
.citation-card{{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:start;gap:12px;border-left:4px solid var(--accent);padding:10px 12px;background:var(--citation-card)}}.citation-reference{{font-weight:750;color:var(--accent)}}.citation-details{{margin-top:12px}}
a{{color:var(--link)}}details{{border:1px solid var(--border);border-radius:9px;padding:10px;margin:10px 0;background:var(--details-background)}}summary{{cursor:pointer;font-weight:650}}
.citation-popover{{position:fixed;z-index:1000;width:min(420px,calc(100vw - 16px));max-height:calc(100vh - 16px);overflow:auto;padding:14px;border:1px solid var(--popover-border);border-radius:12px;background:var(--popover-background);color:var(--popover-text);box-shadow:0 16px 40px var(--popover-shadow);pointer-events:none;overflow-wrap:anywhere;white-space:normal}}.citation-popover[hidden]{{display:none}}
.citation-popover-source{{font-weight:750;margin:0 0 8px}}.citation-popover-grid{{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:3px 10px;margin:0 0 10px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}}.citation-popover-grid dt{{color:var(--popover-muted)}}.citation-popover-grid dd{{margin:0}}
.citation-popover-snippet{{margin:0;padding:10px;border-radius:8px;background:var(--popover-snippet);white-space:pre-wrap;font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}.citation-popover-snippet.missing{{color:var(--popover-muted);font-style:italic}}.citation-popover-snippet mark{{border-radius:2px;background:var(--mark-background);color:var(--mark-text);padding:0 1px}}.citation-popover-action{{margin:9px 0 0;color:var(--link);font-size:12px;font-weight:700}}
.answer-content{{white-space:normal;overflow:visible;max-height:none}}.answer-content ul,.answer-content ol{{padding-left:24px}}
@media(max-width:700px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}.header-top{{display:block}}.theme-switcher{{margin-top:18px}}}}
</style></head><body><main>
<header><div class="header-top"><div class="header-copy"><p class="eyebrow">CIAL Knowledge OS</p><h1>Phase 4 · Reranking & Evidence Selection</h1><p>Offline execution report: Hybrid Retrieval → RRF → Reranking → Evidence Selection → Context → Answer</p></div>
<div class="theme-switcher" role="group" aria-label="Report color theme">
<span class="sr-only">Report theme</span>
<button type="button" data-theme-choice="light" aria-label="Use light theme" aria-pressed="false">Light</button>
<button type="button" data-theme-choice="dark" aria-label="Use dark theme" aria-pressed="false">Dark</button>
<button type="button" data-theme-choice="system" aria-label="Use system theme" aria-pressed="false">System</button>
</div></div></header>
<section><h2>Executive Summary</h2><div class="metrics">{_cards([
("Questions", summary.get("question_count", len(rows))),
("Successful", summary.get("successful_questions", 0)),
("Average context tokens", _number(metrics.get("average_context_tokens", 0))),
("Average selected evidence tokens", _number(metrics.get("average_selected_evidence_tokens", 0))),
("Average token reduction", _number(metrics.get("average_token_reduction_percent", 0)) + "%"),
("Selected chunks", metrics.get("selected_chunk_count", 0)),
("Discarded chunks", metrics.get("discarded_chunk_count", 0)),
("Fallback questions", metrics.get("fallback_question_count", 0)),
("Weak-evidence questions", metrics.get("weak_evidence_question_count", 0)),
("Unsupported queries", metrics.get("unsupported_query_count", 0)),
("Insufficient evidence", metrics.get("insufficient_evidence_count", 0)),
("Extractive fallbacks", metrics.get("extractive_fallback_count", 0)),
("Blocked fallbacks", metrics.get("fallback_blocked_count", 0)),
*indexing_cards,
])}</div><h3>Decision diagnostics</h3><ul>{''.join(diagnostics) or '<li>No diagnostics available.</li>'}</ul></section>
{readiness_html}
{ocr_html}
<section><h2>Answers</h2><p>Full generated answers are rendered below without preview truncation. Evidence selection reduces irrelevant context, not answer depth.</p>{''.join(answer_sections)}</section>
<section><h2>Citations</h2><p>Structured, clickable citation evidence is included with each answer card above.</p></section>
<section><h2>Reranking Trace</h2>{''.join(reranking_sections)}</section>
<section><h2>Evidence Selection</h2>{''.join(selection_sections)}</section>
<section><h2>Token Reduction</h2>{chart_html['tokens']}</section>
<section><h2>Latency Breakdown</h2>{chart_html['latency']}</section>
<section><h2>Evidence Quality</h2>{chart_html['strengths']}{''.join(quality_sections)}</section>
<section><h2>Source Diversity</h2>{chart_html['diversity']}</section>
<section><h2>Selected vs Discarded Chunks</h2>{chart_html['selection']}</section>
<section><h2>Discard Reason Breakdown</h2>{chart_html['discard_reasons']}{_table(
    [
        {"reason": label, "count": value}
        for label, value in charts["discard_reasons"]
    ],
    (("reason", "Exact discard reason"), ("count", "Chunk count")),
)}</section>
<section><h2>Candidate Pool Funnel</h2>{chart_html['funnel']}</section>
<section><h2>Phase 3 vs Phase 4 Comparison</h2><p>{html.escape(comparison_note)}</p></section>
<section><h2>Context and Debug Details</h2>{''.join(debug_sections)}</section>
</main>
<div id="citation-popover" class="citation-popover" role="tooltip" aria-hidden="true" hidden>
  <p class="citation-popover-source" data-field="source"></p>
  <dl class="citation-popover-grid">
    <dt>Page</dt><dd data-field="page"></dd>
    <dt>Chunk</dt><dd data-field="chunk"></dd>
    <dt>Reranker score</dt><dd data-field="score"></dd>
    <dt>Evidence strength</dt><dd data-field="strength"></dd>
    <dt>Retrieved via</dt><dd data-field="retriever"></dd>
  </dl>
  <p class="citation-popover-snippet" data-field="snippet"></p>
  <p class="citation-popover-action" data-field="action"></p>
</div>
<script type="application/json" id="citation-preview-data">{_json_for_html(citation_preview_data)}</script>
<script>{_POPOVER_SCRIPT}</script>
</body></html>"""
    target.write_text(document, encoding="utf-8")
    return target
