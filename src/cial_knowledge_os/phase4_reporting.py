"""Standalone, offline Phase 4 reports and decision visualizations."""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .phase3_reporting import render_safe_markdown


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _bar_svg(
    values: Sequence[tuple[str, float]],
    *,
    title: str,
    color: str = "#2563eb",
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
        '<style>.title{font:600 16px system-ui;fill:#172033}.label,.value{'
        'font:12px system-ui;fill:#344054}</style>'
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
            _bar_svg(values, title=title, color=color),
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


def _citations(citations: Sequence[Mapping[str, Any]]) -> str:
    if not citations:
        return '<p class="muted">No citations were produced.</p>'
    cards = []
    for citation in citations:
        label = (
            f"{citation.get('source_file') or citation.get('source') or 'Unknown'}"
            f" · Page {citation.get('page_number') or 'N/A'}"
            f" · Chunk {citation.get('chunk_id') or 'N/A'}"
        )
        link = citation.get("pdf_link")
        action = (
            f'<a href="{html.escape(str(link), quote=True)}">Open PDF</a>'
            if link
            else '<span class="muted">No PDF link</span>'
        )
        cards.append(
            f'<div class="citation-card"><strong>{html.escape(label)}</strong>{action}</div>'
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
    answer_sections = []
    reranking_sections = []
    selection_sections = []
    quality_sections = []
    debug_sections = []
    diagnostics = []
    for index, row in enumerate(rows, start=1):
        trace = traces[index - 1] if index <= len(traces) else {}
        question = str(row.get("question") or trace.get("question") or "")
        answer_sections.append(
            f'<article class="answer-card"><p class="eyebrow">Question {index}</p>'
            f"<h3>{html.escape(question)}</h3>"
            f'<div class="answer-content">{render_safe_markdown(str(row.get("answer") or trace.get("answer") or ""))}</div>'
            f"<h4>Citations</h4>{_citations(trace.get('citations') or [])}</article>"
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
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 4 Reranking and Evidence Selection</title>
<style>
:root{{--bg:#f3f6fb;--panel:#fff;--ink:#172033;--muted:#667085;--line:#dfe5ef;--accent:#1d4ed8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}}header{{padding:34px;border-radius:18px;background:linear-gradient(135deg,#102a56,#1d4ed8);color:white}}
h1{{margin:0 0 8px;font-size:32px}}h2{{margin-top:0}}section,.answer-card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 6px 18px #102a560d}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:18px}}.metric{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:14px}}.metric span{{display:block;color:var(--muted);font-size:12px}}.metric strong{{font-size:22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eef3fb;position:sticky;top:0}}
.eyebrow{{text-transform:uppercase;letter-spacing:.08em;color:var(--accent);font-weight:700}}.muted{{color:var(--muted)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#dbeafe;padding:14px;border-radius:9px}}
.citation-list{{display:grid;gap:8px}}.citation-card{{display:flex;justify-content:space-between;gap:12px;border-left:4px solid var(--accent);padding:10px 12px;background:#f8fafc}}a{{color:#1d4ed8}}details{{border:1px solid var(--line);border-radius:9px;padding:10px;margin:10px 0}}summary{{cursor:pointer;font-weight:650}}
.answer-content{{white-space:normal;overflow:visible;max-height:none}}.answer-content ul,.answer-content ol{{padding-left:24px}}code{{background:#eef2ff;padding:2px 5px;border-radius:4px}}@media(max-width:700px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><p class="eyebrow" style="color:#bfdbfe">CIAL Knowledge OS</p><h1>Phase 4 · Reranking & Evidence Selection</h1><p>Offline execution report: Hybrid Retrieval → RRF → Reranking → Evidence Selection → Context → Answer</p></header>
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
])}</div><h3>Decision diagnostics</h3><ul>{''.join(diagnostics) or '<li>No diagnostics available.</li>'}</ul></section>
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
</main></body></html>"""
    target.write_text(document, encoding="utf-8")
    return target
