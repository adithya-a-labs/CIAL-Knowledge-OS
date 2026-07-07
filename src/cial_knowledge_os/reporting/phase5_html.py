"""Self-contained Phase 5 HTML decision intelligence report."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .html_citations import (
    render_fallback_citation_chips,
    render_markdown_with_citations,
    render_source_cards,
)
from .phase5_visuals import (
    render_agent_latency_chart,
    render_citation_coverage_map,
    render_consensus_flow,
    render_critic_findings_chart,
    render_evidence_strength_chart,
    render_modality_mix_chart,
    render_readiness_badge,
    render_risk_matrix,
    render_score_card_grid,
    render_source_diversity_chart,
    render_verification_chart,
)


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _panel(title: str, question: str, body: str, *, wide: bool = False) -> str:
    return (
        f'<section class="panel{" wide" if wide else ""}">'
        f"<h3>{_esc(title)}</h3><p class=\"decision-question\">{_esc(question)}</p>"
        f"{body}</section>"
    )


def render_phase5_answer(record: Mapping[str, Any], index: int = 1) -> str:
    citation_values = record.get("citations")
    citations = (
        [item for item in citation_values if isinstance(item, Mapping)]
        if isinstance(citation_values, Sequence)
        and not isinstance(citation_values, (str, bytes))
        else []
    )
    selected_values = record.get("selected_evidence")
    selected_evidence = (
        [item for item in selected_values if isinstance(item, Mapping)]
        if isinstance(selected_values, Sequence)
        and not isinstance(selected_values, (str, bytes))
        else []
    )
    answer_html, citation_records, resolved_ids = (
        render_markdown_with_citations(
            str(record.get("answer") or ""),
            citations,
            selected_evidence,
            question_index=index,
        )
    )
    sources_html = render_source_cards(
        citation_records,
        question_index=index,
        resolved_ids=resolved_ids,
    )
    fallback_chips = (
        render_fallback_citation_chips(
            citation_records,
            question_index=index,
        )
        if not resolved_ids
        else ""
    )
    evidence_items = []
    for item in selected_evidence:
        evidence = item if isinstance(item, Mapping) else {}
        evidence_items.append(
            '<div class="evidence-item">'
            f'<strong>{_esc(evidence.get("modality") or "text")}</strong>'
            f'<span>{_esc(evidence.get("relative_path") or evidence.get("source") or "unknown source")}</span>'
            f'<span>page {_esc(evidence.get("page") or "n/a")}</span>'
            f'<p>{_esc(evidence.get("caption") or evidence.get("ocr_text") or evidence.get("content") or "")}</p>'
            f'<code>{_esc(evidence.get("image_path") or "")}</code>'
            "</div>"
        )
    evidence_detail = (
        "<h4>Selected multimodal evidence</h4>"
        '<div class="evidence-list">'
        + "".join(evidence_items)
        + "</div>"
    )
    detail = (
        f'<details><summary>Detailed agent trace and reviews</summary>'
        f"{evidence_detail}"
        f'<h4>Intent</h4><pre>{_json(record.get("query_intent") or {})}</pre>'
        f'<h4>Response plan</h4><pre>{_json(record.get("response_plan") or {})}</pre>'
        f'<h4>Critic</h4><pre>{_json(record.get("critic_review") or {})}</pre>'
        f'<h4>Compliance</h4><pre>{_json(record.get("compliance_review") or {})}</pre>'
        f'<h4>Risk</h4><pre>{_json(record.get("risk_review") or {})}</pre>'
        f'<h4>Verification</h4><pre>{_json(record.get("evidence_verification") or {})}</pre>'
        f'<h4>Trace</h4><pre>{_json(record.get("phase5_trace") or {})}</pre>'
        "</details>"
    )
    return (
        f'<article class="answer" id="answer-{index}">'
        f'<header><span class="eyebrow">Answer {index}</span>'
        f'<h2>{_esc(record.get("question") or "Question")}</h2></header>'
        f'<section class="final-answer"><span class="eyebrow">Grounded response</span>'
        f'<h3>Final answer</h3><div class="answer-text">{answer_html}</div>'
        f"{fallback_chips}"
        f'<section class="answer-sources"><h3>Sources</h3>{sources_html}</section>'
        "</section>"
        f"{render_readiness_badge(record)}"
        f"{render_score_card_grid(record)}"
        '<details class="diagnostic-sections"><summary>Decision dashboard and agent diagnostics</summary>'
        '<div class="dashboard">'
        + _panel("Evidence strength", "Is the evidence strong enough?", render_evidence_strength_chart(record))
        + _panel("Verification", "Are there unsupported claims?", render_verification_chart(record))
        + _panel("Risk matrix", "What risks are involved?", render_risk_matrix(record), wide=True)
        + _panel("Agent latency", "Where was processing time spent?", render_agent_latency_chart(record))
        + _panel("Consensus flow", "Did the agents agree, and was revision needed?", render_consensus_flow(record))
        + _panel("Critic findings", "What answer-quality gaps remain?", render_critic_findings_chart(record))
        + _panel("Citation coverage", "Which answer sections are supported?", render_citation_coverage_map(record), wide=True)
        + _panel("Modality mix", "Which evidence modalities support the answer?", render_modality_mix_chart(record))
        + _panel("Source diversity", "How concentrated are the supporting sources?", render_source_diversity_chart(record), wide=True)
        + "</div></details>" + detail + "</article>"
    )


def render_phase5_html(records: Sequence[Mapping[str, Any]]) -> str:
    answers = "".join(render_phase5_answer(record, i) for i, record in enumerate(records, 1))
    return f"""<!doctype html>
<html lang="en" data-theme="system"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIAL Knowledge OS — Phase 5 Decision Intelligence</title>
<style>
:root{{--bg:#f3f6fa;--panel:#fff;--text:#172033;--muted:#5b6475;--line:#d8dee9;
--accent:#1769aa;--accent2:#2f8f67;--warn:#a85d00;--danger:#b42318;--track:#e8edf3;
--code:#eef2f7;--code-text:#172033;--citation:#e7f0ff;--citation-text:#174ea6;
--citation-border:#8bb2e8;--source:#f8fafc}}
[data-theme="dark"]{{--bg:#10141c;--panel:#181e29;--text:#eef3fb;--muted:#aeb8ca;
--line:#344054;--accent:#62b0eb;--accent2:#63c69d;--warn:#f0ad4e;--danger:#ff766f;--track:#283142;
--code:#0b1220;--code-text:#dbeafe;--citation:#172554;--citation-text:#bfdbfe;
--citation-border:#3b82f6;--source:#141c29}}
@media(prefers-color-scheme:dark){{[data-theme="system"]{{--bg:#10141c;--panel:#181e29;--text:#eef3fb;--muted:#aeb8ca;--line:#344054;--accent:#62b0eb;--accent2:#63c69d;--warn:#f0ad4e;--danger:#ff766f;--track:#283142;--code:#0b1220;--code-text:#dbeafe;--citation:#172554;--citation-text:#bfdbfe;--citation-border:#3b82f6;--source:#141c29}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
.top{{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;padding:14px max(20px,5vw);background:var(--panel);border-bottom:1px solid var(--line)}}button{{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:8px 12px;border-radius:8px}}
main{{max-width:1280px;margin:auto;padding:28px 20px}}.answer{{margin-bottom:36px}}h1,h2,h3,h4{{line-height:1.2}}.eyebrow,.decision-question,.score-card span,.mini-grid span{{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}}
.decision-banner{{display:flex;justify-content:space-between;align-items:center;padding:18px 22px;border:1px solid var(--line);border-left:7px solid var(--warn);border-radius:12px;background:var(--panel)}}.decision-banner.ready{{border-left-color:var(--accent2)}}.decision-banner.rejected,.decision-banner.unsupported,.decision-banner.insufficient{{border-left-color:var(--danger)}}.decision-banner strong{{display:block;font-size:1.35rem}}.readiness-score span{{font-size:2rem;font-weight:750}}
.final-answer,.panel,details{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:16px}}.final-answer{{padding:26px;border-top:5px solid var(--accent);box-shadow:0 8px 26px #00000012}}.final-answer>h3{{font-size:1.55rem;margin:.25rem 0 1rem}}.answer-text{{font-size:1.02rem;line-height:1.7}}.answer-text h1,.answer-text h2,.answer-text h3,.answer-text h4,.answer-text h5,.answer-text h6{{margin:1.5em 0 .55em}}.answer-text p{{margin:.8em 0}}.answer-text ul,.answer-text ol{{padding-left:1.6rem;margin:.75em 0}}.answer-text li{{margin:.3em 0}}.answer-text code{{padding:.12em .35em;border-radius:4px;background:var(--code);color:var(--code-text)}}.answer-text pre{{white-space:pre;overflow:auto;background:var(--code);color:var(--code-text);padding:14px;border:1px solid var(--line);border-radius:9px}}.answer-text pre code{{padding:0;background:transparent}}.markdown-table-wrap{{overflow:auto;margin:1rem 0;border:1px solid var(--line);border-radius:9px}}.markdown-table{{margin:0}}.markdown-table th{{background:var(--track)}}.score-grid,.mini-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:16px}}.score-card,.mini-grid>div{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}.score-card strong,.mini-grid strong{{display:block;font-size:1.2rem}}
.dashboard{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.panel.wide{{grid-column:1/-1}}.decision-question{{text-transform:none;letter-spacing:0}}.bar-row{{display:grid;grid-template-columns:minmax(90px,1.3fr) 3fr 52px;gap:8px;align-items:center;margin:8px 0}}.bar-row>span{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.track{{height:10px;background:var(--track);border-radius:9px;overflow:hidden}}.track i{{display:block;height:100%;background:var(--accent);border-radius:9px}}.donut-row{{display:grid;grid-template-columns:120px 1fr;gap:20px;align-items:center}}.donut{{--p:0;width:110px;aspect-ratio:1;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle,#0000 54%,var(--panel) 55%),conic-gradient(var(--accent2) calc(var(--p)*1%),var(--track) 0)}}.donut strong{{font-size:1.25rem}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}.pill{{padding:3px 8px;border-radius:12px;background:var(--track)}}.pill.high{{color:var(--danger)}}.pill.medium{{color:var(--warn)}}.pill.low{{color:var(--accent2)}}.flow{{display:flex;align-items:center;gap:9px;overflow:auto}}.flow-node{{white-space:nowrap;border:1px solid var(--line);border-radius:20px;padding:7px 11px}}.flow-node.active{{border-color:var(--accent)}}.flow-node.final{{background:var(--accent);color:white}}.arrow{{color:var(--muted)}}.coverage-map>div{{display:grid;grid-template-columns:1fr 2fr 150px;gap:10px;margin:9px 0}}meter{{width:100%;accent-color:var(--accent2)}}.source-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}.citation-chips{{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:16px 0 4px}}.citation-chips-label{{color:var(--muted);font-size:.8rem;font-weight:700}}.answer-sources{{margin-top:26px;padding-top:18px;border-top:1px solid var(--line)}}.source-card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.source-card{{scroll-margin-top:80px;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:9px;padding:13px;background:var(--source)}}.source-card:target{{outline:3px solid var(--accent);outline-offset:3px}}.source-card-head{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.citation-reference{{font-weight:800;color:var(--accent)}}.source-used,.source-unused{{font-size:.72rem;padding:2px 7px;border-radius:999px;background:var(--track);color:var(--muted)}}.source-used{{color:var(--accent2)}}.source-card h4{{margin:.6rem 0 .25rem;overflow-wrap:anywhere}}.source-meta{{color:var(--muted);font-size:.8rem;margin:.25rem 0}}.source-preview{{display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden;margin:.65rem 0;white-space:pre-wrap}}.source-action{{font-size:.82rem;font-weight:700;color:var(--accent)}}.citation-chip{{display:inline-flex;align-items:center;padding:1px 6px;border:1px solid var(--citation-border);border-radius:999px;background:var(--citation);color:var(--citation-text);font-size:.78em;font-weight:800;line-height:1.4;text-decoration:none;vertical-align:.08em;white-space:nowrap}}.citation-chip:hover{{filter:brightness(.96)}}.unresolved-citation{{border-style:dashed;color:var(--danger);cursor:help}}.diagnostic-sections>summary{{font-size:1.05rem}}.evidence-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}.evidence-item{{border:1px solid var(--line);border-radius:8px;padding:10px}}.evidence-item span,.evidence-item code{{display:block;color:var(--muted);overflow-wrap:anywhere}}pre{{overflow:auto;background:var(--bg);padding:12px;border-radius:8px}}summary{{cursor:pointer;font-weight:700}}.muted{{color:var(--muted)}}
@media(max-width:760px){{.dashboard{{grid-template-columns:1fr}}.panel.wide{{grid-column:auto}}.donut-row{{grid-template-columns:1fr}}.coverage-map>div{{grid-template-columns:1fr}}}}
@media print{{.top button{{display:none}}.top{{position:static}}.answer{{break-after:page}}details{{display:block}}}}
</style></head><body><nav class="top"><strong>CIAL Knowledge OS · Phase 5</strong>
<button id="theme" type="button" aria-label="Change color theme">Theme: system</button></nav>
<main><h1>Decision Intelligence Dashboard</h1>{answers}</main>
<script>
(()=>{{const root=document.documentElement,b=document.getElementById('theme');
const values=['system','light','dark'];let current=localStorage.getItem('cial-theme')||'system';
const apply=v=>{{current=v;root.dataset.theme=v;b.textContent='Theme: '+v;localStorage.setItem('cial-theme',v)}};
apply(current);b.addEventListener('click',()=>apply(values[(values.indexOf(current)+1)%values.length]));}})();
</script></body></html>"""


def write_phase5_html(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_phase5_html(records), encoding="utf-8")
    return target
