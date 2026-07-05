"""Reusable, offline-safe decision intelligence visualizations."""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def decision_recommendation(record: Mapping[str, Any]) -> tuple[str, str]:
    status = str(
        record.get("final_status") or record.get("answer_status") or ""
    ).casefold()
    consensus = str(
        _mapping(record.get("consensus_decision")).get("decision") or ""
    ).casefold()
    verification = _number(
        _mapping(record.get("evidence_verification")).get("verification_rate")
    )
    risk = str(_mapping(record.get("risk_review")).get("risk_level") or "").casefold()
    unsupported = _list(
        _mapping(record.get("evidence_verification")).get("unsupported_claims")
    )
    high_unsupported = any(
        str(_mapping(item).get("severity") or "").casefold() == "high"
        for item in unsupported
    )
    if status == "insufficient_evidence":
        return "Insufficient evidence", "insufficient"
    if status == "unsupported_query":
        return "Unsupported query", "unsupported"
    if consensus == "reject" or status == "rejected":
        return "Rejected by validation", "rejected"
    if consensus == "accept" and verification >= 0.85 and risk != "high":
        return "Ready for decision", "ready"
    if consensus == "accept" and verification >= 0.65 and not high_unsupported:
        return "Usable with caution", "caution"
    return "Usable with caution", "caution"


def readiness_score(record: Mapping[str, Any]) -> int:
    evidence = _list(record.get("selected_evidence"))
    compliance = bool(_mapping(record.get("compliance_review")).get("passed"))
    risk = str(_mapping(record.get("risk_review")).get("risk_level") or "high")
    verification = _number(
        _mapping(record.get("evidence_verification")).get("verification_rate")
    )
    severity = str(_mapping(record.get("critic_review")).get("severity") or "high")
    consensus = str(
        _mapping(record.get("consensus_decision")).get("decision") or ""
    )
    scores = [_number(_mapping(item).get("score"), 0.0) for item in evidence]
    sufficiency = min(1.0, len(evidence) / 3) * (
        min(1.0, max(0.0, fmean(scores))) if scores else 0.5
    )
    risk_score = {"low": 1.0, "medium": 0.6, "high": 0.0}.get(risk, 0.3)
    critic_score = {"low": 1.0, "medium": 0.55, "high": 0.0}.get(severity, 0.3)
    consensus_score = {"accept": 1.0, "revise_once": 0.5, "reject": 0.0}.get(
        consensus, 0.0
    )
    result = (
        sufficiency * 0.2
        + float(compliance) * 0.2
        + risk_score * 0.15
        + verification * 0.25
        + critic_score * 0.1
        + consensus_score * 0.1
    )
    return round(max(0.0, min(1.0, result)) * 100)


def render_readiness_badge(record: Mapping[str, Any]) -> str:
    label, kind = decision_recommendation(record)
    score = readiness_score(record)
    return (
        f'<div class="decision-banner {kind}" role="status">'
        f'<div><span class="eyebrow">Decision recommendation</span>'
        f'<strong>{_esc(label)}</strong></div>'
        f'<div class="readiness-score" aria-label="Answer readiness {score} percent">'
        f'<span>{score}</span><small>/100</small></div></div>'
    )


def render_score_card_grid(record: Mapping[str, Any]) -> str:
    verification = _mapping(record.get("evidence_verification"))
    cards = [
        ("Readiness", f"{readiness_score(record)}/100"),
        ("Verification", f"{_number(verification.get('verification_rate')):.0%}"),
        ("Unsupported claims", len(_list(verification.get("unsupported_claims")))),
        ("Risk", _mapping(record.get("risk_review")).get("risk_level", "unknown")),
        ("Consensus", _mapping(record.get("consensus_decision")).get("decision", "unknown")),
        ("Revision used", "Yes" if record.get("revision_used") else "No"),
    ]
    return '<div class="score-grid">' + "".join(
        f'<div class="score-card"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>'
        for label, value in cards
    ) + "</div>"


def _bar_chart(
    rows: Sequence[tuple[str, float]], *, maximum: float | None = None
) -> str:
    max_value = maximum or max((value for _, value in rows), default=1.0) or 1.0
    return '<div class="bars">' + "".join(
        '<div class="bar-row">'
        f'<span title="{_esc(label)}">{_esc(label)}</span>'
        f'<div class="track"><i style="width:{max(0, min(100, value/max_value*100)):.1f}%"></i></div>'
        f'<b>{value:g}</b></div>'
        for label, value in rows
    ) + "</div>"


def render_evidence_strength_chart(record: Mapping[str, Any]) -> str:
    evidence = [_mapping(item) for item in _list(record.get("selected_evidence"))]
    scores = [_number(item.get("score")) for item in evidence if item.get("score") is not None]
    sources = {
        str(item.get("relative_path") or item.get("source") or "")
        for item in evidence
        if item.get("relative_path") or item.get("source")
    }
    cards = [
        ("Selected", len(evidence)),
        ("Average score", f"{fmean(scores):.2f}" if scores else "n/a"),
        ("Strongest", f"{max(scores):.2f}" if scores else "n/a"),
        ("Weakest", f"{min(scores):.2f}" if scores else "n/a"),
        ("Distinct sources", len(sources)),
    ]
    return '<div class="mini-grid">' + "".join(
        f'<div><span>{_esc(k)}</span><strong>{_esc(v)}</strong></div>' for k, v in cards
    ) + "</div>"


def render_verification_chart(record: Mapping[str, Any]) -> str:
    value = _mapping(record.get("evidence_verification"))
    verified = len(_list(value.get("verified_claims")))
    unsupported = len(_list(value.get("unsupported_claims")))
    mismatches = len(_list(value.get("citation_mismatches")))
    rate = _number(value.get("verification_rate"))
    return (
        f'<div class="donut-row"><div class="donut" style="--p:{rate*100:.1f}" '
        f'aria-label="Verification rate {rate:.0%}"><strong>{rate:.0%}</strong></div>'
        + _bar_chart(
            [("Verified", verified), ("Unsupported", unsupported), ("Mismatches", mismatches)]
        )
        + "</div>"
    )


def render_risk_matrix(record: Mapping[str, Any]) -> str:
    categories = [
        "operational", "cybersecurity", "aviation safety", "governance", "compliance"
    ]
    risks = [_mapping(item) for item in _list(_mapping(record.get("risk_review")).get("risks"))]
    by_category = {
        str(item.get("category") or "").replace("_", " ").casefold(): item
        for item in risks
    }
    rows = []
    for category in categories:
        risk = by_category.get(category, {})
        severity = str(risk.get("severity") or "not assessed")
        likelihood = str(risk.get("likelihood") or "not assessed")
        mitigation = str(risk.get("mitigation_status") or "not specified")
        rows.append(
            f"<tr><th>{_esc(category.title())}</th>"
            f'<td><span class="pill {severity.casefold()}">{_esc(severity)}</span></td>'
            f"<td>{_esc(likelihood)}</td><td>{_esc(mitigation)}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Risk</th><th>Severity</th>'
        "<th>Likelihood</th><th>Mitigation</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def render_agent_latency_chart(record: Mapping[str, Any]) -> str:
    labels = {
        "query_analyzer": "Query analyzer",
        "response_planner": "Response planner",
        "prompt_composer": "Prompt composer",
        "draft_generator": "Draft generator",
        "critic_agent": "Critic",
        "compliance_agent": "Compliance",
        "risk_agent": "Risk",
        "evidence_verifier": "Evidence verifier",
        "consensus_engine": "Consensus engine",
    }
    events = _list(_mapping(record.get("phase5_trace")).get("events"))
    totals: Counter[str] = Counter()
    for event in events:
        item = _mapping(event)
        totals[str(item.get("agent_name") or "")] += _number(item.get("latency_ms"))
    return _bar_chart([(label, round(totals[name], 2)) for name, label in labels.items()])


def render_consensus_flow(record: Mapping[str, Any]) -> str:
    consensus = _mapping(record.get("consensus_decision"))
    decision = str(consensus.get("decision") or "unknown")
    final_status = str(consensus.get("final_status") or record.get("final_status") or "unknown")
    revision = bool(record.get("revision_used"))
    nodes = [
        ("Validated", ""),
        ("Revision used" if revision else "No revision", "active" if revision else ""),
        (decision.replace("_", " ").title(), "active"),
        (final_status.replace("_", " ").title(), "final"),
    ]
    return '<div class="flow" aria-label="Consensus flow">' + '<span class="arrow">→</span>'.join(
        f'<span class="flow-node {kind}">{_esc(label)}</span>' for label, kind in nodes
    ) + "</div>"


def render_critic_findings_chart(record: Mapping[str, Any]) -> str:
    issues = _list(_mapping(record.get("critic_review")).get("issues"))
    categories = {
        "Completeness": ("complete", "missing section"),
        "Structure": ("structure", "organization"),
        "Missing caveats": ("caveat",),
        "Repetition": ("repeat",),
        "Weak reasoning": ("reason", "priorit"),
        "Unanswered parts": ("unanswered", "not answer"),
    }
    text = [str(_mapping(item).get("type") or _mapping(item).get("description") or item).casefold() for item in issues]
    counts = [
        (label, float(sum(any(term in issue for term in terms) for issue in text)))
        for label, terms in categories.items()
    ]
    return _bar_chart(counts)


def _answer_sections(answer: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, list[str]]] = [("Answer", [])]
    for line in answer.splitlines():
        if line.lstrip().startswith("#"):
            parts.append((line.lstrip("# ").strip() or "Section", []))
        else:
            parts[-1][1].append(line)
    return [(name, "\n".join(lines)) for name, lines in parts if any(lines)]


def render_citation_coverage_map(record: Mapping[str, Any]) -> str:
    sections = _answer_sections(str(record.get("answer") or ""))
    rows = []
    for name, text in sections:
        claims = max(1, len([x for x in re.split(r"[.!?]+", text) if len(x.split()) >= 4]))
        citations = len(re.findall(r"\[\d+\]", text))
        ratio = citations / claims
        label = "citation-dense" if ratio >= 0.8 else "weakly cited" if ratio > 0 else "uncited major claims"
        rows.append((name, ratio, label))
    return '<div class="coverage-map">' + "".join(
        f'<div><span>{_esc(name)}</span><meter min="0" max="1" value="{min(1, ratio):.2f}"></meter>'
        f'<b>{_esc(label)}</b></div>' for name, ratio, label in rows
    ) + "</div>"


def render_source_diversity_chart(record: Mapping[str, Any]) -> str:
    evidence = [_mapping(item) for item in _list(record.get("selected_evidence"))]
    dimensions = [
        ("Category", "category"), ("Collection", "collection"),
        ("Document type", "document_type"), ("Relative path", "relative_path"),
        ("Source file", "source"),
    ]
    panels = []
    for label, key in dimensions:
        values = Counter(
            str(item.get(key) or _mapping(item.get("metadata")).get(key) or "unknown")
            for item in evidence
        )
        panels.append(
            f'<div class="source-panel"><h4>{_esc(label)}</h4>'
            + _bar_chart([(name, float(count)) for name, count in values.most_common(8)])
            + "</div>"
        )
    return '<div class="source-grid">' + "".join(panels) + "</div>"


def render_modality_mix_chart(record: Mapping[str, Any]) -> str:
    counts = Counter(
        str(_mapping(item).get("modality") or "text").casefold()
        for item in _list(record.get("selected_evidence"))
    )
    modalities = (
        "text", "table", "figure", "image", "screenshot", "diagram", "ocr",
        "scanned_region",
    )
    return _bar_chart([(name.upper(), float(counts[name])) for name in modalities])
