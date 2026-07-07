from __future__ import annotations

from cial_knowledge_os.reporting.phase5_html import render_phase5_html


def _record() -> dict:
    return {
        "question": "How should controls be implemented?",
        "answer": (
            "# Implementation plan\n\n"
            "Controls should be verified **before deployment** [1].\n\n"
            "- Assign an owner\n"
            "- Record the review\n\n"
            "| Phase | Owner |\n"
            "| --- | --- |\n"
            "| Review | Safety |\n\n"
            "Use *risk-based* checks and `signed` records [Source 2].\n\n"
            "A missing reference remains visible [99].\n\n"
            "```html\n<script>alert('unsafe')</script>\n```"
        ),
        "citations": [
            {
                "reference_id": 1,
                "source": "CISG-2026-01.pdf",
                "source_file": "CISG-2026-01.pdf",
                "page_number": 47,
                "chunk_id": "151",
                "score": 0.665,
            },
            {
                "reference_id": 2,
                "source": "airport-controls.pdf",
                "source_file": "airport-controls.pdf",
                "page_number": 12,
                "chunk_id": "control-12",
                "score": 0.82,
            },
        ],
        "selected_evidence": [
            {
                "source": "CISG-2026-01.pdf",
                "page": 47,
                "chunk_id": "151",
                "score": 0.665,
                "modality": "text",
                "content": "Independent verification is required before deployment.",
            },
            {
                "source": "airport-controls.pdf",
                "page": 12,
                "chunk_id": "control-12",
                "score": 0.82,
                "modality": "table",
                "content": "Control owners must record and sign completed reviews.",
            },
        ],
        "critic_review": {
            "passed": True,
            "severity": "low",
            "issues": [],
        },
        "compliance_review": {"passed": True},
        "risk_review": {"passed": True, "risk_level": "low", "risks": []},
        "evidence_verification": {
            "passed": True,
            "verification_rate": 1,
            "verified_claims": [{"citations": [1, 2]}],
            "unsupported_claims": [],
            "citation_mismatches": [],
        },
        "consensus_decision": {
            "decision": "accept",
            "final_status": "answered",
        },
        "phase5_trace": {"events": []},
    }


def test_phase5_answer_renders_safe_enterprise_markdown() -> None:
    page = render_phase5_html([_record()])

    assert "<h1>Implementation plan</h1>" in page
    assert "<ul><li>Assign an owner</li><li>Record the review</li></ul>" in page
    assert '<table class="markdown-table">' in page
    assert "<th>Phase</th>" in page
    assert "<td>Safety</td>" in page
    assert "<strong>before deployment</strong>" in page
    assert "<em>risk-based</em>" in page
    assert "<code>signed</code>" in page
    assert "&lt;script&gt;alert" in page
    assert "<script>alert" not in page


def test_inline_citations_link_to_source_cards_and_mark_unresolved() -> None:
    page = render_phase5_html([_record()])

    assert 'class="citation-chip" href="#q1-source-1"' in page
    assert 'class="citation-chip" href="#q1-source-2"' in page
    assert ">[1]</a>" in page
    assert ">[Source 2]</a>" in page
    assert 'class="citation-chip unresolved-citation"' in page
    assert "[99]</span>" in page
    assert 'id="q1-source-1"' in page
    assert 'id="q1-source-2"' in page


def test_missing_markers_get_phase4_compatible_fallback_chips() -> None:
    record = _record()
    record["answer"] = "The evidence supports a controlled rollout."

    page = render_phase5_html([record])

    assert 'class="citation-chips"' in page
    assert "Supporting sources:" in page
    assert 'class="citation-chip" href="#q1-source-1"' in page
    assert 'class="citation-chip" href="#q1-source-2"' in page


def test_source_cards_include_provenance_score_modality_and_preview() -> None:
    page = render_phase5_html([_record()])

    assert "Source 1" in page
    assert "CISG-2026-01.pdf" in page
    assert "Page 47 | Chunk 151 | Score 0.6650 | Modality text" in page
    assert "Independent verification is required before deployment." in page
    assert "Page 12 | Chunk control-12 | Score 0.8200 | Modality table" in page


def test_phase5_dashboard_and_agent_diagnostics_remain_available() -> None:
    page = render_phase5_html([_record()])

    assert "Decision Intelligence Dashboard" in page
    assert "Decision dashboard and agent diagnostics" in page
    assert "Evidence strength" in page
    assert "Verification" in page
    assert "Risk matrix" in page
    assert "Agent latency" in page
    assert "Consensus flow" in page
    assert "Detailed agent trace and reviews" in page
    assert page.index('class="final-answer"') < page.index(
        '<div class="decision-banner'
    )
