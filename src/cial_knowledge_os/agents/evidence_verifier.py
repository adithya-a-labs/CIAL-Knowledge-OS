"""Deterministic-first claim and optional visual evidence verification."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from .base import Agent, AgentResult
from .state import AgentState

_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[(\d+)\]")


class EvidenceVerifier(Agent):
    name = "evidence_verifier"

    def __init__(self, router: Any | None = None) -> None:
        self.router = router

    def run(self, state: AgentState) -> AgentResult:
        started = perf_counter()
        claims = [
            item.strip()
            for item in _CLAIM_SPLIT.split(state.draft_answer)
            if len(item.split()) >= 4
        ]
        verified: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        evidence_count = len(state.selected_evidence)
        for claim in claims:
            refs = [int(item) for item in _CITATION.findall(claim)]
            invalid = [ref for ref in refs if ref < 1 or ref > evidence_count]
            if invalid:
                mismatches.append({"claim": claim, "citations": invalid})
            elif refs:
                verified.append({"claim": claim, "citations": refs})
            else:
                unsupported.append({"claim": claim, "severity": "medium"})
        total = len(verified) + len(unsupported) + len(mismatches)
        rate = len(verified) / total if total else 0.0

        visual = [item for item in state.selected_evidence if item.is_visual]
        vision_status = "not_applicable"
        vision_details: list[dict[str, Any]] = []
        if visual:
            vision_status = "not_configured"
            can_use = bool(
                self.router
                and self.router.supports(
                    self.name, {"vision", "text"}, vision=True
                )
            )
            paths = [item.image_path for item in visual if item.image_path]
            if can_use and paths:
                try:
                    response = self.router.generate(
                        self.name,
                        "Verify whether the supplied visual evidence is consistent "
                        "with the answer. Report discrepancies only.",
                        required_capabilities={"vision", "text"},
                        images=paths,
                    )
                    vision_status = "verified"
                    vision_details.append(
                        {"review": str(getattr(response, "content", response))}
                    )
                except Exception as exc:
                    vision_status = "unavailable"
                    vision_details.append({"error": str(exc)})
        output = {
            "verified_claims": verified,
            "unsupported_claims": unsupported,
            "citation_mismatches": mismatches,
            "verification_rate": round(rate, 4),
            "passed": bool(total) and not unsupported and not mismatches,
            "vision_verification": {
                "status": vision_status,
                "visual_evidence_count": len(visual),
                "details": vision_details,
            },
        }
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            updated_state=state.evolve(evidence_verification=output),
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )
