"""Backward-compatible Phase 5 JSON/CSV/XLSX/HTML batch exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..batch_qa import collect_batch_answers
from ..reporting.phase5_html import write_phase5_html

PHASE5_COLUMNS = [
    "phase5_enabled", "query_intent", "response_format", "critic_passed",
    "compliance_passed", "risk_passed", "verification_rate",
    "consensus_decision", "revision_used", "final_status",
    "agent_latency_total_ms", "model_map",
]


class Phase5Runner:
    def __init__(self, pipeline: Any, output_dir: str | Path) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)

    @staticmethod
    def _row(response: Mapping[str, Any], question: str) -> dict[str, Any]:
        intent = response.get("query_intent") or {}
        plan = response.get("response_plan") or {}
        critic = response.get("critic_review") or {}
        compliance = response.get("compliance_review") or {}
        risk = response.get("risk_review") or {}
        verification = response.get("evidence_verification") or {}
        consensus = response.get("consensus_decision") or {}
        return {
            "question": question,
            "answer": str(response.get("answer") or ""),
            "answer_status": str(response.get("answer_status") or ""),
            "phase5_enabled": bool(response.get("phase5_enabled")),
            "query_intent": str(intent.get("intent") or ""),
            "response_format": str(plan.get("format") or ""),
            "critic_passed": bool(critic.get("passed")),
            "compliance_passed": bool(compliance.get("passed")),
            "risk_passed": bool(risk.get("passed")),
            "verification_rate": float(verification.get("verification_rate") or 0),
            "consensus_decision": str(consensus.get("decision") or ""),
            "revision_used": bool(response.get("revision_used")),
            "final_status": str(response.get("final_status") or ""),
            "agent_latency_total_ms": float(
                response.get("agent_latency_total_ms") or 0
            ),
            "model_map": json.dumps(response.get("model_map") or {}, ensure_ascii=False),
        }

    def run(self, questions: Iterable[str]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        question_list = list(questions)
        collection = collect_batch_answers(
            pipeline=self.pipeline,
            questions=question_list,
        )
        rows = [dict(item) for item in collection.rows]
        responses = [
            (
                dict(response) | {"question": question}
                if isinstance(response, Mapping)
                else {"question": question, "answer": "", "answer_status": "failed"}
            )
            for question, response in zip(question_list, collection.responses)
        ]
        csv_path = self.output_dir / "results.csv"
        fields = list(collection.columns)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        json_path = self.output_dir / "results.json"
        json_path.write_text(
            json.dumps(responses, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        xlsx_path = self.output_dir / "results.xlsx"
        try:
            import pandas as pd
            pd.DataFrame(rows, columns=fields).to_excel(xlsx_path, index=False)
        except ImportError:
            xlsx_path = Path("")
        html_path = self.output_dir / "report.html"
        write_phase5_html(html_path, responses)
        return {
            "csv": csv_path,
            "json": json_path,
            "xlsx": xlsx_path,
            "html": html_path,
        }
