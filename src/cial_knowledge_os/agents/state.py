"""Serializable Phase 5 state and multimodal evidence contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

EvidenceModality = Literal[
    "text", "table", "figure", "image", "screenshot", "diagram", "ocr",
    "scanned_region",
]
_MODALITIES = {
    "text", "table", "figure", "image", "screenshot", "diagram", "ocr",
    "scanned_region",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(slots=True)
class Evidence:
    """One text or visual evidence item.

    ``content`` carries text, table text, OCR, or a textual representation.
    Visual fields are optional, making existing Phase 4 text chunks valid input.
    Unknown source fields are retained in ``metadata``.
    """

    evidence_id: str
    source: str = ""
    relative_path: str = ""
    page: int | None = None
    chunk_id: str = ""
    modality: EvidenceModality = "text"
    content: str = ""
    image_path: str = ""
    ocr_text: str = ""
    caption: str = ""
    bbox: tuple[float, float, float, float] | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must not be blank.")
        normalized = str(self.modality).strip().casefold()
        if normalized not in _MODALITIES:
            raise ValueError(f"Unsupported evidence modality: {self.modality}")
        self.modality = normalized  # type: ignore[assignment]
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain four coordinates.")
            self.bbox = tuple(float(value) for value in self.bbox)
        if self.score is not None:
            self.score = float(self.score)
        if self.page is not None:
            self.page = int(self.page)

    @property
    def textual_content(self) -> str:
        return self.content or self.ocr_text or self.caption

    @property
    def is_visual(self) -> bool:
        return self.modality in {
            "figure", "image", "screenshot", "diagram", "scanned_region"
        } or bool(self.image_path)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, index: int = 0
    ) -> "Evidence":
        metadata_value = value.get("metadata")
        metadata = (
            dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        )
        source = str(
            value.get("source")
            or metadata.get("source")
            or value.get("file_name")
            or metadata.get("file_name")
            or ""
        )
        relative_path = str(
            value.get("relative_path")
            or metadata.get("relative_path")
            or metadata.get("source_file")
            or source
        )
        page = (
            value.get("page")
            if value.get("page") is not None
            else value.get("page_number", metadata.get("page_number"))
        )
        chunk_id = str(
            value.get("chunk_id") or metadata.get("chunk_id") or index
        )
        known = {
            "evidence_id", "source", "relative_path", "page", "page_number",
            "chunk_id", "modality", "content", "text", "image_path",
            "ocr_text", "caption", "bbox", "score", "reranker_score", "metadata",
        }
        metadata.update(
            {str(key): item for key, item in value.items() if key not in known}
        )
        raw_bbox = value.get("bbox", metadata.get("bbox"))
        bbox = (
            tuple(raw_bbox)  # type: ignore[arg-type]
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4
            else None
        )
        return cls(
            evidence_id=str(
                value.get("evidence_id")
                or f"{relative_path or source or 'evidence'}:{page or 0}:{chunk_id}"
            ),
            source=source,
            relative_path=relative_path,
            page=int(page) if page not in (None, "") else None,
            chunk_id=chunk_id,
            modality=str(  # type: ignore[arg-type]
                value.get("modality") or metadata.get("modality") or "text"
            ),
            content=str(
                value.get("content")
                or value.get("text")
                or metadata.get("content")
                or ""
            ),
            image_path=str(
                value.get("image_path") or metadata.get("image_path") or ""
            ),
            ocr_text=str(
                value.get("ocr_text") or metadata.get("ocr_text") or ""
            ),
            caption=str(
                value.get("caption") or metadata.get("caption") or ""
            ),
            bbox=bbox,
            score=(
                value.get("score")
                if value.get("score") is not None
                else value.get("reranker_score")
            ),
            metadata=metadata,
        )


@dataclass(slots=True)
class AgentState:
    """Explicit state shared between Phase 5 agents."""

    question: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: dict[str, Any] = field(default_factory=dict)
    phase4_answer_status: str = ""
    query_intent: dict[str, Any] = field(default_factory=dict)
    response_plan: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[Evidence] = field(default_factory=list)
    selected_evidence: list[Evidence] = field(default_factory=list)
    evidence_quality: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    composed_prompt: str = ""
    draft_answer: str = ""
    critic_review: dict[str, Any] = field(default_factory=dict)
    compliance_review: dict[str, Any] = field(default_factory=dict)
    risk_review: dict[str, Any] = field(default_factory=dict)
    evidence_verification: dict[str, Any] = field(default_factory=dict)
    consensus_decision: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank.")
        self.retrieved_chunks = [
            item if isinstance(item, Evidence) else Evidence.from_mapping(item, index=i)
            for i, item in enumerate(self.retrieved_chunks)
        ]
        self.selected_evidence = [
            item if isinstance(item, Evidence) else Evidence.from_mapping(item, index=i)
            for i, item in enumerate(self.selected_evidence)
        ]

    def evolve(self, **changes: Any) -> "AgentState":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_phase4(
        cls,
        question: str,
        response: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> "AgentState":
        selected = response.get("selected_evidence") or []
        retrieved = response.get("retrieved") or response.get("candidate_pool") or []
        return cls(
            question=question,
            run_id=run_id or uuid4().hex,
            config=dict(config or {}),
            phase4_answer_status=str(response.get("answer_status") or ""),
            retrieved_chunks=[
                Evidence.from_mapping(item, index=i)
                for i, item in enumerate(retrieved)
                if isinstance(item, Mapping)
            ],
            selected_evidence=[
                Evidence.from_mapping(item, index=i)
                for i, item in enumerate(selected)
                if isinstance(item, Mapping)
            ],
            evidence_quality=dict(response.get("evidence_quality") or {}),
            citations=[
                dict(item) for item in response.get("citations") or []
                if isinstance(item, Mapping)
            ],
            draft_answer=str(
                response.get("raw_answer") or response.get("answer") or ""
            ),
            metadata={"phase4_response": dict(response)},
        )
