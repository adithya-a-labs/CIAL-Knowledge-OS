"""Version-safe loaders for offline evaluation benchmark records."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _split_cell(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _boolean(value: Any, *, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    expected_answer: str
    expected_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()
    expected_behavior: str = ""
    category: str = "uncategorized"
    difficulty: str = "unknown"
    should_answer: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Benchmark:
    questions: tuple[BenchmarkQuestion, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def load_benchmark(
    csv_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> Benchmark:
    """Load the current CISG schema and compatible future benchmark schemas."""

    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("Benchmark CSV must contain a 'question' column.")
        questions = []
        for index, row in enumerate(reader, start=1):
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            expected_answer = str(
                row.get("expected_answer") or row.get("answer") or ""
            ).strip()
            known = {
                "question_id", "question", "expected_answer", "answer",
                "expected_keywords", "forbidden_keywords", "expected_behavior",
                "category", "difficulty", "should_answer",
            }
            questions.append(
                BenchmarkQuestion(
                    question_id=str(row.get("question_id") or index),
                    question=question,
                    expected_answer=expected_answer,
                    expected_keywords=_split_cell(row.get("expected_keywords")),
                    forbidden_keywords=_split_cell(row.get("forbidden_keywords")),
                    expected_behavior=str(row.get("expected_behavior") or ""),
                    category=str(row.get("category") or "uncategorized"),
                    difficulty=str(row.get("difficulty") or "unknown"),
                    should_answer=_boolean(row.get("should_answer")),
                    metadata={
                        key: value for key, value in row.items() if key not in known
                    },
                )
            )
    if not questions:
        raise ValueError(f"Benchmark contains no questions: {source}")

    metadata: dict[str, Any] = {}
    if metadata_path is not None:
        metadata_source = Path(metadata_path).expanduser().resolve()
        metadata = json.loads(metadata_source.read_text(encoding="utf-8-sig"))
    return Benchmark(tuple(questions), metadata, source)
