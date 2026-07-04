"""Durable per-question checkpoints for long-running Phase 4 batches."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(key): _json_value(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_question(question: str) -> str:
    """Return stable whitespace/case normalization for checkpoint identity.

    The input is one question and the output is a case-folded single-line
    representation. Original text remains in artifacts. Normalization is used
    only for hashing and resume validation, so answer behavior is unchanged.
    """

    return " ".join(str(question).split()).casefold()


@dataclass(frozen=True, slots=True)
class QuestionIdentity:
    """Identify one question occurrence safely across interrupted runs.

    ``index`` is the original one-based position. ``question_hash`` represents
    normalized text, while ``key`` combines both. Duplicate question text is
    therefore checkpointed and resumed independently.
    """

    index: int
    question: str
    normalized_question: str
    question_hash: str
    key: str

    @classmethod
    def create(cls, index: int, question: str) -> "QuestionIdentity":
        """Build an indexed identity from one original question occurrence."""

        if index <= 0:
            raise ValueError("Question identity index must be greater than zero.")
        normalized = normalize_question(question)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(
            index=index,
            question=str(question),
            normalized_question=normalized,
            question_hash=digest,
            key=f"{index}:{digest}",
        )


class Phase4CheckpointManager:
    """Persist and restore one Phase 4 run at per-question granularity.

    Inputs are an existing or newly created run directory. The manager writes
    ``partial_results.csv``, append-only result/retrieval JSONL files, and an
    atomically replaced ``checkpoint.json``. Outputs are checkpoint identities
    and prior successful row/response pairs used by ``Phase4Runner`` to rebuild
    normal final artifacts. No Phase 1--3 paths or formats are changed.
    """

    def __init__(self, run_path: str | Path) -> None:
        self.run_path = Path(run_path).expanduser().resolve()
        self.partial_results_csv = self.run_path / "partial_results.csv"
        self.partial_results_jsonl = self.run_path / "partial_results.jsonl"
        self.partial_retrieval_jsonl = (
            self.run_path / "partial_retrieval.jsonl"
        )
        self.checkpoint_json = self.run_path / "checkpoint.json"
        self.identities: tuple[QuestionIdentity, ...] = ()
        self._checkpoint: dict[str, Any] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._responses: dict[str, Mapping[str, Any] | None] = {}

    @staticmethod
    def build_identities(
        questions: Sequence[str],
    ) -> tuple[QuestionIdentity, ...]:
        """Return indexed/hash identities preserving every input occurrence."""

        return tuple(
            QuestionIdentity.create(index, question)
            for index, question in enumerate(questions, start=1)
        )

    def initialize(
        self,
        questions: Sequence[str],
        *,
        config: Any,
        resume: bool,
    ) -> None:
        """Create or validate checkpoint state for the complete question list.

        ``questions`` must be the original unsliced occurrence order for this
        run. ``config`` is serialized as a reproducibility snapshot. Resume
        requires an existing checkpoint whose indexed hashes exactly match,
        preventing accidental reuse with a reordered or different question
        file.
        """

        self.run_path.mkdir(parents=True, exist_ok=True)
        self.identities = self.build_identities(questions)
        if resume:
            if not self.checkpoint_json.is_file():
                raise FileNotFoundError(
                    "Cannot resume because checkpoint.json is missing from "
                    f"{self.run_path}."
                )
            self._checkpoint = json.loads(
                self.checkpoint_json.read_text(encoding="utf-8")
            )
            stored_keys = [
                str(item.get("key") or "")
                for item in self._checkpoint.get("question_manifest") or []
            ]
            current_keys = [item.key for item in self.identities]
            if stored_keys != current_keys:
                raise ValueError(
                    "Resume question list does not match checkpoint order and "
                    "content. Use the same question file and --max-questions "
                    "settings as the original run."
                )
            self._results = self._load_latest(
                self.partial_results_jsonl,
                value_key="row",
            )
            self._responses = self._load_latest(
                self.partial_retrieval_jsonl,
                value_key="response",
            )
            return
        if self.checkpoint_json.exists():
            raise FileExistsError(
                f"Checkpoint already exists in new run path: {self.run_path}"
            )
        self._checkpoint = {
            "schema_version": "phase4-checkpoint-v1",
            "run_id": self.run_path.name,
            "run_path": str(self.run_path),
            "status": "in_progress",
            "question_count": len(self.identities),
            "question_manifest": [
                _json_value(identity) for identity in self.identities
            ],
            "completed_questions": [],
            "failed_questions": [],
            "last_completed_index": 0,
            "config_snapshot": _json_value(config),
            "timestamp": self._timestamp(),
        }
        self._write_checkpoint()

    def pending(self) -> tuple[QuestionIdentity, ...]:
        """Return occurrences not successfully completed in this checkpoint."""

        completed = {
            str(item.get("key") or "")
            for item in self._checkpoint.get("completed_questions") or []
        }
        return tuple(
            identity for identity in self.identities
            if identity.key not in completed
        )

    def completed_records(
        self,
    ) -> tuple[list[dict[str, Any]], list[Mapping[str, Any] | None]]:
        """Return successful checkpoint rows/responses in original order."""

        completed = {
            str(item.get("key") or "")
            for item in self._checkpoint.get("completed_questions") or []
        }
        rows: list[dict[str, Any]] = []
        responses: list[Mapping[str, Any] | None] = []
        for identity in self.identities:
            if identity.key not in completed:
                continue
            row = self._results.get(identity.key)
            if row is None:
                raise RuntimeError(
                    "Checkpoint marks a question completed but its partial "
                    f"result is missing: {identity.key}"
                )
            rows.append(dict(row))
            responses.append(self._responses.get(identity.key))
        return rows, responses

    def record(
        self,
        identity: QuestionIdentity,
        row: dict[str, Any],
        response: Mapping[str, Any] | None,
    ) -> None:
        """Durably record one attempt and refresh all checkpoint summaries.

        The indexed identity, batch row, and optional full response are inputs.
        JSONL records are appended before the checkpoint pointer is atomically
        replaced. The partial CSV is rewritten from latest attempts so it
        remains directly inspectable during a run.
        """

        row["__checkpoint_key"] = identity.key
        row["__checkpoint_index"] = identity.index
        result_record = {
            "key": identity.key,
            "index": identity.index,
            "question_hash": identity.question_hash,
            "normalized_question": identity.normalized_question,
            "timestamp": self._timestamp(),
            "row": _json_value(row),
        }
        retrieval_record = {
            "key": identity.key,
            "index": identity.index,
            "question_hash": identity.question_hash,
            "timestamp": self._timestamp(),
            "response": _json_value(response),
        }
        self._append_jsonl(self.partial_results_jsonl, result_record)
        self._append_jsonl(
            self.partial_retrieval_jsonl,
            retrieval_record,
        )
        self._results[identity.key] = dict(result_record["row"])
        self._responses[identity.key] = retrieval_record["response"]
        self._write_partial_csv()

        completed_by_key = {
            str(item.get("key") or ""): dict(item)
            for item in self._checkpoint.get("completed_questions") or []
        }
        failed_by_key = {
            str(item.get("key") or ""): dict(item)
            for item in self._checkpoint.get("failed_questions") or []
        }
        identity_payload = dict(_json_value(identity))
        if row.get("status") == "success":
            completed_by_key[identity.key] = identity_payload
            failed_by_key.pop(identity.key, None)
        else:
            completed_by_key.pop(identity.key, None)
            failed_by_key[identity.key] = identity_payload | {
                "error": str(row.get("error") or "Unknown batch failure"),
                "answer_status": str(row.get("answer_status") or ""),
            }
        self._checkpoint.update(
            {
                "status": "in_progress",
                "completed_questions": self._ordered_values(completed_by_key),
                "failed_questions": self._ordered_values(failed_by_key),
                "last_completed_index": max(
                    (
                        int(item.get("index") or 0)
                        for item in completed_by_key.values()
                    ),
                    default=0,
                ),
                "timestamp": self._timestamp(),
            }
        )
        self._write_checkpoint()

    def finalize(self, artifacts: Mapping[str, Any]) -> None:
        """Mark checkpoint execution complete while retaining resumable failures."""

        self._checkpoint.update(
            {
                "status": "completed_with_failures"
                if self._checkpoint.get("failed_questions")
                else "completed",
                "final_artifacts": _json_value(artifacts),
                "timestamp": self._timestamp(),
            }
        )
        self._write_checkpoint()

    def _load_latest(
        self,
        path: Path,
        *,
        value_key: str,
    ) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        if not path.is_file():
            return latest
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A process can terminate during its final append. Earlier
                # corrupt lines indicate real checkpoint damage; only an
                # incomplete last line is safe to ignore and retry.
                if line_number == len(lines):
                    break
                raise
            latest[str(record["key"])] = record.get(value_key)
        return latest

    def _write_partial_csv(self) -> None:
        ordered = [
            self._results[identity.key]
            for identity in self.identities
            if identity.key in self._results
        ]
        if not ordered:
            return
        fieldnames = [
            key for key in ordered[0]
            if not str(key).startswith("__checkpoint_")
        ]
        temporary = self.partial_results_csv.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(self.partial_results_csv)

    def _write_checkpoint(self) -> None:
        temporary = self.checkpoint_json.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                _json_value(self._checkpoint),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.checkpoint_json)

    @staticmethod
    def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_value(value),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()

    @staticmethod
    def _ordered_values(
        values: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            (dict(item) for item in values.values()),
            key=lambda item: int(item.get("index") or 0),
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()
