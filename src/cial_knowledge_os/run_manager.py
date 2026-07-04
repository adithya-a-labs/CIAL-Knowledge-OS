"""Isolated, non-overwriting Phase 3 run lifecycle and artifact paths."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import Phase3Config, RunArtifactNames


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved paths for every configured artifact in a run."""

    root: Path
    results_csv: Path
    results_xlsx: Path
    report_html: Path
    config_json: Path
    summary_json: Path
    retrieval_json: Path
    metrics_json: Path
    logs: Path
    figures: Path
    context: Path


class RunManager:
    """Create one run directory and own all artifact path construction."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        phase_output_name: str,
        run_prefix: str,
        timestamp_format: str,
        artifact_names: RunArtifactNames,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.phase_output_name = phase_output_name
        self.run_prefix = run_prefix
        self.timestamp_format = timestamp_format
        self.artifact_names = artifact_names
        self.paths: RunPaths | None = None

    @classmethod
    def from_config(cls, config: Phase3Config) -> "RunManager":
        return cls(
            output_root=config.output_root,
            phase_output_name=config.phase_output_name,
            run_prefix=config.run_prefix,
            timestamp_format=config.run_timestamp_format,
            artifact_names=config.artifact_names,
        )

    @classmethod
    def from_existing(
        cls,
        config: Phase3Config,
        run_path: str | Path,
    ) -> "RunManager":
        """Attach the configured artifact contract to an existing run folder.

        Inputs are a Phase 3-compatible config and resume directory. The output
        is a normal ``RunManager`` whose paths point at that directory; missing
        context/figure folders are recreated without deleting or overwriting
        checkpoint data. Normal callers continue using :meth:`from_config`.
        """

        root = Path(run_path).expanduser()
        if not root.is_absolute():
            root = config.project_root / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Resume run folder not found: {root}")
        manager = cls.from_config(config)
        names = config.artifact_names
        figures = (root / names.figures_dir).resolve()
        context = (root / names.context_dir).resolve()
        figures.mkdir(exist_ok=True)
        context.mkdir(exist_ok=True)
        manager.paths = RunPaths(
            root=root,
            results_csv=(root / names.results_csv).resolve(),
            results_xlsx=(root / names.results_xlsx).resolve(),
            report_html=(root / names.report_html).resolve(),
            config_json=(root / names.config_json).resolve(),
            summary_json=(root / names.summary_json).resolve(),
            retrieval_json=(root / names.retrieval_json).resolve(),
            metrics_json=(root / names.metrics_json).resolve(),
            logs=(root / names.logs).resolve(),
            figures=figures,
            context=context,
        )
        return manager

    def create(self, *, timestamp: datetime | None = None) -> RunPaths:
        """Create a timestamped run folder without overwriting an earlier run."""

        if self.paths is not None:
            return self.paths
        effective_time = timestamp or datetime.now().astimezone()
        stamp = effective_time.strftime(self.timestamp_format)
        phase_root = self.output_root / self.phase_output_name
        phase_root.mkdir(parents=True, exist_ok=True)
        base_name = f"{self.run_prefix}_{stamp}"
        candidate = phase_root / base_name
        collision = 1
        while True:
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                collision += 1
                candidate = phase_root / f"{base_name}_{collision:02d}"
        names = self.artifact_names
        figures = candidate / names.figures_dir
        context = candidate / names.context_dir
        figures.mkdir()
        context.mkdir()
        self.paths = RunPaths(
            root=candidate.resolve(),
            results_csv=(candidate / names.results_csv).resolve(),
            results_xlsx=(candidate / names.results_xlsx).resolve(),
            report_html=(candidate / names.report_html).resolve(),
            config_json=(candidate / names.config_json).resolve(),
            summary_json=(candidate / names.summary_json).resolve(),
            retrieval_json=(candidate / names.retrieval_json).resolve(),
            metrics_json=(candidate / names.metrics_json).resolve(),
            logs=(candidate / names.logs).resolve(),
            figures=figures.resolve(),
            context=context.resolve(),
        )
        return self.paths

    def require_paths(self) -> RunPaths:
        if self.paths is None:
            raise RuntimeError("Call RunManager.create() before requesting paths.")
        return self.paths

    def write_json(self, path: str | Path, value: Any) -> Path:
        """Atomically write a UTF-8 JSON artifact."""

        target = Path(path).expanduser().resolve()
        run_root = self.require_paths().root
        if target != run_root and run_root not in target.parents:
            raise ValueError("Run artifacts must remain inside the run directory.")
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                _json_value(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def write_effective_config(
        self,
        config: Phase3Config,
        *,
        benchmark: Mapping[str, Any] | None = None,
        run_overrides: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> Path:
        """Persist all settings needed to reproduce a run."""

        payload = _json_value(config)
        if isinstance(payload, dict):
            payload["timestamp"] = timestamp or datetime.now().astimezone().isoformat()
            payload["benchmark"] = dict(benchmark or {})
            payload["run_overrides"] = dict(run_overrides or {})
        return self.write_json(self.require_paths().config_json, payload)

    def context_path(self, question_index: int, question: str) -> Path:
        """Return a stable, readable Markdown path for one question trace."""

        if question_index <= 0:
            raise ValueError("question_index must be greater than zero.")
        stem = re.sub(r"[^\w.-]+", "_", question.strip(), flags=re.UNICODE)
        stem = stem.strip("._")[:64] or "question"
        filename = self.artifact_names.context_file_template.format(
            index=question_index,
            slug=stem,
        )
        return (self.require_paths().context / filename).resolve()
