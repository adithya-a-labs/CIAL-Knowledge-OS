"""Configuration contracts for the Execution & Observability Framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExecutionOptions:
    enabled: bool = True
    console: bool = True
    rich: str | bool = "auto"
    trace_jsonl: bool = True
    progress_log: bool = True
    telemetry: bool = True
    telemetry_interval_seconds: float = 5.0
    console_refresh_seconds: float = 1.0
    output_dir: Path = Path("outputs/runs")

    @classmethod
    def from_config(cls, config: Any) -> "ExecutionOptions":
        project_root = Path(getattr(config, "project_root", Path.cwd()))
        output = Path(
            getattr(config, "observability_output_dir", None)
            or "outputs/runs"
        ).expanduser()
        if not output.is_absolute():
            output = project_root / output
        return cls(
            enabled=bool(getattr(config, "observability_enabled", True)),
            console=bool(getattr(config, "observability_console", True)),
            rich=getattr(config, "observability_rich", "auto"),
            trace_jsonl=bool(
                getattr(config, "observability_trace_jsonl", True)
            ),
            progress_log=bool(
                getattr(config, "observability_progress_log", True)
            ),
            telemetry=bool(
                getattr(config, "observability_telemetry", True)
            ),
            telemetry_interval_seconds=float(
                getattr(
                    config,
                    "observability_telemetry_interval_seconds",
                    5.0,
                )
            ),
            console_refresh_seconds=float(
                getattr(config, "observability_console_refresh_seconds", 1.0)
            ),
            output_dir=output.resolve(),
        )
