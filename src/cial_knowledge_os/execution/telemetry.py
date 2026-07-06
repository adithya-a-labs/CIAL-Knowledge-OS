"""Best-effort local machine telemetry with no network communication."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any, Callable


class TelemetryCollector:
    def __init__(
        self,
        *,
        project_path: str | Path | None = None,
        psutil_module: Any | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.project_path = Path(project_path or Path.cwd())
        self._psutil = psutil_module
        self._command_runner = command_runner

    def _load_psutil(self) -> Any | None:
        if self._psutil is not None:
            return self._psutil
        try:
            return importlib.import_module("psutil")
        except ImportError:
            return None

    def _gpu(self) -> dict[str, Any] | None:
        try:
            result = self._command_runner(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            first = result.stdout.strip().splitlines()[0]
            values = [value.strip() for value in first.split(",")]
            return {
                "utilization_percent": float(values[0]),
                "vram_used_mb": float(values[1]),
                "vram_total_mb": float(values[2]),
                "temperature_c": float(values[3]),
            }
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return None

    def collect(self) -> dict[str, Any]:
        telemetry: dict[str, Any] = {"local_only": True}
        psutil = self._load_psutil()
        if psutil is not None:
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage(str(self.project_path.anchor or self.project_path))
                process = psutil.Process()
                telemetry.update(
                    {
                        "cpu_percent": float(psutil.cpu_percent(interval=None)),
                        "ram_used_bytes": int(memory.used),
                        "ram_total_bytes": int(memory.total),
                        "ram_percent": float(memory.percent),
                        "disk_used_bytes": int(disk.used),
                        "disk_free_bytes": int(disk.free),
                        "disk_percent": float(disk.percent),
                        "process_memory_bytes": int(process.memory_info().rss),
                    }
                )
            except (AttributeError, OSError):
                telemetry["system_unavailable"] = True
        else:
            telemetry["psutil_available"] = False
        gpu = self._gpu()
        if gpu is not None:
            telemetry["gpu"] = gpu
        else:
            telemetry["gpu_available"] = False
        return telemetry
