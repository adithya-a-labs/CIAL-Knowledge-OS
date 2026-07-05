"""Local-only CPU, memory, disk, process, and optional NVIDIA telemetry."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable


class TelemetryCollector:
    """Collect machine metrics without transmitting them."""

    def __init__(
        self,
        *,
        psutil_module: Any | None = None,
        command_runner: Callable[..., Any] | None = None,
        disk_path: str | Path | None = None,
    ) -> None:
        if psutil_module is None:
            try:
                import psutil as psutil_module
            except ImportError as exc:
                raise RuntimeError(
                    "Live telemetry requires psutil. Install project requirements."
                ) from exc
        self.psutil = psutil_module
        self.command_runner = command_runner or subprocess.run
        self.disk_path = Path(disk_path or Path.cwd().anchor or Path.cwd())
        self.process = self.psutil.Process()
        self.model_stats: dict[str, Any] = {
            "current_model": "",
            "model_latency_ms": None,
            "tokens_generated": None,
            "tokens_per_second": None,
        }

    def update_model(
        self,
        *,
        name: str = "",
        latency_ms: float | None = None,
        tokens_generated: int | None = None,
    ) -> None:
        throughput = None
        if tokens_generated is not None and latency_ms and latency_ms > 0:
            throughput = round(tokens_generated / (latency_ms / 1000), 3)
        self.model_stats = {
            "current_model": name,
            "model_latency_ms": latency_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": throughput,
        }

    def _gpu(self) -> dict[str, Any]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,name",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = self.command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            lines = [
                line.strip()
                for line in str(result.stdout or "").splitlines()
                if line.strip()
            ]
            devices = []
            for line in lines:
                parts = [part.strip() for part in line.split(",", 3)]
                if len(parts) != 4:
                    continue
                devices.append(
                    {
                        "usage_percent": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                        "name": parts[3],
                    }
                )
            if not devices:
                return {"available": False, "devices": []}
            return {
                "available": True,
                "devices": devices,
                "usage_percent": round(
                    sum(item["usage_percent"] for item in devices) / len(devices),
                    2,
                ),
                "memory_used_mb": sum(
                    item["memory_used_mb"] for item in devices
                ),
                "memory_total_mb": sum(
                    item["memory_total_mb"] for item in devices
                ),
            }
        except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
            return {"available": False, "devices": []}

    def collect(self) -> dict[str, Any]:
        try:
            memory = self.psutil.virtual_memory()
            disk = self.psutil.disk_usage(str(self.disk_path))
            process_memory = self.process.memory_info()
            base = {
                "cpu_percent": float(self.psutil.cpu_percent(interval=None)),
                "ram_percent": float(memory.percent),
                "ram_used_bytes": int(memory.used),
                "ram_total_bytes": int(memory.total),
                "disk_percent": float(disk.percent),
                "disk_used_bytes": int(disk.used),
                "disk_total_bytes": int(disk.total),
                "process_memory_bytes": int(process_memory.rss),
                "gpu": self._gpu(),
                **self.model_stats,
            }
        except Exception as exc:
            return {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "gpu": {"available": False, "devices": []},
                **self.model_stats,
            }
        return {"available": True, **base}
