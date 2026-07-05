"""Local host resource and runtime checks used by deployment preflight."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import psutil
import torch


def check_memory(*, warning_threshold_gib: float = 4.0) -> dict[str, Any]:
    available = psutil.virtual_memory().available / (1024**3)
    return {
        "available_gib": round(available, 2),
        "warning": (
            f"Only {available:.2f} GiB of RAM is available."
            if available < warning_threshold_gib
            else None
        ),
    }


def check_disk(
    path: Path,
    *,
    warning_threshold_gib: float = 10.0,
) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free / (1024**3)
    return {
        "path": str(probe),
        "available_gib": round(free, 2),
        "warning": (
            f"Only {free:.2f} GiB of disk space is available at '{probe}'."
            if free < warning_threshold_gib
            else None
        ),
    }


def check_gpu(configured_device: str) -> dict[str, Any]:
    requested = configured_device.strip().casefold().startswith("cuda")
    available = bool(torch.cuda.is_available())
    return {
        "configured_device": configured_device,
        "requested": requested,
        "available": available,
        "error": (
            f"GPU device '{configured_device}' is configured, but CUDA is unavailable."
            if requested and not available
            else None
        ),
    }


def check_offline_flags() -> dict[str, Any]:
    flags: dict[str, bool] = {}
    warnings: list[str] = []
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        raw = os.environ.get(name, "1").strip().casefold()
        enabled = raw not in {"0", "false", "no", "off"}
        flags[name] = enabled
        if not enabled:
            warnings.append(
                f"{name} is disabled; enterprise offline operation requires it."
            )
    return {"flags": flags, "warnings": warnings}


def check_ollama(model_name: str) -> dict[str, Any]:
    try:
        from ollama import list as list_ollama_models

        models = {
            model.model
            for model in list_ollama_models().models
            if model.model is not None
        }
        installed = model_name in models
        return {
            "reachable": True,
            "model": model_name,
            "model_installed": installed,
            "error": (
                None
                if installed
                else f"Configured Ollama model '{model_name}' is not installed."
            ),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "model": model_name,
            "model_installed": False,
            "error": f"Local Ollama is unavailable: {exc}",
        }
