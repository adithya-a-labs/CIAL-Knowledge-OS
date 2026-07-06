"""Non-throwing infrastructure preflight for local/on-prem deployments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qdrant_client import QdrantClient

from ..config import KnowledgeOSConfig
from ..vectorstore import create_qdrant_client
from .qdrant_health import check_qdrant_health
from .system_health import (
    check_disk,
    check_gpu,
    check_memory,
    check_offline_flags,
    check_ollama,
)


def _config_check(config: KnowledgeOSConfig) -> dict[str, Any]:
    errors: list[str] = []
    if config.qdrant_mode not in {"embedded", "server"}:
        errors.append(
            f"Unsupported qdrant_mode '{config.qdrant_mode}'; use embedded or server."
        )
    if not config.qdrant_collection_name.strip():
        errors.append("qdrant_collection_name must not be blank.")
    if config.qdrant_batch_size <= 0:
        errors.append("qdrant_batch_size must be greater than zero.")
    return {"valid": not errors, "errors": errors}


def run_preflight(
    config: KnowledgeOSConfig,
    *,
    embedding_dimension: int | None = None,
    generation_enabled: bool = True,
    client_factory: Callable[[KnowledgeOSConfig], QdrantClient] = create_qdrant_client,
    execution_manager: Any | None = None,
) -> dict[str, Any]:
    """Return a structured report; callers decide whether errors should block."""

    if execution_manager is not None:
        execution_manager.start_stage(
            "preflight", event_type="preflight_started"
        )
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, Any] = {}

    checks["config"] = _config_check(config)
    errors.extend(checks["config"]["errors"])
    checks["qdrant_mode"] = {
        "mode": config.qdrant_mode,
        "server_opt_in": config.qdrant_mode == "server",
    }

    offline = check_offline_flags()
    checks["model_offline_flags"] = offline
    warnings.extend(offline["warnings"])

    memory = check_memory()
    checks["memory"] = memory
    if memory["warning"]:
        warnings.append(memory["warning"])

    disk = check_disk(config.data_dir)
    checks["disk"] = disk
    if disk["warning"]:
        warnings.append(disk["warning"])

    configured_devices = [
        config.embedding_device,
        str(getattr(config, "reranker_device", "")),
    ]
    gpu_device = next(
        (
            device
            for device in configured_devices
            if device.strip().casefold().startswith("cuda")
        ),
        config.embedding_device,
    )
    gpu = check_gpu(gpu_device)
    checks["gpu"] = gpu
    if gpu["error"]:
        errors.append(gpu["error"])

    should_check_qdrant = (
        config.qdrant_mode == "server"
        or (config.qdrant_mode == "embedded" and config.qdrant_dir.exists())
    )
    if should_check_qdrant and checks["config"]["valid"]:
        client: QdrantClient | None = None
        try:
            client = client_factory(config)
            qdrant = check_qdrant_health(
                client,
                config.qdrant_collection_name,
                embedding_dimension=embedding_dimension,
            )
        except Exception as exc:
            qdrant = {
                "reachable": False,
                "collection_name": config.qdrant_collection_name,
                "collection_exists": False,
                "warnings": [],
                "errors": [f"Qdrant server is not reachable: {exc}"],
                "passed": False,
            }
        finally:
            if client is not None:
                client.close()
        checks["qdrant"] = qdrant
        if execution_manager is not None:
            execution_manager.emit(
                "qdrant_health_checked",
                stage="preflight",
                status="completed" if qdrant.get("passed") else "warning",
                payload=qdrant,
                source="infra.preflight",
            )
        warnings.extend(qdrant["warnings"])
        errors.extend(qdrant["errors"])
    else:
        checks["qdrant"] = {
            "reachable": None,
            "collection_name": config.qdrant_collection_name,
            "collection_exists": False,
            "skipped": True,
            "reason": "No embedded collection exists yet.",
            "warnings": [],
            "errors": [],
        }

    if generation_enabled:
        ollama = check_ollama(config.ollama_model_name)
        checks["ollama"] = ollama
        if ollama["error"]:
            errors.append(ollama["error"])
    else:
        checks["ollama"] = {"skipped": True, "reason": "Generation is disabled."}

    result = {
        "passed": not errors,
        "warnings": warnings,
        "errors": errors,
        "checks": checks,
    }
    if execution_manager is not None:
        execution_manager.complete_stage(
            "preflight",
            event_type="preflight_completed",
            passed=result["passed"],
            warnings=len(warnings),
            errors=len(errors),
        )
    return result
