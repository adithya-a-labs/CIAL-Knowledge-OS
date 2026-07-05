from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qdrant_client.models import Distance, VectorParams

from cial_knowledge_os.config import (
    KnowledgeOSConfig,
    resolve_qdrant_batch_size,
)
from cial_knowledge_os.infra.preflight import run_preflight
from cial_knowledge_os.infra.qdrant_health import (
    RED_OPTIMIZER_WARNING,
    parse_collection_health,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collection_info(
    *,
    status: str = "green",
    optimizer_status: str = "green",
    points: int = 12,
    indexed: int = 10,
    dimension: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        optimizer_status=SimpleNamespace(status=optimizer_status),
        points_count=points,
        indexed_vectors_count=indexed,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            )
        ),
    )


class BatchResolutionTests(unittest.TestCase):
    def test_mode_defaults_and_explicit_value(self) -> None:
        self.assertEqual(resolve_qdrant_batch_size("server"), 32)
        self.assertEqual(resolve_qdrant_batch_size("embedded"), 256)
        self.assertEqual(resolve_qdrant_batch_size("server", 7), 7)

    def test_old_configs_still_load(self) -> None:
        config = KnowledgeOSConfig()

        self.assertEqual(config.qdrant_mode, "embedded")
        self.assertEqual(config.qdrant_batch_size, 256)
        self.assertTrue(config.qdrant_upsert_wait)


class QdrantHealthParsingTests(unittest.TestCase):
    def test_parser_handles_collection_health_colors(self) -> None:
        for color in ("green", "yellow", "red"):
            with self.subTest(color=color):
                report = parse_collection_health(
                    _collection_info(status=color),
                    embedding_dimension=3,
                )
                self.assertEqual(report["collection_status"], color)
                self.assertTrue(report["dimension_matches"])

    def test_red_optimizer_has_required_warning(self) -> None:
        report = parse_collection_health(
            _collection_info(optimizer_status="red")
        )

        self.assertEqual(report["optimizer_status"], "red")
        self.assertIn(RED_OPTIMIZER_WARNING, report["warnings"])

    def test_optimizer_error_object_is_normalized_to_red(self) -> None:
        info = _collection_info()
        info.optimizer_status = SimpleNamespace(error="permission denied")

        report = parse_collection_health(info)

        self.assertEqual(report["optimizer_status"], "red")
        self.assertIn(RED_OPTIMIZER_WARNING, report["warnings"])


class PreflightTests(unittest.TestCase):
    def _system_patches(self):
        return (
            patch(
                "cial_knowledge_os.infra.preflight.check_offline_flags",
                return_value={"flags": {}, "warnings": []},
            ),
            patch(
                "cial_knowledge_os.infra.preflight.check_memory",
                return_value={"available_gib": 32.0, "warning": None},
            ),
            patch(
                "cial_knowledge_os.infra.preflight.check_disk",
                return_value={
                    "path": ".",
                    "available_gib": 100.0,
                    "warning": None,
                },
            ),
            patch(
                "cial_knowledge_os.infra.preflight.check_gpu",
                return_value={
                    "configured_device": "auto",
                    "requested": False,
                    "available": False,
                    "error": None,
                },
            ),
        )

    def test_preflight_warns_but_passes_for_red_optimizer(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _collection_info(
            optimizer_status="red"
        )
        patches = self._system_patches()
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(
                project_root=Path(directory),
                qdrant_mode="server",
            )
            with patches[0], patches[1], patches[2], patches[3]:
                report = run_preflight(
                    config,
                    embedding_dimension=3,
                    generation_enabled=False,
                    client_factory=lambda _: client,
                )

        self.assertTrue(report["passed"])
        self.assertIn(RED_OPTIMIZER_WARNING, report["warnings"])
        client.close.assert_called_once_with()

    def test_preflight_errors_when_server_is_unreachable(self) -> None:
        patches = self._system_patches()
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(
                project_root=Path(directory),
                qdrant_mode="server",
            )
            with patches[0], patches[1], patches[2], patches[3]:
                report = run_preflight(
                    config,
                    generation_enabled=False,
                    client_factory=lambda _: (_ for _ in ()).throw(
                        OSError("connection refused")
                    ),
                )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("not reachable" in error for error in report["errors"])
        )


class DeploymentFileTests(unittest.TestCase):
    def test_compose_uses_named_volume(self) -> None:
        compose = (PROJECT_ROOT / "docker-compose.qdrant.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "cial_qdrant_storage:/qdrant/storage",
            compose,
        )
        self.assertIn("\nvolumes:\n  cial_qdrant_storage:", compose)
        self.assertNotIn("./data/qdrant_server:/qdrant/storage", compose)

    def test_generated_artifact_paths_are_ignored(self) -> None:
        paths = [
            "data/bm25/example/index.bin",
            "data/qdrant/example/storage.bin",
            "data/qdrant_server/storage.bin",
            "outputs/report.json",
            "scratch.pkl",
            "scratch.lock",
            "scratch.tmp",
        ]
        for path in paths:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", path],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
