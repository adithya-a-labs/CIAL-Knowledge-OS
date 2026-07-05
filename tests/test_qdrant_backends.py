from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams

from cial_knowledge_os.config import KnowledgeOSConfig, Phase4Config
from cial_knowledge_os.vectorstore import create_qdrant_client, index_chunks
from scripts.migrate_embedded_qdrant_to_server import migrate_collection


class QdrantClientModeTests(unittest.TestCase):
    def test_embedded_mode_remains_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeOSConfig(project_root=Path(directory))
            phase4 = Phase4Config(project_root=Path(directory))

        self.assertEqual(base.qdrant_mode, "embedded")
        self.assertEqual(phase4.qdrant_mode, "embedded")
        self.assertEqual(phase4.qdrant_url, "http://localhost:6333")
        self.assertIsNone(phase4.qdrant_api_key)
        self.assertEqual(phase4.qdrant_collection_name, "cial_phase4")
        self.assertEqual(base.qdrant_batch_size, 256)
        self.assertEqual(phase4.qdrant_batch_size, 256)
        self.assertTrue(phase4.qdrant_upsert_wait)

    def test_server_mode_defaults_to_smaller_batches(self) -> None:
        config = KnowledgeOSConfig(qdrant_mode="server")

        self.assertEqual(config.qdrant_batch_size, 32)
        self.assertTrue(config.qdrant_upsert_wait)

    def test_qdrant_batch_settings_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "qdrant_batch_size"):
            KnowledgeOSConfig(qdrant_batch_size=0)
        with self.assertRaisesRegex(TypeError, "qdrant_upsert_wait"):
            KnowledgeOSConfig(qdrant_upsert_wait=1)  # type: ignore[arg-type]

    @patch("cial_knowledge_os.vectorstore.QdrantClient")
    def test_embedded_mode_creates_path_based_client(
        self, client_class: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            result = create_qdrant_client(config)

        client_class.assert_called_once_with(path=str(config.qdrant_dir))
        self.assertIs(result, client_class.return_value)
        client_class.return_value.get_collections.assert_not_called()

    @patch("cial_knowledge_os.vectorstore.QdrantClient")
    def test_server_mode_creates_url_based_client_and_checks_health(
        self, client_class: MagicMock
    ) -> None:
        config = KnowledgeOSConfig(
            qdrant_mode="server",
            qdrant_url="http://qdrant.internal:6333",
            qdrant_api_key="local-secret",
        )
        result = create_qdrant_client(config)

        client_class.assert_called_once_with(
            url="http://qdrant.internal:6333",
            api_key="local-secret",
        )
        client_class.return_value.get_collections.assert_called_once_with()
        self.assertIs(result, client_class.return_value)

    def test_invalid_mode_raises_value_error(self) -> None:
        config = KnowledgeOSConfig(qdrant_mode="cloud")
        with self.assertRaisesRegex(ValueError, "Unsupported qdrant_mode"):
            create_qdrant_client(config)

    @patch("cial_knowledge_os.vectorstore.QdrantClient")
    def test_unreachable_server_has_actionable_error(
        self, client_class: MagicMock
    ) -> None:
        client_class.return_value.get_collections.side_effect = OSError(
            "connection refused"
        )
        config = KnowledgeOSConfig(qdrant_mode="server")

        with self.assertRaisesRegex(
            RuntimeError,
            r"Qdrant server mode is enabled[\s\S]*docker start cial-qdrant",
        ):
            create_qdrant_client(config)
        client_class.return_value.close.assert_called_once_with()


class QdrantIndexBatchTests(unittest.TestCase):
    def test_server_indexing_splits_points_and_forwards_wait(self) -> None:
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=VectorParams(size=3, distance=Distance.COSINE)
                )
            )
        )
        config = KnowledgeOSConfig(
            qdrant_mode="server",
            qdrant_collection_name="server_batch_test",
            qdrant_batch_size=2,
            qdrant_upsert_wait=False,
        )
        chunks = [
            Document(
                page_content=f"chunk text {index}",
                metadata={
                    "source": "manual.pdf",
                    "page_number": 1,
                    "chunk_id": f"chunk-{index}",
                    "chunk_index": index,
                },
            )
            for index in range(5)
        ]
        embeddings = np.asarray(
            [[float(index), 1.0, 0.0] for index in range(5)],
            dtype=np.float32,
        )

        with self.assertLogs(
            "cial_knowledge_os.vectorstore",
            level="INFO",
        ) as captured:
            index_chunks(client, chunks, embeddings, config)

        self.assertEqual(client.upsert.call_count, 3)
        batches = [
            item.kwargs["points"] for item in client.upsert.call_args_list
        ]
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertTrue(
            all(
                item.kwargs["wait"] is False
                for item in client.upsert.call_args_list
            )
        )
        points = [point for batch in batches for point in batch]
        self.assertEqual(len({point.id for point in points}), 5)
        self.assertEqual(
            [point.payload["text"] for point in points],
            [chunk.page_content for chunk in chunks],
        )
        self.assertEqual(
            [point.payload["metadata"] for point in points],
            [chunk.metadata for chunk in chunks],
        )
        self.assertEqual(len(captured.records), 3)


class MigrationTests(unittest.TestCase):
    def _source(self, count: int = 3) -> MagicMock:
        source = MagicMock()
        source.collection_exists.return_value = True
        source.count.return_value = SimpleNamespace(count=count)
        source.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=VectorParams(size=3, distance=Distance.COSINE),
                    sparse_vectors=None,
                )
            )
        )
        return source

    def test_dry_run_does_not_modify_or_scroll(self) -> None:
        source = self._source(10)
        target = MagicMock()
        target.collection_exists.return_value = False

        summary = migrate_collection(
            source,
            target,
            collection_name="cial_phase4",
            dry_run=True,
        )

        self.assertEqual(summary["source_points"], 10)
        self.assertEqual(summary["migrated_points"], 0)
        source.scroll.assert_not_called()
        target.create_collection.assert_not_called()
        target.upsert.assert_not_called()

    def test_migration_scrolls_and_upserts_in_batches(self) -> None:
        source = self._source(3)
        source.scroll.side_effect = [
            (
                [
                    SimpleNamespace(
                        id="one",
                        vector=[1.0, 0.0, 0.0],
                        payload={"n": 1},
                    ),
                    SimpleNamespace(
                        id="two",
                        vector=[0.0, 1.0, 0.0],
                        payload={"n": 2},
                    ),
                ],
                "next",
            ),
            (
                [
                    SimpleNamespace(
                        id="three",
                        vector=[0.0, 0.0, 1.0],
                        payload={"n": 3},
                    )
                ],
                None,
            ),
        ]
        target = MagicMock()
        target.collection_exists.return_value = False
        target.count.return_value = SimpleNamespace(count=3)

        summary = migrate_collection(
            source,
            target,
            collection_name="cial_phase4",
            batch_size=2,
        )

        self.assertEqual(summary["migrated_points"], 3)
        self.assertEqual(summary["batches"], 2)
        self.assertEqual(
            source.scroll.call_args_list,
            [
                call(
                    collection_name="cial_phase4",
                    limit=2,
                    offset=None,
                    with_payload=True,
                    with_vectors=True,
                ),
                call(
                    collection_name="cial_phase4",
                    limit=2,
                    offset="next",
                    with_payload=True,
                    with_vectors=True,
                ),
            ],
        )
        self.assertEqual(target.upsert.call_count, 2)
        first_points = target.upsert.call_args_list[0].kwargs["points"]
        self.assertEqual([point.id for point in first_points], ["one", "two"])
        self.assertEqual(first_points[0].payload, {"n": 1})
        self.assertEqual(first_points[0].vector, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
