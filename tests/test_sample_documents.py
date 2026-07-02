from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cial_knowledge_os.config import KnowledgeOSConfig
from cial_knowledge_os.loaders import (
    SAMPLE_AIRPORT_DOCUMENTS,
    create_sample_airport_documents,
    load_text_documents,
)
from cial_knowledge_os.rag_pipeline import BasicRAGPipeline


class SampleDocumentGenerationTests(unittest.TestCase):
    def test_pipeline_does_not_recreate_samples_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            config.raw_data_dir.mkdir(parents=True)
            raw_document = config.raw_data_dir / "approved_manual.txt"
            raw_document.write_text("Approved operational content.", encoding="utf-8")

            documents = BasicRAGPipeline(config=config).load()

            self.assertFalse(config.sample_data_dir.exists())
            self.assertEqual(
                [document.metadata["file_name"] for document in documents],
                ["approved_manual.txt"],
            )

    def test_pipeline_creates_samples_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(
                project_root=Path(directory),
                create_sample_documents=True,
            )

            documents = BasicRAGPipeline(config=config).load()

            self.assertTrue(config.sample_data_dir.is_dir())
            self.assertEqual(
                {document.metadata["file_name"] for document in documents},
                set(SAMPLE_AIRPORT_DOCUMENTS),
            )

    def test_existing_sample_directory_is_still_loaded_without_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))
            config.sample_data_dir.mkdir(parents=True)
            existing_document = config.sample_data_dir / "existing_sample.txt"
            existing_document.write_text(
                "Existing sample content.",
                encoding="utf-8",
            )

            documents = load_text_documents(config)

            self.assertEqual(len(documents), 1)
            self.assertEqual(
                documents[0].metadata["file_name"],
                "existing_sample.txt",
            )
            self.assertEqual(
                sorted(path.name for path in config.sample_data_dir.iterdir()),
                ["existing_sample.txt"],
            )

    def test_explicit_sample_creation_utility_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = KnowledgeOSConfig(project_root=Path(directory))

            paths = create_sample_airport_documents(config)

            self.assertEqual(
                {path.name for path in paths},
                set(SAMPLE_AIRPORT_DOCUMENTS),
            )
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
