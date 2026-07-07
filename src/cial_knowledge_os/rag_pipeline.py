"""Simple orchestration for the basic local RAG experiment."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .benchmarking import Timer
from .chunking import chunk_documents
from .citations import build_citations, render_answer_with_citations
from .config import KnowledgeOSConfig
from .embeddings import embed_texts, get_embedding_dimension, load_embedding_model
from .incremental_index import (
    IndexingPlan,
    create_indexing_plan,
    entry_paths,
    write_manifest,
)
from .infra.qdrant_health import parse_collection_health
from .execution import ExecutionManager
from .llm import LocalLLM, create_local_llm, generate_answer
from .loaders import (
    create_sample_airport_documents,
    load_pdf_documents,
    load_pdf_paths,
    load_text_documents,
    scan_configured_file_format_readiness,
)
from .retrieval import format_retrieved_context, search_similar_chunks
from .vectorstore import (
    create_qdrant_client,
    delete_document_chunks,
    ensure_collection,
    index_chunks,
    load_indexed_chunks,
    recreate_collection,
    reset_qdrant_storage,
)

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """Inspectable orchestration with replaceable local model dependencies."""

    def __init__(
        self,
        config: KnowledgeOSConfig | None = None,
        *,
        embedding_model: SentenceTransformer | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.config = config or KnowledgeOSConfig()
        self.embedding_model = embedding_model
        self.llm = llm
        self.client: QdrantClient | None = None
        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.embeddings: np.ndarray | None = None
        self.metrics: dict[str, float] = {}
        self.indexing_plan: IndexingPlan | None = None
        self.indexing_summary: dict[str, Any] = {}
        self.file_format_readiness: dict[str, Any] = {}
        self.ocr_summary: dict[str, Any] = {}
        self.execution_manager = ExecutionManager.disabled()

    @property
    def is_ready_for_answering(self) -> bool:
        """Return whether retrieval dependencies have been initialized."""

        return self.client is not None and self.embedding_model is not None

    def load(self) -> list[Document]:
        if getattr(self.config, "create_sample_documents", False):
            create_sample_airport_documents(self.config)
        started_at = time.perf_counter()
        self.file_format_readiness = scan_configured_file_format_readiness(
            self.config
        )
        text_documents = load_text_documents(self.config)
        pdf_started_at = time.perf_counter()
        manifest_without_vectorstore = (
            self.config.qdrant_mode == "embedded"
            and self.config.document_manifest_path.is_file()
            and not self.config.qdrant_dir.exists()
        )
        self.indexing_plan = create_indexing_plan(
            corpus_root=self.config.knowledge_root,
            manifest_path=self.config.document_manifest_path,
            collection_name=self.config.qdrant_collection_name,
            incremental_enabled=self.config.incremental_indexing_enabled,
            force_rebuild=(
                self.config.force_rebuild_index
                or self.config.reset_vectorstore
                or manifest_without_vectorstore
            ),
        )
        if (
            self.config.incremental_indexing_enabled
            and not self.indexing_plan.force_rebuild
        ):
            paths_to_process = entry_paths(
                self.indexing_plan,
                self.indexing_plan.files_to_process,
            )
            try:
                pdf_documents = load_pdf_paths(
                    paths_to_process,
                    corpus_root=self.indexing_plan.corpus_root,
                    config=self.config,
                )
            except TypeError as exc:
                if "unexpected keyword argument 'config'" not in str(exc):
                    raise
                pdf_documents = load_pdf_paths(
                    paths_to_process,
                    corpus_root=self.indexing_plan.corpus_root,
                )
        else:
            pdf_documents = load_pdf_documents(self.config)
        pdf_elapsed = time.perf_counter() - pdf_started_at
        if pdf_documents:
            self.metrics["pdf_loading_time"] = pdf_elapsed
        self.documents = [*text_documents, *pdf_documents]
        ocr_documents = [
            document
            for document in self.documents
            if document.metadata.get("requires_ocr")
        ]
        scanned_ocr_files = int(self.file_format_readiness.get("ocr_files") or 0)
        ocr_success_count = sum(
            document.metadata.get("ocr_status") == "OCR_SUCCESS"
            for document in ocr_documents
        )
        ocr_failure_count = max(0, scanned_ocr_files - int(ocr_success_count))
        ocr_times = [
            float(document.metadata.get("extraction_time_ms") or 0.0)
            for document in ocr_documents
        ]
        self.ocr_summary = {
            "total_ocr_files_processed": len(ocr_documents),
            "ocr_success_count": int(ocr_success_count),
            "ocr_failure_count": ocr_failure_count,
            "ocr_success_rate": (
                round(int(ocr_success_count) / scanned_ocr_files, 6)
                if scanned_ocr_files
                else 0.0
            ),
            "average_ocr_processing_time_ms": (
                round(sum(ocr_times) / len(ocr_times), 6) if ocr_times else 0.0
            ),
            "total_extracted_characters": sum(
                int(document.metadata.get("extracted_character_count") or 0)
                for document in ocr_documents
            ),
            "total_extracted_words": sum(
                int(document.metadata.get("extracted_word_count") or 0)
                for document in ocr_documents
            ),
            "ocr_engine_used": self.config.ocr_engine,
            "failures": [],
        }
        entries = {
            entry.relative_path: entry
            for entry in self.indexing_plan.files_to_process
        }
        for document in pdf_documents:
            relative_path = str(document.metadata.get("relative_path") or "")
            entry = entries.get(relative_path)
            if entry is not None:
                document.metadata["document_id"] = entry.document_id
        self.indexing_summary = {
            "new_files": len(self.indexing_plan.new),
            "changed_files": len(self.indexing_plan.changed),
            "unchanged_files": len(self.indexing_plan.unchanged),
            "deleted_files": len(self.indexing_plan.deleted),
            "chunks_added": 0,
            "chunks_removed": 0,
            "chunks_reused": sum(
                entry.chunk_count for entry in self.indexing_plan.unchanged
            ),
            "bm25_rebuilt": False,
            "vector_index_updated": False,
            "embedding_time_saved_estimate": 0.0,
            "manifest_path": str(self.config.document_manifest_path),
            "file_format_readiness": self.file_format_readiness,
            "ocr_summary": self.ocr_summary,
        }
        self.metrics["document_loading_time"] = time.perf_counter() - started_at
        if not self.documents and not (
            self.indexing_plan.unchanged
            or self.indexing_plan.deleted
            or self.indexing_plan.force_rebuild
        ):
            raise RuntimeError("No local documents were available for the RAG pipeline.")
        return self.documents

    def chunk(self) -> list[Document]:
        if not self.documents and self.indexing_plan is None:
            raise RuntimeError("Call load() before chunk().")
        with Timer(self.metrics, "chunking_time"):
            self.chunks = chunk_documents(self.documents, self.config)
        return self.chunks

    def embed(self) -> np.ndarray:
        if not self.chunks and self.indexing_plan is None:
            raise RuntimeError("Call chunk() before embed().")
        if self.embedding_model is None:
            self.embedding_model = load_embedding_model(self.config)
        with Timer(self.metrics, "embedding_time"):
            if self.chunks:
                self.embeddings = embed_texts(
                    self.embedding_model,
                    [chunk.page_content for chunk in self.chunks],
                    batch_size=self.config.embedding_batch_size,
                )
            else:
                self.embeddings = np.empty(
                    (0, get_embedding_dimension(self.embedding_model)),
                    dtype=float,
                )
        current_count = len(self.chunks)
        reused_count = int(self.indexing_summary.get("chunks_reused", 0))
        if current_count and reused_count:
            self.indexing_summary["embedding_time_saved_estimate"] = round(
                float(self.metrics.get("embedding_time", 0.0))
                * reused_count
                / current_count,
                6,
            )
        return self.embeddings

    def index(self) -> QdrantClient:
        """Idempotently persist chunks in the configured local Qdrant backend."""

        if self.embeddings is None or self.embedding_model is None:
            raise RuntimeError("Call embed() before index().")
        embedding_dimension = get_embedding_dimension(self.embedding_model)
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != embedding_dimension:
            raise ValueError(
                f"Embedding array shape {self.embeddings.shape} does not match "
                f"the model's output dimension {embedding_dimension}."
            )
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.config.reset_vectorstore:
            reset_qdrant_storage(self.config)
        plan = self.indexing_plan
        total_documents = (
            len(plan.new) + len(plan.changed) + len(plan.unchanged)
            if plan is not None
            else len(self.documents)
        )
        total_chunks = len(self.chunks) + int(
            self.indexing_summary.get("chunks_reused", 0)
        )
        logger.info(
            "indexing_started",
            extra={
                "event": "indexing",
                "qdrant_mode": self.config.qdrant_mode,
                "collection_name": self.config.qdrant_collection_name,
                "force_rebuild_index": self.config.force_rebuild_index,
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "embedding_batch_size": self.config.embedding_batch_size,
                "qdrant_batch_size": self.config.qdrant_batch_size,
            },
        )
        manager = self.execution_manager
        manager.start_stage(
            "indexing",
            event_type="indexing_started",
            documents_discovered=total_documents,
            chunks_created=total_chunks,
            qdrant_mode=self.config.qdrant_mode,
            collection_name=self.config.qdrant_collection_name,
            batch_size=self.config.qdrant_batch_size,
        )
        with Timer(self.metrics, "indexing_time"):
            self.client = create_qdrant_client(self.config)
            try:
                collection_existed = self.client.collection_exists(
                    self.config.qdrant_collection_name
                )
                plan = self.indexing_plan
                previous_point_count = (
                    int(
                        self.client.count(
                            collection_name=self.config.qdrant_collection_name,
                            exact=True,
                        ).count
                    )
                    if collection_existed
                    else 0
                )
                if (
                    plan is not None
                    and plan.unchanged
                    and not plan.force_rebuild
                    and not collection_existed
                ):
                    raise RuntimeError(
                        "Manifest references unchanged chunks, but the active "
                        "vector backend does not contain the expected collection. "
                        "This usually happens after switching from embedded to "
                        "server Qdrant. Set FORCE_REBUILD_INDEX=True once or run "
                        "the migration utility."
                    )
                if (
                    plan is not None
                    and plan.unchanged
                    and not plan.force_rebuild
                    and collection_existed
                    and previous_point_count == 0
                ):
                    raise RuntimeError(
                        "Manifest references unchanged chunks, but the active "
                        "Qdrant collection contains no points. Set "
                        "FORCE_REBUILD_INDEX=True once to restore the index."
                    )
                if self.config.force_rebuild_index:
                    recreate_collection(
                        self.client,
                        self.config,
                        embedding_dimension,
                    )
                else:
                    ensure_collection(
                        self.client,
                        self.config,
                        embedding_dimension,
                    )
                collection_health = parse_collection_health(
                    self.client.get_collection(
                        self.config.qdrant_collection_name
                    ),
                    embedding_dimension=embedding_dimension,
                )
                manager.emit(
                    "qdrant_health_checked",
                    stage="indexing",
                    status=(
                        "completed"
                        if not collection_health["warnings"]
                        else "warning"
                    ),
                    payload=collection_health,
                    source="rag_pipeline.index",
                )
                for warning in collection_health["warnings"]:
                    logger.warning(
                        warning,
                        extra={
                            "event": "qdrant_health",
                            "collection_name": self.config.qdrant_collection_name,
                            "collection_status": collection_health[
                                "collection_status"
                            ],
                            "optimizer_status": collection_health[
                                "optimizer_status"
                            ],
                        },
                    )
                reused_chunks = int(
                    self.indexing_summary.get("chunks_reused", 0)
                )
                if (
                    plan is not None
                    and reused_chunks
                    and not plan.force_rebuild
                    and int(
                        self.client.count(
                            collection_name=self.config.qdrant_collection_name,
                            exact=True,
                        ).count
                    )
                    < reused_chunks
                ):
                    raise RuntimeError(
                        "The Qdrant collection contains fewer points than the "
                        "document manifest expects. Set force_rebuild_index=True "
                        "to restore the index safely."
                    )
                removed = (
                    previous_point_count
                    if self.config.force_rebuild_index
                    else (
                        sum(
                            entry.chunk_count
                            for entry in (plan.previous.values() if plan else ())
                        )
                        if self.config.reset_vectorstore
                        else 0
                    )
                )
                if (
                    plan is not None
                    and self.config.incremental_indexing_enabled
                    and not self.config.force_rebuild_index
                ):
                    old_changed = [
                        plan.previous[entry.relative_path]
                        for entry in plan.changed
                        if entry.relative_path in plan.previous
                    ]
                    for entry in (*old_changed, *plan.deleted):
                        removed += delete_document_chunks(
                            self.client,
                            self.config,
                            document_id=entry.document_id,
                            relative_path=entry.relative_path,
                        )
                index_chunks(
                    self.client,
                    self.chunks,
                    self.embeddings,
                    self.config,
                    execution_manager=self.execution_manager,
                )
                changed = bool(
                    self.chunks or removed or self.config.force_rebuild_index
                )
                self.indexing_summary["chunks_added"] = len(self.chunks)
                self.indexing_summary["chunks_removed"] = removed
                self.indexing_summary["vector_index_updated"] = changed
                chunk_counts = Counter(
                    str(chunk.metadata.get("relative_path") or "")
                    for chunk in self.chunks
                    if chunk.metadata.get("document_id")
                )
                if plan is not None and (
                    self.config.incremental_indexing_enabled
                    or self.config.force_rebuild_index
                ):
                    write_manifest(
                        plan,
                        collection_name=self.config.qdrant_collection_name,
                        chunk_counts=dict(chunk_counts),
                    )
                if self.config.incremental_indexing_enabled:
                    self.chunks = load_indexed_chunks(self.client, self.config)
            except Exception as exc:
                manager.emit(
                    "indexing_failed",
                    stage="indexing",
                    status="failed",
                    error=str(exc),
                    source="rag_pipeline.index",
                )
                self.client.close()
                self.client = None
                raise
        if getattr(self.config, "retrieval_mode", "dense") not in {
            "bm25",
            "hybrid",
        }:
            manager.complete_stage(
                "indexing",
                event_type="indexing_completed",
                metrics={
                    "indexing_latency_seconds": self.metrics.get(
                        "indexing_time", 0.0
                    )
                },
                points_upserted=int(
                    self.indexing_summary.get("chunks_added", 0)
                ),
                **self.indexing_summary,
            )
        logger.info(
            "incremental_indexing_complete",
            extra={"event": "indexing", **self.indexing_summary},
        )
        return self.client

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("Call index() before retrieve().")
        with Timer(self.metrics, "retrieval_latency"):
            return search_similar_chunks(
                self.client,
                question,
                self.embedding_model,
                self.config,
            )

    def answer(self, question: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        results = self.retrieve(question)
        context = format_retrieved_context(results, self.config.max_context_chars)
        if self.llm is None:
            self.llm = create_local_llm(self.config)
        with Timer(self.metrics, "generation_latency"):
            raw_answer = generate_answer(self.llm, question, context)
        self.metrics["total_pipeline_latency"] = time.perf_counter() - started_at
        citations = build_citations(results)
        answer = render_answer_with_citations(raw_answer, citations)
        return {
            "question": question,
            "retrieved": results,
            "context": context,
            "raw_answer": raw_answer,
            "answer": answer,
            "citations": citations,
        }

    def run(self, question: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        if not self.documents:
            self.load()
        if not self.chunks:
            self.chunk()
        if self.embeddings is None:
            self.embed()
        if self.client is None:
            self.index()
        response = self.answer(question)
        self.metrics["total_pipeline_latency"] = time.perf_counter() - started_at
        return response

    def close(self) -> None:
        """Release the Qdrant client and any embedded storage lock."""

        if self.client is not None:
            self.client.close()
            self.client = None
