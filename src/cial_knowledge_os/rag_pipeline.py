"""Simple orchestration for the basic local RAG experiment."""

from __future__ import annotations

import time
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
from .llm import LocalLLM, create_local_llm, generate_answer
from .loaders import (
    create_sample_airport_documents,
    load_pdf_documents,
    load_text_documents,
)
from .retrieval import format_retrieved_context, search_similar_chunks
from .vectorstore import (
    create_qdrant_client,
    ensure_collection,
    index_chunks,
    reset_qdrant_storage,
)


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

    @property
    def is_ready_for_answering(self) -> bool:
        """Return whether retrieval dependencies have been initialized."""

        return self.client is not None and self.embedding_model is not None

    def load(self) -> list[Document]:
        create_sample_airport_documents(self.config)
        started_at = time.perf_counter()
        text_documents = load_text_documents(self.config)
        pdf_started_at = time.perf_counter()
        pdf_documents = load_pdf_documents(self.config)
        pdf_elapsed = time.perf_counter() - pdf_started_at
        if pdf_documents:
            self.metrics["pdf_loading_time"] = pdf_elapsed
        self.documents = [*text_documents, *pdf_documents]
        self.metrics["document_loading_time"] = time.perf_counter() - started_at
        if not self.documents:
            raise RuntimeError("No local documents were available for the RAG pipeline.")
        return self.documents

    def chunk(self) -> list[Document]:
        if not self.documents:
            raise RuntimeError("Call load() before chunk().")
        with Timer(self.metrics, "chunking_time"):
            self.chunks = chunk_documents(self.documents, self.config)
        return self.chunks

    def embed(self) -> np.ndarray:
        if not self.chunks:
            raise RuntimeError("Call chunk() before embed().")
        if self.embedding_model is None:
            self.embedding_model = load_embedding_model(self.config)
        with Timer(self.metrics, "embedding_time"):
            self.embeddings = embed_texts(
                self.embedding_model,
                [chunk.page_content for chunk in self.chunks],
            )
        return self.embeddings

    def index(self) -> QdrantClient:
        """Idempotently persist the current chunks in embedded local Qdrant."""

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
        with Timer(self.metrics, "indexing_time"):
            self.client = create_qdrant_client(self.config)
            try:
                ensure_collection(
                    self.client,
                    self.config,
                    embedding_dimension,
                )
                index_chunks(
                    self.client,
                    self.chunks,
                    self.embeddings,
                    self.config,
                )
            except Exception:
                self.client.close()
                self.client = None
                raise
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
        """Release the embedded Qdrant lock."""

        if self.client is not None:
            self.client.close()
            self.client = None
