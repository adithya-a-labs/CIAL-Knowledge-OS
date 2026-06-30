"""Reusable, local-first components for CIAL Knowledge OS."""

from .benchmarking import Timer, benchmark_pipeline_steps, print_benchmark_table
from .chunking import chunk_documents, summarize_chunks
from .config import KnowledgeOSConfig
from .embeddings import embed_texts, get_embedding_dimension, load_embedding_model
from .llm import build_grounded_prompt, create_local_llm, generate_answer
from .loaders import (
    create_sample_airport_documents,
    load_all_documents,
    load_pdf_documents,
    load_text_documents,
    summarize_documents,
)
from .rag_pipeline import BasicRAGPipeline
from .retrieval import (
    format_retrieved_context,
    print_retrieval_results,
    search_similar_chunks,
)
from .vectorstore import (
    create_qdrant_client,
    index_chunks,
    recreate_collection,
    reset_qdrant_storage,
)
from .visualization import (
    plot_chunk_size_distribution,
    plot_retrieval_scores,
    plot_timing_breakdown,
)

__all__ = [
    "BasicRAGPipeline",
    "KnowledgeOSConfig",
    "Timer",
    "benchmark_pipeline_steps",
    "build_grounded_prompt",
    "chunk_documents",
    "create_local_llm",
    "create_qdrant_client",
    "create_sample_airport_documents",
    "embed_texts",
    "format_retrieved_context",
    "generate_answer",
    "get_embedding_dimension",
    "index_chunks",
    "load_all_documents",
    "load_embedding_model",
    "load_pdf_documents",
    "load_text_documents",
    "plot_chunk_size_distribution",
    "plot_retrieval_scores",
    "plot_timing_breakdown",
    "print_benchmark_table",
    "print_retrieval_results",
    "recreate_collection",
    "reset_qdrant_storage",
    "search_similar_chunks",
    "summarize_chunks",
    "summarize_documents",
]
