"""Reusable, local-first components for CIAL Knowledge OS."""

from .batch_qa import (
    CSV_COLUMNS,
    PHASE2_CSV_COLUMNS,
    PHASE3_CSV_COLUMNS,
    PHASE4_CSV_COLUMNS,
    PHASE5_CSV_COLUMNS,
    BatchAnswerCollection,
    collect_batch_answers,
    export_batch_answers,
)
from .benchmarking import Timer, benchmark_pipeline_steps, print_benchmark_table
from .benchmark_loader import Benchmark, BenchmarkQuestion, load_benchmark
from .chunking import chunk_documents, summarize_chunks
from .citations import (
    build_citations,
    render_answer_with_citations,
    render_citations,
)
from .citation_links import CitationLinkBuilder
from .config import (
    KnowledgeOSConfig,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    RunArtifactNames,
)
from .context_builder import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    ContextBuilder,
    ContextBuildResult,
    compress_context,
    merge_overlapping_chunks,
)
from .embeddings import embed_texts, get_embedding_dimension, load_embedding_model
from .evaluation_metrics import (
    aggregate_experiment,
    evaluate_answer,
    rank_experiments,
)
from .evaluation_report import (
    build_recommendations,
    write_recommendation_report,
)
from .experiment_config import ExperimentConfig, ExperimentGrid
from .experiment_runner import (
    ExperimentRunner,
    ExperimentSweepResult,
    ReconfiguringPipelineFactory,
)
from .file_formats import (
    FORMAT_REGISTRY,
    SupportStatus,
    get_file_extension,
    get_file_format_info,
    is_recognized_file,
    is_supported_file,
    list_formats_by_category,
    list_recognized_formats,
    list_supported_formats,
    scan_file_format_readiness,
    validate_ingestion_file,
)
from .fusion import ReciprocalRankFusion
from .llm import (
    GenerationFailedError,
    build_grounded_prompt,
    create_local_llm,
    generate_answer,
)
from .loaders import (
    IMPLEMENTED_KNOWLEDGE_EXTENSIONS,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    create_sample_airport_documents,
    discover_knowledge_documents,
    load_all_documents,
    load_pdf_documents,
    load_text_documents,
    resolve_corpus_root,
    summarize_documents,
)
from .rag_pipeline import BasicRAGPipeline
from .phase2_pipeline import Phase2RAGPipeline
from .phase3_pipeline import Phase3RAGPipeline
from .phase3_reporting import (
    write_results_csv,
    write_results_xlsx,
    write_latency_svg,
    write_standalone_html,
)
from .phase3_runner import Phase3Runner, Phase3RunResult
from .evidence_quality import EvidenceQualityReport, EvidenceQualityScorer
from .evidence_selector import EvidenceSelectionResult, EvidenceSelector
from .phase4_pipeline import Phase4RAGPipeline, UNSUPPORTED_QUERY_RESPONSE
from .phase4_checkpoint import (
    Phase4CheckpointManager,
    QuestionIdentity,
    normalize_question,
)
from .phase4_reporting import write_phase4_figures, write_phase4_html
from .phase4_runner import Phase4Runner, Phase4RunResult
from .phase4_trace import Phase4Trace, build_phase4_trace, phase4_diagnostics
from .agents import Agent, AgentResult, AgentState, Evidence, ModelRouter
from .orchestration import ConsensusEngine, Phase5Pipeline, Phase5Trace
from .reporting import render_phase5_html, write_phase5_html
from .query_transformations import (
    QueryTransformer,
    QueryVariant,
    expand_query_keywords,
    reformulate_for_domain,
    rewrite_query,
)
from .retrieval import (
    format_retrieved_context,
    print_retrieval_results,
    search_similar_chunks,
)
from .retrieval_postprocessing import (
    deduplicate_results,
    expand_neighbor_chunks,
    retrieve_multiple_queries,
)
from .retrieval_trace import build_question_trace
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
    default_bm25_tokenize,
)
from .reranker import (
    CrossEncoderReranker,
    MockReranker,
    RerankResult,
    Reranker,
)
from .run_manager import RunManager, RunPaths
from .token_budget import (
    DEFAULT_TIKTOKEN_ENCODING,
    TiktokenTokenizer,
    TokenBudgetManager,
    TokenBudgetUsage,
    TokenManager,
    create_token_budget_manager,
    create_token_manager,
)
from .trace_visualization import (
    decision_frame,
    display_question_trace,
    funnel_frame,
    load_question_traces,
    query_variants_frame,
    retrieval_results_frame,
    rrf_contribution_frame,
)
from .vectorstore import (
    create_qdrant_client,
    ensure_collection,
    index_chunks,
    recreate_collection,
    reset_qdrant_storage,
)
from .visualization import (
    batch_retrieval_trace_table,
    citation_quality_table,
    context_stage_counts_table,
    display_context_sections_table,
    display_high_duplicate_chunks_table,
    display_low_score_chunks_table,
    display_query_variant_contribution_table,
    display_retrieval_trace_table,
    display_top_sources_table,
    duplicate_chunk_frequency_table,
    neighbor_expansion_table,
    plot_answer_status_distribution,
    plot_chunk_size_distribution,
    plot_context_compression_ratio,
    plot_context_section_lengths,
    plot_context_stage_counts,
    plot_duplicate_chunk_frequency,
    plot_latency_by_question,
    plot_page_distribution,
    plot_query_variant_contribution,
    plot_retrieval_comparison,
    plot_retrieval_funnel,
    plot_retrieval_scores,
    plot_score_by_query_variant,
    plot_score_distribution,
    plot_source_distribution,
    plot_timing_breakdown,
    query_variants_table,
    retrieval_chunks_table,
    retrieval_comparison_table,
)
from .visualization_dashboard import generate_dashboard, load_dashboard_data

__all__ = [
    "BasicRAGPipeline",
    "Benchmark",
    "BenchmarkQuestion",
    "CSV_COLUMNS",
    "PHASE2_CSV_COLUMNS",
    "PHASE3_CSV_COLUMNS",
    "PHASE4_CSV_COLUMNS",
    "PHASE5_CSV_COLUMNS",
    "BatchAnswerCollection",
    "BM25Retriever",
    "CitationLinkBuilder",
    "DenseRetriever",
    "CrossEncoderReranker",
    "DEFAULT_TIKTOKEN_ENCODING",
    "HybridRetriever",
    "KnowledgeOSConfig",
    "Phase2Config",
    "Phase3Config",
    "Phase4Config",
    "Phase2RAGPipeline",
    "Phase3RAGPipeline",
    "Phase3Runner",
    "Phase3RunResult",
    "Phase4RAGPipeline",
    "Phase4CheckpointManager",
    "Phase4Runner",
    "Phase4RunResult",
    "Phase4Trace",
    "Agent",
    "AgentResult",
    "AgentState",
    "Evidence",
    "ModelRouter",
    "ConsensusEngine",
    "Phase5Pipeline",
    "Phase5Runner",
    "Phase5Trace",
    "ReciprocalRankFusion",
    "Retriever",
    "Reranker",
    "RerankResult",
    "MockReranker",
    "RunArtifactNames",
    "RunManager",
    "RunPaths",
    "SupportStatus",
    "TokenBudgetManager",
    "TokenBudgetUsage",
    "TokenManager",
    "TiktokenTokenizer",
    "ContextBuilder",
    "ContextBuildResult",
    "EvidenceQualityReport",
    "EvidenceQualityScorer",
    "EvidenceSelectionResult",
    "EvidenceSelector",
    "ExperimentConfig",
    "ExperimentGrid",
    "ExperimentRunner",
    "ExperimentSweepResult",
    "FORMAT_REGISTRY",
    "GenerationFailedError",
    "IMPLEMENTED_KNOWLEDGE_EXTENSIONS",
    "INSUFFICIENT_EVIDENCE_RESPONSE",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "UNSUPPORTED_QUERY_RESPONSE",
    "QueryTransformer",
    "QuestionIdentity",
    "QueryVariant",
    "ReconfiguringPipelineFactory",
    "Timer",
    "benchmark_pipeline_steps",
    "batch_retrieval_trace_table",
    "aggregate_experiment",
    "build_grounded_prompt",
    "build_question_trace",
    "build_phase4_trace",
    "build_citations",
    "build_recommendations",
    "chunk_documents",
    "collect_batch_answers",
    "citation_quality_table",
    "create_local_llm",
    "create_qdrant_client",
    "create_sample_airport_documents",
    "create_token_budget_manager",
    "create_token_manager",
    "compress_context",
    "context_stage_counts_table",
    "deduplicate_results",
    "discover_knowledge_documents",
    "default_bm25_tokenize",
    "display_context_sections_table",
    "display_question_trace",
    "display_high_duplicate_chunks_table",
    "display_low_score_chunks_table",
    "display_query_variant_contribution_table",
    "display_retrieval_trace_table",
    "display_top_sources_table",
    "duplicate_chunk_frequency_table",
    "embed_texts",
    "ensure_collection",
    "expand_neighbor_chunks",
    "expand_query_keywords",
    "export_batch_answers",
    "evaluate_answer",
    "format_retrieved_context",
    "funnel_frame",
    "generate_answer",
    "generate_dashboard",
    "get_embedding_dimension",
    "get_file_extension",
    "get_file_format_info",
    "index_chunks",
    "is_recognized_file",
    "is_supported_file",
    "load_all_documents",
    "load_benchmark",
    "load_question_traces",
    "load_dashboard_data",
    "load_embedding_model",
    "normalize_question",
    "load_pdf_documents",
    "load_text_documents",
    "list_formats_by_category",
    "list_recognized_formats",
    "list_supported_formats",
    "merge_overlapping_chunks",
    "neighbor_expansion_table",
    "plot_chunk_size_distribution",
    "plot_answer_status_distribution",
    "plot_context_compression_ratio",
    "plot_context_section_lengths",
    "plot_context_stage_counts",
    "plot_duplicate_chunk_frequency",
    "plot_latency_by_question",
    "plot_page_distribution",
    "plot_query_variant_contribution",
    "plot_retrieval_comparison",
    "plot_retrieval_funnel",
    "plot_retrieval_scores",
    "plot_score_by_query_variant",
    "plot_score_distribution",
    "plot_source_distribution",
    "plot_timing_breakdown",
    "phase4_diagnostics",
    "print_benchmark_table",
    "print_retrieval_results",
    "query_variants_table",
    "query_variants_frame",
    "rank_experiments",
    "recreate_collection",
    "reformulate_for_domain",
    "render_answer_with_citations",
    "render_citations",
    "resolve_corpus_root",
    "reset_qdrant_storage",
    "retrieve_multiple_queries",
    "retrieval_chunks_table",
    "retrieval_results_frame",
    "retrieval_comparison_table",
    "rewrite_query",
    "rrf_contribution_frame",
    "scan_file_format_readiness",
    "search_similar_chunks",
    "summarize_chunks",
    "summarize_documents",
    "validate_ingestion_file",
    "decision_frame",
    "write_recommendation_report",
    "write_results_csv",
    "write_results_xlsx",
    "write_latency_svg",
    "write_standalone_html",
    "write_phase4_figures",
    "write_phase4_html",
    "render_phase5_html",
    "write_phase5_html",
]


def __getattr__(name: str):
    if name == "Phase5Runner":
        from .orchestration.phase5_runner import Phase5Runner

        return Phase5Runner
    raise AttributeError(name)
