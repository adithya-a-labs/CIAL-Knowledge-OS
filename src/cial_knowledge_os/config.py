"""Central configuration for the local CIAL Knowledge OS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

from .token_budget import DEFAULT_TIKTOKEN_ENCODING


def resolve_qdrant_batch_size(
    qdrant_mode: str,
    configured_batch_size: int | None = None,
) -> int:
    """Resolve a bounded Qdrant upsert batch without changing legacy configs."""

    if configured_batch_size is not None:
        return configured_batch_size
    return 32 if qdrant_mode == "server" else 256


def _default_project_root() -> Path:
    current = Path.cwd().resolve()
    return current.parent if current.name == "notebooks" else current


@dataclass(slots=True)
class KnowledgeOSConfig:
    """Configuration shared by notebook experiments and future backend code."""

    project_root: Path = field(default_factory=_default_project_root)
    data_dir: Path | None = None
    sample_data_dir: Path | None = None
    raw_data_dir: Path | None = None
    knowledge_root: Path | None = None
    legacy_pdf_root: Path | None = None
    # Deprecated compatibility alias. New code must use ``knowledge_root`` for
    # ingestion; ``legacy_pdf_root`` exists only for one-time migration.
    pdf_data_dir: Path | None = None
    qdrant_mode: str = "embedded"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_batch_size: int | None = None
    qdrant_upsert_wait: bool = True
    qdrant_dir: Path | None = None
    document_manifest_path: Path | None = None
    qdrant_collection_name: str = "cial_basic_rag"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    ollama_model_name: str = "gemma3:12b"
    tokenizer_encoding_name: str = DEFAULT_TIKTOKEN_ENCODING
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 3
    max_context_chars: int = 3_000
    # Persistence is the safe default: callers must explicitly opt into deleting
    # the local embedded Qdrant data.
    reset_vectorstore: bool = False
    incremental_indexing_enabled: bool = True
    force_rebuild_index: bool = False
    ocr_enabled: bool = True
    ocr_engine: str = "tesseract"
    ocr_preprocessing: bool = True
    ocr_language: str = "eng"
    # Synthetic fixtures are opt-in so normal ingestion cannot contaminate a
    # real corpus when data/sample is absent.
    create_sample_documents: bool = False
    # EOF is local-only and additive. Callers can disable all observers while
    # retaining the same pipeline execution path.
    observability_enabled: bool = True
    observability_console: bool = True
    observability_rich: str | bool = "auto"
    observability_trace_jsonl: bool = True
    observability_progress_log: bool = True
    observability_telemetry: bool = True
    observability_telemetry_interval_seconds: float = 5.0
    observability_console_refresh_seconds: float = 1.0
    observability_output_dir: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.data_dir = self._resolve(self.data_dir, self.project_root / "data")
        self.sample_data_dir = self._resolve(
            self.sample_data_dir, self.data_dir / "sample"
        )
        self.raw_data_dir = self._resolve(self.raw_data_dir, self.data_dir / "raw")
        self.knowledge_root = self._resolve(
            self.knowledge_root,
            self.data_dir / "files",
        )
        legacy_pdf_value = (
            self.legacy_pdf_root
            if self.legacy_pdf_root is not None
            else self.pdf_data_dir
        )
        self.legacy_pdf_root = self._resolve(
            legacy_pdf_value,
            self.data_dir / "pdf",
        )
        self.pdf_data_dir = self.legacy_pdf_root
        self.qdrant_dir = self._resolve(
            self.qdrant_dir,
            self.data_dir / "qdrant" / self.qdrant_collection_name,
        )
        self.document_manifest_path = self._resolve(
            self.document_manifest_path,
            self.data_dir / "indexes" / "document_manifest.json",
        )
        self.observability_output_dir = self._resolve(
            self.observability_output_dir,
            self.project_root / "outputs" / "runs",
        )
        self.qdrant_batch_size = resolve_qdrant_batch_size(
            self.qdrant_mode,
            self.qdrant_batch_size,
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if (
            isinstance(self.qdrant_batch_size, bool)
            or self.qdrant_batch_size <= 0
        ):
            raise ValueError("qdrant_batch_size must be greater than zero.")
        if (
            isinstance(self.embedding_batch_size, bool)
            or self.embedding_batch_size <= 0
        ):
            raise ValueError("embedding_batch_size must be greater than zero.")
        if not isinstance(self.qdrant_upsert_wait, bool):
            raise TypeError("qdrant_upsert_wait must be a boolean.")
        if not self.tokenizer_encoding_name.strip():
            raise ValueError("tokenizer_encoding_name must not be blank.")
        self.tokenizer_encoding_name = self.tokenizer_encoding_name.strip()
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")
        if not self.ocr_engine.strip():
            raise ValueError("ocr_engine must not be blank.")
        self.ocr_engine = self.ocr_engine.strip().casefold()
        if self.ocr_engine not in {"tesseract"}:
            raise ValueError("ocr_engine must be 'tesseract'.")
        if not self.ocr_language.strip():
            raise ValueError("ocr_language must not be blank.")
        self.ocr_language = self.ocr_language.strip()
        if self.observability_telemetry_interval_seconds <= 0:
            raise ValueError(
                "observability_telemetry_interval_seconds must be greater than zero."
            )
        if self.observability_console_refresh_seconds <= 0:
            raise ValueError(
                "observability_console_refresh_seconds must be greater than zero."
            )
        if self.observability_rich not in {True, False, "auto"}:
            raise ValueError("observability_rich must be true, false, or 'auto'.")

    def _resolve(self, value: Path | None, default: Path) -> Path:
        path = Path(value).expanduser() if value is not None else default
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(slots=True)
class Phase2Config(KnowledgeOSConfig):
    """Additive configuration for query transformation and context construction.

    ``KnowledgeOSConfig`` and its Phase 1 defaults remain unchanged. Phase 2 uses
    ``retrieval_top_k`` instead of changing the meaning of the existing ``top_k``.
    """
    qdrant_collection_name: str = "cial_phase2"
    max_context_chars: int = 20_000

    retrieval_top_k: int = 10
    enable_query_rewrite: bool = True
    enable_keyword_expansion: bool = True
    enable_domain_reformulation: bool = True
    enable_multi_query: bool = True
    enable_neighbor_expansion: bool = True
    neighbor_window: int = 1
    enable_overlap_merging: bool = True
    enable_context_compression: bool = True
    max_query_variants: int = 4

    def __post_init__(self) -> None:
        super(Phase2Config, self).__post_init__()
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")
        if self.neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative.")
        if self.max_query_variants <= 0:
            raise ValueError("max_query_variants must be greater than zero.")


@dataclass(frozen=True, slots=True)
class RunArtifactNames:
    """Centralized filenames for one reproducible Phase 3 run bundle."""

    results_csv: str = "results.csv"
    results_xlsx: str = "results.xlsx"
    report_html: str = "report.html"
    config_json: str = "config.json"
    summary_json: str = "summary.json"
    retrieval_json: str = "retrieval.json"
    metrics_json: str = "metrics.json"
    logs: str = "logs.txt"
    figures_dir: str = "figures"
    context_dir: str = "context"
    latency_figure: str = "latency.svg"
    context_file_template: str = "{index:03d}_{slug}.md"

    def __post_init__(self) -> None:
        values = (
            self.results_csv,
            self.results_xlsx,
            self.report_html,
            self.config_json,
            self.summary_json,
            self.retrieval_json,
            self.metrics_json,
            self.logs,
            self.figures_dir,
            self.context_dir,
            self.latency_figure,
            self.context_file_template,
        )
        if any(not value.strip() for value in values):
            raise ValueError("Run artifact names must not be blank.")
        if any(Path(value).name != value for value in values):
            raise ValueError("Run artifact names must be simple names, not paths.")
        if len(set(values)) != len(values):
            raise ValueError("Run artifact names must be unique.")
        try:
            rendered = self.context_file_template.format(index=1, slug="question")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "context_file_template must support {index} and {slug}."
            ) from exc
        if Path(rendered).name != rendered:
            raise ValueError(
                "context_file_template must render a simple filename."
            )


@dataclass(slots=True)
class Phase3Config(Phase2Config):
    """Configuration for hybrid retrieval and reproducible Phase 3 runs.

    Phase 1 and Phase 2 defaults remain untouched. ``max_context_tokens`` is
    intentionally optional: setting it enables tokenizer-aware budgeting while
    ``None`` preserves the Phase 2 character-budget implementation.
    """

    qdrant_collection_name: str = "cial_phase3"
    retrieval_mode: str = "hybrid"
    dense_top_k: int = 10
    bm25_top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    parallel_retrieval: bool = True
    bm25_cache_dir: Path | None = None
    bm25_cache_filename: str = "bm25_index.pkl"
    max_context_tokens: int | None = 4_096
    citation_link_mode: Literal["file", "localhost"] = "file"
    citation_base_url: str | None = None
    output_root: Path | None = None
    benchmark_csv_path: Path | None = None
    benchmark_metadata_path: Path | None = None
    phase_output_name: str = "03_Hybrid_Retrieval"
    run_prefix: str = "run"
    run_timestamp_format: str = "%Y%m%dT%H%M%S"
    artifact_names: RunArtifactNames = field(default_factory=RunArtifactNames)
    log_level: str = "INFO"
    structured_logging: bool = True

    def __post_init__(self) -> None:
        super(Phase3Config, self).__post_init__()
        self.bm25_cache_dir = self._resolve(
            self.bm25_cache_dir,
            self.data_dir / "bm25" / self.qdrant_collection_name,
        )
        self.output_root = self._resolve(
            self.output_root,
            self.project_root / "outputs" / "batch_answers",
        )
        self.benchmark_csv_path = self._resolve(
            self.benchmark_csv_path,
            self.data_dir / "benchmarks" / "cisg" / "benchmark_answers.csv",
        )
        self.benchmark_metadata_path = self._resolve(
            self.benchmark_metadata_path,
            self.data_dir / "benchmarks" / "cisg" / "benchmark_metadata.json",
        )
        if not self.retrieval_mode.strip():
            raise ValueError("retrieval_mode must not be blank.")
        self.retrieval_mode = self.retrieval_mode.strip()
        if not isinstance(self.artifact_names, RunArtifactNames):
            raise TypeError("artifact_names must be a RunArtifactNames instance.")
        if self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be greater than zero.")
        if self.bm25_top_k <= 0:
            raise ValueError("bm25_top_k must be greater than zero.")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")
        if self.dense_weight <= 0 or self.bm25_weight <= 0:
            raise ValueError("Retriever weights must be greater than zero.")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be greater than zero.")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between zero and one.")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")
        if self.citation_link_mode == "localhost" and not self.citation_base_url:
            raise ValueError(
                "citation_base_url is required when citation_link_mode is "
                "'localhost'."
            )
        if not self.phase_output_name.strip():
            raise ValueError("phase_output_name must not be blank.")
        if Path(self.phase_output_name).name != self.phase_output_name:
            raise ValueError("phase_output_name must be a simple directory name.")
        if not self.run_prefix.strip():
            raise ValueError("run_prefix must not be blank.")
        if Path(self.bm25_cache_filename).name != self.bm25_cache_filename:
            raise ValueError("bm25_cache_filename must be a simple filename.")
        normalized_level = self.log_level.upper()
        if normalized_level not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ValueError(
                "log_level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG."
            )
        self.log_level = normalized_level


@dataclass(slots=True)
class Phase4Config(Phase3Config):
    """Configure local reranking, evidence selection, and Phase 4 artifacts.

    Inputs are inherited Phase 3 retrieval/context settings plus local reranker
    model, hardware, score, diversity, redundancy, evidence-budget, and
    grounded-answer presentation choices. The resolved object is consumed by
    :class:`Phase4RAGPipeline` and :class:`Phase4Runner`; effective values are
    persisted in every run bundle.

    Phase 4 deliberately keeps all Phase 1--3 fields valid. Its defaults use a
    new collection/output namespace, allow one-time developer model staging,
    and disable neighbor expansion so evidence that bypassed reranking is not
    silently introduced after selection. Set ``reranker_local_files_only=True``
    for strict enterprise offline operation. Earlier configuration classes and
    their defaults are unchanged.
    """

    qdrant_collection_name: str = "cial_phase4"
    phase_output_name: str = "04_Reranking_and_Evidence_Selection"
    enable_neighbor_expansion: bool = False

    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "auto"
    reranker_batch_size: int = 16
    # Developer mode stages a missing model once, then uses the HF cache.
    # Enterprise deployments set this to True to prohibit all network access.
    reranker_local_files_only: bool = False
    reranker_candidate_top_k: int = 30

    evidence_selection_strategies: tuple[str, ...] = (
        "top_k",
        "reranker_score_threshold",
        "source_diversity",
        "redundancy_reduction",
        "token_budget",
    )
    # Legacy names remain valid; the Phase 4 selector resolves them into the
    # clearer min/max and reranker-specific fields below.
    evidence_max_chunks: int = 8
    evidence_score_threshold: float = -4.0
    min_selected_evidence: int = 3
    max_selected_evidence: int | None = None
    reranker_score_threshold: float | None = None
    fallback_to_top_n_if_empty: bool = True
    fallback_top_n: int = 3
    weak_evidence_answer_allowed: bool = True
    min_fallback_reranker_score: float = 0.35
    allow_extractive_fallback_for_weak_evidence: bool = False
    unsupported_query_detection_enabled: bool = True
    # Evidence precision and answer depth are independent controls. These
    # fields change synthesis instructions, never the selected context.
    answer_detail_level: str = "detailed"
    min_answer_words: int | None = 250
    max_answer_words: int | None = None
    prefer_structured_answers: bool = True
    adaptive_answer_sections: bool = True
    include_decision_notes: bool = True
    generation_retries: int = 2
    retry_cooldown_seconds: float = 20.0
    evidence_token_budget: int = 2_400
    selected_evidence_target_min_tokens: int = 800
    selected_evidence_target_max_tokens: int = 1_500
    evidence_max_chunks_per_source: int = 2
    evidence_redundancy_threshold: float = 0.85
    evidence_strong_threshold: float = 0.65
    evidence_medium_threshold: float = 0.35

    phase4_run_mode: Literal[
        "smoke",
        "manual_qa",
        "benchmark",
        "export_only",
    ] = "manual_qa"
    phase4_trace_mode: Literal["compact", "full"] = "full"
    max_inline_manual_questions: int | None = None
    allow_large_run: bool = False

    def __post_init__(self) -> None:
        super(Phase4Config, self).__post_init__()
        if not self.reranker_model_name.strip():
            raise ValueError("reranker_model_name must not be blank.")
        self.reranker_model_name = self.reranker_model_name.strip()
        if not self.reranker_device.strip():
            raise ValueError("reranker_device must not be blank.")
        self.reranker_device = self.reranker_device.strip()
        if self.reranker_batch_size <= 0:
            raise ValueError("reranker_batch_size must be greater than zero.")
        if self.reranker_candidate_top_k <= 0:
            raise ValueError("reranker_candidate_top_k must be greater than zero.")
        if self.max_selected_evidence is None:
            self.max_selected_evidence = self.evidence_max_chunks
        else:
            self.evidence_max_chunks = self.max_selected_evidence
        if self.reranker_score_threshold is None:
            self.reranker_score_threshold = self.evidence_score_threshold
        else:
            self.evidence_score_threshold = self.reranker_score_threshold
        if self.max_selected_evidence <= 0:
            raise ValueError("max_selected_evidence must be greater than zero.")
        if self.min_selected_evidence <= 0:
            raise ValueError("min_selected_evidence must be greater than zero.")
        if self.min_selected_evidence > self.max_selected_evidence:
            self.min_selected_evidence = self.max_selected_evidence
        if self.fallback_top_n <= 0:
            raise ValueError("fallback_top_n must be greater than zero.")
        if self.fallback_top_n > self.max_selected_evidence:
            self.fallback_top_n = self.max_selected_evidence
        if (
            isinstance(self.min_fallback_reranker_score, bool)
            or not isinstance(
                self.min_fallback_reranker_score,
                (int, float),
            )
            or not isfinite(float(self.min_fallback_reranker_score))
        ):
            raise ValueError(
                "min_fallback_reranker_score must be a finite numeric value."
            )
        if not self.answer_detail_level.strip():
            raise ValueError("answer_detail_level must not be blank.")
        self.answer_detail_level = self.answer_detail_level.strip().casefold()
        if self.answer_detail_level not in {"concise", "balanced", "detailed"}:
            raise ValueError(
                "answer_detail_level must be concise, balanced, or detailed."
            )
        if self.min_answer_words is not None and self.min_answer_words <= 0:
            raise ValueError("min_answer_words must be greater than zero or None.")
        if self.max_answer_words is not None and self.max_answer_words <= 0:
            raise ValueError("max_answer_words must be greater than zero or None.")
        if self.generation_retries < 0:
            raise ValueError("generation_retries must be non-negative.")
        if self.retry_cooldown_seconds < 0:
            raise ValueError("retry_cooldown_seconds must be non-negative.")
        if self.evidence_token_budget <= 0:
            raise ValueError("evidence_token_budget must be greater than zero.")
        if self.selected_evidence_target_min_tokens <= 0:
            raise ValueError(
                "selected_evidence_target_min_tokens must be greater than zero."
            )
        if self.selected_evidence_target_max_tokens <= 0:
            raise ValueError(
                "selected_evidence_target_max_tokens must be greater than zero."
            )
        if (
            self.selected_evidence_target_min_tokens
            > self.selected_evidence_target_max_tokens
        ):
            raise ValueError(
                "selected_evidence_target_min_tokens must not exceed "
                "selected_evidence_target_max_tokens."
            )
        if (
            self.selected_evidence_target_max_tokens
            > self.evidence_token_budget
        ):
            self.selected_evidence_target_max_tokens = (
                self.evidence_token_budget
            )
        if (
            self.selected_evidence_target_min_tokens
            > self.selected_evidence_target_max_tokens
        ):
            self.selected_evidence_target_min_tokens = (
                self.selected_evidence_target_max_tokens
            )
        if self.max_context_tokens is not None:
            if self.evidence_token_budget > self.max_context_tokens:
                raise ValueError(
                    "evidence_token_budget must not exceed max_context_tokens."
                )
        if self.evidence_max_chunks_per_source <= 0:
            raise ValueError(
                "evidence_max_chunks_per_source must be greater than zero."
            )
        if not 0.0 <= self.evidence_redundancy_threshold <= 1.0:
            raise ValueError(
                "evidence_redundancy_threshold must be between zero and one."
            )
        if self.evidence_medium_threshold > self.evidence_strong_threshold:
            raise ValueError(
                "evidence_medium_threshold must not exceed "
                "evidence_strong_threshold."
            )
        allowed_strategies = {
            "top_k",
            "reranker_score_threshold",
            "source_diversity",
            "redundancy_reduction",
            "token_budget",
        }
        unknown = set(self.evidence_selection_strategies) - allowed_strategies
        if unknown:
            raise ValueError(
                "Unknown evidence selection strategies: "
                + ", ".join(sorted(unknown))
            )
        if not self.evidence_selection_strategies:
            raise ValueError("evidence_selection_strategies must not be empty.")
        if (
            self.max_inline_manual_questions is not None
            and self.max_inline_manual_questions <= 0
        ):
            raise ValueError(
                "max_inline_manual_questions must be greater than zero or None."
            )
