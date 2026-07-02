"""Central configuration for the local CIAL Knowledge OS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .token_budget import DEFAULT_TIKTOKEN_ENCODING


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
    pdf_data_dir: Path | None = None
    qdrant_dir: Path | None = None
    qdrant_collection_name: str = "cial_basic_rag"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    ollama_model_name: str = "gemma3:12b"
    tokenizer_encoding_name: str = DEFAULT_TIKTOKEN_ENCODING
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 3
    max_context_chars: int = 3_000
    # Persistence is the safe default: callers must explicitly opt into deleting
    # the local embedded Qdrant data.
    reset_vectorstore: bool = False
    # Synthetic fixtures are opt-in so normal ingestion cannot contaminate a
    # real corpus when data/sample is absent.
    create_sample_documents: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.data_dir = self._resolve(self.data_dir, self.project_root / "data")
        self.sample_data_dir = self._resolve(
            self.sample_data_dir, self.data_dir / "sample"
        )
        self.raw_data_dir = self._resolve(self.raw_data_dir, self.data_dir / "raw")
        self.pdf_data_dir = self._resolve(self.pdf_data_dir, self.data_dir / "pdf")
        self.qdrant_dir = self._resolve(
            self.qdrant_dir,
            self.data_dir / "qdrant" / self.qdrant_collection_name,
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if not self.tokenizer_encoding_name.strip():
            raise ValueError("tokenizer_encoding_name must not be blank.")
        self.tokenizer_encoding_name = self.tokenizer_encoding_name.strip()
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")

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
