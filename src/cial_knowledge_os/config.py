"""Central configuration for the local CIAL Knowledge OS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    qdrant_collection_name: str = "cial_phase2"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    ollama_model_name: str = "gemma3:12b"
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
