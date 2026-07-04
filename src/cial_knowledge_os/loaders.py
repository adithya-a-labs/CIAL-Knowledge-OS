"""Local document ingestion with a recursive, configuration-driven corpus."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import KnowledgeOSConfig

logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".md", ".docx", ".html"}
)
IMPLEMENTED_KNOWLEDGE_EXTENSIONS = frozenset({".pdf"})

SAMPLE_AIRPORT_DOCUMENTS = {
    "terminal_operations_sop.txt": """CIAL Terminal Operations SOP
Version: 1.0
Owner: Airport Operations Control Centre

Passenger queue monitoring must be reviewed every 15 minutes during peak periods.
Duty managers must coordinate with airline supervisors when a queue exceeds the
marked holding area. Escalators, elevators, baggage belts, and passenger boarding
bridges must be visually checked at the start of each shift.
""",
    "runway_maintenance_sop.txt": """CIAL Runway Maintenance SOP
Version: 1.0
Owner: Airside Operations

Routine runway surface inspection must be performed every 6 hours and after heavy
rain, bird-strike reports, foreign-object-debris alerts, or pilot braking-action
complaints. Maintenance teams must obtain ATC clearance before entering the runway
strip and maintain continuous radio contact.
""",
    "electrical_maintenance_manual.txt": """CIAL Electrical Maintenance Manual
Version: 1.0
Owner: Electrical Engineering

Work on energized panels requires voltage-rated insulated gloves, an arc-rated face
shield, flame-resistant clothing, dielectric safety shoes, insulated tools, and
lockout-tagout verification wherever isolation is possible. Emergency shutdown
requires operating the designated stop, notifying electrical control, applying
lockout-tagout, and recording the event.
""",
}


def create_sample_airport_documents(config: KnowledgeOSConfig) -> list[Path]:
    """Explicitly create non-sensitive fixtures without overwriting edits."""

    config.sample_data_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file_name, content in SAMPLE_AIRPORT_DOCUMENTS.items():
        path = config.sample_data_dir / file_name
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _base_metadata(
    path: Path,
    loader_type: str,
    page_number: int | None = None,
    *,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    relative_path: Path | None = None
    if corpus_root is not None:
        try:
            relative_path = resolved.relative_to(corpus_root.resolve())
        except ValueError:
            relative_path = None
    folder_parts = relative_path.parts[:-1] if relative_path is not None else ()
    metadata: dict[str, Any] = {
        # ``source`` and ``file_name`` are legacy Phase 1--4 fields.
        "source": str(resolved),
        "file_name": path.name,
        "source_filename": path.name,
        "absolute_path": str(resolved),
        "relative_path": relative_path.as_posix() if relative_path else path.name,
        "category": folder_parts[0] if folder_parts else None,
        "collection": folder_parts[1] if len(folder_parts) > 1 else None,
        "loader_type": loader_type,
        "document_type": path.suffix.lstrip(".").lower(),
        "access_level": "internal",
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    return metadata


def load_text_documents(config: KnowledgeOSConfig) -> list[Document]:
    """Load UTF-8 text documents from sample and raw local data directories."""

    documents: list[Document] = []
    seen: set[Path] = set()
    for directory in (config.sample_data_dir, config.raw_data_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.txt")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata=_base_metadata(path, "text"),
                )
            )
    return documents


def _load_pdf_with_docling(path: Path, corpus_root: Path) -> list[Document]:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    text = result.document.export_to_markdown().strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata=_base_metadata(path, "docling", corpus_root=corpus_root),
        )
    ]


def _load_pdf_with_pymupdf(path: Path, corpus_root: Path) -> list[Document]:
    import fitz

    documents: list[Document] = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=_base_metadata(
                            path,
                            "pymupdf",
                            index,
                            corpus_root=corpus_root,
                        ),
                    )
                )
    return documents


def _supported_documents(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )


def resolve_corpus_root(config: KnowledgeOSConfig) -> Path:
    """Return the canonical configured corpus root."""

    return config.knowledge_root


def discover_knowledge_documents(
    config: KnowledgeOSConfig,
) -> tuple[Path, list[Path]]:
    """Recursively find recognized files below the active configured corpus."""

    corpus_root = resolve_corpus_root(config)
    paths = _supported_documents(corpus_root)
    for path in paths:
        extension = path.suffix.lower()
        if extension not in IMPLEMENTED_KNOWLEDGE_EXTENSIONS:
            logger.warning(
                "document_type_not_implemented",
                extra={
                    "event": "document_discovery",
                    "source": str(path.resolve()),
                    "document_type": extension.lstrip("."),
                },
            )
    return (
        corpus_root,
        [
            path
            for path in paths
            if path.suffix.lower() in IMPLEMENTED_KNOWLEDGE_EXTENSIONS
        ],
    )


def load_pdf_documents(config: KnowledgeOSConfig) -> list[Document]:
    """Recursively load corpus PDFs, preferring Docling then PyMuPDF."""

    corpus_root, pdf_paths = discover_knowledge_documents(config)
    if not pdf_paths:
        logger.info(
            "pdf_corpus_empty",
            extra={
                "event": "document_loading",
                "knowledge_root": str(config.knowledge_root),
                "active_corpus_root": str(corpus_root),
            },
        )
        return []

    try:
        import docling  # noqa: F401
    except ImportError:
        docling_available = False
    else:
        docling_available = True

    try:
        import fitz  # noqa: F401
    except ImportError:
        pymupdf_available = False
    else:
        pymupdf_available = True

    if not docling_available and not pymupdf_available:
        raise ImportError(
            "PDF ingestion requires a local loader. Install 'docling' (preferred) "
            "or 'PyMuPDF'; no cloud OCR fallback is used."
        )

    documents: list[Document] = []
    for path in pdf_paths:
        if docling_available:
            try:
                docling_documents = _load_pdf_with_docling(path, corpus_root)
                if docling_documents:
                    documents.extend(docling_documents)
                    continue
            except Exception as exc:
                if not pymupdf_available:
                    raise RuntimeError(f"Docling could not load {path.name}: {exc}") from exc
                logger.warning(
                    "docling_pdf_fallback",
                    extra={
                        "event": "document_loading",
                        "source": str(path),
                        "error": str(exc),
                    },
                )
            else:
                if not pymupdf_available:
                    raise RuntimeError(
                        f"Docling extracted no text from {path.name}, and PyMuPDF "
                        "is not installed as a local fallback."
                    )
        try:
            documents.extend(_load_pdf_with_pymupdf(path, corpus_root))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read PDF '{path}'. The file may be corrupted, "
                "encrypted, or unsupported by the configured local loaders. "
                f"Original error: {exc}"
            ) from exc
    return documents


def load_all_documents(config: KnowledgeOSConfig) -> list[Document]:
    """Load all supported local document types."""

    return [*load_text_documents(config), *load_pdf_documents(config)]


def summarize_documents(documents: list[Document]) -> dict[str, Any]:
    """Return an inspectable document summary suitable for notebook display."""

    loader_counts = Counter(
        str(document.metadata.get("loader_type", "unknown")) for document in documents
    )
    total_characters = sum(len(document.page_content) for document in documents)
    return {
        "document_count": len(documents),
        "total_characters": total_characters,
        "average_characters": round(total_characters / len(documents), 1)
        if documents
        else 0.0,
        "loader_counts": dict(sorted(loader_counts.items())),
        "sources": sorted(
            {str(document.metadata.get("file_name", "")) for document in documents}
        ),
    }
