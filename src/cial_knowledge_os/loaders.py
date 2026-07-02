"""Local text and PDF document ingestion."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import KnowledgeOSConfig

logger = logging.getLogger(__name__)

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


def _base_metadata(path: Path, loader_type: str, page_number: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": str(path.resolve()),
        "file_name": path.name,
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


def _load_pdf_with_docling(path: Path) -> list[Document]:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    text = result.document.export_to_markdown().strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata=_base_metadata(path, "docling"),
        )
    ]


def _load_pdf_with_pymupdf(path: Path) -> list[Document]:
    import fitz

    documents: list[Document] = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=_base_metadata(path, "pymupdf", index),
                    )
                )
    return documents


def load_pdf_documents(config: KnowledgeOSConfig) -> list[Document]:
    """Load local PDFs, preferring Docling and falling back to PyMuPDF."""

    pdf_paths = (
        sorted(config.pdf_data_dir.rglob("*.pdf"))
        if config.pdf_data_dir.exists()
        else []
    )
    if not pdf_paths:
        logger.info(
            "pdf_corpus_empty",
            extra={
                "event": "document_loading",
                "pdf_data_dir": str(config.pdf_data_dir),
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
                docling_documents = _load_pdf_with_docling(path)
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
            documents.extend(_load_pdf_with_pymupdf(path))
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
