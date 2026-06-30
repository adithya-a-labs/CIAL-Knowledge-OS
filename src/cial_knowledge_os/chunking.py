"""Metadata-preserving document chunking."""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import KnowledgeOSConfig


def chunk_documents(
    documents: list[Document], config: KnowledgeOSConfig
) -> list[Document]:
    """Split documents and add stable, traceable chunk metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    source_indexes: Counter[str] = Counter()
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_index = source_indexes[source]
        source_indexes[source] += 1
        page = chunk.metadata.get("page_number", "na")
        file_name = str(chunk.metadata.get("file_name", "document"))
        chunk.metadata.update(
            {
                "chunk_id": f"{file_name}:p{page}:c{chunk_index}",
                "chunk_index": chunk_index,
                "chunk_char_count": len(chunk.page_content),
            }
        )
    return chunks


def summarize_chunks(chunks: list[Document]) -> dict[str, Any]:
    """Return chunk-count and character-size diagnostics."""

    sizes = [len(chunk.page_content) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "min_characters": min(sizes) if sizes else 0,
        "max_characters": max(sizes) if sizes else 0,
        "average_characters": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        "source_count": len(
            {str(chunk.metadata.get("source", "")) for chunk in chunks}
        ),
    }
