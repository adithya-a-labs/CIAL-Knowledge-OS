"""Local Sentence Transformers embedding helpers."""

from __future__ import annotations

import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import KnowledgeOSConfig


def _offline_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no"}


def load_embedding_model(config: KnowledgeOSConfig) -> SentenceTransformer:
    """Load an embedding model from local storage only."""

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = config.embedding_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Embedding device '{device}' was requested, but CUDA is unavailable."
        )

    print(
        f"Embedding device: {device} "
        f"(TRANSFORMERS_OFFLINE={_offline_enabled('TRANSFORMERS_OFFLINE')}, "
        f"HF_HUB_OFFLINE={_offline_enabled('HF_HUB_OFFLINE')})"
    )
    try:
        return SentenceTransformer(
            config.embedding_model_name,
            device=device,
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Embedding model '{config.embedding_model_name}' is not available "
            "in the local Hugging Face cache. Download it on an approved connected "
            "machine, transfer the cache to this host, and retry offline."
        ) from exc


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Create normalized local embeddings."""

    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Embedding batch size must be greater than zero.")
    if not texts:
        dimension = get_embedding_dimension(model)
        return np.empty((0, dimension), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Return the model's output vector size."""

    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        dimension = int(embed_texts(model, ["dimension probe"]).shape[1])
    return int(dimension)
