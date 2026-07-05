"""Composable dense, lexical, and hybrid retrieval implementations."""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from .fusion import ReciprocalRankFusion
from .metadata import normalize_result

logger = logging.getLogger(__name__)
Tokenize = Callable[[str], list[str]]


class Retriever(Protocol):
    """Stable retrieval contract used by Phase 3 and future methods."""

    name: str

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]: ...


def default_bm25_tokenize(text: str) -> list[str]:
    """Tokenize lexical text deterministically without external resources."""

    return re.findall(r"[\w]+(?:[-'][\w]+)*", text.casefold(), flags=re.UNICODE)


@dataclass(slots=True)
class DenseRetriever:
    """Adapt the existing dense search function to :class:`Retriever`."""

    search: Callable[[str, int], list[dict[str, Any]]]
    name: str = "dense"

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        results = [normalize_result(item) for item in self.search(query, top_k)]
        for rank, result in enumerate(results, start=1):
            result["retrieval_sources"] = [self.name]
            result["retrieval_ranks"] = {self.name: rank}
            result["retrieval_scores"] = {self.name: result.get("score")}
        logger.info(
            "dense_retrieval_complete",
            extra={"event": "dense_retrieval", "result_count": len(results)},
        )
        return results


@dataclass(frozen=True, slots=True)
class _BM25Cache:
    version: int
    fingerprint: str
    tokenized_corpus: list[list[str]]


class BM25Retriever:
    """In-memory BM25 retriever with validated persistent token cache."""

    name = "bm25"
    _CACHE_VERSION = 1

    def __init__(
        self,
        *,
        tokenizer: Tokenize = default_bm25_tokenize,
        k1: float = 1.5,
        b: float = 0.75,
        cache_path: str | Path | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one.")
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.cache_path = (
            Path(cache_path).expanduser().resolve()
            if cache_path is not None
            else None
        )
        self._chunks: list[dict[str, Any]] = []
        self._index: Any | None = None
        self._fingerprint: str | None = None

    @property
    def is_indexed(self) -> bool:
        return self._index is not None

    @staticmethod
    def _normalize_chunk(chunk: Document | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(chunk, Document):
            metadata = dict(chunk.metadata)
            return normalize_result(
                {
                    "text": chunk.page_content,
                    "metadata": metadata,
                    "source": metadata.get("file_name") or metadata.get("source"),
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_id"),
                }
            )
        return normalize_result(chunk)

    @staticmethod
    def _corpus_fingerprint(chunks: Sequence[Mapping[str, Any]]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(str(chunk.get("source_path") or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("chunk_id") or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("text") or "").encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _load_cached_tokens(self, fingerprint: str) -> list[list[str]] | None:
        if self.cache_path is None or not self.cache_path.is_file():
            return None
        try:
            with self.cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                not isinstance(cached, _BM25Cache)
                or cached.version != self._CACHE_VERSION
                or cached.fingerprint != fingerprint
            ):
                return None
            return cached.tokenized_corpus
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as exc:
            logger.warning(
                "bm25_cache_invalid",
                extra={"event": "bm25_cache", "error": str(exc)},
            )
            return None

    def _save_cached_tokens(
        self,
        fingerprint: str,
        tokenized_corpus: list[list[str]],
    ) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(
                    _BM25Cache(
                        self._CACHE_VERSION,
                        fingerprint,
                        tokenized_corpus,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary.replace(self.cache_path)
        except OSError as exc:
            logger.warning(
                "bm25_cache_write_failed",
                extra={"event": "bm25_cache", "error": str(exc)},
            )
            temporary.unlink(missing_ok=True)

    def index(
        self,
        chunks: Sequence[Document | Mapping[str, Any]],
    ) -> bool:
        """Index the corpus and return whether work was performed."""

        normalized = [self._normalize_chunk(chunk) for chunk in chunks]
        if not normalized:
            performed = self._index is None or bool(self._chunks)
            self._chunks = []
            self._index = []
            self._fingerprint = self._corpus_fingerprint([])
            logger.info(
                "bm25_index_ready",
                extra={
                    "event": "bm25_index",
                    "chunk_count": 0,
                    "cache_hit": False,
                },
            )
            return performed
        fingerprint = self._corpus_fingerprint(normalized)
        if self._index is not None and fingerprint == self._fingerprint:
            logger.debug("bm25_index_reused", extra={"event": "bm25_index"})
            return False
        tokenized = self._load_cached_tokens(fingerprint)
        cache_hit = tokenized is not None
        if tokenized is None:
            tokenized = [self.tokenizer(chunk["text"]) for chunk in normalized]
            self._save_cached_tokens(fingerprint, tokenized)
        if not any(tokenized):
            raise ValueError(
                "Cannot build the BM25 index because all chunks are empty after "
                "tokenization. Check document extraction and tokenizer settings."
            )
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "BM25 retrieval requires rank-bm25. Install the pinned project "
                "dependencies from requirements.txt."
            ) from exc
        self._chunks = normalized
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._fingerprint = fingerprint
        logger.info(
            "bm25_index_ready",
            extra={
                "event": "bm25_index",
                "chunk_count": len(normalized),
                "cache_hit": cache_hit,
            },
        )
        return True

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self._index is None:
            raise RuntimeError(
                "The BM25 index is not initialized. Call index(chunks) before "
                "retrieve()."
            )
        if not self._chunks:
            return []
        tokens = self.tokenizer(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )
        results: list[dict[str, Any]] = []
        for index in ranked_indexes:
            score = float(scores[index])
            if score <= 0:
                continue
            result = dict(self._chunks[index])
            result.update(
                {
                    "score": score,
                    "retrieval_sources": [self.name],
                    "retrieval_ranks": {self.name: len(results) + 1},
                    "retrieval_scores": {self.name: score},
                }
            )
            results.append(result)
            if len(results) >= top_k:
                break
        logger.info(
            "bm25_retrieval_complete",
            extra={"event": "bm25_retrieval", "result_count": len(results)},
        )
        return results


class HybridRetriever:
    """Compose arbitrary retrievers through a configurable fusion strategy."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        fuser: ReciprocalRankFusion,
        candidate_limits: Mapping[str, int],
        parallel: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever.")
        names = [retriever.name for retriever in retrievers]
        if len(names) != len(set(names)):
            raise ValueError("Hybrid retriever names must be unique.")
        self.retrievers = tuple(retrievers)
        self.fuser = fuser
        self.candidate_limits = dict(candidate_limits)
        self.parallel = parallel
        self.last_rankings: dict[str, list[dict[str, Any]]] = {}

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        def search(retriever: Retriever) -> list[dict[str, Any]]:
            candidate_limit = self.candidate_limits.get(retriever.name, top_k)
            return retriever.retrieve(
                query,
                top_k=candidate_limit,
            )

        if self.parallel and len(self.retrievers) > 1:
            with ThreadPoolExecutor(
                max_workers=len(self.retrievers),
                thread_name_prefix="cial-retrieval",
            ) as executor:
                futures = {
                    retriever.name: executor.submit(search, retriever)
                    for retriever in self.retrievers
                }
                rankings = {
                    retriever.name: futures[retriever.name].result()
                    for retriever in self.retrievers
                }
        else:
            rankings = {
                retriever.name: search(retriever)
                for retriever in self.retrievers
            }
        self.last_rankings = {
            name: [dict(result) for result in values]
            for name, values in rankings.items()
        }
        results = self.fuser.fuse(rankings, limit=top_k)
        logger.info(
            "hybrid_retrieval_complete",
            extra={
                "event": "hybrid_retrieval",
                "result_count": len(results),
                "candidate_counts": {
                    name: len(values) for name, values in rankings.items()
                },
            },
        )
        return results
