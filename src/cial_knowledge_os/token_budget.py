"""Centralized, swappable token counting and context-budget management."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
_PACKAGED_TIKTOKEN_ENCODINGS = {"cl100k_base"}
_TIKTOKEN_LOAD_LOCK = threading.Lock()


class Tokenizer(Protocol):
    """Minimal codec contract for tiktoken and future model tokenizers."""

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


@lru_cache(maxsize=8)
def _load_tiktoken_encoding(name: str) -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "Token management requires tiktoken. Install the pinned project "
            "dependencies from requirements.txt."
        ) from exc
    if name not in _PACKAGED_TIKTOKEN_ENCODINGS:
        raise ValueError(
            f"Tiktoken encoding '{name}' is not packaged for offline use. "
            "Use 'cl100k_base' or inject a compatible local tokenizer."
        )
    asset_directory = Path(__file__).with_name("assets")
    if not asset_directory.is_dir():
        raise RuntimeError(
            "The packaged tiktoken vocabulary is missing. Reinstall the project "
            "with package data enabled."
        )
    with _TIKTOKEN_LOAD_LOCK:
        previous_cache = os.environ.get("TIKTOKEN_CACHE_DIR")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(asset_directory)
        try:
            return tiktoken.get_encoding(name)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not initialize packaged tiktoken encoding '{name}': "
                f"{exc}"
            ) from exc
        finally:
            if previous_cache is None:
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
            else:
                os.environ["TIKTOKEN_CACHE_DIR"] = previous_cache


class TiktokenTokenizer:
    """Adapt a named tiktoken encoding to the project tokenizer protocol."""

    def __init__(self, encoding_name: str = DEFAULT_TIKTOKEN_ENCODING) -> None:
        if not encoding_name.strip():
            raise ValueError("encoding_name must not be blank.")
        self.encoding_name = encoding_name.strip()
        self._encoding = _load_tiktoken_encoding(self.encoding_name)

    def encode(self, text: str, **_: Any) -> list[int]:
        return list(self._encoding.encode(text, disallowed_special=()))

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return str(self._encoding.decode(token_ids))


@dataclass(frozen=True, slots=True)
class TokenBudgetUsage:
    """Exact tokenizer usage for one bounded context."""

    budget: int
    used: int
    remaining: int
    truncated_sections: int
    omitted_sections: int
    encoding_name: str


class TokenManager:
    """Single authority for exact token counting and token-safe truncation."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    ) -> None:
        self.tokenizer = tokenizer or TiktokenTokenizer(encoding_name)
        if not callable(getattr(self.tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide an encode(text) method.")
        if not callable(getattr(self.tokenizer, "decode", None)):
            raise TypeError("tokenizer must provide a decode(token_ids) method.")
        self.encoding_name = str(
            getattr(self.tokenizer, "encoding_name", type(self.tokenizer).__name__)
        )

    def token_ids(self, text: str) -> list[int]:
        """Encode text through the configured tokenizer without estimation."""

        try:
            values = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            values = self.tokenizer.encode(text)
        return [int(value) for value in values]

    def count(self, text: str) -> int:
        """Return the exact configured-tokenizer count."""

        return len(self.token_ids(text))

    def remaining(self, *, used_tokens: int, max_tokens: int) -> int:
        """Return remaining capacity and reject an already-overflowed budget."""

        if used_tokens < 0:
            raise ValueError("used_tokens must be non-negative.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if used_tokens > max_tokens:
            raise ValueError(
                f"Token budget overflow: used {used_tokens} tokens with a "
                f"configured maximum of {max_tokens}."
            )
        return max_tokens - used_tokens

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate on tokenizer boundaries to at most ``max_tokens``."""

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative.")
        token_ids = self.token_ids(text)
        if len(token_ids) <= max_tokens:
            return text
        if max_tokens == 0:
            return ""
        selected = token_ids[:max_tokens]
        try:
            value = self.tokenizer.decode(
                selected,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = self.tokenizer.decode(selected)
        truncated = str(value).rstrip()
        while truncated and self.count(truncated) > max_tokens:
            selected = selected[:-1]
            truncated = (
                str(self.tokenizer.decode(selected)).rstrip()
                if selected
                else ""
            )
        return truncated


class TokenBudgetManager(TokenManager):
    """Add a fixed context budget and usage reporting to :class:`TokenManager`."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        max_tokens: int,
        encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        super().__init__(tokenizer, encoding_name=encoding_name)
        self.max_tokens = max_tokens
        self.last_usage = TokenBudgetUsage(
            budget=max_tokens,
            used=0,
            remaining=max_tokens,
            truncated_sections=0,
            omitted_sections=0,
            encoding_name=self.encoding_name,
        )

    def record_usage(
        self,
        *,
        used: int,
        truncated_sections: int,
        omitted_sections: int,
    ) -> TokenBudgetUsage:
        """Record exact usage and calculate remaining context centrally."""

        self.last_usage = TokenBudgetUsage(
            budget=self.max_tokens,
            used=used,
            remaining=self.remaining(
                used_tokens=used,
                max_tokens=self.max_tokens,
            ),
            truncated_sections=truncated_sections,
            omitted_sections=omitted_sections,
            encoding_name=self.encoding_name,
        )
        logger.info(
            "token_budget_complete",
            extra={
                "event": "token_budget",
                "token_budget": self.max_tokens,
                "tokens_used": used,
                "tokens_remaining": self.last_usage.remaining,
                "token_encoding": self.encoding_name,
                "truncated_sections": truncated_sections,
                "omitted_sections": omitted_sections,
            },
        )
        return self.last_usage


def create_token_manager(
    *,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    tokenizer: Tokenizer | None = None,
) -> TokenManager:
    """Create the shared tiktoken-backed manager or an injected replacement."""

    return TokenManager(tokenizer, encoding_name=encoding_name)


def create_token_budget_manager(
    *,
    max_tokens: int,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    tokenizer: Tokenizer | None = None,
) -> TokenBudgetManager:
    """Create one exact token-budget manager for context construction."""

    return TokenBudgetManager(
        tokenizer,
        max_tokens=max_tokens,
        encoding_name=encoding_name,
    )
