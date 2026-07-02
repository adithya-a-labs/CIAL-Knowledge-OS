"""Tokenizer-aware counting, truncation, and context budgeting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Subset shared by local Hugging Face-compatible tokenizers."""

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenBudgetUsage:
    """Inspectable token-budget outcome for one constructed context."""

    budget: int
    used: int
    remaining: int
    truncated_sections: int
    omitted_sections: int


class TokenBudgetManager:
    """Enforce exact limits using an injected local tokenizer.

    The manager deliberately does not load models. Callers supply an already
    loaded tokenizer so context construction cannot trigger downloads or repeat
    expensive initialization.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide an encode(text) method.")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.last_usage = TokenBudgetUsage(
            budget=max_tokens,
            used=0,
            remaining=max_tokens,
            truncated_sections=0,
            omitted_sections=0,
        )

    def token_ids(self, text: str) -> list[int]:
        """Encode without model-specific special tokens where supported."""

        try:
            values = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            values = self.tokenizer.encode(text)
        return [int(value) for value in values]

    def count(self, text: str) -> int:
        """Return the configured tokenizer's exact token count."""

        return len(self.token_ids(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to at most ``max_tokens`` without guessing by chars."""

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative.")
        token_ids = self.token_ids(text)
        if len(token_ids) <= max_tokens:
            return text
        if max_tokens == 0:
            return ""
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError(
                "Tokenizer-aware truncation requires tokenizer.decode(token_ids)."
            )
        selected = token_ids[:max_tokens]
        try:
            value = decode(
                selected,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = decode(selected)
        truncated = str(value).rstrip()
        while truncated and self.count(truncated) > max_tokens:
            selected = selected[:-1]
            truncated = str(decode(selected)).rstrip() if selected else ""
        return truncated

    def record_usage(
        self,
        *,
        used: int,
        truncated_sections: int,
        omitted_sections: int,
    ) -> TokenBudgetUsage:
        """Store and log one completed context-budget calculation."""

        if used > self.max_tokens:
            raise ValueError(
                f"Token budget overflow: used {used} tokens with a configured "
                f"maximum of {self.max_tokens}."
            )
        self.last_usage = TokenBudgetUsage(
            budget=self.max_tokens,
            used=used,
            remaining=self.max_tokens - used,
            truncated_sections=truncated_sections,
            omitted_sections=omitted_sections,
        )
        logger.info(
            "token_budget_complete",
            extra={
                "event": "token_budget",
                "token_budget": self.max_tokens,
                "tokens_used": used,
                "truncated_sections": truncated_sections,
                "omitted_sections": omitted_sections,
            },
        )
        return self.last_usage
