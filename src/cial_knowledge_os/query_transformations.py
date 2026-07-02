"""Deterministic, offline query transformations with pluggable strategies."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .config import Phase2Config

TransformFunction = Callable[[str], str]

_POLITE_PREFIX = re.compile(
    r"^(?:please\s+|could you\s+|can you\s+|would you\s+|i want to know\s+)+",
    re.IGNORECASE,
)
_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "runway": ("airside", "surface inspection", "FOD", "ATC clearance"),
    "queue": ("passenger flow", "terminal operations", "holding area"),
    "electrical": ("energized panel", "PPE", "lockout-tagout", "LOTO"),
    "cyber": ("information security", "controls", "risk", "incident response"),
    "maintenance": ("inspection", "preventive maintenance", "SOP"),
    "safety": ("hazard", "control measure", "PPE", "emergency procedure"),
}


@dataclass(frozen=True, slots=True)
class QueryVariant:
    """One inspectable query representation used for retrieval."""

    technique: str
    query: str

    def as_dict(self) -> dict[str, str]:
        return {"technique": self.technique, "query": self.query}


def _clean(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        raise ValueError("query must not be empty.")
    return cleaned


def rewrite_query(query: str) -> str:
    """Remove conversational framing while preserving user intent."""

    cleaned = _POLITE_PREFIX.sub("", _clean(query)).strip()
    cleaned = re.sub(r"^(?:tell me|explain)\s+(?:about\s+)?", "", cleaned, flags=re.I)
    cleaned = cleaned.rstrip(" ?.!")
    return cleaned[0].upper() + cleaned[1:] if cleaned else _clean(query)


def expand_query_keywords(
    query: str,
    *,
    glossary: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Append deterministic domain synonyms for terms present in the query."""

    cleaned = _clean(query)
    term_map = glossary or _DOMAIN_TERMS
    lowered = cleaned.casefold()
    expansions: list[str] = []
    for term, related_terms in term_map.items():
        if term.casefold() in lowered:
            expansions.extend(str(value) for value in related_terms)
    unique = list(
        dict.fromkeys(
            value for value in expansions if value.casefold() not in lowered
        )
    )
    return cleaned if not unique else f"{cleaned} | Related terms: {', '.join(unique)}"


def reformulate_for_domain(query: str, *, domain: str = "CIAL airport operations") -> str:
    """Frame a question using corpus-specific enterprise terminology."""

    cleaned = rewrite_query(query)
    return (
        f"{domain} policy, SOP, manual, controls, responsibilities, and required "
        f"actions: {cleaned}"
    )


class QueryTransformer:
    """Registry-based query transformer prepared for future local strategies."""

    def __init__(
        self,
        config: Phase2Config,
        *,
        strategies: Mapping[str, TransformFunction] | None = None,
    ) -> None:
        self.config = config
        self._strategies: dict[str, TransformFunction] = {
            "original": _clean,
            "rewritten": rewrite_query,
            "keyword_expanded": expand_query_keywords,
            "domain_reformulation": reformulate_for_domain,
        }
        if strategies:
            self._strategies.update(strategies)

    def register(self, name: str, transform: TransformFunction) -> None:
        """Register a future deterministic or local-model transformation."""

        if not name.strip():
            raise ValueError("strategy name must not be empty.")
        self._strategies[name] = transform

    def transform(self, query: str, technique: str) -> QueryVariant:
        """Apply one named technique independently."""

        try:
            strategy = self._strategies[technique]
        except KeyError as exc:
            raise ValueError(f"Unknown query transformation: {technique}") from exc
        return QueryVariant(technique=technique, query=strategy(query))

    def generate(self, query: str) -> list[QueryVariant]:
        """Generate configured variants in deterministic order."""

        names = ["original"]
        if self.config.enable_query_rewrite:
            names.append("rewritten")
        if self.config.enable_keyword_expansion:
            names.append("keyword_expanded")
        if self.config.enable_domain_reformulation:
            names.append("domain_reformulation")
        if not self.config.enable_multi_query:
            names = names[:1]

        variants: list[QueryVariant] = []
        seen: set[str] = set()
        for name in names:
            variant = self.transform(query, name)
            key = variant.query.casefold()
            if key not in seen:
                variants.append(variant)
                seen.add(key)
            if len(variants) >= self.config.max_query_variants:
                break
        return variants
