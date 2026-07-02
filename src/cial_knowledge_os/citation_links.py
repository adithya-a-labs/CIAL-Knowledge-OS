"""Build portable PDF citation links from evidence metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urljoin

from .metadata import page_number, result_metadata, source_path


class CitationLinkBuilder:
    """Resolve evidence to file or future local-server PDF page links."""

    def __init__(
        self,
        *,
        mode: Literal["file", "localhost"] = "file",
        base_url: str | None = None,
    ) -> None:
        if mode not in {"file", "localhost"}:
            raise ValueError("Citation link mode must be 'file' or 'localhost'.")
        if mode == "localhost" and not base_url:
            raise ValueError("base_url is required for localhost citation links.")
        self.mode = mode
        self.base_url = base_url.rstrip("/") + "/" if base_url else None

    @staticmethod
    def _page_fragment(value: Any) -> str:
        if value is None or value == "":
            return ""
        first = str(value).split(",", maxsplit=1)[0].strip()
        try:
            page = int(first)
        except (TypeError, ValueError):
            return ""
        return f"#page={page}" if page > 0 else ""

    def build(self, result: Mapping[str, Any]) -> str | None:
        """Return a clickable PDF URL, or ``None`` for incomplete metadata."""

        metadata = result_metadata(result)
        source = source_path(result)
        source_file = metadata.get("file_name")
        candidate = Path(source) if source else None
        if candidate is not None and candidate.suffix.casefold() != ".pdf":
            return None
        if candidate is None and source_file:
            candidate = Path(str(source_file))
        if candidate is None or candidate.suffix.casefold() != ".pdf":
            return None
        fragment = self._page_fragment(page_number(result))
        if self.mode == "file":
            if not candidate.is_absolute():
                return None
            return candidate.expanduser().resolve().as_uri() + fragment
        assert self.base_url is not None
        return urljoin(self.base_url, quote(candidate.name)) + fragment
