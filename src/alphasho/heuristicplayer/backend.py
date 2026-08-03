"""Search backend boundary; native implementations may be added without UI changes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from alphasho.search import PlayerSearchProgress

from .search import SearchResult


class SearchBackend(Protocol):
    name: str

    def run(
        self,
        *,
        fallback_usi: str,
        max_depth: int,
        progress_callback: Callable[[PlayerSearchProgress], None] | None = None,
    ) -> SearchResult: ...


class PythonSearchBackend:
    name = "python"

    def __init__(self, search: object) -> None:
        self.search = search

    def run(
        self,
        *,
        fallback_usi: str,
        max_depth: int,
        progress_callback: Callable[[PlayerSearchProgress], None] | None = None,
    ) -> SearchResult:
        return self.search.run(  # type: ignore[no-any-return,union-attr]
            fallback_usi=fallback_usi,
            max_depth=max_depth,
            progress_callback=progress_callback,
        )


__all__ = ["PythonSearchBackend", "SearchBackend"]
