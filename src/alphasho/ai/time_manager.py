"""Compatibility exports for search-neutral timing and reporting types."""

from alphasho.search import (
    MoveSearchInfo,
    PlayerSearchProgress,
    ThinkingBudget,
    allocate_thinking_budget,
)

__all__ = [
    "MoveSearchInfo",
    "PlayerSearchProgress",
    "ThinkingBudget",
    "allocate_thinking_budget",
]
