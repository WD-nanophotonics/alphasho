"""Desktop-match search budgets derived from a shogi clock."""

from __future__ import annotations

from dataclasses import dataclass

from alphasho.domain import (
    ClockState,
    Color,
    ThinkingConfig,
    ThinkingStrategy,
    TimeControl,
    TimeControlMode,
)


@dataclass(frozen=True, slots=True)
class ThinkingBudget:
    time_limit_seconds: float | None
    min_playouts: int
    max_playouts: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class MoveSearchInfo:
    elapsed_seconds: float
    budget_seconds: float | None
    playouts: int
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlayerSearchProgress:
    completed_playouts: int
    max_playouts: int
    elapsed_seconds: float
    budget_seconds: float | None
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]


def allocate_thinking_budget(
    config: ThinkingConfig,
    control: TimeControl,
    clock: ClockState,
    color: Color,
) -> ThinkingBudget:
    """Return a bounded per-move budget without changing the match clock."""

    if config.strategy is ThinkingStrategy.FIXED_PLAYOUT:
        limit = (
            None
            if control.mode is TimeControlMode.NONE or not control.time_forfeit
            else _clock_available_seconds(control, clock, color)
        )
    elif config.strategy is ThinkingStrategy.FIXED_TIME:
        limit = config.move_time_seconds
    else:
        limit = _automatic_seconds(config, control, clock, color)

    if limit is not None:
        limit = max(0.01, float(limit))
        if control.mode is not TimeControlMode.NONE and control.time_forfeit:
            safety = config.safety_margin_ms / 1000.0
            remaining = _clock_available_seconds(control, clock, color)
            limit = min(limit, max(0.01, remaining - safety))

    return ThinkingBudget(
        time_limit_seconds=limit,
        min_playouts=max(1, config.min_playouts),
        max_playouts=max(1, config.max_playouts),
        max_depth=max(1, config.max_depth),
    )


def _automatic_seconds(
    config: ThinkingConfig,
    control: TimeControl,
    clock: ClockState,
    color: Color,
) -> float:
    cap = max(0.01, config.move_time_seconds)
    if control.mode is TimeControlMode.NONE:
        return cap

    main = clock.remaining_for(color) / 1000.0
    overtime = control.for_color(color).overtime_seconds
    if control.mode is TimeControlMode.BYOYOMI:
        if main <= 0:
            calculated = overtime * 0.8
        else:
            reserve = main * 0.05
            calculated = max(0.0, main - reserve) / 40.0 + overtime * 0.8
    else:
        calculated = main / 40.0 + overtime * 0.8
        calculated = min(calculated, main * 0.2)
    return min(cap, max(0.01, calculated))


def _clock_available_seconds(
    control: TimeControl,
    clock: ClockState,
    color: Color,
) -> float:
    remaining = clock.remaining_for(color) / 1000.0
    if control.mode is TimeControlMode.BYOYOMI:
        remaining += clock.overtime_for(color) / 1000.0
    return remaining


__all__ = [
    "MoveSearchInfo",
    "PlayerSearchProgress",
    "ThinkingBudget",
    "allocate_thinking_budget",
]
