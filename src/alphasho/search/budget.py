"""Desktop-match search budgets derived from a shogi clock."""

from __future__ import annotations

import math
from dataclasses import dataclass

from alphasho.domain import (
    ClockState,
    Color,
    ThinkingConfig,
    ThinkingPreset,
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


def allocate_thinking_budget(
    config: ThinkingConfig,
    control: TimeControl,
    clock: ClockState,
    color: Color,
    *,
    move_number: int = 1,
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
        limit = _automatic_seconds(config, control, clock, color, move_number=move_number)

    if limit is not None:
        limit = max(0.01, float(limit))
        if control.mode is not TimeControlMode.NONE and control.time_forfeit:
            safety = config.safety_margin_ms / 1000.0
            remaining = _clock_available_seconds(control, clock, color)
            limit = min(limit, max(0.01, remaining - safety))

    max_playouts = max(1, config.max_playouts)
    if (
        config.strategy is ThinkingStrategy.AUTO_TIME
        and control.mode is not TimeControlMode.NONE
        and limit is not None
    ):
        max_playouts = max(max_playouts, min(10_000_000, math.ceil(limit * 100_000)))

    return ThinkingBudget(
        time_limit_seconds=limit,
        min_playouts=max(1, config.min_playouts),
        max_playouts=max_playouts,
        max_depth=max(1, config.max_depth),
    )


def _automatic_seconds(
    config: ThinkingConfig,
    control: TimeControl,
    clock: ClockState,
    color: Color,
    *,
    move_number: int,
) -> float:
    if control.mode is TimeControlMode.NONE:
        return max(0.01, config.move_time_seconds)

    main = clock.remaining_for(color) / 1000.0
    overtime = max(0.0, float(control.for_color(color).overtime_seconds))
    own_moves_played = max(0, (max(1, move_number) - 1) // 2)
    expected_moves, minimum_moves = {
        ThinkingPreset.QUICK: (80, 30),
        ThinkingPreset.BALANCED: (60, 20),
        ThinkingPreset.DEEP: (40, 15),
        ThinkingPreset.CUSTOM: (60, 20),
    }[config.preset]
    moves_to_go = max(minimum_moves, expected_moves - own_moves_played)
    if control.mode is TimeControlMode.BYOYOMI and main <= 0:
        calculated = clock.overtime_for(color) / 1000.0 * 0.8
    else:
        calculated = main * 0.9 / moves_to_go + overtime * 0.8
    return max(0.01, calculated)


def _clock_available_seconds(
    control: TimeControl,
    clock: ClockState,
    color: Color,
) -> float:
    remaining = clock.remaining_for(color) / 1000.0
    if control.mode is TimeControlMode.BYOYOMI:
        remaining += clock.overtime_for(color) / 1000.0
    return remaining


__all__ = ["ThinkingBudget", "allocate_thinking_budget"]
