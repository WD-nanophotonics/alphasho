"""Deterministic match-clock rules, independent of widgets."""

from __future__ import annotations

import time
from collections.abc import Callable

from alphasho.domain import ClockState, Color, TimeControl, TimeControlMode


class MatchClock:
    def __init__(
        self,
        control: TimeControl,
        *,
        turn: Color = Color.BLACK,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.control = control
        self._now = now
        self._remaining = [
            control.black.main_seconds * 1000,
            control.white.main_seconds * 1000,
        ]
        self._overtime = [
            control.black.overtime_seconds * 1000,
            control.white.overtime_seconds * 1000,
        ]
        self._active = turn
        self._running = control.mode is not TimeControlMode.NONE
        self._started_at = now()
        self._turn_elapsed_ms = 0

    def state(self) -> ClockState:
        remaining = list(self._remaining)
        overtime = list(self._overtime)
        if self._running:
            elapsed = self._elapsed_since_start()
            self._deduct(self._active, elapsed, remaining, overtime)
        expired = self._expired_color(remaining, overtime)
        return ClockState(
            remaining_ms=(max(0, remaining[0]), max(0, remaining[1])),
            overtime_remaining_ms=(max(0, overtime[0]), max(0, overtime[1])),
            active_color=self._active,
            running=self._running,
            expired_color=expired,
        )

    def complete_turn(self, mover: Color) -> int:
        if mover is not self._active:
            raise ValueError("Clock mover does not match the active side")
        self._commit_elapsed()
        elapsed = self._turn_elapsed_ms
        side = self.control.for_color(mover)
        if self.control.mode is TimeControlMode.FISCHER:
            self._remaining[int(mover)] += side.overtime_seconds * 1000
        elif self.control.mode is TimeControlMode.BYOYOMI:
            self._overtime[int(mover)] = side.overtime_seconds * 1000
        self._active = mover.opponent
        self._turn_elapsed_ms = 0
        self._started_at = self._now()
        self._running = self.control.mode is not TimeControlMode.NONE
        return max(0, (elapsed + 999) // 1000)

    def pause(self) -> None:
        if not self._running:
            return
        self._commit_elapsed()
        self._running = False

    def resume(self) -> None:
        if self._running or self.control.mode is TimeControlMode.NONE:
            return
        self._started_at = self._now()
        self._running = True

    def restore(self, state: ClockState) -> None:
        self._remaining = list(state.remaining_ms)
        self._overtime = list(state.overtime_remaining_ms)
        self._active = state.active_color
        self._running = (
            state.running and self.control.mode is not TimeControlMode.NONE
        )
        self._turn_elapsed_ms = 0
        self._started_at = self._now()

    def _commit_elapsed(self) -> None:
        if not self._running:
            return
        elapsed = self._elapsed_since_start()
        self._deduct(self._active, elapsed, self._remaining, self._overtime)
        self._turn_elapsed_ms += elapsed
        self._started_at = self._now()

    def _elapsed_since_start(self) -> int:
        return max(0, int((self._now() - self._started_at) * 1000))

    def _deduct(
        self,
        color: Color,
        elapsed: int,
        remaining: list[int],
        overtime: list[int],
    ) -> None:
        index = int(color)
        if self.control.mode is TimeControlMode.NONE:
            return
        main_used = min(max(0, remaining[index]), elapsed)
        remaining[index] -= main_used
        residual = elapsed - main_used
        if residual <= 0:
            return
        if self.control.mode is TimeControlMode.BYOYOMI:
            overtime[index] -= residual
        else:
            remaining[index] -= residual

    def _expired_color(
        self,
        remaining: list[int],
        overtime: list[int],
    ) -> Color | None:
        if not self.control.time_forfeit or self.control.mode is TimeControlMode.NONE:
            return None
        index = int(self._active)
        if self.control.mode is TimeControlMode.BYOYOMI:
            expired = remaining[index] <= 0 and overtime[index] <= 0
        else:
            expired = remaining[index] <= 0
        return self._active if expired else None
