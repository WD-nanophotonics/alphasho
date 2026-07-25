"""Time-bounded iterative-deepening alpha-beta player."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum

import cshogi

from alphasho.domain import (
    ClockState,
    Color,
    GameSnapshot,
    MoveOption,
    ThinkingConfig,
    ThinkingStrategy,
    TimeControl,
    TimeControlMode,
)

MATE = 1_000_000
INF = MATE + 10_000
PIECE_VALUES = (0, 100, 300, 320, 450, 800, 1_000, 520, 0, 520, 520, 520, 520, 950, 1_150)
HAND_VALUES = (100, 300, 320, 450, 520, 800, 1_000)


class SearchCancelled(RuntimeError):
    pass


class _Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass(slots=True)
class _Entry:
    depth: int
    score: int
    bound: _Bound
    best_move: str | None


@dataclass(frozen=True, slots=True)
class HeuristicSearchInfo:
    elapsed_seconds: float
    budget_seconds: float | None
    playouts: int
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HeuristicSearchProgress:
    completed_playouts: int
    max_playouts: int
    elapsed_seconds: float
    budget_seconds: float | None
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]


class HeuristicPlayer:
    """Classical negamax player with no neural/training dependencies."""

    def __init__(self, thinking: ThinkingConfig | None = None) -> None:
        self.thinking = thinking or ThinkingConfig()
        self.last_search_info: HeuristicSearchInfo | None = None
        self._cancelled = threading.Event()
        self._deadline: float | None = None
        self._start = 0.0
        self._nodes = 0
        self._node_limit = 1
        self._table: dict[str, _Entry] = {}

    def cancel(self) -> None:
        self._cancelled.set()

    def choose_move(self, snapshot: GameSnapshot, legal_moves: Sequence[MoveOption]) -> MoveOption:
        return self._search(snapshot, legal_moves, None, None)

    def choose_move_with_context(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        *,
        clock_state: ClockState,
        time_control: TimeControl,
        color: Color,
        progress_callback: Callable[[HeuristicSearchProgress], None] | None = None,
    ) -> MoveOption:
        seconds = self._time_budget(clock_state, time_control, color)
        return self._search(snapshot, legal_moves, seconds, progress_callback)

    def _search(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        seconds: float | None,
        progress_callback: Callable[[HeuristicSearchProgress], None] | None,
    ) -> MoveOption:
        if not legal_moves:
            raise ValueError("HeuristicPlayer requires at least one legal move")
        board = cshogi.Board(snapshot.initial_sfen or snapshot.sfen)
        if snapshot.initial_sfen:
            for usi in snapshot.move_history:
                board.push_usi(usi)
        if board.sfen() != snapshot.sfen:
            raise ValueError("snapshot replay mismatch")
        self._cancelled.clear()
        self._start = time.monotonic()
        self._deadline = None if seconds is None else self._start + max(0.01, seconds)
        self._nodes = 0
        self._node_limit = max(1, self.thinking.max_playouts)
        self._table.clear()
        best_usi, best_score, completed_depth = legal_moves[0].usi, 0, 0
        pv: tuple[str, ...] = (best_usi,)
        for depth in range(1, max(1, self.thinking.max_depth) + 1):
            try:
                score, usi = self._root(board, depth, best_usi)
            except TimeoutError:
                break
            if usi is not None:
                best_usi, best_score, completed_depth = usi, score, depth
                pv = self._principal_variation(board, depth)
                if progress_callback is not None:
                    progress_callback(self._progress(seconds, depth, score, pv))
            if abs(best_score) >= MATE - 256:
                break
        if self._cancelled.is_set():
            raise SearchCancelled()
        elapsed = time.monotonic() - self._start
        self.last_search_info = HeuristicSearchInfo(
            elapsed, seconds, self._nodes, completed_depth, self._normalized(best_score), pv
        )
        return {move.usi: move for move in legal_moves}.get(best_usi, legal_moves[0])

    def _root(self, board: cshogi.Board, depth: int, preferred: str) -> tuple[int, str | None]:
        alpha, best_score, best_move = -INF, -INF, None
        for move in self._ordered_moves(board, preferred):
            self._check_limits()
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, cshogi.move_to_usi(move)
            alpha = max(alpha, score)
        return best_score, best_move

    def _negamax(self, board: cshogi.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._nodes += 1
        self._check_limits()
        if board.is_game_over():
            return -MATE + ply
        if board.is_draw():
            return 0
        if depth <= 0:
            return self._evaluate(board)
        key = " ".join(board.sfen().split()[:3])
        entry = self._table.get(key)
        original_alpha = alpha
        if entry is not None and entry.depth >= depth:
            if entry.bound is _Bound.EXACT:
                return entry.score
            if entry.bound is _Bound.LOWER:
                alpha = max(alpha, entry.score)
            else:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score
        best, best_usi = -INF, None
        for move in self._ordered_moves(board, entry.best_move if entry else None):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best, best_usi = score, cshogi.move_to_usi(move)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        bound = (
            _Bound.UPPER
            if best <= original_alpha
            else _Bound.LOWER
            if best >= beta
            else _Bound.EXACT
        )
        self._table[key] = _Entry(depth, best, bound, best_usi)
        return best

    @staticmethod
    def _evaluate(board: cshogi.Board) -> int:
        black = 0
        for raw in board.pieces:
            piece = int(raw)
            if piece:
                white = piece >= 16
                value = PIECE_VALUES[piece - 16 if white else piece]
                black += -value if white else value
        for index, value in enumerate(HAND_VALUES):
            black += value * int(board.pieces_in_hand[cshogi.BLACK][index])
            black -= value * int(board.pieces_in_hand[cshogi.WHITE][index])
        score = black if board.turn == cshogi.BLACK else -black
        return score - 35 if board.is_check() else score

    @staticmethod
    def _ordered_moves(board: cshogi.Board, preferred: str | None) -> list[int]:
        def priority(move: int) -> int:
            if cshogi.move_to_usi(move) == preferred:
                return 10_000_000
            captured = int(board.piece(cshogi.move_to(move)))
            captured_type = captured - 16 if captured >= 16 else captured
            return (PIECE_VALUES[captured_type] if captured_type else 0) * 10 + (
                500 if cshogi.move_is_promotion(move) else 0
            )
        return sorted((int(move) for move in board.legal_moves), key=priority, reverse=True)

    def _principal_variation(self, board: cshogi.Board, depth: int) -> tuple[str, ...]:
        clone, result = board.copy(), []
        for _ in range(depth):
            entry = self._table.get(" ".join(clone.sfen().split()[:3]))
            if entry is None or entry.best_move is None:
                break
            legal = {cshogi.move_to_usi(move): int(move) for move in clone.legal_moves}
            move = legal.get(entry.best_move)
            if move is None:
                break
            result.append(entry.best_move)
            clone.push(move)
        return tuple(result)

    def _check_limits(self) -> None:
        if self._cancelled.is_set():
            raise SearchCancelled()
        if self._nodes >= self._node_limit:
            raise TimeoutError
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError

    def _time_budget(self, clock: ClockState, control: TimeControl, color: Color) -> float | None:
        config = self.thinking
        available = clock.remaining_for(color) / 1000.0
        if control.mode is TimeControlMode.BYOYOMI:
            available += clock.overtime_for(color) / 1000.0
        if config.strategy is ThinkingStrategy.FIXED_PLAYOUT:
            seconds = None if control.mode is TimeControlMode.NONE else available
        elif config.strategy is ThinkingStrategy.FIXED_TIME:
            seconds = config.move_time_seconds
        elif control.mode is TimeControlMode.NONE:
            seconds = config.move_time_seconds
        else:
            main = clock.remaining_for(color) / 1000.0
            overtime = control.for_color(color).overtime_seconds
            seconds = overtime * 0.8 if main <= 0 else main * 0.95 / 40.0 + overtime * 0.8
            seconds = min(config.move_time_seconds, seconds)
        if seconds is None:
            return None
        if control.mode is not TimeControlMode.NONE and control.time_forfeit:
            seconds = min(seconds, available - config.safety_margin_ms / 1000.0)
        return max(0.01, seconds)

    def _progress(
        self,
        seconds: float | None,
        depth: int,
        score: int,
        pv: tuple[str, ...],
    ) -> HeuristicSearchProgress:
        return HeuristicSearchProgress(
            self._nodes, self._node_limit, time.monotonic() - self._start,
            seconds, depth, self._normalized(score), pv,
        )

    @staticmethod
    def _normalized(score: int) -> float:
        return math.copysign(1.0, score) if abs(score) >= MATE - 256 else math.tanh(score / 2_000.0)


__all__ = ["HeuristicPlayer", "HeuristicSearchInfo", "HeuristicSearchProgress", "SearchCancelled"]
