"""Move classification and ordering heuristics for alpha-beta search."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import cshogi

from .evaluation import PIECE_VALUES

_HAND_PIECE_TYPE = (1, 2, 3, 4, 7, 5, 6)


class MovePickerStage(StrEnum):
    TT = "tt"
    GOOD_TACTICAL = "good_tactical"
    KILLER_COUNTER = "killer_counter"
    QUIET = "quiet"
    BAD_CAPTURE = "bad_capture"


def move_moving_type(move: int) -> int:
    moving = int(cshogi.move_from_piece_type(move))
    if cshogi.move_is_drop(move):
        return _HAND_PIECE_TYPE[int(cshogi.move_drop_hand_piece(move))]
    return moving


def move_is_tactical(move: int) -> bool:
    return bool(cshogi.move_cap(move) or cshogi.move_is_promotion(move))


class PackedMovePicker:
    """Single-generation staged iterator over native cshogi move integers."""

    QUIET_BUCKETS = 8

    def __init__(
        self,
        legal_moves: list[int],
        *,
        ply: int,
        preferred: int | None,
        killers: tuple[int, ...],
        countermove: int | None,
        history: dict[tuple[int, int], int],
        tactical_only: bool = False,
        see_score: Callable[[int], int] | None = None,
        on_yield: Callable[[str], None] | None = None,
    ) -> None:
        self.generated = len(legal_moves)
        self.yielded = 0
        self.yielded_by_stage: dict[str, int] = {}
        self.stage_counts: dict[str, int] = {}
        self._stages: list[tuple[str, list[int]]] = []
        self._on_yield = on_yield
        del ply

        tt: list[int] = []
        good: list[int] = []
        promotions: list[int] = []
        killer_counter: list[int] = []
        quiet_buckets: list[list[int]] = [
            [] for _ in range(self.QUIET_BUCKETS)
        ]
        bad: list[int] = []
        for move in legal_moves:
            captured = int(cshogi.move_cap(move))
            promotion = bool(cshogi.move_is_promotion(move))
            if tactical_only and not captured and not promotion:
                continue
            if move == preferred:
                tt.append(move)
            elif captured:
                if see_score is None or see_score(move) >= 0:
                    good.append(move)
                else:
                    bad.append(move)
            elif promotion:
                promotions.append(move)
            elif move in killers or move == countermove:
                killer_counter.append(move)
            else:
                moving = move_moving_type(move)
                value = max(0, history.get((moving, int(cshogi.move_to(move))), 0))
                bucket = min(self.QUIET_BUCKETS - 1, value.bit_length() // 3)
                quiet_buckets[bucket].append(move)

        def tactical_key(move: int) -> int:
            captured = int(cshogi.move_cap(move))
            moving = move_moving_type(move)
            return PIECE_VALUES[captured] * 32 - PIECE_VALUES[moving]

        good.sort(key=tactical_key, reverse=True)
        promotions.sort(key=lambda move: PIECE_VALUES[move_moving_type(move)], reverse=True)
        bad.sort(key=tactical_key, reverse=True)
        quiet = [
            move
            for bucket in reversed(quiet_buckets)
            for move in bucket
        ]
        self._stages = [
            (MovePickerStage.TT, tt),
            (MovePickerStage.GOOD_TACTICAL, good + promotions),
            (MovePickerStage.KILLER_COUNTER, killer_counter),
            (MovePickerStage.QUIET, quiet),
            (MovePickerStage.BAD_CAPTURE, bad),
        ]
        self.stage_counts = {str(stage): len(moves) for stage, moves in self._stages}

    def __iter__(self):
        for stage, moves in self._stages:
            name = str(stage)
            for move in moves:
                self.yielded += 1
                self.yielded_by_stage[name] = self.yielded_by_stage.get(name, 0) + 1
                if self._on_yield is not None:
                    self._on_yield(name)
                yield move


@dataclass(frozen=True, slots=True)
class MoveFeatures:
    native: int
    usi: str
    captured_type: int
    moving_type: int
    promotion: bool
    gives_check: bool

    @property
    def tactical(self) -> bool:
        return bool(self.captured_type or self.promotion or self.gives_check)


@dataclass(slots=True)
class MoveOrdering:
    killers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    history: dict[tuple[int, int], int] = field(default_factory=dict)
    countermoves: dict[int, int] = field(default_factory=dict)
    last_classification_seconds: float = 0.0
    last_sorting_seconds: float = 0.0
    last_stage_counts: dict[str, int] = field(default_factory=dict)

    MAX_KILLER_PLIES = 256
    MAX_HISTORY_ENTRIES = 8_192
    MAX_COUNTERMOVES = 8_192

    def packed(
        self,
        legal_moves: list[int],
        *,
        ply: int,
        preferred: int | None = None,
        tactical_only: bool = False,
        countermove: int | None = None,
        see_score: Callable[[int], int] | None = None,
        on_yield: Callable[[str], None] | None = None,
    ) -> PackedMovePicker:
        return PackedMovePicker(
            legal_moves,
            ply=ply,
            preferred=preferred,
            killers=self.killers.get(ply, ()),
            countermove=countermove,
            history=self.history,
            tactical_only=tactical_only,
            see_score=see_score,
            on_yield=on_yield,
        )

    def ordered(
        self,
        board: cshogi.Board,
        *,
        ply: int,
        preferred: int | None = None,
        tactical_only: bool = False,
        include_checks: bool = True,
        advanced: bool = True,
        check_limits: Callable[[], None] | None = None,
        legal_moves: list[int] | None = None,
        include_usi: bool = True,
        countermove: int | None = None,
        see_score: Callable[[MoveFeatures], int] | None = None,
        staged: bool = True,
    ) -> list[MoveFeatures]:
        if check_limits is not None:
            check_limits()
        killers = self.killers.get(ply, ())
        classification_started = time.monotonic()
        features: list[MoveFeatures] = []
        # Checking every move used to push/pop the board here.  That made ordering
        # consume a substantial part of a one-second search and made it impossible
        # to interrupt inside cshogi.  Check status is now discovered naturally
        # after the search pushes a move; ordering uses only cheap move metadata.
        del include_checks
        source = (
            legal_moves
            if legal_moves is not None
            else [int(move) for move in board.legal_moves]
        )
        for raw_move in source:
            move = int(raw_move)
            captured = int(cshogi.move_cap(move))
            promotion = bool(cshogi.move_is_promotion(move))
            if tactical_only and not captured and not promotion:
                continue
            moving = int(cshogi.move_from_piece_type(move))
            if cshogi.move_is_drop(move):
                moving = _HAND_PIECE_TYPE[int(cshogi.move_drop_hand_piece(move))]
            item = MoveFeatures(
                native=move,
                usi=cshogi.move_to_usi(move) if include_usi else "",
                captured_type=captured,
                moving_type=moving,
                promotion=promotion,
                gives_check=False,
            )
            features.append(item)

        self.last_classification_seconds = time.monotonic() - classification_started

        def score(item: MoveFeatures) -> int:
            if item.native == preferred:
                return 100_000_000
            if item.captured_type:
                victim = PIECE_VALUES[item.captured_type]
                if not advanced:
                    return victim * 10
                attacker = PIECE_VALUES[item.moving_type]
                return 70_000_000 + victim * 32 - attacker
            if item.promotion:
                if not advanced:
                    return 500
                return 60_000_000 + PIECE_VALUES[item.moving_type]
            if advanced and item.native in killers:
                return 40_000_000 - killers.index(item.native)
            if advanced:
                return self.history.get((item.moving_type, cshogi.move_to(item.native)), 0)
            return 0

        sorting_started = time.monotonic()
        if advanced and staged:
            preferred_moves: list[MoveFeatures] = []
            good_captures: list[MoveFeatures] = []
            promotions: list[MoveFeatures] = []
            killer_moves: list[MoveFeatures] = []
            quiet: list[MoveFeatures] = []
            bad_captures: list[MoveFeatures] = []
            for item in features:
                if item.native == preferred:
                    preferred_moves.append(item)
                elif item.captured_type:
                    if see_score is None or see_score(item) >= 0:
                        good_captures.append(item)
                    else:
                        bad_captures.append(item)
                elif item.promotion:
                    promotions.append(item)
                elif item.native in killers or item.native == countermove:
                    killer_moves.append(item)
                else:
                    quiet.append(item)
            good_captures.sort(key=score, reverse=True)
            promotions.sort(key=score, reverse=True)
            killer_moves.sort(key=score, reverse=True)
            quiet.sort(key=score, reverse=True)
            bad_captures.sort(key=score, reverse=True)
            features = (
                preferred_moves
                + good_captures
                + promotions
                + killer_moves
                + quiet
                + bad_captures
            )
            self.last_stage_counts = {
                "tt": len(preferred_moves),
                "good_tactical": len(good_captures) + len(promotions),
                "killer_counter": len(killer_moves),
                "quiet": len(quiet),
                "bad_capture": len(bad_captures),
            }
        else:
            features.sort(key=score, reverse=True)
            self.last_stage_counts = {"legacy": len(features)}
        self.last_sorting_seconds = time.monotonic() - sorting_started
        if check_limits is not None:
            check_limits()
        return features

    def record_cutoff(
        self,
        move: MoveFeatures,
        *,
        ply: int,
        depth: int,
        previous_move: int | None = None,
    ) -> None:
        if move.tactical:
            return
        if ply >= self.MAX_KILLER_PLIES:
            return
        old = self.killers.get(ply, ())
        remaining = tuple(value for value in old if value != move.native)[:1]
        self.killers[ply] = (move.native,) + remaining
        key = (move.moving_type, cshogi.move_to(move.native))
        if key in self.history or len(self.history) < self.MAX_HISTORY_ENTRIES:
            self.history[key] = min(1_000_000, self.history.get(key, 0) + depth * depth)
        if previous_move is not None and (
            previous_move in self.countermoves
            or len(self.countermoves) < self.MAX_COUNTERMOVES
        ):
            self.countermoves[previous_move] = move.native

    def record_cutoff_native(
        self,
        move: int,
        *,
        ply: int,
        depth: int,
        previous_move: int | None = None,
    ) -> None:
        if move_is_tactical(move) or ply >= self.MAX_KILLER_PLIES:
            return
        old = self.killers.get(ply, ())
        remaining = tuple(value for value in old if value != move)[:1]
        self.killers[ply] = (move,) + remaining
        key = (move_moving_type(move), int(cshogi.move_to(move)))
        if key in self.history or len(self.history) < self.MAX_HISTORY_ENTRIES:
            self.history[key] = min(
                1_000_000, self.history.get(key, 0) + depth * depth
            )
        if previous_move is not None and (
            previous_move in self.countermoves
            or len(self.countermoves) < self.MAX_COUNTERMOVES
        ):
            self.countermoves[previous_move] = move


__all__ = [
    "MoveFeatures",
    "MoveOrdering",
    "MovePickerStage",
    "PackedMovePicker",
    "move_is_tactical",
    "move_moving_type",
]
