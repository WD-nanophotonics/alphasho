"""Experimental short-game adjudication used only by training positions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from alphasho.domain import Color, PieceType
from alphasho.engine import SearchAction, SearchPosition


class AdjudicationMode(StrEnum):
    REAL_RULES = "real_rules"
    CAPTURE_MATERIAL = "capture_material"


class AdjudicationReason(StrEnum):
    REAL_RULE = "real_rule"
    CAPTURE_THRESHOLD = "capture_threshold"
    MATERIAL_CUTOFF = "material_cutoff"
    MOVE_LIMIT_DRAW = "move_limit_draw"


MATERIAL_WEIGHTS: dict[PieceType, int] = {
    PieceType.PAWN: 1,
    PieceType.LANCE: 3,
    PieceType.KNIGHT: 3,
    PieceType.SILVER: 5,
    PieceType.GOLD: 6,
    PieceType.BISHOP: 8,
    PieceType.ROOK: 10,
    PieceType.KING: 0,
    PieceType.PROMOTED_PAWN: 6,
    PieceType.PROMOTED_LANCE: 6,
    PieceType.PROMOTED_KNIGHT: 6,
    PieceType.PROMOTED_SILVER: 6,
    PieceType.HORSE: 10,
    PieceType.DRAGON: 12,
}


@dataclass(frozen=True, slots=True)
class AdjudicationRules:
    mode: AdjudicationMode = AdjudicationMode.REAL_RULES
    max_moves: int = 512
    capture_threshold: int = 0
    material_scale: float = 12.0

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                **asdict(self),
                "mode": self.mode.value,
                "material_weights": {
                    piece.name: value
                    for piece, value in MATERIAL_WEIGHTS.items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AdjudicationResult:
    reason: AdjudicationReason
    winner: Color | None
    black_value: float
    capture_counts: tuple[int, int]
    material_diff: int

    def value_for(self, color: Color) -> float:
        return self.black_value if color is Color.BLACK else -self.black_value


class TrainingPosition:
    """Search-compatible position with reversible experimental counters."""

    def __init__(
        self,
        rules: AdjudicationRules,
        initial_sfen: str | None = None,
    ) -> None:
        self.rules = rules
        self._position = SearchPosition(initial_sfen)
        self._captures = [0, 0]
        self._capture_stack: list[Color | None] = []

    def clone(self) -> TrainingPosition:
        clone = object.__new__(TrainingPosition)
        clone.rules = self.rules
        clone._position = self._position.clone()
        clone._captures = self._captures.copy()
        clone._capture_stack = self._capture_stack.copy()
        return clone

    @property
    def sfen(self) -> str:
        return self._position.sfen

    @property
    def initial_sfen(self) -> str:
        return self._position.initial_sfen

    @property
    def history(self) -> tuple[str, ...]:
        return self._position.history

    @property
    def turn(self) -> Color:
        return self._position.turn

    @property
    def capture_counts(self) -> tuple[int, int]:
        return (self._captures[0], self._captures[1])

    @property
    def state_key(self) -> tuple[object, ...]:
        return (
            *self._position.state_key,
            self.rules.fingerprint,
            self.capture_counts,
        )

    def legal_actions(self) -> tuple[SearchAction, ...]:
        return self._position.legal_actions()

    def push(self, action: SearchAction) -> None:
        mover = self.turn
        captured = self._position.captured_piece(action)
        self._position.push(action)
        capture_color = mover if captured is not None else None
        self._capture_stack.append(capture_color)
        if capture_color is not None:
            self._captures[int(capture_color)] += 1

    def push_usi(self, usi: str) -> None:
        action = next(
            (candidate for candidate in self.legal_actions() if candidate.usi == usi),
            None,
        )
        if action is None:
            self._position.push_usi(usi)
            return
        self.push(action)

    def pop(self) -> str:
        if not self._capture_stack:
            raise IndexError("Cannot pop the initial training position")
        capture_color = self._capture_stack.pop()
        if capture_color is not None:
            self._captures[int(capture_color)] -= 1
        return self._position.pop()

    def fill_features(self, features1, features2) -> None:  # noqa: ANN001
        self._position.fill_features(features1, features2)

    def material_diff(self) -> int:
        inventory = self._position.piece_inventory()
        totals = [0, 0]
        for color in Color:
            totals[int(color)] = sum(
                inventory[int(color)][int(piece_type)] * weight
                for piece_type, weight in MATERIAL_WEIGHTS.items()
            )
        return totals[int(Color.BLACK)] - totals[int(Color.WHITE)]

    def result(self) -> AdjudicationResult | None:
        real_value = self._position.terminal_value()
        if real_value is not None:
            black_value = (
                real_value if self.turn is Color.BLACK else -real_value
            )
            return self._result(
                AdjudicationReason.REAL_RULE,
                black_value,
            )
        if self.rules.mode is AdjudicationMode.CAPTURE_MATERIAL:
            for color in Color:
                if self._captures[int(color)] >= self.rules.capture_threshold:
                    return self._result(
                        AdjudicationReason.CAPTURE_THRESHOLD,
                        1.0 if color is Color.BLACK else -1.0,
                    )
        if len(self.history) < self.rules.max_moves:
            return None
        if self.rules.mode is AdjudicationMode.REAL_RULES:
            return self._result(AdjudicationReason.MOVE_LIMIT_DRAW, 0.0)
        difference = self.material_diff()
        return self._result(
            AdjudicationReason.MATERIAL_CUTOFF,
            math.tanh(difference / self.rules.material_scale),
            material_diff=difference,
        )

    def terminal_value(self) -> float | None:
        result = self.result()
        return None if result is None else result.value_for(self.turn)

    def _result(
        self,
        reason: AdjudicationReason,
        black_value: float,
        *,
        material_diff: int | None = None,
    ) -> AdjudicationResult:
        winner = None
        if black_value > 0:
            winner = Color.BLACK
        elif black_value < 0:
            winner = Color.WHITE
        return AdjudicationResult(
            reason=reason,
            winner=winner,
            black_value=float(black_value),
            capture_counts=self.capture_counts,
            material_diff=(
                self.material_diff()
                if material_diff is None
                else material_diff
            ),
        )
