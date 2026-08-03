"""Incremental evaluator boundary for Python and future native/NNUE backends."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import cshogi

from .evaluation import (
    CHECK_PENALTY,
    HAND_VALUES,
    MAX_POSITIONAL,
    PIECE_VALUES,
    TEMPO_BONUS,
    EvaluationBreakdown,
    EvaluationMode,
    EvaluationScratch,
    evaluate_breakdown_from_absolute,
    evaluate_fast_absolute,
    piece_square_bonus,
)

_PROMOTED_TYPE = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14}
_HAND_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 5: 5, 6: 6}
_DEMOTED_TYPE = {9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
_HAND_PIECE_TYPE = (1, 2, 3, 4, 7, 5, 6)


@dataclass(slots=True)
class EvaluatorState:
    material: int
    positional_raw: int
    occupied_mask: int
    stack: list[tuple[int, int, int]] = field(default_factory=list)


class IncrementalEvaluator(Protocol):
    version: str

    def initialize(self, board: cshogi.Board) -> EvaluatorState: ...

    def push(self, board: cshogi.Board, move: int, state: EvaluatorState) -> None: ...

    def pop(self, state: EvaluatorState) -> None: ...

    def fast(
        self, board: cshogi.Board, state: EvaluatorState, mode: EvaluationMode
    ) -> int: ...

    def exact(
        self,
        board: cshogi.Board,
        state: EvaluatorState,
        mode: EvaluationMode,
        scratch: EvaluationScratch,
    ) -> int: ...

    @property
    def fingerprint(self) -> str: ...


class HandcraftedEvaluator:
    version = "handcrafted-v3-occupied-mask"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.version.encode("ascii")).hexdigest()

    def initialize(self, board: cshogi.Board) -> EvaluatorState:
        material, positional = evaluate_fast_absolute(board)
        occupied_mask = 0
        for square, piece in enumerate(board.pieces):
            if int(piece):
                occupied_mask |= 1 << square
        return EvaluatorState(material, positional, occupied_mask)

    def push(self, board: cshogi.Board, move: int, state: EvaluatorState) -> None:
        state.stack.append((state.material, state.positional_raw, state.occupied_mask))
        mover = int(board.turn)
        sign = 1 if mover == cshogi.BLACK else -1
        to_square = int(cshogi.move_to(move))
        moving_type = int(cshogi.move_from_piece_type(move))
        if cshogi.move_is_drop(move):
            hand_index = int(cshogi.move_drop_hand_piece(move))
            moving_type = _HAND_PIECE_TYPE[hand_index]
            state.material += sign * (
                PIECE_VALUES[moving_type] - HAND_VALUES[hand_index]
            )
            state.positional_raw += sign * piece_square_bonus(
                moving_type, mover, to_square
            )
            state.occupied_mask |= 1 << to_square
            return
        from_square = int(cshogi.move_from(move))
        state.occupied_mask &= ~(1 << from_square)
        state.occupied_mask |= 1 << to_square
        state.positional_raw -= sign * piece_square_bonus(
            moving_type, mover, from_square
        )
        captured_type = int(cshogi.move_cap(move))
        if captured_type:
            hand_type = _DEMOTED_TYPE.get(captured_type, captured_type)
            state.material += sign * (
                PIECE_VALUES[captured_type] + HAND_VALUES[_HAND_INDEX[hand_type]]
            )
            enemy = cshogi.WHITE if mover == cshogi.BLACK else cshogi.BLACK
            state.positional_raw += sign * piece_square_bonus(
                captured_type, enemy, to_square
            )
        placed_type = (
            _PROMOTED_TYPE[moving_type]
            if cshogi.move_is_promotion(move)
            else moving_type
        )
        state.material += sign * (
            PIECE_VALUES[placed_type] - PIECE_VALUES[moving_type]
        )
        state.positional_raw += sign * piece_square_bonus(
            placed_type, mover, to_square
        )

    def pop(self, state: EvaluatorState) -> None:
        state.material, state.positional_raw, state.occupied_mask = state.stack.pop()

    def fast(
        self, board: cshogi.Board, state: EvaluatorState, mode: EvaluationMode
    ) -> int:
        positional = 0
        check_penalty = 35
        tempo = 0
        if mode is not EvaluationMode.MATERIAL:
            positional = max(-MAX_POSITIONAL, min(MAX_POSITIONAL, state.positional_raw))
            check_penalty = CHECK_PENALTY
            tempo = TEMPO_BONUS
        absolute = state.material + positional
        score = absolute if board.turn == cshogi.BLACK else -absolute
        if board.is_check():
            score -= check_penalty
        return score + tempo

    def exact(
        self,
        board: cshogi.Board,
        state: EvaluatorState,
        mode: EvaluationMode,
        scratch: EvaluationScratch,
    ) -> int:
        return self.breakdown(board, state, mode, scratch).total

    def breakdown(
        self,
        board: cshogi.Board,
        state: EvaluatorState,
        mode: EvaluationMode,
        scratch: EvaluationScratch,
    ) -> EvaluationBreakdown:
        return evaluate_breakdown_from_absolute(
            board,
            state.material,
            state.positional_raw,
            mode=mode,
            scratch=scratch,
            occupied_mask=state.occupied_mask,
        )


__all__ = [
    "EvaluatorState",
    "HandcraftedEvaluator",
    "IncrementalEvaluator",
]
