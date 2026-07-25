"""Mutable, isolated cshogi position used by search and training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import cshogi
import numpy as np
from cshogi import dlshogi

from alphasho.domain import HAND_PIECE_TYPES, Color, GameSnapshot, PieceType
from alphasho.engine.game import IllegalMoveError, InvalidPositionError

if TYPE_CHECKING:
    from numpy.typing import NDArray


POLICY_PLANES = 27
POLICY_SIZE = POLICY_PLANES * 81


@dataclass(frozen=True, slots=True)
class SearchAction:
    """A legal action at one search node.

    ``native`` never comes from serialized or UI input. It is created only
    while enumerating the current cshogi board's legal moves.
    """

    usi: str
    label: int
    native: int


class SearchablePosition(Protocol):
    @property
    def sfen(self) -> str: ...

    @property
    def history(self) -> tuple[str, ...]: ...

    @property
    def turn(self) -> Color: ...

    @property
    def state_key(self) -> tuple[object, ...]: ...

    def clone(self) -> SearchablePosition: ...

    def legal_actions(self) -> tuple[SearchAction, ...]: ...

    def push(self, action: SearchAction) -> None: ...

    def terminal_value(self) -> float | None: ...

    def fill_features(
        self,
        features1: NDArray[np.float32],
        features2: NDArray[np.float32],
    ) -> None: ...


class SearchPosition:
    """An independent replayable position for MCTS.

    The live UI board is never shared with this object. Constructing from a
    snapshot replays the complete USI history from its initial SFEN so cshogi
    retains the information needed for repetition adjudication.
    """

    def __init__(
        self,
        initial_sfen: str | None = None,
        moves: tuple[str, ...] | list[str] = (),
    ) -> None:
        try:
            self._board = cshogi.Board(initial_sfen) if initial_sfen else cshogi.Board()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise InvalidPositionError("invalid_sfen") from exc
        if not self._board.is_ok():
            raise InvalidPositionError("invalid_position")
        self._initial_sfen = self._board.sfen()
        self._history: list[str] = []
        initial_key = self._position_key()
        self._position_keys = [initial_key]
        self._position_counts = Counter({initial_key: 1})
        for usi in moves:
            self.push_usi(usi)

    @classmethod
    def from_snapshot(cls, snapshot: GameSnapshot) -> SearchPosition:
        initial = snapshot.initial_sfen or snapshot.sfen
        moves = snapshot.move_history if snapshot.initial_sfen else ()
        position = cls(initial, moves)
        if position.sfen != snapshot.sfen:
            raise InvalidPositionError("snapshot_replay_mismatch")
        return position

    def clone(self) -> SearchPosition:
        clone = object.__new__(SearchPosition)
        clone._board = self._board.copy()
        clone._initial_sfen = self._initial_sfen
        clone._history = self._history.copy()
        clone._position_keys = self._position_keys.copy()
        clone._position_counts = self._position_counts.copy()
        return clone

    @property
    def sfen(self) -> str:
        return self._board.sfen()

    @property
    def initial_sfen(self) -> str:
        return self._initial_sfen

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    @property
    def state_key(self) -> tuple[object, ...]:
        return (self.sfen, self.history)

    @property
    def turn(self) -> Color:
        return Color(self._board.turn)

    @property
    def move_number(self) -> int:
        return int(self._board.move_number)

    @property
    def repetition_count(self) -> int:
        return self._position_counts[self._position_keys[-1]]

    def legal_actions(self) -> tuple[SearchAction, ...]:
        actions = tuple(
            SearchAction(
                usi=cshogi.move_to_usi(native),
                label=int(dlshogi.make_move_label(native, self._board.turn)),
                native=int(native),
            )
            for native in self._board.legal_moves
        )
        labels = {action.label for action in actions}
        if len(labels) != len(actions):
            raise RuntimeError("cshogi returned colliding legal move labels")
        if any(not 0 <= action.label < POLICY_SIZE for action in actions):
            raise RuntimeError("cshogi returned a move label outside the policy space")
        return actions

    def push(self, action: SearchAction) -> None:
        self.push_usi(action.usi)

    def captured_piece(self, action: SearchAction) -> PieceType | None:
        code = int(self._board.piece(cshogi.move_to(action.native)))
        if code == 0:
            return None
        return PieceType(code - 16 if code >= 16 else code)

    def piece_inventory(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        counts = [[0] * 15, [0] * 15]
        for raw_code in self._board.pieces:
            code = int(raw_code)
            if code == 0:
                continue
            color = Color.WHITE if code >= 16 else Color.BLACK
            piece_type = PieceType(code - 16 if color is Color.WHITE else code)
            counts[int(color)][int(piece_type)] += 1
        for color, hand in enumerate(self._board.pieces_in_hand):
            for piece_type, count in zip(HAND_PIECE_TYPES, hand, strict=True):
                counts[color][int(piece_type)] += int(count)
        return tuple(tuple(row) for row in counts)  # type: ignore[return-value]

    def push_usi(self, usi: str) -> None:
        legal = {action.usi: action for action in self.legal_actions()}
        action = legal.get(str(usi))
        if action is None:
            raise IllegalMoveError("illegal_move", usi=usi)
        self._board.push(action.native)
        self._history.append(action.usi)
        position_key = self._position_key()
        self._position_keys.append(position_key)
        self._position_counts[position_key] += 1

    def pop(self) -> str:
        if not self._history:
            raise IndexError("Cannot pop the initial search position")
        removed_key = self._position_keys.pop()
        self._position_counts[removed_key] -= 1
        if self._position_counts[removed_key] <= 0:
            del self._position_counts[removed_key]
        usi = self._history.pop()
        self._board.pop()
        return usi

    def fill_features(
        self,
        features1: NDArray[np.float32],
        features2: NDArray[np.float32],
    ) -> None:
        expected1 = (dlshogi.FEATURES1_NUM, 9, 9)
        expected2 = (dlshogi.FEATURES2_NUM, 9, 9)
        if features1.shape != expected1 or features2.shape != expected2:
            raise ValueError(
                f"Feature buffers must have shapes {expected1} and {expected2}"
            )
        dlshogi.make_input_features(self._board, features1, features2)

    def terminal_value(self) -> float | None:
        """Return the result from the current side-to-move's perspective."""

        repetition = (
            self._board.is_draw()
            if self.repetition_count >= 4
            else cshogi.NOT_REPETITION
        )
        if repetition == cshogi.REPETITION_DRAW:
            return 0.0
        if repetition in {cshogi.REPETITION_WIN, cshogi.REPETITION_SUPERIOR}:
            return 1.0
        if repetition in {cshogi.REPETITION_LOSE, cshogi.REPETITION_INFERIOR}:
            return -1.0
        if self._board.is_nyugyoku():
            return 1.0
        if self._board.is_game_over():
            return -1.0
        return None

    def _position_key(self) -> str:
        """SFEN board, turn, and hands; the move counter is not repetition state."""

        return " ".join(self._board.sfen().split()[:3])
