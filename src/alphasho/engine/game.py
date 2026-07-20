"""A safe, UI-independent adapter around :mod:`cshogi`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import cshogi

from alphasho.domain import (
    Color,
    GameOutcome,
    GameSnapshot,
    MoveOption,
    OutcomeReason,
    Piece,
    PieceType,
)


class IllegalMoveError(ValueError):
    """Raised before an illegal move can reach cshogi's native ``push``."""

    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = params
        super().__init__(code)


class InvalidPositionError(ValueError):
    """Raised when an SFEN position cannot be loaded safely."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ShogiGame:
    """Owns one cshogi board and exposes immutable application-level state."""

    def __init__(self, sfen: str | None = None) -> None:
        self._board = cshogi.Board()
        self._initial_sfen = self._board.sfen()
        self._history: list[MoveOption] = []
        self._position_keys: list[str] = []
        self._position_counts: Counter[str] = Counter()
        self._outcome = GameOutcome()
        self.reset(sfen)

    def reset(self, sfen: str | None = None) -> GameSnapshot:
        try:
            board = cshogi.Board(sfen) if sfen else cshogi.Board()
        except (RuntimeError, ValueError, TypeError) as exc:
            raise InvalidPositionError("invalid_sfen") from exc
        if not board.is_ok():
            raise InvalidPositionError("invalid_position")
        self._board = board
        self._initial_sfen = board.sfen()
        self._history.clear()
        initial_key = self._position_key()
        self._position_keys = [initial_key]
        self._position_counts = Counter({initial_key: 1})
        self._outcome = GameOutcome()
        self._update_automatic_outcome()
        return self.snapshot()

    def snapshot(self) -> GameSnapshot:
        pieces = tuple(self._piece_from_code(code) for code in self._board.pieces)
        hands = tuple(tuple(int(value) for value in hand) for hand in self._board.pieces_in_hand)
        return GameSnapshot(
            board=pieces,
            hands=hands,  # type: ignore[arg-type]
            turn=Color(self._board.turn),
            move_number=int(self._board.move_number),
            is_check=bool(self._board.is_check()),
            last_move=self._history[-1] if self._history else None,
            move_history=tuple(move.usi for move in self._history),
            sfen=self._board.sfen(),
            outcome=self._outcome,
            initial_sfen=self._initial_sfen,
        )

    def legal_moves(
        self,
        *,
        source_square: int | None = None,
        drop_piece: PieceType | None = None,
    ) -> tuple[MoveOption, ...]:
        if source_square is not None and drop_piece is not None:
            raise ValueError("Specify either source_square or drop_piece, not both")
        if self._outcome.is_over:
            return ()

        moves = tuple(self._move_from_native(move) for move in self._board.legal_moves)
        if source_square is not None:
            return tuple(move for move in moves if move.from_square == source_square)
        if drop_piece is not None:
            return tuple(move for move in moves if move.drop_piece is drop_piece)
        return moves

    def play(self, move: MoveOption | str) -> GameSnapshot:
        requested_usi = move.usi if isinstance(move, MoveOption) else str(move)
        native_by_usi = {
            cshogi.move_to_usi(native): native for native in self._board.legal_moves
        }
        native_move = native_by_usi.get(requested_usi)
        if self._outcome.is_over or native_move is None:
            raise IllegalMoveError("illegal_move", usi=requested_usi)

        safe_move = self._move_from_native(native_move)
        self._board.push(native_move)
        self._history.append(safe_move)
        position_key = self._position_key()
        self._position_keys.append(position_key)
        self._position_counts[position_key] += 1
        self._update_automatic_outcome()
        return self.snapshot()

    def undo(self) -> GameSnapshot:
        if not self._history:
            self._outcome = GameOutcome()
            self._update_automatic_outcome()
            return self.snapshot()
        removed_key = self._position_keys.pop()
        self._position_counts[removed_key] -= 1
        if self._position_counts[removed_key] <= 0:
            del self._position_counts[removed_key]
        self._board.pop()
        self._history.pop()
        self._outcome = GameOutcome()
        self._update_automatic_outcome()
        return self.snapshot()

    def resign(self) -> GameSnapshot:
        if not self._outcome.is_over:
            self._outcome = GameOutcome(
                reason=OutcomeReason.RESIGNATION,
                winner=Color(self._board.turn).opponent,
            )
        return self.snapshot()

    def declare_nyugyoku(self) -> GameSnapshot:
        if self._outcome.is_over:
            return self.snapshot()
        if not self._board.is_nyugyoku():
            raise IllegalMoveError("nyugyoku_unavailable")
        self._outcome = GameOutcome(
            reason=OutcomeReason.NYUGYOKU,
            winner=Color(self._board.turn),
        )
        return self.snapshot()

    def finish(self, outcome: GameOutcome) -> GameSnapshot:
        """Apply a validated application-level terminal result."""
        if outcome.reason is OutcomeReason.ONGOING:
            raise ValueError("A terminal outcome is required")
        if not self._outcome.is_over:
            self._outcome = outcome
        return self.snapshot()

    def _update_automatic_outcome(self) -> None:
        if self._outcome.is_over:
            return

        current_key = self._position_key()
        repetition = (
            self._board.is_draw()
            if self._position_counts.get(current_key, 0) >= 4
            else cshogi.NOT_REPETITION
        )
        turn = Color(self._board.turn)
        if repetition == cshogi.REPETITION_DRAW:
            self._outcome = GameOutcome(OutcomeReason.REPETITION)
        elif repetition == cshogi.REPETITION_WIN:
            self._outcome = GameOutcome(OutcomeReason.PERPETUAL_CHECK, turn)
        elif repetition == cshogi.REPETITION_LOSE:
            self._outcome = GameOutcome(OutcomeReason.PERPETUAL_CHECK, turn.opponent)
        elif repetition == cshogi.REPETITION_SUPERIOR:
            self._outcome = GameOutcome(OutcomeReason.SUPERIOR_REPETITION, turn)
        elif repetition == cshogi.REPETITION_INFERIOR:
            self._outcome = GameOutcome(OutcomeReason.SUPERIOR_REPETITION, turn.opponent)
        elif self._board.is_game_over():
            self._outcome = GameOutcome(OutcomeReason.CHECKMATE, turn.opponent)

    def _position_key(self) -> str:
        """SFEN board, turn, and hands; the move counter is not part of repetition."""
        return " ".join(self._board.sfen().split()[:3])

    @staticmethod
    def _piece_from_code(code: int) -> Piece | None:
        if code == cshogi.NONE:
            return None
        color = Color.WHITE if code >= 16 else Color.BLACK
        piece_type = PieceType(code - 16 if color is Color.WHITE else code)
        return Piece(color=color, piece_type=piece_type)

    @staticmethod
    def _move_from_native(move: int) -> MoveOption:
        usi = cshogi.move_to_usi(move)
        to_square = int(cshogi.move_to(move))
        if cshogi.move_is_drop(move):
            letter_to_piece = {
                "P": PieceType.PAWN,
                "L": PieceType.LANCE,
                "N": PieceType.KNIGHT,
                "S": PieceType.SILVER,
                "G": PieceType.GOLD,
                "B": PieceType.BISHOP,
                "R": PieceType.ROOK,
            }
            return MoveOption(
                usi=usi,
                to_square=to_square,
                drop_piece=letter_to_piece[usi[0]],
            )
        return MoveOption(
            usi=usi,
            from_square=int(cshogi.move_from(move)),
            to_square=to_square,
            promotion=bool(cshogi.move_is_promotion(move)),
        )

    @staticmethod
    def usi_sequence(moves: Iterable[MoveOption]) -> tuple[str, ...]:
        return tuple(move.usi for move in moves)
