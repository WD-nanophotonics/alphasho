"""Immutable types shared by the engine, players, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class Color(IntEnum):
    BLACK = 0
    WHITE = 1

    @property
    def opponent(self) -> Color:
        return Color.WHITE if self is Color.BLACK else Color.BLACK

class PieceType(IntEnum):
    PAWN = 1
    LANCE = 2
    KNIGHT = 3
    SILVER = 4
    BISHOP = 5
    ROOK = 6
    GOLD = 7
    KING = 8
    PROMOTED_PAWN = 9
    PROMOTED_LANCE = 10
    PROMOTED_KNIGHT = 11
    PROMOTED_SILVER = 12
    HORSE = 13
    DRAGON = 14

HAND_PIECE_TYPES: tuple[PieceType, ...] = (
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
)


class OutcomeReason(StrEnum):
    ONGOING = "ongoing"
    CHECKMATE = "checkmate"
    REPETITION = "repetition"
    PERPETUAL_CHECK = "perpetual_check"
    SUPERIOR_REPETITION = "superior_repetition"
    RESIGNATION = "resignation"
    NYUGYOKU = "nyugyoku"
    TIMEOUT = "timeout"
    MOVE_LIMIT = "move_limit"


class GameMode(StrEnum):
    """Legacy presets kept for compatibility with older integrations."""

    LOCAL = "local"
    RANDOM_AI = "random_ai"


class ParticipantKind(StrEnum):
    HUMAN = "human"
    RANDOM_AI = "random_ai"
    ALPHAZERO = "alphazero"
    USI_ENGINE = "usi_engine"


class TimeControlMode(StrEnum):
    NONE = "none"
    BYOYOMI = "byoyomi"
    FISCHER = "fischer"


class StartPositionKind(StrEnum):
    INITIAL = "initial"
    CURRENT = "current"
    FILE = "file"


@dataclass(frozen=True, slots=True)
class ParticipantConfig:
    kind: ParticipantKind = ParticipantKind.HUMAN
    name: str = ""
    engine_id: str | None = None
    model_path: str | None = None


@dataclass(frozen=True, slots=True)
class SideTimeConfig:
    main_seconds: int = 0
    overtime_seconds: int = 0


@dataclass(frozen=True, slots=True)
class TimeControl:
    mode: TimeControlMode = TimeControlMode.NONE
    black: SideTimeConfig = SideTimeConfig()
    white: SideTimeConfig = SideTimeConfig()
    time_forfeit: bool = True

    def for_color(self, color: Color) -> SideTimeConfig:
        return self.black if color is Color.BLACK else self.white


@dataclass(frozen=True, slots=True)
class MatchConfig:
    black: ParticipantConfig = ParticipantConfig()
    white: ParticipantConfig = ParticipantConfig()
    time_control: TimeControl = TimeControl()
    start_position: StartPositionKind = StartPositionKind.INITIAL
    start_sfen: str | None = None
    move_limit: int = 0

    def participant(self, color: Color) -> ParticipantConfig:
        return self.black if color is Color.BLACK else self.white


@dataclass(frozen=True, slots=True)
class UiMessage:
    """Language-neutral message passed from core/controller code to the UI."""

    key: str
    params: tuple[tuple[str, object], ...] = ()

    @classmethod
    def create(cls, key: str, **params: object) -> UiMessage:
        return cls(key=key, params=tuple(params.items()))

    def format_params(self) -> dict[str, object]:
        return dict(self.params)


@dataclass(frozen=True, slots=True)
class Piece:
    color: Color
    piece_type: PieceType


@dataclass(frozen=True, slots=True)
class MoveOption:
    """A legal move represented in a stable, module-safe form."""

    usi: str
    to_square: int
    from_square: int | None = None
    drop_piece: PieceType | None = None
    promotion: bool = False

    @property
    def is_drop(self) -> bool:
        return self.drop_piece is not None


@dataclass(frozen=True, slots=True)
class GameOutcome:
    reason: OutcomeReason = OutcomeReason.ONGOING
    winner: Color | None = None

    @property
    def is_over(self) -> bool:
        return self.reason is not OutcomeReason.ONGOING


@dataclass(frozen=True, slots=True)
class ClockState:
    remaining_ms: tuple[int, int]
    overtime_remaining_ms: tuple[int, int]
    active_color: Color
    running: bool
    expired_color: Color | None = None

    def remaining_for(self, color: Color) -> int:
        return self.remaining_ms[int(color)]

    def overtime_for(self, color: Color) -> int:
        return self.overtime_remaining_ms[int(color)]


@dataclass(slots=True)
class GameRecord:
    initial_sfen: str
    black_name: str = ""
    white_name: str = ""
    moves: list[str] = field(default_factory=list)
    move_times: list[int] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    outcome: GameOutcome = GameOutcome()
    file_path: str | None = None
    dirty: bool = False

    def name_for(self, color: Color) -> str:
        return self.black_name if color is Color.BLACK else self.white_name

@dataclass(frozen=True, slots=True)
class GameSnapshot:
    board: tuple[Piece | None, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    turn: Color
    move_number: int
    is_check: bool
    last_move: MoveOption | None
    move_history: tuple[str, ...]
    sfen: str
    outcome: GameOutcome
    initial_sfen: str = ""

    def piece_at(self, square: int) -> Piece | None:
        if not 0 <= square < 81:
            raise ValueError(f"Square must be between 0 and 80, got {square}")
        return self.board[square]

    def hand_count(self, color: Color, piece_type: PieceType) -> int:
        try:
            index = HAND_PIECE_TYPES.index(piece_type)
        except ValueError as exc:
            raise ValueError(f"{piece_type.name} cannot be held in hand") from exc
        return self.hands[int(color)][index]
