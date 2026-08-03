"""Hand-crafted, CPU-cheap shogi position evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cshogi

PIECE_VALUES = (0, 100, 300, 320, 450, 800, 1_000, 520, 0, 520, 520, 520, 520, 950, 1_150)
HAND_VALUES = (100, 300, 320, 450, 520, 800, 1_000)

CHECK_PENALTY = 45
TEMPO_BONUS = 8
MAX_POSITIONAL = 350
MAX_MOBILITY = 100
MAX_KING_SAFETY = 260
MAX_FULL_RESIDUAL = MAX_MOBILITY + MAX_KING_SAFETY

_GOLD_LIKE = {7, 9, 10, 11, 12}
_SLIDERS = {2, 5, 6, 13, 14}


def _piece_square_formula(piece_type: int, color: int, square: int) -> int:
    file_index, rank = divmod(square, 9)
    center = max(0, 8 - abs(file_index - 4) - abs(rank - 4))
    progress = (6 - rank) if color == cshogi.BLACK else (rank - 2)
    progress = max(-2, min(6, progress))
    if piece_type == 1:
        return progress * 4 + center
    if piece_type in {2, 3, 4}:
        return progress * 2 + center * 2
    if piece_type in _GOLD_LIKE:
        return center * 2 + max(0, progress)
    if piece_type in {5, 6, 13, 14}:
        return center * 3
    if piece_type == 8:
        home_distance = 8 - rank if color == cshogi.BLACK else rank
        return -home_distance * 3 - abs(file_index - 4)
    return 0


_PIECE_SQUARE = tuple(
    tuple(
        tuple(_piece_square_formula(piece, color, square) for square in range(81))
        for piece in range(15)
    )
    for color in (cshogi.BLACK, cshogi.WHITE)
)
_KING_ZONE = tuple(
    tuple(
        target
        for df in (-1, 0, 1)
        for dr in (-1, 0, 1)
        if df or dr
        if 0 <= (square // 9) + df < 9 and 0 <= (square % 9) + dr < 9
        for target in (((square // 9) + df) * 9 + (square % 9) + dr,)
    )
    for square in range(81)
)
_SHIELD_TARGETS = tuple(
    tuple(
        tuple(
            target
            for df in (-1, 0, 1)
            if 0 <= (square // 9) + df < 9
            if 0 <= (square % 9) + (-1 if color == cshogi.BLACK else 1) < 9
            for target in (
                ((square // 9) + df) * 9
                + (square % 9)
                + (-1 if color == cshogi.BLACK else 1),
            )
        )
        for square in range(81)
    )
    for color in (cshogi.BLACK, cshogi.WHITE)
)


def _movement_vectors(
    piece_type: int, color: int
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    forward = -1 if color == cshogi.BLACK else 1
    if piece_type == 1:
        return ((0, forward),), ()
    if piece_type == 2:
        return (), ((0, forward),)
    if piece_type == 3:
        return ((-1, 2 * forward), (1, 2 * forward)), ()
    if piece_type == 4:
        return (
            (0, forward),
            (-1, forward),
            (1, forward),
            (-1, -forward),
            (1, -forward),
        ), ()
    if piece_type in _GOLD_LIKE:
        return (
            (0, forward),
            (-1, forward),
            (1, forward),
            (-1, 0),
            (1, 0),
            (0, -forward),
        ), ()
    if piece_type == 5:
        return (), ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if piece_type == 6:
        return (), ((-1, 0), (1, 0), (0, -1), (0, 1))
    if piece_type == 8:
        return tuple(
            (df, dr)
            for df in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if df or dr
        ), ()
    if piece_type == 13:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ), ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if piece_type == 14:
        return (
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ), ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (), ()


def _build_attack_geometry() -> tuple[object, ...]:
    colors: list[object] = []
    for color in (cshogi.BLACK, cshogi.WHITE):
        pieces: list[object] = []
        for piece_type in range(15):
            step_vectors, ray_vectors = _movement_vectors(piece_type, color)
            squares: list[object] = []
            for square in range(81):
                file_index, rank = divmod(square, 9)
                steps = tuple(
                    (file_index + df) * 9 + rank + dr
                    for df, dr in step_vectors
                    if 0 <= file_index + df < 9 and 0 <= rank + dr < 9
                )
                ray_paths = []
                for df, dr in ray_vectors:
                    path = []
                    f, r = file_index + df, rank + dr
                    while 0 <= f < 9 and 0 <= r < 9:
                        path.append(f * 9 + r)
                        f += df
                        r += dr
                    ray_paths.append(tuple(path))
                squares.append((steps, tuple(ray_paths)))
            pieces.append(tuple(squares))
        colors.append(tuple(pieces))
    return tuple(colors)


_ATTACK_GEOMETRY = _build_attack_geometry()


class EvaluationMode(StrEnum):
    MATERIAL = "material"
    FAST = "fast"
    POSITIONAL = "positional"
    KING_SAFETY = "king_safety"


@dataclass(frozen=True, slots=True)
class EvaluationBreakdown:
    material: int
    positional: int
    mobility: int
    king_safety: int
    check: int
    tempo: int
    total: int


@dataclass(slots=True)
class EvaluationScratch:
    """Reusable attack-map storage for the single-threaded search hot path."""

    attacks: tuple[list[int], list[int]] = field(
        default_factory=lambda: ([0] * 81, [0] * 81)
    )
    touched: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))

    def clear(self) -> None:
        for color in (cshogi.BLACK, cshogi.WHITE):
            attack = self.attacks[color]
            touched = self.touched[color]
            for square in touched:
                attack[square] = 0
            touched.clear()


def evaluate_position(
    board: cshogi.Board,
    mode: EvaluationMode = EvaluationMode.KING_SAFETY,
) -> int:
    """Return a bounded static score from the side-to-move's perspective."""

    return evaluate_breakdown(board, mode).total


def evaluate_breakdown(
    board: cshogi.Board,
    mode: EvaluationMode = EvaluationMode.KING_SAFETY,
    scratch: EvaluationScratch | None = None,
) -> EvaluationBreakdown:
    material, positional_raw = evaluate_fast_absolute(board)
    return evaluate_breakdown_from_absolute(
        board, material, positional_raw, mode=mode, scratch=scratch
    )


def evaluate_breakdown_from_absolute(
    board: cshogi.Board,
    material: int,
    positional_raw: int,
    *,
    mode: EvaluationMode = EvaluationMode.KING_SAFETY,
    scratch: EvaluationScratch | None = None,
    occupied_mask: int | None = None,
) -> EvaluationBreakdown:
    """Evaluate using caller-maintained black-perspective fast components."""

    positional = _clamp(positional_raw, MAX_POSITIONAL)
    workspace = scratch or EvaluationScratch()
    workspace.clear()
    attacks = workspace.attacks
    build_attacks = mode not in {EvaluationMode.MATERIAL, EvaluationMode.FAST}

    squares = _iter_occupied_squares(occupied_mask) if occupied_mask is not None else range(81)
    for square in squares:
        code = int(board.piece(square))
        if code == 0:
            continue
        color = cshogi.WHITE if code >= 16 else cshogi.BLACK
        piece_type = code - 16 if color == cshogi.WHITE else code
        if build_attacks:
            _accumulate_attacks(board, square, piece_type, color, workspace)

    if mode is EvaluationMode.MATERIAL:
        positional = 0
    mobility = 0
    if build_attacks:
        # ``touched`` contains every non-zero attack square exactly once.  It is
        # therefore the same activity measure as scanning both 81-entry arrays,
        # without paying that fixed Python loop at every leaf evaluation.
        black_activity = len(workspace.touched[cshogi.BLACK])
        white_activity = len(workspace.touched[cshogi.WHITE])
        mobility = _clamp((black_activity - white_activity) * 2, MAX_MOBILITY)
    king_safety = 0
    if mode is EvaluationMode.KING_SAFETY:
        kings = (int(board.king_square(cshogi.BLACK)), int(board.king_square(cshogi.WHITE)))
        king_safety = _clamp(
            _king_safety(board, cshogi.BLACK, kings[cshogi.BLACK], attacks)
            - _king_safety(board, cshogi.WHITE, kings[cshogi.WHITE], attacks),
            MAX_KING_SAFETY,
        )

    check = 0
    check_penalty = 35 if mode is EvaluationMode.MATERIAL else CHECK_PENALTY
    if board.is_check():
        check = -check_penalty if board.turn == cshogi.BLACK else check_penalty
    absolute = material + positional + mobility + king_safety + check
    perspective = absolute if board.turn == cshogi.BLACK else -absolute
    tempo = 0 if mode is EvaluationMode.MATERIAL else TEMPO_BONUS
    total = perspective + tempo
    return EvaluationBreakdown(
        material=material if board.turn == cshogi.BLACK else -material,
        positional=positional if board.turn == cshogi.BLACK else -positional,
        mobility=mobility if board.turn == cshogi.BLACK else -mobility,
        king_safety=king_safety if board.turn == cshogi.BLACK else -king_safety,
        check=check if board.turn == cshogi.BLACK else -check,
        tempo=tempo,
        total=total,
    )


def _piece_square_bonus(piece_type: int, color: int, square: int) -> int:
    return _PIECE_SQUARE[color][piece_type][square]


def piece_square_bonus(piece_type: int, color: int, square: int) -> int:
    """Expose the precomputed table to the incremental search position."""

    return _PIECE_SQUARE[color][piece_type][square]


def evaluate_fast_absolute(board: cshogi.Board) -> tuple[int, int]:
    """Return black-perspective material and *unclamped* piece-square totals."""

    material = 0
    positional = 0
    for square, raw in enumerate(board.pieces):
        code = int(raw)
        if code == 0:
            continue
        color = cshogi.WHITE if code >= 16 else cshogi.BLACK
        piece_type = code - 16 if color == cshogi.WHITE else code
        sign = 1 if color == cshogi.BLACK else -1
        material += sign * PIECE_VALUES[piece_type]
        positional += sign * _PIECE_SQUARE[color][piece_type][square]
    for index, value in enumerate(HAND_VALUES):
        material += value * int(board.pieces_in_hand[cshogi.BLACK][index])
        material -= value * int(board.pieces_in_hand[cshogi.WHITE][index])
    return material, positional


def _king_safety(
    board: cshogi.Board,
    color: int,
    king: int,
    attacks: tuple[list[int], list[int]],
) -> int:
    if king < 0:
        return -MAX_KING_SAFETY
    enemy = cshogi.WHITE if color == cshogi.BLACK else cshogi.BLACK
    zone = (king, *_KING_ZONE[king])
    escape = 0
    defenders = 0
    pressure = 0
    for target in _KING_ZONE[king]:
        code = int(board.piece(target))
        if code == 0 and attacks[enemy][target] == 0:
            escape += 1
        defenders += attacks[color][target]
    for target in zone:
        pressure += attacks[enemy][target]

    shield = 0
    for target in _SHIELD_TARGETS[color][king]:
        code = int(board.piece(target))
        if code and (code >= 16) == (color == cshogi.WHITE):
            piece_type = code - 16 if code >= 16 else code
            if piece_type in {1, 4, 7, 9, 10, 11, 12}:
                shield += 1
    return defenders * 4 + escape * 6 + shield * 14 - pressure * 14


def _accumulate_attacks(
    board: cshogi.Board,
    square: int,
    piece_type: int,
    color: int,
    scratch: EvaluationScratch,
) -> None:
    steps, ray_paths = _ATTACK_GEOMETRY[color][piece_type][square]  # type: ignore[index]
    attacks = scratch.attacks[color]
    touched = scratch.touched[color]
    for target in steps:
        if attacks[target] == 0:
            touched.append(target)
        attacks[target] += 1
    for path in ray_paths:
        for target in path:
            if attacks[target] == 0:
                touched.append(target)
            attacks[target] += 1
            if int(board.piece(target)) != 0:
                break


def _clamp(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


def _iter_occupied_squares(mask: int):
    while mask:
        least = mask & -mask
        yield least.bit_length() - 1
        mask ^= least


__all__ = [
    "EvaluationBreakdown",
    "EvaluationMode",
    "EvaluationScratch",
    "HAND_VALUES",
    "MAX_FULL_RESIDUAL",
    "PIECE_VALUES",
    "evaluate_breakdown",
    "evaluate_breakdown_from_absolute",
    "evaluate_fast_absolute",
    "evaluate_position",
    "piece_square_bonus",
]
