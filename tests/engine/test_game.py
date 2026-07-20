import pytest

from alphasho.domain import Color, OutcomeReason, PieceType
from alphasho.engine import IllegalMoveError, InvalidPositionError, ShogiGame
from alphasho.players import RandomPlayer


def test_initial_snapshot_and_normal_move() -> None:
    game = ShogiGame()
    initial = game.snapshot()

    assert initial.turn is Color.BLACK
    assert initial.move_number == 1
    assert len(game.legal_moves()) == 30

    after = game.play("7g7f")
    assert after.turn is Color.WHITE
    assert after.last_move is not None
    assert after.last_move.usi == "7g7f"
    assert after.move_history == ("7g7f",)


def test_illegal_move_never_reaches_native_push() -> None:
    game = ShogiGame()
    original = game.snapshot().sfen

    with pytest.raises(IllegalMoveError):
        game.play("7g7e")

    assert game.snapshot().sfen == original


def test_capture_adds_unpromoted_piece_to_hand_and_undo_restores_sfen() -> None:
    game = ShogiGame()
    states = [game.snapshot().sfen]
    moves = ("7g7f", "3c3d", "7f7e", "3d3e", "7e7d", "3e3f", "7d7c+")
    for move in moves:
        states.append(game.play(move).sfen)

    assert game.snapshot().hand_count(Color.BLACK, PieceType.PAWN) == 1
    assert game.undo().sfen == states[-2]


def test_nifu_file_is_not_a_legal_drop_target() -> None:
    game = ShogiGame("4k4/9/9/9/9/9/4P4/9/4K4 b P 1")
    pawn_drops = game.legal_moves(drop_piece=PieceType.PAWN)

    assert pawn_drops
    assert all(not move.usi.startswith("P*5") for move in pawn_drops)


def test_optional_and_mandatory_promotion() -> None:
    optional = ShogiGame("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
    optional_usi = {move.usi for move in optional.legal_moves(source_square=39)}
    assert {"5d5c", "5d5c+"} <= optional_usi

    mandatory = ShogiGame("k8/4P4/9/9/9/9/9/9/4K4 b - 1")
    mandatory_usi = {move.usi for move in mandatory.legal_moves(source_square=37)}
    assert "5b5a+" in mandatory_usi
    assert "5b5a" not in mandatory_usi


def test_checkmate_is_detected_when_loading_position() -> None:
    game = ShogiGame("8k/7GR/9/9/9/9/9/9/K8 w - 1")

    assert game.snapshot().outcome.reason is OutcomeReason.CHECKMATE
    assert game.snapshot().outcome.winner is Color.BLACK
    assert game.legal_moves() == ()
    assert game.undo().outcome.reason is OutcomeReason.CHECKMATE


def test_repetition_requires_four_occurrences() -> None:
    game = ShogiGame("8k/9/9/9/9/9/9/9/K8 b - 1")
    cycle = ("9i9h", "1a1b", "9h9i", "1b1a")

    for move in cycle * 2:
        game.play(move)
    assert not game.snapshot().outcome.is_over

    for move in cycle:
        game.play(move)
    assert game.snapshot().outcome.reason is OutcomeReason.REPETITION


def test_resign_and_undo_terminal_move() -> None:
    game = ShogiGame()
    game.play("7g7f")
    outcome = game.resign().outcome
    assert outcome.reason is OutcomeReason.RESIGNATION
    assert outcome.winner is Color.BLACK

    snapshot = game.undo()
    assert not snapshot.outcome.is_over
    assert snapshot.move_history == ()


def test_invalid_sfen_is_rejected() -> None:
    with pytest.raises(InvalidPositionError):
        ShogiGame("not an sfen")


def test_random_player_returns_a_supplied_legal_move() -> None:
    game = ShogiGame()
    legal = game.legal_moves()

    assert RandomPlayer(seed=7).choose_move(game.snapshot(), legal) in legal
