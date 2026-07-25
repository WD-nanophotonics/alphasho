import time

import numpy as np

from alphasho.ai import MCTS, SearchConfig, encode_position, legal_policy_mask
from alphasho.engine import POLICY_SIZE, SearchPosition, ShogiGame


class _ZeroEvaluator:
    def evaluate_batch(self, features1, features2):  # noqa: ANN001
        return (
            np.zeros((len(features1), POLICY_SIZE), dtype=np.float32),
            np.zeros(len(features2), dtype=np.float32),
        )


def test_dlshogi_shapes_labels_and_mask() -> None:
    position = SearchPosition()
    encoded = encode_position(position)
    actions = position.legal_actions()
    mask = legal_policy_mask(position)

    assert encoded.features1.shape == (62, 9, 9)
    assert encoded.features2.shape == (57, 9, 9)
    assert encoded.features1.dtype == np.uint8
    assert encoded.features2.dtype == np.uint8
    assert len({action.label for action in actions}) == len(actions)
    assert all(0 <= action.label < POLICY_SIZE for action in actions)
    assert int(mask.sum()) == len(actions)


def test_search_position_replays_snapshot_and_pop_restores() -> None:
    game = ShogiGame()
    game.play("7g7f")
    first_sfen = game.snapshot().sfen
    game.play("3c3d")
    position = SearchPosition.from_snapshot(game.snapshot())

    assert position.sfen == game.snapshot().sfen
    assert position.history == ("7g7f", "3c3d")
    assert position.pop() == "3c3d"
    assert position.sfen == first_sfen


def test_search_position_requires_four_position_occurrences() -> None:
    position = SearchPosition("8k/9/9/9/9/9/9/9/K8 b - 1")
    cycle = ("9i9h", "1a1b", "9h9i", "1b1a")

    for expected_count in (2, 3):
        for usi in cycle:
            position.push_usi(usi)
        assert position.repetition_count == expected_count
        assert position.terminal_value() is None

    clone = position.clone()
    for usi in cycle:
        clone.push_usi(usi)
    assert clone.repetition_count == 4
    assert clone.terminal_value() == 0.0
    assert position.repetition_count == 3

    for _ in cycle:
        clone.pop()
    assert clone.sfen == position.sfen
    assert clone.history == position.history
    assert clone.repetition_count == 3
    assert clone.terminal_value() is None


def test_screenshot_twofold_repetition_remains_searchable() -> None:
    moves = ("3g3f", "4a4b", "6i6h", "7c7d", "2h1h", "8b9b", "1h2h", "9b8b")
    game = ShogiGame()
    for usi in moves:
        game.play(usi)
    position = SearchPosition.from_snapshot(game.snapshot())

    assert not game.snapshot().outcome.is_over
    assert len(game.legal_moves()) == 28
    assert position.repetition_count == 2
    assert position.terminal_value() is None
    result = MCTS(
        _ZeroEvaluator(),
        SearchConfig(n_playouts=4, batch_size=2),
        seed=11,
    ).search(position, temperature=1.0)
    assert result.choose(np.random.default_rng(11)) in {
        move.usi for move in game.legal_moves()
    }


def test_mcts_returns_only_legal_normalized_actions() -> None:
    position = SearchPosition()
    legal = {action.usi for action in position.legal_actions()}
    search = MCTS(
        _ZeroEvaluator(),
        SearchConfig(n_playouts=4, batch_size=2),
        seed=7,
    )
    result = search.search(position, temperature=1.0)

    assert set(result.action_usis) == legal
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert result.visits.sum() == 4
    assert search.root.virtual_visits == 0
    assert all(node.virtual_visits == 0 for node in search.root.children.values())


def test_mcts_honours_deadline_and_depth_limit() -> None:
    position = SearchPosition()
    search = MCTS(
        _ZeroEvaluator(),
        SearchConfig(n_playouts=100, batch_size=2),
        seed=7,
    )
    result = search.search(
        position,
        max_playouts=100,
        deadline=time.monotonic(),
        max_depth=1,
    )
    assert result.stats.playouts == 0
    assert result.choose(np.random.default_rng(7)) in {
        action.usi for action in position.legal_actions()
    }

    result = search.search(position, max_playouts=8, max_depth=1)
    assert result.stats.playouts == 8
    assert result.stats.max_depth <= 1


def test_promotion_drop_and_terminal_labels_use_cshogi() -> None:
    promotion = SearchPosition(
        "4k4/9/9/4P4/9/9/9/9/4K4 b - 1"
    )
    choices = {
        action.usi: action.label
        for action in promotion.legal_actions()
        if action.usi.startswith("5d5c")
    }
    assert set(choices) == {"5d5c", "5d5c+"}
    assert choices["5d5c"] != choices["5d5c+"]

    drops = SearchPosition("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    assert any(action.usi.startswith("P*") for action in drops.legal_actions())

    checkmated = SearchPosition("8k/7GR/9/9/9/9/9/9/K8 w - 1")
    assert checkmated.terminal_value() == -1.0
