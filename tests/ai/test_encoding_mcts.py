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
