import hashlib
import math

import numpy as np

from alphasho.ai import MCTS, SearchConfig
from alphasho.domain import Color, PieceType
from alphasho.engine import POLICY_SIZE
from alphasho.training import (
    SEARCH_RULES_VERSION,
    AdjudicationMode,
    AdjudicationReason,
    AdjudicationRules,
    StartPositionConfig,
    TrainingPosition,
    ruleset_id,
)
from alphasho.training.adjudication import MATERIAL_WEIGHTS
from alphasho.training.positions import load_tactical_positions


class _ZeroEvaluator:
    def evaluate_batch(self, features1, features2):  # noqa: ANN001
        return (
            np.zeros((len(features1), POLICY_SIZE), dtype=np.float32),
            np.zeros(len(features2), dtype=np.float32),
        )


def _rules(
    *,
    max_moves: int = 8,
    threshold: int = 1,
) -> AdjudicationRules:
    return AdjudicationRules(
        AdjudicationMode.CAPTURE_MATERIAL,
        max_moves,
        threshold,
        12.0,
    )


def test_material_weights_and_color_symmetry() -> None:
    assert MATERIAL_WEIGHTS == {
        PieceType.PAWN: 1,
        PieceType.LANCE: 3,
        PieceType.KNIGHT: 3,
        PieceType.SILVER: 5,
        PieceType.BISHOP: 8,
        PieceType.ROOK: 10,
        PieceType.GOLD: 6,
        PieceType.KING: 0,
        PieceType.PROMOTED_PAWN: 6,
        PieceType.PROMOTED_LANCE: 6,
        PieceType.PROMOTED_KNIGHT: 6,
        PieceType.PROMOTED_SILVER: 6,
        PieceType.HORSE: 10,
        PieceType.DRAGON: 12,
    }
    black = TrainingPosition(
        _rules(),
        "4k4/9/9/9/4P4/9/9/9/4K4 b - 1",
    )
    white = TrainingPosition(
        _rules(),
        "4k4/9/9/9/4p4/9/9/9/4K4 w - 1",
    )
    assert black.material_diff() == 1
    assert white.material_diff() == -1
    promoted_and_hand = TrainingPosition(
        _rules(),
        "4k4/9/9/9/4+P4/9/9/9/4K4 b R 1",
    )
    mirrored = TrainingPosition(
        _rules(),
        "4k4/9/9/9/4+p4/9/9/9/4K4 w r 1",
    )
    assert promoted_and_hand.material_diff() == 16
    assert mirrored.material_diff() == -16


def test_capture_counts_clone_pop_and_drop() -> None:
    position = TrainingPosition(
        _rules(threshold=3),
        "4k4/9/9/4pr3/4P4/9/9/9/4K4 b - 1",
    )
    position.push_usi("5e5d")
    assert position.capture_counts == (1, 0)
    clone = position.clone()
    clone.push_usi("4d5d")
    assert clone.capture_counts == (1, 1)
    assert position.capture_counts == (1, 0)
    clone.pop()
    assert clone.capture_counts == (1, 0)
    position.pop()
    assert position.capture_counts == (0, 0)

    drop = TrainingPosition(
        _rules(threshold=3),
        "4k4/9/9/9/9/9/9/9/4K4 b P 1",
    )
    drop.push_usi("P*5e")
    assert drop.capture_counts == (0, 0)


def test_terminal_priority_threshold_and_continuous_cutoff() -> None:
    checkmated = TrainingPosition(
        _rules(max_moves=1),
        "8k/7GR/9/9/9/9/9/9/K8 w - 1",
    )
    real = checkmated.result()
    assert real is not None
    assert real.reason is AdjudicationReason.REAL_RULE
    assert real.winner is Color.BLACK

    threshold = TrainingPosition(
        _rules(),
        "4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1",
    )
    threshold.push_usi("5e5d")
    result = threshold.result()
    assert result is not None
    assert result.reason is AdjudicationReason.CAPTURE_THRESHOLD
    assert result.black_value == 1.0

    cutoff = TrainingPosition(
        _rules(max_moves=1, threshold=9),
        "4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1",
    )
    cutoff.push_usi("5e5d")
    result = cutoff.result()
    assert result is not None
    assert result.reason is AdjudicationReason.MATERIAL_CUTOFF
    assert result.material_diff == 2
    assert math.isclose(result.black_value, math.tanh(2 / 12))
    assert result.value_for(Color.WHITE) == -result.black_value


def test_mcts_sees_capture_threshold_terminal() -> None:
    position = TrainingPosition(
        _rules(),
        "4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1",
    )
    result = MCTS(
        _ZeroEvaluator(),
        SearchConfig(n_playouts=8, batch_size=4),
        seed=4,
    ).search(position, temperature=1e-3)
    chosen = result.action_usis[int(result.probabilities.argmax())]
    assert chosen == "5e5d"


def test_rules_fingerprint_changes_with_training_semantics() -> None:
    assert _rules(threshold=1).fingerprint != _rules(threshold=2).fingerprint
    assert _rules(max_moves=8).fingerprint != _rules(max_moves=9).fingerprint
    rules = _rules()
    starts = StartPositionConfig()
    legacy = hashlib.sha256(
        f"{rules.fingerprint}:{starts.fingerprint()}".encode()
    ).hexdigest()[:20]
    assert SEARCH_RULES_VERSION == 2
    assert ruleset_id(rules, starts) != legacy


def test_training_position_does_not_end_on_twofold_repetition() -> None:
    position = TrainingPosition(
        _rules(max_moves=64, threshold=9),
        "8k/9/9/9/9/9/9/9/K8 b - 1",
    )
    cycle = ("9i9h", "1a1b", "9h9i", "1b1a")

    for usi in cycle:
        position.push_usi(usi)
    assert position.result() is None

    for _ in range(2):
        for usi in cycle:
            position.push_usi(usi)
    result = position.result()
    assert result is not None
    assert result.reason is AdjudicationReason.REAL_RULE
    assert result.winner is None


def test_tactical_pool_has_color_pairs_and_legal_captures() -> None:
    positions = load_tactical_positions(
        "configs/training/tactical_positions.jsonl"
    )
    assert len(positions) >= 16
    names = {position.name for position in positions}
    for name in names:
        if name.startswith("black-"):
            assert name.replace("black-", "white-", 1) in names
