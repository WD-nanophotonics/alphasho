from dataclasses import replace

import pytest

from alphasho.training import load_training_config, validate_training_config
from alphasho.training.positions import (
    evaluation_suite_id,
    load_evaluation_positions,
)


def test_baseline_evaluation_suite_pairs_ten_unique_positions() -> None:
    config = load_training_config("configs/training/baseline.toml")

    positions = config.evaluation_positions
    assert len(positions) == 10
    assert len({position.name for position in positions}) == 10
    assert len({position.sfen for position in positions}) == 10
    assert config.evaluation_games == 20
    assert config.evaluation_suite_id == evaluation_suite_id(positions)

    assigned = [
        config.evaluation_position_for_game(index)
        for index in range(config.evaluation_games)
    ]
    for pair_index, position in enumerate(positions):
        assert assigned[pair_index * 2] == position
        assert assigned[pair_index * 2 + 1] == position


def test_evaluation_suite_hash_depends_on_order_and_content() -> None:
    positions = load_evaluation_positions(
        "configs/training/evaluation_positions.jsonl"
    )
    assert evaluation_suite_id(positions) == evaluation_suite_id(positions)
    assert evaluation_suite_id(positions) != evaluation_suite_id(
        tuple(reversed(positions))
    )


def test_evaluation_game_count_must_match_two_per_position(tmp_path) -> None:
    config = load_training_config("configs/training/baseline.toml")
    with pytest.raises(ValueError, match="two games per evaluation position"):
        validate_training_config(replace(config, evaluation_games=18))
