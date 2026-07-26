import numpy as np

from alphasho.ai import (
    AlphaZeroPlayer,
    ModelConfig,
    ShogiPolicyValueNet,
    TorchEvaluator,
    encode_position,
    load_inference_checkpoint,
    save_inference_checkpoint,
)
from alphasho.ai.time_manager import allocate_thinking_budget
from alphasho.domain import (
    ClockState,
    Color,
    SideTimeConfig,
    ThinkingConfig,
    ThinkingPreset,
    ThinkingStrategy,
    TimeControl,
    TimeControlMode,
)
from alphasho.engine import SearchPosition, ShogiGame
from alphasho.training.data import ShardStore, TrainingSample
from alphasho.training.trainer import PolicyValueTrainer


def _sample() -> TrainingSample:
    position = SearchPosition()
    encoded = encode_position(position)
    action = position.legal_actions()[0]
    return TrainingSample(
        encoded.features1,
        encoded.features2,
        np.asarray([action.label], dtype=np.uint16),
        np.asarray([1.0], dtype=np.float16),
        0,
        {"usi": action.usi},
    )


def test_model_train_checkpoint_and_shard_roundtrip(tmp_path) -> None:
    model = ShogiPolicyValueNet(ModelConfig(8, 1, 16))
    trainer = PolicyValueTrainer(
        model,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    sample = _sample()
    before = next(model.parameters()).detach().clone()
    metrics = trainer.train([sample], steps=1, batch_size=1, seed=3)

    assert np.isfinite(metrics.loss)
    assert not np.array_equal(
        before.cpu().numpy(),
        next(model.parameters()).detach().cpu().numpy(),
    )

    store = ShardStore(tmp_path / "replay")
    entry = store.write([sample], source="test", model_version="v0")
    loaded = store.load_entry(entry)
    assert len(loaded) == 1
    assert loaded[0].metadata["usi"] == sample.metadata["usi"]
    assert np.array_equal(loaded[0].features1, sample.features1)

    checkpoint = tmp_path / "best.pt"
    expected, _ = TorchEvaluator(model, "cpu").evaluate_batch(
        sample.features1[None],
        sample.features2[None],
    )
    save_inference_checkpoint(
        checkpoint,
        model,
        search_config={"n_playouts": 2},
    )
    _loaded_model, evaluator, metadata = load_inference_checkpoint(
        checkpoint,
        device="cpu",
    )
    actual, _ = evaluator.evaluate_batch(
        sample.features1[None],
        sample.features2[None],
    )
    assert np.allclose(actual, expected)
    assert metadata["search_config"]["n_playouts"] == 2


def test_same_checkpoint_creates_independent_players_and_searches_cycle(tmp_path) -> None:
    checkpoint = tmp_path / "shared.pt"
    save_inference_checkpoint(
        checkpoint,
        ShogiPolicyValueNet(ModelConfig(8, 1, 16)),
        search_config={"n_playouts": 2, "batch_size": 1},
    )
    black = AlphaZeroPlayer(checkpoint, device="cpu", seed=1)
    white = AlphaZeroPlayer(checkpoint, device="cpu", seed=2)
    assert black is not white
    assert black._mcts is not white._mcts

    game = ShogiGame()
    for usi in ("3g3f", "4a4b", "6i6h", "7c7d", "2h1h", "8b9b", "1h2h", "9b8b"):
        game.play(usi)
    move = black.choose_move(game.snapshot(), game.legal_moves())
    assert move in game.legal_moves()


def test_ui_thinking_budget_uses_clock_and_presets() -> None:
    balanced = ThinkingConfig.for_preset(ThinkingPreset.BALANCED)
    untimed = allocate_thinking_budget(
        balanced,
        TimeControl(),
        ClockState((0, 0), (0, 0), Color.BLACK, False),
        Color.BLACK,
    )
    assert untimed.time_limit_seconds == 5.0
    assert untimed.max_playouts == 10_000

    byoyomi = TimeControl(
        mode=TimeControlMode.BYOYOMI,
        black=SideTimeConfig(0, 10),
        white=SideTimeConfig(0, 10),
    )
    clock = ClockState((0, 0), (10_000, 10_000), Color.BLACK, True)
    automatic = allocate_thinking_budget(
        balanced,
        byoyomi,
        clock,
        Color.BLACK,
    )
    assert automatic.time_limit_seconds == 8.0
    assert automatic.max_playouts == 800_000

    one_hour = allocate_thinking_budget(
        balanced,
        TimeControl(
            mode=TimeControlMode.FISCHER,
            black=SideTimeConfig(3600, 0),
            white=SideTimeConfig(3600, 0),
        ),
        ClockState((3_600_000, 3_600_000), (0, 0), Color.BLACK, True),
        Color.BLACK,
        move_number=1,
    )
    assert one_hour.time_limit_seconds == 54.0
    assert one_hour.max_playouts == 5_400_000

    late_game = allocate_thinking_budget(
        balanced,
        TimeControl(
            mode=TimeControlMode.FISCHER,
            black=SideTimeConfig(3600, 0),
            white=SideTimeConfig(3600, 0),
        ),
        ClockState((3_600_000, 3_600_000), (0, 0), Color.BLACK, True),
        Color.BLACK,
        move_number=81,
    )
    assert late_game.time_limit_seconds == 162.0

    fixed = allocate_thinking_budget(
        ThinkingConfig(
            strategy=ThinkingStrategy.FIXED_PLAYOUT,
            max_playouts=321,
        ),
        TimeControl(),
        clock,
        Color.BLACK,
    )
    assert fixed.time_limit_seconds is None
    assert fixed.max_playouts == 321
