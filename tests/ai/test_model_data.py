import numpy as np

from alphasho.ai import (
    ModelConfig,
    ShogiPolicyValueNet,
    TorchEvaluator,
    encode_position,
    load_inference_checkpoint,
    save_inference_checkpoint,
)
from alphasho.engine import SearchPosition
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
