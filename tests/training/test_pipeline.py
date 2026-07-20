import json
from types import SimpleNamespace

import numpy as np
import pytest

import alphasho.atomic_io as atomic_io
from alphasho.ai import ModelConfig, ShogiPolicyValueNet
from alphasho.atomic_io import ExclusiveFileLock, FileLockError
from alphasho.domain import GameRecord
from alphasho.engine import ShogiGame, save_record
from alphasho.training import (
    AlphaZeroTrainer,
    MonitoringConfig,
    PipelineState,
    QuickTrainingSettings,
    ShardStore,
    TrainingLanguage,
    TrainingReporter,
    TrainingSample,
    TrainingStage,
    load_recent_samples,
    render_training_message,
    validate_message_catalog,
)
from alphasho.training.orchestrator import recover_pipeline_state
from alphasho.training.supervised import ingest_supervised_corpus
from alphasho.training.trainer import PolicyValueTrainer


def _sample() -> TrainingSample:
    return TrainingSample(
        np.zeros((62, 9, 9), dtype=np.uint8),
        np.zeros((57, 9, 9), dtype=np.uint8),
        np.asarray([0], dtype=np.uint16),
        np.asarray([1], dtype=np.float16),
        0,
        {"usi": "7g7f"},
    )


def test_atomic_replace_retries_sharing_violation(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "state.json"
    temporary = tmp_path / "state.json.tmp"
    destination.write_text("old", encoding="utf-8")
    temporary.write_text("new", encoding="utf-8")
    real_replace = atomic_io.os.replace
    calls = 0

    def flaky_replace(source, target) -> None:  # noqa: ANN001
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, "locked", str(target))
        real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    attempts = atomic_io.replace_with_retry(
        temporary,
        destination,
        retry_delays=(0, 0),
    )

    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_atomic_replace_failure_preserves_old_and_temporary(
    monkeypatch,
    tmp_path,
) -> None:
    destination = tmp_path / "state.json"
    temporary = tmp_path / "state.json.tmp"
    destination.write_text("old", encoding="utf-8")
    temporary.write_text("new", encoding="utf-8")

    def always_locked(_source, target) -> None:  # noqa: ANN001
        raise PermissionError(5, "locked", str(target))

    monkeypatch.setattr(atomic_io.os, "replace", always_locked)
    with pytest.raises(PermissionError):
        atomic_io.replace_with_retry(
            temporary,
            destination,
            retry_delays=(),
        )

    assert destination.read_text(encoding="utf-8") == "old"
    assert temporary.read_text(encoding="utf-8") == "new"


def test_newer_temporary_pipeline_state_is_recovered(tmp_path) -> None:
    path = tmp_path / "pipeline_state.json"
    path.write_text(
        json.dumps(
            PipelineState(
                revision=4,
                completed_iterations=15,
                active_iteration=16,
                stage=TrainingStage.SELFPLAY,
                selfplay_completed=0,
            ).to_dict()
        ),
        encoding="utf-8",
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            PipelineState(
                revision=5,
                completed_iterations=15,
                active_iteration=16,
                stage=TrainingStage.SELFPLAY,
                selfplay_completed=1,
            ).to_dict()
        ),
        encoding="utf-8",
    )

    state, recovered, _attempts = recover_pipeline_state(path)

    assert recovered
    assert state is not None
    assert state.revision == 5
    assert state.selfplay_completed == 1
    assert not temporary.exists()


def test_profile_lock_rejects_second_trainer(tmp_path) -> None:
    first = ExclusiveFileLock(tmp_path / ".trainer.lock")
    second = ExclusiveFileLock(tmp_path / ".trainer.lock")
    first.acquire()
    try:
        with pytest.raises(FileLockError):
            second.acquire()
    finally:
        first.release()


def test_resumed_evaluation_is_not_skipped_after_frequency_change() -> None:
    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer.config = SimpleNamespace(iterations=200, check_frequency=10)
    trainer._state = PipelineState(
        completed_iterations=5,
        active_iteration=6,
        stage=TrainingStage.EVALUATION,
    )
    calls: list[str] = []
    trainer._emit = lambda *_args, **_kwargs: None
    trainer._require_current_model = lambda: object()
    trainer._run_tactical_benchmark = (
        lambda *_args, **_kwargs: calls.append("benchmark")
    )
    trainer._run_evaluation = lambda _iteration: (
        calls.append("evaluation") or (None, False)
    )
    trainer._complete_iteration = lambda *_args, **_kwargs: calls.append(
        "complete"
    )

    trainer._run_iteration(6)

    assert calls == ["benchmark", "evaluation", "complete"]


def test_quick_settings_reporter_resume_and_idempotent_shard(tmp_path) -> None:
    settings = QuickTrainingSettings(
        profile="smoke",
        device="cpu",
        iterations=3,
        model_preset="smoke",
        playouts=7,
        c_puct=1.7,
        inference_batch_size=3,
        games_per_iteration=2,
        dirichlet_alpha=0.2,
        dirichlet_weight=0.1,
        training_batch_size=4,
        replay_buffer_size=99,
        train_steps_per_iteration=6,
        learning_rate=0.002,
        check_frequency=2,
        evaluation_games=4,
        promotion_score=0.6,
    )
    config = settings.build_config()
    assert config.search.n_playouts == 7
    assert config.replay_capacity == 99

    reporter = TrainingReporter(
        tmp_path / "logs",
        "smoke",
        MonitoringConfig(
            console_level="ERROR",
            output_language=TrainingLanguage.CHINESE,
        ),
    )
    reporter.emit(
        TrainingStage.STARTUP,
        "new-run",
        iteration=1,
        loss=1.25,
    )
    reporter.close()
    status = json.loads(
        (tmp_path / "logs" / "status.json").read_text("utf-8")
    )
    assert status["fields"]["loss"] == 1.25
    assert status["message_key"] == "startup.new-run"
    assert status["language"] == "zh"
    assert "未发现训练状态" in status["message"]
    assert (tmp_path / "logs" / "training.zh.log").is_file()
    metric = json.loads(
        (tmp_path / "logs" / "smoke.jsonl").read_text("utf-8")
    )
    assert metric["schema_version"] == 2
    assert metric["fields"]["loss"] == 1.25

    model_config = ModelConfig(8, 1, 16)
    original = PolicyValueTrainer(
        ShogiPolicyValueNet(model_config),
        device="cpu",
        learning_rate=0.01,
        weight_decay=0.2,
    )
    checkpoint = tmp_path / "training.pt"
    original.save_state(checkpoint, iteration=2, metadata={"stage": "training"})
    resumed = PolicyValueTrainer(
        ShogiPolicyValueNet(model_config),
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.01,
    )
    resumed.load_state(
        checkpoint,
        learning_rate=0.001,
        weight_decay=0.01,
    )
    assert resumed.optimizer.param_groups[0]["lr"] == 0.001

    store = ShardStore(tmp_path / "replay")
    first = store.write(
        [_sample()],
        source="selfplay",
        model_version="best-0",
        shard_id="iteration-000001-game-000001",
        ruleset_id="rule-a",
    )
    second = store.write(
        [_sample()],
        source="selfplay",
        model_version="best-0",
        shard_id="iteration-000001-game-000001",
        ruleset_id="rule-a",
    )
    assert first.path == second.path
    assert len(store.entries()) == 1
    assert len(load_recent_samples([store], 10, ruleset_id="rule-a")) == 1


def test_training_message_catalog_renders_three_languages() -> None:
    validate_message_catalog()
    fields = {"step": 3, "total_steps": 10}
    rendered = {
        language: render_training_message(
            "training.training-heartbeat",
            language,
            fields,
        )
        for language in TrainingLanguage
    }
    assert "still running" in rendered[TrainingLanguage.ENGLISH]
    assert "仍在运行" in rendered[TrainingLanguage.CHINESE]
    assert "実行中" in rendered[TrainingLanguage.JAPANESE]


def test_csa_supervised_ingestion(tmp_path) -> None:
    game = ShogiGame()
    game.play("7g7f")
    outcome = game.resign()
    record = GameRecord(
        initial_sfen=game.snapshot().initial_sfen,
        black_name="black",
        white_name="white",
        moves=["7g7f"],
        outcome=outcome.outcome,
    )
    records = tmp_path / "records"
    records.mkdir()
    save_record(record, records / "sample.csa")
    train_store = ShardStore(tmp_path / "train")

    stats = ingest_supervised_corpus(
        (records,),
        train_store,
        ShardStore(tmp_path / "validation"),
        validation_fraction=0,
        shard_size=8,
    )
    samples = train_store.load_entry(train_store.entries()[0])

    assert stats.games == 1
    assert samples[0].value == 1
    assert samples[0].metadata["usi"] == "7g7f"
