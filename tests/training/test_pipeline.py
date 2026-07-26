import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import alphasho.atomic_io as atomic_io
from alphasho.ai import ModelConfig, ShogiPolicyValueNet, save_inference_checkpoint
from alphasho.atomic_io import ExclusiveFileLock, FileLockError
from alphasho.domain import GameRecord
from alphasho.engine import ShogiGame, save_record
from alphasho.training import (
    AlphaZeroTrainer,
    IncompatibleRunError,
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


def test_incompatible_rules_stop_before_checkpoint_write(tmp_path) -> None:
    config = replace(
        QuickTrainingSettings().build_config(),
        artifact_root=tmp_path / "cpu-debug",
    )
    config.artifact_root.mkdir(parents=True)
    state_path = config.artifact_root / "checkpoints" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"revision": 17}', encoding="utf-8")
    original = state_path.read_bytes()
    effective = {
        "ruleset_id": "different-rules",
        "model": config.model.to_dict(),
    }
    (config.artifact_root / "effective_config.json").write_text(
        json.dumps(effective), encoding="utf-8"
    )

    with pytest.raises(IncompatibleRunError):
        AlphaZeroTrainer(config).run()

    assert state_path.read_bytes() == original
    assert not (state_path.parent / "training_state.pt").exists()


def test_replay_without_pipeline_state_is_rejected(tmp_path) -> None:
    config = replace(
        QuickTrainingSettings().build_config(),
        artifact_root=tmp_path / "cpu-debug",
    )
    store = ShardStore(config.artifact_root / "selfplay")
    store.write(
        [_sample()],
        source="selfplay",
        model_version="iteration-0",
        shard_id="iteration-000001-game-000001",
        ruleset_id=config.ruleset_id,
    )

    with pytest.raises(IncompatibleRunError, match="pipeline_state"):
        AlphaZeroTrainer(config).run()

    assert not (config.artifact_root / "checkpoints" / "pipeline_state.json").exists()


def test_incompatible_model_stops_before_checkpoint_write(tmp_path) -> None:
    config = replace(
        QuickTrainingSettings().build_config(),
        artifact_root=tmp_path / "cpu-debug",
    )
    checkpoint = config.artifact_root / "checkpoints" / "best.pt"
    save_inference_checkpoint(
        checkpoint,
        ShogiPolicyValueNet(ModelConfig(16, 1, 32)),
        metadata={"ruleset_id": config.ruleset_id},
    )
    original = checkpoint.read_bytes()

    with pytest.raises(IncompatibleRunError, match="model structure"):
        AlphaZeroTrainer(config).run()

    assert checkpoint.read_bytes() == original
    assert not (checkpoint.parent / "pipeline_state.json").exists()


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
        evaluation_required=True,
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


def test_resumed_evaluation_trims_results_after_game_count_reduction() -> None:
    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer.config = SimpleNamespace(evaluation_games=20)
    trainer._state = PipelineState(
        active_iteration=5,
        stage=TrainingStage.EVALUATION,
        evaluation_results=[
            {
                "game_index": index,
                "outcome": (
                    "candidate"
                    if index < 5
                    else "best"
                    if index < 12
                    else "draw"
                ),
            }
            for index in range(91)
        ],
    )
    persisted: list[int] = []
    events: list[tuple[str, dict[str, object]]] = []
    trainer._persist_state = lambda: persisted.append(
        trainer._state.evaluation_completed
    )
    trainer._emit = lambda _stage, event, **fields: events.append(
        (event, fields)
    )

    trainer._reconcile_evaluation_progress()

    assert [item["game_index"] for item in trainer._state.evaluation_results] == list(
        range(20)
    )
    assert trainer._state.evaluation_completed == 20
    assert trainer._state.candidate_wins == 5
    assert trainer._state.best_wins == 7
    assert trainer._state.draws == 8
    assert persisted == [20]
    assert events == [
        (
            "evaluation-progress-trimmed",
            {
                "level": "WARNING",
                "iteration": 5,
                "previous_games": 91,
                "retained_games": 20,
                "evaluation_games": 20,
            },
        )
    ]

    trainer._reconcile_evaluation_progress()
    assert persisted == [20]


def test_changed_evaluation_suite_discards_only_partial_evaluation() -> None:
    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer._state = PipelineState(
        active_iteration=86,
        stage=TrainingStage.EVALUATION,
        evaluation_suite_id="legacy-suite",
        evaluation_results=[
            {"game_index": 7, "outcome": "candidate"},
            {"game_index": 2, "outcome": "best"},
        ],
    )
    persisted: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    trainer._persist_state = lambda: persisted.append(
        trainer._state.evaluation_suite_id
    )
    trainer._emit = lambda _stage, event, **fields: events.append(
        (event, fields)
    )

    trainer._reconcile_evaluation_suite("diverse-suite")

    assert trainer._state.evaluation_suite_id == "diverse-suite"
    assert trainer._state.evaluation_results == []
    assert trainer._state.evaluation_completed == 0
    assert trainer._state.candidate_wins == 0
    assert trainer._state.best_wins == 0
    assert persisted == ["diverse-suite"]
    assert events[0][0] == "evaluation-suite-reset"
    assert events[0][1]["previous_games"] == 2

    trainer._reconcile_evaluation_suite("diverse-suite")
    assert persisted == ["diverse-suite"]


def test_off_cycle_iteration_skips_evaluation() -> None:
    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer.config = SimpleNamespace(iterations=200, check_frequency=5)
    trainer._state = PipelineState(
        completed_iterations=0,
        active_iteration=1,
        stage=TrainingStage.EVALUATION,
        evaluation_required=False,
    )
    calls: list[str] = []
    trainer._emit = (
        lambda _stage, event, **_kwargs: calls.append(event)
    )
    trainer._run_evaluation = lambda _iteration: (
        calls.append("evaluation") or (None, False)
    )
    trainer._complete_iteration = lambda *_args, **_kwargs: calls.append(
        "complete"
    )

    trainer._run_iteration(1)

    assert calls == ["iteration-start", "evaluation-skipped", "complete"]


def test_new_iteration_preserves_checkpoint_revision() -> None:
    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer.config = SimpleNamespace(iterations=10, check_frequency=5)
    trainer._state = PipelineState(
        revision=99,
        completed_iterations=1,
    )
    trainer._emit = lambda *_args, **_kwargs: None
    revisions: list[int] = []
    trainer._persist_state = lambda: revisions.append(
        trainer._state.revision
    )

    def stop_after_initialization(_iteration: int) -> None:
        raise RuntimeError("stop")

    trainer._run_selfplay = stop_after_initialization

    with pytest.raises(RuntimeError, match="stop"):
        trainer._run_iteration(2)

    assert revisions == [99]


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

    legacy_payload = torch.load(checkpoint, weights_only=False)
    legacy_payload.pop("torch_cuda_random_state_all", None)
    legacy_checkpoint = tmp_path / "legacy-training.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    resumed.load_state(legacy_checkpoint)

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


@pytest.mark.parametrize("language", list(TrainingLanguage))
def test_reporter_supplies_iteration_to_message_templates(
    tmp_path,
    language: TrainingLanguage,
) -> None:
    reporter = TrainingReporter(
        tmp_path / language.value,
        "cpu_debug",
        MonitoringConfig(
            console_level="ERROR",
            output_language=language,
        ),
    )
    started = reporter.emit(
        TrainingStage.SELFPLAY,
        "iteration-start",
        iteration=35,
        iterations=2000,
    )
    completed = reporter.emit(
        TrainingStage.COMPLETED,
        "iteration-complete",
        iteration=35,
        iterations=2000,
        seconds=5.0,
        loss=1.25,
        promoted=False,
        evaluated=True,
    )
    reporter.close()

    assert "35/2000" in started.message
    assert "35/2000" in completed.message
    events = [
        json.loads(line)
        for line in (
            tmp_path / language.value / "cpu_debug.jsonl"
        ).read_text("utf-8").splitlines()
    ]
    assert [event["iteration"] for event in events] == [35, 35]
    assert events[0]["fields"] == {"iterations": 2000}
    assert "iteration" not in events[1]["fields"]


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
