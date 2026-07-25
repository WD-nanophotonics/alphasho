"""Observable, resumable AlphaZero training controller."""

from __future__ import annotations

import copy
import json
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from alphasho.ai import (
    SearchCancelled,
    ShogiPolicyValueNet,
    TorchEvaluator,
    load_inference_checkpoint,
    save_inference_checkpoint,
)
from alphasho.ai.model import resolve_device
from alphasho.atomic_io import (
    ExclusiveFileLock,
    FileLockError,
    atomic_copy,
    atomic_write_json,
    recover_newer_temporary,
)
from alphasho.training.adjudication import AdjudicationMode
from alphasho.training.benchmark import (
    TacticalBenchmarkResult,
    evaluate_tactical_benchmark,
)
from alphasho.training.config import (
    MODEL_PRESETS,
    QuickTrainingSettings,
    TrainingConfig,
    effective_config_dict,
)
from alphasho.training.data import ShardStore, load_recent_samples
from alphasho.training.evaluation import EvaluationResult, evaluate_candidate
from alphasho.training.messages import render_training_message
from alphasho.training.parallel import (
    CompletedEvaluation,
    CompletedSelfPlay,
    InferenceMetrics,
    resolve_worker_count,
    run_parallel_evaluation,
    run_parallel_selfplay,
)
from alphasho.training.reporting import (
    TrainerStatus,
    TrainingReporter,
    TrainingStage,
    write_effective_config,
)
from alphasho.training.supervised import ingest_supervised_corpus
from alphasho.training.trainer import (
    PolicyValueTrainer,
    TrainMetrics,
    TrainStepMetrics,
)


@dataclass(slots=True)
class PipelineState:
    schema_version: int = 4
    revision: int = 0
    completed_iterations: int = 0
    active_iteration: int = 0
    stage: TrainingStage = TrainingStage.STARTUP
    selfplay_completed: int = 0
    selfplay_completed_games: list[int] = field(default_factory=list)
    train_steps_completed: int = 0
    evaluation_completed: int = 0
    candidate_wins: int = 0
    best_wins: int = 0
    draws: int = 0
    evaluation_results: list[dict[str, Any]] = field(default_factory=list)
    evaluation_required: bool | None = None
    best_iteration: int = 0
    consecutive_no_promotion: int = 0

    def __post_init__(self) -> None:
        if self.selfplay_completed and not self.selfplay_completed_games:
            self.selfplay_completed_games = list(range(self.selfplay_completed))
        if self.evaluation_completed and not self.evaluation_results:
            outcomes = (
                ["candidate"] * self.candidate_wins
                + ["best"] * self.best_wins
                + ["draw"] * self.draws
            )
            outcomes.extend(
                ["draw"] * max(0, self.evaluation_completed - len(outcomes))
            )
            self.evaluation_results = [
                {
                    "game_index": index,
                    "outcome": outcomes[index],
                    "terminal": "legacy",
                    "captures": 0,
                    "absolute_material_diff": 0,
                }
                for index in range(self.evaluation_completed)
            ]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["stage"] = self.stage.value
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PipelineState:
        fields = {
            key: value
            for key, value in values.items()
            if key in cls.__dataclass_fields__
        }
        fields["stage"] = TrainingStage(
            fields.get("stage", TrainingStage.STARTUP)
        )
        source_schema = int(fields.get("schema_version", 1))
        if source_schema < 3:
            selfplay_completed = int(fields.get("selfplay_completed", 0))
            fields["selfplay_completed_games"] = list(
                range(selfplay_completed)
            )
            evaluation_completed = int(fields.get("evaluation_completed", 0))
            outcomes = (
                ["candidate"] * int(fields.get("candidate_wins", 0))
                + ["best"] * int(fields.get("best_wins", 0))
                + ["draw"] * int(fields.get("draws", 0))
            )
            outcomes.extend(
                ["draw"] * max(0, evaluation_completed - len(outcomes))
            )
            fields["evaluation_results"] = [
                {
                    "game_index": index,
                    "outcome": outcomes[index],
                    "terminal": "legacy",
                    "captures": 0,
                    "absolute_material_diff": 0,
                }
                for index in range(evaluation_completed)
            ]
        if "evaluation_required" not in values:
            fields["evaluation_required"] = (
                fields["stage"] is TrainingStage.EVALUATION
                if source_schema < 3
                else None
            )
        fields["schema_version"] = 4
        fields["selfplay_completed_games"] = sorted(
            {int(item) for item in fields.get("selfplay_completed_games", [])}
        )
        fields["evaluation_results"] = sorted(
            list(fields.get("evaluation_results", [])),
            key=lambda item: int(item["game_index"]),
        )
        fields["selfplay_completed"] = len(fields["selfplay_completed_games"])
        fields["evaluation_completed"] = len(fields["evaluation_results"])
        fields["candidate_wins"] = sum(
            item.get("outcome") == "candidate"
            for item in fields["evaluation_results"]
        )
        fields["best_wins"] = sum(
            item.get("outcome") == "best"
            for item in fields["evaluation_results"]
        )
        fields["draws"] = sum(
            item.get("outcome") == "draw"
            for item in fields["evaluation_results"]
        )
        return cls(**fields)


class IncompatibleRunError(RuntimeError):
    """Raised before mutation when persisted artifacts do not match a run."""


class AlphaZeroTrainer:
    """High-level lifecycle used by the compact user training pipeline."""

    def __init__(
        self,
        settings: QuickTrainingSettings | TrainingConfig,
        reporter: TrainingReporter | None = None,
    ) -> None:
        self.settings = settings
        self.config = (
            settings.build_config()
            if isinstance(settings, QuickTrainingSettings)
            else settings
        )
        self.paths = self._artifact_paths()
        self._reporter = reporter
        self._owns_reporter = reporter is None
        self._stop_event = threading.Event()
        self._state = PipelineState()
        self._best_model: ShogiPolicyValueNet | None = None
        self._current_model: ShogiPolicyValueNet | None = None
        self._network_trainer: PolicyValueTrainer | None = None
        self._run_lock = ExclusiveFileLock(
            self.config.artifact_root / ".trainer.lock"
        )
        self._last_checkpoint = 0.0
        self._run_started = 0.0
        self._checkpoint_ready = False

    def run(self) -> None:
        self._run_started = time.monotonic()
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        try:
            self._run_lock.acquire()
        except FileLockError as exc:
            if self._reporter is not None:
                self._emit(
                    TrainingStage.ERROR,
                    "trainer-lock-conflict",
                    level="ERROR",
                    detail=str(exc),
                )
            else:
                message = render_training_message(
                    "error.trainer-lock-conflict",
                    self.config.monitoring.output_language,
                    {"detail": str(exc)},
                )
                print(f"[ERROR][STARTUP] {message}", flush=True)
            raise
        if self._reporter is None:
            self._reporter = TrainingReporter(
                self.paths["logs"],
                self.config.name,
                self.config.monitoring,
            )
        previous_config = self._read_effective_config()
        try:
            self._install_signal_handlers()
            self._report_startup(previous_config)
            self._audit_persisted_run(previous_config)
            self._initialize_models()
            self.resume()
            self._audit_resumed_replay()
            self._checkpoint_ready = True
            self._write_effective_config()
            self._run_supervised_warmup()
            self._ensure_initial_best()
            self._ensure_reference_model()
            self._ensure_initial_benchmark()
            if self._state.completed_iterations >= self.config.iterations:
                self._emit(
                    TrainingStage.COMPLETED,
                    "already-complete",
                    completed_iterations=self._state.completed_iterations,
                    iterations=self.config.iterations,
                )
                return
            while self._state.completed_iterations < self.config.iterations:
                self._run_iteration(self._state.completed_iterations + 1)
        except (KeyboardInterrupt, SearchCancelled):
            self.request_stop()
            if self._checkpoint_ready:
                self._safe_checkpoint("interrupted", raise_on_error=False)
            self._emit(
                TrainingStage.STOPPED,
                "stopped",
                level="WARNING",
                iteration=self._state.active_iteration,
            )
        except Exception as exc:
            if self._checkpoint_ready:
                self._safe_checkpoint("error", raise_on_error=False)
            self._emit(
                TrainingStage.ERROR,
                "fatal-error",
                level="ERROR",
                iteration=self._state.active_iteration,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            raise
        finally:
            self._run_lock.release()
            if self._reporter is not None and self._owns_reporter:
                self._reporter.close()

    def resume(self) -> None:
        state_path = self.paths["checkpoints"] / "pipeline_state.json"
        network_path = self.paths["checkpoints"] / "training_state.pt"
        loaded_state, recovered, attempts = recover_pipeline_state(state_path)
        if loaded_state is not None:
            self._state = loaded_state
            if self._state.evaluation_required is None:
                self._state.evaluation_required = (
                    self._state.active_iteration > 0
                    and self._state.active_iteration
                    % self.config.check_frequency
                    == 0
                )
        if recovered:
            self._emit(
                TrainingStage.CHECKPOINT,
                "temporary-state-recovered",
                level="WARNING",
                iteration=self._state.active_iteration,
                active_iteration=self._state.active_iteration,
                selfplay_completed=self._state.selfplay_completed,
                replace_attempts=attempts,
                state=self._state.to_dict(),
            )
        if network_path.exists():
            network_trainer = self._require_network_trainer()
            completed, metadata = network_trainer.load_state(
                network_path,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            if loaded_state is None:
                self._state.completed_iterations = completed
            if metadata.get("stage") == TrainingStage.TRAINING.value:
                self._state.train_steps_completed = max(
                    self._state.train_steps_completed,
                    int(metadata.get("train_step", 0)),
                )
            self._emit(
                TrainingStage.STARTUP,
                "resume",
                iteration=self._state.active_iteration,
                completed_iterations=self._state.completed_iterations,
                current_stage=self._state.stage.value,
                checkpoint=str(network_path),
                state=self._state.to_dict(),
            )
        else:
            self._emit(
                TrainingStage.STARTUP,
                "new-run",
            )

    def request_stop(self) -> None:
        self._stop_event.set()

    def status(self) -> TrainerStatus:
        if self._reporter is None:
            return TrainerStatus(
                stage=self._state.stage,
                iteration=self._state.active_iteration,
                event="created",
                updated_at=time.time(),
                fields=self._state.to_dict(),
            )
        return self._reporter.status()

    def _run_iteration(self, iteration: int) -> None:
        if self._state.active_iteration not in {0, iteration}:
            raise RuntimeError(
                "Pipeline state points to a different active iteration"
            )
        if self._state.active_iteration == 0:
            self._state = PipelineState(
                revision=self._state.revision,
                completed_iterations=self._state.completed_iterations,
                active_iteration=iteration,
                stage=TrainingStage.SELFPLAY,
                evaluation_required=(
                    iteration % self.config.check_frequency == 0
                ),
                best_iteration=self._state.best_iteration,
                consecutive_no_promotion=self._state.consecutive_no_promotion,
            )
            self._persist_state()
        iteration_started = time.monotonic()
        self._emit(
            TrainingStage.SELFPLAY,
            "iteration-start",
            iteration=iteration,
            iterations=self.config.iterations,
        )
        if self._state.stage is TrainingStage.SELFPLAY:
            self._run_selfplay(iteration)
        if self._state.stage is TrainingStage.TRAINING:
            train_metrics = self._run_network_training(iteration)
        else:
            train_metrics = None
        evaluation: EvaluationResult | None = None
        promoted = False
        evaluation_due = bool(self._state.evaluation_required)
        if self._state.stage is TrainingStage.EVALUATION and evaluation_due:
            self._run_tactical_benchmark(
                self._require_current_model(),
                iteration=iteration,
                label="candidate",
            )
            evaluation, promoted = self._run_evaluation(iteration)
            if iteration == self.config.iterations:
                self._run_reference_evaluation(iteration)
        elif self._state.stage is TrainingStage.EVALUATION:
            self._emit(
                TrainingStage.EVALUATION,
                "evaluation-skipped",
                iteration=iteration,
                check_frequency=self.config.check_frequency,
            )
        self._complete_iteration(
            iteration,
            train_metrics,
            evaluation,
            promoted,
            time.monotonic() - iteration_started,
        )

    def _run_selfplay(self, iteration: int) -> None:
        best_model = self._require_best_model()
        evaluator = TorchEvaluator(best_model, device=self.config.device)
        store = ShardStore(self.paths["selfplay"])
        wins = [0, 0]
        draws = 0
        terminal_counts: dict[str, int] = {}
        total_captures = 0
        total_abs_material = 0
        total_abs_value = 0.0
        nonzero_samples = 0
        generated_samples = 0
        stage_started = time.monotonic()
        completed_before = set(self._state.selfplay_completed_games)
        pending_games = [
            index
            for index in range(self.config.selfplay_games)
            if index not in completed_before
        ]
        worker_count = resolve_worker_count(
            self.config.parallel.selfplay_workers,
            len(pending_games),
        )
        self._emit(
            TrainingStage.SELFPLAY,
            "parallel-start",
            iteration=iteration,
            workers=worker_count,
            pending_games=len(pending_games),
            gpu_batch_size=self.config.parallel.gpu_batch_size,
            batch_wait_ms=self.config.parallel.batch_wait_ms,
        )

        def complete_game(
            completed_game: CompletedSelfPlay,
            inference: InferenceMetrics,
        ) -> None:
            nonlocal draws
            nonlocal total_captures, total_abs_material, total_abs_value
            nonlocal generated_samples, nonzero_samples
            game_index = completed_game.game_index
            result = completed_game.result
            entry = None
            if result.samples:
                entry = store.write(
                    list(result.samples),
                    source="selfplay",
                    model_version=f"best-{self._state.best_iteration}",
                    shard_id=(
                        f"iteration-{iteration:06d}-"
                        f"game-{game_index + 1:06d}"
                    ),
                    ruleset_id=self.config.ruleset_id,
                )
            terminal_counts[result.terminal] = (
                terminal_counts.get(result.terminal, 0) + 1
            )
            total_captures += sum(result.adjudication.capture_counts)
            total_abs_material += abs(result.adjudication.material_diff)
            total_abs_value += abs(result.adjudication.black_value)
            generated_samples += len(result.samples)
            nonzero_samples += sum(
                abs(sample.value) > 1e-6 for sample in result.samples
            )
            if result.winner is None:
                draws += 1
            else:
                wins[int(result.winner)] += 1
            self._state.selfplay_completed_games = sorted(
                {*self._state.selfplay_completed_games, game_index}
            )
            self._state.selfplay_completed = len(
                self._state.selfplay_completed_games
            )
            self._persist_state()
            elapsed = time.monotonic() - stage_started
            completed = self._state.selfplay_completed
            completed_here = completed - len(completed_before)
            eta = elapsed / max(completed_here, 1) * (
                self.config.selfplay_games - completed
            )
            replay_samples = sum(
                item.samples
                for item in store.entries()
                if item.ruleset_id == self.config.ruleset_id
            )
            self._emit(
                TrainingStage.SELFPLAY,
                "game-complete",
                iteration=iteration,
                game=completed,
                source_game=game_index + 1,
                games=self.config.selfplay_games,
                moves=len(result.moves),
                winner=None if result.winner is None else int(result.winner),
                terminal=result.terminal,
                black_value=result.adjudication.black_value,
                capture_counts=result.adjudication.capture_counts,
                material_diff=result.adjudication.material_diff,
                seconds=completed_game.elapsed_seconds,
                eta_seconds=eta,
                games_per_hour=completed_here / max(elapsed, 1e-9) * 3600,
                inference_requests=inference.requests,
                inference_batches=inference.batches,
                inference_leaves=inference.leaves,
                gpu_average_batch_size=inference.average_batch_size,
                gpu_inference_seconds=inference.inference_seconds,
                inference_queue_seconds=inference.queue_seconds,
                playouts=result.playouts,
                aggregate_playouts_per_second=(
                    result.playouts
                    / max(completed_game.elapsed_seconds, 1e-9)
                ),
                tree_selection_seconds=result.selection_seconds,
                feature_encoding_seconds=result.encoding_seconds,
                evaluator_wait_seconds=result.evaluator_seconds,
                tree_expansion_seconds=result.expansion_seconds,
                replay_samples=replay_samples,
                shard=None if entry is None else str(entry.path),
            )
            if self._checkpoint_due(
                completed,
                self.config.checkpoint.selfplay_games,
            ):
                self._safe_checkpoint("selfplay-progress")

        run_parallel_selfplay(
            evaluator,
            self.config.search,
            game_indices=pending_games,
            configured_workers=self.config.parallel.selfplay_workers,
            parallel_config=self.config.parallel,
            seed_for_game=lambda game_index: (
                self.config.seed + iteration * 10_000 + game_index
            ),
            initial_sfen_for_game=self.config.start_positions.choose,
            model_version=f"best-{self._state.best_iteration}",
            rules=self.config.adjudication,
            ruleset_id=self.config.ruleset_id,
            temperature_moves=self.config.temperature_moves,
            stop_event=self._stop_event,
            on_complete=complete_game,
        )
        if draws == self.config.selfplay_games:
            self._emit(
                TrainingStage.SELFPLAY,
                "all-draw-warning",
                level="WARNING",
                iteration=iteration,
            )
        completed_games = max(1, sum(terminal_counts.values()))
        terminal_rates = {
            reason: count / completed_games
            for reason, count in terminal_counts.items()
        }
        nonzero_rate = nonzero_samples / max(generated_samples, 1)
        if (
            self.config.name == "smoke"
            and generated_samples
            and nonzero_samples == 0
        ):
            raise RuntimeError(
                "Smoke generated only zero-value samples"
            )
        self._emit(
            TrainingStage.SELFPLAY,
            "selfplay-diagnostics",
            iteration=iteration,
            terminal_counts=terminal_counts,
            terminal_rates=terminal_rates,
            mean_captures=total_captures / completed_games,
            mean_absolute_material_diff=(
                total_abs_material / completed_games
            ),
            mean_absolute_value=total_abs_value / completed_games,
            nonzero_value_rate=nonzero_rate,
            ruleset_id=self.config.ruleset_id,
        )
        self._state.stage = TrainingStage.TRAINING
        self._persist_state()

    def _run_network_training(self, iteration: int) -> TrainMetrics:
        selfplay_store = ShardStore(self.paths["selfplay"])
        replay = load_recent_samples(
            [selfplay_store],
            self.config.replay_capacity,
            ruleset_id=self.config.ruleset_id,
        )
        if self.config.adjudication.mode is AdjudicationMode.REAL_RULES:
            supervised = load_recent_samples(
                [ShardStore(self.paths["datasets"] / "supervised-train")],
                self.config.replay_capacity,
            )
            replay = (replay + supervised)[: self.config.replay_capacity]
        if not replay:
            raise ValueError("Replay buffer is empty after self-play")
        if len(replay) < self.config.batch_size:
            self._emit(
                TrainingStage.TRAINING,
                "small-replay-warning",
                level="WARNING",
                iteration=iteration,
                replay_samples=len(replay),
                batch_size=self.config.batch_size,
            )
        self._emit(
            TrainingStage.TRAINING,
            "training-start",
            iteration=iteration,
            completed_steps=self._state.train_steps_completed,
            train_steps=self.config.train_steps,
            replay_samples=len(replay),
            replay_capacity=self.config.replay_capacity,
            batch_size=self.config.batch_size,
        )
        start_step = self._state.train_steps_completed

        def progress(metrics: TrainStepMetrics) -> None:
            self._state.train_steps_completed = metrics.step
            eta = (
                metrics.elapsed_seconds
                / max(metrics.step - start_step, 1)
                * (metrics.total_steps - metrics.step)
            )
            self._emit(
                TrainingStage.TRAINING,
                "train-step",
                iteration=iteration,
                **asdict(metrics),
                eta_seconds=eta,
            )
            if (
                metrics.predicted_entropy
                < self.config.monitoring.entropy_warning_low
                or metrics.predicted_entropy
                > self.config.monitoring.entropy_warning_high
            ):
                self._emit(
                    TrainingStage.TRAINING,
                    "entropy-warning",
                    level="WARNING",
                    iteration=iteration,
                    entropy=metrics.predicted_entropy,
                )

        def checkpoint(step: int) -> None:
            self._state.train_steps_completed = step
            self._safe_checkpoint("training-progress")

        def step_completed(step: int) -> None:
            self._state.train_steps_completed = step

        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._training_heartbeat,
            args=(heartbeat_stop, iteration),
            name="alphasho-training-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            metrics = self._require_network_trainer().train(
                replay,
                steps=self.config.train_steps,
                batch_size=self.config.batch_size,
                seed=self.config.seed + iteration,
                start_step=start_step,
                log_every_steps=self.config.monitoring.train_log_every_steps,
                progress_callback=progress,
                step_callback=step_completed,
                checkpoint_callback=checkpoint,
                checkpoint_every_steps=self.config.checkpoint.train_steps,
                stop_event=self._stop_event,
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
        self._state.train_steps_completed = self.config.train_steps
        save_inference_checkpoint(
            self.paths["checkpoints"] / "candidate.pt",
            self._require_current_model(),
            search_config=self.config.search.to_dict(),
            metadata={
                "profile": self.config.name,
                "iteration": iteration,
                "ruleset_id": self.config.ruleset_id,
            },
        )
        self._safe_checkpoint("training-complete")
        self._emit(
            TrainingStage.TRAINING,
            "training-complete",
            iteration=iteration,
            **asdict(metrics),
        )
        self._state.stage = TrainingStage.EVALUATION
        self._persist_state()
        return metrics

    def _run_evaluation(
        self,
        iteration: int,
    ) -> tuple[EvaluationResult, bool]:
        self._reconcile_evaluation_progress()
        self._emit(
            TrainingStage.EVALUATION,
            "evaluation-start",
            iteration=iteration,
            completed_games=self._state.evaluation_completed,
            evaluation_games=self.config.evaluation_games,
            promotion_score=self.config.promotion_score,
        )
        existing = {
            int(item["game_index"])
            for item in self._state.evaluation_results
        }
        pending_games = [
            index
            for index in range(self.config.evaluation_games)
            if index not in existing
        ]
        worker_count = resolve_worker_count(
            self.config.parallel.evaluation_workers,
            len(pending_games),
        )
        self._emit(
            TrainingStage.EVALUATION,
            "parallel-start",
            iteration=iteration,
            workers=worker_count,
            pending_games=len(pending_games),
            gpu_batch_size=self.config.parallel.gpu_batch_size,
            batch_wait_ms=self.config.parallel.batch_wait_ms,
        )
        started = time.monotonic()

        def complete_game(
            completed_game: CompletedEvaluation,
            inference: InferenceMetrics,
        ) -> None:
            game = completed_game.result
            adjudication = game.adjudication
            if adjudication.winner is None:
                outcome = "draw"
            elif adjudication.winner is game.candidate_color:
                outcome = "candidate"
            else:
                outcome = "best"
            record = {
                "game_index": game.game_index,
                "outcome": outcome,
                "candidate_color": int(game.candidate_color),
                "winner": (
                    None
                    if adjudication.winner is None
                    else int(adjudication.winner)
                ),
                "terminal": adjudication.reason.value,
                "captures": sum(adjudication.capture_counts),
                "absolute_material_diff": abs(adjudication.material_diff),
            }
            records = {
                int(item["game_index"]): item
                for item in self._state.evaluation_results
            }
            records[game.game_index] = record
            self._state.evaluation_results = [
                records[index] for index in sorted(records)
            ]
            self._refresh_evaluation_totals()
            self._persist_state()
            elapsed = time.monotonic() - started
            self._emit(
                TrainingStage.EVALUATION,
                "evaluation-game",
                iteration=iteration,
                game=self._state.evaluation_completed,
                source_game=game.game_index + 1,
                games=self.config.evaluation_games,
                candidate_color=int(game.candidate_color),
                winner=record["winner"],
                terminal=record["terminal"],
                candidate_wins=self._state.candidate_wins,
                best_wins=self._state.best_wins,
                draws=self._state.draws,
                score=(
                    self._state.candidate_wins + 0.5 * self._state.draws
                )
                / self._state.evaluation_completed,
                elapsed_seconds=elapsed,
                eta_seconds=elapsed
                / max(self._state.evaluation_completed - len(existing), 1)
                * (
                    self.config.evaluation_games
                    - self._state.evaluation_completed
                ),
                seconds=completed_game.elapsed_seconds,
                gpu_average_batch_size=inference.average_batch_size,
                inference_batches=inference.batches,
                inference_leaves=inference.leaves,
            )
            if self._checkpoint_due(
                self._state.evaluation_completed,
                self.config.checkpoint.evaluation_games,
            ):
                self._safe_checkpoint("evaluation-progress")

        run_parallel_evaluation(
            TorchEvaluator(self._require_current_model(), self.config.device),
            TorchEvaluator(self._require_best_model(), self.config.device),
            self.config.search,
            game_indices=pending_games,
            configured_workers=self.config.parallel.evaluation_workers,
            parallel_config=self.config.parallel,
            rules=self.config.adjudication,
            initial_sfen_for_game=lambda game_index: (
                self.config.start_positions.initial_sfens()[
                    (game_index // 2)
                    % len(self.config.start_positions.initial_sfens())
                ]
            ),
            seed_for_game=lambda game_index: (
                self.config.seed + iteration * 1_000_000 + game_index
            ),
            stop_event=self._stop_event,
            on_complete=complete_game,
        )
        result = self._aggregate_evaluation_result()
        promoted = result.score >= self.config.promotion_score
        if promoted:
            self._require_best_model().load_state_dict(
                self._require_current_model().state_dict()
            )
            self._state.best_iteration = iteration
            self._state.consecutive_no_promotion = 0
            self._save_best(iteration)
        else:
            self._state.consecutive_no_promotion += 1
        level = "INFO" if promoted else "WARNING"
        self._emit(
            TrainingStage.EVALUATION,
            "evaluation-complete",
            level=level,
            iteration=iteration,
            candidate_wins=result.candidate_wins,
            best_wins=result.best_wins,
            draws=result.draws,
            score=result.score,
            promotion_score=self.config.promotion_score,
            promoted=promoted,
            termination_counts=dict(result.termination_counts),
            mean_captures=result.mean_captures,
            mean_absolute_material_diff=result.mean_absolute_material_diff,
        )
        if self._state.consecutive_no_promotion >= 3:
            self._emit(
                TrainingStage.EVALUATION,
                "promotion-stall-warning",
                level="WARNING",
                iteration=iteration,
                consecutive_no_promotion=(
                    self._state.consecutive_no_promotion
                ),
            )
        return result, promoted

    def _reconcile_evaluation_progress(self) -> None:
        """Keep resumable results inside the currently configured game range."""

        records = self._state.evaluation_results
        retained = [
            item
            for item in records
            if 0 <= int(item["game_index"]) < self.config.evaluation_games
        ]
        if len(retained) == len(records):
            return
        previous_games = len(records)
        self._state.evaluation_results = sorted(
            retained,
            key=lambda item: int(item["game_index"]),
        )
        self._refresh_evaluation_totals()
        self._persist_state()
        self._emit(
            TrainingStage.EVALUATION,
            "evaluation-progress-trimmed",
            level="WARNING",
            iteration=self._state.active_iteration,
            previous_games=previous_games,
            retained_games=self._state.evaluation_completed,
            evaluation_games=self.config.evaluation_games,
        )

    def _refresh_evaluation_totals(self) -> None:
        records = self._state.evaluation_results
        self._state.evaluation_completed = len(records)
        self._state.candidate_wins = sum(
            item.get("outcome") == "candidate" for item in records
        )
        self._state.best_wins = sum(
            item.get("outcome") == "best" for item in records
        )
        self._state.draws = sum(
            item.get("outcome") == "draw" for item in records
        )

    def _aggregate_evaluation_result(self) -> EvaluationResult:
        self._refresh_evaluation_totals()
        records = self._state.evaluation_results
        termination_counts: dict[str, int] = {}
        for item in records:
            terminal = str(item.get("terminal", "unknown"))
            termination_counts[terminal] = termination_counts.get(terminal, 0) + 1
        games = len(records)
        if games <= 0:
            raise RuntimeError("Evaluation completed without any game result")
        return EvaluationResult(
            self._state.candidate_wins,
            self._state.best_wins,
            self._state.draws,
            games,
            tuple(sorted(termination_counts.items())),
            sum(float(item.get("captures", 0)) for item in records) / games,
            sum(
                float(item.get("absolute_material_diff", 0))
                for item in records
            )
            / games,
        )

    def _complete_iteration(
        self,
        iteration: int,
        train_metrics: TrainMetrics | None,
        evaluation: EvaluationResult | None,
        promoted: bool,
        elapsed: float,
    ) -> None:
        self._state.completed_iterations = iteration
        self._state.active_iteration = 0
        self._state.stage = TrainingStage.STARTUP
        self._state.selfplay_completed = 0
        self._state.selfplay_completed_games = []
        self._state.train_steps_completed = 0
        self._state.evaluation_completed = 0
        self._state.candidate_wins = 0
        self._state.best_wins = 0
        self._state.draws = 0
        self._state.evaluation_results = []
        self._state.evaluation_required = None
        self._persist_state()
        self._safe_checkpoint("iteration-complete")
        self._emit(
            TrainingStage.COMPLETED,
            "iteration-complete",
            iteration=iteration,
            iterations=self.config.iterations,
            seconds=elapsed,
            loss=None if train_metrics is None else train_metrics.loss,
            score=None if evaluation is None else evaluation.score,
            promoted=promoted,
            evaluated=evaluation is not None,
        )

    def _run_supervised_warmup(self) -> None:
        train_store = ShardStore(self.paths["datasets"] / "supervised-train")
        if (
            self._state.completed_iterations > 0
            or not self.config.supervised_paths
            or self.config.supervised_epochs <= 0
            or train_store.entries()
        ):
            return
        self._emit(
            TrainingStage.SUPERVISED,
            "ingest-start",
        )
        validation_store = ShardStore(
            self.paths["datasets"] / "supervised-validation"
        )
        stats = ingest_supervised_corpus(
            self.config.supervised_paths,
            train_store,
            validation_store,
            validation_fraction=self.config.validation_fraction,
            shard_size=self.config.shard_size,
        )
        self._emit(
            TrainingStage.SUPERVISED,
            "ingest-complete",
            files=stats.files,
            games=stats.games,
            samples=stats.samples,
            skipped=stats.skipped,
        )
        samples = load_recent_samples([train_store], self.config.replay_capacity)
        if not samples:
            return
        metrics = self._require_network_trainer().train(
            samples,
            steps=self.config.supervised_epochs
            * max(1, len(samples) // self.config.batch_size),
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            log_every_steps=self.config.monitoring.train_log_every_steps,
            stop_event=self._stop_event,
        )
        self._require_best_model().load_state_dict(
            self._require_current_model().state_dict()
        )
        self._save_best(0)
        self._emit(
            TrainingStage.SUPERVISED,
            "warmup-complete",
            **asdict(metrics),
        )

    def _initialize_models(self) -> None:
        best_path = self.paths["checkpoints"] / "best.pt"
        if best_path.exists():
            best_model, _evaluator, _metadata = load_inference_checkpoint(
                best_path,
                device=self.config.device,
            )
            if best_model.config != self.config.model:
                raise ValueError(
                    "Model architecture is incompatible with the checkpoint: "
                    f"configured={self.config.model.to_dict()}, "
                    f"checkpoint={best_model.config.to_dict()}"
                )
            self._best_model = best_model
        else:
            self._best_model = ShogiPolicyValueNet(self.config.model)
        self._current_model = copy.deepcopy(self._best_model)
        self._network_trainer = PolicyValueTrainer(
            self._current_model,
            device=self.config.device,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        device = resolve_device(self.config.device)
        parameters = sum(
            parameter.numel() for parameter in self._current_model.parameters()
        )
        fields: dict[str, object] = {
            "device": str(device),
            "parameters": parameters,
            "model": self.config.model.to_dict(),
        }
        if device.type == "cuda":
            fields["device_name"] = torch.cuda.get_device_name(device)
            fields["allocated_memory"] = torch.cuda.memory_allocated(device)
        elif self.config.name == "baseline":
            self._emit(
                TrainingStage.STARTUP,
                "cpu-baseline-warning",
                level="WARNING",
                **fields,
            )
        self._emit(
            TrainingStage.STARTUP,
            "model-ready",
            **fields,
        )

    def _audit_persisted_run(self, previous: dict[str, Any] | None) -> None:
        """Reject incompatible persisted data before loading or writing models."""

        expected_model = self.config.model.to_dict()
        if previous is not None:
            if previous.get("ruleset_id") != self.config.ruleset_id:
                self._incompatible(
                    "effective_config ruleset_id does not match the current rules"
                )
            if previous.get("model") != expected_model:
                self._incompatible(
                    "effective_config model structure does not match the current model"
                )

        checkpoint_names = (
            "best.pt",
            "candidate.pt",
            "iteration-0.pt",
            "training_state.pt",
        )
        for name in checkpoint_names:
            path = self.paths["checkpoints"] / name
            if not path.exists():
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model_config = payload.get("model_config")
                metadata = payload.get("metadata", {})
            except Exception as exc:
                self._incompatible(f"cannot inspect {name}: {exc}")
            if model_config != expected_model:
                self._incompatible(f"{name} uses a different model structure")
            if (
                not isinstance(metadata, dict)
                or metadata.get("ruleset_id") != self.config.ruleset_id
            ):
                self._incompatible(f"{name} does not belong to the current ruleset")

        entries = ShardStore(self.paths["selfplay"]).entries()
        state_path = self.paths["checkpoints"] / "pipeline_state.json"
        temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
        if entries and not (state_path.exists() or temporary_state.exists()):
            self._incompatible(
                "replay data exists but pipeline_state.json is missing; "
                "refusing to overwrite iteration 1"
            )
        mismatched = [
            entry.shard_id
            for entry in entries
            if entry.ruleset_id != self.config.ruleset_id
        ]
        if mismatched:
            self._incompatible(
                "replay data belongs to another ruleset: " + ", ".join(mismatched[:3])
            )

    def _audit_resumed_replay(self) -> None:
        entries = ShardStore(self.paths["selfplay"]).entries()
        if not entries or self._state.active_iteration <= 0:
            return
        prefix = f"iteration-{self._state.active_iteration:06d}-game-"
        active_numbers: set[int] = set()
        for entry in entries:
            if entry.shard_id.startswith(prefix):
                try:
                    active_numbers.add(int(entry.shard_id.removeprefix(prefix)))
                except ValueError:
                    self._incompatible(f"invalid replay shard id {entry.shard_id!r}")
        expected = {
            game_index + 1
            for game_index in self._state.selfplay_completed_games
        }
        if active_numbers != expected:
            self._incompatible(
                "pipeline state and replay progress disagree for active iteration "
                f"{self._state.active_iteration}: state={sorted(expected)}, "
                f"replay={sorted(active_numbers)}"
            )

    def _incompatible(self, detail: str) -> None:
        self._emit(
            TrainingStage.ERROR,
            "incompatible-run",
            level="ERROR",
            detail=detail,
        )
        raise IncompatibleRunError(detail)

    def _report_startup(self, previous: dict[str, Any] | None) -> None:
        quick = _quick_summary(self.config)
        self._emit(
            TrainingStage.STARTUP,
            "startup",
            profile=self.config.name,
            quick_settings=quick,
            config_path=str(self.config.source_path),
        )
        self._emit(
            TrainingStage.STARTUP,
            "settings",
            **quick,
        )
        self._emit(
            TrainingStage.STARTUP,
            "schedule",
            **quick,
            model_preset=_model_preset_name(self.config),
        )
        if previous is not None:
            changes = _config_changes(previous, effective_config_dict(self.config))
            if changes:
                self._emit(
                    TrainingStage.STARTUP,
                    "config-changed",
                    level="WARNING",
                    changes=changes,
                    changes_count=len(changes),
                )
        if self.config.monitoring.console_level == "DEBUG":
            self._emit(
                TrainingStage.STARTUP,
                "full-config",
                config_json=json.dumps(
                    effective_config_dict(self.config),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                level="DEBUG",
            )

    def _ensure_initial_best(self) -> None:
        best_path = self.paths["checkpoints"] / "best.pt"
        if not best_path.exists():
            self._save_best(self._state.best_iteration)

    def _ensure_reference_model(self) -> None:
        reference_path = self.paths["checkpoints"] / "iteration-0.pt"
        if reference_path.exists():
            return
        save_inference_checkpoint(
            reference_path,
            self._require_best_model(),
            search_config=self.config.search.to_dict(),
            metadata={
                "profile": self.config.name,
                "iteration": 0,
                "ruleset_id": self.config.ruleset_id,
                "kind": "frozen-reference",
            },
        )
        self._emit(
            TrainingStage.CHECKPOINT,
            "reference-saved",
            checkpoint=str(reference_path),
        )

    def _ensure_initial_benchmark(self) -> None:
        path = self.paths["diagnostics"] / "benchmark-initial.json"
        if path.exists() or self.config.benchmark_path is None:
            return
        result = self._run_tactical_benchmark(
            self._require_best_model(),
            iteration=0,
            label="initial",
        )
        if result is not None:
            write_effective_config(path, asdict(result))

    def _run_tactical_benchmark(
        self,
        model: ShogiPolicyValueNet,
        *,
        iteration: int,
        label: str,
    ) -> TacticalBenchmarkResult | None:
        if self.config.benchmark_path is None:
            return None
        result = evaluate_tactical_benchmark(
            TorchEvaluator(model, self.config.device),
            self.config.benchmark_path,
        )
        initial_path = self.paths["diagnostics"] / "benchmark-initial.json"
        initial_probability = result.capture_probability
        if initial_path.exists():
            initial_values = json.loads(initial_path.read_text(encoding="utf-8"))
            initial_probability = float(
                initial_values.get(
                    "capture_probability",
                    result.capture_probability,
                )
            )
        improvement = result.capture_probability - initial_probability
        passed = (
            result.capture_top1_accuracy >= 0.8
            or improvement >= 0.3
        )
        self._emit(
            TrainingStage.EVALUATION,
            "tactical-benchmark",
            level=(
                "WARNING"
                if (
                    self.config.name == "tactical_overfit"
                    and iteration == self.config.iterations
                    and not passed
                )
                else "INFO"
            ),
            iteration=iteration,
            label=label,
            **asdict(result),
            capture_probability_improvement=improvement,
            diagnostic_passed=passed,
        )
        write_effective_config(
            self.paths["diagnostics"]
            / f"benchmark-{label}-{iteration:06d}.json",
            {
                **asdict(result),
                "capture_probability_improvement": improvement,
                "diagnostic_passed": passed,
            },
        )
        return result

    def _run_reference_evaluation(self, iteration: int) -> None:
        if self.config.reference_games <= 0:
            return
        reference_path = self.paths["checkpoints"] / "iteration-0.pt"
        reference_model, _evaluator, _metadata = load_inference_checkpoint(
            reference_path,
            device=self.config.device,
        )
        self._emit(
            TrainingStage.EVALUATION,
            "reference-evaluation-start",
            iteration=iteration,
            reference_games=self.config.reference_games,
        )

        def progress(event: str, fields: dict[str, object]) -> None:
            if event == "search-heartbeat":
                self._emit(
                    TrainingStage.EVALUATION,
                    "reference-heartbeat",
                    iteration=iteration,
                    **fields,
                )
            elif event == "game-complete":
                self._emit(
                    TrainingStage.EVALUATION,
                    "reference-game",
                    iteration=iteration,
                    **fields,
                )

        result = evaluate_candidate(
            TorchEvaluator(self._require_current_model(), self.config.device),
            TorchEvaluator(reference_model, self.config.device),
            self.config.search,
            games=self.config.reference_games,
            rules=self.config.adjudication,
            initial_sfens=self.config.start_positions.initial_sfens(),
            seed=self.config.seed + 9_000_000,
            progress_callback=progress,
            stop_event=self._stop_event,
            heartbeat_seconds=self.config.monitoring.heartbeat_seconds,
        )
        passed = result.score >= 0.60
        self._emit(
            TrainingStage.EVALUATION,
            "reference-evaluation-complete",
            level="INFO" if passed else "WARNING",
            iteration=iteration,
            score=result.score,
            candidate_wins=result.candidate_wins,
            reference_wins=result.best_wins,
            draws=result.draws,
            diagnostic_passed=passed,
            termination_counts=dict(result.termination_counts),
            mean_captures=result.mean_captures,
            mean_absolute_material_diff=result.mean_absolute_material_diff,
        )

    def _save_best(self, iteration: int) -> None:
        best_path = self.paths["checkpoints"] / "best.pt"
        save_inference_checkpoint(
            best_path,
            self._require_best_model(),
            search_config=self.config.search.to_dict(),
            metadata={
                "profile": self.config.name,
                "iteration": iteration,
                "ruleset_id": self.config.ruleset_id,
            },
        )
        published = (
            self.config.artifact_root.parent
            / "checkpoints"
            / self.config.published_model_name
        )
        _copy_atomic(best_path, published)
        self._emit(
            TrainingStage.CHECKPOINT,
            "best-saved",
            iteration=iteration,
            checkpoint=str(best_path),
            published=str(published),
        )

    def _safe_checkpoint(
        self,
        reason: str,
        *,
        raise_on_error: bool = True,
    ) -> None:
        try:
            if self._network_trainer is not None:
                self._network_trainer.save_state(
                    self.paths["checkpoints"] / "training_state.pt",
                    iteration=self._state.completed_iterations,
                    metadata={
                        "profile": self.config.name,
                        "stage": self._state.stage.value,
                        "active_iteration": self._state.active_iteration,
                        "train_step": self._state.train_steps_completed,
                        "reason": reason,
                        "ruleset_id": self.config.ruleset_id,
                    },
                )
            self._persist_state()
            self._last_checkpoint = time.monotonic()
            self._emit(
                TrainingStage.CHECKPOINT,
                "checkpoint",
                iteration=self._state.active_iteration,
                reason=reason,
                current_stage=self._state.stage.value,
                state=self._state.to_dict(),
            )
        except Exception as exc:
            self._emit(
                TrainingStage.ERROR,
                "checkpoint-failed",
                level="ERROR",
                detail=str(exc),
            )
            if raise_on_error:
                raise RuntimeError("Could not save training checkpoint") from exc

    def _checkpoint_due(self, count: int, every: int) -> bool:
        return (
            count % max(1, every) == 0
            or time.monotonic() - self._last_checkpoint
            >= self.config.checkpoint.interval_seconds
        )

    def _training_heartbeat(
        self,
        stop_event: threading.Event,
        iteration: int,
    ) -> None:
        interval = max(0.1, self.config.monitoring.heartbeat_seconds)
        while not stop_event.wait(interval):
            self._emit(
                TrainingStage.TRAINING,
                "training-heartbeat",
                iteration=iteration,
                step=self._state.train_steps_completed,
                total_steps=self.config.train_steps,
            )

    def _persist_state(self) -> None:
        path = self.paths["checkpoints"] / "pipeline_state.json"
        self._state.schema_version = 4
        self._state.selfplay_completed = len(
            self._state.selfplay_completed_games
        )
        self._refresh_evaluation_totals()
        self._state.revision += 1
        attempts = atomic_write_json(
            path,
            self._state.to_dict(),
        )
        if attempts > 1:
            self._emit(
                TrainingStage.CHECKPOINT,
                "checkpoint-retried",
                level="WARNING",
                iteration=self._state.active_iteration,
                attempts=attempts,
                checkpoint=str(path),
            )

    def _write_effective_config(self) -> None:
        write_effective_config(
            self.config.artifact_root / "effective_config.json",
            effective_config_dict(self.config),
        )

    def _read_effective_config(self) -> dict[str, Any] | None:
        path = self.config.artifact_root / "effective_config.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def stop_handler(_signum, _frame) -> None:  # noqa: ANN001
            self.request_stop()

        for signal_name in ("SIGTERM",):
            signal_value = getattr(signal, signal_name, None)
            if signal_value is not None:
                signal.signal(signal_value, stop_handler)

    def _emit(
        self,
        stage: TrainingStage,
        event: str,
        *,
        message_key: str | None = None,
        level: str = "INFO",
        iteration: int = 0,
        **fields: Any,
    ) -> None:
        if self._reporter is not None:
            self._reporter.emit(
                stage,
                event,
                message_key=message_key,
                level=level,
                iteration=iteration,
                **fields,
            )

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise SearchCancelled("Training was stopped")

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            "selfplay": self.config.artifact_root / "selfplay",
            "datasets": self.config.artifact_root / "datasets",
            "checkpoints": self.config.artifact_root / "checkpoints",
            "logs": self.config.artifact_root / "logs",
            "diagnostics": self.config.artifact_root / "diagnostics",
        }

    def _require_best_model(self) -> ShogiPolicyValueNet:
        if self._best_model is None:
            raise RuntimeError("Best model has not been initialized")
        return self._best_model

    def _require_current_model(self) -> ShogiPolicyValueNet:
        if self._current_model is None:
            raise RuntimeError("Current model has not been initialized")
        return self._current_model

    def _require_network_trainer(self) -> PolicyValueTrainer:
        if self._network_trainer is None:
            raise RuntimeError("Network trainer has not been initialized")
        return self._network_trainer


def _quick_summary(config: TrainingConfig) -> dict[str, Any]:
    return {
        "profile": config.name,
        "device": config.device,
        "iterations": config.iterations,
        "start_position_mode": config.start_positions.mode.value,
        "adjudication_mode": config.adjudication.mode.value,
        "max_moves": config.max_moves,
        "capture_threshold": config.adjudication.capture_threshold,
        "ruleset_id": config.ruleset_id,
        "playouts": config.search.n_playouts,
        "c_puct": config.search.c_puct,
        "inference_batch_size": config.search.batch_size,
        "games_per_iteration": config.selfplay_games,
        "dirichlet_alpha": config.search.dirichlet_alpha,
        "dirichlet_weight": config.search.dirichlet_epsilon,
        "training_batch_size": config.batch_size,
        "replay_buffer_size": config.replay_capacity,
        "train_steps_per_iteration": config.train_steps,
        "learning_rate": config.learning_rate,
        "check_frequency": config.check_frequency,
        "evaluation_games": config.evaluation_games,
        "promotion_score": config.promotion_score,
    }


def _model_preset_name(config: TrainingConfig) -> str:
    for name, preset in MODEL_PRESETS.items():
        if config.model == preset:
            return name
    return "custom"


def _config_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    before = _flatten(previous)
    after = _flatten(current)
    ignored = {"source_path"}
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if key not in ignored and before.get(key) != after.get(key)
    }


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def recover_pipeline_state(
    path: str | Path,
) -> tuple[PipelineState | None, bool, int]:
    """Load and, when necessary, commit the newest valid pipeline state."""

    def load(candidate: Path) -> PipelineState:
        values = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Pipeline state must be a JSON object")
        return PipelineState.from_dict(values)

    return recover_newer_temporary(
        path,
        validator=load,
        revision=lambda state: state.revision,
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    atomic_copy(source, destination)
