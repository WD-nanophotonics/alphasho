"""Versioned TOML configuration for AlphaSho training."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from alphasho.ai import ModelConfig, SearchConfig
from alphasho.training.adjudication import (
    AdjudicationMode,
    AdjudicationRules,
)
from alphasho.training.messages import TrainingLanguage
from alphasho.training.positions import (
    EvaluationPosition,
    StartPositionConfig,
    TrainingStartMode,
    evaluation_suite_id,
    load_evaluation_positions,
    ruleset_id,
)


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    console_level: str = "INFO"
    output_language: TrainingLanguage = TrainingLanguage.ENGLISH
    heartbeat_seconds: float = 10.0
    train_log_every_steps: int = 10
    log_file_max_mb: int = 20
    log_backups: int = 5
    entropy_warning_low: float = 0.25
    entropy_warning_high: float = 7.6


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    interval_seconds: float = 120.0
    selfplay_games: int = 1
    train_steps: int = 25
    evaluation_games: int = 2


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    selfplay_workers: int = 0
    evaluation_workers: int = 0
    gpu_batch_size: int = 128
    batch_wait_ms: float = 2.0
    request_queue_size: int = 64


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    name: str
    seed: int
    device: str
    iterations: int
    model: ModelConfig
    search: SearchConfig
    artifact_root: Path
    supervised_paths: tuple[Path, ...]
    supervised_epochs: int
    validation_fraction: float
    selfplay_games: int
    adjudication: AdjudicationRules
    start_positions: StartPositionConfig
    temperature_moves: int
    replay_capacity: int
    shard_size: int
    batch_size: int
    train_steps: int
    learning_rate: float
    weight_decay: float
    evaluation_games: int
    evaluation_positions_path: Path | None
    promotion_score: float
    check_frequency: int
    monitoring: MonitoringConfig
    checkpoint: CheckpointConfig
    parallel: ParallelConfig
    published_model_name: str
    benchmark_path: Path | None
    reference_games: int
    source_path: Path

    @property
    def max_moves(self) -> int:
        return self.adjudication.max_moves

    @property
    def ruleset_id(self) -> str:
        return ruleset_id(self.adjudication, self.start_positions)

    @property
    def evaluation_positions(self) -> tuple[EvaluationPosition, ...]:
        if self.evaluation_positions_path is None:
            return ()
        return load_evaluation_positions(self.evaluation_positions_path)

    @property
    def evaluation_suite_id(self) -> str:
        positions = self.evaluation_positions
        return "standard" if not positions else evaluation_suite_id(positions)

    def evaluation_position_for_game(
        self,
        game_index: int,
    ) -> EvaluationPosition | None:
        positions = self.evaluation_positions
        if not positions:
            return None
        return positions[(game_index // 2) % len(positions)]


MODEL_PRESETS: dict[str, ModelConfig] = {
    "smoke": ModelConfig(32, 2, 64),
    "baseline": ModelConfig(128, 10, 256),
    "large": ModelConfig(256, 20, 256),
}


@dataclass(frozen=True, slots=True)
class QuickTrainingSettings:
    profile: str = "cpu_debug"
    device: str = "cpu"
    output_language: str = "en"
    iterations: int = 20
    model_preset: str = "smoke"
    start_position_mode: str = "standard"
    adjudication_mode: str = "capture_material"
    max_moves: int = 64
    capture_threshold: int = 3
    playouts: int = 16
    c_puct: float = 1.5
    inference_batch_size: int = 8
    games_per_iteration: int = 8
    selfplay_workers: int = 0
    evaluation_workers: int = 0
    gpu_batch_size: int = 128
    gpu_batch_wait_ms: float = 2.0
    dirichlet_alpha: float = 0.15
    dirichlet_weight: float = 0.25
    training_batch_size: int = 64
    replay_buffer_size: int = 50_000
    train_steps_per_iteration: int = 32
    learning_rate: float = 0.001
    check_frequency: int = 2
    evaluation_games: int = 20
    promotion_score: float = 0.55

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> QuickTrainingSettings:
        names = {
            "profile": "PROFILE",
            "device": "DEVICE",
            "output_language": "OUTPUT_LANGUAGE",
            "iterations": "ITERATIONS",
            "model_preset": "MODEL_PRESET",
            "start_position_mode": "START_POSITION_MODE",
            "adjudication_mode": "ADJUDICATION_MODE",
            "max_moves": "MAX_MOVES",
            "capture_threshold": "CAPTURE_THRESHOLD",
            "playouts": "PLAYOUTS",
            "c_puct": "C_PUCT",
            "inference_batch_size": "INFERENCE_BATCH_SIZE",
            "games_per_iteration": "GAMES_PER_ITERATION",
            "selfplay_workers": "SELFPLAY_WORKERS",
            "evaluation_workers": "EVALUATION_WORKERS",
            "gpu_batch_size": "GPU_BATCH_SIZE",
            "gpu_batch_wait_ms": "GPU_BATCH_WAIT_MS",
            "dirichlet_alpha": "DIRICHLET_ALPHA",
            "dirichlet_weight": "DIRICHLET_WEIGHT",
            "training_batch_size": "TRAINING_BATCH_SIZE",
            "replay_buffer_size": "REPLAY_BUFFER_SIZE",
            "train_steps_per_iteration": "TRAIN_STEPS_PER_ITERATION",
            "learning_rate": "LEARNING_RATE",
            "check_frequency": "CHECK_FREQUENCY",
            "evaluation_games": "EVALUATION_GAMES",
            "promotion_score": "PROMOTION_SCORE",
        }
        missing = [constant for constant in names.values() if constant not in values]
        if missing:
            raise ValueError(
                "Missing quick training constants: " + ", ".join(missing)
            )
        return cls(**{field: values[constant] for field, constant in names.items()})

    def build_config(self) -> TrainingConfig:
        root = Path(__file__).resolve().parents[3]
        profile_path = root / "configs" / "training" / f"{self.profile}.toml"
        if not profile_path.exists():
            raise ValueError(f"profile: unknown training profile {self.profile!r}")
        base = load_training_config(profile_path)
        model = MODEL_PRESETS.get(self.model_preset)
        if model is None:
            choices = ", ".join(MODEL_PRESETS)
            raise ValueError(
                f"model_preset: {self.model_preset!r} is invalid; choose {choices}"
            )
        try:
            start_mode = TrainingStartMode(self.start_position_mode)
        except ValueError as exc:
            raise ValueError(
                "start_position_mode: choose standard or tactical"
            ) from exc
        try:
            adjudication_mode = AdjudicationMode(self.adjudication_mode)
        except ValueError as exc:
            raise ValueError(
                "adjudication_mode: choose real_rules or capture_material"
            ) from exc
        start_path = base.start_positions.path
        if start_mode is TrainingStartMode.TACTICAL and start_path is None:
            start_path = root / "configs" / "training" / "tactical_positions.jsonl"
        config = replace(
            base,
            name=self.profile,
            device=self.device,
            monitoring=replace(
                base.monitoring,
                output_language=TrainingLanguage.parse(self.output_language),
            ),
            iterations=self.iterations,
            model=model,
            adjudication=replace(
                base.adjudication,
                mode=adjudication_mode,
                max_moves=self.max_moves,
                capture_threshold=self.capture_threshold,
            ),
            start_positions=replace(
                base.start_positions,
                mode=start_mode,
                path=start_path,
            ),
            search=replace(
                base.search,
                n_playouts=self.playouts,
                c_puct=self.c_puct,
                batch_size=self.inference_batch_size,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_weight,
            ),
            parallel=replace(
                base.parallel,
                selfplay_workers=self.selfplay_workers,
                evaluation_workers=self.evaluation_workers,
                gpu_batch_size=self.gpu_batch_size,
                batch_wait_ms=self.gpu_batch_wait_ms,
            ),
            selfplay_games=self.games_per_iteration,
            replay_capacity=self.replay_buffer_size,
            batch_size=self.training_batch_size,
            train_steps=self.train_steps_per_iteration,
            learning_rate=self.learning_rate,
            check_frequency=self.check_frequency,
            evaluation_games=self.evaluation_games,
            promotion_score=self.promotion_score,
            published_model_name=(
                "best.pt"
                if self.profile == "baseline"
                and adjudication_mode is AdjudicationMode.REAL_RULES
                else "debug-best.pt"
            ),
        )
        validate_training_config(config)
        return config


def load_training_config(path: str | Path) -> TrainingConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    project_root = Path(__file__).resolve().parents[3]
    run = _section(values, "run")
    data = _section(values, "data")
    training = _section(values, "training")
    evaluation = _section(values, "evaluation")
    monitoring = _section(values, "monitoring")
    checkpoint = _section(values, "checkpoint")
    parallel = _section(values, "parallel")
    adjudication = _section(values, "adjudication")
    starts = _section(values, "start_positions")
    publication = _section(values, "publication")
    diagnostics = _section(values, "diagnostics")
    raw_paths = data.get("supervised_paths", [])
    if not isinstance(raw_paths, list):
        raise ValueError("data.supervised_paths must be a list")
    supervised_paths = tuple(
        _resolve_path(project_root, str(item)) for item in raw_paths
    )
    config = TrainingConfig(
        name=str(run.get("name", source.stem)),
        seed=int(run.get("seed", 20260720)),
        device=str(run.get("device", "auto")),
        iterations=int(run.get("iterations", 1)),
        model=ModelConfig.from_dict(_section(values, "model")),
        search=SearchConfig.from_dict(_section(values, "search")),
        artifact_root=_resolve_path(
            project_root, str(run.get("artifact_root", "artifacts"))
        ),
        supervised_paths=supervised_paths,
        supervised_epochs=int(data.get("supervised_epochs", 0)),
        validation_fraction=float(data.get("validation_fraction", 0.1)),
        selfplay_games=int(data.get("selfplay_games", 1)),
        adjudication=AdjudicationRules(
            mode=AdjudicationMode(
                adjudication.get("mode", "real_rules")
            ),
            max_moves=int(
                adjudication.get(
                    "max_moves",
                    data.get("max_moves", 512),
                )
            ),
            capture_threshold=int(
                adjudication.get("capture_threshold", 0)
            ),
            material_scale=float(
                adjudication.get("material_scale", 12.0)
            ),
        ),
        start_positions=StartPositionConfig(
            mode=TrainingStartMode(starts.get("mode", "standard")),
            path=(
                _resolve_path(project_root, str(starts["path"]))
                if starts.get("path")
                else None
            ),
        ),
        temperature_moves=int(data.get("temperature_moves", 30)),
        replay_capacity=int(data.get("replay_capacity", 500_000)),
        shard_size=int(data.get("shard_size", 512)),
        batch_size=int(training.get("batch_size", 256)),
        train_steps=int(training.get("train_steps", 32)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        evaluation_games=int(evaluation.get("games", 20)),
        evaluation_positions_path=(
            _resolve_path(project_root, str(evaluation["positions_path"]))
            if evaluation.get("positions_path")
            else None
        ),
        promotion_score=float(evaluation.get("promotion_score", 0.55)),
        check_frequency=int(evaluation.get("check_frequency", 1)),
        monitoring=MonitoringConfig(
            console_level=str(monitoring.get("console_level", "INFO")).upper(),
            output_language=TrainingLanguage.parse(
                str(monitoring.get("output_language", "en"))
            ),
            heartbeat_seconds=float(monitoring.get("heartbeat_seconds", 10.0)),
            train_log_every_steps=int(
                monitoring.get("train_log_every_steps", 10)
            ),
            log_file_max_mb=int(monitoring.get("log_file_max_mb", 20)),
            log_backups=int(monitoring.get("log_backups", 5)),
            entropy_warning_low=float(
                monitoring.get("entropy_warning_low", 0.25)
            ),
            entropy_warning_high=float(
                monitoring.get("entropy_warning_high", 7.6)
            ),
        ),
        checkpoint=CheckpointConfig(
            interval_seconds=float(checkpoint.get("interval_seconds", 120.0)),
            selfplay_games=int(checkpoint.get("selfplay_games", 1)),
            train_steps=int(checkpoint.get("train_steps", 25)),
            evaluation_games=int(checkpoint.get("evaluation_games", 2)),
        ),
        parallel=ParallelConfig(
            selfplay_workers=int(parallel.get("selfplay_workers", 0)),
            evaluation_workers=int(parallel.get("evaluation_workers", 0)),
            gpu_batch_size=int(parallel.get("gpu_batch_size", 128)),
            batch_wait_ms=float(parallel.get("batch_wait_ms", 2.0)),
            request_queue_size=int(parallel.get("request_queue_size", 64)),
        ),
        published_model_name=str(
            publication.get("model_name", "best.pt")
        ),
        benchmark_path=(
            _resolve_path(project_root, str(diagnostics["benchmark_path"]))
            if diagnostics.get("benchmark_path")
            else None
        ),
        reference_games=int(diagnostics.get("reference_games", 0)),
        source_path=source,
    )
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    errors: list[str] = []
    positive_integers = {
        "iterations": config.iterations,
        "playouts": config.search.n_playouts,
        "inference_batch_size": config.search.batch_size,
        "games_per_iteration": config.selfplay_games,
        "training_batch_size": config.batch_size,
        "replay_buffer_size": config.replay_capacity,
        "train_steps_per_iteration": config.train_steps,
        "check_frequency": config.check_frequency,
        "evaluation_games": config.evaluation_games,
        "max_moves": config.max_moves,
        "shard_size": config.shard_size,
        "monitoring.train_log_every_steps": (
            config.monitoring.train_log_every_steps
        ),
        "monitoring.log_file_max_mb": config.monitoring.log_file_max_mb,
        "monitoring.log_backups": config.monitoring.log_backups,
        "checkpoint.selfplay_games": config.checkpoint.selfplay_games,
        "checkpoint.train_steps": config.checkpoint.train_steps,
        "checkpoint.evaluation_games": config.checkpoint.evaluation_games,
        "parallel.gpu_batch_size": config.parallel.gpu_batch_size,
        "parallel.request_queue_size": config.parallel.request_queue_size,
    }
    errors.extend(
        f"{name}: expected a positive integer, got {value}"
        for name, value in positive_integers.items()
        if value <= 0
    )
    if config.parallel.selfplay_workers < 0:
        errors.append("parallel.selfplay_workers: expected 0 or a positive integer")
    if config.parallel.evaluation_workers < 0:
        errors.append("parallel.evaluation_workers: expected 0 or a positive integer")
    if config.parallel.gpu_batch_size < config.search.batch_size:
        errors.append(
            "parallel.gpu_batch_size: must be at least search.batch_size "
            f"({config.search.batch_size})"
        )
    if config.parallel.batch_wait_ms < 0:
        errors.append("parallel.batch_wait_ms: expected a non-negative value")
    if config.replay_capacity < config.batch_size:
        errors.append(
            "replay_buffer_size: must be at least training_batch_size "
            f"({config.batch_size}), got {config.replay_capacity}"
        )
    if not 0 <= config.validation_fraction < 1:
        errors.append(
            "validation_fraction: expected a value in [0, 1), "
            f"got {config.validation_fraction}"
        )
    if config.evaluation_games % 2:
        errors.append(
            "evaluation_games: must be even for paired colors, "
            f"got {config.evaluation_games}"
        )
    if config.evaluation_positions_path is not None:
        if not config.evaluation_positions_path.is_file():
            errors.append(
                "evaluation.positions_path: file not found: "
                f"{config.evaluation_positions_path}"
            )
        else:
            try:
                position_count = len(config.evaluation_positions)
            except (OSError, ValueError) as exc:
                errors.append(f"evaluation.positions_path: {exc}")
            else:
                expected_games = position_count * 2
                if config.evaluation_games != expected_games:
                    errors.append(
                        "evaluation_games: must equal two games per evaluation "
                        f"position ({expected_games}), got "
                        f"{config.evaluation_games}"
                    )
    if config.supervised_epochs < 0:
        errors.append(
            "supervised_epochs: expected a non-negative integer, "
            f"got {config.supervised_epochs}"
        )
    if config.temperature_moves < 0:
        errors.append(
            "temperature_moves: expected a non-negative integer, "
            f"got {config.temperature_moves}"
        )
    if config.temperature_moves > config.max_moves:
        errors.append(
            "temperature_moves: must not exceed max_moves "
            f"({config.max_moves}), got {config.temperature_moves}"
        )
    if (
        config.adjudication.mode is AdjudicationMode.CAPTURE_MATERIAL
        and config.adjudication.capture_threshold <= 0
    ):
        errors.append(
            "capture_threshold: expected a positive integer in "
            "capture_material mode"
        )
    if config.adjudication.capture_threshold < 0:
        errors.append(
            "capture_threshold: expected a non-negative integer, "
            f"got {config.adjudication.capture_threshold}"
        )
    positive_ranges = {
        "dirichlet_alpha": config.search.dirichlet_alpha,
        "learning_rate": config.learning_rate,
        "c_puct": config.search.c_puct,
        "monitoring.heartbeat_seconds": config.monitoring.heartbeat_seconds,
        "checkpoint.interval_seconds": config.checkpoint.interval_seconds,
        "material_scale": config.adjudication.material_scale,
    }
    for name, value in positive_ranges.items():
        if value <= 0:
            errors.append(
                f"{name}: expected a value greater than 0, got {value}"
            )
    unit_ranges = {
        "dirichlet_weight": config.search.dirichlet_epsilon,
        "promotion_score": config.promotion_score,
    }
    for name, value in unit_ranges.items():
        if not 0 <= value <= 1:
            errors.append(
                f"{name}: expected a value in [0, 1], got {value}"
            )
    if config.weight_decay < 0:
        errors.append(
            f"weight_decay: expected a non-negative value, got {config.weight_decay}"
        )
    if config.search.virtual_loss < 0:
        errors.append(
            "search.virtual_loss: expected a non-negative value, "
            f"got {config.search.virtual_loss}"
        )
    if not (
        0
        <= config.monitoring.entropy_warning_low
        < config.monitoring.entropy_warning_high
    ):
        errors.append(
            "monitoring entropy range: expected 0 <= low < high, got "
            f"{config.monitoring.entropy_warning_low} and "
            f"{config.monitoring.entropy_warning_high}"
        )
    if config.model.channels <= 0:
        errors.append(
            f"model.channels: expected a positive integer, got {config.model.channels}"
        )
    if config.model.residual_blocks < 0:
        errors.append(
            "model.residual_blocks: expected a non-negative integer, "
            f"got {config.model.residual_blocks}"
        )
    if config.model.value_hidden <= 0:
        errors.append(
            "model.value_hidden: expected a positive integer, "
            f"got {config.model.value_hidden}"
        )
    if config.start_positions.mode is TrainingStartMode.TACTICAL:
        if (
            config.start_positions.path is None
            or not config.start_positions.path.is_file()
        ):
            errors.append(
                "start_positions.path: tactical mode requires an existing JSONL file"
            )
    if config.reference_games < 0 or config.reference_games % 2:
        errors.append(
            "diagnostics.reference_games: expected a non-negative even integer, "
            f"got {config.reference_games}"
        )
    if (
        config.benchmark_path is not None
        and not config.benchmark_path.is_file()
    ):
        errors.append(
            f"diagnostics.benchmark_path: file not found: {config.benchmark_path}"
        )
    if (
        not config.published_model_name.endswith(".pt")
        or Path(config.published_model_name).name
        != config.published_model_name
    ):
        errors.append(
            "publication.model_name: expected a simple .pt filename, "
            f"got {config.published_model_name!r}"
        )
    if config.published_model_name == "best.pt" and (
        config.name != "baseline"
        or config.adjudication.mode is not AdjudicationMode.REAL_RULES
    ):
        errors.append(
            "publication.model_name: only the real-rules baseline profile "
            "may publish best.pt"
        )
    if config.monitoring.console_level not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    }:
        errors.append(
            "monitoring.console_level: choose DEBUG, INFO, WARNING, or ERROR"
        )
    if errors:
        raise ValueError("Invalid training configuration:\n- " + "\n- ".join(errors))


def effective_config_dict(config: TrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    values["artifact_root"] = str(config.artifact_root)
    values["supervised_paths"] = [str(path) for path in config.supervised_paths]
    values["start_positions"]["path"] = (
        None
        if config.start_positions.path is None
        else str(config.start_positions.path)
    )
    values["benchmark_path"] = (
        None if config.benchmark_path is None else str(config.benchmark_path)
    )
    values["evaluation_positions_path"] = (
        None
        if config.evaluation_positions_path is None
        else str(config.evaluation_positions_path)
    )
    values["evaluation_suite_id"] = config.evaluation_suite_id
    values["ruleset_id"] = config.ruleset_id
    values["source_path"] = str(config.source_path)
    return values


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    section = values.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a TOML table")
    return section


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
