"""Immediate console, rotating text, JSONL, and status reporting."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from alphasho.atomic_io import atomic_write_json
from alphasho.training.config import MonitoringConfig
from alphasho.training.messages import (
    TrainingLanguage,
    default_message_key,
    render_training_message,
)

REPORTER_MESSAGE_FIELDS = frozenset(
    {
        "iteration",
        "stage",
        "event",
        "level",
        "message_key",
        "language",
    }
)


class TrainingStage(StrEnum):
    STARTUP = "startup"
    SUPERVISED = "supervised"
    SELFPLAY = "selfplay"
    TRAINING = "training"
    EVALUATION = "evaluation"
    CHECKPOINT = "checkpoint"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    timestamp: float
    level: str
    stage: TrainingStage
    event: str
    message_key: str
    language: TrainingLanguage
    message: str
    iteration: int = 0
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainerStatus:
    stage: TrainingStage = TrainingStage.STARTUP
    iteration: int = 0
    event: str = "created"
    message_key: str = ""
    language: TrainingLanguage = TrainingLanguage.ENGLISH
    message: str = ""
    updated_at: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)


class TrainingReporter:
    def __init__(
        self,
        log_directory: str | Path,
        profile: str,
        config: MonitoringConfig,
    ) -> None:
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.config = config
        self.language = TrainingLanguage.parse(config.output_language)
        self.metrics_path = self.log_directory / f"{profile}.jsonl"
        self.status_path = self.log_directory / "status.json"
        self._lock = threading.RLock()
        self._heartbeats: dict[str, float] = {}
        self._status = TrainerStatus(updated_at=time.time())
        self._logger = logging.getLogger(f"alphasho.training.{profile}.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        formatter = logging.Formatter("%(message)s")

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, config.console_level))
        console.setFormatter(formatter)
        self._logger.addHandler(console)

        text_file = RotatingFileHandler(
            self.log_directory / f"training.{self.language.value}.log",
            maxBytes=config.log_file_max_mb * 1024 * 1024,
            backupCount=config.log_backups,
            encoding="utf-8",
        )
        text_file.setLevel(logging.DEBUG)
        text_file.setFormatter(formatter)
        self._logger.addHandler(text_file)

    def emit(
        self,
        stage: TrainingStage,
        event: str,
        *,
        message_key: str | None = None,
        level: str = "INFO",
        iteration: int = 0,
        **fields: Any,
    ) -> TrainingEvent:
        now = time.time()
        normalized = _json_safe(fields)
        normalized_level = level.upper()
        resolved_key = message_key or default_message_key(stage.value, event)
        reporter_fields = {
            "iteration": iteration,
            "stage": stage.value,
            "event": event,
            "level": normalized_level,
            "message_key": resolved_key,
            "language": self.language.value,
        }
        if frozenset(reporter_fields) != REPORTER_MESSAGE_FIELDS:
            raise RuntimeError("Training Reporter message field contract changed")
        render_context = {**normalized, **reporter_fields}
        message = render_training_message(
            resolved_key,
            self.language,
            render_context,
        )
        training_event = TrainingEvent(
            now,
            normalized_level,
            stage,
            event,
            resolved_key,
            self.language,
            message,
            iteration,
            normalized,
        )
        prefix = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        iteration_text = f"[iter {iteration}]" if iteration else ""
        line = (
            f"[{prefix}][{training_event.level}][{stage.value.upper()}]"
            f"{iteration_text} {message}"
        )
        with self._lock:
            self._logger.log(
                getattr(logging, training_event.level, logging.INFO),
                line,
            )
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "time": now,
                            "level": training_event.level,
                            "stage": stage.value,
                            "event": event,
                            "iteration": iteration,
                            "message_key": resolved_key,
                            "language": self.language.value,
                            "message": message,
                            "fields": normalized,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
            self._status = TrainerStatus(
                stage,
                iteration,
                event,
                resolved_key,
                self.language,
                message,
                now,
                normalized,
            )
            _atomic_json(self.status_path, asdict(self._status))
        return training_event

    def should_heartbeat(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            previous = self._heartbeats.get(key, 0.0)
            if now - previous < self.config.heartbeat_seconds:
                return False
            self._heartbeats[key] = now
            return True

    def status(self) -> TrainerStatus:
        with self._lock:
            return self._status

    def close(self) -> None:
        with self._lock:
            handlers = tuple(self._logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                self._logger.removeHandler(handler)


def write_effective_config(path: str | Path, values: dict[str, Any]) -> None:
    _atomic_json(Path(path), values)


def _atomic_json(path: Path, values: Any) -> None:
    atomic_write_json(path, _json_safe(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, StrEnum)):
        return str(value)
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return str(value)
    return value
