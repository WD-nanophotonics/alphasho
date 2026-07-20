"""KIF/CSA corpus ingestion for optional supervised warm-up."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alphasho.ai import encode_position
from alphasho.domain import Color
from alphasho.engine import RecordError, SearchPosition, load_record
from alphasho.training.data import ShardStore, TrainingSample


@dataclass(frozen=True, slots=True)
class CorpusStats:
    files: int = 0
    games: int = 0
    samples: int = 0
    skipped: int = 0


def ingest_supervised_corpus(
    paths: tuple[Path, ...],
    train_store: ShardStore,
    validation_store: ShardStore,
    *,
    validation_fraction: float,
    shard_size: int,
) -> CorpusStats:
    files = _record_files(paths)
    train_pending: list[TrainingSample] = []
    validation_pending: list[TrainingSample] = []
    games = samples = skipped = 0
    for path in files:
        try:
            record = load_record(path)
            if not record.outcome.is_over:
                skipped += 1
                continue
            game_samples = _samples_from_record(path, record)
        except (OSError, UnicodeError, RecordError, RuntimeError, ValueError):
            skipped += 1
            continue
        target = (
            validation_pending
            if _is_validation(path, validation_fraction)
            else train_pending
        )
        target.extend(game_samples)
        games += 1
        samples += len(game_samples)
        if len(train_pending) >= shard_size:
            _flush(train_store, train_pending, "supervised-train", shard_size)
        if len(validation_pending) >= shard_size:
            _flush(
                validation_store,
                validation_pending,
                "supervised-validation",
                shard_size,
            )
    _flush(train_store, train_pending, "supervised-train", shard_size, all_items=True)
    _flush(
        validation_store,
        validation_pending,
        "supervised-validation",
        shard_size,
        all_items=True,
    )
    return CorpusStats(len(files), games, samples, skipped)


def _samples_from_record(path: Path, record) -> list[TrainingSample]:  # noqa: ANN001
    position = SearchPosition(record.initial_sfen)
    pending: list[tuple[TrainingSample, Color]] = []
    for ply, usi in enumerate(record.moves):
        action = next(
            (candidate for candidate in position.legal_actions() if candidate.usi == usi),
            None,
        )
        if action is None:
            raise ValueError(f"Illegal move at ply {ply + 1}: {usi}")
        encoded = encode_position(position)
        pending.append(
            (
                TrainingSample(
                    encoded.features1,
                    encoded.features2,
                    np.asarray([action.label], dtype=np.uint16),
                    np.asarray([1.0], dtype=np.float16),
                    0,
                    {
                        "source": str(path),
                        "ply": ply,
                        "usi": usi,
                        "kind": "supervised",
                    },
                ),
                position.turn,
            )
        )
        position.push(action)
    winner = record.outcome.winner
    result: list[TrainingSample] = []
    for sample, turn in pending:
        sample.value = 0 if winner is None else (1 if winner is turn else -1)
        result.append(sample)
    return result


def _record_files(paths: tuple[Path, ...]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".kif", ".kifu", ".csa"}:
            result.append(path)
        elif path.is_dir():
            result.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".kif", ".kifu", ".csa"}
            )
    return sorted(set(result))


def _is_validation(path: Path, fraction: float) -> bool:
    if fraction <= 0:
        return False
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < fraction


def _flush(
    store: ShardStore,
    pending: list[TrainingSample],
    source: str,
    shard_size: int,
    *,
    all_items: bool = False,
) -> None:
    while pending and (all_items or len(pending) >= shard_size):
        count = min(shard_size, len(pending))
        store.write(pending[:count], source=source, model_version="human-record")
        del pending[:count]
