"""MCTS self-play generation."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from alphasho.ai import (
    MCTS,
    SearchCancelled,
    SearchConfig,
    SearchProgress,
    encode_position,
)
from alphasho.ai.mcts import BatchEvaluator
from alphasho.domain import Color
from alphasho.training.adjudication import (
    AdjudicationResult,
    AdjudicationRules,
    TrainingPosition,
)
from alphasho.training.data import TrainingSample


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    samples: tuple[TrainingSample, ...]
    winner: Color | None
    moves: tuple[str, ...]
    terminal: str
    adjudication: AdjudicationResult
    playouts: int = 0
    selection_seconds: float = 0.0
    encoding_seconds: float = 0.0
    evaluator_seconds: float = 0.0
    expansion_seconds: float = 0.0


def play_self_game(
    evaluator: BatchEvaluator,
    search_config: SearchConfig,
    *,
    seed: int,
    model_version: str,
    rules: AdjudicationRules,
    initial_sfen: str | None,
    ruleset_id: str,
    temperature_moves: int,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
    stop_event: threading.Event | None = None,
    heartbeat_seconds: float = 10.0,
) -> SelfPlayResult:
    position = TrainingPosition(rules, initial_sfen)
    search = MCTS(evaluator, search_config, seed=seed)
    rng = np.random.default_rng(seed)
    pending: list[tuple[TrainingSample, Color]] = []
    moves: list[str] = []
    playouts = 0
    selection_seconds = 0.0
    encoding_seconds = 0.0
    evaluator_seconds = 0.0
    expansion_seconds = 0.0
    ply = 0
    while position.result() is None:
        if stop_event is not None and stop_event.is_set():
            raise SearchCancelled("Self-play was stopped")
        encoded = encode_position(position)
        temperature = 1.0 if ply < temperature_moves else 1e-3
        result = search.search(
            position,
            temperature=temperature,
            add_root_noise=True,
            cancel_event=stop_event,
            progress_callback=(
                (
                    lambda search_progress, current_ply=ply: progress_callback(
                        "search-heartbeat",
                        _search_fields(current_ply, search_progress),
                    )
                )
                if progress_callback is not None
                else None
            ),
            heartbeat_seconds=heartbeat_seconds,
        )
        usi = result.choose(rng)
        playouts += result.stats.playouts
        selection_seconds += result.stats.selection_seconds
        encoding_seconds += result.stats.encoding_seconds
        evaluator_seconds += result.stats.evaluator_seconds
        expansion_seconds += result.stats.expansion_seconds
        pending.append(
            (
                TrainingSample(
                    encoded.features1,
                    encoded.features2,
                    result.action_labels.astype(np.uint16),
                    result.probabilities.astype(np.float16),
                    0.0,
                    {
                        "kind": "selfplay",
                        "model_version": model_version,
                        "seed": seed,
                        "ply": ply,
                        "usi": usi,
                        "root_value": result.root_value,
                        "search": search_config.to_dict(),
                        "ruleset_id": ruleset_id,
                        "initial_sfen": position.initial_sfen,
                    },
                ),
                position.turn,
            )
        )
        position.push_usi(usi)
        moves.append(usi)
        search.advance(usi, position)
        if progress_callback is not None:
            progress_callback(
                "move",
                {
                    "ply": ply + 1,
                    "usi": usi,
                    "playouts": result.stats.playouts,
                    "new_nodes": result.stats.new_nodes,
                    "search_seconds": result.stats.elapsed_seconds,
                    "playouts_per_second": result.stats.playouts_per_second,
                    "average_batch_size": result.stats.average_batch_size,
                    "capture_counts": position.capture_counts,
                },
            )
        ply += 1

    adjudication = position.result()
    if adjudication is None:
        raise RuntimeError("Self-play ended without an adjudication result")
    samples: list[TrainingSample] = []
    for sample, turn in pending:
        sample.value = adjudication.value_for(turn)
        sample.metadata.update(
            {
                "terminal": adjudication.reason.value,
                "terminal_black_value": adjudication.black_value,
                "terminal_material_diff": adjudication.material_diff,
                "terminal_capture_counts": adjudication.capture_counts,
            }
        )
        samples.append(sample)
    return SelfPlayResult(
        tuple(samples),
        adjudication.winner,
        tuple(moves),
        adjudication.reason.value,
        adjudication,
        playouts,
        selection_seconds,
        encoding_seconds,
        evaluator_seconds,
        expansion_seconds,
    )


def _search_fields(ply: int, progress: SearchProgress) -> dict[str, object]:
    return {
        "ply": ply + 1,
        "completed_playouts": progress.completed_playouts,
        "total_playouts": progress.total_playouts,
        "search_seconds": progress.elapsed_seconds,
        "new_nodes": progress.new_nodes,
        "evaluation_batches": progress.evaluation_batches,
        "evaluated_leaves": progress.evaluated_leaves,
    }
