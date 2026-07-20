"""Paired-color candidate evaluation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from alphasho.ai import MCTS, SearchCancelled, SearchConfig, TorchEvaluator
from alphasho.domain import Color
from alphasho.training.adjudication import (
    AdjudicationResult,
    AdjudicationRules,
    TrainingPosition,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    candidate_wins: int
    best_wins: int
    draws: int
    games: int
    termination_counts: tuple[tuple[str, int], ...] = ()
    mean_captures: float = 0.0
    mean_absolute_material_diff: float = 0.0

    @property
    def score(self) -> float:
        return (self.candidate_wins + 0.5 * self.draws) / self.games


def evaluate_candidate(
    candidate: TorchEvaluator,
    best: TorchEvaluator,
    search_config: SearchConfig,
    *,
    games: int,
    rules: AdjudicationRules,
    initial_sfens: tuple[str | None, ...],
    seed: int,
    start_game: int = 0,
    candidate_wins: int = 0,
    best_wins: int = 0,
    draws: int = 0,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
    stop_event: threading.Event | None = None,
    heartbeat_seconds: float = 10.0,
) -> EvaluationResult:
    started = time.monotonic()
    termination_counts: dict[str, int] = {}
    total_captures = 0
    total_absolute_material = 0
    for game_index in range(start_game, games):
        if stop_event is not None and stop_event.is_set():
            raise SearchCancelled("Evaluation was stopped")
        candidate_color = Color.BLACK if game_index % 2 == 0 else Color.WHITE
        initial_sfen = initial_sfens[(game_index // 2) % len(initial_sfens)]
        adjudication = _play_game(
            candidate,
            best,
            candidate_color,
            search_config,
            rules=rules,
            initial_sfen=initial_sfen,
            seed=seed + game_index,
            progress_callback=(
                (
                    lambda event, fields, current_game=game_index: progress_callback(
                        event,
                        {"game": current_game + 1, **fields},
                    )
                )
                if progress_callback is not None
                else None
            ),
            stop_event=stop_event,
            heartbeat_seconds=heartbeat_seconds,
        )
        winner = adjudication.winner
        reason = adjudication.reason.value
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
        total_captures += sum(adjudication.capture_counts)
        total_absolute_material += abs(adjudication.material_diff)
        if winner is None:
            draws += 1
        elif winner is candidate_color:
            candidate_wins += 1
        else:
            best_wins += 1
        if progress_callback is not None:
            completed = game_index + 1
            elapsed = time.monotonic() - started
            progress_callback(
                "game-complete",
                {
                    "game": completed,
                    "games": games,
                    "candidate_color": int(candidate_color),
                    "winner": None if winner is None else int(winner),
                    "terminal": adjudication.reason.value,
                    "black_value": adjudication.black_value,
                    "material_diff": adjudication.material_diff,
                    "capture_counts": adjudication.capture_counts,
                    "candidate_wins": candidate_wins,
                    "best_wins": best_wins,
                    "draws": draws,
                    "score": (
                        candidate_wins + 0.5 * draws
                    ) / completed,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / max(completed - start_game, 1)
                    * (games - completed),
                },
            )
    completed_here = max(1, games - start_game)
    return EvaluationResult(
        candidate_wins,
        best_wins,
        draws,
        games,
        tuple(sorted(termination_counts.items())),
        total_captures / completed_here,
        total_absolute_material / completed_here,
    )


def _play_game(
    candidate: TorchEvaluator,
    best: TorchEvaluator,
    candidate_color: Color,
    search_config: SearchConfig,
    *,
    rules: AdjudicationRules,
    initial_sfen: str | None,
    seed: int,
    progress_callback: Callable[[str, dict[str, object]], None] | None,
    stop_event: threading.Event | None,
    heartbeat_seconds: float,
) -> AdjudicationResult:
    position = TrainingPosition(rules, initial_sfen)
    searches = {
        candidate_color: MCTS(candidate, search_config, seed=seed),
        candidate_color.opponent: MCTS(best, search_config, seed=seed + 100_000),
    }
    ply = 0
    while position.result() is None:
        if stop_event is not None and stop_event.is_set():
            raise SearchCancelled("Evaluation game was stopped")
        result = searches[position.turn].search(
            position,
            temperature=1e-3,
            add_root_noise=False,
            cancel_event=stop_event,
            progress_callback=(
                (
                    lambda search_progress, current_ply=ply: progress_callback(
                        "search-heartbeat",
                        {
                            "ply": current_ply + 1,
                            "completed_playouts": search_progress.completed_playouts,
                            "total_playouts": search_progress.total_playouts,
                            "search_seconds": search_progress.elapsed_seconds,
                            "new_nodes": search_progress.new_nodes,
                        },
                    )
                )
                if progress_callback is not None
                else None
            ),
            heartbeat_seconds=heartbeat_seconds,
        )
        usi = result.action_usis[int(result.probabilities.argmax())]
        position.push_usi(usi)
        if progress_callback is not None:
            progress_callback(
                "move",
                {
                    "ply": ply + 1,
                    "usi": usi,
                    "playouts": result.stats.playouts,
                    "playouts_per_second": result.stats.playouts_per_second,
                    "search_seconds": result.stats.elapsed_seconds,
                },
            )
        for search in searches.values():
            search.advance(usi, position)
        ply += 1
    adjudication = position.result()
    if adjudication is None:
        raise RuntimeError("Evaluation ended without an adjudication result")
    return adjudication
