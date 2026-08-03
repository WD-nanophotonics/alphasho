"""Desktop Player adapter for the classical alpha-beta search."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cshogi

from alphasho.domain import (
    ClockState,
    Color,
    GameSnapshot,
    MoveOption,
    ThinkingConfig,
    TimeControl,
)
from alphasho.search import (
    MoveSearchInfo,
    PlayerSearchProgress,
    allocate_thinking_budget,
)

from .backend import PythonSearchBackend
from .search import AlphaBetaSearch, SearchCancelled, SearchTuning

HeuristicSearchInfo = MoveSearchInfo
HeuristicSearchProgress = PlayerSearchProgress


@dataclass(frozen=True, slots=True)
class _RebuiltSearchPosition:
    board: cshogi.Board
    repetition_counts: Counter[int]
    history_plies: int


class HeuristicPlayer:
    """Time-bounded classical search with no neural or training dependencies."""

    def __init__(
        self,
        thinking: ThinkingConfig | None = None,
        *,
        _search_tuning: SearchTuning | None = None,
    ) -> None:
        self.thinking = thinking or ThinkingConfig()
        self.last_search_info: MoveSearchInfo | None = None
        self._cancelled = threading.Event()
        self._search_tuning = _search_tuning

    def cancel(self) -> None:
        self._cancelled.set()

    def choose_move(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
    ) -> MoveOption:
        return self._search(
            snapshot,
            legal_moves,
            None,
            self.thinking.max_playouts,
            self.thinking.max_depth,
            None,
        )

    def choose_move_with_context(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        *,
        clock_state: ClockState,
        time_control: TimeControl,
        color: Color,
        progress_callback: Callable[[PlayerSearchProgress], None] | None = None,
    ) -> MoveOption:
        budget = allocate_thinking_budget(
            self.thinking,
            time_control,
            clock_state,
            color,
            move_number=snapshot.move_number,
        )
        return self._search(
            snapshot,
            legal_moves,
            budget.time_limit_seconds,
            budget.max_playouts,
            budget.max_depth,
            progress_callback,
        )

    def _search(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        seconds: float | None,
        node_limit: int,
        max_depth: int,
        progress_callback: Callable[[PlayerSearchProgress], None] | None,
    ) -> MoveOption:
        if not legal_moves:
            raise ValueError("HeuristicPlayer requires at least one legal move")
        started = time.monotonic()
        deadline = None if seconds is None else started + max(0.01, seconds)
        self._cancelled.clear()
        rebuilt = _rebuild_search_position(snapshot)
        setup_elapsed = time.monotonic() - started
        search = AlphaBetaSearch(
            rebuilt.board,
            deadline=deadline,
            node_limit=node_limit,
            cancelled=self._cancelled,
            budget_seconds=seconds,
            repetition_counts=rebuilt.repetition_counts,
            started_at=started,
            setup_elapsed_seconds=setup_elapsed,
            position_history_plies=rebuilt.history_plies,
            tuning=self._search_tuning,
        )
        result = PythonSearchBackend(search).run(
            fallback_usi=legal_moves[0].usi,
            max_depth=max_depth,
            progress_callback=progress_callback,
        )
        elapsed = time.monotonic() - started
        deadline_overshoot_ms = (
            result.deadline_overshoot_ms
            if seconds is None
            else max(result.deadline_overshoot_ms, (elapsed - seconds) * 1_000.0, 0.0)
        )
        self.last_search_info = MoveSearchInfo(
            elapsed_seconds=elapsed,
            budget_seconds=seconds,
            playouts=result.nodes,
            max_depth=result.completed_depth,
            root_value=result.normalized_score,
            principal_variation=result.principal_variation,
            main_nodes=result.main_nodes,
            quiescence_nodes=result.quiescence_nodes,
            tt_hits=result.tt_hits,
            beta_cutoffs=result.beta_cutoffs,
            nodes_per_second=result.nodes / max(elapsed, 1e-9),
            completed_root_moves=result.completed_root_moves,
            used_fallback=result.used_fallback,
            quiescence_truncations=result.quiescence_truncations,
            attempted_depth=result.attempted_depth,
            fallback_reason=result.fallback_reason,
            deadline_overshoot_ms=deadline_overshoot_ms,
            q_depth_truncations=result.q_depth_truncations,
            q_leaf_truncations=result.q_leaf_truncations,
            q_budget_truncations=result.q_budget_truncations,
            q_evasion_nodes=result.q_evasion_nodes,
            move_ordering_seconds=result.move_ordering_seconds,
            legal_generation_seconds=result.legal_generation_seconds,
            checking_probe_seconds=result.checking_probe_seconds,
            evaluation_seconds=result.evaluation_seconds,
            evaluation_calls=result.evaluation_calls,
            ordering_calls=result.ordering_calls,
            ordered_moves=result.ordered_moves,
            setup_elapsed_seconds=result.setup_elapsed_seconds,
            position_history_plies=result.position_history_plies,
            timeout_stage=result.timeout_stage,
            incremental_evaluation_calls=result.incremental_evaluation_calls,
            full_evaluation_calls=result.full_evaluation_calls,
            lazy_beta_skips=result.lazy_beta_skips,
            lazy_alpha_skips=result.lazy_alpha_skips,
            cpu_elapsed_seconds=result.cpu_elapsed_seconds,
            cpu_evaluation_seconds=result.cpu_evaluation_seconds,
            cpu_ordering_seconds=result.cpu_ordering_seconds,
            cpu_see_seconds=result.cpu_see_seconds,
            cpu_tt_seconds=result.cpu_tt_seconds,
            cpu_search_overhead_seconds=result.cpu_search_overhead_seconds,
            move_picker_generated=result.move_picker_generated,
            move_picker_yielded=result.move_picker_yielded,
            move_picker_yielded_by_stage=result.move_picker_yielded_by_stage,
            full_evaluation_bound_skips=result.full_evaluation_bound_skips,
            occupied_piece_iterations=result.occupied_piece_iterations,
            mate_status=result.mate_status,
            mate_nodes=result.mate_nodes,
            mate_elapsed_seconds=result.mate_elapsed_seconds,
            mate_principal_variation=result.mate_principal_variation,
            scheduler_wait_seconds=result.scheduler_wait_seconds,
            timing_cause=result.timing_cause,
            classification_seconds=result.classification_seconds,
            sorting_seconds=result.sorting_seconds,
            pvs_null_window_searches=result.pvs_null_window_searches,
            pvs_researches=result.pvs_researches,
            aspiration_fail_low=result.aspiration_fail_low,
            aspiration_fail_high=result.aspiration_fail_high,
            aspiration_researches=result.aspiration_researches,
            see_queries=result.see_queries,
            see_prunes=result.see_prunes,
            lmr_reductions=result.lmr_reductions,
            lmr_researches=result.lmr_researches,
            null_move_attempts=result.null_move_attempts,
            null_move_cutoffs=result.null_move_cutoffs,
            null_move_verifications=result.null_move_verifications,
            razor_prunes=result.razor_prunes,
            reverse_futility_prunes=result.reverse_futility_prunes,
            move_picker_stage_counts=result.move_picker_stage_counts,
            move_picker_stage_seconds=result.move_picker_stage_seconds,
        )
        legal_by_usi = {move.usi: move for move in legal_moves}
        return legal_by_usi.get(result.best_move, legal_moves[0])


def _rebuild_search_position(snapshot: GameSnapshot) -> _RebuiltSearchPosition:
    board = cshogi.Board(snapshot.initial_sfen or snapshot.sfen)
    counts: Counter[int] = Counter({int(board.zobrist_hash()): 1})
    history_plies = 0
    if snapshot.initial_sfen:
        for usi in snapshot.move_history:
            board.push_usi(usi)
            counts[int(board.zobrist_hash())] += 1
            history_plies += 1
    if board.sfen() != snapshot.sfen:
        raise ValueError("snapshot replay mismatch")
    return _RebuiltSearchPosition(board, counts, history_plies)


def _board_from_snapshot(snapshot: GameSnapshot) -> cshogi.Board:
    """Compatibility wrapper for callers that only need the rebuilt board."""

    return _rebuild_search_position(snapshot).board


__all__ = [
    "HeuristicPlayer",
    "HeuristicSearchInfo",
    "HeuristicSearchProgress",
    "SearchCancelled",
]
