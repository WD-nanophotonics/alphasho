"""Iterative-deepening negamax search for the classical player."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import cshogi

from alphasho.search import PlayerSearchProgress

from .evaluation import (
    HAND_VALUES,
    MAX_FULL_RESIDUAL,
    PIECE_VALUES,
    EvaluationMode,
    EvaluationScratch,
    evaluate_position,
)
from .evaluator import HandcraftedEvaluator, IncrementalEvaluator
from .mate_search import BoundedDfpnMateSolver, MateStatus
from .move_ordering import (
    MoveFeatures,
    MoveOrdering,
    PackedMovePicker,
    move_moving_type,
)

MATE = 1_000_000
INF = MATE + 10_000
MATE_WINDOW = 512
MAX_QUIESCENCE_DEPTH = 4
MAX_QUIESCENCE_NODES_PER_LEAF = 32
MIN_QUIESCENCE_NODES_PER_ITERATION = 128
QUIESCENCE_TO_MAIN_RATIO = 1
DELTA_MARGIN = 100
MAX_TT_ENTRIES = 200_000
_PROMOTED_TYPE = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14}
_HAND_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 5: 5, 6: 6}
_DEMOTED_TYPE = {9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
_HAND_PIECE_TYPE = (1, 2, 3, 4, 7, 5, 6)
SearchMove = MoveFeatures | int


def _native(move: SearchMove) -> int:
    return move if isinstance(move, int) else move.native


def _usi(move: SearchMove) -> str:
    return cshogi.move_to_usi(_native(move))


def _captured(move: SearchMove) -> int:
    return int(cshogi.move_cap(move)) if isinstance(move, int) else move.captured_type


def _moving(move: SearchMove) -> int:
    return move_moving_type(move) if isinstance(move, int) else move.moving_type


def _promotion(move: SearchMove) -> bool:
    return bool(cshogi.move_is_promotion(move)) if isinstance(move, int) else move.promotion


class SearchCancelled(RuntimeError):
    pass


class _SearchTimeout(RuntimeError):
    pass


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


class NodeType(IntEnum):
    NON_PV = 0
    PV = 1


@dataclass(slots=True)
class TranspositionEntry:
    depth: int
    score: int
    bound: Bound
    best_move: int | None


@dataclass(frozen=True, slots=True)
class SearchTuning:
    use_quiescence: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.KING_SAFETY
    advanced_ordering: bool = True
    q_max_depth: int = MAX_QUIESCENCE_DEPTH
    q_leaf_nodes: int = MAX_QUIESCENCE_NODES_PER_LEAF
    q_min_iteration: int = MIN_QUIESCENCE_NODES_PER_ITERATION
    q_to_main_ratio: int = QUIESCENCE_TO_MAIN_RATIO
    q_deep_evaluation_mode: EvaluationMode | None = EvaluationMode.FAST
    use_incremental_evaluation: bool = True
    use_lazy_full: bool = False
    reference_evaluation: bool = False
    use_pvs: bool = False
    use_aspiration: bool = False
    use_staged_move_picker: bool = False
    use_see: bool = False
    use_lmr: bool = False
    use_null_move: bool = False
    use_shallow_pruning: bool = False
    use_root_tactical: bool = True
    move_picker_mode: str = "legacy"
    evaluation_schedule: str = "exact"
    use_mate_probe: bool = False
    mate_max_ply: int = 7
    mate_node_limit: int = 10_000
    mate_time_fraction: float = 0.025


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_move: str
    score: int
    completed_depth: int
    principal_variation: tuple[str, ...]
    elapsed_seconds: float
    main_nodes: int
    quiescence_nodes: int
    tt_hits: int
    beta_cutoffs: int
    completed_root_moves: int
    used_fallback: bool
    quiescence_truncations: int
    attempted_depth: int
    fallback_reason: str | None
    deadline_overshoot_ms: float
    q_depth_truncations: int
    q_leaf_truncations: int
    q_budget_truncations: int
    q_evasion_nodes: int
    move_ordering_seconds: float
    legal_generation_seconds: float
    checking_probe_seconds: float
    evaluation_seconds: float
    evaluation_calls: int
    ordering_calls: int
    ordered_moves: int
    setup_elapsed_seconds: float
    position_history_plies: int
    timeout_stage: str | None
    incremental_evaluation_calls: int
    full_evaluation_calls: int
    lazy_beta_skips: int
    lazy_alpha_skips: int
    cpu_elapsed_seconds: float
    cpu_evaluation_seconds: float
    cpu_ordering_seconds: float
    scheduler_wait_seconds: float
    timing_cause: str | None
    classification_seconds: float
    sorting_seconds: float
    pvs_null_window_searches: int
    pvs_researches: int
    aspiration_fail_low: int
    aspiration_fail_high: int
    aspiration_researches: int
    see_queries: int
    see_prunes: int
    lmr_reductions: int
    lmr_researches: int
    null_move_attempts: int
    null_move_cutoffs: int
    null_move_verifications: int
    razor_prunes: int
    reverse_futility_prunes: int
    move_picker_stage_counts: dict[str, int]
    move_picker_stage_seconds: dict[str, float]
    cpu_see_seconds: float
    cpu_tt_seconds: float
    cpu_search_overhead_seconds: float
    move_picker_generated: int
    move_picker_yielded: int
    move_picker_yielded_by_stage: dict[str, int]
    full_evaluation_bound_skips: int
    occupied_piece_iterations: int
    mate_status: str | None
    mate_nodes: int
    mate_elapsed_seconds: float
    mate_principal_variation: tuple[str, ...]

    @property
    def nodes(self) -> int:
        return self.main_nodes + self.quiescence_nodes

    @property
    def normalized_score(self) -> float:
        if abs(self.score) >= MATE - MATE_WINDOW:
            return math.copysign(1.0, self.score)
        return math.tanh(self.score / 2_000.0)


class _Position:
    """Board plus incremental repetition counts for TT discrimination."""

    def __init__(
        self,
        board: cshogi.Board,
        repetition_counts: Counter[int] | None = None,
        evaluator: IncrementalEvaluator | None = None,
    ) -> None:
        self.board = board
        self.evaluator = evaluator or HandcraftedEvaluator()
        self.evaluator_state = self.evaluator.initialize(board)
        self.null_plies = 0
        if repetition_counts is not None:
            self.counts = repetition_counts.copy()
            return
        clone = board.copy()
        counts: Counter[int] = Counter()
        counts[int(clone.zobrist_hash())] += 1
        while clone.history:
            clone.pop()
            counts[int(clone.zobrist_hash())] += 1
        self.counts = counts

    @property
    def key(self) -> tuple[int, int]:
        hash_value = int(self.board.zobrist_hash())
        return hash_value, min(3, self.counts[hash_value])

    def push(self, move: int) -> None:
        self.evaluator.push(self.board, move, self.evaluator_state)
        self.board.push(move)
        self.counts[int(self.board.zobrist_hash())] += 1

    def pop(self) -> None:
        hash_value = int(self.board.zobrist_hash())
        self.counts[hash_value] -= 1
        if self.counts[hash_value] <= 0:
            del self.counts[hash_value]
        self.board.pop()
        self.evaluator.pop(self.evaluator_state)

    def push_pass(self) -> None:
        self.evaluator_state.stack.append(
            (self.material, self.positional_raw, self.evaluator_state.occupied_mask)
        )
        self.board.push_pass()
        self.null_plies += 1

    def pop_pass(self) -> None:
        self.board.pop_pass()
        self.null_plies -= 1
        self.evaluator.pop(self.evaluator_state)

    @property
    def material(self) -> int:
        return self.evaluator_state.material

    @property
    def positional_raw(self) -> int:
        return self.evaluator_state.positional_raw


class AlphaBetaSearch:
    def __init__(
        self,
        board: cshogi.Board,
        *,
        deadline: float | None,
        node_limit: int,
        cancelled: threading.Event,
        budget_seconds: float | None,
        repetition_counts: Counter[int] | None = None,
        started_at: float | None = None,
        setup_elapsed_seconds: float = 0.0,
        position_history_plies: int = 0,
        max_tt_entries: int = MAX_TT_ENTRIES,
        tuning: SearchTuning | None = None,
        evaluator: IncrementalEvaluator | None = None,
    ) -> None:
        self.position = _Position(board, repetition_counts, evaluator)
        self.hard_deadline = deadline
        reserve = 0.0
        if deadline is not None and budget_seconds is not None:
            reserve = max(0.005, min(0.100, budget_seconds * 0.15))
        self.deadline = None if deadline is None else max(time.monotonic(), deadline - reserve)
        self.node_limit = max(1, node_limit)
        self.cancelled = cancelled
        self.budget_seconds = budget_seconds
        self.max_tt_entries = max(1, max_tt_entries)
        self.tuning = tuning or SearchTuning()
        self.started = started_at if started_at is not None else time.monotonic()
        self.started_cpu = time.process_time()
        self.setup_elapsed_seconds = max(0.0, setup_elapsed_seconds)
        self.position_history_plies = max(0, position_history_plies)
        self.timeout_stage: str | None = None
        self.main_nodes = 0
        self.quiescence_nodes = 0
        self.tt_hits = 0
        self.beta_cutoffs = 0
        self.completed_root_moves = 0
        self.quiescence_truncations = 0
        self.q_depth_truncations = 0
        self.q_leaf_truncations = 0
        self.q_budget_truncations = 0
        self.q_evasion_nodes = 0
        self.move_ordering_seconds = 0.0
        self.legal_generation_seconds = 0.0
        self.checking_probe_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.cpu_evaluation_seconds = 0.0
        self.evaluation_calls = 0
        self.incremental_evaluation_calls = 0
        self.full_evaluation_calls = 0
        self.lazy_beta_skips = 0
        self.lazy_alpha_skips = 0
        self.cpu_ordering_seconds = 0.0
        self.classification_seconds = 0.0
        self.sorting_seconds = 0.0
        self.pvs_null_window_searches = 0
        self.pvs_researches = 0
        self.aspiration_fail_low = 0
        self.aspiration_fail_high = 0
        self.aspiration_researches = 0
        self.see_queries = 0
        self.see_prunes = 0
        self.lmr_reductions = 0
        self.lmr_researches = 0
        self.null_move_attempts = 0
        self.null_move_cutoffs = 0
        self.null_move_verifications = 0
        self.razor_prunes = 0
        self.reverse_futility_prunes = 0
        self.move_picker_stage_counts: Counter[str] = Counter()
        self.move_picker_stage_seconds: dict[str, float] = {}
        self.cpu_see_seconds = 0.0
        self.cpu_tt_seconds = 0.0
        self.move_picker_generated = 0
        self.move_picker_yielded = 0
        self.move_picker_yielded_by_stage: Counter[str] = Counter()
        self.full_evaluation_bound_skips = 0
        self.occupied_piece_iterations = 0
        self.mate_status: str | None = None
        self.mate_nodes = 0
        self.mate_elapsed_seconds = 0.0
        self.mate_principal_variation: tuple[str, ...] = ()
        self.ordering_calls = 0
        self.ordered_moves = 0
        self.attempted_depth = 0
        self.table: dict[tuple[int, int], TranspositionEntry] = {}
        self.evaluation_cache: dict[tuple[int, EvaluationMode], int] = {}
        self.evaluation_scratch = EvaluationScratch()
        self.ordering = MoveOrdering()
        self._partial_move: str | None = None
        self._partial_score = -INF
        self._last_limit_reason: str | None = None
        self._iteration_main_start = 0
        self._iteration_q_start = 0
        self._iteration_q_evasion_start = 0
        self._root_features: list[SearchMove] | None = None
        self._root_checking_moves: list[int] = []

    def run(
        self,
        *,
        fallback_usi: str,
        max_depth: int,
        progress_callback: Callable[[PlayerSearchProgress], None] | None = None,
    ) -> SearchResult:
        best_usi = fallback_usi
        best_score = 0
        completed_depth = 0
        used_fallback = False
        pv = (fallback_usi,)
        fallback_reason: str | None = None

        self.attempted_depth = 1
        self._partial_move = None
        self._partial_score = -INF
        self.completed_root_moves = 0
        try:
            self.timeout_stage = "position_setup"
            self._check_limits(force=True)
            self.timeout_stage = "root_ordering"
            root_order = self._ordered(
                ply=0,
                preferred=None,
                include_checks=False,
                advanced=self.tuning.advanced_ordering,
            )
            root_order = list(root_order)
            self._root_features = root_order
            self.timeout_stage = "root_scan"
            score, usi = self._root_baseline(root_order)
        except _SearchTimeout:
            if self._partial_move is not None:
                best_usi = self._partial_move
                best_score = self._partial_score
                fallback_reason = f"root_scan_{self._last_limit_reason or 'interrupted'}"
            elif self.timeout_stage == "position_setup":
                fallback_reason = (
                    f"position_setup_{self._last_limit_reason or 'interrupted'}"
                )
            else:
                fallback_reason = f"before_root_scan_{self._last_limit_reason or 'interrupted'}"
            pv = (best_usi,)
            used_fallback = True
        else:
            self.timeout_stage = None
            best_usi, best_score, completed_depth = usi, score, 1
            mate_pv = self._run_mate_probe()
            if mate_pv:
                best_usi = mate_pv[0]
                best_score = MATE - len(mate_pv)
                native = self._legal_native(best_usi)
                if native is not None:
                    self._store(
                        self.position.key,
                        TranspositionEntry(1, best_score, Bound.EXACT, native),
                    )
                pv = mate_pv
            else:
                pv = self.principal_variation(1) or (best_usi,)
            if progress_callback is not None:
                progress_callback(self._progress(1, score, pv))

        for depth in range(2, max(1, max_depth) + 1):
            if completed_depth == 0:
                break
            if abs(best_score) >= MATE - MATE_WINDOW:
                break
            self.attempted_depth = depth
            self._partial_move = None
            self._partial_score = -INF
            self.completed_root_moves = 0
            self._iteration_main_start = self.main_nodes
            self._iteration_q_start = self.quiescence_nodes
            self._iteration_q_evasion_start = self.q_evasion_nodes
            try:
                self.timeout_stage = f"iterative_depth_{depth}"
                score, usi = self._root_iteration(depth, best_usi, best_score)
            except _SearchTimeout:
                break
            best_usi, best_score, completed_depth = usi, score, depth
            pv = self.principal_variation(depth)
            if not pv:
                pv = (best_usi,)
            if progress_callback is not None:
                progress_callback(self._progress(depth, score, pv))
            if abs(score) >= MATE - MATE_WINDOW:
                self.timeout_stage = None
                break
            self.timeout_stage = None
        if self.cancelled.is_set():
            raise SearchCancelled()
        finished = time.monotonic()
        finished_cpu = time.process_time()
        cpu_elapsed = max(0.0, finished_cpu - self.started_cpu)
        wall_elapsed = finished - self.started
        scheduler_wait = max(0.0, wall_elapsed - cpu_elapsed)
        overshoot_ms = (
            0.0
            if self.hard_deadline is None
            else max(0.0, (finished - self.hard_deadline) * 1_000.0)
        )
        return SearchResult(
            best_move=best_usi,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=pv,
            elapsed_seconds=wall_elapsed,
            main_nodes=self.main_nodes,
            quiescence_nodes=self.quiescence_nodes,
            tt_hits=self.tt_hits,
            beta_cutoffs=self.beta_cutoffs,
            completed_root_moves=self.completed_root_moves,
            used_fallback=used_fallback,
            quiescence_truncations=self.quiescence_truncations,
            attempted_depth=self.attempted_depth,
            fallback_reason=fallback_reason,
            deadline_overshoot_ms=overshoot_ms,
            q_depth_truncations=self.q_depth_truncations,
            q_leaf_truncations=self.q_leaf_truncations,
            q_budget_truncations=self.q_budget_truncations,
            q_evasion_nodes=self.q_evasion_nodes,
            move_ordering_seconds=self.move_ordering_seconds,
            legal_generation_seconds=self.legal_generation_seconds,
            checking_probe_seconds=self.checking_probe_seconds,
            evaluation_seconds=self.evaluation_seconds,
            evaluation_calls=self.evaluation_calls,
            ordering_calls=self.ordering_calls,
            ordered_moves=self.ordered_moves,
            setup_elapsed_seconds=self.setup_elapsed_seconds,
            position_history_plies=self.position_history_plies,
            timeout_stage=self.timeout_stage,
            incremental_evaluation_calls=self.incremental_evaluation_calls,
            full_evaluation_calls=self.full_evaluation_calls,
            lazy_beta_skips=self.lazy_beta_skips,
            lazy_alpha_skips=self.lazy_alpha_skips,
            cpu_elapsed_seconds=cpu_elapsed,
            cpu_evaluation_seconds=self.cpu_evaluation_seconds,
            cpu_ordering_seconds=self.cpu_ordering_seconds,
            scheduler_wait_seconds=scheduler_wait,
            timing_cause=self._timing_cause(wall_elapsed, cpu_elapsed),
            classification_seconds=self.classification_seconds,
            sorting_seconds=self.sorting_seconds,
            pvs_null_window_searches=self.pvs_null_window_searches,
            pvs_researches=self.pvs_researches,
            aspiration_fail_low=self.aspiration_fail_low,
            aspiration_fail_high=self.aspiration_fail_high,
            aspiration_researches=self.aspiration_researches,
            see_queries=self.see_queries,
            see_prunes=self.see_prunes,
            lmr_reductions=self.lmr_reductions,
            lmr_researches=self.lmr_researches,
            null_move_attempts=self.null_move_attempts,
            null_move_cutoffs=self.null_move_cutoffs,
            null_move_verifications=self.null_move_verifications,
            razor_prunes=self.razor_prunes,
            reverse_futility_prunes=self.reverse_futility_prunes,
            move_picker_stage_counts=dict(self.move_picker_stage_counts),
            move_picker_stage_seconds=dict(self.move_picker_stage_seconds),
            cpu_see_seconds=self.cpu_see_seconds,
            cpu_tt_seconds=self.cpu_tt_seconds,
            cpu_search_overhead_seconds=max(
                0.0,
                cpu_elapsed
                - self.cpu_evaluation_seconds
                - self.cpu_ordering_seconds
                - self.cpu_see_seconds
                - self.cpu_tt_seconds,
            ),
            move_picker_generated=self.move_picker_generated,
            move_picker_yielded=self.move_picker_yielded,
            move_picker_yielded_by_stage=dict(self.move_picker_yielded_by_stage),
            full_evaluation_bound_skips=self.full_evaluation_bound_skips,
            occupied_piece_iterations=self.occupied_piece_iterations,
            mate_status=self.mate_status,
            mate_nodes=self.mate_nodes,
            mate_elapsed_seconds=self.mate_elapsed_seconds,
            mate_principal_variation=self.mate_principal_variation,
        )

    def _root_baseline(self, moves: list[SearchMove]) -> tuple[int, str]:
        best_score = -INF
        best: SearchMove | None = None
        base_score = (
            None if self.tuning.use_root_tactical else self._evaluate_baseline()
        )
        for move in moves:
            self._check_limits(force=True)
            self.main_nodes += 1
            if base_score is not None:
                score = base_score + self._root_material_delta(move)
            else:
                self.position.push(_native(move))
                try:
                    self._check_limits(force=True)
                    if self.position.board.is_check():
                        self._root_checking_moves.append(_native(move))
                    terminal = self._rule_terminal_score(1, baseline=True)
                    replies_started = time.monotonic()
                    has_reply = next(iter(self.position.board.legal_moves), None) is not None
                    self.legal_generation_seconds += time.monotonic() - replies_started
                    if not has_reply:
                        score = MATE - 1
                    elif terminal is not None:
                        score = -terminal
                    else:
                        score = -self._evaluate_incremental(EvaluationMode.FAST)
                finally:
                    self.position.pop()
            if score > best_score:
                best_score, best = score, move
                self._partial_move, self._partial_score = _usi(move), score
            self.completed_root_moves += 1
            self._check_limits(force=True)
        if best is None:
            raise _SearchTimeout()
        self._store(
            self.position.key,
            TranspositionEntry(1, _score_to_tt(best_score, 0), Bound.EXACT, _native(best)),
        )
        return best_score, _usi(best)

    def _run_mate_probe(self) -> tuple[str, ...]:
        if not self.tuning.use_mate_probe:
            return ()
        if (
            self.budget_seconds is None
            or self.budget_seconds < 0.5
            or self.position.board.is_check()
            or not self._root_checking_moves
        ):
            self.mate_status = "not_triggered"
            return ()
        allowance = min(
            0.050,
            max(0.0, self.budget_seconds * self.tuning.mate_time_fraction),
        )
        deadline = time.monotonic() + allowance
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        solver = BoundedDfpnMateSolver(
            self.position.board,
            max_ply=self.tuning.mate_max_ply,
            node_limit=self.tuning.mate_node_limit,
            deadline=deadline,
            cancelled=self.cancelled.is_set,
            root_checks=self._root_checking_moves,
        )
        result = solver.solve()
        self.mate_status = result.status.value
        self.mate_nodes = result.nodes
        self.mate_elapsed_seconds = result.elapsed_seconds
        self.mate_principal_variation = result.principal_variation
        return (
            result.principal_variation
            if result.status is MateStatus.PROVEN
            else ()
        )

    @staticmethod
    def _root_material_delta(move: SearchMove) -> int:
        delta = 0
        captured = _captured(move)
        moving = _moving(move)
        if captured:
            hand_type = _DEMOTED_TYPE.get(captured, captured)
            delta += PIECE_VALUES[captured]
            delta += HAND_VALUES[_HAND_INDEX[hand_type]]
        if _promotion(move):
            promoted = _PROMOTED_TYPE[moving]
            delta += PIECE_VALUES[promoted] - PIECE_VALUES[moving]
        return delta

    def _root_iteration(
        self, depth: int, preferred_usi: str, previous_score: int
    ) -> tuple[int, str]:
        if (
            not self.tuning.use_aspiration
            or depth < 4
            or abs(previous_score) >= MATE - MATE_WINDOW
        ):
            return self._root(depth, preferred_usi, -INF, INF)
        delta = 50
        alpha = max(-INF, previous_score - delta)
        beta = min(INF, previous_score + delta)
        failures = 0
        while True:
            score, usi = self._root(depth, preferred_usi, alpha, beta)
            if score <= alpha:
                self.aspiration_fail_low += 1
                self.aspiration_researches += 1
                failures += 1
                if failures >= 2:
                    alpha, beta = -INF, INF
                else:
                    delta *= 2
                    alpha = max(-INF, score - delta)
                    beta = min(INF, beta)
                continue
            if score >= beta:
                self.aspiration_fail_high += 1
                self.aspiration_researches += 1
                failures += 1
                if failures >= 2:
                    alpha, beta = -INF, INF
                else:
                    delta *= 2
                    alpha = max(-INF, alpha)
                    beta = min(INF, score + delta)
                continue
            return score, usi

    def _root(
        self,
        depth: int,
        preferred_usi: str,
        window_alpha: int,
        window_beta: int,
    ) -> tuple[int, str]:
        self._check_limits(force=True)
        preferred = self._legal_native(preferred_usi)
        alpha = window_alpha
        beta = window_beta
        best_score = -INF
        best: SearchMove | None = None
        root_features = self._root_features or self._ordered(ply=0, preferred=preferred)
        root_list = list(root_features)
        if self.tuning.move_picker_mode == "packed":
            moves = sorted(
                root_list,
                key=lambda move: _native(move) != preferred,
            )
        else:
            moves = self._sort_features(root_list, ply=0, preferred=preferred)
        for move_index, move in enumerate(moves):
            self._check_limits(force=True)
            native = _native(move)
            self.position.push(native)
            try:
                if self.tuning.use_pvs and move_index > 0:
                    self.pvs_null_window_searches += 1
                    score = -self._negamax(
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        1,
                        NodeType.NON_PV,
                        native,
                    )
                    if alpha < score < beta:
                        self.pvs_researches += 1
                        score = -self._negamax(
                            depth - 1,
                            -beta,
                            -alpha,
                            1,
                            NodeType.PV,
                            native,
                        )
                else:
                    score = -self._negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        NodeType.PV,
                        native,
                    )
            finally:
                self.position.pop()
            if score > best_score:
                best_score, best = score, move
                self._partial_move, self._partial_score = _usi(move), score
            alpha = max(alpha, score)
            self.completed_root_moves += 1
            if alpha >= beta:
                self.beta_cutoffs += 1
                break
        if best is None:
            raise _SearchTimeout()
        self._store(
            self.position.key,
            TranspositionEntry(
                depth,
                _score_to_tt(best_score, 0),
                Bound.UPPER
                if best_score <= window_alpha
                else Bound.LOWER
                if best_score >= window_beta
                else Bound.EXACT,
                _native(best),
            ),
        )
        return best_score, _usi(best)

    def _negamax(
        self,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        node_type: NodeType,
        previous_move: int | None,
        *,
        allow_null: bool = True,
    ) -> int:
        if depth <= 0:
            self.main_nodes += 1
            self._check_limits(force=True)
            if not self.tuning.use_quiescence:
                terminal = self._rule_terminal_score(ply)
                return self._evaluate() if terminal is None else terminal
            if not self.position.board.is_check() and self._q_budget_exhausted():
                self.q_budget_truncations += 1
                self.quiescence_truncations += 1
                score, _bound = self._bounded_static(alpha, beta)
                return score
            return self._quiescence(alpha, beta, ply, 0, self._ordinary_q_nodes)
        self.main_nodes += 1
        self._check_limits(force=True)
        terminal = self._rule_terminal_score(ply)
        if terminal is not None:
            return terminal

        # Mate-distance pruning is exact and prevents searching a line whose
        # best possible mate is already worse than a known one.
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        key = self.position.key
        entry = self._tt_get(key)
        original_alpha = alpha
        original_beta = beta
        if entry is not None and entry.depth >= depth:
            self.tt_hits += 1
            score = _score_from_tt(entry.score, ply)
            if entry.bound is Bound.EXACT:
                return score
            if entry.bound is Bound.LOWER:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        board = self.position.board
        in_check = bool(board.is_check())
        static_eval: int | None = None
        repetition_sensitive = self.position.counts[int(board.zobrist_hash())] >= 2
        pruning_window = abs(alpha) < MATE - MATE_WINDOW and abs(beta) < MATE - MATE_WINDOW
        if (
            not in_check
            and node_type is NodeType.NON_PV
            and pruning_window
            and not repetition_sensitive
            and (self.tuning.use_shallow_pruning or self.tuning.use_null_move)
        ):
            static_eval = self._evaluate_incremental(EvaluationMode.FAST)

        if self.tuning.use_shallow_pruning and static_eval is not None:
            if depth <= 2 and static_eval + 180 * depth <= alpha:
                qscore = self._quiescence(alpha, beta, ply, 0, self._ordinary_q_nodes)
                if qscore <= alpha:
                    self.razor_prunes += 1
                    return qscore
            if depth <= 3 and static_eval - 120 * depth >= beta:
                self.reverse_futility_prunes += 1
                return static_eval

        if (
            self.tuning.use_null_move
            and allow_null
            and node_type is NodeType.NON_PV
            and depth >= 4
            and not in_check
            and static_eval is not None
            and static_eval >= beta
            and self._has_non_pawn_material()
        ):
            reduction = 3 if depth >= 8 else 2
            self.null_move_attempts += 1
            self.position.push_pass()
            try:
                null_score = -self._negamax(
                    depth - reduction - 1,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    NodeType.NON_PV,
                    None,
                    allow_null=False,
                )
            finally:
                self.position.pop_pass()
            if null_score >= beta:
                if depth >= 8:
                    self.null_move_verifications += 1
                    verified = self._negamax(
                        depth - reduction,
                        beta - 1,
                        beta,
                        ply,
                        NodeType.NON_PV,
                        previous_move,
                        allow_null=False,
                    )
                    if verified < beta:
                        null_score = verified
                if null_score >= beta:
                    self.null_move_cutoffs += 1
                    return null_score

        best = -INF
        best_move: SearchMove | None = None
        preferred = entry.best_move if entry else None
        moves = self._ordered(
            ply=ply,
            preferred=preferred,
            advanced=self.tuning.advanced_ordering,
            countermove=self.ordering.countermoves.get(previous_move)
            if previous_move is not None
            else None,
            staged=self.tuning.use_staged_move_picker,
            see_score=self._see_ordering_score if self.tuning.use_see else None,
        )
        for move_index, move in enumerate(moves):
            self._check_limits(force=True)
            native = _native(move)
            captured = _captured(move)
            promotion = _promotion(move)
            self.position.push(native)
            try:
                gives_check = bool(self.position.board.is_check())
                reduced = (
                    self.tuning.use_lmr
                    and node_type is NodeType.NON_PV
                    and depth >= 3
                    and move_index >= 3
                    and native != preferred
                    and not captured
                    and not promotion
                    and native not in self.ordering.killers.get(ply, ())
                    and not gives_check
                    and not in_check
                )
                score: int
                if reduced:
                    self.lmr_reductions += 1
                    score = -self._negamax(
                        depth - 2,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        NodeType.NON_PV,
                        native,
                    )
                    if score > alpha:
                        self.lmr_researches += 1
                        score = -self._negamax(
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            NodeType.NON_PV,
                            native,
                        )
                elif (
                    self.tuning.use_pvs
                    and node_type is NodeType.PV
                    and move_index > 0
                ):
                    self.pvs_null_window_searches += 1
                    score = -self._negamax(
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        NodeType.NON_PV,
                        native,
                    )
                else:
                    child_type = (
                        NodeType.PV
                        if node_type is NodeType.PV and move_index == 0
                        else NodeType.NON_PV
                    )
                    score = -self._negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        child_type,
                        native,
                    )
                if (
                    self.tuning.use_pvs
                    and node_type is NodeType.PV
                    and move_index > 0
                    and alpha < score < beta
                ):
                    self.pvs_researches += 1
                    score = -self._negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        NodeType.PV,
                        native,
                    )
            finally:
                self.position.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.beta_cutoffs += 1
                if self.tuning.advanced_ordering:
                    if isinstance(move, int):
                        self.ordering.record_cutoff_native(
                            move,
                            ply=ply,
                            depth=depth,
                            previous_move=previous_move,
                        )
                    else:
                        self.ordering.record_cutoff(
                            move,
                            ply=ply,
                            depth=depth,
                            previous_move=previous_move,
                        )
                break
            self._check_limits(force=True)

        if best_move is None:
            return -MATE + ply
        bound = (
            Bound.UPPER
            if best <= original_alpha
            else Bound.LOWER
            if best >= original_beta
            else Bound.EXACT
        )
        self._store(
            key,
            TranspositionEntry(depth, _score_to_tt(best, ply), bound, _native(best_move)),
        )
        return best

    def _quiescence(
        self,
        alpha: int,
        beta: int,
        ply: int,
        qdepth: int,
        leaf_start: int,
    ) -> int:
        self.quiescence_nodes += 1
        self._check_limits(force=True)
        terminal = self._rule_terminal_score(ply)
        if terminal is not None:
            return terminal
        board = self.position.board
        in_check = bool(board.is_check())
        if in_check:
            self.q_evasion_nodes += 1
        stand_pat: int | None = None
        if not in_check:
            evaluation_mode = (
                self.tuning.evaluation_mode
                if qdepth == 0 or self.tuning.q_deep_evaluation_mode is None
                else self.tuning.q_deep_evaluation_mode
            )
            stand_pat, static_bound = self._bounded_static(
                alpha, beta, evaluation_mode
            )
            if static_bound is Bound.LOWER:
                return beta
            if static_bound is Bound.EXACT:
                if stand_pat >= beta:
                    return stand_pat
                alpha = max(alpha, stand_pat)
            else:
                # A static fail-low cannot improve alpha, but tactical captures
                # and promotions must still be searched.
                stand_pat = None
            if qdepth >= self.tuning.q_max_depth:
                self.q_depth_truncations += 1
                self.quiescence_truncations += 1
                return alpha
            if self._ordinary_q_nodes - leaf_start >= self.tuning.q_leaf_nodes:
                self.q_leaf_truncations += 1
                self.quiescence_truncations += 1
                return alpha
            iteration_main = self.main_nodes - self._iteration_main_start
            iteration_q = self._ordinary_q_nodes - (
                self._iteration_q_start - self._iteration_q_evasion_start
            )
            q_budget = max(
                self.tuning.q_min_iteration,
                self.tuning.q_to_main_ratio * max(1, iteration_main),
            )
            if iteration_q > q_budget:
                self.q_budget_truncations += 1
                self.quiescence_truncations += 1
                return alpha

        moves = self._ordered(
            ply=ply,
            tactical_only=not in_check,
            include_checks=False,
            advanced=self.tuning.advanced_ordering,
        )
        if not in_check:
            moves = [move for move in moves if _captured(move) or _promotion(move)]
        if in_check and not moves:
            return -MATE + ply
        for move in moves:
            if (
                self.tuning.use_see
                and not in_check
                and _captured(move)
                and not _promotion(move)
                and not self._see_ge(move, -PIECE_VALUES[1])
            ):
                self.see_prunes += 1
                continue
            if (
                not in_check
                and stand_pat is not None
                and _captured(move)
                and not _promotion(move)
                and stand_pat + PIECE_VALUES[_captured(move)] + DELTA_MARGIN <= alpha
            ):
                continue
            self._check_limits(force=True)
            self.position.push(_native(move))
            try:
                score = -self._quiescence(
                    -beta,
                    -alpha,
                    ply + 1,
                    qdepth + 1,
                    leaf_start,
                )
            finally:
                self.position.pop()
            if score >= beta:
                self.beta_cutoffs += 1
                return score
            alpha = max(alpha, score)
            self._check_limits(force=True)
        return alpha

    def principal_variation(self, depth: int) -> tuple[str, ...]:
        clone = self.position.board.copy()
        result: list[str] = []
        counts = self.position.counts.copy()
        for _ in range(depth):
            hash_value = int(clone.zobrist_hash())
            entry = self._tt_get((hash_value, min(3, counts[hash_value])))
            if entry is None or entry.best_move is None:
                break
            legal = {int(move) for move in clone.legal_moves}
            if entry.best_move not in legal:
                break
            usi = cshogi.move_to_usi(entry.best_move)
            result.append(usi)
            clone.push(entry.best_move)
            counts[int(clone.zobrist_hash())] += 1
        return tuple(result)

    def _rule_terminal_score(self, ply: int, *, baseline: bool = False) -> int | None:
        board = self.position.board
        hash_value = int(board.zobrist_hash())
        if self.position.null_plies == 0 and self.position.counts[hash_value] >= 4:
            repetition = board.is_draw()
            if repetition == cshogi.REPETITION_DRAW:
                return 0
            if repetition in {cshogi.REPETITION_WIN, cshogi.REPETITION_SUPERIOR}:
                return MATE - ply
            if repetition in {cshogi.REPETITION_LOSE, cshogi.REPETITION_INFERIOR}:
                return -MATE + ply
        if baseline:
            return None
        if board.is_nyugyoku():
            return MATE - ply
        return None

    def _ordered(
        self, *, ply: int, **values: object
    ) -> list[MoveFeatures] | PackedMovePicker:
        generation_started = time.monotonic()
        legal_moves = [int(move) for move in self.position.board.legal_moves]
        self.legal_generation_seconds += time.monotonic() - generation_started
        self._check_limits(force=True)
        started = time.monotonic()
        started_cpu = time.process_time()
        try:
            if self.tuning.move_picker_mode == "packed":
                preferred = values.get("preferred")
                countermove = values.get("countermove")
                tactical_only = bool(values.get("tactical_only", False))
                see_score = (
                    self._see_ordering_score if self.tuning.use_see else None
                )

                def record_yield(stage: str) -> None:
                    self.move_picker_yielded += 1
                    self.move_picker_yielded_by_stage[stage] += 1

                picker = self.ordering.packed(
                    legal_moves,
                    ply=ply,
                    preferred=preferred if isinstance(preferred, int) else None,
                    countermove=countermove if isinstance(countermove, int) else None,
                    tactical_only=tactical_only,
                    see_score=see_score,
                    on_yield=record_yield,
                )
                self.ordering_calls += 1
                self.ordered_moves += picker.generated
                self.move_picker_generated += picker.generated
                for stage, count in picker.stage_counts.items():
                    self.move_picker_stage_counts[stage] += count
                return picker
            result = self.ordering.ordered(
                self.position.board,
                ply=ply,
                check_limits=lambda: self._check_limits(force=True),
                legal_moves=legal_moves,
                include_usi=ply == 0,
                **values,
            )
            self.ordering_calls += 1
            self.ordered_moves += len(result)
            self.move_picker_generated += len(result)
            self.move_picker_yielded += len(result)
            self.classification_seconds += self.ordering.last_classification_seconds
            self.sorting_seconds += self.ordering.last_sorting_seconds
            for stage, count in self.ordering.last_stage_counts.items():
                self.move_picker_stage_counts[stage] += count
            self.move_picker_stage_seconds["classification"] = (
                self.move_picker_stage_seconds.get("classification", 0.0)
                + self.ordering.last_classification_seconds
            )
            self.move_picker_stage_seconds["sorting"] = (
                self.move_picker_stage_seconds.get("sorting", 0.0)
                + self.ordering.last_sorting_seconds
            )
            return result
        finally:
            self.move_ordering_seconds += time.monotonic() - started
            self.cpu_ordering_seconds += time.process_time() - started_cpu

    @staticmethod
    def _placed_piece_type(move: SearchMove) -> int:
        moving = _moving(move)
        if _promotion(move):
            return _PROMOTED_TYPE.get(moving, moving)
        return moving

    def _see_ordering_score(self, move: SearchMove) -> int:
        captured = _captured(move)
        if not captured:
            return 0
        victim_hand = _DEMOTED_TYPE.get(captured, captured)
        gain = PIECE_VALUES[captured] + HAND_VALUES[_HAND_INDEX[victim_hand]]
        placed = self._placed_piece_type(move)
        loss_hand = _DEMOTED_TYPE.get(placed, placed)
        loss = PIECE_VALUES[placed]
        if loss_hand in _HAND_INDEX:
            loss += HAND_VALUES[_HAND_INDEX[loss_hand]]
        return gain - loss

    def _see_ge(self, move: SearchMove, threshold: int) -> bool:
        """Conservative legal-recapture SEE; uncertainty always keeps the move."""

        self.see_queries += 1
        started_cpu = time.process_time()
        try:
            return self._see_ge_impl(move, threshold)
        finally:
            self.cpu_see_seconds += time.process_time() - started_cpu

    def _see_ge_impl(self, move: SearchMove, threshold: int) -> bool:
        if not _captured(move) or _moving(move) == 8:
            return True
        estimate = self._see_ordering_score(move)
        if estimate >= threshold:
            return True
        native = _native(move)
        target = int(cshogi.move_to(native))
        self.position.push(native)
        try:
            if self.position.board.is_check():
                return True
            recaptures = [
                int(reply)
                for reply in self.position.board.legal_moves
                if int(cshogi.move_to(reply)) == target and int(cshogi.move_cap(reply))
            ]
            if not recaptures:
                return True
            for reply in recaptures:
                self._check_limits(force=True)
                self.position.push(reply)
                try:
                    can_continue = any(
                        int(cshogi.move_to(answer)) == target
                        and int(cshogi.move_cap(answer))
                        for answer in self.position.board.legal_moves
                    )
                finally:
                    self.position.pop()
                if not can_continue:
                    return estimate >= threshold
            return True
        finally:
            self.position.pop()

    def _has_non_pawn_material(self) -> bool:
        turn = int(self.position.board.turn)
        for raw in self.position.board.pieces:
            code = int(raw)
            if not code:
                continue
            color = cshogi.WHITE if code >= 16 else cshogi.BLACK
            piece_type = code - 16 if color == cshogi.WHITE else code
            if color == turn and piece_type not in {1, 8}:
                return True
        hand = self.position.board.pieces_in_hand[turn]
        return any(int(hand[index]) for index in range(1, len(hand)))

    def _sort_features(
        self,
        features: list[SearchMove],
        *,
        ply: int,
        preferred: int | None,
    ) -> list[SearchMove]:
        killers = self.ordering.killers.get(ply, ())

        def score(item: SearchMove) -> int:
            native = _native(item)
            captured = _captured(item)
            moving = _moving(item)
            if native == preferred:
                return 100_000_000
            if captured:
                return (
                    70_000_000
                    + PIECE_VALUES[captured] * 32
                    - PIECE_VALUES[moving]
                )
            if _promotion(item):
                return 60_000_000 + PIECE_VALUES[moving]
            if native in killers:
                return 40_000_000 - killers.index(native)
            return self.ordering.history.get((moving, cshogi.move_to(native)), 0)

        started = time.monotonic()
        started_cpu = time.process_time()
        preferred_moves = [item for item in features if _native(item) == preferred]
        captures = [
            item for item in features if _native(item) != preferred and _captured(item)
        ]
        promotions = [
            item
            for item in features
            if _native(item) != preferred and not _captured(item) and _promotion(item)
        ]
        killer_moves = [
            item
            for item in features
            if _native(item) != preferred
            and not _captured(item)
            and not _promotion(item)
            and _native(item) in killers
        ]
        quiet = [
            item
            for item in features
            if _native(item) != preferred
            and not _captured(item)
            and not _promotion(item)
            and _native(item) not in killers
        ]
        captures.sort(key=score, reverse=True)
        promotions.sort(key=score, reverse=True)
        killer_moves.sort(key=score, reverse=True)
        quiet.sort(key=score, reverse=True)
        result = preferred_moves + captures + promotions + killer_moves + quiet
        elapsed = time.monotonic() - started
        self.move_ordering_seconds += elapsed
        self.sorting_seconds += elapsed
        self.cpu_ordering_seconds += time.process_time() - started_cpu
        self.ordering_calls += 1
        self.ordered_moves += len(result)
        return result

    def _store(self, key: tuple[int, int], entry: TranspositionEntry) -> None:
        started_cpu = time.process_time()
        try:
            previous = self.table.get(key)
            if previous is not None and previous.depth > entry.depth:
                return
            if previous is None and len(self.table) >= self.max_tt_entries:
                return
            self.table[key] = entry
        finally:
            self.cpu_tt_seconds += time.process_time() - started_cpu

    def _tt_get(self, key: tuple[int, int]) -> TranspositionEntry | None:
        started_cpu = time.process_time()
        try:
            return self.table.get(key)
        finally:
            self.cpu_tt_seconds += time.process_time() - started_cpu

    def _evaluate(self, mode: EvaluationMode | None = None) -> int:
        self._check_limits(force=True)
        evaluation_mode = mode or self.tuning.evaluation_mode
        if (
            self.tuning.use_incremental_evaluation
            and not self.tuning.reference_evaluation
            and evaluation_mode in {EvaluationMode.FAST, EvaluationMode.MATERIAL}
        ):
            return self._evaluate_incremental(evaluation_mode)
        key = (int(self.position.board.zobrist_hash()), evaluation_mode)
        cached = self.evaluation_cache.get(key)
        if cached is not None:
            return cached
        started = time.monotonic()
        started_cpu = time.process_time()
        self.evaluation_calls += 1
        if evaluation_mode is EvaluationMode.KING_SAFETY:
            self.full_evaluation_calls += 1
            self.occupied_piece_iterations += (
                self.position.evaluator_state.occupied_mask.bit_count()
            )
        try:
            if (
                self.tuning.use_incremental_evaluation
                and not self.tuning.reference_evaluation
            ):
                score = self.position.evaluator.exact(
                    self.position.board,
                    self.position.evaluator_state,
                    evaluation_mode,
                    self.evaluation_scratch,
                )
            else:
                score = evaluate_position(self.position.board, evaluation_mode)
        finally:
            self.evaluation_seconds += time.monotonic() - started
            self.cpu_evaluation_seconds += time.process_time() - started_cpu
        self._check_limits(force=True)
        if len(self.evaluation_cache) < self.max_tt_entries:
            self.evaluation_cache[key] = score
        return score

    def _evaluate_incremental(self, mode: EvaluationMode) -> int:
        key = (int(self.position.board.zobrist_hash()), mode)
        cached = self.evaluation_cache.get(key)
        if cached is not None:
            return cached
        started = time.monotonic()
        started_cpu = time.process_time()
        self.evaluation_calls += 1
        self.incremental_evaluation_calls += 1
        try:
            result = self.position.evaluator.fast(
                self.position.board,
                self.position.evaluator_state,
                mode,
            )
            if len(self.evaluation_cache) < self.max_tt_entries:
                self.evaluation_cache[key] = result
            return result
        finally:
            self.evaluation_seconds += time.monotonic() - started
            self.cpu_evaluation_seconds += time.process_time() - started_cpu

    def _bounded_static(
        self,
        alpha: int,
        beta: int,
        mode: EvaluationMode | None = None,
    ) -> tuple[int, Bound]:
        evaluation_mode = mode or self.tuning.evaluation_mode
        lazy_full = (
            self.tuning.use_lazy_full
            or self.tuning.evaluation_schedule == "lazy_full"
        )
        if not lazy_full or evaluation_mode is not EvaluationMode.KING_SAFETY:
            return self._evaluate(evaluation_mode), Bound.EXACT
        fast = self._evaluate_incremental(EvaluationMode.FAST)
        lower = fast - MAX_FULL_RESIDUAL
        upper = fast + MAX_FULL_RESIDUAL
        if lower >= beta:
            self.lazy_beta_skips += 1
            self.full_evaluation_bound_skips += 1
            return beta, Bound.LOWER
        if upper <= alpha:
            self.lazy_alpha_skips += 1
            self.full_evaluation_bound_skips += 1
            return alpha, Bound.UPPER
        return self._evaluate(EvaluationMode.KING_SAFETY), Bound.EXACT

    def _evaluate_baseline(self) -> int:
        self._check_limits(force=True)
        started = time.monotonic()
        self.evaluation_calls += 1
        try:
            score = evaluate_position(self.position.board, EvaluationMode.MATERIAL)
        finally:
            self.evaluation_seconds += time.monotonic() - started
        self._check_limits(force=True)
        return score

    def _legal_native(self, usi: str) -> int | None:
        for move in self.position.board.legal_moves:
            native = int(move)
            if cshogi.move_to_usi(native) == usi:
                return native
        return None

    def _check_limits(self, *, force: bool = False) -> None:
        total = self.main_nodes + self.quiescence_nodes
        if total >= self.node_limit:
            self._last_limit_reason = "node_limit"
            raise _SearchTimeout()
        if self.cancelled.is_set():
            raise SearchCancelled()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._last_limit_reason = "deadline"
            raise _SearchTimeout()

    def _timing_cause(self, wall_elapsed: float, cpu_elapsed: float) -> str | None:
        if self.budget_seconds is None:
            return None
        tolerance = max(0.050, self.budget_seconds * 0.05)
        wall_invalid = wall_elapsed > self.budget_seconds + tolerance
        cpu_invalid = cpu_elapsed > self.budget_seconds
        if not wall_invalid:
            return None
        scheduler_wait = max(0.0, wall_elapsed - cpu_elapsed)
        if cpu_invalid and scheduler_wait > tolerance:
            return "mixed"
        if cpu_invalid:
            return "search_cpu_overrun"
        if wall_invalid:
            return "scheduler_stall"
        return None

    def _progress(
        self,
        depth: int,
        score: int,
        pv: tuple[str, ...],
    ) -> PlayerSearchProgress:
        elapsed = time.monotonic() - self.started
        nodes = self.main_nodes + self.quiescence_nodes
        normalized = (
            math.copysign(1.0, score)
            if abs(score) >= MATE - MATE_WINDOW
            else math.tanh(score / 2_000.0)
        )
        return PlayerSearchProgress(
            completed_playouts=nodes,
            max_playouts=self.node_limit,
            elapsed_seconds=elapsed,
            budget_seconds=self.budget_seconds,
            max_depth=depth,
            root_value=normalized,
            principal_variation=pv,
            main_nodes=self.main_nodes,
            quiescence_nodes=self.quiescence_nodes,
            tt_hits=self.tt_hits,
            beta_cutoffs=self.beta_cutoffs,
            nodes_per_second=nodes / max(elapsed, 1e-9),
            completed_root_moves=self.completed_root_moves,
            used_fallback=False,
            quiescence_truncations=self.quiescence_truncations,
            attempted_depth=self.attempted_depth,
            fallback_reason=None,
            deadline_overshoot_ms=0.0,
            q_depth_truncations=self.q_depth_truncations,
            q_leaf_truncations=self.q_leaf_truncations,
            q_budget_truncations=self.q_budget_truncations,
            q_evasion_nodes=self.q_evasion_nodes,
            move_ordering_seconds=self.move_ordering_seconds,
            legal_generation_seconds=self.legal_generation_seconds,
            checking_probe_seconds=self.checking_probe_seconds,
            evaluation_seconds=self.evaluation_seconds,
            evaluation_calls=self.evaluation_calls,
            ordering_calls=self.ordering_calls,
            ordered_moves=self.ordered_moves,
            setup_elapsed_seconds=self.setup_elapsed_seconds,
            position_history_plies=self.position_history_plies,
            timeout_stage=self.timeout_stage,
            incremental_evaluation_calls=self.incremental_evaluation_calls,
            full_evaluation_calls=self.full_evaluation_calls,
            lazy_beta_skips=self.lazy_beta_skips,
            lazy_alpha_skips=self.lazy_alpha_skips,
            cpu_elapsed_seconds=max(0.0, time.process_time() - self.started_cpu),
            cpu_evaluation_seconds=self.cpu_evaluation_seconds,
            cpu_ordering_seconds=self.cpu_ordering_seconds,
            scheduler_wait_seconds=max(
                0.0,
                elapsed - max(0.0, time.process_time() - self.started_cpu),
            ),
            timing_cause=None,
            classification_seconds=self.classification_seconds,
            sorting_seconds=self.sorting_seconds,
            pvs_null_window_searches=self.pvs_null_window_searches,
            pvs_researches=self.pvs_researches,
            aspiration_fail_low=self.aspiration_fail_low,
            aspiration_fail_high=self.aspiration_fail_high,
            aspiration_researches=self.aspiration_researches,
            see_queries=self.see_queries,
            see_prunes=self.see_prunes,
            lmr_reductions=self.lmr_reductions,
            lmr_researches=self.lmr_researches,
            null_move_attempts=self.null_move_attempts,
            null_move_cutoffs=self.null_move_cutoffs,
            null_move_verifications=self.null_move_verifications,
            razor_prunes=self.razor_prunes,
            reverse_futility_prunes=self.reverse_futility_prunes,
            move_picker_stage_counts=dict(self.move_picker_stage_counts),
            move_picker_stage_seconds=dict(self.move_picker_stage_seconds),
            cpu_see_seconds=self.cpu_see_seconds,
            cpu_tt_seconds=self.cpu_tt_seconds,
            cpu_search_overhead_seconds=max(
                0.0,
                max(0.0, time.process_time() - self.started_cpu)
                - self.cpu_evaluation_seconds
                - self.cpu_ordering_seconds
                - self.cpu_see_seconds
                - self.cpu_tt_seconds,
            ),
            move_picker_generated=self.move_picker_generated,
            move_picker_yielded=self.move_picker_yielded,
            move_picker_yielded_by_stage=dict(self.move_picker_yielded_by_stage),
            full_evaluation_bound_skips=self.full_evaluation_bound_skips,
            occupied_piece_iterations=self.occupied_piece_iterations,
            mate_status=self.mate_status,
            mate_nodes=self.mate_nodes,
            mate_elapsed_seconds=self.mate_elapsed_seconds,
            mate_principal_variation=self.mate_principal_variation,
        )

    @property
    def _ordinary_q_nodes(self) -> int:
        return self.quiescence_nodes - self.q_evasion_nodes

    def _q_budget_exhausted(self) -> bool:
        iteration_main = self.main_nodes - self._iteration_main_start
        iteration_q = self._ordinary_q_nodes - (
            self._iteration_q_start - self._iteration_q_evasion_start
        )
        return iteration_q >= max(
            self.tuning.q_min_iteration,
            self.tuning.q_to_main_ratio * max(1, iteration_main),
        )


def _score_to_tt(score: int, ply: int) -> int:
    if score >= MATE - MATE_WINDOW:
        return score + ply
    if score <= -MATE + MATE_WINDOW:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE - MATE_WINDOW:
        return score - ply
    if score <= -MATE + MATE_WINDOW:
        return score + ply
    return score


__all__ = [
    "AlphaBetaSearch",
    "Bound",
    "INF",
    "MATE",
    "NodeType",
    "SearchCancelled",
    "SearchResult",
    "SearchTuning",
    "TranspositionEntry",
]
