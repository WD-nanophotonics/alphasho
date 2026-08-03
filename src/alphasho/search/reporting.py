"""Shared progress and result values for desktop search players."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoveSearchInfo:
    elapsed_seconds: float
    budget_seconds: float | None
    playouts: int
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]
    main_nodes: int = 0
    quiescence_nodes: int = 0
    tt_hits: int = 0
    beta_cutoffs: int = 0
    nodes_per_second: float = 0.0
    completed_root_moves: int = 0
    used_fallback: bool = False
    quiescence_truncations: int = 0
    attempted_depth: int = 0
    fallback_reason: str | None = None
    deadline_overshoot_ms: float = 0.0
    q_depth_truncations: int = 0
    q_leaf_truncations: int = 0
    q_budget_truncations: int = 0
    q_evasion_nodes: int = 0
    move_ordering_seconds: float = 0.0
    legal_generation_seconds: float = 0.0
    checking_probe_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    evaluation_calls: int = 0
    ordering_calls: int = 0
    ordered_moves: int = 0
    setup_elapsed_seconds: float = 0.0
    position_history_plies: int = 0
    timeout_stage: str | None = None
    incremental_evaluation_calls: int = 0
    full_evaluation_calls: int = 0
    lazy_beta_skips: int = 0
    lazy_alpha_skips: int = 0
    cpu_elapsed_seconds: float = 0.0
    cpu_evaluation_seconds: float = 0.0
    cpu_ordering_seconds: float = 0.0
    cpu_see_seconds: float = 0.0
    cpu_tt_seconds: float = 0.0
    cpu_search_overhead_seconds: float = 0.0
    move_picker_generated: int = 0
    move_picker_yielded: int = 0
    move_picker_yielded_by_stage: dict[str, int] | None = None
    full_evaluation_bound_skips: int = 0
    occupied_piece_iterations: int = 0
    mate_status: str | None = None
    mate_nodes: int = 0
    mate_elapsed_seconds: float = 0.0
    mate_principal_variation: tuple[str, ...] = ()
    scheduler_wait_seconds: float = 0.0
    timing_cause: str | None = None
    classification_seconds: float = 0.0
    sorting_seconds: float = 0.0
    pvs_null_window_searches: int = 0
    pvs_researches: int = 0
    aspiration_fail_low: int = 0
    aspiration_fail_high: int = 0
    aspiration_researches: int = 0
    see_queries: int = 0
    see_prunes: int = 0
    lmr_reductions: int = 0
    lmr_researches: int = 0
    null_move_attempts: int = 0
    null_move_cutoffs: int = 0
    null_move_verifications: int = 0
    razor_prunes: int = 0
    reverse_futility_prunes: int = 0
    move_picker_stage_counts: dict[str, int] | None = None
    move_picker_stage_seconds: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class PlayerSearchProgress:
    completed_playouts: int
    max_playouts: int
    elapsed_seconds: float
    budget_seconds: float | None
    max_depth: int
    root_value: float
    principal_variation: tuple[str, ...]
    main_nodes: int = 0
    quiescence_nodes: int = 0
    tt_hits: int = 0
    beta_cutoffs: int = 0
    nodes_per_second: float = 0.0
    completed_root_moves: int = 0
    used_fallback: bool = False
    quiescence_truncations: int = 0
    attempted_depth: int = 0
    fallback_reason: str | None = None
    deadline_overshoot_ms: float = 0.0
    q_depth_truncations: int = 0
    q_leaf_truncations: int = 0
    q_budget_truncations: int = 0
    q_evasion_nodes: int = 0
    move_ordering_seconds: float = 0.0
    legal_generation_seconds: float = 0.0
    checking_probe_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    evaluation_calls: int = 0
    ordering_calls: int = 0
    ordered_moves: int = 0
    setup_elapsed_seconds: float = 0.0
    position_history_plies: int = 0
    timeout_stage: str | None = None
    incremental_evaluation_calls: int = 0
    full_evaluation_calls: int = 0
    lazy_beta_skips: int = 0
    lazy_alpha_skips: int = 0
    cpu_elapsed_seconds: float = 0.0
    cpu_evaluation_seconds: float = 0.0
    cpu_ordering_seconds: float = 0.0
    cpu_see_seconds: float = 0.0
    cpu_tt_seconds: float = 0.0
    cpu_search_overhead_seconds: float = 0.0
    move_picker_generated: int = 0
    move_picker_yielded: int = 0
    move_picker_yielded_by_stage: dict[str, int] | None = None
    full_evaluation_bound_skips: int = 0
    occupied_piece_iterations: int = 0
    mate_status: str | None = None
    mate_nodes: int = 0
    mate_elapsed_seconds: float = 0.0
    mate_principal_variation: tuple[str, ...] = ()
    scheduler_wait_seconds: float = 0.0
    timing_cause: str | None = None
    classification_seconds: float = 0.0
    sorting_seconds: float = 0.0
    pvs_null_window_searches: int = 0
    pvs_researches: int = 0
    aspiration_fail_low: int = 0
    aspiration_fail_high: int = 0
    aspiration_researches: int = 0
    see_queries: int = 0
    see_prunes: int = 0
    lmr_reductions: int = 0
    lmr_researches: int = 0
    null_move_attempts: int = 0
    null_move_cutoffs: int = 0
    null_move_verifications: int = 0
    razor_prunes: int = 0
    reverse_futility_prunes: int = 0
    move_picker_stage_counts: dict[str, int] | None = None
    move_picker_stage_seconds: dict[str, float] | None = None


__all__ = ["MoveSearchInfo", "PlayerSearchProgress"]
