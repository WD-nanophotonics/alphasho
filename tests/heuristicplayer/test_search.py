import random
import threading
import time
from collections import Counter

import cshogi
import pytest

from alphasho.domain import (
    ClockState,
    ThinkingConfig,
    ThinkingPreset,
    ThinkingStrategy,
    TimeControl,
)
from alphasho.engine import ShogiGame
from alphasho.heuristicplayer import HeuristicPlayer
from alphasho.heuristicplayer.evaluation import (
    MAX_FULL_RESIDUAL,
    EvaluationMode,
    evaluate_breakdown,
    evaluate_breakdown_from_absolute,
    evaluate_fast_absolute,
    evaluate_position,
)
from alphasho.heuristicplayer.move_ordering import MoveOrdering
from alphasho.heuristicplayer.player import _rebuild_search_position
from alphasho.heuristicplayer.search import (
    MATE,
    AlphaBetaSearch,
    Bound,
    SearchTuning,
    _Position,
    _score_from_tt,
    _score_to_tt,
)

_FAILED_AUDIT_INITIAL_SFEN = (
    "lns4nl/1r1gksgb1/pp1p1pppp/2p1p4/9/1P4P2/"
    "P1PPPPNPP/1B2G1SR1/LNS1KG2L b - 13"
)
_FAILED_AUDIT_HISTORY = tuple(
    "7g7f 3c3d 8h2b+ 3b2b B*4f 8a7c 1g1f B*4h 5i4h 4c4d "
    "B*5f 5b4c 5f7d 1c1d 2g2f 9c9d 3f3e 3d3e 4f3e P*3c "
    "P*3f 3c3d 3e4f 9d9e 3f3e 3d3e 4f3e P*3c P*3f 3c3d "
    "3e4f 1a1b 3f3e 3d3e 4f3e P*3c P*3f 3c3d 3e4f 8c8d "
    "3g2e 4d4e 4f3g 2c2d 2e1c+ 2a1c 7d6e 7c6e 6g6f 6e5g+ "
    "4h5g B*3i 4i3i N*5e P*5f 9e9f 5f5e 9f9g+ N*7d 1c2e "
    "2f2e 8b7b B*8c 7b7c N*3e".split()
)


def _fixed(nodes: int, depth: int = 8) -> ThinkingConfig:
    return ThinkingConfig(
        strategy=ThinkingStrategy.FIXED_PLAYOUT,
        preset=ThinkingPreset.CUSTOM,
        max_playouts=nodes,
        max_depth=depth,
    )


def test_completed_search_reports_root_pv_and_diagnostics() -> None:
    game = ShogiGame()
    player = HeuristicPlayer(_fixed(2_000))

    move = player.choose_move(game.snapshot(), game.legal_moves())

    info = player.last_search_info
    assert info is not None
    assert info.principal_variation
    assert info.principal_variation[0] == move.usi
    assert info.playouts == info.main_nodes + info.quiescence_nodes
    assert info.playouts <= 2_000
    assert info.quiescence_nodes > 0
    assert info.nodes_per_second > 0
    assert info.attempted_depth >= info.max_depth
    assert not info.used_fallback
    assert info.quiescence_truncations >= 0
    assert info.evaluation_calls > 0
    assert info.evaluation_seconds >= 0.0
    assert info.legal_generation_seconds >= 0.0
    assert info.checking_probe_seconds == 0.0
    assert info.ordering_calls > 0
    assert info.ordered_moves > 0
    assert info.incremental_evaluation_calls > 0
    assert info.cpu_elapsed_seconds >= 0.0
    assert info.cpu_evaluation_seconds >= 0.0
    assert info.scheduler_wait_seconds >= 0.0


def test_incremental_fast_matches_reference_across_push_and_pop() -> None:
    randomizer = random.Random(20260802)
    board = cshogi.Board()
    position = _Position(board, Counter({int(board.zobrist_hash()): 1}))
    expected_stack: list[tuple[int, int]] = []

    for _ply in range(120):
        assert (position.material, position.positional_raw) == evaluate_fast_absolute(board)
        incremental = evaluate_breakdown_from_absolute(
            board,
            position.material,
            position.positional_raw,
            mode=EvaluationMode.FAST,
        ).total
        assert incremental == evaluate_position(board, EvaluationMode.FAST)
        occupied_full = evaluate_breakdown_from_absolute(
            board,
            position.material,
            position.positional_raw,
            mode=EvaluationMode.KING_SAFETY,
            occupied_mask=position.evaluator_state.occupied_mask,
        )
        assert occupied_full == evaluate_breakdown(board)
        moves = list(board.legal_moves)
        if not moves:
            break
        expected_stack.append((position.material, position.positional_raw))
        position.push(int(randomizer.choice(moves)))

    while expected_stack:
        position.pop()
        assert (position.material, position.positional_raw) == expected_stack.pop()


def test_lazy_full_bounds_are_mathematically_safe() -> None:
    board = cshogi.Board()
    search = AlphaBetaSearch(
        board,
        deadline=None,
        node_limit=10_000,
        cancelled=threading.Event(),
        budget_seconds=None,
        tuning=SearchTuning(use_lazy_full=True),
    )
    fast = search._evaluate_incremental(EvaluationMode.FAST)
    exact = evaluate_position(board, EvaluationMode.KING_SAFETY)

    lower_score, lower_bound = search._bounded_static(
        -1_000_000, fast - MAX_FULL_RESIDUAL
    )
    upper_score, upper_bound = search._bounded_static(
        fast + MAX_FULL_RESIDUAL, 1_000_000
    )

    assert lower_bound is Bound.LOWER
    assert exact >= lower_score
    assert upper_bound is Bound.UPPER
    assert exact <= upper_score
    assert search.lazy_beta_skips == 1
    assert search.lazy_alpha_skips == 1


def test_color_rotated_evaluation_has_same_side_to_move_score() -> None:
    black = cshogi.Board("4k4/9/9/9/9/9/4P4/9/4K4 b - 1")
    white = cshogi.Board("4k4/9/4p4/9/9/9/9/9/4K4 w - 1")

    assert evaluate_position(black) == evaluate_position(white)


def test_pawn_shield_improves_king_safety_component() -> None:
    shielded = cshogi.Board("4k4/9/9/9/9/9/9/3PPP3/4K4 b - 1")
    exposed = cshogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")

    assert evaluate_breakdown(shielded).king_safety > evaluate_breakdown(exposed).king_safety


def test_mate_scores_round_trip_through_transposition_table() -> None:
    for score in (MATE - 7, -MATE + 7, 123, -456):
        stored = _score_to_tt(score, 5)
        assert _score_from_tt(stored, 5) == score


def test_material_ablation_matches_historical_static_evaluation() -> None:
    from benchmarks.legacy_3262cc8 import LegacyHeuristicPlayer

    for sfen in (
        ShogiGame().snapshot().sfen,
        "4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1",
    ):
        board = cshogi.Board(sfen)
        assert evaluate_position(board, EvaluationMode.MATERIAL) == LegacyHeuristicPlayer._evaluate(
            board
        )


def test_killer_move_leads_other_quiet_moves() -> None:
    board = cshogi.Board()
    ordering = MoveOrdering()
    target = next(move for move in ordering.ordered(board, ply=2) if move.usi == "7g7f")

    ordering.record_cutoff(target, ply=2, depth=4)
    ordered = ordering.ordered(board, ply=2)

    assert ordered[0].usi == "7g7f"
    assert ordering.history


def test_move_ordering_never_pushes_or_probes_check() -> None:
    real = cshogi.Board()

    class _OrderingBoard:
        legal_moves = [int(move) for move in real.legal_moves]

        def piece(self, square: int) -> int:
            return int(real.piece(square))

        def push(self, _move: int) -> None:
            raise AssertionError("ordering must not push moves")

        def is_check(self) -> bool:
            raise AssertionError("ordering must not probe check")

    ordered = MoveOrdering().ordered(  # type: ignore[arg-type]
        _OrderingBoard(), ply=0, include_checks=True
    )

    assert ordered
    assert all(not move.gives_check for move in ordered)


def test_checked_quiescence_searches_evasions_and_returns_legal_move() -> None:
    game = ShogiGame("4k4/9/9/4r4/9/9/9/9/4K4 b - 1")
    board = cshogi.Board(game.snapshot().sfen)
    search = AlphaBetaSearch(
        board,
        deadline=None,
        node_limit=2_000,
        cancelled=threading.Event(),
        budget_seconds=None,
    )

    result = search.run(fallback_usi=game.legal_moves()[0].usi, max_depth=2)

    assert result.best_move in {move.usi for move in game.legal_moves()}
    assert result.quiescence_nodes > 0
    assert result.q_evasion_nodes > 0


def test_completed_root_baseline_survives_later_node_timeout() -> None:
    game = ShogiGame()
    search = AlphaBetaSearch(
        cshogi.Board(game.snapshot().sfen),
        deadline=None,
        node_limit=50,
        cancelled=threading.Event(),
        budget_seconds=None,
    )

    result = search.run(fallback_usi=game.legal_moves()[0].usi, max_depth=8)

    assert result.completed_depth == 1
    assert result.attempted_depth == 2
    assert not result.used_fallback
    assert result.fallback_reason is None
    assert result.principal_variation[0] == result.best_move


def test_root_tactical_scan_selects_any_mate_in_one() -> None:
    game = ShogiGame("8k/9/7G1/9/9/9/9/9/K8 b R 1")
    player = HeuristicPlayer(_fixed(500, depth=8))

    move = player.choose_move(game.snapshot(), game.legal_moves())

    assert move.usi in {f"R*{file_index}a" for file_index in range(3, 10)}
    info = player.last_search_info
    assert info is not None
    assert info.root_value == 1.0
    assert info.max_depth == 1


def test_pvs_and_selective_profiles_keep_root_move_at_fixed_depth() -> None:
    game = ShogiGame()
    baseline = AlphaBetaSearch(
        cshogi.Board(),
        deadline=None,
        node_limit=1_000_000,
        cancelled=threading.Event(),
        budget_seconds=None,
        tuning=SearchTuning(),
    ).run(fallback_usi=game.legal_moves()[0].usi, max_depth=3)
    candidate = AlphaBetaSearch(
        cshogi.Board(),
        deadline=None,
        node_limit=1_000_000,
        cancelled=threading.Event(),
        budget_seconds=None,
        tuning=SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
        ),
    ).run(fallback_usi=game.legal_moves()[0].usi, max_depth=3)

    assert candidate.best_move == baseline.best_move
    assert candidate.completed_depth == baseline.completed_depth == 3
    assert candidate.pvs_null_window_searches > 0
    assert candidate.pvs_researches <= candidate.pvs_null_window_searches


def test_packed_hotpath_matches_legacy_ordering_without_selective_search() -> None:
    game = ShogiGame()
    results = []
    for mode in ("legacy", "packed"):
        results.append(
            AlphaBetaSearch(
                cshogi.Board(),
                deadline=None,
                node_limit=1_000_000,
                cancelled=threading.Event(),
                budget_seconds=None,
                tuning=SearchTuning(
                    use_quiescence=False,
                    move_picker_mode=mode,
                    evaluation_schedule="lazy_full",
                ),
            ).run(fallback_usi=game.legal_moves()[0].usi, max_depth=3)
        )

    legacy, packed = results
    assert packed.best_move == legacy.best_move
    assert packed.score == legacy.score
    assert packed.principal_variation == legacy.principal_variation
    assert packed.move_picker_generated >= packed.move_picker_yielded > 0


def test_null_move_does_not_change_repetition_counts() -> None:
    board = cshogi.Board()
    position = _Position(board, Counter({int(board.zobrist_hash()): 1}))
    original_counts = position.counts.copy()
    original_fast = (position.material, position.positional_raw)

    position.push_pass()
    assert position.null_plies == 1
    assert position.counts == original_counts
    position.pop_pass()

    assert position.null_plies == 0
    assert position.counts == original_counts
    assert (position.material, position.positional_raw) == original_fast


def test_conservative_see_rejects_unrecapturable_rook_for_pawn_trade() -> None:
    board = cshogi.Board("8k/9/4g4/4p4/4R4/9/9/9/K8 b - 1")
    search = AlphaBetaSearch(
        board,
        deadline=None,
        node_limit=10_000,
        cancelled=threading.Event(),
        budget_seconds=None,
        tuning=SearchTuning(use_see=True, use_staged_move_picker=True),
    )
    move = next(item for item in search._ordered(ply=0) if item.usi == "5e5d")

    assert search._see_ordering_score(move) < -100
    assert not search._see_ge(move, -100)
    assert search.see_queries == 1


def test_expired_entry_deadline_returns_fallback_instead_of_raising() -> None:
    game = ShogiGame()
    fallback = game.legal_moves()[0].usi
    search = AlphaBetaSearch(
        cshogi.Board(game.snapshot().sfen),
        deadline=time.monotonic() - 1.0,
        node_limit=10_000,
        cancelled=threading.Event(),
        budget_seconds=1.0,
        setup_elapsed_seconds=1.0,
    )

    result = search.run(fallback_usi=fallback, max_depth=8)

    assert result.best_move == fallback
    assert result.principal_variation == (fallback,)
    assert result.completed_depth == 0
    assert result.used_fallback
    assert result.fallback_reason == "position_setup_deadline"
    assert result.timeout_stage == "position_setup"


def test_snapshot_rebuild_precomputes_historical_repetition_counts() -> None:
    game = ShogiGame("8k/9/9/9/9/9/9/9/K8 b - 1")
    for usi in ("9i9h", "1a1b", "9h9i", "1b1a") * 2:
        game.play(usi)

    rebuilt = _rebuild_search_position(game.snapshot())
    clone = rebuilt.board.copy()
    expected: Counter[int] = Counter()
    expected[int(clone.zobrist_hash())] += 1
    while clone.history:
        clone.pop()
        expected[int(clone.zobrist_hash())] += 1

    assert rebuilt.history_plies == 8
    assert rebuilt.repetition_counts == expected


def test_repetition_probe_is_skipped_before_four_occurrences() -> None:
    game = ShogiGame()
    search = AlphaBetaSearch(
        cshogi.Board(game.snapshot().sfen),
        deadline=None,
        node_limit=10_000,
        cancelled=threading.Event(),
        budget_seconds=None,
    )

    class _FakeBoard:
        draw_calls = 0

        def zobrist_hash(self) -> int:
            return 42

        def is_draw(self) -> int:
            self.draw_calls += 1
            return 0

    fake = _FakeBoard()
    search.position.board = fake  # type: ignore[assignment]
    search.position.counts = Counter({42: 3})

    assert search._rule_terminal_score(0, baseline=True) is None
    assert fake.draw_calls == 0


def test_failed_audit_long_history_does_not_leak_timeout() -> None:
    game = ShogiGame(_FAILED_AUDIT_INITIAL_SFEN)
    for usi in _FAILED_AUDIT_HISTORY:
        game.play(usi)
    snapshot = game.snapshot()
    player = HeuristicPlayer(
        ThinkingConfig(
            strategy=ThinkingStrategy.FIXED_TIME,
            preset=ThinkingPreset.CUSTOM,
            move_time_seconds=1.0,
            max_playouts=10_000_000,
            max_depth=128,
            safety_margin_ms=0,
        )
    )

    move = player.choose_move_with_context(
        snapshot,
        game.legal_moves(),
        clock_state=ClockState(
            (3_600_000, 3_600_000),
            (0, 0),
            snapshot.turn,
            True,
        ),
        time_control=TimeControl(),
        color=snapshot.turn,
    )

    info = player.last_search_info
    assert move in game.legal_moves()
    assert info is not None
    assert info.position_history_plies == len(_FAILED_AUDIT_HISTORY)
    assert info.setup_elapsed_seconds >= 0.0
    assert info.max_depth >= 1
    assert not info.used_fallback


@pytest.mark.parametrize(
    "sfen",
    (
        "lnsg1g1nl/2r4s1/p2pkp1pp/1pp2Pp2/9/1bP3PB1/"
        "P2PP2PP/4G3R/LNSK1GS1L w N2P 34",
        "l5Snl/1s1g1k3/pp1p1pgpp/4p4/2p6/1+bb4RP/"
        "P3PPNP1/4GKSR1/LNS5L w gn5p 46",
        "lnsgkg2l/2r4bn/pp1pp1sp1/2p3p1l/8R/4PPPP1/"
        "PPPP1K3/1B3G1S1/LNS2G3 b N3P 41",
        "2kg4l/2s6/ppnpb3p/9/9/5PG2/P1+b1P1KPP/"
        "3+r4R/2+p2GS1L w N2L5Pg2s2n3p 64",
    ),
)
def test_audit_regressions_complete_depth_and_respect_short_deadline(sfen: str) -> None:
    game = ShogiGame(sfen)
    player = HeuristicPlayer(
        ThinkingConfig(
            strategy=ThinkingStrategy.FIXED_TIME,
            preset=ThinkingPreset.CUSTOM,
            move_time_seconds=0.1,
            max_playouts=10_000_000,
            max_depth=128,
            safety_margin_ms=0,
        )
    )

    snapshot = game.snapshot()
    started = time.monotonic()
    player.choose_move_with_context(
        snapshot,
        game.legal_moves(),
        clock_state=ClockState(
            (3_600_000, 3_600_000),
            (0, 0),
            snapshot.turn,
            True,
        ),
        time_control=TimeControl(),
        color=snapshot.turn,
    )
    elapsed = time.monotonic() - started

    info = player.last_search_info
    assert info is not None
    assert info.max_depth >= 1
    assert not info.used_fallback
    assert elapsed <= 0.15
    assert info.deadline_overshoot_ms <= 50.0
    assert info.q_budget_truncations >= 0


def test_regression_complex_position_marks_incomplete_root_as_fallback() -> None:
    game = ShogiGame(
        "ln1g2sn1/2s2kg2/9/8R/pPP2PP2/4P2P1/"
        "P2P5/1R3SG2/L3GK3 b 2BS2N2L9P 87"
    )
    player = HeuristicPlayer(_fixed(256, depth=128))

    player.choose_move(game.snapshot(), game.legal_moves())

    info = player.last_search_info
    assert info is not None
    assert info.max_depth == 0
    assert info.used_fallback
    assert info.completed_root_moves > 0
    assert info.fallback_reason == "root_scan_node_limit"
    assert info.principal_variation[0] in {move.usi for move in game.legal_moves()}


def test_quiescence_limit_is_visible_in_diagnostics() -> None:
    game = ShogiGame("4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1")
    player = HeuristicPlayer(_fixed(10_000, depth=8))

    player.choose_move(game.snapshot(), game.legal_moves())

    info = player.last_search_info
    assert info is not None
    assert info.quiescence_truncations > 0
    assert info.quiescence_truncations == (
        info.q_depth_truncations
        + info.q_leaf_truncations
        + info.q_budget_truncations
    )


@pytest.mark.parametrize(
    ("sfen", "expected"),
    (
        (
            "2g1kgsn1/l2s3Pl/pp1ppp3/2p5p/3n3p1/P3PlP2/"
            "BPPSr3P/5K3/LN1G1GSN+b b RP2p 45",
            ("R*4g",),
        ),
        (
            "1ng1g1k1R/l4s2+S/pp2ppr1p/3P2pB1/5+B2P/P2N2PPL/"
            "1P+p1L4/5GS2/LN1GK4 w SN2P3p 78",
            ("P*2a", "3a3b"),
        ),
    ),
)
def test_opening_eight_forced_loss_regressions_are_stable(
    sfen: str, expected: tuple[str, ...]
) -> None:
    game = ShogiGame(sfen)
    player = HeuristicPlayer(_fixed(10_000))

    move = player.choose_move(game.snapshot(), game.legal_moves())

    info = player.last_search_info
    assert move.usi in expected
    assert info is not None
    assert info.root_value == -1.0
    assert info.max_depth >= 2
    assert not info.used_fallback
