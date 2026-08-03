from __future__ import annotations

import threading
import time

import cshogi

from alphasho.heuristicplayer.mate_search import BoundedDfpnMateSolver, MateStatus
from alphasho.heuristicplayer.move_ordering import MoveOrdering
from alphasho.heuristicplayer.search import AlphaBetaSearch, SearchTuning

MATE_THREE = (
    "l2g5/3r5/pp+R6/2p3p2/2b1NG3/2k1GP3/"
    "PP1PP3P/1S1KB4/LN3S2L b G2S2N6Pl2p 99"
)
MATE_FIVE = (
    "4s2g1/2+r6/+R1s2k1p1/1plp1p3/4p4/PP1P1+p1g1/"
    "3+bn3+l/1K5+b1/+p4g3 w Sgs3n2l8p 144"
)
MATE_SEVEN = (
    "4s2g1/2+r6/+R1s2k1p1/1plp1p3/4p4/PP1P1+p1g1/"
    "K2+bn3+l/7+b1/+p4g3 w g2s3n2l8p 142"
)


def test_bounded_mate_solver_proves_three_five_and_seven_ply() -> None:
    for sfen, depth in ((MATE_THREE, 3), (MATE_FIVE, 5), (MATE_SEVEN, 7)):
        result = BoundedDfpnMateSolver(
            cshogi.Board(sfen),
            max_ply=depth,
            node_limit=10_000,
            deadline=None,
        ).solve()
        assert result.status is MateStatus.PROVEN
        assert len(result.principal_variation) == depth


def test_bounded_mate_solver_returns_unknown_at_resource_limit() -> None:
    result = BoundedDfpnMateSolver(
        cshogi.Board(MATE_SEVEN),
        max_ply=7,
        node_limit=1,
        deadline=None,
    ).solve()
    assert result.status is MateStatus.UNKNOWN
    assert not result.principal_variation


def test_bounded_mate_solver_treats_path_repetition_as_unknown() -> None:
    board = cshogi.Board(MATE_THREE)
    solver = BoundedDfpnMateSolver(
        board,
        max_ply=3,
        node_limit=100,
        deadline=None,
    )
    proof = solver._search(  # noqa: SLF001 - direct path-state regression
        True,
        3,
        {int(board.zobrist_hash())},
    )
    assert proof.status is MateStatus.UNKNOWN


def test_bounded_mate_solver_disproves_quiet_initial_position() -> None:
    result = BoundedDfpnMateSolver(
        cshogi.Board(),
        max_ply=3,
        node_limit=10_000,
        deadline=None,
    ).solve()
    assert result.status is MateStatus.DISPROVEN


def test_bounded_mate_solver_respects_illegal_pawn_drop_mate() -> None:
    board = cshogi.Board("8k/9/6NG1/9/9/9/9/9/K8 b P 1")
    assert "P*1b" not in {cshogi.move_to_usi(int(move)) for move in board.legal_moves}
    result = BoundedDfpnMateSolver(
        board,
        max_ply=1,
        node_limit=10_000,
        deadline=None,
    ).solve()
    assert result.status is MateStatus.DISPROVEN


def test_packed_move_picker_is_complete_unique_and_history_bucketed() -> None:
    board = cshogi.Board()
    legal = [int(move) for move in board.legal_moves]
    ordering = MoveOrdering()
    target = int(board.move_from_usi("7g7f"))
    history_key = (
        int(cshogi.move_from_piece_type(target)),
        int(cshogi.move_to(target)),
    )
    ordering.history[history_key] = 10_000
    picker = ordering.packed(legal, ply=0)
    yielded = list(picker)
    assert len(yielded) == len(set(yielded)) == len(legal)
    quiet = [move for move in yielded if not cshogi.move_cap(move)]
    assert quiet[0] == target
    assert picker.generated == len(legal)
    assert picker.yielded == len(legal)


def test_search_mate_probe_returns_proven_pv_without_fallback() -> None:
    board = cshogi.Board(MATE_THREE)
    fallback = cshogi.move_to_usi(next(iter(board.legal_moves)))
    search = AlphaBetaSearch(
        board,
        deadline=time.monotonic() + 1.0,
        node_limit=1_000_000,
        cancelled=threading.Event(),
        budget_seconds=1.0,
        tuning=SearchTuning(
            use_mate_probe=True,
            mate_max_ply=3,
            mate_node_limit=10_000,
        ),
    )
    result = search.run(fallback_usi=fallback, max_depth=8)
    assert result.mate_status == MateStatus.PROVEN.value
    assert len(result.mate_principal_variation) == 3
    assert result.best_move == result.mate_principal_variation[0]
    assert not result.used_fallback
