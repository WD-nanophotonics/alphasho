from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cshogi

from alphasho.heuristicplayer.mate_search import BoundedDfpnMateSolver, MateStatus
from benchmarks.yaneuraou_usi import parse_info, sha256_file


def test_parse_multipv_score_and_pv() -> None:
    parsed = parse_info(
        "info depth 17 nodes 123456 multipv 2 score cp -87 upperbound "
        "pv 7g7f 3c3d"
    )
    assert parsed is not None
    assert parsed.multipv == 2
    assert parsed.score_cp == -87
    assert parsed.score_mate is None
    assert parsed.upperbound
    assert parsed.principal_variation == ("7g7f", "3c3d")


def test_parse_mate_score() -> None:
    parsed = parse_info("info depth 9 score mate 5 pv G*5b 4a5b R*4b")
    assert parsed is not None
    assert parsed.score_mate == 5


def test_sha256_file() -> None:
    target = Path(__file__)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_frozen_mate_corpus_is_legal_and_proven() -> None:
    corpus = Path("configs/benchmark/mate_positions.jsonl")
    for line in corpus.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        board = cshogi.Board(row["sfen"])
        result = BoundedDfpnMateSolver(
            board,
            max_ply=int(row["max_ply"]),
            node_limit=50_000,
            deadline=None,
        ).solve()
        assert result.status.value == row["expected_status"], row["name"]
        if result.status is MateStatus.PROVEN:
            assert len(result.principal_variation) == row["proven_ply"]
            assert result.principal_variation[0] in row["legal_first_moves"]
