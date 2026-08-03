"""Deterministic benchmark for the bounded short-mate proof search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cshogi

from alphasho.heuristicplayer.mate_search import BoundedDfpnMateSolver

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "configs" / "benchmark" / "mate_positions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--node-limit", type=int, default=10_000)
    parser.add_argument("--milliseconds", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/abp_mate_solver.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for line in args.corpus.read_text(encoding="utf-8").splitlines():
        expected = json.loads(line)
        board = cshogi.Board(expected["sfen"])
        result = BoundedDfpnMateSolver(
            board,
            max_ply=int(expected["max_ply"]),
            node_limit=args.node_limit,
            deadline=time.monotonic() + args.milliseconds / 1_000.0,
        ).solve()
        first = result.principal_variation[0] if result.principal_variation else None
        solved = result.status.value == expected["expected_status"] and (
            result.status.value != "proven" or first in expected["legal_first_moves"]
        )
        row = {
            "name": expected["name"],
            "expected": expected["expected_status"],
            "status": result.status.value,
            "solved": solved,
            "nodes": result.nodes,
            "elapsed_seconds": result.elapsed_seconds,
            "pv": result.principal_variation,
            "proof_number": result.proof_number,
            "disproof_number": result.disproof_number,
        }
        rows.append(row)
        print(
            f"MATE {expected['name']} status={result.status.value} "
            f"nodes={result.nodes} elapsed_ms={result.elapsed_seconds * 1_000:.3f}",
            flush=True,
        )
    report = {
        "corpus": str(args.corpus.resolve()),
        "node_limit": args.node_limit,
        "milliseconds": args.milliseconds,
        "positions": len(rows),
        "solved": sum(bool(row["solved"]) for row in rows),
        "false_proofs": sum(
            row["status"] == "proven" and row["expected"] != "proven" for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MATE REPORT SAVED {args.output.resolve()}", flush=True)
    return 0 if report["solved"] == report["positions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

