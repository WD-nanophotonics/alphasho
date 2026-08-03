"""Reproducible fixed-position search-efficiency ablation for heuristic ABP."""

from __future__ import annotations

import argparse
import atexit
import json
import statistics
from pathlib import Path

from heuristic_strength import (
    DEFAULT_OPENINGS,
    AblationProfile,
    ProcessPriority,
    _choose,
    load_openings,
    make_player,
)

from alphasho.engine import ShogiGame

DEFAULT_PROFILES = (
    AblationProfile.HOTPATH_BASELINE,
    AblationProfile.PVS_HOTPATH,
    AblationProfile.SEE_HOTPATH,
    AblationProfile.LMR_HOTPATH,
    AblationProfile.MATE_PROBE,
    AblationProfile.NEXT_FULL_CANDIDATE,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=[profile.value for profile in AblationProfile],
        default=[profile.value for profile in DEFAULT_PROFILES],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/abp_search_efficiency_ablation.json"),
    )
    parser.add_argument("--normal-priority", action="store_true")
    args = parser.parse_args()

    priority = ProcessPriority(
        requested="normal" if args.normal_priority else "above_normal"
    )
    if not args.normal_priority:
        priority.apply()
        atexit.register(priority.restore)

    openings = load_openings(args.openings, limit=args.positions)
    report: dict[str, object] = {
        "seconds": args.seconds,
        "positions": [opening.name for opening in openings],
        "process_priority": {
            "requested": priority.requested,
            "applied": priority.applied,
            "original": priority.original,
            "error": priority.error,
        },
        "profiles": {},
    }
    profile_results: dict[str, object] = {}
    for profile_name in args.profiles:
        profile = AblationProfile(profile_name)
        print(f"PROFILE START {profile.value}", flush=True)
        rows: list[dict[str, object]] = []
        for index, opening in enumerate(openings, start=1):
            game = ShogiGame(opening.sfen)
            player = make_player(profile, args.seconds)
            move, elapsed = _choose(player, game)
            info = player.last_search_info
            nodes = int(getattr(info, "playouts", 0))
            main = int(getattr(info, "main_nodes", nodes))
            qnodes = int(getattr(info, "quiescence_nodes", 0))
            row = {
                "opening": opening.name,
                "move": move.usi,
                "depth": int(getattr(info, "max_depth", 0)),
                "attempted_depth": int(getattr(info, "attempted_depth", 0)),
                "nodes": nodes,
                "main_nodes": main,
                "quiescence_nodes": qnodes,
                "elapsed_seconds": elapsed,
                "nps": nodes / max(elapsed, 1e-9),
                "fallback": bool(getattr(info, "used_fallback", False)),
                "timing_cause": getattr(info, "timing_cause", None),
                "timing_invalid": getattr(info, "timing_cause", None) is not None,
                "cpu_ordering_seconds": float(
                    getattr(info, "cpu_ordering_seconds", 0.0)
                ),
                "cpu_evaluation_seconds": float(
                    getattr(info, "cpu_evaluation_seconds", 0.0)
                ),
                "cpu_see_seconds": float(getattr(info, "cpu_see_seconds", 0.0)),
                "cpu_tt_seconds": float(getattr(info, "cpu_tt_seconds", 0.0)),
                "cpu_search_overhead_seconds": float(
                    getattr(info, "cpu_search_overhead_seconds", 0.0)
                ),
                "pvs_null_window_searches": int(
                    getattr(info, "pvs_null_window_searches", 0)
                ),
                "pvs_researches": int(getattr(info, "pvs_researches", 0)),
                "see_queries": int(getattr(info, "see_queries", 0)),
                "see_prunes": int(getattr(info, "see_prunes", 0)),
                "lmr_reductions": int(getattr(info, "lmr_reductions", 0)),
                "lmr_researches": int(getattr(info, "lmr_researches", 0)),
                "null_move_attempts": int(
                    getattr(info, "null_move_attempts", 0)
                ),
                "null_move_cutoffs": int(getattr(info, "null_move_cutoffs", 0)),
                "razor_prunes": int(getattr(info, "razor_prunes", 0)),
                "reverse_futility_prunes": int(
                    getattr(info, "reverse_futility_prunes", 0)
                ),
                "move_picker_generated": int(
                    getattr(info, "move_picker_generated", 0)
                ),
                "move_picker_yielded": int(
                    getattr(info, "move_picker_yielded", 0)
                ),
                "full_evaluation_bound_skips": int(
                    getattr(info, "full_evaluation_bound_skips", 0)
                ),
                "occupied_piece_iterations": int(
                    getattr(info, "occupied_piece_iterations", 0)
                ),
                "full_evaluation_calls": int(
                    getattr(info, "full_evaluation_calls", 0)
                ),
                "incremental_evaluation_calls": int(
                    getattr(info, "incremental_evaluation_calls", 0)
                ),
                "mate_status": getattr(info, "mate_status", None),
                "mate_nodes": int(getattr(info, "mate_nodes", 0)),
                "mate_elapsed_seconds": float(
                    getattr(info, "mate_elapsed_seconds", 0.0)
                ),
            }
            rows.append(row)
            print(
                f"POSITION {index}/{len(openings)} {profile.value} "
                f"depth={row['depth']} nodes={nodes} move={move.usi}",
                flush=True,
            )
        valid_rows = [row for row in rows if not row["timing_invalid"]]
        aggregate_rows = valid_rows or rows
        total_nodes = sum(int(row["nodes"]) for row in aggregate_rows)
        total_elapsed = sum(float(row["elapsed_seconds"]) for row in aggregate_rows)
        qnodes = sum(int(row["quiescence_nodes"]) for row in aggregate_rows)
        profile_results[profile.value] = {
            "valid_positions": len(valid_rows),
            "timing_invalid_positions": len(rows) - len(valid_rows),
            "median_depth": statistics.median(
                int(row["depth"]) for row in aggregate_rows
            ),
            "mean_nodes": total_nodes / len(aggregate_rows),
            "nps": total_nodes / max(total_elapsed, 1e-9),
            "q_share": qnodes / max(1, total_nodes),
            "fallbacks": sum(bool(row["fallback"]) for row in rows),
            "mean_cpu_ordering_ms": statistics.fmean(
                float(row["cpu_ordering_seconds"]) * 1_000.0
                for row in aggregate_rows
            ),
            "mean_cpu_evaluation_ms": statistics.fmean(
                float(row["cpu_evaluation_seconds"]) * 1_000.0
                for row in aggregate_rows
            ),
            "mean_cpu_see_ms": statistics.fmean(
                float(row["cpu_see_seconds"]) * 1_000.0 for row in aggregate_rows
            ),
            "mean_cpu_tt_ms": statistics.fmean(
                float(row["cpu_tt_seconds"]) * 1_000.0 for row in aggregate_rows
            ),
            "mean_cpu_search_overhead_ms": statistics.fmean(
                float(row["cpu_search_overhead_seconds"]) * 1_000.0
                for row in aggregate_rows
            ),
            "full_evaluation_calls": sum(
                int(row["full_evaluation_calls"]) for row in aggregate_rows
            ),
            "full_evaluation_bound_skips": sum(
                int(row["full_evaluation_bound_skips"]) for row in aggregate_rows
            ),
            "move_picker_generated": sum(
                int(row["move_picker_generated"]) for row in aggregate_rows
            ),
            "move_picker_yielded": sum(
                int(row["move_picker_yielded"]) for row in aggregate_rows
            ),
            "mate_proven": sum(
                row["mate_status"] == "proven" for row in aggregate_rows
            ),
            "rows": rows,
        }
    report["profiles"] = profile_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    priority.restore()
    print(f"REPORT SAVED {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
