"""Optional paired games between heuristic ABP and a local YaneuraOu USI engine."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from alphasho.domain import Color
from alphasho.engine import ShogiGame
from benchmarks.heuristic_strength import (
    DEFAULT_OPENINGS,
    AblationProfile,
    _choose,
    load_openings,
    make_player,
)
from benchmarks.yaneuraou_usi import YaneuraOuUsi


@dataclass(frozen=True, slots=True)
class MatchGame:
    opening: str
    candidate_color: str
    result: str
    reason: str
    plies: int
    moves: tuple[dict[str, object], ...]


def _play(
    engine: YaneuraOuUsi,
    *,
    opening_name: str,
    opening_sfen: str,
    candidate_color: Color,
    profile: AblationProfile,
    seconds: float,
    oracle_nodes: int,
    max_plies: int,
) -> MatchGame:
    game = ShogiGame(opening_sfen)
    candidate = make_player(profile, seconds)
    rows: list[dict[str, object]] = []
    engine.new_game()
    for _ply in range(max_plies):
        snapshot = game.snapshot()
        if snapshot.outcome.is_over:
            break
        started = time.monotonic()
        if snapshot.turn is candidate_color:
            move, elapsed = _choose(candidate, game)
            info = candidate.last_search_info
            rows.append(
                {
                    "player": "candidate",
                    "usi": move.usi,
                    "elapsed_seconds": elapsed,
                    "depth": int(getattr(info, "max_depth", 0)),
                    "nodes": int(getattr(info, "playouts", 0)),
                    "fallback": bool(getattr(info, "used_fallback", False)),
                }
            )
            game.play(move)
        else:
            result = engine.analyse(
                snapshot.initial_sfen,
                nodes=oracle_nodes,
                moves=snapshot.move_history,
            )
            elapsed = time.monotonic() - started
            if result.bestmove is None:
                game.resign()
                rows.append(
                    {
                        "player": "yaneuraou",
                        "usi": "resign",
                        "elapsed_seconds": elapsed,
                    }
                )
                break
            rows.append(
                {
                    "player": "yaneuraou",
                    "usi": result.bestmove,
                    "elapsed_seconds": elapsed,
                    "oracle_nodes": oracle_nodes,
                    "pv": result.analyses[0].principal_variation
                    if result.analyses
                    else (),
                }
            )
            game.play(result.bestmove)
    snapshot = game.snapshot()
    if not snapshot.outcome.is_over:
        result_name = "unresolved"
        reason = "benchmark_move_limit"
    elif snapshot.outcome.winner is None:
        result_name = "rule_draw"
        reason = snapshot.outcome.reason.value
    else:
        result_name = (
            "candidate_win"
            if snapshot.outcome.winner is candidate_color
            else "oracle_win"
        )
        reason = snapshot.outcome.reason.value
    return MatchGame(
        opening=opening_name,
        candidate_color=candidate_color.name,
        result=result_name,
        reason=reason,
        plies=len(rows),
        moves=tuple(rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--profile", choices=[p.value for p in AblationProfile], default="full")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--oracle-nodes", type=int, default=100_000)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    openings = load_openings(args.openings, limit=args.positions)
    games: list[MatchGame] = []
    with YaneuraOuUsi(args.engine) as engine:
        oracle = {"sha256": engine.binary_sha256, "ids": engine.ids}
        for opening in openings:
            for color in (Color.BLACK, Color.WHITE):
                game = _play(
                    engine,
                    opening_name=opening.name,
                    opening_sfen=opening.sfen,
                    candidate_color=color,
                    profile=AblationProfile(args.profile),
                    seconds=args.seconds,
                    oracle_nodes=args.oracle_nodes,
                    max_plies=args.max_plies,
                )
                games.append(game)
                print(
                    f"MATCH {opening.name} candidate={color.name} result={game.result}",
                    flush=True,
                )
    report = {
        "oracle": oracle,
        "candidate_profile": args.profile,
        "seconds": args.seconds,
        "oracle_nodes": args.oracle_nodes,
        "games": [asdict(game) for game in games],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MATCH REPORT SAVED {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
