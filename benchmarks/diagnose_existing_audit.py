"""Re-search decisive games from an existing ABP audit without resuming it."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from heuristic_strength import diagnose_loss, load_completed_games


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--openings", type=int, default=2)
    parser.add_argument("--base-seconds", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, nargs="+", default=(1.0, 5.0))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    games = load_completed_games(args.run_dir / "games.jsonl")
    opening_names = list(dict.fromkeys(game.opening for game in games))[: args.openings]
    selected = [
        game
        for game in games
        if game.opening in opening_names and game.outcome_kind == "decisive"
    ]
    output = args.output or args.run_dir / "diagnosis-existing.json"
    report: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": str(args.run_dir.resolve()),
        "opening_names": opening_names,
        "base_seconds": args.base_seconds,
        "research_seconds": args.seconds,
        "games": [],
    }
    rows: list[dict[str, object]] = []
    for index, game in enumerate(selected, start=1):
        print(
            f"DIAGNOSE {index}/{len(selected)} {game.opening}/{game.candidate_color} "
            f"winner={game.winner}",
            flush=True,
        )
        rows.append(
            {
                "opening": game.opening,
                "candidate_color": game.candidate_color,
                "winner": game.winner,
                "candidate_won": game.winner == game.candidate_color,
                "diagnosis": diagnose_loss(
                    game,
                    base_seconds=args.base_seconds,
                    seconds=tuple(args.seconds),
                ),
            }
        )
    report["games"] = rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DIAGNOSIS SAVED {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
