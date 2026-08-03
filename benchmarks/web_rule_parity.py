"""Generate a deterministic cshogi oracle corpus for the static Web rules.

The generated JSONL is intentionally not committed: cshogi is a development
oracle and is never bundled into the GitHub Pages artifact.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cshogi


def generate(output: Path, positions: int, seed: int) -> None:
    randomizer = random.Random(seed)
    board = cshogi.Board()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    ply = 0
    with output.open("w", encoding="utf-8") as stream:
        while written < positions:
            legal = list(board.legal_moves)
            row = {
                "sfen": board.sfen(),
                "legal": sorted(cshogi.move_to_usi(move) for move in legal),
                "check": bool(board.is_check()),
                "gameOver": bool(board.is_game_over()),
                "draw": int(board.is_draw()),
                "nyugyoku": bool(board.is_nyugyoku()),
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if not legal or board.is_game_over() or board.is_draw() or ply >= 400:
                board.reset()
                ply = 0
                continue
            board.push(randomizer.choice(legal))
            ply += 1
    print(f"wrote {written} parity positions to {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("web/.generated/parity.jsonl"))
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    generate(args.output, args.positions, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
