"""Build frozen analysis/mate corpora with a user-supplied YaneuraOu binary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cshogi

from benchmarks.yaneuraou_usi import YaneuraOuUsi


def _load_positions(path: Path, limit: int | None) -> list[dict[str, object]]:
    positions: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            values = json.loads(line)
            sfen = str(values["sfen"])
            cshogi.Board(sfen)
            positions.append(
                {"name": str(values.get("name", f"position_{line_number}")), "sfen": sfen}
            )
            if limit is not None and len(positions) >= limit:
                break
    if not positions:
        raise ValueError(f"No positions found in {path}")
    return positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--mate-ms", type=int, default=0)
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()

    options: dict[str, str] = {}
    for raw in args.option:
        if "=" not in raw:
            parser.error(f"--option requires NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        options[name] = value

    positions = _load_positions(args.input, args.limit)
    rows: list[dict[str, object]] = []
    with YaneuraOuUsi(args.engine) as engine:
        engine.new_game()
        fingerprint = {
            "binary_sha256": engine.binary_sha256,
            "engine_ids": engine.ids,
            "nodes": args.nodes,
            "multipv": args.multipv,
            "options": options,
        }
        for index, position in enumerate(positions, start=1):
            analysis = engine.analyse(
                str(position["sfen"]),
                nodes=args.nodes,
                multipv=args.multipv,
                options=options,
            )
            row: dict[str, object] = {
                **position,
                "oracle": fingerprint,
                "bestmove": analysis.bestmove,
                "multipv": [
                    {
                        "rank": item.multipv,
                        "score_cp": item.score_cp,
                        "score_mate": item.score_mate,
                        "nodes": item.nodes,
                        "depth": item.depth,
                        "pv": item.principal_variation,
                    }
                    for item in analysis.analyses
                ],
            }
            if args.mate_ms > 0:
                mate = engine.solve_mate(str(position["sfen"]), milliseconds=args.mate_ms)
                row["mate"] = {
                    "status": mate.status,
                    "pv": mate.principal_variation,
                    "milliseconds": args.mate_ms,
                }
            rows.append(row)
            print(
                f"ORACLE {index}/{len(positions)} {position['name']} "
                f"bestmove={analysis.bestmove}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "positions": len(rows),
        "oracle": rows[0]["oracle"],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"CORPUS SAVED {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

