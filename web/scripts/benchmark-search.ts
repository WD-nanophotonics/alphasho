import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { Position } from "../src/engine/position";
import { AlphaBetaSearch } from "../src/engine/search";

const rows = readFileSync(resolve("../configs/training/evaluation_positions.jsonl"), "utf8")
  .trim().split(/\r?\n/).map((line) => JSON.parse(line) as { name: string; sfen: string });
const profiles = [
  { milliseconds: 100, maxDepth: 2 },
  { milliseconds: 500, maxDepth: 6 },
  { milliseconds: 2_000, maxDepth: 64 }
];
let failures = 0;
for (const profile of profiles) {
  const summaries: Array<{ elapsed: number; depth: number; nodes: number }> = [];
  for (const row of rows) {
    const position = new Position(row.sfen);
    const legal = position.legalUsiMoves();
    const start = performance.now();
    const result = new AlphaBetaSearch(position).run(profile);
    const elapsed = performance.now() - start;
    const tolerance = Math.max(50, profile.milliseconds * 0.1);
    if (!legal.includes(result.move) || elapsed > profile.milliseconds + tolerance || position.sfen() !== row.sfen) {
      failures += 1;
      console.error(JSON.stringify({ name: row.name, profile, elapsed, move: result.move, legal: legal.includes(result.move), restored: position.sfen() === row.sfen }));
    }
    summaries.push({ elapsed, depth: result.depth, nodes: result.nodes });
  }
  const sortedDepths = summaries.map((item) => item.depth).sort((a, b) => a - b);
  console.log(JSON.stringify({ budgetMs: profile.milliseconds, positions: summaries.length,
    maxElapsedMs: Math.max(...summaries.map((item) => item.elapsed)).toFixed(1),
    medianDepth: sortedDepths[Math.floor(sortedDepths.length / 2)],
    averageNodes: Math.round(summaries.reduce((sum, item) => sum + item.nodes, 0) / summaries.length) }));
}
if (failures) process.exitCode = 1;
