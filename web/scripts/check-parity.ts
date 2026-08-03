import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { Position } from "../src/engine/position";

interface OracleRow {
  sfen: string;
  legal: string[];
  check: boolean;
  gameOver: boolean;
  draw: number;
  nyugyoku: boolean;
}

const path = resolve(process.argv[2] ?? ".generated/parity.jsonl");
const lines = createInterface({ input: createReadStream(path, "utf8"), crlfDelay: Infinity });
let checked = 0;
let failures = 0;

for await (const line of lines) {
  if (!line.trim()) continue;
  const oracle = JSON.parse(line) as OracleRow;
  const position = new Position(oracle.sfen);
  const legal = position.legalUsiMoves().sort();
  const mismatch = position.sfen() !== oracle.sfen || JSON.stringify(legal) !== JSON.stringify(oracle.legal) ||
    position.inCheck(position.turn) !== oracle.check || (legal.length === 0) !== oracle.gameOver ||
    position.canDeclareEnteringKing() !== oracle.nyugyoku;
  if (mismatch) {
    failures += 1;
    if (failures <= 10) {
      const missing = oracle.legal.filter((move) => !legal.includes(move));
      const extra = legal.filter((move) => !oracle.legal.includes(move));
      console.error(JSON.stringify({ sfen: oracle.sfen, missing, extra,
        expectedCheck: oracle.check, actualCheck: position.inCheck(position.turn),
        expectedNyugyoku: oracle.nyugyoku, actualNyugyoku: position.canDeclareEnteringKing() }));
    }
  }
  checked += 1;
}

console.log(`checked=${checked} failures=${failures}`);
if (failures) process.exitCode = 1;
