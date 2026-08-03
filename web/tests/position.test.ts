import { describe, expect, it } from "vitest";
import { Position, rankOf } from "../src/engine/position";
import { BLACK, PieceType, START_SFEN, WHITE } from "../src/engine/types";
import { hashKey, positionHash } from "../src/engine/hash";
import { referenceFastAbsolute } from "../src/engine/evaluation";

describe("Position", () => {
  it("round-trips the initial SFEN and exposes the canonical 30 moves", () => {
    const position = new Position();
    expect(position.sfen()).toBe(START_SFEN);
    expect(position.legalUsiMoves()).toHaveLength(30);
  });

  it("pushes and pops without changing the root", () => {
    const position = new Position();
    for (const usi of ["7g7f", "3c3d", "8h2b+"]) position.pushUsi(usi);
    expect(position.hands[BLACK * 8 + PieceType.Bishop]).toBe(1);
    position.pop(); position.pop(); position.pop();
    expect(position.sfen()).toBe(START_SFEN);
  });

  it("keeps the double 32-bit Zobrist hash exact across push and pop", () => {
    const position = new Position();
    for (const usi of ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e"]) {
      position.pushUsi(usi);
      expect(position.hashKey()).toBe(hashKey(positionHash(position.board, position.hands, position.turn)));
      expect(position.fastAbsolute).toBe(referenceFastAbsolute(position));
    }
    while (position.moveNumber > 1) {
      position.pop();
      expect(position.hashKey()).toBe(hashKey(positionHash(position.board, position.hands, position.turn)));
      expect(position.fastAbsolute).toBe(referenceFastAbsolute(position));
    }
  });

  it("enforces nifu and dead-square drop rules", () => {
    const position = new Position("4k4/9/9/9/9/9/4P4/9/4K4 b PLN 1");
    const drops = position.legalUsiMoves().filter((move) => move.includes("*"));
    expect(drops.some((move) => move.startsWith("P*5"))).toBe(false);
    expect(drops.filter((move) => move.startsWith("L*")).every((move) => rankOf(position.parseUsi(move).to) !== 0)).toBe(true);
    expect(drops.filter((move) => move.startsWith("N*")).every((move) => rankOf(position.parseUsi(move).to) > 1)).toBe(true);
  });

  it("rejects a pawn-drop mate (uchi-fuzume)", () => {
    const position = new Position("1rp1GG+P1p/5s3/+Pl2bk1N1/2Pn4l/3p1GpS1/p2PpG1p1/1+p2Bl3/3S1N1+s1/+l+n1K4R b P5p 263");
    expect(position.legalUsiMoves()).not.toContain("P*6d");
  });

  it("forces pawn promotion on the last rank", () => {
    const position = new Position("4k4/4P4/9/9/9/9/9/9/4K4 b - 1");
    const moves = position.legalUsiMoves().filter((move) => move.startsWith("5b5a"));
    expect(moves).toEqual(["5b5a+"]);
  });

  it("classifies fourfold repetition and perpetual check separately", () => {
    const draw = new Position("4k4/9/9/9/9/9/9/9/4K4 b - 1");
    const cycle = ["5i5h", "5a5b", "5h5i", "5b5a"];
    for (let count = 0; count < 3; count += 1) for (const move of cycle) draw.pushUsi(move);
    expect(draw.repetitionOutcome()).toEqual({ winner: null, reason: "repetition" });

    const perpetual = new Position("4k4/5R3/9/9/9/9/9/9/K8 b - 1");
    const checkingCycle = ["4b5b", "5a4a", "5b4b", "4a5a"];
    for (let count = 0; count < 3; count += 1) for (const move of checkingCycle) perpetual.pushUsi(move);
    expect(perpetual.repetitionOutcome()).toEqual({ winner: WHITE, reason: "perpetual-check" });
  });

  it("detects entering-king only with the required camp and points", () => {
    const noDeclaration = new Position("4K4/PPPPPPPPP/9/9/9/9/9/9/4k4 b RBG2S2N2L9P 1");
    expect(noDeclaration.canDeclareEnteringKing(BLACK)).toBe(false);
    const declaration = new Position("4K4/PPPPPPPPP/4G4/9/9/9/9/9/4k4 b RBG2S2N2L8P 1");
    expect(declaration.canDeclareEnteringKing(BLACK)).toBe(true);
    expect(declaration.canDeclareEnteringKing(WHITE)).toBe(false);
  });

  it("rejects malformed SFEN and USI", () => {
    expect(() => new Position("not sfen")).toThrow();
    expect(() => new Position().parseUsi("hello")).toThrow();
  });
});
