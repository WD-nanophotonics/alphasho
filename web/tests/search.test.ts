import { describe, expect, it, vi } from "vitest";
import { Position } from "../src/engine/position";
import { AlphaBetaSearch } from "../src/engine/search";

describe("browser ABP", () => {
  it("returns a legal move and a non-empty PV", () => {
    const position = new Position();
    const result = new AlphaBetaSearch(position.clone()).run({ milliseconds: 100, maxDepth: 2 });
    expect(position.legalUsiMoves()).toContain(result.move);
    expect(result.principalVariation[0]).toBe(result.move);
    expect(result.depth).toBeGreaterThanOrEqual(1);
  });

  it("publishes only completed iterative depths", () => {
    const callback = vi.fn();
    const result = new AlphaBetaSearch(new Position()).run({ milliseconds: 250, maxDepth: 3 }, callback);
    for (const call of callback.mock.calls) expect(call[0].depth).toBeGreaterThanOrEqual(2);
    expect(result.depth).toBeGreaterThanOrEqual(1);
  });

  it("unwinds every pushed move when the deadline interrupts recursion", () => {
    const position = new Position();
    const root = position.sfen();
    const result = new AlphaBetaSearch(position).run({ milliseconds: 10, maxDepth: 64 });
    expect(position.sfen()).toBe(root);
    expect(position.legalUsiMoves()).toContain(result.move);
  });

  it("root safety scan selects a legal one-ply mate", () => {
    const position = new Position("8k/9/7G1/9/9/9/9/9/K8 b R 1");
    const result = new AlphaBetaSearch(position.clone()).run({ milliseconds: 250, maxDepth: 4 });
    expect(position.legalUsiMoves()).toContain(result.move);
    const move = position.parseUsi(result.move);
    position.push(move);
    expect(position.legalMoves()).toHaveLength(0);
  });
});
