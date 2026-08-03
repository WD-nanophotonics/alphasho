import { describe, expect, it, vi } from "vitest";
import { Position } from "../src/engine/position";
import { BLACK } from "../src/engine/types";
import { APP_VERSION, PROTOCOL_VERSION, createRoomCode, envelope, isNetworkMessage, validateMoveProposal } from "../src/online/protocol";

describe("online protocol", () => {
  it("uses crypto-backed ten-character room codes", () => {
    vi.stubGlobal("crypto", { getRandomValues: (array: Uint8Array) => { array.forEach((_, index) => { array[index] = index; }); return array; } });
    expect(createRoomCode()).toMatch(/^[A-Z2-9]{10}$/);
    vi.unstubAllGlobals();
  });

  it("rejects unknown versions", () => {
    const hello = { ...envelope("ABCDEFGHIJ"), type: "hello", role: "host" };
    expect(isNetworkMessage(hello)).toBe(true);
    expect(isNetworkMessage({ ...hello, protocolVersion: PROTOCOL_VERSION + 1 })).toBe(false);
    expect(isNetworkMessage({ ...hello, appVersion: `${APP_VERSION}-old` })).toBe(false);
  });

  it("authoritatively rejects stale, duplicated, wrong-turn and illegal proposals", () => {
    const position = new Position();
    const hash = position.hashKey();
    expect(validateMoveProposal(position, 1, 1, 0, hash, "7g7f")).toEqual({ ok: true });
    expect(validateMoveProposal(position, 1, 0, 0, hash, "7g7f")).toEqual({ ok: false, code: "sequence" });
    expect(validateMoveProposal(position, 1, 1, 0, "bad", "7g7f")).toEqual({ ok: false, code: "hash" });
    expect(validateMoveProposal(position, BLACK, 1, 0, hash, "7g7f")).toEqual({ ok: false, code: "turn" });
    expect(validateMoveProposal(position, 1, 1, 0, hash, "7g7e")).toEqual({ ok: false, code: "illegal-move" });
  });
});
