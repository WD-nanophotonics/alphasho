import { Color, PieceType } from "./types";

export interface Hash64 { hi: number; lo: number }

let seed = 0x9e3779b9;
function random32(): number {
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return seed >>> 0;
}

const BOARD_KEYS: Hash64[][] = Array.from({ length: 81 }, () =>
  Array.from({ length: 29 }, () => ({ hi: random32(), lo: random32() }))
);
const HAND_KEYS: Hash64[][][] = Array.from({ length: 2 }, () =>
  Array.from({ length: 8 }, () =>
    Array.from({ length: 19 }, () => ({ hi: random32(), lo: random32() })))
);
const TURN_KEY = { hi: random32(), lo: random32() };

function pieceIndex(piece: number): number { return piece + 14; }

export function positionHash(board: Int8Array, hands: Uint8Array, turn: Color): Hash64 {
  let hi = 0;
  let lo = 0;
  for (let square = 0; square < 81; square += 1) {
    const piece = board[square] ?? 0;
    if (!piece) continue;
    const key = BOARD_KEYS[square]?.[pieceIndex(piece)];
    if (key) { hi ^= key.hi; lo ^= key.lo; }
  }
  for (let color = 0; color < 2; color += 1) {
    for (let type = PieceType.Pawn; type <= PieceType.Rook; type += 1) {
      const count = hands[color * 8 + type] ?? 0;
      const key = HAND_KEYS[color]?.[type]?.[Math.min(count, 18)];
      if (key) { hi ^= key.hi; lo ^= key.lo; }
    }
  }
  if (turn === 1) { hi ^= TURN_KEY.hi; lo ^= TURN_KEY.lo; }
  return { hi: hi >>> 0, lo: lo >>> 0 };
}

export function hashKey(hash: Hash64): string {
  return `${hash.hi.toString(16).padStart(8, "0")}${hash.lo.toString(16).padStart(8, "0")}`;
}

export function xorBoard(hash: Hash64, square: number, piece: number): void {
  if (!piece) return;
  const key = BOARD_KEYS[square]?.[pieceIndex(piece)];
  if (key) { hash.hi = (hash.hi ^ key.hi) >>> 0; hash.lo = (hash.lo ^ key.lo) >>> 0; }
}

export function xorHand(hash: Hash64, color: Color, type: PieceType, oldCount: number, newCount: number): void {
  const oldKey = HAND_KEYS[color]?.[type]?.[Math.min(oldCount, 18)];
  const newKey = HAND_KEYS[color]?.[type]?.[Math.min(newCount, 18)];
  if (oldKey) { hash.hi ^= oldKey.hi; hash.lo ^= oldKey.lo; }
  if (newKey) { hash.hi ^= newKey.hi; hash.lo ^= newKey.lo; }
  hash.hi >>>= 0; hash.lo >>>= 0;
}

export function xorTurn(hash: Hash64): void {
  hash.hi = (hash.hi ^ TURN_KEY.hi) >>> 0;
  hash.lo = (hash.lo ^ TURN_KEY.lo) >>> 0;
}
