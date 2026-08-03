import {
  BLACK, Color, GameOutcome, HAND_TYPES, HandPiece, Move, PieceType, START_SFEN,
  WHITE, colorOf, isPromoted, opposite, promote, signedPiece, typeOf, unpromote
} from "./types";
import { Hash64, hashKey, positionHash, xorBoard, xorHand, xorTurn } from "./hash";
import { PIECE_VALUE, boardPieceValue } from "./eval-tables";

const SFEN_PIECES: Record<string, PieceType> = {
  P: PieceType.Pawn, L: PieceType.Lance, N: PieceType.Knight, S: PieceType.Silver,
  G: PieceType.Gold, B: PieceType.Bishop, R: PieceType.Rook, K: PieceType.King
};
const TYPE_TO_SFEN: Partial<Record<PieceType, string>> = Object.fromEntries(
  Object.entries(SFEN_PIECES).map(([key, value]) => [value, key])
);
const DROP_TYPES: Record<string, HandPiece> = {
  P: PieceType.Pawn, L: PieceType.Lance, N: PieceType.Knight, S: PieceType.Silver,
  G: PieceType.Gold, B: PieceType.Bishop, R: PieceType.Rook
};

interface Undo {
  move: Move;
  moved: number;
  captured: number;
  previousMoveNumber: number;
  hash: string;
  gaveCheck: boolean;
  previousHash: Hash64;
  previousFastAbsolute: number;
}

interface RepetitionRecord { hash: string; mover: Color | null; gaveCheck: boolean }

export function fileOf(square: number): number { return Math.floor(square / 9); }
export function rankOf(square: number): number { return square % 9; }
export function squareOf(file: number, rank: number): number { return file * 9 + rank; }
export function squareToUsi(square: number): string {
  return `${9 - fileOf(square)}${String.fromCharCode(97 + rankOf(square))}`;
}
export function usiToSquare(text: string): number {
  const file = 9 - Number(text[0]);
  const rank = (text.charCodeAt(1) || 97) - 97;
  if (file < 0 || file > 8 || rank < 0 || rank > 8) throw new Error(`Invalid USI square: ${text}`);
  return squareOf(file, rank);
}

export class Position {
  readonly board = new Int8Array(81);
  readonly hands = new Uint8Array(16);
  turn: Color = BLACK;
  moveNumber = 1;
  private readonly undoStack: Undo[] = [];
  private readonly repetition: RepetitionRecord[] = [];
  private readonly counts = new Map<string, number>();
  private currentHash: Hash64 = { hi: 0, lo: 0 };
  fastAbsolute = 0;

  constructor(sfen = START_SFEN) { this.setSfen(sfen); }

  static fromHistory(history: readonly string[], expectedSfen?: string): Position {
    const position = new Position();
    for (const usi of history) position.pushUsi(usi);
    if (expectedSfen && position.sfen() !== expectedSfen) throw new Error("History does not match the supplied SFEN");
    return position;
  }

  clone(): Position {
    const copy = new Position(this.sfen());
    copy.repetition.length = 0;
    copy.repetition.push(...this.repetition);
    copy.counts.clear();
    for (const [key, count] of this.counts) copy.counts.set(key, count);
    return copy;
  }

  setSfen(sfen: string): void {
    const [boardPart, turnPart, handPart, movePart] = sfen.trim().split(/\s+/);
    if (!boardPart || !turnPart || !handPart) throw new Error(`Invalid SFEN: ${sfen}`);
    this.board.fill(0);
    this.hands.fill(0);
    const ranks = boardPart.split("/");
    if (ranks.length !== 9) throw new Error(`Invalid SFEN board: ${boardPart}`);
    ranks.forEach((row, rank) => {
      let file = 0;
      let promoted = false;
      for (const char of row) {
        if (/\d/.test(char)) { file += Number(char); continue; }
        if (char === "+") { promoted = true; continue; }
        const base = SFEN_PIECES[char.toUpperCase()];
        if (!base || file > 8) throw new Error(`Invalid SFEN piece: ${char}`);
        const type = promoted ? promote(base) : base;
        const color: Color = char === char.toUpperCase() ? BLACK : WHITE;
        this.board[squareOf(file, rank)] = signedPiece(color, type);
        file += 1;
        promoted = false;
      }
      if (file !== 9 || promoted) throw new Error(`Invalid SFEN rank: ${row}`);
    });
    this.turn = turnPart === "b" ? BLACK : WHITE;
    if (handPart !== "-") {
      let countText = "";
      for (const char of handPart) {
        if (/\d/.test(char)) { countText += char; continue; }
        const type = DROP_TYPES[char.toUpperCase()];
        if (!type) throw new Error(`Invalid SFEN hand: ${char}`);
        const color: Color = char === char.toUpperCase() ? BLACK : WHITE;
        this.hands[color * 8 + type] = Number(countText || "1");
        countText = "";
      }
      if (countText) throw new Error(`Invalid SFEN hand count: ${handPart}`);
    }
    this.moveNumber = Math.max(1, Number(movePart ?? "1") || 1);
    this.currentHash = positionHash(this.board, this.hands, this.turn);
    this.fastAbsolute = 0;
    for (let square = 0; square < 81; square += 1) {
      const piece = this.board[square] ?? 0;
      if (!piece) continue;
      const color = colorOf(piece);
      this.fastAbsolute += (color === BLACK ? 1 : -1) * boardPieceValue(typeOf(piece), square, color);
    }
    for (const color of [BLACK, WHITE] as const) for (const type of HAND_TYPES)
      this.fastAbsolute += (color === BLACK ? 1 : -1) * (this.hands[color * 8 + type] ?? 0) * (PIECE_VALUE[type] ?? 0);
    this.undoStack.length = 0;
    this.repetition.length = 0;
    this.counts.clear();
    const hash = this.hashKey();
    this.repetition.push({ hash, mover: null, gaveCheck: false });
    this.counts.set(hash, 1);
  }

  sfen(): string {
    const rows: string[] = [];
    for (let rank = 0; rank < 9; rank += 1) {
      let row = "";
      let empty = 0;
      for (let file = 0; file < 9; file += 1) {
        const piece = this.board[squareOf(file, rank)] ?? 0;
        if (!piece) { empty += 1; continue; }
        if (empty) { row += empty; empty = 0; }
        const type = typeOf(piece);
        const base = unpromote(type);
        let symbol = TYPE_TO_SFEN[base] ?? "";
        if (colorOf(piece) === WHITE) symbol = symbol.toLowerCase();
        row += `${isPromoted(type) ? "+" : ""}${symbol}`;
      }
      if (empty) row += empty;
      rows.push(row);
    }
    let hand = "";
    for (const color of [BLACK, WHITE] as const) {
      for (const type of [...HAND_TYPES].reverse()) {
        const count = this.hands[color * 8 + type] ?? 0;
        if (!count) continue;
        let symbol = TYPE_TO_SFEN[type] ?? "";
        if (color === WHITE) symbol = symbol.toLowerCase();
        hand += `${count > 1 ? count : ""}${symbol}`;
      }
    }
    return `${rows.join("/")} ${this.turn === BLACK ? "b" : "w"} ${hand || "-"} ${this.moveNumber}`;
  }

  hashKey(): string { return hashKey(this.currentHash); }

  setTurn(color: Color): void {
    if (this.turn !== color) { this.turn = color; xorTurn(this.currentHash); }
  }

  parseUsi(usi: string): Move {
    if (/^[PLNSGBR]\*[1-9][a-i]$/.test(usi)) {
      const type = DROP_TYPES[usi[0] ?? ""];
      if (!type) throw new Error(`Invalid USI drop: ${usi}`);
      return { from: null, to: usiToSquare(usi.slice(2)), piece: type, promote: false, captured: PieceType.Empty };
    }
    if (!/^[1-9][a-i][1-9][a-i]\+?$/.test(usi)) throw new Error(`Invalid USI move: ${usi}`);
    const from = usiToSquare(usi.slice(0, 2));
    const to = usiToSquare(usi.slice(2, 4));
    const piece = typeOf(this.board[from] ?? 0);
    const captured = typeOf(this.board[to] ?? 0);
    return { from, to, piece, promote: usi.endsWith("+"), captured };
  }

  moveToUsi(move: Move): string {
    if (move.from === null) return `${TYPE_TO_SFEN[move.piece]}*${squareToUsi(move.to)}`;
    return `${squareToUsi(move.from)}${squareToUsi(move.to)}${move.promote ? "+" : ""}`;
  }

  pushUsi(usi: string): void {
    const move = this.parseUsi(usi);
    if (!this.legalMoves().some((candidate) => this.moveToUsi(candidate) === usi)) throw new Error(`Illegal move: ${usi}`);
    this.push(move);
  }

  push(move: Move): void {
    const mover = this.turn;
    const previousHash = this.hashKey();
    const previousHashValue = { ...this.currentHash };
    const previousFastAbsolute = this.fastAbsolute;
    const sign = mover === BLACK ? 1 : -1;
    let moved = 0;
    let captured = 0;
    if (move.from === null) {
      moved = signedPiece(mover, move.piece);
      const handIndex = mover * 8 + move.piece;
      const oldCount = this.hands[handIndex] ?? 0;
      this.hands[handIndex] = oldCount - 1;
      xorHand(this.currentHash, mover, move.piece, oldCount, oldCount - 1);
      xorBoard(this.currentHash, move.to, moved);
      this.fastAbsolute += sign * (boardPieceValue(move.piece, move.to, mover) - (PIECE_VALUE[move.piece] ?? 0));
    } else {
      moved = this.board[move.from] ?? 0;
      captured = this.board[move.to] ?? 0;
      const originalType = typeOf(moved);
      this.fastAbsolute -= sign * boardPieceValue(originalType, move.from, mover);
      xorBoard(this.currentHash, move.from, moved);
      xorBoard(this.currentHash, move.to, captured);
      this.board[move.from] = 0;
      if (captured) {
        const handIndex = mover * 8 + unpromote(typeOf(captured));
        const oldCount = this.hands[handIndex] ?? 0;
        this.hands[handIndex] = oldCount + 1;
        xorHand(this.currentHash, mover, unpromote(typeOf(captured)), oldCount, oldCount + 1);
        const capturedColor = opposite(mover);
        this.fastAbsolute -= (capturedColor === BLACK ? 1 : -1) * boardPieceValue(typeOf(captured), move.to, capturedColor);
        this.fastAbsolute += sign * (PIECE_VALUE[unpromote(typeOf(captured))] ?? 0);
      }
      if (move.promote) moved = signedPiece(mover, promote(typeOf(moved)));
      xorBoard(this.currentHash, move.to, moved);
      this.fastAbsolute += sign * boardPieceValue(typeOf(moved), move.to, mover);
    }
    this.board[move.to] = moved;
    this.turn = opposite(mover);
    xorTurn(this.currentHash);
    this.moveNumber += 1;
    const gaveCheck = this.inCheck(this.turn);
    const currentHash = this.hashKey();
    this.undoStack.push({ move, moved, captured, previousMoveNumber: this.moveNumber - 1, hash: previousHash, gaveCheck,
      previousHash: previousHashValue, previousFastAbsolute });
    this.repetition.push({ hash: currentHash, mover, gaveCheck });
    this.counts.set(currentHash, (this.counts.get(currentHash) ?? 0) + 1);
  }

  pop(): Move {
    const undo = this.undoStack.pop();
    if (!undo) throw new Error("Cannot pop the root position");
    const current = this.repetition.pop();
    if (current) {
      const count = (this.counts.get(current.hash) ?? 1) - 1;
      if (count) this.counts.set(current.hash, count); else this.counts.delete(current.hash);
    }
    this.turn = opposite(this.turn);
    this.currentHash = undo.previousHash;
    this.fastAbsolute = undo.previousFastAbsolute;
    this.moveNumber = undo.previousMoveNumber;
    if (undo.move.from === null) {
      this.board[undo.move.to] = 0;
      const handIndex = this.turn * 8 + undo.move.piece;
      this.hands[handIndex] = (this.hands[handIndex] ?? 0) + 1;
    } else {
      this.board[undo.move.from] = signedPiece(this.turn, undo.move.piece);
      this.board[undo.move.to] = undo.captured;
      if (undo.captured) {
        const handIndex = this.turn * 8 + unpromote(typeOf(undo.captured));
        this.hands[handIndex] = (this.hands[handIndex] ?? 0) - 1;
      }
    }
    return undo.move;
  }

  kingSquare(color: Color): number {
    const king = signedPiece(color, PieceType.King);
    return this.board.indexOf(king);
  }

  inCheck(color: Color): boolean {
    const king = this.kingSquare(color);
    return king >= 0 && this.isSquareAttacked(king, opposite(color));
  }

  isSquareAttacked(target: number, byColor: Color): boolean {
    for (let from = 0; from < 81; from += 1) {
      const piece = this.board[from] ?? 0;
      if (piece && colorOf(piece) === byColor && this.attacks(from, target, typeOf(piece), byColor)) return true;
    }
    return false;
  }

  private attacks(from: number, to: number, type: PieceType, color: Color): boolean {
    const dx = fileOf(to) - fileOf(from);
    const dy = (rankOf(to) - rankOf(from)) * (color === BLACK ? 1 : -1);
    const gold = (dx === 0 && dy === -1) || (Math.abs(dx) === 1 && dy === -1) ||
      (Math.abs(dx) === 1 && dy === 0) || (dx === 0 && dy === 1);
    if (type === PieceType.Pawn) return dx === 0 && dy === -1;
    if (type === PieceType.Knight) return Math.abs(dx) === 1 && dy === -2;
    if (type === PieceType.Silver) return (Math.abs(dx) === 1 && Math.abs(dy) === 1) || (dx === 0 && dy === -1);
    if (type === PieceType.Gold || (type >= PieceType.ProPawn && type <= PieceType.ProSilver)) return gold;
    if (type === PieceType.King) return Math.max(Math.abs(dx), Math.abs(dy)) === 1;
    if ((type === PieceType.Horse && ((Math.abs(dx) === 1 && dy === 0) || (dx === 0 && Math.abs(dy) === 1))) ||
        (type === PieceType.Dragon && Math.abs(dx) === 1 && Math.abs(dy) === 1)) return true;
    const bishopLike = type === PieceType.Bishop || type === PieceType.Horse;
    const rookLike = type === PieceType.Rook || type === PieceType.Dragon;
    const lanceLike = type === PieceType.Lance && dx === 0 && dy < 0;
    if (!((bishopLike && Math.abs(dx) === Math.abs(dy)) ||
          (rookLike && (dx === 0 || dy === 0)) || lanceLike)) return false;
    const steps = Math.max(Math.abs(dx), Math.abs(dy));
    const sx = Math.sign(dx);
    const absoluteDy = rankOf(to) - rankOf(from);
    const sy = Math.sign(absoluteDy);
    for (let step = 1; step < steps; step += 1) {
      if (this.board[squareOf(fileOf(from) + sx * step, rankOf(from) + sy * step)]) return false;
    }
    return true;
  }

  private addStep(moves: Move[], from: number, dx: number, relativeDy: number, color: Color, type: PieceType): void {
    const file = fileOf(from) + dx;
    const rank = rankOf(from) + relativeDy * (color === BLACK ? 1 : -1);
    if (file < 0 || file > 8 || rank < 0 || rank > 8) return;
    const to = squareOf(file, rank);
    const target = this.board[to] ?? 0;
    if (target && colorOf(target) === color) return;
    this.addPromotionVariants(moves, from, to, type, target);
  }

  private addRay(moves: Move[], from: number, dx: number, relativeDy: number, color: Color, type: PieceType): void {
    const dy = relativeDy * (color === BLACK ? 1 : -1);
    for (let distance = 1; distance < 9; distance += 1) {
      const file = fileOf(from) + dx * distance;
      const rank = rankOf(from) + dy * distance;
      if (file < 0 || file > 8 || rank < 0 || rank > 8) break;
      const to = squareOf(file, rank);
      const target = this.board[to] ?? 0;
      if (target && colorOf(target) === color) break;
      this.addPromotionVariants(moves, from, to, type, target);
      if (target) break;
    }
  }

  private addPromotionVariants(moves: Move[], from: number, to: number, type: PieceType, target: number): void {
    const promotable = type <= PieceType.Rook && type !== PieceType.Gold && type !== PieceType.King;
    const sourceRank = rankOf(from);
    const targetRank = rankOf(to);
    const inZone = this.turn === BLACK ? sourceRank <= 2 || targetRank <= 2 : sourceRank >= 6 || targetRank >= 6;
    const last = this.turn === BLACK ? targetRank === 0 : targetRank === 8;
    const lastTwo = this.turn === BLACK ? targetRank <= 1 : targetRank >= 7;
    const forced = (type === PieceType.Pawn || type === PieceType.Lance) ? last : type === PieceType.Knight && lastTwo;
    const captured = typeOf(target);
    if (!forced) moves.push({ from, to, piece: type, promote: false, captured });
    if (promotable && inZone) moves.push({ from, to, piece: type, promote: true, captured });
  }

  private pseudoBoardMoves(): Move[] {
    const moves: Move[] = [];
    for (let from = 0; from < 81; from += 1) {
      const piece = this.board[from] ?? 0;
      if (!piece || colorOf(piece) !== this.turn) continue;
      const type = typeOf(piece);
      const step = (dx: number, dy: number) => this.addStep(moves, from, dx, dy, this.turn, type);
      const ray = (dx: number, dy: number) => this.addRay(moves, from, dx, dy, this.turn, type);
      if (type === PieceType.Pawn) step(0, -1);
      else if (type === PieceType.Lance) ray(0, -1);
      else if (type === PieceType.Knight) { step(-1, -2); step(1, -2); }
      else if (type === PieceType.Silver) { step(-1, -1); step(0, -1); step(1, -1); step(-1, 1); step(1, 1); }
      else if (type === PieceType.Gold || (type >= PieceType.ProPawn && type <= PieceType.ProSilver)) {
        step(-1, -1); step(0, -1); step(1, -1); step(-1, 0); step(1, 0); step(0, 1);
      } else if (type === PieceType.King) {
        for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) if (dx || dy) step(dx, dy);
      } else {
        if (type === PieceType.Bishop || type === PieceType.Horse) {
          ray(-1, -1); ray(1, -1); ray(-1, 1); ray(1, 1);
        }
        if (type === PieceType.Rook || type === PieceType.Dragon) {
          ray(0, -1); ray(0, 1); ray(-1, 0); ray(1, 0);
        }
        if (type === PieceType.Horse) { step(-1, 0); step(1, 0); step(0, -1); step(0, 1); }
        if (type === PieceType.Dragon) { step(-1, -1); step(1, -1); step(-1, 1); step(1, 1); }
      }
    }
    return moves;
  }

  private pseudoDrops(): Move[] {
    const moves: Move[] = [];
    for (const type of HAND_TYPES) {
      if (!(this.hands[this.turn * 8 + type] ?? 0)) continue;
      for (let to = 0; to < 81; to += 1) {
        if (this.board[to]) continue;
        const rank = rankOf(to);
        const last = this.turn === BLACK ? rank === 0 : rank === 8;
        const lastTwo = this.turn === BLACK ? rank <= 1 : rank >= 7;
        if ((type === PieceType.Pawn || type === PieceType.Lance) && last) continue;
        if (type === PieceType.Knight && lastTwo) continue;
        if (type === PieceType.Pawn) {
          const file = fileOf(to);
          let nifu = false;
          for (let r = 0; r < 9; r += 1) {
            if (this.board[squareOf(file, r)] === signedPiece(this.turn, PieceType.Pawn)) { nifu = true; break; }
          }
          if (nifu) continue;
        }
        moves.push({ from: null, to, piece: type, promote: false, captured: PieceType.Empty });
      }
    }
    return moves;
  }

  legalMoves(options: { skipPawnDropMate?: boolean; check?: () => void } = {}): Move[] {
    const color = this.turn;
    const legal: Move[] = [];
    for (const move of [...this.pseudoBoardMoves(), ...this.pseudoDrops()]) {
      options.check?.();
      this.push(move);
      let selfCheck = false;
      let pawnDropMate = false;
      try {
        selfCheck = this.inCheck(color);
        if (!selfCheck && !options.skipPawnDropMate && move.from === null && move.piece === PieceType.Pawn && this.inCheck(this.turn))
          pawnDropMate = this.legalMoves({ skipPawnDropMate: true, check: options.check }).length === 0;
      } finally { this.pop(); }
      if (!selfCheck && !pawnDropMate) legal.push(move);
      options.check?.();
    }
    return legal;
  }

  legalUsiMoves(): string[] { return this.legalMoves().map((move) => this.moveToUsi(move)); }

  repetitionOutcome(): GameOutcome | null {
    const hash = this.hashKey();
    if ((this.counts.get(hash) ?? 0) < 4) return null;
    const occurrences: number[] = [];
    this.repetition.forEach((record, index) => { if (record.hash === hash) occurrences.push(index); });
    const start = occurrences.at(-4) ?? 0;
    for (const checkingColor of [BLACK, WHITE] as const) {
      let madeMove = false;
      let continuous = true;
      for (let index = start + 1; index < this.repetition.length; index += 1) {
        const record = this.repetition[index];
        if (record?.mover === checkingColor) {
          madeMove = true;
          if (!record.gaveCheck) { continuous = false; break; }
        }
      }
      if (madeMove && continuous) return { winner: opposite(checkingColor), reason: "perpetual-check" };
    }
    return { winner: null, reason: "repetition" };
  }

  canDeclareEnteringKing(color = this.turn): boolean {
    if (this.inCheck(color)) return false;
    const king = this.kingSquare(color);
    if (king < 0 || (color === BLACK ? rankOf(king) > 2 : rankOf(king) < 6)) return false;
    let campPieces = 0;
    let points = 0;
    for (let square = 0; square < 81; square += 1) {
      const piece = this.board[square] ?? 0;
      if (!piece || colorOf(piece) !== color || typeOf(piece) === PieceType.King) continue;
      const inCamp = color === BLACK ? rankOf(square) <= 2 : rankOf(square) >= 6;
      if (inCamp) {
        const base = unpromote(typeOf(piece));
        points += base === PieceType.Bishop || base === PieceType.Rook ? 5 : 1;
        campPieces += 1;
      }
    }
    for (const type of HAND_TYPES) {
      const count = this.hands[color * 8 + type] ?? 0;
      points += count * (type === PieceType.Bishop || type === PieceType.Rook ? 5 : 1);
    }
    // cshogi follows the JSA declaration rule: ten non-king pieces in the
    // opponent camp and at least 28 points, irrespective of color.
    return campPieces >= 10 && points >= 28;
  }

  outcome(): GameOutcome | null {
    const repetition = this.repetitionOutcome();
    if (repetition) return repetition;
    if (this.legalMoves().length === 0) return { winner: opposite(this.turn), reason: "checkmate" };
    return null;
  }
}
