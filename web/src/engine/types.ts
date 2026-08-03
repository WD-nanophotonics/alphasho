export const BLACK = 0 as const;
export const WHITE = 1 as const;
export type Color = typeof BLACK | typeof WHITE;

export enum PieceType {
  Empty = 0,
  Pawn = 1,
  Lance = 2,
  Knight = 3,
  Silver = 4,
  Gold = 5,
  Bishop = 6,
  Rook = 7,
  King = 8,
  ProPawn = 9,
  ProLance = 10,
  ProKnight = 11,
  ProSilver = 12,
  Horse = 13,
  Dragon = 14
}

export type HandPiece = PieceType.Pawn | PieceType.Lance | PieceType.Knight |
  PieceType.Silver | PieceType.Gold | PieceType.Bishop | PieceType.Rook;

export interface Move {
  from: number | null;
  to: number;
  piece: PieceType;
  promote: boolean;
  captured: PieceType;
}

export type OutcomeReason = "checkmate" | "resignation" | "repetition" |
  "perpetual-check" | "entering-king";

export interface GameOutcome {
  winner: Color | null;
  reason: OutcomeReason;
}

export interface GameState {
  schemaVersion: 1;
  sfen: string;
  history: string[];
  turn: Color;
  outcome: GameOutcome | null;
  lastMove: string | null;
}

export interface SearchBudget {
  milliseconds: number;
  maxDepth: number;
}

export interface SearchProgress {
  depth: number;
  score: number;
  nodes: number;
  elapsedMs: number;
  principalVariation: string[];
}

export interface PlayerResult extends SearchProgress {
  move: string;
}

export interface BrowserPlayer {
  chooseMove(
    state: GameState,
    budget: SearchBudget,
    signal: AbortSignal,
    onProgress?: (progress: SearchProgress) => void
  ): Promise<PlayerResult>;
  dispose(): void;
}

export const HAND_TYPES: readonly HandPiece[] = [
  PieceType.Pawn,
  PieceType.Lance,
  PieceType.Knight,
  PieceType.Silver,
  PieceType.Gold,
  PieceType.Bishop,
  PieceType.Rook
];

export const START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

export function opposite(color: Color): Color {
  return color === BLACK ? WHITE : BLACK;
}

export function colorOf(piece: number): Color {
  return piece > 0 ? BLACK : WHITE;
}

export function typeOf(piece: number): PieceType {
  return Math.abs(piece) as PieceType;
}

export function signedPiece(color: Color, type: PieceType): number {
  return color === BLACK ? type : -type;
}

export function promote(type: PieceType): PieceType {
  if (type >= PieceType.Pawn && type <= PieceType.Silver) return (type + 8) as PieceType;
  if (type === PieceType.Bishop || type === PieceType.Rook) return (type + 7) as PieceType;
  return type;
}

export function unpromote(type: PieceType): PieceType {
  if (type >= PieceType.ProPawn && type <= PieceType.ProSilver) return (type - 8) as PieceType;
  if (type === PieceType.Horse || type === PieceType.Dragon) return (type - 7) as PieceType;
  return type;
}

export function isPromoted(type: PieceType): boolean {
  return type >= PieceType.ProPawn;
}
