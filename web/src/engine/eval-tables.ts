import { BLACK, Color, PieceType } from "./types";

const fileOf = (square: number) => Math.floor(square / 9);
const rankOf = (square: number) => square % 9;

export const PIECE_VALUE: Readonly<Record<number, number>> = {
  [PieceType.Pawn]: 100, [PieceType.Lance]: 300, [PieceType.Knight]: 320,
  [PieceType.Silver]: 450, [PieceType.Gold]: 540, [PieceType.Bishop]: 780,
  [PieceType.Rook]: 900, [PieceType.King]: 20_000,
  [PieceType.ProPawn]: 540, [PieceType.ProLance]: 540, [PieceType.ProKnight]: 540,
  [PieceType.ProSilver]: 540, [PieceType.Horse]: 930, [PieceType.Dragon]: 1_050
};

export function positionalValue(type: PieceType, square: number, color: Color): number {
  const rank = color === BLACK ? rankOf(square) : 8 - rankOf(square);
  const center = 4 - Math.abs(4 - fileOf(square));
  if (type === PieceType.King) return rank >= 6 ? 12 + center * 2 : -rank * 3;
  const advance = (8 - rank) * (type <= PieceType.Silver ? 3 : 1);
  return Math.min(40, advance + center * 3);
}

export function boardPieceValue(type: PieceType, square: number, color: Color): number {
  return (PIECE_VALUE[type] ?? 0) + positionalValue(type, square, color);
}
