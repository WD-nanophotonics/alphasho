import { BLACK, Color, HAND_TYPES, PieceType, WHITE, colorOf, typeOf, unpromote } from "./types";
import { Position, fileOf, rankOf, squareOf } from "./position";
import { PIECE_VALUE, boardPieceValue } from "./eval-tables";
export { PIECE_VALUE } from "./eval-tables";

export function referenceFastAbsolute(position: Position): number {
  let score = 0;
  for (let square = 0; square < 81; square += 1) {
    const piece = position.board[square] ?? 0;
    if (!piece) continue;
    const color = colorOf(piece);
    score += (color === BLACK ? 1 : -1) * boardPieceValue(typeOf(piece), square, color);
  }
  for (const color of [BLACK, WHITE] as const) for (const type of HAND_TYPES)
    score += (color === BLACK ? 1 : -1) * (position.hands[color * 8 + type] ?? 0) * (PIECE_VALUE[type] ?? 0);
  return score;
}

export function fastEvaluation(position: Position): number {
  const perspective = position.turn === BLACK ? position.fastAbsolute : -position.fastAbsolute;
  return perspective + (position.inCheck(position.turn) ? -35 : 0) + 8;
}

function neighborhoodSafety(position: Position, color: Color): number {
  const king = position.kingSquare(color);
  if (king < 0) return -260;
  let defenders = 0;
  let pressure = 0;
  let escapes = 0;
  let shield = 0;
  const forward = color === BLACK ? -1 : 1;
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      if (!dx && !dy) continue;
      const file = fileOf(king) + dx;
      const rank = rankOf(king) + dy;
      if (file < 0 || file > 8 || rank < 0 || rank > 8) continue;
      const square = squareOf(file, rank);
      if (position.isSquareAttacked(square, color)) defenders += 1;
      if (position.isSquareAttacked(square, color === BLACK ? WHITE : BLACK)) pressure += 1;
      if (!position.board[square] && !position.isSquareAttacked(square, color === BLACK ? WHITE : BLACK)) escapes += 1;
      if (dy === forward) {
        const piece = position.board[square] ?? 0;
        if (piece && colorOf(piece) === color && unpromote(typeOf(piece)) === PieceType.Pawn) shield += 1;
      }
    }
  }
  return Math.max(-260, Math.min(260, defenders * 7 + escapes * 9 + shield * 15 - pressure * 20));
}

export interface EvaluationBreakdown {
  fast: number;
  mobility: number;
  kingSafety: number;
  total: number;
}

export function evaluate(position: Position, check?: () => void): number { return evaluationBreakdown(position, check).total; }

export function evaluationBreakdown(position: Position, check?: () => void): EvaluationBreakdown {
  const fast = fastEvaluation(position);
  check?.();
  const ownMoves = position.legalMoves({ skipPawnDropMate: true, check }).length;
  const clone = position.clone();
  clone.setTurn(clone.turn === BLACK ? WHITE : BLACK);
  const enemyMoves = clone.legalMoves({ skipPawnDropMate: true, check }).length;
  const mobility = Math.max(-100, Math.min(100, (ownMoves - enemyMoves) * 2));
  const own = neighborhoodSafety(position, position.turn);
  const enemy = neighborhoodSafety(position, position.turn === BLACK ? WHITE : BLACK);
  const kingSafety = Math.max(-260, Math.min(260, own - enemy));
  return { fast, mobility, kingSafety, total: fast + mobility + kingSafety };
}
