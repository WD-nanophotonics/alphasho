import { Position, usiToSquare, fileOf } from '../engine/position';
import { Color, Move, START_SFEN } from '../engine/types';
export type RookRoute = 'ibisha' | 'furibisha' | 'other' | 'undetermined';
function family(square: number, color: Color): RookRoute {
  const file = color === 0 ? 9 - fileOf(square) : fileOf(square) + 1;
  return file <= 4 ? 'ibisha' : file <= 8 ? 'furibisha' : 'other';
}
export function guidedRoots(history: string[], color: Color, target: RookRoute): { roots: string[]; route: RookRoute; reason: string } {
  const p = new Position(START_SFEN); let square = usiToSquare(color === 0 ? '2h' : '8b');
  let route: RookRoute = 'ibisha', active = false, event = '';
  for (let ply = 0; ply < history.length; ply++) {
    const move = p.parseUsi(history[ply]!);
    if (ply < 40 && !event) {
      if (p.turn !== color && move.to === square) { event = 'captured'; if (!active) route = 'undetermined'; }
      else if (p.turn === color && move.from === square) {
        active = true; if (move.promote) event = 'promoted'; else { square = move.to; route = family(square, color); }
      }
    }
    p.push(move);
  }
  const legal = p.legalMoves(); const encode = (moves: Move[]) => moves.map(m => p.moveToUsi(m));
  if (history.length >= 40 || event || p.inCheck(color)) return { roots: encode(legal), route, reason: event || 'ordinary-or-check' };
  if (target === 'furibisha' && route !== target) {
    const pawn = color === 0 ? '7g7f' : '3c3d';
    if (history.length < 2 && encode(legal).includes(pawn)) return { roots: [pawn], route, reason: 'pawn-lead-in' };
    const transfers = legal.filter(m => m.from === square && !m.promote && family(m.to, color) === target);
    if (transfers.length) return { roots: encode(transfers), route, reason: 'rook-transfer' };
  }
  const allowed = legal.filter(m => m.from !== square || (m.promote && route === target) || family(m.to, color) === target);
  return { roots: encode(allowed.length ? allowed : legal), route, reason: 'route-guard' };
}
