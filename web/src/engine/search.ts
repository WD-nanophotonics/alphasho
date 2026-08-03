import { evaluate, fastEvaluation, PIECE_VALUE } from "./evaluation";
import { Position } from "./position";
import { Move, PieceType, PlayerResult, SearchBudget, SearchProgress } from "./types";

const INF = 1_000_000;
const MATE = 900_000;
const TT_LIMIT = 100_000;

interface TTEntry { depth: number; score: number; flag: -1 | 0 | 1; move?: string }
class Timeout extends Error {}

export class AlphaBetaSearch {
  private readonly tt = new Map<string, TTEntry>();
  private readonly killers = new Map<number, string[]>();
  private readonly history = new Map<string, number>();
  private nodes = 0;
  private deadline = 0;
  private stopped = false;
  private qNodes = 0;
  private completedDepth = 0;
  private bestMove = "";
  private bestScore = 0;
  private bestPv: string[] = [];

  constructor(private readonly position: Position) {}

  cancel(): void { this.stopped = true; }

  run(budget: SearchBudget, onProgress?: (progress: SearchProgress) => void): PlayerResult {
    const started = performance.now();
    this.deadline = started + Math.max(10, budget.milliseconds - Math.min(50, Math.max(5, budget.milliseconds * 0.08)));
    const rootMoves = this.position.legalMoves();
    if (!rootMoves.length) throw new Error("No legal moves");
    this.bestMove = this.position.moveToUsi(rootMoves[0] as Move);
    try { this.rootSafetyScan(rootMoves); } catch (error) { if (!(error instanceof Timeout)) throw error; }
    if (this.bestScore >= MATE - 100) {
      return { move: this.bestMove, depth: 1, score: this.bestScore, nodes: this.nodes + this.qNodes,
        elapsedMs: performance.now() - started, principalVariation: [this.bestMove] };
    }
    for (let depth = 2; depth <= budget.maxDepth; depth += 1) {
      try {
        const result = this.searchRoot(rootMoves, depth);
        this.completedDepth = depth;
        this.bestMove = result.move;
        this.bestScore = result.score;
        this.bestPv = result.pv;
        onProgress?.({ depth, score: result.score, nodes: this.nodes + this.qNodes,
          elapsedMs: performance.now() - started, principalVariation: result.pv });
      } catch (error) {
        if (error instanceof Timeout) break;
        throw error;
      }
    }
    return { move: this.bestMove, depth: Math.max(1, this.completedDepth), score: this.bestScore,
      nodes: this.nodes + this.qNodes, elapsedMs: performance.now() - started,
      principalVariation: this.bestPv.length ? this.bestPv : [this.bestMove] };
  }

  private check(): void {
    if (this.stopped || performance.now() >= this.deadline) throw new Timeout();
  }

  private rootSafetyScan(moves: Move[]): void {
    let best = -INF;
    let completed = 0;
    for (const move of this.orderMoves(moves, 1)) {
      this.check();
      const usi = this.position.moveToUsi(move);
      this.position.push(move);
      let score: number;
      try {
        const replies = this.position.legalMoves({ check: () => this.check() });
        score = replies.length === 0 ? MATE - 1 : -fastEvaluation(this.position);
      } finally { this.position.pop(); }
      completed += 1;
      if (score > best) { best = score; this.bestMove = usi; this.bestScore = score; this.bestPv = [usi]; }
      if (score >= MATE - 100) { this.completedDepth = 1; return; }
    }
    if (completed) this.completedDepth = 1;
  }

  private searchRoot(moves: Move[], depth: number): { move: string; score: number; pv: string[] } {
    let alpha = -INF;
    let bestMove = this.bestMove;
    let bestPv = [bestMove];
    for (const move of this.orderMoves(moves, 0, this.bestMove)) {
      this.check();
      const usi = this.position.moveToUsi(move);
      this.position.push(move);
      let score: number;
      try { score = -this.negamax(depth - 1, -INF, -alpha, 1); }
      finally { this.position.pop(); }
      if (score > alpha) {
        alpha = score;
        bestMove = usi;
        bestPv = [usi, ...this.extractPv(depth - 1)];
      }
    }
    this.store(this.position.hashKey(), { depth, score: alpha, flag: 0, move: bestMove });
    return { move: bestMove, score: alpha, pv: bestPv };
  }

  private negamax(depth: number, alphaInput: number, beta: number, ply: number): number {
    this.check();
    this.nodes += 1;
    const repetition = this.position.repetitionOutcome();
    if (repetition) return repetition.winner === null ? 0 : repetition.winner === this.position.turn ? MATE - ply : -MATE + ply;
    let alpha = alphaInput;
    const key = this.position.hashKey();
    const entry = this.tt.get(key);
    if (entry && entry.depth >= depth) {
      if (entry.flag === 0) return entry.score;
      if (entry.flag === 1 && entry.score >= beta) return entry.score;
      if (entry.flag === -1 && entry.score <= alpha) return entry.score;
    }
    if (depth <= 0) return this.quiescence(alpha, beta, 0, ply, 0);
    const moves = this.position.legalMoves({ check: () => this.check() });
    if (!moves.length) return -MATE + ply;
    let best = -INF;
    let bestMove = "";
    let moveIndex = 0;
    for (const move of this.orderMoves(moves, ply, entry?.move)) {
      this.check();
      const usi = this.position.moveToUsi(move);
      this.position.push(move);
      let score: number;
      try { score = -this.negamax(depth - 1, -beta, -alpha, ply + 1); }
      finally { this.position.pop(); }
      if (score > best) { best = score; bestMove = usi; }
      if (score > alpha) alpha = score;
      if (alpha >= beta) {
        if (!move.captured && !move.promote) {
          const list = this.killers.get(ply) ?? [];
          this.killers.set(ply, [usi, ...list.filter((item) => item !== usi)].slice(0, 2));
          this.history.set(usi, Math.min(1_000_000, (this.history.get(usi) ?? 0) + depth * depth));
        }
        break;
      }
      moveIndex += 1;
    }
    const flag: -1 | 0 | 1 = best <= alphaInput ? -1 : best >= beta ? 1 : 0;
    this.store(key, { depth, score: best, flag, move: bestMove });
    return best;
  }

  private quiescence(alphaInput: number, beta: number, qDepth: number, ply: number, leafNodes: number): number {
    this.check();
    this.qNodes += 1;
    let alpha = alphaInput;
    const inCheck = this.position.inCheck(this.position.turn);
    if (!inCheck) {
      const standPat = evaluate(this.position, () => this.check());
      if (standPat >= beta) return beta;
      if (standPat > alpha) alpha = standPat;
      if (qDepth >= 4 || leafNodes >= 32) return alpha;
    }
    let moves = this.position.legalMoves({ check: () => this.check() });
    if (!moves.length) return -MATE + ply;
    if (!inCheck) moves = moves.filter((move) => move.captured !== PieceType.Empty || move.promote);
    for (const move of this.orderMoves(moves, ply)) {
      this.check();
      if (!inCheck && move.captured && !move.promote && (PIECE_VALUE[move.captured] ?? 0) + 100 < (PIECE_VALUE[move.piece] ?? 0)) continue;
      this.position.push(move);
      let score: number;
      try { score = -this.quiescence(-beta, -alpha, qDepth + 1, ply + 1, leafNodes + 1); }
      finally { this.position.pop(); }
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  private orderMoves(moves: Move[], ply: number, preferred?: string): Move[] {
    const killers = this.killers.get(ply) ?? [];
    return [...moves].sort((a, b) => this.moveScore(b, preferred, killers) - this.moveScore(a, preferred, killers));
  }

  private moveScore(move: Move, preferred: string | undefined, killers: string[]): number {
    const usi = this.position.moveToUsi(move);
    if (usi === preferred) return 10_000_000;
    if (move.captured) return 1_000_000 + (PIECE_VALUE[move.captured] ?? 0) * 10 - (PIECE_VALUE[move.piece] ?? 0);
    if (move.promote) return 800_000;
    const killerIndex = killers.indexOf(usi);
    if (killerIndex >= 0) return 600_000 - killerIndex;
    return this.history.get(usi) ?? 0;
  }

  private extractPv(depth: number): string[] {
    const pv: string[] = [];
    const pushed: Move[] = [];
    for (let index = 0; index < depth; index += 1) {
      const entry = this.tt.get(this.position.hashKey());
      if (!entry?.move) break;
      const move = this.position.legalMoves().find((candidate) => this.position.moveToUsi(candidate) === entry.move);
      if (!move) break;
      pv.push(entry.move);
      this.position.push(move);
      pushed.push(move);
    }
    while (pushed.length) { this.position.pop(); pushed.pop(); }
    return pv;
  }

  private store(key: string, entry: TTEntry): void {
    if (this.tt.size >= TT_LIMIT) this.tt.delete(this.tt.keys().next().value as string);
    this.tt.set(key, entry);
  }
}
