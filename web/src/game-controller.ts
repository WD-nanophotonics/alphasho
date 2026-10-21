import { Position } from "./engine/position";
import { BrowserPlayer, Color, GameOutcome, GameState, Move, START_SFEN, WHITE, opposite } from "./engine/types";
import { DIFFICULTIES, HeuristicWorkerPlayer } from "./player/heuristic-player";

import { NnueWorkerPlayer } from "./player/nnue-player";
import { NNUE_LEVELS } from "./player/nnue-config";
import { guidedRoots, type RookRoute } from "./player/rook-guide";
const guidedRoute = (history: string[], color: Color) => guidedRoots(history, color, 'ibisha').route;
export type EngineKind = "abp" | "nnue";
export type Mode = "cpu" | "local" | "online";
export type Difficulty = keyof typeof DIFFICULTIES;

export interface ControllerState {
  mode: Mode;
  position: Position;
  history: string[];
  sfenHistory: string[];
  outcome: GameOutcome | null;
  humanColor: Color | null;
  thinking: boolean;
  progress: { depth: number; nodes: number } | null;
}

export class GameController {
  player: BrowserPlayer = new HeuristicWorkerPlayer();
  engineKind: EngineKind = "abp";
  nnLevel = 5;
  rookTarget?: RookRoute;
  error = "";
  loading: number | null = null;
  paused = false;
  nnTrace: Array<{ ply: number; move: string; elapsedMs: number; nodes: number; depth: number }> = [];
  private makePlayer(target?: RookRoute): void {
    this.player.dispose(); this.error = ""; this.loading = null;
    if (this.engineKind === "nnue") {
      const player = new NnueWorkerPlayer(target ?? this.rookTarget); this.rookTarget = player.target;
      player.onLoading = value => { if (this.player !== player || !this.state.thinking) return; const step = Math.floor(value * 100); if (this.loading !== step) { this.loading = step; this.onChange(); } };
      this.player = player;
    } else this.player = new HeuristicWorkerPlayer();
  }
  private abort: AbortController | null = null;
  state: ControllerState;
  onChange: () => void = () => undefined;
  difficulty: Difficulty = "normal";

  constructor() {
    this.state = { mode: "local", position: new Position(), history: [], sfenHistory: [START_SFEN],
      outcome: null, humanColor: null, thinking: false, progress: null };
  }

  start(mode: Mode, humanColor: Color | null = null, difficulty: Difficulty = "normal"): void {
    this.cancelSearch();
    this.difficulty = difficulty; this.paused = false; this.rookTarget = undefined; this.nnTrace = []; this.makePlayer();
    this.state = { mode, position: new Position(), history: [], sfenHistory: [START_SFEN], outcome: null,
      humanColor, thinking: false, progress: null };
    this.persist(); this.onChange();
    if (mode === "cpu" && humanColor === WHITE) void this.requestCpuMove();
  }

  load(sfen: string, history: string[], mode: Mode, humanColor: Color | null, outcome: GameOutcome | null): void {
    this.cancelSearch();
    this.nnTrace = [];
    this.makePlayer(this.player instanceof NnueWorkerPlayer ? this.player.target : undefined);
    const position = history.length ? Position.fromHistory(history, sfen) : new Position(sfen);
    this.state = { mode, position, history: [...history], sfenHistory: [START_SFEN, sfen],
      outcome, humanColor, thinking: false, progress: null };
    // Restoring a CPU turn must expose an explicit resume button after a reload.
    this.paused = mode === 'cpu' && !outcome && position.turn !== humanColor;
    this.persist(); this.onChange();
  }

  legalMovesFrom(square: number | null, drop?: number): Move[] {
    return this.state.position.legalMoves().filter((move) => drop ? move.from === null && move.piece === drop : move.from === square);
  }

  play(move: Move): void {
    if (this.state.outcome || this.state.thinking) return;
    const legal = this.state.position.legalMoves().find((candidate) => this.state.position.moveToUsi(candidate) === this.state.position.moveToUsi(move));
    if (!legal) throw new Error("Illegal move");
    this.apply(legal);
    if (!this.state.outcome && this.state.mode === "cpu") void this.requestCpuMove();
  }

  playUsi(usi: string): void {
    const move = this.state.position.legalMoves().find((candidate) => this.state.position.moveToUsi(candidate) === usi);
    if (!move) throw new Error(`Illegal move: ${usi}`);
    this.apply(move);
  }

  private apply(move: Move): void {
    const usi = this.state.position.moveToUsi(move);
    this.state.position.push(move);
    this.state.history.push(usi);
    this.state.sfenHistory.push(this.state.position.sfen());
    this.state.outcome = this.state.position.outcome();
    this.state.progress = null;
    this.persist(); this.onChange();
  }

  private async requestCpuMove(): Promise<void> {
    if (this.paused || this.state.thinking || this.state.mode !== "cpu" || this.state.outcome || this.state.position.turn === this.state.humanColor) return;
    const abort = new AbortController(); this.abort = abort;
    this.error = ""; this.state.thinking = true; this.onChange();
    try {
      const level = NNUE_LEVELS[this.nnLevel - 1]!;
      const budget = this.engineKind === "nnue" ? { milliseconds: level.milliseconds, maxDepth: level.maxDepth, level: this.nnLevel } : DIFFICULTIES[this.difficulty];
      const result = await this.player.chooseMove(this.gameState(), budget, abort.signal,
        progress => { if (this.abort !== abort) return; this.loading = null; this.state.progress = { depth: progress.depth, nodes: progress.nodes }; this.onChange(); });
      if (this.abort !== abort || abort.signal.aborted) return;
      if (this.engineKind === 'nnue') this.nnTrace.push({ ply: this.state.history.length + 1, move: result.move, elapsedMs: result.elapsedMs, nodes: result.nodes, depth: result.depth });
      this.abort = null; this.loading = null; this.state.thinking = false;
      this.playUsi(result.move);
    } catch (error) {
      if (this.abort !== abort) return;
      this.abort = null; this.loading = null; this.state.thinking = false;
      if (!(error instanceof DOMException && error.name === "AbortError")) this.error = error instanceof Error ? error.message : String(error);
      this.onChange();
    }
  }
  retryCpu(): void { this.paused = false; void this.requestCpuMove(); }
  exportNnGame(): object {
    const player = this.player instanceof NnueWorkerPlayer ? this.player : null;
    return { schemaVersion: 1, time: new Date().toISOString(), device: navigator.userAgent, level: this.nnLevel,
      model: player?.manifest, requestedRoute: this.rookTarget, actualRoute: this.state.humanColor === null ? null : guidedRoute(this.state.history, opposite(this.state.humanColor)),
      routeAudit: player?.routeAudit, moves: this.nnTrace, history: this.state.history, outcome: this.state.outcome,
      completed: !!this.state.outcome, traceComplete: this.nnTrace.length === this.state.history.filter((_, ply) => ply % 2 !== this.state.humanColor).length,
      sfen: this.state.position.sfen(), note: 'Manual game record. Requested route is not the observed route. No certified Elo.' };
  }
  pauseCpu(): void { this.paused = true; this.cancelSearch(); this.state.thinking = false; this.state.progress = null; this.loading = null; this.onChange(); }
  useAbp(): void { this.cancelSearch(); this.engineKind = "abp"; this.makePlayer(); this.paused = false; this.state.thinking = false; this.persist(); void this.requestCpuMove(); }

  canHumanMove(): boolean {
    if (this.state.outcome || this.state.thinking) return false;
    return this.state.mode === "local" || this.state.mode === "online" || this.state.position.turn === this.state.humanColor;
  }

  undo(): void {
    if (this.state.mode === "online" || !this.state.history.length) return;
    this.cancelSearch();
    const count = this.state.mode === "cpu" && this.state.position.turn === this.state.humanColor && this.state.history.length >= 2 ? 2 : 1;
    for (let index = 0; index < count && this.state.history.length; index += 1) {
      this.state.position.pop(); this.state.history.pop(); this.state.sfenHistory.pop();
    }
    this.state.outcome = null; this.state.thinking = false; this.state.progress = null;
    this.nnTrace = this.nnTrace.filter(move => move.ply <= this.state.history.length);
    this.persist(); this.onChange();
    if (this.state.mode === 'cpu' && this.state.position.turn !== this.state.humanColor) void this.requestCpuMove();
  }

  resign(color = this.state.position.turn): void {
    this.cancelSearch();
    this.state.outcome = { winner: opposite(color), reason: "resignation" };
    this.persist(); this.onChange();
  }

  declareEnteringKing(): void {
    if (!this.state.outcome && this.state.position.canDeclareEnteringKing()) {
      this.cancelSearch();
      this.state.outcome = { winner: this.state.position.turn, reason: "entering-king" };
      this.persist(); this.onChange();
    }
  }

  gameState(): GameState {
    return { schemaVersion: 1, sfen: this.state.position.sfen(), history: [...this.state.history],
      turn: this.state.position.turn, outcome: this.state.outcome, lastMove: this.state.history.at(-1) ?? null };
  }

  private cancelSearch(): void { const old = this.abort; this.abort = null; old?.abort(); this.player.dispose(); }
  dispose(): void { this.cancelSearch(); }

  private persist(): void {
    if (this.state.mode === "online") return;
    localStorage.setItem("alphasho.game", JSON.stringify({ schemaVersion: 1, mode: this.state.mode,
      sfen: this.state.position.sfen(), history: this.state.history, outcome: this.state.outcome,
      humanColor: this.state.humanColor, difficulty: this.difficulty, engineKind: this.engineKind, nnLevel: this.nnLevel, rookTarget: this.player instanceof NnueWorkerPlayer ? this.player.target : undefined }));
  }
}
