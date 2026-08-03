import { Position } from "./engine/position";
import { Color, GameOutcome, GameState, Move, START_SFEN, WHITE, opposite } from "./engine/types";
import { DIFFICULTIES, HeuristicWorkerPlayer } from "./player/heuristic-player";

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
  readonly player = new HeuristicWorkerPlayer();
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
    this.difficulty = difficulty;
    this.state = { mode, position: new Position(), history: [], sfenHistory: [START_SFEN], outcome: null,
      humanColor, thinking: false, progress: null };
    this.persist(); this.onChange();
    if (mode === "cpu" && humanColor === WHITE) void this.requestCpuMove();
  }

  load(sfen: string, history: string[], mode: Mode, humanColor: Color | null, outcome: GameOutcome | null): void {
    this.cancelSearch();
    const position = history.length ? Position.fromHistory(history, sfen) : new Position(sfen);
    this.state = { mode, position, history: [...history], sfenHistory: [START_SFEN, sfen],
      outcome, humanColor, thinking: false, progress: null };
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
    if (this.state.mode !== "cpu" || this.state.outcome || this.state.position.turn === this.state.humanColor) return;
    this.abort = new AbortController();
    this.state.thinking = true; this.onChange();
    try {
      const result = await this.player.chooseMove(this.gameState(), DIFFICULTIES[this.difficulty], this.abort.signal,
        (progress) => { this.state.progress = { depth: progress.depth, nodes: progress.nodes }; this.onChange(); });
      this.state.thinking = false;
      this.playUsi(result.move);
    } catch (error) {
      this.state.thinking = false;
      if (!(error instanceof DOMException && error.name === "AbortError")) console.error(error);
      this.onChange();
    }
  }

  canHumanMove(): boolean {
    if (this.state.outcome || this.state.thinking) return false;
    return this.state.mode === "local" || this.state.mode === "online" || this.state.position.turn === this.state.humanColor;
  }

  undo(): void {
    if (this.state.mode === "online" || !this.state.history.length) return;
    this.cancelSearch();
    const count = this.state.mode === "cpu" && this.state.history.length >= 2 ? 2 : 1;
    for (let index = 0; index < count && this.state.history.length; index += 1) {
      this.state.position.pop(); this.state.history.pop(); this.state.sfenHistory.pop();
    }
    this.state.outcome = null; this.state.thinking = false; this.state.progress = null;
    this.persist(); this.onChange();
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

  private cancelSearch(): void { this.abort?.abort(); this.abort = null; this.player.dispose(); }
  dispose(): void { this.cancelSearch(); }

  private persist(): void {
    if (this.state.mode === "online") return;
    localStorage.setItem("alphasho.game", JSON.stringify({ schemaVersion: 1, mode: this.state.mode,
      sfen: this.state.position.sfen(), history: this.state.history, outcome: this.state.outcome,
      humanColor: this.state.humanColor, difficulty: this.difficulty }));
  }
}
