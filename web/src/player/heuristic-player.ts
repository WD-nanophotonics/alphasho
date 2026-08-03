import { BrowserPlayer, GameState, PlayerResult, SearchBudget, SearchProgress } from "../engine/types";

type WorkerReply =
  | { type: "search/progress"; id: number; progress: SearchProgress }
  | { type: "search/result"; id: number; result: PlayerResult }
  | { type: "search/error"; id: number; message: string };

export class HeuristicWorkerPlayer implements BrowserPlayer {
  private worker: Worker | null = null;
  private serial = 0;

  chooseMove(state: GameState, budget: SearchBudget, signal: AbortSignal,
    onProgress?: (progress: SearchProgress) => void): Promise<PlayerResult> {
    this.dispose();
    const id = ++this.serial;
    const worker = new Worker(new URL("./heuristic-worker.ts", import.meta.url), { type: "module" });
    this.worker = worker;
    return new Promise((resolve, reject) => {
      const cleanup = () => { signal.removeEventListener("abort", abort); worker.terminate(); if (this.worker === worker) this.worker = null; };
      const abort = () => { worker.postMessage({ type: "search/cancel", id }); cleanup(); reject(new DOMException("Search cancelled", "AbortError")); };
      signal.addEventListener("abort", abort, { once: true });
      worker.onmessage = (event: MessageEvent<WorkerReply>) => {
        if (event.data.id !== id) return;
        if (event.data.type === "search/progress") onProgress?.(event.data.progress);
        else if (event.data.type === "search/result") { cleanup(); resolve(event.data.result); }
        else { cleanup(); reject(new Error(event.data.message)); }
      };
      worker.onerror = (event) => { cleanup(); reject(new Error(event.message || "ABP worker crashed")); };
      worker.postMessage({ type: "search/start", id, sfen: state.sfen, history: state.history, budget });
    });
  }

  dispose(): void { this.worker?.terminate(); this.worker = null; }
}

export const DIFFICULTIES = {
  easy: { milliseconds: 100, maxDepth: 2 },
  normal: { milliseconds: 500, maxDepth: 6 },
  hard: { milliseconds: 2_000, maxDepth: 64 }
} as const;
