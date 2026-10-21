import type { BrowserPlayer, GameState, PlayerResult, SearchBudget, SearchProgress } from '../engine/types';
import { Position } from '../engine/position';
import { loadModel } from './nnue-model';
import { guidedRoots, type RookRoute } from './rook-guide';
import type { NnueManifest } from './nnue-config';
export interface NnueResult extends PlayerResult { engineElapsedMs: number; memoryBytes: number }
export class NnueWorkerPlayer implements BrowserPlayer {
  private worker: Worker | null = null;
  private initialized: Promise<void> | null = null;
  private serial = 0;
  private generation = 0;
  private pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void; progress?: (p: SearchProgress) => void }>();
  manifest: NnueManifest | null = null;
  loadMetrics = { modelMs: 0, workerMs: 0 };
  onLoading: (fraction: number) => void = () => {};
  readonly target: RookRoute;
  readonly routeAudit: Array<{ ply: number; requested: RookRoute; actual: RookRoute; reason: string }> = [];
  constructor(target?: RookRoute) { this.target = target ?? (crypto.getRandomValues(new Uint8Array(1))[0]! & 1 ? 'furibisha' : 'ibisha'); }
  static supported(): boolean { return globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined' && typeof WebAssembly !== 'undefined' && WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11])); }
  private async init(signal: AbortSignal): Promise<void> {
    if (this.initialized) return this.initialized;
    if (!NnueWorkerPlayer.supported()) throw new Error('NNUE requires cross-origin isolation, WebAssembly SIMD and a modern browser');
    const generation = this.generation;
    this.initialized = (async () => {
      const loadStart = performance.now();
      const { manifest, bytes } = await loadModel(signal, p => this.onLoading(p)); this.manifest = manifest;
      this.loadMetrics.modelMs = performance.now() - loadStart;
      const workerStart = performance.now();
      signal.throwIfAborted(); this.worker = new Worker(new URL('./nnue-worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = ({ data }) => {
        const pending = this.pending.get(data.id); if (!pending) return;
        if (data.type === 'progress') pending.progress?.(data.progress);
        else { this.pending.delete(data.id); if (data.type === 'error') pending.reject(new Error(data.message)); else pending.resolve(data.result ?? data.cp ?? data); }
      };
      this.worker.onerror = event => { this.fail(new Error(event.message || 'NNUE Worker crashed')); };
      await this.call({ type: 'init', base: new URL(import.meta.env.BASE_URL, location.origin).href, bytes }, [bytes], undefined, 60000);
      this.loadMetrics.workerMs = performance.now() - workerStart;
    })().catch(error => { if (generation === this.generation) this.dispose(); throw error; });
    return this.initialized;
  }
  private call(data: Record<string, unknown>, transfer: Transferable[] = [], progress?: (p: SearchProgress) => void, timeout = 10000): Promise<unknown> {
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.fail(new Error('NNUE Worker response timeout')); }, timeout);
      this.pending.set(id, { resolve: v => { clearTimeout(timer); resolve(v); }, reject: e => { clearTimeout(timer); reject(e); }, progress });
      this.worker!.postMessage({ ...data, id }, transfer);
    });
  }
  async chooseMove(state: GameState, budget: SearchBudget, signal: AbortSignal, onProgress?: (p: SearchProgress) => void): Promise<PlayerResult> {
    const generation = this.generation;
    const abort = () => this.fail(new DOMException('Search cancelled', 'AbortError'));
    signal.addEventListener('abort', abort, { once: true });
    try {
      signal.throwIfAborted(); await this.init(signal); signal.throwIfAborted();
      const start = performance.now();
      const p = state.history.length ? Position.fromHistory(state.history, state.sfen) : new Position(state.sfen);
      let roots: string[] | undefined;
      if (budget.guidance !== false && (state.history.length || state.sfen === new Position().sfen())) {
        const guide = guidedRoots(state.history, state.turn, this.target); this.routeAudit.push({ ply: state.history.length, requested: this.target, actual: guide.route, reason: guide.reason });
        if (guide.roots.length < p.legalMoves().length) roots = guide.roots;
      }
      const position = state.history.length ? 'position startpos moves ' + state.history.join(' ') : 'position sfen ' + state.sfen;
      const remaining = budget.milliseconds - (performance.now() - start);
      if (remaining <= 5) throw new Error('NNUE preparation exceeded move budget');
      const result = await this.call({ type: 'go', position, budget: { ...budget, milliseconds: remaining }, level: budget.level ?? 10, roots, benchmark: budget.benchmark }, [], onProgress, remaining + 250) as NnueResult;
      signal.throwIfAborted(); if (performance.now() - start > budget.milliseconds + 250) throw new Error('NNUE returned an expired result');
      if (!p.legalMoves().some(m => p.moveToUsi(m) === result.move)) throw new Error('NNUE returned an illegal move');
      result.elapsedMs = performance.now() - start; return result;
    } catch (error) { if (generation === this.generation) this.dispose(); throw error; }
    finally { signal.removeEventListener('abort', abort); }
  }
  async evaluate(position: string, signal: AbortSignal): Promise<number> { await this.init(signal); return await this.call({ type: 'eval', position }) as number; }
  private fail(error: Error) { this.generation++; for (const p of this.pending.values()) p.reject(error); this.pending.clear(); this.worker?.terminate(); this.worker = null; this.initialized = null; }
  dispose(): void { this.fail(new DOMException('Search cancelled', 'AbortError')); }
}
