import './style.css';
import { registerSW } from 'virtual:pwa-register';
import { NnueWorkerPlayer, type NnueResult } from './player/nnue-player';
import type { GameState, SearchBudget } from './engine/types';
import { Position } from './engine/position';
const root = document.querySelector<HTMLDivElement>('#benchmark')!;
root.innerHTML = `<main class="about-shell"><h1>AlphaSho NN 第一代 · Benchmark</h1><p>iPhone Safari / Android Chrome：请在手机本身运行。浏览器模拟不代表手机算力。</p><p id="capability"></p><p id="device"></p><p id="model"></p><p id="status">Ready / 准备就绪</p><button id="quick">Quick / 快速基准</button> <button id="full">Full 48-position benchmark</button> <button id="parity">1024-position parity</button> <button id="stress">10-minute stress</button> <button id="stop">Stop</button> <button id="download">Export JSON</button><p><a href="./">返回对局 / Play</a></p><label>真实设备 / Physical device <input id="physical" type="checkbox"></label><p>设备型号 / Model <input id="device-model" placeholder="例如 iPhone 13 / Pixel 8"></p><p>人工验收：分别完成1档、10档各一盘，再勾选。</p><label><input id="level1" type="checkbox">Level 1 complete</label> <label><input id="level10" type="checkbox">Level 10 complete</label><pre id="results" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></main>`;
const text = (id: string, value: string) => { document.getElementById(id)!.textContent = value; };
text('capability', `crossOriginIsolated=${crossOriginIsolated}; SharedArrayBuffer=${typeof SharedArrayBuffer !== 'undefined'}; secure=${isSecureContext}; serviceWorker=${'serviceWorker' in navigator}; controlled=${!!navigator.serviceWorker?.controller}`);
text('device', navigator.userAgent);
const buildLabel = document.createElement('p'); buildLabel.textContent = `Controller ${NnueWorkerPlayer.policyHash.slice(0,12)} · WASM ${NnueWorkerPlayer.engineSha.slice(0,12)}`; document.getElementById('device')!.after(buildLabel);
interface Entry { kind: string; sfen: string; nodes?: number; elapsedMs?: number; engineElapsedMs?: number; nps?: number; depth?: number; memoryBytes?: number; cp?: number; expectedCp?: number; ok?: boolean }
let player: NnueWorkerPlayer | null = null, abort: AbortController | null = null;
const report = { schemaVersion: 1, policyHash: NnueWorkerPlayer.policyHash, engineWasmSha256: NnueWorkerPlayer.engineSha, startedAt: '', device: navigator.userAgent, entries: [] as Entry[], error: '', finished: false, runs: [] as Array<{ kind: string; startedAt: string; finishedAt: string; elapsedMs: number; loading: unknown; finished: boolean; error: string; entries: Entry[] }>, physicalDevice: false, deviceModel: '', level1GameCompleted: false, level10GameCompleted: false, model: null as unknown, loading: null as unknown };
async function run(kind: string) {
  if (abort) return;
  abort = new AbortController(); const signal = abort.signal; report.startedAt = new Date().toISOString(); report.error = ''; report.finished = false; report.entries = [];
  const runStarted = performance.now();
  player = new NnueWorkerPlayer(); player.onLoading = p => text('status', `Model ${Math.floor(p * 100)}%`);
  try {
    const fixtures = await fetch(import.meta.env.BASE_URL + 'nnue/benchmark.json', { signal }).then(r => r.json()) as Array<{ sfen: string; category: string }>;
    if (kind === 'parity') {
      const fixtures = await fetch(import.meta.env.BASE_URL + 'nnue/parity.json', { signal }).then(r => r.json()) as Array<{ position: string; sfen: string; cp: number }>;
      for (const f of fixtures) { signal.throwIfAborted(); const cp = await player.evaluate(f.position, signal); const ok = cp === f.cp; report.entries.push({ kind, sfen: f.sfen, cp, expectedCp: f.cp, ok }); text('status', `${report.entries.length}/${fixtures.length}; mismatch=${report.entries.filter(e => !e.ok).length}`); if (!ok) throw new Error('Integer evaluation mismatch'); }
    } else {
      const budgets: SearchBudget[] = kind === 'full' ? [{ milliseconds: 10000, maxDepth: 128, nodes: 10000 }, { milliseconds: 10000, maxDepth: 128, nodes: 50000 }, ...[100, 500, 2000, 5000].map(milliseconds => ({ milliseconds, maxDepth: 128 }))] : [500, 2000, 5000].map(milliseconds => ({ milliseconds, maxDepth: 128 }));
      const positions = kind === 'full' ? fixtures : fixtures.filter((_, i) => i % 6 === 0);
      const repeats = kind === 'full' ? 3 : 1; const deadline = performance.now() + 600000;
      stressLoop: do {
        for (let repeat = 0; repeat < repeats; repeat++) for (const f of positions) for (const budget of budgets) {
          signal.throwIfAborted(); const p = new Position(f.sfen);
          const state: GameState = { schemaVersion: 1, sfen: f.sfen, history: [], turn: p.turn, outcome: null, lastMove: null };
          const result = await player.chooseMove(state, { ...budget, level: 10, guidance: false, benchmark: true }, signal) as NnueResult;
          report.entries.push({ kind: `${kind}/${budget.nodes ? 'nodes-' + budget.nodes : budget.milliseconds + 'ms'}`, sfen: f.sfen, nodes: result.nodes, depth: result.depth, elapsedMs: result.elapsedMs, engineElapsedMs: result.engineElapsedMs, nps: result.nodes / Math.max(.001, result.elapsedMs / 1000), memoryBytes: result.memoryBytes });
          text('status', `${report.entries.length} searches · ${Math.round(report.entries.at(-1)!.nps!)} actual nodes/s`);
          text('results', JSON.stringify(report.entries.slice(-3), null, 2));
          if (kind === 'stress' && performance.now() >= deadline) break stressLoop;
        }
      } while (kind === 'stress' && performance.now() < deadline);
    }
    report.finished = true; text('status', 'Complete / 已完成');
  } catch (e) { report.error = e instanceof Error ? e.message : String(e); text('status', 'Stopped / ' + report.error); }
  finally { report.runs.push({ kind, startedAt: report.startedAt, finishedAt: new Date().toISOString(), elapsedMs: performance.now() - runStarted, loading: player?.loadMetrics, finished: report.finished, error: report.error, entries: report.entries.slice() }); report.model = player?.manifest; report.loading = player?.loadMetrics; text('model', JSON.stringify(player?.manifest)); player?.dispose(); player = null; abort = null; text('results', JSON.stringify(report, null, 2)); }
}
for (const id of ['quick', 'full', 'parity', 'stress']) document.getElementById(id)!.onclick = () => { void run(id); };
document.getElementById('stop')!.onclick = () => abort?.abort();
document.getElementById('download')!.onclick = () => {
  report.physicalDevice = (document.getElementById('physical') as HTMLInputElement).checked;
  report.deviceModel = (document.getElementById('device-model') as HTMLInputElement).value;
  report.level1GameCompleted = (document.getElementById('level1') as HTMLInputElement).checked;
  report.level10GameCompleted = (document.getElementById('level10') as HTMLInputElement).checked;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'alphasho-nn-device-report.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.addEventListener('visibilitychange', () => { if (document.hidden) abort?.abort(); });
registerSW({ immediate: true, onRegisteredSW: (url, registration) => { text('status', 'SW ' + url + ' ' + String(registration?.active?.state ?? registration?.installing?.state)); registration?.installing?.addEventListener('statechange', () => text('status', 'SW ' + String(registration?.active?.state ?? registration?.installing?.state))); }, onRegisterError: error => text('status', 'Service Worker: ' + String(error)) });
if ('serviceWorker' in navigator) {
  const isolationKey = 'alphasho.benchmark-isolation-v1:' + import.meta.env.BASE_URL;
  const reload = () => { if (!crossOriginIsolated && !sessionStorage.getItem(isolationKey)) { sessionStorage.setItem(isolationKey, '1'); location.reload(); } };
  navigator.serviceWorker.addEventListener('controllerchange', reload); void navigator.serviceWorker.ready.then(reload);
}
