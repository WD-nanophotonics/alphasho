/// <reference lib="webworker" />
import { NNUE_LEVELS, parseInfo, selectCandidate, type Candidate } from './nnue-config';
interface Module {
  FS: { mkdir(path: string): void; writeFile(path: string, bytes: Uint8Array): void };
  postMessage(command: string): void; addMessageListener(listener: (line: string) => void): void;
  HEAPU8: Uint8Array;
  ccall(name: string, result: string, args: string[], values: string[]): number;
}
let engine: Module | null = null;
let listener: ((line: string) => void) | null = null;
function wait(prefix: string, seconds = 45): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const lines: string[] = []; const timer = setTimeout(() => { listener = null; reject(new Error('Engine timeout: ' + prefix)); }, seconds * 1000);
    listener = line => {
      if (line.includes('Error!')) { clearTimeout(timer); listener = null; reject(new Error(line)); return; }
      lines.push(line);
      if (line.startsWith(prefix)) { clearTimeout(timer); listener = null; resolve(lines); }
    };
  });
}
async function initialize(base: string, bytes: ArrayBuffer) {
  const { default: factory } = await import(/* @vite-ignore */ base + 'nnue/engine/yaneuraou.js');
  engine = await factory({ print: (line: string) => listener?.(line), printErr: (line: string) => listener?.('Error! ' + line), locateFile: (file: string) => base + 'nnue/engine/' + file });
  engine!.addMessageListener(line => listener?.(line));
  engine!.postMessage = command => { engine!.ccall('alphasho_usi_command', 'number', ['string'], [command]); };

  engine!.FS.mkdir('/eval'); engine!.FS.writeFile('/eval/nn.bin', new Uint8Array(bytes));
  let ready = wait('usiok'); engine!.postMessage('usi'); await ready;
  for (const [key, value] of Object.entries({ USI_Hash: 16, Threads: 1, MultiPV: 1, EvalDir: '/eval', FV_SCALE: 16, BookFile: 'no_book', USI_Ponder: 'false', NetworkDelay: 0, NetworkDelay2: 0, MinimumThinkingTime: 1000, ResignValue: 99999, PvInterval: 0, GenerateAllLegalMoves: 'true', EnteringKingRule: 'NoEnteringKing' })) engine!.postMessage(`setoption name ${key} value ${value}`);
  ready = wait('readyok'); engine!.postMessage('isready'); await ready;
  engine!.postMessage('position startpos'); ready = wait('eval = '); engine!.postMessage('e'); await ready;
  engine!.postMessage('usinewgame');
}
self.onmessage = async ({ data }) => {
  const { id } = data;
  try {
    if (data.type === 'init') { await initialize(data.base, data.bytes); self.postMessage({ type: 'ready', id, memoryBytes: engine!.HEAPU8.byteLength }); return; }
    if (!engine) throw new Error('Engine not initialized');
    if (data.type === 'stop') { engine.postMessage('stop'); return; }
    if (data.type === 'eval') {
      engine.postMessage(data.position); const promise = wait('eval = ', 10); engine.postMessage('e');
      const lines = await promise; self.postMessage({ type: 'eval', id, cp: Number(lines.at(-1)!.split('=').at(-1)) }); return;
    }
    if (data.type !== 'go') return;
    if (listener) throw new Error('Concurrent engine command');
    const level = Number(data.level ?? 10), config = NNUE_LEVELS[level - 1]; if (!config) throw new Error('Invalid NNUE level');
    const budget = Number(data.budget?.milliseconds ?? config.milliseconds);
    const started = performance.now(); const candidates = new Map<string, Candidate>(); let last: Candidate | null = null; let finished = false; let lastProgress = -Infinity;
    const stop = setTimeout(() => engine?.postMessage('stop'), Math.max(5, budget - 50));
    const hard = setTimeout(() => { if (!finished) { listener = null; self.postMessage({ type: 'error', id, message: 'Search exceeded total deadline' }); } }, budget + 250);
    listener = line => {
      if (finished) return;
      const c = parseInfo(line);
      if (c) { candidates.set(`${c.depth}/${c.rank}`, c); last = c; if (performance.now() - lastProgress >= 100) { lastProgress = performance.now(); self.postMessage({ type: 'progress', id, progress: { depth: c.depth, nodes: c.nodes, elapsedMs: performance.now() - started, score: c.cp ?? (c.mate! > 0 ? 30000 : -30000), principalVariation: c.pv } }); } }
      if (line.includes('Error!')) { finished = true; clearTimeout(stop); clearTimeout(hard); listener = null; self.postMessage({ type: 'error', id, message: line }); }
      if (!line.startsWith('bestmove ')) return;
      finished = true; clearTimeout(stop); clearTimeout(hard); listener = null;
      const best = line.split(/\s+/)[1]!;
      const move = data.benchmark ? best : selectCandidate([...candidates.values()], best, level);
      self.postMessage({ type: 'result', id, result: { move, depth: last?.depth ?? 0, nodes: last?.nodes ?? 0, score: last?.cp ?? 0, elapsedMs: performance.now() - started, engineElapsedMs: last?.elapsedMs ?? 0, principalVariation: last?.pv ?? [], memoryBytes: engine!.HEAPU8.byteLength, candidates: [...candidates.values()] } });
    };
    engine.postMessage(`setoption name MultiPV value ${data.benchmark ? 1 : config.multiPV}`);
    engine.postMessage(data.position);
    let go = `go movetime ${Math.max(5, budget - 50)} depth ${data.budget?.maxDepth ?? config.maxDepth}`;
    if (data.budget?.nodes) go += ` nodes ${data.budget.nodes}`;
    if (data.roots?.length) go += ' searchmoves ' + data.roots.join(' ');
    engine.postMessage(go);
  } catch (error) { self.postMessage({ type: 'error', id, message: error instanceof Error ? error.message : String(error) }); }
};
