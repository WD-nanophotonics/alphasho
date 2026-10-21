import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import factory from '../public/nnue/engine/yaneuraou.js';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = path.join(root, 'public/nnue');
const lines = []; let consumer = null;
const output = line => { lines.push(line); if (lines.length > 100) lines.shift(); consumer?.(line); };
const module = await factory({ print: output, printErr: output, wasmBinary: fs.readFileSync(path.join(base, 'engine/yaneuraou.wasm')), locateFile: file => path.join(base, 'engine', file) });
console.log('WASM initialized');
module.addMessageListener(output);
module.postMessage = command => module.ccall('alphasho_usi_command', 'number', ['string'], [command]);

function command(commands, prefix, seconds = 30) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { consumer = null; reject(new Error('Timeout ' + prefix + ': ' + lines.slice(-5).join('\n'))); }, seconds * 1000);
    consumer = line => {
      if (line.includes('Error!')) { clearTimeout(timer); consumer = null; reject(new Error(line)); }
      else if (line.startsWith(prefix)) { clearTimeout(timer); consumer = null; resolve(line); }
    };
    try { for (const c of commands) module.postMessage(c); } catch (error) { clearTimeout(timer); consumer=null; reject(error); }
  });
}
try {
  module.FS.mkdir('/eval'); module.FS.writeFile('/eval/nn.bin', fs.readFileSync(path.join(base, 'nn.bin')));
  await command(['usi'], 'usiok');
  await command(['setoption name USI_Hash value 16', 'setoption name Threads value 1', 'setoption name EvalDir value /eval', 'setoption name FV_SCALE value 16', 'setoption name BookFile value no_book', 'setoption name GenerateAllLegalMoves value true', 'setoption name NetworkDelay value 0', 'setoption name NetworkDelay2 value 0', 'setoption name MinimumThinkingTime value 1000', 'isready'], 'readyok');
  const fixtures = JSON.parse(fs.readFileSync(path.join(base, 'parity.json'))); const mismatches = []; let checked = 0;
  for (const [i, f] of fixtures.entries()) {
    if (process.env.NNUE_PROBE && i >= 5) break;
    if (process.env.NNUE_PROBE) { const fresh = await command(['position sfen '+f.sfen, 'e'], 'eval = '); console.log('fresh', i, fresh, 'native', f.cp); }
    const line = await command([f.position, 'e'], 'eval = '); const cp = Number(line.split('=').at(-1));
    if (cp !== f.cp) mismatches.push({ i, cp, expected: f.cp, sfen: f.sfen });
    checked++;
  }
  const report = { runtime: 'node-wasm', node: process.version, positions: checked, expectedPositions: fixtures.length, full: checked===fixtures.length, mismatches, memoryBytes: module.HEAPU8.byteLength, time: new Date().toISOString(), lastLines: lines.slice(-5) };
  fs.writeFileSync(path.join(root, process.env.NNUE_PROBE?'nnue-reports/wasm-probe.json':'nnue-reports/wasm-parity.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify({ ...report, mismatches: report.mismatches.slice(0, 5), mismatchCount: report.mismatches.length }));
  if (mismatches.length) process.exitCode = 1;
} catch (error) { console.error('AUDIT ERROR', error, lines.slice(-20)); process.exitCode=1; } finally { module.terminate(); }
