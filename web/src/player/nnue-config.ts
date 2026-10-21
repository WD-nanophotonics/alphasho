export const NNUE_LEVELS = [
  [100, 1, 8, 1000, 600], [150, 2, 8, 800, 450], [250, 3, 6, 600, 300],
  [400, 4, 4, 400, 200], [600, 6, 4, 250, 120], [900, 8, 3, 150, 80],
  [1300, 12, 2, 80, 40], [2000, 16, 2, 40, 20], [3000, 64, 1, 0, 0], [5000, 128, 1, 0, 0]
].map(([milliseconds, maxDepth, multiPV, lossLimit, temperature]) => ({
  milliseconds: milliseconds!, maxDepth: maxDepth!, multiPV: multiPV!, lossLimit: lossLimit!, temperature: temperature!
}));
export interface NnueManifest {
  schemaVersion: 1; modelVersion: string; round: number; format: string; bytes: number;
  sha256: string; modelUrl: string; engineVersion: string; fvScale: number;
}
export interface Candidate {
  depth: number; rank: number; cp?: number; mate?: number; bound: boolean;
  pv: string[]; nodes: number; elapsedMs: number;
}
export function parseInfo(line: string): Candidate | null {
  const p = line.trim().split(/\s+/); if (p[0] !== 'info' || !p.includes('pv') || !p.includes('score')) return null;
  const num = (k: string, d = 0) => p.includes(k) ? Number(p[p.indexOf(k) + 1]) : d;
  const c: Candidate = { depth: num('depth'), rank: num('multipv', 1), nodes: num('nodes'),
    elapsedMs: num('time'), pv: p.slice(p.indexOf('pv') + 1), bound: p.includes('lowerbound') || p.includes('upperbound') };
  if (p.includes('cp')) c.cp = num('cp'); if (p.includes('mate')) c.mate = num('mate');
  return c;
}
export function selectCandidate(candidates: Candidate[], best: string, level: number, random = Math.random): string {
  const config = NNUE_LEVELS[level - 1]; if (!config || config.multiPV === 1) return best;
  // Scores from different completed depths or bound-only lines must never be compared.
  const depths = [...new Set(candidates.map(c => c.depth))].sort((a, b) => b - a);
  for (const depth of depths) {
    const same = candidates.filter(c => c.depth === depth && !c.bound && c.pv[0]);
    const first = same.find(c => c.rank === 1 && c.pv[0] === best);
    if (!first) continue;
    // A deeper forced mate must not fall through to an older cp-scored pool.
    if (first.mate !== undefined) return best;
    if (first.cp === undefined) continue;
    const pool = same.filter(c => c.cp !== undefined && Number.isFinite(c.cp) && c.mate === undefined && first.cp! - c.cp! <= config.lossLimit);
    if (pool.length < 2) continue;
    const weights = pool.map(c => Math.exp((c.cp! - first.cp!) / config.temperature));
    let value = random() * weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < pool.length; i++) { value -= weights[i]!; if (value <= 0) return pool[i]!.pv[0]!; }
  }
  return best;
}
