import { describe, it, expect } from 'vitest';
import { NNUE_LEVELS, parseInfo, selectCandidate } from '../src/player/nnue-config';
import { guidedRoots } from '../src/player/rook-guide';
describe('NNUE safety and levels', () => {
  it('uses ten increasing budgets with no weakening in the top two levels', () => {
    expect(NNUE_LEVELS).toHaveLength(10); expect(NNUE_LEVELS[9]!.milliseconds).toBe(5000);
    expect(NNUE_LEVELS.slice(8).every(l => l.multiPV === 1)).toBe(true);
    expect(NNUE_LEVELS.every((l, i) => i === 0 || l.milliseconds > NNUE_LEVELS[i - 1]!.milliseconds)).toBe(true);
  });
  it('never mixes bounded, mate, or different-depth candidates', () => {
    const a = parseInfo('info depth 3 multipv 1 score cp 100 nodes 300 time 3 pv 7g7f')!;
    const b = parseInfo('info depth 2 multipv 2 score cp 90 nodes 200 time 2 pv 2g2f')!;
    const c = parseInfo('info depth 3 multipv 2 score cp 90 upperbound nodes 300 time 3 pv 2g2f')!;
    expect(selectCandidate([a, b, c], '7g7f', 1, () => .99)).toBe('7g7f');
    const mate = parseInfo('info depth 3 multipv 1 score mate 3 nodes 300 time 3 pv 7g7f')!;
    expect(selectCandidate([mate, { ...b, depth: 3 }], '7g7f', 1, () => .99)).toBe('7g7f');
  });
  it('can weaken among reliable candidates and keeps top levels strongest', () => {
    const a = parseInfo('info depth 3 multipv 1 score cp 100 nodes 300 time 3 pv 7g7f')!;
    const b = parseInfo('info depth 3 multipv 2 score cp 90 nodes 300 time 3 pv 2g2f')!;
    expect(selectCandidate([a, b], '7g7f', 1, () => .99)).toBe('2g2f');
    expect(selectCandidate([a, b], '7g7f', 10, () => .99)).toBe('7g7f');
  });
  it('rotates rook-family guidance for both colors', () => {
    expect(guidedRoots([], 0, 'furibisha').roots).toEqual(['7g7f']);
    expect(guidedRoots(['7g7f'], 1, 'furibisha').roots).toEqual(['3c3d']);
    const h = ['7g7f', '3c3d', '2h6h', '8b4b'];
    expect(guidedRoots(h, 0, 'furibisha').route).toBe('furibisha');
    expect(guidedRoots([...h, '6g6f'], 1, 'furibisha').route).toBe('furibisha');
  });
});
