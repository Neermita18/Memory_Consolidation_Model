import { describe, expect, it } from 'vitest';
import { DEFAULT_LAYOUT } from './layout.ts';
import { BrainSim } from './sim.ts';

const run = (s: BrainSim, phase: Parameters<BrainSim['start']>[0]) => {
  s.start(phase);
  s.runToEnd();
};

describe('layout', () => {
  it('creates the requested populations and engrams', () => {
    const s = new BrainSim();
    expect(s.n).toBe(DEFAULT_LAYOUT.hipCount + DEFAULT_LAYOUT.ctxCount);
    for (const region of ['hip', 'ctx']) {
      for (let k = 0; k < DEFAULT_LAYOUT.ensembles; k++) {
        const size = s.neurons.filter((n) => n.region === region && n.ensemble === k).length;
        expect(size).toBe(DEFAULT_LAYOUT.ensembleSize);
      }
    }
  });
});

describe('memory consolidation', () => {
  it('starts with no memory trace and fails recall', () => {
    const s = new BrainSim();
    run(s, 'recall');
    expect(s.strength('hc')).toBe(0);
    expect(s.lastRecall.score).toBe(1); // only the cued element
  });

  it('encoding builds hippocampal links fast and cortical links weakly', () => {
    const s = new BrainSim();
    run(s, 'encode');
    expect(s.strength('hc')).toBeGreaterThan(0.9);
    expect(s.strength('cc')).toBeLessThan(0.5);
  });

  it('a cue without consolidation does not complete the sequence', () => {
    const s = new BrainSim();
    run(s, 'encode');
    run(s, 'recall');
    expect(s.lastRecall.score).toBeLessThan(s.lastRecall.total);
  });

  it('sleep strengthens cortical links and reduces hippocampal dependence', () => {
    const s = new BrainSim();
    run(s, 'encode');
    const cc = s.strength('cc');
    const hc = s.strength('hc');
    run(s, 'consolidate');
    expect(s.strength('cc')).toBeGreaterThan(cc + 0.2);
    expect(s.strength('hc')).toBeLessThan(hc);
  });

  it('after sleep, cortex completes the whole sequence with the hippocampus silenced', () => {
    const s = new BrainSim();
    s.silenceHipInRecall = true;
    run(s, 'encode');
    run(s, 'consolidate');
    run(s, 'recall');
    expect(s.lastRecall).toEqual({ score: 4, total: 4 });
  });

  it('emits ripple and ensemble events during sleep', () => {
    const s = new BrainSim();
    run(s, 'encode');
    const types: string[] = [];
    s.onEvent = (e) => types.push(e.type);
    run(s, 'consolidate');
    expect(types.filter((t) => t === 'ripple')).toHaveLength(24);
    expect(types).toContain('ensemble');
  });

  it('is deterministic for a given seed', () => {
    const a = new BrainSim();
    const b = new BrainSim();
    for (const s of [a, b]) {
      run(s, 'encode');
      run(s, 'consolidate');
    }
    expect(Array.from(a.edgeW)).toEqual(Array.from(b.edgeW));
  });
});
