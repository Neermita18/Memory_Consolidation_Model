import { describe, expect, it } from 'vitest';
import { ensembleFreq } from './scale.ts';

describe('ensembleFreq', () => {
  it('uses the root for the first cortical element', () => {
    expect(ensembleFreq('ctx', 0)).toBeCloseTo(220);
  });
  it('rises across the motif', () => {
    const f = [0, 1, 2, 3].map((k) => ensembleFreq('ctx', k));
    expect([...f].sort((a, b) => a - b)).toEqual(f);
  });
  it('puts hippocampal notes an octave above cortical ones', () => {
    expect(ensembleFreq('hip', 2) / ensembleFreq('ctx', 2)).toBeCloseTo(2);
  });
});
