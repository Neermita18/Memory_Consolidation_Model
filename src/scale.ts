import type { Region } from './layout.ts';

/** The "memory" is a four-note motif: one note per element of the experience. */
export const MOTIF_SEMITONES = [0, 4, 7, 12];

export function ensembleFreq(region: Region, index: number, root = 220): number {
  const semis = MOTIF_SEMITONES[index % MOTIF_SEMITONES.length] + (region === 'hip' ? 12 : 0);
  return root * Math.pow(2, semis / 12);
}
