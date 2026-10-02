export type Region = 'hip' | 'ctx';

export interface Neuron {
  id: number;
  region: Region;
  /** Index of the memory ensemble this neuron belongs to, or -1 for background neurons. */
  ensemble: number;
  pos: [number, number, number];
}

export interface LayoutConfig {
  hipCount: number;
  ctxCount: number;
  ensembles: number;
  ensembleSize: number;
  seed: number;
}

export const DEFAULT_LAYOUT: LayoutConfig = {
  hipCount: 64,
  ctxCount: 360,
  ensembles: 4,
  ensembleSize: 8,
  seed: 7,
};

/** The "experience" is a sequence of elements, each stored in a different cortical area. */
export const ENSEMBLE_LABELS = ['Sight', 'Sound', 'Place', 'Meaning'];
export const ENSEMBLE_COLORS = ['#ff7aa2', '#ffd166', '#7bffb2', '#8fa8ff'];

/** Half-extents of the brain ellipsoid (x = left/right, y = up, z = front). */
export const BRAIN_RADII: [number, number, number] = [0.95, 0.8, 1.3];

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A C-shaped arch standing in for the left (side 0) / right (side 1) hippocampus. */
export function hippocampusPoint(side: 0 | 1, t: number): [number, number, number] {
  const s = side === 0 ? -1 : 1;
  const phi = Math.PI * (0.15 + 0.8 * t);
  return [s * (0.38 + 0.1 * t), -0.15 + 0.35 * Math.sin(phi), -0.1 + 0.45 * Math.cos(phi)];
}

const ANCHORS: [number, number, number][] = [
  [0, 0.25, -1], // back  -> Sight
  [1, -0.2, 0.15], // side  -> Sound
  [0.25, 1, -0.3], // top   -> Place
  [-0.1, 0.35, 1], // front -> Meaning
];

export function buildNeurons(cfg: LayoutConfig = DEFAULT_LAYOUT): Neuron[] {
  const rng = mulberry32(cfg.seed);
  const neurons: Neuron[] = [];

  // Hippocampus
  const perSide = cfg.hipCount / 2;
  const stride = Math.floor(cfg.hipCount / cfg.ensembles);
  for (let i = 0; i < cfg.hipCount; i++) {
    const side = (i % 2) as 0 | 1;
    const t = (i >> 1) / (perSide - 1);
    const p = hippocampusPoint(side, t);
    const k = Math.floor(i / stride);
    const within = i - k * stride;
    neurons.push({
      id: i,
      region: 'hip',
      ensemble: k < cfg.ensembles && within < cfg.ensembleSize ? k : -1,
      pos: [p[0] + (rng() - 0.5) * 0.05, p[1] + (rng() - 0.5) * 0.05, p[2] + (rng() - 0.5) * 0.05],
    });
  }

  // Cortex: Fibonacci sphere squashed into a brain-shaped ellipsoid
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let j = 0; j < cfg.ctxCount; j++) {
    const y = 1 - (2 * (j + 0.5)) / cfg.ctxCount;
    const r = Math.sqrt(1 - y * y);
    const phi = j * golden;
    const d = [Math.cos(phi) * r, y, Math.sin(phi) * r];
    const k = 0.93 + rng() * 0.05;
    neurons.push({
      id: cfg.hipCount + j,
      region: 'ctx',
      ensemble: -1,
      pos: [d[0] * BRAIN_RADII[0] * k, d[1] * BRAIN_RADII[1] * k, d[2] * BRAIN_RADII[2] * k],
    });
  }

  // Cortical engrams: the nearest free neurons to each anchor direction
  const ctx = neurons.filter((n) => n.region === 'ctx');
  for (let k = 0; k < cfg.ensembles; k++) {
    const a = ANCHORS[k % ANCHORS.length];
    const len = Math.hypot(a[0], a[1], a[2]);
    const target = [
      (a[0] / len) * BRAIN_RADII[0],
      (a[1] / len) * BRAIN_RADII[1],
      (a[2] / len) * BRAIN_RADII[2],
    ];
    const dist = (n: Neuron) =>
      Math.hypot(n.pos[0] - target[0], n.pos[1] - target[1], n.pos[2] - target[2]);
    ctx
      .filter((n) => n.ensemble === -1)
      .sort((p, q) => dist(p) - dist(q))
      .slice(0, cfg.ensembleSize)
      .forEach((n) => (n.ensemble = k));
  }
  return neurons;
}
