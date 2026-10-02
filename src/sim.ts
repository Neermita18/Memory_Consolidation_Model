import { buildNeurons, DEFAULT_LAYOUT, mulberry32 } from './layout.ts';
import type { LayoutConfig, Neuron, Region } from './layout.ts';

export type Phase = 'idle' | 'encode' | 'consolidate' | 'recall';
export type EdgeKind = 'hh' | 'hc' | 'cc'; // hippocampus->hippocampus, hippocampus->cortex, cortex->cortex

export interface Edge {
  pre: number;
  post: number;
  kind: EdgeKind;
}

export type SimEvent =
  | { type: 'ensemble'; region: Region; index: number }
  | { type: 'ripple' }
  | { type: 'phase'; phase: Phase }
  | { type: 'recall'; score: number; total: number };

/** All times in ms; the simulation advances in 1 ms steps. */
export const PARAMS = {
  WMAX: 0.3,
  THRESHOLD: 1,
  DELAY: 6,
  REFRACT: 5,
  TAU_V: 20,
  TAU_TRACE: 60,
  NOISE_P: 0.0003,
  DRIVE: 0.15,
  ENC_STIM_MS: 120,
  ENC_GAP_MS: 500,
  ENC_TRIALS: 4,
  RIPPLE_EVERY_MS: 500,
  RIPPLES: 24,
  HC_DECAY_PER_S: 0.05,
  RECALL_CUES: 3,
  RECALL_EVERY_MS: 1000,
  RECALL_WINDOW_MS: 200,
  LR_HIP_ENCODE: 0.02,
  LR_CTX_ENCODE: 0.002,
  LR_CTX_SLEEP: 0.004,
} as const;

const REGION_INDEX: Record<Region, number> = { hip: 0, ctx: 1 };

/**
 * A small spiking network (leaky integrate-and-fire neurons, delayed synapses,
 * trace-based Hebbian plasticity) with a hippocampal and a cortical population.
 *
 *  encode      – the experience drives both regions; hippocampal links form fast, cortical links barely.
 *  consolidate – sleep: sharp-wave ripples replay the sequence from the hippocampus;
 *                replay drives cortex, slowly building direct cortico-cortical links.
 *  recall      – cue the first cortical element (optionally with the hippocampus silenced)
 *                and see how much of the sequence the cortex completes by itself.
 */
export class BrainSim {
  readonly cfg: LayoutConfig;
  readonly neurons: Neuron[];
  readonly n: number;
  readonly edges: Edge[] = [];
  readonly edgeW: Float32Array;

  phase: Phase = 'idle';
  timeMs = 0;
  phaseMs = 0;
  silenceHipInRecall = true;
  lastRecall = { score: 0, total: 0 };

  onSpike?: (id: number) => void;
  onEvent?: (e: SimEvent) => void;

  private readonly rng: () => number;
  private readonly v: Float32Array;
  private readonly trace: Float32Array;
  private readonly refr: Int16Array;
  private readonly gain: Float32Array;
  private readonly pending: number[][];
  private readonly outgoing: number[][];
  private readonly incoming: number[][];
  private readonly byKind: Record<EdgeKind, number[]> = { hh: [], hc: [], cc: [] };
  private readonly members: number[][][]; // [region][ensemble] -> neuron ids
  private readonly acc: Float64Array;
  private readonly lastFire: Float64Array;
  private readonly recallSeen: Uint8Array;
  private hipSilenced = false;
  private lrHip: number = 0;
  private lrCtx: number = 0;
  private readonly traceDecay = Math.exp(-1 / PARAMS.TAU_TRACE);
  private readonly vDecay = 1 - 1 / PARAMS.TAU_V;
  private readonly accDecay = Math.exp(-1 / 8);

  constructor(cfg: LayoutConfig = DEFAULT_LAYOUT) {
    this.cfg = cfg;
    this.rng = mulberry32(cfg.seed + 1000);
    this.neurons = buildNeurons(cfg);
    this.n = this.neurons.length;
    this.v = new Float32Array(this.n);
    this.trace = new Float32Array(this.n);
    this.refr = new Int16Array(this.n);
    this.gain = new Float32Array(this.n).map(() => 0.8 + 0.4 * this.rng());
    this.pending = Array.from({ length: PARAMS.DELAY + 1 }, () => []);
    this.outgoing = Array.from({ length: this.n }, () => []);
    this.incoming = Array.from({ length: this.n }, () => []);

    const E = cfg.ensembles;
    this.members = [0, 1].map(() => Array.from({ length: E }, () => [] as number[]));
    for (const nr of this.neurons) {
      if (nr.ensemble >= 0) this.members[REGION_INDEX[nr.region]][nr.ensemble].push(nr.id);
    }

    // Plastic connections: structured so the sequence is stored in order.
    const connect = (pres: number[], posts: number[], kind: EdgeKind) => {
      for (const pre of pres)
        for (const post of posts) {
          const idx = this.edges.length;
          this.edges.push({ pre, post, kind });
          this.outgoing[pre].push(idx);
          this.incoming[post].push(idx);
          this.byKind[kind].push(idx);
        }
    };
    for (let k = 0; k < E; k++) {
      connect(this.members[0][k], this.members[1][k], 'hc');
      if (k < E - 1) {
        connect(this.members[0][k], this.members[0][k + 1], 'hh');
        connect(this.members[1][k], this.members[1][k + 1], 'cc');
      }
    }
    this.edgeW = new Float32Array(this.edges.length);
    this.acc = new Float64Array(2 * E);
    this.lastFire = new Float64Array(2 * E).fill(-1e9);
    this.recallSeen = new Uint8Array(this.n);
  }

  // ---------- control ----------

  reset(): void {
    this.edgeW.fill(0);
    this.clearState();
    this.phase = 'idle';
    this.phaseMs = 0;
    this.hipSilenced = false;
    this.lastRecall = { score: 0, total: 0 };
    this.emit({ type: 'phase', phase: 'idle' });
  }

  start(phase: Phase): void {
    this.clearState();
    this.phase = phase;
    this.phaseMs = 0;
    this.hipSilenced = phase === 'recall' && this.silenceHipInRecall;
    this.lrHip = phase === 'encode' ? PARAMS.LR_HIP_ENCODE : 0;
    this.lrCtx =
      phase === 'encode' ? PARAMS.LR_CTX_ENCODE : phase === 'consolidate' ? PARAMS.LR_CTX_SLEEP : 0;
    this.emit({ type: 'phase', phase });
  }

  /** Synchronously run the current phase to completion (used by tests). */
  runToEnd(maxMs = 60000): void {
    for (let i = 0; i < maxMs && this.phase !== 'idle'; i++) this.step();
  }

  /** Mean weight of an edge class, as a fraction of the maximum weight (0..1). */
  strength(kind: EdgeKind): number {
    const ids = this.byKind[kind];
    let s = 0;
    for (const i of ids) s += this.edgeW[i];
    return s / ids.length / PARAMS.WMAX;
  }

  get phaseDurationMs(): number {
    const E = this.cfg.ensembles;
    switch (this.phase) {
      case 'encode':
        return PARAMS.ENC_TRIALS * (E * PARAMS.ENC_STIM_MS + PARAMS.ENC_GAP_MS);
      case 'consolidate':
        return PARAMS.RIPPLES * PARAMS.RIPPLE_EVERY_MS;
      case 'recall':
        return PARAMS.RECALL_CUES * PARAMS.RECALL_EVERY_MS;
      default:
        return 1;
    }
  }

  // ---------- dynamics ----------

  step(): void {
    const P = PARAMS;
    this.timeMs++;
    this.phaseMs++;
    this.runScript();

    // Deliver spikes whose synaptic delay has elapsed.
    const slot = this.pending[this.timeMs % (P.DELAY + 1)];
    for (const src of slot)
      for (const ei of this.outgoing[src]) this.v[this.edges[ei].post] += this.edgeW[ei];
    slot.length = 0;

    for (let i = 0; i < this.n; i++) this.trace[i] *= this.traceDecay;

    for (let i = 0; i < this.n; i++) {
      if (this.refr[i] > 0) {
        this.refr[i]--;
        this.v[i] = 0;
        continue;
      }
      if (this.hipSilenced && i < this.cfg.hipCount) {
        this.v[i] = 0;
        continue;
      }
      let vi = this.v[i] * this.vDecay;
      if (this.rng() < P.NOISE_P) vi += 1.1;
      this.v[i] = vi;
      if (vi >= P.THRESHOLD) this.fire(i);
    }

    // Ensemble-level events (for audio / HUD)
    for (let r = 0; r < 2; r++)
      for (let k = 0; k < this.cfg.ensembles; k++) {
        const j = r * this.cfg.ensembles + k;
        this.acc[j] *= this.accDecay;
        if (this.acc[j] >= 4 && this.timeMs - this.lastFire[j] >= 100) {
          this.lastFire[j] = this.timeMs;
          this.emit({ type: 'ensemble', region: r === 0 ? 'hip' : 'ctx', index: k });
        }
      }

    if (this.phase === 'consolidate' && this.timeMs % 10 === 0) {
      const f = 1 - P.HC_DECAY_PER_S * 0.01;
      for (const i of this.byKind.hc) this.edgeW[i] *= f;
    }
  }

  private fire(i: number): void {
    const P = PARAMS;
    this.v[i] = 0;
    this.refr[i] = P.REFRACT;

    for (const ei of this.incoming[i]) {
      const e = this.edges[ei];
      const lr = e.kind === 'cc' ? this.lrCtx : this.lrHip;
      if (lr > 0) this.edgeW[ei] = Math.min(P.WMAX, this.edgeW[ei] + lr * this.trace[e.pre]);
    }

    this.trace[i] = 1;
    this.pending[(this.timeMs + P.DELAY) % (P.DELAY + 1)].push(i);
    this.recallSeen[i] = 1;

    const nr = this.neurons[i];
    if (nr.ensemble >= 0) this.acc[REGION_INDEX[nr.region] * this.cfg.ensembles + nr.ensemble]++;
    this.onSpike?.(i);
  }

  private drive(ensemble: number, regions: number[]): void {
    for (const r of regions)
      for (const id of this.members[r][ensemble]) this.v[id] += PARAMS.DRIVE * this.gain[id];
  }

  private inject(ensemble: number, region: number, amount: number): void {
    for (const id of this.members[region][ensemble]) this.v[id] += amount;
  }

  private runScript(): void {
    const P = PARAMS;
    const t = this.phaseMs;
    const E = this.cfg.ensembles;
    if (this.phase === 'idle') return;
    if (t > this.phaseDurationMs) return this.finish();

    if (this.phase === 'encode') {
      const period = E * P.ENC_STIM_MS + P.ENC_GAP_MS;
      const tt = t % period;
      if (tt < E * P.ENC_STIM_MS) this.drive(Math.floor(tt / P.ENC_STIM_MS), [0, 1]);
    } else if (this.phase === 'consolidate') {
      if (t % P.RIPPLE_EVERY_MS === 100) {
        this.inject(0, 0, 1.5); // sharp-wave ripple seeds hippocampal replay
        this.emit({ type: 'ripple' });
      }
    } else if (this.phase === 'recall') {
      const m = t % P.RECALL_EVERY_MS;
      if (m === 100) {
        this.recallSeen.fill(0);
        this.inject(0, 1, 1.5); // cue only the first cortical element
      } else if (m === 100 + P.RECALL_WINDOW_MS) {
        this.scoreRecall();
      }
    }
  }

  private scoreRecall(): void {
    const need = Math.ceil(this.cfg.ensembleSize / 2);
    let score = 0;
    for (let k = 0; k < this.cfg.ensembles; k++) {
      const seen = this.members[1][k].filter((id) => this.recallSeen[id]).length;
      if (seen >= need) score++;
    }
    this.lastRecall = { score, total: this.cfg.ensembles };
    this.emit({ type: 'recall', ...this.lastRecall });
  }

  private finish(): void {
    this.phase = 'idle';
    this.hipSilenced = false;
    this.lrHip = this.lrCtx = 0;
    this.emit({ type: 'phase', phase: 'idle' });
  }

  private clearState(): void {
    this.v.fill(0);
    this.trace.fill(0);
    this.refr.fill(0);
    this.acc.fill(0);
    this.recallSeen.fill(0);
    for (const s of this.pending) s.length = 0;
  }

  private emit(e: SimEvent): void {
    this.onEvent?.(e);
  }
}
