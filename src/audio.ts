import type { Region } from './layout.ts';
import type { Phase } from './sim.ts';
import { ensembleFreq } from './scale.ts';

const MOODS: Record<Phase, { gain: number; cutoff: number; lfoDepth: number; lfoHz: number }> = {
  idle: { gain: 0.04, cutoff: 300, lfoDepth: 0, lfoHz: 0.5 },
  encode: { gain: 0.06, cutoff: 800, lfoDepth: 0.005, lfoHz: 1.5 },
  consolidate: { gain: 0.09, cutoff: 220, lfoDepth: 0.04, lfoHz: 0.8 }, // slow "delta-wave" swell
  recall: { gain: 0.05, cutoff: 600, lfoDepth: 0.01, lfoHz: 1 },
};

/**
 * Web Audio sonification: a drone whose colour follows the brain state, a pitched
 * note whenever an ensemble fires (the motif IS the memory), and a noise burst per ripple.
 * Notes within a region are spaced >= 70 ms so time-compressed replay stays audible as a melody.
 */
export class AudioEngine {
  enabled = false;
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private droneGain!: GainNode;
  private droneFilter!: BiquadFilterNode;
  private lfo!: OscillatorNode;
  private lfoDepth!: GainNode;
  private noise!: AudioBuffer;
  private lastStart: Record<Region, number> = { hip: 0, ctx: 0 };

  async setEnabled(on: boolean): Promise<void> {
    if (on && !this.ctx) this.build();
    if (!this.ctx) return;
    if (on) await this.ctx.resume();
    this.enabled = on;
    this.master.gain.setTargetAtTime(on ? 0.8 : 0, this.ctx.currentTime, 0.05);
  }

  setPhase(phase: Phase): void {
    if (!this.ctx) return;
    const m = MOODS[phase];
    const t = this.ctx.currentTime;
    this.droneGain.gain.setTargetAtTime(m.gain, t, 0.4);
    this.droneFilter.frequency.setTargetAtTime(m.cutoff, t, 0.4);
    this.lfoDepth.gain.setTargetAtTime(m.lfoDepth, t, 0.4);
    this.lfo.frequency.setTargetAtTime(m.lfoHz, t, 0.4);
  }

  ensemble(region: Region, index: number): void {
    if (!this.enabled || !this.ctx) return;
    const now = this.ctx.currentTime;
    const start = Math.max(now, this.lastStart[region] + 0.07);
    if (start - now > 0.5) return; // drop notes if replay outruns the queue
    this.lastStart[region] = start;

    const osc = this.ctx.createOscillator();
    osc.type = region === 'hip' ? 'triangle' : 'sine';
    osc.frequency.value = ensembleFreq(region, index);
    const g = this.ctx.createGain();
    const peak = region === 'hip' ? 0.05 : 0.09;
    const decay = region === 'hip' ? 0.25 : 0.6;
    g.gain.setValueAtTime(0.0001, start);
    g.gain.linearRampToValueAtTime(peak, start + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, start + decay);
    osc.connect(g).connect(this.master);
    osc.start(start);
    osc.stop(start + decay + 0.05);
  }

  ripple(): void {
    if (!this.enabled || !this.ctx) return;
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 2400;
    bp.Q.value = 4;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(0.07, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
    src.connect(bp).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + 0.15);
  }

  private build(): void {
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.connect(ctx.destination);

    this.droneFilter = ctx.createBiquadFilter();
    this.droneFilter.type = 'lowpass';
    this.droneFilter.frequency.value = 300;
    this.droneGain = ctx.createGain();
    this.droneGain.gain.value = MOODS.idle.gain;
    this.droneFilter.connect(this.droneGain).connect(this.master);

    for (const [type, freq] of [['sine', 55], ['sine', 82.41], ['triangle', 110]] as const) {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = freq;
      o.connect(this.droneFilter);
      o.start();
    }
    this.lfo = ctx.createOscillator();
    this.lfo.frequency.value = 0.5;
    this.lfoDepth = ctx.createGain();
    this.lfoDepth.gain.value = 0;
    this.lfo.connect(this.lfoDepth).connect(this.droneGain.gain);
    this.lfo.start();

    this.noise = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.2), ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
}
