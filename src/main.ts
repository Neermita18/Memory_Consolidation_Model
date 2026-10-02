import './style.css';
import { AudioEngine } from './audio.ts';
import { ENSEMBLE_COLORS, ENSEMBLE_LABELS } from './layout.ts';
import { BrainScene } from './scene.ts';
import { BrainSim } from './sim.ts';
import type { Phase, SimEvent } from './sim.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const LABELS: Record<Phase, string> = {
  idle: 'Idle',
  encode: 'Awake · encoding',
  consolidate: 'Asleep · consolidation',
  recall: 'Awake · recall test',
};
const CAPTIONS: Record<Phase, string> = {
  idle: 'An experience is a sequence of elements spread across cortex. Press “Experience” to learn one.',
  encode:
    'The experience arrives. Cortical areas fire in order while the hippocampus binds them together quickly (orange links). Direct cortical links barely form.',
  consolidate:
    'During sleep the hippocampus replays the sequence in sharp-wave ripples, time-compressed. Each replay nudges cortex to wire the elements together itself (green links).',
  recall:
    'Only the first element is cued. If the cortex has consolidated the memory it completes the sequence on its own — ideally even with the hippocampus silenced.',
};

const sim = new BrainSim();
const view = new BrainScene($<HTMLCanvasElement>('scene'), sim);
const audio = new AudioEngine();

// Chips: one per memory element, lit when its cortical ensemble fires
const chips = ENSEMBLE_LABELS.map((label, i) => {
  const el = document.createElement('div');
  el.className = 'chip';
  el.textContent = label;
  el.style.setProperty('--c', ENSEMBLE_COLORS[i]);
  $('chips').append(el);
  return el;
});

let queue: Phase[] = [];
function begin(phase: Phase): void {
  sim.silenceHipInRecall = $<HTMLInputElement>('silence').checked;
  sim.start(phase);
}

sim.onEvent = (e: SimEvent) => {
  switch (e.type) {
    case 'ensemble':
      audio.ensemble(e.region, e.index);
      if (e.region === 'ctx') {
        chips[e.index].classList.add('on');
        setTimeout(() => chips[e.index].classList.remove('on'), 160);
      }
      break;
    case 'ripple':
      audio.ripple();
      view.flashRipple();
      break;
    case 'phase':
      $('phase-label').textContent = LABELS[e.phase];
      $('caption').textContent = CAPTIONS[e.phase];
      audio.setPhase(e.phase);
      setBusy(e.phase !== 'idle');
      if (e.phase === 'idle' && queue.length) begin(queue.shift()!);
      break;
    case 'recall':
      $('recall-text').textContent =
        `Recall: ${e.score} of ${e.total} elements completed after the cue` +
        (sim.silenceHipInRecall ? ' (hippocampus off).' : '.');
      break;
  }
};

function setBusy(busy: boolean): void {
  for (const id of ['btn-encode', 'btn-sleep', 'btn-recall', 'btn-demo'])
    $<HTMLButtonElement>(id).disabled = busy;
}

$('btn-encode').onclick = () => begin('encode');
$('btn-sleep').onclick = () => begin('consolidate');
$('btn-recall').onclick = () => begin('recall');
$('btn-demo').onclick = () => {
  sim.reset();
  queue = ['consolidate', 'recall'];
  begin('encode');
};
$('btn-reset').onclick = () => {
  queue = [];
  sim.reset();
  $('recall-text').textContent = 'No recall test yet.';
};
$<HTMLInputElement>('audio').onchange = (ev) => audio.setEnabled((ev.target as HTMLInputElement).checked);
$<HTMLInputElement>('rotate').onchange = (ev) => (view.autoRotate = (ev.target as HTMLInputElement).checked);

$('caption').textContent = CAPTIONS.idle;

// Main loop: advance the sim in 1 ms steps scaled by the speed slider
let last = performance.now();
let carry = 0;
function frame(now: number): void {
  const dt = Math.min(64, now - last);
  last = now;
  carry += dt * Number($<HTMLInputElement>('speed').value);
  const steps = Math.min(400, Math.floor(carry));
  carry -= steps;
  for (let i = 0; i < steps; i++) sim.step();

  view.render(dt);
  const hip = sim.strength('hc');
  const ctx = sim.strength('cc');
  $('bar-hip').style.width = `${hip * 100}%`;
  $('bar-ctx').style.width = `${ctx * 100}%`;
  $('val-hip').textContent = `${Math.round(hip * 100)}%`;
  $('val-ctx').textContent = `${Math.round(ctx * 100)}%`;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
