import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { BRAIN_RADII, ENSEMBLE_COLORS, hippocampusPoint } from './layout.ts';
import { BrainSim, PARAMS } from './sim.ts';
import type { EdgeKind } from './sim.ts';

const VERT = /* glsl */ `
attribute float aGlow; attribute float aBase; attribute vec3 aColor;
uniform float uScale;
varying float vGlow; varying vec3 vColor;
void main() {
  vGlow = aGlow; vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = (aBase + aGlow * 0.11) * uScale / -mv.z;
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = /* glsl */ `
varying float vGlow; varying vec3 vColor;
void main() {
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  float a = smoothstep(0.5, 0.0, d); a *= a;
  vec3 col = mix(vColor, vec3(1.0), clamp(vGlow - 0.3, 0.0, 1.0) * 0.7);
  gl_FragColor = vec4(col, a * (0.3 + 0.7 * vGlow));
}`;

const EDGE_COLORS: Record<EdgeKind, THREE.Color> = {
  hh: new THREE.Color('#ff9a3c'),
  hc: new THREE.Color('#ff6a3d'),
  cc: new THREE.Color('#3dffc4'),
};

export class BrainScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
  private readonly scene = new THREE.Scene();
  private readonly controls: OrbitControls;
  private readonly glow: Float32Array;
  private readonly glowAttr: THREE.BufferAttribute;
  private readonly edgeColors: THREE.BufferAttribute;
  private readonly uniforms = { uScale: { value: 800 } };
  private readonly hippoMats: THREE.MeshBasicMaterial[] = [];
  private rippleLevel = 0;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly sim: BrainSim) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.camera.position.set(3.2, 1.5, 3.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 9;

    this.scene.add(this.makeBrainShell(), this.makeHippocampus());

    // Neurons
    const n = sim.n;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const base = new Float32Array(n);
    this.glow = new Float32Array(n);
    const tmp = new THREE.Color();
    sim.neurons.forEach((nr, i) => {
      pos.set(nr.pos, i * 3);
      if (nr.ensemble >= 0) tmp.set(ENSEMBLE_COLORS[nr.ensemble]);
      else tmp.set(nr.region === 'hip' ? '#a8662a' : '#2f5f8f');
      col.set([tmp.r, tmp.g, tmp.b], i * 3);
      base[i] = nr.ensemble >= 0 ? 0.05 : 0.026;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('aBase', new THREE.BufferAttribute(base, 1));
    this.glowAttr = new THREE.BufferAttribute(this.glow, 1);
    this.glowAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aGlow', this.glowAttr);
    const pts = new THREE.Points(
      geo,
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: this.uniforms,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    pts.frustumCulled = false;
    this.scene.add(pts);

    // Synapses of the memory trace: brightness = weight, pulses when the presynaptic neuron fires
    const lp = new Float32Array(sim.edges.length * 6);
    sim.edges.forEach((e, i) => {
      lp.set(sim.neurons[e.pre].pos, i * 6);
      lp.set(sim.neurons[e.post].pos, i * 6 + 3);
    });
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
    this.edgeColors = new THREE.BufferAttribute(new Float32Array(sim.edges.length * 6), 3);
    this.edgeColors.setUsage(THREE.DynamicDrawUsage);
    lg.setAttribute('color', this.edgeColors);
    const lines = new THREE.LineSegments(
      lg,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    lines.frustumCulled = false;
    this.scene.add(lines);

    sim.onSpike = (id) => (this.glow[id] = 1);
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  set autoRotate(on: boolean) {
    this.controls.autoRotate = on;
  }

  flashRipple(): void {
    this.rippleLevel = 1;
  }

  render(dtMs: number): void {
    const decay = Math.exp(-dtMs / 260);
    for (let i = 0; i < this.glow.length; i++) this.glow[i] *= decay;
    this.glowAttr.needsUpdate = true;

    const { edges, edgeW } = this.sim;
    const arr = this.edgeColors.array as Float32Array;
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      const w = edgeW[i] / PARAMS.WMAX;
      const s = w * (0.22 + 0.9 * this.glow[e.pre]);
      const c = EDGE_COLORS[e.kind];
      arr[i * 6] = arr[i * 6 + 3] = c.r * s;
      arr[i * 6 + 1] = arr[i * 6 + 4] = c.g * s;
      arr[i * 6 + 2] = arr[i * 6 + 5] = c.b * s;
    }
    this.edgeColors.needsUpdate = true;

    this.rippleLevel *= Math.exp(-dtMs / 200);
    for (const m of this.hippoMats) m.opacity = 0.1 + 0.5 * this.rippleLevel;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.max(1, Math.floor(this.canvas.clientWidth));
  const h = Math.max(1, Math.floor(this.canvas.clientHeight));
  this.renderer.setPixelRatio(dpr);
  this.renderer.setSize(w, h, false);
  this.camera.aspect = w / h;

  const shift = w > 900 ? 160 : 0; // pixels to move the brain right (0 on narrow screens)
  this.camera.setViewOffset(w, h, -shift, 0, w, h);

  this.uniforms.uScale.value = h * dpr * 1.2;
}

  private makeBrainShell(): THREE.Group {
    const geo = new THREE.IcosahedronGeometry(1, 4);
    const p = geo.attributes.position as THREE.BufferAttribute;
    const d = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      d.fromBufferAttribute(p, i).normalize();
      const bump =
        0.035 *
        (Math.sin(9 * d.x + 2 * d.y) * Math.sin(8 * d.z + 3 * d.y) +
          0.6 * Math.sin(13 * d.y + 5 * d.z) * Math.sin(7 * d.x));
      const fissure = 1 - 0.1 * Math.exp(-Math.pow(d.x / 0.07, 2)) * Math.max(0, d.y);
      const r = (1 + bump) * fissure;
      p.setXYZ(i, d.x * r * BRAIN_RADII[0], d.y * r * BRAIN_RADII[1], d.z * r * BRAIN_RADII[2]);
    }
    const g = new THREE.Group();
    g.add(
      new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({ color: 0x35557f, wireframe: true, transparent: true, opacity: 0.28, depthWrite: false }),
      ),
      new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({ color: 0x07101f, transparent: true, opacity: 0.5, side: THREE.BackSide, depthWrite: false }),
      ),
    );
    return g;
  }

  private makeHippocampus(): THREE.Group {
    const g = new THREE.Group();
    for (const side of [0, 1] as const) {
      const pts = Array.from({ length: 11 }, (_, i) => new THREE.Vector3(...hippocampusPoint(side, i / 10)));
      const mat = new THREE.MeshBasicMaterial({
        color: 0xff9a3c,
        transparent: true,
        opacity: 0.1,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      this.hippoMats.push(mat);
      g.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, 0.035, 8), mat));
    }
    return g;
  }
}
