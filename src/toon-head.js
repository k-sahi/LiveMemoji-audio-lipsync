// toon-head.js — a 3D cartoon head built from a photo's traits (toon-analyze.js).
//
// Everything is real 3D geometry, so it animates cleanly:
//   head      a sculpted ellipsoid (face width, jaw taper, cheeks); the chin drops with the jaw
//   eyes      eyeballs with irises that glance around, and eyelids that close for blinks
//   brows     curved tubes that lift with emphasis
//   mouth     lips, dark inside, teeth and tongue, re-shaped every frame from the
//             person's own lip outline + the captured real-mouth motion (visemes.js)
//   hair      none / short / curly / long, in the photo's hair colour and volume
//   extras    nose, ears, glasses, mustache, neck and shirt
// Units: the head is about 2 wide; the caller scales the whole group.

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';

const SHAPES = ['ah', 'ee', 'oo', 'mm', 'ff', 'smile'];
const EYE_Y = 0.12;
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
const srgb = ([r, g, b], k = 1) => new THREE.Color().setRGB(Math.min(1, r * k / 255), Math.min(1, g * k / 255), Math.min(1, b * k / 255), THREE.SRGBColorSpace);

// Make photo colours a bit more "cartoon": a little brighter and cleaner.
function cartoonColor(rgb, { lift = 1.08, sat = 1.12 } = {}) {
  const c = srgb(rgb);
  const hsl = {}; c.getHSL(hsl);
  c.setHSL(hsl.h, Math.min(1, hsl.s * sat), Math.min(0.92, hsl.l * lift));
  return c;
}

// Style presets: what makes a cartoon read as feminine, masculine or neutral.
// (Chosen by the person; the photo only offers a first suggestion.)
export const LOOKS = {
  feminine: {
    eye: 1.1, lash: 1.9, flicks: 4, lid: -0.06, brow: 0.6, browThin: true, arch: 1.8, browLift: 0.03,
    upperLip: 1.2, lowerLip: 1.3, bow: 1, mouthW: 0.9, lipTint: [208, 66, 104], lipMix: 0.5, gloss: true,
    nose: 0.75, taper: 0.14, jawPow: 1.1, jawAngle: 0, chin: -0.02, cheekbone: 0.08, browRidge: 0,
    blush: 0.38, neck: 0.8, shoulders: 1.55, adamsApple: false, facialHair: 'none',
  },
  masculine: {
    eye: 0.88, lash: 0.6, flicks: 0, lid: 0.12, brow: 1.6, browThin: false, arch: 0.25, browLift: -0.035,
    upperLip: 0.6, lowerLip: 0.9, bow: 0, mouthW: 1.04, lipTint: null, lipMix: 0.6, gloss: false,
    nose: 1.3, taper: -0.16, jawPow: 3.4, jawAngle: 0.08, chin: 0.08, cheekbone: 0.02, browRidge: 0.04,
    blush: 0, neck: 1.28, shoulders: 2.1, adamsApple: true, facialHair: 'stubble',
  },
  neutral: {
    eye: 1, lash: 1.1, flicks: 0, lid: 0, brow: 1, browThin: true, arch: 1, browLift: 0,
    upperLip: 1, lowerLip: 1.1, bow: 0.4, mouthW: 1, lipTint: null, lipMix: 0.15, gloss: false,
    nose: 1, taper: 0, jawPow: 1.6, jawAngle: 0.02, chin: 0.02, cheekbone: 0.05, browRidge: 0.01,
    blush: 0.12, neck: 1, shoulders: 1.8, adamsApple: false, facialHair: 'none',
  },
};

/**
 * A first suggestion from the photo. Deliberately cautious: only a clear, full beard
 * suggests masculine (a mustache-like shadow alone is too often wrong); long hair suggests
 * feminine; everything else starts neutral. Always editable.
 */
export function suggestLook(traits) {
  const a = traits.accessories;
  if (a.beard && a.mustache) return 'masculine';
  if (traits.hair.style === 'long' || traits.hair.style === 'bob') return 'feminine';
  return 'neutral';
}

/** Facial hair to start with: what the photo clearly shows, otherwise the style's default. */
export function suggestFacialHair(traits, look) {
  const a = traits.accessories;
  // Shadows and hands can look like facial hair, so a single cue only counts for the masculine style.
  if (a.beard && a.mustache) return look === 'feminine' ? 'none' : 'beard';
  if (look === 'masculine' && a.beard) return 'goatee';
  if (look === 'masculine' && a.mustache) return 'mustache';
  return LOOKS[look]?.facialHair ?? 'none';
}

// Caricature: push each measurement away from the average, like a caricature artist.
const EXAGGERATE = 2.2;
const caricature = (value, avg, min, max) => Math.min(max, Math.max(min, 1 + (value / avg - 1) * EXAGGERATE));

/** A tube whose thickness follows fn(t), t = 0..1 along the curve (brows, mustaches, lashes, locks). */
function taperTube(curve, segs, radius, radial, fn, flat = 1) {
  const g = new THREE.TubeGeometry(curve, segs, radius, radial, false);
  const p = g.attributes.position;
  for (let i = 0; i <= segs; i++) {
    const c = curve.getPointAt(i / segs), k = Math.max(0.05, fn(i / segs));
    for (let j = 0; j <= radial; j++) {
      const n = i * (radial + 1) + j;
      p.setXYZ(n, c.x + (p.getX(n) - c.x) * k, c.y + (p.getY(n) - c.y) * k, c.z + (p.getZ(n) - c.z) * k * flat);
    }
  }
  g.computeVertexNormals();
  return g;
}
const bell = (lo, hi) => (t) => lo + (hi - lo) * Math.sin(Math.PI * t);

export class ToonHead {
  /**
   * @param traits result of analyzeFace()
   * @param visemeFile captured mouth motion (lipsync-visemes-v1) or null for built-in shapes
   */
  constructor(traits, visemeFile = null, { look = 'neutral' } = {}) {
    this.t = traits;
    this.look = LOOKS[look] ?? LOOKS.neutral;
    this.lookName = look;
    const L = this.look;
    this.group = new THREE.Group();
    this.head = new THREE.Group();          // everything that turns with the head
    this.head.position.y = -0.95;            // pivot at the top of the neck
    this.group.add(this.head);
    this.inner = new THREE.Group();          // undo the pivot offset for children
    this.inner.position.y = 0.95;
    this.head.add(this.inner);

    const f = traits.face;
    // Shape of the head, from the photo's proportions (typical values in comments).
    // Typical values (in eye-distance units) are the "average face"; differences get exaggerated.
    this.wx = caricature(f.cheekWidth / Math.max(1, f.height), 0.77, 0.82, 1.24);             // face width / height
    this.eyeSpread = caricature(2.25 / Math.max(1, f.cheekWidth), 1, 0.84, 1.18);             // eyes far apart vs close
    this.eyeScale = caricature(f.eyeWidth, 0.5, 0.78, 1.32) * L.eye;                           // eye size
    this.noseScale = caricature(f.noseWidth, 0.5, 0.7, 1.55) * L.nose;                         // nose width
    this.mouthScale = Math.min(1.12, Math.max(0.85, 1 + (f.mouthWidth / 0.85 - 1) * 1.2)) * L.mouthW; // mouth width (mildly exaggerated)
    this.jawPow = L.jawPow;
    this.hy = 1.16;
    this.dz = 1.0;
    this.taper = Math.min(0.55, Math.max(0.06, 0.28 + (0.75 - f.jawRatio) * 1.4 * EXAGGERATE + L.taper)); // narrow vs wide jaw
    this.k = 0.72 * this.wx;                                                                          // eye-distance → head units

    this.skin = cartoonColor(traits.colors.skin, { lift: 1.06, sat: 1.05 });
    this.hairColor = cartoonColor(traits.colors.hair, { lift: 1.15, sat: 1.0 });
    this.mat = {
      skin: new THREE.MeshStandardMaterial({ color: this.skin, roughness: 0.55, emissive: this.skin.clone().multiplyScalar(0.08) }),
      skinDark: new THREE.MeshStandardMaterial({ color: this.skin.clone().multiplyScalar(0.93), roughness: 0.55 }),
      hair: new THREE.MeshStandardMaterial({ color: this.hairColor, roughness: 0.75 }),
      brow: new THREE.MeshStandardMaterial({ color: cartoonColor(traits.colors.brows, { lift: 1.05 }), roughness: 0.8 }),
      lips: new THREE.MeshStandardMaterial({ color: lipColor(traits, L), roughness: L.gloss ? 0.22 : 0.6, side: THREE.DoubleSide }),
      mouth: new THREE.MeshBasicMaterial({ color: 0x3a0d16, side: THREE.DoubleSide }),
      teeth: new THREE.MeshStandardMaterial({ color: 0xf6f3ec, roughness: 0.35, side: THREE.DoubleSide }),
      tongue: new THREE.MeshStandardMaterial({ color: 0xc0505f, roughness: 0.5, side: THREE.DoubleSide }),
      white: new THREE.MeshStandardMaterial({ color: 0xfbfbfb, roughness: 0.25 }),
      iris: new THREE.MeshStandardMaterial({ color: cartoonColor(traits.colors.iris, { lift: 1.2, sat: 1.25 }), roughness: 0.3 }),
      pupil: new THREE.MeshStandardMaterial({ color: 0x0b0b0e, roughness: 0.2 }),
      shine: new THREE.MeshBasicMaterial({ color: 0xffffff }),
      lash: new THREE.MeshStandardMaterial({ color: srgb(traits.colors.brows, 0.6), roughness: 0.8 }),
      shirt: new THREE.MeshStandardMaterial({ color: cartoonColor(traits.colors.shirt, { lift: 1.05 }), roughness: 0.8 }),
      frame: new THREE.MeshStandardMaterial({ color: 0x202024, roughness: 0.3, metalness: 0.2 }),
    };

    this.addLights();
    this.buildHead();
    this.buildBody();
    this.buildEyes();
    this.buildBrowsNoseEars();
    this.setupMouth(visemeFile);
    this.setHairStyle(traits.hair.style);
    this.setGlasses(traits.accessories.glasses);
    this.setFacialHair(suggestFacialHair(traits, look));
  }

  // ---- geometry helpers ------------------------------------------------------------------
  // jawPow: higher = squarer jaw (width holds, then drops near the chin); lower = softer V shape.
  halfWidthAt(y) { return this.wx * (1 - this.taper * Math.pow(Math.min(1, Math.max(0, -y / this.hy)), this.jawPow ?? 1.5)); }
  /** z of the face surface at (x, y), so features sit on the head. */
  surfaceZ(x, y) {
    const a = x / this.halfWidthAt(y), b = y / this.hy;
    return this.dz * Math.sqrt(Math.max(0, 1 - a * a - b * b));
  }

  addLights() {
    this.group.add(new THREE.HemisphereLight(0xffffff, 0x5a4650, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-2, 3, 5); this.group.add(key);
    const fill = new THREE.DirectionalLight(0xfff1e6, 0.6); fill.position.set(3, 0, 3); this.group.add(fill);
    const rim = new THREE.DirectionalLight(0xdde8ff, 1.0); rim.position.set(2, 2, -4); this.group.add(rim);
  }

  buildHead() {
    const g = new THREE.SphereGeometry(1, 128, 96); // fine enough for smooth beard edges
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const yy = y * this.hy;
      x *= this.halfWidthAt(yy);
      // soft cheeks and a slightly flatter back of the head
      const L = this.look;
      const cy = L.cheekbone > 0.06 ? -0.12 : -0.25;   // feminine: higher, rounder cheekbones
      const cheek = L.cheekbone * (Math.exp(-((x - 0.55) ** 2 + (yy - cy) ** 2) / 0.06) + Math.exp(-((x + 0.55) ** 2 + (yy - cy) ** 2) / 0.06));
      // jaw corners pushed out (square jaw), chin pushed forward, brow ridge
      x *= 1 + L.jawAngle * Math.exp(-((yy + 0.72) ** 2) / 0.025) * (z > -0.2 ? 1 : 0.5);
      const chin = L.chin * Math.exp(-(x * x) / 0.09 - ((yy + 1.0) ** 2) / 0.03);
      const ridge = L.browRidge * Math.exp(-((yy - EYE_Y - 0.3) ** 2) / 0.006) * Math.exp(-(x * x) / 0.5);
      z = z * this.dz * (z < 0 ? 0.92 : 1) + (z > 0 ? cheek + chin + ridge : 0);
      p.setXYZ(i, x, yy, z);
    }
    g.computeVertexNormals();
    this.headGeo = g;
    this.headBase = Float32Array.from(p.array);
    this.headMesh = new THREE.Mesh(g, this.mat.skin);
    this.inner.add(this.headMesh);
  }

  buildBody() {
    const nk = this.look.neck;
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.4 * nk, 0.48 * nk, 0.9, 32), this.mat.skinDark);
    neck.position.set(0, -1.35, -0.08);
    const torso = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), this.mat.shirt);
    torso.scale.set(this.look.shoulders, 0.85, 1.05);
    torso.position.set(0, -2.25, -0.15);
    this.torso = torso;
    this.group.add(neck, torso);
    if (this.look.adamsApple) {
      const bump = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), this.mat.skinDark);
      bump.scale.set(1, 1.3, 0.8);
      bump.position.set(0, -1.32, -0.08 + 0.4 * nk);
      this.group.add(bump);
    }
  }

  buildEyes() {
    const f = this.t.face;
    const r = 0.19 * this.wx * this.eyeScale, L = this.look;
    this.eyeR = r;
    this.eyes = [];
    for (const side of [-1, 1]) {
      const x = side * 0.5 * this.k * this.eyeSpread, y = EYE_Y;
      const root = new THREE.Group();
      root.position.set(x, y, this.surfaceZ(x, y) - r * 0.42);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24), this.mat.white);
      const look = new THREE.Group(); // rotates for glances
      const cap = (radius, theta, mat) => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 12, 0, Math.PI * 2, 0, theta), mat);
        m.rotation.x = Math.PI / 2; // the cap's pole points forward (+z)
        return m;
      };
      look.add(cap(r * 1.004, 0.62, this.mat.iris), cap(r * 1.008, 0.27, this.mat.pupil));
      const shine = new THREE.Mesh(new THREE.SphereGeometry(r * 0.12, 12, 8), this.mat.shine);
      shine.position.set(-r * 0.22, r * 0.26, r * 0.93);
      const shine2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.06, 10, 6), this.mat.shine);
      shine2.position.set(r * 0.2, -r * 0.18, r * 0.97);
      look.add(shine, shine2);
      // Upper lid: a skin-coloured half shell that rotates down over the eye; lashes on its edge.
      const lid = new THREE.Group();
      lid.add(new THREE.Mesh(new THREE.SphereGeometry(r * 1.06, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), this.mat.skin));
      const lash = new THREE.Mesh(new THREE.TorusGeometry(r * 1.07, r * 0.07 * L.lash, 8, 40, Math.PI), this.mat.lash);
      lash.rotation.x = Math.PI / 2;
      lid.add(lash);
      if (L.flicks) {
        // Lash flicks at the outer corner (they ride on the lid, so they blink too).
        for (const [a, len] of [[0.12, 0.42], [0.3, 0.4], [0.48, 0.34], [0.66, 0.26]].slice(0, L.flicks)) {
          const ang = side > 0 ? a : Math.PI - a;          // outer side of each eye
          const base = new THREE.Vector3(Math.cos(ang) * r * 1.07, 0, Math.sin(ang) * r * 1.07);
          const dir = new THREE.Vector3(Math.cos(ang) * 0.8, 0.35, Math.sin(ang) * 0.5).normalize().multiplyScalar(r * len);
          const tip = base.clone().add(dir);
          const mid = base.clone().lerp(tip, 0.5).add(new THREE.Vector3(0, r * 0.06, 0));
          lid.add(new THREE.Mesh(taperTube(new THREE.QuadraticBezierCurve3(base, mid, tip), 8, r * 0.05, 6, (t) => 1 - 0.85 * t), this.mat.lash));
        }
      }
      // Lower lid: a thin shell covering the bottom of the eye.
      const lower = new THREE.Mesh(new THREE.SphereGeometry(r * 1.05, 32, 12, 0, Math.PI * 2, Math.PI * 0.78, Math.PI * 0.22), this.mat.skin);
      lower.rotation.x = -0.15;
      root.add(ball, look, lid, lower);
      this.inner.add(root);
      this.eyes.push({ root, look, lid });
    }
    const ratio = f.eyeOpen / Math.max(0.2, f.eyeWidth); // ≈ 0.3 typical
    this.lidRest = -(0.6 + Math.max(-0.12, Math.min(0.15, (ratio - 0.3) * 1.2))) + L.lid;
  }

  buildBrowsNoseEars() {
    const f = this.t.face, r = this.eyeR;
    this.brows = [];
    const browY = EYE_Y + r * 1.2 + Math.max(0.02, Math.min(0.12, (-(f.browY) - 0.3) * this.k * 0.6)) + this.look.browLift;
    for (const side of [-1, 1]) {
      const ex = side * 0.5 * this.k * this.eyeSpread, L = this.look;
      // inner end → peak → tail. Feminine: high arch, thin tapering tail. Masculine: low, straight, heavy.
      const pts = [[ex - side * r * 0.95, browY - 0.01], [ex + side * r * 0.25, browY + 0.07 * L.arch], [ex + side * r * 1.1, browY - 0.04 * L.arch]]
        .map(([x, y]) => new THREE.Vector3(x, y, this.surfaceZ(x, y) + 0.03));
      const curve = new THREE.QuadraticBezierCurve3(...pts);
      const thick = L.browThin ? (t) => 1.15 - 0.85 * t * t : (t) => 1 - 0.15 * t;
      const brow = new THREE.Mesh(taperTube(curve, 24, 0.034 * L.brow, 8, thick, 0.6), this.mat.brow);
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.034 * L.brow * thick(0), 10, 8), this.mat.brow);
      cap.position.copy(pts[0]); cap.scale.z = 0.6;
      const g = new THREE.Group(); g.add(brow, cap);
      this.inner.add(g);
      this.brows.push({ g, side });
    }
    // Nose
    const nw = 0.105 * this.wx * this.noseScale;
    const ny = EYE_Y - this.k * 0.8 * Math.max(0.55, Math.min(0.95, f.noseY));
    const nose = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), this.mat.skinDark);
    nose.scale.set(nw, nw * 0.85, nw * 0.9);
    nose.position.set(0, ny, this.surfaceZ(0, ny) + nw * 0.25);
    this.inner.add(nose);
    this.noseY = ny;
    // Ears
    for (const side of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), this.mat.skin);
      ear.scale.set(0.12, 0.24, 0.17);
      ear.position.set(side * this.halfWidthAt(EYE_Y - 0.12) * 0.97, EYE_Y - 0.12, -0.05);
      this.inner.add(ear);
    }
    // Blush: soft pink on the cheeks.
    if (this.look.blush > 0) {
      const mat = new THREE.MeshBasicMaterial({ color: 0xff6f86, transparent: true, opacity: this.look.blush, depthWrite: false });
      for (const side of [-1, 1]) {
        const x = side * 0.52 * this.wx, y = EYE_Y - 0.36;
        const d = new THREE.Mesh(new THREE.CircleGeometry(0.15, 32), mat);
        d.scale.set(1.25, 0.8, 1);
        d.position.set(x, y, this.surfaceZ(x, y) + 0.012);
        d.lookAt(x * 2.2, y * 1.2, this.surfaceZ(x, y) * 2.2 + 0.5);
        d.renderOrder = 5;
        this.inner.add(d);
      }
    }
  }

  // ---- mouth -------------------------------------------------------------------------------
  setupMouth(visemeFile) {
    const m = this.t.mouth, f = this.t.face;
    this.mouthY = EYE_Y - this.k * 0.8 * Math.max(0.9, Math.min(1.5, f.mouthY));
    this.mScale = this.k * 0.88 * this.mouthScale;
    // Neutral lips from the photo, with any gap closed (paired inner points meet in the middle).
    const inner = m.inner.map((p) => p.slice()), outer = m.outer.map((p) => p.slice());
    for (let i = 1; i < 10; i++) {
      const j = 20 - i, mid = (inner[i][1] + inner[j][1]) / 2;
      const gap = inner[j][1] - inner[i][1];
      if (gap > 0) { inner[i][1] = mid - 0.004; inner[j][1] = mid + 0.004; }
    }
    // Lip fullness: push the outer lip line away from the mouth's centre line.
    const L = this.look;
    for (const p of outer) p[1] *= p[1] < 0 ? L.upperLip : L.lowerLip;
    if (L.bow) {
      // Cupid's bow: the two peaks of the upper lip rise, the middle dips.
      const h = Math.abs(outer[5][1] - inner[5][1]);
      outer[4][1] -= 0.28 * h * L.bow; outer[6][1] -= 0.28 * h * L.bow; outer[5][1] += 0.12 * h * L.bow;
    }
    this.neutral = { inner, outer };
    this.deltas = {};
    const idx = [...m.innerIdx, ...m.outerIdx];
    if (visemeFile?.deltas) {
      for (const k of SHAPES) {
        const d = visemeFile.deltas[k];
        if (d) this.deltas[k] = idx.map((i) => [d[i * 3], d[i * 3 + 1]]);
      }
    } else {
      this.deltas = builtinMouthShapes(inner, outer);
    }
    this.mouthParts = {};
    for (const [name, mat, order] of [['inside', this.mat.mouth, 1], ['tongue', this.mat.tongue, 2], ['lowerTeeth', this.mat.teeth, 3],
      ['upperTeeth', this.mat.teeth, 3], ['lips', this.mat.lips, 4]]) {
      // Drawn on top of the skin (no depth test): when the jaw drops, the moved chin skin
      // would otherwise hide the opening. The mouth is on the front of the face, so nothing
      // else ever needs to cover it.
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat.clone());
      mesh.material.depthTest = false;
      mesh.renderOrder = 10 + order;
      mesh.frustumCulled = false;
      this.inner.add(mesh);
      this.mouthParts[name] = mesh;
    }
  }

  /** Current lip outlines (head coords) for the given shape weights. */
  lipPoints(weights) {
    const E = 0.95; // how far the mouth moves (1 = like the real captured mouth)
    const n = 20;
    const cur = (arr, off) => arr.map((p, i) => {
      let x = p[0], y = p[1];
      for (const [k, w] of Object.entries(weights)) {
        const d = this.deltas[k];
        if (!d || !w) continue;
        x += E * w * d[off + i][0];
        y += E * w * d[off + i][1];
      }
      return new THREE.Vector2(x * this.mScale, this.mouthY - y * this.mScale);
    });
    // Keep each outline a simple loop: no lip point may pass beyond the mouth corners
    // (strong "oo" rounding pulls the corners in faster than their neighbours).
    const tidy = (pts) => {
      const left = Math.min(pts[0].x, pts[10].x), right = Math.max(pts[0].x, pts[10].x), pad = (right - left) * 0.04;
      for (let i = 1; i < 20; i++) if (i !== 10) pts[i].x = Math.min(right - pad, Math.max(left + pad, pts[i].x));
      return pts;
    };
    return { inner: tidy(cur(this.neutral.inner, 0)), outer: tidy(cur(this.neutral.outer, n)) };
  }

  updateMouth(weights) {
    const { inner, outer } = this.lipPoints(weights);
    const smoothLoop = (pts, n) => new THREE.SplineCurve([...pts, pts[0]]).getSpacedPoints(n).slice(0, n);
    const outerS = smoothLoop(outer, 64), innerS = smoothLoop(inner, 64);
    // Opening size: average gap between paired upper/lower inner points.
    let gap = 0;
    for (let i = 3; i <= 7; i++) gap += (inner[i].y - inner[20 - i].y) / 5;
    const open = gap > 0.012;

    const zOn = (v, off) => this.surfaceZ(v.x, v.y) + off;
    const setShape = (mesh, contour, holes = [], off) => {
      const tris = THREE.ShapeUtils.triangulateShape(contour, holes);
      const all = [...contour, ...holes.flat()];
      const pos = new Float32Array(all.length * 3);
      all.forEach((v, i) => pos.set([v.x, v.y, zOn(v, off)], i * 3));
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setIndex(tris.flat());
      geo.computeVertexNormals();
      mesh.geometry.dispose();
      mesh.geometry = geo;
    };

    if (open) {
      setShape(this.mouthParts.lips, outerS, [innerS], 0.022);
      setShape(this.mouthParts.inside, innerS, [], 0.008);
      // Teeth and tongue: bands that hug the upper / lower inner lip, never wider than the opening.
      const upper = new THREE.SplineCurve(inner.slice(0, 11)).getSpacedPoints(24);   // corner → top → corner
      const lowerPts = [inner[10], ...inner.slice(11), inner[0]];                     // corner → bottom → corner
      const lower = new THREE.SplineCurve(lowerPts).getSpacedPoints(24);
      const yAt = (curve, x) => {                                                     // y of a curve at x
        let best = curve[0];
        for (const v of curve) if (Math.abs(v.x - x) < Math.abs(best.x - x)) best = v;
        return best.y;
      };
      const teethH = 0.075 * this.wx;
      const band = (edge, other, h, down) => edge.map((v) => {
        const o = yAt(other, v.x);
        return new THREE.Vector2(v.x, down ? Math.max(v.y - h, o + 0.004) : Math.min(v.y + h, o - 0.004));
      }).reverse();
      setShape(this.mouthParts.upperTeeth, [...upper, ...band(upper, lower, teethH, true)], [], 0.014);
      setShape(this.mouthParts.lowerTeeth, [...lower, ...band(lower, upper, teethH * 0.55, false)], [], 0.012);
      const tongueTop = lower.map((v) => new THREE.Vector2(v.x * 0.8, Math.min(v.y + gap * 0.45, yAt(upper, v.x) - 0.01)));
      setShape(this.mouthParts.tongue, [...lower.map((v) => new THREE.Vector2(v.x * 0.8, v.y)), ...tongueTop.reverse()], [], 0.011);
      for (const k of ['inside', 'upperTeeth', 'lowerTeeth', 'tongue']) this.mouthParts[k].visible = true;
    } else {
      // Closed: solid lips (the seam between them is drawn by the shading of the two lips meeting).
      setShape(this.mouthParts.lips, outerS, [], 0.022);
      for (const k of ['inside', 'upperTeeth', 'lowerTeeth', 'tongue']) this.mouthParts[k].visible = false;
    }
    // The chin follows the lower lip.
    const neutralLow = this.mouthY - this.neutral.inner[15][1] * this.mScale;
    this.updateJaw(0.6 * Math.max(0, neutralLow - inner[15].y)); // the chin moves less than the lower lip
    if (this.mustache) this.mustache.position.y = Math.max(0, inner[5].y - (this.mouthY - this.neutral.inner[5][1] * this.mScale)) * 0.8;
  }

  updateJaw(drop) {
    const p = this.headGeo.attributes.position, base = this.headBase;
    const top = this.mouthY + 0.02, bottom = this.mouthY - 0.25;
    for (let i = 0; i < p.count; i++) {
      const x = base[i * 3], y = base[i * 3 + 1], z = base[i * 3 + 2];
      const w = smooth(top, bottom, y) * (1 - smooth(0.5 * this.wx, 0.95 * this.wx, Math.abs(x))) * smooth(-0.3, 0.3, z);
      p.setXYZ(i, x, y - drop * w, z - drop * 0.1 * w);
    }
    p.needsUpdate = true;
    if (drop > 0 || this._jawWasOpen) {
      this.headGeo.computeVertexNormals();
      this.fitFacialHair();
    }
    this._jawWasOpen = drop > 0;
  }

  // ---- hair --------------------------------------------------------------------------------
  setHairStyle(style) {
    if (this.hair) { this.inner.remove(this.hair); this.hair.traverse((o) => o.geometry?.dispose()); }
    this.hairStyle = style;
    this.hair = new THREE.Group();
    this.inner.add(this.hair);
    if (style === 'none') return;
    const h = this.t.hair, look = this.lookName;
    const fem = look === 'feminine', masc = look === 'masculine';
    const tight = style === 'buzz' ? 1.02 : style === 'bun' ? 1.035 : masc && style === 'short' ? 1.05 : 1.09;
    const vol = style === 'curly' ? 1.18 + 0.35 * Math.min(0.6, h.volumeTop) + (fem ? 0.06 : 0) : tight;
    const foreheadY = EYE_Y + this.eyeR * 1.2 + (style === 'curly' ? 0.3 : 0.22) + (masc ? 0.04 : 0);
    // Masculine hairlines recede a little at the temples (an "M"); feminine ones are rounder.
    const temple = masc ? 0.22 : fem ? 0.02 : 0.08;
    // A shell over the skull, cut along a hairline (high at the forehead, low at the back).
    const g = new THREE.SphereGeometry(1, 160, 110); // fine mesh so the hairline cut is smooth
    const p = g.attributes.position;
    const keep = new Uint8Array(p.count);
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i) * this.hy, z = p.getZ(i); // y: head units
      const front = smooth(0.15, 0.85, z);                     // 1 at the face, 0 at the back
      const line = foreheadY * front + (style === 'buzz' || style === 'bun' ? -0.45 : -0.62) * (1 - front) + temple * Math.abs(x) * front;
      keep[i] = y > line ? 1 : 0;
      p.setXYZ(i, x * this.halfWidthAt(y) * vol, y * vol * 0.98 + (style === 'curly' ? 0.05 : 0), z * this.dz * vol * (z < 0 ? 0.95 : 1));
    }
    const idx = g.index.array, kept = [];
    for (let t = 0; t < idx.length; t += 3) if (keep[idx[t]] && keep[idx[t + 1]] && keep[idx[t + 2]]) kept.push(idx[t], idx[t + 1], idx[t + 2]);
    g.setIndex(kept);
    g.computeVertexNormals();
    const shell = new THREE.Mesh(g, style === 'buzz' ? this.mat.hair.clone() : this.mat.hair);
    if (style === 'buzz') { shell.material.color.multiplyScalar(0.8); shell.material.roughness = 1; }
    this.hair.add(shell);
    const rand = mulberry32(7);
    const onHead = (x, y, lift = 0) => new THREE.Vector3(x, y, this.surfaceZ(x, Math.min(y, this.hy * 0.98)) * vol + lift);
    if (masc && style !== 'curly') {
      // short sideburns in front of the ears
      for (const side of [-1, 1]) {
        const sb = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), this.mat.hair);
        const x = side * this.halfWidthAt(EYE_Y - 0.05) * 0.9;
        sb.scale.set(0.07, 0.2, 0.09);
        sb.position.set(x, EYE_Y - 0.02, this.surfaceZ(x, EYE_Y) * 0.95);
        this.hair.add(sb);
      }
    }
    if (style === 'buzz') return;
    if (style === 'bun') {
      const bun = new THREE.Mesh(new THREE.SphereGeometry(0.36, 28, 20), this.mat.hair);
      bun.position.set(0, this.hy * 1.02, -0.3);
      this.hair.add(bun);
      return;
    }
    if (fem && style !== 'curly') {
      // Side-swept fringe: thick locks sweeping from a side parting across the forehead.
      for (const [dy, r, sc] of [[0, 0.15, 1], [-0.07, 0.11, 0.85], [0.08, 0.12, 0.9]]) {
        const pts = [[-0.25, foreheadY + 0.26 + dy], [0.15, foreheadY + 0.1 + dy], [0.55, foreheadY - 0.05 + dy], [0.86, foreheadY - 0.32 + dy]]
          .map(([x, y]) => onHead(x * this.wx * sc, y, 0.02));
        this.hair.add(new THREE.Mesh(taperTube(new THREE.CatmullRomCurve3(pts), 30, r, 12, bell(0.35, 1), 0.55), this.mat.hair));
      }
      // locks framing the face in front of the ears
      const len = style === 'long' ? 0.95 : style === 'bob' ? 0.55 : 0.35;
      for (const side of [-1, 1]) {
        const pts = [[0.86, foreheadY - 0.1], [0.92, EYE_Y - 0.1], [0.86, EYE_Y - 0.1 - len]]
          .map(([x, y]) => { const xx = side * x * this.halfWidthAt(Math.max(-this.hy * 0.9, y)); return new THREE.Vector3(xx, y, Math.max(0.15, this.surfaceZ(xx * 0.97, y) * 0.9)); });
        this.hair.add(new THREE.Mesh(taperTube(new THREE.CatmullRomCurve3(pts), 24, 0.13, 10, (t) => 0.9 - 0.5 * t, 0.7), this.mat.hair));
      }
    } else if (masc && style !== 'curly') {
      // Quiff: hair pushed up and back at the front.
      for (const x of [-0.32, -0.12, 0.08, 0.28]) {
        const xx = x * this.wx, y0 = foreheadY - 0.03 - x * x * 0.4;
        const pts = [onHead(xx, y0, 0), onHead(xx + 0.04, y0 + 0.16, 0.1), onHead(xx + 0.09, y0 + 0.3, 0.06)];
        this.hair.add(new THREE.Mesh(taperTube(new THREE.CatmullRomCurve3(pts), 16, 0.13, 10, (t) => 0.95 - 0.5 * t, 0.8), this.mat.hair));
      }
    } else for (let k = -4; k <= 4; k++) {
      // Fringe: soft bumps along the front hairline.
      const x = k * 0.13 * this.wx, y = foreheadY + 0.03 - (k * k) * 0.006;
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 14), this.mat.hair);
      b.scale.set(1.35, 0.7, 0.75);
      b.position.set(x, y, this.surfaceZ(x, y) * vol * 0.97);
      this.hair.add(b);
    }
    if (style === 'curly') {
      // Lots of small curls scattered over the hair shell, in slightly varied shades.
      const shades = [0.85, 1, 1.15].map((s) => new THREE.MeshStandardMaterial({ color: this.hairColor.clone().multiplyScalar(s), roughness: 0.85 }));
      const ball = new THREE.SphereGeometry(1, 12, 10);
      for (let n = 0; n < (fem ? 260 : 170); n++) {
        const u = rand() * Math.PI * 2, v = Math.acos(1 - rand() * (fem ? 1.6 : 1.25));   // upper part of a sphere
        let x = Math.sin(v) * Math.cos(u), y = Math.cos(v), z = Math.sin(v) * Math.sin(u);
        const yy = y * this.hy;
        const front = smooth(0.15, 0.85, z);
        const low = fem ? -1.0 : -0.55;   // feminine curls fall past the jaw
        if (yy < (foreheadY + 0.06) * front + low * (1 - front)) continue;
        const s = 0.12 + rand() * 0.1;
        const c = new THREE.Mesh(ball, shades[n % 3]);
        c.scale.setScalar(s);
        c.position.set(x * this.halfWidthAt(yy) * (vol + 0.05), yy * vol + 0.05, z * this.dz * (vol + 0.05));
        this.hair.add(c);
      }
    }
    if (style === 'bob') {
      // Chin-length: fuller sides that end around the jaw, rounded at the back.
      for (const side of [-1, 1]) {
        const lock = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), this.mat.hair);
        lock.scale.set(0.3, 0.62, 0.62);
        lock.position.set(side * this.wx * 0.9, -0.28, -0.12);
        this.hair.add(lock);
      }
      const back = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), this.mat.hair);
      back.scale.set(this.wx * 1.06, 0.95, 0.75);
      back.position.set(0, -0.15, -0.35);
      this.hair.add(back);
    }
    if (style === 'long') {
      for (const side of [-1, 1]) {
        const ll = masc ? 0.55 : fem ? 1.6 : 1.1;
        const lock = new THREE.Mesh(new THREE.CapsuleGeometry(fem ? 0.32 : 0.26, ll, 8, 16), this.mat.hair);
        lock.position.set(side * this.wx * 0.92, -0.2 - ll / 2, -0.12);
        lock.rotation.z = side * 0.08;
        this.hair.add(lock);
      }
      const back = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), this.mat.hair);
      back.scale.set(this.wx * (fem ? 1.12 : 1.05), masc ? 1.05 : fem ? 1.85 : 1.5, 0.6);
      back.position.set(0, masc ? -0.3 : fem ? -0.75 : -0.55, -0.48);
      this.hair.add(back);
    }
  }

  setGlasses(on) {
    if (this.glasses) { this.inner.remove(this.glasses); this.glasses = null; }
    if (!on) return;
    const r = this.eyeR, gl = new THREE.Group();
    const ex = 0.5 * this.k * this.eyeSpread;
    const fz = this.surfaceZ(ex, EYE_Y) + r * 0.25;
    for (const side of [-1, 1]) {
      // Frames fit around the eye but never overlap each other in the middle.
      const R = Math.min(r * 1.38, (ex - 0.05) / 1.12);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(R, 0.03, 10, 48), this.mat.frame);
      ring.scale.set(1.12, Math.max(0.8, Math.min(1.05, (r * 1.25) / R)), 1);
      ring.position.set(side * ex, EYE_Y, fz);
      gl.add(ring);
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.9), this.mat.frame);
      arm.position.set(side * (ex + r * 1.5), EYE_Y + 0.02, fz - 0.45);
      gl.add(arm);
    }
    const bridge = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, Math.max(0.05, 2 * (ex - Math.min(r * 1.38, (ex - 0.05) / 1.12) * 1.12) + 0.02), 8), this.mat.frame);
    bridge.rotation.z = Math.PI / 2;
    bridge.position.set(0, EYE_Y + 0.04, fz + 0.02);
    gl.add(bridge);
    this.glasses = gl;
    this.inner.add(gl);
  }

  /**
   * Facial hair: 'none' | 'stubble' | 'mustache' | 'goatee' | 'beard'.
   * Beard, goatee and stubble are a layer over the face mesh (so they follow the jaw when the
   * mouth opens), thickest in the middle of the area and fading out at its edges.
   */
  setFacialHair(style) {
    for (const k of ['mustache', 'beard']) if (this[k]) { this.inner.remove(this[k]); this[k].traverse((o) => o.geometry?.dispose()); this[k] = null; }
    this.facialHair = style;
    if (style === 'none') return;
    const o = this.neutral.outer, s = this.mScale;
    const hw = Math.max(...o.map((q) => Math.abs(q[0]))) * s;          // half mouth width
    const top = this.mouthY - Math.min(...o.map((q) => q[1])) * s;     // top of the upper lip
    const bot = this.mouthY - Math.max(...o.map((q) => q[1])) * s;     // bottom of the lower lip
    if (style !== 'mustache') {
      const base = this.headBase, n = base.length / 3;
      const T = { stubble: 0.006, goatee: 0.07, beard: 0.085 }[style];
      const under = this.noseY - 0.07;
      const mask = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const x = base[i * 3], y = base[i * 3 + 1], z = base[i * 3 + 2], ax = Math.abs(x);
        let m = smooth(-0.4, -0.1, z);
        // top edge: under the nose in the middle, rising to the ears (sideburns) at the sides
        const sideTop = style === 'goatee' ? under : under + (EYE_Y - 0.05 - under) * smooth(0.35 * this.wx, 0.8 * this.wx, ax);
        m *= smooth(sideTop + 0.03, sideTop - 0.04, y);
        if (style === 'goatee') m *= smooth(hw * 1.25, hw * 0.95, ax);
        // keep the lips clear
        const e = Math.hypot(x / (hw * 1.12), (y - (top + bot) / 2) / ((top - bot) / 2 + 0.035));
        m *= smooth(0.95, 1.2, e);
        mask[i] = m;
      }
      const idx = this.headGeo.index.array, kept = [];
      for (let t = 0; t < idx.length; t += 3) if (mask[idx[t]] > 0.01 || mask[idx[t + 1]] > 0.01 || mask[idx[t + 2]] > 0.01) kept.push(idx[t], idx[t + 1], idx[t + 2]);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(base.length), 3));
      g.setIndex(kept);
      // Edges fade out (vertex alpha) instead of ending in a hard, jagged line.
      const rgba = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) rgba.set([1, 1, 1, smooth(0.02, 0.45, mask[i])], i * 4);
      g.setAttribute('color', new THREE.BufferAttribute(rgba, 4));
      // fuzzy texture: thickness varies a little from place to place
      this.beardLift = Float32Array.from(mask, (m, i) => T * m * (style === 'stubble' ? 1 : 0.8 + 0.35 * Math.abs(Math.sin(base[i * 3] * 37 + base[i * 3 + 1] * 23))));
      const dark = this.hairColor.clone().multiplyScalar(style === 'stubble' ? 0.6 : 1);
      const mat = style === 'stubble'
        ? new THREE.MeshStandardMaterial({ color: dark, roughness: 1, vertexColors: true, transparent: true, opacity: 0.45, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 })
        : new THREE.MeshStandardMaterial({ color: dark, roughness: 0.9, vertexColors: true, transparent: true, polygonOffset: true, polygonOffsetFactor: -1 });
      this.beard = new THREE.Mesh(g, mat);
      this.beard.renderOrder = style === 'stubble' ? 6 : 0;
      this.inner.add(this.beard);
      this.fitFacialHair();
    }
    if (style === 'mustache' || style === 'goatee' || style === 'beard') {
      // A thick mustache that is fullest in the middle and tapers to points at the corners.
      const at = (i, lift, out = 1.08) => { const x = o[i][0] * s * out, y = this.mouthY - o[i][1] * s + lift; return new THREE.Vector3(x, y, this.surfaceZ(x, y) + 0.04); };
      const curve = new THREE.CatmullRomCurve3([at(0, -0.03, 1.2), at(2, 0.05), at(5, 0.075), at(8, 0.05), at(10, -0.03, 1.2)]);
      const m = new THREE.Mesh(taperTube(curve, 40, 0.06, 10, bell(0.25, 1.05), 0.7), this.mat.hair.clone());
      m.material.depthTest = false; m.renderOrder = 16;
      this.mustache = new THREE.Group();
      this.mustache.add(m);
      this.inner.add(this.mustache);
    }
  }

  /** Re-place the beard layer on the (possibly jaw-dropped) face. */
  fitFacialHair() {
    if (!this.beard) return;
    const hp = this.headGeo.attributes.position, hn = this.headGeo.attributes.normal, bp = this.beard.geometry.attributes.position;
    for (let i = 0; i < hp.count; i++) {
      const l = this.beardLift[i];
      bp.setXYZ(i, hp.getX(i) + hn.getX(i) * l, hp.getY(i) + hn.getY(i) * l, hp.getZ(i) + hn.getZ(i) * l);
    }
    bp.needsUpdate = true;
    this.beard.geometry.computeVertexNormals();
    this.beard.geometry.computeBoundingSphere();
  }

  // ---- per-frame update ----------------------------------------------------------------------
  update({ weights = {}, blink = 0, brow = 0, glance = { x: 0, y: 0 }, yaw = 0, pitch = 0, roll = 0, breath = 0 }) {
    this.updateMouth(weights);
    for (const e of this.eyes) {
      e.lid.rotation.x = this.lidRest + blink * (-this.lidRest + 0.42) - brow * 0.08;
      e.look.rotation.y = glance.x * 0.32;
      e.look.rotation.x = -glance.y * 0.22;
    }
    for (const b of this.brows) {
      b.g.position.y = brow * 0.07 + (weights.smile ?? 0) * 0.015;
      b.g.rotation.z = -b.side * brow * 0.06;
    }
    this.head.rotation.set(pitch, yaw, roll);
    this.torso.scale.y = 0.85 * (1 + breath);
  }

  dispose() {
    this.group.traverse((o) => o.geometry?.dispose());
    Object.values(this.mat).forEach((m) => m.dispose());
  }
}

/** Mouth shapes to use when no real-mouth capture is available (canonical eye-distance units). */
function builtinMouthShapes(inner, outer) {
  const n = 20;
  const make = (fn) => [...inner.map((p, i) => fn(p, i, true)), ...outer.map((p, i) => fn(p, i, false))];
  const lowerHalf = (i) => i > 10;                       // indices 11..19 are the lower lip
  const mid = (i) => Math.sin((Math.PI * (i % 10)) / 10); // 0 at the corners, 1 in the middle
  return {
    ah: make((p, i) => [0, lowerHalf(i) ? 0.55 * mid(i) : (i === 0 || i === 10 ? 0.2 : -0.05 * mid(i))]),
    ee: make((p) => [Math.sign(p[0]) * 0.14 * Math.abs(p[0]) / 0.4, 0.06 * Math.sign(p[1])]),
    oo: make((p, i) => [-0.4 * p[0], (lowerHalf(i) ? 0.3 : -0.02) * mid(i)]),
    mm: make(() => [0, 0]),
    ff: make((p, i) => [0, lowerHalf(i) ? -0.05 * mid(i) : 0.02]),
    smile: make((p) => [Math.sign(p[0]) * 0.1 * Math.abs(p[0]) / 0.4, -0.08 * Math.abs(p[0]) / 0.4]),
  };
}

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function lipColor(traits, L) {
  const base = cartoonColor(traits.colors.lips, { lift: 1.05, sat: 1.15 });
  const toward = L.lipTint ? srgb(L.lipTint) : cartoonColor(traits.colors.skin, { lift: 0.95 });
  return base.lerp(toward, L.lipMix);
}
