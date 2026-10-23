// avatar.js — Step 6: your picture as a 3D face that lip-syncs to your voice.
//
//   avatar.png ──MediaPipe face (once)──► 478 3D landmarks ──avatar-geometry.js──► textured face mesh
//              ──MediaPipe hand (once)──► 21 hand points ──avatar-hand.js──► hand layer + background without the hand
//   mic ──AudioWorklet──► MFCCs ──trained model──► 27 mouth blendshapes ──deform()──► moving mesh
//
// Layers, back to front:
//   background (picture with the hand painted out) → mouth interior → face mesh → hand
// Only the FACE turns and nods; hair and body stay put. That's the classic 2.5D trick:
// one picture has no information about the back of the head, but the face has depth.

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
import { createLandmarker, createHandLandmarker } from './facemodel.js';
import { buildFace, deform } from './avatar-geometry.js';
import { liftHand } from './avatar-hand.js';
import { Predictor, MODEL_KEY, MODEL_URL } from './predictor.js';
import { startMicFrames } from './audio.js';
import { createFeatureExtractor } from './features.js';

const tf = window.tf;
const $ = (id) => document.getElementById(id);
const status = $('status');

// ---- Three.js setup -------------------------------------------------------------
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
// Orthographic camera: no perspective, so at rest every pixel lines up exactly
// with the original picture. Turning the face still reveals its 3D shape.
const VIEW_HEIGHT = 1.06;
const camera = new THREE.OrthographicCamera(-0.5, 0.5, VIEW_HEIGHT / 2, -VIEW_HEIGHT / 2, -10, 10);
camera.position.set(0, 0, 2);
const root = new THREE.Group();
scene.add(root);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  const halfW = (VIEW_HEIGHT / 2) * (w / h);
  camera.left = -halfW;
  camera.right = halfW;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

// ---- state ---------------------------------------------------------------------
let face = null;          // buildFace() result
let faceGeo = null;
let faceRoot = null;      // group the face turns with (pivot near the middle of the face)
let mouthParts = null;
let handRoot = null;      // group the hand waves with (pivot at the wrist)
let predictor = null;
let values = null;
const voice = { level: 0, fast: 0, slow: 0, lastLoudAt: 0, talking: false };
const gain = () => Number($('gain').value);

// ---- 1. build the avatar from a picture ---------------------------------------------
async function loadAvatar(src) {
  status.textContent = 'loading picture…';
  const img = new Image();
  img.src = src;
  await img.decode();

  status.textContent = 'finding face landmarks…';
  const faceFinder = await createLandmarker('IMAGE');
  const res = faceFinder.detect(img);
  faceFinder.close();
  if (!res.faceLandmarks.length) throw new Error('no face found in this picture; try a front-facing one');

  status.textContent = 'looking for a hand…';
  let hand = null;
  try {
    const handFinder = await createHandLandmarker();
    const hr = handFinder.detect(img);
    handFinder.close();
    hand = hr.landmarks?.[0] ?? null;
  } catch (err) {
    console.warn('hand detection failed', err);
  }

  const aspect = img.naturalWidth / img.naturalHeight;
  face = buildFace(res.faceLandmarks[0], aspect, { depth: Number($('depth').value) });

  // Pixels: original (for the face), and, if there's a hand, a hand layer + a background without it.
  const W = img.naturalWidth, H = img.naturalHeight;
  const cnv = document.createElement('canvas');
  cnv.width = W; cnv.height = H;
  const ctx = cnv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const tex = (source) => {
    const t = new THREE.CanvasTexture(source);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return t;
  };
  const faceTexture = tex(cnv);
  let bgTexture = faceTexture, handTexture = null, lifted = null;
  if (hand) {
    status.textContent = 'lifting the hand off the picture…';
    await new Promise((r) => setTimeout(r, 0)); // let the status text paint
    lifted = liftHand(ctx.getImageData(0, 0, W, H), hand);
    const layer = (rgba) => {
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      c.getContext('2d').putImageData(new ImageData(rgba, W, H), 0, 0);
      return c;
    };
    bgTexture = tex(layer(lifted.bgRGBA));
    handTexture = tex(layer(lifted.handRGBA));
  }

  // Remove the previous avatar, if any.
  root.clear();

  const flat = (map, order, extra = {}) => new THREE.MeshBasicMaterial({ map, depthTest: false, depthWrite: false, ...extra });

  // Background.
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(aspect, 1), flat(bgTexture));
  bg.renderOrder = 0;
  root.add(bg);

  // Face (+ mouth interior) in a group that turns around the middle of the face.
  let cx = 0, cy = 0;
  for (let i = 0; i < face.count; i++) { cx += face.rest[i * 3]; cy += face.rest[i * 3 + 1]; }
  cx /= face.count; cy /= face.count;
  faceRoot = new THREE.Group();
  faceRoot.position.set(cx, cy, face.info.zBack);
  const faceInner = new THREE.Group();
  faceInner.position.set(-cx, -cy, -face.info.zBack);
  faceRoot.add(faceInner);
  root.add(faceRoot);

  faceGeo = new THREE.BufferGeometry();
  faceGeo.setAttribute('position', new THREE.BufferAttribute(face.rest.slice(), 3));
  faceGeo.setAttribute('uv', new THREE.BufferAttribute(face.uv, 2));
  faceGeo.setIndex(new THREE.BufferAttribute(face.indices, 1));
  const faceMesh = new THREE.Mesh(faceGeo, new THREE.MeshBasicMaterial({ map: faceTexture, side: THREE.DoubleSide }));
  faceMesh.renderOrder = 10; // drawn after the mouth interior, so that only shows between the lips
  faceInner.add(faceMesh);
  buildMouth(faceInner);

  // Hand on top, pivoting at the wrist.
  handRoot = null;
  if (lifted) {
    const [px, py] = lifted.pivot;
    handRoot = new THREE.Group();
    handRoot.position.set((px - 0.5) * aspect, 0.5 - py, 0);
    const handPlane = new THREE.Mesh(new THREE.PlaneGeometry(aspect, 1), flat(handTexture, 20, { transparent: true }));
    handPlane.position.set(-(px - 0.5) * aspect, -(0.5 - py), 0);
    handPlane.renderOrder = 20;
    handRoot.add(handPlane);
    root.add(handRoot);
  }
  $('waveBtn').disabled = !handRoot;

  status.textContent = `avatar ready: ${face.count} points, ${face.info.triangles} triangles` +
    (face.info.closedBy > 0 ? ', lips closed for the resting pose' : '') +
    (handRoot ? `, hand found${lifted.colorKeyed ? '' : ' (busy background: rough cut-out)'}` : ', no hand found');
}

// Inside of the mouth. Drawn BEFORE the face and ignoring depth; the face is
// drawn on top, so these only ever show through the hole between the lips.
function buildMouth(parent) {
  const { C, Wm, H, zBack } = face.info;
  const behind = (d) => Math.max(zBack + 0.002, C.z - d * H);
  const mat = (color) => new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false });

  const cavity = new THREE.Mesh(new THREE.CircleGeometry(1, 32), mat(0x2a0a0e));
  cavity.scale.set(0.95 * Wm, 0.75 * Wm, 1);
  cavity.position.set(C.x, C.y - 0.15 * Wm, behind(0.045));

  const teethGeo = new THREE.PlaneGeometry(1.45 * Wm, 0.32 * Wm);
  const upperTeeth = new THREE.Mesh(teethGeo, mat(0xf1ece2));
  upperTeeth.position.set(C.x, C.y + 0.1 * Wm, behind(0.012));
  const lowerTeeth = new THREE.Mesh(teethGeo, mat(0xe6e0d4));
  lowerTeeth.position.set(C.x, C.y - 0.3 * Wm, behind(0.016));

  const tongue = new THREE.Mesh(new THREE.CircleGeometry(1, 24), mat(0xa8404c));
  tongue.scale.set(0.6 * Wm, 0.28 * Wm, 1);
  tongue.position.set(C.x, C.y - 0.45 * Wm, behind(0.03));

  [cavity, tongue, lowerTeeth, upperTeeth].forEach((m, i) => { m.renderOrder = 1 + i; });
  parent.add(cavity, upperTeeth, lowerTeeth, tongue);
  mouthParts = { lowerTeeth, tongue, lowerY: lowerTeeth.position.y, tongueY: tongue.position.y };
}

// ---- 2. model + microphone ------------------------------------------------------------
async function loadModel() {
  const found = await Predictor.load(tf, [MODEL_KEY, MODEL_URL]);
  if (!found) {
    $('modelInfo').innerHTML = '⚠️ no trained model found. Train one on the <a href="train.html">Step 5 page</a> ' +
      'or put <code>lipsync-model.json</code> + <code>.weights.bin</code> in a <code>model</code> folder.';
    return;
  }
  predictor = found.predictor;
  const where = found.source.startsWith('indexeddb') ? 'trained in this browser' : 'from model/ folder';
  $('modelInfo').textContent = `model: ${where} | score ${predictor.meta.valR2} | ` +
    `trained ${new Date(predictor.meta.trainedAt).toLocaleDateString()}`;
  $('micBtn').disabled = false;
}

$('micBtn').addEventListener('click', async (e) => {
  e.target.disabled = true;
  try {
    let features = null;
    const mic = await startMicFrames({
      fps: 30,
      windowSize: 1024,
      onFrame: (samples) => {
        if (!features) return;
        const { mfcc, rms } = features.compute(samples);
        const level = Math.min(1, Math.sqrt(rms) * 4);
        voice.level = 0.8 * voice.level + 0.2 * level;
        voice.fast = 0.6 * voice.fast + 0.4 * level;   // reacts within ~100 ms
        voice.slow = 0.97 * voice.slow + 0.03 * level;  // ~1 s average
        onVoiceLevel(level);
        predictor.calibrate = $('calib').checked;
        values = predictor.push(mfcc, Number($('smooth').value));
      },
    });
    features = createFeatureExtractor({ sampleRate: mic.sampleRate, fftSize: 1024 });
    e.target.textContent = '🎙 Listening';
    const trainedMic = predictor.meta.mic;
    $('micInfo').textContent = `mic: ${mic.micLabel}` +
      (trainedMic && trainedMic !== mic.micLabel ? ` ⚠️ (model was trained with: ${trainedMic})` : '');
  } catch (err) {
    $('micInfo').textContent = 'mic error: ' + err.message;
    e.target.disabled = false;
  }
});

// Speech detection, used to start a wave when you begin talking after a pause.
function onVoiceLevel(level) {
  const now = performance.now();
  const loud = level > Math.max(0.12, voice.slow * 1.8);
  if (loud) {
    if (!voice.talking && now - voice.lastLoudAt > 2500 && $('autoWave').checked) startWave(1.6);
    voice.talking = true;
    voice.lastLoudAt = now;
  } else if (now - voice.lastLoudAt > 400) {
    voice.talking = false;
  }
}

// ---- 3. life: blinks, brows, head motion, waving -----------------------------------------
const anim = { blinkStart: -1, nextBlink: performance.now() + 1500, double: false, brow: 0, waveStart: -1, waveLen: 0 };

function blinkAmount(now) {
  if (anim.blinkStart < 0 && now >= anim.nextBlink) { anim.blinkStart = now; anim.double = Math.random() < 0.2; }
  if (anim.blinkStart < 0) return 0;
  const t = now - anim.blinkStart;          // close 70 ms, hold 40 ms, open 120 ms
  let b = t < 70 ? t / 70 : t < 110 ? 1 : t < 230 ? 1 - (t - 110) / 120 : 0;
  if (t >= 230) {
    if (anim.double) { anim.double = false; anim.blinkStart = now + 90; }
    else { anim.blinkStart = -1; anim.nextBlink = now + 2200 + Math.random() * 3800; }
    b = 0;
  }
  return Math.max(0, b);
}

function startWave(seconds) {
  if (!handRoot) return;
  anim.waveStart = performance.now();
  anim.waveLen = seconds * 1000;
}
$('waveBtn').addEventListener('click', () => startWave(2.4));

function waveAngle(now) {
  const idle = 0.035 * Math.sin(now / 900);
  if (anim.waveStart < 0) return idle;
  const t = now - anim.waveStart;
  if (t > anim.waveLen) { anim.waveStart = -1; return idle; }
  const env = Math.sin(Math.PI * (t / anim.waveLen)); // ease in and out
  return idle + env * 0.32 * Math.sin((t / 1000) * 2 * Math.PI * 1.7);
}

// ---- 4. animation loop -----------------------------------------------------------------
const names = () => predictor?.meta.targetNames ?? [];
function blendshape(name) {
  if (!values) return 0;
  const i = names().indexOf(name);
  return i < 0 ? 0 : Math.min(1, values[i] * gain());
}

// Drag to turn the face (limited: one picture only shows its front).
let drag = null, yaw = 0, pitch = 0;
canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw, pitch }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  yaw = clamp(drag.yaw + (e.clientX - drag.x) * 0.004, -0.3, 0.3);
  pitch = clamp(drag.pitch + (e.clientY - drag.y) * 0.003, -0.2, 0.2);
});
canvas.addEventListener('pointerup', () => { drag = null; });
canvas.addEventListener('dblclick', () => { yaw = pitch = 0; });

const debugRows = $('debug');
function frame(t) {
  requestAnimationFrame(frame);
  if (face) {
    const now = performance.now();
    const alive = $('alive').checked;

    // Brows lift on emphasis: when the voice jumps above its recent average.
    const emphasis = clamp((voice.fast - voice.slow) * 5, 0, 1);
    anim.brow += ((alive ? emphasis : 0) - anim.brow) * 0.15;

    const pos = faceGeo.attributes.position.array;
    const { jawDrop } = deform(face, blendshape, pos, { blink: alive ? blinkAmount(now) : 0, brow: anim.brow });
    faceGeo.attributes.position.needsUpdate = true;
    faceGeo.computeBoundingSphere();
    mouthParts.lowerTeeth.position.y = mouthParts.lowerY - jawDrop;
    mouthParts.tongue.position.y = mouthParts.tongueY - jawDrop * 0.85;

    // Head: slow idle drift + small nods that follow the rhythm of speech.
    const s = t / 1000;
    const idle = alive ? 1 : 0;
    faceRoot.rotation.y = yaw + idle * (0.05 * Math.sin(s * 0.43) + 0.02 * Math.sin(s * 1.3));
    faceRoot.rotation.x = pitch + idle * (0.025 * Math.sin(s * 0.31) - 0.07 * emphasis - 0.02 * voice.level);
    faceRoot.rotation.z = idle * 0.02 * Math.sin(s * 0.27);

    if (handRoot) handRoot.rotation.z = alive || anim.waveStart >= 0 ? waveAngle(now) : 0;
  }
  renderer.render(scene, camera);

  if (values && $('showDebug').checked) {
    debugRows.textContent = ['jawOpen', 'mouthFunnel', 'mouthPucker', 'mouthSmileLeft', 'mouthLowerDownLeft', 'mouthUpperUpLeft']
      .map((n) => `${n.padEnd(19)} ${'█'.repeat(Math.round(blendshape(n) * 20)).padEnd(20, '·')} ${blendshape(n).toFixed(2)}`)
      .join('\n');
  } else if (!$('showDebug').checked) {
    debugRows.textContent = '';
  }
}
requestAnimationFrame(frame);

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ---- 5. picture controls ---------------------------------------------------------------
let currentSrc = 'avatar.png';
$('photoInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  currentSrc = URL.createObjectURL(file);
  try { await loadAvatar(currentSrc); } catch (err) { status.textContent = 'error: ' + err.message; }
});
$('depth').addEventListener('change', async () => {
  if (face) try { await loadAvatar(currentSrc); } catch (err) { status.textContent = 'error: ' + err.message; }
});

// ---- start ---------------------------------------------------------------------------
resize();
loadModel();
loadAvatar(currentSrc).catch((err) => {
  status.textContent = `couldn't load avatar.png (${err.message}). Choose your picture with the button below, ` +
    'or save it as avatar.png in the project folder.';
});
