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
import { buildFace, deform, attachVisemes, turnFace } from './avatar-geometry.js';
import { faceFrame } from './visemes.js';
import { analyzeFace } from './toon-analyze.js';
import { ToonHead, suggestLook, suggestFacialHair } from './toon-head.js';
import { liftHand } from './avatar-hand.js';
import { Predictor, MODEL_KEY, MODEL_URL } from './predictor.js';
import { startMicFrames } from './audio.js';
import { createFeatureExtractor } from './features.js';
import { createVoiceEnvelope } from './voice-envelope.js';
import { createFormantTracker, vowelShape } from './formants.js';
import { createListener } from './listening.js';
import { createLatencyMeter, STAGES } from './latency.js';
import { createTalker, defaultModel } from './talk.js';
import { createSeer } from './seeing.js';

const tf = window.tf;
const $ = (id) => document.getElementById(id);
const status = $('status');

// ---- Three.js setup -------------------------------------------------------------
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, stencil: true }); // stencil: mouth mask
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
// Orthographic camera: no perspective, so at rest every pixel lines up exactly
// with the original picture. Turning the face still reveals its 3D shape.
const VIEW_HEIGHT = 0.94; // slightly smaller than the picture (1.0), so its edges stay out of view while it moves
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
let face = null;          // buildFace() result (photo mode)
let toon = null;          // ToonHead (cartoon mode)
let toonTraits = null;
let faceGeo = null;
let faceRoot = null;      // group the face turns with (pivot near the middle of the face)
let mouthParts = null;
let handRoot = null;      // group the hand waves with (pivot at the wrist)
let predictor = null;
let values = null;
const voice = { level: 0, fast: 0, slow: 0, lastLoudAt: 0, talking: false, heardAt: 0, startedAt: 0 };
const envelope = createVoiceEnvelope();
const listener = createListener();
const latency = createLatencyMeter();   // per-stage timing of voice → face
let pendingFrame = null;                   // the newest audio result, waiting to be drawn
let micInputMs = 0;                        // mic/driver latency, when the browser reports it   // reacts to pauses, "umm" and the end of your turn
let formants = null;
const vowel = { open: 0, round: 0, spread: 0, fric: 0, voiced: false }; // smoothed mouth shape from formants
let visemeFile = null;   // captured real-mouth motion (capture.html), if available
const glance = { x: 0, y: 0, tx: 0, ty: 0, next: performance.now() + 2000 };
const gain = () => Number($('gain').value);

// ---- 1. build the avatar from a picture ---------------------------------------------
async function loadAvatar(src) {
  status.textContent = 'loading picture…';
  const original = new Image();
  original.src = src;
  await original.decode();

  status.textContent = 'finding the face…';
  const faceFinder = await createLandmarker('IMAGE');
  let img, res, notes;
  try {
    ({ img, res, notes } = framePicture(original, faceFinder, { crop: src !== DEFAULT_PICTURE }));
  } finally {
    faceFinder.close();
  }

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

  const aspect = img.width / img.height;
  // Build the new face separately; it replaces the old one only once everything is ready.
  const nextFace = buildFace(res.faceLandmarks[0], aspect, { depth: Number($('depth').value) });
  if (visemeFile) attachVisemes(nextFace, visemeFile);
  nextFace.frame = faceFrame((i) => [nextFace.rest[i * 3], nextFace.rest[i * 3 + 1], nextFace.rest[i * 3 + 2]]);

  // Pixels: original (for the face), and, if there's a hand, a hand layer + a background without it.
  const W = img.width, H = img.height;
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

  // Remove the previous avatar, if any, and switch to the new face.
  face = null;
  disposeToon();
  root.clear();
  root.scale.set(1, 1, 1);
  face = nextFace;
  $('toonControls').hidden = true;
  $('depthRow').hidden = false;

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
  root.add(faceMesh); // not in faceRoot: the face turns per-vertex (turnFace) so its outline stays glued to the picture
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

  status.textContent = (notes.length ? notes.join(' ') + ' ' : '') + `avatar ready: ${face.count} points, ${face.info.triangles} triangles` +
    (face.visemes ? ', mouth motion captured from a real face' : ', mouth motion: built-in rules') +
    (face.info.closedBy > 0 ? ', lips closed for the resting pose' : '') +
    (handRoot ? `, hand found${lifted.colorKeyed ? '' : ' (busy background: rough cut-out)'}` : ', no hand found');
}

// ---- getting any picture ready --------------------------------------------------------
const DEFAULT_PICTURE = 'avatar.png';
const BACKDROP = '#1b2030';

/**
 * Prepare a picture: flatten transparency onto a backdrop (Memoji stickers are
 * transparent PNGs), find the face, and crop a square around it so a small face
 * in a big photo still fills the view. Returns the canvas, the face result and
 * friendly notes about anything that may look worse.
 */
function framePicture(original, faceFinder, { crop }) {
  const fit = (w, h, max) => Math.min(1, max / Math.max(w, h));
  const draw = (sx, sy, sw, sh, outMax) => {
    const k = fit(sw, sh, outMax);
    const c = document.createElement('canvas');
    c.width = Math.round(sw * k); c.height = Math.round(sh * k);
    const g = c.getContext('2d');
    g.fillStyle = BACKDROP; g.fillRect(0, 0, c.width, c.height);
    g.drawImage(original, sx, sy, sw, sh, 0, 0, c.width, c.height);
    return c;
  };
  const W0 = original.naturalWidth, H0 = original.naturalHeight;
  let img = draw(0, 0, W0, H0, 1600);
  let res = faceFinder.detect(img);
  if (!res.faceLandmarks.length) {
    throw new Error('no face found. Try a clear, front-facing photo where the face is not covered (no sunglasses or hands over it).');
  }
  if (crop) {
    const lm = res.faceLandmarks[0];
    const xs = lm.map((p) => p.x * W0), ys = lm.map((p) => p.y * H0);
    const fw = Math.max(...xs) - Math.min(...xs), fh = Math.max(...ys) - Math.min(...ys);
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2 + 0.12 * fh;
    const side = Math.min(W0, H0, 2.4 * Math.max(fw, fh));
    const sx = Math.min(Math.max(0, cx - side / 2), W0 - side), sy = Math.min(Math.max(0, cy - side / 2), H0 - side);
    img = draw(sx, sy, side, side, 1024);
    res = faceFinder.detect(img);
    if (!res.faceLandmarks.length) throw new Error('lost the face after cropping. Try a photo where the face is larger.');
  } else if (Math.max(img.width, img.height) > 1024) {
    img = draw(0, 0, W0, H0, 1024);
    res = faceFinder.detect(img);
  }
  const notes = [];
  const m = res.facialTransformationMatrixes?.[0]?.data;
  if (m) {
    const yaw = Math.abs(Math.atan2(m[8], m[10]) * 57.3);
    if (yaw > 20) notes.push(`⚠️ The face is turned ${yaw.toFixed(0)}° sideways; a front-facing photo looks better.`);
  }
  const jaw = res.faceBlendshapes?.[0]?.categories.find((c) => c.categoryName === 'jawOpen')?.score ?? 0;
  if (jaw > 0.35) notes.push('ℹ️ The mouth is open in this photo; a closed-mouth photo looks most natural.');
  return { img, res, notes };
}

// ---- cartoon mode: photo → 3D cartoon (toon-analyze.js + toon-head.js) ----------------------
function disposeToon() {
  if (!toon) return;
  root.remove(toon.group);
  toon.dispose();
  toon = null;
}

async function loadToon(src) {
  status.textContent = 'loading picture…';
  const original = new Image();
  original.src = src;
  await original.decode();
  status.textContent = 'reading the face…';
  const faceFinder = await createLandmarker('IMAGE');
  let img, res, notes;
  try {
    ({ img, res, notes } = framePicture(original, faceFinder, { crop: true }));
  } finally {
    faceFinder.close();
  }
  const pixels = img.getContext('2d').getImageData(0, 0, img.width, img.height);
  toonTraits = analyzeFace(pixels, res.faceLandmarks[0]);

  // Swap the scene over to the cartoon.
  face = faceGeo = mouthParts = null;
  handRoot = null;
  disposeToon();
  root.clear();
  root.scale.set(1, 1, 1);
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({ map: gradientTexture() }));
  bg.position.z = -3;
  root.add(bg);
  // Show what was detected; people can correct any of it.
  $('toonLook').value = suggestLook(toonTraits);
  $('facialHair').value = suggestFacialHair(toonTraits, $('toonLook').value);
  $('hairStyle').value = toonTraits.hair.style;
  $('toonGlasses').checked = toonTraits.accessories.glasses;
  buildToon();
  $('waveBtn').disabled = true;
  $('toonControls').hidden = false;
  $('depthRow').hidden = true;
  $('sourceThumb').src = img.toDataURL('image/jpeg', 0.85);
  const hairName = { none: 'no hair', short: 'short hair', bob: 'chin-length hair', curly: 'curly hair', long: 'long hair' }[toonTraits.hair.style];
  const fh = $('facialHair').value;
  status.textContent = (notes.length ? notes.join(' ') + ' ' : '') +
    `cartoon ready: ${$('toonLook').value} style, ${hairName}${toonTraits.accessories.glasses ? ', glasses' : ''}${fh !== 'none' ? ', ' + fh : ''}` +
    (visemeFile ? ', mouth moves with captured real-mouth motion' : '') + '. Wrong guess? Change it in the panel.';
}

function gradientTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d'), grad = g.createRadialGradient(128, 100, 10, 128, 128, 180);
  grad.addColorStop(0, '#3a4560'); grad.addColorStop(1, '#141822');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// (Re)build the cartoon from the traits + the current choices in the panel.
function buildToon() {
  disposeToon();
  toon = new ToonHead(toonTraits, visemeFile, { look: $('toonLook').value });
  toon.setHairStyle($('hairStyle').value);
  toon.setGlasses($('toonGlasses').checked);
  toon.setFacialHair($('facialHair').value);
  toon.group.scale.setScalar(0.2);
  toon.group.position.y = 0.075;
  root.add(toon.group);
}
$('toonLook').addEventListener('change', () => {
  if (!toonTraits || !toon) return;
  // Switching style also switches facial hair to that style's default (unless the photo clearly shows some).
  $('facialHair').value = suggestFacialHair(toonTraits, $('toonLook').value);
  buildToon();
});
$('showLatency').addEventListener('change', (e) => { $('latencyBox').hidden = !e.target.checked; if (e.target.checked) showLatency(); });
$('facialHair').addEventListener('change', (e) => toon?.setFacialHair(e.target.value));
$('hairStyle').addEventListener('change', (e) => toon?.setHairStyle(e.target.value));
$('toonGlasses').addEventListener('change', (e) => toon?.setGlasses(e.target.checked));
$('cartoonMode').addEventListener('change', () => { if (currentSrc !== DEFAULT_PICTURE) useOwnPicture(currentSrc); });

// ---- trying your own face: selfie, upload, drag-and-drop ----------------------------------
async function useOwnPicture(src) {
  currentSrc = src;
  $('resetBtn').hidden = src === DEFAULT_PICTURE;
  try {
    if (src !== DEFAULT_PICTURE && $('cartoonMode').checked) await loadToon(src);
    else await loadAvatar(src);
    $('scene').scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (err) {
    status.textContent = '⚠️ ' + err.message;
    if (src !== DEFAULT_PICTURE) { currentSrc = DEFAULT_PICTURE; $('resetBtn').hidden = true; }
  }
}

$('selfieBtn').addEventListener('click', async (e) => {
  const btn = e.target, video = $('selfieVideo');
  btn.disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720, facingMode: 'user' } });
    video.srcObject = stream;
    video.hidden = false;
    await video.play();
    for (let n = 3; n > 0; n--) { $('selfieCount').textContent = `Look at the camera, mouth closed… ${n}`; await new Promise((r) => setTimeout(r, 1000)); }
    const c = document.createElement('canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    $('selfieCount').textContent = '📸';
    await useOwnPicture(c.toDataURL('image/jpeg', 0.92));
  } catch (err) {
    $('selfieCount').textContent = err.name === 'NotAllowedError' ? '⚠️ Camera access was blocked.' : '⚠️ ' + err.message;
  } finally {
    stream?.getTracks().forEach((t) => t.stop());
    video.hidden = true;
    btn.disabled = false;
    setTimeout(() => { if ($('selfieCount').textContent === '📸') $('selfieCount').textContent = ''; }, 1500);
  }
});

$('resetBtn').addEventListener('click', () => useOwnPicture(DEFAULT_PICTURE));

// Drag a picture onto the avatar.
canvas.addEventListener('dragover', (e) => { e.preventDefault(); canvas.classList.add('dropping'); });
canvas.addEventListener('dragleave', () => canvas.classList.remove('dropping'));
canvas.addEventListener('drop', (e) => {
  e.preventDefault();
  canvas.classList.remove('dropping');
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('image/'));
  if (file) useOwnPicture(URL.createObjectURL(file));
});

// Inside of the mouth. First a MASK shaped exactly like the opening between the
// inner lips is drawn into the stencil buffer (invisible). The dark cavity, teeth
// and tongue are then drawn only where that mask is. So they always follow the
// shape of the lips and can never poke out past them (no "fangs" at the corners).
const INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
function buildMouth(parent) {
  const { C, Wm, H, zBack } = face.info;
  const behind = (d) => Math.max(zBack + 0.002, C.z - d * H);
  const inMouth = (color) => new THREE.MeshBasicMaterial({
    color, depthTest: false, depthWrite: false,
    stencilWrite: true, stencilRef: 1, stencilFunc: THREE.EqualStencilFunc,
    stencilFail: THREE.KeepStencilOp, stencilZFail: THREE.KeepStencilOp, stencilZPass: THREE.KeepStencilOp,
  });

  // Mask: a fan of triangles from the mouth centre to each inner-lip point.
  const n = INNER_LIP.length;
  const maskGeo = new THREE.BufferGeometry();
  maskGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((n + 1) * 3), 3));
  const idx = [];
  for (let i = 0; i < n; i++) idx.push(n, i, (i + 1) % n);
  maskGeo.setIndex(idx);
  const mask = new THREE.Mesh(maskGeo, new THREE.MeshBasicMaterial({
    colorWrite: false, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
    stencilWrite: true, stencilRef: 1, stencilFunc: THREE.AlwaysStencilFunc, stencilZPass: THREE.ReplaceStencilOp,
  }));
  mask.renderOrder = 0.5;
  mask.frustumCulled = false;

  // Dark mouth: a radial gradient (darker at the back) so it has depth.
  const g = document.createElement('canvas'); g.width = g.height = 64;
  const gc = g.getContext('2d'), grad = gc.createRadialGradient(32, 26, 4, 32, 32, 32);
  grad.addColorStop(0, '#12030a'); grad.addColorStop(1, '#4a1420');
  gc.fillStyle = grad; gc.fillRect(0, 0, 64, 64);
  const cavityMat = inMouth(0xffffff); cavityMat.map = new THREE.CanvasTexture(g);
  const cavity = new THREE.Mesh(new THREE.PlaneGeometry(2.6 * Wm, 2.2 * Wm), cavityMat);
  cavity.position.set(C.x, C.y - 0.2 * Wm, behind(0.045));

  // Teeth: rounded strips (ellipse segments) under the upper lip and above the lower lip.
  const teethShape = (w, h, up) => {
    const sh = new THREE.Shape();
    sh.moveTo(-w / 2, 0);
    sh.quadraticCurveTo(0, (up ? -1 : 1) * h * 0.35, w / 2, 0);
    sh.lineTo(w / 2, (up ? 1 : -1) * h);
    sh.lineTo(-w / 2, (up ? 1 : -1) * h);
    return new THREE.ShapeGeometry(sh);
  };
  const upperTeeth = new THREE.Mesh(teethShape(1.5 * Wm, 0.5 * Wm, true), inMouth(0xf4f0e8));
  upperTeeth.position.set(C.x, C.y - 0.1 * Wm, behind(0.012));
  const lowerTeeth = new THREE.Mesh(teethShape(1.3 * Wm, 0.5 * Wm, false), inMouth(0xe4ddd0));
  lowerTeeth.position.set(C.x, C.y - 0.06 * Wm, behind(0.016));

  const tongue = new THREE.Mesh(new THREE.CircleGeometry(1, 32), inMouth(0xb24a58));
  tongue.scale.set(0.75 * Wm, 0.35 * Wm, 1);
  tongue.position.set(C.x, C.y - 0.45 * Wm, behind(0.03));

  [cavity, tongue, lowerTeeth, upperTeeth].forEach((m, i) => { m.renderOrder = 1 + i; m.frustumCulled = false; });
  parent.add(mask, cavity, upperTeeth, lowerTeeth, tongue);
  mouthParts = { mask, upperTeeth, lowerTeeth, tongue, upperY: upperTeeth.position.y, lowerY: lowerTeeth.position.y, tongueY: tongue.position.y, z: behind(0) };
}

// Each frame: reshape the mask to the current inner-lip points (slightly enlarged so no seam shows).
function updateMouthMask(pos) {
  const arr = mouthParts.mask.geometry.attributes.position.array;
  const n = INNER_LIP.length;
  let cx = 0, cy = 0;
  for (const i of INNER_LIP) { cx += pos[i * 3] / n; cy += pos[i * 3 + 1] / n; }
  INNER_LIP.forEach((i, k) => {
    arr[k * 3] = cx + (pos[i * 3] - cx) * 1.06;
    arr[k * 3 + 1] = cy + (pos[i * 3 + 1] - cy) * 1.12;
    arr[k * 3 + 2] = mouthParts.z;
  });
  arr[n * 3] = cx; arr[n * 3 + 1] = cy; arr[n * 3 + 2] = mouthParts.z;
  mouthParts.mask.geometry.attributes.position.needsUpdate = true;
}

// ---- 2. model + microphone ------------------------------------------------------------
async function loadModel() {
  const found = await Predictor.load(tf, [MODEL_KEY, MODEL_URL]);
  if (!found) {
    $('modelInfo').innerHTML = '⚠️ trained model not found: the mouth will follow loudness only. ' +
      'Train one on the <a href="train.html">Train page</a> or add <code>model/lipsync-model.json</code>.';
    $('micBtn').disabled = false;
    return;
  }
  predictor = found.predictor;
  const where = found.source.startsWith('indexeddb') ? 'trained in this browser'
    : found.predictor.meta.trainedWith === 'pytorch' ? 'trained in PyTorch' : 'from model/ folder';
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
      onFrame: (samples, endFrame) => {
        if (!features) return;
        // When was this sound actually made? Count back from now:
        //   how long ago the frame's last sample entered the audio graph (browser buffering + message hop),
        //   + half the 1024-sample window (its middle is "the sound"),
        //   + the mic's own input latency, if the browser reports it.
        const t0 = performance.now();
        const halfWindow = (samples.length / 2 / mic.sampleRate) * 1000;
        const buffered = Math.max(0, (mic.ctx.currentTime - endFrame / mic.sampleRate) * 1000);
        const soundAt = t0 - buffered - halfWindow - micInputMs;
        latency.record('audio in', t0 - soundAt);
        const { mfcc, rms } = features.compute(samples);
        const t1 = performance.now();
        latency.record('features', t1 - t0);
        envelope.update(rms);
        const f = formants.analyze(samples);
        latency.record('formants', performance.now() - t1);
        updateVowel(f);
        if (!talker.busy) listener.hear({ now: performance.now(), loud: envelope.value > 0.15, shape: f && envelope.value > 0.05 ? vowelShape(f) : null });
        if (envelope.value > 0.15) voice.heardAt = performance.now();
        const level = Math.min(1, Math.sqrt(rms) * 4);
        voice.level = 0.8 * voice.level + 0.2 * level;
        voice.fast = 0.6 * voice.fast + 0.4 * level;   // reacts within ~100 ms
        voice.slow = 0.97 * voice.slow + 0.03 * level;  // ~1 s average
        onVoiceLevel(level);
        if (predictor) {
          predictor.calibrate = $('calib').checked;
          const t2 = performance.now();
          values = predictor.push(mfcc, Number($('smooth').value));
          latency.record('model', performance.now() - t2);
        }
        pendingFrame = { soundAt, readyAt: performance.now() };
      },
    });
    micInputMs = (mic.stream.getAudioTracks()[0]?.getSettings?.().latency ?? 0) * 1000;
    features = createFeatureExtractor({ sampleRate: mic.sampleRate, fftSize: 1024 });
    formants = createFormantTracker(mic.sampleRate);
    e.target.textContent = '🎙 Listening';
    voice.startedAt = performance.now();
    $('micInfo').textContent = `🎙 ${mic.micLabel}: talk and watch the mouth follow you`;
    $('meterRow').hidden = false;
  } catch (err) {
    const denied = err.name === 'NotAllowedError' || err.name === 'SecurityError';
    $('micInfo').textContent = denied
      ? '⚠️ Microphone access was blocked. Click the icon at the left of the address bar, allow the microphone, and reload.'
      : err.name === 'NotFoundError' ? '⚠️ No microphone found on this device.' : '⚠️ mic error: ' + err.message;
    e.target.disabled = false;
  }
});

// Mouth shape from the voice's resonances (see formants.js). Voiced sounds get
// their vowel shape; hiss like "s"/"f"/"sh" gets a narrow, slightly spread mouth.
function updateVowel(f) {
  const env = envelope.value;
  let target;
  if (f && env > 0.05) target = { ...vowelShape(f), fric: 0, voiced: true };
  else if (env > 0.15) target = { open: 0, round: 0, spread: 0.35, fric: 1, voiced: false };
  else target = { open: vowel.open, round: 0, spread: 0, fric: 0, voiced: false };
  for (const k of ['open', 'round', 'spread', 'fric']) vowel[k] += (target[k] - vowel[k]) * 0.35;
  vowel.voiced = target.voiced;
}

// How much of each captured real-mouth shape to use right now.
// What drives the mouth: your voice through the mic, or, while the Memoji is talking, its own words.
function mouthSource() {
  return talker.mouth(performance.now()) ??
    { env: envelope.value, open: vowel.open, round: vowel.round, spread: vowel.spread, fric: vowel.fric, mic: true };
}

// The trained model's blendshapes, rescaled to 0..1 mouth-shape weights.
// Each blendshape has its own typical range in the recordings (median → 95th percentile),
// e.g. jawOpen 0.035 → 0.44, so raw values would barely move the mouth.
const MODEL_RANGE = {
  jawOpen: [0.035, 0.44], mouthFunnel: [0.055, 0.31], mouthPucker: [0.048, 0.99],
  mouthStretchLeft: [0.006, 0.1], mouthStretchRight: [0.023, 0.24],
};
function modelVisemes() {
  if (!values || !predictor) return null;
  const v = (n) => { const [lo, hi] = MODEL_RANGE[n]; return clamp((predictor.get(values, n) - lo) / (hi - lo), 0, 1.3); };
  return {
    ah: v('jawOpen'),
    oo: Math.max(v('mouthFunnel'), 0.6 * v('mouthPucker')),
    ee: (v('mouthStretchLeft') + v('mouthStretchRight')) / 2,
  };
}

function visemeWeights() {
  const src = mouthSource();
  const env = src.env;
  const speaking = Math.min(1, env / 0.12);
  const k = gain() / 1.6; // the expressiveness slider (1.6 = default)
  const c = (v) => Math.min(1.2, Math.max(0, v * k));
  // Rules (loudness + formants) and the trained model, blended by the "mouth driven by" slider.
  // The model only knows the mic's voice, so it's off while the Memoji talks.
  const rule = {
    ah: Math.pow(env, 0.75) * (0.15 + 0.85 * src.open),
    oo: src.round * speaking,
    ee: 0.9 * src.spread * speaking,
  };
  const m = src.mic ? modelVisemes() : null;
  const mix = m ? Number($('modelMix').value) : 0;
  const blend = (key) => (1 - mix) * rule[key] + mix * m[key] * speaking;
  return {
    // Jaw: never less than loudness says, so it stays instant even though the model looks ahead.
    ah: c(mix ? Math.max(blend('ah'), rule.ah * 0.7) : rule.ah),
    oo: c(mix ? blend('oo') : rule.oo),
    ee: c(mix ? blend('ee') : rule.ee),
    ff: c(0.8 * src.fric * speaking),
    smile: c(0.12 + 0.25 * anim.brow + listenPose.smile + (seePose?.smile ?? 0)),
  };
}

// Eyes glance around now and then, like a person thinking or listening.
function updateGlance(now) {
  if (now > glance.next) {
    const r = Math.random();
    [glance.tx, glance.ty] = r < 0.45 ? [0, 0] : r < 0.65 ? [-1, 0] : r < 0.85 ? [1, 0] : [Math.random() * 2 - 1, -0.6];
    glance.next = now + 1200 + Math.random() * 2800;
  }
  glance.x += (glance.tx - glance.x) * 0.3;   // quick, like a real eye movement
  glance.y += (glance.ty - glance.y) * 0.3;
}
function moveEyes(pos) {
  const { ux, iod } = face.frame;
  const amp = 0.045 * iod;
  const dx = (glance.x * ux[0] + glance.y * ux[1]) * amp, dy = (glance.x * ux[1] - glance.y * ux[0]) * amp;
  for (let i = 468; i < Math.min(478, face.count); i++) { pos[i * 3] += dx; pos[i * 3 + 1] += dy; }
}

// Speech detection, used to start a wave when you begin talking after a pause.
function onVoiceLevel(level) {
  const now = performance.now();
  const loud = envelope.value > 0.15; // relative to this mic's own background level, so it works on quiet mics too
  if (loud) {
    if (!voice.talking && now - voice.lastLoudAt > 2500 && $('autoWave').checked && !talker.busy) startWave(1.6);
    voice.talking = true;
    voice.lastLoudAt = now;
  } else if (now - voice.lastLoudAt > 400) {
    voice.talking = false;
  }
}

// ---- 3. life: blinks, brows, head motion, waving -----------------------------------------
const anim = { talk: 0, blinkStart: -1, nextBlink: performance.now() + 1500, double: false, brow: 0, waveStart: -1, waveLen: 0 };

function blinkAmount(now) {
  if (anim.blinkStart < 0 && now >= anim.nextBlink) { anim.blinkStart = now; anim.double = Math.random() < 0.2; }
  if (anim.blinkStart < 0) return 0;
  const t = now - anim.blinkStart;          // close 55 ms, hold 30 ms, open 95 ms (real blinks take ~150–200 ms)
  let b = t < 55 ? t / 55 : t < 85 ? 1 : t < 180 ? 1 - (t - 85) / 95 : 0;
  if (t >= 180) {
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
  // Resting sway, plus gestures that follow your speech rhythm while you talk.
  const idle = 0.05 * Math.sin(now / 900) + anim.talk * (0.1 * Math.sin(now / 260) + 0.12 * envelope.value);
  if (anim.waveStart < 0) return idle;
  const t = now - anim.waveStart;
  if (t > anim.waveLen) { anim.waveStart = -1; return idle; }
  const env = Math.sin(Math.PI * (t / anim.waveLen)); // ease in and out
  return idle + env * 0.32 * Math.sin((t / 1000) * 2 * Math.PI * 1.7);
}

// ---- 4. animation loop -----------------------------------------------------------------
const names = () => predictor?.meta.targetNames ?? [];
// How the mouth is driven:
//  - OPENING comes from loudness (voice-envelope.js): works on any mic, instantly.
//  - SHAPE (rounding, smiles, lip movement) comes from the neural network.
//  - In silence everything relaxes, so the mouth rests closed.
// The model alone was trained on one mic and predicts cautious, small values,
// so on its own the mouth barely moved for other people.
const OPENING = new Set(['jawOpen', 'mouthLowerDownLeft', 'mouthLowerDownRight']);
function blendshape(name) {
  const src = mouthSource();
  const env = src.env;
  const i = values && src.mic ? names().indexOf(name) : -1;   // the model only knows the mic's voice
  const model = i < 0 ? 0 : values[i] * gain();
  const speaking = Math.min(1, env / 0.12);              // 0 in silence, 1 once clearly talking
  // How far it opens: loudness × how open the vowel is ("ah" wide, "ee"/"oo" small).
  const open = Math.pow(env, 0.75) * (0.22 + 0.65 * src.open);
  const round = src.round * speaking, spread = src.spread * speaking;
  if (name === 'jawOpen') return Math.min(1, Math.max(model * speaking * 0.5, open));
  if (OPENING.has(name)) return Math.min(1, Math.max(model * speaking * 0.5, 0.45 * open));
  if (name === 'mouthUpperUpLeft' || name === 'mouthUpperUpRight') return Math.min(1, Math.max(model * speaking, 0.3 * open));
  if (name === 'mouthFunnel') return Math.min(1, Math.max(model * speaking, 0.9 * round));
  if (name === 'mouthPucker') return Math.min(1, Math.max(model * speaking, 0.6 * round));
  if (name === 'mouthStretchLeft' || name === 'mouthStretchRight') return Math.min(1, Math.max(model * speaking, 0.7 * spread));
  if (name === 'mouthSmileLeft' || name === 'mouthSmileRight') return Math.min(1, Math.max(model * speaking, 0.3 * spread));
  return Math.min(1, model * (0.25 + 0.75 * speaking));
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
const NO_POSE = { roll: 0, yaw: 0, pitch: 0, brow: 0, smile: 0, gaze: null, label: '' };
let listenPose = NO_POSE;
const seer = createSeer($('seeVideo'));
let seePose = null;
$('seeMode').addEventListener('change', async (e) => {
  if (e.target.checked) {
    await seer.start();
    if (seer.state.error) { e.target.checked = false; $('seeStatus').textContent = '⚠️ ' + seer.state.error; }
    else $('seeStatus').textContent = '';
  } else seer.stop();
  $('seeVideo').hidden = !seer.state.on;
});
// While the AI is thinking: eyes up and to the side, a slight tilt.
const THINK_POSE = { roll: 0.06, yaw: -0.04, pitch: -0.03, brow: 0.3, smile: 0.1, gaze: { x: -0.5, y: -0.6 }, label: '' };

// ---- Conversation mode -------------------------------------------------------------------
// How you spoke this turn, as a short note for the AI (it's told to use it gently, never quote it).
function speakingCues(seconds, text) {
  const words = text.split(/\s+/).filter(Boolean).length;
  const c = listener.takeCounts();
  const parts = [];
  if (seconds > 1.5) parts.push(`~${Math.round((words / seconds) * 60)} words/min`);
  if (voice.startedAt) {
    if (c.umm) parts.push(`${c.umm} "umm"${c.umm > 1 ? 's' : ''}`);
    if (c.pause) parts.push(`${c.pause} hesitation pause${c.pause > 1 ? 's' : ''}`);
    if (!c.umm && !c.pause) parts.push('no hesitation');
  }
  const saw = seer.takeNotes();
  const notes = [];
  if (parts.length) notes.push(`[how they spoke: ${parts.join(', ')}]`);
  if (saw.length) notes.push(`[what you saw on their face while they spoke: they ${saw.join(', ')}]`);
  return notes.join(' ');
}
const talker = createTalker({ getCues: speakingCues, onUpdate: renderTalk });
Object.defineProperty(talker, 'busy', { get: () => ['thinking', 'speaking'].includes(talker.state.state) });

function renderTalk(st) {
  $('talkBtn').textContent = st.active ? '⏹ Stop talking' : '▶ Start talking';
  const line = st.error ? '⚠️ ' + st.error
    : st.state === 'listening' ? '👂 Listening, just talk.' + (st.interim || st.heard ? ` <i>"${esc((st.heard + ' ' + st.interim).trim())}"</i>` : '')
    : st.state === 'thinking' ? '🤔 Thinking…'
    : st.state === 'speaking' ? '🗣 Speaking. Start talking to interrupt.' : '';
  const t = st.timing;
  const speed = t?.firstSound
    ? `<br>⚡ Started replying <b>${((t.firstSound - t.userEnd) / 1000).toFixed(1)} s</b> after you finished ` +
      `(AI's first words ${Math.round(t.firstToken - t.userEnd)} ms, voice ${Math.round(t.firstSound - t.firstToken)} ms).`
    : '';
  $('talkStatus').innerHTML = line + speed;
  $('talkLog').innerHTML = st.transcript.slice(-6)
    .map((m) => `<div class="msg ${m.who}"><b>${m.who === 'you' ? 'You' : 'Memoji'}:</b> ${esc(m.text)}</div>`).join('');
}
const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// Voices: prefer natural-sounding English ones.
function loadVoices() {
  const vs = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'));
  if (!vs.length) return;
  const sel = $('talkVoice');
  const keep = sel.value;
  sel.innerHTML = vs.map((v) => `<option>${esc(v.name)}</option>`).join('');
  const best = vs.find((v) => /natural|aria|jenny|samantha|google us english|zira/i.test(v.name)) ?? vs[0];
  sel.value = vs.some((v) => v.name === keep) ? keep : best.name;
}
if ('speechSynthesis' in window) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }

// Remembered settings (only if the visitor asked us to remember them).
try {
  const saved = JSON.parse(localStorage.getItem('lipsync.talk') ?? 'null');
  if (saved) { $('talkProvider').value = saved.provider; $('talkKey').value = saved.key; $('talkRemember').checked = true; }
} catch {}
$('talkProvider').addEventListener('change', () => { $('talkModel').placeholder = defaultModel($('talkProvider').value); });
$('talkModel').placeholder = defaultModel($('talkProvider').value);

$('talkBtn').addEventListener('click', () => {
  if (talker.state.active) { talker.stop(); return; }
  const provider = $('talkProvider').value, key = $('talkKey').value.trim();
  if (!key) { talker.state.error = 'Paste an API key first.'; renderTalk(talker.state); return; }
  try {
    if ($('talkRemember').checked) localStorage.setItem('lipsync.talk', JSON.stringify({ provider, key }));
    else localStorage.removeItem('lipsync.talk');
  } catch {}
  // The mic also feeds the "how you spoke" notes (pauses, umms), so start it too.
  if (!voice.startedAt && !$('micBtn').disabled) $('micBtn').click();
  const voiceObj = speechSynthesis.getVoices().find((v) => v.name === $('talkVoice').value) ?? null;
  talker.start({ provider, key, model: $('talkModel').value.trim(), voice: voiceObj });
});
const convState = $('convState');

// Latency panel: one row per stage with typical (p50) and worst-case (p95) times, plus a bar.
function showLatency() {
  const box = $('latencyBox');
  const e2e = latency.stats('voice → jaw'), shape = latency.stats('voice → shape');
  if (!e2e.n) { box.innerHTML = '<div class="latNote">Start the mic and talk to measure.</div>'; return; }
  const ms = (v) => v == null ? '–' : v < 10 ? v.toFixed(1) : Math.round(v);
  const max = Math.max(1, ...STAGES.map(([n]) => latency.stats(n).p50 ?? 0));
  const rows = STAGES.map(([name, what]) => {
    const st = latency.stats(name);
    if (name === 'look-ahead' && !st.p50) return '';
    return `<tr title="${what}"><td>${name}</td><td><div class="latBar" style="width:${Math.round(((st.p50 ?? 0) / max) * 100)}%"></div></td>` +
      `<td>${ms(st.p50)}</td><td>${ms(st.p95)}</td></tr>`;
  }).join('');
  const compute = ['features', 'formants', 'model'].reduce((a, n) => a + (latency.stats(n).p50 ?? 0), 0);
  const smooth = Number($('smooth').value);
  const ease = smooth > 0.01 ? -(1000 / 30) / Math.log(smooth) : 0;
  box.innerHTML =
    `<div class="latHead">voice → jaw <b>${ms(e2e.p50)} ms</b> · voice → mouth shape <b>${ms(shape.p50)} ms</b></div>` +
    `<div class="latNote">typical (p50); worst 5% (p95): ${ms(e2e.p95)} ms / ${ms(shape.p95)} ms</div>` +
    `<table><tr><th>stage</th><th></th><th>p50</th><th>p95</th></tr>${rows}</table>` +
    `<div class="latNote">Compute per audio frame: ${ms(compute)} ms of the 33 ms budget ` +
    `(${compute < 33 ? '✅ fits' : '⚠️ over'}). Smoothing adds ~${Math.round(ease)} ms of easing on top. ` +
    (latency.stats('look-ahead').p50 ? `Look-ahead only delays the mouth <i>shape</i>; the jaw opening follows loudness without waiting. ` : '') +
    (micInputMs ? `Includes ${ms(micInputMs)} ms of mic latency reported by the browser. ` : `The mic's own hardware delay isn't reported by this browser, so isn't included. `) +
    `The screen's own delay isn't visible to the browser.</div>`;
}
function frame(t) {
  requestAnimationFrame(frame);
  const drawStart = performance.now();
  const consumed = pendingFrame;
  pendingFrame = null;
  if ((face && faceGeo && mouthParts) || toon) {
    const now = performance.now();
    const alive = $('alive').checked;

    // Brows lift on emphasis: when the voice jumps above its recent average.
    const emphasis = clamp((voice.fast - voice.slow) * 5, 0, 1);
    anim.brow += ((alive ? emphasis : 0) - anim.brow) * 0.15;

    // Head: slow idle drift + small nods that follow the rhythm of speech.
    const s = t / 1000;
    const idle = alive ? 1 : 0;
    // Talking energy: rises while you speak, fades ~1 s after you stop.
    anim.talk += ((voice.talking ? 1 : 0) - anim.talk) * 0.05;
    const env = envelope.value;
    const head = {
      yaw: yaw + idle * (0.06 * Math.sin(s * 0.43) + 0.03 * Math.sin(s * 1.3) + 0.05 * anim.talk * Math.sin(s * 2.1)),
      pitch: pitch + idle * (0.025 * Math.sin(s * 0.31) - 0.08 * emphasis - 0.06 * env),
      roll: idle * (0.02 * Math.sin(s * 0.27) + 0.03 * anim.talk * Math.sin(s * 1.7)),
    };
    // Listening behaviour: tilt and wait on hesitations, nod when you carry on or finish.
    const listening = alive && $('listenMode').checked;
    const talkState = talker.state.state;
    listenPose = talkState === 'thinking' ? THINK_POSE : talkState === 'speaking' ? NO_POSE
      : listening ? listener.pose(now) : NO_POSE;
    head.yaw += listenPose.yaw;
    head.pitch += listenPose.pitch;
    head.roll += listenPose.roll;
    anim.brow = Math.max(anim.brow, listenPose.brow);

    // Seeing you: eye contact, and reactions to your expression (when the webcam is on).
    seer.tick(now);
    seePose = alive ? seer.pose() : null;
    if (seePose) {
      head.yaw += seePose.yaw;
      head.pitch += seePose.pitch;
      if (!listenPose.roll) head.roll += seePose.roll;
      anim.brow = Math.max(anim.brow, seePose.brow);
    }
    const label = talkState === 'thinking' ? '🤔 thinking…' : talkState === 'speaking' ? '🗣 speaking (talk to interrupt)'
      : (listening && listenPose.label) || seePose?.label || '';
    if (convState.textContent !== label) { convState.textContent = label; convState.hidden = !label; }

    const blink = alive ? blinkAmount(now) : 0;
    if (alive) updateGlance(now); else { glance.x = glance.y = 0; }
    // Gaze priority: "thinking" look > eye contact with you > its own idle glances.
    const gaze = listenPose.gaze ?? (seePose && talkState !== 'thinking' ? seePose.gaze : null);
    if (gaze) {
      const k = listenPose.gaze ? 0.12 : 0.2;
      glance.x += (gaze.x - glance.x) * k;
      glance.y += (gaze.y - glance.y) * k;
    }

    if (toon) {
      // Cartoon mode: a real 3D rig, so every part just gets its value.
      toon.update({
        weights: visemeWeights(), blink, brow: anim.brow, glance,
        yaw: head.yaw * 1.4, pitch: head.pitch * 1.3, roll: head.roll,
        breath: idle * 0.012 * Math.sin(s * 2 * Math.PI * 0.22),
      });
    } else {
      faceRoot.rotation.set(head.pitch, head.yaw, head.roll);
      const pos = faceGeo.attributes.position.array;
      const { jawDrop, upperLift = 0 } = deform(face, blendshape, pos, {
        blink, brow: anim.brow, visemes: face.visemes ? visemeWeights() : null,
      });
      if (alive) moveEyes(pos);
      updateMouthMask(pos); // before turning: the mask lives in faceRoot, which turns as a whole
      turnFace(face, pos, [faceRoot.position.x, faceRoot.position.y, faceRoot.position.z], head);
      faceGeo.attributes.position.needsUpdate = true;
      faceGeo.computeBoundingSphere();
      // Teeth only when the lips are actually apart (no specks through closed lips).
      const apart = jawDrop > 0.006 * face.info.H;
      mouthParts.upperTeeth.visible = mouthParts.lowerTeeth.visible = mouthParts.tongue.visible = apart;
      mouthParts.upperTeeth.position.y = mouthParts.upperY + upperLift;
      mouthParts.lowerTeeth.position.y = mouthParts.lowerY - jawDrop;
      mouthParts.tongue.position.y = mouthParts.tongueY - jawDrop * 0.85;
    }

    // Whole avatar (hair, body, hand): a gentle sway and a small bounce on syllables.
    // The view is zoomed in, so the edges stay hidden.
    root.rotation.z = idle * (0.012 * Math.sin(s * 0.5) + 0.018 * anim.talk * Math.sin(s * 1.9));
    root.position.y = idle * (0.004 * Math.sin(s * 0.8) - 0.012 * env);
    root.position.x = idle * 0.006 * anim.talk * Math.sin(s * 1.1);
    if (!toon) root.scale.set(1, 1 + idle * 0.004 * Math.sin(s * 2 * Math.PI * 0.22), 1); // breathing

    if (handRoot) handRoot.rotation.z = alive || anim.waveStart >= 0 ? waveAngle(now) : 0;
  }
  renderer.render(scene, camera);

  // ---- latency bookkeeping
  const drawEnd = performance.now();
  latency.record('draw', drawEnd - drawStart);
  if (consumed) {
    const lookAhead = predictor ? (predictor.meta.future ?? 0) * (1000 / 30) : 0;
    latency.record('look-ahead', lookAhead);
    latency.record('wait', drawStart - consumed.readyAt);
    latency.record('voice → jaw', drawEnd - consumed.soundAt);                 // opening follows loudness at once
    latency.record('voice → shape', drawEnd - consumed.soundAt + lookAhead);  // the model's mouth shape waits for look-ahead
  }
  if ($('showLatency').checked && drawEnd - (frame.latAt ?? 0) > 250) { frame.latAt = drawEnd; showLatency(); }

  if (voice.startedAt) {
    $('meter').style.width = Math.min(100, envelope.aboveFloor * 3.3) + '%';
    const now = performance.now();
    if (!voice.heardAt && now - voice.startedAt > 6000) {
      $('micInfo').textContent = "⚠️ I can't hear you yet. Check the right microphone is selected " +
        '(icon at the right of the address bar) and that it isn\'t muted.';
    } else if (voice.heardAt && $('micInfo').textContent.startsWith('⚠️ I can')) {
      $('micInfo').textContent = '🎙 hearing you: keep talking';
    }
  }

  if (values && $('showDebug').checked) {
    const vw = face?.visemes ? visemeWeights() : null;
    debugRows.textContent = `vowel: open ${vowel.open.toFixed(2)}  round ${vowel.round.toFixed(2)}  spread ${vowel.spread.toFixed(2)}\n` +
      (vw ? 'captured shapes: ' + Object.entries(vw).map(([k, v]) => `${k} ${v.toFixed(2)}`).join('  ') + '\n' : '') + ['jawOpen', 'mouthFunnel', 'mouthPucker', 'mouthSmileLeft', 'mouthLowerDownLeft', 'mouthUpperUpLeft']
      .map((n) => `${n.padEnd(19)} ${'█'.repeat(Math.round(blendshape(n) * 20)).padEnd(20, '·')} ${blendshape(n).toFixed(2)}`)
      .join('\n');
  } else if (!$('showDebug').checked) {
    debugRows.textContent = '';
  }
}
requestAnimationFrame(frame);

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ---- 5. picture controls ---------------------------------------------------------------
let currentSrc = DEFAULT_PICTURE;
$('photoInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) useOwnPicture(URL.createObjectURL(file));
  e.target.value = ''; // so choosing the same file again still triggers
});
$('depth').addEventListener('change', async () => {
  if (face && !toon) try { await loadAvatar(currentSrc); } catch (err) { status.textContent = 'error: ' + err.message; }
});

// ---- start ---------------------------------------------------------------------------
resize();
loadModel();
(async () => {
  // Captured mouth motion: this browser's own capture first, then the repo's copy.
  try { visemeFile = JSON.parse(localStorage.getItem('lipsync-visemes')); } catch { visemeFile = null; }
  if (!visemeFile) {
    try { const r = await fetch('model/visemes.json', { cache: 'no-cache' }); if (r.ok) visemeFile = await r.json(); } catch { /* none yet */ }
  }
  if (visemeFile?.format !== 'lipsync-visemes-v1') visemeFile = null;
  await loadAvatar(currentSrc);
})().catch((err) => {
  status.textContent = `couldn't load avatar.png (${err.message}). Choose your picture with the button below, ` +
    'or save it as avatar.png in the project folder.';
});
