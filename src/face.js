// face.js — Step 3: MediaPipe FaceLandmarker on the webcam + an avatar-image checker.
//
// FaceLandmarker gives us, per face:
//   faceLandmarks            478 points {x, y, z}. x,y are 0..1 of the image; z is depth
//                            (negative = closer to camera), roughly in the same scale as x.
//   faceBlendshapes          52 scores 0..1 named like Apple ARKit: jawOpen, mouthFunnel, ...
//   facialTransformationMatrixes  4x4 head pose (rotation + translation).
//
// In Step 4 the blendshapes become the TRAINING LABELS for our audio model.
// In Step 6 the landmarks on your avatar image become a 3D mesh.

import { FaceLandmarker, DrawingUtils, createLandmarker } from './facemodel.js';

const $ = (id) => document.getElementById(id);

// ---- A. live webcam --------------------------------------------------------

const video = $('video');
const overlay = $('overlay');
const camStatus = $('camStatus');
const bars = $('bars');
const mouthOnly = $('mouthOnly');

const isMouth = (name) => name.startsWith('jaw') || name.startsWith('mouth');
let barEls = null; // name -> {row, fill, val}

$('camBtn').addEventListener('click', async (e) => {
  e.target.disabled = true;
  try {
    camStatus.textContent = 'loading face model (first time takes a few seconds)…';
    const landmarker = await createLandmarker('VIDEO');
    camStatus.textContent = 'starting camera…';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    e.target.textContent = '📷 Tracking';
    runWebcam(landmarker);
  } catch (err) {
    camStatus.textContent = 'error: ' + err.message;
    e.target.disabled = false;
  }
});

mouthOnly.addEventListener('change', () => {
  if (!barEls) return;
  for (const [name, { row }] of Object.entries(barEls)) {
    row.style.display = mouthOnly.checked && !isMouth(name) ? 'none' : '';
  }
});

function runWebcam(landmarker) {
  const ctx = overlay.getContext('2d');
  const draw = new DrawingUtils(ctx);
  let lastVideoTime = -1;
  let frames = 0;
  let fpsT0 = performance.now();

  function frame() {
    // Only run the model when the camera has delivered a new frame.
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const t0 = performance.now();
      const res = landmarker.detectForVideo(video, t0);
      const ms = performance.now() - t0;

      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (res.faceLandmarks.length) {
        drawFaceMesh(draw, res.faceLandmarks[0]);
        updateBars(res.faceBlendshapes[0].categories);
      }

      frames++;
      const now = performance.now();
      if (now - fpsT0 > 500) {
        const fps = (frames * 1000) / (now - fpsT0);
        camStatus.textContent = res.faceLandmarks.length
          ? `face found | ${res.faceLandmarks[0].length} landmarks | ` +
            `${res.faceBlendshapes[0].categories.length} blendshapes | ` +
            `model ${ms.toFixed(1)} ms | ${fps.toFixed(0)} fps`
          : 'no face in view';
        frames = 0;
        fpsT0 = now;
      }
    }
    requestAnimationFrame(frame);
  }
  frame();
}

function drawFaceMesh(draw, landmarks) {
  draw.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: '#ffffff30', lineWidth: 0.5 });
  draw.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#8ab4ff', lineWidth: 1.5 });
  draw.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#8ab4ff', lineWidth: 1.5 });
  draw.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#8ab4ff', lineWidth: 1.5 });
  draw.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#ff6a5a', lineWidth: 2 });
}

function updateBars(categories) {
  if (!barEls) {
    barEls = {};
    for (const { categoryName: name } of categories) {
      const row = document.createElement('div');
      row.className = 'bar' + (isMouth(name) ? ' mouth' : '');
      row.innerHTML = `<span>${name}</span><div class="track"><div class="fill"></div></div><span>0.00</span>`;
      if (mouthOnly.checked && !isMouth(name)) row.style.display = 'none';
      bars.appendChild(row);
      barEls[name] = { row, fill: row.querySelector('.fill'), val: row.lastElementChild };
    }
  }
  for (const { categoryName: name, score } of categories) {
    const el = barEls[name];
    el.fill.style.width = (score * 100).toFixed(1) + '%';
    el.val.textContent = score.toFixed(2);
  }
}

// ---- B. avatar image checker ------------------------------------------------

const photoCanvas = $('photoCanvas');
const photoStatus = $('photoStatus');
const checks = $('checks');
let imageLandmarker = null;

$('photoInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  checks.innerHTML = '';
  try {
    photoStatus.textContent = 'loading face model…';
    imageLandmarker ??= await createLandmarker('IMAGE');

    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    // Draw the image (scaled to fit 460px) — not mirrored, it's a photo.
    const scale = Math.min(1, 460 / Math.max(img.naturalWidth, img.naturalHeight));
    photoCanvas.width = Math.round(img.naturalWidth * scale);
    photoCanvas.height = Math.round(img.naturalHeight * scale);
    const ctx = photoCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0, photoCanvas.width, photoCanvas.height);

    photoStatus.textContent = 'detecting…';
    const res = imageLandmarker.detect(img);
    reportAvatar(res, img, ctx);
  } catch (err) {
    photoStatus.textContent = 'error: ' + err.message;
  }
});

function reportAvatar(res, img, ctx) {
  const found = res.faceLandmarks.length > 0;
  const items = [];
  const add = (ok, text) => items.push(`<li>${ok === null ? '➖' : ok ? '✅' : '⚠️'} ${text}</li>`);

  const minSide = Math.min(img.naturalWidth, img.naturalHeight);
  add(minSide >= 300, `resolution ${img.naturalWidth}×${img.naturalHeight}` + (minSide >= 300 ? '' : ' (bigger is better, aim for 400px+)'));

  if (!found) {
    add(false, '<b>no face detected</b>. MediaPipe can\'t map this picture. Try a more front-facing, neutral-expression version, or a crop that\'s mostly the face.');
    photoStatus.textContent = 'no face found';
    checks.innerHTML = items.join('');
    return;
  }

  const lm = res.faceLandmarks[0];
  add(true, `<b>face detected</b>: ${lm.length} 3D points mapped`);

  // Head yaw (left/right turn) from the pose matrix. data is column-major 4x4.
  const m = res.facialTransformationMatrixes?.[0]?.data;
  if (m) {
    const yaw = Math.abs((Math.atan2(m[8], m[10]) * 180) / Math.PI);
    add(yaw < 15, `head turned ${yaw.toFixed(0)}°` + (yaw < 15 ? ' (front-facing)' : ' (straighter is better, under 15°)'));
  }

  const score = (name) => res.faceBlendshapes[0].categories.find((c) => c.categoryName === name)?.score ?? 0;
  const jaw = score('jawOpen');
  add(jaw < 0.2, `jawOpen ${jaw.toFixed(2)}` + (jaw < 0.2 ? ' (mouth closed, ideal)' : ' (an open mouth makes animating harder)'));

  // Draw the mesh, colouring points by depth so you can see the 3D shape:
  // bright = close to the camera (nose tip), dark = far (cheeks, ears).
  const draw = new DrawingUtils(ctx);
  draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: '#ffffff40', lineWidth: 0.6 });
  draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#ff6a5a', lineWidth: 1.5 });
  let zMin = Infinity, zMax = -Infinity;
  for (const p of lm) { zMin = Math.min(zMin, p.z); zMax = Math.max(zMax, p.z); }
  for (const p of lm) {
    const t = 1 - (p.z - zMin) / (zMax - zMin || 1);   // 1 = closest
    ctx.fillStyle = `hsl(${40 + t * 20}, 100%, ${20 + t * 60}%)`;
    ctx.fillRect(p.x * ctx.canvas.width - 1, p.y * ctx.canvas.height - 1, 2, 2);
  }
  add(null, 'dot brightness = depth: bright dots are closer to the camera. That depth becomes the 3D shape in Step 6.');

  photoStatus.textContent = 'done';
  checks.innerHTML = items.join('');
}
