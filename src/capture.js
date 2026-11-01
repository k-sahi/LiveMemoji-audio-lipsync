// capture.js — record how YOUR mouth really moves, once, so any avatar can move the same way.
//
// For each mouth shape we collect ~1.2 s of webcam frames, remove head movement
// (align to the neutral face using points that don't move when you talk), average,
// and store how every landmark moved. See visemes.js for the math.

import { FaceLandmarker, DrawingUtils, createLandmarker } from './facemodel.js';
import { SHAPES, toPixels, alignToNeutral, meanPoints, toCanonicalDelta, describeCapture } from './visemes.js';

export const STORAGE_KEY = 'lipsync-visemes';
const $ = (id) => document.getElementById(id);

const STEPS = [
  { key: 'rest', emoji: '😐', say: 'Relax your face, lips gently closed', tip: 'This is your neutral face. Everything else is measured from it.' },
  { key: 'ah', emoji: '😮', say: 'Say a long "aaaah"', tip: 'Open wide, like at the doctor.' },
  { key: 'ee', emoji: '😁', say: 'Say a long "eeeee"', tip: 'Stretch your lips sideways, teeth almost together.' },
  { key: 'oo', emoji: '😗', say: 'Say a long "ooooo"', tip: 'Round your lips into a small O.' },
  { key: 'mm', emoji: '🤐', say: 'Say "mmmmm" with lips pressed', tip: 'Press your lips together firmly.' },
  { key: 'ff', emoji: '😬', say: 'Say a long "fffff"', tip: 'Rest your lower lip against your upper teeth.' },
  { key: 'smile', emoji: '😊', say: 'Big smile, mouth closed', tip: 'A natural, wide smile.' },
];
const READY_MS = 2200, HOLD_MS = 1300;

const video = $('video'), overlay = $('overlay'), status = $('status');
let landmarker = null, latest = null, W = 640, H = 480;

$('camBtn').addEventListener('click', async (e) => {
  e.target.disabled = true;
  try {
    status.textContent = 'loading face model…';
    landmarker = await createLandmarker('VIDEO');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    video.srcObject = stream;
    await video.play();
    W = overlay.width = video.videoWidth;
    H = overlay.height = video.videoHeight;
    track();
    $('goBtn').disabled = false;
    status.textContent = 'Face the camera straight on, in good light, then click Start capture.';
  } catch (err) {
    status.textContent = 'camera error: ' + err.message;
    e.target.disabled = false;
  }
});

// Head angle from MediaPipe's pose matrix: we only accept frames facing the camera.
function headAngles(m) {
  if (!m) return { yaw: 0, pitch: 0 };
  return { yaw: Math.atan2(m[8], m[10]) * 57.3, pitch: Math.asin(-Math.max(-1, Math.min(1, m[9]))) * 57.3 };
}

function track() {
  const ctx = overlay.getContext('2d');
  const draw = new DrawingUtils(ctx);
  let last = -1;
  const frame = () => {
    if (video.currentTime !== last) {
      last = video.currentTime;
      const res = landmarker.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, W, H);
      if (res.faceLandmarks.length) {
        const lm = res.faceLandmarks[0];
        const ang = headAngles(res.facialTransformationMatrixes?.[0]?.data);
        const facing = Math.abs(ang.yaw) < 12 && Math.abs(ang.pitch) < 15;
        latest = { pts: toPixels(lm, W, H), facing, time: performance.now() };
        draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: facing ? '#3ecf6e' : '#f5a524', lineWidth: 2 });
        $('pose').textContent = facing ? '✅ facing the camera' : `↪ turn toward the camera (yaw ${ang.yaw.toFixed(0)}°, pitch ${ang.pitch.toFixed(0)}°)`;
      } else {
        latest = null;
        $('pose').textContent = '⚠️ no face found';
      }
    }
    requestAnimationFrame(frame);
  };
  frame();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function collect(ms) {
  const frames = [];
  const end = performance.now() + ms;
  let lastTime = 0;
  while (performance.now() < end) {
    if (latest && latest.facing && latest.time !== lastTime) { frames.push(latest.pts); lastTime = latest.time; }
    await sleep(15);
  }
  return frames;
}

$('goBtn').addEventListener('click', async (e) => {
  e.target.disabled = true;
  $('result').innerHTML = '';
  const raw = {};
  for (let s = 0; s < STEPS.length; s++) {
    const step = STEPS[s];
    $('card').innerHTML = `<div class="emoji">${step.emoji}</div><div class="say">${step.say}</div><div class="tip">${step.tip}</div>`;
    $('progress').style.width = `${(s / STEPS.length) * 100}%`;
    $('phase').textContent = `${s + 1}/${STEPS.length}: get ready…`;
    await sleep(READY_MS);
    $('phase').textContent = `${s + 1}/${STEPS.length}: 🔴 hold it…`;
    $('card').classList.add('recording');
    raw[step.key] = await collect(HOLD_MS);
    $('card').classList.remove('recording');
    if (raw[step.key].length < 10) {
      $('card').innerHTML = '<div class="say">Lost your face for that one. Let\'s try again.</div>';
      await sleep(1200);
      s--;
    }
  }
  $('progress').style.width = '100%';
  $('phase').textContent = 'done!';
  $('card').innerHTML = '<div class="emoji">🎉</div><div class="say">Captured!</div>';
  finish(raw);
  e.target.disabled = false;
});

function finish(raw) {
  // Neutral: align all rest frames to the first one, then average (removes small head wobble).
  const ref = raw.rest[0];
  const neutral = meanPoints(raw.rest.map((f) => alignToNeutral(f, ref)));
  const deltas = {};
  for (const k of SHAPES) deltas[k] = toCanonicalDelta(meanPoints(raw[k].map((f) => alignToNeutral(f, neutral))), neutral);

  const report = describeCapture(deltas);
  $('result').innerHTML = '<h3>How much each shape moved your mouth</h3><table>' + SHAPES.map((k) => {
    const { mouth, still } = report[k];
    const ok = mouth > 0.06 && still < 0.05;
    return `<tr><td>${STEPS.find((s) => s.key === k).emoji} ${k}</td><td><div class="track"><div class="fill" style="width:${Math.min(100, mouth * 250)}%"></div></div></td>` +
      `<td>${ok ? '✅' : mouth <= 0.06 ? '⚠️ small, exaggerate it a bit more' : '⚠️ your head moved a lot'}</td></tr>`;
  }).join('') + '</table>';

  const file = {
    format: 'lipsync-visemes-v1',
    capturedAt: new Date().toISOString(),
    units: 'eye-distance; x along the eye line, y down the face, z toward/away from camera (MediaPipe sign)',
    points: neutral.length,
    deltas: Object.fromEntries(SHAPES.map((k) => [k, Array.from(deltas[k], (v) => Math.round(v * 1e4) / 1e4)])),
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(file)); } catch { /* storage full or blocked: download still works */ }
  window.__capture = file;
  $('saveInfo').textContent = 'Saved in this browser: the avatar page uses it right away. ' +
    'Download it and put it in the model folder so visitors get your mouth motion too.';
  $('dlBtn').disabled = false;
}

$('dlBtn').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(window.__capture)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'visemes.json';
  a.click();
  URL.revokeObjectURL(a.href);
});
