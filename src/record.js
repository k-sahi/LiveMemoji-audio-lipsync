// record.js — Step 4: record synced (audio features, face blendshapes) pairs.
//
// Two independent streams, matched by timestamp:
//   AUDIO  An AudioWorklet on the audio thread cuts the mic into windows at
//          exactly 30 per second (see audio-worklet.js). Each window becomes
//          13 MFCCs, stamped with the real time it was heard.
//   FACE   Every camera frame, MediaPipe gives 52 blendshapes, stamped with
//          the time it was seen. We keep the last ~2 seconds in a history.
//
// Each audio frame is paired with the face result closest in time. Audio frames
// wait in a short queue (MATCH_DELAY_MS) so the face results around them exist.
//
// Why is audio the clock? At playback (Step 6) there's no camera, only the mic,
// so training data must be sampled exactly like live audio will be: 30 Hz, regular.
//
// Output JSON: one frame = { t: ms, m: [13 MFCCs], b: [52 blendshapes] | null, r: rms }.
// b is null when no face was close enough in time; Step 5 skips those as
// labels but still uses their audio as context for neighbouring frames.

import { startMicFrames } from './audio.js';
import { createFeatureExtractor } from './features.js';
import { FaceLandmarker, DrawingUtils, createLandmarker } from './facemodel.js';

const FPS = 30;
const FACE_MATCH_MS = 60;    // nearest face must be within this many ms of the audio frame
const MATCH_DELAY_MS = 150;  // how long audio frames wait for face results to catch up
const MAX_LAG_FRAMES = 30;   // delay search range for the quality report (±1 s)

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const startBtn = $('startBtn');
const recBtn = $('recBtn');
const dlBtn = $('dlBtn');
const clearBtn = $('clearBtn');
const status = $('status');
const micInfo = $('micInfo');
const levelFill = $('levelFill');
const jawFill = $('jawFill');
const faceDot = $('faceDot');
const timer = $('timer');
const summary = $('summary');

// ---- recording scripts -------------------------------------------------------
// Each take covers different sounds. Variety matters more than length:
// the model can only learn mouth shapes it has heard.
const TAKES = [
  {
    title: 'Take 1: vowels + read sentences',
    html: `
      <p><i>Hold each for ~2 seconds, mouth clearly shaped:</i><br>
      <b>aah … eee … ooo … ehh … ihh … uhh</b></p>
      <p><i>Slide smoothly between them:</i><br><b>ah-ee-oo … oo-ee-ah … ah-ee-oo</b></p>
      <p class="pause">[ silent, mouth closed, 4 seconds ]</p>
      <p>The source of the huge river is the clear spring. Help the woman get back to her feet.
      A pot of tea helps to pass the evening. Smoky fires lack flame and heat.
      The soft cushion broke the man's fall. The salt breeze came across from the sea.
      The girl at the booth sold fifty bonds.</p>
      <p class="pause">[ silent, 4 seconds ]</p>`,
  },
  {
    title: 'Take 2: consonants + tongue-twisters',
    html: `
      <p><i>Lips and teeth sounds, ~1 second each:</i><br>
      <b>mmm … nnn … fff … vvv … sss … zzz … shh … thh</b></p>
      <p><b>ma pa ba · wa fa va · la ta da · ka ga ya</b> (twice, then faster)</p>
      <p class="pause">[ silent, 4 seconds ]</p>
      <p>Betty Botter bought some butter, but she said the butter's bitter.
      Toy boat, toy boat, toy boat. Fresh fried fish, fish fresh fried.
      Which wristwatches are Swiss wristwatches? A big black bug bit a big black bear.
      Lovely lemon liniment. Peppered pumpkin pie for Polly.</p>
      <p class="pause">[ silent, 4 seconds ]</p>`,
  },
  {
    title: 'Take 3: talk naturally',
    html: `
      <p><i>No script. Talk for about 60–90 seconds like you're on the phone with a friend.</i></p>
      <p>Ideas: what you ate today · a show or game you're into · why you're building this project ·
      a place you'd love to travel.</p>
      <p>Mix it up: ask a question, laugh, say something quietly, say something with energy,
      pause to think ("hmm…").</p>
      <p class="pause">[ end with 4 seconds of silence ]</p>`,
  },
];

let takeIndex = 0;
function renderTakes() {
  $('takeTabs').innerHTML = TAKES.map((t, i) =>
    `<button class="tab${i === takeIndex ? ' active' : ''}" data-i="${i}">${i + 1}</button>`).join('');
  $('takeTitle').textContent = TAKES[takeIndex].title;
  $('script').innerHTML = TAKES[takeIndex].html;
}
$('takeTabs').addEventListener('click', (e) => {
  if (e.target.dataset.i == null || recording) return;
  takeIndex = Number(e.target.dataset.i);
  renderTakes();
});
renderTakes();

// ---- shared state ------------------------------------------------------------
let mic = null;
let features = null;
let meta = null;              // recording settings saved into the JSON
let blendshapeNames = null;
const faceHistory = [];       // [{ time, bs: Float32Array(52) }], newest last
const pending = [];           // audio frames waiting to be matched with a face
let recording = false;
let recStartFrame = null;     // audio sample index where this recording began
let frames = [];
let lastReport = null;
const live = { rms: 0, faceTime: 0, jaw: 0 };

// ---- start: load model, open mic + camera --------------------------------------
startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  try {
    status.textContent = 'loading face model…';
    const landmarker = await createLandmarker('VIDEO');

    status.textContent = 'requesting microphone + camera…';
    mic = await startMicFrames({ fps: FPS, windowSize: 1024, onFrame: onAudioFrame });
    features = createFeatureExtractor({ sampleRate: mic.sampleRate, fftSize: 1024 });
    const cam = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
    });
    video.srcObject = cam;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    meta = {
      fps: mic.fps,
      sampleRate: mic.sampleRate,
      hop: mic.hop,
      fftSize: features.fftSize,
      numMfcc: features.numCoeffs,
      numMelFilters: features.numFilters,
      window: 'hann',
      mic: mic.micLabel,
    };

    micInfo.innerHTML = mic.micLooksLaggy
      ? `⚠️ mic: <b>${mic.micLabel}</b>. Bluetooth mics add 150–400 ms of delay. Use your laptop's built-in mic or a wired one if you can (change it in Windows sound settings, then reload).`
      : `🎙 mic: ${mic.micLabel}`;

    runFaceLoop(landmarker);
    runUiLoop();

    startBtn.textContent = '✅ Ready';
    recBtn.disabled = false;
    status.textContent = 'ready. Check the face dot is green and the level meter moves when you talk.';
  } catch (err) {
    status.textContent = 'error: ' + err.message;
    startBtn.disabled = false;
  }
});

// ---- FACE loop (main thread, every camera frame) ---------------------------------
function runFaceLoop(landmarker) {
  const ctx = overlay.getContext('2d');
  const draw = new DrawingUtils(ctx);
  let lastVideoTime = -1;

  function frame() {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      const res = landmarker.detectForVideo(video, now);
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      if (res.faceLandmarks.length) {
        const cats = res.faceBlendshapes[0].categories;
        blendshapeNames ??= cats.map((c) => c.categoryName);
        const bs = Float32Array.from(cats, (c) => c.score);
        faceHistory.push({ time: now, bs });
        while (faceHistory.length && faceHistory[0].time < now - 2000) faceHistory.shift();
        live.faceTime = now;
        live.jaw = bs[blendshapeNames.indexOf('jawOpen')];
        const lm = res.faceLandmarks[0];
        draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#8ab4ff80', lineWidth: 1 });
        draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#ff6a5a', lineWidth: 2 });
      }
    }
    requestAnimationFrame(frame);
  }
  frame();
}

// ---- AUDIO frames (arrive from the audio thread, exactly 30 per second) ----------
function onAudioFrame(samples, endFrame) {
  const { mfcc, rms } = features.compute(samples);
  live.rms = rms;

  if (recording) {
    recStartFrame ??= endFrame;
    pending.push({
      endFrame,
      time: mic.audioFrameToPerfTime(endFrame),
      m: Array.from(mfcc, (v) => round(v, 2)),   // copy! compute() reuses its buffer
      r: round(rms, 4),
    });
  }
  matchPending(false);
}

// Pair queued audio frames with the closest face result in time.
function matchPending(flushAll) {
  const now = performance.now();
  while (pending.length && (flushAll || pending[0].time < now - MATCH_DELAY_MS)) {
    const a = pending.shift();
    let best = null;
    let bestDt = Infinity;
    for (const f of faceHistory) {
      const dt = Math.abs(f.time - a.time);
      if (dt < bestDt) { bestDt = dt; best = f; }
    }
    frames.push({
      t: Math.round(((a.endFrame - recStartFrame) / mic.sampleRate) * 1000),
      m: a.m,
      b: best && bestDt <= FACE_MATCH_MS ? Array.from(best.bs, (v) => round(v, 3)) : null,
      r: a.r,
    });
  }
}

// ---- UI loop -------------------------------------------------------------------
function runUiLoop() {
  function frame() {
    const faceOk = performance.now() - live.faceTime < 150;
    levelFill.style.width = Math.min(100, Math.sqrt(live.rms) * 250) + '%';
    jawFill.style.width = (live.jaw * 100).toFixed(0) + '%';
    faceDot.className = 'dot ' + (faceOk ? 'ok' : 'bad');
    if (recording) {
      const s = frames.length / FPS;
      const noFace = frames.filter((f) => !f.b).length;
      timer.textContent = `⏺ ${fmtTime(s)} | ${frames.length} frames` + (noFace ? ` | ${noFace} without face` : '');
    }
    requestAnimationFrame(frame);
  }
  frame();
}

// ---- record / stop / download / clear ------------------------------------------------
recBtn.addEventListener('click', () => {
  if (!recording) {
    if (frames.length && !confirm('Discard the current recording and start a new one?')) return;
    frames = [];
    pending.length = 0;
    recStartFrame = null;
    lastReport = null;
    summary.innerHTML = '';
    recording = true;
    recBtn.textContent = '⏹ Stop';
    recBtn.classList.add('recording');
    dlBtn.disabled = clearBtn.disabled = true;
  } else {
    recording = false;
    matchPending(true);
    recBtn.textContent = '⏺ Record';
    recBtn.classList.remove('recording');
    dlBtn.disabled = clearBtn.disabled = frames.length === 0;
    timer.textContent = `stopped | ${frames.length} frames (${fmtTime(frames.length / FPS)})`;
    showSummary();
  }
});

dlBtn.addEventListener('click', () => {
  const data = {
    format: 'lipsync-take-v2',
    take: takeIndex + 1,
    createdAt: new Date().toISOString(),
    ...meta,
    estimatedLagFrames: lastReport?.lag ?? 0,
    blendshapeNames,
    frames,
  };
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `take${takeIndex + 1}-${stamp}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

clearBtn.addEventListener('click', () => {
  if (!confirm('Clear this recording?')) return;
  frames = [];
  summary.innerHTML = '';
  timer.textContent = 'cleared';
  dlBtn.disabled = clearBtn.disabled = true;
});

// ---- quality report -----------------------------------------------------------
// A quick sanity check that the audio and face data actually relate to each other.
function showSummary() {
  if (!frames.length) return;
  const labelled = frames.filter((f) => f.b);
  const faceFrac = labelled.length / frames.length;

  // "Voiced" = noticeably louder than the quietest 10% of the recording.
  const rmsSorted = frames.map((f) => f.r).sort((a, b) => a - b);
  const floor = rmsSorted[Math.floor(rmsSorted.length * 0.1)];
  const voicedFrac = frames.filter((f) => f.r > floor * 3).length / frames.length;

  // Correlate loudness (log RMS) with jawOpen at different time shifts.
  // The best shift estimates the delay between the mic and the camera.
  // Negative = the face moved BEFORE the mic heard it (mic is late).
  const jawIdx = blendshapeNames ? blendshapeNames.indexOf('jawOpen') : -1;
  let best = { lag: 0, r: 0 };
  if (jawIdx >= 0) {
    const loud = frames.map((f) => Math.log(f.r + 1e-6));
    best = { lag: 0, r: -Infinity };
    for (let lag = -MAX_LAG_FRAMES; lag <= MAX_LAG_FRAMES; lag++) {
      const xs = [], ys = [];
      for (let i = 0; i < frames.length; i++) {
        const j = i + lag;
        if (j < 0 || j >= frames.length || !frames[j].b) continue;
        xs.push(loud[i]);
        ys.push(frames[j].b[jawIdx]);
      }
      const r = pearson(xs, ys);
      if (r > best.r) best = { lag, r };
    }
  }
  lastReport = best;
  const lagMs = Math.round((best.lag * 1000) / FPS);

  const ok = (cond) => (cond ? '✅' : '⚠️');
  summary.innerHTML = `
    <li>${ok(frames.length >= 1800)} ${frames.length} frames (${fmtTime(frames.length / FPS)})${frames.length < 1800 ? ' (aim for 60–90 s per take)' : ''}</li>
    <li>${ok(faceFrac > 0.95)} face matched in ${(faceFrac * 100).toFixed(0)}% of frames${faceFrac <= 0.95 ? ' (keep your mouth in view, improve lighting)' : ''}</li>
    <li>${ok(voicedFrac > 0.4 && voicedFrac < 0.9)} speaking in ~${(voicedFrac * 100).toFixed(0)}% of frames (some silence is good, the model needs to learn "mouth closed")</li>
    <li>${ok(best.r > 0.3)} loudness ↔ jawOpen correlation <b>${best.r.toFixed(2)}</b>${best.r <= 0.3 ? ' (low: speak a bit louder / open your mouth naturally)' : ''}</li>
    <li>${ok(Math.abs(lagMs) <= 200)} mic vs camera delay ≈ <b>${lagMs} ms</b>${Math.abs(lagMs) > 200 ? ' (large: Bluetooth mic? Step 5 will correct it, but a wired/built-in mic is better)' : ' (Step 5 corrects this automatically)'}</li>`;
}

// ---- helpers --------------------------------------------------------------------
function pearson(xs, ys) {
  const n = xs.length;
  if (n < 10) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy || 1);
}
function round(v, d) { const k = 10 ** d; return Math.round(v * k) / k; }
function fmtTime(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; }
