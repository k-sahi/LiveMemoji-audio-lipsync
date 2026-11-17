// seeing.js: the Memoji watches YOU through the webcam and reacts like a person would.
//
//   your face (webcam) → MediaPipe face landmarks + 52 blendshapes, ~15 times a second
//     • where you are      → it keeps eye contact, turning its eyes and head toward you
//     • you smile          → it smiles back
//     • brows pulled down  → you look unsure: it tilts its head with a concerned look
//     • brows up, eyes wide→ you look surprised: its brows go up too
//     • you look away      → it waits for you to come back
//
// In talk mode these also become short notes for the AI ("they smiled", "they looked unsure"),
// so replies can respond to your face, not just your words.
// Everything runs on this computer; the video is never sent anywhere.

import { createLandmarker } from './facemodel.js';

const EVERY_MS = 66;                    // ~15 analyses per second is plenty for expressions
const ease = (a, b, k) => a + (b - a) * k;

export function createSeer(video) {
  const s = {
    on: false, starting: false, error: '',
    stream: null, lm: null, lastRun: 0,
    present: false, lastSeen: 0,
    at: { x: 0, y: 0 },                 // where your face is, -1..1 (screen directions)
    smile: 0, unsure: 0, surprise: 0,   // smoothed 0..1
    flags: { smile: false, unsure: false, surprise: false, away: false },
    notes: new Set(),                   // what happened since the last takeNotes()
    tilt: 1,
  };

  async function start() {
    if (s.on || s.starting) return;
    s.starting = true; s.error = '';
    try {
      s.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240, facingMode: 'user' }, audio: false });
      video.srcObject = s.stream;
      await video.play();
      s.lm ??= await createLandmarker('VIDEO');
      s.on = true;
    } catch (err) {
      s.error = err.name === 'NotAllowedError' ? 'Camera permission was blocked.' : 'Camera unavailable: ' + err.message;
      stop();
    }
    s.starting = false;
  }

  function stop() {
    s.on = false;
    s.stream?.getTracks().forEach((t) => t.stop());
    s.stream = null;
    video.srcObject = null;
    s.present = false;
    Object.keys(s.flags).forEach((k) => (s.flags[k] = false));
  }

  // One analysis of the current webcam frame (throttled).
  function tick(now) {
    if (!s.on || video.readyState < 2 || now - s.lastRun < EVERY_MS) return;
    s.lastRun = now;
    let res;
    try { res = s.lm.detectForVideo(video, now); } catch { return; }
    const face = res.faceLandmarks?.[0];
    if (!face) {
      s.present = false;
      if (now - s.lastSeen > 1200) setFlag('away', true, 'looked away');
      s.smile = ease(s.smile, 0, 0.3); s.unsure = ease(s.unsure, 0, 0.3); s.surprise = ease(s.surprise, 0, 0.3);
      setFlag('smile', false); setFlag('unsure', false); setFlag('surprise', false);   // can't read a face that isn't there
      return;
    }
    s.present = true;
    s.lastSeen = now;
    setFlag('away', false);

    const b = Object.fromEntries((res.faceBlendshapes?.[0]?.categories ?? []).map((c) => [c.categoryName, c.score]));
    const avg = (a, c) => ((b[a] ?? 0) + (b[c] ?? 0)) / 2;
    // The webcam image is not mirrored: your right appears on its left. Flip x so -1..1 means screen left..right.
    const nose = face[168];
    s.at.x = ease(s.at.x, Math.max(-1, Math.min(1, (0.5 - nose.x) * 2.5)), 0.4);
    s.at.y = ease(s.at.y, Math.max(-1, Math.min(1, (nose.y - 0.45) * 2.5)), 0.4);

    s.smile = ease(s.smile, avg('mouthSmileLeft', 'mouthSmileRight'), 0.35);
    s.unsure = ease(s.unsure, Math.min(1, avg('browDownLeft', 'browDownRight') * 1.6), 0.3);
    s.surprise = ease(s.surprise, Math.min(1, avg('browOuterUpLeft', 'browOuterUpRight') * 0.8 + avg('eyeWideLeft', 'eyeWideRight') * 0.6), 0.3);

    // Hysteresis: switch on above one level, off below a lower one, so it doesn't flicker.
    setFlag('smile', s.flags.smile ? s.smile > 0.3 : s.smile > 0.45, 'smiled');
    setFlag('unsure', s.flags.unsure ? s.unsure > 0.25 : s.unsure > 0.4, 'looked unsure');
    setFlag('surprise', s.flags.surprise ? s.surprise > 0.3 : s.surprise > 0.45, 'looked surprised');
  }

  function setFlag(name, on, note) {
    if (on && !s.flags[name]) {
      if (note) s.notes.add(note);
      if (name === 'unsure') s.tilt = Math.random() < 0.5 ? -1 : 1;
    }
    s.flags[name] = on;
  }

  /** Body language to add on top of the normal animation, or null when not watching. */
  function pose() {
    if (!s.on) return null;
    const f = s.flags;
    const away = f.away;
    return {
      // Eye contact: look at where you are (unless you've gone; then gaze drifts back to centre).
      gaze: away ? { x: 0, y: 0 } : { x: s.at.x, y: s.at.y },
      yaw: away ? 0 : s.at.x * 0.08,
      pitch: away ? 0 : s.at.y * 0.04,
      roll: f.unsure ? 0.09 * s.tilt : 0,
      brow: Math.max(f.unsure ? 0.45 : 0, f.surprise ? 0.8 : 0),
      smile: f.smile ? 0.35 + 0.4 * s.smile : 0,
      label: away ? '👀 waiting for you to come back'
        : f.surprise ? '😮 you look surprised'
        : f.unsure ? '🤔 you look unsure'
        : f.smile ? '😊 smiling back' : '',
    };
  }

  /** What it saw since the last call, as a short note for the AI (then starts fresh). */
  function takeNotes() {
    const out = [...s.notes];
    s.notes.clear();
    return out;
  }

  return { start, stop, tick, pose, takeNotes, state: s };
}
