// latency.js: measure how long each stage of the voice → face pipeline takes, live.
//
// Every audio frame is timed through the pipeline:
//
//   audio in      sound reaches the mic → the frame arrives in JavaScript
//                 (half the 1024-sample window + browser audio buffering + the worklet's message)
//   features      Hann window + FFT + mel filterbank + DCT → 13 MFCCs
//   formants      LPC (Levinson-Durbin) → F1, F2
//   model         the neural network's prediction (TensorFlow.js)
//   look-ahead    frames the model waits for to see a little "future" audio (if trained that way)
//   wait          until the next screen refresh picks up the new mouth shape
//   draw          moving the face + rendering it with Three.js (WebGL)
//
// It keeps the last few seconds of each and reports p50 (typical) and p95 (worst 5%).
// Not included: the microphone hardware/driver and the display itself, which the browser can't see.

export function createLatencyMeter({ keep = 300 } = {}) {
  const series = new Map();

  function record(name, ms) {
    if (!Number.isFinite(ms) || ms < 0) return;
    let s = series.get(name);
    if (!s) series.set(name, (s = []));
    s.push(ms);
    if (s.length > keep) s.shift();
  }

  function pct(arr, p) {
    if (!arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
  }

  function stats(name) {
    const s = series.get(name) ?? [];
    return { p50: pct(s, 0.5), p95: pct(s, 0.95), n: s.length };
  }

  return { record, stats, reset: () => series.clear() };
}

export const STAGES = [
  ['audio in', 'half the 21 ms window + browser audio buffering (+ mic latency if reported)'],
  ['features', 'FFT + mel filterbank + MFCC'],
  ['formants', 'LPC vowel tracking'],
  ['model', 'neural network (TensorFlow.js)'],
  ['look-ahead', 'model waits for future frames'],
  ['wait', 'until the next screen refresh'],
  ['draw', 'animate + render (Three.js / WebGL)'],
];
