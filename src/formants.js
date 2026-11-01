// formants.js — estimate the voice's first two resonances (formants) from a frame.
//
// Your mouth and tongue shape a tube; its resonances are the formants:
//   F1 (≈250–900 Hz) rises as the mouth OPENS        ("ee" low, "ah" high)
//   F2 (≈700–2800 Hz) rises as lips SPREAD / tongue moves forward ("oo" low, "ee" high)
// Unlike MFCC-based models, formants come from the shape of the sound itself,
// so they work on any microphone.
//
// Method: Linear Predictive Coding (LPC), the classic speech-science approach.
//  1. Downsample to ~12 kHz (formants we need are all below 4 kHz).
//  2. Pre-emphasis + Hamming window.
//  3. Autocorrelation → Levinson-Durbin → an all-pole filter whose peaks are the formants.
//  4. Evaluate the filter's spectrum and pick the first two peaks.

export function createFormantTracker(sampleRate, { order = 12 } = {}) {
  const dec = Math.max(1, Math.round(sampleRate / 12000));
  const fs = sampleRate / dec;
  const NPTS = 256;               // frequency points from 0 to 4 kHz
  const fMax = Math.min(4000, fs / 2);
  // Precompute cos/sin for evaluating A(e^{jw}) at each frequency point.
  const cosT = [], sinT = [];
  for (let k = 0; k < NPTS; k++) {
    const w = (2 * Math.PI * (k * fMax / NPTS)) / fs;
    cosT.push(Float64Array.from({ length: order + 1 }, (_, i) => Math.cos(w * i)));
    sinT.push(Float64Array.from({ length: order + 1 }, (_, i) => Math.sin(w * i)));
  }

  /** @param {Float32Array} samples one analysis window (e.g. 1024 samples)
   *  @returns {{f1:number, f2:number}|null} null if no clear formants (silence/noise) */
  function analyze(samples) {
    // 0. Voiced or not? Hiss ("s", "f", "sh") crosses zero far more often than a voice
    //    does. Formants only make sense for voiced sounds.
    let zc = 0;
    for (let i = 1; i < samples.length; i++) if ((samples[i] >= 0) !== (samples[i - 1] >= 0)) zc++;
    if (zc / samples.length > 0.12 * (48000 / sampleRate)) return null;
    // 1. Downsample by averaging (a crude but sufficient low-pass).
    const n = Math.floor(samples.length / dec);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < dec; j++) s += samples[i * dec + j];
      x[i] = s / dec;
    }
    // 2. Pre-emphasis (boosts highs so F2 isn't swamped by F1) + Hamming window.
    for (let i = n - 1; i > 0; i--) x[i] -= 0.97 * x[i - 1];
    for (let i = 0; i < n; i++) x[i] *= 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
    // 3. Autocorrelation + Levinson-Durbin.
    const r = new Float64Array(order + 1);
    for (let lag = 0; lag <= order; lag++) {
      let s = 0;
      for (let i = lag; i < n; i++) s += x[i] * x[i - lag];
      r[lag] = s;
    }
    if (r[0] < 1e-9) return null;
    r[0] *= 1 + 1e-9;
    const a = new Float64Array(order + 1); a[0] = 1;
    let err = r[0];
    for (let i = 1; i <= order; i++) {
      let acc = r[i];
      for (let j = 1; j < i; j++) acc += a[j] * r[i - j];
      const k = -acc / err;
      const prev = a.slice();
      for (let j = 1; j < i; j++) a[j] = prev[j] + k * prev[i - j];
      a[i] = k;
      err *= 1 - k * k;
      if (err <= 0) return null;
    }
    // 4. Spectrum of 1/A and its peaks.
    const mag = new Float64Array(NPTS);
    for (let k = 0; k < NPTS; k++) {
      let re = 0, im = 0;
      for (let i = 0; i <= order; i++) { re += a[i] * cosT[k][i]; im -= a[i] * sinT[k][i]; }
      mag[k] = -10 * Math.log10(re * re + im * im + 1e-12);
    }
    const hz = (k) => (k * fMax) / NPTS;
    const peaks = [];
    for (let k = 1; k < NPTS - 1; k++) {
      if (mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] && hz(k) > 180) {
        // Parabolic interpolation for a finer frequency estimate.
        const d = (mag[k - 1] - mag[k + 1]) / (2 * (mag[k - 1] - 2 * mag[k] + mag[k + 1]) || 1);
        peaks.push(hz(k + d));
      }
    }
    const f1 = peaks.find((f) => f >= 200 && f <= 1100);
    if (f1 === undefined) return null;
    const f2 = peaks.find((f) => f > f1 + 250 && f <= 3200);
    if (f2 === undefined) return null;
    return { f1, f2 };
  }

  return { analyze };
}

/** Map formants to mouth shape (0..1 each). Ranges cover typical adult voices. */
export function vowelShape({ f1, f2 }) {
  const c = (v) => Math.min(1, Math.max(0, v));
  return {
    open: c((f1 - 300) / 450),     // "ee"/"oo" ≈ 0 … "ah" ≈ 1
    // Low F2 means rounded lips ("oo"), unless the mouth is wide open ("ah" has a low F2 too).
    round: c((1350 - f2) / 500) * (1 - 0.8 * c((f1 - 300) / 450)),
    spread: c((f2 - 1900) / 600),  // "ee"/"ih" high
  };
}
