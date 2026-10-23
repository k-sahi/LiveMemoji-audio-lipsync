// mfcc.js — Mel-Frequency Cepstral Coefficients.
//
// Goal: squash a 512-bin spectrum into 13 numbers that describe the SHAPE
// of the spectrum (where the formants are = mouth shape) and mostly ignore
// pitch (which note you're singing) and fine noise.
//
// Pipeline per frame:
//   1. dB spectrum  → linear power             (undo the log the analyser applied)
//   2. power        → 26 mel-band energies     (triangular filters, spaced like human hearing)
//   3. energies     → log energies             (loudness perception is logarithmic)
//   4. log energies → DCT → keep first 13      (decorrelate + keep only the smooth shape)
//
// c0 ≈ overall loudness. c1 ≈ spectral tilt (bright vs dull). Higher
// coefficients describe finer bumps in the spectrum's envelope (formants).

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

export function createMFCC({
  sampleRate,
  fftSize,
  numFilters = 26,
  numCoeffs = 13,
  minHz = 80,     // below this is mostly hum / rumble, not speech
  maxHz = 7600,   // speech energy above this is small
} = {}) {
  const nBins = fftSize / 2;
  const binHz = sampleRate / fftSize;
  maxHz = Math.min(maxHz, sampleRate / 2);

  // ---- 1. Mel filterbank --------------------------------------------------
  // numFilters + 2 points, evenly spaced on the MEL scale (so they bunch up at
  // low Hz, where our ears, and the formants, need more resolution).
  const melMin = hzToMel(minHz);
  const melMax = hzToMel(maxHz);
  const edgesHz = [];
  for (let i = 0; i < numFilters + 2; i++) {
    edgesHz.push(melToHz(melMin + ((melMax - melMin) * i) / (numFilters + 1)));
  }

  // Each filter is a triangle: rises from edge[m-1] to a peak at edge[m],
  // falls to zero at edge[m+1]. Stored sparsely: first bin + weights.
  const filters = [];
  for (let m = 1; m <= numFilters; m++) {
    const [left, center, right] = [edgesHz[m - 1], edgesHz[m], edgesHz[m + 1]];
    const start = Math.max(0, Math.floor(left / binHz));
    const end = Math.min(nBins - 1, Math.ceil(right / binHz));
    const weights = new Float32Array(end - start + 1);
    for (let k = start; k <= end; k++) {
      const f = k * binHz;
      if (f > left && f <= center) weights[k - start] = (f - left) / (center - left);
      else if (f > center && f < right) weights[k - start] = (right - f) / (right - center);
    }
    filters.push({ start, weights, centerHz: center });
  }

  // ---- 2. DCT-II matrix (orthonormal) -------------------------------------
  // Row n is a cosine with n half-periods across the 26 bands. Projecting the
  // log-energies onto it measures "how much of that wiggle" the envelope has.
  const dct = [];
  for (let n = 0; n < numCoeffs; n++) {
    const row = new Float32Array(numFilters);
    const scale = Math.sqrt((n === 0 ? 1 : 2) / numFilters);
    for (let m = 0; m < numFilters; m++) {
      row[m] = scale * Math.cos((Math.PI * n * (m + 0.5)) / numFilters);
    }
    dct.push(row);
  }

  // Reused buffers. compute() OVERWRITES these every call, so copy them
  // if you need to keep a frame (we will, when recording in Step 4).
  const power = new Float32Array(nBins);
  const mel = new Float32Array(numFilters);
  const mfcc = new Float32Array(numCoeffs);

  /** @param {Float32Array} dbSpectrum from AnalyserNode.getFloatFrequencyData */
  function compute(dbSpectrum) {
    // 1. dB → linear power. (-Infinity dB becomes 0, which is fine.)
    for (let k = 0; k < nBins; k++) power[k] = 10 ** (dbSpectrum[k] / 10);

    // 2 + 3. Weighted sum under each triangle, then natural log.
    // The tiny epsilon stops log(0) = -Infinity during total silence.
    for (let m = 0; m < numFilters; m++) {
      const { start, weights } = filters[m];
      let e = 0;
      for (let j = 0; j < weights.length; j++) e += power[start + j] * weights[j];
      mel[m] = Math.log(e + 1e-10);
    }

    // 4. DCT, keeping only the first numCoeffs outputs.
    for (let n = 0; n < numCoeffs; n++) {
      const row = dct[n];
      let s = 0;
      for (let m = 0; m < numFilters; m++) s += row[m] * mel[m];
      mfcc[n] = s;
    }
    return { mel, mfcc };
  }

  return { compute, filters, numFilters, numCoeffs, binHz, minHz, maxHz };
}
