// features.js — raw audio window → 13 MFCCs. The ONE feature pipeline used for
// recording (Step 4), training (Step 5) and the live avatar (Step 6).
//
//   samples (1024) → Hann window → FFT → power spectrum (dB) → mfcc.js → 13 numbers

import { createFFT } from './fft.js';
import { createMFCC } from './mfcc.js';

export function createFeatureExtractor({ sampleRate, fftSize = 1024 }) {
  const N = fftSize;
  const fft = createFFT(N);
  const mfcc = createMFCC({ sampleRate, fftSize: N });

  // Hann window: fades the edges of the window to zero so the FFT doesn't see
  // a fake "click" where the snippet was cut out of the stream.
  const hann = new Float32Array(N);
  for (let n = 0; n < N; n++) hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));

  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const db = new Float32Array(N / 2);

  /** @param {Float32Array} samples exactly fftSize samples, oldest first */
  function compute(samples) {
    let sumSq = 0;
    for (let n = 0; n < N; n++) {
      re[n] = samples[n] * hann[n];
      im[n] = 0;
      sumSq += samples[n] * samples[n];
    }
    fft(re, im);
    for (let k = 0; k < N / 2; k++) {
      const power = (re[k] * re[k] + im[k] * im[k]) / (N * N);
      db[k] = 10 * Math.log10(power + 1e-20);
    }
    const { mfcc: coeffs } = mfcc.compute(db);
    return { mfcc: coeffs, rms: Math.sqrt(sumSq / N), db };
  }

  return { compute, numCoeffs: mfcc.numCoeffs, numFilters: mfcc.numFilters, fftSize: N, sampleRate };
}
