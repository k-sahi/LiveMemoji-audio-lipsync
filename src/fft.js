// fft.js — iterative radix-2 Fast Fourier Transform (in place).
// Turns N time samples into N complex frequency bins in O(N log N).
// Steps 1–2 used the browser's AnalyserNode to do this for us. From Step 4 on
// we compute it ourselves, so recording and playback use the exact same math.

export function createFFT(N) {
  const levels = Math.log2(N);
  if (!Number.isInteger(levels)) throw new Error('FFT size must be a power of 2');

  // Bit-reversal permutation table: the classic FFT reorders its input first.
  const rev = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
    rev[i] = r;
  }
  // Precomputed "twiddle factors" e^(-2πik/N).
  const cos = new Float64Array(N / 2);
  const sin = new Float64Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    cos[k] = Math.cos((2 * Math.PI * k) / N);
    sin[k] = -Math.sin((2 * Math.PI * k) / N);
  }

  /** In-place FFT of (re, im), both Float32Array/Float64Array of length N. */
  return function fft(re, im) {
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    // Combine butterflies of size 2, 4, 8, … N.
    for (let size = 2; size <= N; size *= 2) {
      const half = size / 2;
      const step = N / size;
      for (let start = 0; start < N; start += size) {
        for (let k = 0; k < half; k++) {
          const a = start + k;
          const b = a + half;
          const wr = cos[k * step], wi = sin[k * step];
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr;        im[a] += ti;
        }
      }
    }
  };
}
