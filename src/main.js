// main.js — Step 2: mic → spectrum → mel bands → 13 MFCCs, all visualized live.
import { startMic } from './audio.js';
import { createMFCC } from './mfcc.js';

const $ = (id) => document.getElementById(id);
const btn = $('start');
const stats = $('stats');
const mfccText = $('mfccText');
const wave = $('wave').getContext('2d');
const spec = $('spec').getContext('2d');
const bank = $('bank').getContext('2d');
const melCtx = $('mel').getContext('2d');
const mfccCtx = $('mfcc').getContext('2d');

const SPEC_MAX_HZ = 8000; // x-axis range for spectrum + filterbank plots

btn.addEventListener('click', async () => {
  btn.disabled = true;
  try {
    const { analyser, sampleRate } = await startMic();
    const mfcc = createMFCC({ sampleRate, fftSize: analyser.fftSize });
    drawFilterbank(mfcc);
    btn.textContent = '🎙 Listening';
    loop(analyser, sampleRate, mfcc);
  } catch (err) {
    stats.textContent = 'mic error: ' + err.message;
    btn.disabled = false;
  }
});

function loop(analyser, sampleRate, mfcc) {
  const timeData = new Float32Array(analyser.fftSize);
  const freqData = new Float32Array(analyser.frequencyBinCount);
  const binHz = sampleRate / analyser.fftSize;
  const norm = createRunningNorm(mfcc.numCoeffs);
  let frameNo = 0;

  function frame() {
    analyser.getFloatTimeDomainData(timeData);
    analyser.getFloatFrequencyData(freqData);

    const t0 = performance.now();
    const { mel, mfcc: coeffs } = mfcc.compute(freqData);
    const ms = performance.now() - t0;

    const z = norm.update(coeffs);

    drawWave(timeData);
    drawSpectrum(freqData, binHz);
    scrollColumn(melCtx, mel, melColor);
    scrollColumn(mfccCtx, z, divergingColor);

    if (frameNo++ % 6 === 0) {
      // Update text ~10x/sec so it's readable.
      stats.textContent =
        `sample rate ${sampleRate} Hz | window ${analyser.fftSize} samples | ` +
        `${analyser.frequencyBinCount} FFT bins → ${mfcc.numFilters} mel bands → ` +
        `${mfcc.numCoeffs} MFCCs | compute ${ms.toFixed(2)} ms`;
      mfccText.textContent = Array.from(coeffs, (v, i) =>
        `c${i}`.padStart(3) + ' ' + v.toFixed(1).padStart(7)).join('   ');
    }
    requestAnimationFrame(frame);
  }
  frame();
}

// Running mean / std per coefficient (exponential moving average).
// Turns each MFCC into a z-score: "how unusual is this value right now?"
// This is for display, but the model will need the same idea in Step 5:
// neural nets train badly when inputs have wildly different scales
// (c0 is around -80, while c7 is around ±3).
function createRunningNorm(n, alpha = 0.01) {
  const mean = new Float32Array(n);
  const vari = new Float32Array(n).fill(1);
  const out = new Float32Array(n);
  let first = true;
  return {
    update(x) {
      for (let i = 0; i < n; i++) {
        if (first) mean[i] = x[i];
        const d = x[i] - mean[i];
        mean[i] += alpha * d;
        vari[i] = (1 - alpha) * vari[i] + alpha * d * d;
        out[i] = d / Math.sqrt(vari[i] + 1e-6);
      }
      first = false;
      return out;
    },
  };
}

// ---- drawing helpers -------------------------------------------------------

function drawWave(data) {
  const { width: w, height: h } = wave.canvas;
  wave.clearRect(0, 0, w, h);
  wave.strokeStyle = '#4f8cff';
  wave.lineWidth = 2;
  wave.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / data.length) * w;
    const y = h / 2 - data[i] * (h / 2);
    i ? wave.lineTo(x, y) : wave.moveTo(x, y);
  }
  wave.stroke();
}

function drawSpectrum(data, binHz) {
  const { width: w, height: h } = spec.canvas;
  spec.clearRect(0, 0, w, h);
  const maxBin = Math.min(data.length, Math.floor(SPEC_MAX_HZ / binHz));
  const barW = w / maxBin;
  for (let i = 0; i < maxBin; i++) {
    const v = clamp01((data[i] + 100) / 90);
    spec.fillStyle = `hsl(${200 + v * 120}, 80%, ${30 + v * 40}%)`;
    spec.fillRect(i * barW, h - v * h, barW + 0.5, v * h);
  }
  drawKHzGrid(spec);
}

// Static plot of the 26 triangular mel filters, on the same Hz axis as the spectrum.
function drawFilterbank({ filters, binHz }) {
  const { width: w, height: h } = bank.canvas;
  bank.clearRect(0, 0, w, h);
  const xOf = (hz) => (hz / SPEC_MAX_HZ) * w;
  filters.forEach(({ start, weights }, m) => {
    bank.strokeStyle = m % 2 ? '#f5a524' : '#4f8cff';
    bank.lineWidth = 1.5;
    bank.beginPath();
    for (let j = 0; j < weights.length; j++) {
      const x = xOf((start + j) * binHz);
      const y = h - 4 - weights[j] * (h - 20);
      j ? bank.lineTo(x, y) : bank.moveTo(x, y);
    }
    bank.stroke();
  });
  drawKHzGrid(bank);
}

function drawKHzGrid(ctx) {
  const { width: w, height: h } = ctx.canvas;
  ctx.fillStyle = '#666';
  ctx.font = '11px system-ui';
  for (let k = 1; k < SPEC_MAX_HZ / 1000; k++) {
    const x = (k * 1000 / SPEC_MAX_HZ) * w;
    ctx.fillRect(x, 0, 1, h);
    ctx.fillText(`${k}k`, x + 3, 12);
  }
}

// Scrolling heatmap: shift the image left, paint the newest frame as a column
// on the right. Row 0 (lowest band / c0) is at the bottom.
function scrollColumn(ctx, values, colorFn) {
  const { width: w, height: h } = ctx.canvas;
  const step = 2;
  ctx.drawImage(ctx.canvas, -step, 0);
  const rowH = h / values.length;
  for (let i = 0; i < values.length; i++) {
    ctx.fillStyle = colorFn(values[i]);
    ctx.fillRect(w - step, h - (i + 1) * rowH, step, Math.ceil(rowH));
  }
}

// Log mel energy: roughly -25 (silence) … -5 (loud).
function melColor(v) {
  const t = clamp01((v + 25) / 20);
  return `hsl(${260 - t * 220}, 85%, ${8 + t * 52}%)`;
}

// z-score: blue = below its average, red = above, black = typical.
function divergingColor(z) {
  const t = clamp01(Math.abs(z) / 2.5);
  return `hsl(${z > 0 ? 8 : 212}, 85%, ${6 + t * 50}%)`;
}

function clamp01(x) { return Math.max(0, Math.min(1, x)); }
