// dataset.js — turn recorded takes into (input, target) training pairs.
//
// INPUT  for audio frame i: MFCCs of frames i-PAST … i+FUTURE (a 9-frame window
//        = 300 ms of sound), each coefficient normalised to mean 0, std 1.
//        9 × 13 = 117 numbers.
// TARGET for frame i: the jaw + mouth blendshapes the camera saw at frame i+lag.
//        `lag` corrects the mic-vs-camera delay measured when recording.
//
// Why a window, not one frame? A single 21 ms snapshot can't tell "p" from
// silence. The frames around it show the burst of sound, the transition, the vowel.
// Why only 2 future frames? At live time, future frames haven't happened yet.
// Each future frame we need = 33 ms of delay before the avatar can react.
//
// Validation: the LAST 15% of every take is held out. We never test on random
// frames, because neighbouring frames are near-copies of each other: a model
// could "memorise" its way to a great score without learning anything.

export const DEFAULTS = { past: 6, future: 2, valFrac: 0.15, maxLag: 10 };

export const isMouthShape = (name) => name.startsWith('jaw') || name.startsWith('mouth');

/** Check one parsed JSON file. Returns { take } or { error }. */
export function validateTake(json, fileName) {
  if (!json || !Array.isArray(json.frames)) return { error: `${fileName}: not a recording file` };
  if (json.format !== 'lipsync-take-v2') {
    return { error: `${fileName}: made with the old recorder (format ${json.format ?? 'unknown'}). Re-record it with the current record.html.` };
  }
  if (Math.abs(json.fps - 30) > 0.5) return { error: `${fileName}: expected 30 fps, got ${json.fps}` };
  if (json.numMfcc !== 13) return { error: `${fileName}: expected 13 MFCCs, got ${json.numMfcc}` };
  if (!Array.isArray(json.blendshapeNames) || json.blendshapeNames.length !== 52) {
    return { error: `${fileName}: missing blendshape names (was a face ever detected?)` };
  }
  const labelled = json.frames.filter((f) => f.b).length;
  if (labelled < 100) return { error: `${fileName}: only ${labelled} frames with a face` };
  return { take: { ...json, fileName } };
}

/** Build normalised train / validation arrays from validated takes. */
export function buildDataset(takes, opts = {}) {
  const { past, future, valFrac, maxLag } = { ...DEFAULTS, ...opts };
  const win = past + 1 + future;
  const names = takes[0].blendshapeNames;
  for (const t of takes) {
    if (t.blendshapeNames.join() !== names.join()) throw new Error(`${t.fileName}: different blendshape list`);
  }
  const targetIdx = names.map((n, i) => (isMouthShape(n) ? i : -1)).filter((i) => i >= 0);
  const targetNames = targetIdx.map((i) => names[i]);
  const C = takes[0].numMfcc;

  // 1. Normalisation statistics, from TRAINING frames only (no peeking at validation).
  const sum = new Float64Array(C), sumSq = new Float64Array(C);
  let count = 0;
  for (const t of takes) {
    const cut = Math.floor(t.frames.length * (1 - valFrac));
    for (let i = 0; i < cut; i++) {
      const m = t.frames[i].m;
      for (let c = 0; c < C; c++) { sum[c] += m[c]; sumSq[c] += m[c] * m[c]; }
      count++;
    }
  }
  const mean = Array.from(sum, (s) => s / count);
  const std = Array.from(sumSq, (s, c) => Math.sqrt(Math.max(s / count - mean[c] ** 2, 1e-6)));

  // 2. Windows + targets.
  const tr = { x: [], y: [] }, va = { x: [], y: [] };
  const perTake = [];
  for (const t of takes) {
    const F = t.frames, n = F.length;
    const cut = Math.floor(n * (1 - valFrac));
    const lag = Math.max(-maxLag, Math.min(maxLag, Math.round(t.estimatedLagFrames ?? 0)));
    const norm = F.map((f) => f.m.map((v, c) => (v - mean[c]) / std[c]));
    let nTr = 0, nVa = 0;
    for (let i = past; i < n - future; i++) {
      const j = i + lag;
      if (j < 0 || j >= n || !F[j].b) continue;
      const lo = Math.min(i - past, j), hi = Math.max(i + future, j);
      // A sample must sit entirely on one side of the cut, or it would leak.
      const set = hi < cut ? tr : lo >= cut ? va : null;
      if (!set) continue;
      const x = new Array(win * C);
      for (let w = 0; w < win; w++) for (let c = 0; c < C; c++) x[w * C + c] = norm[i - past + w][c];
      set.x.push(x);
      set.y.push(targetIdx.map((k) => F[j].b[k]));
      set === tr ? nTr++ : nVa++;
    }
    perTake.push({ fileName: t.fileName, frames: n, lag, train: nTr, val: nVa });
  }

  const flat = (rows, width) => {
    const out = new Float32Array(rows.length * width);
    rows.forEach((r, i) => out.set(r, i * width));
    return out;
  };
  const inputSize = win * C;
  return {
    xTrain: flat(tr.x, inputSize), yTrain: flat(tr.y, targetNames.length), nTrain: tr.x.length,
    xVal: flat(va.x, inputSize), yVal: flat(va.y, targetNames.length), nVal: va.x.length,
    inputSize, targetNames, mean, std, past, future, numMfcc: C, perTake,
  };
}

export function pearson(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return saa > 1e-12 && sbb > 1e-12 ? sab / Math.sqrt(saa * sbb) : NaN;
}
