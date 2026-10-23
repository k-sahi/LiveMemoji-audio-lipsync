// predictor.js — run a trained model on live MFCC frames.
// Shared by the training page (live test) and the avatar (Step 6).
//
// Feed it one 13-MFCC frame at a time (30 per second). It keeps the last
// `past + 1 + future` frames, normalises them exactly as in training, and
// returns the predicted jaw/mouth blendshapes for the frame `future` steps ago.
// (That's the price of looking ahead: future × 33 ms of delay.)

export const MODEL_KEY = 'indexeddb://lipsync-model';
export const MODEL_URL = 'model/lipsync-model.json'; // committed copy for the public demo

export class Predictor {
  constructor(tf, model, meta) {
    this.tf = tf;
    this.model = model;
    this.meta = meta;
    this.C = meta.numMfcc;
    this.win = meta.past + 1 + meta.future;
    this.buf = [];                                   // normalised frames, oldest first
    this.input = new Float32Array(this.win * this.C);
    this.smoothed = new Float32Array(meta.targetNames.length);

    // Mic calibration (cepstral mean normalisation), EXPERIMENTAL, off by default.
    // Every mic "colours" the sound, shifting the average MFCCs. This measures
    // the live mic's average and uses it in place of the training average.
    // Tested offline on my recordings: it did NOT rescue a model trained on one
    // mic when used with another (score stayed ~0). Recording training takes
    // with several mics is the real fix. Kept as a toggle to experiment with.
    this.calibrate = false;
    this.liveMean = Float32Array.from(meta.mean);
    this.liveFrames = 0;
  }

  /** 0..1, how much of the first few seconds of calibration is done. */
  get calibration() { return Math.min(1, this.liveFrames / 90); }

  /** Load from the browser (just trained) first, then from the repo's model/ folder. */
  static async load(tf, sources = [MODEL_KEY, MODEL_URL]) {
    for (const src of sources) {
      try {
        const model = src.startsWith('indexeddb://') ? await tf.loadLayersModel(src) : await loadModelFile(tf, src);
        const meta = model.getUserDefinedMetadata?.();
        if (meta && meta.targetNames) return { predictor: new Predictor(tf, model, meta), source: src };
      } catch { /* not there, try the next source */ }
    }
    return null;
  }

  /**
   * @param {Float32Array|number[]} mfcc one frame of 13 MFCCs
   * @param {number} smoothing 0 = raw, 0.5 = gentle, 0.8 = heavy (exponential moving average)
   * @returns {Float32Array|null} blendshape values (order = meta.targetNames), null while warming up
   */
  push(mfcc, smoothing = 0.4) {
    const { std } = this.meta;
    let mean = this.meta.mean;
    if (this.calibrate) {
      // Plain average for the first 3 s, then a slow moving average (~15 s memory)
      // so the estimate settles quickly but doesn't chase every word.
      this.liveFrames++;
      const a = this.liveFrames <= 90 ? 1 / this.liveFrames : 1 / 450;
      for (let c = 0; c < this.C; c++) this.liveMean[c] += a * (mfcc[c] - this.liveMean[c]);
      if (this.liveFrames >= 30) mean = this.liveMean;
    }
    const f = new Float32Array(this.C);
    for (let c = 0; c < this.C; c++) f[c] = (mfcc[c] - mean[c]) / std[c];
    this.buf.push(f);
    if (this.buf.length > this.win) this.buf.shift();
    if (this.buf.length < this.win) return null;

    this.buf.forEach((frame, w) => this.input.set(frame, w * this.C));
    const out = this.tf.tidy(() =>
      this.model.predict(this.tf.tensor2d(this.input, [1, this.input.length])).dataSync());
    for (let k = 0; k < out.length; k++) {
      this.smoothed[k] = smoothing * this.smoothed[k] + (1 - smoothing) * out[k];
    }
    return this.smoothed;
  }

  /** Convenience: value by blendshape name. */
  get(values, name) {
    const i = this.meta.targetNames.indexOf(name);
    return i >= 0 && values ? values[i] : 0;
  }

  reset() {
    this.buf = [];
    this.smoothed.fill(0);
    this.liveMean = Float32Array.from(this.meta.mean);
    this.liveFrames = 0;
  }
}

// ---- single-file model format -------------------------------------------------------
// TensorFlow.js normally saves a model as TWO files (.json + .weights.bin), and
// browsers often block the second download. We pack everything (architecture,
// learned weights as base64, and our metadata) into ONE .json file instead.

/** Pack a trained model into one downloadable JSON Blob. */
export async function exportModelFile(tf, model) {
  let artifacts;
  await model.save(tf.io.withSaveHandler(async (a) => {
    artifacts = a;
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));
  // weightData can be one ArrayBuffer or a list of them, depending on the TF.js version.
  const parts = Array.isArray(artifacts.weightData) ? artifacts.weightData : [artifacts.weightData];
  const bytes = new Uint8Array(parts.reduce((n, b) => n + b.byteLength, 0));
  let offset = 0;
  for (const b of parts) { bytes.set(new Uint8Array(b), offset); offset += b.byteLength; }
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  const file = {
    format: 'lipsync-model-v1',
    modelTopology: artifacts.modelTopology,
    weightSpecs: artifacts.weightSpecs,
    weightDataBase64: btoa(binary),
    userDefinedMetadata: artifacts.userDefinedMetadata ?? model.getUserDefinedMetadata(),
  };
  return new Blob([JSON.stringify(file)], { type: 'application/json' });
}

/** Load a model from a URL: our single-file format, or TF.js's usual two-file format. */
export async function loadModelFile(tf, url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const json = await res.json();
  if (json.format !== 'lipsync-model-v1') return tf.loadLayersModel(url); // two-file format
  const binary = atob(json.weightDataBase64);
  const weightData = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) weightData[i] = binary.charCodeAt(i);
  return tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: json.modelTopology,
    weightSpecs: json.weightSpecs,
    weightData: weightData.buffer,
    userDefinedMetadata: json.userDefinedMetadata,
  }));
}
