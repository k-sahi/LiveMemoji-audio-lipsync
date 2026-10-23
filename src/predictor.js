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
        const model = await tf.loadLayersModel(src);
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
