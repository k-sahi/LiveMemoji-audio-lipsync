// train.js — Step 5: train a small neural network in the browser with TensorFlow.js.
//
//   117 inputs (9 frames × 13 MFCCs)
//     → GaussianNoise 0.1   (training only: jitters inputs so the net can't memorise exact numbers)
//     → Dense 128, ReLU → Dropout 0.2
//     → Dense 64,  ReLU → Dropout 0.2
//     → Dense 27,  sigmoid  (one output per jaw/mouth blendshape, squashed into 0..1)
//
// Loss: mean squared error between predicted and camera-measured blendshapes.
// We keep the weights from the epoch with the LOWEST VALIDATION loss: after
// that point the network starts memorising the training takes instead of
// learning patterns that transfer to new speech.

import { validateTake, buildDataset, pearson } from './dataset.js';
import { Predictor, MODEL_KEY } from './predictor.js';
import { startMicFrames } from './audio.js';
import { createFeatureExtractor } from './features.js';

const tf = window.tf;
const $ = (id) => document.getElementById(id);

const EPOCHS = 60;
const BATCH = 128;
const SHOW = ['jawOpen', 'mouthFunnel', 'mouthPucker', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthLowerDownLeft', 'mouthUpperUpLeft', 'mouthStretchLeft', 'mouthClose', 'mouthRollLower'];

let takes = [];
let predictor = null;

// ---- 1. load files ------------------------------------------------------------
const drop = $('drop');
$('fileInput').addEventListener('change', (e) => addFiles(e.target.files));
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); });
$('clearFiles').addEventListener('click', () => { takes = []; renderFiles([]); });

async function addFiles(fileList) {
  const messages = [];
  for (const file of fileList) {
    try {
      const { take, error } = validateTake(JSON.parse(await file.text()), file.name);
      if (error) messages.push(`❌ ${error}`);
      else if (takes.some((t) => t.createdAt === take.createdAt)) messages.push(`➖ ${file.name}: already added`);
      else takes.push(take);
    } catch (err) {
      messages.push(`❌ ${file.name}: ${err.message}`);
    }
  }
  renderFiles(messages);
}

function renderFiles(messages) {
  const total = takes.reduce((s, t) => s + t.frames.length, 0);
  $('fileList').innerHTML =
    takes.map((t) => `<li>✅ ${t.fileName}: take ${t.take}, ${(t.frames.length / 30).toFixed(0)} s, ` +
      `mic delay ${Math.round((t.estimatedLagFrames ?? 0) * 1000 / 30)} ms</li>`).join('') +
    messages.map((m) => `<li>${m}</li>`).join('');
  const mins = total / 30 / 60;
  $('dataSummary').textContent = takes.length
    ? `${takes.length} take(s), ${total} frames = ${mins.toFixed(1)} min of speech ` +
      (mins < 3 ? '(works, but 5+ min gives a much better avatar)' : mins < 6 ? '(good)' : '(great)')
    : 'no takes loaded';
  $('trainBtn').disabled = takes.length === 0;
}

// ---- 2. train -----------------------------------------------------------------------
$('trainBtn').addEventListener('click', train);

async function train() {
  $('trainBtn').disabled = true;
  const log = $('trainStatus');
  try {
    log.textContent = 'building dataset…';
    await tf.nextFrame();
    const ds = buildDataset(takes);
    if (ds.nVal < 50) throw new Error('not enough validation frames, record a bit longer');

    const xTrain = tf.tensor2d(ds.xTrain, [ds.nTrain, ds.inputSize]);
    const yTrain = tf.tensor2d(ds.yTrain, [ds.nTrain, ds.targetNames.length]);
    const xVal = tf.tensor2d(ds.xVal, [ds.nVal, ds.inputSize]);
    const yVal = tf.tensor2d(ds.yVal, [ds.nVal, ds.targetNames.length]);

    // Baseline: always predicting the training average. Our model must beat this.
    const baseline = tf.tidy(() => yVal.sub(yTrain.mean(0)).square().mean().dataSync()[0]);

    const model = tf.sequential();
    model.add(tf.layers.gaussianNoise({ stddev: 0.1, inputShape: [ds.inputSize] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: ds.targetNames.length, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });
    $('modelSummary').textContent =
      `${ds.inputSize} → 128 → 64 → ${ds.targetNames.length} | ${model.countParams().toLocaleString()} weights | ` +
      `${ds.nTrain} training / ${ds.nVal} validation examples | baseline loss ${baseline.toFixed(5)}`;

    const history = { loss: [], val: [] };
    let best = { val: Infinity, epoch: -1, weights: null };

    await model.fit(xTrain, yTrain, {
      epochs: EPOCHS,
      batchSize: BATCH,
      shuffle: true,
      validationData: [xVal, yVal],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          history.loss.push(logs.loss);
          history.val.push(logs.val_loss);
          if (logs.val_loss < best.val) {
            best.weights?.forEach((w) => w.dispose());
            best = { val: logs.val_loss, epoch, weights: model.getWeights().map((w) => w.clone()) };
          }
          log.textContent = `epoch ${epoch + 1}/${EPOCHS} | train loss ${logs.loss.toFixed(5)} | ` +
            `validation loss ${logs.val_loss.toFixed(5)} | best so far: epoch ${best.epoch + 1}`;
          drawLossChart(history, baseline, best.epoch);
          await tf.nextFrame();
        },
      },
    });

    model.setWeights(best.weights);
    best.weights.forEach((w) => w.dispose());

    // ---- evaluate on validation data ----
    const pred = tf.tidy(() => model.predict(xVal).arraySync());
    const truth = yVal.arraySync();
    const K = ds.targetNames.length;
    const corr = ds.targetNames.map((name, k) => ({
      name,
      r: pearson(pred.map((p) => p[k]), truth.map((t) => t[k])),
      spread: Math.sqrt(truth.reduce((s, t) => s + t[k] * t[k], 0) / truth.length),
    }));
    const r2 = 1 - best.val / baseline;
    showEvaluation(corr.filter((c) => SHOW.includes(c.name)), r2, best.epoch);
    [xTrain, yTrain, xVal, yVal].forEach((t) => t.dispose());

    // ---- save: the settings needed to use the model travel INSIDE the model file ----
    const meta = {
      version: 1,
      trainedAt: new Date().toISOString(),
      fps: 30,
      sampleRate: takes[0].sampleRate,
      mic: takes[0].mic,
      fftSize: takes[0].fftSize,
      numMfcc: ds.numMfcc,
      past: ds.past,
      future: ds.future,
      mean: ds.mean,
      std: ds.std,
      targetNames: ds.targetNames,
      trainFrames: ds.nTrain,
      valR2: Number(r2.toFixed(3)),
      valCorr: Object.fromEntries(corr.map((c) => [c.name, Number((c.r || 0).toFixed(3))])),
    };
    model.setUserDefinedMetadata(meta);
    await model.save(MODEL_KEY);
    predictor = new Predictor(tf, model, meta);
    log.textContent = `done. Kept epoch ${best.epoch + 1}. Saved in this browser; the avatar page will find it.`;
    $('downloadBtn').disabled = false;
    $('liveBtn').disabled = false;
    $('savedInfo').textContent = `saved model: trained ${new Date(meta.trainedAt).toLocaleString()}, score ${meta.valR2}`;
  } catch (err) {
    log.textContent = 'error: ' + err.message;
    console.error(err);
  } finally {
    $('trainBtn').disabled = takes.length === 0;
  }
}

function showEvaluation(rows, r2, epoch) {
  const verdict = r2 > 0.3 ? 'strong' : r2 > 0.12 ? 'good' : r2 > 0.03 ? 'learning, more data will help' : 'weak, record more takes';
  $('evalSummary').innerHTML =
    `Score on unseen speech: <b>${r2.toFixed(3)}</b> (${verdict}). 0 = no better than always guessing the average, 1 = perfect.`;
  $('evalTable').innerHTML = '<tr><th>blendshape</th><th>prediction ↔ camera correlation</th></tr>' +
    rows.map(({ name, r }) => {
      const v = Number.isFinite(r) ? r : 0;
      return `<tr><td>${name}</td><td><div class="track"><div class="fill" style="width:${Math.max(0, v) * 100}%"></div></div> ${Number.isFinite(r) ? r.toFixed(2) : 'n/a'}</td></tr>`;
    }).join('');
}

function drawLossChart({ loss, val }, baseline, bestEpoch) {
  const cv = $('lossChart');
  const ctx = cv.getContext('2d');
  const { width: w, height: h } = cv;
  ctx.clearRect(0, 0, w, h);
  const all = [...loss, ...val, baseline];
  const max = Math.max(...all) * 1.05, min = Math.min(...all) * 0.95;
  const x = (i) => 30 + (i / (EPOCHS - 1)) * (w - 40);
  const y = (v) => h - 20 - ((v - min) / (max - min || 1)) * (h - 35);
  ctx.strokeStyle = '#555'; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(30, y(baseline)); ctx.lineTo(w - 10, y(baseline)); ctx.stroke();
  ctx.setLineDash([]);
  const line = (arr, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
  };
  line(loss, '#4f8cff');
  line(val, '#f5a524');
  if (bestEpoch >= 0) {
    ctx.fillStyle = '#3ecf6e';
    ctx.beginPath(); ctx.arc(x(bestEpoch), y(val[bestEpoch]), 4, 0, 7); ctx.fill();
  }
  ctx.font = '12px system-ui';
  ctx.fillStyle = '#4f8cff'; ctx.fillText('training loss', 40, 14);
  ctx.fillStyle = '#f5a524'; ctx.fillText('validation loss', 140, 14);
  ctx.fillStyle = '#888'; ctx.fillText('- - baseline (guess the average)', 260, 14);
  ctx.fillStyle = '#3ecf6e'; ctx.fillText('● kept', 470, 14);
}

// ---- 3. download (to commit for the public demo) ------------------------------------
$('downloadBtn').addEventListener('click', async () => {
  if (predictor) await predictor.model.save('downloads://lipsync-model');
});

// ---- 4. live test -------------------------------------------------------------------
$('liveBtn').addEventListener('click', async (e) => {
  e.target.disabled = true;
  try {
    let features = null;
    const mic = await startMicFrames({
      fps: 30,
      windowSize: 1024,
      onFrame: (samples) => {
        if (!features || !predictor) return;
        const values = predictor.push(features.compute(samples).mfcc, Number($('smooth').value));
        if (values) drawLive(values);
      },
    });
    features = createFeatureExtractor({ sampleRate: mic.sampleRate, fftSize: 1024 });
    e.target.textContent = '🎙 Live';
    $('liveInfo').textContent = `mic: ${mic.micLabel} | latency ≈ ${predictor.meta.future * 33} ms of look-ahead + smoothing`;
  } catch (err) {
    $('liveInfo').textContent = 'error: ' + err.message;
    e.target.disabled = false;
  }
});

let liveRows = null;
function drawLive(values) {
  const names = predictor.meta.targetNames;
  if (!liveRows) {
    $('liveBars').innerHTML = SHOW.filter((n) => names.includes(n)).map((n) =>
      `<div class="bar"><span>${n}</span><div class="track"><div class="fill" data-n="${n}"></div></div></div>`).join('');
    liveRows = [...document.querySelectorAll('#liveBars .fill')];
  }
  for (const el of liveRows) el.style.width = (predictor.get(values, el.dataset.n) * 100).toFixed(0) + '%';
  drawMouth(values);
}

// A simple 2D mouth, just to see the predictions move. Step 6 replaces this with your 3D avatar.
function drawMouth(v) {
  const g = (n) => predictor.get(v, n);
  const cv = $('mouth');
  const ctx = cv.getContext('2d');
  const { width: w, height: h } = cv;
  ctx.clearRect(0, 0, w, h);
  const open = Math.min(1, g('jawOpen') * 1.6 + (g('mouthLowerDownLeft') + g('mouthLowerDownRight')) * 0.4);
  const round = Math.min(1, g('mouthFunnel') + g('mouthPucker') * 0.6);
  const smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2 + (g('mouthStretchLeft') + g('mouthStretchRight')) / 4;
  const halfW = 70 * (1 - round * 0.45) * (1 + smile * 0.35);
  const halfH = 6 + open * 55;
  const cx = w / 2, cy = h / 2;
  ctx.fillStyle = '#3a0d12';
  ctx.strokeStyle = '#d9534f';
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(cx - halfW, cy - smile * 10);
  ctx.quadraticCurveTo(cx, cy - halfH * 0.8, cx + halfW, cy - smile * 10);
  ctx.quadraticCurveTo(cx, cy + halfH * 1.2, cx - halfW, cy - smile * 10);
  ctx.fill();
  ctx.stroke();
}

// ---- on load: is there already a saved model? -------------------------------------------
(async () => {
  const found = await Predictor.load(tf, [MODEL_KEY]);
  if (found) {
    predictor = found.predictor;
    $('savedInfo').textContent =
      `saved model found: trained ${new Date(predictor.meta.trainedAt).toLocaleString()}, score ${predictor.meta.valR2}`;
    $('liveBtn').disabled = false;
    $('downloadBtn').disabled = false;
  }
})();
