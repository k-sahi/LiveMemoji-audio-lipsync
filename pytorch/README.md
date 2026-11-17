# PyTorch training pipeline

The browser demo trains its model with TensorFlow.js (`train.html`). This folder trains the
**same model in PyTorch**, on the same recordings, and exports it in the exact file format the
browser loads, so a PyTorch-trained model drops straight into the live demo.

| File | What it does |
|---|---|
| `dataset.py` | Line-by-line port of `src/dataset.js`: 9-frame MFCC windows, normalisation from training frames only, the leak-free "last 15% of each take" validation split |
| `model.py` | The network: 117 → 128 → 64 → 27, ReLU, dropout 0.2, input noise 0.1, sigmoid outputs |
| `train.py` | Adam + MSE, keeps the epoch with the lowest validation loss, reports R² and per-shape correlations, times inference, exports |
| `export.py` | Writes the browser's single-file `lipsync-model-v1` format (PyTorch `[out,in]` weights → TF.js `[in,out]` kernels) |
| `sweep.py` | The experiment: look-ahead (0–100 ms) vs accuracy, several seeds each → `out/results.md` |
| `benchmark_onnx.py` | Exports to ONNX, checks outputs match PyTorch, times PyTorch vs ONNX Runtime |

## Checked against the browser

- Same dataset: 4,796 training / 850 validation windows from the three takes, identical MFCC mean/std.
- Re-scoring the browser-trained model in Python gives **R² 0.576**, exactly what the browser reported.
- Exporting the browser model's weights through `export.py` reproduces the original file **byte for byte**.

## Run it

```bash
cd pytorch
pip install -r requirements.txt
python train.py                 # same settings as the browser (look-ahead 2 frames)
python sweep.py                 # look-ahead vs accuracy table → out/results.md
python train.py --future 0 --out ../model/lipsync-model.json   # use the PyTorch model in the demo
python benchmark_onnx.py out/lipsync-model-f0.pt               # optional
```

## The experiment: is look-ahead worth its delay?

The model sees `past` frames before and `future` frames after the moment it predicts. Every
future frame is 33 ms the avatar must wait before showing the mouth shape. The browser model
used `future = 2` (67 ms). The sweep measures what that delay actually buys.

Results from `sweep.py` (PyTorch, 3 seeds per setting, past = 6 frames, 60 epochs):

| look-ahead | delay | validation R² | jawOpen r | mouthFunnel r | mouthPucker r | mouthClose r | CPU / frame |
|---|---|---|---|---|---|---|---|
| 0 frames | 0 ms | 0.589 ± 0.005 | 0.31 | 0.46 | 0.70 | 0.22 | 62 µs |
| 1 frame | 33 ms | 0.599 ± 0.010 | 0.45 | 0.46 | 0.72 | 0.23 | 81 µs |
| 2 frames | 67 ms | 0.585 ± 0.008 | 0.48 | 0.63 | 0.72 | 0.26 | 73 µs |
| 3 frames | 100 ms | 0.591 ± 0.009 | 0.45 | 0.49 | 0.70 | 0.25 | 83 µs |

What it shows:
- **The overall score hides the trade-off.** R² is flat (0.585–0.599, within run-to-run noise),
  but the jaw needs at least one frame of look-ahead (r 0.31 → 0.45), and lip rounding ("oo")
  peaks at two frames (0.63).
- **One frame (33 ms) is the sweet spot here:** the best overall R², the jaw keeps its accuracy,
  and the mouth shape responds 33 ms sooner than the browser default of two frames.
- **Inference is not the bottleneck:** 60–80 µs per frame on a laptop CPU, about 0.2% of the
  33 ms real-time budget. The delay that matters is the look-ahead, not the compute.
- Caveats: about 3 minutes of one speaker, 850 validation frames, 3 seeds. The per-shape
  correlations are noisy; more recordings would firm this up.
