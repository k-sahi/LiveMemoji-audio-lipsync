"""dataset.py: turn recorded takes into (input, target) training pairs.

A line-by-line port of src/dataset.js, so the PyTorch model sees EXACTLY the data the
browser trainer sees:

  INPUT  for audio frame i: the 13 MFCCs of frames i-past ... i+future, each normalised
         (mean 0, std 1, statistics from TRAINING frames only). 9 frames x 13 = 117 numbers
         with the default past=6, future=2.
  TARGET for frame i: the 27 jaw/mouth blendshapes the camera saw at frame i+lag
         (lag = the mic-vs-camera delay measured when recording).
  SPLIT  the LAST 15% of every take is validation. A window must sit entirely on one side
         of the cut: neighbouring frames are near-copies, so mixing them would leak.

`future` is the knob this project experiments with: each future frame the model needs
is 33 ms the avatar must wait before it can show the mouth shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULTS = dict(past=6, future=2, val_frac=0.15, max_lag=10)


def js_round(x: float) -> int:
    """JavaScript's Math.round (halves round up), so the lag matches the browser exactly."""
    return int(np.floor(x + 0.5))


def is_mouth_shape(name: str) -> bool:
    return name.startswith("jaw") or name.startswith("mouth")


def load_takes(folder: str | Path) -> list[dict]:
    """Load and check every take-*.json in a folder (same checks as validateTake in JS)."""
    takes = []
    for path in sorted(Path(folder).glob("*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        if t.get("format") != "lipsync-take-v2":
            print(f"skipping {path.name}: not a lipsync-take-v2 recording")
            continue
        assert abs(t["fps"] - 30) < 0.5, f"{path.name}: expected 30 fps"
        assert t["numMfcc"] == 13, f"{path.name}: expected 13 MFCCs"
        assert len(t["blendshapeNames"]) == 52, f"{path.name}: missing blendshape names"
        labelled = sum(1 for f in t["frames"] if f.get("b"))
        assert labelled >= 100, f"{path.name}: only {labelled} frames with a face"
        t["fileName"] = path.name
        takes.append(t)
    if not takes:
        raise SystemExit(f"no recordings found in {folder}")
    return takes


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    target_names: list[str]
    mean: list[float]
    std: list[float]
    past: int
    future: int
    num_mfcc: int
    per_take: list[dict] = field(default_factory=list)

    @property
    def input_size(self) -> int:
        return self.x_train.shape[1]


def build_dataset(takes: list[dict], **opts) -> Dataset:
    o = {**DEFAULTS, **opts}
    past, future, val_frac, max_lag = o["past"], o["future"], o["val_frac"], o["max_lag"]
    win = past + 1 + future
    names = takes[0]["blendshapeNames"]
    for t in takes:
        if t["blendshapeNames"] != names:
            raise ValueError(f"{t['fileName']}: different blendshape list")
    target_idx = [i for i, n in enumerate(names) if is_mouth_shape(n)]
    target_names = [names[i] for i in target_idx]
    C = takes[0]["numMfcc"]

    # 1. Normalisation statistics from TRAINING frames only (no peeking at validation).
    s = np.zeros(C)
    ss = np.zeros(C)
    count = 0
    for t in takes:
        cut = int(np.floor(len(t["frames"]) * (1 - val_frac)))
        m = np.array([f["m"] for f in t["frames"][:cut]], dtype=np.float64)
        s += m.sum(0)
        ss += (m * m).sum(0)
        count += len(m)
    mean = s / count
    std = np.sqrt(np.maximum(ss / count - mean**2, 1e-6))

    # 2. Windows + targets.
    tr_x, tr_y, va_x, va_y, per_take = [], [], [], [], []
    for t in takes:
        F = t["frames"]
        n = len(F)
        cut = int(np.floor(n * (1 - val_frac)))
        lag = max(-max_lag, min(max_lag, js_round(t.get("estimatedLagFrames") or 0)))
        norm = (np.array([f["m"] for f in F], dtype=np.float64) - mean) / std
        n_tr = n_va = 0
        for i in range(past, n - future):
            j = i + lag
            if j < 0 or j >= n or not F[j].get("b"):
                continue
            lo, hi = min(i - past, j), max(i + future, j)
            if hi < cut:
                xs, ys = tr_x, tr_y
                n_tr += 1
            elif lo >= cut:
                xs, ys = va_x, va_y
                n_va += 1
            else:
                continue  # straddles the cut: would leak
            xs.append(norm[i - past : i + future + 1].reshape(-1))
            ys.append([F[j]["b"][k] for k in target_idx])
        per_take.append(dict(fileName=t["fileName"], frames=n, lag=lag, train=n_tr, val=n_va))

    f32 = lambda a, w: np.asarray(a, dtype=np.float32).reshape(-1, w)
    return Dataset(
        x_train=f32(tr_x, win * C), y_train=f32(tr_y, len(target_names)),
        x_val=f32(va_x, win * C), y_val=f32(va_y, len(target_names)),
        target_names=target_names, mean=mean.tolist(), std=std.tolist(),
        past=past, future=future, num_mfcc=C, per_take=per_take,
    )


def r2_score(pred: np.ndarray, truth: np.ndarray, train_mean: np.ndarray) -> float:
    """Same score as the browser: 1 - MSE(model) / MSE(always predicting the training average)."""
    mse = float(((pred - truth) ** 2).mean())
    baseline = float(((truth - train_mean) ** 2).mean())
    return 1 - mse / baseline


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else float("nan")
