"""sweep.py: how much accuracy does each frame of look-ahead buy?

Trains the model with future = 0, 1, 2, 3 frames (0–100 ms of look-ahead), several seeds each
(the dataset is small, so one run can be lucky), and writes a results table:

  python sweep.py              # → out/results.md

This is the latency/quality trade-off every real-time speech system faces: waiting to hear
a little more makes predictions better, but the face reacts later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset import load_takes
from train import train

SHAPES = ["jawOpen", "mouthFunnel", "mouthPucker", "mouthClose"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data")
    p.add_argument("--futures", default="0,1,2,3")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()

    takes = load_takes(a.data)
    rows = []
    for f in [int(v) for v in a.futures.split(",")]:
        runs = [train(takes, future=f, epochs=a.epochs, seed=s, quiet=True) for s in range(a.seeds)]
        r2 = np.array([r["r2"] for r in runs])
        corr = {n: np.nanmean([r["corr"][n] for r in runs]) for n in SHAPES}
        rows.append((f, f * 1000 / 30, r2.mean(), r2.std(), corr, np.median([r["infer_us"] for r in runs])))
        print(f"future={f} ({f * 33:.0f} ms): R² {r2.mean():.3f} ± {r2.std():.3f}")

    lines = ["| look-ahead | delay | validation R² | " + " | ".join(f"{n} r" for n in SHAPES) + " | CPU / frame |",
             "|---|---|---|" + "---|" * len(SHAPES) + "---|"]
    for f, ms, m, sd, corr, us in rows:
        lines.append(f"| {f} frames | {ms:.0f} ms | {m:.3f} ± {sd:.3f} | " +
                     " | ".join(f"{corr[n]:.2f}" for n in SHAPES) + f" | {us:.0f} µs |")
    table = "\n".join(lines)
    Path("out").mkdir(exist_ok=True)
    Path("out/results.md").write_text(
        f"# Look-ahead vs accuracy\n\n{a.seeds} seeds per setting, past = 6 frames, {a.epochs} epochs.\n\n{table}\n",
        encoding="utf-8")
    print("\n" + table + "\n\nwritten to out/results.md")


if __name__ == "__main__":
    main()
