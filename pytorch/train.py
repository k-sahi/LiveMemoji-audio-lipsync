"""train.py: train the audio → mouth model in PyTorch and export it for the browser demo.

  python train.py                      # default: past=6, future=2 (same as the browser trainer)
  python train.py --future 1           # less look-ahead: 33 ms faster mouth shapes
  python train.py --future 0 --out ../model/lipsync-model.json   # replace the demo's model

What it does:
  1. builds exactly the browser's dataset (dataset.py),
  2. trains with Adam + MSE, keeping the epoch with the lowest VALIDATION loss,
  3. reports R² (same definition as the browser) and per-shape correlations,
  4. measures CPU inference time per frame,
  5. writes a .pt checkpoint and a lipsync-model-v1 .json the browser loads unchanged.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_dataset, load_takes, pearson, r2_score
from export import export_browser_model
from model import LipSyncMLP

KEY_SHAPES = ["jawOpen", "mouthFunnel", "mouthPucker", "mouthClose", "mouthSmileLeft", "mouthStretchLeft"]


def train(takes, past=6, future=2, epochs=60, batch=128, lr=1e-3, seed=0, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = build_dataset(takes, past=past, future=future)
    xt, yt = torch.from_numpy(ds.x_train), torch.from_numpy(ds.y_train)
    xv, yv = torch.from_numpy(ds.x_val), torch.from_numpy(ds.y_val)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = LipSyncMLP(ds.input_size, len(ds.target_names))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    baseline = float(((ds.y_val - ds.y_train.mean(0)) ** 2).mean())   # always guessing the average
    best = dict(val=float("inf"), epoch=-1, state=None)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        if val < best["val"]:
            best = dict(val=val, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        if not quiet:
            print(f"epoch {epoch + 1:3d}/{epochs} | train {total / len(xt):.5f} | val {val:.5f} "
                  f"| R² {1 - val / baseline:.3f} | best epoch {best['epoch'] + 1}")

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pred = model(xv).numpy()
    r2 = r2_score(pred, ds.y_val, ds.y_train.mean(0))
    corr = {n: pearson(pred[:, k], ds.y_val[:, k]) for k, n in enumerate(ds.target_names)}

    # CPU inference time for ONE frame, the way the live demo calls it.
    x1 = xv[:1]
    with torch.no_grad():
        for _ in range(50):
            model(x1)
        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            model(x1)
            times.append((time.perf_counter() - t0) * 1e6)
    infer_us = float(np.median(times))

    return dict(model=model, ds=ds, r2=r2, corr=corr, best_epoch=best["epoch"] + 1,
                infer_us=infer_us, lookahead_ms=future * 1000 / 30)


def save(result, takes, out: Path):
    ds, model = result["ds"], result["model"]
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        fps=30, sampleRate=takes[0]["sampleRate"], mic=takes[0]["mic"], fftSize=takes[0]["fftSize"],
        numMfcc=ds.num_mfcc, past=ds.past, future=ds.future, mean=ds.mean, std=ds.std,
        targetNames=ds.target_names, trainFrames=int(len(ds.x_train)),
        valR2=round(result["r2"], 3),
        valCorr={n: round(0 if np.isnan(r) else r, 3) for n, r in result["corr"].items()},
        trainedWith="pytorch",
    )
    export_browser_model(model.linear_layers(), meta, str(out))
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out.with_suffix(".pt"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data", help="folder with your take-*.json recordings")
    p.add_argument("--past", type=int, default=6)
    p.add_argument("--future", type=int, default=2, help="look-ahead frames (each = 33 ms of delay)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="where to write the browser model (default: out/lipsync-model-f<future>.json)")
    a = p.parse_args()

    takes = load_takes(a.data)
    r = train(takes, a.past, a.future, a.epochs, a.batch, a.lr, a.seed)
    out = Path(a.out or f"out/lipsync-model-f{a.future}.json")
    save(r, takes, out)

    print(f"\nkept epoch {r['best_epoch']} | validation R² {r['r2']:.3f} | look-ahead {r['lookahead_ms']:.0f} ms "
          f"| CPU inference {r['infer_us']:.0f} µs per frame")
    for n in KEY_SHAPES:
        print(f"  {n:18s} r = {r['corr'][n]:.2f}")
    print(f"\nsaved {out} (load it in the browser) and {out.with_suffix('.pt')} (PyTorch checkpoint)")


if __name__ == "__main__":
    main()
