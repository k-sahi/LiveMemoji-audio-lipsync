"""benchmark_onnx.py: export a trained checkpoint to ONNX and compare inference speed.

  pip install onnx onnxruntime
  python benchmark_onnx.py out/lipsync-model-f1.pt

ONNX is the portable model format used to deploy models outside PyTorch (ONNX Runtime on
servers, in browsers via onnxruntime-web, on phones). This checks the exported model gives
the same outputs as PyTorch and times one-frame inference in both.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from model import LipSyncMLP


def time_us(fn, n=2000):
    for _ in range(100):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e6)
    return np.percentile(t, 50), np.percentile(t, 95)


def main(ckpt_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("first: pip install onnx onnxruntime")

    ckpt = torch.load(ckpt_path, weights_only=False)
    meta = ckpt["meta"]
    n_in = (meta["past"] + 1 + meta["future"]) * meta["numMfcc"]
    model = LipSyncMLP(n_in, len(meta["targetNames"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = ckpt_path.replace(".pt", ".onnx")
    x = torch.randn(1, n_in)
    torch.onnx.export(model, x, onnx_path, input_names=["mfcc_window"], output_names=["blendshapes"],
                      dynamic_axes={"mfcc_window": {0: "batch"}, "blendshapes": {0: "batch"}}, dynamo=False)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"mfcc_window": x.numpy()})[0]
    print(f"exported {onnx_path} | max difference vs PyTorch: {np.abs(out - ref).max():.2e}")

    with torch.no_grad():
        pt50, pt95 = time_us(lambda: model(x))
    xn = x.numpy()
    ox50, ox95 = time_us(lambda: sess.run(None, {"mfcc_window": xn}))
    print(f"PyTorch      one frame: p50 {pt50:6.1f} µs, p95 {pt95:6.1f} µs")
    print(f"ONNX Runtime one frame: p50 {ox50:6.1f} µs, p95 {ox95:6.1f} µs")
    print(f"(the whole real-time budget per frame is 33,333 µs)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/lipsync-model-f2.pt")
