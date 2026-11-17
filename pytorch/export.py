"""export.py: save trained weights as the browser's single-file model (lipsync-model-v1).

The page loads models with TensorFlow.js (src/predictor.js). Rather than converting through
ONNX/TF, this writes the same JSON the browser trainer writes: a Keras-style Sequential
topology, the weights as one base64 blob, and the metadata the predictor needs
(past/future window, MFCC mean/std, blendshape names, score).

PyTorch's nn.Linear stores weight as [out, in]; Keras/TF.js Dense kernels are [in, out],
so every weight matrix is transposed on the way out.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import numpy as np


def _dense(i: int, units: int, activation: str) -> dict:
    return {"class_name": "Dense", "config": {
        "units": units, "activation": activation, "use_bias": True,
        "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "normal", "seed": None}},
        "bias_initializer": {"class_name": "Zeros", "config": {}},
        "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
        "kernel_constraint": None, "bias_constraint": None, "name": f"dense_Dense{i}", "trainable": True}}


def _dropout(i: int, rate: float) -> dict:
    return {"class_name": "Dropout", "config": {"rate": rate, "noise_shape": None, "seed": None, "name": f"dropout_Dropout{i}", "trainable": True}}


def export_browser_model(layers: list[tuple[np.ndarray, np.ndarray]], meta: dict, path: str,
                         activations=("relu", "relu", "sigmoid"), noise=0.1, dropout=0.2) -> None:
    """layers: [(weight [out,in], bias [out]), ...] in PyTorch layout."""
    input_size = layers[0][0].shape[1]
    topo_layers = [{"class_name": "GaussianNoise", "config": {
        "stddev": noise, "name": "gaussian_noise_GaussianNoise1", "trainable": True,
        "batch_input_shape": [None, input_size], "dtype": "float32"}}]
    specs, blobs = [], []
    for i, ((w, b), act) in enumerate(zip(layers, activations), start=1):
        topo_layers.append(_dense(i, w.shape[0], act))
        if i < len(layers):
            topo_layers.append(_dropout(i, dropout))
        kernel = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)   # [in, out]
        bias = np.asarray(b, dtype=np.float32)
        specs += [{"name": f"dense_Dense{i}/kernel", "shape": list(kernel.shape), "dtype": "float32"},
                  {"name": f"dense_Dense{i}/bias", "shape": list(bias.shape), "dtype": "float32"}]
        blobs += [kernel.tobytes(), bias.tobytes()]
    model = {
        "format": "lipsync-model-v1",
        "modelTopology": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": topo_layers},
                          "keras_version": "tfjs-layers 4.22.0", "backend": "tensor_flow.js"},
        "weightSpecs": specs,
        "weightDataBase64": base64.b64encode(b"".join(blobs)).decode("ascii"),
        "userDefinedMetadata": {"version": 1, "trainedAt": datetime.now(timezone.utc).isoformat(), **meta},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)


def load_browser_model(path: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Read a lipsync-model-v1 file back into PyTorch-layout (weight [out,in], bias) pairs."""
    m = json.load(open(path, encoding="utf-8"))
    raw = base64.b64decode(m["weightDataBase64"])
    arrays, off = {}, 0
    for spec in m["weightSpecs"]:
        n = int(np.prod(spec["shape"]))
        arrays[spec["name"]] = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).reshape(spec["shape"])
        off += n * 4
    layers, i = [], 1
    while f"dense_Dense{i}/kernel" in arrays:
        layers.append((arrays[f"dense_Dense{i}/kernel"].T.copy(), arrays[f"dense_Dense{i}/bias"].copy()))
        i += 1
    return layers, m["userDefinedMetadata"]


def numpy_forward(layers, x, activations=("relu", "relu", "sigmoid")) -> np.ndarray:
    """Inference without PyTorch (used to check exports and the old browser-trained model)."""
    for (w, b), act in zip(layers, activations):
        x = x @ w.T + b
        x = np.maximum(x, 0) if act == "relu" else 1 / (1 + np.exp(-x))
    return x
