"""model.py: the audio → mouth network, in PyTorch.

Same architecture as the browser trainer (src/train.js), so results compare 1:1:

  117 inputs (9 frames x 13 MFCCs with past=6, future=2)
    → Gaussian noise 0.1 (training only: jitters inputs so it can't memorise exact numbers)
    → Linear 128 + ReLU → Dropout 0.2
    → Linear 64  + ReLU → Dropout 0.2
    → Linear 27  + sigmoid (one value per jaw/mouth blendshape, 0..1)
"""
from __future__ import annotations

import torch
from torch import nn


class LipSyncMLP(nn.Module):
    def __init__(self, input_size: int, n_out: int, hidden=(128, 64), dropout: float = 0.2, noise: float = 0.1):
        super().__init__()
        self.noise = noise
        dims = [input_size, *hidden]
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], n_out)
        # Glorot/Xavier init, like Keras/TF.js Dense layers, so training behaves like the browser version.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return torch.sigmoid(self.head(self.body(x)))

    def linear_layers(self):
        """(weight [out, in], bias [out]) for every Linear, as numpy, for export.py."""
        return [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
                for m in self.modules() if isinstance(m, nn.Linear)]
