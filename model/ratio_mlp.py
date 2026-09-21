"""PMF-ratio MLP: unconstrained log B_k, k=1..K. Same trunk as 1d_mlp.MLP."""
from __future__ import annotations

import importlib

import torch.nn as nn

_mlp = importlib.import_module("model.1d_mlp")
ConditionalLayer = _mlp.ConditionalLayer
timestep_embedding = _mlp.timestep_embedding


class RatioMLP(nn.Module):
    """(z, t) -> (b_1,...,b_K),  b_k = log B_k,  B_k = [p_γ(z+k)/p_γ(z)] / γ^k."""

    def __init__(self, in_dim=1, hidden=128, out_dim=3, layers=3, continuous_t=True):
        super().__init__()
        self.hidden = hidden
        self.continuous_t = continuous_t
        temb_dim = 4 * hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, temb_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(temb_dim, temb_dim),
        )
        self.in_fc = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [ConditionalLayer(hidden, hidden, temb_dim) for _ in range(layers)]
        )
        self.out_fc = nn.Sequential(nn.LeakyReLU(0.02), nn.Linear(hidden, out_dim))

    def forward(self, x, t):
        if self.continuous_t:
            t = 1000.0 * t
        t_emb = self.time_mlp(timestep_embedding(t, self.hidden))
        h = self.in_fc(x)
        for layer in self.layers:
            h = layer(h, t_emb)
        return self.out_fc(h)
