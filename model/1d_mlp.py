from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sin/cos of the raw scalar t. t shape (B,)."""
    half = dim // 2
    freqs = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * freqs)
    args = torch.outer(t.to(dtype=torch.float32), freqs)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temb_dim):
        super().__init__()
        self.act = nn.LeakyReLU(0.02)
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_fc = nn.Linear(in_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.out_fc = nn.Linear(out_dim, out_dim)
        self.proj = nn.Linear(temb_dim, out_dim)

    def forward(self, x, t_emb):
        h = self.in_fc(self.act(self.in_norm(x)))
        h = h + self.proj(self.act(t_emb))
        return self.out_fc(self.act(self.out_norm(h)))


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=128, out_dim=1, layers=3, continuous_t=True):
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
        # x: (B, 1) raw z    t: (B,) in [0, 1] if continuous_t
        if self.continuous_t:
            t = 1000.0 * t
        t_emb = self.time_mlp(timestep_embedding(t, self.hidden))
        h = self.in_fc(x)
        for layer in self.layers:
            h = layer(h, t_emb)
        return F.softplus(self.out_fc(h))

