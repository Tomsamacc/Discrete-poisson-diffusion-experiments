import torch
import torch.nn as nn


class PoissonDiscreteDiffusionModel(nn.Module):
    def __init__(
        self,
        net,
        lbd=10.0,
        z_rescale=False,
        clip_z=True,
        clip_range=None,
        normalize=None,
    ):
        super().__init__()
        self.net = net
        self.lbd = lbd
        self.z_rescale = z_rescale
        self.clip_z = clip_z
        self.clip_range = clip_range
        self.normalize = normalize

    def prepare_z(self, z, alpha=None):
        """Copy of z for the MLP. Caller z is not modified."""
        z_in = z
        if self.clip_z:
            z_in = z_in.clamp_min(0.0)
        if self.z_rescale:
            if alpha is None:
                raise ValueError("z_rescale=True needs alpha (or γ/λ)")
            rate = self.lbd * alpha
            if rate.ndim == 1:
                rate = rate.view(-1, 1)
            z_in = z_in / rate.clamp_min(1e-12)
        return z_in

    def clip_xhat(self, xhat):
        xhat = xhat.clamp_min(0.0)
        if self.clip_range is not None:
            xhat = xhat.clamp(self.clip_range[0], self.clip_range[1])
        return xhat

    def encode_x(self, x):
        """x_0 -> network/loss coordinates. Inverse of decode_x."""
        if self.normalize is None:
            return x
        mean, std = self.normalize
        return (x - mean) / std

    def decode_x(self, x):
        if self.normalize is None:
            return x
        mean, std = self.normalize
        return x * std + mean

    def forward(self, z, t, alpha=None):
        z_in = self.prepare_z(z, alpha=alpha)
        xhat = self.net(z_in, t)
        return self.clip_xhat(xhat)
