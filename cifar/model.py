import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_ITD = Path("/home/tomsama/scratch/itdpdm/refs/ITdiffusion")
if str(_ITD) not in sys.path:
    sys.path.insert(0, str(_ITD))

if "diffusers" not in sys.modules:
    _diffusers = types.ModuleType("diffusers")

    class _UNet2DModel(nn.Module):
        pass

    _diffusers.UNet2DModel = _UNet2DModel
    sys.modules["diffusers"] = _diffusers

from utilsitd.unet import UNetModel


def net_input(z, gamma, gmin):
    g = gamma.reshape(-1).float().clamp_min(float(gmin))
    zin = z.clamp_min(0) / g.view(-1, 1, 1, 1)
    return zin, g.log()


class CifarPoissonNet(nn.Module):
    def __init__(self, n_heads, gmin):
        super().__init__()
        self.gmin = float(gmin)
        self.backbone = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=128,
            num_res_blocks=3,
            attention_resolutions=(2, 4),
            dropout=0.3,
            channel_mult=(1, 2, 2, 2),
            num_heads=4,
            use_scale_shift_norm=True,
        )
        conv = self.backbone.out[-1]
        nn.init.kaiming_uniform_(conv.weight, a=5**0.5)
        if conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            bound = fan_in**-0.5
            nn.init.uniform_(conv.bias, -bound, bound)
        self.heads = nn.ModuleList(
            [nn.Conv2d(128, 3, kernel_size=3, padding=1) for _ in range(int(n_heads))]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z, gamma):
        zin, logg = net_input(z, gamma, self.gmin)
        feat = self.backbone(zin, logg)
        return torch.stack([head(feat) for head in self.heads], dim=1)
