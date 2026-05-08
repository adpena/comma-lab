"""HNeRVDecoder — verbatim copy of the PR106/PR107 architecture.

The factorized_hnerv_v1 runtime reconstructs floats via SVD components and
loads them into this exact model. Strict-scorer-rule: this module imports
no scorer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HNeRVDecoder(nn.Module):
    def __init__(self, latent_dim=28, base_channels=36, eval_size=(384, 512)):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z):
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)
