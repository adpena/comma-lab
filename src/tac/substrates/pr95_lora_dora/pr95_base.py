# SPDX-License-Identifier: MIT
"""Byte-faithful re-port of PR95 hnerv_muon `src/model.py` (55 LOC original).

Source: experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/model.py

The verbatim port is required because:
  (a) PR95 intake clones MUST stay pristine per Catalog #109 — we cannot
      import directly from the intake.
  (b) Substrate's runtime tree must be self-contained per HNeRV parity
      discipline lesson 9 (runtime closure).
  (c) The architecture is 55 LOC and reviewable in 30 seconds per
      lesson 12.

Forward shape: latent (B, 28) -> RGB pair (B, 2, 3, 384, 512) in [0, 255].
Parameter count: 228,958 (matches `archive_provenance.json` claim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HNeRVDecoder(nn.Module):
    """Byte-faithful copy of PR95's HNeRVDecoder.

    Architecture (per `src/model.py` of PR95 hnerv_muon):
    - Stem: Linear(latent_dim, C*6*8) where C = base_channels
    - 6 upsample stages: Conv(in_ch, out_ch*4, 3x3) + PixelShuffle(2) +
        bilinear_skip + sin(out + identity)
    - Channel taper [C, C, C, int(0.75C), int(0.58C), int(0.5C), int(0.5C)]
    - Refine: Conv 3x3 dilation=2 (final_ch -> final_ch//2) +
        Conv 3x3 (final_ch//2 -> final_ch); residual `0.1 * sin(refine(x))`
    - Two RGB heads (separate frame 0 / frame 1): Conv 3x3 -> sigmoid * 255
    """

    def __init__(self, latent_dim: int = 28, base_channels: int = 36,
                 eval_size: tuple[int, int] = (384, 512)):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels

        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58),
                         int(C * 0.5), int(C * 0.5)]

        self.stem = nn.Linear(latent_dim,
                              self.channels[0] * self.base_h * self.base_w)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                else nn.Identity()
            )
        self.ps = nn.PixelShuffle(2)

        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=True):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear",
                                     align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)
