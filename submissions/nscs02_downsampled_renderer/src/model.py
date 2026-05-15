"""NSCS02 standalone decoder (mirror of tac.substrates.nscs02_downsampled_renderer.architecture).

Per HNeRV parity discipline lesson 9 (runtime closure) the inflate
runtime tree must be self-contained. This module is a byte-faithful
duplicate of ``src/tac/substrates/nscs02_downsampled_renderer/architecture.py``
with no ``tac.*`` imports. Parity is enforced by
``src/tac/tests/test_substrate_nscs02_downsampled_renderer.py::test_submission_decoder_byte_parity_with_substrate_decoder``.

Architecture: 5 PixelShuffle stages from (6, 8) -> (192, 256).
Channel taper [C, C, 0.75C, 0.58C, 0.5C, 0.5C].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSCS02Decoder(nn.Module):
    def __init__(self, latent_dim=28, base_channels=36, render_hw=(192, 256)):
        super().__init__()
        self.render_hw = render_hw
        self.base_h, self.base_w = 6, 8
        C = base_channels

        self.channels = [C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        n_stages = 5
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_stages):
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
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)
