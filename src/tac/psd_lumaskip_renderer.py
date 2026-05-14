# SPDX-License-Identifier: MIT
"""PSD-LumaSkip postfilter — PoseNet-aware luma-skip variant of PSD.

This module implements the Lane PSD-LumaSkip architecture approved by the
Council Phase A review on 2026-04-30 (see
``.omx/research/council_lane_psd_lumaskip_design_20260430.md``). It is a
SCAFFOLD landing only — no GPU dispatch is authorized until a separate
council convenes with empirical predicted-band evidence.

Background — why this lane exists
---------------------------------
Vanilla PSD (``PSDPostFilter`` in ``architectures.py``) operates on RGB at
camera-size, ``PixelUnshuffle(2)`` to half resolution, processes a 12-channel
intermediate, then ``PixelShuffle(2)`` back to RGB. This bottleneck is
*aligned* with SegNet's stride-2 stem (the 12.8% SegNet improvement
historically measured at 2026-04-11) but *destroys* the high-frequency luma
content that FastViT-PoseNet's attention layers actually attend to via the
YUV6 polyphase decomposition. Empirical regression: 5x worse PoseNet
distortion (0.011 vs Lane G v3's 0.000931) → score 1.49 vs Lane G v3's 1.05.

Mechanism — why luma-skip might recover the regression
------------------------------------------------------
``rgb_to_yuv6`` in ``upstream/frame_utils.py`` decomposes the luma plane Y
into four polyphase planes ``(y00, y10, y01, y11)`` via stride-2 sampling.
This decomposition is *exact*: the full-resolution Y is fully recoverable
from the four half-resolution planes. PoseNet sees 8-of-12 channels per
frame-pair as luma polyphase (66% of input channels). High-frequency Y at
384x512 (after PoseNet's bilinear resize) is what the attention layers use
for vehicle micro-motion and lane-marker phase predictions.

PSD-LumaSkip preserves a **full-resolution luma path** that bypasses the
PSD bottleneck entirely. The chroma+RGB-residual still flows through PSD's
half-res bottleneck (chroma is already half-res inside YUV6 anyway, so the
downscale is largely free for chroma). At the output, both residuals add
back to the input RGB before the bilinear resize that feeds PoseNet.

Architecture (council-approved per Phase A memo §3 and §4)
---------------------------------------------------------
    INPUT: x = RGB(B, 3, H, W) at camera-size [0, 255]

    Step 1: x_norm = x / 255

    Step 2: Luma path (full-resolution, lightweight ~1.5K params at
            luma_hidden=16):
        y_in        = 0.299 R + 0.587 G + 0.114 B          # (B, 1, H, W) full res
        h_y         = Conv(1 -> luma_hidden, 3x3) + ReLU
        h_y         = Conv(luma_hidden -> luma_hidden, 3x3 dilated=2) + ReLU
        y_residual  = Conv(luma_hidden -> 1, 3x3, zero-init)
        if use_learned_luma_projection:
            y_residual_3ch = Conv(1 -> 3, 1x1, init from broadcast prior) (y_residual)
        else:
            y_residual_3ch = y_residual.expand(-1, 3, -1, -1)

    Step 3: Chroma path (PSD bottleneck, the PSD half-res workhorse):
        h_down      = PixelUnshuffle(2)(x_norm)            # (B, 12, H/2, W/2)
        h           = Conv(12 -> hidden, 3x3) + ReLU
        h           = Conv(hidden -> hidden, 3x3 dilated=2) + ReLU
        h           = Conv(hidden -> hidden, 3x3) + ReLU
        rgb_resid_h = Conv(hidden -> 12, 3x3, zero-init)
        rgb_residual = PixelShuffle(2)(rgb_resid_h)        # (B, 3, H, W)

    Step 4: Combine + clamp:
        output = clamp(x_norm + rgb_residual + y_residual_3ch, 0, 1) * 255

The two output convs (``self.conv4`` chroma and ``self.luma_out`` luma) are
both zero-init so the postfilter starts as identity. EMA snapshot+restore
plumbs through the canonical ``tac.training.EMA`` path (verified via
``test_psd_lumaskip_renderer.test_ema_compatibility``).

References
----------
- Council Phase A memo: ``.omx/research/council_lane_psd_lumaskip_design_20260430.md``
- Council #271 KILL of vanilla PSD: ``.omx/research/council_lane_7_psd_dispatch_review_20260430.md``
- Reactivation Criterion #1: ``.omx/research/lane_7_psd_kill_memo_20260430.md`` Section 3
- Upstream scorer pipeline: ``upstream/modules.py`` (PoseNet/SegNet) and
  ``upstream/frame_utils.py`` (``rgb_to_yuv6`` polyphase decomposition)
- Vanilla PSD reference: ``src/tac/architectures.py:106`` (``PSDPostFilter``)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Standard rec.601 luma weights (matches `rgb_to_yuv6` in upstream/frame_utils.py)
_LUMA_R: float = 0.299
_LUMA_G: float = 0.587
_LUMA_B: float = 0.114


class PSDLumaSkipPostFilter(nn.Module):
    """PoseNet-aware luma-skip variant of PSD.

    Two parallel residual paths added to the input RGB:

    1. **Chroma+RGB-residual path** (PSD bottleneck): ``PixelUnshuffle(2)`` →
       4-conv body at half resolution → ``PixelShuffle(2)`` back to full res.
       Carries the SegNet-aligned attack signal (12.8% historical SegNet
       advantage of vanilla PSD).

    2. **Luma path** (full-resolution skip): rec.601 luma extraction → small
       3-conv stack at FULL resolution → optional learned 1×1 projection back
       to 3 channels. Carries the FastViT-PoseNet-aligned correction signal
       that the half-res bottleneck cannot preserve.

    Args:
        hidden: chroma path hidden channels (PSD-equivalent; default 64
            matches ``PSDPostFilter``).
        kernel: conv kernel size (default 3, with dilation=2 on conv2 of
            chroma path matching PSDPostFilter).
        luma_hidden: luma path hidden channels (Fridrich Phase A
            recommendation: 16). Smaller values (e.g. 8) under-allocate
            capacity to luma; larger values (e.g. 32) consume rate without
            evidence of additional gain.
        use_learned_luma_projection: when True, the 1-channel luma residual
            is projected to 3 channels via a learned 1×1 conv initialized
            with the broadcast prior (each output equal to input). When
            False, the 1-channel residual is `.expand`-broadcast to 3
            channels (zero parameters but no learned per-channel gain).

    Notes:
        - All output convs (``self.conv4`` chroma, ``self.luma_out``) are
          zero-init so the postfilter starts as identity (residuals are
          zero at init).
        - The architecture is residual-add (correction added to input),
          NOT replacement. So at init, output == input (after clamp).
        - Both H and W must be divisible by 2 (PixelUnshuffle constraint).
        - Param count target at hidden=64, luma_hidden=16: ~95-100K.
        - Compatible with ``tac.training.EMA`` decay=0.997 (canonical).

    Cross-references:
        - Phase A council memo §3 (scaffold spec) and §4 (predicted bands)
        - Polyphase preservation argument: Phase A memo §0 F1+F2
    """

    def __init__(
        self,
        hidden: int = 64,
        kernel: int = 3,
        luma_hidden: int = 16,
        use_learned_luma_projection: bool = True,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if luma_hidden < 1:
            raise ValueError(f"luma_hidden must be >= 1, got {luma_hidden}")
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError(f"kernel must be a positive odd int, got {kernel}")

        self.hidden = hidden
        self.luma_hidden = luma_hidden
        self.use_learned_luma_projection = use_learned_luma_projection

        pad = kernel // 2

        # ── Chroma + RGB residual path (PSDPostFilter geometry) ──
        # Mirrors PSDPostFilter exactly so the chroma path retains PSD's
        # SegNet-aligned 12.8% advantage. PixelUnshuffle/PixelShuffle on RGB.
        self.down = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.up = nn.PixelShuffle(2)

        # ── Luma skip path (full-resolution, lightweight) ──
        # NEVER apply any downsampling primitive on this path — see
        # ``check_psd_lumaskip_preserves_luma_resolution`` in the preflight.
        self.luma_in = nn.Conv2d(1, luma_hidden, kernel, padding=pad, bias=True)
        self.luma_mid = nn.Conv2d(luma_hidden, luma_hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.luma_out = nn.Conv2d(luma_hidden, 1, kernel, padding=pad, bias=True)

        if use_learned_luma_projection:
            # Learned 1×1 from luma-residual to 3-channel correction.
            # Initialized with the broadcast prior (each output channel = input)
            # so behaviour at init matches the simpler `.expand(-1, 3, -1, -1)`
            # path. Subsequent training can learn per-channel scaling
            # (e.g. luma correction may need to be slightly stronger on R or
            # B depending on the failure mode).
            self.luma_project = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        else:
            self.luma_project = None  # type: ignore[assignment]

        self.act = nn.ReLU(inplace=False)

        # ── Zero-init the residual outputs so the filter starts as identity ──
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.luma_out.weight)
        nn.init.zeros_(self.luma_out.bias)

        if self.luma_project is not None:
            # Broadcast-prior init: each of the 3 output channels equals the
            # 1-channel input. Bias zero. After zero residual at luma_out the
            # initial luma_project output is zero anyway, but the prior makes
            # the gradient flow symmetric across R/G/B at the start of
            # training rather than randomly biased.
            with torch.no_grad():
                self.luma_project.weight.zero_()
                self.luma_project.weight[:, 0, 0, 0] = 1.0
                self.luma_project.bias.zero_()

    # ---------------------------------------------------------------- forward

    def _luma_extract(self, x_norm: torch.Tensor) -> torch.Tensor:
        """rec.601 luma extraction matching upstream `rgb_to_yuv6`.

        Args:
            x_norm: (B, 3, H, W) RGB in [0, 1]

        Returns:
            (B, 1, H, W) luma in [0, 1]
        """
        y = (
            x_norm[:, 0:1] * _LUMA_R
            + x_norm[:, 1:2] * _LUMA_G
            + x_norm[:, 2:3] * _LUMA_B
        )
        return y

    def forward_luma_path(self, y_in: torch.Tensor) -> torch.Tensor:
        """Forward through the full-resolution luma skip path.

        Args:
            y_in: (B, 1, H, W) luma in [0, 1]

        Returns:
            (B, 1, H, W) luma residual (zero at init due to luma_out zero-init)
        """
        h = self.act(self.luma_in(y_in))
        h = self.act(self.luma_mid(h))
        return self.luma_out(h)

    def forward_chroma_path(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Forward through the PSD half-res bottleneck (chroma+RGB residual).

        Args:
            x_norm: (B, 3, H, W) RGB in [0, 1]

        Returns:
            (B, 3, H, W) RGB residual (zero at init due to conv4 zero-init)
        """
        h = self.down(x_norm)            # (B, 12, H/2, W/2)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.conv4(h)                # (B, 12, H/2, W/2)
        return self.up(h)                # (B, 3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dual-path postfilter.

        Args:
            x: (B, 3, H, W) RGB in [0, 255]. H and W must be divisible by 2.

        Returns:
            (B, 3, H, W) RGB in [0, 255], clamped.

        Raises:
            ValueError: if input shape is not 4D or H/W not divisible by 2.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"PSDLumaSkipPostFilter expects (B, 3, H, W); got {tuple(x.shape)}"
            )
        h, w = x.shape[2], x.shape[3]
        if h % 2 or w % 2:
            raise ValueError(
                f"PixelUnshuffle(2) requires H and W divisible by 2; got H={h}, W={w}"
            )

        x_norm = x / 255.0

        # Luma path (full-resolution skip)
        y_in = self._luma_extract(x_norm)
        y_residual = self.forward_luma_path(y_in)
        if self.luma_project is not None:
            y_residual_3ch = self.luma_project(y_residual)
        else:
            y_residual_3ch = y_residual.expand(-1, 3, -1, -1)

        # Chroma path (PSD half-res bottleneck)
        rgb_residual = self.forward_chroma_path(x_norm)

        # Combine and clamp back to [0, 255]
        output = (x_norm + rgb_residual + y_residual_3ch).clamp(0, 1) * 255.0
        return output

    # ---------------------------------------------------------------- helpers

    def num_parameters(self) -> int:
        """Return the total parameter count (for predicted-band sanity)."""
        return sum(p.numel() for p in self.parameters())

    def luma_path_params(self) -> int:
        """Return the parameter count of the luma skip path only."""
        n = sum(p.numel() for p in self.luma_in.parameters())
        n += sum(p.numel() for p in self.luma_mid.parameters())
        n += sum(p.numel() for p in self.luma_out.parameters())
        if self.luma_project is not None:
            n += sum(p.numel() for p in self.luma_project.parameters())
        return n

    def chroma_path_params(self) -> int:
        """Return the parameter count of the chroma+RGB-residual path only."""
        n = sum(p.numel() for p in self.conv1.parameters())
        n += sum(p.numel() for p in self.conv2.parameters())
        n += sum(p.numel() for p in self.conv3.parameters())
        n += sum(p.numel() for p in self.conv4.parameters())
        return n
