# SPDX-License-Identifier: MIT
"""Configurable-taper HNeRV decoder — a FAITHFUL GENERALIZATION of the vendored PR95
``hnerv_muon`` decoder (``experiments/results/public_pr_intake_full/.../src/model.py``),
whose ONLY change is that the 7-stage channel schedule (the "taper") is an explicit
argument instead of the hardcoded HNeRV ratio ``[C,C,C,.75C,.58C,.5C,.5C]``.

WHY: the gate-2 d_seg sensitivity map showed ~81% of the base_ch=20 decoder's capacity
is mis-allocated to the INSENSITIVE low-res early stages, while the d_seg-CRITICAL
high-res band (``rgb_1`` / ``skips.2-4`` / ``blocks.4``) is starved. Conv params are
``in·out·k²`` — independent of spatial resolution — so REALLOCATING channels from the
early stages to the mid-late stages is ~byte-neutral yet feeds capacity exactly where
d_seg is decided. This is rate-aware capacity allocation = the contest objective
(minimise d_seg at fixed bytes). It is basis-agnostic (the same reallocation helps a
frontier-class basis).

PARITY CONTRACT (the faithful-generalization gate): with the DEFAULT taper (``channels``
= the vendored HNeRV ratio for the given ``base_channels``) this decoder is
architecturally IDENTICAL to the vendored one — same module names, same param shapes,
and bit-identical ``forward`` output given the same weights — so it round-trips through
the schedule-AGNOSTIC vendored codec (``build_archive`` / ``parse_archive`` /
``partition_params_for_muon`` / ``apply_qat``) unchanged. Verified by
``tests/test_configurable_taper_decoder.py`` (load vendored weights → this decoder →
byte-identical forward).

NO originality claim: this is the vendored PR95 decoder with a configurable taper, fully
attributed. It is a TOOL for the taper RD experiment, not a submission.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def vendored_taper(base_channels: int) -> list[int]:
    """The exact vendored HNeRV channel schedule for ``base_channels`` (the parity
    default). MUST match ``model.HNeRVDecoder.__init__`` line-for-line:
    ``[C, C, C, int(C*0.75), int(C*0.58), int(C*0.5), int(C*0.5)]``."""
    C = int(base_channels)
    return [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]


def decoder_param_count(channels: list[int], latent_dim: int = 28) -> int:
    """Exact trainable-param count for a taper schedule — so a d_seg-aware taper can be
    byte-MATCHED to the baseline (the archive is ~int8(params) + latents, so param count
    is the rate proxy). Mirrors the module construction below exactly:
      stem   : Linear(latent_dim, channels[0]*48)        = latent_dim*c0*48 + c0*48
      blocks : Conv2d(c_i, c_{i+1}*4, 3)                  = c_i*(c_{i+1}*4)*9 + c_{i+1}*4
      skips  : Conv2d(c_i, c_{i+1}, 1) if c_i!=c_{i+1}    = c_i*c_{i+1} + c_{i+1}, else 0
      refine : Conv2d(cf, cf//2, 3) + Conv2d(cf//2, cf, 3)
      rgb_0/1: Conv2d(cf, 3, 3)  (x2)
    """
    if len(channels) != 7:
        raise ValueError(f"taper must have 7 stages; got {len(channels)}: {channels}")
    if any(int(c) <= 0 for c in channels):
        raise ValueError(f"all taper channels must be positive; got {channels}")
    if int(channels[-1]) < 2:
        raise ValueError(
            f"taper final channel must be >= 2 (refine uses final_ch//2 as its intermediate "
            f"width → final_ch//2==0 builds an invalid Conv2d); got channels[-1]={channels[-1]}"
        )
    c = [int(x) for x in channels]
    base_h, base_w = 6, 8
    n = 0
    # stem
    n += latent_dim * (c[0] * base_h * base_w) + (c[0] * base_h * base_w)
    # blocks + skips
    for i in range(6):
        ci, co = c[i], c[i + 1]
        n += ci * (co * 4) * 9 + (co * 4)
        if ci != co:
            n += ci * co + co
    # refine (2 convs)
    cf = c[-1]
    n += cf * (cf // 2) * 9 + (cf // 2)
    n += (cf // 2) * cf * 9 + cf
    # rgb_0 + rgb_1
    n += 2 * (cf * 3 * 9 + 3)
    return n


def dseg_aware_taper(base_channels: int, latent_dim: int = 28) -> list[int]:
    """A d_seg-AWARE taper for ``base_channels``: reallocate capacity from the insensitive
    low-res early stages (0-2) to the d_seg-critical mid-late stages (3-5) where the gate-2
    sensitivity map found d_seg lives (``skips.2-4`` / ``blocks.4``). The FINAL channel is
    kept near the baseline final (raising it inflates ``refine`` quadratically). Tuned to be
    byte-MATCHED to ``vendored_taper`` within ~3% so the A/B isolates ALLOCATION from total
    capacity. ``latent_dim`` is threaded into the byte-match target so the match is EXACT at
    any latent_dim (the stem param count = latent_dim·channels[0]·48 depends on it). For
    base_ch=20 baseline [20,20,20,15,11,10,10] this yields a flatter, late-weighted schedule
    of equal-ish param count."""
    C = int(base_channels)
    base = vendored_taper(C)
    target = decoder_param_count(base, latent_dim=latent_dim)
    # Reallocation shape: pull the early stages DOWN, push the mid-late stages UP, hold the
    # final near baseline. Expressed as ratios of C; tuned then byte-matched by a 1-D scan
    # on the early-stage shrink so the total lands within ~3% of the baseline param count.
    # mid-late raised (stages 3,4,5), final held; early (0,1,2) shrunk to pay for it.
    def schedule(shrink: float) -> list[int]:
        return [
            max(1, int(C * shrink)),          # stage0 (insensitive low-res)
            max(1, int(C * shrink)),          # stage1
            max(1, int(C * (shrink + 0.05))), # stage2 (skips.2 starts mattering)
            max(1, int(C * 0.95)),            # stage3 (skips.3 — d_seg-critical)
            max(1, int(C * 0.95)),            # stage4 (skips.4 / blocks.4 — d_seg-critical)
            max(1, int(C * 0.70)),            # stage5 (feeds final)
            max(1, int(C * 0.50)),            # stage6 final (held ~baseline to bound refine)
        ]
    # 1-D byte-match scan over the early-stage shrink ratio (param count at the SAME
    # latent_dim as the target, so the match is exact regardless of latent_dim).
    best, best_gap = None, None
    s = 0.40
    while s <= 0.95 + 1e-9:
        cand = schedule(s)
        gap = abs(decoder_param_count(cand, latent_dim=latent_dim) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = cand, gap
        s += 0.01
    return best


class ConfigurableTaperHNeRVDecoder(nn.Module):
    """The vendored PR95 HNeRV decoder with an EXPLICIT channel schedule.

    ``channels``: the 7-stage taper. If None, defaults to ``vendored_taper(base_channels)``
    → bit-identical architecture to the vendored decoder (the parity contract). The
    ``__init__`` body REPLICATES the vendored ``model.HNeRVDecoder.__init__`` verbatim
    (module order, layer types, the Identity-vs-1x1-skip choice) so the state_dict keys +
    shapes match and the codec is unchanged. ``forward`` is byte-identical to the vendored
    forward (same ops in the same order)."""

    def __init__(
        self,
        latent_dim: int = 28,
        base_channels: int = 36,
        eval_size: tuple[int, int] = (384, 512),
        channels: list[int] | None = None,
    ):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        if channels is None:
            channels = vendored_taper(base_channels)
        if len(channels) != 7:
            raise ValueError(f"taper must have 7 stages; got {len(channels)}: {channels}")
        if any(int(c) <= 0 for c in channels):
            raise ValueError(f"all taper channels must be positive; got {channels}")
        if int(channels[-1]) < 2:
            raise ValueError(
                f"taper final channel must be >= 2 (refine uses final_ch//2 as its "
                f"intermediate width → final_ch//2==0 builds an invalid Conv2d); got "
                f"channels[-1]={channels[-1]}"
            )
        self.channels = [int(c) for c in channels]

        # --- verbatim vendored construction (only self.channels is the variable) -------
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
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)
