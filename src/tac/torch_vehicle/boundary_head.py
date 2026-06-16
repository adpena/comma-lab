# SPDX-License-Identifier: MIT
"""Boundary-conditioned / depthwise-separable HEAD variant for the hungriest tensor.

THE PROBLEM (from the gate-2 d_seg sensitivity map ``reports/dseg_sensitivity_map_n600.json``):
``rgb_1`` — the frame-1 output head SegNet reads to decide d_seg — has the highest
sensitivity density (delta_per_kparam ~3.88e-4 at the 384x512 stage; the HIGH band is
~2.84x under-provisioned relative to its sensitivity). The naive fix is to widen the final
taper channel ``final_ch``, but the vendored ``refine`` block is two FULL 3x3 convs whose
parameter cost is ``O(final_ch^2)`` — so widening the head to gain capacity at the
d_seg-critical output blows up the rate quadratically. (Verified: refine params 588 -> 2328
-> 9264 for final_ch 8 -> 16 -> 32, i.e. ~4x per 2x width = quadratic.)

THE MECHANISM (this module): a DEPTHWISE-SEPARABLE refine — a per-channel k x k depthwise
conv + a 1x1 pointwise mix (MobileNet/Xception factorization, Howard 2017 / Chollet 2017).
A separable conv approximates a full conv at ``O(C·k^2 + C·C')`` instead of ``O(C·C'·k^2)``.
With this factorization the head can carry MORE EFFECTIVE DEPTH (stack n separable stages)
at cost that grows ~LINEARLY in the number of stages (each stage is ``~C·k^2 + C^2`` and we
keep the pointwise mix at the SAME width so it does not re-introduce the quadratic-in-width
blow-up the full ``refine`` has when you widen ``final_ch``). The point is: gain head
capacity along the DEPTH axis (cheap, O(stages)) rather than the WIDTH axis (the C^2 trap).

THE OPTIONAL SECONDARY (boundary-conditioned low-rank residual): a small rank-r additive
correction to the ``rgb_1`` output, ``U(V(x))`` with V: cf -> r (1x1), U: r -> 3 (1x1) —
capacity concentrated as a low-rank head residual. r is tiny (default 4) so it is cheap;
it exists to spend a few params PURELY on the contested-argmax boundary pixels where d_seg
is decided (ties to Lever-D + the boundary-math seg core). DEFAULT OFF.

PARITY CONTRACT (the faithful-default gate): when ``enabled=False`` (the default) the head
is the EXACT vendored ``refine`` + ``rgb_0`` + ``rgb_1`` path, BIT-IDENTICAL given the same
weights. The separable head is a strictly OPTIONAL A/B arm. The production A/B decides if it
pays — rgb_1's high *output sensitivity* is EXPECTED (it writes the scored pixels) and is
NOT proof of under-capacity; this module proves only the MECHANISM (capacity at O(C) not
O(C^2)) + parity, NOT a d_seg win. No score claim is made here.

NUMPY-PORTABLE FORWARD NOTE (the inflate path is torch-free): every op here is a standard
conv2d / depthwise-conv2d (groups=C) / 1x1 conv / sigmoid / sin — all expressible as
``numpy`` correlate2d-per-channel (depthwise) + tensordot (1x1) + elementwise. A reference
implementation needs only: depthwise = per-channel 2D correlation with the channel's k x k
kernel + bias; pointwise/1x1 = ``einsum('bchw,oc->bohw')`` + bias; sin/sigmoid elementwise.
The default path is the vendored refine (already numpy-portable), so the inflate runtime is
unchanged unless this arm is selected for a byte-closed candidate.

NO originality claim: depthwise-separable convs are Howard 2017 (MobileNets) / Chollet 2017
(Xception); the low-rank head residual is a standard rank-factorized 1x1. This is a TOOL for
the head-capacity A/B, attributed, not a submission.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-used verbatim from configurable_taper_decoder so the two stay in lockstep — the
# parent decoder carries the taper schedule, the parity contract, and the codec round-trip.
from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder


# ---------------------------------------------------------------------------
# param-count helpers (exact; mirror the module construction below)
# ---------------------------------------------------------------------------
def full_refine_param_count(final_ch: int) -> int:
    """Exact param count of the VENDORED full-conv ``refine`` block:
    ``Conv2d(cf, cf//2, 3) + Conv2d(cf//2, cf, 3)`` = ``O(cf^2)`` (the quadratic).
    Matches ``decoder_param_count``'s refine term line-for-line."""
    cf = int(final_ch)
    if cf < 2:
        raise ValueError(f"final_ch must be >= 2 (uses cf//2); got {cf}")
    return cf * (cf // 2) * 9 + (cf // 2) + (cf // 2) * cf * 9 + cf


def separable_refine_param_count(final_ch: int, n_stages: int = 2) -> int:
    """Exact param count of the depthwise-separable ``refine`` with ``n_stages``
    separable stages. Each stage = depthwise Conv2d(cf, cf, 3, groups=cf) [+bias] +
    pointwise Conv2d(cf, cf, 1) [+bias]:
      depthwise : cf*1*3*3 + cf      = 9*cf + cf  = 10*cf
      pointwise : cf*cf*1*1 + cf     = cf^2 + cf
    so per-stage = ``cf^2 + 11*cf`` and total = ``n_stages*(cf^2 + 11*cf)``.

    Note: the pointwise 1x1 is ``O(cf^2)`` PER STAGE, but it is held at the SAME width
    (cf->cf) — so capacity grows ALONG DEPTH (``n_stages``) at ``O(n_stages)``, NOT along
    width (which is the vendored refine's C^2 trap when you widen ``final_ch`` to feed the
    head). The mechanism claim is: add head depth at O(stages); do NOT widen at O(width^2).
    """
    cf = int(final_ch)
    if cf < 1:
        raise ValueError(f"final_ch must be >= 1; got {cf}")
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1; got {n_stages}")
    per_stage = cf * cf + 11 * cf
    return n_stages * per_stage


def lowrank_head_residual_param_count(final_ch: int, rank: int) -> int:
    """Exact param count of the optional boundary low-rank head residual:
    ``V: Conv2d(cf, r, 1) [+bias]`` then ``U: Conv2d(r, 3, 1) [+bias]``:
      V : cf*r + r
      U : r*3 + 3
    = ``cf*r + r + 3*r + 3`` = ``cf*r + 4*r + 3``. O(cf) in width."""
    cf, r = int(final_ch), int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1; got {r}")
    return cf * r + 4 * r + 3


# ---------------------------------------------------------------------------
# the separable refine + optional low-rank head residual
# ---------------------------------------------------------------------------
class DepthwiseSeparableRefine(nn.Module):
    """A depthwise-separable replacement for the vendored full-conv ``refine``.

    ``n_stages`` separable stages, each = depthwise 3x3 (groups=cf) + pointwise 1x1
    (cf->cf). Forward returns a residual the decoder adds as ``x + 0.1*sin(refine(x))``
    (same composition rule as the vendored decoder), so it is a drop-in for ``self.refine``.

    Capacity-at-linear-cost: stacking stages grows params ``O(n_stages)``; the width cf is
    held = ``final_ch`` (NOT widened), so the C^2-in-width blow-up of widening the vendored
    refine is avoided.
    """

    def __init__(self, final_ch: int, n_stages: int = 2, kernel_size: int = 3):
        super().__init__()
        cf = int(final_ch)
        if cf < 1:
            raise ValueError(f"final_ch must be >= 1; got {cf}")
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1; got {n_stages}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}")
        self.final_ch = cf
        self.n_stages = int(n_stages)
        pad = kernel_size // 2
        stages: list[nn.Module] = []
        for _ in range(self.n_stages):
            stages.append(
                nn.ModuleList(
                    [
                        # depthwise: one k x k kernel per channel (groups=cf) -> O(cf)
                        nn.Conv2d(cf, cf, kernel_size, padding=pad, groups=cf),
                        # pointwise: 1x1 cross-channel mix at the SAME width cf -> O(cf^2)/stage
                        nn.Conv2d(cf, cf, 1),
                    ]
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, (dw, pw) in enumerate(self.stages):
            y = pw(dw(out))
            # intermediate non-linearity between stages (NeRF-style sin, matching the
            # decoder's activation family); the LAST stage stays linear so the decoder's
            # own ``x + 0.1*sin(refine(x))`` composition is the activation on the residual.
            out = torch.sin(y) if i < self.n_stages - 1 else y
        return out


class BoundaryLowRankHeadResidual(nn.Module):
    """Optional rank-r additive correction to a 3-channel RGB head output.

    ``U(V(x))``: V is 1x1 cf->r, U is 1x1 r->3. The correction is concentrated in a
    low-rank subspace — a few params spent purely on refining the head where d_seg is
    decided (the contested-argmax boundary). Returns a 3-channel residual ADDED to the
    head's pre-sigmoid logits (so it shifts the argmax boundary without a full conv)."""

    def __init__(self, final_ch: int, rank: int = 4):
        super().__init__()
        cf, r = int(final_ch), int(rank)
        if r < 1:
            raise ValueError(f"rank must be >= 1; got {r}")
        self.v = nn.Conv2d(cf, r, 1)
        self.u = nn.Conv2d(r, 3, 1)
        # zero-init U so the residual starts at EXACTLY zero -> enabling it does not
        # perturb a warm-started head until training moves it (graceful A/B start).
        nn.init.zeros_(self.u.weight)
        nn.init.zeros_(self.u.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(self.v(x))


# ---------------------------------------------------------------------------
# the head-variant decoder wrapper (DEFAULT-OFF, parity-preserving)
# ---------------------------------------------------------------------------
class BoundaryHeadTaperDecoder(ConfigurableTaperHNeRVDecoder):
    """``ConfigurableTaperHNeRVDecoder`` with an OPTIONAL boundary/separable HEAD.

    When ``boundary_head_enabled=False`` (the DEFAULT) this is IDENTICAL to the parent —
    same modules, same state_dict keys, same forward — so it is bit-identical to the
    vendored decoder under the default taper (the parity contract is inherited unchanged).

    When ``boundary_head_enabled=True`` it swaps ``self.refine`` for a
    ``DepthwiseSeparableRefine`` (capacity along depth at O(stages), not width at O(C^2))
    and OPTIONALLY adds a ``BoundaryLowRankHeadResidual`` to the ``rgb_1`` logits. The
    decoder's forward composition (``x + 0.1*sin(refine(x))``; ``f1 = sigmoid(rgb_1(x))``)
    is preserved; only the refine block's internal factorization changes (+ the optional
    rgb_1 residual). This is an OPTIONAL A/B ARM, wired into the production launcher LATER
    by P2/integration — NOT here.
    """

    def __init__(
        self,
        latent_dim: int = 28,
        base_channels: int = 36,
        eval_size: tuple[int, int] = (384, 512),
        channels: list[int] | None = None,
        boundary_head_enabled: bool = False,
        separable_n_stages: int = 2,
        lowrank_residual_enabled: bool = False,
        lowrank_rank: int = 4,
    ):
        super().__init__(
            latent_dim=latent_dim,
            base_channels=base_channels,
            eval_size=eval_size,
            channels=channels,
        )
        self.boundary_head_enabled = bool(boundary_head_enabled)
        self.lowrank_residual_enabled = bool(lowrank_residual_enabled)
        self.separable_n_stages = int(separable_n_stages)
        self.lowrank_rank = int(lowrank_rank)
        final_ch = self.channels[-1]
        if self.boundary_head_enabled:
            # swap the vendored full-conv refine for the depthwise-separable refine.
            self.refine = DepthwiseSeparableRefine(
                final_ch, n_stages=self.separable_n_stages
            )
        if self.lowrank_residual_enabled:
            self.rgb_1_boundary = BoundaryLowRankHeadResidual(
                final_ch, rank=self.lowrank_rank
            )

    def forward(self, z):
        # When disabled, defer ENTIRELY to the parent forward (bit-identical guarantee).
        if not self.boundary_head_enabled and not self.lowrank_residual_enabled:
            return super().forward(z)

        # Enabled path: replicate the parent forward, but the refine is the separable one
        # (already swapped into self.refine) and the rgb_1 logits get the optional residual.
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=True):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        rgb1_logits = self.rgb_1(x)
        if self.lowrank_residual_enabled:
            rgb1_logits = rgb1_logits + self.rgb_1_boundary(x)
        f1 = torch.sigmoid(rgb1_logits) * 255.0
        return torch.stack([f0, f1], dim=1)


__all__ = [
    "BoundaryHeadTaperDecoder",
    "BoundaryLowRankHeadResidual",
    "DepthwiseSeparableRefine",
    "full_refine_param_count",
    "lowrank_head_residual_param_count",
    "separable_refine_param_count",
]
