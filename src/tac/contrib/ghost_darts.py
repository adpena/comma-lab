"""Lane GH-DARTS — Ghost convolution ratio search via DARTS.

Lane GH ships :class:`GhostConv2d` (Han et al. CVPR 2020) with
``ratio=2`` hard-coded. The "halve params" choice is conventional but
arbitrary at the chosen base_ch=36, mid_ch=60. Per CLAUDE.md "no
arbitrary architecture knobs" — this module wraps :class:`GhostConv2d`
in a :class:`DARTSCell` over a discrete set of ratios so the optimal
ratio is *learned*, not picked.

Candidate set
-------------

ratio ∈ {1.5, 2.0, 2.5, 3.0, 4.0}

Why these five?
  * **1.5** — softer reduction; keeps more channel-mixing capacity.
  * **2.0** — Han et al.'s default; the literature anchor.
  * **2.5, 3.0, 4.0** — progressively tighter parameter budgets.

Note that :class:`GhostConv2d` enforces ``ratio ≥ 2`` because its
implementation uses a ceil-divide that requires integer arithmetic. We
relax this guard inside :class:`GhostConvVariant` by snapping
non-integer ratios to ``ceil`` for the c_intrinsic computation, while
still tracking the *fractional* ratio for parameter-count accounting.
For ratio=1.5 the ceil-divide gives c_intrinsic = ceil(c_out / 1.5),
which is approximately 0.67·c_out — roughly the same effective param
budget as a non-ghost Conv2d with c_out·0.67 channels.

Geometry
--------

Each candidate produces an output tensor of shape ``(B, c_out, H, W)``
matching the original :class:`GhostConv2d`. The DARTS mixture is
therefore well-defined: ``out = Σ softmax(α/T)_i · ghost_i(x)``.

Param count of the supernet
---------------------------

The supernet *during search* contains all 5 candidates simultaneously.
Total params ≈ Σ_i params(ghost_i). The discovered arch (after retrain
from scratch) uses only the argmax candidate — back to Lane A's normal
~144K renderer footprint.

Tests verify forward/backward + that α converges decisively (KL ≥ 2 nats)
on a synthetic regression toward a known-best ratio.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tac.darts import (
    DARTSAnnealSchedule,
    DARTSCell,
    DARTSAlphaTrajectory,
    DARTSOptimizer,
)


__all__ = [
    "GhostConvVariant",
    "GhostRatioDARTSCell",
    "GHOST_RATIO_CANDIDATES",
    "build_ghost_ratio_supernet",
]


# Discovered-set anchor. Locked here so the test contract is stable.
GHOST_RATIO_CANDIDATES: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0)


class GhostConvVariant(nn.Module):
    """GhostConv2d-equivalent that accepts fractional ratios.

    For ``ratio < 2`` the original :class:`tac.renderer.GhostConv2d`
    raises (its ceil-divide assumes integer reduction). Here we use
    ``c_intrinsic = ceil(c_out / ratio)`` directly, which produces a
    well-defined Conv2d for any real ratio ≥ 1.

    Forward shape contract: ``(B, c_in, H, W) → (B, c_out, H', W')``
    where ``H', W'`` match a standard ``Conv2d(c_in, c_out, kernel,
    stride=stride, padding=padding)`` with the same kwargs.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        *,
        ratio: float,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if ratio < 1.0:
            raise ValueError(f"GhostConvVariant ratio must be ≥ 1.0, got {ratio}")
        # Per Han et al. CVPR 2020 §3.1 Eq. 1: each intrinsic feature map
        # produces (ratio-1) ghost feature maps via cheap linear ops, so
        # total output = c_intrinsic + c_intrinsic·(ratio-1) = c_intrinsic·ratio.
        # We invert: c_intrinsic = ceil(c_out / ratio). The ghost branch
        # must then produce ENOUGH channels so that c_intrinsic + ghost ≥ c_out
        # — we always slice down to c_out at the end.
        #
        # For fractional ratios (e.g. 2.5), use a multi-group ghost branch:
        # ghost_out_per_intrinsic = ceil((c_out - c_intrinsic) / c_intrinsic),
        # then slice. This guarantees the cat is ≥ c_out for any ratio ≥ 1.
        c_intrinsic = max(1, math.ceil(c_out / ratio))
        c_intrinsic = min(c_intrinsic, c_out)
        # How many ghost feature maps must each intrinsic produce so that
        # the cat covers c_out? At minimum 1 (so the ghost branch is real).
        ghost_needed = max(0, c_out - c_intrinsic)
        if ghost_needed == 0:
            # ratio=1: vacuous ghost (no params, identity slice). We still
            # construct a tiny ghost-equivalent conv (1 ch) so the param-
            # counting story is non-degenerate, but it contributes 0 chans.
            ghost_per = 0
        else:
            ghost_per = max(1, math.ceil(ghost_needed / c_intrinsic))
        self.c_out = c_out
        self.c_intrinsic = c_intrinsic
        self.ghost_per_intrinsic = ghost_per
        self.ratio = float(ratio)
        # Primary conv at the requested spatial resolution (handles stride).
        self.primary = nn.Conv2d(
            c_in,
            c_intrinsic,
            kernel,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        # Cheap depthwise ghost branch (Han et al. §3.1: depthwise 3x3 over
        # intrinsic maps, stride=1). With ghost_per > 1 we use groups=
        # c_intrinsic so each intrinsic channel produces ghost_per ghost
        # channels via its OWN depthwise filter (this is exactly the
        # "linear transformation per intrinsic" prescribed in Eq. 1).
        if ghost_per > 0:
            self.ghost = nn.Conv2d(
                c_intrinsic,
                c_intrinsic * ghost_per,
                3,
                stride=1,
                padding=1,
                groups=c_intrinsic,
                bias=False,
                padding_mode=padding_mode,
            )
        else:
            self.ghost = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intrinsic = self.primary(x)
        if self.ghost is not None:
            ghost = self.ghost(intrinsic)
            out = torch.cat([intrinsic, ghost], dim=1)
        else:
            out = intrinsic
        # Always slice to c_out (the cat may overshoot for some ratios).
        if out.shape[1] != self.c_out:
            out = out[:, : self.c_out]
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GhostRatioDARTSCell(DARTSCell):
    """:class:`DARTSCell` over Ghost convolution ratios.

    Each candidate is a :class:`GhostConvVariant` with the same I/O
    shape but a different ratio. The mixture is the standard DARTS
    softmax-weighted sum.

    Args (forwarded to every candidate):
        c_in: input channels
        c_out: output channels
        kernel: primary conv kernel size
        stride / padding / bias / padding_mode: standard Conv2d kwargs
        candidate_ratios: tuple of N ratios to search over (default
            :data:`GHOST_RATIO_CANDIDATES`)
        anneal: temperature schedule (default
            :class:`DARTSAnnealSchedule`)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        candidate_ratios: tuple[float, ...] = GHOST_RATIO_CANDIDATES,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        if len(candidate_ratios) < 2:
            raise ValueError(
                f"GhostRatioDARTSCell needs ≥ 2 candidate ratios, got {len(candidate_ratios)}"
            )
        ops = [
            GhostConvVariant(
                c_in,
                c_out,
                kernel,
                ratio=r,
                stride=stride,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
            )
            for r in candidate_ratios
        ]
        names = tuple(f"ratio_{r:g}" for r in candidate_ratios)
        super().__init__(ops=ops, anneal=anneal, names=names)
        self.candidate_ratios: tuple[float, ...] = tuple(float(r) for r in candidate_ratios)
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel

    def discrete_arch_ratio(self) -> float:
        """Return the discovered ghost ratio (for retraining-from-scratch)."""
        return self.candidate_ratios[self.discrete_arch()]

    def candidate_param_counts(self) -> dict[str, int]:
        """Per-candidate parameter count — for budget-vs-score reporting."""
        return {
            name: int(op.param_count())
            for name, op in zip(self.names, self.ops)
        }


# ── Convenience: 3-cell supernet for the Lane GH stem/down/down2 ────────


class GhostRatioSupernet(nn.Module):
    """Tiny stand-alone supernet used by the search loop and tests.

    Mirrors the three GhostConv2d insertion sites in
    :class:`tac.renderer.MaskRenderer` (stem, down, down2). For the
    *full* search you would integrate :class:`GhostRatioDARTSCell` into
    the actual MaskRenderer; this stand-alone supernet is what the test
    harness exercises and what the synthetic-data convergence check
    runs through.

    Args:
        c_in: input channels (e.g. 6 = embed_dim for Lane GH).
        widths: tuple of (stem, down, down2) output widths. Defaults to
            the Lane GH baseline (36, 60, 60).
        candidate_ratios: passed to every :class:`GhostRatioDARTSCell`.
    """

    def __init__(
        self,
        c_in: int = 6,
        widths: tuple[int, int, int] = (36, 60, 60),
        candidate_ratios: tuple[float, ...] = GHOST_RATIO_CANDIDATES,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        super().__init__()
        w0, w1, w2 = widths
        self.stem = GhostRatioDARTSCell(
            c_in, w0, kernel=3, padding=1, candidate_ratios=candidate_ratios, anneal=anneal,
        )
        self.down = GhostRatioDARTSCell(
            w0, w1, kernel=3, stride=2, padding=1, candidate_ratios=candidate_ratios, anneal=anneal,
        )
        self.down2 = GhostRatioDARTSCell(
            w1, w2, kernel=3, stride=2, padding=1, candidate_ratios=candidate_ratios, anneal=anneal,
        )
        # A trivial regression head so the synthetic search can run a
        # full forward → loss → backward cycle without the rest of the
        # MaskRenderer plumbing.
        self.head = nn.Conv2d(w2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down(x)
        x = self.down2(x)
        return self.head(x)

    def temperature_anneal(self, epoch: int, total_epochs: int) -> float:
        """Anneal every cell's temperature in lock-step."""
        T = None
        for cell in (self.stem, self.down, self.down2):
            T = cell.temperature_anneal(epoch, total_epochs)
        return float(T)  # type: ignore[arg-type]

    def discrete_archs(self) -> tuple[float, float, float]:
        return (
            self.stem.discrete_arch_ratio(),
            self.down.discrete_arch_ratio(),
            self.down2.discrete_arch_ratio(),
        )


def build_ghost_ratio_supernet(
    c_in: int = 6,
    widths: tuple[int, int, int] = (36, 60, 60),
    candidate_ratios: tuple[float, ...] = GHOST_RATIO_CANDIDATES,
    anneal: DARTSAnnealSchedule | None = None,
) -> GhostRatioSupernet:
    """Construct a :class:`GhostRatioSupernet` with explicit defaults."""
    return GhostRatioSupernet(
        c_in=c_in,
        widths=widths,
        candidate_ratios=candidate_ratios,
        anneal=anneal,
    )


# ── Trajectory-recorder convenience for callers ─────────────────────────


def make_trajectory(
    cell: GhostRatioDARTSCell,
) -> DARTSAlphaTrajectory:
    """Construct a :class:`DARTSAlphaTrajectory` keyed by ratio names."""
    return DARTSAlphaTrajectory(op_names=cell.names)


def build_ghost_arch_optimizer(
    supernet: GhostRatioSupernet,
    lr: float = 3e-4,
) -> DARTSOptimizer:
    """Convenience: a :class:`DARTSOptimizer` over every cell's α."""
    arch_params: list[nn.Parameter] = []
    for cell in (supernet.stem, supernet.down, supernet.down2):
        arch_params.extend(cell.arch_parameters())
    return DARTSOptimizer(arch_params=arch_params, lr=lr)
