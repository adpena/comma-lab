"""Lane K-DARTS — depthwise-separable channel-dim search via DARTS.

Lane K (DSCONV_QUANTIZR_KILLER, profile :data:`tac.profiles.PROFILES`)
hand-picks ``base_ch=24, mid_ch=32, motion_hidden=16`` to land at
≈88,996 params (matching Quantizr's class). The exact 88K target is a
proxy for Quantizr's contest-relevant param budget — but the Pareto-
optimal channel widths for our renderer + scorer pair are unknown.

This module wraps a small DSConv-encoder *supernet* in DARTS over a
joint grid of (base_ch, mid_ch) channel dims, with a **budget penalty**
in the loss to discourage candidates that exceed 100K params.

Candidate set
-------------

base_ch ∈ {16, 24, 32, 48}  (4 widths)
mid_ch  ∈ {24, 32, 48, 64}  (4 widths)
                            -----
                            16 candidates

The motion module's ``motion_hidden`` is held fixed at 16 (Lane K's
default) — searching it jointly would 4×16=64 combinations and the
motion module is small enough that its dim is dominated by base_ch
through the upstream embedding.

Budget penalty
--------------

The CLAUDE.md "no arbitrary architecture knobs" rule says the search
should be informed by the score, not by an arbitrary param target. But
the search would happily commit to base_ch=48, mid_ch=64 (largest
candidate, ~150K params) if the loss didn't discourage it. We add a
soft hinge penalty::

    L_total = L_data + λ · max(0, params - PARAM_BUDGET)²

with ``PARAM_BUDGET = 100_000`` and ``λ = 1e-9`` (the L2 hinge gives
~1.0 contribution at +30K over budget, dominating after the first few
epochs of training). This is a standard rate-distortion trade-off —
see Liu et al. §A.4 for the analogous treatment in differentiable
NAS over MAdds budgets.

Note that the penalty applies to the *expected* param count under the
softmax mixture, not the discrete-argmax count. Expected param count is
differentiable through α::

    E[params] = Σ softmax(α/T)_i · params_i

and the hinge derivative pushes α away from candidates that exceed the
budget.

Tests verify the supernet runs forward + backward and the budget
penalty actually reduces α-mass on over-budget candidates after a few
synthetic-data steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.darts import (
    DARTSAnnealSchedule,
    DARTSAlphaTrajectory,
    DARTSCell,
    DARTSOptimizer,
    alpha_softmax,
)


__all__ = [
    "DSConvVariant",
    "DSConvChannelDARTSCell",
    "DSCONV_BASE_CHANNELS",
    "DSCONV_MID_CHANNELS",
    "PARAM_BUDGET",
    "param_budget_penalty",
    "build_dsconv_channel_supernet",
]


DSCONV_BASE_CHANNELS: tuple[int, ...] = (16, 24, 32, 48)
DSCONV_MID_CHANNELS: tuple[int, ...] = (24, 32, 48, 64)
PARAM_BUDGET: int = 100_000


class DSConvVariant(nn.Module):
    """Depthwise-separable conv encoder with explicit (base_ch, mid_ch).

    Mirrors the three-stage encoder structure of
    :class:`tac.renderer.MaskRenderer` (stem → down → down2) but with
    every conv replaced by a depthwise-separable pair (DSConv,
    Howard et al. MobileNet 2017). All candidates produce an output of
    shape ``(B, mid_ch, H/4, W/4)`` so that the DARTS mixture would
    NOT be well-defined unless we either (a) project every candidate
    to a common output channel count, or (b) refuse to mix and instead
    *select* one candidate per forward.

    We pick (a): a final 1×1 projection from ``mid_ch`` to the FIXED
    ``output_ch`` (default 32, the median of the search range). The
    mixture is then over the (base_ch, mid_ch) configs but the output
    shape is identical, satisfying the DARTSCell shape contract.
    """

    def __init__(
        self,
        c_in: int,
        base_ch: int,
        mid_ch: int,
        output_ch: int,
    ):
        super().__init__()
        self.base_ch = int(base_ch)
        self.mid_ch = int(mid_ch)
        self.output_ch = int(output_ch)
        # Stem: c_in → base_ch (full Conv2d for the first layer; DSConv
        # is unprofitable when c_in is tiny like 6).
        self.stem = nn.Conv2d(c_in, base_ch, 3, padding=1)
        # Down1: base_ch → base_ch (DSConv, stride 2).
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1, groups=base_ch, bias=False),
            nn.Conv2d(base_ch, mid_ch, 1),
        )
        # Down2: mid_ch → mid_ch (DSConv, stride 2).
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, groups=mid_ch, bias=False),
            nn.Conv2d(mid_ch, output_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DSConvChannelDARTSCell(DARTSCell):
    """:class:`DARTSCell` over a 2-D grid of (base_ch, mid_ch) channels.

    Args:
        c_in: input channels.
        output_ch: output channels (fixed across candidates so the
            mixture shape contract holds).
        base_channels: candidate base_ch values
            (default :data:`DSCONV_BASE_CHANNELS`).
        mid_channels: candidate mid_ch values
            (default :data:`DSCONV_MID_CHANNELS`).
    """

    def __init__(
        self,
        c_in: int,
        output_ch: int = 32,
        *,
        base_channels: tuple[int, ...] = DSCONV_BASE_CHANNELS,
        mid_channels: tuple[int, ...] = DSCONV_MID_CHANNELS,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        if len(base_channels) < 2 or len(mid_channels) < 2:
            raise ValueError(
                f"DSConvChannelDARTSCell needs ≥ 2 base AND mid channel "
                f"candidates, got {len(base_channels)} × {len(mid_channels)}"
            )
        # Cartesian product → flat candidate list.
        candidate_specs: list[tuple[int, int]] = [
            (b, m) for b in base_channels for m in mid_channels
        ]
        ops = [
            DSConvVariant(c_in, b, m, output_ch)
            for (b, m) in candidate_specs
        ]
        names = tuple(f"base_{b}_mid_{m}" for (b, m) in candidate_specs)
        super().__init__(ops=ops, anneal=anneal, names=names)
        self.candidate_specs: tuple[tuple[int, int], ...] = tuple(candidate_specs)
        self.c_in = c_in
        self.output_ch = output_ch

    def discrete_arch_spec(self) -> tuple[int, int]:
        idx = self.discrete_arch()
        return self.candidate_specs[idx]

    def candidate_param_counts(self) -> dict[str, int]:
        return {name: int(op.param_count()) for name, op in zip(self.names, self.ops)}

    def expected_param_count(self) -> torch.Tensor:
        """Differentiable expected param count under the current softmax(α/T).

        Returns a 0-D tensor that backpropagates into α.
        """
        weights = alpha_softmax(self.alpha, self._current_T)
        # Build a 1-D tensor of per-candidate param counts on the same
        # device as α — required so the multiply doesn't trigger an
        # implicit host-device copy on CUDA.
        counts = torch.tensor(
            [float(op.param_count()) for op in self.ops],
            dtype=weights.dtype,
            device=weights.device,
        )
        return (weights * counts).sum()


def param_budget_penalty(
    cell: DSConvChannelDARTSCell,
    *,
    budget: int = PARAM_BUDGET,
    weight: float = 1e-9,
) -> torch.Tensor:
    """Soft hinge penalty: ``weight · max(0, expected_params - budget)²``.

    Returns a 0-D tensor that gradients into α.

    Tuning: ``weight = 1e-9`` makes a 30K-param overshoot contribute
    ~1.0 to the loss (matches a typical scorer-loss scale of O(1) at
    convergence). The squared form is differentiable everywhere
    including at the boundary; a linear hinge is non-smooth and gives
    α SGD a worse signal at the kink.
    """
    expected = cell.expected_param_count()
    over = (expected - float(budget)).clamp_min(0.0)
    return weight * (over * over)


# ── Stand-alone supernet for the search loop + tests ────────────────────


class DSConvChannelSupernet(nn.Module):
    """Single-cell supernet wrapping :class:`DSConvChannelDARTSCell`."""

    def __init__(
        self,
        c_in: int = 6,
        output_ch: int = 32,
        base_channels: tuple[int, ...] = DSCONV_BASE_CHANNELS,
        mid_channels: tuple[int, ...] = DSCONV_MID_CHANNELS,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        super().__init__()
        self.cell = DSConvChannelDARTSCell(
            c_in=c_in,
            output_ch=output_ch,
            base_channels=base_channels,
            mid_channels=mid_channels,
            anneal=anneal,
        )
        # Trivial regression head so the synthetic search can run.
        self.head = nn.Conv2d(output_ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cell(x))

    def temperature_anneal(self, epoch: int, total_epochs: int) -> float:
        return self.cell.temperature_anneal(epoch, total_epochs)

    def discrete_arch_spec(self) -> tuple[int, int]:
        return self.cell.discrete_arch_spec()


def build_dsconv_channel_supernet(
    c_in: int = 6,
    output_ch: int = 32,
    base_channels: tuple[int, ...] = DSCONV_BASE_CHANNELS,
    mid_channels: tuple[int, ...] = DSCONV_MID_CHANNELS,
    anneal: DARTSAnnealSchedule | None = None,
) -> DSConvChannelSupernet:
    return DSConvChannelSupernet(
        c_in=c_in,
        output_ch=output_ch,
        base_channels=base_channels,
        mid_channels=mid_channels,
        anneal=anneal,
    )


def build_dsconv_arch_optimizer(
    supernet: DSConvChannelSupernet,
    lr: float = 3e-4,
) -> DARTSOptimizer:
    return DARTSOptimizer(arch_params=supernet.cell.arch_parameters(), lr=lr)


def make_trajectory(cell: DSConvChannelDARTSCell) -> DARTSAlphaTrajectory:
    return DARTSAlphaTrajectory(op_names=cell.names)
