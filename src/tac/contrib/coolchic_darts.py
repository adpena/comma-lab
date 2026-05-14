# SPDX-License-Identifier: MIT
"""Lane I-DARTS — Cool-Chic dim search via DARTS.

:class:`tac.contrib.coolchic_renderer.CoolChicLatentRenderer` (Lane I)
takes two arch-knob arguments that have been hand-picked since the
profile was committed:

    * ``hidden`` (default 32) — width of the synthesis decoder MLP.
    * ``latent_shapes`` (default ((6,8), (12,16), (24,32))) — the
      multi-resolution latent grid sizes.

This module wraps the renderer in a DARTS supernet over a joint grid
of (hidden_dim, latent_grid_res) so the optimal pair is *learned*.

Candidate set
-------------

hidden_dim       ∈ {8, 16, 24, 32}     (4 widths)
latent_grid_res  ∈ {(8,6,4), (12,8,6), (16,12,8), (24,16,12)}
                                        (4 base widths → fixed
                                         3-band cascade with /2 between
                                         bands; values are W and the
                                         tuple is (W_finest, W_mid,
                                         W_coarse). H is W * 0.75 to
                                         match the 384×512 → 4:3
                                         aspect of the contest video.)
                                          -----
                                          16 candidates

All candidates produce a ``(B, 3, H, W)`` RGB output (the
:class:`CoolChicLatentRenderer` contract is fixed at 3-channel sigmoid
output regardless of internal hidden_dim or latent_shapes), so the
mixture shape is well-defined.

Mathematical rigor
------------------

Cool-Chic / C3 is a self-compression *codec*: rate is dominated by the
quantizable latents and the synthesis decoder weights. The search
trades distortion (more hidden_dim, finer latent grid → lower
distortion) against rate (more parameters → larger renderer.bin).

The discovered (hidden, latent_res) pair is the operating point on the
Cool-Chic rate-distortion frontier that minimizes the DARTS validation
loss — typically a frame reconstruction MSE.

Note on retrain-from-scratch
----------------------------

Per CLAUDE.md "DARTS doesn't transfer weights" rule, the discovered
arch is RETRAINED from scratch. The supernet weights are coupling-noise
(every candidate has been gradient-trained against an averaged-target
that includes the sister candidates). The retrain step uses the
discovered ``(hidden, latent_shapes)`` to instantiate a fresh
:class:`CoolChicLatentRenderer` and trains it with the standard Lane I
recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.contrib.coolchic_renderer import CoolChicLatentRenderer
from tac.darts import (
    DARTSAnnealSchedule,
    DARTSAlphaTrajectory,
    DARTSCell,
    DARTSOptimizer,
)


__all__ = [
    "COOLCHIC_HIDDEN_CANDIDATES",
    "COOLCHIC_LATENT_GRID_CANDIDATES",
    "CoolChicVariant",
    "CoolChicDARTSCell",
    "build_coolchic_supernet",
]


COOLCHIC_HIDDEN_CANDIDATES: tuple[int, ...] = (8, 16, 24, 32)

# Each entry is a 3-band latent cascade: ((H, W) finest, (H, W) mid,
# (H, W) coarse). The aspect ratio (H ≈ 0.75·W) matches the 384×512
# contest video. The /2 between bands matches the original Cool-Chic
# multi-resolution design.
COOLCHIC_LATENT_GRID_CANDIDATES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((6, 8), (3, 4), (2, 3)),       # 32×24 finest equivalent
    ((9, 12), (5, 6), (3, 4)),      # 48×36
    ((12, 16), (6, 8), (4, 6)),     # 64×48 (Lane I default class)
    ((18, 24), (9, 12), (6, 8)),    # 96×72
)


class CoolChicVariant(nn.Module):
    """A :class:`CoolChicLatentRenderer` instance with explicit dims.

    Forward shape contract: ``(B, H, W)`` integer mask → ``(B, 3, H, W)``
    sigmoid'd RGB output. This matches the
    :class:`CoolChicLatentRenderer` contract exactly so DARTS mixing is
    well-defined (the variant is a thin wrapper that pins the dims).
    """

    def __init__(
        self,
        hidden: int,
        latent_shapes: tuple[tuple[int, int], ...],
        *,
        num_classes: int = 5,
        embed_dim: int = 6,
        latent_ch: int = 8,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.latent_shapes = tuple((int(h), int(w)) for h, w in latent_shapes)
        self.renderer = CoolChicLatentRenderer(
            num_classes=num_classes,
            class_embed_dim=embed_dim,
            latent_ch=latent_ch,
            hidden=self.hidden,
            latent_shapes=self.latent_shapes,
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.renderer(masks)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CoolChicDARTSCell(DARTSCell):
    """:class:`DARTSCell` over (hidden_dim, latent_grid_res) pairs.

    Args:
        hidden_candidates: candidate hidden_dim widths.
        latent_grid_candidates: candidate 3-band latent cascades.
    """

    def __init__(
        self,
        hidden_candidates: tuple[int, ...] = COOLCHIC_HIDDEN_CANDIDATES,
        latent_grid_candidates: tuple[tuple[tuple[int, int], ...], ...]
            = COOLCHIC_LATENT_GRID_CANDIDATES,
        *,
        num_classes: int = 5,
        embed_dim: int = 6,
        latent_ch: int = 8,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        if len(hidden_candidates) < 2 or len(latent_grid_candidates) < 2:
            raise ValueError(
                f"CoolChicDARTSCell needs ≥ 2 hidden AND ≥ 2 latent grid "
                f"candidates, got {len(hidden_candidates)} × "
                f"{len(latent_grid_candidates)}"
            )
        candidate_specs: list[tuple[int, tuple[tuple[int, int], ...]]] = [
            (h, lg)
            for h in hidden_candidates
            for lg in latent_grid_candidates
        ]
        ops = [
            CoolChicVariant(
                hidden=h,
                latent_shapes=lg,
                num_classes=num_classes,
                embed_dim=embed_dim,
                latent_ch=latent_ch,
            )
            for (h, lg) in candidate_specs
        ]
        # Compact human-readable name = h + finest-band W (the human-
        # interpretable summary of the 3-band cascade).
        names = tuple(
            f"hidden_{h}_grid_{lg[0][1]}x{lg[0][0]}"
            for (h, lg) in candidate_specs
        )
        super().__init__(ops=ops, anneal=anneal, names=names)
        self.candidate_specs = tuple(candidate_specs)

    def discrete_arch_spec(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        return self.candidate_specs[self.discrete_arch()]

    def candidate_param_counts(self) -> dict[str, int]:
        return {name: int(op.param_count()) for name, op in zip(self.names, self.ops)}


# ── Stand-alone supernet for tests + the search loop ───────────────────


class CoolChicSupernet(nn.Module):
    """Single-cell supernet wrapping :class:`CoolChicDARTSCell`."""

    def __init__(
        self,
        hidden_candidates: tuple[int, ...] = COOLCHIC_HIDDEN_CANDIDATES,
        latent_grid_candidates: tuple[tuple[tuple[int, int], ...], ...]
            = COOLCHIC_LATENT_GRID_CANDIDATES,
        num_classes: int = 5,
        embed_dim: int = 6,
        latent_ch: int = 8,
        anneal: DARTSAnnealSchedule | None = None,
    ):
        super().__init__()
        self.cell = CoolChicDARTSCell(
            hidden_candidates=hidden_candidates,
            latent_grid_candidates=latent_grid_candidates,
            num_classes=num_classes,
            embed_dim=embed_dim,
            latent_ch=latent_ch,
            anneal=anneal,
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.cell(masks)

    def temperature_anneal(self, epoch: int, total_epochs: int) -> float:
        return self.cell.temperature_anneal(epoch, total_epochs)

    def discrete_arch_spec(self):
        return self.cell.discrete_arch_spec()


def build_coolchic_supernet(
    hidden_candidates: tuple[int, ...] = COOLCHIC_HIDDEN_CANDIDATES,
    latent_grid_candidates: tuple[tuple[tuple[int, int], ...], ...]
        = COOLCHIC_LATENT_GRID_CANDIDATES,
    num_classes: int = 5,
    embed_dim: int = 6,
    latent_ch: int = 8,
    anneal: DARTSAnnealSchedule | None = None,
) -> CoolChicSupernet:
    return CoolChicSupernet(
        hidden_candidates=hidden_candidates,
        latent_grid_candidates=latent_grid_candidates,
        num_classes=num_classes,
        embed_dim=embed_dim,
        latent_ch=latent_ch,
        anneal=anneal,
    )


def build_coolchic_arch_optimizer(
    supernet: CoolChicSupernet,
    lr: float = 3e-4,
) -> DARTSOptimizer:
    return DARTSOptimizer(arch_params=supernet.cell.arch_parameters(), lr=lr)


def make_trajectory(cell: CoolChicDARTSCell) -> DARTSAlphaTrajectory:
    return DARTSAlphaTrajectory(op_names=cell.names)
