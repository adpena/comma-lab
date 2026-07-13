# SPDX-License-Identifier: MIT
"""Typed, default-off contract for the #484 PRE-SE composition reopen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tac.witness_dsl.pre_se_locus_policy_20260713 import PreSELocusPolicy


@dataclass(frozen=True)
class PreSEMultiSourceReopenPolicy:
    """Seal the two coupled reopen bars without authorizing a live lever."""

    mode: str = "pre_se_multi_source_reopen_20260713"
    base: PreSELocusPolicy = field(default_factory=PreSELocusPolicy)
    feature_sources: tuple[str, ...] = (
        "base-42-shallow-chart",
        "block2-pre-se-144",
        "block3-pre-se-288",
        "sensitivity-2",
    )
    feature_count: int = 476
    tile_grid: tuple[int, int] = (2, 2)
    tile_halo_alignment: int = 8
    donated_global_mode: str = "full-frame-SE-gates-once-then-broadcast"
    prior_output: str = "experiments/results/pre_se_locus_20260713"
    source_round5_output: str = (
        "experiments/results/replace_round5_deeper_nonlinear_20260713"
    )
    local_cached_measurement_only: bool = True
    live_training_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.mode != "pre_se_multi_source_reopen_20260713":
            raise ValueError("multi-source reopen mode is sealed")
        if self.feature_count != 42 + 144 + 288 + 2:
            raise ValueError("multi-source feature width drifted")
        if self.tile_grid != (2, 2) or self.tile_halo_alignment != 8:
            raise ValueError("tile proof geometry drifted")
        if not self.local_cached_measurement_only:
            raise ValueError("#484 reopen has local cached authority only")
        if self.live_training_enabled or self.paid_or_remote_dispatch_enabled:
            raise ValueError("#484 reopen cannot actuate training or dispatch")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local costate evidence cannot claim score or promotion")

    def compile_measurement_contract(self) -> dict[str, Any]:
        """Compile the inherited apples-to-apples contract and its only deltas."""

        inherited = self.base.compile_measurement_contract()
        base = self.base.base
        return {
            **inherited,
            "mode": self.mode,
            "feature_sources": list(self.feature_sources),
            "feature_count": self.feature_count,
            "feature_source_rule": (
                "concatenate shared base-42 once, block2 PRE-SE 144, block3 PRE-SE 288, "
                "and shared sensitivity-2 once; no other chart, support, fit, or metric change"
            ),
            "retained_mass_bar": base.retained_mass_bar,
            "requested_area_fraction": base.requested_area_fraction,
            "realized_area_fraction": base.realized_area_fraction,
            "selected_prefix_cells": base.selected_prefix_cells,
            "round4_oracle_retained_mass": base.compile_measurement_contract()[
                "round4_oracle_retained_mass"
            ],
            "convex_solver": inherited["convex_solver"],
            "nonlinear_training": (
                "unchanged three deterministic 476-to-32-to-20 pair-gated ReLU MLP seeds; "
                "unchanged train-only dev retained-mass early stopping"
            ),
            "tileability_proof": {
                "mode": self.donated_global_mode,
                "grid": list(self.tile_grid),
                "halo": "derive receptive field from the executable encoder graph then round up to stride",
                "pass_rule": (
                    "all composed PRE-SE core tiles equal the corresponding full-frame tensors "
                    "with the same full-frame SE gates donated; charge unique reductions and SE MLPs once"
                ),
            },
            "reopen_rule": (
                "retained mass >= 0.47 on either sealed rung AND "
                "tileable-modulo-cheap-globals confirmed"
            ),
            "retained_mass_failure_rule": (
                "RETAINED-MASS-FAMILY-KILL only with req-R evidence covering at least two "
                "formulations plus the structural information-bottleneck reason"
            ),
            "prior_output": self.prior_output,
            "source_round5_output": self.source_round5_output,
            "live_trainer_argv": [],
            "constant_provenance": {
                **inherited["constant_provenance"],
                "composition": "SOURCE operator directive 2026-07-13",
                "all_bars_splits_and_optimizers": "MEASURED-INHERITED protected PRE-SE n600 receipt",
                "tile_halo": "DERIVED from executable convolution kernel/stride recurrence",
                "SE_union": "MEASURED from executable module order; shared ancestors deduplicated",
            },
        }


__all__ = ["PreSEMultiSourceReopenPolicy"]
