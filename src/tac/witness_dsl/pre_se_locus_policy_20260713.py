# SPDX-License-Identifier: MIT
"""Typed, default-off policy for the Round-5 PRE-SE feature-locus probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tac.witness_dsl.replace_round5_deeper_nonlinear_policy import (
    ReplaceRound5DeeperNonlinearPolicy,
)


@dataclass(frozen=True)
class PreSELocusSpec:
    """One independently fitted feature-source locus."""

    name: str
    stage_index: int
    block_index: int
    module: str
    channels: int
    spatial_divisor: int

    @property
    def feature_count(self) -> int:
        # Round-5 base-42 + this PRE-SE tensor + sensitivity-2.
        return 42 + self.channels + 2


LOCUS_SPECS = (
    PreSELocusSpec(
        name="block2-pre-se",
        stage_index=1,
        block_index=2,
        module="encoder.model.blocks.1.2.se",
        channels=144,
        spatial_divisor=2,
    ),
    PreSELocusSpec(
        name="block3-pre-se",
        stage_index=2,
        block_index=2,
        module="encoder.model.blocks.2.2.se",
        channels=288,
        spatial_divisor=4,
    ),
)


@dataclass(frozen=True)
class PreSELocusPolicy:
    """Freeze the only allowed delta from Round 5: the feature tap."""

    mode: str = "pre_se_locus_20260713"
    base: ReplaceRound5DeeperNonlinearPolicy = field(
        default_factory=ReplaceRound5DeeperNonlinearPolicy
    )
    loci: tuple[PreSELocusSpec, ...] = LOCUS_SPECS
    source_round5_receipt: str = (
        "experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json"
    )
    source_round5_targets: str = (
        "experiments/results/replace_round5_deeper_nonlinear_20260713/train_targets"
    )
    fresh_heldout_exact_calls: int = 120
    inherited_train_exact_targets: int = 480
    live_training_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.mode != "pre_se_locus_20260713":
            raise ValueError("PRE-SE policy mode is sealed")
        if self.loci != LOCUS_SPECS:
            raise ValueError("PRE-SE loci are sealed")
        if (self.inherited_train_exact_targets, self.fresh_heldout_exact_calls) != (480, 120):
            raise ValueError("the n600 exact-target custody split is sealed")
        if self.live_training_enabled or self.paid_or_remote_dispatch_enabled:
            raise ValueError("PRE-SE probe has local measurement authority only")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local costate evidence cannot claim score or promotion")

    def compile_measurement_contract(self) -> dict[str, Any]:
        """Compile the inherited Round-5 contract with only locus fields replaced."""

        base = self.base.compile_measurement_contract()
        payload = {
            **base,
            "mode": self.mode,
            "loci": [asdict(spec) | {"feature_count": spec.feature_count} for spec in self.loci],
            "cut_modules": [spec.module for spec in self.loci],
            "cut_names": [spec.name for spec in self.loci],
            "deep_channels": [spec.channels for spec in self.loci],
            "deep_feature_count_by_locus": {
                spec.name: spec.feature_count for spec in self.loci
            },
            "source_round5_receipt": self.source_round5_receipt,
            "source_round5_targets": self.source_round5_targets,
            "fresh_heldout_exact_calls": self.fresh_heldout_exact_calls,
            "inherited_train_exact_targets": self.inherited_train_exact_targets,
            "feature_source_rule": (
                "fit block2-pre-se and block3-pre-se independently; each uses the same base-42 "
                "chart and sensitivity-2 columns as Round 5"
            ),
            "tap_rule": (
                "forward-pre input of the last MBConv SE in each Round-5 encoder stage; "
                "this is the post-depthwise activation before that MBConv's own SE"
            ),
            "tileability_gate": (
                "PASS only if the tap has no upstream full-frame reduction dependency; "
                "being before its own SE is necessary but not sufficient"
            ),
            "convex_solver": base["convex_solver"],
            "nonlinear_training": (
                "same Round-5 three deterministic pair-gated 32-hidden ReLU MLPs, fitted "
                "separately at each locus with train-only dev retained-mass early stop"
            ),
            "primary_metric": base["primary_metric"],
            "live_trainer_argv": [],
            "constant_provenance": {
                **base["constant_provenance"],
                "pre_se_loci": (
                    "SOURCE operator 2026-07-13 precise untested cell; DERIVED live timm "
                    "EfficientNet-B2 module graph"
                ),
                "n600_target_reuse": (
                    "MEASURED-INHERITED immutable Round-5 480 train targets plus fresh exact "
                    "120 heldout targets under the identical deterministic assignment"
                ),
                "all_other_rules": "MEASURED-INHERITED Round-5 sealed contract",
            },
        }
        return payload


__all__ = ["LOCUS_SPECS", "PreSELocusPolicy", "PreSELocusSpec"]
