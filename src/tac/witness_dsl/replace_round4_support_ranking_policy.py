# SPDX-License-Identifier: MIT
"""Typed, default-off policy for the round-4 support-ranking replay."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ROUND3_ORACLE_RETAINED_MASS = 0.5278150212253758
ROUND3_PREFIX_FLOP_FRACTION = 0.005714118050141177
REQUESTED_AREA_FRACTION = 0.047
PREFIX_CELL_COUNT = 192 * 256
SELECTED_PREFIX_CELLS = math.ceil(REQUESTED_AREA_FRACTION * PREFIX_CELL_COUNT)
REALIZED_AREA_FRACTION = SELECTED_PREFIX_CELLS / PREFIX_CELL_COUNT
RETAINED_MASS_BAR = 0.47
CALIBRATION_BIN_COUNT = 16
HELDOUT_ECE_REFUSAL_BAR = 0.05


@dataclass(frozen=True)
class ReplaceRound4SupportRankingPolicy:
    """Seal the finite convex ladder before any round-4 teacher call."""

    mode: str = "replace_round4_support_ranking"
    seed: int = 455
    n_pairs: int = 600
    checkpoint_count: int = 3
    holdout_period: int = 5
    train_lattice_stride_on_prefix: int = 4
    heldout_lattice_stride_on_prefix: int = 1
    teacher_batch_size: int = 1
    requested_area_fraction: float = REQUESTED_AREA_FRACTION
    retained_mass_bar: float = RETAINED_MASS_BAR
    calibration_bin_count: int = CALIBRATION_BIN_COUNT
    heldout_ece_refusal_bar: float = HELDOUT_ECE_REFUSAL_BAR
    rung_order: tuple[str, ...] = (
        "weighted-topk-global-84",
        "weighted-topk-pair-block-44",
        "pairwise-rank-pair-block-44",
    )
    source_class_sensitivity: tuple[float, ...] = (2.2, 32.0, 0.26, 1.0, 0.0)
    prefix_module: str = "encoder.model.blocks.0.0.bn1"
    prefix_cut: str = "pre_squeeze_excite_pre_global_pool"
    global_feature_count: int = 84
    block_feature_count: int = 44
    ordered_pair_count: int = 20
    fallback: str = "full_exact_teacher"
    live_training_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    fore_weighting_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode != "replace_round4_support_ranking":
            raise ValueError("round-4 policy mode is sealed")
        if (self.seed, self.n_pairs, self.checkpoint_count) != (455, 600, 3):
            raise ValueError("seed, n600 cohort, and three checkpoint stages are sealed")
        if (self.holdout_period, self.train_lattice_stride_on_prefix) != (5, 4):
            raise ValueError("split and training lattice are preregistered")
        if self.heldout_lattice_stride_on_prefix != 1 or self.teacher_batch_size != 1:
            raise ValueError("heldout geometry and teacher batch are sealed")
        if self.requested_area_fraction != REQUESTED_AREA_FRACTION:
            raise ValueError("matched support area is preregistered")
        if self.retained_mass_bar != RETAINED_MASS_BAR:
            raise ValueError("oracle-derived retained-mass bar is preregistered")
        if self.calibration_bin_count != 16 or self.heldout_ece_refusal_bar != 0.05:
            raise ValueError("calibration policy is preregistered")
        if self.rung_order != (
            "weighted-topk-global-84",
            "weighted-topk-pair-block-44",
            "pairwise-rank-pair-block-44",
        ):
            raise ValueError("finite rung order is sealed")
        if self.source_class_sensitivity != (2.2, 32.0, 0.26, 1.0, 0.0):
            raise ValueError("class sensitivity provenance would change")
        if self.prefix_module != "encoder.model.blocks.0.0.bn1":
            raise ValueError("prefix module is inherited and sealed")
        if self.prefix_cut != "pre_squeeze_excite_pre_global_pool":
            raise ValueError("prefix locality may not be weakened")
        if (self.global_feature_count, self.block_feature_count, self.ordered_pair_count) != (
            84,
            44,
            20,
        ):
            raise ValueError("feature and block widths are preregistered")
        if self.fallback != "full_exact_teacher":
            raise ValueError("failed localizers must fall back to the exact teacher")
        if self.live_training_enabled or not self.research_only:
            raise ValueError("round 4 has no live-training authority")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local gradient evidence cannot claim score or promotion")
        if self.fore_weighting_enabled:
            raise ValueError("fixed replay lacks transition-complete FORE custody")

    @property
    def train_state_count(self) -> int:
        return self.n_pairs - self.n_pairs // self.holdout_period

    @property
    def heldout_state_count(self) -> int:
        return self.n_pairs // self.holdout_period

    @property
    def selected_prefix_cells(self) -> int:
        return SELECTED_PREFIX_CELLS

    @property
    def realized_area_fraction(self) -> float:
        return REALIZED_AREA_FRACTION

    @property
    def conditional_composed_label_coefficient(self) -> float:
        prefix = ROUND3_PREFIX_FLOP_FRACTION
        return prefix + (1.0 - prefix) * self.realized_area_fraction

    @property
    def conditional_variable_cost_reduction_x(self) -> float:
        return 1.0 / self.conditional_composed_label_coefficient

    def compile_measurement_contract(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "train_state_count": self.train_state_count,
                "heldout_state_count": self.heldout_state_count,
                "prefix_cell_count": PREFIX_CELL_COUNT,
                "selected_prefix_cells": self.selected_prefix_cells,
                "realized_area_fraction": self.realized_area_fraction,
                "round3_oracle_retained_mass": ROUND3_ORACLE_RETAINED_MASS,
                "primary_metric": (
                    "aggregate exact input-costate L2-square mass retained by deterministic "
                    "top-score masks on heldout states"
                ),
                "winner_rule": "maximum retained mass; exact tie uses preregistered rung order",
                "convex_solver": (
                    "float64 symmetric eigendecomposition Moore-Penrose minimum-norm "
                    "normal-equation optimum; no ridge sweep"
                ),
                "calibration": (
                    "train-only 16-quantile Jeffreys reliability bins, monotone PAV, "
                    "piecewise-linear prediction"
                ),
                "economics": {
                    "form": "C_teacher=A+c_label*D",
                    "prefix_fraction": ROUND3_PREFIX_FLOP_FRACTION,
                    "conditional_c_label": self.conditional_composed_label_coefficient,
                    "conditional_reduction_x": self.conditional_variable_cost_reduction_x,
                    "wall_clock_claim": False,
                    "reason": (
                        "exact teacher contains global squeeze-excite dependencies; the composed "
                        "coefficient is conditional until a sparse exact receiver exists"
                    ),
                },
                "live_trainer_argv": [],
                "constant_provenance": {
                    "area_and_bar": "SOURCE operator round-4 contract and round-3 oracle",
                    "split_and_prefix": "MEASURED-INHERITED round-3 replay custody",
                    "class_sensitivity_0_1_2_4": "MEASURED committed waterfill directive",
                    "class_sensitivity_3": (
                        "ASSUMED neutral feature value; no committed numeric coefficient; "
                        "class-pair block isolates it"
                    ),
                    "feature_widths": "DERIVED from sealed base plus explicit pair channels",
                    "solver_threshold": "DERIVED float64 numerical rank floor",
                    "calibration_bins": "ASSUMED one bounded diagnostic; no sweep",
                    "FORE": "REFUSED current replay; no transitions or coverage receipt",
                },
            }
        )
        return payload


__all__ = [
    "CALIBRATION_BIN_COUNT",
    "HELDOUT_ECE_REFUSAL_BAR",
    "PREFIX_CELL_COUNT",
    "REALIZED_AREA_FRACTION",
    "REQUESTED_AREA_FRACTION",
    "RETAINED_MASS_BAR",
    "ROUND3_ORACLE_RETAINED_MASS",
    "ROUND3_PREFIX_FLOP_FRACTION",
    "SELECTED_PREFIX_CELLS",
    "ReplaceRound4SupportRankingPolicy",
]
