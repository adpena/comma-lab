# SPDX-License-Identifier: MIT
"""Typed, default-off policy for REPLACE round-5 deeper/nonlinear localization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

REQUESTED_AREA_FRACTION = 0.047
PREFIX_CELL_COUNT = 192 * 256
SELECTED_PREFIX_CELLS = math.ceil(REQUESTED_AREA_FRACTION * PREFIX_CELL_COUNT)
REALIZED_AREA_FRACTION = SELECTED_PREFIX_CELLS / PREFIX_CELL_COUNT
RETAINED_MASS_BAR = 0.47
ROUND4_ORACLE_RETAINED_MASS = 0.5278150212253758
ROUND4_WINNER_RETAINED_MASS = 0.20172451295048283


@dataclass(frozen=True)
class ReplaceRound5DeeperNonlinearPolicy:
    """Seal every decision rule before the first round-5 teacher call."""

    mode: str = "replace_round5_deeper_nonlinear"
    seed: int = 455
    n_pairs: int = 600
    checkpoint_count: int = 3
    holdout_period: int = 5
    nonlinear_dev_modulus: int = 10
    nonlinear_dev_remainder: int = 1
    train_lattice_stride_on_prefix: int = 4
    heldout_lattice_stride_on_prefix: int = 1
    teacher_batch_size: int = 1
    requested_area_fraction: float = REQUESTED_AREA_FRACTION
    retained_mass_bar: float = RETAINED_MASS_BAR
    rung_order: tuple[str, ...] = (
        "convex-deeper-pair-block-mp",
        "nonlinear-pair-gated-mlp-ensemble",
    )
    cut_modules: tuple[str, ...] = (
        "encoder.model.blocks.1",
        "encoder.model.blocks.2",
    )
    cut_names: tuple[str, ...] = ("block2-post-se", "block3-post-se")
    deep_channels: tuple[int, ...] = (24, 48)
    deep_feature_count: int = 116
    ordered_pair_count: int = 20
    source_class_sensitivity: tuple[float, ...] = (2.2, 32.0, 0.26, 1.0, 0.0)
    nonlinear_seeds: tuple[int, ...] = (455, 456, 457)
    nonlinear_hidden_width: int = 32
    nonlinear_batch_size: int = 4096
    nonlinear_max_epochs: int = 60
    nonlinear_patience: int = 8
    nonlinear_min_delta: float = 0.0005
    nonlinear_learning_rate: float = 0.003
    nonlinear_weight_decay: float = 1e-5
    nonlinear_seed_std_bar: float = 0.03
    teacher_started_call_budget: int = 600
    query_total_fraction: float = 0.05
    query_targeted_fraction: float = 0.04
    query_random_audit_fraction: float = 0.01
    disagreement_error_ratio_bar: float = 1.25
    branch_horizons: tuple[int, ...] = (0, 1, 2, 4)
    fallback: str = "full-exact-teacher"
    live_training_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    fore_weighting_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode != "replace_round5_deeper_nonlinear":
            raise ValueError("round-5 policy mode is sealed")
        if (self.seed, self.n_pairs, self.checkpoint_count) != (455, 600, 3):
            raise ValueError("seed, n600 cohort, and checkpoint stages are sealed")
        if (self.holdout_period, self.nonlinear_dev_modulus, self.nonlinear_dev_remainder) != (
            5,
            10,
            1,
        ):
            raise ValueError("heldout and train-only nonlinear-dev splits are sealed")
        if self.train_lattice_stride_on_prefix != 4 or self.heldout_lattice_stride_on_prefix != 1:
            raise ValueError("training and heldout lattice geometry is sealed")
        if self.teacher_batch_size != 1:
            raise ValueError("one unique state per exact teacher call is required")
        if self.requested_area_fraction != REQUESTED_AREA_FRACTION:
            raise ValueError("matched support area is preregistered")
        if self.retained_mass_bar != RETAINED_MASS_BAR:
            raise ValueError("oracle-derived retained-mass bar is preregistered")
        if self.rung_order != (
            "convex-deeper-pair-block-mp",
            "nonlinear-pair-gated-mlp-ensemble",
        ):
            raise ValueError("the finite round-5 rung order is sealed")
        if self.cut_modules != ("encoder.model.blocks.1", "encoder.model.blocks.2"):
            raise ValueError("post-SE cut modules are sealed")
        if self.cut_names != ("block2-post-se", "block3-post-se"):
            raise ValueError("post-SE cut names are sealed")
        if self.deep_channels != (24, 48) or self.deep_feature_count != 116:
            raise ValueError("deeper feature geometry is sealed")
        if self.ordered_pair_count != 20:
            raise ValueError("ordered class-pair count is sealed")
        if self.source_class_sensitivity != (2.2, 32.0, 0.26, 1.0, 0.0):
            raise ValueError("class sensitivity provenance would change")
        if self.nonlinear_seeds != (455, 456, 457):
            raise ValueError("the nonlinear multi-seed audit is sealed")
        if (
            self.nonlinear_hidden_width,
            self.nonlinear_batch_size,
            self.nonlinear_max_epochs,
            self.nonlinear_patience,
        ) != (32, 4096, 60, 8):
            raise ValueError("nonlinear capacity and early-stop cap are sealed")
        if (
            self.nonlinear_min_delta,
            self.nonlinear_learning_rate,
            self.nonlinear_weight_decay,
            self.nonlinear_seed_std_bar,
        ) != (0.0005, 0.003, 1e-5, 0.03):
            raise ValueError("nonlinear optimizer and stability gate are sealed")
        if self.teacher_started_call_budget != 600:
            raise ValueError("campaign-honest exact-call budget is sealed")
        if (
            self.query_total_fraction,
            self.query_targeted_fraction,
            self.query_random_audit_fraction,
        ) != (0.05, 0.04, 0.01):
            raise ValueError("query/audit budget is sealed")
        if not math.isclose(
            self.query_targeted_fraction + self.query_random_audit_fraction,
            self.query_total_fraction,
        ):
            raise ValueError("targeted and randomized query fractions must close")
        if self.disagreement_error_ratio_bar != 1.25:
            raise ValueError("disagreement ranking gate is sealed")
        if self.branch_horizons != (0, 1, 2, 4):
            raise ValueError("equal-call branch-horizon candidates are sealed")
        if self.fallback != "full-exact-teacher":
            raise ValueError("failed localizers must fall back to the exact teacher")
        if self.live_training_enabled or not self.research_only:
            raise ValueError("round 5 has no live-training authority")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local costate evidence cannot claim score or promotion")
        if self.fore_weighting_enabled:
            raise ValueError("fixed replay lacks transition-complete FORE custody")

    @property
    def train_state_count(self) -> int:
        return self.n_pairs - self.n_pairs // self.holdout_period

    @property
    def heldout_state_count(self) -> int:
        return self.n_pairs // self.holdout_period

    @property
    def nonlinear_dev_state_count(self) -> int:
        return sum(
            index % self.holdout_period != 0
            and index % self.nonlinear_dev_modulus == self.nonlinear_dev_remainder
            for index in range(self.n_pairs)
        )

    @property
    def nonlinear_core_state_count(self) -> int:
        return self.train_state_count - self.nonlinear_dev_state_count

    @property
    def selected_prefix_cells(self) -> int:
        return SELECTED_PREFIX_CELLS

    @property
    def realized_area_fraction(self) -> float:
        return REALIZED_AREA_FRACTION

    def compile_measurement_contract(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "train_state_count": self.train_state_count,
                "heldout_state_count": self.heldout_state_count,
                "nonlinear_core_state_count": self.nonlinear_core_state_count,
                "nonlinear_dev_state_count": self.nonlinear_dev_state_count,
                "prefix_cell_count": PREFIX_CELL_COUNT,
                "selected_prefix_cells": self.selected_prefix_cells,
                "realized_area_fraction": self.realized_area_fraction,
                "round4_oracle_retained_mass": ROUND4_ORACLE_RETAINED_MASS,
                "round4_winner_retained_mass": ROUND4_WINNER_RETAINED_MASS,
                "primary_metric": (
                    "aggregate exact input-costate L2-square mass retained by deterministic "
                    "top-score masks on the untouched 120-state heldout split"
                ),
                "convex_gate": (
                    "PASS iff convex-deeper-pair-block-mp retained mass is at least 0.47"
                ),
                "nonlinear_gate": (
                    "PASS iff ensemble retained mass is at least 0.47, heldout per-seed "
                    "retained-mass population std is at most 0.03, all three seeds finish, "
                    "and campaign-honest exact teacher starts are at most 600"
                ),
                "winner_rule": (
                    "measure both fixed rungs in preregistered EV order; maximum admitted "
                    "retained mass wins, with exact ties resolved by rung order; add no third "
                    "rung after heldout inspection"
                ),
                "convex_solver": (
                    "twenty float64 symmetric-eigendecomposition Moore-Penrose minimum-norm "
                    "RankRLS block optima; no regularization or width sweep"
                ),
                "nonlinear_training": (
                    "three deterministic 116->32->20 pair-gated ReLU MLPs; fixed optimizer; "
                    "train-only dev retained-mass early stop; no heldout selection"
                ),
                "branch_horizon_ticket": {
                    "candidate_horizons": list(self.branch_horizons),
                    "equal_exact_call_budgets": True,
                    "current_fixed_replay_status": "blocked-not-identified",
                    "required_transition_tuple": ["Z", "A", "R", "Z_prime"],
                },
                "query_real_ticket": {
                    "targeted_fraction": self.query_targeted_fraction,
                    "randomized_positive_propensity_audit_fraction": (
                        self.query_random_audit_fraction
                    ),
                    "total_fraction": self.query_total_fraction,
                    "error_ratio_bar": self.disagreement_error_ratio_bar,
                    "live_default": "refuse-live-research-only-fixed-replay",
                },
                "live_trainer_argv": [],
                "constant_provenance": {
                    "area_and_bar": "SOURCE operator round-5 contract and round-3 oracle",
                    "split": "MEASURED-INHERITED fixed round-4 V9 n600 replay custody",
                    "cut_modules": "DERIVED first two encoder stages after first SE",
                    "class_pair_structure": "MEASURED-INHERITED round-4 relative winner",
                    "mlp_capacity_and_stability": "ASSUMED one bounded nonlinear rung; no sweep",
                    "query_budget": "SOURCE DIG-S1 charter 5-percent falsifier; split preregistered",
                    "branch_horizons": "DERIVED bounded MVE/STEVE design set; unmeasured here",
                    "FORE": "REFUSED current replay; no Z,A,R,Z-prime transitions",
                },
            }
        )
        return payload


__all__ = [
    "PREFIX_CELL_COUNT",
    "REALIZED_AREA_FRACTION",
    "REQUESTED_AREA_FRACTION",
    "RETAINED_MASS_BAR",
    "ROUND4_ORACLE_RETAINED_MASS",
    "ROUND4_WINNER_RETAINED_MASS",
    "SELECTED_PREFIX_CELLS",
    "ReplaceRound5DeeperNonlinearPolicy",
]
