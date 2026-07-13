# SPDX-License-Identifier: MIT
"""Typed, default-off policy for the round-2 frozen-replay convex head."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

_BINARY32_FRACTION_BITS = 23
_IDEAL_WORST_CASE_CONTRACTION_DENOMINATOR = 3
_DERIVED_FIT_EPOCHS = math.ceil(
    _BINARY32_FRACTION_BITS
    * math.log(2.0)
    / math.log(float(_IDEAL_WORST_CASE_CONTRACTION_DENOMINATOR))
)


@dataclass(frozen=True)
class FrozenReplayConvexHeadPolicy:
    """Compile the measurement contract without inventing a trainer flag.

    The state lattice and held-out split are explicitly ASSUMED probe design
    choices.  The fit horizon is derived from the worst-case ideal contraction
    ``1/3``: enough steps to reduce the initial error by one binary32 fraction
    ulp.  Curvature, step size, and the realized fp32 contraction are derived
    from the frozen feature covariance rather than accepted from the caller.
    """

    mode: str = "frozen_replay_convex_head"
    seed: int = 455
    n_pairs: int = 600
    checkpoint_count: int = 3
    holdout_period: int = 5
    train_lattice_stride: int = 8
    fit_epochs: int = _DERIVED_FIT_EPOCHS
    teacher_batch_size: int = 1
    minimum_teacher_call_amortization_x: float = 5.0
    operator_early_regime_cosine_bar: float = -0.16153190769629602
    legacy_nonnegative_policy_overlay: float = 0.0
    fallback: str = "full_exact_teacher"
    live_training_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.mode != "frozen_replay_convex_head":
            raise ValueError("mode is sealed for the registered formulation")
        if self.seed != 455 or self.n_pairs != 600 or self.checkpoint_count != 3:
            raise ValueError("seed, n600 cohort, and three-stage replay are sealed")
        if self.holdout_period != 5 or self.train_lattice_stride != 8:
            raise ValueError("changing an ASSUMED probe design requires a new formulation instance")
        if self.fit_epochs != _DERIVED_FIT_EPOCHS:
            raise ValueError("fit_epochs must preserve the binary32/worst-case-contraction derivation")
        if self.teacher_batch_size != 1:
            raise ValueError(
                "exact per-state costates require teacher_batch_size=1 because the committed teacher "
                "uses mean cross-entropy over the batch"
            )
        if (
            self.minimum_teacher_call_amortization_x != 5.0
            or self.operator_early_regime_cosine_bar != -0.16153190769629602
            or self.legacy_nonnegative_policy_overlay != 0.0
        ):
            raise ValueError("decision thresholds are preregistered")
        if self.fallback != "full_exact_teacher":
            raise ValueError("a failed research head must fall back to the exact teacher")
        if self.live_training_enabled or not self.research_only:
            raise ValueError("the round-2 probe has no live-training authority")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local training-gradient evidence cannot claim score or promotion authority")

    @property
    def train_state_count(self) -> int:
        return self.n_pairs - self.n_pairs // self.holdout_period

    @property
    def heldout_state_count(self) -> int:
        return self.n_pairs // self.holdout_period

    @property
    def effective_training_state_steps(self) -> int:
        """Number of labeled state uses, distinct from optimizer-step count."""

        return self.train_state_count * self.fit_epochs

    def compile_measurement_contract(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "train_state_count": self.train_state_count,
                "heldout_state_count": self.heldout_state_count,
                "effective_training_state_steps": self.effective_training_state_steps,
                "optimizer_steps": self.fit_epochs,
                "exact_target_costate_tensor_absent": True,
                "feature_includes_source_labels_and_margins": True,
                "full_batch_deterministic_gradient_descent": True,
                "curvature_policy": "DERIVED from realized numpy-fp32 X'X/n",
                "ridge_policy": "DERIVED lambda=lambda_max(X'X/n)",
                "step_policy": "DERIVED eta=2/(mu+L)",
                "constant_provenance": {
                    "seed": "SOURCE task-455 lineage",
                    "n_pairs": "SOURCE operator n600 evidence rule",
                    "checkpoint_count": "DERIVED available cold V9 trajectory stages",
                    "holdout_period": "ASSUMED deterministic 20-percent split; not tuned",
                    "train_lattice_stride": "ASSUMED compute-bounded fixed lattice; not tuned",
                    "fit_epochs": (
                        "DERIVED ceil(binary32_fraction_bits*ln(2)/ln(3)) from ideal gamma<=1/3"
                    ),
                    "teacher_batch_size": (
                        "DERIVED exact-label parity: committed teacher mean reduction requires batch size 1"
                    ),
                    "decision_thresholds": (
                        "SOURCE operator literal early saved-regime bar and 5x call floor; "
                        "legacy nonnegative predicate is diagnostic only"
                    ),
                },
                "authority": "numpy-fp32 local macOS-CPU training-gradient evidence only",
                "live_trainer_argv": [],
            }
        )
        return payload


__all__ = ["FrozenReplayConvexHeadPolicy"]
