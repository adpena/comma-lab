# SPDX-License-Identifier: MIT
"""Typed control law for task #455's on-policy scorer surrogate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tac.scorer_surrogate.onpolicy_costate import FULL_BUILD_BLOCKER

_PROVIDER_FEATURE_GROUPS = ("current_frame", "frame_delta", "anchor_costate")
_UINT8_BITS = 8
_IEEE754_BINARY32_FRACTION_BITS = 23


@dataclass(frozen=True)
class OnPolicyScorerSurrogatePolicy:
    """Fail-closed policy; it emits no live-trainer flag.

    The 95% target is operator-supplied by the task.  Cadence is therefore
    self-derived rather than tuned: ``ceil(1 / (1 - target_skip_fraction))``.
    Admission remains event-conditioned on exact through-R teacher descent.
    """

    target_exact_teacher_skip_fraction: float = 0.95
    control_cadences: tuple[int, ...] = (1, 4)
    input_channels: int = 9
    requires_exact_ce_descent: bool = True
    requires_nonnegative_costate_cosine: bool = True
    requires_sequence_endpoint_dseg_nonworsening: bool = True
    requires_sequence_endpoint_dpose_nonworsening: bool = True
    ema_decay: float = 0.997
    fallback: str = "exact_anchor_refresh_only"
    research_only: bool = True
    full_build_blocker: str = FULL_BUILD_BLOCKER
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.target_exact_teacher_skip_fraction < 1.0):
            raise ValueError("target_exact_teacher_skip_fraction must lie in (0, 1)")
        if any(not isinstance(value, int) or value < 1 for value in self.control_cadences):
            raise ValueError("control_cadences must contain integers >= 1")
        if self.input_channels != 9:
            raise ValueError("the provider contract requires 9 input channels")
        if not (0.99 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must lie in [0.99, 1.0)")
        if self.fallback != "exact_anchor_refresh_only":
            raise ValueError("non-anchor exact fallback would invalidate forward replacement")
        if not self.research_only or self.full_build_blocker != FULL_BUILD_BLOCKER:
            raise ValueError("task #455 remains research-only until one cadence passes the matched three-regime gate")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("the training surrogate cannot carry score or promotion authority")

    @property
    def derived_target_cadence(self) -> int:
        return math.ceil(1.0 / (1.0 - self.target_exact_teacher_skip_fraction))

    @property
    def measured_cadences(self) -> tuple[int, ...]:
        return tuple(sorted({*self.control_cadences, self.derived_target_cadence}))

    @property
    def hidden_channels(self) -> int:
        """First-probe default; not a derived or admitted capacity law."""

        return self.input_channels - 1

    @property
    def fit_steps_per_anchor(self) -> int:
        """First-probe default; the named capacity recess must replace it."""

        return self.hidden_channels

    @property
    def measurement_horizon(self) -> int:
        """Two target cycles ensure the recurring anchor fit is timed once."""

        return 2 * self.derived_target_cadence

    @property
    def matched_window_steps(self) -> int:
        """Cheap smoke horizon; never the decisive full-cadence horizon.

        The corrected probe is deliberately bounded and independent of the
        earlier arbitrary 40-step completion rule.  Its length is the ceiling
        of the square root of the target cadence, so changing the operator's
        skip target automatically changes the fidelity horizon.
        """

        return math.ceil(math.sqrt(self.derived_target_cadence))

    @property
    def decisive_window_steps(self) -> int:
        """One complete anchor interval required by the replacement-fidelity gate."""

        return self.derived_target_cadence

    @property
    def frame_channels(self) -> int:
        """Derive RGB channels from current/delta/anchor-costate feature groups."""

        feature_groups = len(_PROVIDER_FEATURE_GROUPS)
        if self.input_channels % feature_groups:
            raise ValueError("provider input channels must divide into three feature groups")
        return self.input_channels // feature_groups

    @property
    def amortized_hidden_channels(self) -> int:
        """Double the first-form residual width after its locality falsification."""

        return 2 * self.hidden_channels

    @property
    def branch_kernel_sizes(self) -> tuple[int, ...]:
        """First two nontrivial odd receptive-field supports, derived from RGB rank."""

        return tuple(2 * index + 1 for index in range(1, self.frame_channels))

    @property
    def dense_optimizer_steps_per_observation(self) -> int:
        """One optimizer update per independently parameterized receptive-field branch."""

        return len(self.branch_kernel_sizes)

    @property
    def dense_ema_decay(self) -> float:
        """EMA horizon equals the complete dense student-owned collection window."""

        return 1.0 - 1.0 / self.matched_window_steps

    def compile_corrected_measurement_contract(self) -> dict[str, object]:
        """Compile the premise-falsification repair without emitting live argv."""

        return {
            "mode": "dense_onpolicy_multireceptive_input_costate_matched_window",
            "supersedes_method": "onpolicy_scorer_surrogate_probe.v2_verdict_method_only",
            "target_anchor_cadence": self.derived_target_cadence,
            "matched_window_steps": self.matched_window_steps,
            "decisive_window_steps": self.decisive_window_steps,
            "admissible_window_steps": [self.matched_window_steps, self.decisive_window_steps],
            "dense_collection_steps": self.matched_window_steps,
            "dense_optimizer_steps_per_observation": self.dense_optimizer_steps_per_observation,
            "common_control_schedule_required": True,
            "exact_control_law": (
                "fractional halving from the explicit maximum until strict CE descent plus "
                "non-worsening exact through-R d_seg and d_pose; bounded exhaustion is BLOCKED, and only "
                "an exact zero renderer-gradient certificate is a terminal floor"
            ),
            "exact_metric_trace_every_step": True,
            "deterministic_repeat_noise_floor_required": True,
            "ema_provider_is_admission_authority": True,
            "resume_preserves_anchor_frame_and_costate": True,
            "isolated_timing_surfaces": [
                "exact_forward_only",
                "exact_costate_forward_backward",
                "anchor_fit",
                "surrogate_inference",
                "renderer_vjp",
                "whole_matched_window",
            ],
            "architecture": {
                "frame_channels": self.frame_channels,
                "input_channels": self.input_channels,
                "hidden_channels": self.amortized_hidden_channels,
                "branch_kernel_sizes": list(self.branch_kernel_sizes),
                "frame_value_scale": float((1 << _UINT8_BITS) - 1),
                "normalization_floor": math.ldexp(1.0, -_IEEE754_BINARY32_FRACTION_BITS),
                "mse_weight": 1.0,
                "cosine_weight": 1.0,
                "ema_decay": self.dense_ema_decay,
                "admission_min_relative_improvement": 0.0,
            },
            "derivations": {
                "matched_window": "ceil(sqrt(target_anchor_cadence)) smoke only",
                "decisive_window": "target_anchor_cadence",
                "hidden_channels": "2*(first_form_input_channels-1)",
                "branch_kernel_sizes": "first frame_channels-1 nontrivial odd supports",
                "ema_decay": "1-1/dense_collection_steps",
                "optimizer_steps": "number_of_receptive_field_branches",
                "frame_value_scale": "2^uint8_bits-1",
                "normalization_floor": "IEEE754_binary32_unit_roundoff",
                "loss_weights": "equal_reference_and_direction_debt",
            },
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "live_trainer_argv": [],
        }

    def compile_measurement_contract(self) -> dict[str, object]:
        return {
            "mode": "onpolicy_nonlinear_input_costate",
            "measured_cadences": list(self.measured_cadences),
            "target_exact_teacher_skip_fraction": self.target_exact_teacher_skip_fraction,
            "derived_target_cadence": self.derived_target_cadence,
            "measurement_horizon": self.measurement_horizon,
            "fit_steps_per_anchor": self.fit_steps_per_anchor,
            "hidden_channels": self.hidden_channels,
            "capacity_control_law": "constant first-probe defaults: hidden=8; fit_steps=8",
            "capacity_default_status": "ASSUMED_NOT_DERIVED",
            "capacity_recess_measurement": "shared_horizon_width_fit_grid_hidden_4_8_16_steps_4_8_16",
            "cadence_recess_measurement": "K4_cadence_interpolation_canary",
            "ema_decay": self.ema_decay,
            "admission_predicate": {
                "exact_cycle_ce_descent": self.requires_exact_ce_descent,
                "nonnegative_costate_cosine": self.requires_nonnegative_costate_cosine,
                "sequence_endpoint_dseg_nonworsening": self.requires_sequence_endpoint_dseg_nonworsening,
                "sequence_endpoint_dpose_nonworsening": self.requires_sequence_endpoint_dpose_nonworsening,
            },
            "fallback": self.fallback,
            "research_only": self.research_only,
            "full_build_blocker": self.full_build_blocker,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "live_trainer_argv": [],
        }
