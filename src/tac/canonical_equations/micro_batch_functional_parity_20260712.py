# SPDX-License-Identifier: MIT
"""Canonical micro-batch functional-parity and training-admission law (2026-07-12).

Historical batch-dependent floating-point drift remains measured and true. The operator's
2026-07-12 waiver changes the TRAINING admission predicate, not those measurements and not score
authority: bit identity is waived for the training loop only. Each pair's spatially weighted term
must be normalized independently before the batch mean; a global batch denominator silently lets
large-support pairs dominate and is not functionally equivalent to serial per-pair training.

No empirical speed anchor is registered here yet. A measured wall-clock receipt and n600 score
validation remain owed; callers must not use this law to promote a score or move the frontier.
"""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "micro_batch_functional_parity_training_admission_v1"
EPSILON = 1e-6
_SPEC = ".omx/research/micro_batch_v9_unlock_20260712_implementation_spec.md"
_PREDICTED = "[training-policy/source-verified; speed empirical owed]"


def per_pair_weighted_batch_mean(
    numerator: np.ndarray, weight: np.ndarray, *, eps: float = EPSILON
) -> float:
    """Return ``mean_b(sum_x numerator[b,x] / (sum_x weight[b,x] + eps))``.

    ``numerator`` is the already-weighted per-element loss numerator. Batch is axis 0 and every
    remaining axis is the per-pair spatial/channel domain. The two arrays must have identical shape;
    weights are finite and non-negative. This deliberately does *not* compute the tempting global
    ratio ``sum_{b,x} numerator / sum_{b,x} weight``.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(weight, dtype=np.float64)
    if num.shape != den.shape or num.ndim < 2 or num.shape[0] < 1:
        raise ValueError(
            f"numerator and weight must share shape (B,...), B>=1; got {num.shape} vs {den.shape}"
        )
    if not np.isfinite(num).all() or not np.isfinite(den).all():
        raise ValueError("numerator and weight must be finite")
    if (den < 0.0).any():
        raise ValueError("weight must be non-negative")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"eps must be finite and >0, got {eps!r}")
    spatial_axes = tuple(range(1, num.ndim))
    per_pair = np.sum(num, axis=spatial_axes) / (
        np.sum(den, axis=spatial_axes) + float(eps)
    )
    return float(np.mean(per_pair))


def training_admission_predicate(
    *,
    loss_delta: float,
    loss_tolerance: float,
    gradient_delta: float,
    gradient_tolerance: float,
    measured_speedup: float,
    scope: str = "training",
    requests_score_authority: bool = False,
) -> bool:
    """Training-only admission under the operator waiver.

    Admission requires functional loss and gradient parity within predeclared tolerances and a
    measured wall-clock speedup strictly above 1.0. Any score-authority request refuses: exact
    byte-closed n600 evaluation is a separate, non-waived gate.
    """
    values = (loss_delta, loss_tolerance, gradient_delta, gradient_tolerance, measured_speedup)
    if not all(np.isfinite(float(v)) for v in values):
        return False
    if float(loss_tolerance) < 0.0 or float(gradient_tolerance) < 0.0:
        return False
    return bool(
        scope == "training"
        and not requests_score_authority
        and abs(float(loss_delta)) <= float(loss_tolerance)
        and abs(float(gradient_delta)) <= float(gradient_tolerance)
        and float(measured_speedup) > 1.0
    )


def build_micro_batch_functional_parity_training_admission_v1() -> CanonicalEquation:
    """Build the central per-pair-normalization and training-only admission law."""
    policy_anchor = EmpiricalAnchor(
        anchor_id="micro_batch_training_only_drift_waiver_policy_20260712",
        measurement_utc="2026-07-12T00:00:00Z",
        inputs={
            "scope": "training only",
            "historical_drift": "preserved; scorer/reduction batch dependence remains measured",
            "normalization": "per-pair weighted ratios, then mean over batch",
        },
        predicted_output={
            "admission": "functional loss parity AND gradient parity AND measured speedup > 1"
        },
        empirical_output={
            "operator_waiver": "bit identity waived for training only",
            "speed_receipt": "OWED; do not append a registry empirical row yet",
            "score_authority": "NONE; byte-closed n600 exact validation owed",
        },
        residual=0.0,
        source_artifact=_SPEC,
        measurement_method="operator-policy/source inspection; empirical speed measurement pending",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="micro_batch_functional_parity_training_admission.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="Apple_Metal_MLX_training_surface_unmeasured",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Micro-batch per-pair functional parity and training-only admission",
        one_line_summary=(
            "Normalize every pair independently, mean across pairs, and admit training only after "
            "loss/gradient parity plus measured speedup; drift waiver grants no score authority."
        ),
        latex_form=(
            r"L_B=\frac{1}{B}\sum_{b=1}^{B}"
            r"\frac{\sum_x N_{b,x}}{\sum_x W_{b,x}+\varepsilon};\quad "
            r"A_{train}=\mathbf{1}[\Delta_L\leq\tau_L\land\Delta_g\leq\tau_g"
            r"\land s_{wall}>1\land\neg A_{score}]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.micro_batch_functional_parity_20260712:"
            "per_pair_weighted_batch_mean"
        ),
        domain_of_validity={
            "vehicle": ["level_set_witness_v9_cgauge"],
            "scope": "training only; floating-point trajectory drift explicitly waived",
            "normalization": "per-pair numerator/denominator before batch mean; never global denominator",
            "training_admission": "functional loss/gradient parity plus measured wall-clock speedup",
            "score_authority": "none; byte-closed exact n600 validation remains owed",
            "frontier_authority": "reports/latest.md; unchanged by this training-only law",
        },
        units_in={
            "numerator": "weighted_loss_elements_by_pair",
            "weight": "nonnegative_weight_elements_by_pair",
            "measured_speedup": "serial_walltime_over_batched_walltime",
        },
        units_out={"L_B": "mean_per_pair_loss", "A_train": "boolean"},
        empirical_anchors=(policy_anchor,),
        predicted_vs_empirical_residual={"speed_receipt_owed": 0.0},
        last_calibration_utc="2026-07-12T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.levelset_micro_batch_loss",
            "experiments.train_levelset_witness_realized_through_R_mlx",
        ),
        canonical_producers=("tools.micro_batch_bit_identity_probe",),
        provenance=build_provenance_for_predicted(
            model_id="micro_batch_functional_parity_training_admission.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="Apple_Metal_MLX_training_surface_unmeasured",
        ),
    )


def populate_micro_batch_functional_parity_training_admission_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Idempotent append-only registration; caller must not invoke before the benchmark is final."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_micro_batch_functional_parity_training_admission_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "micro_batch_functional_parity_20260712; training-only drift waiver; functional parity "
            "+ measured speed required; no score authority"
        ),
    )
    return eq


__all__ = [
    "EPSILON",
    "EQUATION_ID",
    "build_micro_batch_functional_parity_training_admission_v1",
    "per_pair_weighted_batch_mean",
    "populate_micro_batch_functional_parity_training_admission_v1",
    "training_admission_predicate",
]
