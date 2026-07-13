# SPDX-License-Identifier: MIT
"""Canonical finite-sample tail discipline for control-driving interpolators.

This module is a MEANS surface.  It selects a regularization strength by an
empirical upper-tail loss subject to an explicit mean gate; it does not confer
contest-score or control-actuation authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "control_interpolator_tail_cvar_mean_gate_v1"
FINITE_BOUND_ID = "fixed_design_correlated_gaussian_ridge_tail_v1"
_UTC = "2026-07-13T23:30:00Z"
_MEMO = ".omx/research/quant_tail_reliability_20260713.md"
_RECEIPT = ".omx/research/quant_tail_reliability_receipt_20260713.json"
_AXIS = "[macOS-CPU advisory; NumPy-fp32 decision; MEANS; no score authority]"


class TailReliabilityError(ValueError):
    """A tail metric, gate, or finite-design bound is not well posed."""


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise TailReliabilityError(f"{name} must be nonempty and finite")
    return array


def empirical_cvar(losses: Sequence[float] | np.ndarray, *, alpha: float) -> float:
    """Return the exact upper-tail integral of the empirical quantile function.

    Fractional boundary mass is retained, so this definition is stable when
    ``(1-alpha)*n`` is not an integer.  At a tail mass below one observation the
    empirical CVaR is the observed maximum; this is deliberately conservative.
    """

    values = np.sort(_finite_vector(losses, name="losses"))[::-1]
    if not 0.0 <= float(alpha) < 1.0:
        raise TailReliabilityError("alpha must lie in [0, 1)")
    tail_mass = (1.0 - float(alpha)) * values.size
    if tail_mass <= 1.0:
        return float(values[0])
    full = min(math.floor(tail_mass + 1.0e-12), values.size)
    fraction = tail_mass - full
    numerator = float(values[:full].sum(dtype=np.float64))
    if fraction > 1.0e-12 and full < values.size:
        numerator += fraction * float(values[full])
    return numerator / tail_mass


def retained_mass_tail_summary(
    retained_mass: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> dict[str, float | int]:
    """Summarize beneficial retained mass and its harmful shortfall tail."""

    mass = _finite_vector(retained_mass, name="retained_mass")
    if np.any((mass < 0.0) | (mass > 1.0)):
        raise TailReliabilityError("retained mass must lie in [0, 1]")
    error = 1.0 - mass
    return {
        "sample_count": int(mass.size),
        "retained_mass_mean": float(mass.mean(dtype=np.float64)),
        "retained_mass_median": float(np.quantile(mass, 0.50, method="linear")),
        "retained_mass_q10": float(np.quantile(mass, 0.10, method="linear")),
        "retained_mass_q05": float(np.quantile(mass, 0.05, method="linear")),
        "retained_mass_q01": float(np.quantile(mass, 0.01, method="linear")),
        "retained_mass_worst": float(np.min(mass)),
        "shortfall_mean": float(error.mean(dtype=np.float64)),
        "shortfall_p90": float(np.quantile(error, 0.90, method="linear")),
        "shortfall_p95": float(np.quantile(error, 0.95, method="linear")),
        "shortfall_p99": float(np.quantile(error, 0.99, method="linear")),
        "shortfall_worst": float(np.max(error)),
        "shortfall_cvar": empirical_cvar(error, alpha=alpha),
        "cvar_alpha": float(alpha),
    }


def loss_tail_summary(
    losses: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> dict[str, float | int]:
    """Summarize an arbitrary nonnegative loss for mean/tail selection."""

    values = _finite_vector(losses, name="losses")
    if np.any(values < 0.0):
        raise TailReliabilityError("losses may not be negative")
    return {
        "sample_count": int(values.size),
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.quantile(values, 0.50, method="linear")),
        "p90": float(np.quantile(values, 0.90, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "worst": float(np.max(values)),
        "cvar": empirical_cvar(values, alpha=alpha),
        "cvar_alpha": float(alpha),
    }


@dataclass(frozen=True)
class TailLambdaSelection:
    lambda_value: float
    cvar: float
    mean_loss: float
    eligible_count: int
    mean_limit: float
    alpha: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "equation_id": EQUATION_ID,
            "lambda": self.lambda_value,
            "cvar": self.cvar,
            "mean_loss": self.mean_loss,
            "eligible_count": self.eligible_count,
            "mean_limit": self.mean_limit,
            "alpha": self.alpha,
        }


def select_tail_lambda(
    rows: Sequence[Mapping[str, Any]],
    *,
    mean_reference: float,
    alpha: float,
    mean_tolerance: float = 0.0,
    require_positive: bool = True,
) -> TailLambdaSelection:
    """Implement ``argmin_lambda CVaR_alpha(L)`` subject to a mean gate.

    Each row must contain ``lambda`` and ``losses``.  Ties are broken by p99,
    mean, then the smaller lambda, in that order.  Lambda zero can remain in a
    diagnostic curve while being excluded from a load-bearing selection.
    """

    if not math.isfinite(float(mean_reference)) or float(mean_reference) < 0.0:
        raise TailReliabilityError("mean_reference must be finite and nonnegative")
    if not math.isfinite(float(mean_tolerance)) or float(mean_tolerance) < 0.0:
        raise TailReliabilityError("mean_tolerance must be finite and nonnegative")
    mean_limit = float(mean_reference) + float(mean_tolerance)
    eligible: list[tuple[float, float, float, float]] = []
    for row in rows:
        lambda_value = float(row["lambda"])
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise TailReliabilityError("lambda must be finite and nonnegative")
        losses = _finite_vector(row["losses"], name="row losses")
        mean_loss = float(losses.mean(dtype=np.float64))
        if (not require_positive or lambda_value > 0.0) and mean_loss <= mean_limit + 1.0e-15:
            eligible.append(
                (
                    empirical_cvar(losses, alpha=alpha),
                    float(np.quantile(losses, 0.99, method="linear")),
                    mean_loss,
                    lambda_value,
                )
            )
    if not eligible:
        raise TailReliabilityError("no lambda satisfies the declared mean gate")
    cvar, _p99, mean_loss, lambda_value = min(eligible)
    return TailLambdaSelection(
        lambda_value=lambda_value,
        cvar=cvar,
        mean_loss=mean_loss,
        eligible_count=len(eligible),
        mean_limit=mean_limit,
        alpha=float(alpha),
    )


def correlated_gaussian_quadratic_upper_tail(
    quadratic_matrix: Sequence[Sequence[float]] | np.ndarray,
    linear_vector: Sequence[float] | np.ndarray,
    *,
    delta: float,
) -> dict[str, float | str]:
    """Finite fixed-design noncentral-Gaussian quadratic upper-tail bound.

    For ``u ~ N(0,I)`` and

    ``Q-EQ = u' A u - tr(A) + 2 c' u``, with ``A`` positive semidefinite,

    the returned excess ``B`` satisfies ``P(Q-EQ >= B) <= delta``.  This is a
    conditional finite-sample statement: callers must construct ``A`` and ``c``
    from their fixed design, covariance, loss weights, and deterministic bias.
    """

    matrix = np.asarray(quadratic_matrix, dtype=np.float64)
    vector = _finite_vector(linear_vector, name="linear_vector")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise TailReliabilityError("quadratic_matrix must be square")
    if matrix.shape[0] != vector.size or not np.isfinite(matrix).all():
        raise TailReliabilityError("quadratic and linear geometries disagree")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(eigenvalues), initial=0.0))
    )
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise TailReliabilityError("quadratic_matrix must be positive semidefinite")
    if not 0.0 < float(delta) < 1.0:
        raise TailReliabilityError("delta must lie in (0, 1)")
    t = math.log(1.0 / float(delta))
    frobenius = float(np.linalg.norm(matrix, ord="fro"))
    operator = max(0.0, float(np.max(eigenvalues, initial=0.0)))
    linear = float(np.linalg.norm(vector))
    variance_geometry = frobenius * frobenius + 2.0 * linear * linear
    excess = 2.0 * math.sqrt(variance_geometry * t) + 2.0 * operator * t
    return {
        "equation_id": FINITE_BOUND_ID,
        "assumption": "fixed design; correlated Gaussian noise with known covariance whitening",
        "delta": float(delta),
        "log_one_over_delta": t,
        "frobenius_A": frobenius,
        "operator_A": operator,
        "l2_c": linear,
        "upper_tail_excess": excess,
    }


def build_control_interpolator_tail_cvar_mean_gate_v1() -> CanonicalEquation:
    """Build the typed decision law with the cached 2026-07-13 anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "remeasure on state/trajectory-block holdout; require a closed lambda bracket; "
            "for PRE-SE preserve official n120 raw arrays; for control preserve counterfactual regret"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macos_arm64_numpy_cached_research",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="quant_tail_cached_prese_and_organ_20260713",
        measurement_utc=_UTC,
        inputs={
            "pre_se_scope": "420 core / 60 train-only dev real states x seeds 455/456/457",
            "organ_scope": "nine cached intervals / seven past-only walk-forward folds",
            "alpha": 0.95,
            "mean_gate": "no empirical mean regression versus declared reference",
            "official_n120_new_lambda": "BLOCKED_MISSING_RAW_ARRAYS",
        },
        predicted_output={
            "law": "positive ridge may suppress severe RankRLS tail errors",
            "selection": "argmin empirical CVaR subject to mean gate",
        },
        empirical_output={
            "pre_se_block2": {"lambda": 1.0, "p95": 0.8687327671206475,
                               "p99": 0.885778230777784, "cvar95": 0.8805590344095054},
            "pre_se_block3": {"lambda": 0.3, "p95": 0.9196654967792378,
                               "p99": 0.9329094311657196, "cvar95": 0.9291047003046485},
            "organ_A": {"lambda": 1000.0, "bracket_closed": False},
            "organ_P": {"lambda": 0.1, "own_default_cvar_relative": -0.041117334701275765},
            "organ_Q": {"lambda": 0.1, "own_default_cvar_relative": -0.0410981616509446},
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "cached NumPy-fp32 RankRLS cross-fit and cached organ walk-forward; "
            "canonical selector reproduced the measured argmin"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Tail-CVaR lambda selection subject to an empirical mean gate",
        one_line_summary=(
            "Every control-driving interpolator selects positive lambda by block-held-out "
            "CVaR subject to no mean regression and reports a tail quantile beside its mean."
        ),
        latex_form=(
            r"\lambda^*=\arg\min_{\lambda\in\Lambda,\lambda>0}"
            r"\operatorname{CVaR}_{\alpha}(L_\lambda)\quad\mathrm{s.t.}\quad"
            r"\widehat{\mathbb E}L_\lambda\le\widehat{\mathbb E}L_{\rm ref}+\epsilon_{\rm mean}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.control_interpolator_tail_reliability_20260713:"
            "select_tail_lambda"
        ),
        domain_of_validity={
            "included": (
                "control-driving interpolator with declared nonnegative loss",
                "state/trajectory-block holdout",
                "closed positive-lambda bracket",
                "declared reference, mean tolerance, tail level, and numerical authority",
            ),
            "excluded": (
                "score or archive promotion",
                "selection on an official final holdout",
                "open-boundary lambda adoption",
                "forecast-regret proxy relabeled as counterfactual control regret",
            ),
            "verdict_scope": "MEANS decision law; numeric lambdas are surface-specific",
            "authority": _AXIS,
        },
        units_in={"lambda": "regularization units", "loss": "declared surface loss"},
        units_out={"lambda_star": "regularization units", "cvar": "declared surface loss"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"selector_reproduction": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.control_tail_reliability_policy_20260713",
            "tools.measure_quant_tail_reliability_20260713",
        ),
        canonical_producers=("tools.measure_quant_tail_reliability_20260713", _RECEIPT),
        provenance=provenance,
    )


def build_fixed_design_correlated_gaussian_ridge_tail_v1() -> CanonicalEquation:
    """Build the explicit finite-design noncentral Gaussian tail law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "numeric closure requires residual covariance innovations and the top-k "
            "score-boundary margin envelope on a block-held-out surface"
        ),
        measurement_axis="[DERIVED finite-design law; no numeric or score authority]",
        hardware_substrate="substrate_independent_fixed_design_math",
        captured_at_utc=_UTC,
    )
    return CanonicalEquation(
        equation_id=FINITE_BOUND_ID,
        name="Finite fixed-design correlated-Gaussian ridge quadratic tail",
        one_line_summary=(
            "A noncentral Gaussian quadratic has an explicit Frobenius/operator/linear "
            "upper-tail radius; retained-mass transfer additionally needs a top-k margin envelope."
        ),
        latex_form=(
            r"\Pr\{Q-\mathbb EQ\ge2\sqrt{(\|A\|_F^2+2\|c\|_2^2)t}"
            r"+2\|A\|_{op}t\}\le e^{-t}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.control_interpolator_tail_reliability_20260713:"
            "correlated_gaussian_quadratic_upper_tail"
        ),
        domain_of_validity={
            "included": (
                "finite fixed design",
                "known correlated Gaussian covariance whitening",
                "positive-semidefinite quadratic loss",
            ),
            "excluded": (
                "guessed covariance",
                "arbitrary dependent non-Gaussian whitened coordinates",
                "numeric retained-mass claim without a measured top-k boundary-margin envelope",
                "borrowed proportional-asymptotic rate",
            ),
            "verdict_scope": "DERIVED symbolic close; present cached numeric close=false",
            "req_R": (
                "per-state residual innovations, covariance custody, state independence/cluster "
                "model, and score-boundary exact-mass envelope"
            ),
        },
        units_in={"A": "squared loss", "c": "squared-loss square root", "delta": "probability"},
        units_out={"upper_tail_excess": "squared loss"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=("tools.measure_quant_tail_reliability_20260713",),
        canonical_producers=(_MEMO,),
        provenance=provenance,
    )


def populate_control_tail_reliability_equations(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> tuple[CanonicalEquation, CanonicalEquation]:
    """Append both laws through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = (
        build_control_interpolator_tail_cvar_mean_gate_v1(),
        build_fixed_design_correlated_gaussian_ridge_tail_v1(),
    )
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="quant-tail-reliability; MEANS; cached local; score_claim=false; pointer unmoved",
        )
    return equations


__all__ = [
    "EQUATION_ID",
    "FINITE_BOUND_ID",
    "TailLambdaSelection",
    "TailReliabilityError",
    "build_control_interpolator_tail_cvar_mean_gate_v1",
    "build_fixed_design_correlated_gaussian_ridge_tail_v1",
    "correlated_gaussian_quadratic_upper_tail",
    "empirical_cvar",
    "loss_tail_summary",
    "populate_control_tail_reliability_equations",
    "retained_mass_tail_summary",
    "select_tail_lambda",
]
