# SPDX-License-Identifier: MIT
"""Closed task-space variational law for the frozen contest scorer.

The state is the evaluator's finite-dimensional task output, not a symbolic
composition of every CNN layer.  Segmentation lives in the exact rank-four
quotient of the final affine SegNet head; pose lives in the first-six PoseNet
output coordinate ``xi``; rate is the exact archive byte count.  The decoder
and realization operator couple those coordinates through an explicit
feasibility constraint rather than being hidden inside the action.

This module is research-only.  It deliberately refuses to turn an unknown
description language or an unmeasured archive rate-distortion function into a
numeric constrained minimum.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

RATE_DENOMINATOR_BYTES = 37_545_489
RATE_WEIGHT = 25
SEG_WEIGHT = 100.0
POSE_WEIGHT = 10.0
BYTE_CAP = 154_600
TARGET_SCORE = 0.15
RATE_PRICE_EXACT = Fraction(RATE_WEIGHT, RATE_DENOMINATOR_BYTES)

RECEIPT_PATH = ".omx/research/closed_scorer_variational_de_20260721T173654Z.json"
MEMO_PATH = ".omx/research/closed_scorer_variational_de_20260721T173654Z.md"
CALIBRATION_UTC = "2026-07-21T17:36:54Z"

TASKSPACE_EQUATION_ID = "closed_scorer_taskspace_variational_functional_v1"
STATIONARITY_EQUATION_ID = "closed_scorer_viscosity_kkt_stationarity_v1"
REACHABILITY_EQUATION_ID = "closed_scorer_archive_reachability_bound_v1"


@dataclass(frozen=True)
class ReachabilityCertificate:
    """Exact arithmetic that can be certified without inventing a decoder."""

    byte_cap: int
    target_score: float
    rate_at_cap: Fraction
    residual_distortion_budget: float
    status: str
    missing_witness: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_cap": self.byte_cap,
            "target_score": self.target_score,
            "rate_at_cap_exact": f"{self.rate_at_cap.numerator}/{self.rate_at_cap.denominator}",
            "rate_at_cap": float(self.rate_at_cap),
            "residual_distortion_budget": self.residual_distortion_budget,
            "status": self.status,
            "missing_witness": list(self.missing_witness),
        }


def _nonnegative_finite(value: float | int, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def rate_term_exact(archive_bytes: int) -> Fraction:
    """Return ``25 B / 37_545_489`` without floating-point rounding."""

    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int):
        raise TypeError("archive_bytes must be an integer")
    if archive_bytes < 0:
        raise ValueError("archive_bytes must be >= 0")
    return Fraction(RATE_WEIGHT * archive_bytes, RATE_DENOMINATOR_BYTES)


def closed_scorer_action(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """Evaluate the frozen score on already-realized task coordinates.

    Authority is not inferred: the caller remains responsible for exact
    archive/receiver/R/scorer custody of ``d_seg`` and ``d_pose``.
    """

    seg = _nonnegative_finite(d_seg, "d_seg")
    pose = _nonnegative_finite(d_pose, "d_pose")
    return SEG_WEIGHT * seg + math.sqrt(POSE_WEIGHT * pose) + float(rate_term_exact(archive_bytes))


def power_laguerre_labels(
    quotient_points: np.ndarray,
    sites: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Assign exact affine-head cells in rank-four Laguerre coordinates.

    For affine logits ``z_c(q)=a_c.q+b_c``, take ``s_c=a_c/2`` and
    ``omega_c=b_c+||s_c||^2``.  Maximizing the expression below is identical
    to minimizing the corresponding power distance.
    """

    q = np.asarray(quotient_points, dtype=np.float64)
    s = np.asarray(sites, dtype=q.dtype)
    w = np.asarray(weights, dtype=q.dtype)
    if q.ndim != 2 or s.ndim != 2 or q.shape[1] != s.shape[1]:
        raise ValueError("quotient_points and sites must have compatible rank-2 shapes")
    if w.shape != (s.shape[0],):
        raise ValueError("weights must contain one value per site")
    if not np.isfinite(q).all() or not np.isfinite(s).all() or not np.isfinite(w).all():
        raise ValueError("Laguerre inputs must be finite")
    # Explicit reduction avoids platform-BLAS warning/drift at this tiny fixed
    # rank and mirrors the frozen-head native-f32 diagnostic.
    dot = np.sum(q[:, None, :] * s[None, :, :], axis=2, dtype=q.dtype)
    scores = 2.0 * dot + w[None, :] - np.sum(s * s, axis=1, dtype=q.dtype)[None, :]
    return np.argmax(scores, axis=1).astype(np.int64, copy=False)


def categorical_bregman_debt(logits: np.ndarray, target_class: np.ndarray) -> np.ndarray:
    """Return ``D_F*(e_c || softmax(z)) = logsumexp(z)-z_c``.

    ``F*(p)=sum p log p``.  Its vertex debt is finite categorical KL, and
    selecting the minimum-debt vertex is exactly the scorer's argmax cell.
    This finite divergence must not be conflated with a Fisher-natural
    inverse or the V9 model-factorization claim.
    """

    z = np.asarray(logits, dtype=np.float64)
    c = np.asarray(target_class, dtype=np.int64)
    if z.ndim != 2 or c.shape != (z.shape[0],):
        raise ValueError("logits must be NxC and target_class must be N")
    if np.any(c < 0) or np.any(c >= z.shape[1]):
        raise ValueError("target_class is outside the logit class range")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    z_max = np.max(z, axis=1, keepdims=True)
    log_partition = z_max[:, 0] + np.log(np.sum(np.exp(z - z_max), axis=1))
    return log_partition - z[np.arange(z.shape[0]), c]


def bregman_voronoi_labels(logits: np.ndarray) -> np.ndarray:
    """Select the negative-entropy Bregman vertex of minimum debt."""

    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must be NxC")
    classes = np.arange(z.shape[1], dtype=np.int64)
    debts = np.stack(
        [categorical_bregman_debt(z, np.full(z.shape[0], c, dtype=np.int64)) for c in classes],
        axis=1,
    )
    return np.argmin(debts, axis=1).astype(np.int64, copy=False)


def pose_task_quadratic(xi: np.ndarray, xi_target: np.ndarray) -> float:
    """Exact frozen pose MSE in scorer-output coordinates, Hessian ``I/3``."""

    value = np.asarray(xi, dtype=np.float64)
    target = np.asarray(xi_target, dtype=np.float64)
    if value.shape != target.shape or value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("xi and xi_target must both be Nx6")
    if not np.isfinite(value).all() or not np.isfinite(target).all():
        raise ValueError("xi and xi_target must be finite")
    return float(np.mean((value - target) ** 2))


def stationarity_residual(
    *,
    relaxed_seg_gradient: Iterable[float],
    pose_debt_gradient: Iterable[float],
    code_length_gradient: Iterable[float],
    d_pose: float,
    byte_multiplier: float = 0.0,
) -> dict[str, Any]:
    """Evaluate differentiable-stratum KKT stationarity in a shared code chart.

    Argmax jumps themselves are interpreted in the viscosity/subgradient
    sense.  The archive hard-cap multiplier ``mu_B`` is distinct from the
    fixed objective byte price ``25/N``.
    """

    seg = np.asarray(tuple(relaxed_seg_gradient), dtype=np.float64)
    pose = np.asarray(tuple(pose_debt_gradient), dtype=np.float64)
    rate = np.asarray(tuple(code_length_gradient), dtype=np.float64)
    if seg.ndim != 1 or pose.shape != seg.shape or rate.shape != seg.shape:
        raise ValueError("all gradients must be same-length vectors")
    if not np.isfinite(seg).all() or not np.isfinite(pose).all() or not np.isfinite(rate).all():
        raise ValueError("all gradients must be finite")
    pose_debt = _nonnegative_finite(d_pose, "d_pose")
    if pose_debt == 0.0 and np.any(pose != 0.0):
        raise ValueError("sqrt pose term is nondifferentiable at zero with nonzero pose residual")
    pose_scale = 0.0 if pose_debt == 0.0 else math.sqrt(POSE_WEIGHT) / (2.0 * math.sqrt(pose_debt))
    mu_b = _nonnegative_finite(byte_multiplier, "byte_multiplier")
    vector = SEG_WEIGHT * seg + pose_scale * pose + (float(RATE_PRICE_EXACT) + mu_b) * rate
    return {
        "stationarity_vector": vector.tolist(),
        "l2_residual": float(np.linalg.norm(vector)),
        "pose_scale": pose_scale,
        "objective_rate_price_exact": f"{RATE_PRICE_EXACT.numerator}/{RATE_PRICE_EXACT.denominator}",
        "hard_byte_cap_multiplier_mu_B": mu_b,
        "interpretation": "SMOOTH_STRATUM_ONLY; CELL_INTERFACES_REQUIRE_VISCOSITY_SUBGRADIENT",
    }


def reachability_certificate(
    *, byte_cap: int = BYTE_CAP, target_score: float = TARGET_SCORE
) -> ReachabilityCertificate:
    """Certify exact budget arithmetic and refuse an unevidenced minimum."""

    target = _nonnegative_finite(target_score, "target_score")
    rate = rate_term_exact(byte_cap)
    remaining = target - float(rate)
    return ReachabilityCertificate(
        byte_cap=byte_cap,
        target_score=target,
        rate_at_cap=rate,
        residual_distortion_budget=remaining,
        status="UNRESOLVED_REQUIRES_BYTE_CLOSED_WITNESS" if remaining >= 0.0 else "IMPOSSIBLE_AT_BYTE_CAP",
        missing_witness=(
            "legal deterministic decoder/description language",
            "exact archive bytes <= cap",
            "receiver parse-back and runtime custody",
            "realized-through-R frozen SegNet/PoseNet debts",
        ),
    )


def _provenance(path: str, *, axis: str) -> Any:
    return build_provenance_for_research_sidecar(
        sidecar_path=path,
        reactivation_criteria=(
            "promotion requires exact archive bytes, receiver parse-back, runtime custody, "
            "and contest-CPU/CUDA frozen-evaluator replay"
        ),
        measurement_axis=axis,
        hardware_substrate="macos_arm64_cpu_or_derived_math",
        captured_at_utc=CALIBRATION_UTC,
    )


def _load_fidelity_anchor() -> EmpiricalAnchor:
    receipt_path = Path(RECEIPT_PATH)
    payload = json.loads(receipt_path.read_text())
    if payload.get("schema") != "closed_scorer_variational_de_fidelity.v1":
        raise ValueError("D1 fidelity receipt schema drift")
    fidelity = payload["d1_fidelity"]
    residual = float(fidelity["max_normalized_residual"])
    return EmpiricalAnchor(
        anchor_id="closed_scorer_rank4_bregman_real_tiles_seed1234_20260721",
        measurement_utc=str(payload["written_at_utc"]),
        inputs={
            "seed": payload["seed"],
            "heldout_tiles": fidelity["heldout_tiles"],
            "heldout_pixels": fidelity["heldout_pixels"],
            "segnet_weights_sha256": payload["source_custody"]["segnet_weights_sha256"],
            "task_coordinate": "rank4_final_head_quotient",
        },
        predicted_output={
            "native_f32_power_vs_live_argmax_disagreement_rate": 0.0,
            "bregman_voronoi_vs_live_argmax_disagreement_rate": 0.0,
        },
        empirical_output={
            "native_f32_power_vs_live_argmax_disagreement_rate": fidelity[
                "native_f32_power_vs_live_argmax_disagreement_rate"
            ],
            "bregman_voronoi_vs_live_argmax_disagreement_rate": fidelity[
                "bregman_voronoi_vs_live_argmax_disagreement_rate"
            ],
            "cached_target_vs_live_argmax_disagreement_rate": fidelity[
                "cached_target_vs_live_argmax_disagreement_rate"
            ],
            "axis": payload["authority"],
        },
        residual=residual,
        source_artifact=RECEIPT_PATH,
        measurement_method=(
            "20 deterministic held-out 32x32 tiles from four real gt_n600 frames; frozen "
            "SegNet CPU forward, captured final-head features, independent rank-four quotient "
            "convolution, native-f32 Laguerre assignment, and negative-entropy Bregman assignment"
        ),
        provenance=_provenance(RECEIPT_PATH, axis="[macOS-CPU advisory real frozen-SegNet tiles]"),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _pending_anchor(anchor_id: str, statement: str) -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc=CALIBRATION_UTC,
        inputs={"statement": statement},
        predicted_output={"status": "NUMERIC_WITNESS_OWED"},
        empirical_output={"status": "NOT_MEASURED"},
        residual=1.0,
        source_artifact=MEMO_PATH,
        measurement_method="derived task-space law; archive/receiver measurement intentionally absent",
        provenance=_provenance(MEMO_PATH, axis="[DERIVED; archive witness owed]"),
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
    )


def build_taskspace_equation() -> CanonicalEquation:
    anchor = _load_fidelity_anchor()
    return CanonicalEquation(
        equation_id=TASKSPACE_EQUATION_ID,
        name="Closed frozen-scorer task-space variational functional",
        one_line_summary="Rank-4 Laguerre/Bregman Seg cells + PoseNet xi quadratic + exact archive MDL, coupled only by realized decoder feasibility.",
        latex_form=(
            r"\mathcal S[C]=100|\Omega|^{-1}\sum_p\mathbf 1[\mathcal L(q_p(C))\ne c_p^*]"
            r"+\sqrt{10\,\|\xi(C)-\xi^*\|_2^2/(6N)}+25L_{MDL}(C)/37545489,\ "
            r"(q,\xi)=(\Pi_4 h_{Seg},h_{Pose})\circ R_8\circ G(C)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.closed_scorer_variational_de_20260721:closed_scorer_action"
        ),
        domain_of_validity={
            "seg_domain": "frozen SegNet final affine head rank-4 quotient after exact realization",
            "seg_geometry": "Laguerre power cells; equivalent negative-entropy Bregman vertex cells",
            "pose_domain": "first-six frozen PoseNet output xi; exact output-space Hessian I/3",
            "rate_domain": "exact legal archive bytes/MDL only",
            "coupling": "(q,xi)=task_map(realize_uint8_R(inflate(C)))",
            "excludes": ["symbolic full-CNN Euler-Lagrange expansion", "proxy entropy as exact bytes", "smooth ODE across argmax jumps"],
            "research_only": True,
            "pointer_moved": False,
        },
        units_in={"d_seg": "fraction", "d_pose": "mean_squared_PoseNet_output", "archive_bytes": "bytes"},
        units_out={"score": "contest_score_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"real_frozen_segnet_20_tile_max": anchor.residual},
        last_calibration_utc=CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("einstein_kolmogorov_ultra.U1", "tac.witness_autoconfig"),
        canonical_producers=("segnet_head_rank4_linear_flipdist_v1", "fisher_curvature_equals_categorical_fisher_trace_caustic_v1", RECEIPT_PATH),
        provenance=_provenance(RECEIPT_PATH, axis="[macOS-CPU advisory real frozen-SegNet tiles]"),
    )


def build_stationarity_equation() -> CanonicalEquation:
    anchor = _pending_anchor(
        "closed_scorer_stationarity_archive_witness_owed_20260721",
        "viscosity-HJ/SE(3)/entropy KKT must be evaluated on a byte-closed decoder chart",
    )
    return CanonicalEquation(
        equation_id=STATIONARITY_EQUATION_ID,
        name="Viscosity/subgradient KKT stationarity for closed scorer action",
        one_line_summary="Smooth-stratum code gradients combine Seg cell force, SE(3) pose force, and exact byte price; cell jumps use viscosity/subgradients.",
        latex_form=(
            r"0\in100\partial_C D_{seg}+\frac{\sqrt{10}}{2\sqrt{D_{pose}}}J_\xi^T(\xi-\xi^*)"
            r"+(25/37545489+\mu_B)\partial_C L_{MDL}+N_{\mathcal F}(C),\ \mu_B(L-154600)=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.closed_scorer_variational_de_20260721:stationarity_residual"
        ),
        domain_of_validity={
            "smooth": "within fixed argmax-cell strata",
            "nonsmooth": "viscosity Hamilton-Jacobi/subgradient at separatrices",
            "pose": "SE(3) geodesic xi chart with measured scorer pullback required",
            "rate": "entropy-optimal coder with exact length/parse-back residual gate",
            "lambda_star": "25/37545489 is objective byte price; hard-cap KKT multiplier is distinct mu_B",
            "research_only": True,
        },
        units_in={"gradients": "task_debt_per_code_coordinate", "d_pose": "mean_squared_PoseNet_output", "byte_multiplier": "score_per_byte"},
        units_out={"stationarity_residual": "score_per_code_coordinate"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"byte_closed_stationarity": 1.0},
        last_calibration_utc=CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("einstein_kolmogorov_ultra.U1",),
        canonical_producers=(TASKSPACE_EQUATION_ID, "flip_margin_step_law_v1", "instant_projected_input_adjoint_v1"),
        provenance=_provenance(MEMO_PATH, axis="[DERIVED; archive witness owed]"),
    )


def build_reachability_equation() -> CanonicalEquation:
    anchor = _pending_anchor(
        "closed_scorer_sub015_reachability_witness_owed_20260721",
        "154600-byte sub-0.15 witness and exact constrained minimum are not present",
    )
    return CanonicalEquation(
        equation_id=REACHABILITY_EQUATION_ID,
        name="Archive-constrained lower-bound and sub-0.15 reachability certificate",
        one_line_summary="Exact byte arithmetic is closed; the numeric constrained minimum remains unresolved until a legal byte-closed witness or valid lower-bound relaxation lands.",
        latex_form=(
            r"S^*_{154600}=\inf_{C\in\mathcal A,\,L(C)\le154600}\mathcal S[C],\ "
            r"25(154600)/37545489=0.102941794\ldots,\ "
            r"100D_{seg}+\sqrt{10D_{pose}}<0.047058206\ldots"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.closed_scorer_variational_de_20260721:reachability_certificate"
        ),
        domain_of_validity={
            "legal_archive_cap_bytes": BYTE_CAP,
            "target_score": TARGET_SCORE,
            "S_floor_0_118_status": "EMPIRICAL_ACHIEVER_UPPER_BOUND_NOT_PROVED_INFIMUM",
            "numeric_minimum_status": "UNRESOLVED_REQUIRES_DESCRIPTION_LANGUAGE_AND_EXACT_RECEIVER",
            "research_only": True,
            "pointer_moved": False,
        },
        units_in={"archive_bytes": "bytes", "target_score": "contest_score_units"},
        units_out={"rate_at_cap": "contest_score_units", "residual_distortion_budget": "contest_score_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"exact_archive_constrained_minimum": 1.0},
        last_calibration_utc=CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("einstein_kolmogorov_ultra.U2", "einstein_kolmogorov_ultra.U3"),
        canonical_producers=(TASKSPACE_EQUATION_ID, "scorer_conditional_joint_rate_distortion_floor_v1"),
        provenance=_provenance(MEMO_PATH, axis="[DERIVED exact arithmetic; reachability unresolved]"),
    )


def populate_closed_scorer_variational_equations(
    *, path: str | Path | None = None, lock_path: str | Path | None = None,
    agent: str | None = None, subagent_id: str | None = None,
) -> tuple[CanonicalEquation, ...]:
    """Append U1/U2/U3 equations to the canonical registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = (build_taskspace_equation(), build_stationarity_equation(), build_reachability_equation())
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="closed scorer variational DE; research-only; pointer unchanged; MAIN review required",
        )
    return equations


__all__ = [
    "BYTE_CAP", "RATE_PRICE_EXACT", "REACHABILITY_EQUATION_ID", "STATIONARITY_EQUATION_ID",
    "TARGET_SCORE", "TASKSPACE_EQUATION_ID", "bregman_voronoi_labels", "categorical_bregman_debt",
    "closed_scorer_action", "populate_closed_scorer_variational_equations", "pose_task_quadratic",
    "power_laguerre_labels", "rate_term_exact", "reachability_certificate", "stationarity_residual",
]
