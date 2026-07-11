# SPDX-License-Identifier: MIT
"""Canonical equation: SPD-cone water-filled pose-section codec Pareto-dominates the
#140 constant-levels low-rank codec (v1).

THE LAW (measured, advisory — a RATE law on the stored pose SECTION, not a d_pose law).
=======================================================================================
The stored pose targets form a small ``(n_pairs, pose_dim)`` matrix whose centered
covariance ``C`` is SPD. The #140 low-rank codec (``encode_pose_section_lowrank``) stores
the top-``rank`` SVD eigenbasis and quantizes each retained principal mode with the SAME
``levels`` (constant per-mode bit DEPTH) + a HARD rank truncation. The SPD-cone codec
(``encode_pose_section_spd``) uses the SAME SVD basis but the rate-distortion-OPTIMAL
allocation for a Gaussian source under MSE — reverse WATER-FILLING: a constant
quantization STEP (equal per-mode distortion ``θ``) with graceful (soft) mode-dropping
below the water level.

MEASURED on the REAL contest pose section (600×6, ``gt_targets_n600.pt``): at the #140
shipped default's operating point (rank-4/levels-511 = 2563 B @ MSE 2.70e-5) the SPD codec
reaches ≤ that MSE in **1869 B (-27.1%)**, or 9.3× lower MSE at matched bytes — a PARETO
rate cut (fewer bytes at no-worse fidelity ⇒ contest ``d_pose`` CANNOT worsen). On the RD
frontier (both codecs tuned to their best curve) the SPD frontier dominates across the
near-lossless regime and the byte-gap **scales with the covariance's Hilbert projective
distance** ``d_H = log(λ_max/λ_min)`` (Nielsen 2307.10644): anisotropic spectrum →
large gap, near-isotropic → the frontiers nearly coincide. So the mechanism IS the
SPD-cone water-filling geometry.

HONEST SCOPE (CLAUDE.md NO-FAKE class 8): the section-reconstruction MSE maps to contest
``d_pose`` only through the byte-closed decode; the NET-score effect is confirmed only by a
byte-closed ``upstream/evaluate.py``. This is a candidate RATE refinement (composes with
the stored-pose sidecar path), NOT a pointer move. Pointer 0.19108282 UNMOVED.

Artifact: ``.omx/research/spd_cone_pose_codec_ab_measured.json``.
"""
from __future__ import annotations

import math

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pose_spd_cone_waterfill_rate_v1"
_UTC = "2026-07-10T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_ARTIFACT = ".omx/research/spd_cone_pose_codec_ab_measured.json"


def hilbert_projective_distance(cov_eigs) -> float:
    """``d_H = log(λ_max/λ_min)`` — the SPD-cone anisotropy that predicts the byte-gap
    (large ⇒ SPD water-filling wins big; ~0 ⇒ near-parity with constant-levels)."""
    e = np.asarray(cov_eigs, dtype=np.float64)
    e = e[e > 1e-12]
    if e.size == 0:
        return 0.0
    return float(math.log(float(e.max()) / float(e.min())))


def build_pose_spd_cone_waterfill_rate_v1() -> CanonicalEquation:
    """Builder for ``pose_spd_cone_waterfill_rate_v1`` (Catalog #344 registry)."""
    anchor = EmpiricalAnchor(
        anchor_id="spd_vs_lowrank_real_pose_section_matched_mse_20260710",
        measurement_utc=_UTC,
        inputs={
            "pose_section": "gt_targets_n600.pt::pose (600x6)",
            "baseline_codec": "encode_pose_section_lowrank(rank=4,levels=511)",
            "baseline_bytes": 2563,
            "baseline_mse": 2.704832e-05,
            "cov_eigenvalues": [1.5810e00, 1.7196e-03, 9.1503e-04, 3.7846e-04, 7.3770e-05, 4.2498e-05],
            "hilbert_projective_distance": 10.52,
            "measurement_axis": _ADVISORY,
        },
        predicted_output={
            "note": "SPD Pareto-dominates: fewer bytes at <= baseline MSE (rate cut, d_pose safe)",
        },
        empirical_output={
            "spd_matched_mse_bytes": 1869,
            "spd_matched_mse_mse": 2.679873e-05,
            "byte_fraction_saved": 0.271,
            "spd_matched_bytes_mse_ratio": 9.3,
            "frontier_gap_bytes_at_2p7e5_mse": 1024,
        },
        residual=0.0,  # first empirical observation; no prior prediction
        source_artifact=_ARTIFACT,
        measurement_method="encode_roundtrip_byte_and_mse_on_real_pose_section_frontier_and_matched_point",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_ARTIFACT,
            reactivation_criteria="byte_closed_upstream_evaluate_on_archive_with_pose_codec_spd",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="SPD-cone water-filled pose-section codec vs #140 low-rank (Pareto rate)",
        one_line_summary=(
            "Equal-distortion (constant-step) water-filling on the SPD pose covariance "
            "Pareto-beats constant-levels low-rank; byte-gap scales with Hilbert d_H."
        ),
        latex_form=(
            r"L_i = \mathrm{range}_i / \sqrt{12\,\theta},\quad "
            r"D_i = \min(\lambda_i, \theta),\quad "
            r"\Delta\text{bytes}(A) \uparrow \text{ with } d_H = \log(\lambda_{\max}/\lambda_{\min})"
        ),
        python_callable_module_path=(
            "tac.torch_vehicle.pose_spd_codec:spd_pose_section_fidelity"
        ),
        domain_of_validity={
            "pose_section_shape": "(n_pairs, pose_dim) with n_pairs >> pose_dim",
            "pose_dim_range": [2, 12],
            "n_pairs_range": [100, 600],
            "distortion": "reconstruction_MSE (maps to d_pose only via byte-closed decode)",
            "measurement_axes": ["[macOS-CPU advisory]"],
        },
        units_in={
            "stored_pose": "float_pose_target_matrix",
            "water_level": "float_per_mode_MSE_theta",
        },
        units_out={
            "bytes": "int_pose_section_byte_count",
            "mse": "float_reconstruction_mse",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "spd_vs_lowrank_real_pose_matched_mse": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.torch_vehicle.pose_film.build_archive_with_pose",  # pose_codec='spd'
            "tac.torch_vehicle.pose_film.parse_pose_section",
        ),
        canonical_producers=(
            "tac.torch_vehicle.pose_spd_codec.spd_pose_section_fidelity",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_ARTIFACT,
            reactivation_criteria="byte_closed_upstream_evaluate_on_archive_with_pose_codec_spd",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "build_pose_spd_cone_waterfill_rate_v1",
    "hilbert_projective_distance",
]
