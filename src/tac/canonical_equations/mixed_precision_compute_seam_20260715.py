# SPDX-License-Identifier: MIT
"""Canonical equation: bf16/fp16 compute-seam gradient-quality gate (#509 batch 3).

THE BUILD-OWED THIS FORMALIZES (memo ``.omx/research/wallclock_burndown_build_20260715.md`` §3
+ ``m5max_unconstrained_leverage_campaign_20260715`` constraint 1): the witness trainer had no
dtype seam — all forward/backward ran fp32 while M-series GPUs execute bf16/fp16 at ~2x the
fp32 rate (ESTIMATE-flagged; the max-Metal ceiling row from precision alone is ~1.5-1.8x sec/ep
IF grad-accum is ~60-70% of wall — both shares awaiting the #480 attribution receipts).

THE LAW (the admission gate, C0-lesson-aware): a low-precision compute arm is ADMISSIBLE for
training only when its POST-NORMALIZE update direction agrees with the same-master fp32
reference —

    cos_t = <u_lowp_t, u_ref_t> / (||u_lowp_t|| ||u_ref_t||),   r_t = ||u_lowp_t|| / ||u_ref_t||
    ADMIT  iff  median_t(cos_t) >= cos_min  AND  q_lo <= median_t(r_t) <= q_hi

where u = the update AFTER the trainer's clip+normalize pipeline (the C0 confound,
``perparam_normalize_masks_all_norm_clipping_c0_confound_20260715``: per-param normalize
divides out uniform per-tensor scales, so any PRE-normalize comparison grades a direction the
optimizer never applies), measured over the first N optimizer steps ALONG the fp32 reference
trajectory (the QC mode applies the reference update, so gradient quality is never conflated
with trajectory divergence).

Thresholds (cos_min=0.99, r in [0.9, 1.1]) are PROPOSED admission constants, operator-adjustable
— ASSUMED_AWAITING_VERIFICATION until the bounded n24 QC run measures the actual cosine/rel-norm
distribution (the anchor below is OWED; the run may justify tightening or loosening).

Mechanism: ``tac.witness_control.compute_dtype_seam`` (fp32 masters; cast-inside-the-trace;
entry shims; scorer/render/verdict/decode fp32; resume-safe — nothing persisted). DSL leg:
``curriculum_dsl.ComputeDtype`` (``--compute-dtype {fp32,bf16,fp16}``, default fp32 =
byte-identical). Sister law: ``autoclip_percentile_threshold_v1`` (the same burn-down's
magnitude-law cure; the QC gate refuses autoclip to keep the reference replication faithful).

means != ends: sec/ep lever candidate; NO score claim; every number ``[macOS-MLX advisory]``
NON-PROMOTABLE; pointer UNMOVED.
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

EQUATION_ID = "bf16_compute_seam_gradient_quality_v1"

_UTC = "2026-07-15T00:00:00Z"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/wallclock_burndown_build_20260715.md"

# PROPOSED admission constants (operator-adjustable; awaiting the n24 QC distribution).
COS_MIN_PROPOSED = 0.99
REL_NORM_BAND_PROPOSED = (0.9, 1.1)
# The max-Metal precision ceiling ESTIMATE (flagged, never a measurement): bf16 rate ~2x fp32
# on M-series GPU => sec/ep ceiling ~1.5-1.8x IF grad-accum is ~60-70% of wall.
FP16_RATE_VS_FP32_ESTIMATE = 2.0
SECEP_CEILING_ESTIMATE_BAND = (1.5, 1.8)


def gradient_quality_gate(
    cosines: np.ndarray,
    rel_norms: np.ndarray,
    cos_min: float = COS_MIN_PROPOSED,
    rel_band: tuple[float, float] = REL_NORM_BAND_PROPOSED,
) -> dict:
    """The law's pure math: median-based admission verdict over per-step QC rows.

    Deterministic fp64 numpy. Inputs are the per-step ``cosine`` / ``rel_norm`` fields of
    ``compute_dtype_quality.jsonl`` (produced by the trainer's
    ``--compute-dtype-quality-check`` mode via
    ``tac.witness_control.compute_dtype_seam.update_direction_stats``)."""
    c = np.asarray(cosines, np.float64)
    r = np.asarray(rel_norms, np.float64)
    if c.size == 0 or r.size == 0 or c.size != r.size:
        raise ValueError(
            f"gradient_quality_gate requires matched non-empty cosine/rel_norm arrays "
            f"(got {c.size} / {r.size})")
    lo, hi = float(rel_band[0]), float(rel_band[1])
    if not (0.0 < lo < hi):
        raise ValueError(f"rel_band must satisfy 0 < lo < hi, got {rel_band!r}")
    med_c = float(np.median(c))
    med_r = float(np.median(r))
    admit = bool(med_c >= float(cos_min) and lo <= med_r <= hi)
    return {
        "admit": admit,
        "median_cosine": med_c,
        "median_rel_norm": med_r,
        "n_steps": int(c.size),
        "cos_min": float(cos_min),
        "rel_band": [lo, hi],
        "worst_cosine": float(np.min(c)),
    }


def build_bf16_compute_seam_gradient_quality_v1() -> CanonicalEquation:
    """Build the compute-seam gradient-quality gate law with its OWED anchor."""
    anchor_qc_owed = EmpiricalAnchor(
        anchor_id="bf16_seam_n24_quality_check_owed_20260715",
        measurement_utc=_UTC,
        inputs={
            "arm": ("--compute-dtype bf16 --compute-dtype-quality-check 50 (n24, "
                    "--grad-clip-mode fixed, seed OFF, incumbent --grad-normalize per-param)"),
            "comparator": ("tac.witness_control.compute_dtype_seam.update_direction_stats "
                           "(POST-normalize; fp32-reference trajectory)"),
            "proposed_gate": {"cos_min": COS_MIN_PROPOSED,
                              "rel_band": list(REL_NORM_BAND_PROPOSED)},
        },
        predicted_output={
            "hypothesis": ("bf16 (fp32-range exponent) passes the direction gate; the sec/ep win "
                           "is a SEPARATE paired bench (ceiling ESTIMATE ~1.5-1.8x, flagged). "
                           "fp16 (no loss scaling) may fail — exposed for the matrix only."),
        },
        empirical_output={
            "status": ("OWED — the bounded n24 QC run is the admission measurement; NO adoption "
                       "claim before it; means != ends; pointer UNMOVED"),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="bounded n24 governed QC run (launch pending at registration time)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="bf16_compute_seam.gradient_quality_gate",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("bf16/fp16 compute-seam gradient-quality gate: admit iff median post-normalize "
              "update-direction cosine >= cos_min and median rel-norm in band, vs the "
              "same-master fp32 reference"),
        one_line_summary=(
            "Mixed-precision admission law: POST-normalize update-direction agreement (cosine + "
            "rel-norm medians) vs a same-master fp32 reference — the C0-aware --compute-dtype gate."
        ),
        latex_form=(
            r"\mathrm{ADMIT}\iff \mathrm{med}_t\,\cos(u^{lp}_t,u^{32}_t)\ge c_{\min}\ \wedge\ "
            r"q_{lo}\le \mathrm{med}_t\,\|u^{lp}_t\|/\|u^{32}_t\|\le q_{hi}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.mixed_precision_compute_seam_20260715:gradient_quality_gate"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "v9_cgauge_*"],
            "lever": ("tac.witness_dsl.curriculum_dsl.ComputeDtype "
                      "(--compute-dtype / --compute-dtype-quality-check)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("TRAINING-path only (2026-07-15 relaxed-identity directive): the seam keeps "
                     "the frozen-scorer forwards, render/R, verdict, EMA, checkpoints, and decode "
                     "fp32; masters restored after every traced call (resume-safe, nothing "
                     "persisted). Thresholds PROPOSED until the n24 QC distribution is measured. "
                     "fp16 has NO loss scaling — bf16 is the recommended arm."),
        },
        units_in={"cosines": "dimensionless", "rel_norms": "ratio",
                  "cos_min": "dimensionless", "rel_band": "ratio"},
        units_out={"admit": "bool", "median_cosine": "dimensionless",
                   "median_rel_norm": "ratio"},
        empirical_anchors=(anchor_qc_owed,),
        predicted_vs_empirical_residual={
            "n24_quality_check_owed": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # DSL leg: ComputeDtype factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the seam consumer
        ),
        canonical_producers=(
            "tac.witness_control.compute_dtype_seam",
        ),
        provenance=build_provenance_for_predicted(
            model_id="bf16_compute_seam_gradient_quality.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_bf16_compute_seam_gradient_quality_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the compute-seam quality gate (latest-row-wins).

    EQUATIONS leg of the #509 batch-3 bf16 seam; DSL leg = ``curriculum_dsl.ComputeDtype``;
    mechanism = ``tac.witness_control.compute_dtype_seam``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_bf16_compute_seam_gradient_quality_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="mixed_precision_compute_seam_20260715 (equations leg of the #509 batch-3 bf16 "
              "seam; DSL leg = ComputeDtype; n24 QC run owed)",
    )
    return eq


__all__ = [
    "COS_MIN_PROPOSED",
    "EQUATION_ID",
    "FP16_RATE_VS_FP32_ESTIMATE",
    "REL_NORM_BAND_PROPOSED",
    "SECEP_CEILING_ESTIMATE_BAND",
    "build_bf16_compute_seam_gradient_quality_v1",
    "gradient_quality_gate",
    "populate_bf16_compute_seam_gradient_quality_equation",
]
