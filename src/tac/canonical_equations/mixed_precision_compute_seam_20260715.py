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

Thresholds (cos_min=0.99, r in [0.9, 1.1]): the bounded n24 QC run is MEASURED (anchor
``bf16_seam_n24_quality_check_measured_20260715``, 60 steps): gate ADMIT — median cosine
0.992538 (clearance +0.0025 over the bar; p10 0.9869 sits BELOW it), median rel_norm 1.0000
(pinned ~1 by construction under per-param normalize). CE-stage/first-60-steps scope only;
stage-boundary re-runs + the paired sec/ep speed bench remain OWED before any adoption.

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
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

_ADVISORY_MLX = "[macOS-MLX research-signal] NON-PROMOTABLE"

EQUATION_ID = "bf16_compute_seam_gradient_quality_v1"

_UTC = "2026-07-15T00:00:00Z"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/wallclock_burndown_build_20260715.md"

# Admission constants (operator-adjustable). First anchor MEASURED 2026-07-15: median
# cosine 0.992538 clears cos_min by +0.0025 (p10 0.9869 below the bar — median law holds).
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
    """Build the compute-seam gradient-quality gate law with its MEASURED n24 QC anchor."""
    anchor_qc_measured = EmpiricalAnchor(
        anchor_id="bf16_seam_n24_quality_check_measured_20260715",
        measurement_utc="2026-07-15T20:55:00Z",
        inputs={
            "arm": ("--dsl-lever ComputeDtypeBf16QCGate (bf16 + 60-step QC window + fixed clip + "
                    "seed OFF; incumbent --grad-normalize per-param), config v9_cgauge_ideal_mod19, "
                    "gt_n24, seed 0 — experiments/results/levelset_n24_bf16qc_20260715"),
            "comparator": ("tac.witness_control.compute_dtype_seam.update_direction_stats "
                           "(POST-normalize; fp32-reference trajectory)"),
            "gate": {"cos_min": COS_MIN_PROPOSED, "rel_band": list(REL_NORM_BAND_PROPOSED)},
        },
        predicted_output={
            "hypothesis": ("bf16 (fp32-range exponent) passes the direction gate; the sec/ep win "
                           "is a SEPARATE paired bench (ceiling ESTIMATE ~1.5-1.8x, flagged). "
                           "fp16 (no loss scaling) may fail — exposed for the matrix only."),
        },
        empirical_output={
            "gate_verdict": "ADMIT",
            "n_steps": 60,
            "median_cosine": 0.992538,
            "median_rel_norm": 1.0000003,
            "distribution": {
                "cosine_min": 0.9454, "cosine_p10": 0.9869, "cosine_p90": 0.9962,
                "cosine_max": 0.9971, "frac_ge_0p99": 0.783, "frac_ge_0p95": 0.983,
                "rel_norm_range": [1.000000, 1.000012],
            },
            "per_group_cosine_median": {
                "code": 0.9913, "film": 0.9813, "hidden": 0.9954, "in_proj": 0.9930,
                "out_sdf": 0.9984, "out_tex": 0.9976, "palette": 0.9973,
                "pose_carrier": 0.0,
            },
            "per_group_notes": ("pose_carrier cosine 0.0 = INACTIVE group (zero grads in the "
                                "trunk phase, w_pose 0) not a quality failure; film worst early "
                                "step 0.5623, settles to 0.9813 median; rel_norm pinned ~1.0 BY "
                                "CONSTRUCTION under per-param normalize (the C0 lesson is why the "
                                "gate grades DIRECTION)"),
            "honest_scope": ("median-law ADMIT clears cos_min 0.99 by 0.0025 with p10 BELOW the "
                             "bar (0.9869) — the admission holds under the registered median law "
                             "but is NOT a large margin; first-60-steps CE-stage only; later "
                             "stages (l7/Muon/finisher) un-measured; the SPEED effect is NOT "
                             "measured by this run (QC computes both grads + steps with the fp32 "
                             "reference => its sec/ep is not the bf16 number)"),
            "verdict_scope": ("FORMULATION — n24, single seed, CE-stage first 60 opt steps, "
                              "per-param lineage, seed-islands off; [macOS-MLX research-signal] "
                              "NON-PROMOTABLE; no score/adoption claim; pointer UNMOVED"),
        },
        residual=0.002538,  # measured median_cosine minus the 0.99 gate line (the clearance)
        source_artifact="experiments/results/levelset_n24_bf16qc_20260715/compute_dtype_quality.jsonl",
        measurement_method=("bounded n24 governed QC run via the launcher-composable "
                            "ComputeDtypeBf16QCGate; 60 per-step receipts; gate evaluated by "
                            "gradient_quality_gate (median law); sanctioned stop after ep25 "
                            "verdict (fp32-reference trajectory d_seg 0.018829)"),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="experiments/results/levelset_n24_bf16qc_20260715/compute_dtype_quality.jsonl",
            reactivation_criteria=("re-run QC at stage boundaries (l7/Muon/finisher norms differ); "
                                   "paired sec/ep bench (bf16 no-QC vs fp32) = the speed anchor "
                                   "OWED before any adoption; fp16 arm un-measured"),
            measurement_axis=_ADVISORY_MLX,
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
        empirical_anchors=(anchor_qc_measured,),
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
