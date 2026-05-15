#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cathedral autopilot RECOMMENDER - ranked technique selection from evidence.

Complement (not duplicate) of the existing rate-distortion autopilot at
``experiments/run_cathedral_autopilot.py``:

  experiments/run_cathedral_autopilot.py - solves the rate-distortion
    inner problem on a SYNTHETIC substrate (gradient descent + dual
    ascent on the Lagrangian) to validate the optimization machinery
  tools/cathedral_autopilot.py (this) - RECOMMENDS the next technique
    to apply on a REAL operator state, using the pre-built technique
    catalog + Shannon ladder + dispatch-advisor + floor-explorer

Both consume the canonical contest formula in
``tac.contest_rate_distortion_system`` (codex-built).

This recommender ingests an operator state (d_seg, d_pose, archive_bytes)
+ optional target_score + optional prior-execution evidence, and emits a
ranked playbook of next actions. It composes:

  - ``tac.contest_rate_distortion_system`` - canonical contest formula (codex)
  - ``tac.score_geometry`` - torch-free analytics + inverse curves
  - ``tac.score_geometry_floor_explorer`` - what-if technique floors
  - ``tac.score_geometry_stacking`` - Volterra cross-axis interaction
  - The meta-Lagrangian search (``tac.optimizer.meta_lagrangian``)
  - The PR101 entropy-floor triage reports
  - The charged Markov-table and iid empirical floor findings

The recommender does NOT dispatch GPU jobs. It outputs a structured plan
with concrete shell commands the operator (or
``tools/parallel_dispatch_top_k.py``) can execute. Pure CPU + math; no
scorer load; no contest score claims.

Feedback-loop mode (``--prior-plan-output``): given a prior plan's
execution evidence, the recommender re-ranks techniques by EMPIRICAL
gain-per-cost from prior dispatches, not just predicted. This is the
continual-learning hook the operator asked for.

Usage::

    .venv/bin/python tools/cathedral_autopilot.py plan \\
        --d-seg 6.7e-4 --d-pose 3.4e-5 --archive-bytes 178144 \\
        --target-score 0.190 \\
        --output reports/cathedral_autopilot_plan.json

    .venv/bin/python tools/cathedral_autopilot.py plan-from-pareto \\
        --pareto-json reports/pareto_3axis.json \\
        --target-score 0.155 \\
        --output reports/cathedral_autopilot_plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.authority_contract import normalize_score_authority_fields  # noqa: E402
from tac.optimization.candidate_evidence_contract import (  # noqa: E402
    has_positive_exact_cuda_evidence_marker,
    promotable_exact_cuda_evidence_blockers,
)
from tac.optimization.cuda_cpu_axis_calibration import CudaCpuCalibration  # noqa: E402
from tac.score_geometry import (  # noqa: E402
    contest_score,
    equal_score_curve_archive_bytes,
    equal_score_curve_d_pose,
    importance_flip_threshold,
    information_floor,
    operating_regime,
    score_decomposition,
    score_gradient,
)

TOOL_NAME = "tools/cathedral_autopilot.py"
SCHEMA_VERSION = "cathedral_autopilot.v1"
EVIDENCE_GRADE = "[CPU-prep planning-only]"


# Empirically-anchored substrate baselines (from session memory)
PR101_SUBSTRATE = {
    "label": "PR101_HNeRV_FT_microcodec",
    "n_elements": 228_958,
    "n_quant": 127,
    "archive_overhead_bytes": 16_094,
    "weighted_avg_empirical_bits_per_element": 5.5843,
    "uniform_floor_bytes": 216_109,
    "empirical_floor_bytes": 175_916,
    "brotli_optuna_optimum": 178_144,
    "per_tensor_aac": 178_181,
    "iid_per_tensor_floor": 159_822,
    "markov1_oracle_floor_bytes": 152_106,
    "markov1_sparse_varint_table_archive_bytes": 200_219,
    "markov1_dense_u16_table_archive_bytes": 209_103,
}


# Implementation-vs-model-gap audit (2026-05-08): every catalog row carries
# a ``model_spec`` declaring the implementation contract that "faithful test"
# means. The audit at
# ``feedback_implementation_vs_model_gap_audit_20260508.md`` documented 4
# falsifications that tested INFERIOR representatives of their technique
# class (capacity mismatch, substrate mismatch, shape-family mismatch,
# variant mismatch). The model_spec field structurally extincts that bug
# class:
#   - capacity_constraint: e.g., "<=200 params" for tiny_nn
#   - architecture_class: canonical class name (e.g., "factorized_softmax")
#   - substrate_constraint: e.g., "1d_quantized_symbols" / "2d_natural_image"
#   - canonical_shape_family: e.g., "kaiming+laplace+spike-and-slab"
#   - variant_required: list of variants the lane class subsumes
#
# The new preflight check `check_evidence_implementation_matches_model_spec`
# scans `reports/cathedral_autopilot_evidence.jsonl` and verifies each row's
# tested implementation matches the corresponding catalog model_spec. The
# in-process self-protection layer in ``update_catalog_from_evidence`` warns
# (does not break) when an evidence row's source string indicates a
# divergent implementation.
#
# Encoder-side technique catalog (from grand-council synthesis)
ENCODER_TECHNIQUES = [
    {
        "name": "brotli_optuna_default",
        "predicted_archive_bytes": 178_144,
        "cost_hours": 0.5,
        "cost_dollars": 0.0,
        "risk": "lossless",
        "evidence_grade": "[contest-CUDA]",
        "description": "Current best brotli + Optuna q=11/lgwin=16/lgblock=19",
        "model_spec": {
            "capacity_constraint": "n/a",
            "architecture_class": "brotli_lossless",
            "substrate_constraint": "1d_quantized_symbols",
            "canonical_shape_family": "n/a_lossless",
            "variant_required": ["q=11_lgwin=16_lgblock=19"],
        },
    },
    {
        "name": "per_tensor_adaptive_aac",
        "predicted_archive_bytes": 178_181,
        "cost_hours": 1.0,
        "cost_dollars": 0.0,
        "risk": "lossless",
        "evidence_grade": "[CPU-prep]",
        "description": "Adaptive arithmetic coding per-tensor; ties brotli within 37 B",
        "model_spec": {
            "capacity_constraint": "n/a",
            "architecture_class": "adaptive_arithmetic_coding",
            "substrate_constraint": "1d_quantized_symbols",
            "canonical_shape_family": "per_tensor_empirical_pmf",
            "variant_required": ["per_tensor_pmf"],
        },
    },
    {
        "name": "tiny_nn_pmf_predictor",
        "predicted_archive_bytes": 167_000,
        "cost_hours": 3.0,
        "cost_dollars": 0.0,
        "risk": "lossless",
        "evidence_grade": "[predicted]",
        "description": "200-param MLP predicting per-tensor PMF; ~400B model + AAC",
        "model_spec": {
            # Implementation-vs-model-gap audit found rank=8 (~5K params)
            # was tested against the predicted "200-param MLP". Pin the
            # capacity bound the catalog row actually claims.
            "capacity_constraint": "<=200_params",
            "architecture_class": "MLP",
            "substrate_constraint": "1d_quantized_symbols",
            "canonical_shape_family": "tensor_id_layer_class_features",
            "variant_required": ["mlp_under_200_params"],
        },
    },
    {
        "name": "compressai_balle_hyperprior",
        "predicted_archive_bytes": 158_000,
        "cost_hours": 4.0,
        "cost_dollars": 5.0,
        "risk": "lossless",
        "evidence_grade": "[subagent-predicted]",
        "description": "Balle scale-hyperprior NN; subagent-verdict joint floor",
        "model_spec": {
            # Implementation-vs-model-gap audit found ScaleHyperprior tested
            # on PR101 INT8-reshaped-as-pseudo-2D substrate; the model is
            # SUBSTRATE-MISMATCHED (ScaleHyperprior expects 2d natural image).
            "capacity_constraint": "5KB_to_10KB_compressed_hyperprior",
            "architecture_class": "ScaleHyperprior",
            "substrate_constraint": "2d_natural_image",
            "canonical_shape_family": "hyperprior_side_info_GDN_nonlinearity",
            "variant_required": [
                "scale_hyperprior",
                "mean_scale_hyperprior",
            ],
        },
    },
    {
        "name": "kalle_fold_mixture_canonical_shapes",
        "predicted_archive_bytes": 173_500,
        "cost_hours": 2.0,
        "cost_dollars": 0.0,
        "risk": "lossless",
        "evidence_grade": "[predicted]",
        "description": "4-component mixture (Gaussian+Laplace+sparse+uniform) on PMFs",
        "model_spec": {
            # Implementation-vs-model-gap audit found generic
            # Gaussian/Laplace/Cauchy mixture tested instead of NN-weight-
            # distribution canonical shapes (Kaiming/Laplace+outliers/
            # spike-and-slab). Pin the canonical-shape-family contract.
            "capacity_constraint": "<=8_components",
            "architecture_class": "mixture_of_canonical_NN_PMF_shapes",
            "substrate_constraint": "1d_quantized_symbols",
            "canonical_shape_family": "kaiming+laplace_with_outliers+spike_and_slab+truncated_normal",
            "variant_required": [
                "nn_weight_distribution_basis",
            ],
        },
    },
    {
        "name": "shared_canonical_pmf_clusters",
        "predicted_archive_bytes": 179_046,
        "cost_hours": 2.0,
        "cost_dollars": 0.0,
        "risk": "lossless",
        "evidence_grade": "[CPU-prep empirical; planning-only]",
        "description": "Shared FP16 PMF clusters over PR101 quantized tensors; measured negative",
        "model_spec": {
            "capacity_constraint": "shared_clusters_under_2KB",
            "architecture_class": "shared_pmf_clusters",
            "substrate_constraint": "1d_quantized_symbols",
            "canonical_shape_family": "fp16_cluster_centers",
            "variant_required": ["fp16_shared_clusters"],
        },
    },
]


# Architecture-side technique catalog (5-10x more headroom)
ARCH_TECHNIQUES = [
    {
        "name": "sparsity_alpha_0.7_imp_retrain",
        "predicted_archive_bytes": 65_000,
        "cost_hours": 24.0,
        "cost_dollars": 25.0,
        "risk": "training_side",
        "evidence_grade": "[predicted]",
        "description": "70% sparsity via IMP retraining; arch unchanged",
        "model_spec": {
            "capacity_constraint": "70_percent_zero_weights",
            "architecture_class": "iterative_magnitude_pruning",
            "substrate_constraint": "renderer_weights",
            "canonical_shape_family": "sparse_mask_bool_indicator",
            "variant_required": ["imp_retrain_with_finetune"],
        },
    },
    {
        "name": "arch_shrink_x0.4_quantizr_class",
        "predicted_archive_bytes": 80_000,
        "cost_hours": 12.0,
        "cost_dollars": 15.0,
        "risk": "architectural",
        "evidence_grade": "[predicted]",
        "description": "88K-element renderer (Quantizr-class); full retrain",
        "model_spec": {
            "capacity_constraint": "~88K_params_renderer",
            "architecture_class": "FiLM_depthwise_separable_CNN",
            "substrate_constraint": "renderer_weights",
            "canonical_shape_family": "quantizr_88k_class",
            "variant_required": ["full_retrain"],
        },
    },
    {
        "name": "self_compress_neural_codec",
        "predicted_archive_bytes": 90_000,
        "cost_hours": 18.0,
        "cost_dollars": 20.0,
        "risk": "architectural",
        "evidence_grade": "[predicted]",
        "description": "Selfcomp/Quantizr-style: renderer is its own decoder",
        "model_spec": {
            "capacity_constraint": "renderer_self_decodes_no_separate_decoder",
            "architecture_class": "self_compressing_neural_codec",
            "substrate_constraint": "renderer_weights_bind_to_decoder",
            "canonical_shape_family": "selfcomp_block_fp_self_decode",
            "variant_required": ["self_decode_path_built"],
        },
    },
    {
        "name": "lossy_coarsening_analytical",
        "predicted_archive_bytes": 156_344,
        "cost_hours": 3.0,
        "cost_dollars": 5.0,
        "risk": "lossy_high",
        "evidence_grade": "[MPS-research-signal; requires exact CUDA review]",
        "description": (
            "Per-tensor K-step lossy coarsening of PR101 quantized renderer "
            "symbols; byte proxy only until exact CUDA review clears a measured "
            "config"
        ),
        "model_spec": {
            "capacity_constraint": "28_uint8_per_tensor_K_values_plus_brotli_payload",
            "architecture_class": "analytical_per_tensor_K_coarsening",
            "substrate_constraint": "PR101_1d_quantized_renderer_symbols",
            "canonical_shape_family": "round_to_nearest_per_tensor_step_size_K",
            "variant_required": ["per_tensor_K_budget_search"],
        },
    },
    {
        "name": "lossy_int4_quantization",
        "predicted_archive_bytes": 105_440,
        "cost_hours": 6.0,
        "cost_dollars": 8.0,
        "risk": "lossy_high",
        "evidence_grade": "[predicted]",
        "description": "n_quant=15 (int4) with QAT/LSQ retuning",
        "model_spec": {
            # Implementation-vs-model-gap audit found NAIVE PTQ tested
            # against a row that names QAT/LSQ. The lane class subsumes
            # multiple variants — only some have been faithfully tested.
            "capacity_constraint": "n_quant=15_int4",
            "architecture_class": "low_bit_quantization",
            "substrate_constraint": "renderer_weights",
            "canonical_shape_family": "int4_per_block_or_per_channel_scales",
            "variant_required": [
                "naive_ptq",
                "qat",
                "lsq",
                "per_channel_scales",
                "mixed_precision_int4_int6_int8",
                "gptq",
                "awq",
            ],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Track-registry-derived catalog rows (T19 follow-on, 2026-05-09).
#
# Per ``feedback_unified_solver_integration_landed_20260509.md`` deferred
# wire-in #2: cathedral autopilot must SEE T20 (KL pose distill loss),
# T22 (temporal consistency regularizer), and Lane-12-v2 (NeRV-as-renderer)
# even though they are NOT yet dispatch-eligible (entry conditions list
# trainer wire-in / Phase B preconditions).
#
# Design split:
#   - LOSS_MODIFIER_TECHNIQUES: training-time loss/regularizer atoms that
#     do NOT directly emit an archive but modify the trainer's objective.
#     Visible to the autopilot for dispatch-queue planning even when not
#     yet promotable.
#   - REPRESENTATION_LANES: architecture-class atoms (Lane-12-v2 NeRV,
#     etc.) that DO emit a distinct archive grammar. Filtered out of
#     ``recommended_top_3`` until ``promotion_eligible == True``.
#
# Both lists carry the same schema as ENCODER_TECHNIQUES / ARCH_TECHNIQUES
# so the existing ``_rank_techniques`` machinery can consume them. The
# `predicted_archive_bytes` for loss-modifiers defaults to the current
# baseline (no archive change) so they never rank ahead of a real
# byte-saving technique unless an empirical anchor lands.
#
# These catalogs are SEEDED from ``tac.track_registry.TRACK_REGISTRY`` to
# preserve single-source-of-truth (CLAUDE.md "Meta-Lagrangian/Pareto solver"
# non-negotiable) — the registry is the typed-row contract; the autopilot
# catalogs derive their entries from it and add planner-specific fields
# (cost_dollars, predicted_archive_bytes, evidence_grade for ranking).
# ─────────────────────────────────────────────────────────────────────────────


def _now_utc_iso() -> str:
    """Return the current UTC time in ISO-8601 (seconds resolution).

    Used to stamp ``last_updated_utc`` on catalog rows so operators can
    detect stale rows (added 2026-05-09,
    lane_check_125_backfill_and_production_hardening_polish).
    """
    import datetime as _dt
    return _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")


def _seed_loss_modifier_catalog_from_registry() -> list[dict[str, Any]]:
    """Derive loss-modifier catalog rows from ``tac.track_registry``.

    Loss modifiers (T20, T22) are training-time atoms — they do not change
    archive bytes directly. The autopilot exposes them in
    ``loss_modifier_technique_ranking`` so the operator can see they are
    visible to the planner even before trainer wire-in clears their
    entry conditions.

    Returns:
        list of catalog rows (one per loss-modifier track in registry).
        Each row carries ``last_updated_utc`` so stale rows can be detected.
    """
    try:
        from tac.track_registry import TRACK_REGISTRY
    except ImportError as exc:
        # Soft-fail if registry import broken — autopilot still works with
        # the legacy ENCODER/ARCH catalogs, but log loudly so the operator
        # sees that loss-modifier / representation-lane visibility is gone.
        print(
            f"[cathedral_autopilot] WARN: tac.track_registry import failed "
            f"({exc}); loss-modifier / representation-lane catalogs will be "
            f"empty. Loss-modifiers (T20/T22) and representation lanes "
            f"(Lane-12-v2) will NOT be visible to the planner this run.",
            file=sys.stderr,
        )
        return []

    rows: list[dict[str, Any]] = []
    for track_id, entry in TRACK_REGISTRY.items():
        # Only include tracks that EXPLICITLY name cathedral_autopilot in
        # their planner_visibility AND are loss-modifier-shaped (loss_term
        # OR regularizer phase).
        if "cathedral_autopilot" not in entry.planner_visibility:
            continue
        phase_str = entry.phase.value if hasattr(entry.phase, "value") else str(entry.phase)
        if phase_str not in ("loss_term", "regularizer"):
            continue
        rows.append({
            "name": track_id,
            "track_id": track_id,
            "track_kind": "loss_modifier",
            "track_phase": phase_str,
            "track_pareto_axis": (
                entry.pareto_axis.value
                if hasattr(entry.pareto_axis, "value") else str(entry.pareto_axis)
            ),
            # Loss modifiers don't change archive bytes by themselves —
            # default to baseline so they never rank ahead of byte-savers
            # unless empirical evidence lands.
            "predicted_archive_bytes": 178_144,  # PR101 brotli baseline
            "cost_hours": 0.0,  # cost is in trainer wire-in, not autopilot dispatch
            "cost_dollars": 0.0,
            "risk": "training_side",
            "evidence_grade": entry.evidence_grade,
            "description": entry.kind_summary,
            "module_path": entry.module_path,
            "entry_conditions": list(entry.entry_conditions),
            "promotion_eligible": entry.promotion_eligible,
            "landed_commit_or_memo": entry.landed_commit_or_memo,
            "last_updated_utc": _now_utc_iso(),
            "model_spec": {
                "capacity_constraint": "n/a_loss_term",
                "architecture_class": "loss_modifier",
                "substrate_constraint": "training_objective",
                "canonical_shape_family": phase_str,
                "variant_required": [track_id],
            },
        })
    return rows


def _seed_representation_lane_catalog_from_registry() -> list[dict[str, Any]]:
    """Derive representation-lane catalog rows from ``tac.track_registry``.

    Representation lanes (Lane-12-v2 NeRV-as-renderer) DO emit distinct
    archive grammars. The autopilot exposes them in
    ``representation_lane_ranking`` but filters out of
    ``recommended_top_3`` until ``promotion_eligible == True``.

    Returns:
        list of catalog rows (one per architecture-phase track in registry
        that names cathedral_autopilot in planner_visibility).
    """
    try:
        from tac.track_registry import TRACK_REGISTRY
    except ImportError:
        return []

    rows: list[dict[str, Any]] = []
    for track_id, entry in TRACK_REGISTRY.items():
        if "cathedral_autopilot" not in entry.planner_visibility:
            continue
        phase_str = entry.phase.value if hasattr(entry.phase, "value") else str(entry.phase)
        if phase_str != "architecture":
            continue
        rows.append({
            "name": track_id,
            "track_id": track_id,
            "track_kind": "representation_lane",
            "track_phase": phase_str,
            "track_pareto_axis": (
                entry.pareto_axis.value
                if hasattr(entry.pareto_axis, "value") else str(entry.pareto_axis)
            ),
            # Representation lanes have NOT been measured empirically yet —
            # use 178,144 as a neutral baseline so the row participates in
            # ranking without claiming a byte saving until empirical anchor.
            "predicted_archive_bytes": 178_144,
            "cost_hours": 24.0,  # NeRV training is multi-hour
            "cost_dollars": 30.0,  # rough Lightning T4 estimate
            "risk": "architectural",
            "evidence_grade": entry.evidence_grade,
            "description": entry.kind_summary,
            "module_path": entry.module_path,
            "entry_conditions": list(entry.entry_conditions),
            "promotion_eligible": entry.promotion_eligible,
            "landed_commit_or_memo": entry.landed_commit_or_memo,
            "last_updated_utc": _now_utc_iso(),
            "model_spec": {
                "capacity_constraint": "research_only_until_phase_b",
                "architecture_class": "nerv_class_renderer",
                "substrate_constraint": "renderer_replacement",
                "canonical_shape_family": "implicit_neural_representation",
                "variant_required": [track_id],
            },
        })
    return rows


# Pre-computed at import time so build_plan can consume them efficiently.
LOSS_MODIFIER_TECHNIQUES = _seed_loss_modifier_catalog_from_registry()
REPRESENTATION_LANES = _seed_representation_lane_catalog_from_registry()


@dataclass
class AutopilotPlan:
    """The autopilot's recommended action plan for a given operator state."""
    schema: str
    tool: str
    evidence_grade: str
    operator_state: dict[str, Any]
    score_geometry: dict[str, Any]
    encoder_technique_ranking: list[dict[str, Any]] = field(default_factory=list)
    arch_technique_ranking: list[dict[str, Any]] = field(default_factory=list)
    # T19 follow-on (2026-05-09): track-registry-derived catalog rankings.
    # Loss modifiers (T20/T22) and representation lanes (Lane-12-v2) are
    # SURFACED to the planner even before they are dispatch-eligible so the
    # operator can see what's queued for trainer wire-in / phase B preconditions.
    loss_modifier_technique_ranking: list[dict[str, Any]] = field(default_factory=list)
    representation_lane_ranking: list[dict[str, Any]] = field(default_factory=list)
    recommended_top_3: list[dict[str, Any]] = field(default_factory=list)
    target_score_gap_analysis: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    axis_priorities: list[dict[str, Any]] = field(default_factory=list)
    evidence_semantics_report: dict[str, Any] = field(default_factory=dict)
    validation_queue: list[dict[str, Any]] = field(default_factory=list)


def _technique_score_after(
    *, baseline_bytes: int, technique_bytes: int,
    d_seg: float, d_pose: float,
) -> float:
    """Score if technique replaces baseline encoder, holding distortion fixed.

    Returns the CUDA-axis score (legacy default — see :func:
    `_technique_score_after_dual_axis` for CPU-axis primary form).
    """
    return contest_score(d_seg, d_pose, technique_bytes)


def _technique_score_after_dual_axis(
    *, baseline_bytes: int, technique_bytes: int,
    d_seg_cuda: float, d_pose_cuda: float,
    architecture_class: str = "hnerv",
) -> dict[str, Any]:
    """CPU + CUDA axis predicted score after a (lossless) encoder swap.

    The contest leaderboard ranks by ``--device cpu`` (per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA"). This helper returns
    BOTH the CUDA-axis score (legacy) and a CPU-axis prediction band so
    the recommender can primary-rank by CPU.

    Args:
        baseline_bytes: current archive bytes (unused; kept for symmetry).
        technique_bytes: predicted archive bytes after technique applies.
        d_seg_cuda: CUDA-axis seg distortion at the operating point.
        d_pose_cuda: CUDA-axis pose distortion at the operating point.
        architecture_class: calibration class (default ``"hnerv"``).

    Returns:
        Dict with keys ``predicted_cuda_score``, ``predicted_cpu_score``,
        ``predicted_cpu_score_lo``, ``predicted_cpu_score_hi``,
        ``predicted_cpu_score_calibration`` (one of ``"hnerv-anchored"``
        / ``"extrapolated"``).
    """
    cuda_score = contest_score(d_seg_cuda, d_pose_cuda, technique_bytes)
    cal = CudaCpuCalibration(architecture_class=architecture_class)
    band = cal.predict_cpu_from_cuda(
        cuda_score,
        d_pose_cuda=d_pose_cuda,
        d_seg_cuda=d_seg_cuda,
        archive_bytes=technique_bytes,
    )
    return {
        "predicted_cuda_score": cuda_score,
        "predicted_cpu_score": band.score_point,
        "predicted_cpu_score_lo": band.score_lo,
        "predicted_cpu_score_hi": band.score_hi,
        "predicted_cpu_score_calibration": band.calibration_quality,
    }


# Feedback-loop / continual-learning helpers


@dataclass
class TechniqueEvidence:
    """One empirical observation of a technique's actual performance.

    Fields ``empirical_archive_bytes`` and (optional) ``empirical_score`` are
    posterior measurements that override the catalog's prior
    ``predicted_archive_bytes`` once enough observations exist.
    """
    technique: str
    empirical_archive_bytes: int | None = None
    empirical_score: float | None = None
    empirical_d_seg: float | None = None
    empirical_d_pose: float | None = None
    evidence_grade: str = ""
    evidence_marker: str = ""
    evidence_semantics: str = ""
    device_axis: str = ""
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    runtime_manifest: str = ""
    hardware: str = ""
    substrate_class: str = ""
    architecture_family: str = ""
    decoder_path: str = ""
    loader_drift_probe_path: str = ""
    network_drift_probe_path: str = ""
    decoder_pose_ratio_cuda_over_cpu: float | None = None
    network_pose_ratio_cuda_over_cpu: float | None = None
    score_claim: bool | None = None
    score_contest_cpu: float | None = None
    score_contest_cuda: float | None = None
    sample_count: int | None = None
    seg_distortion: float | None = None
    pose_distortion: float | None = None
    rate_term: float | None = None
    recomputed_score: float | None = None
    exact_eval_command: str = ""
    log_path: str = ""
    dispatch_claim_status: str = ""
    promotion_eligible: bool | None = None
    rank_or_kill_eligible: bool | None = None
    ready_for_exact_eval_dispatch: bool | None = None
    exact_cuda_auth_eval: bool | None = None
    contest_dispatch_verdict: str = ""
    measured_config_status: str = ""
    family_falsified: bool | None = None
    method_family_retired: bool | None = None
    exact_result_review_path: str = ""
    falsification_scope: str = ""
    reactivation_criteria: list[str] = field(default_factory=list)
    dispatch_blockers: list[str] = field(default_factory=list)
    score_affecting_payload_changed: bool | None = None
    charged_bits_changed: bool | None = None
    cuda_eval_worth_testing: bool | None = None
    byte_proxy_only: bool | None = None
    proxy_row: bool | None = None
    source: str = ""
    timestamp: str = ""


def _strict_json_bool(
    row: dict[str, Any],
    key: str,
    schema_blockers: list[str],
) -> bool | None:
    """Parse JSON booleans without accepting truthy strings or integers."""

    if key not in row or row.get(key) is None:
        return None
    value = row[key]
    if isinstance(value, bool):
        return value
    schema_blockers.append(f"invalid_evidence_schema_boolean:{key}")
    return None


def _evidence_dispatch_blockers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(blocker) for blocker in value if str(blocker)]
    return [f"invalid_evidence_schema_dispatch_blockers:{type(value).__name__}"]


def _first_finite_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if row.get(key) is None:
            continue
        return float(row[key])
    return None


def _first_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        return int(value)
    return None


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


_LAST_EVIDENCE_LOAD_DIAGNOSTICS: list[dict[str, Any]] = []


def _load_evidence(path: Path) -> list[TechniqueEvidence]:
    """Read JSONL or JSON-array of evidence rows. Skips malformed rows."""
    global _LAST_EVIDENCE_LOAD_DIAGNOSTICS
    _LAST_EVIDENCE_LOAD_DIAGNOSTICS = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: list[tuple[int, Any]] = []
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            _LAST_EVIDENCE_LOAD_DIAGNOSTICS.append({
                "line_number": 1,
                "reason": f"json_parse_error:{exc}",
            })
            return []
        if not isinstance(payload, list):
            _LAST_EVIDENCE_LOAD_DIAGNOSTICS.append({
                "line_number": 1,
                "reason": "json_payload_not_array",
            })
            return []
        rows = [(idx + 1, item) for idx, item in enumerate(payload)]
    else:
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                _LAST_EVIDENCE_LOAD_DIAGNOSTICS.append({
                    "line_number": line_number,
                    "reason": f"json_parse_error:{exc}",
                })
    out: list[TechniqueEvidence] = []
    for line_number, r in rows:
        if not isinstance(r, dict) or "technique" not in r:
            _LAST_EVIDENCE_LOAD_DIAGNOSTICS.append({
                "line_number": line_number,
                "reason": "row_not_object_or_missing_technique",
            })
            continue
        try:
            schema_blockers: list[str] = []
            dispatch_blockers = _evidence_dispatch_blockers(r.get("dispatch_blockers"))
            out.append(TechniqueEvidence(
                technique=str(r["technique"]),
                empirical_archive_bytes=_first_int(
                    r, "empirical_archive_bytes", "archive_bytes"
                ),
                empirical_score=(
                    float(r["empirical_score"])
                    if r.get("empirical_score") is not None else None
                ),
                empirical_d_seg=(
                    float(r["empirical_d_seg"])
                    if r.get("empirical_d_seg") is not None else None
                ),
                empirical_d_pose=(
                    float(r["empirical_d_pose"])
                    if r.get("empirical_d_pose") is not None else None
                ),
            evidence_grade=str(r.get("evidence_grade", "")),
            evidence_marker=str(r.get("evidence_marker", "")),
            evidence_semantics=str(r.get("evidence_semantics", "")),
            device_axis=str(r.get("device_axis", "") or r.get("device", "")),
            archive_sha256=str(
                r.get("archive_sha256")
                or r.get("archive_sha")
                or ""
            ),
            runtime_tree_sha256=str(
                r.get("runtime_tree_sha256")
                or r.get("inflate_runtime_tree_sha256")
                or r.get("runtime_tree_sha")
                or ""
            ),
            runtime_manifest=str(
                r.get("runtime_manifest")
                or r.get("inflate_runtime_manifest")
                or ""
            ),
            hardware=str(r.get("hardware", "") or r.get("runner", "")),
            substrate_class=str(
                r.get("substrate_class")
                or r.get("architecture_class")
                or r.get("family")
                or ""
            ),
            architecture_family=str(
                r.get("architecture_family")
                or r.get("model_family")
                or ""
            ),
            decoder_path=str(r.get("decoder_path", "") or r.get("inflate_decoder", "")),
            loader_drift_probe_path=str(r.get("loader_drift_probe_path", "")),
            network_drift_probe_path=str(r.get("network_drift_probe_path", "")),
            decoder_pose_ratio_cuda_over_cpu=(
                float(r["decoder_pose_ratio_cuda_over_cpu"])
                if r.get("decoder_pose_ratio_cuda_over_cpu") is not None else None
            ),
            network_pose_ratio_cuda_over_cpu=(
                float(r["network_pose_ratio_cuda_over_cpu"])
                if r.get("network_pose_ratio_cuda_over_cpu") is not None else None
            ),
            score_claim=_strict_json_bool(r, "score_claim", schema_blockers),
            score_contest_cpu=(
                _first_finite_float(
                    r,
                    "score_contest_cpu",
                    "contest_cpu_score",
                    "canonical_score_contest_cpu",
                )
            ),
            score_contest_cuda=(
                _first_finite_float(
                    r,
                    "score_contest_cuda",
                    "contest_cuda_score",
                    "canonical_score_contest_cuda",
                )
            ),
            sample_count=_first_int(r, "sample_count", "n_samples", "num_samples"),
            seg_distortion=_first_finite_float(
                r, "seg_distortion", "segnet_distortion", "avg_segnet_dist", "d_seg"
            ),
            pose_distortion=_first_finite_float(
                r, "pose_distortion", "posenet_distortion", "avg_posenet_dist", "d_pose"
            ),
            rate_term=_first_finite_float(
                r, "rate_term", "rate", "compression_rate", "archive_rate_ratio"
            ),
            recomputed_score=_first_finite_float(
                r,
                "recomputed_score",
                "score_recomputed_from_components",
                "canonical_score_recomputed",
                "contest_cuda_score_recomputed",
            ),
            exact_eval_command=_first_text(r, "exact_eval_command", "eval_command"),
            log_path=_first_text(r, "log_path", "auth_eval_log", "log_file"),
            dispatch_claim_status=_first_text(
                r,
                "dispatch_claim_status",
                "dispatch_claim_latest_status",
                "dispatch_status",
            ),
            promotion_eligible=_strict_json_bool(
                r, "promotion_eligible", schema_blockers
            ),
            rank_or_kill_eligible=_strict_json_bool(
                r, "rank_or_kill_eligible", schema_blockers
            ),
            ready_for_exact_eval_dispatch=_strict_json_bool(
                r, "ready_for_exact_eval_dispatch", schema_blockers
            ),
            exact_cuda_auth_eval=_strict_json_bool(
                r, "exact_cuda_auth_eval", schema_blockers
            ),
            contest_dispatch_verdict=str(r.get("contest_dispatch_verdict", "")),
            measured_config_status=str(r.get("measured_config_status", "")),
            family_falsified=_strict_json_bool(
                r, "family_falsified", schema_blockers
            ),
            method_family_retired=_strict_json_bool(
                r, "method_family_retired", schema_blockers
            ),
            exact_result_review_path=str(
                r.get("exact_result_review_path")
                or r.get("exact_result_review_packet")
                or ""
            ),
            falsification_scope=str(r.get("falsification_scope", "")),
            reactivation_criteria=[
                str(item) for item in r.get("reactivation_criteria", [])
            ] if isinstance(r.get("reactivation_criteria", []), list) else [],
            score_affecting_payload_changed=_strict_json_bool(
                r, "score_affecting_payload_changed", schema_blockers
            ),
            charged_bits_changed=_strict_json_bool(
                r, "charged_bits_changed", schema_blockers
            ),
            cuda_eval_worth_testing=_strict_json_bool(
                r, "cuda_eval_worth_testing", schema_blockers
            ),
            byte_proxy_only=_strict_json_bool(r, "byte_proxy_only", schema_blockers),
            proxy_row=_strict_json_bool(r, "proxy_row", schema_blockers),
            dispatch_blockers=list(
                dict.fromkeys(
                    [
                        *dispatch_blockers,
                        *(
                            [
                                "invalid_evidence_schema_non_promotable",
                                *schema_blockers,
                            ]
                            if schema_blockers
                            else []
                        ),
                    ]
                )
            ),
            source=str(r.get("source", "")),
            timestamp=str(r.get("timestamp", "")),
            ))
        except (TypeError, ValueError) as exc:
            _LAST_EVIDENCE_LOAD_DIAGNOSTICS.append({
                "line_number": line_number,
                "technique": str(r.get("technique", "")),
                "reason": f"row_cast_error:{type(exc).__name__}:{exc}",
            })
    return out


def _is_explicitly_promotable_evidence(evidence: TechniqueEvidence) -> bool:
    """Return true only for rows that explicitly opt into promotion semantics.

    Absence of custody fields must fail closed. CPU/MPS/proxy rows may still
    update planning byte estimates, but they cannot be labeled promotable unless
    the producer explicitly records score/promotion/dispatch readiness and has
    no dispatch blockers.
    """
    return not promotable_exact_cuda_evidence_blockers(asdict(evidence))


def _promotability_blockers(evidence: TechniqueEvidence) -> list[str]:
    """Machine-readable blockers for exact-CUDA promotion semantics."""

    return promotable_exact_cuda_evidence_blockers(asdict(evidence))


def _is_explicitly_contest_cpu_evidence(evidence: TechniqueEvidence) -> bool:
    """Return true for official Linux/x86 contest-CPU score rows.

    These rows are valid CPU-axis score evidence, but they are intentionally
    separate from ``_is_explicitly_promotable_evidence`` because they do not
    replace exact CUDA custody for CUDA promotion, dispatch, or broad
    retirement decisions.
    """
    axis = _device_axis_for_evidence(evidence)
    if axis != "contest_cpu":
        return False
    axis_blockers = _device_axis_evidence_blockers(evidence, axis)
    text = " ".join(
        part.strip().lower()
        for part in (
            evidence.device_axis,
            evidence.evidence_grade,
            evidence.evidence_marker,
            evidence.evidence_semantics,
            evidence.source,
            evidence.hardware,
        )
        if part
    )
    proxy_marker = any(
        marker in text
        for marker in (
            "macos",
            "mps",
            "cpu-prep",
            "cpu_prep",
            "proxy",
            "prediction",
            "predicted",
            "research-signal",
            "research_signal",
        )
    )
    return (
        evidence.score_claim is True
        and evidence.rank_or_kill_eligible is True
        and not evidence.dispatch_blockers
        and not axis_blockers
        and not proxy_marker
    )


def _is_exact_negative_or_retired_evidence(evidence: TechniqueEvidence) -> bool:
    """Return true for exact CUDA rows that supersede proxy byte anchors.

    Exact negative evidence retires the measured config only unless a separate
    review explicitly proves a broader method/family conclusion. The planner's
    responsibility here is narrower: do not keep ranking an older CPU/MPS byte
    anchor as promising after exact CUDA showed the scored archive regressed.
    """
    text = " ".join(
        part.strip().lower()
        for part in (
            evidence.evidence_grade,
            evidence.evidence_marker,
            evidence.evidence_semantics,
            evidence.source,
            evidence.contest_dispatch_verdict,
            evidence.measured_config_status,
            evidence.exact_result_review_path,
        )
        if part
    )
    exact_cuda = has_positive_exact_cuda_evidence_marker(
        asdict(evidence),
        include_negative_grade=True,
    )
    if not exact_cuda:
        return False
    return any(
        marker in text
        for marker in (
            "a-negative",
            "exact-negative",
            "exact_negative",
            "measured_config_retired",
            "measured-config-retired",
            "score_negative",
            "score-negative",
            "cuda_negative",
            "cuda-negative",
        )
    )


def _is_active_ranking_blocked(row: dict[str, Any]) -> bool:
    """Return true when evidence may inform audit state but not ranking."""
    return bool(
        row.get("retired_from_active_ranking")
        or row.get("active_ranking_blocked")
        or (
            row.get("empirical_anchor_promotable") is False
            and row.get("rank_or_kill_eligible") is False
        )
    )


def _catalog_names(catalogs: list[list[dict[str, Any]]]) -> set[str]:
    return {
        str(row.get("name", ""))
        for catalog in catalogs
        for row in catalog
        if row.get("name")
    }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _device_axis_for_evidence(evidence: TechniqueEvidence) -> str:
    text = " ".join(
        part.strip().lower()
        for part in (
            evidence.device_axis,
            evidence.evidence_grade,
            evidence.evidence_marker,
            evidence.evidence_semantics,
            evidence.source,
            evidence.hardware,
        )
        if part
    )
    if "macos-cpu" in text or "macos_cpu" in text or "apple silicon" in text:
        return "macos_cpu_advisory"
    if (
        "contest-cpu" in text
        or "contest_cpu" in text
        or ("linux x86_64" in text and "cpu" in text)
        or ("ubuntu-24.04" in text and "cpu" in text)
    ):
        return "contest_cpu"
    if "contest-cuda" in text or "contest_cuda" in text or "exact_cuda" in text:
        return "contest_cuda"
    if evidence.device_axis.strip().lower() in {"cpu", "cuda", "mps"}:
        return evidence.device_axis.strip().lower()
    return ""


def _device_axis_evidence_blockers(
    evidence: TechniqueEvidence,
    axis: str,
) -> list[str]:
    blockers: list[str] = []
    if not evidence.archive_sha256:
        blockers.append("archive_sha256_required")
    if not evidence.runtime_tree_sha256:
        blockers.append("runtime_tree_sha256_required")
    if evidence.sample_count is None or evidence.sample_count < 600:
        blockers.append("sample_count_ge_600_required")
    if not evidence.hardware:
        blockers.append("hardware_required")
    if not evidence.exact_eval_command:
        blockers.append("exact_eval_command_required")
    if not evidence.log_path:
        blockers.append("log_path_required")
    if evidence.empirical_archive_bytes is None or evidence.empirical_archive_bytes <= 0:
        blockers.append("archive_bytes_required")
    if evidence.empirical_d_seg is None:
        blockers.append("seg_component_required")
    if evidence.empirical_d_pose is None:
        blockers.append("pose_component_required")
    grade = evidence.evidence_grade.lower()
    if axis == "contest_cpu":
        if evidence.score_contest_cpu is None and evidence.empirical_score is None:
            blockers.append("contest_cpu_score_required")
        if "contest-cpu" not in grade and "contest_cpu" not in grade:
            blockers.append("contest_cpu_grade_required")
    elif axis == "contest_cuda":
        if evidence.score_contest_cuda is None and evidence.empirical_score is None:
            blockers.append("contest_cuda_score_required")
        if not (
            "contest-cuda" in grade
            or "contest_cuda" in grade
            or "exact-cuda" in grade
            or "exact_cuda" in grade
            or grade.strip() in {"a", "a++"}
        ):
            blockers.append("contest_cuda_grade_required")
    return blockers


def summarize_device_axis_evidence(
    evidence: list[TechniqueEvidence],
) -> dict[str, Any]:
    """Summarize paired CPU/CUDA evidence without changing active ranking.

    The selector must not prefer CPU-targeted or CUDA-targeted architectures
    from one-sided rows. macOS CPU rows are useful high-throughput research
    proxies after the PR107 Linux-vs-M5 check (6e-6 score gap), but a
    device-axis comparison becomes actionable only for a specific
    archive/runtime pair when both ``contest_cpu`` and
    ``contest_cuda`` rows share archive SHA and runtime-tree SHA. Even then the
    comparison is a diagnostic prior, not a replacement for the normal
    evidence-grade promotion rules.
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    class_values: dict[str, list[float]] = {}
    component_values: dict[str, dict[str, list[float]]] = {}
    unpaired: list[dict[str, Any]] = []
    advisory_count = 0
    for row in evidence:
        axis = _device_axis_for_evidence(row)
        if not axis:
            continue
        if axis == "macos_cpu_advisory":
            advisory_count += 1
        key_ready = bool(row.archive_sha256 and row.runtime_tree_sha256)
        key = (row.technique, row.archive_sha256, row.runtime_tree_sha256)
        axis_blockers = (
            _device_axis_evidence_blockers(row, axis)
            if axis in {"contest_cpu", "contest_cuda"}
            else []
        )
        if axis in {"contest_cpu", "contest_cuda"} and axis_blockers:
            unpaired.append(
                {
                    "technique": row.technique,
                    "device_axis": axis,
                    "archive_sha256": row.archive_sha256,
                    "runtime_tree_sha256": row.runtime_tree_sha256,
                    "evidence_grade": row.evidence_grade,
                    "source": row.source,
                    "reason": "incomplete_device_axis_evidence",
                    "blockers": axis_blockers,
                }
            )
            continue
        if axis in {"contest_cpu", "contest_cuda"} and key_ready:
            bucket = groups.setdefault(
                key,
                {
                    "technique": row.technique,
                    "archive_sha256": row.archive_sha256,
                    "runtime_tree_sha256": row.runtime_tree_sha256,
                    "axes": {},
                    "sources": [],
                },
            )
            score = row.score_contest_cpu if axis == "contest_cpu" else row.score_contest_cuda
            if score is None:
                score = row.empirical_score
            axis_payload = {
                "score": score,
                "d_seg": row.empirical_d_seg,
                "d_pose": row.empirical_d_pose,
                "archive_bytes": row.empirical_archive_bytes,
                "hardware": row.hardware,
                "evidence_grade": row.evidence_grade,
                "source": row.source,
            }
            existing = bucket["axes"].get(axis)
            if existing is not None and existing != axis_payload:
                bucket.setdefault("axis_conflicts", []).append({
                    "device_axis": axis,
                    "existing": existing,
                    "conflicting": axis_payload,
                    "reason": "conflicting_duplicate_axis_rows_same_archive_runtime",
                })
            else:
                bucket["axes"][axis] = axis_payload
            if row.source:
                bucket["sources"].append(row.source)
        else:
            unpaired.append(
                {
                    "technique": row.technique,
                    "device_axis": axis,
                    "archive_sha256": row.archive_sha256,
                    "runtime_tree_sha256": row.runtime_tree_sha256,
                    "evidence_grade": row.evidence_grade,
                    "source": row.source,
                    "reason": (
                        "macos_cpu_research_proxy_needs_linux_contest_cpu_promotion"
                        if axis == "macos_cpu_advisory"
                        else "missing_archive_or_runtime_pairing_key"
                    ),
                }
            )

    paired: list[dict[str, Any]] = []
    for bucket in groups.values():
        axes = bucket["axes"]
        if bucket.get("axis_conflicts"):
            unpaired.append(
                {
                    "technique": bucket["technique"],
                    "device_axis": ",".join(sorted(axes)),
                    "archive_sha256": bucket["archive_sha256"],
                    "runtime_tree_sha256": bucket["runtime_tree_sha256"],
                    "evidence_grade": "; ".join(
                        sorted({
                            str(axis_row.get("evidence_grade", ""))
                            for axis_row in axes.values()
                            if axis_row.get("evidence_grade")
                        })
                    ),
                    "source": "; ".join(bucket["sources"]),
                    "reason": "conflicting_duplicate_axis_rows_same_archive_runtime",
                    "axis_conflicts": bucket["axis_conflicts"],
                }
            )
            continue
        if "contest_cpu" not in axes or "contest_cuda" not in axes:
            unpaired.append(
                {
                    "technique": bucket["technique"],
                    "device_axis": ",".join(sorted(axes)),
                    "archive_sha256": bucket["archive_sha256"],
                    "runtime_tree_sha256": bucket["runtime_tree_sha256"],
                    "evidence_grade": "; ".join(
                        sorted({
                            str(axis_row.get("evidence_grade", ""))
                            for axis_row in axes.values()
                            if axis_row.get("evidence_grade")
                        })
                    ),
                    "source": "; ".join(bucket["sources"]),
                    "reason": "needs_both_contest_cpu_and_contest_cuda_same_archive_runtime",
                }
            )
            continue
        cpu = axes["contest_cpu"]
        cuda = axes["contest_cuda"]
        cpu_score = cpu.get("score")
        cuda_score = cuda.get("score")
        comparison: dict[str, Any] = {
            "technique": bucket["technique"],
            "archive_sha256": bucket["archive_sha256"],
            "runtime_tree_sha256": bucket["runtime_tree_sha256"],
            "contest_cpu": cpu,
            "contest_cuda": cuda,
            "source": "; ".join(bucket["sources"]),
        }
        if cpu_score is not None and cuda_score is not None:
            comparison["score_gap_cuda_minus_cpu"] = float(cuda_score) - float(cpu_score)
        cpu_pose = cpu.get("d_pose")
        cuda_pose = cuda.get("d_pose")
        cpu_seg = cpu.get("d_seg")
        cuda_seg = cuda.get("d_seg")
        if cpu_pose and cuda_pose:
            pose_ratio = float(cuda_pose) / float(cpu_pose)
            comparison["pose_ratio_cuda_over_cpu"] = pose_ratio
            technique_rows = [
                row for row in evidence
                if row.technique == bucket["technique"]
                and row.archive_sha256 == bucket["archive_sha256"]
                and row.runtime_tree_sha256 == bucket["runtime_tree_sha256"]
            ]
            substrate_classes = sorted({
                row.substrate_class for row in technique_rows if row.substrate_class
            })
            comparison["substrate_classes"] = substrate_classes
            for substrate_class in substrate_classes or ["unknown_substrate"]:
                class_values.setdefault(substrate_class, []).append(pose_ratio)
            for row in technique_rows:
                substrate_class = row.substrate_class or "unknown_substrate"
                bucket_components = component_values.setdefault(
                    substrate_class,
                    {"decoder": [], "network": []},
                )
                if row.decoder_pose_ratio_cuda_over_cpu is not None:
                    bucket_components["decoder"].append(row.decoder_pose_ratio_cuda_over_cpu)
                if row.network_pose_ratio_cuda_over_cpu is not None:
                    bucket_components["network"].append(row.network_pose_ratio_cuda_over_cpu)
        if cpu_seg and cuda_seg:
            comparison["seg_ratio_cuda_over_cpu"] = float(cuda_seg) / float(cpu_seg)
        paired.append(comparison)

    substrate_profiles = []
    for substrate_class, values in sorted(class_values.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        components = component_values.get(substrate_class, {"decoder": [], "network": []})
        decoder_values = components.get("decoder", [])
        network_values = components.get("network", [])
        substrate_profiles.append(
            {
                "substrate_class": substrate_class,
                "n": len(values),
                "pose_ratio_mean_cuda_over_cpu": mean,
                "pose_ratio_variance": variance,
                "pose_ratio_values": values,
                "decoder_pose_ratio_mean_cuda_over_cpu": (
                    sum(decoder_values) / len(decoder_values)
                    if decoder_values else None
                ),
                "network_pose_ratio_mean_cuda_over_cpu": (
                    sum(network_values) / len(network_values)
                    if network_values else None
                ),
                "posterior_status": (
                    "needs_more_anchors" if len(values) < 3 else "empirical_profile"
                ),
                "outlier_policy": (
                    "New same-class anchors outside 3 sigma should fork a new "
                    "substrate class instead of mutating the class mean silently."
                ),
            }
        )

    return {
        "policy": (
            "Do not assign CPU-vs-GPU architecture or stack priority until "
            "both contest-CPU and contest-CUDA are measured on the same "
            "archive SHA and runtime-tree SHA. macOS CPU may accelerate "
            "parallel dev sweeps after the PR107 GHA-vs-M5 check, but it "
            "remains a research proxy until promoted by Linux x86_64 "
            "contest-CPU."
        ),
        "learning_policy": (
            "Treat CPU/CUDA drift as a Bayesian per-substrate-class profile. "
            "Update class posteriors only from paired same-archive/runtime "
            "anchors; loader and network drift probes are independent optional "
            "components of R_total."
        ),
        "priority_status": (
            "paired_archive_specific_diagnostics_available"
            if paired else "insufficient_paired_axis_evidence"
        ),
        "paired_comparison_count": len(paired),
        "paired_comparisons": paired,
        "substrate_class_profiles": substrate_profiles,
        "unpaired_device_axis_evidence_count": len(unpaired),
        "unpaired_device_axis_evidence": unpaired,
        "macos_cpu_advisory_count": advisory_count,
    }


def summarize_evidence_semantics(
    evidence: list[TechniqueEvidence] | None,
    catalogs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize evidence rows that the catalog cannot safely rank.

    Unknown technique names fail closed: they are visible in reports, but they
    are never turned into ranked recommendations without a catalog row or an
    explicit alias/parent mapping.
    """
    ev = evidence or []
    known_names = _catalog_names(catalogs)
    unknown_by_name: dict[str, dict[str, Any]] = {}
    cataloged_exact_negative: set[str] = set()
    contest_cpu_score_claim_rows = 0

    for row in ev:
        if _is_explicitly_contest_cpu_evidence(row):
            contest_cpu_score_claim_rows += 1
        if row.technique in known_names:
            if _is_exact_negative_or_retired_evidence(row):
                cataloged_exact_negative.add(row.technique)
            continue

        bucket = unknown_by_name.setdefault(
            row.technique,
            {
                "technique": row.technique,
                "n_rows": 0,
                "exact_negative_rows": 0,
                "proxy_or_planning_rows": 0,
                "sources": [],
                "dispatch_blockers": [],
                "evidence_grades": [],
                "verdicts": [],
                "empirical_archive_bytes": [],
                "min_empirical_archive_bytes": None,
                "latest_timestamp": "",
                "score_affecting_payload_changed_any": False,
                "charged_bits_changed_any": False,
                "cuda_eval_worth_testing_any": False,
                "cuda_eval_worth_testing_false": False,
                "byte_proxy_only_any": False,
                "proxy_row_any": False,
                "status": "unknown_technique_not_ranked",
            },
        )
        bucket["n_rows"] += 1
        if _is_exact_negative_or_retired_evidence(row):
            bucket["exact_negative_rows"] += 1
        if not _is_explicitly_promotable_evidence(row):
            bucket["proxy_or_planning_rows"] += 1
        if row.source and row.source not in bucket["sources"]:
            bucket["sources"].append(row.source)
        if row.evidence_grade and row.evidence_grade not in bucket["evidence_grades"]:
            bucket["evidence_grades"].append(row.evidence_grade)
        if (
            row.contest_dispatch_verdict
            and row.contest_dispatch_verdict not in bucket["verdicts"]
        ):
            bucket["verdicts"].append(row.contest_dispatch_verdict)
        if row.empirical_archive_bytes is not None:
            bucket["empirical_archive_bytes"].append(row.empirical_archive_bytes)
            current_min = bucket["min_empirical_archive_bytes"]
            bucket["min_empirical_archive_bytes"] = (
                row.empirical_archive_bytes
                if current_min is None
                else min(current_min, row.empirical_archive_bytes)
            )
        if row.timestamp and row.timestamp > bucket["latest_timestamp"]:
            bucket["latest_timestamp"] = row.timestamp
        if row.score_affecting_payload_changed is True:
            bucket["score_affecting_payload_changed_any"] = True
        if row.charged_bits_changed is True:
            bucket["charged_bits_changed_any"] = True
        if row.cuda_eval_worth_testing is True:
            bucket["cuda_eval_worth_testing_any"] = True
        if row.cuda_eval_worth_testing is False:
            bucket["cuda_eval_worth_testing_false"] = True
        if row.byte_proxy_only is True:
            bucket["byte_proxy_only_any"] = True
        if row.proxy_row is True:
            bucket["proxy_row_any"] = True
        for blocker in row.dispatch_blockers:
            if blocker not in bucket["dispatch_blockers"]:
                bucket["dispatch_blockers"].append(blocker)

    unknown = sorted(unknown_by_name.values(), key=lambda item: item["technique"])
    return {
        "n_evidence_rows": len(ev),
        "rejected_evidence_row_count": len(_LAST_EVIDENCE_LOAD_DIAGNOSTICS),
        "evidence_load_diagnostics": list(_LAST_EVIDENCE_LOAD_DIAGNOSTICS),
        "device_axis_report": summarize_device_axis_evidence(ev),
        "unknown_evidence_row_count": sum(item["n_rows"] for item in unknown),
        "unknown_evidence_technique_count": len(unknown),
        "unknown_evidence_techniques": unknown,
        "unknown_exact_negative_row_count": sum(
            item["exact_negative_rows"] for item in unknown
        ),
        "contest_cpu_score_claim_row_count": contest_cpu_score_claim_rows,
        "cataloged_exact_negative_techniques": sorted(cataloged_exact_negative),
        "active_ranking_blocked_techniques": [],
    }


def attach_active_ranking_summary(
    report: dict[str, Any],
    catalogs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    out = dict(report)
    blocked = sorted({
        str(row.get("name", ""))
        for catalog in catalogs
        for row in catalog
        if row.get("name") and _is_active_ranking_blocked(row)
    })
    out["active_ranking_blocked_techniques"] = blocked
    return out


def detect_evidence_model_spec_mismatch(
    spec: dict[str, Any] | None,
    evidence: TechniqueEvidence,
) -> list[str]:
    """Self-protection: flag evidence rows that diverge from declared model_spec.

    Returns a list of human-readable mismatch reasons (empty list = match).
    Implementation: we conservatively grep the evidence ``source`` /
    ``evidence_semantics`` strings for known divergent-implementation
    fingerprints called out in the audit memo. The check is INTENTIONALLY
    coarse — it flags suspicions, never blocks. The hard preflight check
    (``check_evidence_implementation_matches_model_spec``) is the one that
    eventually flips to STRICT once the live count is 0.

    Reference: ``feedback_implementation_vs_model_gap_audit_20260508.md``.
    """
    if not spec:
        return []
    text = " ".join(
        part for part in (
            evidence.source,
            evidence.evidence_semantics,
            evidence.evidence_marker,
            evidence.evidence_grade,
        ) if part
    ).lower()
    if not text:
        return []
    reasons: list[str] = []
    ascii_text = text.replace("\u00d7", "x")

    capacity = str(spec.get("capacity_constraint", "")).lower()
    arch_class = str(spec.get("architecture_class", "")).lower()
    substrate = str(spec.get("substrate_constraint", "")).lower()
    shape_family = str(spec.get("canonical_shape_family", "")).lower()
    variants = [str(v).lower() for v in spec.get("variant_required", []) or []]

    # tiny_nn capacity mismatch — audit memo §2.
    if ("200" in capacity and "param" in capacity) and (
        "rank=8" in text or "rank=32" in text or "rank=16" in text
        or "rank=64" in text or "factorized" in text
    ):
        reasons.append(
            "CAPACITY_MISMATCH: model_spec declares ~200-param MLP, "
            "evidence source mentions rank-K factorized softmax "
            "(typically thousands of params)"
        )

    # ScaleHyperprior substrate mismatch — audit memo §3.
    if "scalehyperprior" in arch_class.replace("_", ""):
        substrate_2d = "2d_natural_image" in substrate
        if substrate_2d and (
            "1d" in text
            or "pseudo-image" in text
            or "1x1x" in ascii_text
            or "reshape" in text
        ):
            reasons.append(
                "SUBSTRATE_MISMATCH: model_spec declares 2d_natural_image, "
                "evidence source mentions 1D / pseudo-image reshape"
            )
        if substrate_2d and (
            "weight" in text or "pmf" in text or "symbol" in text
            or "int8" in text
        ):
            reasons.append(
                "SUBSTRATE_MISMATCH: model_spec declares 2d_natural_image, "
                "evidence source mentions weight/pmf/symbol substrate"
            )

    # kalle_fold canonical-shape-family mismatch — audit memo §1.
    if "spike_and_slab" in shape_family or "kaiming" in shape_family:
        normalized = text.replace(" ", "")
        if (
            "gaussian+laplace+delta+uniform" in normalized
            or "cauchy" in text
            or "generic" in text
            or "intuition" in text
            or "my own picked" in text
        ):
            reasons.append(
                "SHAPE_FAMILY_MISMATCH: model_spec declares "
                "kaiming/laplace+outliers/spike-and-slab basis, evidence "
                "source mentions generic Gaussian/Laplace/Cauchy"
            )

    # lossy_int4 variant exhaustion — audit memo §4.
    if variants and len(variants) >= 2:
        seen: list[str] = []
        for variant in variants:
            tokens = [tok for tok in variant.replace("=", "_").split("_") if tok]
            for tok in tokens:
                if tok and tok in text:
                    seen.append(variant)
                    break
        has_negated_variant_inventory = (
            any(phrase in text for phrase in ("not tested", "not exercised", "unexercised"))
            and any(tok in text for tok in ("qat", "gptq", "awq", "lsq", "mixed-precision", "per-channel"))
        )
        if (has_negated_variant_inventory or not seen) and any(
            tok in text
            for tok in ("naive_ptq", "ptq", "qat", "gptq", "awq", "lsq", "mixed-precision")
        ):
            reasons.append(
                "VARIANT_PARTIAL_EXHAUSTION: model_spec lists "
                f"{len(variants)} required variants, evidence row "
                "contains a partial or explicitly negated variant "
                "inventory"
            )
    return reasons


def update_catalog_from_evidence(
    catalog: list[dict[str, Any]],
    evidence: list[TechniqueEvidence],
    *,
    log_warnings: bool = True,
) -> list[dict[str, Any]]:
    """Return a NEW catalog with empirical anchors blended over predictions.

    Per-technique median over observed empirical_archive_bytes replaces the
    predicted value; evidence_grade is upgraded to ``[empirical-anchor-N]``
    where N is the count of observations. Original predicted value is kept
    in ``catalog_prior_bytes`` for forensic comparison.

    Self-protection layer (added 2026-05-08 per implementation-vs-model
    gap audit): every incoming evidence row is checked against the catalog
    row's ``model_spec``. Mismatches produce WARNINGs to stderr (not
    blocking — the byte-anchor measurement is still real); a
    ``model_spec_mismatch_reasons`` field is attached to the catalog row,
    and the row is forced to ``empirical_anchor_promotable=False``. The
    hard guard is the preflight check
    ``check_evidence_implementation_matches_model_spec``.
    """
    by_name: dict[str, list[TechniqueEvidence]] = {}
    for e in evidence:
        by_name.setdefault(e.technique, []).append(e)
    out = []
    for t in catalog:
        obs = by_name.get(t["name"], [])
        new_t = dict(t)
        # Self-protection: scan every observation against the model_spec.
        spec = t.get("model_spec")
        per_obs_mismatches: list[dict[str, Any]] = []
        for e in obs:
            reasons = detect_evidence_model_spec_mismatch(spec, e)
            if reasons:
                per_obs_mismatches.append({
                    "source": e.source,
                    "timestamp": e.timestamp,
                    "reasons": reasons,
                })
                if log_warnings:
                    print(
                        f"WARNING [cathedral_autopilot]: technique="
                        f"{t['name']!r} evidence row diverges from "
                        f"model_spec — reasons: {reasons}; source="
                        f"{e.source!r}",
                        file=sys.stderr,
                    )
        if per_obs_mismatches:
            new_t["model_spec_mismatch_reasons"] = per_obs_mismatches
            new_t["model_spec_mismatch_count"] = len(per_obs_mismatches)
        exact_negative_obs = [
            e for e in obs if _is_exact_negative_or_retired_evidence(e)
        ]
        promotable_obs = [
            e for e in obs if _is_explicitly_promotable_evidence(e)
        ]
        broad_exact_negative_obs = [
            e for e in exact_negative_obs
            if e.family_falsified is True or e.method_family_retired is True
        ]
        blocking_exact_negative_obs = (
            broad_exact_negative_obs
            if broad_exact_negative_obs
            else ([] if promotable_obs else exact_negative_obs)
        )
        if blocking_exact_negative_obs:
            exact_negative_obs = blocking_exact_negative_obs
            supporting_non_promotable_obs = [
                e for e in obs
                if (
                    not _is_exact_negative_or_retired_evidence(e)
                    and not _is_explicitly_promotable_evidence(e)
                )
            ]
            blockers = sorted({
                blocker
                for e in obs
                for blocker in e.dispatch_blockers
            })
            blockers.append(
                "exact_negative_result_requires_reactivation_before_empirical_byte_anchor"
            )
            new_t["catalog_prior_bytes"] = t["predicted_archive_bytes"]
            new_t["exact_negative_evidence_n"] = len(exact_negative_obs)
            new_t["exact_negative_sources"] = [
                e.source for e in exact_negative_obs if e.source
            ]
            new_t["exact_negative_review_packets"] = [
                e.exact_result_review_path
                for e in exact_negative_obs
                if e.exact_result_review_path
            ]
            new_t["exact_negative_falsification_scopes"] = _ordered_unique([
                e.falsification_scope for e in exact_negative_obs
            ])
            new_t["measured_config_statuses"] = _ordered_unique([
                e.measured_config_status for e in exact_negative_obs
            ])
            new_t["reactivation_criteria"] = _ordered_unique([
                criterion
                for e in exact_negative_obs
                for criterion in e.reactivation_criteria
            ])
            if supporting_non_promotable_obs:
                new_t["supporting_non_promotable_evidence_n"] = len(
                    supporting_non_promotable_obs
                )
                new_t["supporting_non_promotable_sources"] = [
                    e.source for e in supporting_non_promotable_obs if e.source
                ]
            new_t["empirical_anchor_promotable"] = False
            new_t["empirical_anchor_score_claim"] = False
            new_t["ready_for_exact_eval_dispatch"] = False
            new_t["rank_or_kill_eligible"] = False
            new_t["measured_config_retired"] = True
            new_t["active_ranking_blocked"] = True
            new_t["active_ranking_block_reason"] = (
                "exact_negative_measured_config_requires_reactivation"
            )
            new_t["family_falsified"] = any(
                e.family_falsified is True for e in exact_negative_obs
            )
            new_t["method_family_retired"] = any(
                e.method_family_retired is True for e in exact_negative_obs
            )
            new_t["measured_config_retired_only"] = (
                not new_t["family_falsified"]
                and not new_t["method_family_retired"]
            )
            new_t["exact_negative_classification"] = (
                "measured_config_retired_only"
                if new_t["measured_config_retired_only"]
                else "broader_family_or_method_retirement_claimed"
            )
            new_t["retired_from_active_ranking"] = True
            new_t["evidence_grade"] = (
                f"[exact-negative-N{len(exact_negative_obs)}; "
                "reactivation-required]"
            )
            new_t["dispatch_blockers"] = sorted(set(blockers))
            out.append(new_t)
            continue
        rankable_obs = promotable_obs if promotable_obs else obs
        empirical_bytes = [
            e.empirical_archive_bytes for e in rankable_obs
            if e.empirical_archive_bytes is not None
        ]
        if empirical_bytes:
            empirical_bytes.sort()
            n = len(empirical_bytes)
            median = empirical_bytes[n // 2]
            explicit_promotable = bool(promotable_obs) and all(
                _is_explicitly_promotable_evidence(e) for e in rankable_obs
            )
            blockers = sorted({
                blocker
                for e in rankable_obs
                for blocker in e.dispatch_blockers
            })
            promotability_blockers = _ordered_unique([
                blocker
                for e in rankable_obs
                if not _is_explicitly_promotable_evidence(e)
                for blocker in _promotability_blockers(e)
            ])
            rankable_mismatches: list[dict[str, Any]] = []
            for e in rankable_obs:
                reasons = detect_evidence_model_spec_mismatch(spec, e)
                if reasons:
                    rankable_mismatches.append({
                        "source": e.source,
                        "timestamp": e.timestamp,
                        "reasons": reasons,
                    })
            new_t["catalog_prior_bytes"] = t["predicted_archive_bytes"]
            new_t["predicted_archive_bytes"] = median
            new_t["empirical_anchor_n"] = n
            new_t["empirical_anchor_bytes"] = median
            new_t["empirical_anchor_sources"] = [
                e.source for e in rankable_obs if e.source
            ]
            new_t["empirical_anchor_evidence_semantics"] = sorted({
                e.evidence_semantics for e in rankable_obs if e.evidence_semantics
            })
            supporting_non_promotable_obs = [
                e for e in obs
                if (
                    e not in rankable_obs
                    and not _is_exact_negative_or_retired_evidence(e)
                )
            ]
            if supporting_non_promotable_obs:
                new_t["supporting_non_promotable_evidence_n"] = len(
                    supporting_non_promotable_obs
                )
                new_t["supporting_non_promotable_sources"] = [
                    e.source for e in supporting_non_promotable_obs if e.source
                ]
            if exact_negative_obs:
                new_t["retired_measured_config_evidence_n"] = len(exact_negative_obs)
                new_t["retired_measured_config_sources"] = [
                    e.source for e in exact_negative_obs if e.source
                ]
                new_t["retired_measured_config_statuses"] = _ordered_unique([
                    e.measured_config_status for e in exact_negative_obs
                ])
                new_t["retired_measured_config_reactivation_criteria"] = (
                    _ordered_unique([
                        criterion
                        for e in exact_negative_obs
                        for criterion in e.reactivation_criteria
                    ])
                )
            if explicit_promotable and not rankable_mismatches:
                new_t["evidence_grade"] = f"[empirical-anchor-N{n}]"
                new_t["empirical_anchor_promotable"] = True
            else:
                grade_suffix = "; planning-only"
                if rankable_mismatches:
                    grade_suffix = "; planning-only; model_spec_mismatch"
                new_t["evidence_grade"] = (
                    f"[empirical-anchor-N{n}{grade_suffix}]"
                )
                new_t["empirical_anchor_promotable"] = False
                new_t["empirical_anchor_score_claim"] = False
                new_t["ready_for_exact_eval_dispatch"] = False
                new_t["rank_or_kill_eligible"] = False
                new_t["active_ranking_blocked"] = True
                new_t["active_ranking_block_reason"] = (
                    "non_promotable_empirical_anchor_not_rank_or_kill_evidence"
                )
                merged_blockers = list(blockers) if blockers else [
                    "empirical_anchor_not_promotable_without_explicit_exact_eval_custody"
                ]
                for blocker in promotability_blockers:
                    if blocker not in merged_blockers:
                        merged_blockers.append(blocker)
                if per_obs_mismatches and (
                    "model_spec_mismatch_pending_faithful_implementation"
                    not in merged_blockers
                ):
                    merged_blockers.append(
                        "model_spec_mismatch_pending_faithful_implementation"
                    )
                new_t["dispatch_blockers"] = merged_blockers
        out.append(new_t)
    return out


# Ranking


def _rank_techniques(
    techniques: list[dict[str, Any]], *,
    d_seg: float, d_pose: float, current_archive_bytes: int,
    current_score: float, target_score: float | None,
    min_score_delta: float = 0.0,
    rank_axis: str = "dual",
    architecture_class: str = "hnerv",
    current_score_axis: str = "cuda",
) -> list[dict[str, Any]]:
    """Score each technique by predicted score gain per cost-dollar.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": every row now
    carries dual-axis predictions. The legacy ``predicted_score_*`` keys
    remain CUDA-axis-anchored (back-compat); new ``predicted_cuda_score`` /
    ``predicted_cpu_score`` keys are added.

    Args:
        d_seg, d_pose: distortion at the *current* operating point.
        current_score: current contest score.
        rank_axis: ``"dual"`` (default), ``"cuda"``, or ``"cpu"``. The
            dual-axis mode primary-ranks by the smaller predicted improvement
            across CUDA and CPU so neither scorer path is silently privileged.
        architecture_class: calibration class for the CPU-axis prediction
            (default ``"hnerv"``).
        current_score_axis: which axis ``current_score`` was measured on
            (``"cuda"`` default, or ``"cpu"`` if the operator passed a
            CPU-anchored baseline).
        min_score_delta: high-signal filter on the *primary-axis* delta.

    high-signal filter: techniques with ``predicted_score_delta`` strictly
    below ``min_score_delta`` are dropped (always keeping the top-1 row so
    callers never get an empty list).
    """
    if rank_axis not in {"cuda", "cpu", "dual"}:
        raise ValueError(
            f"rank_axis must be 'cuda', 'cpu', or 'dual'; got {rank_axis!r}"
        )
    if current_score_axis not in {"cuda", "cpu"}:
        raise ValueError(
            f"current_score_axis must be 'cuda' or 'cpu'; got {current_score_axis!r}"
        )
    cal = CudaCpuCalibration(architecture_class=architecture_class)
    # Compute the "current score" on the OPPOSITE axis for delta computation.
    if current_score_axis == "cuda":
        d_seg_cuda = d_seg
        d_pose_cuda = d_pose
        d_seg_cpu = d_seg / cal.r_seg
        d_pose_cpu = d_pose / cal.r_pose
        current_cuda_score = current_score
        current_cpu_band = cal.predict_cpu_from_cuda(
            current_score,
            d_pose_cuda=d_pose_cuda,
            d_seg_cuda=d_seg_cuda,
            archive_bytes=current_archive_bytes,
        )
        current_cpu_score = current_cpu_band.score_point
    else:
        d_seg_cpu = d_seg
        d_pose_cpu = d_pose
        d_seg_cuda = d_seg * cal.r_seg
        d_pose_cuda = d_pose * cal.r_pose
        current_cpu_score = current_score
        current_cuda_band = cal.predict_cuda_from_cpu(
            current_score,
            d_pose_cpu=d_pose_cpu,
            d_seg_cpu=d_seg_cpu,
            archive_bytes=current_archive_bytes,
        )
        current_cuda_score = current_cuda_band.score_point
    rows: list[dict[str, Any]] = []
    for t in techniques:
        active_ranking_blocked = _is_active_ranking_blocked(t)
        retired_from_active_ranking = bool(t.get("retired_from_active_ranking"))
        if active_ranking_blocked or t["predicted_archive_bytes"] >= current_archive_bytes:
            score_after = current_cuda_score
            score_delta = 0.0
            cpu_score_after = current_cpu_score
            cpu_score_lo = current_cpu_score
            cpu_score_hi = current_cpu_score
            cpu_calibration = "no-change"
        else:
            # CUDA-axis prediction (legacy back-compat keys).
            score_after = _technique_score_after(
                baseline_bytes=current_archive_bytes,
                technique_bytes=t["predicted_archive_bytes"],
                d_seg=d_seg_cuda, d_pose=d_pose_cuda,
            )
            score_delta = current_cuda_score - score_after
            if current_score_axis == "cpu":
                # The supplied distortions are already CPU-axis components;
                # do not rebase them a second time.
                cpu_score_after = _technique_score_after(
                    baseline_bytes=current_archive_bytes,
                    technique_bytes=t["predicted_archive_bytes"],
                    d_seg=d_seg_cpu, d_pose=d_pose_cpu,
                )
                cpu_score_lo = cpu_score_after
                cpu_score_hi = cpu_score_after
                cpu_calibration = "cpu-axis-direct"
            else:
                # CPU-axis prediction band from measured CUDA components.
                dual = _technique_score_after_dual_axis(
                    baseline_bytes=current_archive_bytes,
                    technique_bytes=t["predicted_archive_bytes"],
                    d_seg_cuda=d_seg_cuda, d_pose_cuda=d_pose_cuda,
                    architecture_class=architecture_class,
                )
                cpu_score_after = dual["predicted_cpu_score"]
                cpu_score_lo = dual["predicted_cpu_score_lo"]
                cpu_score_hi = dual["predicted_cpu_score_hi"]
                cpu_calibration = dual["predicted_cpu_score_calibration"]
        cpu_score_delta = current_cpu_score - cpu_score_after
        cost_dollars = max(t["cost_dollars"], 0.5)  # floor for pure-CPU items
        # Pick primary delta based on rank_axis. Existing back-compat key
        # ``predicted_score_delta`` continues to be CUDA-axis. Dual-axis mode
        # is conservative: rank by the smaller predicted gain and target-gap
        # by the worse predicted score across CUDA/CPU.
        if rank_axis == "cuda":
            primary_delta = score_delta
            primary_score_after = score_after
        elif rank_axis == "cpu":
            primary_delta = cpu_score_delta
            primary_score_after = cpu_score_after
        else:
            primary_delta = min(score_delta, cpu_score_delta)
            primary_score_after = max(score_after, cpu_score_after)
        gain_per_dollar = primary_delta / cost_dollars if cost_dollars > 0 else 0.0
        gap_to_target = (
            primary_score_after - target_score if target_score is not None else None
        )
        row = {
            **t,
            "current_score_baseline": current_score,
            "predicted_score_after": score_after,
            "predicted_score_delta": score_delta,
            "predicted_cuda_score": score_after,
            "predicted_cuda_score_delta": score_delta,
            "predicted_cpu_score": cpu_score_after,
            "predicted_cpu_score_lo": cpu_score_lo,
            "predicted_cpu_score_hi": cpu_score_hi,
            "predicted_cpu_score_delta": cpu_score_delta,
            "predicted_cpu_score_calibration": cpu_calibration,
            "rank_axis": rank_axis,
            "current_score_axis": current_score_axis,
            "primary_score_after": primary_score_after,
            "primary_score_delta": primary_delta,
            "gain_per_dollar": gain_per_dollar,
            "gain_per_hour": primary_delta / max(t["cost_hours"], 0.1),
            "predicted_gap_to_target": gap_to_target,
            "reaches_target": (
                gap_to_target is not None and gap_to_target <= 0
            ),
            "active_ranking_blocked": active_ranking_blocked,
            "retired_from_active_ranking": retired_from_active_ranking,
        }
        rows.append(_attach_fail_closed_dispatch_fields(row))
    # Primary-axis ranking; default dual mode uses the conservative smaller
    # predicted gain across CUDA and CPU.
    rows.sort(key=lambda r: (
        r["active_ranking_blocked"],
        -r["primary_score_delta"],
        r["cost_dollars"],
    ))
    if min_score_delta > 0.0 and rows:
        kept = [r for r in rows if r["primary_score_delta"] >= min_score_delta]
        if not kept:
            kept = [rows[0]]  # always preserve top-1
        rows = kept
    return rows


def _score_delta_if_validated(
    *,
    candidate_archive_bytes: int | None,
    current_archive_bytes: int,
    current_score: float,
    d_seg: float,
    d_pose: float,
) -> float:
    """Rate-only upside if a non-promotable byte anchor is later validated.

    This is explicitly not active ranking evidence. It exists so proxy/MPS/CPU
    and unknown-catalog signals remain visible while still failing closed for
    promotion and dispatch.
    """
    if candidate_archive_bytes is None:
        return 0.0
    if candidate_archive_bytes <= 0:
        return 0.0
    if candidate_archive_bytes >= current_archive_bytes:
        return 0.0
    return current_score - contest_score(d_seg, d_pose, candidate_archive_bytes)


def _validation_status_for_unknown(bucket: dict[str, Any]) -> str:
    min_bytes = bucket.get("min_empirical_archive_bytes")
    try:
        min_bytes_int = int(min_bytes) if min_bytes is not None else None
    except (TypeError, ValueError):
        min_bytes_int = None
    if min_bytes_int is not None and min_bytes_int <= 0:
        return "unknown_invalid_or_missing_archive_bytes"
    text = " ".join(
        str(item).lower()
        for part in (
            [bucket.get("status", "")],
            bucket.get("verdicts", []),
            bucket.get("dispatch_blockers", []),
        )
        for item in part
        if item
    )
    if bucket.get("exact_negative_rows"):
        return "unknown_exact_negative_or_retired"
    if bucket.get("byte_proxy_only_any") or "byte_proxy" in text or "retracted" in text:
        return "unknown_byte_proxy_needs_packetization"
    if bucket.get("cuda_eval_worth_testing_false"):
        return "unknown_not_cuda_worth_testing_until_reactivated"
    if bucket.get("cuda_eval_worth_testing_any"):
        return "unknown_candidate_needs_catalog_or_alias_before_dispatch"
    return "unknown_planning_signal_needs_catalog_or_alias"


def _attach_fail_closed_dispatch_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Make score/dispatch authority explicit on ranked and recommended rows."""

    normalize_score_authority_fields(row)
    score_claim = row["score_claim"]
    ready = row["ready_for_exact_eval_dispatch"]
    blockers = row.get("dispatch_blockers")
    if not isinstance(blockers, list):
        blockers = [] if blockers in (None, "") else [str(blockers)]
    if not ready:
        evidence_grade = str(row.get("evidence_grade") or "").lower()
        if "predicted" in evidence_grade and not any(
            str(blocker).startswith("predicted_row_requires_exact_eval_before_dispatch")
            for blocker in blockers
        ):
            blockers.append("predicted_row_requires_exact_eval_before_dispatch")
        if not blockers and not score_claim:
            blockers.append("score_claim_false_requires_exact_eval_before_dispatch")
        row.setdefault("recommended_action", "build_or_exact_eval_before_dispatch")
        row.setdefault(
            "active_ranking_block_reason",
            ";".join(str(blocker) for blocker in blockers)
            or "ready_for_exact_eval_dispatch_false",
        )
        row.setdefault("dispatch_recommendation_status", "blocked_until_exact_eval")
    else:
        row.setdefault("recommended_action", "dispatch_exact_eval_after_claim")
        row.setdefault("dispatch_recommendation_status", "ready_for_exact_eval_dispatch")
    row["dispatch_blockers"] = blockers
    return row


def build_validation_queue(
    *,
    ranked_catalogs: list[list[dict[str, Any]]],
    evidence_report: dict[str, Any],
    d_seg: float,
    d_pose: float,
    current_archive_bytes: int,
    current_score: float,
) -> list[dict[str, Any]]:
    """Preserve blocked/proxy/cross-paradigm signals without ranking them.

    The queue answers "what should we validate next if we want no signal
    loss?" It never turns non-promotable evidence into a dispatch decision.
    Active rankings still require promotable exact evidence.
    """
    rows: list[dict[str, Any]] = []
    for catalog in ranked_catalogs:
        for item in catalog:
            if not _is_active_ranking_blocked(item):
                continue
            candidate_bytes = item.get(
                "empirical_anchor_bytes",
                item.get("predicted_archive_bytes"),
            )
            try:
                candidate_bytes_int = (
                    int(candidate_bytes) if candidate_bytes is not None else None
                )
            except (TypeError, ValueError):
                candidate_bytes_int = None
            retired = bool(item.get("retired_from_active_ranking"))
            potential_delta = (
                0.0 if retired else _score_delta_if_validated(
                    candidate_archive_bytes=candidate_bytes_int,
                    current_archive_bytes=current_archive_bytes,
                    current_score=current_score,
                    d_seg=d_seg,
                    d_pose=d_pose,
                )
            )
            rows.append({
                "technique": item.get("name"),
                "queue_source": "cataloged_blocked_candidate",
                "validation_status": (
                    "reactivation_required_exact_negative"
                    if retired
                    else item.get(
                        "active_ranking_block_reason",
                        "blocked_pending_exact_custody",
                    )
                ),
                "archive_bytes_if_validated": candidate_bytes_int,
                "potential_score_delta_if_validated": potential_delta,
                "evidence_grade": item.get("evidence_grade", ""),
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "score_claim": False,
                "dispatch_blockers": item.get("dispatch_blockers", []),
                "sources": item.get("empirical_anchor_sources", []),
            })

    for bucket in evidence_report.get("unknown_evidence_techniques", []):
        candidate_bytes = bucket.get("min_empirical_archive_bytes")
        try:
            candidate_bytes_int = (
                int(candidate_bytes) if candidate_bytes is not None else None
            )
        except (TypeError, ValueError):
            candidate_bytes_int = None
        status = _validation_status_for_unknown(bucket)
        potential_delta = _score_delta_if_validated(
            candidate_archive_bytes=candidate_bytes_int,
            current_archive_bytes=current_archive_bytes,
            current_score=current_score,
            d_seg=d_seg,
            d_pose=d_pose,
        )
        if status in {
            "unknown_invalid_or_missing_archive_bytes",
            "unknown_byte_proxy_needs_packetization",
            "unknown_exact_negative_or_retired",
            "unknown_not_cuda_worth_testing_until_reactivated",
        }:
            potential_delta = 0.0
        rows.append({
            "technique": bucket.get("technique"),
            "queue_source": "unknown_evidence_candidate",
            "validation_status": status,
            "archive_bytes_if_validated": candidate_bytes_int,
            "potential_score_delta_if_validated": potential_delta,
            "evidence_grade": "; ".join(bucket.get("evidence_grades", [])),
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "score_claim": False,
            "dispatch_blockers": bucket.get("dispatch_blockers", []),
            "sources": bucket.get("sources", []),
            "latest_timestamp": bucket.get("latest_timestamp", ""),
            "n_rows": bucket.get("n_rows", 0),
        })

    status_priority = {
        "non_promotable_empirical_anchor_not_rank_or_kill_evidence": 0,
        "blocked_pending_exact_custody": 0,
        "unknown_candidate_needs_catalog_or_alias_before_dispatch": 1,
        "unknown_planning_signal_needs_catalog_or_alias": 2,
        "unknown_byte_proxy_needs_packetization": 3,
        "unknown_not_cuda_worth_testing_until_reactivated": 4,
        "unknown_invalid_or_missing_archive_bytes": 4,
        "reactivation_required_exact_negative": 5,
        "exact_negative_measured_config_requires_reactivation": 5,
        "unknown_exact_negative_or_retired": 5,
    }
    rows.sort(key=lambda row: (
        status_priority.get(str(row.get("validation_status")), 2),
        -float(row.get("potential_score_delta_if_validated") or 0.0),
        int(row.get("archive_bytes_if_validated") or 10**18),
        str(row.get("technique") or ""),
    ))
    return rows


def _maybe_axis_priorities(
    d_seg: float, d_pose: float, archive_bytes: int,
    target_score: float | None,
) -> list[dict[str, Any]] | None:
    """B2 wiring: pull dispatch_advisor's per-axis priority advice.

    Returns None if dispatch_advisor isn't importable from this Python
    path; otherwise returns a list of axis priority dicts ranked by
    marginal score-per-unit-effort. The advisor lives in tools/, so this
    must do a path-flex import (the autopilot can be invoked from anywhere).
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import dispatch_advisor as _adv
        adv = _adv.advise_candidate(
            label="autopilot_inline",
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
            target_score=target_score,
        )
        return [dict(p) for p in adv.axis_priorities]
    except (ImportError, AttributeError, TypeError):
        return None


def build_plan(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    target_score: float | None = None,
    label: str = "current_candidate",
    prior_evidence: list[TechniqueEvidence] | None = None,
    min_score_delta: float = 0.0,
    include_axis_priorities: bool = True,
    rank_axis: str = "dual",
    current_score_axis: str = "cuda",
    architecture_class: str = "hnerv",
) -> AutopilotPlan:
    """Build a complete autopilot plan for the supplied operator state.

    ``prior_evidence`` is the continual-learning hook: any catalog entry
    whose ``name`` appears in the evidence list is updated to use the
    median empirical archive size before ranking.

    ``min_score_delta`` is the high-signal filter: techniques predicted
    to deliver less than this score-delta are dropped (top-1 always kept
    so callers don't get an empty list).
    """
    score = contest_score(d_seg, d_pose, archive_bytes)
    decomp = score_decomposition(d_seg, d_pose, archive_bytes)
    regime = operating_regime(d_pose)
    grad = score_gradient(d_seg, d_pose)
    flip = importance_flip_threshold()

    enc_catalog = ENCODER_TECHNIQUES
    arch_catalog = ARCH_TECHNIQUES
    # T19 follow-on (2026-05-09): include track-registry-derived catalogs
    # so loss modifiers (T20/T22) and representation lanes (Lane-12-v2) are
    # visible to the planner. Empty lists when registry import fails (soft
    # degradation per registry contract).
    loss_modifier_catalog = LOSS_MODIFIER_TECHNIQUES
    representation_lane_catalog = REPRESENTATION_LANES
    evidence_report = summarize_evidence_semantics(
        prior_evidence,
        [
            ENCODER_TECHNIQUES,
            ARCH_TECHNIQUES,
            loss_modifier_catalog,
            representation_lane_catalog,
        ],
    )
    if prior_evidence:
        enc_catalog = update_catalog_from_evidence(enc_catalog, prior_evidence)
        arch_catalog = update_catalog_from_evidence(arch_catalog, prior_evidence)
        loss_modifier_catalog = update_catalog_from_evidence(
            loss_modifier_catalog, prior_evidence
        )
        representation_lane_catalog = update_catalog_from_evidence(
            representation_lane_catalog, prior_evidence
        )
        evidence_report = attach_active_ranking_summary(
            evidence_report,
            [
                enc_catalog,
                arch_catalog,
                loss_modifier_catalog,
                representation_lane_catalog,
            ],
        )

    encoder_ranked = _rank_techniques(
        enc_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=min_score_delta,
        rank_axis=rank_axis,
        current_score_axis=current_score_axis,
        architecture_class=architecture_class,
    )
    arch_ranked = _rank_techniques(
        arch_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=min_score_delta,
        rank_axis=rank_axis,
        current_score_axis=current_score_axis,
        architecture_class=architecture_class,
    )
    # T19 follow-on (2026-05-09): rank loss-modifier + representation-lane
    # catalogs through the SAME _rank_techniques machinery so they participate
    # in the planner's ranking pipeline. Loss modifiers default to baseline
    # bytes so they never displace a real byte-saver in the top-3 unless an
    # empirical anchor is supplied.
    loss_modifier_ranked = _rank_techniques(
        loss_modifier_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=0.0,  # always include loss-modifiers; do not filter
        rank_axis=rank_axis,
        current_score_axis=current_score_axis,
        architecture_class=architecture_class,
    ) if loss_modifier_catalog else []
    representation_lane_ranked = _rank_techniques(
        representation_lane_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=0.0,
        rank_axis=rank_axis,
        current_score_axis=current_score_axis,
        architecture_class=architecture_class,
    ) if representation_lane_catalog else []
    # Combined top-3: best by score-delta across encoder+arch lists. Loss
    # modifiers and representation lanes are SURFACED separately and only
    # rolled into recommended_top_3 when promotion_eligible AND not
    # active_ranking_blocked (guard against premature promotion).
    promotable_loss_mods = [
        r for r in loss_modifier_ranked
        if r.get("promotion_eligible") and not r["active_ranking_blocked"]
    ]
    promotable_repr_lanes = [
        r for r in representation_lane_ranked
        if r.get("promotion_eligible") and not r["active_ranking_blocked"]
    ]
    combined = sorted(
        encoder_ranked + arch_ranked + promotable_loss_mods + promotable_repr_lanes,
        key=lambda r: (
            r["active_ranking_blocked"],
            -r["primary_score_delta"],
            r["cost_dollars"],
        ),
    )
    top_3 = combined[:3]
    validation_queue = build_validation_queue(
        ranked_catalogs=[encoder_ranked, arch_ranked],
        evidence_report=evidence_report,
        d_seg=d_seg,
        d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
    )

    # Target-score gap analysis
    gap = {}
    if target_score is not None:
        gap["target_score"] = target_score
        gap["score_gap"] = score - target_score
        # Pose-only path
        required_pose = equal_score_curve_d_pose(target_score, d_seg, archive_bytes)
        gap["pose_only_required_d_pose"] = required_pose
        gap["pose_only_feasible"] = required_pose is not None
        if required_pose is not None and required_pose > 0:
            gap["pose_only_improvement_factor"] = d_pose / required_pose
        # Bytes-only path
        required_bytes = equal_score_curve_archive_bytes(target_score, d_seg, d_pose)
        gap["bytes_only_required_archive_bytes"] = required_bytes
        gap["bytes_only_feasible"] = required_bytes is not None
        if required_bytes is not None:
            gap["bytes_only_savings_required"] = archive_bytes - required_bytes
        # Information-theoretic floor at this byte budget
        gap["information_floor_at_current_bytes"] = information_floor(archive_bytes)

    notes = [
        f"Operating regime: {regime.advice}",
        f"Importance flip threshold: d_pose = {flip:.2e}",
        f"Score decomposition: seg={decomp.seg_term:.5f} + pose={decomp.pose_term:.5f} + rate={decomp.rate_term:.5f}",
    ]
    if target_score is not None and score <= target_score:
        notes.append(f"ALREADY AT TARGET: current score {score:.5f} <= target {target_score:.5f}")
    elif target_score is not None:
        notes.append(
            f"To reach {target_score:.5f}: need -{score - target_score:.5f} score points. "
            f"Top-3 techniques rank by {rank_axis}-axis predicted score-delta."
        )
    notes.append(
        "PR101 hand-coded entropy probes are saturated near the 178 KB brotli/AAC band: "
        "the best charged Markov-1 table rerun is 200,219 B, so the currently tested "
        "table-transmission path loses. This is not a global no-ML impossibility proof; "
        "sub-178 KB still needs a new charged coder, a shared decoder model, or lower-entropy "
        "weights. Architecture has 5-10x more predicted headroom."
    )
    unknown_count = int(evidence_report.get("unknown_evidence_technique_count", 0))
    if unknown_count:
        unknown_names = [
            row["technique"]
            for row in evidence_report.get("unknown_evidence_techniques", [])
        ][:8]
        notes.append(
            "Evidence feed contains "
            f"{unknown_count} unknown technique name(s) that are reported "
            f"but not ranked until cataloged or aliased: {', '.join(unknown_names)}"
        )
    blocked_names = evidence_report.get("active_ranking_blocked_techniques", [])
    if blocked_names:
        notes.append(
            "Non-promotable, proxy, or exact-negative empirical anchors are "
            "assigned zero active ranking delta: "
            f"{', '.join(blocked_names[:8])}"
        )
    exact_neg_names = evidence_report.get("cataloged_exact_negative_techniques", [])
    if exact_neg_names:
        notes.append(
            "Exact-negative evidence retires only the measured config unless "
            "family_falsified/method_family_retired is explicitly true: "
            f"{', '.join(exact_neg_names[:8])}"
        )
    if validation_queue:
        notes.append(
            "Validation queue preserves blocked/proxy/cross-paradigm signals "
            "without promoting them; queue rows are not score claims and cannot "
            "dispatch until blockers close."
        )
    # T19 follow-on (2026-05-09): surface gated track-registry entries.
    gated_loss_mods = [
        r["name"] for r in loss_modifier_ranked
        if not r.get("promotion_eligible")
    ]
    gated_repr_lanes = [
        r["name"] for r in representation_lane_ranked
        if not r.get("promotion_eligible")
    ]
    if gated_loss_mods:
        notes.append(
            f"Loss-modifier tracks visible but GATED (entry conditions pending): "
            f"{', '.join(gated_loss_mods)}. See track_registry.entry_conditions "
            f"for unblock requirements."
        )
    if gated_repr_lanes:
        notes.append(
            f"Representation lanes visible but GATED (Phase B preconditions / "
            f"contest-CUDA anchor pending): {', '.join(gated_repr_lanes)}. See "
            f"track_registry.entry_conditions for unblock requirements."
        )

    return AutopilotPlan(
        schema=SCHEMA_VERSION,
        tool=TOOL_NAME,
        evidence_grade=EVIDENCE_GRADE,
        operator_state={
            "label": label,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": archive_bytes,
            "current_score": score,
            "rank_axis": rank_axis,
            "current_score_axis": current_score_axis,
            "architecture_class": architecture_class,
        },
        score_geometry={
            "decomposition": {
                "seg_term": decomp.seg_term,
                "pose_term": decomp.pose_term,
                "rate_term": decomp.rate_term,
            },
            "gradient": {
                "d_seg": grad.d_seg,
                "d_pose": grad.d_pose,
                "d_bytes": grad.d_bytes,
            },
            "operating_regime": {
                "pose_dominates": regime.pose_dominates,
                "seg_dominates": regime.seg_dominates,
                "marginal_ratio_seg_over_pose": regime.marginal_ratio_seg_over_pose,
                "advice": regime.advice,
            },
            "flip_threshold": flip,
        },
        encoder_technique_ranking=encoder_ranked,
        arch_technique_ranking=arch_ranked,
        loss_modifier_technique_ranking=loss_modifier_ranked,
        representation_lane_ranking=representation_lane_ranked,
        recommended_top_3=top_3,
        target_score_gap_analysis=gap,
        notes=notes,
        axis_priorities=(
            _maybe_axis_priorities(d_seg, d_pose, archive_bytes, target_score)
            if include_axis_priorities else None
        ) or [],
        evidence_semantics_report=evidence_report,
        validation_queue=validation_queue,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Plan from a single (d_seg, d_pose, B) point")
    p_plan.add_argument("--label", default="current_candidate")
    p_plan.add_argument("--d-seg", type=float, required=True)
    p_plan.add_argument("--d-pose", type=float, required=True)
    p_plan.add_argument("--archive-bytes", type=int, required=True)
    p_plan.add_argument("--target-score", type=float, default=None)
    p_plan.add_argument("--output", type=Path, default=None)
    p_plan.add_argument("--summary-text", action="store_true")
    p_plan.add_argument("--prior-evidence", type=Path, default=None,
                        help="JSONL/JSON of TechniqueEvidence rows; updates catalog before ranking")
    p_plan.add_argument("--min-score-delta", type=float, default=0.0,
                        help="High-signal filter: drop techniques predicting <delta score (top-1 always kept)")
    p_plan.add_argument(
        "--rank-axis",
        choices=["dual", "cuda", "cpu"],
        default="dual",
        help=(
            "Primary ranking axis; default dual ranks by the smaller predicted "
            "CUDA/CPU gain while both axes remain reported"
        ),
    )
    p_plan.add_argument("--current-score-axis", "--score-axis",
                        choices=["cuda", "cpu"], default="cuda",
                        help="Axis of the supplied d_seg/d_pose/current state")
    p_plan.add_argument("--architecture-class", default="hnerv",
                        help="CPU/CUDA calibration class or profile label")

    p_pareto = sub.add_parser("plan-from-pareto",
                              help="Plan for every candidate in 3-axis Pareto JSON")
    p_pareto.add_argument("--pareto-json", type=Path, required=True)
    p_pareto.add_argument("--target-score", type=float, default=None)
    p_pareto.add_argument("--output", type=Path, required=True)
    p_pareto.add_argument("--prior-evidence", type=Path, default=None)
    p_pareto.add_argument("--min-score-delta", type=float, default=0.0)
    p_pareto.add_argument("--rank-axis", choices=["dual", "cuda", "cpu"], default="dual")
    p_pareto.add_argument("--current-score-axis", "--score-axis",
                          choices=["cuda", "cpu"], default="cuda")
    p_pareto.add_argument("--architecture-class", default="hnerv")

    p_evid = sub.add_parser("evidence-update",
                            help="Print catalog updated by an evidence file (no plan)")
    p_evid.add_argument("--prior-evidence", type=Path, required=True)
    p_evid.add_argument("--output", type=Path, default=None)
    p_evid.add_argument("--catalog", choices=["encoder", "arch", "both"], default="both")

    args = parser.parse_args(argv)

    prior_ev = (
        _load_evidence(args.prior_evidence)
        if getattr(args, "prior_evidence", None) is not None
        else None
    )

    if args.cmd == "plan":
        plan = build_plan(
            d_seg=args.d_seg,
            d_pose=args.d_pose,
            archive_bytes=args.archive_bytes,
            target_score=args.target_score,
            label=args.label,
            prior_evidence=prior_ev,
            min_score_delta=args.min_score_delta,
            rank_axis=args.rank_axis,
            current_score_axis=args.current_score_axis,
            architecture_class=args.architecture_class,
        )
        payload = asdict(plan)
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        if args.summary_text:
            print(_render_plan_summary(plan))
        else:
            print(text)
        return 0

    if args.cmd == "plan-from-pareto":
        if not args.pareto_json.is_file():
            raise SystemExit(f"pareto json not found: {args.pareto_json}")
        pareto_payload = json.loads(args.pareto_json.read_text(encoding="utf-8"))
        candidates = pareto_payload.get("candidates", [])
        plans: list[dict[str, Any]] = []
        for c in candidates:
            try:
                plan = build_plan(
                    d_seg=float(c["d_seg"]),
                    d_pose=float(c["d_pose"]),
                    archive_bytes=int(c["archive_bytes"]),
                    target_score=args.target_score,
                    label=str(c.get("label", "?")),
                    prior_evidence=prior_ev,
                    min_score_delta=args.min_score_delta,
                    rank_axis=args.rank_axis,
                    current_score_axis=args.current_score_axis,
                    architecture_class=args.architecture_class,
                )
            except (KeyError, ValueError):
                continue
            plans.append(asdict(plan))
        manifest = {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "evidence_grade": EVIDENCE_GRADE,
            "input_pareto_json": str(args.pareto_json),
            "n_plans": len(plans),
            "target_score": args.target_score,
            "min_score_delta": args.min_score_delta,
            "rank_axis": args.rank_axis,
            "current_score_axis": args.current_score_axis,
            "architecture_class": args.architecture_class,
            "n_evidence_rows": len(prior_ev) if prior_ev else 0,
            "plans": plans,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {len(plans)} autopilot plans to {args.output}")
        return 0

    if args.cmd == "evidence-update":
        ev = prior_ev or []
        out: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "n_evidence_rows": len(ev),
        }
        updated_catalogs: list[list[dict[str, Any]]] = []
        report = summarize_evidence_semantics(
            ev,
            [ENCODER_TECHNIQUES, ARCH_TECHNIQUES],
        )
        if args.catalog in ("encoder", "both"):
            out["encoder_catalog"] = update_catalog_from_evidence(ENCODER_TECHNIQUES, ev)
            updated_catalogs.append(out["encoder_catalog"])
        if args.catalog in ("arch", "both"):
            out["arch_catalog"] = update_catalog_from_evidence(ARCH_TECHNIQUES, ev)
            updated_catalogs.append(out["arch_catalog"])
        out["evidence_semantics_report"] = attach_active_ranking_summary(
            report,
            updated_catalogs,
        )
        text = json.dumps(out, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"wrote evidence-updated catalog to {args.output}")
        else:
            print(text)
        return 0

    return 2


def _render_plan_summary(plan: AutopilotPlan) -> str:
    s = plan.operator_state
    sg = plan.score_geometry
    lines: list[str] = []
    lines.append(f"=== Cathedral Autopilot Plan: {s['label']} ===")
    lines.append(
        f"State: d_seg={s['d_seg']:.4e}, d_pose={s['d_pose']:.4e}, "
        f"B={s['archive_bytes']:,} -> score {s['current_score']:.5f}"
    )
    lines.append(
        f"Decomposition: seg={sg['decomposition']['seg_term']:.5f} + "
        f"pose={sg['decomposition']['pose_term']:.5f} + "
        f"rate={sg['decomposition']['rate_term']:.5f}"
    )
    lines.append(f"Regime: {sg['operating_regime']['advice']}")
    if plan.target_score_gap_analysis:
        gap = plan.target_score_gap_analysis
        if "target_score" in gap:
            lines.append(
                f"Target {gap['target_score']:.5f}: gap {gap['score_gap']:+.5f}"
            )
            if gap.get("bytes_only_feasible"):
                lines.append(
                    f"  Bytes-only path: shrink to "
                    f"{gap['bytes_only_required_archive_bytes']:,} bytes "
                    f"(saves {gap['bytes_only_savings_required']:,})"
                )
            if gap.get("pose_only_feasible"):
                f_imp = gap.get("pose_only_improvement_factor")
                f_imp_s = f"{f_imp:.2f}x" if f_imp is not None else "(?)"
                lines.append(
                    f"  Pose-only path: improve d_pose to "
                    f"{gap['pose_only_required_d_pose']:.4e} "
                    f"(currently {s['d_pose']:.4e}; factor {f_imp_s})"
                )
    lines.append("")
    rank_axis = s.get("rank_axis", "dual")
    lines.append(f"TOP-3 RECOMMENDED ACTIONS (ranked by {rank_axis}-axis score-delta then cost):")
    for i, rec in enumerate(plan.recommended_top_3, 1):
        lines.append(
            f"  {i}. {rec['name']:<40s}  "
            f"Delta={rec['primary_score_delta']:+.5f}  "
            f"${rec['cost_dollars']}  ({rec['cost_hours']}h)  "
            f"[{rec['evidence_grade']}]"
        )
        lines.append(f"     -> {rec['description']}")
    lines.append("")
    lines.append("ENCODER TECHNIQUES (top 3 by score-delta):")
    for r in plan.encoder_technique_ranking[:3]:
        lines.append(
            f"  {r['name']:<40s}  bytes={r['predicted_archive_bytes']:,}  "
            f"score={r['primary_score_after']:.5f}  Delta={r['primary_score_delta']:+.5f}"
        )
    lines.append("")
    lines.append("ARCHITECTURE TECHNIQUES (top 3 by score-delta):")
    for r in plan.arch_technique_ranking[:3]:
        lines.append(
            f"  {r['name']:<40s}  bytes={r['predicted_archive_bytes']:,}  "
            f"score={r['primary_score_after']:.5f}  Delta={r['primary_score_delta']:+.5f}"
        )
    if plan.validation_queue:
        lines.append("")
        lines.append(
            "VALIDATION QUEUE (blocked/non-promotable signals; not score claims):"
        )
        for r in plan.validation_queue[:5]:
            candidate_bytes = r.get("archive_bytes_if_validated")
            candidate_s = f"{candidate_bytes:,}" if candidate_bytes is not None else "?"
            lines.append(
                f"  {r['technique']:<48s} "
                f"bytes_if_valid={candidate_s:<9s} "
                f"potential_delta={r['potential_score_delta_if_validated']:+.5f} "
                f"status={r['validation_status']}"
            )
    lines.append("")
    lines.append("NOTES:")
    for n in plan.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
