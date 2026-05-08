#!/usr/bin/env python3
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
    recommended_top_3: list[dict[str, Any]] = field(default_factory=list)
    target_score_gap_analysis: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    axis_priorities: list[dict[str, Any]] = field(default_factory=list)


def _technique_score_after(
    *, baseline_bytes: int, technique_bytes: int,
    d_seg: float, d_pose: float,
) -> float:
    """Score if technique replaces baseline encoder, holding distortion fixed."""
    return contest_score(d_seg, d_pose, technique_bytes)


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
    score_claim: bool | None = None
    promotion_eligible: bool | None = None
    rank_or_kill_eligible: bool | None = None
    ready_for_exact_eval_dispatch: bool | None = None
    dispatch_blockers: list[str] = field(default_factory=list)
    source: str = ""
    timestamp: str = ""


def _load_evidence(path: Path) -> list[TechniqueEvidence]:
    """Read JSONL or JSON-array of evidence rows. Skips malformed rows."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: list[Any]
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: list[TechniqueEvidence] = []
    for r in rows:
        if not isinstance(r, dict) or "technique" not in r:
            continue
        out.append(TechniqueEvidence(
            technique=str(r["technique"]),
            empirical_archive_bytes=(
                int(r["empirical_archive_bytes"])
                if r.get("empirical_archive_bytes") is not None else None
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
            score_claim=(
                bool(r["score_claim"]) if r.get("score_claim") is not None else None
            ),
            promotion_eligible=(
                bool(r["promotion_eligible"])
                if r.get("promotion_eligible") is not None else None
            ),
            rank_or_kill_eligible=(
                bool(r["rank_or_kill_eligible"])
                if r.get("rank_or_kill_eligible") is not None else None
            ),
            ready_for_exact_eval_dispatch=(
                bool(r["ready_for_exact_eval_dispatch"])
                if r.get("ready_for_exact_eval_dispatch") is not None else None
            ),
            dispatch_blockers=[
                str(blocker) for blocker in r.get("dispatch_blockers", [])
            ] if isinstance(r.get("dispatch_blockers", []), list) else [],
            source=str(r.get("source", "")),
            timestamp=str(r.get("timestamp", "")),
        ))
    return out


def _is_explicitly_promotable_evidence(evidence: TechniqueEvidence) -> bool:
    """Return true only for rows that explicitly opt into promotion semantics.

    Absence of custody fields must fail closed. CPU/MPS/proxy rows may still
    update planning byte estimates, but they cannot be labeled promotable unless
    the producer explicitly records score/promotion/dispatch readiness and has
    no dispatch blockers.
    """
    text = " ".join(
        part.strip().lower()
        for part in (
            evidence.evidence_grade,
            evidence.evidence_marker,
            evidence.evidence_semantics,
            evidence.source,
        )
        if part
    )
    exact_cuda = (
        "contest-cuda" in text
        or "contest_cuda" in text
        or "exact_cuda_auth_eval" in text
        or evidence.evidence_grade.strip().lower() in {"a", "a++"}
    )
    proxy_marker = any(
        marker in text
        for marker in (
            "mps",
            "cpu-prep",
            "cpu_prep",
            "proxy",
            "prediction",
            "predicted",
            "forensic",
            "research-signal",
            "research_signal",
        )
    )
    return (
        evidence.score_claim is True
        and evidence.promotion_eligible is True
        and evidence.rank_or_kill_eligible is True
        and evidence.ready_for_exact_eval_dispatch is True
        and not evidence.dispatch_blockers
        and exact_cuda
        and not proxy_marker
    )


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

    capacity = str(spec.get("capacity_constraint", "")).lower()
    arch_class = str(spec.get("architecture_class", "")).lower()
    substrate = str(spec.get("substrate_constraint", "")).lower()
    shape_family = str(spec.get("canonical_shape_family", "")).lower()
    variants = [str(v).lower() for v in spec.get("variant_required", []) or []]

    # tiny_nn capacity mismatch — audit memo §2.
    if "200" in capacity and "param" in capacity:
        if (
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
            or "1×1×" in text
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
        if not seen:
            if any(
                tok in text
                for tok in ("naive_ptq", "ptq", "qat", "gptq", "awq", "lsq")
            ):
                reasons.append(
                    "VARIANT_PARTIAL_EXHAUSTION: model_spec lists "
                    f"{len(variants)} required variants, evidence row "
                    "appears to test a single variant only"
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
        empirical_bytes = [
            e.empirical_archive_bytes for e in obs
            if e.empirical_archive_bytes is not None
        ]
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
        if empirical_bytes:
            empirical_bytes.sort()
            n = len(empirical_bytes)
            median = empirical_bytes[n // 2]
            explicit_promotable = all(
                _is_explicitly_promotable_evidence(e) for e in obs
            )
            blockers = sorted({
                blocker
                for e in obs
                for blocker in e.dispatch_blockers
            })
            new_t["catalog_prior_bytes"] = t["predicted_archive_bytes"]
            new_t["predicted_archive_bytes"] = median
            new_t["empirical_anchor_n"] = n
            new_t["empirical_anchor_bytes"] = median
            new_t["empirical_anchor_sources"] = [e.source for e in obs if e.source]
            new_t["empirical_anchor_evidence_semantics"] = sorted({
                e.evidence_semantics for e in obs if e.evidence_semantics
            })
            if explicit_promotable and not per_obs_mismatches:
                new_t["evidence_grade"] = f"[empirical-anchor-N{n}]"
                new_t["empirical_anchor_promotable"] = True
            else:
                grade_suffix = "; planning-only"
                if per_obs_mismatches:
                    grade_suffix = "; planning-only; model_spec_mismatch"
                new_t["evidence_grade"] = (
                    f"[empirical-anchor-N{n}{grade_suffix}]"
                )
                new_t["empirical_anchor_promotable"] = False
                new_t["empirical_anchor_score_claim"] = False
                new_t["ready_for_exact_eval_dispatch"] = False
                new_t["rank_or_kill_eligible"] = False
                merged_blockers = list(blockers) if blockers else [
                    "empirical_anchor_not_promotable_without_explicit_exact_eval_custody"
                ]
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
) -> list[dict[str, Any]]:
    """Score each technique by predicted score gain per cost-dollar.

    high-signal filter: techniques with ``predicted_score_delta`` strictly
    below ``min_score_delta`` are dropped (always keeping the top-1 row so
    callers never get an empty list).
    """
    rows: list[dict[str, Any]] = []
    for t in techniques:
        if t["predicted_archive_bytes"] >= current_archive_bytes:
            score_after = current_score
            score_delta = 0.0
        else:
            score_after = _technique_score_after(
                baseline_bytes=current_archive_bytes,
                technique_bytes=t["predicted_archive_bytes"],
                d_seg=d_seg, d_pose=d_pose,
            )
            score_delta = current_score - score_after
        cost_dollars = max(t["cost_dollars"], 0.5)  # floor for pure-CPU items
        gain_per_dollar = score_delta / cost_dollars if cost_dollars > 0 else 0.0
        gap_to_target = (
            score_after - target_score if target_score is not None else None
        )
        rows.append({
            **t,
            "current_score_baseline": current_score,
            "predicted_score_after": score_after,
            "predicted_score_delta": score_delta,
            "gain_per_dollar": gain_per_dollar,
            "gain_per_hour": score_delta / max(t["cost_hours"], 0.1),
            "predicted_gap_to_target": gap_to_target,
            "reaches_target": (
                gap_to_target is not None and gap_to_target <= 0
            ),
        })
    rows.sort(key=lambda r: (-r["predicted_score_delta"], r["cost_dollars"]))
    if min_score_delta > 0.0 and rows:
        kept = [r for r in rows if r["predicted_score_delta"] >= min_score_delta]
        if not kept:
            kept = [rows[0]]  # always preserve top-1
        rows = kept
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
    if prior_evidence:
        enc_catalog = update_catalog_from_evidence(enc_catalog, prior_evidence)
        arch_catalog = update_catalog_from_evidence(arch_catalog, prior_evidence)

    encoder_ranked = _rank_techniques(
        enc_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=min_score_delta,
    )
    arch_ranked = _rank_techniques(
        arch_catalog,
        d_seg=d_seg, d_pose=d_pose,
        current_archive_bytes=archive_bytes,
        current_score=score,
        target_score=target_score,
        min_score_delta=min_score_delta,
    )
    # Combined top-3: best by score-delta across both lists
    combined = sorted(
        encoder_ranked + arch_ranked,
        key=lambda r: (-r["predicted_score_delta"], r["cost_dollars"]),
    )
    top_3 = combined[:3]

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
            f"Top-3 techniques rank by predicted score-delta."
        )
    notes.append(
        "PR101 hand-coded entropy probes are saturated near the 178 KB brotli/AAC band: "
        "the best charged Markov-1 table rerun is 200,219 B, so the currently tested "
        "table-transmission path loses. This is not a global no-ML impossibility proof; "
        "sub-178 KB still needs a new charged coder, a shared decoder model, or lower-entropy "
        "weights. Architecture has 5-10x more predicted headroom."
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
        recommended_top_3=top_3,
        target_score_gap_analysis=gap,
        notes=notes,
        axis_priorities=(
            _maybe_axis_priorities(d_seg, d_pose, archive_bytes, target_score)
            if include_axis_priorities else None
        ) or [],
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

    p_pareto = sub.add_parser("plan-from-pareto",
                              help="Plan for every candidate in 3-axis Pareto JSON")
    p_pareto.add_argument("--pareto-json", type=Path, required=True)
    p_pareto.add_argument("--target-score", type=float, default=None)
    p_pareto.add_argument("--output", type=Path, required=True)
    p_pareto.add_argument("--prior-evidence", type=Path, default=None)
    p_pareto.add_argument("--min-score-delta", type=float, default=0.0)

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
        if args.catalog in ("encoder", "both"):
            out["encoder_catalog"] = update_catalog_from_evidence(ENCODER_TECHNIQUES, ev)
        if args.catalog in ("arch", "both"):
            out["arch_catalog"] = update_catalog_from_evidence(ARCH_TECHNIQUES, ev)
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
    lines.append("TOP-3 RECOMMENDED ACTIONS (ranked by score-delta then cost):")
    for i, rec in enumerate(plan.recommended_top_3, 1):
        lines.append(
            f"  {i}. {rec['name']:<40s}  "
            f"Delta={rec['predicted_score_delta']:+.5f}  "
            f"${rec['cost_dollars']}  ({rec['cost_hours']}h)  "
            f"[{rec['evidence_grade']}]"
        )
        lines.append(f"     -> {rec['description']}")
    lines.append("")
    lines.append("ENCODER TECHNIQUES (top 3 by score-delta):")
    for r in plan.encoder_technique_ranking[:3]:
        lines.append(
            f"  {r['name']:<40s}  bytes={r['predicted_archive_bytes']:,}  "
            f"score={r['predicted_score_after']:.5f}  Delta={r['predicted_score_delta']:+.5f}"
        )
    lines.append("")
    lines.append("ARCHITECTURE TECHNIQUES (top 3 by score-delta):")
    for r in plan.arch_technique_ranking[:3]:
        lines.append(
            f"  {r['name']:<40s}  bytes={r['predicted_archive_bytes']:,}  "
            f"score={r['predicted_score_after']:.5f}  Delta={r['predicted_score_delta']:+.5f}"
        )
    lines.append("")
    lines.append("NOTES:")
    for n in plan.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
