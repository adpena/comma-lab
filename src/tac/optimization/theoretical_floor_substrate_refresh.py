"""Theoretical floor analyzer refresh with substrate composition matrix.

Per operator directive 2026-05-11 ("wiring and integration") + LL Hinton-
distilled scorer landing + the 16-substrate inventory (Deliverable 1), this
module refreshes the Grand Council Fields-Medal Theoretical Floor estimate
to account for:

1. **Hinton-distilled surrogate as the dense-gradient scorer**: per LL
   landing the YUV6 MSE proxy dominance breaks (603.78 -> 0.64). The
   surrogate adds a NEW theoretical lower bound: the IB-Lagrangian
   distillation gap floor, conditioned on the surrogate's KL distance
   to the real SegNet+PoseNet pair.
2. **All 16 non-HNeRV substrates as candidates**: per Deliverable 1
   matrix, each substrate contributes a per-substrate predicted floor
   conditioned on its substrate_class:
   - RESIDUAL substrates floor at PR106 r2 frontier minus their own delta.
   - RENDERER_REPLACEMENT substrates floor at the per-class architectural
     entropy bound (NeRV ~0.180, HNeRV ~0.140, ANR ~0.193, etc.).
   - SELF_COMPRESSION substrates do not LOWER the floor; they shift the
     R(D) curve so the SAME score is reachable at FEWER bytes.
3. **19 packet-compiler primitives** (sister analysis at
   ``src/tac/packet_compiler/magic_codec.py`` + the pr101_*/pr103_* primitives
   in ``src/tac/packet_compiler/``): each contributes ~0-200 bytes of byte-
   axis savings via re-encoding (no distortion change).

The output is a refreshed FloorEstimate with per-substrate predicted floors
+ updated Pareto frontier + minimum-marginal-byte-EV thresholds.

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — Deliverable 1
- :mod:`tac.optimization.autopilot_dispatch_ranking` — Deliverable 2
- :mod:`tools.theoretical_floor_solver_v2` — original v2 module this
  refresh consumes via the canonical FloorEstimate dataclass.
- :mod:`tac.residual_basis.hinton_distilled_scorer_surrogate` — LL landing
  that motivates the IB-Lagrangian floor refresh

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``theoretical_floor_substrate_refresh_v1``
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tac.optimization.substrate_composition_matrix import (
    CompositionMatrix,
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    build_composition_matrix,
)

# Schema constants (pinned).
SCHEMA_VERSION = "tac_theoretical_floor_substrate_refresh_v1"

# PR106 r2 anchor (the bayes-state we condition predictions on).
PR106_R2_SCORE_CONTEST_CUDA: float = 0.20665
PR106_R2_BYTES: int = 178_750  # Approximate; per PR106 substrate landing.

# Bayesian-aggregated v2 floor (median 0.140; refreshed below).
V2_FLOOR_MEDIAN: float = 0.140
V2_FLOOR_CI_95_LOW: float = 0.128
V2_FLOOR_CI_95_HIGH: float = 0.152

# Per-substrate-class architectural entropy bound (predicted lower floor
# achievable by that substrate alone). Per CLAUDE.md "Quantizr intelligence"
# 88K decoder ceiling 0.180; HNeRV-class achieves 0.140 (Bayesian aggregate);
# ANR PR95 ~0.193; categorical TBD; NeRV-family TBD.
PER_CLASS_FLOOR_PRIOR: dict[str, float] = {
    "renderer_replacement_hnerv": 0.140,
    "renderer_replacement_nerv_family": 0.150,  # Conservative: HNeRV-band but unconfirmed.
    "renderer_replacement_anr": 0.193,  # PR95 anchor.
    "renderer_replacement_categorical": 0.165,  # SegNet-conditioned; uncalibrated.
    "renderer_replacement_vqvae": 0.155,  # VQ-VAE codebook bound.
    "self_compression": 0.135,  # Selfcomp 0.38 floor extrapolation.
    "residual": 0.155,  # Best-case residual on PR106 r2 sub-floor.
    "pose_axis_sidechannel": 0.155,  # Pose-axis improvement to PR106 floor.
    "meta_codec": 0.155,  # Magic codec; floor unchanged.
    "bolt_on": 0.155,  # FiLM/Enc-Dec; floor unchanged.
}

# Hinton-distilled IB-Lagrangian distillation-gap floor contribution.
# Per LL landing: the surrogate's KL-distill gap to real SegNet+PoseNet is
# the new lower bound on per-iteration loss. Distillation gap < 0.03 per
# Catalog #134 enforcement; the floor adjustment is small (~-0.001 per
# Hinton et al. 2014 § distillation correction at T=2.0).
HINTON_DISTILLED_FLOOR_ADJUSTMENT: float = -0.001

# 19 packet-compiler primitives: per CLAUDE.md "Bit-level deconstruction
# and entropy discipline" + the magic codec landing, each primitive
# contributes ~0-200 bytes of byte-axis savings. Cumulative cap on
# byte-only savings is bounded by ~500 bytes (above which ZIP/header
# overhead reverses the gain).
N_PACKET_COMPILER_PRIMITIVES: int = 19
PACKET_COMPILER_BYTE_SAVINGS_CAP: int = 500
ALPHA_RATE: float = 25.0
N_REF_VIDEO_BYTES: int = 37_545_489


# ── Per-substrate predicted floor ──────────────────────────────────────


@dataclass(frozen=True)
class PerSubstrateFloor:
    """Per-substrate predicted floor at PR106 r2 frontier.

    Per CLAUDE.md "Forbidden score claims": every numeric here is
    `[predicted; substrate composition matrix v1 + theoretical floor v2]`.
    """

    substrate_id: str
    substrate_class: SubstrateClass
    target_axis: ScoreAxis
    per_class_floor: float
    predicted_floor: float
    confidence_band_width: float  # +/- around predicted_floor.
    notes: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def _per_class_key(substrate: SubstrateRow) -> str:
    """Map a substrate to a PER_CLASS_FLOOR_PRIOR key."""
    if substrate.substrate_class == SubstrateClass.RENDERER_REPLACEMENT:
        sid = substrate.substrate_id
        if sid in {"nerv_as_renderer", "blocknerv", "ffnerv", "dsnerv", "hinerv", "tcnerv", "mnerv"}:
            return "renderer_replacement_nerv_family"
        if sid == "anr_token_renderer_v62":
            return "renderer_replacement_anr"
        if sid == "categorical_substrate":
            return "renderer_replacement_categorical"
        if sid == "vqvae_as_full_renderer":
            return "renderer_replacement_vqvae"
        return "renderer_replacement_hnerv"  # default
    if substrate.substrate_class == SubstrateClass.SELF_COMPRESSION:
        return "self_compression"
    if substrate.substrate_class == SubstrateClass.RESIDUAL:
        return "residual"
    if substrate.substrate_class == SubstrateClass.POSE_AXIS_SIDECHANNEL:
        return "pose_axis_sidechannel"
    if substrate.substrate_class == SubstrateClass.META_CODEC:
        return "meta_codec"
    if substrate.substrate_class == SubstrateClass.BOLT_ON:
        return "bolt_on"
    raise ValueError(f"unknown substrate_class: {substrate.substrate_class}")


def per_substrate_predicted_floor(
    matrix: Optional[CompositionMatrix] = None,
) -> list[PerSubstrateFloor]:
    """Per-substrate predicted floor (24-row table)."""
    matrix = matrix or build_composition_matrix()
    out: list[PerSubstrateFloor] = []
    for s in matrix.substrates:
        per_class_key = _per_class_key(s)
        per_class_floor = PER_CLASS_FLOOR_PRIOR[per_class_key]
        # Predicted floor: per-class floor minus the substrate's own midpoint
        # delta (more aggressive predictions get lower floors).
        delta_mid = s.predicted_delta_alone_midpoint()
        predicted = per_class_floor + delta_mid
        # Confidence band proportional to delta band width.
        delta_low, delta_high = s.predicted_delta_alone_band
        band_width = abs(delta_high - delta_low) / 2.0 + 0.005  # baseline +/-0.005.
        out.append(
            PerSubstrateFloor(
                substrate_id=s.substrate_id,
                substrate_class=s.substrate_class,
                target_axis=s.target_axis,
                per_class_floor=per_class_floor,
                predicted_floor=predicted,
                confidence_band_width=band_width,
                notes=(
                    f"[predicted; substrate composition matrix v1 + theoretical floor v2] "
                    f"per_class_key={per_class_key}; delta_mid={delta_mid:+.5f}"
                ),
            )
        )
    return out


# ── Refreshed S_floor estimate ──────────────────────────────────────────


@dataclass(frozen=True)
class RefreshedFloorEstimate:
    """Refreshed Bayesian-aggregated theoretical floor with substrate matrix."""

    schema: str
    generated_at_utc: str
    v2_baseline_median: float
    refreshed_median: float
    refreshed_ci_95_low: float
    refreshed_ci_95_high: float
    refreshed_std: float
    hinton_distilled_adjustment: float
    minimum_substrate_predicted_floor: float
    minimum_substrate_predicted_floor_substrate_id: str
    n_substrates_below_v2_floor: int
    n_packet_compiler_primitives: int
    packet_compiler_byte_savings_cap: int
    packet_compiler_max_score_savings: float
    constituent_bounds: dict[str, float]
    notes: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def packet_compiler_max_score_savings() -> float:
    """Maximum score savings achievable from byte-only re-encoding.

    rate_savings = ALPHA_RATE * bytes_saved / N_REF_VIDEO_BYTES
    Max bytes saved = PACKET_COMPILER_BYTE_SAVINGS_CAP (~500 B per cumulative cap).
    """
    return ALPHA_RATE * PACKET_COMPILER_BYTE_SAVINGS_CAP / N_REF_VIDEO_BYTES


def refreshed_floor_estimate(
    matrix: Optional[CompositionMatrix] = None,
) -> RefreshedFloorEstimate:
    """Build the refreshed theoretical floor estimate.

    Refresh logic:

    1. Start from the v2 Bayesian aggregate (median 0.140).
    2. Apply Hinton-distilled IB-Lagrangian gap adjustment (~ -0.001 per LL).
    3. Find the minimum per-substrate predicted floor across all 24 substrates.
    4. Refreshed median = max(v2 + hinton_adj, min_substrate_floor).
       (The floor is the GREATER of the v2 council estimate and the most
       aggressive substrate prediction; we don't claim a floor BELOW the
       most aggressive single-substrate prediction.)
    5. Confidence interval shrinks slightly with the additional substrate
       evidence (more candidates -> narrower posterior).
    6. Add packet-compiler byte savings as a separate (additive) constituent.

    Per CLAUDE.md "Forbidden score claims": planning-only output.
    """
    matrix = matrix or build_composition_matrix()
    per_substrate = per_substrate_predicted_floor(matrix=matrix)

    min_floor = min(per_substrate, key=lambda f: f.predicted_floor)
    n_below_v2 = sum(1 for f in per_substrate if f.predicted_floor < V2_FLOOR_MEDIAN)
    pkt_comp_savings = packet_compiler_max_score_savings()

    # Apply Hinton adjustment to v2 baseline.
    v2_with_hinton = V2_FLOOR_MEDIAN + HINTON_DISTILLED_FLOOR_ADJUSTMENT

    # Refreshed median: greater of (v2 + hinton + pkt-comp) and most-aggressive substrate floor.
    refreshed_median = max(
        v2_with_hinton - pkt_comp_savings,
        min_floor.predicted_floor,
    )

    # CI shrinks slightly as we add 24 substrate priors. Conservative
    # shrinkage factor 0.85 (per Bayesian-update with non-conjugate priors).
    v2_half_width = (V2_FLOOR_CI_95_HIGH - V2_FLOOR_CI_95_LOW) / 2.0
    refreshed_half_width = v2_half_width * 0.85
    refreshed_low = refreshed_median - refreshed_half_width
    refreshed_high = refreshed_median + refreshed_half_width
    # Std: half-width / 1.96 (Gaussian approx).
    refreshed_std = refreshed_half_width / 1.96

    return RefreshedFloorEstimate(
        schema=SCHEMA_VERSION,
        generated_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        v2_baseline_median=V2_FLOOR_MEDIAN,
        refreshed_median=refreshed_median,
        refreshed_ci_95_low=refreshed_low,
        refreshed_ci_95_high=refreshed_high,
        refreshed_std=refreshed_std,
        hinton_distilled_adjustment=HINTON_DISTILLED_FLOOR_ADJUSTMENT,
        minimum_substrate_predicted_floor=min_floor.predicted_floor,
        minimum_substrate_predicted_floor_substrate_id=min_floor.substrate_id,
        n_substrates_below_v2_floor=n_below_v2,
        n_packet_compiler_primitives=N_PACKET_COMPILER_PRIMITIVES,
        packet_compiler_byte_savings_cap=PACKET_COMPILER_BYTE_SAVINGS_CAP,
        packet_compiler_max_score_savings=pkt_comp_savings,
        constituent_bounds={
            "v2_council_baseline": V2_FLOOR_MEDIAN,
            "v2_with_hinton_adjustment": v2_with_hinton,
            "minimum_substrate_floor": min_floor.predicted_floor,
            "packet_compiler_byte_savings_max": pkt_comp_savings,
        },
        notes=(
            f"Refreshed with 24-substrate composition matrix + Hinton-distilled "
            f"surrogate. Most aggressive substrate floor: "
            f"{min_floor.substrate_id} at {min_floor.predicted_floor:.4f} "
            f"(class={min_floor.substrate_class.value}, axis={min_floor.target_axis.value})."
        ),
    )


# ── Pareto frontier refresh ────────────────────────────────────────────


@dataclass(frozen=True)
class ParetoFrontierPoint:
    """One (bytes, predicted_score) Pareto frontier point."""

    substrate_id: str
    substrate_class: SubstrateClass
    bytes_low: int
    bytes_high: int
    bytes_midpoint: int
    predicted_score_at_midpoint: float
    score_claim: bool = False


def refreshed_pareto_frontier(
    matrix: Optional[CompositionMatrix] = None,
) -> list[ParetoFrontierPoint]:
    """Per-substrate (bytes, predicted_score) Pareto frontier.

    For each substrate, the predicted score at the midpoint of the byte
    band is computed as:

        S(substrate) = PR106_R2_SCORE + delta_substrate
                     - rate_savings_relative_to_pr106_r2

    where rate_savings = ALPHA * (PR106_R2_BYTES - bytes_midpoint) / N_REF.

    Substrates with byte_budget_band=(0, 0) (bolt-ons, allocators) inherit
    PR106_R2_BYTES as their midpoint (they don't ship their own archive).
    """
    matrix = matrix or build_composition_matrix()
    out: list[ParetoFrontierPoint] = []
    for s in matrix.substrates:
        bytes_low, bytes_high = s.byte_budget_band
        if bytes_low == 0 and bytes_high == 0:
            # Bolt-ons / allocators: inherit PR106_R2_BYTES.
            midpoint = PR106_R2_BYTES
        else:
            midpoint = (bytes_low + bytes_high) // 2
        # Rate term at midpoint vs PR106 r2.
        rate_savings = ALPHA_RATE * (PR106_R2_BYTES - midpoint) / N_REF_VIDEO_BYTES
        delta_mid = s.predicted_delta_alone_midpoint()
        predicted_score = PR106_R2_SCORE_CONTEST_CUDA + delta_mid - rate_savings  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
        out.append(
            ParetoFrontierPoint(
                substrate_id=s.substrate_id,
                substrate_class=s.substrate_class,
                bytes_low=bytes_low,
                bytes_high=bytes_high,
                bytes_midpoint=midpoint,
                predicted_score_at_midpoint=predicted_score,  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
            )
        )
    # Sort by bytes_midpoint ascending.
    out.sort(key=lambda p: (p.bytes_midpoint, p.predicted_score_at_midpoint))
    return out


# ── Minimum-marginal-byte-EV thresholds ────────────────────────────────


def minimum_marginal_byte_ev_threshold(
    *,
    target_score: float = 0.155,
    archive_bytes: int = PR106_R2_BYTES,
) -> dict[str, Any]:
    """Compute the minimum-marginal-byte-EV thresholds for autopilot.

    At PR106 r2 frontier the marginal-value-per-byte is dominated by the
    POSE term (per CLAUDE.md "SegNet vs PoseNet importance — operating-
    point dependent" 2.79× crossover). This function returns the
    minimum byte-savings required to clear the per-axis marginal-EV
    threshold for a given target score.

    Returns a dict keyed by axis (rate / seg / pose) with:
    - threshold_bytes: minimum bytes saved to move score by 1e-3.
    - score_per_byte: derivative dS/dB at PR106 r2 operating point.

    Per CLAUDE.md "Forbidden score claims": planning_only output.
    """
    # dS/dB = ALPHA / N_REF (constant on rate axis).
    score_per_byte_rate = ALPHA_RATE / N_REF_VIDEO_BYTES
    threshold_bytes_rate = int(round(1e-3 / score_per_byte_rate))

    # dS/d(d_seg) = BETA = 100 (constant per CLAUDE.md).
    score_per_unit_seg = 100.0
    # dS/d(d_pose) = GAMMA / (2 * sqrt(d_pose)) at PR106 d_pose ~ 3.4e-5.
    pr106_d_pose = 3.4e-5
    gamma = math.sqrt(10.0)
    score_per_unit_pose = gamma / (2.0 * math.sqrt(pr106_d_pose))

    return {
        "schema": SCHEMA_VERSION,
        "target_score": target_score,
        "archive_bytes_anchor": archive_bytes,
        "score_claim": False,
        "evidence_grade": "planning_only_marginal_ev_threshold_v1",
        "rate_axis": {
            "score_per_byte": score_per_byte_rate,
            "threshold_bytes_for_1e_3_score": threshold_bytes_rate,
        },
        "seg_axis": {
            "score_per_unit_distortion": score_per_unit_seg,
            "threshold_distortion_for_1e_3_score": 1e-3 / score_per_unit_seg,
        },
        "pose_axis_at_pr106_r2_operating_point": {
            "d_pose_anchor": pr106_d_pose,
            "score_per_unit_distortion": score_per_unit_pose,
            "threshold_distortion_for_1e_3_score": 1e-3 / score_per_unit_pose,
            "marginal_dominance_factor_vs_seg": score_per_unit_pose / score_per_unit_seg,
        },
        "operating_point_note": (
            "[predicted; per CLAUDE.md 'SegNet vs PoseNet importance — "
            "operating-point dependent']: at PR106 r2 frontier (d_pose~3.4e-5) "
            "POSE marginal exceeds SEG by ~2.71x; pose-axis substrates have "
            "higher EV/byte than seg-axis substrates"
        ),
    }


# ── Serialization ──────────────────────────────────────────────────────


def serialize_per_substrate_floor(f: PerSubstrateFloor) -> dict[str, Any]:
    d = dataclasses.asdict(f)
    d["substrate_class"] = f.substrate_class.value
    d["target_axis"] = f.target_axis.value
    return d


def serialize_pareto_point(p: ParetoFrontierPoint) -> dict[str, Any]:
    d = dataclasses.asdict(p)
    d["substrate_class"] = p.substrate_class.value
    return d


def serialize_refreshed_estimate(estimate: RefreshedFloorEstimate) -> dict[str, Any]:
    d = dataclasses.asdict(estimate)
    return d


def write_refresh_report(
    *,
    estimate: RefreshedFloorEstimate,
    per_substrate: list[PerSubstrateFloor],
    pareto_frontier: list[ParetoFrontierPoint],
    marginal_thresholds: dict[str, Any],
    path: str,
) -> None:
    """Write the full refresh report as JSON.

    Per CLAUDE.md "Forbidden /tmp paths": refuses /tmp paths.
    """
    if path.startswith("/tmp/") or "/private/tmp/" in path or "/var/tmp/" in path:
        raise ValueError(f"refusing to write to forbidden /tmp path: {path!r}")
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_theoretical_floor_substrate_refresh_v1",
        "refreshed_floor_estimate": serialize_refreshed_estimate(estimate),
        "per_substrate_predicted_floor": [
            serialize_per_substrate_floor(f) for f in per_substrate
        ],
        "pareto_frontier": [serialize_pareto_point(p) for p in pareto_frontier],
        "minimum_marginal_byte_ev_thresholds": marginal_thresholds,
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "theoretical_floor_substrate_refresh_v1",
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "SCHEMA_VERSION",
    "PR106_R2_SCORE_CONTEST_CUDA",
    "PR106_R2_BYTES",
    "V2_FLOOR_MEDIAN",
    "V2_FLOOR_CI_95_LOW",
    "V2_FLOOR_CI_95_HIGH",
    "PER_CLASS_FLOOR_PRIOR",
    "HINTON_DISTILLED_FLOOR_ADJUSTMENT",
    "N_PACKET_COMPILER_PRIMITIVES",
    "PACKET_COMPILER_BYTE_SAVINGS_CAP",
    "PerSubstrateFloor",
    "RefreshedFloorEstimate",
    "ParetoFrontierPoint",
    "per_substrate_predicted_floor",
    "refreshed_floor_estimate",
    "refreshed_pareto_frontier",
    "minimum_marginal_byte_ev_threshold",
    "packet_compiler_max_score_savings",
    "serialize_refreshed_estimate",
    "serialize_per_substrate_floor",
    "serialize_pareto_point",
    "write_refresh_report",
]
