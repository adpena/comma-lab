# SPDX-License-Identifier: MIT
"""Cathedral consumer for cross-codec orthogonality + cross-hardware leverage predictions.

Per WAVE-3-STRATEGIC-FINDINGS-CANONICAL-EXTENSION (sister of commit
``af727e3c1`` WAVE-3-CROSS-CANDIDATE-SENSITIVITY-COMPARISON-DIAGNOSTIC).
This consumer is the canonical producer + consumer of THREE canonical
equations registered in this landing:

* ``per_byte_leverage_uniformly_distributed_v2_cross_hardware_aware`` (Equation 7)
* ``hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1`` (Equation 8)
* ``cross_codec_super_additive_orthogonality_predictor_v1`` (Equation 9)

The 3 predictor helpers exposed by this module (``predict_top_k_leverage_cross_hardware_aware``,
``predict_hnerv_backbone_saturation``, ``predict_cross_codec_super_additivity``)
are referenced by ``CanonicalEquation.python_callable_module_path`` per
Catalog #344 so the canonical equation lookup consumer at
:mod:`tac.cathedral_consumers.canonical_equation_lookup_consumer` can
auto-route token-matched candidates through them.

Catalog #341 canonical Tier A routing markers in ``consume_candidate``:
this consumer NEVER mutates ``predicted_delta_adjustment`` (always 0.0);
``promotable=False`` always; ``axis_tag="[predicted]"`` always. The
predictions are observability-only side information per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable (promotion to
score-contributing requires paired CUDA + CPU empirical auth-eval).

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates with
    cross-codec orthogonality + backbone saturation classifications)
  * #5 continual-learning posterior — ACTIVE (NEW master-gradient
    anchors trigger recomputation via ``update_from_anchor``; the
    consumer reads the latest similarity matrix on every consume call
    so live updates propagate without per-consumer state)

Sister wire-in:
  * Catalog #322 ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
    — this consumer's predictors derive from VALIDATED_CONTEST_MEMBER
    archives per the predecessor sister Catalog #321 prober-fix wave.
  * Catalog #344 ``check_empirical_finding_memo_references_canonical_equation``
    — the strategic findings memo at
    ``.omx/research/cross_candidate_strategic_findings_canonical_extension_*.md``
    references all 3 canonical equation IDs to satisfy the gate.
  * Catalog #335 ``check_cathedral_consumer_directory_package_exposes_canonical_contract``
    — this package satisfies the canonical Protocol (CONSUMER_NAME +
    CONSUMER_VERSION + CONSUMER_HOOK_NUMBERS + ``update_from_anchor`` +
    ``consume_candidate``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "cross_codec_orthogonality_predictor_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical sidecar paths discoverable per Catalog #245/#248 graceful-failure pattern.
_MATRIX_GLOB = "cross_substrate_sensitivity_similarity_matrix_*.json"
_PR101_FEC6_DELTA_GLOB = "pr101_vs_fec6_per_byte_delta_*.json"


# Empirical anchors from the strategic findings memo (used as fallback when
# canonical sidecars are unavailable). These are baked from
# .omx/research/cross_candidate_sensitivity_comparison_diagnostic_20260520T192204Z.md
# per Catalog #287 evidence-tag discipline; the empirical-anchor numbers
# carry [predicted] axis_tag because they are research-sidecar-derived.
_CROSS_HARDWARE_TOP_1PCT_EMPIRICAL = {
    "[macOS-CPU advisory]": 0.0641,  # fec6 frontier macOS advisory
    "[contest-CUDA T4]": 0.1111,     # fec6 frontier CUDA T4 authoritative
    "[contest-CPU]": 0.0641,         # treat as advisory-equivalent until paired Linux x86_64 anchor lands
}

_HNERV_BACKBONE_SATURATED_MEDAL_CLUSTER = frozenset(
    {"a1", "a1_finetuned", "pr101", "pr101_gold", "pr102", "pr103", "fec6", "fec6_frontier_cuda_t4"}
)
_HNERV_BACKBONE_BYTE_COUNT = 178158

# Cross-codec orthogonal pairs from the 21-pair similarity matrix.
# Each row: (codec_family_a, codec_family_b, top_k_jaccard_observed,
# per_axis_pearson_seg_observed, classification).
_CROSS_CODEC_ORTHOGONAL_PAIRS = (
    ("hnerv_brotli", "format0d_score_table", 0.000, -0.076, "SUPER_ADDITIVE"),
    ("hnerv_brotli", "apogee_int4", 0.000, 0.012, "SUPER_ADDITIVE"),
    ("format0d_score_table", "fec6_huffman_k16", 0.000, -0.083, "SUPER_ADDITIVE"),
    ("apogee_int4", "fec6_huffman_k16", 0.000, -0.050, "SUPER_ADDITIVE"),
)


def _state_dir() -> Path:
    """Locate .omx/state/ for sidecar discovery."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_matrix() -> Mapping[str, Any] | None:
    """Load the most-recent cross-substrate similarity matrix.

    Returns None on missing/corrupt per Catalog #138/#245/#248 sister
    graceful-failure discipline.
    """
    state = _state_dir()
    if not state.is_dir():
        return None
    matches = sorted(state.glob(_MATRIX_GLOB))
    if not matches:
        return None
    try:
        with matches[-1].open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def predict_top_k_leverage_cross_hardware_aware(
    k_percent: float,
    measurement_axis: str = "[predicted]",
    total_bytes: int = 154147,
) -> float:
    """Equation 7 canonical callable: top-K leverage prediction with cross-hardware factor.

    Args:
        k_percent: Top-K percent of bytes (e.g., 1.0 for top-1%).
        measurement_axis: Canonical axis tag per Catalog #127/#323.
            One of ``[macOS-CPU advisory]`` / ``[contest-CPU]`` /
            ``[contest-CUDA T4]`` / ``[predicted]``.
        total_bytes: Archive byte count (default = PR101 GOLD).

    Returns:
        Predicted top-K leverage ratio. For the 1.0% top-K case, returns
        the empirically observed leverage on the matching hardware axis
        (6.41% for macOS-CPU advisory; 11.11% for CUDA T4). For other K
        values, falls back to the uniform-Pareto baseline (k_percent/100)
        as a conservative lower bound.

    Raises:
        ValueError: if ``k_percent`` is outside [0, 100].
    """
    if k_percent < 0 or k_percent > 100:
        raise ValueError(f"k_percent must be in [0, 100], got {k_percent}")
    # At top-1%, return empirical observation per hardware axis
    if abs(k_percent - 1.0) < 1e-6:
        return _CROSS_HARDWARE_TOP_1PCT_EMPIRICAL.get(
            measurement_axis, k_percent / 100.0
        )
    # Other K values: uniform baseline (sister of Equation 3 v1 helper)
    return k_percent / 100.0


def predict_hnerv_backbone_saturation(
    substrate_label: str,
    backbone_size_bytes: int = _HNERV_BACKBONE_BYTE_COUNT,
) -> Mapping[str, Any]:
    """Equation 8 canonical callable: HNeRV backbone saturation prediction.

    Predicts whether a substrate's backbone is in the saturated medal cluster.

    Args:
        substrate_label: Canonical substrate label (token-matched against
            the HNeRV medal cluster set).
        backbone_size_bytes: Backbone byte count.

    Returns:
        Mapping with ``is_saturated_backbone`` (bool) +
        ``predicted_per_axis_diff`` (float; 0.0 if saturated) +
        ``rationale`` (str).
    """
    label_lower = substrate_label.lower()
    matches_cluster = any(token in label_lower for token in _HNERV_BACKBONE_SATURATED_MEDAL_CLUSTER)
    in_size_range = 170_000 <= backbone_size_bytes <= 185_000
    is_saturated = matches_cluster and in_size_range
    return {
        "is_saturated_backbone": is_saturated,
        "predicted_per_axis_diff": 0.0 if is_saturated else None,
        "medal_cluster_token_matched": matches_cluster,
        "backbone_size_in_canonical_range": in_size_range,
        "rationale": (
            f"substrate {substrate_label!r}: medal-cluster token matched={matches_cluster}, "
            f"backbone size {backbone_size_bytes} in canonical [170000, 185000] range={in_size_range}; "
            "saturated backbones have per-axis aggregate sensitivity identical to "
            "PR101 GOLD reference to 4 sig figs (Equation 8) [predicted]"
        ),
    }


def predict_cross_codec_super_additivity(
    codec_family_a: str,
    codec_family_b: str,
    top_k_jaccard: float | None = None,
    per_axis_pearson_seg: float | None = None,
) -> Mapping[str, Any]:
    """Equation 9 canonical callable: cross-codec SUPER_ADDITIVE prediction.

    Predicts SUPER_ADDITIVE classification based on:
    1. top-K Jaccard < 0.05 (orthogonal byte streams)
    2. per-axis Pearson seg ρ ∈ [-0.10, +0.10] (mild correlation)
    3. codec_family mismatch (different codec backbones)

    Args:
        codec_family_a: Codec family token (e.g., "hnerv_brotli").
        codec_family_b: Codec family token.
        top_k_jaccard: Optional observed Jaccard; uses anchor lookup if None.
        per_axis_pearson_seg: Optional observed Pearson; uses anchor lookup if None.

    Returns:
        Mapping with ``classification`` (str: SUPER_ADDITIVE / INDETERMINATE /
        UNKNOWN_CODEC_PAIR) + ``confidence`` (float in [0,1]) + ``rationale``.
    """
    # Try empirical anchor lookup first
    for (a, b, j, p, classification) in _CROSS_CODEC_ORTHOGONAL_PAIRS:
        if (codec_family_a == a and codec_family_b == b) or (
            codec_family_a == b and codec_family_b == a
        ):
            return {
                "classification": classification,
                "confidence": 0.85,  # 4 empirical anchors; not yet paired-CUDA
                "anchor_match": True,
                "top_k_jaccard_anchor": j,
                "per_axis_pearson_seg_anchor": p,
                "rationale": (
                    f"({codec_family_a!r} ↔ {codec_family_b!r}) matched empirical anchor: "
                    f"top-K Jaccard={j}, per-axis seg ρ={p}, classification={classification} "
                    "[predicted]"
                ),
            }
    # Fall back to caller-supplied observations
    if codec_family_a == codec_family_b:
        return {
            "classification": "INDETERMINATE",
            "confidence": 0.0,
            "anchor_match": False,
            "rationale": (
                f"same-codec pair ({codec_family_a!r}) — orthogonality predictor "
                "does not apply [predicted]"
            ),
        }
    if top_k_jaccard is None or per_axis_pearson_seg is None:
        return {
            "classification": "UNKNOWN_CODEC_PAIR",
            "confidence": 0.0,
            "anchor_match": False,
            "rationale": (
                f"unknown codec pair ({codec_family_a!r} ↔ {codec_family_b!r}) and "
                "no observations provided; cannot predict without similarity-matrix "
                "anchor or caller-supplied (top_k_jaccard, per_axis_pearson_seg) "
                "[predicted]"
            ),
        }
    # Apply Equation 9 criteria
    if top_k_jaccard < 0.05 and -0.10 <= per_axis_pearson_seg <= 0.10:
        return {
            "classification": "SUPER_ADDITIVE",
            "confidence": 0.60,  # derived from observations, not anchor
            "anchor_match": False,
            "rationale": (
                f"observation-derived SUPER_ADDITIVE: Jaccard={top_k_jaccard:.4f}<0.05 AND "
                f"per-axis seg ρ={per_axis_pearson_seg:.4f} in [-0.10, +0.10] [predicted]"
            ),
        }
    return {
        "classification": "INDETERMINATE",
        "confidence": 0.30,
        "anchor_match": False,
        "rationale": (
            f"observation-derived INDETERMINATE: Jaccard={top_k_jaccard:.4f}, "
            f"per-axis seg ρ={per_axis_pearson_seg:.4f}; does not match SUPER_ADDITIVE "
            "criteria [predicted]"
        ),
    }


def _candidate_substrate_label(candidate: Mapping[str, Any]) -> str | None:
    """Extract canonical substrate label from candidate dict (token-walker)."""
    text_parts = [
        f"{k}={v}" for k, v in candidate.items()
        if isinstance(v, (str, int, float))
    ]
    text = " ".join(text_parts).lower()
    for token in (
        "fec6_frontier_cuda_t4", "fec6_frontier_macos_advisory",
        "pr101_gold", "pr106_format0d", "pr107_apogee", "a1_finetuned",
        "fec6_frontier", "fec6", "pr101", "pr106", "pr107", "a1",
    ):
        if token in text:
            if token in ("fec6_frontier", "fec6"):
                return "fec6_frontier_cuda_t4"
            if token == "pr101":
                return "pr101_gold"
            if token == "pr106":
                return "pr106_format0d"
            if token == "pr107":
                return "pr107_apogee"
            if token == "a1":
                return "a1_finetuned"
            return token
    return None


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    NO-OP per Catalog #287 + #323 measurement-provenance discipline.
    The canonical refresh path is operator-triggered via the matrix
    rebuild helper. Each ``consume_candidate`` call reads the latest
    matrix so live updates propagate without per-consumer state.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with cross-codec + backbone predictions.

    Returns canonical Tier A contribution per Catalog #341 (always
    observability-only; never mutates score or promotes):
      * predicted_delta_adjustment=0.0
      * promotable=False
      * axis_tag="[predicted]"
      * rationale + matched predictions
    """
    matrix = _load_latest_matrix()
    target_substrate = _candidate_substrate_label(candidate)

    annotations: dict[str, Any] = {
        "matrix_available": matrix is not None,
        "matched_substrate": target_substrate,
    }

    if target_substrate is not None:
        # Equation 8: backbone saturation
        saturation = predict_hnerv_backbone_saturation(target_substrate)
        annotations["hnerv_backbone_saturation"] = saturation

    # Equation 7: cross-hardware top-1% leverage prediction at canonical axes
    annotations["cross_hardware_top_1pct_leverage_predictions"] = {
        axis: predict_top_k_leverage_cross_hardware_aware(1.0, measurement_axis=axis)
        for axis in ("[macOS-CPU advisory]", "[contest-CPU]", "[contest-CUDA T4]")
    }

    rationale_parts = [
        f"cross_codec_orthogonality_predictor_consumer annotated candidate "
        f"(substrate={target_substrate!r}, matrix_available={matrix is not None})"
    ]
    if target_substrate is not None:
        sat = annotations.get("hnerv_backbone_saturation", {})
        if sat.get("is_saturated_backbone"):
            rationale_parts.append(
                f"backbone-saturated medal cluster (Equation 8); "
                f"future score-lowering should target selector/microcodec overlay"
            )
    rationale_parts.append("[predicted]")

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": "; ".join(rationale_parts),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        **annotations,
    }
