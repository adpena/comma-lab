# SPDX-License-Identifier: MIT
"""Catalog #324-grade predicted-band axis-attribution discipline (Path 3 C' Phase 3 — UNWIND cargo-cult #12).

Per Path 3 C' Phase 2 substrate-design decision memo
(``.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md``
commit ``bac0ec05d``) Section 3d.

**Cargo-cult #12 (CARGO-CULTED-CRITICAL)**: the existing v8 substrate
design memo (``.omx/research/nscs06_v8_chroma_lut_design_20260521.md``)
declares ``predicted_band = -0.002706`` per canonical equation #26 closed
form. But this is RATE-AXIS ONLY; the SEG and POSE axis contributions are
UNMODELED. The 6 prior v8 Modal dispatches (rc=22/rc=1) are receipts that
the unmodeled seg+pose terms can DOMINATE the rate-axis savings (worst
case: v6 score 105.15 ≫ rate-axis 1.96).

Per Catalog #324 ``check_no_predicted_band_without_post_training_tier_c_validation``:
a predicted band must EITHER be validated post-training Tier-C OR explicitly
declared as ``pending_post_training`` with reactivation criteria pinned.
The existing v8 substrate recipe correctly declares
``predicted_band_validation_status: pending_post_training`` per the design
memo frontmatter, but the substrate-side helper does NOT explicitly DECOMPOSE
the prediction by axis with UNKNOWN tags on seg+pose.

This module provides the canonical axis-attribution helper. The function
:func:`predicted_delta_s_with_axis_attribution` returns:

    {
        "rate_axis": -0.002706,             # canonical equation #26 closed form
        "seg_axis": "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324",
        "pose_axis": "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324",
        "total_axis": "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324",
    }

This makes the axis-attribution EXPLICIT at every consumer surface (autopilot
ranker, lane registry evidence, smoke metadata JSON). The operator-eye
prediction is now correctly bounded: the rate-axis savings is the only
component with a closed-form prediction; the seg+pose components require
empirical paired-smoke per Catalog #324 reactivation criteria.

CLAUDE.md compliance:
- Catalog #324 predicted-band-from-incomplete-Tier-C self-protect anchor
- Catalog #287 + #323 canonical Provenance (axis-tagged at every layer)
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW module; no
  mutation of sister `procedural_variant.py::predicted_delta_s` which
  remains the canonical rate-axis-only predictor)
- Catalog #297 + #220 sister extinction at the prediction surface
  (rate-axis only does not satisfy operational-mechanism-contract; the
  axis-attribution discipline structurally surfaces the gap)

6-hook wire-in declaration per Catalog #125:
* hook #1 sensitivity-map = ACTIVE (per-axis decomposition IS canonical
  sensitivity surface per Catalog #356; downstream consumers can route
  rate-axis through Pareto polytope while seg+pose route through UNKNOWN
  posterior-pending channel)
* hook #2 Pareto constraint = ACTIVE PRIMARY (axis-decomposition enables
  Dykstra alternating projections on rate+seg+pose constraints)
* hook #3 bit-allocator = N/A (predictor; no bit allocation)
* hook #4 cathedral autopilot dispatch = ACTIVE (ranker consumes axis-tagged
  prediction; UNKNOWN seg+pose tags signal high-uncertainty operating point)
* hook #5 continual-learning posterior = ACTIVE PRIMARY (first paired-smoke
  empirical anchor populates seg+pose; replaces UNKNOWN tags with measured
  values per Catalog #324 post-training Tier-C validation discipline)
* hook #6 probe-disambiguator = ACTIVE (axis-attribution IS the canonical
  disambiguator between rate-axis-closed-form prediction vs total-score
  empirical anchor)
"""

from __future__ import annotations

from typing import Any, Final, Union

from .procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    predicted_delta_s as _canonical_rate_axis_predicted_delta_s,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN",
    "PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN",
    "SEG_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN",
    "POSE_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN",
    "TOTAL_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN",
    "predicted_delta_s_with_axis_attribution",
    "axis_attribution_to_dict_for_metadata_json",
    "is_predicted_band_validated_post_training",
]


# Canonical UNKNOWN-pending tokens per Catalog #324 + #287 + #323
SEG_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN: Final[str] = (
    "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324"
)
"""Canonical token for SEG-axis prediction pending paired-smoke per Catalog #324."""

POSE_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN: Final[str] = (
    "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324"
)
"""Canonical token for POSE-axis prediction pending paired-smoke per Catalog #324."""

TOTAL_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN: Final[str] = (
    "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324"
)
"""Canonical token for TOTAL prediction pending paired-smoke per Catalog #324.

The total cannot be computed when seg+pose are UNKNOWN; the rate-axis
contribution alone is NOT a meaningful total because the contest score
formula is `25 * archive_bytes / 37_545_489 + 100 * seg + sqrt(10 * pose)`
and the seg+pose terms can be 10-100× larger than the rate-axis term."""

PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN: Final[str] = "pending_post_training"
"""Canonical recipe-frontmatter token per Catalog #324."""

PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN: Final[str] = "validated_post_training"
"""Canonical recipe-frontmatter token per Catalog #324 (set post-paired-smoke)."""


def predicted_delta_s_with_axis_attribution() -> dict[str, Union[float, str]]:
    """Return v8 chroma_lut predicted ΔS DECOMPOSED by axis per Catalog #324.

    Sister of the existing canonical
    ``tac.substrates.nscs06_v8_chroma_lut.procedural_variant.predicted_delta_s``
    (rate-axis only). THIS helper extends the prediction surface with EXPLICIT
    axis-attribution, ensuring downstream consumers (autopilot ranker, lane
    registry evidence emitters, smoke metadata JSON) carry the UNKNOWN tags
    on seg+pose axes per Catalog #324 discipline.

    Returns:
        Dict with 4 keys:
        - ``rate_axis``: float; canonical equation #26 closed-form prediction
          ``-0.002706`` for nscs06_v8_chroma_lut.
        - ``seg_axis``: str; canonical UNKNOWN token per Catalog #324.
        - ``pose_axis``: str; canonical UNKNOWN token per Catalog #324.
        - ``total_axis``: str; canonical UNKNOWN token (cannot total
          rate+UNKNOWN+UNKNOWN).
        - ``canonical_equation_in_domain_context``: str; ``nscs06_v8_chroma_lut``
          per canonical equation #26.
        - ``validation_status``: str; ``pending_post_training`` until paired
          smoke per Catalog #324.

    Example:
        >>> result = predicted_delta_s_with_axis_attribution()
        >>> isinstance(result["rate_axis"], float)
        True
        >>> result["seg_axis"] == "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324"
        True
    """
    return {
        "rate_axis": float(_canonical_rate_axis_predicted_delta_s()),
        "seg_axis": SEG_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
        "pose_axis": POSE_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
        "total_axis": TOTAL_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
        "canonical_equation_in_domain_context": CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        "validation_status": PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN,
    }


def axis_attribution_to_dict_for_metadata_json() -> dict[str, Any]:
    """Return axis-attributed prediction as canonical metadata-JSON-ready dict.

    Same as :func:`predicted_delta_s_with_axis_attribution` but with the
    additional `axis_tag` + `score_claim` + `promotion_eligible` fields per
    Catalog #287 + #323 canonical Provenance for embedding in smoke metadata
    JSON / archive manifests / cathedral autopilot ranker rows.
    """
    base = predicted_delta_s_with_axis_attribution()
    base.update(
        {
            "axis_tag": "[prediction; canonical-equation-26-grounded; rate-axis-only; seg+pose-pending-per-catalog-324]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "predicted",
            "predicted_band_validation_status": PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN,
            "reactivation_criteria": (
                "post-training Tier-C re-measurement on landed paired-smoke "
                "archive sha via tools/mdl_scorer_conditional_ablation.py --tier c "
                "+ paired contest-CUDA + contest-CPU anchors per CLAUDE.md "
                "'Submission auth eval - BOTH CPU AND CUDA' non-negotiable"
            ),
        }
    )
    return base


def is_predicted_band_validated_post_training(
    axis_attribution_dict: dict[str, Any],
) -> bool:
    """Return True iff axis_attribution_dict declares validated_post_training status.

    Canonical Catalog #324 consumer-side helper: downstream consumers check
    this predicate before treating the prediction as actionable for promotion.
    UNKNOWN tags on seg+pose keep this False; only post-paired-smoke
    re-emission with measured seg+pose values + ``validated_post_training``
    flips it True.

    Args:
        axis_attribution_dict: dict produced by
            :func:`predicted_delta_s_with_axis_attribution` (or an updated
            successor that replaces UNKNOWN tags with measured floats).

    Returns:
        True iff (a) `validation_status` is ``validated_post_training``,
        AND (b) all 3 axis fields (`rate_axis`, `seg_axis`, `pose_axis`)
        are finite floats (no UNKNOWN tokens).
    """
    if not isinstance(axis_attribution_dict, dict):
        return False
    status = axis_attribution_dict.get("validation_status")
    if status != PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN:
        return False
    for axis_key in ("rate_axis", "seg_axis", "pose_axis"):
        v = axis_attribution_dict.get(axis_key)
        if not isinstance(v, (int, float)):
            return False
        # Reject NaN/Inf as non-validated per Catalog #287.
        if v != v:  # NaN
            return False
        if v in (float("inf"), float("-inf")):
            return False
    return True
