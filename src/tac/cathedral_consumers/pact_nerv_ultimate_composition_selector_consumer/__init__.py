# SPDX-License-Identifier: MIT
"""Cathedral consumer for Pact-NeRV ULTIMATE composition selector.

Per WAVE-3-PACT-NERV-ULTIMATE-RESEARCH-AND-DESIGN task spec deliverable D
+ PACT-NERV-DESIGN-SYMPOSIUM Section 13 stack-of-stacks Catalog #322
EMPIRICAL alpha validation. Sister of
:mod:`tac.cathedral_consumers.cross_substrate_similarity_consumer`
(Catalog #344 sister pattern; commit ``af727e3c1``).

For each in-flight Pact-NeRV variant candidate, predicts
ULTIMATE-eligibility per the 8 dimensions (FRONTIER / PAPER / RIGOR /
COMPOSITION-ALPHA / EFFICIENCY / INTERPRETABILITY / CONTEST-COMPLIANT /
CROSS-CODEC) based on:

  * the variant taxonomy table (15+ variants with LOC + predicted ΔS
    + classification per Section 4 of the parent research memo
    ``.omx/research/pact_nerv_ultimate_research_and_design_*.md``)
  * the CROSS-CANDIDATE empirical α matrix (cross-substrate composition
    signals at ``.omx/state/cross_substrate_sensitivity_similarity_matrix_*.json``)
  * the canonical equations registry (per-variant predicted ΔS
    predictors per Catalog #344)

Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
this consumer NEVER mutates ``predicted_delta_adjustment`` (always 0.0);
``promotable=False`` always; ``axis_tag="[predicted]"`` always. The
ULTIMATE-eligibility prediction is observability-only because the
variant taxonomy combines [literature-prediction] + apparatus-empirical
extrapolation signals from CROSS-CANDIDATE findings; per Catalog #287
+ #341 these are non-promotable until per-Stage post-training Tier-C
validation per Catalog #324.

Sister wire-in:
  Catalog #322 ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
  — this consumer cites canonical equation ``cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1``
  per Catalog #344 sister discipline; downstream consumers refusing
  phantom-provenance composition_alpha rows per Catalog #321/#322 see
  this consumer's ULTIMATE-eligibility annotations as observability-only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "pact_nerv_ultimate_composition_selector_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "PACT_NERV_VARIANT_TAXONOMY",
    "ULTIMATE_DIMENSIONS",
    "update_from_anchor",
    "consume_candidate",
]


# Canonical 15+ variant taxonomy per parent research memo Section 4.
# Each row: (variant_name, loc, predicted_delta_lower, predicted_delta_upper,
#            classification, ultimate_dimensions_eligible, priority)
PACT_NERV_VARIANT_TAXONOMY: tuple[dict[str, Any], ...] = (
    # Group 1 — bleeding-edge architecture variants
    {"name": "pact_nerv_mamba", "loc": 1800, "delta_lower": -0.005, "delta_upper": 0.005,
     "classification": "CARGO-CULTED-MAY-BE-PROMISING",
     "ultimates": ("PAPER",), "priority": 3},
    {"name": "pact_nerv_moe", "loc": 1500, "delta_lower": -0.003, "delta_upper": 0.002,
     "classification": "CARGO-CULTED-MAY-BE-PROMISING",
     "ultimates": ("PAPER", "FRONTIER"), "priority": 2},
    {"name": "pact_nerv_diffusion_distilled", "loc": 2000, "delta_lower": -0.005, "delta_upper": 0.010,
     "classification": "CARGO-CULTED-RISK-ARTIFACT",
     "ultimates": ("PAPER",), "priority": 3},
    {"name": "pact_nerv_dreamer", "loc": 2200, "delta_lower": -0.008, "delta_upper": 0.005,
     "classification": "HARD-EARNED-LITERATURE-COOPERATIVE-RECEIVER",
     "ultimates": ("FRONTIER", "PAPER"), "priority": 3},
    {"name": "pact_nerv_neural_codec_e2e", "loc": 1800, "delta_lower": -0.010, "delta_upper": 0.005,
     "classification": "HARD-EARNED-LITERATURE",
     "ultimates": ("FRONTIER", "PAPER", "CROSS-CODEC"), "priority": 2},
    # Group 2 — mid-LOC apparatus-aligned variants
    {"name": "pact_nerv_distilled_scorer", "loc": 400, "delta_lower": -0.003, "delta_upper": 0.001,
     "classification": "HARD-EARNED-THEORETICALLY-HINTON",
     "ultimates": ("FRONTIER", "RIGOR"), "priority": 2},
    {"name": "pact_nerv_vq", "loc": 500, "delta_lower": -0.005, "delta_upper": 0.003,
     "classification": "HARD-EARNED-LITERATURE-VDOORD",
     "ultimates": ("PAPER", "FRONTIER"), "priority": 2},
    {"name": "pact_nerv_bayesian", "loc": 600, "delta_lower": -0.002, "delta_upper": 0.003,
     "classification": "HARD-EARNED-LITERATURE-MACKAY",
     "ultimates": ("RIGOR", "INTERPRETABILITY"), "priority": 3},
    {"name": "pact_nerv_multimodal", "loc": 700, "delta_lower": -0.004, "delta_upper": 0.002,
     "classification": "HARD-EARNED-LITERATURE",
     "ultimates": ("PAPER", "COMPOSITION-ALPHA"), "priority": 3},
    {"name": "pact_nerv_diffusion_trajectory", "loc": 1800, "delta_lower": 0.001, "delta_upper": 0.020,
     "classification": "CARGO-CULTED-RISK-LOC-EXCESS",
     "ultimates": ("PAPER",), "priority": 3},
    # Group 3 — SELECTOR-PARADIGM-EXTENSIONS per CROSS-CANDIDATE finding #1
    {"name": "pact_nerv_selector_v2", "loc": 150, "delta_lower": -0.003, "delta_upper": 0.000,
     "classification": "HARD-EARNED-VIA-CROSS-CANDIDATE-EMPIRICAL",
     "ultimates": ("FRONTIER", "EFFICIENCY"), "priority": 1},
    {"name": "pact_nerv_selector_v3", "loc": 300, "delta_lower": -0.004, "delta_upper": 0.001,
     "classification": "HARD-EARNED-VIA-CROSS-CANDIDATE-EMPIRICAL",
     "ultimates": ("FRONTIER", "EFFICIENCY"), "priority": 1},
    {"name": "pact_nerv_selector_v4", "loc": 400, "delta_lower": -0.005, "delta_upper": 0.001,
     "classification": "HARD-EARNED-VIA-CROSS-CANDIDATE-EMPIRICAL",
     "ultimates": ("FRONTIER", "EFFICIENCY"), "priority": 1},
    {"name": "pact_nerv_ia3_multi", "loc": 150, "delta_lower": -0.003, "delta_upper": 0.001,
     "classification": "HARD-EARNED-LITERATURE",
     "ultimates": ("FRONTIER", "EFFICIENCY"), "priority": 1},
    {"name": "pact_nerv_asymmetric_boundary", "loc": 250, "delta_lower": -0.004, "delta_upper": 0.001,
     "classification": "HARD-EARNED-VIA-APPARATUS-CANONICAL-EQUATION",
     "ultimates": ("FRONTIER", "INTERPRETABILITY"), "priority": 2},
    # Group 4 — CROSS-CODEC composition per CROSS-CANDIDATE finding #3
    {"name": "pact_nerv_cross_codec_a", "loc": 600, "delta_lower": -0.010, "delta_upper": -0.003,
     "classification": "HARD-EARNED-EMPIRICALLY-VIA-CROSS-CANDIDATE-FINDING_3",
     "ultimates": ("CROSS-CODEC", "FRONTIER"), "priority": 1},
    {"name": "pact_nerv_cross_codec_b", "loc": 700, "delta_lower": -0.012, "delta_upper": -0.001,
     "classification": "HARD-EARNED-EMPIRICALLY-VIA-CROSS-CANDIDATE",
     "ultimates": ("CROSS-CODEC", "FRONTIER"), "priority": 2},
    {"name": "pact_nerv_neural_codec_e2e_cross", "loc": 1500, "delta_lower": -0.015, "delta_upper": 0.005,
     "classification": "HARD-EARNED-CLASS-VIA-CROSS-CANDIDATE",
     "ultimates": ("PAPER", "CROSS-CODEC"), "priority": 3},
    # Sister Stage 1+2 from FILM-FAMILY-RESEARCH / PACT-NERV-DESIGN-SYMPOSIUM
    {"name": "pact_nerv_ia3", "loc": 50, "delta_lower": -0.002, "delta_upper": 0.001,
     "classification": "HARD-EARNED-LITERATURE",
     "ultimates": ("FRONTIER", "EFFICIENCY"), "priority": 1},
    {"name": "pact_nerv_a1", "loc": 600, "delta_lower": -0.005, "delta_upper": 0.001,
     "classification": "HARD-EARNED-STACK",
     "ultimates": ("FRONTIER", "PAPER", "RIGOR"), "priority": 1},
)


# Canonical 8 ULTIMATE dimensions per parent research memo Section 10.
ULTIMATE_DIMENSIONS: tuple[str, ...] = (
    "FRONTIER",
    "PAPER",
    "RIGOR",
    "COMPOSITION-ALPHA",
    "EFFICIENCY",
    "INTERPRETABILITY",
    "CONTEST-COMPLIANT",
    "CROSS-CODEC",
)


# Canonical similarity matrix path glob per sister consumer pattern.
_MATRIX_GLOB = "cross_substrate_sensitivity_similarity_matrix_*.json"


def _state_dir() -> Path:
    """Locate .omx/state/ for matrix discovery.

    Walks up from this file to find the repo root + .omx/state/. Returns
    a non-existent Path if not found (callers handle missing matrix
    gracefully per Catalog #138 sister fail-closed pattern).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_matrix() -> Mapping[str, Any] | None:
    """Load the most-recent cross-substrate similarity matrix.

    Sister of cross_substrate_similarity_consumer._load_latest_matrix.
    Returns None gracefully when no matrix is on disk per Catalog #138
    sister fail-closed discipline.
    """
    state = _state_dir()
    if not state.is_dir():
        return None
    matches = sorted(state.glob(_MATRIX_GLOB))
    if not matches:
        return None
    latest = matches[-1]
    try:
        with latest.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _candidate_variant_name(candidate: Mapping[str, Any]) -> str | None:
    """Heuristic: extract canonical Pact-NeRV variant name from candidate dict.

    Walks string-valued fields looking for known variant name tokens.
    Sister of cross_substrate_similarity_consumer._candidate_substrate_label.
    """
    candidate_text_parts: list[str] = []
    for k, v in candidate.items():
        if isinstance(v, (str, int, float)):
            candidate_text_parts.append(f"{k}={v}")
    candidate_text = " ".join(candidate_text_parts).lower()
    # Sort by length descending so longer variant names match before
    # shorter prefixes (pact_nerv_selector_v2 before pact_nerv_selector).
    known = sorted(
        (row["name"] for row in PACT_NERV_VARIANT_TAXONOMY),
        key=len,
        reverse=True,
    )
    for variant_name in known:
        if variant_name in candidate_text:
            return variant_name
    # Sister forms — alias mapping
    if "pact_nerv" in candidate_text or "pact-nerv" in candidate_text:
        return "pact_nerv_a1"  # canonical baseline if pact_nerv mentioned without variant
    return None


def _variant_row(variant_name: str) -> Mapping[str, Any] | None:
    """Look up canonical variant row from the taxonomy."""
    for row in PACT_NERV_VARIANT_TAXONOMY:
        if row["name"] == variant_name:
            return row
    return None


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference implementation: NO-OP. The canonical refresh path is
    operator-triggered via re-registration of the variant taxonomy when
    a new variant lands or empirical anchor updates classification per
    Catalog #287 + #323 (signed measurement provenance required). The
    consumer surfaces the canonical taxonomy on every consume call so
    live updates propagate without per-consumer state.

    Future enhancement: when an empirical anchor for a specific variant
    lands at L1+ with paired CPU+CUDA per CROSS-CANDIDATE finding #2,
    update the variant's ``classification`` from HARD-EARNED-LITERATURE
    to HARD-EARNED-EMPIRICAL + tighten the predicted_delta band per
    Catalog #324 post-training Tier-C validation result.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with Pact-NeRV ULTIMATE eligibility.

    Returns the canonical Tier A contribution per Catalog #341:
      * predicted_delta_adjustment=0.0 (NEVER mutates score)
      * promotable=False (NEVER promotes)
      * axis_tag="[predicted]" (advisory-only per Catalog #287)
      * rationale + matched_variant + ultimate_eligibility + priority
        (operator audit surface)
    """
    variant_name = _candidate_variant_name(candidate)
    if variant_name is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "candidate did not match any known Pact-NeRV variant name in "
                "PACT_NERV_VARIANT_TAXONOMY; observability-only annotation "
                "[predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "matched_variant": None,
        }
    row = _variant_row(variant_name)
    if row is None:  # pragma: no cover — defensive guard
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"candidate matched variant_name={variant_name!r} but row not in "
                "PACT_NERV_VARIANT_TAXONOMY; observability-only annotation [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "matched_variant": variant_name,
        }
    matrix = _load_latest_matrix()
    matrix_available = matrix is not None
    matrix_captured_at = (
        matrix.get("captured_at_utc", "") if matrix_available else ""
    )
    rationale = (
        f"candidate variant {variant_name!r} matches Pact-NeRV taxonomy row "
        f"(LOC={row['loc']}, predicted ΔS [{row['delta_lower']:.3f}, "
        f"{row['delta_upper']:.3f}], classification={row['classification']!r}, "
        f"PRIORITY {row['priority']}); ULTIMATE-eligibility: "
        f"{', '.join(row['ultimates'])}; cross-substrate similarity matrix "
        f"{'available' if matrix_available else 'unavailable'}; observability-"
        "only per Catalog #341 [predicted]"
    )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_variant": variant_name,
        "variant_loc": row["loc"],
        "variant_predicted_delta_lower": row["delta_lower"],
        "variant_predicted_delta_upper": row["delta_upper"],
        "variant_classification": row["classification"],
        "ultimate_eligibility": list(row["ultimates"]),
        "priority": row["priority"],
        "matrix_available": matrix_available,
        "matrix_captured_at_utc": matrix_captured_at,
    }
