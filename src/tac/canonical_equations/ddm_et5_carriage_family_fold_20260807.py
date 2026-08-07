# SPDX-License-Identifier: MIT
"""ET5 restricted-carriage fold law."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_et5_restricted_carriage_family_fold_v1"
PRICING_RECEIPT_PATH = ".omx/research/ddm_et5_20260807/pricing_receipt.json"
SOURCE_ARTIFACT = ".omx/research/ddm_et5_20260807/RECEIPT.md"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def restricted_carriage_fold_decision(
    *,
    best_b_per_full_patch_flip: float,
    waterline_b_per_flip: float,
    waterfill_selected_count: int,
    realization_ran: bool,
) -> dict[str, float | bool | str]:
    """Return ET5's scorer-free fold decision from priced carriage fields."""

    if best_b_per_full_patch_flip < 0 or waterline_b_per_flip <= 0:
        raise ValueError("bytes-per-flip values must be non-negative with positive waterline")
    over_waterline = float(best_b_per_full_patch_flip) / float(waterline_b_per_flip)
    folded = over_waterline > 1.0 and int(waterfill_selected_count) == 0 and not realization_ran
    return {
        "folded": folded,
        "over_waterline_ratio": over_waterline,
        "selected_count": int(waterfill_selected_count),
        "owed_reopen_condition": (
            "new restriction/coder measures <= W on stratified n>=32"
            if folded
            else "materialize all-600 and validate restricted-patch argmax"
        ),
    }


def restricted_carriage_fold_decision_from_receipt(
    *, pricing_receipt_path: str | Path = PRICING_RECEIPT_PATH
) -> dict[str, float | bool | str]:
    receipt = _read_json(pricing_receipt_path)
    best = receipt["verdict"]["best_projection"]["coder_row"]
    waterfill = best["waterfill_subset_if_full_patch_flips_retained"]
    return restricted_carriage_fold_decision(
        best_b_per_full_patch_flip=float(best["B_per_full_patch_flip"]),
        waterline_b_per_flip=float(receipt["waterline_B_per_flip"]),
        waterfill_selected_count=int(waterfill["selected_count"]),
        realization_ran=False,
    )


def build_ddm_et5_restricted_carriage_family_fold_v1(
    *, pricing_receipt_path: str | Path = PRICING_RECEIPT_PATH
) -> CanonicalEquation:
    receipt = _read_json(pricing_receipt_path)
    best_projection = receipt["verdict"]["best_projection"]
    best = best_projection["coder_row"]
    waterfill = best["waterfill_subset_if_full_patch_flips_retained"]
    decision = restricted_carriage_fold_decision_from_receipt(
        pricing_receipt_path=pricing_receipt_path
    )
    provenance = build_provenance_for_research_sidecar(
        SOURCE_ARTIFACT,
        reactivation_criteria=(
            "reopen only if a new restriction/coder measures <= W on a stratified n>=32 "
            "sample, followed by all-600 materialization and scorer validation"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="scorer_free_byte_pricing",
        captured_at_utc="2026-08-07T10:39:22Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="et5_best_restricted_patch_carriage_rate_dead_20260807",
        measurement_utc="2026-08-07T10:39:22Z",
        inputs={
            "pricing_receipt_path": str(pricing_receipt_path),
            "selection": receipt["selection"],
            "restriction": best_projection["restriction"],
            "coder": best["coder"],
            "waterline_B_per_flip": receipt["waterline_B_per_flip"],
        },
        predicted_output={
            "folded": True,
            "waterfill_selected_count": 0,
            "realization_should_fire": False,
        },
        empirical_output={
            "status": receipt["verdict"]["status"],
            "verdict_scope": receipt["verdict"]["verdict_scope"],
            "B_per_full_patch_flip": best["B_per_full_patch_flip"],
            "over_waterline_ratio": decision["over_waterline_ratio"],
            "projected_n600_bytes": best["projected_n600_bytes"],
            "projected_net_delta_S": best["net_delta_S_if_full_patch_flips_retained"],
            "waterfill_selected_count": waterfill["selected_count"],
        },
        residual=0.0,
        source_artifact=str(pricing_receipt_path),
        measurement_method=(
            "scorer-free deterministic stratified n32 ET4 patch restriction/coder pricing; "
            "no realization or score promotion"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="ET5 restricted-carriage family fold",
        one_line_summary=(
            "Every measured ET4 restricted-patch carriage candidate exceeded W before "
            "realization; waterfill selected 0/32, so the family folds at instance scope."
        ),
        latex_form=r"\text{fold}\iff \min_i B_i/F_i > W \wedge |\{i:B_i/F_i\le W\}|=0",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_et5_carriage_family_fold_20260807:"
            "restricted_carriage_fold_decision"
        ),
        domain_of_validity={
            "included": [
                "ET4 correction field on tq1c parent",
                "stratified random n32 scorer-free description pricing",
            ],
            "excluded": [
                "restricted-patch realization authority",
                "new grammar families not measured in ET5",
                "claiming the solve family is dead",
            ],
            "verdict_scope": receipt["verdict"]["verdict_scope"],
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "best_b_per_full_patch_flip": "bytes per projected full-patch flip",
            "waterline_b_per_flip": "bytes per flip",
            "waterfill_selected_count": "pairs",
            "realization_ran": "bool",
        },
        units_out={
            "folded": "bool",
            "over_waterline_ratio": "unitless",
            "owed_reopen_condition": "string",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"et5_fold_decision_residual": 0.0},
        last_calibration_utc="2026-08-07T10:39:22Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "campaign_984_composition_route",
            "ET4 carriage successor selection",
            "scorer_slot_fire_order",
        ),
        canonical_producers=(
            ".omx/research/ddm_et5_20260807/pricing_receipt.json",
            ".omx/research/ddm_et5_20260807/CAMPAIGN_984_ROUTE.md",
        ),
        provenance=provenance,
    )


def populate_ddm_et5_restricted_carriage_family_fold_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_ddm_et5_restricted_carriage_family_fold_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cq1 registration: ET5 restricted-carriage fold verdict",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "PRICING_RECEIPT_PATH",
    "SOURCE_ARTIFACT",
    "build_ddm_et5_restricted_carriage_family_fold_v1",
    "populate_ddm_et5_restricted_carriage_family_fold_v1",
    "restricted_carriage_fold_decision",
    "restricted_carriage_fold_decision_from_receipt",
]
