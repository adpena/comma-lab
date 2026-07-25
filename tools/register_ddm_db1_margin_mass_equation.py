#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the exact fixed-atlas SN1 N(delta) duplicate-budget law."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.canonical_equations.equation import (  # noqa: E402
    RECALIBRATE_ON_PARAMETER_REFIT,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

EQUATION_ID = "ddm_sn1_margin_mass_duplicate_budget_bounds_v1"


def build_equation(receipt_path: Path) -> CanonicalEquation:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    margin = receipt["sn1_at1_margin_mass"]
    incidence_total = int(margin["total_ordered_boundary_incidences"])
    unique_total = int(margin["total_unique_boundary_pixels"])
    duplicate_budget = int(margin["total_duplicate_incidences"])
    if incidence_total - unique_total != duplicate_budget:
        raise ValueError("receipt duplicate-budget identity failed")
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=receipt_path,
        reactivation_criteria=(
            "remain fixed-atlas research-only until coordinate-resolved target-error membership "
            "and live descent replenishment are measured"
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory artifacts; no new scorer invocation]",
        hardware_substrate="m5_max_macos_cpu_artifact_analysis",
        captured_at_utc=receipt["written_at_utc"],
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_sn1_n600_complete_boundary_duplicate_budget_20260725",
        measurement_utc=receipt["written_at_utc"],
        inputs={
            "ordered_boundary_incidences": incidence_total,
            "unique_boundary_pixels": unique_total,
            "class_count": 5,
            "pair_count": 600,
            "sn1_receipt_sha256": margin["sn1_custody"]["receipt"]["sha256"],
            "at1_atlas_payload_sha256": margin["at1_custody"][
                "atlas_canonical_payload_sha256"
            ],
        },
        predicted_output={
            "at_delta_infinity_lower": unique_total,
            "at_delta_infinity_upper": unique_total,
            "identity": "D=I(infinity)-B",
        },
        empirical_output={
            "duplicate_budget": duplicate_budget,
            "at_delta_infinity_unique_count": unique_total,
            "bound_collapses_to_equality": True,
        },
        residual=0.0,
        source_artifact=str(receipt_path),
        measurement_method="hash_verified_sn1_ordered_incidence_and_unique_boundary_accounting",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="SN1 fixed-boundary margin mass with exact global duplicate budget",
        one_line_summary=(
            "For the fixed SN1 boundary atlas, subtracting the complete duplicate-incidence "
            "budget gives a tight lower bound on unique near-flip pixels."
        ),
        latex_form=(
            r"\max(0,I(\delta)-[I(\infty)-B])\leq N(\delta)"
            r"\leq\min(I(\delta),B)"
        ),
        python_callable_module_path="tac.analysis.ddm_db1_decay_bounds:unique_count_bounds",
        domain_of_validity={
            "included": [
                "hash-bound SN1 n600 ordered predicted-boundary margin shards",
                "AT1 exact rank-4 head distance d2=margin/head-pair-norm",
                "fixed atlas threshold queries",
            ],
            "excluded": [
                "target-error-conditioned correctable mass",
                "non-boundary pixels",
                "live descent boundary replenishment",
                "steps-to-target or terminal descent predictions",
                "contest score or promotion",
            ],
            "evidence_axis": (
                "[macOS-CPU frozen-scorer advisory artifacts; no new scorer invocation]"
            ),
        },
        units_in={
            "incidence_count": "ordered_boundary_incidence_count",
            "total_incidence_count": "ordered_boundary_incidence_count",
            "unique_total": "unique_boundary_pixel_count",
        },
        units_out={
            "lower": "unique_boundary_pixel_count",
            "upper": "unique_boundary_pixel_count",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"complete_atlas_duplicate_identity": 0.0},
        last_calibration_utc=receipt["written_at_utc"],
        next_recalibration_trigger=RECALIBRATE_ON_PARAMETER_REFIT,
        canonical_consumers=(
            "tac.analysis.ddm_db1_decay_bounds.analyze_margin_mass",
            "tools.analyze_ddm_db1_decay_bounds",
        ),
        canonical_producers=(
            "tac.analysis.segnet_internal_telemetry.extract_ordered_pair_boundary_samples",
            "tools.register_ddm_db1_margin_mass_equation",
        ),
        provenance=provenance,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    equation = build_equation(args.receipt)
    if args.dry_run:
        print(json.dumps(equation.to_dict(), indent=2, sort_keys=True))
        return 0
    register_canonical_equation(
        equation,
        agent="codex",
        subagent_id="ddm_db1_decay_bounds_20260725T121605Z",
        notes=(
            "FEED-603-db1 exact fixed-atlas margin-mass law; no V19C decay law "
            "registered because proposal-order extrapolation is not transferable"
        ),
    )
    print(json.dumps({"registered": EQUATION_ID, "anchor_count": 1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
