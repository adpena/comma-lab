#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Refine ``renderer_seg_pose_coupling_shipped_object_v1`` with ddm_pr1's measurement.

WHAT THIS FIXES.  The registered law carries
``domain_of_validity["carrier_recovery_measured"] = [5.87, 8.0]``.  Neither number was
measured for a RENDERER change: 8.0x is ``ddm_jg5``'s n600 carrier recovery on TOKEN
edits and 5.87x is ``ddm_fcd2``'s.  ``ddm_ft1``'s closing table applied 8.0x to a
renderer change and concluded "81x over the ceiling"; the shipping chain's terminal
pose re-solve (``FIRE_ORDER.sh`` FO-2) was gated on "realized d_seg DOWN" and never ran.
This tool folds the MEASURED post-re-solve number in, and splits the law's domain so a
future reader cannot mistake a PRE-re-solve k for a POST-re-solve one again.

The refinement adds:

* ``domain_of_validity_included`` -- the PRE-re-solve reading of k, which both existing
  anchors are, and which is the correct input to a "what will this cost before the
  carrier is re-tuned" question.
* ``domain_of_validity_excluded`` -- using a PRE-re-solve k inside a promotion or
  closure verdict without the terminal re-solve, and transferring a TOKEN-edit carrier
  recovery factor onto a renderer change.
* ``carrier_recovery_measured_renderer_change`` -- the number this arm measured.

Then it appends the ddm_pr1 anchor. Every value is read from the arm's own report JSON;
nothing is retyped, so a transcription slip cannot enter the registry.

CONTAINMENT: pure read + JSONL append. No scorer, no Modal, no Metal, no launch. The
own-vehicle frontier is UNMOVED -- this is the equations leg, not a lever.

Usage:
    .venv/bin/python tools/register_ddm_pr1_coupling_refinement_20260904.py \\
        --report <pr1 report json> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.equation import (  # noqa: E402
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    EmpiricalAnchor,
)
from tac.canonical_equations.registry import (  # noqa: E402
    update_equation_with_domain_refinement,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

EQUATION_ID = "renderer_seg_pose_coupling_shipped_object_v1"
AGENT = "ddm_pr1_pose_resolve_on_renderer_change_20260904"
AXIS = "[macOS-CPU advisory]"


def build_anchor(report: dict, report_path: Path) -> EmpiricalAnchor:
    coupling = report["coupling"]
    recovery = report["recovery"]
    d_pose = report["d_pose"]
    closing = report["closing_arithmetic"]
    k_post = float(coupling["post_re_solve"])
    k_pre = float(coupling["pre_re_solve"])
    return EmpiricalAnchor(
        anchor_id="pr1_terminal_pose_resolve_on_ft1_step600_renderer_change_20260904",
        measurement_utc="2026-09-04T00:00:00Z",
        inputs={
            "object": (
                "ddm_ft1 step-600 seg-only fine-tune of the shipped SM3R semantic "
                "renderer, exported and parsed back through the shipped receiver, "
                "composed with the afr1 pose carrier and then RE-SOLVED"
            ),
            "edit_kind": (
                "TRAINED seg-only expected-flip margin loss, no pose term in the loop, "
                "followed by the shipping chain's TERMINAL pose re-solve"
            ),
            "solver": (
                "ddm_jg5.refine_pair -- br1 damped Gauss-Newton on the shipped 12-dim "
                "basis and signed-int12 lattice with the +-2 polish, under jg5's DERIVED "
                "materiality stop evaluated at the afr1 operating point 6.37e-06"
            ),
            "solver_not_used": (
                "ddm_up2.solve_pair_realized -- jg5 Sec 4 records its +-2 single-"
                "coordinate radius as a truncation br1 was built to escape; using it "
                "would measure the solver's ceiling and report it as the carrier's"
            ),
            "n_pairs": int(report["pairs"]),
            "pair_selection": report["pair_selection"],
            "batch_size": int(report["batch_size"]),
            "gt_lineage": "DALI",
            "delta_d_seg": float(coupling["delta_d_seg_used"]),
            "delta_d_seg_source": coupling["delta_d_seg_source"],
            "base_d_pose_as_shipped": float(d_pose["base_as_shipped"]),
            "base_vs_t4_receipt_relative": float(d_pose["base_vs_t4_relative"]),
            "archive_sha256": report["instrument"]["archive_sha256"],
            "report_path": str(report_path),
        },
        predicted_output={
            "prior_law": (
                "the registered law's closing table applied jg5's TOKEN-edit carrier "
                "recovery of 8.0x to a renderer change and concluded 81x over the "
                "ceiling; the recovery for a renderer change was never measured"
            ),
            "transferred_recovery_used_by_ft1": 8.0,
            "charter_predicted_post_re_solve_coupling_below": 20.0,
            "charter_falsifier_recovery_below": 3.0,
        },
        empirical_output={
            "coupling_pre_re_solve": k_pre,
            "coupling_post_re_solve": k_post,
            "carrier_recovery_mean_based": float(recovery["mean_based"]),
            "carrier_recovery_median_per_pair": float(recovery["median_per_pair"]),
            "d_pose_candidate_stale_carrier": float(d_pose["candidate_stale_carrier"]),
            "d_pose_candidate_re_solved": float(d_pose["candidate_re_solved"]),
            "payable_pose_ceiling_at_25pct_cut": float(closing["payable_pose_ceiling"]),
            "k_post_payable_bar_at_25pct_cut": float(closing["k_post_payable_bar"]),
            "overshoot_multiple_at_25pct_cut": float(closing["overshoot_multiple"]),
            "seg_only_move_payable_at_25pct_cut": bool(closing["payable"]),
            "charter_prediction_holds": bool(
                report["charter_prediction"]["prediction_holds"]
            ),
            "charter_falsifier_fired": bool(
                report["charter_prediction"]["falsifier_fired"]
            ),
        },
        # The law's registered band is a PRE-re-solve band; this anchor's own
        # quantity is the POST-re-solve k, which the band never modelled. The
        # residual is therefore reported against the quantity the anchor shares
        # with the band -- its PRE-re-solve k -- and never against a centre that
        # was fitted to a different quantity.
        residual=abs(k_pre - 190.38926383452008) / 190.38926383452008,
        source_artifact=str(report_path),
        measurement_method=(
            "n600 per-pair d_pose at one declared batch shape (8) on the frozen CPU-torch "
            "PoseNet against the DALI GT, for three code/render combinations: the shipped "
            "renderer with the shipped carrier, the candidate renderer with the STALE "
            "shipped carrier, and the candidate renderer with the carrier RE-SOLVED per "
            "pair by jg5's Gauss-Newton refinement; the candidate's odd frames are "
            "rendered through the shipped receiver's own frame-1 path "
            "(cpr1/inflate.py:315-326) and frame 0 through its carrier + selector path"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=str(report_path),
            reactivation_criteria=(
                "re-measure on a JOINT (pose-priced) objective, and on a candidate whose "
                "realized d_seg FALLS -- both anchors and this one moved d_seg UP, so the "
                "direction-symmetry assumption is still stated rather than measured"
            ),
            measurement_axis=AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_domain_extension(report: dict) -> dict:
    recovery = report["recovery"]
    coupling = report["coupling"]
    closing = report["closing_arithmetic"]
    return {
        "domain_of_validity_included": (
            "k as a PRE-re-solve coupling: the d_pose a seg-only renderer move costs "
            "BEFORE the shipping chain's terminal carrier re-solve runs. Both registered "
            "anchors (rf1 166.81, ft1 217.30) are this quantity.",
            "the shipped SM3R semantic renderer at its own size (36,130 B section)",
            "edits whose gradient direction prices d_seg ALONE",
        ),
        "domain_of_validity_excluded": (
            "using a PRE-re-solve k inside a closure, promotion or refusal verdict "
            "WITHOUT running the terminal pose re-solve: the shipping chain re-solves "
            "the carrier after every seg change, so a pre-re-solve k prices an object "
            "that never ships",
            "transferring a TOKEN-edit carrier recovery factor (jg5 8.0x, fcd2 5.87x) "
            "onto a RENDERER change: a renderer weight moves all 600 renders coherently "
            "while a token edit moves a subset locally, and ddm_pr1 measured the two "
            "recoveries to be different",
            "reading the mean-based recovery as the typical pair's recovery: the "
            "per-pair distribution is heavy-tailed and the n600 mean is owned by the "
            "worst-recovering pairs",
        ),
        "carrier_recovery_measured_renderer_change": {
            "mean_based": float(recovery["mean_based"]),
            "median_per_pair": float(recovery["median_per_pair"]),
            "n_pairs": int(report["pairs"]),
            "solver": "ddm_jg5.refine_pair (br1 GN + polish, derived materiality stop)",
            "source": "ddm_pr1",
        },
        "coupling_post_re_solve_measured": float(coupling["post_re_solve"]),
        "post_re_solve_payability_bar": {
            "seg_cut_fraction": float(closing["seg_cut_fraction"]),
            "k_post_must_be_at_most": float(closing["k_post_payable_bar"]),
            "derivation": (
                "at dB = 0 the promotion condition is sqrt(10 d_pose_new) < pose_leg_T4 "
                "+ 100 |delta d_seg|; solving for k_post = (d_pose_new - d_pose_base) / "
                "|delta d_seg| gives the bar"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != "tac.ddm_pr1.report.v1":
        raise SystemExit(f"{args.report} is not a ddm_pr1 report")

    extension = build_domain_extension(report)
    anchor = build_anchor(report, args.report)
    print(json.dumps(extension, indent=2))
    print(f"anchor {anchor.anchor_id}: {json.dumps(anchor.empirical_output, indent=2)}")
    if args.dry_run:
        print("DRY-RUN: registry not touched.")
        return 0

    update_equation_with_domain_refinement(
        EQUATION_ID,
        domain_of_validity_extension=extension,
        rationale=(
            "The law's closing arithmetic imported jg5's 8.0x TOKEN-edit carrier "
            "recovery onto a RENDERER change and never ran the terminal re-solve the "
            "shipping chain actually performs. ddm_pr1 measured that re-solve on the "
            "ft1 step-600 candidate at n600 and the recovery is not the transferred "
            "number, so the domain now separates the PRE-re-solve k both existing "
            "anchors measure from the POST-re-solve k a closure verdict needs."
        ),
        agent=AGENT,
        notes=(
            "ddm_pr1 domain refinement: PRE- vs POST-re-solve k are different "
            "quantities; a token-edit recovery factor does not transfer to a renderer "
            "change. Advisory axis; NON-PROMOTABLE; pointer unmoved."
        ),
    )
    print(f"domain refined on {EQUATION_ID}")
    update_equation_with_empirical_anchor(
        EQUATION_ID,
        anchor,
        agent=AGENT,
        notes=(
            "ddm_pr1 post-re-solve anchor: the terminal pose re-solve measured, not "
            "transferred. Advisory axis; NON-PROMOTABLE; pointer unmoved."
        ),
    )
    print(f"anchor appended to {EQUATION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
