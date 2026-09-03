#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the ddm_dr1 n600 delta_R EmpiricalAnchor on the canonical law.

WHY
---
``margin_band_satisficing_threshold_v1`` (m_safe = headroom * delta_R) carries
exactly one anchor: ``margin_band_delta_r_noise_floor_n96_20260708``, measured on
a CONTIGUOUS 96-frame PREFIX of the 600-pair cohort.  A prefix of a skewed
population is a different population ([[m88]]), so ddm_dr1 re-measured delta_R at
n600 (all pairs) with the SAME tool.  This script appends the n600 anchor through
the canonical registry helper.  The JSONL is never hand-edited.

RESIDUAL SEMANTICS (deliberate, and the one judgement call here)
---------------------------------------------------------------
``EmpiricalAnchor.residual`` feeds the equation's Bayesian posterior over the
LAW's predicted-vs-empirical error.  The law here is ``m_safe = headroom *
delta_R`` — exact fp arithmetic — so its residual is 0.0 at every anchor, exactly
as the n96 anchor records.  The n96->n600 deviation of delta_R is NOT a law
error: it is a change in the measured INPUT.  Putting it in ``residual`` would
teach the posterior that the multiplication is inaccurate, which is false.  It is
therefore recorded as data inside ``empirical_output`` (with the pre-registered
falsifier verdict) and left out of the residual.

AUTHORITY: [PyAV frames . macOS-CPU advisory . NON-PROMOTABLE].  This sets a
lever constant.  It is not a score and it moves no pointer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Pre-registered in the ddm_dr1 charter (commit 4870d475c): the n96 value and the
# +/-10% band whose violation re-grades every m_safe derived from it.
N96_DELTA_R = 0.019590163230895963
FALSIFIER_REL_TOL = 0.10
ANCHOR_ID = "margin_band_delta_r_noise_floor_n600_20260904"
AXIS = "[PyAV frames . macOS-CPU advisory . frozen CPU-torch SegNet . NON-PROMOTABLE]"


def falsifier_verdict(delta_r_n600: float, *, n96: float = N96_DELTA_R,
                      rel_tol: float = FALSIFIER_REL_TOL) -> dict[str, Any]:
    """Read the pre-registered falsifier out explicitly."""
    lo = n96 * (1.0 - rel_tol)
    hi = n96 * (1.0 + rel_tol)
    fired = not (lo <= delta_r_n600 <= hi)
    return {
        "prediction": f"n600 delta_R within +/-{rel_tol:.0%} of the n96 value",
        "n96_delta_R": n96,
        "band": [lo, hi],
        "n600_delta_R": delta_r_n600,
        "relative_deviation": (delta_r_n600 - n96) / n96,
        "ratio_n600_over_n96": delta_r_n600 / n96,
        "falsifier_fired": fired,
        "consequence_if_fired": (
            "the n96 constant was prefix-biased; every m_safe derived from it "
            "(fh1 R3, hg1, nx1) is re-graded and the law's artifact pointer must move"
        ),
    }


def build_n600_anchor(report: dict, receipts: dict | None, *, report_path: str,
                      measurement_utc: str, gt_npz_sha256: str | None = None):
    """Build the n600 EmpiricalAnchor from the tool's own output JSONs."""
    from tac.canonical_equations.equation import (
        VERIFIED_VIA_EMPIRICAL_ANCHOR,
        EmpiricalAnchor,
    )
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        margin_safe_threshold,
        minimum_integer_headroom,
    )
    from tac.provenance.builders import build_provenance_for_research_sidecar

    delta_r = float(report["delta_R"])
    full_r = float(report["cross_check_full_R_vs_gt_direct"]["annulus"]["p95"])
    headroom = minimum_integer_headroom(delta_r, full_r)
    m_safe = margin_safe_threshold(delta_r, headroom)

    per_class = (receipts or {}).get("per_class_annulus_pooled") or {}
    sub_band = (receipts or {}).get("sub_band_sensitivity") or {}

    def _p95(row) -> float | None:
        """A receipt slice with no pixels (row None) or a null p95 has no value."""
        if not row or row.get("p95") is None:
            return None
        return float(row["p95"])

    def _class_m_safe(row) -> float | None:
        # A degenerate all-zero perturbation has no derivable threshold:
        # margin_safe_threshold refuses delta <= 0, and one degenerate class
        # must not crash the whole registration.
        p95 = _p95(row)
        if p95 is None or p95 <= 0.0:
            return None
        return margin_safe_threshold(p95, headroom)

    per_class_m_safe = {name: _class_m_safe(row) for name, row in per_class.items()}

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=report_path,
        reactivation_criteria=(
            "re-run tools/measure_delta_R_noise_floor.py if the GT cohort, the frozen "
            "SegNet weights, or the R chain change; re-derive the minimum headroom; "
            "the headroom-2-versus-3 A/B remains OPEN_UNMEASURED"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_cpu_torch",
        captured_at_utc=measurement_utc,
    )
    return EmpiricalAnchor(
        anchor_id=ANCHOR_ID,
        measurement_utc=measurement_utc,
        inputs={
            "artifact": report_path,
            "measurement_tool": "tools/measure_delta_R_noise_floor.py",
            "n_frames": int(report["n_frames"]),
            "annulus_band": float(report["band"]),
            "cohort": "ALL 600 pairs — not a prefix",
            "gt_npz": report["gt_npz"],
            "gt_npz_sha256": gt_npz_sha256,
            "gt_frame_lineage": (
                "PyAV (gt_n600.npz); 20,671 argmax sites differ from the DALI cache — "
                "delta_R uses the FRAMES and SegNet's own margins, not the lstars table"
            ),
            "torch_num_threads": (receipts or {}).get("torch_num_threads"),
        },
        predicted_output={
            "law": "m_safe = headroom * delta_R",
            "headroom_policy": (
                "smallest integer factor covering measured full-R annulus p95"
            ),
            "pre_registered_prediction": (
                "n600 delta_R within +/-10% of the n96 prefix value 0.019590163230895963"
            ),
        },
        empirical_output={
            "delta_R": delta_r,
            "full_R_annulus_p95": full_r,
            "derived_headroom": headroom,
            "derived_m_safe": m_safe,
            "annulus_area_frac": float(report["annulus_area_frac"]),
            "per_class_annulus_p95": {
                name: _p95(row) for name, row in per_class.items()
            },
            "per_class_derived_m_safe": per_class_m_safe,
            "sub_band_delta_R": {
                name: _p95(row) for name, row in sub_band.items()
            },
            "prefix_bias_check": falsifier_verdict(delta_r),
            "headroom_3_status": "OPEN_UNMEASURED_TREATMENT_NOT_DEFAULT",
            "treatment_delta_s": "UNMEASURED",
        },
        # See RESIDUAL SEMANTICS in the module docstring: the law is exact
        # arithmetic, so its residual stays 0.0; the n96->n600 input change is
        # recorded as data above, never as law error.
        residual=0.0,
        source_artifact=report_path,
        measurement_method=(
            "frozen CPU-torch SegNet margin perturbation over all 600 pairs; delta_R is "
            "the annulus p95 of the uint8-at-camera isolation |margin(round(x_c)) - "
            "margin(x_c)|; full-R p95 is the conservative coverage cross-check. The "
            "instrument is top1-top2; hg1 MEASURED the exact signed margin agreeing to "
            "~1% on the same statistics (GT is runner-up on 98.018% of flips)"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance=(
            "deterministic: fixed GT frames, frozen weights, CPU torch at a fixed "
            "thread count reproduce the JSON byte-identically (verified at n=8)"
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="reports/delta_R_noise_floor_n600.json")
    ap.add_argument("--receipts", default=None)
    ap.add_argument("--gt-npz-sha256", default=None)
    ap.add_argument("--measurement-utc", required=True,
                    help="UTC of the measurement run (the tool embeds no clock)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the anchor without touching the registry")
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    receipts = (
        json.loads(Path(args.receipts).read_text(encoding="utf-8"))
        if args.receipts else None
    )
    anchor = build_n600_anchor(
        report, receipts,
        report_path=args.report,
        measurement_utc=args.measurement_utc,
        gt_npz_sha256=args.gt_npz_sha256,
    )
    verdict = anchor.empirical_output["prefix_bias_check"]
    print(json.dumps({
        "anchor_id": anchor.anchor_id,
        "delta_R": anchor.empirical_output["delta_R"],
        "derived_headroom": anchor.empirical_output["derived_headroom"],
        "derived_m_safe": anchor.empirical_output["derived_m_safe"],
        "falsifier": verdict,
    }, indent=2))
    if args.dry_run:
        print("[dry-run] registry untouched")
        return 0

    from tac.canonical_equations import (
        append_empirical_anchor_to_equation_with_posterior_update,
    )
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        EQUATION_ID,
    )

    equation, posterior = append_empirical_anchor_to_equation_with_posterior_update(
        EQUATION_ID,
        anchor,
        agent="claude",
        subagent_id="ddm_dr1",
        notes=(
            "ddm_dr1: delta_R re-measured at n600 (all pairs) with the SAME tool; the "
            "sole prior anchor was a contiguous n96 PREFIX. NON-PROMOTABLE advisory."
        ),
    )
    print(json.dumps({
        "registered": equation.equation_id,
        "anchors_now": [a.anchor_id for a in equation.empirical_anchors],
        "posterior": str(posterior)[:400],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
