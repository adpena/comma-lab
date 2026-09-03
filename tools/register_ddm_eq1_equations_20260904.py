#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register ddm_eq1's two backfilled canonical equations into the JSONL registry
(``.omx/state/canonical_equations_registry.jsonl``), so the EQUATIONS leg of the triality
carries two laws that until now lived only in memo prose.

1. ``renderer_seg_pose_coupling_shipped_object_v1`` — the shipped SM3R renderer's
   seg->pose coupling, TWO independent anchors: rf1 166.81 (un-retrained structural)
   and ft1 217.30 (trained seg-only fine-tune), 1.303x apart. At dB = 0 a 25% seg cut
   funds d_pose 1.694e-05 and costs >= 8.4e-03, so the seg-only renderer formulation is
   CLOSED by arithmetic at BOTH ends of the measured band. Joint (pose-priced)
   formulations are explicitly OUTSIDE the domain.

2. ``annulus_restricted_prefix_bias_detector_v1`` — dr1's n96->n600 result read as a
   DETECTOR: the annulus-restricted p95 moved +11.698% while the same field's all-pixel
   p95 moved +0.451% (25.94x amplification) and its all-pixel mean moved -0.004%. The
   prefix positive control reproduces the independent n96 artifact bit-identically, so
   the deviation is 100% cohort. A global-statistic sanity check therefore has no power
   against a restricted-set prefix bias.

CONTAINMENT: pure build + JSONL append; NO scorer, NO Modal, NO Metal, NO launch. The
own-vehicle frontier is UNMOVED — this is apparatus, not a lever. Idempotent in the
registry's append-only sense: re-running appends another 'registered' event and
``query_equations`` returns the latest payload per equation_id.

Usage:
    .venv/bin/python tools/register_ddm_eq1_equations_20260904.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904 import (  # noqa: E402
    build_annulus_restricted_prefix_bias_detector_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.canonical_equations.renderer_seg_pose_coupling_20260903 import (  # noqa: E402
    build_renderer_seg_pose_coupling_shipped_object_v1,
)

_NOTES = {
    "renderer_seg_pose_coupling_shipped_object_v1": (
        "ddm_eq1 equations-leg backfill. Two independent arms measured the same coupling "
        "on the shipped semantic renderer and neither reached the equations leg: rf1 "
        "166.80837961844966 (DERIVED from its published component table, un-retrained "
        "structural swap) and ft1 217.30366224024704 (MEASURED, retained "
        "verdict_ft1_step600.json, trained seg-only fine-tune). The law closes the "
        "seg-only renderer formulation by arithmetic at BOTH band ends; JOINT "
        "pose-priced formulations stay OUT of domain and OPEN. Advisory axis; "
        "NON-PROMOTABLE; pointer unmoved."
    ),
    "annulus_restricted_prefix_bias_detector_v1": (
        "ddm_eq1 equations-leg backfill. The third costume of [[m88]], after the TIME "
        "axis (wallclock_fixed_cost_prefix_bias_v1) and the SEED axis "
        "(seed_ensemble_falsifier_band_v1): a restricted-set statistic inherits a prefix "
        "bias a global-statistic check cannot see (+11.698% annulus vs +0.451% "
        "all-pixel, 25.94x). Positive control bit-identical, so the deviation is 100% "
        "cohort and 0% instrument. APPARATUS law; NON-PROMOTABLE; pointer unmoved."
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build + validate but do NOT append to the registry",
    )
    args = ap.parse_args(argv)

    equations = [
        build_renderer_seg_pose_coupling_shipped_object_v1(),
        build_annulus_restricted_prefix_bias_detector_v1(),
    ]
    for eq in equations:
        print(f"built {eq.equation_id}: {eq.one_line_summary}")
        print(f"  producers={eq.canonical_producers}")
        print(f"  consumers={eq.canonical_consumers}")
        print(f"  anchors={[anchor.anchor_id for anchor in eq.empirical_anchors]}")
    if args.dry_run:
        print("DRY-RUN: not registered.")
        return 0
    for eq in equations:
        register_canonical_equation(
            eq,
            agent="ddm_eq1_equations_leg_backfill_20260904",
            notes=_NOTES[eq.equation_id],
        )
        print(f"registered {eq.equation_id} into .omx/state/canonical_equations_registry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
