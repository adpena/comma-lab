#!/usr/bin/env python
"""ddm_md4 — the TERMINAL READ of md3's fired different-START burn cell.

md1's instrument (``experiments/ddm_md1_micro_to_macro.py``) and md2's overlap tool
(``experiments/ddm_md2_persistent_site_overlap.py``) do the measurement; both are used
UNCHANGED.  This module adds only the two things neither of them can supply:

``--mode resume-control``
    The cell was killed by a wall-clock cap at step 4,552 and resumed from
    ``periodic_004544.pt``.  The partition is read off checkpoints that straddle that
    boundary, so the boundary is a named confound until it is measured.  This mode reads the
    retained checkpoints around it and reports (a) whether ``config_identity_sha256`` is
    IDENTICAL on both sides — the trainer's own identity receipt — and (b) the per-16-step
    relative L2 displacement of the EMA shadow and of the live weights across the boundary
    against the local distribution of the same quantity inside each segment.  A boundary step
    that is an ordinary-sized step is continuous in the only sense the partition uses.

``--mode verdict``
    Reads md1's ``ANALYSIS_*.json`` and md2's ``OVERLAP_*.json`` for the new cell and its two
    comparators and emits the verdict arithmetic against the PRE-REGISTERED thresholds of
    ``.omx/research/charters/ddm_md3_different_initialisation_cell_20260905.md``
    (J >= 0.70 -> DATA-ANCHORED; J <= 0.45 -> INIT-ANCHORED; else INDETERMINATE), with the
    persistent-set Jaccard ceiling RE-DERIVED from the measured step-0 pools rather than
    copied from md3's memo, and with the ceiling's own size assumption checked against the
    now-measured persistent set.

$0.  CPU only.  Reads retained payloads; writes only under ``--out-store``.  No score claim.
Every number carries the md1/md2 axis: reconstructed from retained checkpoints, n32 sealed
selection, NOT contest authority.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "ddm_md4_terminal_read.v1"
AXIS = (
    "[macOS-CPU advisory; site sets reconstructed by ddm_md1_micro_to_macro from retained "
    "16-step checkpoints; n32 sealed stratified selection; not contest authority]"
)
DATA_ANCHORED_AT_OR_ABOVE = 0.70
INIT_ANCHORED_AT_OR_BELOW = 0.45
# The persistent-set Jaccard ceiling md3's memo §4 reported for the r10_live rung.  Cited for
# comparison only; this module re-derives the value from the measured step-0 pools and reports
# whether the citation still agrees.  It is never used as an input to any other number.
MD3_MEMO_CEILING = 0.8082


class MD4Error(RuntimeError):
    """ddm_md4 refuses rather than emitting an unlabelled or unfaithful number."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MD4Error(f"required artifact absent: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def _custody(elapsed_s: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "contest_eval_invocations": 0,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "elapsed_s": elapsed_s,
        "peak_rss_gib": _peak_rss_gib(),
        "host": {"platform": platform.platform()},
    }


# --------------------------------------------------------------------------- resume control
def _relative_l2(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Relative L2 between two flat state dicts; refuses unless their key sets are EQUAL."""

    import torch

    if set(a) != set(b):
        # A silent set-intersection here would understate the displacement by dropping the
        # keys that differ, which is exactly the confound this control exists to detect.
        missing = sorted(set(a) ^ set(b))[:8]
        raise MD4Error(f"state dicts have different tensor key sets; symmetric difference {missing}")
    keys = sorted(a)
    if not keys:
        raise MD4Error("state dicts share no tensor keys")
    num = 0.0
    den = 0.0
    for key in keys:
        ta, tb = a[key], b[key]
        if not hasattr(ta, "shape") or tuple(ta.shape) != tuple(tb.shape):
            raise MD4Error(f"tensor {key} shape mismatch across checkpoints")
        da = ta.detach().to(torch.float64).reshape(-1)
        db = tb.detach().to(torch.float64).reshape(-1)
        num += float(((da - db) ** 2).sum())
        den += float((db**2).sum())
    if den <= 0.0:
        raise MD4Error("reference norm is zero; relative L2 undefined")
    return math.sqrt(num / den)


def run_resume_control(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    ckpt_dir = Path(args.checkpoint_dir)
    boundary = int(args.boundary_step)
    every = int(args.checkpoint_every)
    window = int(args.window)

    steps = [boundary + every * k for k in range(-window, window + 1)]
    steps = [s for s in steps if s > 0]
    loaded: dict[int, dict[str, Any]] = {}
    for step in steps:
        path = ckpt_dir / f"periodic_{step:06d}.pt"
        if not path.is_file():
            raise MD4Error(f"checkpoint absent, cannot run the boundary control: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["completed_steps"]) != step:
            raise MD4Error(
                f"{path.name} declares completed_steps={payload['completed_steps']}, expected {step}"
            )
        loaded[step] = {
            "path": str(path),
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
            "config_identity_sha256": payload["config_identity_sha256"],
            "live": payload["live_state_dict"],
            "shadow": payload["ema"]["shadow"],
            "ema_num_updates": int(payload["ema"]["num_updates"]),
        }

    identities = {loaded[s]["config_identity_sha256"] for s in steps}
    rows = []
    for lo, hi in pairwise(steps):
        rows.append(
            {
                "from_step": lo,
                "to_step": hi,
                "crosses_resume_boundary": lo <= boundary < hi,
                "relative_l2_shadow": _relative_l2(loaded[hi]["shadow"], loaded[lo]["shadow"]),
                "relative_l2_live": _relative_l2(loaded[hi]["live"], loaded[lo]["live"]),
            }
        )

    interior = [r for r in rows if not r["crosses_resume_boundary"]]
    crossing = [r for r in rows if r["crosses_resume_boundary"]]
    if len(crossing) != 1:
        raise MD4Error(f"expected exactly one boundary-crossing interval, got {len(crossing)}")

    def _ratio(field: str) -> float | None:
        others = [r[field] for r in interior]
        if not others:
            return None
        mean = sum(others) / len(others)
        return crossing[0][field] / mean if mean > 0 else None

    receipt = {
        **_custody(time.perf_counter() - started),
        "mode": "resume-control",
        "checkpoint_dir": str(ckpt_dir),
        "boundary_step": boundary,
        "checkpoint_every_steps": every,
        "steps_read": steps,
        "config_identity_sha256_unique_values": sorted(identities),
        "config_identity_identical_across_boundary": len(identities) == 1,
        "ema_num_updates_by_step": {str(s): loaded[s]["ema_num_updates"] for s in steps},
        "mtime_utc_by_step": {str(s): loaded[s]["mtime_utc"] for s in steps},
        "intervals": rows,
        "boundary_over_interior_mean_shadow": _ratio("relative_l2_shadow"),
        "boundary_over_interior_mean_live": _ratio("relative_l2_live"),
        "interpretation_note": (
            "A boundary interval whose displacement matches the interior intervals, with an "
            "identical config identity and a monotone EMA update counter, is continuous in the "
            "only sense the trajectory partition uses. This is a control, not a bit-identity "
            "proof: bit identity would require re-running the killed segment."
        ),
    }
    out = Path(args.out_store) / "RESUME_BOUNDARY_CONTROL.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt["receipt_path"] = str(out)
    return receipt


# --------------------------------------------------------------------------------- verdict
def _persistent_block(analysis: dict[str, Any], forward: str) -> dict[str, Any]:
    block = analysis["forwards"][forward]
    persistent = block["classes"]["PERSISTENT"]
    return {
        "sites": persistent["sites"],
        "terminal_wrong_sites": persistent["terminal_wrong_sites"],
        "terminal_share_of_error": persistent["terminal_share_of_error"],
        "terminal_d_seg_contribution": persistent["terminal_d_seg_contribution"],
        "terminal_d_seg_hat": block["terminal_d_seg_hat"],
        "calibration_gate_max_abs_integer_residual": block["bridge"][
            "calibration_gate_max_abs_integer_residual"
        ],
    }


def _verdict_word(jaccard: float) -> str:
    if jaccard >= DATA_ANCHORED_AT_OR_ABOVE:
        return "DATA-ANCHORED"
    if jaccard <= INIT_ANCHORED_AT_OR_BELOW:
        return "INIT-ANCHORED"
    return "INDETERMINATE"


def run_verdict(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    store = Path(args.store)
    lineage = args.gt_lineage
    new_cell = args.cell
    cold_cell = args.cold_cell
    order_cell = args.order_cell

    new_analysis = _read_json(store / f"ANALYSIS_{new_cell}_{lineage}.json")
    cold_analysis = _read_json(Path(args.cold_store) / f"ANALYSIS_{cold_cell}_{lineage}.json")
    order_analysis = _read_json(Path(args.order_store) / f"ANALYSIS_{order_cell}_{lineage}.json")
    overlap_cold = _read_json(
        store / f"OVERLAP_{new_cell}_vs_{cold_cell}_{lineage}.json"
    )
    overlap_order = _read_json(
        store / f"OVERLAP_{new_cell}_vs_{order_cell}_{lineage}.json"
    )
    step0 = _read_json(Path(args.step0_overlap))

    candidates = {
        row["candidate_id"]: row for row in step0["blocks"][lineage]["candidates"]
    }
    if args.start_candidate not in candidates:
        raise MD4Error(f"start candidate {args.start_candidate!r} absent from the step-0 receipt")
    start = candidates[args.start_candidate]
    incumbent = step0["blocks"][lineage]["incumbent"]

    pool_new = start["pool_vs_incumbent"]["a_sites"]
    pool_cold = incumbent["pool_sites"]
    persistent_cold_step0 = incumbent["persistent_sites"]
    intersection_cap = start["incumbent_persistent_still_wrong_here"]

    share_cold = persistent_cold_step0 / pool_cold
    assumed_persistent_new = int(pool_new * share_cold)
    ceiling_assumed_share = intersection_cap / (
        persistent_cold_step0 + assumed_persistent_new - intersection_cap
    )
    ceiling_unconstrained_size = intersection_cap / persistent_cold_step0

    forwards: dict[str, Any] = {}
    for forward in ("shadow", "live"):
        new_block = _persistent_block(new_analysis, forward)
        measured_persistent_new = new_block["sites"]
        # The step-0 receipt's pool and its |P_cold cap pool_new| cap are read off the SHADOW
        # forward of the cold control (11,842 sites, DALI).  The live forward has a different
        # persistent set, so mixing that cap with a live-forward size would be a unit error.
        # The ceiling is therefore emitted for the shadow forward only.
        ceiling_measured_size = (
            intersection_cap / (persistent_cold_step0 + measured_persistent_new - intersection_cap)
            if forward == "shadow"
            else None
        )
        cold_overlap = overlap_cold["forwards"][forward]["overlap"]["PERSISTENT"]
        order_overlap = overlap_order["forwards"][forward]["overlap"]["PERSISTENT"]
        forwards[forward] = {
            "new_cell": new_block,
            "cold_control": _persistent_block(cold_analysis, forward),
            "data_order_control": _persistent_block(order_analysis, forward),
            "jaccard_vs_cold": cold_overlap["jaccard"],
            "jaccard_vs_data_order": order_overlap["jaccard"],
            "intersection_vs_cold": cold_overlap["intersection_sites"],
            "intersection_over_chance_vs_cold": cold_overlap["intersection_over_chance"],
            "new_covered_by_cold_fraction": cold_overlap["a_covered_by_b_fraction"],
            "cold_covered_by_new_fraction": cold_overlap["b_covered_by_a_fraction"],
            "lane_touching_share_of_terminal_wrong": overlap_cold["forwards"][forward]["lane_a"][
                "lane_touching_share_of_terminal_wrong"
            ],
            "gt_lane_enrichment": overlap_cold["forwards"][forward]["lane_a"]["gt_lane_enrichment"],
            "overpaint_birth_step_by_class": overlap_cold["birth"][
                "overpaint_birth_step_by_class"
            ]["a"][forward],
            "excursion_peak_step": overlap_cold["forwards"][forward]["a_peak_step"],
            "jaccard_ceiling_with_measured_persistent_size": ceiling_measured_size,
            "jaccard_ceiling_note": (
                "re-derived from the measured step-0 pools"
                if forward == "shadow"
                else "not emitted: the step-0 cap is shadow-referenced; see jaccard_ceiling_rederived"
            ),
            "verdict_word": _verdict_word(cold_overlap["jaccard"]),
        }

    authority = forwards[args.authority_forward]
    receipt = {
        **_custody(time.perf_counter() - started),
        "mode": "verdict",
        "gt_lineage": lineage,
        "authority_forward": args.authority_forward,
        "cells": {"new": new_cell, "cold": cold_cell, "data_order": order_cell},
        "pre_registered_thresholds": {
            "data_anchored_at_or_above": DATA_ANCHORED_AT_OR_ABOVE,
            "init_anchored_at_or_below": INIT_ANCHORED_AT_OR_BELOW,
            "predicted_persistent_share_band": [0.55, 0.65],
            "charter": ".omx/research/charters/ddm_md3_different_initialisation_cell_20260905.md",
        },
        "step0_inputs": {
            "start_candidate": args.start_candidate,
            "start_state_sha256": start["state_sha256"],
            "relative_l2_from_incumbent": start["relative_l2_from_incumbent"],
            "pool_sites_new": pool_new,
            "pool_sites_cold": pool_cold,
            "pool_jaccard": start["pool_vs_incumbent"]["jaccard"],
            "cold_persistent_sites_step0_reference": persistent_cold_step0,
            "cold_persistent_still_wrong_at_new_start": intersection_cap,
            "cold_persistent_containment_at_new_start": start["incumbent_persistent_containment"],
        },
        "jaccard_ceiling_rederived": {
            "definition": (
                "max J of the two PERSISTENT sets given the measured step-0 pools: the "
                "intersection cannot exceed |P_cold cap pool_new|, and the union is at least "
                "|P_cold| + |P_new| - that cap"
            ),
            "with_assumed_persistent_share_of_pool": ceiling_assumed_share,
            "assumed_persistent_new_sites": assumed_persistent_new,
            "assumed_share_of_pool": share_cold,
            "with_unconstrained_persistent_size": ceiling_unconstrained_size,
            "md3_memo_reported": MD3_MEMO_CEILING,
            "rederivation_matches_md3_memo_to_4dp": (
                round(ceiling_assumed_share, 4) == MD3_MEMO_CEILING
            ),
        },
        "forwards": forwards,
        "verdict": {
            "jaccard_vs_cold": authority["jaccard_vs_cold"],
            "word": authority["verdict_word"],
            "persistent_share_of_terminal_error": authority["new_cell"][
                "terminal_share_of_error"
            ],
            "persistent_share_inside_predicted_band": (
                0.55 <= authority["new_cell"]["terminal_share_of_error"] <= 0.65
            ),
        },
    }
    out = store / f"TERMINAL_READ_{lineage}.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt["receipt_path"] = str(out)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("resume-control", "verdict"), required=True)
    parser.add_argument("--out-store", default=None, help="resume-control: the only directory written")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--boundary-step", type=int, default=4544)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--store", default=None, help="verdict: md4's store (holds the new cell)")
    parser.add_argument("--cell", default="seed_20260902_different_init_r10_live_control_native100")
    parser.add_argument("--cold-store", default="/Volumes/APDataStore/pact/ddm_md1_micro_macro")
    parser.add_argument("--cold-cell", default="cold_control_seed_20260902")
    parser.add_argument(
        "--order-store", default="/Volumes/APDataStore/pact/ddm_md3_different_initialisation"
    )
    parser.add_argument("--order-cell", default="data_order_control_seed_20260903")
    parser.add_argument(
        "--step0-overlap",
        default="/Volumes/APDataStore/pact/ddm_md3_different_initialisation/STEP0_POOL_OVERLAP.json",
    )
    parser.add_argument("--start-candidate", default="r10_live")
    parser.add_argument("--gt-lineage", choices=("dali", "pyav"), default="dali")
    parser.add_argument("--authority-forward", choices=("shadow", "live"), default="shadow")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "resume-control":
        if not args.checkpoint_dir or not args.out_store:
            raise MD4Error("--mode resume-control requires --checkpoint-dir and --out-store")
        receipt = run_resume_control(args)
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in (
                        "receipt_path",
                        "config_identity_identical_across_boundary",
                        "boundary_over_interior_mean_shadow",
                        "boundary_over_interior_mean_live",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        if not args.store:
            raise MD4Error("--mode verdict requires --store")
        receipt = run_verdict(args)
        print(json.dumps({"receipt_path": receipt["receipt_path"], **receipt["verdict"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
