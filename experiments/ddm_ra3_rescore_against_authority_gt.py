#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-RA3: re-score every retained ra3 candidate against the AUTHORITY-TRACKING GT, and re-do
the per-pair acceptance on that quantity -- at zero additional PoseNet forwards.

WHY THIS COSTS NOTHING.  ``ddm_ra3_subspace_trust_region_refit.py`` persisted the generated
``pose6`` for all 600 pairs x 12 candidates (``ra3_r11_pose6_by_mu.float64.npy``) plus both
quantized rows, because ALWAYS-KEEP-THE-PAYLOAD is P0.  ``d_pose`` is a pure function of
(generated pose6, GT pose6), so swapping the GT is arithmetic.  Had this arm persisted only the
scalar ``d_pose`` values -- the exact measure-and-discard defect the rule forbids -- the
instrument fix would have cost a 40-minute re-run of 8,400 scorer forwards.

THE FIX, and why the old numbers needed it (``pi2``, landed ed153d0203).  The advisory
instrument reads TWO GROUND TRUTHS: seg loads a cached DALI/nvdec-lineage argmax, while pose has
no cache and re-decodes with PyAV every run.  One instrument, two lineages -- that is the entire
~21x pose offset.  Scorer-forward CPU-vs-T4 drift is 3.572e-12 and is falsified as the cause.
The fix is not a conversion factor (the implied factor is NOT constant -- 7.55x to 11.45x across
rn1's rows) but a FIXED REFERENCE: ``gt_cache_dali.pt["pose"]``, MEASURED to track the contest
authority at 1.00081x.

WHAT CHANGES BEYOND THE NUMBERS.  The per-pair realised acceptance in the parent tool chose each
pair's trust-region radius by the measured error against the PyAV GT.  On the authority-tracking
GT the argmin can land on a different radius, so the acceptance is RE-DONE here rather than
re-labelled -- and the accepted coefficient payload is re-assembled and re-persisted to match.

Both scorings are reported. The PyAV column keeps ra3's cell commensurable with ra2's 2x2 (all
four of those cells were measured against the PyAV GT, so the offset is common and the
within-table comparisons are sound). The DALI column is the one that may be priced.

Axis: PyAV column [macOS-CPU advisory]; DALI column [authority-tracking, 1.00081x].
score_claim=false, promotable=false. No scorer runs, no Modal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PI2_SOURCE = REPO / "experiments/ddm_pi2_pose_axis_attribution.py"
RA3_RETAINED = Path("/Volumes/APDataStore/pact/ddm_ra3/retained")
JC1_RETAINED = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")

#: MEASURED by pi2: tracks the contest authority at 1.00081x.
GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
GT_CACHE_DALI_SHA_PIN = "a91d9825"

FRAMES, POSE_DIMS = 600, 6
S_PER_BYTE = 25.0 / 37_545_489.0
CARRIER_BYTES, CARRIER_DIM = 22_161, 12


class RescoreRefusal(RuntimeError):
    """Fail-closed refusal: missing custody, wrong shape, or a broken control."""


def _load_pi2():
    spec = importlib.util.spec_from_file_location("ddm_pi2", PI2_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_pi2"] = module
    spec.loader.exec_module(module)
    return module


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def retain_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    _atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def d_pose(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """upstream's pose distortion; identical convention to pi2 stage_crossaxis (mean over dims)."""
    return float(((generated - ground_truth) ** 2).mean(axis=1).mean())


def dS_pose(candidate: float, base: float) -> float:
    """Score delta of the pose term between two d_pose values on the SAME axis."""
    return float(np.sqrt(10.0 * candidate) - np.sqrt(10.0 * base))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ra3-retained", type=Path, default=RA3_RETAINED)
    parser.add_argument("--jc1-retained", type=Path, default=JC1_RETAINED)
    parser.add_argument("--gt-cache-dali", type=Path, default=GT_CACHE_DALI)
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--bytes-back", type=float, default=CARRIER_BYTES / CARRIER_DIM,
                        help="ra2's credit convention, kept so ratios stay comparable")
    parser.add_argument("--output", type=Path,
                        default=Path("/Volumes/APDataStore/pact/ddm_ra3"))
    args = parser.parse_args()

    pi2 = _load_pi2()
    import torch

    cache_path = Path(args.gt_cache_dali)
    if not cache_path.exists():
        raise RescoreRefusal(f"authority GT cache missing: {cache_path}")
    cache_sha = pi2.sha256_file(cache_path)
    if not cache_sha.startswith(GT_CACHE_DALI_SHA_PIN):
        raise RescoreRefusal(
            f"GT cache custody FAILED: {cache_sha} does not start {GT_CACHE_DALI_SHA_PIN}")
    gt_dali = torch.load(cache_path, map_location="cpu")["pose"].double().numpy()
    if gt_dali.shape != (FRAMES, POSE_DIMS):
        raise RescoreRefusal(f"authority GT shape {gt_dali.shape} != {(FRAMES, POSE_DIMS)}")

    gt_av = np.load(Path(args.jc1_retained) / "pose6_groundtruth.float64.npy")
    pose_base = np.load(Path(args.jc1_retained) / "pose6_generated.float64.npy")
    pose_all = np.load(Path(args.ra3_retained) / f"ra3_r{args.rank}_pose6_by_mu.float64.npy")
    if pose_all.shape[0] != FRAMES or pose_all.shape[2] != POSE_DIMS:
        raise RescoreRefusal(f"retained pose6 stack has shape {pose_all.shape}")
    n_cand = pose_all.shape[1]

    receipt_path = Path(args.output) / f"RA3_TRUST_REGION_REFIT_r{args.rank}.json"
    parent = json.loads(receipt_path.read_text())
    labels = [row["label"] for row in parent["per_candidate_measured"]]
    if len(labels) != n_cand:
        raise RescoreRefusal("retained pose6 stack does not match the parent receipt's labels")

    # CONTROL: re-scoring against the PyAV GT must reproduce the parent receipt exactly --
    # proves this tool's arithmetic is the parent's, so any change is the GT and nothing else.
    control = []
    for slot, label in enumerate(labels):
        here = d_pose(pose_all[:, slot, :], gt_av)
        there = parent["per_candidate_measured"][slot]["d_pose_measured"]
        control.append({"label": label, "reproduced": here, "parent": there,
                        "relative_error": abs(here - there) / there})
    worst = max(row["relative_error"] for row in control)
    if worst > 1e-12:
        raise RescoreRefusal(
            f"PyAV re-scoring does not reproduce the parent receipt (worst {worst:.3e})")

    base_av, base_dali = d_pose(pose_base, gt_av), d_pose(pose_base, gt_dali)
    credit = args.bytes_back * S_PER_BYTE

    rows = []
    for slot, label in enumerate(labels):
        av, dali = d_pose(pose_all[:, slot, :], gt_av), d_pose(pose_all[:, slot, :], gt_dali)
        rows.append({
            "label": label,
            "d_pose_pyav_advisory": av, "ratio_pyav": av / base_av,
            "d_pose_dali_authority": dali, "ratio_dali": dali / base_dali,
            "dS_pose_authority": dS_pose(dali, base_dali),
        })

    # RE-DO the acceptance on the authority-tracking quantity: the argmin can move.
    sq_dali = ((pose_all - gt_dali[:, None, :]) ** 2).mean(axis=2)      # (600, n_cand)
    slot_dali = sq_dali.argmin(axis=1)
    accepted_dali = float(sq_dali[np.arange(FRAMES), slot_dali].mean())
    sq_av = ((pose_all - gt_av[:, None, :]) ** 2).mean(axis=2)
    slot_av = sq_av.argmin(axis=1)
    moved = int((slot_dali != slot_av).sum())

    candidates = np.stack(
        [np.load(Path(args.ra3_retained) / f"ra3_r{args.rank}_{label}.float64.npy")
         for label in labels], axis=1)                                  # (600, n_cand, 12)
    accepted_coeff = candidates[np.arange(FRAMES), slot_dali]
    retained = {
        "accepted_on_authority_gt": retain_array(
            Path(args.ra3_retained) / f"ra3_r{args.rank}_accepted_authority_gt.float64.npy",
            accepted_coeff),
        "accepted_slot_authority_gt": retain_array(
            Path(args.ra3_retained) / f"ra3_r{args.rank}_accepted_slot_authority_gt.int32.npy",
            slot_dali.astype(np.int32)),
    }

    dS = dS_pose(accepted_dali, base_dali)
    receipt = {
        "arm": "ddm_ra3",
        "schema": "ddm_ra3_rescore_against_authority_gt.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "axis": "[authority-tracking GT, MEASURED 1.00081x vs contest-CUDA]",
        "score_claim": False,
        "promotable": False,
        "measurement_status": "RESCORED_FROM_RETAINED_POSE6_NO_NEW_FORWARDS",
        "gt_cache": {"path": str(cache_path), "sha256": cache_sha,
                     "bytes": cache_path.stat().st_size},
        "pyav_reproduction_control": {"worst_relative_error": worst, "rows": control},
        "base": {"d_pose_pyav_advisory": base_av, "d_pose_dali_authority": base_dali,
                 "pyav_over_dali": base_av / base_dali},
        "rank": args.rank, "bytes_back": args.bytes_back, "rate_credit_S": credit,
        "rows": rows,
        "accepted_on_authority_gt": {
            "d_pose": accepted_dali,
            "ratio_vs_base": accepted_dali / base_dali,
            "dS_pose": dS,
            "net_dS": dS - credit,
            "pose_cost_over_rate_credit": dS / credit,
            "breaks_even": bool(dS <= credit),
            "pairs_whose_accepted_radius_moved_vs_pyav": moved,
            "slot_histogram": {labels[s]: int((slot_dali == s).sum()) for s in range(n_cand)},
        },
        "retained": retained,
    }
    _atomic_bytes(
        Path(args.output) / f"RA3_AUTHORITY_GT_RESCORE_r{args.rank}.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())

    print(f"PyAV reproduction control: worst relative error {worst:.2e}  (must be <= 1e-12)")
    print(f"base d_pose: PyAV {base_av:.8f}   AUTHORITY-GT {base_dali:.8e}   "
          f"ratio {base_av / base_dali:.2f}x")
    print(f"\n{'candidate':>12s} {'d_pose PyAV':>13s} {'ratio':>8s} "
          f"{'d_pose AUTHORITY':>17s} {'ratio':>8s} {'dS_pose':>10s}")
    for row in rows:
        print(f"{row['label']:>12s} {row['d_pose_pyav_advisory']:13.8f} "
              f"{row['ratio_pyav']:7.3f}x {row['d_pose_dali_authority']:17.8e} "
              f"{row['ratio_dali']:7.3f}x {row['dS_pose_authority']:+10.6f}")
    acc = receipt["accepted_on_authority_gt"]
    print(f"\nACCEPTED on the AUTHORITY GT: d_pose {acc['d_pose']:.8e}  "
          f"{acc['ratio_vs_base']:.3f}x base")
    print(f"  dS_pose {acc['dS_pose']:+.6f}   rate credit {credit:+.6f}   "
          f"net dS {acc['net_dS']:+.6f}")
    print(f"  pose cost / rate credit = {acc['pose_cost_over_rate_credit']:.1f}x   "
          f"breaks even: {acc['breaks_even']}")
    print(f"  pairs whose accepted radius MOVED when the GT was fixed: {moved}/600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
