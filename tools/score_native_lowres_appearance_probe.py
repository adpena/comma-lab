#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reactivation-path probe: does a low-res appearance carrier recover pose AND hold seg?

The palette bridge collapses pose (d_pose 12.66 vs GT-frame1 floor 0.0). The verdict's
reactivation path is a pose-carrying appearance section. This probe measures, on the exact
CPU-torch scorer, the d_seg AND d_pose of a frame1 = bilinear-upsample(downsample(GT, factor))
across factors — the cheapest pose-preserving granularity — so the next build knows the
appearance section's rate-vs-(seg,pose) curve.

This is a DIAGNOSTIC (not the carrier itself): it uses GT-derived low-res appearance to bound
what a pose-carrying section costs. The generator's argmax (the seg carrier) is held; the
question is purely the appearance section's pose-recovery vs byte curve.

Authority ``[local CPU-torch advisory]`` — non-promotable. $0, no GPU, no MPS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
_UPSTREAM = REPO_ROOT / "upstream"
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(_HARNESS), str(_UPSTREAM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.bitmask_dseg import d_seg_reference  # noqa: E402
from tac.boundary_math.legal_frame_bridge import (  # noqa: E402
    lowres_appearance_carrier,
    upsample_appearance,
)

CAMERA_H, CAMERA_W = 874, 1164
_CONTEST_TOTAL_BYTES = 37_545_489


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _segnet():
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg = SegNet().eval().to("cpu")
    seg.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    return seg


def run(targets_dir: Path, out_dir: Path, n_pairs: int, factors: list[int]) -> dict[str, Any]:
    import render_and_score_lib as L

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    H, W = meta["seg_input_hw"]
    gt_argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r",
                          shape=(n_built, H, W))
    pairs = list(range(min(n_pairs, n_built)))
    segnet = _segnet()
    scorer = L.ExactScorer()
    gt_pairs = L.decode_gt_pairs(pairs)

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    rows = []
    for factor in factors:
        d_seg_list, d_pose_list, bytes_list = [], [], []
        for pi in pairs:
            gt1 = np.asarray(gt_pairs[pi][1])  # (camera,3) GT frame1
            gt0 = np.asarray(gt_pairs[pi][0])
            lowres, coded = lowres_appearance_carrier(gt1, factor=factor)
            recon1 = upsample_appearance(lowres, CAMERA_H, CAMERA_W)  # (camera,3) uint8
            # seg: argmax of the upsampled low-res frame1 vs L*.
            a_recon, _ = measure_segnet_argmax(segnet, recon1)
            ls = np.asarray(gt_argmax[pi]).astype(np.int64)
            d_seg_list.append(float(d_seg_reference(a_recon.astype(np.int64), ls)))
            # pose: GT frame0 + reconstructed frame1.
            comp = torch.stack([
                torch.from_numpy(gt0.transpose(2, 0, 1)).float(),
                torch.from_numpy(recon1.transpose(2, 0, 1)).float(),
            ])
            gt_pair_t = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
            pose_d, _ = scorer.score_batch(gt_pair_t, L.comp_pair_to_bthwc(comp))
            d_pose_list.append(float(pose_d[0]))
            bytes_list.append(coded)
        # per-pair appearance bytes * n_pairs (carried once per pair).
        mean_bytes = float(np.mean(bytes_list))
        total_appearance = round(mean_bytes * 600)  # extrapolated 600-pair carrier
        rows.append({
            "factor": factor,
            "lowres_hw": [max(1, CAMERA_H // factor), max(1, CAMERA_W // factor)],
            "mean_d_seg": float(np.mean(d_seg_list)),
            "mean_d_pose": float(np.mean(d_pose_list)),
            "mean_appearance_bytes_per_pair": mean_bytes,
            "appearance_bytes_600pair": total_appearance,
            "rate_term_appearance_only": 25.0 * total_appearance / _CONTEST_TOTAL_BYTES,
        })
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in rows[-1].items()}), flush=True)

    result = {
        "subagent": "score_native_first_candidate_55_56",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "n_pairs": len(pairs),
        "wall_s": round(time.time() - t0, 1),
        "factor_sweep": rows,
        "note": ("DIAGNOSTIC: GT-derived low-res frame1 appearance carrier. Bounds the pose-vs-byte "
                 "curve for the pose-carrying appearance section the palette bridge lacked."),
        "provenance": {"axis_tag": "[local CPU-torch advisory]", "promotable": False,
                       "gt_decode": "frame_utils.yuv420_to_rgb"},
    }
    (out_dir / "lowres_appearance_probe.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "experiments/results/score_native_candidate_20260610")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--factors", type=int, nargs="*", default=[2, 4, 8, 16])
    args = ap.parse_args(argv)
    run(args.targets_dir, args.out_dir, args.n_pairs, args.factors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
