#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""frame1 dual-fidelity probe — the actual wall (task #61).

frame1 has the DUAL constraint: SegNet argmax (d_seg) AND PoseNet luma (d_pose). This probe measures
the JOINT (d_seg, d_pose, bytes) tradeoff for a sequence of frame1 representations, to find the byte
floor frame1 needs to hold BOTH terms — and whether that floor preserves the score-native rate
advantage or converges to HNeRV-class (the DEFER-to-lever-C gate).

Representations probed (frame1 only; frame0 = the best frame0 carrier OR GT0):
  * gt1                : GT frame1 (d_seg=0, d_pose=0) — the sanity ceiling.
  * lowres_gt_f{K}     : box-downsample GT1 by K, brotli byte-count, bilinear up. Sweeps the luma
                         fidelity vs byte curve. d_seg measured (does coarse luma flip argmax?) +
                         d_pose measured.
  * palette            : the #57 dead end (seg-argmax label map → fixed RGB per class). d_seg low,
                         d_pose ~12 (pose-blind).

Each row reports the FULL candidate S = 100*d_seg + sqrt(10*d_pose) + 25*B/D using the chosen frame0.
Authority [local CPU-torch advisory]. NO MPS. $0. Non-promotable. NO-FAKE: exact DistortionNet d_seg
AND d_pose on the decoded frames; brotli byte counts of the actual payloads.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
for _p in (REPO_ROOT, REPO_ROOT / "src", _HARNESS, REPO_ROOT / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAMERA_H, CAMERA_W = 874, 1164
_CONTEST_TOTAL_BYTES = 37_545_489


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _score_pair(scorer, L, gt_pairs, pi, f0_chw, f1_chw):
    import torch

    comp = torch.stack([f0_chw, f1_chw])
    gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
    pose_d, seg_d = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
    return float(pose_d[0]), float(seg_d[0])


def _lowres_frame(gt_chw, factor, *, luma_only=False):
    """Box-downsample (3,H,W) GT frame by factor, bilinear up; return (rec_chw, brotli_bytes)."""
    import brotli
    import torch
    import torch.nn.functional as F

    chw = gt_chw.unsqueeze(0)  # (1,3,H,W)
    hh, ww = CAMERA_H // factor, CAMERA_W // factor
    low = F.interpolate(chw, size=(hh, ww), mode="area")
    low_u8 = low.clamp(0, 255).round().to(torch.uint8)
    blob = brotli.compress(low_u8.numpy().tobytes(), quality=11)
    up = F.interpolate(low_u8.float(), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    return up[0], len(blob)


def main(argv: list[str] | None = None) -> int:
    import render_and_score_lib as L
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--lowres-factors", default="16,8,4,2")
    # the best frame0 carrier (npz) to compose with; if absent, use GT0 (isolates frame1).
    ap.add_argument("--frame0-carrier-npz", type=Path, default=None)
    ap.add_argument("--frame0-bytes", type=int, default=23572)  # the dense frame0 carrier byte cost
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.targets_dir / "targets_meta.json").read_text())
    pairs = list(range(min(args.n_pairs, int(meta["num_pairs_built"]))))
    gt_pairs = L.decode_gt_pairs(pairs)
    scorer = L.ExactScorer()
    factors = [int(x) for x in args.lowres_factors.split(",") if x.strip()]

    # frame0: GT0 (isolate frame1's dual constraint) — and separately note the carrier byte cost.
    f0_of = {pi: gt_pairs[pi][0].float().permute(2, 0, 1) for pi in pairs}
    f0_extra_bytes = 0
    f0_note = "GT0 (frame1 isolation)"
    if args.frame0_carrier_npz and Path(args.frame0_carrier_npz).exists():
        from tac.boundary_math.amortized_luma_carrier import (
            build_coords,
            carrier_frame,
            load_carrier_npz,
            measure_carrier_bytes,
        )
        params, cfg = load_carrier_npz(args.frame0_carrier_npz)
        coords = build_coords(CAMERA_H, CAMERA_W)
        for j, pi in enumerate(pairs):
            fc = carrier_frame(params, cfg, coords, j, CAMERA_H, CAMERA_W)
            f0_of[pi] = torch.from_numpy(fc.transpose(2, 0, 1)).float()
        f0_extra_bytes = measure_carrier_bytes(params, cfg).total_bytes
        f0_note = f"frame0 carrier {Path(args.frame0_carrier_npz).name}"
    else:
        f0_extra_bytes = args.frame0_bytes  # account the carrier rate even if using GT0 for f0 pixels

    rows: list[dict[str, Any]] = []

    def _row(method, f1_fn, f1_bytes):
        dps, dss = [], []
        for pi in pairs:
            f1 = f1_fn(pi)
            dp, ds = _score_pair(scorer, L, gt_pairs, pi, f0_of[pi], f1)
            dps.append(dp)
            dss.append(ds)
        mdp, mds = float(np.mean(dps)), float(np.mean(dss))
        total_bytes = int(f1_bytes + f0_extra_bytes)
        S = 100.0 * mds + float(np.sqrt(10.0 * mdp)) + 25.0 * total_bytes / _CONTEST_TOTAL_BYTES
        r = {"method": method, "mean_d_pose": mdp, "mean_d_seg": mds,
             "frame1_bytes": int(f1_bytes), "frame0_bytes": int(f0_extra_bytes),
             "total_bytes": total_bytes, "seg_term": 100.0 * mds,
             "pose_term": float(np.sqrt(10.0 * mdp)), "rate_term": 25.0 * total_bytes / _CONTEST_TOTAL_BYTES,
             "S": S}
        rows.append(r)
        print(f"[f1] {method:20s} d_seg={mds:.5f} d_pose={mdp:.5f} f1B={int(f1_bytes):>8d} S={S:.4f}",
              flush=True)
        return r

    # gt1 sanity (d_seg=0, d_pose=0; bytes only the f0 carrier).
    _row("gt1", lambda pi: gt_pairs[pi][1].float().permute(2, 0, 1), 0)

    # lowres GT1 sweep — the luma-vs-byte curve; both d_seg AND d_pose measured.
    for f in factors:
        cache = {}
        bcache = []
        for pi in pairs:
            rec, b = _lowres_frame(gt_pairs[pi][1].float().permute(2, 0, 1), f)
            cache[pi] = rec
            bcache.append(b)
        _row(f"lowres_gt1_f{f}", lambda pi, c=cache: c[pi], int(np.mean(bcache)) * len(pairs)
             if False else int(np.sum(bcache)))

    out = {
        "subagent": "task61_ptnc_frame1_dual_fidelity", "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]", "promotion_eligible": False, "score_claim": False,
        "n_pairs": len(pairs), "frame0_note": f0_note, "frame0_bytes_accounted": f0_extra_bytes,
        "rows": rows,
    }
    (args.out_dir / "frame1_dual_fidelity.json").write_text(json.dumps(out, indent=2))
    print("\n=== FRAME1 DUAL-FIDELITY RD ===")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
