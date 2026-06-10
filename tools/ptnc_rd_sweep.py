#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PTNC RD sweep — d_pose vs bytes for {dense, ptnc} anchors across capacity + frame slot (task #61).

Produces the RD curve the verdict memo reports: for each (anchor_mode, capacity) it trains the carrier
and records (carrier_bytes, exact d_pose, d_pose_per_kb). The headline comparison is PTNC vs dense at
matched capacity — does the Jacobian-saliency anchor reduce d_pose-per-byte (break the 0.0036 ceiling)?

Also computes the low-res-GT-luma reference point (the #56 hint, d_pose 0.0007 at higher byte): a
factor-K box-downsample of GT frame0 luma, byte-counted (brotli of the downsampled uint8), bilinear
upsample, exact d_pose. This is the non-learned RD baseline PTNC must beat per byte.

Authority ``[local CPU-torch advisory]``. NO MPS. $0. Non-promotable. NO-FAKE: every d_pose is the exact
PoseNet MSE on the decoded frame; every byte cost is brotli of the actual payload.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
for _p in (REPO_ROOT, REPO_ROOT / "src", _HARNESS, REPO_ROOT / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAMERA_H, CAMERA_W = 874, 1164
_CONTEST_TOTAL_BYTES = 37_545_489

# capacity grid (hidden_dim, mod_dim) — small/tiny only (the #57 non-monotone finding: bigger != better).
_CAPACITY = {
    "tiny": {"hidden_dim": 48, "mod_dim": 16, "n_fourier": 24, "n_hidden": 3},
    "small": {"hidden_dim": 64, "mod_dim": 24, "n_fourier": 24, "n_hidden": 3},
    "mid": {"hidden_dim": 96, "mod_dim": 32, "n_fourier": 32, "n_hidden": 3},
}


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lowres_gt_point(gt_pairs, pairs, scorer, factor: int, frame_slot: int) -> dict[str, Any]:
    """Non-learned RD reference: box-downsample GT frame luma by ``factor``, byte-count, bilinear up."""
    import brotli
    import render_and_score_lib as L
    import torch
    import torch.nn.functional as F

    d_pose_list: list[float] = []
    payload_chunks: list[bytes] = []
    for pi in pairs:
        gt = gt_pairs[pi][frame_slot]  # (H,W,3) uint8
        chw = gt.float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        hh, ww = CAMERA_H // factor, CAMERA_W // factor
        low = F.interpolate(chw, size=(hh, ww), mode="area")
        low_u8 = low.clamp(0, 255).round().to(torch.uint8)
        payload_chunks.append(low_u8.numpy().tobytes())
        up = F.interpolate(low_u8.float(), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        rec = up[0]  # (3,H,W)
        other = gt_pairs[pi][1 - frame_slot].float().permute(2, 0, 1)
        comp = torch.stack([rec, other]) if frame_slot == 0 else torch.stack([other, rec])
        gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
        pose_d, _ = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
        d_pose_list.append(float(pose_d[0]))
    blob = brotli.compress(b"".join(payload_chunks), quality=11)
    total_bytes = len(blob)
    mean_dp = float(np.mean(d_pose_list))
    return {
        "method": f"lowres_gt_f{factor}", "factor": factor, "frame_slot": frame_slot,
        "carrier_bytes": total_bytes, "mean_d_pose": mean_dp,
        "rate_term": 25.0 * total_bytes / _CONTEST_TOTAL_BYTES,
        "pose_term_sqrt10": float(np.sqrt(10.0 * mean_dp)),
        "d_pose_per_kb": float(mean_dp / max(1.0, total_bytes / 1024.0)),
        "per_pair_d_pose": d_pose_list,
    }


def main(argv: list[str] | None = None) -> int:
    from ptnc_train_pose_carrier import train  # the sweep reuses the single trainer

    from tac.boundary_math.amortized_luma_carrier import LumaCarrierConfig

    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frame-slot", type=int, default=0, choices=(0, 1))
    ap.add_argument("--modes", default="dense,ptnc")
    ap.add_argument("--capacities", default="tiny,small")
    ap.add_argument("--lowres-factors", default="8,4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    caps = [c.strip() for c in args.capacities.split(",") if c.strip()]
    factors = [int(x) for x in args.lowres_factors.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for cap in caps:
        for mode in modes:
            knobs = _CAPACITY[cap]
            cfg = LumaCarrierConfig(num_pairs=args.n_pairs, quant_bits=8, **knobs)
            sub = args.out_dir / f"slot{args.frame_slot}_{mode}_{cap}"
            res = train(
                args.targets_dir, sub, cfg, n_pairs=args.n_pairs, epochs=args.epochs, lr=3e-3,
                frame_slot=args.frame_slot, anchor_mode=mode, anchor_floor=0.02, anchor_gamma=1.0,
                pose_weight=50.0, anchor_weight=1.0, seed=args.seed, eval_every=max(20, args.epochs),
            )
            rows.append({
                "method": f"carrier_{mode}", "capacity": cap, "anchor_mode": mode,
                "frame_slot": args.frame_slot, "carrier_bytes": res["byte_account"]["total_bytes"],
                "mean_d_pose": res["exact_mean_d_pose"], "rate_term": res["rate_term_carrier_only"],
                "pose_term_sqrt10": res["pose_term_contribution_sqrt10"],
                "d_pose_per_kb": res["d_pose_per_kb"], "param_count": res["param_count"],
                "parity_pass": res["portability_parity"]["parity_pass"],
            })
            print(f"[sweep] {mode}/{cap} d_pose={res['exact_mean_d_pose']:.6f} "
                  f"bytes={res['byte_account']['total_bytes']} dp/kb={res['d_pose_per_kb']:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # low-res GT reference points (non-learned baseline).
    import render_and_score_lib as L
    meta = json.loads((args.targets_dir / "targets_meta.json").read_text())
    pairs = list(range(min(args.n_pairs, int(meta["num_pairs_built"]))))
    gt_pairs = L.decode_gt_pairs(pairs)
    scorer = L.ExactScorer()
    for f in factors:
        pt = _lowres_gt_point(gt_pairs, pairs, scorer, f, args.frame_slot)
        rows.append(pt)
        print(f"[sweep] lowres_gt_f{f} d_pose={pt['mean_d_pose']:.6f} bytes={pt['carrier_bytes']} "
              f"dp/kb={pt['d_pose_per_kb']:.6f}", flush=True)

    out = {
        "subagent": "task61_ptnc_frame1_dual_fidelity", "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]", "promotion_eligible": False, "score_claim": False,
        "n_pairs": args.n_pairs, "epochs": args.epochs, "frame_slot": args.frame_slot,
        "rows": rows, "wall_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "rd_sweep.json").write_text(json.dumps(out, indent=2))
    print("\n=== RD SWEEP ===")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
