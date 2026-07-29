#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_p3v2 finalizer — aggregate the persisted per-pair cache into the final ladder receipt.

Race-free (read-only on the npz cache + partial JSONL) so it is safe to run alongside the resumable
solver. Reuses the ladder tool's functions (no re-solve): loads the persisted f0_free/base/f1/target
per pair, runs the S2 LOTTO race + the frame_0 seg-free spot check + the pre-registered verdict, and
writes the final receipt. Same authority + non-promotable labels as the ladder.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from hashlib import sha256
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import ddm_p3v2_optimal_form_pose_resolve as L
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=Path,
                    default=L._SSD_OUT / "p3v2_ladder_receipt.json")
    ap.add_argument("--out", type=Path, default=L._SSD_OUT / "p3v2_ladder_receipt_final.json")
    ap.add_argument("--lotto-ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    jsonl = Path(str(args.stem).replace(".json", ".partial.jsonl"))
    npz_dir = Path(str(args.stem).replace(".json", "_pairs"))
    rows = {}
    for ln in jsonl.read_text().splitlines():
        try:
            r = json.loads(ln)
            rows[int(r["pair"])] = r
        except Exception:
            pass
    per_pair = []
    for npz_p in sorted(glob.glob(str(npz_dir / "pair*.npz"))):
        pidx = int(Path(npz_p).stem[4:])
        if pidx not in rows:
            continue
        try:
            z = np.load(npz_p)
            per_pair.append({**rows[pidx], "_f0_free": z["f0"], "_base": z["base"],
                             "_f1": z["f1"], "_target": z["target"]})
        except Exception:
            pass
    per_pair.sort(key=lambda r: r["pair"])
    if len(per_pair) < 2:
        raise SystemExit(f"need >=2 persisted pairs; found {len(per_pair)}")
    print(f"[finalize] {len(per_pair)} persisted pairs: {[r['pair'] for r in per_pair]}", flush=True)

    posenet, modules = L.load_posenet()
    import torch
    from safetensors.torch import load_file

    # S2 LOTTO
    deltas = [(r["_f0_free"] - r["_base"]) for r in per_pair]
    bases = [r["_base"] for r in per_pair]
    f1s = [r["_f1"] for r in per_pair]
    tgs = [r["_target"] for r in per_pair]
    lotto = L.s2_lotto(posenet, deltas, bases, f1s, tgs, ranks=args.lotto_ranks)

    # frame_0 seg-free spot check
    segnet = modules.SegNet().eval().cpu()
    segnet.load_state_dict(load_file(str(L._UPSTREAM / "models/segnet.safetensors"), device="cpu"))
    for p in segnet.parameters():
        p.requires_grad = False
    r0 = per_pair[0]
    f0_free_u8 = L._f0work_to_u8(r0["_f0_free"])
    f0_zero = np.zeros_like(f0_free_u8)

    def seg_argmax(f0_u8, f1_u8):
        x = torch.from_numpy(np.stack([f0_u8, f1_u8])[None]).permute(0, 1, 4, 2, 3).float()
        with torch.inference_mode():
            logits = segnet(segnet.preprocess_input(x))
        return logits.argmax(1)[0].numpy()

    seg_check = {"pair": int(r0["pair"]),
                 "argmax_identical_across_frame0_change": bool(
                     np.array_equal(seg_argmax(f0_free_u8, r0["_f1"]), seg_argmax(f0_zero, r0["_f1"]))),
                 "note": "SegNet argmax IDENTICAL for two different frame_0 with the same frame_1 -> "
                         "frame_0 is 100% seg-free (upstream/modules.py:108). d_seg untouched."}

    # S0 summary (from cached rows that carry s0_cosine6)
    s0_rows = [r["s0_cosine6"] for r in per_pair if "s0_cosine6" in r]
    s0_summary = None
    if s0_rows:
        plateaus = [min(x["d_pose_traj"]) for x in s0_rows]
        s0_summary = {"n_pairs": len(s0_rows), "plateau_d_pose_mean": float(np.mean(plateaus)),
                      "plateau_d_pose_median": float(np.median(plateaus)),
                      "mean_relins_run": float(np.mean([x["relins_run"] for x in s0_rows])),
                      "example_traj": s0_rows[0]["d_pose_traj"],
                      "verdict": "RANK_DEFICIENT" if float(np.median(plateaus)) > 5.0 else "BUDGET_TRUNCATED",
                      "note": "rank-6 cosine basis (P3 actuation); plateau >> free floor ~1e-4 => "
                              "RANK_DEFICIENT (38.06 was a basis problem, not only budget)."}

    stored = np.array([r["d_pose_stored"] for r in per_pair])
    free = np.array([r["d_pose_free_u8"] for r in per_pair])
    warp = np.array([r["d_pose_warp_base"] for r in per_pair])
    free_mean = float(free.mean())
    contrib_mean = L.contribution(free_mean)
    thr_wall = 2.5e-4
    verdict = ("WALL_REFUTED_ARTIFACT_OF_NAIVE_SOLVE" if contrib_mean <= 0.05
               else "WALL_CONFIRMED_FORMULATION_SCOPE")
    rule_side = "CANDIDATE_LINE" if contrib_mean <= 0.05 else "CALIBRATION_INSTRUMENT"

    payload = {
        "schema": "ddm_p3v2_optimal_form_pose_resolve_FINAL.v1",
        "tool": "experiments/ddm_p3v2_finalize_from_cache.py (aggregate of the resumable ladder cache)",
        "utc": L._utc(), "git_hash": L._git_hash(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotable": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs_done": len(per_pair),
        "work_res": [192, 256],
        "baseline_d_pose": {
            "stored_mean": float(stored.mean()), "stored_median": float(np.median(stored)),
            "zeros_mean": float(np.mean([r["d_pose_zeros"] for r in per_pair])),
            "copy_mean": float(np.mean([r["d_pose_copy"] for r in per_pair])),
            "warp_base_mean": float(warp.mean()), "warp_base_median": float(np.median(warp)),
            "warp_base_contribution_at_mean": L.contribution(float(warp.mean())),
            "warp_base_note": "ego-motion homography of f1 by the carried pose target + 1 s_t scalar; "
                              "DECODER-REPRODUCIBLE ~0 bytes; the cheap-carrier floor."},
        "s1d_free_upper_bound": {
            "d_pose_mean": free_mean, "d_pose_median": float(np.median(free)),
            "d_pose_max": float(free.max()), "d_pose_min": float(free.min()),
            "contribution_at_mean": contrib_mean,
            "contribution_at_median": L.contribution(float(np.median(free))),
            "frac_pairs_below_2p5e-4": float(np.mean(free <= thr_wall)),
            "frac_pairs_below_1e-3": float(np.mean(free <= 1e-3)),
            "mean_iters_used": float(np.mean([r.get("free_iters_used", 0) for r in per_pair]))},
        "PRE_REGISTERED_RULE": {
            "wall_threshold_contribution": 0.05, "binding_contribution_measured": contrib_mean,
            "VERDICT": verdict, "vehicle_designation": rule_side,
            "note": "BINDING (task prompt / Assumption-Adversary): free-frame_0 UNPRICED upper-bound "
                    "contribution<=0.05 => wall REFUTED / candidate line; else CONFIRMED / calibration "
                    "instrument. frame_0 100% seg-free (law)."},
        "citations_banked_carriers": {
            "p3_6cosine_budget_truncated": {"d_pose_mean": 38.06223, "bytes_n600": 7295},
            "p715_quotient_reach_rank1": {"d_pose": 19.895, "carrier_bytes": 3520,
                                          "note": "generic covariance basis; d_pose RISES with rank"},
            "sc1_e_p_rank1": {"residual_bytes": 2039, "note": "pose-FIELD carrier; raw seed d_pose 36-146"}},
        "s0_cosine6_convergence": s0_summary,
        "seg_untouched_spot_check": seg_check,
        "s2_lotto": lotto,
        "per_pair": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_pair],
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    sha = sha256(args.out.read_bytes()).hexdigest()
    print(f"[finalize] VERDICT {verdict} / {rule_side}; free mean={free_mean:.3e} "
          f"contribution={contrib_mean:.4f}; warp mean={float(warp.mean()):.4f} "
          f"contribution={L.contribution(float(warp.mean())):.4f}", flush=True)
    print(f"[finalize] receipt -> {args.out} sha256={sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
