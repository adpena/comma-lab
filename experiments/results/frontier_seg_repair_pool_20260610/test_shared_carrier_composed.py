#!/usr/bin/env python
"""Decisive shared-carrier composed test (#51): the ONLY carrier that could clear THE LAW.

The per-pair gradient repair fails the LAW at pool scale (value/cost 0.43) because
per-pair position entropy (~3 B/flip) exceeds the 1.27 B/flip break-even. The ONLY
way to break the position bottleneck is a SHARED correction field: code the position
set ONCE, apply it to all 600 frame-1s. The optimistic upper bound (K>=10 high-freq
flip pixels) gives value/cost ~6 — IF the shared correction (a) fixes the targeted
flips in every pair AND (b) creates no new flips where the pixel was correct.

The single-pair receptive-field tests showed sparse corrections CREATE flips. This
tool MEASURES whether a shared correction at the systematically-mis-rendered pixels
(those flipping in >=K pairs) reduces the COMPOSED pool d_seg, evaluated on a
representative pair sample (pairs WHERE the pixels flip + pairs where they don't).

Shared correction direction: the per-pixel MEAN gradient-margin sign over the pairs
where the pixel flips (the systematic boundary-error direction). One field, one cost.

AXIS [macOS-CPU advisory]. NO MPS. $0 local. This decides dispatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R2 = Path(__file__).resolve().parent.parent / "pr110pp_r2_nonmps_candidate_20260609" / "analysis"
sys.path.insert(0, str(R2))
import render_and_score_lib as L  # noqa: E402
from tac.optimization.frame1_joint_safe_cone import measure_pair_distortion  # noqa: E402

SCH, SCW = 384, 512
N_CONTEST = 37_545_489


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-map-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--freq-k", type=int, default=10, help="target pixels flipping in >=K pairs")
    ap.add_argument("--n-eval-pairs", type=int, default=60, help="representative eval sample")
    ap.add_argument("--steps", type=str, default="4,8,16,32")
    args = ap.parse_args()

    flip_dir = Path(args.flip_map_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [float(x) for x in args.steps.split(",")]

    # 1. build per-pixel flip frequency + per-pixel systematic gradient-sign direction.
    files = sorted(flip_dir.glob("flip_idx_pair*.npy"))
    freq = np.zeros(SCH * SCW, dtype=np.int32)
    for f in files:
        idx = np.load(f)
        if idx.size:
            freq[idx] += 1
    target_flat = np.where(freq >= args.freq_k)[0]
    target_mask = np.zeros(SCH * SCW, dtype=bool)
    target_mask[target_flat] = True
    target_mask_hw = target_mask.reshape(SCH, SCW)
    n_target = int(target_mask.sum())
    print(f"[shared] {n_target} pixels flip in >={args.freq_k} pairs (systematic boundary errors)",
          flush=True)

    # 2. eval sample: a spread of pairs (includes high-flip + low-flip to catch new flips).
    per_pair = json.loads((flip_dir / "flip_map_per_pair.json").read_text())
    by_flip = sorted(per_pair, key=lambda r: -r["n_flip"])
    n = args.n_eval_pairs
    sample = sorted(set([r["pair_index"] for r in by_flip[: n // 2]]
                        + [r["pair_index"] for r in by_flip[-(n // 2):]]))

    renderer = L.FrontierRenderer()
    scorer = L.ExactScorer()
    seg = scorer.net.segnet
    comp_all = renderer.render_baseline_pairs(sample)
    gt_all = L.decode_gt_pairs(sample)

    def argmax_of(f):
        pair = torch.stack([f, f])[None].float()
        with torch.inference_mode():
            return seg(seg.preprocess_input(pair)).argmax(1)[0].cpu().numpy()

    # 3. accumulate the per-pixel systematic gradient-sign direction (scorer grid),
    #    averaged over the eval-sample pairs where each target pixel flips.
    grad_sign_accum = np.zeros((3, SCH, SCW), dtype=np.float64)
    grad_count = np.zeros((SCH, SCW), dtype=np.float64)
    t0 = time.time()
    for pi in sample:
        comp = comp_all[pi]
        gt = gt_all[pi].float()
        comp_f1 = comp[1].clone()
        gt_f1 = gt[1].permute(2, 0, 1)
        Hc, Wc = comp_f1.shape[1], comp_f1.shape[2]
        comp_am = argmax_of(comp_f1)
        gt_am = argmax_of(gt_f1)
        flip = (comp_am != gt_am)
        flip_t = torch.from_numpy(flip.astype(np.float32))
        gt_am_t = torch.from_numpy(gt_am).long()
        x = comp_f1.clone().requires_grad_(True)
        pair = torch.stack([x, x])[None].float()
        logits = seg(seg.preprocess_input(pair))[0]
        gt_logit = logits.gather(0, gt_am_t[None])[0]
        top_logit = logits.max(0).values
        ((gt_logit - top_logit) * flip_t).sum().backward()
        # grad is camera-res; down-project to scorer grid (mean-pool) for the shared field
        grad_cam = torch.sign(x.grad.detach())  # (3,Hc,Wc)
        grad_sc = torch.nn.functional.interpolate(
            grad_cam[None], size=(SCH, SCW), mode="bilinear")[0].numpy()  # (3,384,512)
        # only accumulate at target pixels that flip in THIS pair
        flip_target = flip & target_mask_hw
        grad_sign_accum[:, flip_target] += grad_sc[:, flip_target]
        grad_count[flip_target] += 1.0

    # systematic shared direction = sign of the mean gradient at each target pixel
    shared_dir = np.zeros((3, SCH, SCW), dtype=np.float64)
    nz = grad_count > 0
    for c in range(3):
        shared_dir[c, nz] = np.sign(grad_sign_accum[c, nz])
    shared_target = nz & target_mask_hw  # pixels with a defined shared direction
    n_shared = int(shared_target.sum())
    print(f"[shared] shared correction defined at {n_shared} pixels "
          f"(setup {time.time() - t0:.1f}s)", flush=True)

    # 4. for each step, apply the SAME shared correction to EVERY eval pair, measure
    #    composed pool d_seg (mean over the eval sample) + d_pose. One carrier cost.
    base_segs = []
    base_poses = []
    for pi in sample:
        comp = comp_all[pi]
        gt_for = gt_all[pi].float().unsqueeze(0)
        comp_for = L.comp_pair_to_bthwc(comp)
        bs, bp = measure_pair_distortion(scorer.net, gt_for, comp_for)
        base_segs.append(bs)
        base_poses.append(bp)
    base_seg_mean = float(np.mean(base_segs))
    base_pose_mean = float(np.mean(base_poses))

    # carrier bytes: shared position set + shared per-pixel int8 value, coded ONCE.
    try:
        import brotli
    except ImportError:
        import brotlicffi as brotli  # type: ignore
    sflat = np.sort(np.where(shared_target.reshape(-1))[0])
    deltas = np.diff(np.concatenate([[0], sflat]))
    dtype = "<u2" if (deltas.size and deltas.max() < 65536) else "<u4"
    pos_b = len(brotli.compress(deltas.astype(dtype).tobytes(), quality=11)) if deltas.size else 0

    results = []
    Hc, Wc = comp_all[sample[0]][1].shape[1], comp_all[sample[0]][1].shape[2]
    shared_dir_cam = torch.nn.functional.interpolate(
        torch.from_numpy(shared_dir).float()[None], size=(Hc, Wc), mode="nearest")[0]
    shared_mask_cam = torch.nn.functional.interpolate(
        torch.from_numpy(shared_target.astype(np.float32))[None, None], size=(Hc, Wc),
        mode="nearest")[0, 0] > 0.5

    for ss in steps:
        # value coding (the int8 correction magnitudes are uniform = ss*sign, brotli tiny)
        val_arr = (ss * shared_dir.transpose(1, 2, 0)[shared_target]).astype(np.int16)
        val_arr = np.clip(val_arr, -127, 127).astype(np.int8)
        val_b = len(brotli.compress(val_arr.tobytes(), quality=11))
        cbytes = pos_b + val_b + 16
        rep_segs = []
        rep_poses = []
        for pi in sample:
            comp = comp_all[pi]
            comp_f1 = comp[1].clone()
            corr = ss * shared_dir_cam
            rep_f1 = torch.where(shared_mask_cam[None].expand(3, -1, -1),
                                 (comp_f1 + corr).clamp(0, 255), comp_f1).round()
            rep_pair = comp.clone()
            rep_pair[1] = rep_f1
            gt_for = gt_all[pi].float().unsqueeze(0)
            rep_for = L.comp_pair_to_bthwc(rep_pair)
            rs, rp = measure_pair_distortion(scorer.net, gt_for, rep_for)
            rep_segs.append(rs)
            rep_poses.append(rp)
        rep_seg_mean = float(np.mean(rep_segs))
        rep_pose_mean = float(np.mean(rep_poses))
        d_seg = rep_seg_mean - base_seg_mean
        d_pose = rep_pose_mean - base_pose_mean
        seg_term = 100.0 * d_seg
        pose_term = float(np.sqrt(10 * max(rep_pose_mean, 0)) - np.sqrt(10 * max(base_pose_mean, 0)))
        rate_term = 25.0 * cbytes / N_CONTEST
        dS = seg_term + pose_term + rate_term
        results.append({"step": ss, "base_seg_mean": base_seg_mean, "rep_seg_mean": rep_seg_mean,
                        "d_seg_composed": d_seg, "d_pose_composed": d_pose,
                        "carrier_bytes": cbytes, "seg_term": seg_term, "pose_term": pose_term,
                        "rate_term": rate_term, "dS_composed": dS,
                        "composed_better": bool(dS < 0)})
        print(f"  step={ss}: composed d_seg {base_seg_mean:.6e}->{rep_seg_mean:.6e} "
              f"(Δ{d_seg:+.2e}) d_pose Δ{d_pose:+.2e} bytes={cbytes} dS={dS:+.3e} "
              f"{'BETTER' if dS < 0 else 'worse'}", flush=True)

    summary = {
        "schema": "frontier_seg_repair_shared_carrier_composed.v1",
        "frontier_archive_sha256_16": "b7106c9bdbb8a2df",
        "freq_k": args.freq_k, "n_target_pixels": n_target, "n_shared_defined": n_shared,
        "n_eval_pairs": len(sample), "eval_pairs": sample,
        "carrier_position_bytes": pos_b,
        "base_seg_mean_eval": base_seg_mean,
        "results": results,
        "best_dS": min((r["dS_composed"] for r in results), default=None),
        "any_composed_better": any(r["composed_better"] for r in results),
        "axis_tag": "[macOS-CPU advisory]",
        "note": "eval on n_eval_pairs sample; pool ΔS extrapolation in memo; on-host ratifies",
        "provenance": {"score_claim": False, "promotion_eligible": False,
                       "hardware_substrate": "local_macos_cpu"},
        "total_seconds": time.time() - t0,
    }
    (out_dir / "shared_carrier_composed_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
