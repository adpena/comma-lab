#!/usr/bin/env python
"""Gradient-margin seg-repair screening on the REAL frontier (#51, corrected lever).

The appearance-gap repair (Class-3 module's implementation) INCREASES d_seg on
the real frontier (the GT-appearance direction is NOT the SegNet logit-margin
direction; receptive-field coupling). The CORRECT lever is the SegNet logit
gradient: step the rendered frame-1 along sign(∂(gt_class_logit - top_logit)/∂x)
at the flip pixels (the true margin-normal). This tool screens THAT lever with
full LAW accounting (d_seg, d_pose, honest carrier bytes).

Sweeps step amplitude per pair, picks the per-pair best LAW ΔS, and reports the
pool-wide economics: can the gradient repair clear THE LAW (1.27 B/flip break-even)?

AXIS [macOS-CPU advisory]. On-host replay ratifies. NO MPS. $0 local.
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

try:
    import brotli
except ImportError:  # pragma: no cover
    import brotlicffi as brotli  # type: ignore

N_CONTEST = 37_545_489
RATE_COEF = 25.0
SEG_W = 100.0
POSE_TEN = 10.0
SCH, SCW = 384, 512


def carrier_bytes_gradient(flip_cam_mask: np.ndarray, sign_correction_cam: np.ndarray,
                           step_scale: float) -> int:
    """Honest carrier bytes for a sparse gradient-sign correction at CAMERA res.

    Grammar: the correction is `step_scale * sign(grad)` at the flip-mask camera
    pixels. We code: sorted camera-position deltas (uint32, brotli) + per-pixel
    3-channel sign in {-1,0,1} packed 2 bits/channel (brotli) + 1 float step_scale.
    Camera positions are coarser to address (Hc*Wc ~ 1e6) so we instead store the
    SCORER-grid flip mask (384x512) + per-scorer-pixel sign, and the inflate patch
    upsamples (nearest) — far cheaper than camera positions.
    """
    # Down-project the camera correction to scorer grid for coding (nearest).
    import torch as _t
    sc_mask = _t.nn.functional.interpolate(
        _t.from_numpy(flip_cam_mask.astype(np.float32))[None, None],
        size=(SCH, SCW), mode="nearest")[0, 0].numpy() > 0.5
    flat = np.where(sc_mask.reshape(-1))[0].astype(np.int64)
    if flat.size == 0:
        return 0
    s = np.sort(flat)
    deltas = np.diff(np.concatenate([[0], s]))
    dtype = "<u2" if deltas.max() < 65536 else "<u4"
    pos_coded = brotli.compress(deltas.astype(dtype).tobytes(), quality=11)
    # signs per scorer pixel per channel: down-sample the camera sign via nearest
    sign_sc = _t.nn.functional.interpolate(
        _t.from_numpy(np.sign(sign_correction_cam).astype(np.float32)).permute(2, 0, 1)[None],
        size=(SCH, SCW), mode="nearest")[0].permute(1, 2, 0).numpy()
    sign_at = np.sign(sign_sc.reshape(-1, 3)[s]).astype(np.int8)  # in {-1,0,1}
    # pack 3 ternary -> 1 byte each (coarse; brotli compresses the redundancy)
    sign_coded = brotli.compress((sign_at + 1).astype(np.uint8).tobytes(), quality=11)
    header = 4 + 4 + 4 + 4 + 4  # magic + n + 2 len prefixes + step float
    return header + len(pos_coded) + len(sign_coded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="comma list OR 'top:N' from flip map")
    ap.add_argument("--flip-map-dir", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=str, default="2,4,8,16,32")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [float(x) for x in args.steps.split(",")]

    if args.pairs.startswith("top:"):
        n = int(args.pairs.split(":")[1])
        pp = json.loads((Path(args.flip_map_dir) / "flip_map_per_pair.json").read_text())
        sel = [r["pair_index"] for r in sorted(pp, key=lambda r: -r["n_flip"])[:n] if r["n_flip"] > 0]
    else:
        sel = [int(x) for x in args.pairs.split(",")]

    t0 = time.time()
    renderer = L.FrontierRenderer()
    scorer = L.ExactScorer()
    seg = scorer.net.segnet
    comp_all = renderer.render_baseline_pairs(sel)
    gt_all = L.decode_gt_pairs(sel)
    print(f"[grad-screen] {len(sel)} pairs, steps={steps}", flush=True)

    def argmax_of(f):
        pair = torch.stack([f, f])[None].float()
        with torch.inference_mode():
            return seg(seg.preprocess_input(pair)).argmax(1)[0].cpu().numpy()

    rows = []
    n_accept = 0
    for pi in sel:
        comp = comp_all[pi]
        gt = gt_all[pi].float()
        comp_f1 = comp[1].clone()
        gt_f1 = gt[1].permute(2, 0, 1)
        Hc, Wc = comp_f1.shape[1], comp_f1.shape[2]
        comp_am = argmax_of(comp_f1)
        gt_am = argmax_of(gt_f1)
        flip_sc = torch.from_numpy((comp_am != gt_am).astype(np.float32))
        n_flip = int(flip_sc.sum())
        if n_flip == 0:
            continue
        gt_am_t = torch.from_numpy(gt_am).long()

        # one gradient eval (margin loss); reuse sign across step scales
        x = comp_f1.clone().requires_grad_(True)
        pair = torch.stack([x, x])[None].float()
        logits = seg(seg.preprocess_input(pair))[0]  # (5,384,512)
        gt_logit = logits.gather(0, gt_am_t[None])[0]
        top_logit = logits.max(0).values
        ((gt_logit - top_logit) * flip_sc).sum().backward()
        grad = x.grad.detach()  # (3,Hc,Wc)
        sign_grad = torch.sign(grad)
        flip_cam = torch.nn.functional.interpolate(
            flip_sc[None, None], size=(Hc, Wc), mode="nearest")[0, 0] > 0.5

        # baseline distortion
        comp_for = L.comp_pair_to_bthwc(comp)
        gt_for = gt.unsqueeze(0)
        base_seg, base_pose = measure_pair_distortion(scorer.net, gt_for, comp_for)

        best = None
        for ss in steps:
            corr = ss * sign_grad
            rep_f1 = torch.where(flip_cam[None].expand(3, -1, -1),
                                 (comp_f1 + corr).clamp(0, 255), comp_f1).round()
            rep_pair = comp.clone()
            rep_pair[1] = rep_f1
            rep_for = L.comp_pair_to_bthwc(rep_pair)
            rep_seg, rep_pose = measure_pair_distortion(scorer.net, gt_for, rep_for)
            d_seg_delta = rep_seg - base_seg
            d_pose_delta = rep_pose - base_pose
            corr_cam = (corr * flip_cam[None]).permute(1, 2, 0).numpy()
            cbytes = carrier_bytes_gradient(flip_cam.numpy(), corr_cam, ss)
            seg_term = SEG_W * d_seg_delta
            pose_term = float(np.sqrt(POSE_TEN * max(rep_pose, 0)) - np.sqrt(POSE_TEN * max(base_pose, 0)))
            rate_term = RATE_COEF * cbytes / N_CONTEST
            dS = seg_term + pose_term + rate_term
            cand = {"step": ss, "d_seg_delta": d_seg_delta, "d_pose_delta": d_pose_delta,
                    "carrier_bytes": cbytes, "seg_term": seg_term, "pose_term": pose_term,
                    "rate_term": rate_term, "dS": dS, "n_flip": n_flip,
                    "flips_fixed_est": int(-d_seg_delta * SCH * SCW)}
            if best is None or dS < best["dS"]:
                best = cand
        best["pair_index"] = pi
        best["accepted"] = bool(best["d_seg_delta"] < 0 and best["dS"] < 0)
        if best["accepted"]:
            n_accept += 1
        rows.append(best)
        print(f"  pair {pi}: nflip={n_flip} best step={best['step']} dSeg={best['d_seg_delta']:.2e} "
              f"(fixed~{best['flips_fixed_est']}) dPose={best['d_pose_delta']:.2e} "
              f"bytes={best['carrier_bytes']} dS={best['dS']:.3e} "
              f"{'ACCEPT' if best['accepted'] else 'reject'}", flush=True)

    summary = {
        "schema": "frontier_seg_repair_gradient_screen.v1",
        "frontier_archive_sha256_16": "b7106c9bdbb8a2df",
        "n_pairs": len(rows), "n_accepted": n_accept,
        "steps_swept": steps,
        "axis_tag": "[macOS-CPU advisory]",
        "provenance": {"score_claim": False, "promotion_eligible": False,
                       "hardware_substrate": "local_macos_cpu"},
        "total_seconds": time.time() - t0,
    }
    (out_dir / "grad_screen_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "grad_screen_rows.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
