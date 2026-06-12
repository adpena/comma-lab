# SPDX-License-Identifier: MIT
"""Residual argmax-FLIP delta runner (the Yousfi eureka) — the decisive measure.

Lane ``lane_cool_chic_residual_flip_delta_20260611``. Stops the Cool-Chic base
at d_seg ~= 0.0276 (the n48 48x64 config) and codes ONLY its residual argmax
flips vs L* as a sparse RGB witness perturbation, byte-closes the delta, and
RE-SEGMENTS the perturbed witness through the EXACT eval roundtrip to measure:

  - the BIGGEST RISK FIRST: does the RGB delta NET-reduce d_seg under the exact
    re-segmentation (not a relabel), or do new off-target flips cancel the gain?
  - the delta byte cost (is it proportional to flips = cheap?), and the new S.

torch-CPU ONLY; NEVER MPS; GT via upstream/frame_utils.py yuv420_to_rgb only.
Every row [macOS-CPU advisory] / promotable=False.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
from tac.residual_basis.cool_chic_byte_close import (
    byte_close,
    carrier_to_state,
    inflate_numpy,
    render_pair_numpy,
)
from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.residual_basis.residual_flip_delta import (
    apply_flip_delta,
    build_appearance_flip_delta,
    byte_close_flip_delta,
    inflate_flip_delta,
)
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n{n}.pt"
ARCHIVE_DENOM = 37_545_489.0


def _eval_roundtrip(flat_bchw: torch.Tensor, scorer_hw: tuple[int, int]) -> torch.Tensor:
    """The SAME uint8/resize eval roundtrip the scorer + base runner use."""
    cam_h = max(round(scorer_hw[0] * 874 / 384), scorer_hw[0] + 1)
    cam_w = max(round(scorer_hw[1] * 1164 / 512), scorer_hw[1] + 1)
    return apply_eval_roundtrip_during_training(
        flat_bchw, simulate_uint8=True, simulate_resize=True, ste_round=True,
        target_h=cam_h, target_w=cam_w,
    )


@torch.no_grad()
def _witnesses_and_argmax(state, net, n, scorer_hw):
    """Render each pair, build the WITNESS (post-interpolate, pre-roundtrip) RGB
    for frame0/frame1, and the frame1 SegNet argmax AFTER the exact roundtrip.

    Returns:
        witness1_list: list of (H_w, W_w, 3) numpy frame1 witnesses (pre-roundtrip).
        witness0_list: same for frame0 (needed to rebuild the pair for re-score).
        argmax1: (n, 384, 512) int64 base SegNet argmax of each frame1.
    """
    h, w = scorer_hw
    witness0, witness1, argmaxes = [], [], []
    for i in range(n):
        rgb0, rgb1 = render_pair_numpy(state, i)  # (3,H,W) [0,255]
        t = torch.from_numpy(np.stack([rgb0, rgb1], axis=0)).float()  # (2,3,H,W)
        ti = F.interpolate(t, size=scorer_hw, mode="bilinear", align_corners=False)
        w0 = ti[0].permute(1, 2, 0).numpy()  # (H_w,W_w,3) witness frame0
        w1 = ti[1].permute(1, 2, 0).numpy()  # witness frame1
        witness0.append(w0)
        witness1.append(w1)
        # re-segment frame1 witness through the exact roundtrip
        rt = _eval_roundtrip(ti, scorer_hw)  # (2,3,H_w,W_w)
        bchw = rt.reshape(1, 2, 3, h, w)
        bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
        _, segnet_in = net.preprocess_input(bhwc)
        seg_out = net.segnet(segnet_in)  # (1,5,384,512)
        argmaxes.append(seg_out.argmax(dim=1)[0].numpy().astype(np.int64))
    return witness1, witness0, np.stack(argmaxes, axis=0)


@torch.no_grad()
def _rescore_witnesses(witness0_list, witness1_list, net, seg_targets, pose_targets, scorer_hw):
    """Re-score a set of (frame0,frame1) witnesses through the exact roundtrip.

    Returns (d_seg, d_pose). d_seg is mean argmax-disagreement vs L* (seg_targets).
    """
    h, w = scorer_hw
    n = len(witness1_list)
    pairs = []
    for i in range(n):
        w0 = torch.from_numpy(witness0_list[i]).float().permute(2, 0, 1)  # (3,H,W)
        w1 = torch.from_numpy(witness1_list[i]).float().permute(2, 0, 1)
        pairs.append(torch.stack([w0, w1], axis=0))  # (2,3,H,W)
    t = torch.stack(pairs, axis=0)  # (n,2,3,H,W)
    flat = t.reshape(n * 2, 3, h, w)
    flat = _eval_roundtrip(flat, scorer_hw)
    bchw = flat.reshape(n, 2, 3, h, w)
    bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
    posenet_in, segnet_in = net.preprocess_input(bhwc)
    seg_out = net.segnet(segnet_in)
    pred = seg_out.argmax(dim=1)
    d_seg = float((pred != seg_targets.long()).float().mean().item())
    d_pose = None
    if pose_targets is not None:
        pose_out = net.posenet(posenet_in)
        d_pose = float(F.mse_loss(pose_out["pose"][:, :6], pose_targets).item())
    return d_seg, d_pose


def fit_and_byte_close_base(net, seg, pose, *, scorer_hw, epochs, grid_step, delta_step, seed=0):
    """Re-fit + byte-close the n48 48x64 base config (the d_seg~0.0276 base)."""
    torch.manual_seed(seed)
    n_pairs = int(seg.shape[0])
    spec = CoolChicGridSpec(base_h=48, base_w=64, n_grids=4, channels_per_grid=3)
    carrier = CoolChicPairCarrier(n_pairs=n_pairs, spec=spec, synth_hidden=24, out_hw=(96, 128))
    cfg = ScoreAwareLoopConfig(
        epochs=epochs, batch_size=n_pairs, scorer_hw=scorer_hw, pose_enabled=pose is not None,
        eval_every=max(epochs // 4, 1), seg_loss_form="ce_seg_loss", decoder_lr=3e-3,
        latent_lr_mult=10.0, ema_decay=0.99, seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg, pose, cfg)
    t0 = time.time()
    tr.train()
    wall = time.time() - t0
    tr._ema_for_eval.apply_to(tr.model)
    state = carrier_to_state(carrier)
    blob = byte_close(state, grid_step=grid_step, delta_step=delta_step)
    base_state = inflate_numpy(blob)
    return base_state, len(blob), wall


def decode_gt_witnesses(net, n, scorer_hw, video_path):
    """Decode GT frame0/frame1 pairs (yuv420_to_rgb) and build witnesses at scorer_hw."""
    from tac.boundary_math.seg_core import decode_gt_frame1_pairs

    h, w = scorer_hw
    gt_w0, gt_w1 = [], []
    for _idx, f0, f1 in decode_gt_frame1_pairs(video_path, n_pairs=n):
        t = torch.from_numpy(np.stack([f0, f1], axis=0).astype(np.float64)).float()  # (2,H,W,3)
        t = t.permute(0, 3, 1, 2)  # (2,3,H,W)
        ti = F.interpolate(t, size=scorer_hw, mode="bilinear", align_corners=False)
        gt_w0.append(ti[0].permute(1, 2, 0).numpy())
        gt_w1.append(ti[1].permute(1, 2, 0).numpy())
    return gt_w0, gt_w1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--delta-step", type=float, default=0.05)
    ap.add_argument("--l-inf", type=float, default=64.0)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_residual_flip_delta_20260611/flip_delta_advisory_s.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    scorer_hw = (args.scorer_h, args.scorer_w)
    video_path = "upstream/videos/0.mkv"

    cache = CACHE.format(n=48 if args.n_pairs <= 48 else 100)
    d = torch.load(cache, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")
    n = int(seg.shape[0])

    print(f"[1/5] fit+byte-close base (n={n}, scorer={scorer_hw}, ep={args.epochs})", flush=True)
    base_state, base_bytes, wall = fit_and_byte_close_base(
        net, seg, pose, scorer_hw=scorer_hw, epochs=args.epochs,
        grid_step=args.grid_step, delta_step=args.delta_step,
    )

    print("[2/5] re-segment base witnesses + GT witnesses", flush=True)
    base_w1, base_w0, base_argmax = _witnesses_and_argmax(base_state, net, n, scorer_hw)
    gt_w0, gt_w1 = decode_gt_witnesses(net, n, scorer_hw, video_path)
    lstar = seg.numpy().astype(np.int64)  # (n,384,512) GT SegNet argmax

    # base d_seg / d_pose (no delta) — should reproduce the ~0.0276 base row.
    d_seg_base, d_pose_base = _rescore_witnesses(base_w0, base_w1, net, seg, pose, scorer_hw)
    print(f"      base d_seg={d_seg_base:.4f} d_pose={d_pose_base:.4e} bytes={base_bytes}", flush=True)

    print("[3/5] BIGGEST-RISK sweep: does ANY (strength,l_inf,dilate) NET-reduce d_seg?", flush=True)
    # The Yousfi-named risk: a sparse RGB nudge at flip pixels must FIX flips
    # under the EXACT re-segmentation without inducing NEW off-target flips
    # through the resize. We sweep the operating point (the appearance-gap
    # fraction + L-inf cap + tube dilation) and measure, for each, the NET
    # re-segmented d_seg + the new off-target flip count. The decisive plan is
    # the operating point with the LOWEST net d_seg that also beats the base.
    sweep_grid = [
        # (strength, l_inf, dilate) — gentle->strong; small l_inf = minimal nudge
        (1.0, 4.0, 0), (1.0, 2.0, 0), (0.5, 4.0, 0),
        (1.0, 8.0, 0), (0.5, 8.0, 0),
        (1.0, 16.0, 0), (1.0, 64.0, 0),
        # full GT-substitution CEILING reference: replace flip pixels with GT.
        (1.0, 255.0, 0),
    ]
    sweep_rows = []
    best = None
    for (st, li, dl) in sweep_grid:
        pl = [
            build_appearance_flip_delta(
                base_w1[i], gt_w1[i], base_argmax[i], lstar[i], pair_idx=i,
                l_inf_budget=li, dilate=dl, strength=st,
            )
            for i in range(n)
        ]
        pert = [apply_flip_delta(base_w1[i], pl[i]) for i in range(n)]
        off = 0
        for i in range(n):
            t = torch.from_numpy(pert[i]).float().permute(2, 0, 1)
            pair = torch.stack([t, t], dim=0)  # (2,3,H,W)
            rt = _eval_roundtrip(pair, scorer_hw)
            bhwc = rt.reshape(1, 2, 3, *scorer_hw).permute(0, 1, 3, 4, 2).contiguous()
            _, segnet_in = net.preprocess_input(bhwc)
            am = net.segnet(segnet_in).argmax(dim=1)[0].numpy().astype(np.int64)
            was_flip = base_argmax[i] != lstar[i]
            now_flip = am != lstar[i]
            off += int((now_flip & ~was_flip).sum())
        d_seg_sw, _ = _rescore_witnesses(base_w0, pert, net, seg, None, scorer_hw)
        k = sum(p.flat_idx.size for p in pl)
        srow = {"strength": st, "l_inf": li, "dilate": dl, "d_seg": d_seg_sw,
                "new_offtarget": off, "k_total": k}
        sweep_rows.append(srow)
        print(f"      st={st} li={li} dl={dl}: d_seg={d_seg_sw:.4f} "
              f"(base {d_seg_base:.4f}) off-target={off} k={k}", flush=True)
        if best is None or d_seg_sw < best["d_seg"]:
            best = srow

    # Pick the decisive operating point: lowest net re-segmented d_seg.
    plans = [
        build_appearance_flip_delta(
            base_w1[i], gt_w1[i], base_argmax[i], lstar[i], pair_idx=i,
            l_inf_budget=best["l_inf"], dilate=best["dilate"], strength=best["strength"],
        )
        for i in range(n)
    ]
    target_flips_total = sum(p.target_flip_pixels for p in plans)
    new_offtarget_total = best["new_offtarget"]
    args.l_inf = best["l_inf"]  # byte-close uses the chosen operating point's scale
    pert_w1_fp = [apply_flip_delta(base_w1[i], plans[i]) for i in range(n)]
    d_seg_fp, d_pose_fp = _rescore_witnesses(base_w0, pert_w1_fp, net, seg, pose, scorer_hw)

    print("[4/5] byte-close delta + numpy inflate parity", flush=True)
    delta_blob = byte_close_flip_delta(plans, l_inf_budget=args.l_inf)
    delta_bytes = len(delta_blob)
    plans2 = inflate_flip_delta(delta_blob)
    # parity: inflated plan reproduces the applied perturbation bit-exact (post-quant).
    parity_max = 0.0
    pert_w1_bc = []
    for i in range(n):
        a = apply_flip_delta(base_w1[i], plans2[i])
        pert_w1_bc.append(a)
        # compare vs a fresh quantized apply of the original plan to isolate inflate parity
        parity_max = max(parity_max, float(np.abs(a - apply_flip_delta(base_w1[i], _requantize(plans[i], args.l_inf))).max()))

    print("[5/5] RE-SCORE byte-closed delta-applied witness (the decisive measure)", flush=True)
    d_seg_bc, d_pose_bc = _rescore_witnesses(base_w0, pert_w1_bc, net, seg, pose, scorer_hw)

    total_bytes = base_bytes + delta_bytes
    pose_term = math.sqrt(10.0 * d_pose_bc) if d_pose_bc is not None else 0.0
    rate_term = 25.0 * total_bytes / ARCHIVE_DENOM
    advisory_s = 100.0 * d_seg_bc + pose_term + rate_term

    flips_before_total = sum(p.n_flips_before for p in plans)
    delta_bytes_per_flip = delta_bytes / max(flips_before_total, 1)

    row = {
        "config": {
            "base_hw": [48, 64], "scorer_hw": list(scorer_hw), "epochs": args.epochs,
            "chosen_operating_point": best,
        },
        "biggest_risk_sweep": sweep_rows,
        "base_bytes": base_bytes,
        "delta_bytes": delta_bytes,
        "total_bytes": total_bytes,
        "delta_inflate_parity_max_255scale": parity_max,
        "d_seg_base_no_delta": d_seg_base,
        "d_seg_fullprec_delta": d_seg_fp,
        "d_seg_byteclosed_delta": d_seg_bc,
        "d_pose_base_no_delta": d_pose_base,
        "d_pose_byteclosed_delta": d_pose_bc,
        "flips_before_total_384x512": flips_before_total,
        "target_flip_pixels_witness": target_flips_total,
        "new_offtarget_flips_witness": new_offtarget_total,
        "delta_bytes_per_flip_384x512": delta_bytes_per_flip,
        "advisory_s": advisory_s,
        "advisory_s_terms": {
            "seg_100xdseg": 100.0 * d_seg_bc,
            "pose_sqrt10dpose": pose_term,
            "rate_25xbytes": rate_term,
        },
        "train_wall_seconds": round(wall, 1),
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "lane": "lane_cool_chic_residual_flip_delta_20260611",
                "axis_tag": "[macOS-CPU advisory]",
                "promotable": False,
                "frontier_for_reference": 0.19110,
                "n_pairs": n,
                "row": row,
            },
            f, indent=2,
        )
    print(
        f"\nRESULT: base d_seg {d_seg_base:.4f} -> delta-applied bc {d_seg_bc:.4f} "
        f"(fp {d_seg_fp:.4f}) | delta {delta_bytes}B (~{delta_bytes_per_flip:.3f} B/flip) | "
        f"new off-target flips {new_offtarget_total} (target tube {target_flips_total}px) | "
        f"d_pose {d_pose_base:.3e}->{d_pose_bc:.3e} | S={advisory_s:.4f} "
        f"[seg {row['advisory_s_terms']['seg_100xdseg']:.4f} + pose {pose_term:.4f} + "
        f"rate {rate_term:.2e}]",
        flush=True,
    )
    print("FLIP-DELTA ADVISORY-S DONE", args.out, flush=True)


def _requantize(plan, l_inf):
    """Return a plan whose values are quantized exactly like byte-close (for parity)."""
    from tac.residual_basis.residual_flip_delta import FlipDeltaPlan, _quantize_values

    qv, scale = _quantize_values(plan.values, l_inf)
    return FlipDeltaPlan(
        pair_idx=plan.pair_idx, witness_hw=plan.witness_hw, flat_idx=plan.flat_idx,
        values=qv.astype(np.float64) * scale, n_flips_before=plan.n_flips_before,
        target_flip_pixels=plan.target_flip_pixels,
    )


if __name__ == "__main__":
    main()
