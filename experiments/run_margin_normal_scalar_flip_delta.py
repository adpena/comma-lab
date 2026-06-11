# SPDX-License-Identifier: MIT
"""Cheap margin-normal SCALAR flip-delta runner (lever-D reactivation of v2).

The v2 surgical repair atom NET-reduces d_seg but codes the full 3-channel RGB gap
(~187 B/flip => rate-blocked).  This runner builds the SAME canonical repair atom,
then encodes the boundary crossing as the CHEAP one-channel margin-normal scalar
(``tac.residual_basis.margin_normal_scalar_delta``) and answers the biggest risk:

  does a cheap one-channel scalar nudge still flip the SegNet argmax through
  uint8/resize, the way the full 3-channel RGB gap did?

Apples-to-apples: on the SAME well-trained byte-closed base it measures BOTH the
RGB-delta (v2 path) and the cheap-scalar path -- byte-closed d_seg, MEASURED bytes/
flip, new off-target flips, advisory S, and the 600-pair rate projection.

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
from tac.optimization.frame1_seg_repair_atoms import (
    Frame1SegRepairAtomError,
    RepairTargets,
    SegRepairAtomConfig,
    generate_seg_repair_atom,
)
from tac.residual_basis.cool_chic_byte_close import (
    byte_close,
    carrier_to_state,
    inflate_numpy,
    render_pair_numpy,
)
from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.residual_basis.margin_normal_scalar_delta import (
    apply_margin_normal_scalar,
    build_margin_normal_scalar_plan,
    byte_close_margin_normal_scalar,
    inflate_margin_normal_scalar,
)
from tac.residual_basis.residual_flip_delta import (
    FlipDeltaPlan,
    apply_flip_delta,
    byte_close_flip_delta,
    inflate_flip_delta,
)
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n{n}.pt"
ARCHIVE_DENOM = 37_545_489.0
CONTEST_WATER_LEVEL_B_PER_FLIP = 1.27


def _eval_roundtrip(flat_bchw, scorer_hw):
    cam_h = max(round(scorer_hw[0] * 874 / 384), scorer_hw[0] + 1)
    cam_w = max(round(scorer_hw[1] * 1164 / 512), scorer_hw[1] + 1)
    return apply_eval_roundtrip_during_training(
        flat_bchw, simulate_uint8=True, simulate_resize=True, ste_round=True,
        target_h=cam_h, target_w=cam_w,
    )


@torch.no_grad()
def _resegment(witness_hwc, net, scorer_hw):
    t = torch.from_numpy(np.asarray(witness_hwc)).float().permute(2, 0, 1)
    pair = torch.stack([t, t], dim=0)
    rt = _eval_roundtrip(pair, scorer_hw)
    bhwc = rt.reshape(1, 2, 3, *scorer_hw).permute(0, 1, 3, 4, 2).contiguous()
    _, segnet_in = net.preprocess_input(bhwc)
    return net.segnet(segnet_in).argmax(dim=1)[0].numpy().astype(np.int64)


@torch.no_grad()
def _rescore(witness0_list, witness1_list, net, seg_targets, pose_targets, scorer_hw):
    h, w = scorer_hw
    n = len(witness1_list)
    pairs = []
    for i in range(n):
        w0 = torch.from_numpy(witness0_list[i]).float().permute(2, 0, 1)
        w1 = torch.from_numpy(witness1_list[i]).float().permute(2, 0, 1)
        pairs.append(torch.stack([w0, w1], axis=0))
    t = torch.stack(pairs, axis=0).reshape(n * 2, 3, h, w)
    t = _eval_roundtrip(t, scorer_hw)
    bhwc = t.reshape(n, 2, 3, h, w).permute(0, 1, 3, 4, 2).contiguous()
    posenet_in, segnet_in = net.preprocess_input(bhwc)
    pred = net.segnet(segnet_in).argmax(dim=1)
    d_seg = float((pred != seg_targets.long()).float().mean().item())
    d_pose = None
    if pose_targets is not None:
        d_pose = float(F.mse_loss(net.posenet(posenet_in)["pose"][:, :6], pose_targets).item())
    return d_seg, d_pose


def fit_and_byte_close_base(net, seg, pose, *, scorer_hw, epochs, seed=0):
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
    blob = byte_close(state)
    return inflate_numpy(blob), len(blob), wall


def _witnesses(state, net, n, scorer_hw):
    w0, w1 = [], []
    for i in range(n):
        rgb0, rgb1 = render_pair_numpy(state, i)
        t = torch.from_numpy(np.stack([rgb0, rgb1], axis=0)).float()
        ti = F.interpolate(t, size=scorer_hw, mode="bilinear", align_corners=False)
        w0.append(ti[0].permute(1, 2, 0).numpy())
        w1.append(ti[1].permute(1, 2, 0).numpy())
    return w0, w1


def _gt_witnesses(n, scorer_hw, video_path):
    from tac.boundary_math.seg_core import decode_gt_frame1_pairs

    g0, g1 = [], []
    for _idx, f0, f1 in decode_gt_frame1_pairs(video_path, n_pairs=n):
        t = torch.from_numpy(np.stack([f0, f1], axis=0).astype(np.float64)).float().permute(0, 3, 1, 2)
        ti = F.interpolate(t, size=scorer_hw, mode="bilinear", align_corners=False)
        g0.append(ti[0].permute(1, 2, 0).numpy())
        g1.append(ti[1].permute(1, 2, 0).numpy())
    return g0, g1


def _atom_to_rgb_plan(atom, witness_hw, l_inf):
    """v2 path: full (H,W,3) correction -> sparse 3-channel RGB plan (the ~187 B/flip)."""
    corr = atom.correction
    h, w = witness_hw
    ch_list = []
    for c in range(3):
        plane = corr[:, :, c]
        ph, pw = plane.shape
        if (ph, pw) != (h, w):
            ys = (np.arange(h) * ph / h).astype(np.int64).clip(0, ph - 1)
            xs = (np.arange(w) * pw / w).astype(np.int64).clip(0, pw - 1)
            plane = plane[ys][:, xs]
        ch_list.append(plane)
    cw = np.clip(np.stack(ch_list, axis=2), -l_inf, l_inf)
    nz = np.abs(cw) > 0.5
    ys, xs, cs = np.nonzero(nz)
    flat = (ys * (w * 3) + xs * 3 + cs).astype(np.int64)
    order = np.argsort(flat)
    return FlipDeltaPlan(
        pair_idx=atom.pair_index, witness_hw=witness_hw,
        flat_idx=flat[order], values=cw[ys, xs, cs][order].astype(np.float64),
        n_flips_before=atom.n_support_pixels, target_flip_pixels=int(nz.any(axis=2).sum()),
    )


def _offtarget(reseg, base_argmax_i, lstar_i):
    was = base_argmax_i != lstar_i
    now = reseg != lstar_i
    return int((now & ~was).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--l-inf", type=float, default=48.0)
    ap.add_argument("--correction-fraction", type=float, default=1.0)
    ap.add_argument("--support-top-fraction", type=float, default=0.25)
    ap.add_argument("--boundary-margin-percentile", type=float, default=0.3)
    ap.add_argument(
        "--screen-the-law", action="store_true",
        help="ship a pair's cheap-scalar atom only if it STRICTLY reduces that pair's "
             "re-segmented (eval-roundtrip) flip count vs L* (THE LAW on the real surface)",
    )
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_residual_flip_delta_20260611/margin_normal_scalar_advisory_s.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    scorer_hw = (384, 512)
    video_path = "upstream/videos/0.mkv"

    cache = CACHE.format(n=48 if args.n_pairs <= 48 else 100)
    d = torch.load(cache, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")
    n = int(seg.shape[0])

    print(f"[1/6] fit+byte-close base (n={n}, ep={args.epochs})", flush=True)
    base_state, base_bytes, wall = fit_and_byte_close_base(
        net, seg, pose, scorer_hw=scorer_hw, epochs=args.epochs)

    print("[2/6] witnesses + base/GT argmax", flush=True)
    base_w0, base_w1 = _witnesses(base_state, net, n, scorer_hw)
    gt_w0, gt_w1 = _gt_witnesses(n, scorer_hw, video_path)
    lstar = seg.numpy().astype(np.int64)
    base_argmax = np.stack([_resegment(base_w1[i], net, scorer_hw) for i in range(n)], axis=0)
    d_seg_base, d_pose_base = _rescore(base_w0, base_w1, net, seg, pose, scorer_hw)
    print(f"      base d_seg={d_seg_base:.4f} d_pose={d_pose_base:.4e} bytes={base_bytes}", flush=True)

    print("[3/6] canonical surgical repair atoms (shared by both encodings)", flush=True)
    cfg = SegRepairAtomConfig(
        boundary_margin_percentile=args.boundary_margin_percentile,
        support_top_fraction=args.support_top_fraction,
        correction_fraction=args.correction_fraction,
        min_support_pixels=16,
    )
    rgb_plans, mns_plans = [], []
    rgb_off, mns_off, n_support_total = 0, 0, 0
    rgb_w1, mns_w1 = [], []
    mns_atoms_shipped, mns_atoms_screened_out = 0, 0
    empty_mns = lambda: build_margin_normal_scalar_plan(  # noqa: E731
        np.zeros((*scorer_hw, 3)), pair_idx=i, witness_hw=scorer_hw, l_inf_budget=args.l_inf)
    for i in range(n):
        try:
            targets = RepairTargets.measure(
                segnet=net.segnet, rendered_frame1_hwc_unit255=base_w1[i],
                gt_frame1_hwc_unit255=gt_w1[i],
                fragile_mask=np.zeros((384, 512), dtype=bool),
            )
            atom = generate_seg_repair_atom(pair_index=i, targets=targets, config=cfg, prefer_mlx=False)
            n_support_total += atom.n_support_pixels
            # v2 RGB plan
            rgb_plan = _atom_to_rgb_plan(atom, scorer_hw, args.l_inf)
            rgb_rep = atom.apply(base_w1[i])
            # cheap scalar plan (collapse to dominant channel)
            mns_plan = build_margin_normal_scalar_plan(
                atom.correction, pair_idx=i, witness_hw=scorer_hw, l_inf_budget=args.l_inf)
            mns_rep = apply_margin_normal_scalar(base_w1[i], mns_plan)
        except Frame1SegRepairAtomError:
            rgb_plan = FlipDeltaPlan(i, scorer_hw, np.zeros((0,), np.int64), np.zeros((0,), np.float64), 0, 0)
            mns_plan = empty_mns()
            rgb_rep = np.asarray(base_w1[i], dtype=np.float64)
            mns_rep = np.asarray(base_w1[i], dtype=np.float64)
        # THE LAW on the REAL re-segmentation surface (eval-roundtrip): only ship a
        # cheap-scalar atom if it STRICTLY reduces THIS pair's re-segmented flip count.
        mns_reseg = _resegment(mns_rep, net, scorer_hw)
        if args.screen_the_law and mns_plan.n_support_pixels > 0:
            flips_base = int((base_argmax[i] != lstar[i]).sum())
            flips_pert = int((mns_reseg != lstar[i]).sum())
            if flips_pert >= flips_base:  # not a rent-payer -> ship nothing for this pair
                mns_plan = empty_mns()
                mns_rep = np.asarray(base_w1[i], dtype=np.float64)
                mns_reseg = base_argmax[i]
                mns_atoms_screened_out += 1
            else:
                mns_atoms_shipped += 1
        elif mns_plan.n_support_pixels > 0:
            mns_atoms_shipped += 1
        rgb_plans.append(rgb_plan)
        mns_plans.append(mns_plan)
        rgb_w1.append(rgb_rep)
        mns_w1.append(mns_rep)
        rgb_off += _offtarget(_resegment(rgb_rep, net, scorer_hw), base_argmax[i], lstar[i])
        mns_off += _offtarget(mns_reseg, base_argmax[i], lstar[i])

    print("[4/6] BIGGEST-RISK: full-precision re-segmentation (RGB vs cheap scalar)", flush=True)
    d_seg_rgb_fp, _ = _rescore(base_w0, rgb_w1, net, seg, None, scorer_hw)
    d_seg_mns_fp, _ = _rescore(base_w0, mns_w1, net, seg, None, scorer_hw)
    print(f"      RGB fp d_seg={d_seg_rgb_fp:.4f} (off {rgb_off}) | "
          f"cheap-scalar fp d_seg={d_seg_mns_fp:.4f} (off {mns_off}) | base {d_seg_base:.4f}",
          flush=True)

    print("[5/6] byte-close BOTH + parity + re-score byte-closed", flush=True)
    rgb_blob = byte_close_flip_delta(rgb_plans, l_inf_budget=args.l_inf)
    rgb_dec = inflate_flip_delta(rgb_blob)
    rgb_bc = [apply_flip_delta(base_w1[i], rgb_dec[i]) for i in range(n)]
    d_seg_rgb_bc, d_pose_rgb_bc = _rescore(base_w0, rgb_bc, net, seg, pose, scorer_hw)

    mns_blob = byte_close_margin_normal_scalar(mns_plans, l_inf_budget=args.l_inf)
    mns_dec = inflate_margin_normal_scalar(mns_blob)
    mns_bc = [apply_margin_normal_scalar(base_w1[i], mns_dec[i]) for i in range(n)]
    mns_parity = max(
        float(np.max(np.abs(apply_margin_normal_scalar(base_w1[i], mns_plans[i]) - mns_bc[i])))
        for i in range(n)
    )
    d_seg_mns_bc, d_pose_mns_bc = _rescore(base_w0, mns_bc, net, seg, pose, scorer_hw)

    print("[6/6] advisory S + 600-pair rate projection", flush=True)

    def _adv_s(d_seg, d_pose, delta_bytes):
        pose_term = math.sqrt(10.0 * d_pose) if d_pose is not None else 0.0
        rate_term = 25.0 * (base_bytes + delta_bytes) / ARCHIVE_DENOM
        return 100.0 * d_seg + pose_term + rate_term, pose_term, rate_term

    rgb_bytes, mns_bytes = len(rgb_blob), len(mns_blob)
    s_rgb, pt_rgb, rt_rgb = _adv_s(d_seg_rgb_bc, d_pose_rgb_bc, rgb_bytes)
    s_mns, pt_mns, rt_mns = _adv_s(d_seg_mns_bc, d_pose_mns_bc, mns_bytes)
    bpf_rgb = rgb_bytes / max(n_support_total, 1)
    bpf_mns = mns_bytes / max(n_support_total, 1)
    # 600-pair delta-byte projection (linear in support; header amortizes => upper bound)
    proj_mns_600 = mns_bytes * (600.0 / n)
    proj_rgb_600 = rgb_bytes * (600.0 / n)

    row = {
        "config": {"base_hw": [48, 64], "scorer_hw": list(scorer_hw), "epochs": args.epochs,
                   "l_inf": args.l_inf, "correction_fraction": args.correction_fraction,
                   "support_top_fraction": args.support_top_fraction,
                   "boundary_margin_percentile": args.boundary_margin_percentile},
        "base_bytes": base_bytes, "support_pixels_total": n_support_total,
        "d_seg_base_no_delta": d_seg_base, "d_pose_base_no_delta": d_pose_base,
        "rgb_delta": {
            "delta_bytes": rgb_bytes, "bytes_per_flip": bpf_rgb,
            "d_seg_fullprec": d_seg_rgb_fp, "d_seg_byteclosed": d_seg_rgb_bc,
            "d_pose_byteclosed": d_pose_rgb_bc, "new_offtarget_flips": rgb_off,
            "advisory_s": s_rgb, "proj_delta_bytes_600": proj_rgb_600,
        },
        "margin_normal_scalar": {
            "delta_bytes": mns_bytes, "bytes_per_flip": bpf_mns,
            "byteclose_apply_parity_max_255scale": mns_parity,
            "d_seg_fullprec": d_seg_mns_fp, "d_seg_byteclosed": d_seg_mns_bc,
            "d_pose_byteclosed": d_pose_mns_bc, "new_offtarget_flips": mns_off,
            "advisory_s": s_mns, "advisory_s_terms": {"seg": 100.0 * d_seg_mns_bc, "pose": pt_mns, "rate": rt_mns},
            "proj_delta_bytes_600": proj_mns_600,
            "contest_water_level_b_per_flip": CONTEST_WATER_LEVEL_B_PER_FLIP,
            "under_water_level": bool(bpf_mns < CONTEST_WATER_LEVEL_B_PER_FLIP),
            "screen_the_law": bool(args.screen_the_law),
            "atoms_shipped": mns_atoms_shipped,
            "atoms_screened_out": mns_atoms_screened_out,
        },
        "net_d_seg_reduction_mns": d_seg_base - d_seg_mns_bc,
        "net_d_seg_reduction_rgb": d_seg_base - d_seg_rgb_bc,
        "train_wall_seconds": round(wall, 1),
        "axis_tag": "[macOS-CPU advisory]", "promotable": False, "score_claim": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"lane": "lane_cool_chic_residual_flip_delta_20260611",
                   "axis_tag": "[macOS-CPU advisory]", "promotable": False,
                   "method": "margin_normal_scalar_vs_rgb_delta", "n_pairs": n, "row": row}, f, indent=2)
    print(f"\nRESULT base d_seg {d_seg_base:.4f}", flush=True)
    print(f"  RGB-delta:   bc d_seg {d_seg_rgb_bc:.4f} (fp {d_seg_rgb_fp:.4f}) | "
          f"{bpf_rgb:.1f} B/flip | off {rgb_off} | S={s_rgb:.4f}", flush=True)
    print(f"  cheap scalar: bc d_seg {d_seg_mns_bc:.4f} (fp {d_seg_mns_fp:.4f}) | "
          f"{bpf_mns:.2f} B/flip | off {mns_off} | S={s_mns:.4f} | "
          f"under-water={bpf_mns < CONTEST_WATER_LEVEL_B_PER_FLIP}", flush=True)
    print("MARGIN-NORMAL-SCALAR DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
