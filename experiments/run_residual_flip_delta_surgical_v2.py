# SPDX-License-Identifier: MIT
"""Residual flip-delta v2 — SURGICAL canonical repair atom (the corrected Yousfi move).

v1 (run_residual_flip_delta_advisory_s.py) coded the FULL appearance gap over the
WHOLE flip tube and the sweep proved every operating point made d_seg WORSE: a
broad RGB nudge through the eval-roundtrip resize induces ~7x more NEW off-target
flips than it fixes.  SEARCH-AND-FAMILIARIZE redirect: the canonical
``tac.optimization.frame1_seg_repair_atoms`` already does this correctly — it
restricts the correction to RECOVERABLE flips (small-margin boundary-tube ∪
thin-class ∪ fragile), ranks by recoverable-margin leverage, keeps only the top
fraction, and screens each atom against THE LAW (100*Δd_seg + Δpose + rate < 0).

This v2 reuses ``generate_seg_repair_atom`` to build the surgical correction, then
RE-SEGMENTS the corrected witness through the EXACT eval-roundtrip (the biggest
risk: does the surgical nudge NET-reduce d_seg under uint8/resize without new
off-target flips?), byte-closes the sparse correction, and reports the advisory S.

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


def _eval_roundtrip(flat_bchw, scorer_hw):
    cam_h = max(round(scorer_hw[0] * 874 / 384), scorer_hw[0] + 1)
    cam_w = max(round(scorer_hw[1] * 1164 / 512), scorer_hw[1] + 1)
    return apply_eval_roundtrip_during_training(
        flat_bchw, simulate_uint8=True, simulate_resize=True, ste_round=True,
        target_h=cam_h, target_w=cam_w,
    )


@torch.no_grad()
def _resegment(witness_hwc, net, scorer_hw):
    """Frame1 SegNet argmax (384x512) of a witness, through the EXACT roundtrip."""
    t = torch.from_numpy(np.asarray(witness_hwc)).float().permute(2, 0, 1)
    pair = torch.stack([t, t], dim=0)  # (2,3,H,W)
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


def _atom_to_plan(atom, witness_hw, l_inf):
    """Convert a canonical repair atom's (H,W,3) correction to a sparse byte-close plan."""
    corr = atom.correction  # (H,W,3) at scorer grid 384x512
    # resize correction to witness res (the frame we actually perturb + re-segment)
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
    cw = np.stack(ch_list, axis=2)  # (h,w,3)
    cw = np.clip(cw, -l_inf, l_inf)
    nz = np.abs(cw) > 0.5
    ys, xs, cs = np.nonzero(nz)
    flat = (ys * (w * 3) + xs * 3 + cs).astype(np.int64)
    order = np.argsort(flat)
    return FlipDeltaPlan(
        pair_idx=atom.pair_index, witness_hw=witness_hw,
        flat_idx=flat[order], values=cw[ys, xs, cs][order].astype(np.float64),
        n_flips_before=atom.n_support_pixels, target_flip_pixels=int(nz.any(axis=2).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--l-inf", type=float, default=48.0)
    ap.add_argument("--correction-fraction", type=float, default=1.0)
    ap.add_argument("--support-top-fraction", type=float, default=0.25)
    ap.add_argument("--boundary-margin-percentile", type=float, default=0.3)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_residual_flip_delta_20260611/surgical_v2_advisory_s.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    # The witness lives at the SegNet-native 384x512 grid: the canonical repair
    # atom builds region/flip/support from the SegNet argmax (always 384x512) and
    # the appearance gap on the SAME grid, so the perturbation + re-segmentation
    # must also be at 384x512 (no separate downscale). preprocess_input then
    # resizes to 512x384 internally, exactly as the scorer does.
    scorer_hw = (384, 512)
    video_path = "upstream/videos/0.mkv"

    cache = CACHE.format(n=48 if args.n_pairs <= 48 else 100)
    d = torch.load(cache, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")
    n = int(seg.shape[0])

    print(f"[1/5] fit+byte-close base (n={n}, ep={args.epochs})", flush=True)
    base_state, base_bytes, wall = fit_and_byte_close_base(
        net, seg, pose, scorer_hw=scorer_hw, epochs=args.epochs)

    print("[2/5] witnesses + base/GT argmax", flush=True)
    base_w0, base_w1 = _witnesses(base_state, net, n, scorer_hw)
    gt_w0, gt_w1 = _gt_witnesses(n, scorer_hw, video_path)
    lstar = seg.numpy().astype(np.int64)
    base_argmax = np.stack([_resegment(base_w1[i], net, scorer_hw) for i in range(n)], axis=0)
    d_seg_base, d_pose_base = _rescore(base_w0, base_w1, net, seg, pose, scorer_hw)
    print(f"      base d_seg={d_seg_base:.4f} d_pose={d_pose_base:.4e} bytes={base_bytes}", flush=True)

    print("[3/5] canonical surgical repair atoms + BIGGEST-RISK re-segmentation", flush=True)
    cfg = SegRepairAtomConfig(
        boundary_margin_percentile=args.boundary_margin_percentile,
        support_top_fraction=args.support_top_fraction,
        correction_fraction=args.correction_fraction,
        min_support_pixels=16,
    )
    plans, new_off, n_support_total = [], 0, 0
    pert_w1 = []
    for i in range(n):
        # RepairTargets at the WITNESS resolution (the frame we perturb + re-segment).
        # rendered argmax is measured at 384x512; downstream atom support is at the
        # render argmax grid. We build targets on the witness frames directly.
        try:
            # witness is 384x512 == the SegNet argmax grid; fragile_mask + gap +
            # rendered_argmax all share that grid (we don't use the #35 cone here).
            targets = RepairTargets.measure(
                segnet=net.segnet, rendered_frame1_hwc_unit255=base_w1[i],
                gt_frame1_hwc_unit255=gt_w1[i],
                fragile_mask=np.zeros((384, 512), dtype=bool),
            )
            atom = generate_seg_repair_atom(pair_index=i, targets=targets, config=cfg, prefer_mlx=False)
            repaired = atom.apply(base_w1[i])  # (h,w,3) corrected witness (correction resized to witness)
            plan = _atom_to_plan(atom, (scorer_hw[0], scorer_hw[1]), args.l_inf)
        except Frame1SegRepairAtomError:
            repaired = np.asarray(base_w1[i], dtype=np.float64)
            plan = FlipDeltaPlan(i, scorer_hw, np.zeros((0,), np.int64),
                                 np.zeros((0,), np.float64), 0, 0)
        plans.append(plan)
        n_support_total += plan.n_flips_before
        pert_w1.append(repaired)
        # off-target accounting under the EXACT re-segmentation.
        am = _resegment(repaired, net, scorer_hw)
        was = base_argmax[i] != lstar[i]
        now = am != lstar[i]
        new_off += int((now & ~was).sum())

    d_seg_fp, _ = _rescore(base_w0, pert_w1, net, seg, None, scorer_hw)
    print(f"      surgical fp d_seg={d_seg_fp:.4f} (base {d_seg_base:.4f}) "
          f"new off-target={new_off} support={n_support_total}", flush=True)

    print("[4/5] byte-close surgical correction + parity", flush=True)
    blob = byte_close_flip_delta(plans, l_inf_budget=args.l_inf)
    delta_bytes = len(blob)
    plans2 = inflate_flip_delta(blob)
    pert_bc = [apply_flip_delta(base_w1[i], plans2[i]) for i in range(n)]

    print("[5/5] RE-SCORE byte-closed surgical witness", flush=True)
    d_seg_bc, d_pose_bc = _rescore(base_w0, pert_bc, net, seg, pose, scorer_hw)

    total_bytes = base_bytes + delta_bytes
    pose_term = math.sqrt(10.0 * d_pose_bc) if d_pose_bc is not None else 0.0
    rate_term = 25.0 * total_bytes / ARCHIVE_DENOM
    advisory_s = 100.0 * d_seg_bc + pose_term + rate_term

    row = {
        "config": {"base_hw": [48, 64], "scorer_hw": list(scorer_hw), "epochs": args.epochs,
                   "l_inf": args.l_inf, "correction_fraction": args.correction_fraction,
                   "support_top_fraction": args.support_top_fraction,
                   "boundary_margin_percentile": args.boundary_margin_percentile},
        "base_bytes": base_bytes, "delta_bytes": delta_bytes, "total_bytes": total_bytes,
        "d_seg_base_no_delta": d_seg_base, "d_seg_fullprec_surgical": d_seg_fp,
        "d_seg_byteclosed_surgical": d_seg_bc,
        "d_pose_base_no_delta": d_pose_base, "d_pose_byteclosed_surgical": d_pose_bc,
        "support_pixels_total": n_support_total, "new_offtarget_flips": new_off,
        "delta_bytes_per_support_px": delta_bytes / max(n_support_total, 1),
        "advisory_s": advisory_s,
        "advisory_s_terms": {"seg_100xdseg": 100.0 * d_seg_bc, "pose_sqrt10dpose": pose_term,
                             "rate_25xbytes": rate_term},
        "train_wall_seconds": round(wall, 1),
        "axis_tag": "[macOS-CPU advisory]", "promotable": False, "score_claim": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"lane": "lane_cool_chic_residual_flip_delta_20260611",
                   "axis_tag": "[macOS-CPU advisory]", "promotable": False,
                   "method": "surgical_canonical_repair_atom", "n_pairs": n, "row": row}, f, indent=2)
    print(f"\nRESULT(surgical): base d_seg {d_seg_base:.4f} -> bc {d_seg_bc:.4f} (fp {d_seg_fp:.4f}) | "
          f"delta {delta_bytes}B | new off-target {new_off} (support {n_support_total}px) | "
          f"d_pose {d_pose_base:.3e}->{d_pose_bc:.3e} | S={advisory_s:.4f}", flush=True)
    print("SURGICAL-V2 DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
