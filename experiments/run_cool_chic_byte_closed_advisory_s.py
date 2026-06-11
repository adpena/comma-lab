# SPDX-License-Identifier: MIT
"""Cool-Chic BYTE-CLOSED advisory-S runner — the decisive measure.

Lane ``lane_cool_chic_score_aware_basis_20260611``. Converts B1's ARM byte
ESTIMATE into a REAL byte-closed archive + first real advisory S row:

  1. Re-fit the Cool-Chic carrier (live-SegNet + PoseNet ``ScoreAwareTrainer``)
     for each grid config (pose ENABLED so d_pose is part of the fit).
  2. Byte-close it (``cool_chic_byte_close.byte_close``): quantize latents ->
     range-code (REAL coder) + int8-pack synth weights -> single blob.
  3. ``inflate_numpy`` the blob (pure numpy, the portability contract).
  4. RE-SCORE the byte-closed reloaded render through the EXACT frozen
     DistortionNet: d_seg = re-segmented argmax-disagreement on the inflated
     RGB; d_pose = MSE of PoseNet output on the inflated 2-frame pair (the
     contest d_pose, whose Lagrangian term is sqrt(10*d_pose)).
  5. advisory S = 100*d_seg + sqrt(10*d_pose) + 25*real_bytes/37_545_489.

Honesty (CLAUDE.md "Forbidden score claims" / MPS / MLX rules):
  - torch-CPU ONLY (GPU busy; NO MPS — corrupts SegNet ~2x).
  - Every row ``[macOS-CPU advisory]`` / ``promotable=False``. The d_seg/d_pose
    are LIVE frozen-scorer measurements on the BYTE-CLOSED render, but contest S
    still requires Linux-x86_64 evaluate.py on the n600 archive.
  - Reports REAL bytes vs ARM estimate (the gap factor) explicitly.
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
from tac.score_aware_loop.live_segnet_loss import exact_d_seg_from_logits
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n{n}.pt"
ARCHIVE_DENOM = 37_545_489.0


@torch.no_grad()
def score_byte_closed(
    state, net, seg_targets, pose_targets, *, scorer_hw, eval_roundtrip=True
):
    """Re-score the BYTE-CLOSED numpy render through the frozen scorer.

    Renders each pair from the inflated numpy state, runs the SAME eval
    roundtrip the trainer uses, forwards the frozen SegNet (d_seg = re-segmented
    argmax disagreement) and PoseNet (d_pose = MSE on first 6 dims).
    """
    n = int(seg_targets.shape[0])
    h, w = scorer_hw
    pairs = []
    for i in range(n):
        rgb0, rgb1 = render_pair_numpy(state, i)  # (3,H,W) numpy in [0,255]
        pairs.append(np.stack([rgb0, rgb1], axis=0))  # (2,3,H,W)
    arr = np.stack(pairs, axis=0)  # (n,2,3,h0,w0)
    t = torch.from_numpy(arr).float()
    b = t.shape[0]
    flat = t.reshape(b * 2, 3, t.shape[-2], t.shape[-1])
    flat = F.interpolate(flat, size=scorer_hw, mode="bilinear", align_corners=False)
    if eval_roundtrip:
        cam_h = max(round(scorer_hw[0] * 874 / 384), scorer_hw[0] + 1)
        cam_w = max(round(scorer_hw[1] * 1164 / 512), scorer_hw[1] + 1)
        flat = apply_eval_roundtrip_during_training(
            flat, simulate_uint8=True, simulate_resize=True, ste_round=True,
            target_h=cam_h, target_w=cam_w,
        )
    else:
        flat = flat.clamp(0.0, 255.0)
    bchw = flat.reshape(b, 2, 3, h, w)
    bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
    posenet_in, segnet_in = net.preprocess_input(bhwc)
    seg_out = net.segnet(segnet_in)
    d_seg = exact_d_seg_from_logits(seg_out, seg_targets.long())
    d_pose = None
    if pose_targets is not None:
        pose_out = net.posenet(posenet_in)
        pose_pred = pose_out["pose"][:, :6]
        d_pose = float(F.mse_loss(pose_pred, pose_targets).item())
    return d_seg, d_pose


def run_config(
    net, seg, pose, *, base_h, base_w, n_grids, cpg, synth_hidden, out_hw,
    scorer_hw, epochs, decoder_lr, grid_step, delta_step, seed=0,
):
    torch.manual_seed(seed)
    n_pairs = int(seg.shape[0])
    spec = CoolChicGridSpec(base_h=base_h, base_w=base_w, n_grids=n_grids, channels_per_grid=cpg)
    carrier = CoolChicPairCarrier(n_pairs=n_pairs, spec=spec, synth_hidden=synth_hidden, out_hw=out_hw)
    cfg = ScoreAwareLoopConfig(
        epochs=epochs, batch_size=n_pairs, scorer_hw=scorer_hw, pose_enabled=pose is not None,
        eval_every=max(epochs // 4, 1), seg_loss_form="ce_seg_loss", decoder_lr=decoder_lr,
        latent_lr_mult=10.0, ema_decay=0.99, seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg, pose, cfg)
    t0 = time.time()
    res = tr.train()
    train_wall = time.time() - t0

    # Apply the EMA shadow to the carrier (the inference checkpoint per CLAUDE.md
    # "EMA"): byte-close the shadow weights, not the live final-epoch weights.
    ema_orig = tr._ema_for_eval.apply_to(tr.model)
    arm_estimate = carrier.charged_bytes()  # ARM estimate of the SHADOW
    state = carrier_to_state(carrier)
    # restore live weights (defensive; we are done with the carrier anyway).
    carrier.load_state_dict(ema_orig)

    blob = byte_close(state, grid_step=grid_step, delta_step=delta_step)
    real_bytes = len(blob)
    state2 = inflate_numpy(blob)

    # parity proof: full-precision numpy render vs byte-closed numpy render.
    parity_max_delta = 0.0
    for i in range(min(n_pairs, 4)):
        r0a, r1a = render_pair_numpy(state, i)
        r0b, r1b = render_pair_numpy(state2, i)
        parity_max_delta = max(parity_max_delta, float(np.abs(r0a - r0b).max()),
                               float(np.abs(r1a - r1b).max()))

    # RE-SCORE the byte-closed reloaded render (the decisive measure).
    d_seg_bc, d_pose_bc = score_byte_closed(state2, net, seg, pose, scorer_hw=scorer_hw)
    # also the full-precision (pre-byte-close) re-score, to isolate quant loss.
    d_seg_fp, d_pose_fp = score_byte_closed(state, net, seg, pose, scorer_hw=scorer_hw)

    rate_term = 25.0 * real_bytes / ARCHIVE_DENOM
    pose_term = math.sqrt(10.0 * d_pose_bc) if d_pose_bc is not None else 0.0
    advisory_s = 100.0 * d_seg_bc + pose_term + rate_term

    return {
        "config": {
            "base_hw": [base_h, base_w], "n_grids": n_grids, "channels_per_grid": cpg,
            "synth_hidden": synth_hidden, "out_hw": list(out_hw), "scorer_hw": list(scorer_hw),
            "epochs": epochs, "decoder_lr": decoder_lr,
            "grid_step": grid_step, "delta_step": delta_step,
        },
        "arm_estimate_bytes": arm_estimate,
        "real_bytes": real_bytes,
        "real_vs_arm_factor": real_bytes / max(arm_estimate["total_bytes"], 1e-9),
        "parity_max_delta_255scale": parity_max_delta,
        "d_seg_initial": res["d_seg_initial"],
        "d_seg_best_ema_train": res["d_seg_best_ema"],
        "d_seg_fullprec_rescored": d_seg_fp,
        "d_seg_byteclosed_rescored": d_seg_bc,
        "d_pose_fullprec_rescored": d_pose_fp,
        "d_pose_byteclosed_rescored": d_pose_bc,
        "advisory_s": advisory_s,
        "advisory_s_terms": {
            "seg_100xdseg": 100.0 * d_seg_bc,
            "pose_sqrt10dpose": pose_term,
            "rate_25xbytes": rate_term,
        },
        "train_wall_seconds": round(train_wall, 1),
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--delta-step", type=float, default=0.05)
    ap.add_argument("--only-config", type=int, default=-1,
                    help="if >=0, run only configs[i] and APPEND to the out JSON")
    ap.add_argument("--epochs-override", type=int, default=-1)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_score_aware_basis_20260611/byte_closed_advisory_s.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cache = CACHE.format(n=48 if args.n_pairs <= 48 else 100)
    d = torch.load(cache, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")

    # Configs: byte-close the curve tail (smaller grids cheap) + the basin config
    # (48x64 grids = B1's deep-fit config). epochs scaled so total wall is bounded.
    configs = [
        # (base_h, base_w, n_grids, cpg, synth_hidden, out_h, out_w, epochs)
        (24, 32, 3, 2, 16, 48, 64, 30),
        (24, 32, 4, 2, 16, 64, 96, 30),
        (48, 64, 4, 3, 24, 96, 128, 45),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Resume-by-inspection: load any existing rows so single-config calls append.
    rows = []
    if args.only_config >= 0 and Path(args.out).exists():
        try:
            rows = json.load(open(args.out)).get("rows", [])
        except Exception:
            rows = []
    if args.only_config >= 0:
        configs = [configs[args.only_config]]
    for (bh, bw, ng, cpg, sh, oh, ow, ep) in configs:
        if args.epochs_override >= 0:
            ep = args.epochs_override
        row = run_config(
            net, seg, pose, base_h=bh, base_w=bw, n_grids=ng, cpg=cpg,
            synth_hidden=sh, out_hw=(oh, ow), scorer_hw=(args.scorer_h, args.scorer_w),
            epochs=ep, decoder_lr=3e-3, grid_step=args.grid_step, delta_step=args.delta_step,
        )
        rows.append(row)
        c = row["config"]
        print(
            f"base={bh}x{bw} g{ng}c{cpg} h{sh} ep{ep}: "
            f"REAL {row['real_bytes']}B (ARM est {row['arm_estimate_bytes']['total_bytes']:.0f}B, "
            f"x{row['real_vs_arm_factor']:.2f}) | parity {row['parity_max_delta_255scale']:.3f} | "
            f"d_seg bc={row['d_seg_byteclosed_rescored']:.4f} (fp {row['d_seg_fullprec_rescored']:.4f}, "
            f"train_ema {row['d_seg_best_ema_train']:.4f}) | "
            f"d_pose bc={row['d_pose_byteclosed_rescored']:.2e} | "
            f"S={row['advisory_s']:.4f} "
            f"[seg {row['advisory_s_terms']['seg_100xdseg']:.4f} + "
            f"pose {row['advisory_s_terms']['pose_sqrt10dpose']:.4f} + "
            f"rate {row['advisory_s_terms']['rate_25xbytes']:.2e}] "
            f"{row['train_wall_seconds']}s",
            flush=True,
        )
        with open(args.out, "w") as f:
            json.dump(
                {
                    "lane": "lane_cool_chic_score_aware_basis_20260611",
                    "axis_tag": "[macOS-CPU advisory]",
                    "promotable": False,
                    "frontier_for_reference": 0.19110,
                    "n_pairs": args.n_pairs,
                    "rows": rows,
                },
                f, indent=2,
            )
    print("BYTE-CLOSED ADVISORY-S DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
