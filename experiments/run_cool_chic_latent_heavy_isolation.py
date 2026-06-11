# SPDX-License-Identifier: MIT
"""DECISIVE optimal-capacity-allocation test: latent-heavy vs synth-heavy.

The arm-C confound (run_fields_medal_hinge_abc.py): arm C bumped synth_hidden
16->48 AND grids 4x2->5x3 TOGETHER and got only -10% d_seg for 2.7x bytes. But
the byte split shows arm C was SYNTH-byte-dominated (latent ~611B / weight
~2116B = 78% synth). So arm C never tested LATENT capacity in isolation.

This driver isolates the two axes. Cool-Chic codes per-frame LATENTS cheaply
(ARM entropy-coded shared grids) but stores the SYNTH once (int8, expensive per
param). The decisive question: does pushing the cheaply-coded LATENT capacity UP
(finer/more shared grids + more channels) while holding the SYNTH MODEST reach
the corrected sub-frontier d_seg bar (~0.0011-0.0017) at a LOW byte-closed rate,
where arm-C's synth-heavy point could not?

Reuses run_arm() + ScoreAwareTrainer.exact_d_seg (EXACT re-segmented render,
torch-CPU AUTHORITY) + the EXACT upstream/modules.py SegNet. [macOS-CPU
advisory] non-promotable; NO FAKE.

Latent capacity is in the SHARED frame0 grids (frame1 = feat0 + coarse delta, so
finer feat0 directly adds frame1 / SegNet-term capacity). synth_hidden held at a
MODEST 16 across the latent sweep so byte growth is in the cheap latent term.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt"


def run_arm(
    net, seg, pose, *, label, seg_loss_form, base_h, base_w, n_grids, cpg,
    synth_hidden, out_hw, scorer_hw, epochs, decoder_lr, seed,
):
    torch.manual_seed(seed)
    n = int(seg.shape[0])
    spec = CoolChicGridSpec(base_h=base_h, base_w=base_w, n_grids=n_grids, channels_per_grid=cpg)
    carrier = CoolChicPairCarrier(n_pairs=n, spec=spec, synth_hidden=synth_hidden, out_hw=out_hw)
    cfg = ScoreAwareLoopConfig(
        epochs=epochs, batch_size=n, scorer_hw=scorer_hw, pose_enabled=pose is not None,
        eval_every=max(epochs // 8, 1), seg_loss_form=seg_loss_form, decoder_lr=decoder_lr,
        latent_lr_mult=10.0, ema_decay=0.99, seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg, pose, cfg)

    # NO-FAKE gradient-flow guard.
    idx = torch.arange(n)
    loss_parts = tr.compute_loss(idx)
    loss_parts["total"].backward()
    grad_w1 = float(carrier.w1.grad.abs().sum()) if carrier.w1.grad is not None else 0.0
    grad_grid = float(carrier.latent_grids[0].grad.abs().sum()) if carrier.latent_grids[0].grad is not None else 0.0
    carrier.zero_grad(set_to_none=True)

    bytes_breakdown = carrier.charged_bytes()
    t0 = time.time()
    res = tr.train()
    wall = time.time() - t0
    d_seg_final_ema = tr.exact_d_seg(use_ema=True)
    d_seg_final_live = tr.exact_d_seg(use_ema=False)  # EMA-lag guard
    bytes_post = carrier.charged_bytes()  # post-train ARM bytes (the real split)
    return {
        "label": label,
        "seg_loss_form": seg_loss_form,
        "config": {
            "base_hw": [base_h, base_w], "n_grids": n_grids, "channels_per_grid": cpg,
            "synth_hidden": synth_hidden, "out_hw": list(out_hw),
            "scorer_hw": list(scorer_hw), "epochs": epochs, "decoder_lr": decoder_lr,
        },
        "charged_bytes_init": bytes_breakdown,
        "charged_bytes_post": bytes_post,
        "grad_w1_abs_sum_first_step": grad_w1,
        "grad_grid0_abs_sum_first_step": grad_grid,
        "d_seg_initial": res["d_seg_initial"],
        "d_seg_best_ema": res["d_seg_best_ema"],
        "d_seg_final_ema_exact": d_seg_final_ema,
        "d_seg_final_live_exact": d_seg_final_live,
        "descended": res["descended"],
        "trajectory": res["trajectory"],
        "train_wall_seconds": round(wall, 1),
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }


def _p(r):
    cbp = r["charged_bytes_post"]
    print(
        f"{r['label']}: d_seg init={r['d_seg_initial']:.4f} best_ema={r['d_seg_best_ema']:.4f} "
        f"final_ema={r['d_seg_final_ema_exact']:.4f} live={r['d_seg_final_live_exact']:.4f} | "
        f"grad_grid={r['grad_grid0_abs_sum_first_step']:.2e} | "
        f"POST bytes total={cbp['total_bytes']:.0f} (latent {cbp['latent_bytes']:.0f} + w {cbp['weight_bytes']:.0f}) "
        f"latent_frac={cbp['latent_bytes']/max(cbp['total_bytes'],1):.2f} | {r['train_wall_seconds']}s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_score_aware_basis_20260611/latent_heavy_isolation.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")
    scorer_hw = (args.scorer_h, args.scorer_w)
    HINGE = "hard_pixel_curriculum_seg_loss"  # the default hinge curriculum

    rows = []
    # Decisive 4-arm isolation. out_hw held at (96,128) across ALL arms so render
    # res is NOT a confound — only synth_hidden vs grid resolution/channels move.
    OHW = (96, 128)
    arms = [
        # (label, base_h, base_w, n_grids, cpg, synth_hidden)
        # REF synth-heavy (arm-C-equivalent: big synth, coarse latents).
        ("REF_synth_heavy_armC", 24, 32, 5, 3, 48),
        # latent-heavy sweep: MODEST synth=16, push shared-grid latent capacity UP.
        ("L0_compact",           24, 32, 4, 2, 16),
        ("L2_finer",             40, 56, 6, 4, 16),
        ("L3_latent_max",        48, 64, 6, 6, 16),
    ]
    for (label, bh, bw, ng, cpg, sh) in arms:
        rows.append(run_arm(
            net, seg, pose, label=label, seg_loss_form=HINGE,
            base_h=bh, base_w=bw, n_grids=ng, cpg=cpg, synth_hidden=sh, out_hw=OHW,
            scorer_hw=scorer_hw, epochs=args.epochs, decoder_lr=3e-3, seed=0,
        ))
        _p(rows[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "lane": "lane_cool_chic_score_aware_basis_20260611",
            "test": "latent_heavy_vs_synth_heavy_capacity_isolation",
            "axis_tag": "[macOS-CPU advisory]",
            "promotable": False,
            "n_pairs": args.n_pairs,
            "epochs": args.epochs,
            "corrected_bar_d_seg": [0.0011, 0.0017],
            "basin_d_seg_reference": 5.6e-4,
            "rows": rows,
        }, f, indent=2)
    print("LATENT-HEAVY ISOLATION DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
