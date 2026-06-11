# SPDX-License-Identifier: MIT
"""FIELDS-MEDAL decisive A/B/C: does the margin-polytope hinge (+ capacity) break
the Cool-Chic d_seg base-convergence wall, or is the basin capacity-bound?

A: CE baseline (the existing ce_seg_loss path) — re-confirm the plateau.
B: margin-polytope hinge + hard-pixel curriculum, SAME synth size as A.
C: hinge + curriculum + a TEXTURE-RICHER synth (more synth_hidden + grids/channels).

Decisive observable: the EXACT re-segmented-render d_seg (argmax(live SegNet on
render) != GT SegNet argmax) via ScoreAwareTrainer.exact_d_seg — NOT the
surrogate. torch-CPU is the AUTHORITY. [macOS-CPU advisory] non-promotable.

Honesty: optimize against the EXACT upstream/modules.py SegNet (load_frozen_
distortion_net) + GT pose; report C's byte cost; verdict is honest about whether
B/C reach d_seg < ~3e-3 (basin-adjacent) or all plateau ~0.008-0.03.
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

    # NO-FAKE gradient-flow guard: one forward/backward must produce a nonzero
    # gradient on the carrier synthesis weights (the hinge is not inert).
    idx = torch.arange(n)
    loss_parts = tr.compute_loss(idx)
    loss_parts["total"].backward()
    grad_w1 = float(carrier.w1.grad.abs().sum()) if carrier.w1.grad is not None else 0.0
    carrier.zero_grad(set_to_none=True)

    bytes_breakdown = carrier.charged_bytes()
    t0 = time.time()
    res = tr.train()
    wall = time.time() - t0
    # exact final EMA-shadow d_seg (the inference checkpoint, re-segmented render).
    d_seg_final = tr.exact_d_seg(use_ema=True)
    return {
        "label": label,
        "seg_loss_form": seg_loss_form,
        "config": {
            "base_hw": [base_h, base_w], "n_grids": n_grids, "channels_per_grid": cpg,
            "synth_hidden": synth_hidden, "out_hw": list(out_hw),
            "scorer_hw": list(scorer_hw), "epochs": epochs, "decoder_lr": decoder_lr,
        },
        "charged_bytes": bytes_breakdown,
        "grad_w1_abs_sum_first_step": grad_w1,
        "d_seg_initial": res["d_seg_initial"],
        "d_seg_best_ema": res["d_seg_best_ema"],
        "d_seg_final_ema_exact": d_seg_final,
        "descended": res["descended"],
        "clip_fired_fraction_final": res["clip_fired_fraction_final"],
        "trajectory": res["trajectory"],
        "train_wall_seconds": round(wall, 1),
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_score_aware_basis_20260611/fields_medal_hinge_abc.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")
    scorer_hw = (args.scorer_h, args.scorer_w)

    rows = []
    # A: CE baseline, compact synth.
    rows.append(run_arm(
        net, seg, pose, label="A_ce_baseline", seg_loss_form="ce_seg_loss",
        base_h=24, base_w=32, n_grids=4, cpg=2, synth_hidden=16, out_hw=(64, 96),
        scorer_hw=scorer_hw, epochs=args.epochs, decoder_lr=3e-3, seed=0,
    ))
    _p(rows[-1])
    # B: hinge + hard-pixel curriculum, SAME synth size as A.
    rows.append(run_arm(
        net, seg, pose, label="B_hinge_curriculum_same_size",
        seg_loss_form="hard_pixel_curriculum_seg_loss",
        base_h=24, base_w=32, n_grids=4, cpg=2, synth_hidden=16, out_hw=(64, 96),
        scorer_hw=scorer_hw, epochs=args.epochs, decoder_lr=3e-3, seed=0,
    ))
    _p(rows[-1])
    # C: hinge + curriculum + texture-richer synth (more hidden + grids/channels).
    rows.append(run_arm(
        net, seg, pose, label="C_hinge_texture_richer",
        seg_loss_form="hard_pixel_curriculum_seg_loss",
        base_h=24, base_w=32, n_grids=5, cpg=3, synth_hidden=48, out_hw=(96, 128),
        scorer_hw=scorer_hw, epochs=args.epochs, decoder_lr=3e-3, seed=0,
    ))
    _p(rows[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "lane": "lane_cool_chic_score_aware_basis_20260611",
            "test": "fields_medal_margin_polytope_hinge_decisive_abc",
            "axis_tag": "[macOS-CPU advisory]",
            "promotable": False,
            "n_pairs": args.n_pairs,
            "basin_d_seg_reference": 5.6e-4,
            "rows": rows,
        }, f, indent=2)
    print("ABC DONE", args.out, flush=True)


def _p(r):
    cb = r["charged_bytes"]
    print(
        f"{r['label']}: d_seg init={r['d_seg_initial']:.4f} best_ema={r['d_seg_best_ema']:.4f} "
        f"final_exact={r['d_seg_final_ema_exact']:.4f} | grad_w1={r['grad_w1_abs_sum_first_step']:.3e} | "
        f"bytes total={cb['total_bytes']:.0f} (latent {cb['latent_bytes']:.0f} + w {cb['weight_bytes']:.0f}) | "
        f"clip={r['clip_fired_fraction_final']} | {r['train_wall_seconds']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
