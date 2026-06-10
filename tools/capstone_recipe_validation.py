# SPDX-License-Identifier: MIT
"""Real-scorer recipe-validation for the capstone VQ-NeRV + FiLM-pose (Task #78).

Runs the joint d_seg/d_pose descent on the EXACT EfficientNet-B2 SegNet +
FastViT PoseNet at 384x512 with eval_roundtrip, on a small pair subset ($0,
local torch-CPU scorer + MLX-GPU decode). NO MPS. GT only via
frame_utils.yuv420_to_rgb (build_gt_targets).

This is the recipe-validation row the prior capstone memo left pending. Emits a
scorer_quotient_candidate_row.v1 JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from tac.capstone_vq_nerv.capstone_trainer import (
    CapstoneTrainConfig,
    CapstoneTrainer,
)
from tac.capstone_vq_nerv.vq_nerv_bundle import (
    CapstoneVqNervBundle,
    CapstoneVqNervConfig,
)
from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--base-channels", type=int, default=36)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--muon-lr", type=float, default=2.0e-4)
    ap.add_argument("--adamw-lr", type=float, default=3.0e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--commitment-weight", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    print(
        f"[capstone] loading frozen scorer + {args.max_pairs} GT pairs "
        f"(base_ch={args.base_channels})...",
        file=sys.stderr,
        flush=True,
    )
    dnet = load_frozen_distortion_net(device="cpu")
    seg_targets, pose_targets, n_pairs = build_gt_targets(
        dnet, max_pairs=args.max_pairs, device="cpu"
    )
    t_targets = time.time() - t0
    print(
        f"[capstone] GT targets cached: n_pairs={n_pairs} in {t_targets:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    pose_store = pose_targets.numpy().astype(np.float32)

    bundle_cfg = CapstoneVqNervConfig(
        num_pairs=n_pairs,
        base_channels=args.base_channels,
        seed=args.seed,
    )
    bundle = CapstoneVqNervBundle(bundle_cfg)
    param_count = sum(
        int(np.prod(v.shape))
        for _, v in _flatten(bundle.trainable_parameters())
    )
    decoder_param_count = sum(
        int(np.prod(v.shape))
        for k, v in _flatten(bundle.trainable_parameters())
        if "decoder" in k
    )
    print(
        f"[capstone] bundle: total trainable params={param_count} "
        f"decoder params={decoder_param_count}",
        file=sys.stderr,
        flush=True,
    )

    bridge = TorchScorerBridge(
        dnet,
        seg_targets,
        pose_targets,
        seg_loss_form="ce_seg_loss",
        seg_weight=100.0,
        pose_weight=1.0,
        eval_roundtrip=True,
    )

    train_cfg = CapstoneTrainConfig(
        epochs=args.epochs,
        batch_size=min(8, n_pairs),
        eval_every=args.eval_every,
        seed=args.seed,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        grad_clip=args.grad_clip,
        grad_clip_muon=args.grad_clip,
        ema_decay=args.ema_decay,
        commitment_weight=args.commitment_weight,
        use_ema_for_eval=False,  # eval LIVE weights (the #82 EMA-lag landmine).
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, train_cfg)

    t1 = time.time()
    summary = trainer.train()
    t_train = time.time() - t1

    row = {
        "schema": "scorer_quotient_candidate_row.v1",
        "substrate": "capstone_vq_nerv_film_pose",
        "axis_tag": "[macOS-MLX research-signal]",
        "promotable": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "n_pairs": n_pairs,
        "base_channels": args.base_channels,
        "epochs": args.epochs,
        "muon_lr": args.muon_lr,
        "adamw_lr": args.adamw_lr,
        "grad_clip": args.grad_clip,
        "commitment_weight": args.commitment_weight,
        "total_trainable_params": param_count,
        "decoder_params": decoder_param_count,
        "d_seg_initial": summary["d_seg_initial"],
        "d_seg_final": summary["d_seg_final"],
        "d_seg_best": summary["d_seg_best"],
        "d_pose_initial": summary["d_pose_initial"],
        "d_pose_final": summary["d_pose_final"],
        "seg_descended": summary["seg_descended"],
        "pose_held": summary["pose_held"],
        "train_seconds": round(t_train, 1),
        "targets_seconds": round(t_targets, 1),
        "trajectory": summary["trajectory"],
    }
    print(json.dumps(row, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(row, fh, indent=2)
        print(f"[capstone] wrote {args.out}", file=sys.stderr, flush=True)
    return 0


def _flatten(tree):
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


if __name__ == "__main__":
    raise SystemExit(main())
