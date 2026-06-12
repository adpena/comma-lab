#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Resumable, detached capstone PR95-curriculum daemon (the LOCAL-MLX-DREAM run).

The dream (operator, standing): a super-long local MLX run that descends d_seg to
the basin on the n600 capstone basis, byte-closes, and is directly portable to a
contest exact-eval — sustained over days, RESUMABLE so a death costs at most one
in-flight checkpoint.

This driver wraps the canonical ``CapstoneTrainer`` + the PR95 8-stage curriculum
with PER-EPOCH checkpoint+resume (``tac.capstone_vq_nerv.checkpoint``) and a
done-marker-on-exit. It is the missing resumable piece: the standing
``run_capstone_campaign.py`` runs the curriculum in one monolithic pass with NO
mid-run checkpoint, so a SIGURG/OOM/reboot loses the whole run (the gap that lost
the earlier 2x2 capacity-ablation arms). Here, every epoch's end atomically
overwrites the single live checkpoint; on restart the driver loads it and
continues the EXACT trajectory (proven bit-identical by
``test_checkpoint.py::test_resume_is_bit_identical_to_uninterrupted_run``).

Launch (detached daemon, survives the session):

    nohup .venv/bin/python experiments/run_capstone_resumable_curriculum.py \
        --max-pairs 600 --base-channels 20 --carrier stored_latent \
        --curriculum-total-epochs 2000 \
        --optimizer-schedule muon_throughout --muon-lr 0.03 --grad-clip 50 \
        --scorer-backend mlx_gpu --authority-recheck-every 50 \
        --checkpoint-every 1 --eval-every 10 \
        --out-dir experiments/results/capstone_n600_resumable_$(date +%Y%m%dT%H%M) \
        < /dev/null > .omx/tmp/capstone_n600_resumable.log 2>&1 &
    disown

Resume: re-run the SAME command with the SAME --out-dir; the driver detects the
checkpoint and continues. Idempotent: if the done-marker is present it exits 0.

Authority: the per-step gradient is torch-CPU (default) or MLX-GPU (research
signal); the reported d_seg/d_pose are ALWAYS torch-CPU (the eval authority).
$0, local, no MPS. NOT a pointer move — a sub-threshold advisory here GATES a
paired contest-CPU+CUDA exact eval (the only pointer-moving step).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _stream_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--carrier", default="stored_latent", choices=("vq_index", "stored_latent"))
    ap.add_argument("--curriculum-total-epochs", type=int, default=2000)
    ap.add_argument("--optimizer-schedule", default="muon_throughout",
                    choices=("pr95_adamw_then_muon", "muon_throughout"))
    ap.add_argument("--muon-lr", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--scorer-backend", default="torch_cpu_bridge",
                    choices=("torch_cpu_bridge", "mlx_gpu"))
    ap.add_argument("--authority-recheck-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=1, help="epochs between checkpoints")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.scorer_backend == "mlx_gpu":
        import os
        os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")

    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.checkpoint import (
        CheckpointPosition,
        checkpoint_exists,
        is_done,
        load_checkpoint,
        save_checkpoint,
        write_done_marker,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle, CapstoneVqNervConfig
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum, resolve_use_muon
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    out = Path(args.out_dir)
    ckpt_dir = out / "checkpoint"
    traj_path = out / "trajectory.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    if is_done(out):
        print(f"[done] {out} already has a done-marker; exiting 0.", flush=True)
        return

    torch.set_num_threads(min(6, torch.get_num_threads()))

    # Build the canonical trainer (mirrors run_capstone_campaign.main).
    net = load_frozen_distortion_net(device=args.device)
    cache = Path(args.targets_cache) / f"gt_targets_n{args.max_pairs}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location=args.device, weights_only=False)
        seg_t, pose_t = blob["seg"][: args.max_pairs], blob["pose"][: args.max_pairs]
        print(f"[targets] loaded cache {cache}", flush=True)
    else:
        seg_t, pose_t, _ = build_gt_targets(args.max_pairs, device=args.device)
        seg_t, pose_t = seg_t[: args.max_pairs], pose_t[: args.max_pairs]
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seg": seg_t, "pose": pose_t, "n": args.max_pairs}, cache)
        print(f"[targets] built + cached {cache}", flush=True)

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=args.max_pairs, base_channels=args.base_channels,
                             carrier=args.carrier, seed=args.seed)
    )
    bridge = TorchScorerBridge(net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
                               seg_weight=args.seg_weight, pose_weight=args.pose_weight,
                               eval_roundtrip=True)
    pose_store = pose_t.float().cpu().numpy()
    cfg = CapstoneTrainConfig(
        epochs=1, batch_size=args.batch_size, seg_weight=args.seg_weight,
        pose_weight=args.pose_weight, muon_lr=args.muon_lr, grad_clip=args.grad_clip,
        grad_clip_muon=args.grad_clip, eval_every=args.eval_every, seed=args.seed,
        scorer_backend=args.scorer_backend, authority_recheck_every=args.authority_recheck_every,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)

    stages = build_pr95_8stage_curriculum(total_epochs=args.curriculum_total_epochs)

    # Resume position.
    resume_stage, resume_epoch = 0, 0
    if checkpoint_exists(ckpt_dir):
        pos = load_checkpoint(trainer, ckpt_dir)
        resume_stage, resume_epoch = pos.stage_index, pos.epoch_in_stage
        print(f"[resume] loaded checkpoint at stage={resume_stage} epoch_in_stage={resume_epoch}",
              flush=True)
    else:
        d0 = trainer.exact_d_seg(use_ema=False)
        p0 = trainer.mean_d_pose(use_ema=False)
        print(f"[init] n={args.max_pairs} base_ch={args.base_channels} d_seg={d0:.6f} "
              f"d_pose={p0:.6f} backend={args.scorer_backend} stages={len(stages)} "
              f"total_epochs={sum(s.epochs for s in stages)}", flush=True)
        _stream_row(traj_path, {"event": "init", "exact_d_seg": d0, "mean_d_pose": p0,
                                "n_pairs": args.max_pairs, "base_channels": args.base_channels})

    t_start = time.time()
    best_d_seg = float("inf")
    rng = np.random.RandomState(args.seed)

    for si in range(resume_stage, len(stages)):
        spec = stages[si]
        trainer.configure_stage(spec, optimizer_schedule=args.optimizer_schedule)
        # Mid-stage resume: configure_stage reset _current_epoch=0; fast-forward the
        # cosine epoch pointer (the optimizer/EMA/weights are already restored from
        # the checkpoint, so we resume the SAME state — only the epoch counter and the
        # per-epoch permutation advance from `start_epoch`).
        start_epoch = resume_epoch if si == resume_stage else 0
        use_muon = resolve_use_muon(spec, args.optimizer_schedule)
        print(f"[stage {si}/{len(stages)-1}] {spec.name} epochs={spec.epochs} "
              f"start_epoch={start_epoch} seg_loss={spec.seg_loss_form} use_muon={use_muon}",
              flush=True)
        for epoch in range(start_epoch, spec.epochs):
            trainer._current_epoch = epoch
            lr_scale = trainer._lr_scale_for_epoch(epoch)
            perm = rng.permutation(trainer.n_pairs)
            for s in range(0, trainer.n_pairs, args.batch_size):
                idx = perm[s : s + args.batch_size]
                # trainer.step() ALREADY updates the VQ codebook EMA
                # (ema_update_from_last) AND the weight-EMA shadow (_ema.update)
                # internally (capstone_trainer.py:485-487) — do NOT double-update.
                trainer.step(idx, lr_scale=lr_scale)

            global_epoch = sum(s.epochs for s in stages[:si]) + epoch + 1
            if (epoch + 1) % args.eval_every == 0 or epoch == spec.epochs - 1:
                d = trainer.exact_d_seg()
                p = trainer.mean_d_pose()
                best_d_seg = min(best_d_seg, d)
                row = {"stage_index": si, "stage": spec.name, "epoch_in_stage": epoch + 1,
                       "global_epoch": global_epoch, "exact_d_seg": d, "mean_d_pose": p,
                       "best_d_seg": best_d_seg, "lr_scale": lr_scale,
                       "elapsed_s": time.time() - t_start}
                _stream_row(traj_path, row)
                print(f"[stage {si} ep {epoch+1}/{spec.epochs} g{global_epoch}] "
                      f"d_seg={d:.6f} d_pose={p:.6f} best={best_d_seg:.6f} "
                      f"lr={lr_scale:.4g} ({row['elapsed_s']:.0f}s)", flush=True)

            if (epoch + 1) % args.checkpoint_every == 0:
                # Checkpoint AFTER completing this epoch -> resume runs epoch+1..end.
                save_checkpoint(trainer, ckpt_dir, CheckpointPosition(si, epoch + 1),
                                extra={"global_epoch": global_epoch, "best_d_seg": best_d_seg})
        resume_epoch = 0  # subsequent stages start fresh

    # Final EMA-shadow d_seg (the export/inference point) + done marker.
    final_d_seg = trainer.exact_d_seg()
    final_d_pose = trainer.mean_d_pose()
    summary = {"final_d_seg": final_d_seg, "final_d_pose": final_d_pose,
               "best_d_seg": best_d_seg, "stages": len(stages),
               "total_epochs": sum(s.epochs for s in stages),
               "elapsed_s": time.time() - t_start, "scorer_backend": args.scorer_backend,
               "n_pairs": args.max_pairs, "base_channels": args.base_channels}
    write_done_marker(out, summary)
    print(f"[DONE] final d_seg={final_d_seg:.6f} d_pose={final_d_pose:.6f} "
          f"best={best_d_seg:.6f} elapsed={summary['elapsed_s']:.0f}s", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
