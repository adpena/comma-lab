#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Thin launcher for the base_ch=20 n600 basin on the SPLIT-BY-HEAD gradient
backend (SegNet fwd+bwd on MPS / PoseNet fwd+bwd on the CPU authority, the two
frame cotangents summed at the frame tensor — the pose-axis salvage).

Why a thin launcher (not ``python -m tac.torch_vehicle.run``): the production
``run.py`` always uses the UNCAPPED ``precompute_targets`` (~2.5 h CPU to build
all 600 GT targets). This launcher reuses an EXISTING, byte-identical full-600
target cache (verified bit-identical to a freshly-computed n48 cache: seg equal,
pose max-abs-diff 0.0) via the driver's capped+cached targets path, so the basin
starts training immediately instead of re-paying the precompute. Everything else
delegates to :class:`tac.torch_vehicle.driver.TorchVehicleDriver` (resumable
per-epoch checkpoints, best-by-canonical-score, full per-epoch torch-CPU exact
d_seg/d_pose/rate JSONL telemetry, DONE-marker-on-exit).

Authority: the per-step SegNet gradient runs on MPS (the 90x lever; validated
bit-identical on d_seg by the descent-equivalence gate); the per-step PoseNet
gradient runs on the CPU AUTHORITY (zero pose drift). The EXACT d_seg/d_pose
that pick BEST + seed telemetry ALWAYS run on the CPU authority (``--device``).
``[macOS-CPU advisory]`` NON-PROMOTABLE — a sub-frontier basin GATES, never IS,
a paired contest-CPU+CUDA exact eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="run dir (resumes if a checkpoint is present)")
    p.add_argument("--total-epoch-budget", type=int, default=None,
                   help="proportional epoch budget across the 8 PR95 stages "
                        "(None=full faithful 29,650)")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None,
                   help="override per-stage eval cadence (None=per-stage default)")
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"],
                   help="SegNet-path gradient backend (mps=Apple GPU 90x lever).")
    p.add_argument("--n-pairs", type=int, default=600,
                   help="number of pairs (the basin uses all 600).")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"),
                   help="dir holding gt_targets_n<N>.pt (byte-identical to a fresh "
                        "compute; reused to skip the ~2.5h precompute).")
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split-by-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SPLIT-BY-HEAD gradient backend (SegNet bwd on train_device / "
             "PoseNet bwd on the CPU authority — the 7-11x pose-axis salvage, "
             "DEFAULT for safety). Pass --no-split-by-head for the FULL-MPS "
             "basin (BOTH scorer heads on train_device — the 104x lever, "
             "admissible per the optimizer-chaos verdict "
             "mps_pose_drift_patchable_verdict_20260612.md; still CPU-authority "
             "BEST-tracked so a real late divergence is caught LIVE).",
    )
    p.add_argument("--dashboard", action="store_true",
                   help="print the dashboard for an existing run and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.dashboard:
        from tac.torch_vehicle.telemetry import render_dashboard

        print(render_dashboard(args.out_dir))
        return 0

    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = import_vendored("data").get_default_video_path()

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        out_dir=args.out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        device=args.device,
        train_device=args.train_device,
        split_by_head=args.split_by_head,  # True=pose-axis salvage; False=full-MPS
        seed=args.seed,
    )
    # Reuse the byte-identical full-600 target cache (skips the ~2.5h precompute).
    scorer = RealScorerContext(
        video_path,
        device=args.device,
        train_device=args.train_device,
        split_by_head=args.split_by_head,
        max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    driver = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle())
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
