# SPDX-License-Identifier: MIT
"""Production CLI for the resumable P2 torch-vehicle curriculum run.

Thin entry point (delegates to :class:`tac.torch_vehicle.driver.TorchVehicleDriver`
per the "tac stays clean; thin CLIs delegate" rule). Resumes automatically from
``--out-dir`` if a checkpoint is present; writes a DONE marker on completion.

Example (base_ch=20 rate-win config, faithful PR95 curriculum, $0 local smoke)::

    python -m tac.torch_vehicle.run \\
        --base-channels 20 --ema-decay 0.999 \\
        --total-epoch-budget 200 --eval-every 25 \\
        --out-dir experiments/results/torch_vehicle_n600_bc20 \\
        --device cpu

For the $100 Modal run (post-symposium) the same command runs on CUDA with the
full or budget-compressed curriculum; a SIGKILL/OOM/preempt loses <= one
checkpoint interval and a re-launch resumes the EXACT trajectory.

Authority: in-loop d_seg/d_pose are ``[contest-CPU advisory]`` NON-PROMOTABLE.
The leaderboard score is authoritative ONLY after ``upstream/evaluate.py`` on the
byte-closed ``best/best_archive.bin``. NO MPS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable P2 torch-vehicle (vendored PR95) curriculum run.")
    p.add_argument("--base-channels", type=int, default=20, help="decoder base channels (20=rate-win, 36=PR95-proven)")
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--out-dir", type=Path, required=True, help="run dir (resumes if a checkpoint is present)")
    p.add_argument("--total-epoch-budget", type=int, default=None,
                   help="proportional epoch budget across the 8 stages (None=full 29,650)")
    p.add_argument("--ema-decay", type=float, default=0.999, help="PR95-faithful EMA decay (0.999)")
    p.add_argument("--eval-every", type=int, default=None, help="override per-stage eval cadence")
    p.add_argument("--checkpoint-every-epochs", type=int, default=1, help="a death costs <= this many epochs")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu TRUSTED; cuda for Modal. NO mps.")
    p.add_argument("--video-path", type=Path, default=None,
                   help="contest video (default: vendored data.get_default_video_path())")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dashboard", action="store_true", help="print the dashboard for an existing run and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.dashboard:
        from tac.torch_vehicle.telemetry import render_dashboard

        print(render_dashboard(args.out_dir))
        return 0

    # Import the driver + the REAL scorer context lazily (the scorer pulls in the
    # frozen SegNet/PoseNet weights — only needed for an actual run).
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        # Use the vendored default-video resolver (single-video memorization).
        from tac.torch_vehicle.vendored_imports import import_vendored

        data = import_vendored("data")
        video_path = data.get_default_video_path()

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        out_dir=args.out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        device=args.device,
        seed=args.seed,
    )
    scorer = RealScorerContext(video_path, device=args.device)
    driver = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle())
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
