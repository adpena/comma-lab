#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Lever-2 A/B runner: score-domain argmax-flip seg surrogate vs vendored CE,
forked from the IMMUTABLE base_ch=20 basin snapshot.

THE EXPERIMENT (Layer 2 of the operator's three-layer stack; Lever 2 of
``.omx/research/incurriculum_levers_design_floor_chasing_20260612.md``, ranked
#1): does routing the seg term through the differentiable score-domain
argmax-flip surrogate (``tac.losses.core.segnet_surrogate_per_pixel``, here
``soft_cosine``) produce LOWER d_seg than PR95's vendored cross-entropy, from the
SAME init, at matched epochs?

ARMS (the off-arm is the LIVE BASIN — we run only the on-arm here):
  * ON  (this script): resume from the frozen forkpoint, enable
    ``seg_surrogate=soft_cosine`` on the active stage, train N epochs MPS, eval CPU.
  * OFF (the live control): the daemon in
    ``experiments/results/torch_vehicle_full_mps_basin_bc20_n600`` continuing CE
    from the SAME forkpoint epoch. Diff this arm's trajectory vs the basin's
    recorded ``torch_vehicle_trajectory.jsonl`` over the SAME global-epoch window.

SAFETY (non-negotiable, per the task brief + CLAUDE.md):
  * The fork is COPIED into a NEW out-dir; the live control out-dir is NEVER
    touched. The daemon (pid 33911) is never killed/restarted.
  * Authority is CPU (``--device cpu``); the SegNet gradient runs on MPS
    (``--train-device mps``, the 104x lever). Every number is
    ``[contest-CPU advisory]`` NON-PROMOTABLE — a sub-frontier basin GATES, never
    IS, a paired contest-CPU+CUDA exact eval (CLAUDE.md "MPS auth eval is NOISE" +
    "MPS is a VALID TRAINING-GRADIENT device").
  * The seg-surrogate field is DEFAULT-PRESERVING in the driver; this script
    OPTS IN by mutating the curriculum's active-stage spec only.

Resume is idempotent: re-running continues from this out-dir's own checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

_FORKPOINT = Path("experiments/results/forkpoints/basin_bc20_20260612T121523Z")
_STATE = "torch_vehicle_checkpoint_state.pt"
_MANIFEST = "torch_vehicle_checkpoint_manifest.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forkpoint", type=Path, default=_FORKPOINT,
                   help="immutable snapshot dir to fork from (state .pt + manifest)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="NEW run dir for the lever arm (NEVER the control out-dir)")
    p.add_argument("--seg-surrogate", default="soft_cosine",
                   choices=["soft_cosine", "fisher_rao", "sinkhorn"],
                   help="score-domain d_seg surrogate to enable on the active stage")
    p.add_argument("--seg-temperature", type=float, default=1.0,
                   help="prediction softmax temperature for the surrogate "
                        "(1.0=unit; anneal toward hard argmax with T->small)")
    p.add_argument("--extra-epochs", type=int, default=400,
                   help="how many MORE epochs (beyond the fork epoch) to train")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=25,
                   help="eval cadence (match the basin's stage-1 eval_every=25)")
    p.add_argument("--checkpoint-every-epochs", type=int, default=5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"],
                   help="SegNet-path gradient backend (mps=Apple GPU 104x lever).")
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"))
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="non-blocking CPU authority eval (the basin uses this).")
    return p


def _seed_forkpoint(forkpoint: Path, out_dir: Path) -> dict:
    """Copy the frozen checkpoint into a NEW out-dir (idempotent: only on first
    run — if out_dir already has a checkpoint, resume from ITS own state)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_state = out_dir / _STATE
    dst_manifest = out_dir / _MANIFEST
    if dst_state.exists() and dst_manifest.exists():
        # Resuming from this arm's OWN checkpoint (do not clobber progress).
        return json.loads(dst_manifest.read_text())
    src_state = forkpoint / _STATE
    src_manifest = forkpoint / _MANIFEST
    if not (src_state.exists() and src_manifest.exists()):
        raise FileNotFoundError(
            f"forkpoint {forkpoint} is missing {_STATE} or {_MANIFEST}"
        )
    shutil.copy2(src_state, dst_state)
    shutil.copy2(src_manifest, dst_manifest)
    return json.loads(dst_manifest.read_text())


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    out_dir: Path = args.out_dir
    control = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600").resolve()
    if out_dir.resolve() == control:
        raise SystemExit(
            "REFUSED: --out-dir is the LIVE CONTROL basin. Pick a NEW dir; the "
            "control must never be contaminated."
        )

    man = _seed_forkpoint(args.forkpoint, out_dir)
    fork_stage = int(man["stage_index"])
    fork_epoch_in_stage = int(man["epoch_in_stage"])
    print(
        f"[lever2-ab] forked from stage={fork_stage} epoch_in_stage="
        f"{fork_epoch_in_stage} best_score={man['best_score']:.6f} "
        f"best_ep={man['best_ep']} -> out_dir={out_dir}",
        flush=True,
    )

    from tac.torch_vehicle.curriculum import build_curriculum
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

    # Build the faithful PR95 curriculum, then CAP the active stage so the run
    # ends a bounded ``--extra-epochs`` past the fork (this arm time-shares the
    # GPU with the live basin; keep it bounded). All later stages are dropped
    # (epochs=fork+extra means the active stage ends at the cap, and we truncate
    # the curriculum at the active stage so the run exits cleanly with a DONE).
    curriculum = build_curriculum(
        total_epoch_budget=None,  # full faithful schedule; we cap the active stage
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
    )
    target_epochs = fork_epoch_in_stage + int(args.extra_epochs)
    active = curriculum[fork_stage]
    # OPT IN to the lever on the active stage + cap its epoch count; everything
    # else (seg_loss_fn fallback, all other fields) is the vendored projection.
    active = replace(
        active,
        epochs=target_epochs,
        seg_surrogate=args.seg_surrogate,
        seg_temperature=args.seg_temperature,
    )
    # Truncate the curriculum at the active stage (we A/B within stage 1 only;
    # the run ends with a DONE marker when the active stage's capped epochs finish).
    capped_curriculum = [*curriculum[:fork_stage], active]
    print(
        f"[lever2-ab] active stage='{active.name}' capped to {target_epochs} epochs "
        f"(fork at {fork_epoch_in_stage} + {args.extra_epochs} extra); "
        f"seg_surrogate={args.seg_surrogate} T={args.seg_temperature}",
        flush=True,
    )

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=None,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        device=args.device,
        train_device=args.train_device,
        split_by_head=False,  # match the live basin (--no-split-by-head full-MPS)
        async_eval=args.async_eval,
        seed=args.seed,
    )
    scorer = RealScorerContext(
        video_path,
        device=args.device,
        train_device=args.train_device,
        split_by_head=False,
        max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=capped_curriculum
    )
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
