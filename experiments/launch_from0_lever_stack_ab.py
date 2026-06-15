#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""From-EPOCH-0 decisive A/B(/C) for the Track-A distortion lever stack (#116).

Three arms from a SHARED bit-identical epoch-0 init (same ``--seed``; the driver's
RNG-neutral FiLM build keeps the latent/decoder draw bit-shared across arms):

  * ``control``        — vendored PR95 curriculum, levers OFF (CE seg, no FiLM, no
                         margin-weight). The baseline the lever stack must beat.
  * ``stack``          — the FULL distortion lever stack across ALL 8 stages:
                         Lever-2 soft-cosine @ fast-cool (T 1.0→0.3 hold_frac=0.3, the
                         #119-confirmed schedule) + Lever-3 v2 residual pose-FiLM
                         (``pose_film_version=2``, rgb_0-only, d_seg/d_pose decoupled) +
                         Lever-5 margin-weight (τ=0.3).
  * ``stack_ce_early`` — the CE-early variant: stages 0-3 (CE/tau/smooth/QAT, the
                         COARSE-structure phase) stay vendored CE (no seg lever, no
                         margin); stages 4-7 (C1a/λ/σ/muon, the REFINEMENT phase) get the
                         full seg lever stack. Lever-3 v2 FiLM is ON throughout (identity-
                         init → harmless early). Tests the soft_cosine-from-epoch-0
                         hypothesis (the gate showed soft_cosine@T=1.0 is untargeted vs CE;
                         early broad margins may prefer CE) per "multiple contenders → run
                         both".

WHY a from-0 A/B (not forkpoint-resume): #116 is the decisive "does the lever stack,
trained from scratch, beat the vendored baseline on the contest SCORE?" test. The anneal
schedule was calibrated on a CONVERGED basin; this run measures the stack across the whole
trajectory, so the early-soft_cosine question is part of what it answers.

AUTHORITY (CLAUDE.md): exact d_seg/d_pose that pick BEST + seed telemetry ALWAYS run on
the CPU authority (``--device cpu``; NO MPS). ``--train-device`` defaults to CPU (fully
trusted, zero device-placement risk). Split-by-head (MPS SegNet grad + CPU PoseNet grad,
the ~90x lever) is OPT-IN and gated on the v2-FiLM split-parity check
(``test_pose_film_v2_wire_in`` / a dedicated parity smoke) — the v2 FiLM is on the pose
path, so its gradient under the MPS/CPU split must be verified descent-equivalent FIRST.
Every in-loop number is ``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed
archive runs through ``upstream/evaluate.py``.

Run one arm per invocation (each is its OWN out-dir; resumable per-epoch; DONE-marker on
exit), launched as a DETACHED nohup daemon (NOT tool-background — the harness SIGURG-kills
bg bash at ~3 min). Example:
    nohup bash -c '.venv/bin/python experiments/launch_from0_lever_stack_ab.py \
        --arm stack --out-dir experiments/results/from0_ab_<stamp>/stack \
        --total-epoch-budget 2400 --go' </dev/null >.../stack.outer.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

# Stage index split: PR95 8-stage curriculum (0-indexed): 0=CE 1=tau_softplus 2=smooth
# 3=QAT (the COARSE-structure phase) | 4=C1a-L7 5=lambda_sweep 6=sigma_sweep 7=muon
# (the REFINEMENT phase). The CE-early arm applies the seg lever stack only on >= this.
_REFINEMENT_FIRST_STAGE = 4

# The #119-confirmed seg lever config (fast-cool → 0.3 hold + Lever-5 τ).
_SEG_SURROGATE = "soft_cosine"
_SEG_T_START = 1.0
_SEG_T_END = 0.3
_SEG_T_HOLD_FRAC = 0.3
# Lever-5 margin-weight bandwidth (coherence fix 2026-06-15): τ=1.0, NOT 0.3. The
# soft_cosine gradient ALREADY self-concentrates on the boundary via its q_g(1-q_g)/T
# factor; an aggressive explicit τ=0.3 double-concentrates AND mis-scales (SegNet boundary
# margins are ~0.5-2 logits, but exp(-m/0.3)≈0.04 already at m=1 → weights only the
# un-fixable m<0.3 noise). τ≈1.0 matches the flip-prone margin band and lets the low-T
# self-concentration lead with a gentle explicit nudge (the co-tuned, coherent setting).
_MARGIN_WEIGHT_TAU = 1.0

# The 5 arms (shared epoch-0 init via the same --seed). Two orthogonal lever axes:
#   SEG axis (Lever-2 soft-cosine fast-cool + Lever-5 margin) and FiLM axis (Lever-3 v2):
#     control        : SEG off,  FiLM off   (vendored baseline)
#     stack          : SEG all,  FiLM on    (full stack)
#     stack_ce_early : SEG late, FiLM on    (CE coarse, lever-stack refinement)
#     seg_only       : SEG all,  FiLM off   (stack − FiLM  ⇒ isolates Lever-3's effect)
#     film_only      : SEG off,  FiLM on    (control + FiLM ⇒ isolates Lever-3 alone)
# Lever-3's contribution is thus measured TWO ways (stack−seg_only and film_only−control);
# the seg-side is stack−film? no: seg-side = seg_only−control. Full attribution, one launch.
_ARMS = ("control", "stack", "stack_lovasz", "stack_ce_early", "seg_only", "film_only")
_SEG_OVERLAY_ALL = {"stack", "seg_only"}  # soft_cosine fast-cool seg lever stack on EVERY stage
_SEG_OVERLAY_LATE = {"stack_ce_early"}  # seg lever stack on the refinement stages only
# lovász-softmax (Berman 2018) is the convex envelope of the Jaccard/IoU loss — the
# SUBMODULAR/overlap complement to the MODULAR soft_cosine (whose Lovász degenerates to a
# per-pixel mean). It attacks d_seg through a different gradient geometry, so it is the
# highest-value NEW arm against the soft_cosine plateau. T=1 fixed (the envelope lives on
# the plain simplex → no anneal) and NO Lever-5 margin (lovász is a scalar loss → the
# per-pixel margin weight is a no-op for it; see test_lovasz_ignores_per_pixel_margin_weight).
_LOVASZ_ARMS = {"stack_lovasz"}  # lovász on EVERY stage; FiLM on; no anneal, no margin
_FILM_ON_ARMS = {"stack", "stack_lovasz", "stack_ce_early", "film_only"}  # Lever-3 v2 pose-FiLM
_LIVE_BASIN_DIRS = {
    Path("experiments/results/forkpoints").resolve(),
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=_ARMS,
                   help="which arm to run (one per invocation; same --seed across arms)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="arm run dir (resumes if a checkpoint is present). MUST be a NEW "
                        "dir, NOT the live basin/forkpoint tree.")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--total-epoch-budget", type=int, default=2400,
                   help="BOUNDED proportional budget across the 8 PR95 stages for the "
                        "first decisive read (None=full faithful 29,650). Resumable to "
                        "extend.")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="gradient backend. DEFAULT cpu (trusted). mps (split-by-head) is "
                        "OPT-IN + requires the v2-FiLM split-parity check to pass first.")
    p.add_argument("--split-by-head", action="store_true",
                   help="MPS SegNet grad + CPU PoseNet grad. Requires --train-device mps "
                        "AND a passing v2-FiLM split-parity check (the v2 FiLM is on the "
                        "pose path).")
    p.add_argument("--async-eval", action="store_true")
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--pose-film-hidden", type=int, default=8)
    p.add_argument("--pose-grad-every-k", type=int, default=1,
                   help="POSE-THROTTLE speed lever: compute the split-by-head PoseNet "
                        "gradient only every k-th epoch (1=every epoch=byte-identical). "
                        "Pose costs ~51%% of the epoch but is solved early — k=4 reclaims "
                        "~40%% on the solved tail. d_seg (the A/B verdict) is unaffected.")
    p.add_argument("--pose-grad-resume-threshold", type=float, default=0.0,
                   help="Warm/self-protect guard: force pose every epoch while the running "
                        "pose_mse exceeds this (still converging / drifted); else cadence. "
                        "0.0=disabled (pure cadence). Recommended 0.005 with k>1.")
    p.add_argument("--seed", type=int, default=0,
                   help="SHARED across arms → bit-identical epoch-0 init (apples-to-apples).")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"))
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the real-scorer arm (loads the heavy scorer).")
    p.add_argument("--print-plan", action="store_true",
                   help="print the per-stage lever overlay for this arm and exit ($0).")
    return p


def _overlay_levers(specs: list, arm: str) -> list:
    """Return a NEW spec list with the distortion lever stack overlaid per ``arm``.

    ``control`` / ``film_only`` → specs UNCHANGED (vendored CE; FiLM, if any, is set at
       the cfg level, NOT the seg overlay).
    ``stack`` / ``seg_only`` → EVERY stage gets soft-cosine + fast-cool anneal + Lever-5 τ.
    ``stack_ce_early`` → only stages >= _REFINEMENT_FIRST_STAGE get the seg lever stack;
       the coarse stages stay vendored CE (seg_surrogate=None, no margin-weight).
    """
    if arm in _LOVASZ_ARMS:
        # lovász-softmax overlay: the submodular/IoU convex envelope on EVERY stage. T=1
        # fixed (no fast-cool — the envelope is on the plain simplex), no margin-weight
        # (scalar loss). FiLM is set at the cfg level (this arm is in _FILM_ON_ARMS).
        return [
            dataclasses.replace(s, seg_surrogate="lovasz", seg_temperature=1.0)
            for s in specs
        ]
    if arm not in _SEG_OVERLAY_ALL and arm not in _SEG_OVERLAY_LATE:
        return list(specs)  # control / film_only: no seg overlay
    out = []
    for i, s in enumerate(specs):
        seg_levers_on = arm in _SEG_OVERLAY_ALL or i >= _REFINEMENT_FIRST_STAGE
        if seg_levers_on:
            out.append(
                dataclasses.replace(
                    s,
                    seg_surrogate=_SEG_SURROGATE,
                    seg_temperature=_SEG_T_START,
                    seg_temperature_end=_SEG_T_END,
                    seg_temperature_hold_frac=_SEG_T_HOLD_FRAC,
                    margin_weight_tau=_MARGIN_WEIGHT_TAU,
                )
            )
        else:
            out.append(s)  # coarse stage: vendored CE, no seg lever, no margin-weight
    return out


def _plan(specs: list, arm: str) -> list[dict]:
    over = _overlay_levers(specs, arm)
    return [
        {
            "stage": i,
            "name": s.name,
            "epochs": s.epochs,
            "seg_surrogate": s.seg_surrogate,
            "seg_temperature": s.seg_temperature,
            "seg_temperature_end": s.seg_temperature_end,
            "seg_temperature_hold_frac": s.seg_temperature_hold_frac,
            "margin_weight_tau": s.margin_weight_tau,
        }
        for i, s in enumerate(over)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
    )
    overlaid = _overlay_levers(specs, args.arm)

    if args.print_plan:
        print(json.dumps({"arm": args.arm, "plan": _plan(specs, args.arm)}, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    for live in _LIVE_BASIN_DIRS:
        try:
            if live in out_dir.resolve().parents or out_dir.resolve() == live:
                raise SystemExit(
                    f"REFUSED: --out-dir {out_dir} is inside the live basin tree {live}. "
                    "Use a NEW results dir for the A/B."
                )
        except FileNotFoundError:
            pass
    if args.split_by_head and args.train_device != "mps":
        raise SystemExit("--split-by-head requires --train-device mps")
    if args.train_device == "mps" and not args.split_by_head:
        raise SystemExit(
            "--train-device mps must be used WITH --split-by-head (the PoseNet path stays "
            "on the CPU authority; MPS corrupts PoseNet 23x). And the v2-FiLM split-parity "
            "check must pass first."
        )
    if not args.go:
        raise SystemExit(
            "REFUSED without --go (the real-scorer arm loads the heavy SegNet/PoseNet). "
            "Use --print-plan for the $0 lever-overlay preview."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
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

    scorer = RealScorerContext(
        video_path, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    film_on = args.arm in _FILM_ON_ARMS
    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        pose_film_enabled=film_on, pose_film_hidden=args.pose_film_hidden,
        pose_film_version=2 if film_on else 1, seed=args.seed,
        pose_grad_every_k=args.pose_grad_every_k,
        pose_grad_resume_threshold=args.pose_grad_resume_threshold,
    )
    print(json.dumps({
        "arm": args.arm, "film_on": film_on, "pose_film_version": cfg.pose_film_version,
        "device": args.device, "train_device": args.train_device,
        "split_by_head": args.split_by_head, "n_pairs": args.n_pairs,
        "total_epoch_budget": args.total_epoch_budget, "seed": args.seed,
        "out_dir": str(out_dir), "plan": _plan(specs, args.arm),
    }, indent=2), flush=True)

    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=overlaid
    )
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
