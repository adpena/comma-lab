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
    p.add_argument(
        "--async-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the CPU AUTHORITY exact eval (byte-close + 600-pair "
             "d_seg/d_pose/rate/score, ~13 min) in a BACKGROUND THREAD off a "
             "point-in-time CPU snapshot of the EMA shadow, so the GPU/MPS training "
             "loop is NOT blocked. The eval is the IDENTICAL eval on the IDENTICAL "
             "snapshot — only non-blocking — so the authority numbers are bit-for-bit "
             "the same as the sync path (the no-regression test proves it). At most "
             "ONE eval is in-flight (over-cadence evals self-throttle: skip + log); "
             "the in-flight thread is JOINED on completion. DEFAULT off (the sync "
             "path is byte-identical unchanged).",
    )
    p.add_argument(
        "--muon-lr-floor-fix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="STAGE-8 apparatus fix (BUG-B): give the stage-8 Muon optimizer its "
             "OWN cosine annealing floor keyed to muon_lr, instead of sharing the "
             "AdamW eta_min (which floors against adamw_lr and prematurely caps the "
             "Muon-finetune d_seg polish). Default OFF preserves byte-identical "
             "behavior for stages 1-7 (AdamW, use_muon=False) and all existing "
             "callers; pass --muon-lr-floor-fix for the decisive full-curriculum run "
             "so the stage-8 d_seg-finishing polish is PR95-faithful. See "
             "apparatus_audit_pr95_breakthrough_blocker_20260619T214001Z.md.",
    )
    p.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="Override the Muon LR on the use_muon stage(s) (stage 8 in the PR95 "
             "curriculum). Default None = vendored curriculum muon_lr (2e-4). The driver "
             "reads spec.muon_lr for BOTH the Muon optimizer LR (driver ~1791) and the "
             "stage-8 cosine floor (driver ~1852: lr_floor_ratio/muon_lr, with "
             "--muon-lr-floor-fix), so this single override keeps both consistent. "
             "Used to disambiguate the 2e-4-flat-on-600-pairs finding (2026-06-23): is "
             "2e-4 too timid on 600 pairs, or was the N=16 probe a memorization artifact? "
             "Non-use_muon stages ignore muon_lr (no-op there).",
    )
    p.add_argument(
        "--defer-batch-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="THROUGHPUT lever (non-split): accumulate the per-batch loss/pose/grad-norm "
             "logging scalars ON-DEVICE and read them with ONE .item() at epoch end, "
             "instead of ~3 device->host syncs per batch (75 batches/epoch -> ~225 "
             "pipeline stalls/epoch; each .item() flushes the MPS command buffer + "
             "blocks). PROVEN bit-identical (gradients/weights/EMA byte-for-byte) by "
             "src/tac/torch_vehicle/tests/test_batch_sync_deferral_bit_identical.py — "
             "the only change is WHEN logging scalars are read. Default OFF (byte-"
             "identical); pass --defer-batch-sync for the faster MPS basin run.",
    )
    p.add_argument(
        "--seg-margin-hinge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace every stage's seg surrogate with the flip-targeting MARGIN-HINGE "
             "(lens A: 0.643x CE residual d_seg; the #1 detector-informed loss lever). "
             "Validated at n600 (d_seg 0.00221 < basin CE 0.00251) while keeping THIS "
             "launcher's full-MPS pose path that actually trains pose (d_pose 0.00034) — "
             "unlike the bind-all split-by-head+equimarginal path which broke pose "
             "(d_pose 0.8 from a 0.00034 KD-warm start). Default OFF = byte-identical "
             "vendored curriculum.",
    )
    p.add_argument(
        "--weight-entropy-penalty-lambda",
        type=float,
        default=0.0,
        help="Ballé end-to-end rate-distortion lever: λ for the learned-prior decoder "
             "WEIGHT-ENTROPY penalty (λ·rate_term added to the loss; pulls the codec "
             "INT8 weight-symbol distribution toward LOW entropy → lower deployed bytes "
             "— the only rate lever left now the post-hoc coder is at the lossless "
             "floor). DEFAULT 0.0 = OFF / byte-identical (the live basin is unaffected). "
             "[contest-CPU advisory] NON-PROMOTABLE; the real bytes come from the "
             "byte-closed codec (λ-on/off A/B at equal d_seg/d_pose).",
    )
    p.add_argument(
        "--weight-entropy-penalty-stage-min",
        type=int,
        default=0,
        help="first curriculum stage index (0-based) the weight-entropy penalty is "
             "active (mirrors C1a's late-stage ramp; default 0 = active from stage 0). "
             "Only consulted when --weight-entropy-penalty-lambda > 0.",
    )
    p.add_argument(
        "--weight-entropy-penalty-waterfill",
        action="store_true",
        help="WATERFILL the weight-entropy penalty across tensors (KKT reverse-water-fill: "
             "byte_share_t/(sensitivity_t+eps), normalized to the same aggregate budget) "
             "instead of a UNIFORM λ. Concentrates rate pressure on big-byte / "
             "low-d_seg-d_pose-sensitivity tensors. Default OFF = uniform = byte-identical "
             "loss term. Only consulted when --weight-entropy-penalty-lambda > 0.",
    )
    p.add_argument(
        "--weight-entropy-penalty-stack-c1a",
        action="store_true",
        help="STACK the penalty ON TOP of C1a (cat_entropy_v2) instead of SUPERSEDING it. "
             "NOT recommended: the two penalize the SAME codec-grid symbol entropy and a "
             "$0 probe shows stacking reaches WORSE entropy than either alone. Default "
             "(absent) = the penalty SUPERSEDES C1a when active (the empirically-correct "
             "behavior). Only consulted when --weight-entropy-penalty-lambda > 0.",
    )
    p.add_argument(
        "--stage-lr-warmup-frac",
        type=float,
        default=0.0,
        help="E#5 STAGE-TRANSITION POSE-KICK FIX: per-stage LR WARMUP fraction. DEFAULT "
             "0.0 = OFF / byte-identical (every stage starts at the cosine peak, the "
             "legacy faithful path). When > 0, the first frac·stage_epochs of EACH stage "
             "linearly ramp LR floor→cosine (warmup-after-restart) so the shared trunk is "
             "eased in at the stage boundary instead of slammed to peak (MEASURED d_pose "
             "spike 0.00021→0.00142, 6.8×, at the stage-1→2 boundary; 6 more transitions "
             "remain). Applies to BOTH AdamW + Muon groups. Must be in [0.0, 0.5]. "
             "[contest-CPU advisory] NON-PROMOTABLE; the win is the boundary-kick reduction.",
    )
    p.add_argument(
        "--stage-lr-warmup-start-ratio",
        type=float,
        default=0.1,
        help="the LR floor (fraction of stage peak) the E#5 warmup ramp starts from at "
             "each stage's first epoch (default 0.1 = 10% of peak). Only consulted when "
             "--stage-lr-warmup-frac > 0. Must be in (0.0, 1.0].",
    )
    p.add_argument(
        "--taper-channels",
        type=str,
        default=None,
        help="Yousfi STEP-2 CAPACITY+RATE lever: comma-separated per-block channel taper "
             "(e.g. '16,16,17,19,19,14,10' = lens-C dseg_aware_taper(20), reallocating the "
             "~69%% of capacity wasted in SegNet-invisible low-res stages INTO the high-res "
             "boundary band where the d_seg flips live, at byte-neutral cost). DEFAULT None = "
             "the vendored taper [20,20,20,15,11,10,10] (byte-identical). This is the d_seg-"
             "NEUTRAL rate lever (the entropy-penalty rate lever was net-negative: weight "
             "distortion costs ~374x more on d_seg than it saves on rate).",
    )
    p.add_argument(
        "--kd-warm-start-dir",
        type=Path,
        default=None,
        help="KD-warm-start dir (a teacher run's out-dir, e.g. the margin_hinge basin). The "
             "driver builds a FROZEN teacher decoder + distills its rendered pairs into the "
             "re-tapered student over the first --kd-warm-epochs of stage 0 (so a taper "
             "change inherits the teacher's progress instead of a weeks-from-scratch run). "
             "DEFAULT None = no KD (fresh/seed init). Teacher must match base_channels/"
             "latent_dim (strict); the taper MAY differ (that's the point).",
    )
    p.add_argument("--kd-warm-epochs", type=int, default=300,
                   help="KD warm-up epochs (only consulted when --kd-warm-start-dir is set).")
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
        async_eval=args.async_eval,  # True=non-blocking background CPU authority eval
        muon_lr_floor_fix=args.muon_lr_floor_fix,  # stage-8 Muon own-floor (BUG-B fix)
        defer_batch_sync=args.defer_batch_sync,  # throughput: 1 device→host sync/epoch
        #   (non-split), proven BIT-IDENTICAL by test_batch_sync_deferral_bit_identical
        # Ballé rate lever (default 0.0 = OFF / byte-identical; the live basin uses 0.0).
        weight_entropy_penalty_lambda=args.weight_entropy_penalty_lambda,
        weight_entropy_penalty_stage_min=args.weight_entropy_penalty_stage_min,
        weight_entropy_penalty_waterfill=args.weight_entropy_penalty_waterfill,
        # supersede C1a when the penalty is active (default True; --stack-c1a flips it off).
        weight_entropy_penalty_supersedes_c1a=not args.weight_entropy_penalty_stack_c1a,
        # E#5 per-stage LR warmup (default 0.0 = OFF / byte-identical).
        stage_lr_warmup_frac=args.stage_lr_warmup_frac,
        stage_lr_warmup_start_ratio=args.stage_lr_warmup_start_ratio,
        # Yousfi step-2: d_seg-aware taper (None = vendored, byte-identical) + KD-warm.
        taper_channels=(
            [int(c) for c in args.taper_channels.split(",")]
            if args.taper_channels
            else None
        ),
        kd_warm_start_dir=args.kd_warm_start_dir,
        kd_warm_epochs=args.kd_warm_epochs,
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
    # Optional seg-surrogate lever: build the curriculum + replace every stage's seg
    # surrogate with margin_hinge (the validated detector-informed d_seg lever), passed
    # to the driver via its ``curriculum=`` hook. None → driver builds the vendored
    # curriculum internally (byte-identical default). This rides on THIS launcher's
    # full-MPS pose path (the one that trains pose), NOT the bind-all split-by-head path.
    curriculum = None
    if args.seg_margin_hinge or args.muon_lr is not None:
        import dataclasses

        from tac.torch_vehicle.curriculum import build_curriculum

        specs = build_curriculum(
            total_epoch_budget=args.total_epoch_budget,
            ema_decay=args.ema_decay,
            eval_every=args.eval_every,
        )
        if args.seg_margin_hinge:
            specs = [dataclasses.replace(s, seg_surrogate="margin_hinge") for s in specs]
        if args.muon_lr is not None:
            # Override the Muon LR on the use_muon stage(s) only (stage 8). The driver reads
            # spec.muon_lr for BOTH the optimizer LR and the cosine floor, so this single
            # replace keeps them consistent; non-Muon stages ignore muon_lr (no-op).
            specs = [
                dataclasses.replace(s, muon_lr=args.muon_lr) if s.use_muon else s
                for s in specs
            ]
        curriculum = specs

    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
