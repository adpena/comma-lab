#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""MEASURE the real per-epoch training wall-clock of the torch_vehicle split-by-head
path (the small-basis 600-pair config), so arm_b's budget can be planned in MEASURED
seconds/epoch instead of estimated minutes (CLAUDE.md "Long-burn ... timing-smoke
command that measures seconds/epoch").

This is a TIMING tool ONLY. It emits ZERO score claims. The d_seg/d_pose numbers it
would produce are irrelevant; it drives the EXACT production ``_train_one_epoch`` code
path (real decoder, real frozen SegNet on the train device, real CPU-authority PoseNet)
and times each epoch with ``time.perf_counter``. NO eval runs inside the timed loop —
the eval cost is measured SEPARATELY (``--time-one-eval``) so the train-epoch and eval
costs are never confounded.

The dominant per-epoch cost on the split-by-head path is the CPU-authority PoseNet
(FastViT) forward+backward (the ~51%% lever). This tool isolates it by timing three
pose-cadence configs against the IDENTICAL decoder/scorer/basis:

  * ``k1``       — pose every epoch (the naive ``--pose-grad-every-k 1`` baseline).
  * ``apgc``     — the adaptive pose-grad controller at the recommended production band
                   (``--pose-grad-adaptive --pose-grad-floor-tol 0.08 --pose-grad-k-max 8``);
                   measured at the FLOOR steady state (pose throttled to the sparsest
                   cadence) AND with pose forced (the drift-arrest worst case), so the
                   APGC time band is bounded on both ends.
  * ``poseoff``  — pose backward never computed (``compute_pose=False`` every epoch); the
                   pure SegNet-only-on-MPS lower bound. Isolates the pose-backward fraction.

WHY drive ``_train_one_epoch`` directly (not the full launcher): the launcher pays the
warm-baseline eval + curriculum + checkpoint I/O once, none of which is per-epoch train
cost. Timing the inner epoch fn is the cleanest apples-to-apples per-epoch number and
needs no edits to driver.py / the launcher (which a sister subagent is editing).

AUTHORITY: none. This emits wall-clock only. ``--train-device mps`` is the gradient-only
speed path (CLAUDE.md: MPS is a valid TRAINING-GRADIENT device, never a score authority).

Run from a STABLE checkout (do not import the half-edited launcher). Example:
    .venv/bin/python tools/measure_training_throughput.py \
        --warm-start-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
        --n-pairs 600 --train-device mps --split-by-head \
        --warmup-epochs 1 --timed-epochs 4 \
        --configs k1,apgc_floor,apgc_forced,poseoff \
        --out-json experiments/results/timing_smoke/throughput.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

_DEFAULT_CONFIGS = ("k1", "apgc_floor", "apgc_forced", "poseoff")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--warm-start-dir",
        type=Path,
        required=True,
        help="converged best/ dir (best_ema_decoder.pt + best_ema_latents.pt) — the "
        "decoder/latents are loaded so the timed epochs run on a realistic basin.",
    )
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument(
        "--n-pairs",
        type=int,
        default=600,
        help="basis size (MUST match the warm-start checkpoint's basis).",
    )
    p.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--split-by-head", action="store_true", default=True)
    p.add_argument("--no-split-by-head", dest="split_by_head", action="store_false")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="untimed epochs first (Metal kernel compile + cache warm — the FIRST MPS "
        "epoch pays one-time compilation, so it is excluded from the measured median).",
    )
    p.add_argument(
        "--timed-epochs",
        type=int,
        default=4,
        help="epochs to time per config (the median is the reported sec/epoch).",
    )
    p.add_argument(
        "--configs",
        default=",".join(_DEFAULT_CONFIGS),
        help="comma list from {k1, apgc_floor, apgc_forced, poseoff}.",
    )
    p.add_argument(
        "--pose-grad-floor-tol",
        type=float,
        default=0.08,
        help="APGC deadband (production recommended 0.08).",
    )
    p.add_argument(
        "--pose-grad-k-max",
        type=int,
        default=8,
        help="APGC sparsest cadence / measurement-floor (production recommended 8).",
    )
    p.add_argument(
        "--time-one-eval",
        action="store_true",
        help="ALSO time ONE exact CPU-authority eval (the --eval-every cost) so the "
        "eval overhead is quantified separately from the train epochs.",
    )
    p.add_argument(
        "--seg-weight-base",
        type=float,
        default=100.0,
        help="stage seg_weight (matches the vendored muon-finetune default).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--targets-cache",
        type=Path,
        default=Path("experiments/results/capstone_gt_targets_cache"),
    )
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--go",
        action="store_true",
        help="REQUIRED to launch (loads the heavy SegNet/PoseNet). Without it, prints "
        "the measurement plan and exits ($0).",
    )
    return p


def _make_driver(args, *, pose_adaptive: bool, pose_every_k: int):
    """Build a real TorchVehicleDriver + RealScorerContext + a single refinement stage,
    warm-started from the converged checkpoint. Returns (driver, rt, spec, scorer)."""
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

    scorer = RealScorerContext(
        video_path,
        device=args.device,
        train_device=args.train_device,
        split_by_head=args.split_by_head,
        max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )

    # The faithful refinement stage (Muon + QAT + C1a + sigma) — the last PR95 stage.
    base_refine = build_curriculum(
        total_epoch_budget=800, ema_decay=0.999, eval_every=10
    )[-1]
    import dataclasses

    # init_latents_random so the warm-start branch fires; small epoch count (we drive
    # epochs manually anyway).
    spec = dataclasses.replace(
        base_refine,
        epochs=args.timed_epochs + args.warmup_epochs + 1,
        init_latents_random=True,
        seg_weight=float(args.seg_weight_base),
    )

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        out_dir=Path("experiments/results/timing_smoke/_scratch_driver"),
        total_epoch_budget=spec.epochs,
        ema_decay=0.999,
        eval_every=spec.epochs + 1,  # never eval inside this driver
        device=args.device,
        train_device=args.train_device,
        split_by_head=args.split_by_head,
        seed=args.seed,
        pose_grad_every_k=max(1, pose_every_k),
        pose_grad_adaptive=pose_adaptive,
        pose_grad_floor_tol=args.pose_grad_floor_tol,
        pose_grad_k_max=args.pose_grad_k_max,
        warm_start_dir=Path(args.warm_start_dir),
    )
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[spec]
    )

    # Build the stage runtime exactly as run() does for stage 0 (warm-started decoder).
    decoder = driver._new_decoder()
    latents = driver._load_warm_start_into(decoder)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    return driver, rt, spec, scorer


def _time_config(args, name: str) -> dict:
    """Time ``warmup_epochs`` (untimed) + ``timed_epochs`` (timed) of the real
    ``_train_one_epoch`` for one pose-cadence config. Returns the per-epoch stats."""
    if name == "k1":
        pose_adaptive, pose_every_k, force_pose_off = False, 1, False
    elif name == "apgc_floor":
        # APGC at steady-state floor: simulate "pose at floor" so the controller picks
        # the sparsest cadence (the throttle is ACTIVE → most epochs skip the pose bwd).
        pose_adaptive, pose_every_k, force_pose_off = True, 1, "apgc_floor"
    elif name == "apgc_forced":
        # APGC drift-arrest worst case: pose forced every epoch (== k1 cost) — bounds the
        # APGC band's upper end (when pose is drifting the controller pays full freight).
        pose_adaptive, pose_every_k, force_pose_off = True, 1, "apgc_forced"
    elif name == "poseoff":
        pose_adaptive, pose_every_k, force_pose_off = False, 10**9, "poseoff"
    else:
        raise SystemExit(f"unknown config {name!r}")

    driver, rt, spec, _scorer = _make_driver(
        args, pose_adaptive=pose_adaptive, pose_every_k=pose_every_k
    )

    # apgc_floor — simulate the STEADY-STATE "pose solidly at floor" condition the
    # throttle is designed for. The honest way to hold that condition: before EVERY epoch
    # RE-SEED the full controller state to the at-floor belief (floor==last_mse so dev==1
    # → slack==1 → k==k_max), set ``_global_epoch`` to a NON-multiple of k_max so the
    # ``epoch % k`` cadence SKIPS, and stamp ``_last_pose_epoch`` so the measurement-floor
    # (``epoch - last_pose_epoch >= k_max``) does NOT force a compute. Without the per-epoch
    # re-seed, the first epoch's real (large) basin pose_mse overwrites ``_last_pose_mse``
    # and the controller falls into permanent drift-arrest (dev huge), masking the throttle.
    # This bounds the LOW end of the APGC time band (most epochs skip pose). apgc_forced
    # (unseeded) bounds the HIGH end (controller pays full freight every epoch).
    apgc_floor = force_pose_off == "apgc_floor"

    def _seed_at_floor() -> None:
        driver._pose_floor = 1.0e-6
        driver._last_pose_mse = 1.0e-6
        driver._pose_mse_hist = [1.0e-6, 1.0e-6, 1.0e-6]
        # global_epoch ≡ 1 (mod k_max) → (epoch % k_max) != 0 → cadence SKIP; and
        # (epoch - last_pose_epoch) = 1 < k_max → measurement-floor does NOT force.
        driver._global_epoch = 1
        driver._last_pose_epoch = 0

    per_epoch_secs: list[float] = []
    pose_computed_flags: list[bool] = []
    total_epochs = args.warmup_epochs + args.timed_epochs
    for e in range(total_epochs):
        if apgc_floor:
            _seed_at_floor()
        # Detect whether the (expensive) CPU-authority pose backward ran THIS epoch via
        # the change in ``_last_pose_mse`` — it is overwritten EVERY time the pose path
        # computes (both the static k<=1 path and the APGC path), whereas
        # ``_last_pose_epoch`` is APGC-only. Use a sentinel object id snapshot so a
        # repeated identical pose_mse value (unlikely but possible) is still counted via
        # the always-fresh float identity from ``.item()``.
        pose_mse_before = driver._last_pose_mse
        pose_id_before = id(driver._last_pose_mse)
        t0 = time.perf_counter()
        driver._train_one_epoch(rt, spec, epoch_in_stage=e)
        # Force any queued MPS work to finish before stopping the clock.
        _mps_sync(args.train_device)
        dt = time.perf_counter() - t0
        driver._global_epoch += 1
        # pose ran iff _last_pose_mse changed (new float object) or went from None->value.
        pose_recomputed = (
            driver._last_pose_mse is not None
            and (
                pose_mse_before is None
                or id(driver._last_pose_mse) != pose_id_before
            )
        )
        if e >= args.warmup_epochs:
            per_epoch_secs.append(dt)
            pose_computed_flags.append(bool(pose_recomputed))
        print(
            f"  [{name}] epoch {e} ({'WARMUP' if e < args.warmup_epochs else 'TIMED'}): "
            f"{dt:.2f}s  pose_recomputed={pose_recomputed}",
            flush=True,
        )

    return {
        "config": name,
        "timed_epochs": len(per_epoch_secs),
        "per_epoch_secs": [round(s, 3) for s in per_epoch_secs],
        "median_sec_per_epoch": round(median(per_epoch_secs), 3)
        if per_epoch_secs
        else None,
        "min_sec_per_epoch": round(min(per_epoch_secs), 3) if per_epoch_secs else None,
        "max_sec_per_epoch": round(max(per_epoch_secs), 3) if per_epoch_secs else None,
        "pose_computed_per_timed_epoch": pose_computed_flags,
        "pose_computed_frac": round(
            sum(pose_computed_flags) / max(1, len(pose_computed_flags)), 3
        ),
    }


def _mps_sync(train_device: str) -> None:
    import torch

    if str(train_device).startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif str(train_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_one_eval(args) -> dict:
    """Time ONE exact CPU-authority eval (the --eval-every cost) off the warm checkpoint."""
    import torch

    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = import_vendored("data").get_default_video_path()
    scorer = RealScorerContext(
        video_path,
        device=args.device,
        train_device=args.train_device,
        split_by_head=args.split_by_head,
        max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    vbundle = import_vendored_bundle()
    dec = (
        vbundle.HNeRVDecoder(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            eval_size=(384, 512),
        )
        .to(args.device)
        .eval()
    )
    sd = torch.load(
        Path(args.warm_start_dir) / "best_ema_decoder.pt",
        map_location=args.device,
        weights_only=False,
    )
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    dec.load_state_dict({k: v.to(args.device) for k, v in sd.items()})
    lat = torch.load(
        Path(args.warm_start_dir) / "best_ema_latents.pt",
        map_location=args.device,
        weights_only=False,
    )
    if isinstance(lat, dict):
        lat = lat.get("latents", next(iter(lat.values())))
    archive_bytes = 0
    wab = Path(args.warm_start_dir) / "best_archive.bin"
    if wab.exists():
        archive_bytes = wab.stat().st_size
    t0 = time.perf_counter()
    scorer.exact_eval(dec, lat.to(args.device), archive_bytes)
    dt = time.perf_counter() - t0
    print(f"  [eval] one exact CPU-authority eval: {dt:.1f}s", flush=True)
    return {"one_eval_secs": round(dt, 2), "n_pairs": args.n_pairs}


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    plan = {
        "warm_start_dir": str(args.warm_start_dir),
        "n_pairs": args.n_pairs,
        "train_device": args.train_device,
        "split_by_head": args.split_by_head,
        "warmup_epochs": args.warmup_epochs,
        "timed_epochs": args.timed_epochs,
        "configs": configs,
        "apgc_floor_tol": args.pose_grad_floor_tol,
        "apgc_k_max": args.pose_grad_k_max,
        "authority": "NONE — wall-clock timing only, ZERO score claims",
    }
    if not args.go:
        print(json.dumps({"plan": plan, "note": "pass --go to run"}, indent=2))
        return 0

    print(json.dumps({"plan": plan}, indent=2), flush=True)
    results: list[dict] = []
    for name in configs:
        print(f"=== timing config: {name} ===", flush=True)
        results.append(_time_config(args, name))

    eval_row = None
    if args.time_one_eval:
        print("=== timing one exact eval ===", flush=True)
        eval_row = _time_one_eval(args)

    out = {"plan": plan, "results": results, "eval": eval_row}
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
