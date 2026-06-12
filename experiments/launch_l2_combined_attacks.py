#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Launch the COMBINED Layer-2-attacks A/B arm for the base_ch=20 HNeRV basin —
resume from the IMMUTABLE basin forkpoint into a NEW out-dir with BOTH wired-in
score-aware attacks ENABLED:

  * Lever 2 — the score-domain ARGMAX-FLIP seg surrogate
    (``StageSpec.seg_surrogate`` routed through
    ``tac.losses.core.segnet_surrogate_per_pixel``; the contest d_seg is the
    per-pixel SegNet argmax-disagreement rate, so the surrogate concentrates the
    gradient where the argmax actually flips, unlike PR95's vendored CE which
    spends capacity on confident-interior pixels — HNeRV parity L6).
  * Lever 3 — the pose-FiLM STORE (Quantizr STORE-pose lesson;
    ``cfg.pose_film_enabled``): store the 6 GT pose scalars per pair as side
    information (Wyner-Ziv) and FiLM-condition the decoder on them, so d_pose
    collapses toward the stored-pose quant floor and the decoder capacity is
    freed for d_seg + rate. The stored pose is range-coded into an ADDITIVE
    archive section (~1 KB charged); the vendored codec stays pristine.

ALL FIVE Layer-2 levers are now WIRED + default-OFF + composable. Pass
``--levers all`` to enable EVERY lever (seg surrogate + the per-epoch T-anneal +
pose-FiLM + rate surrogate + score-aware QAT + margin-weight) in one arm; the
default (no ``--levers all``) keeps the original seg+pose-FiLM pair so existing
invocations are unchanged. The five:

  * **Lever 1** — differentiable brotli-rate surrogate
    (``StageSpec.rate_lambda_w`` / ``rate_lambda_lat`` →
    ``tac.losses.rate_surrogate``: order-1 conditional weight entropy + latent
    temporal-delta entropy; the dominant rate-term attack).
  * **Lever 2** — score-domain argmax-flip seg surrogate + the per-epoch
    TEMPERATURE-ANNEAL (``seg_surrogate`` / ``seg_temperature`` /
    ``seg_temperature_end`` → cosine T anneal toward hard argmax).
  * **Lever 3** — pose-FiLM STORE (``cfg.pose_film_enabled``; Wyner-Ziv
    side-info; additive ~1 KB pose codec section).
  * **Lever 4** — score-aware QAT (``score_aware_qat`` → per-tensor
    sensitivity-weighted INT8 grid; the water-filling bit-allocator).
  * **Lever 5** — margin-weighted seg promotion (``margin_weight_tau`` →
    ``exp(−margin/τ)`` boundary weight; capacity concentrates where d_seg flips).

See the design memo
``.omx/research/incurriculum_levers_design_floor_chasing_20260612.md``.

THE EXPERIMENT (the operator's question "should we resume with the seg and pose
attacks wired in? all the layer 2 perhaps?" — YES, this is that arm):
  * ARM (this script): resume from the FROZEN forkpoint
    ``experiments/results/forkpoints/basin_bc20_<stamp>/`` (the immutable init
    shared by every lever arm) into a NEW out-dir, enable BOTH attacks, train N
    epochs, eval CPU.
  * CONTROL (the live basin): the daemon (pid 33911) continuing the vendored CE +
    no-FiLM path from the SAME forkpoint epoch in
    ``experiments/results/torch_vehicle_full_mps_basin_bc20_n600/``. Diff this
    arm's trajectory vs the basin's recorded ``torch_vehicle_trajectory.jsonl``
    over the SAME global-epoch window (matched-epoch A/B — immune to wall-clock,
    so valid under GPU contention).

A/B discipline (clean delta attribution):
  * The arm resumes from the FROZEN forkpoint snapshot. Because pose-FiLM CHANGES
    the decoder architecture (the FiLM wrapper adds ``pose_film.*`` params + a
    ``stored_pose`` buffer), the vendored forkpoint checkpoint is REMAPPED into
    the FiLM-wrapper key layout (``decoder.*`` + identity-init ``pose_film.*`` +
    a ``stored_pose`` buffer seeded from the GT pose) and written as a FORK-SEED
    checkpoint (optimizer state intentionally ``None`` — the AdamW/Muon momentum
    cannot transfer across the FiLM architecture change; fresh optimizers start
    from these weights at the SAME curriculum position). FiLM is IDENTITY at init
    (``fc2`` zero-init → gamma=1/beta=0), so the first FiLM step is bit-equal to
    the baseline continuation — the pose-axis A/B delta is FiLM-only. The seg
    surrogate is opted in by mutating the active-stage ``StageSpec`` only
    (default-preserving in the driver) — no checkpoint change.
  * The control out-dir is NEVER touched; the daemon (pid 33911) is never
    killed/restarted.

Authority: per-step SegNet+PoseNet gradient on MPS (the 104x lever; full-MPS to
MATCH the live basin which runs ``--no-split-by-head``); the EXACT d_seg/d_pose
that pick BEST + seed telemetry ALWAYS run on the CPU authority (``--device``).
``[macOS-CPU advisory]`` NON-PROMOTABLE — a sub-frontier basin GATES, never IS, a
paired contest-CPU+CUDA exact eval (CLAUDE.md "MPS auth eval is NOISE" + "MPS is a
VALID TRAINING-GRADIENT device").

``--self-test`` runs a tiny ~3-epoch SYNTHETIC-scorer end-to-end check (both
attacks active → train → byte-close WITH the pose section → archive parses →
pose-section round-trips) and exits — no heavy real-scorer load, no contention
with the live basin daemon. ``--go`` is REQUIRED to launch the full real-scorer
arm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

_CONTROL_RUN_DIR = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600")
_DEFAULT_FORKPOINT = Path("experiments/results/forkpoints/basin_bc20_20260612T121523Z")
_VALID_SURROGATES = ("soft_cosine", "fisher_rao", "sinkhorn")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forkpoint", type=Path, default=_DEFAULT_FORKPOINT,
                   help="immutable basin forkpoint dir (read-only seed; NEVER mutated).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="NEW combined-arm out-dir (default: "
                        "experiments/results/l2_combined_seg_pose_ab_<stamp>). MUST "
                        "NOT be the live control run dir.")
    # -- ALL-LEVERS switch (the one command that enables every Layer-2 lever) --
    p.add_argument("--levers", choices=["seg_pose", "all"], default="seg_pose",
                   help="seg_pose (default) = the original Lever 2 (seg surrogate) + "
                        "Lever 3 (pose-FiLM) pair. all = ALSO enable Lever 1 (rate "
                        "surrogate), the Lever-2 T-anneal, Lever 4 (score-aware QAT), "
                        "and Lever 5 (margin-weighted seg) — the full Layer-2 stack. "
                        "The per-lever flags below override the --levers all defaults.")
    # -- Lever 2 (seg surrogate + anneal) --
    p.add_argument("--seg-surrogate", default="soft_cosine", choices=list(_VALID_SURROGATES),
                   help="score-domain argmax-flip d_seg surrogate on the active stage.")
    p.add_argument("--seg-temperature", type=float, default=1.0,
                   help="prediction softmax temperature for the surrogate (1.0=unit; "
                        "the START temperature when --seg-temperature-end is set).")
    p.add_argument("--seg-temperature-end", type=float, default=None,
                   help="Lever-2 ANNEAL endpoint: cosine-anneal the prediction softmax "
                        "temperature from --seg-temperature toward this value over each "
                        "stage (the memo's 'T: 1.0 -> 0.05 toward hard argmax'). None "
                        "(default) = STATIC temperature. --levers all sets 0.05.")
    # -- Lever 1 (differentiable brotli-rate surrogate) --
    p.add_argument("--rate-lambda-w", type=float, default=0.0,
                   help="Lever-1 weight-entropy rate coefficient (order-1 conditional "
                        "weight entropy; the dominant rate-term attack). 0=off. "
                        "--levers all sets a small late-stage value.")
    p.add_argument("--rate-lambda-lat", type=float, default=0.0,
                   help="Lever-1 latent temporal-delta rate coefficient (currently "
                        "unexploited latent byte lever). 0=off. --levers all sets a "
                        "small value.")
    # -- Lever 4 (score-aware QAT) --
    p.add_argument("--score-aware-qat", action=argparse.BooleanOptionalAction, default=None,
                   help="Lever-4 sensitivity-weighted INT8 grid (engages only in "
                        "use_qat stages). None (default) follows --levers (off for "
                        "seg_pose, on for all). Explicit --score-aware-qat / "
                        "--no-score-aware-qat overrides.")
    p.add_argument("--qat-sensitivity-decay", type=float, default=0.99,
                   help="EMA decay for the per-tensor score-sensitivity (Lever 4).")
    # -- Lever 5 (margin-weighted seg) --
    p.add_argument("--margin-weight-tau", type=float, default=None,
                   help="Lever-5 exp(-margin/tau) boundary weight on the seg surrogate "
                        "(capacity concentrates where d_seg flips). None (default) "
                        "follows --levers (off for seg_pose, ~2.0 for all).")
    # -- Lever 3 (pose-FiLM) --
    p.add_argument("--pose-film-hidden", type=int, default=8,
                   help="FiLM MLP bottleneck width.")
    # -- arch / budget --
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--extra-epochs", type=int, default=400,
                   help="how many MORE epochs (beyond the fork epoch) to train in the "
                        "active stage (this arm time-shares the GPU with the live "
                        "basin; keep it bounded). Ignored if --total-epoch-budget is "
                        "set (which runs the full faithful 8-stage curriculum).")
    p.add_argument("--total-epoch-budget", type=int, default=None,
                   help="proportional epoch budget across the 8 PR95 stages "
                        "(None=cap the ACTIVE stage at fork+extra-epochs and exit; the "
                        "faithful full 29,650 schedule is used for stage-boundary "
                        "alignment with the baseline when this is None). Pass the "
                        "basin's faithful 29650 to line up ALL stage boundaries.")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=25,
                   help="eval cadence (match the basin's stage-1 eval_every=25).")
    p.add_argument("--checkpoint-every-epochs", type=int, default=5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"],
                   help="scorer-path gradient backend (mps=Apple GPU 104x lever).")
    p.add_argument("--split-by-head", action=argparse.BooleanOptionalAction, default=False,
                   help="DEFAULT False to MATCH the live basin (--no-split-by-head "
                        "full-MPS). Pass --split-by-head for the pose-axis salvage "
                        "(SegNet grad on MPS / PoseNet grad on the CPU authority).")
    p.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="non-blocking CPU authority eval (the basin uses this).")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"),
                   help="dir holding gt_targets_n<N>.pt (reused to skip the precompute).")
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", action="store_true",
                   help="tiny ~3-epoch SYNTHETIC-scorer end-to-end check then exit; "
                        "does NOT load the real scorer or launch the full arm.")
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the FULL real-scorer arm.")
    p.add_argument("--dashboard", action="store_true",
                   help="print the dashboard for an existing run and exit.")
    return p


def _default_out_dir() -> Path:
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(f"experiments/results/l2_combined_seg_pose_ab_{stamp}")


def _resolve_lever_overrides(args) -> dict:
    """Resolve the active-stage ``StageSpec`` lever overrides from ``--levers`` +
    per-flag overrides. ``--levers all`` turns on EVERY Layer-2 lever with sensible
    late-stage defaults; explicit per-flag args override. Returns the dict of
    ``StageSpec`` field overrides to apply on the active (+later) stages.

    Lever 3 (pose-FiLM) is a CONFIG (cfg.pose_film_enabled), enabled separately in
    main(); it is NOT a StageSpec field, so it is not in this dict."""
    all_on = args.levers == "all"
    # Lever 2 (always on in both modes — it is the seg attack).
    overrides = {
        "seg_surrogate": args.seg_surrogate,
        "seg_temperature": args.seg_temperature,
    }
    # Lever 2 anneal: --levers all defaults the endpoint to 0.05 (the memo's target);
    # an explicit --seg-temperature-end overrides.
    if args.seg_temperature_end is not None:
        overrides["seg_temperature_end"] = args.seg_temperature_end
    elif all_on:
        overrides["seg_temperature_end"] = 0.05
    # Lever 1 (rate surrogate): --levers all defaults to small late-stage coefficients
    # (decoder-dominated, so the weight term first); explicit flags override.
    rate_w = args.rate_lambda_w if args.rate_lambda_w > 0 else (1e-3 if all_on else 0.0)
    rate_lat = args.rate_lambda_lat if args.rate_lambda_lat > 0 else (1e-3 if all_on else 0.0)
    overrides["rate_lambda_w"] = rate_w
    overrides["rate_lambda_lat"] = rate_lat
    # Lever 4 (score-aware QAT): None follows --levers; explicit bool overrides.
    saq = args.score_aware_qat if args.score_aware_qat is not None else all_on
    overrides["score_aware_qat"] = bool(saq)
    overrides["qat_sensitivity_decay"] = args.qat_sensitivity_decay
    # Lever 5 (margin weight): None follows --levers (~2.0 for all); explicit overrides.
    mtau = args.margin_weight_tau if args.margin_weight_tau is not None else (2.0 if all_on else None)
    overrides["margin_weight_tau"] = mtau
    return overrides


# ---------------------------------------------------------------------------
# Pose-FiLM fork-seed remap (identical mechanism to launch_pose_film_basin.py):
# the vendored forkpoint checkpoint is REMAPPED into the FiLM-wrapper key layout
# so the combined arm starts at the EXACT basin curriculum position with FiLM
# identity at init. The seg surrogate is a separate curriculum mutation (below).
# ---------------------------------------------------------------------------
def _identity_film_init(
    *, base_channels: int, latent_dim: int, n_pairs: int,
    pose_targets: torch.Tensor, pose_film_hidden: int,
) -> dict:
    """Harvest IDENTITY-INIT ``pose_film.*`` weights + the ``stored_pose`` buffer
    (seeded from the GT pose) from a fresh FiLM wrapper — the FiLM seed for the
    fork checkpoint (zero-init fc2 → gamma=1/beta=0 → first step == baseline)."""
    from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper
    from tac.torch_vehicle.vendored_imports import import_vendored

    vendored = import_vendored("model").HNeRVDecoder(
        latent_dim=latent_dim, base_channels=base_channels, eval_size=(384, 512)
    )
    wrapper = PoseFiLMHNeRVWrapper(vendored, n_pairs=n_pairs, film_hidden=pose_film_hidden)
    wrapper.set_stored_pose(pose_targets[:n_pairs].cpu())
    return {
        k: v.detach().cpu().clone()
        for k, v in wrapper.state_dict().items()
        if k.startswith("pose_film.") or k == "stored_pose"
    }


def _seed_film_checkpoint_from_forkpoint(
    forkpoint: Path,
    out_dir: Path,
    *,
    base_channels: int,
    latent_dim: int,
    n_pairs: int,
    pose_targets: torch.Tensor,
    pose_film_hidden: int,
) -> dict:
    """Remap the VENDORED forkpoint checkpoint into a FiLM-wrapper FORK-SEED
    checkpoint in ``out_dir`` (optimizer state ``None`` → fresh optimizers).

    IDEMPOTENT: if ``out_dir`` already holds this arm's OWN checkpoint, resume
    from it (do NOT clobber progress) — return the existing seed info.

    The curriculum position (stage/epoch) + RNG are preserved so the combined arm
    starts at the EXACT basin point the control continues from. FiLM is identity
    at init."""
    from tac.torch_vehicle.checkpoint import (
        TorchCheckpointPosition,
        checkpoint_exists,
        load_checkpoint,
        read_manifest,
        save_checkpoint,
    )

    if checkpoint_exists(out_dir):
        man = read_manifest(out_dir)
        return {
            "resumed_from_own_checkpoint": True,
            "seeded_position": {
                "stage_index": int(man["stage_index"]),
                "epoch_in_stage": int(man["epoch_in_stage"]),
                "stage_name": man.get("stage_name", ""),
            },
        }

    man = read_manifest(forkpoint)
    merged = load_checkpoint(forkpoint, map_location="cpu")
    if int(man["base_channels"]) != base_channels:
        raise ValueError(
            f"forkpoint base_channels={man['base_channels']} != "
            f"--base-channels={base_channels}"
        )

    film_init = _identity_film_init(
        base_channels=base_channels, latent_dim=latent_dim, n_pairs=n_pairs,
        pose_targets=pose_targets, pose_film_hidden=pose_film_hidden,
    )

    def _wrap_decoder_sd(vendored_sd: dict) -> dict:
        out = {f"decoder.{k}": v for k, v in vendored_sd.items()}
        out.update(film_init)  # identity FiLM + stored_pose buffer
        return out

    seed_state = {
        "decoder": _wrap_decoder_sd(merged["decoder"]),
        "latents": merged["latents"],
        "ema_decoder": _wrap_decoder_sd(merged["ema_decoder"]),
        "ema_latents": merged["ema_latents"],
        "adamw": None, "muon": None, "adamw_sched": None, "muon_sched": None,
        "torch_rng": merged.get("torch_rng"),
        "numpy_rng": merged.get("numpy_rng"),
        "base_channels": base_channels, "latent_dim": latent_dim, "n_pairs": n_pairs,
        "stage_name": man.get("stage_name", ""),
        "ema_decay": float(man.get("ema_decay", 0.999)),
        "best_score": float("inf"),  # the combined arm tracks its OWN best fresh
        "best_ep": int(man.get("best_ep", 0)),
        "best_stage": int(man.get("best_stage", 0)),
    }
    position = TorchCheckpointPosition(
        stage_index=int(man["stage_index"]),
        epoch_in_stage=int(man["epoch_in_stage"]),
    )
    save_checkpoint(seed_state, out_dir, position)
    return {
        "resumed_from_own_checkpoint": False,
        "seeded_position": {
            "stage_index": position.stage_index,
            "epoch_in_stage": position.epoch_in_stage,
            "stage_name": man.get("stage_name", ""),
        },
        "film_param_keys": sorted(k for k in film_init if k.startswith("pose_film.")),
        "forkpoint_best_score": float(man.get("best_score", float("inf"))),
    }


def _build_combined_curriculum(
    *,
    fork_stage: int,
    fork_epoch_in_stage: int,
    extra_epochs: int,
    total_epoch_budget: int | None,
    ema_decay: float,
    eval_every: int,
    lever_overrides: dict,
):
    """Build the faithful PR95 curriculum, enable the StageSpec-level levers (seg
    surrogate + anneal + rate surrogate + score-aware QAT + margin weight per
    ``lever_overrides``) on the active stage, and (when ``total_epoch_budget`` is
    None) CAP the active stage at fork+extra-epochs + truncate so the arm exits
    cleanly with a DONE.

    When ``total_epoch_budget`` is given (e.g. the basin's faithful 29650), the
    FULL 8-stage curriculum is kept (stage boundaries line up with the baseline
    for matched-epoch diffing across stages) and the levers are enabled on the
    active stage AND every later stage (so the attack stays on as the curriculum
    advances). ``score_aware_qat`` only ENGAGES in ``use_qat`` stages (the override
    is a no-op on non-QAT stages — byte-identical there)."""
    from tac.torch_vehicle.curriculum import build_curriculum

    curriculum = build_curriculum(
        total_epoch_budget=total_epoch_budget,
        ema_decay=ema_decay,
        eval_every=eval_every,
    )

    def _enable_seg(spec):
        return replace(spec, **lever_overrides)

    if total_epoch_budget is None:
        # Cap the ACTIVE stage at fork+extra; truncate the curriculum there so the
        # arm runs a bounded window and exits with a DONE marker. Seg surrogate on
        # the active stage only (we A/B within the fork stage).
        target_epochs = fork_epoch_in_stage + int(extra_epochs)
        active = _enable_seg(replace(curriculum[fork_stage], epochs=target_epochs))
        capped = [*curriculum[:fork_stage], active]
        return capped, target_epochs

    # Full faithful schedule: enable the seg surrogate on the active stage AND all
    # later stages (the attack persists as the curriculum advances). Earlier
    # already-completed stages are kept as-is (the resume skips them).
    out = []
    for i, spec in enumerate(curriculum):
        out.append(_enable_seg(spec) if i >= fork_stage else spec)
    return out, None


def _run_self_test(args, out_dir: Path) -> int:
    """Tiny SYNTHETIC-scorer end-to-end: 3-epoch ALL-ACTIVE-LEVERS train → byte-close
    WITH the pose section → archive parses → pose-section round-trips. Proves the
    composed lever driver/train/export path runs end-to-end WITHOUT loading the heavy
    real scorer (no basin contention). With ``--levers all`` EVERY Layer-2 lever is
    on (rate surrogate + seg surrogate + T-anneal + score-aware QAT + margin weight +
    pose-FiLM) — the compose-all-five smoke. Starts from a fresh 6-pair init (fast,
    self-contained); the full arm exercises the real-forkpoint resume."""
    import torch.nn.functional as F  # noqa: N812

    from tac.torch_vehicle.curriculum import StageSpec
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.pose_film import parse_pose_section
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    n_pairs = 6
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=0)

    def _ce(s, t):
        return F.cross_entropy(s, t)

    lever_overrides = _resolve_lever_overrides(args)
    all_on = args.levers == "all"
    # Score-aware QAT only engages in use_qat stages; with --levers all, exercise it
    # by turning on use_qat for the smoke (so the QAT block actually runs).
    use_qat = bool(lever_overrides.get("score_aware_qat")) and all_on
    # ALL levers active (StageSpec-level + cfg-level pose-FiLM).
    spec = StageSpec(
        name="self_test", epochs=3, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
        cat_lambda=(0.01 if all_on else 0.0), cat_sigma=0.2,
        use_qat=use_qat, init_latents_random=True, **lever_overrides,
    )
    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=1, device="cpu", seed=0,
        pose_film_enabled=True, pose_film_hidden=args.pose_film_hidden,
    )
    driver = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=[spec]
    )
    summary = driver.run()

    arch_path = out_dir / "best" / "best_archive.bin"
    archive_ok = arch_path.exists()
    pose_ok = False
    seg_attack_active = spec.seg_surrogate in _VALID_SURROGATES
    pose_attack_active = bool(cfg.pose_film_enabled)
    rate_active = spec.rate_lambda_w > 0 or spec.rate_lambda_lat > 0
    anneal_active = spec.seg_temperature_end is not None
    qat_active = bool(spec.score_aware_qat) and spec.use_qat
    margin_active = spec.margin_weight_tau is not None
    if archive_ok:
        pose = parse_pose_section(arch_path.read_bytes(), driver.v.parse_archive)
        pose_ok = pose is not None and tuple(pose.shape) == (n_pairs, 6)
    # PASS requires the end-to-end + pose-section round-trip + the seg & pose attacks;
    # with --levers all it ALSO requires every other lever to be active (compose-five).
    passed = archive_ok and pose_ok and seg_attack_active and pose_attack_active
    if all_on:
        passed = passed and rate_active and anneal_active and qat_active and margin_active
    result = {
        "self_test": "PASS" if passed else "FAIL",
        "levers_mode": args.levers,
        "active_levers": {
            "lever1_rate_surrogate": rate_active,
            "lever2_seg_surrogate": spec.seg_surrogate,
            "lever2_temperature_anneal": anneal_active,
            "lever3_pose_film": pose_attack_active,
            "lever4_score_aware_qat": qat_active,
            "lever5_margin_weight": margin_active,
        },
        "best_score": summary.get("best_score"),
        "best_archive_exists": archive_ok,
        "pose_section_parses": pose_ok,
        "out_dir": str(out_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.dashboard:
        from tac.torch_vehicle.telemetry import render_dashboard

        print(render_dashboard(args.out_dir))
        return 0

    out_dir = Path(args.out_dir or _default_out_dir())
    # HARD GUARD: never write into the live control run dir (basin contention).
    if out_dir.resolve() == _CONTROL_RUN_DIR.resolve():
        raise SystemExit(
            f"REFUSED: --out-dir is the live control run dir {_CONTROL_RUN_DIR}. "
            "The combined A/B MUST use a NEW out-dir (clean delta attribution + safety)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        return _run_self_test(args, out_dir)

    if not args.forkpoint.exists():
        raise SystemExit(f"forkpoint not found: {args.forkpoint}")

    if not args.go:
        raise SystemExit(
            "REFUSED to launch the full real-scorer arm without --go. Re-run with "
            "--go to launch, or --self-test for a tiny end-to-end check."
        )

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

    # Fork-seed the FiLM-wrapper checkpoint from the immutable basin forkpoint.
    seed_info = _seed_film_checkpoint_from_forkpoint(
        args.forkpoint, out_dir,
        base_channels=args.base_channels, latent_dim=args.latent_dim,
        n_pairs=int(scorer.n_pairs), pose_targets=scorer.pose_targets,
        pose_film_hidden=args.pose_film_hidden,
    )
    print(json.dumps({"fork_seed": seed_info}, indent=2, sort_keys=True), flush=True)
    seeded = seed_info["seeded_position"]
    fork_stage = int(seeded["stage_index"])
    fork_epoch_in_stage = int(seeded["epoch_in_stage"])

    # Build the curriculum with the StageSpec-level levers enabled — pose-FiLM
    # (Lever 3) is enabled via the cfg below (it is a cfg, not a StageSpec field).
    lever_overrides = _resolve_lever_overrides(args)
    curriculum, capped_to = _build_combined_curriculum(
        fork_stage=fork_stage,
        fork_epoch_in_stage=fork_epoch_in_stage,
        extra_epochs=args.extra_epochs,
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        lever_overrides=lever_overrides,
    )
    print(json.dumps({
        "levers_mode": args.levers,
        "active_levers": {
            "lever1_rate_lambda_w": lever_overrides.get("rate_lambda_w", 0.0),
            "lever1_rate_lambda_lat": lever_overrides.get("rate_lambda_lat", 0.0),
            "lever2_seg_surrogate": lever_overrides.get("seg_surrogate"),
            "lever2_seg_temperature": lever_overrides.get("seg_temperature"),
            "lever2_seg_temperature_end": lever_overrides.get("seg_temperature_end"),
            "lever3_pose_film_enabled": True,
            "lever4_score_aware_qat": lever_overrides.get("score_aware_qat"),
            "lever5_margin_weight_tau": lever_overrides.get("margin_weight_tau"),
        },
        "fork_stage": fork_stage,
        "fork_epoch_in_stage": fork_epoch_in_stage,
        "active_stage_capped_to": capped_to,
        "total_epoch_budget": args.total_epoch_budget,
        "split_by_head": args.split_by_head,
        "out_dir": str(out_dir),
    }, indent=2, sort_keys=True), flush=True)

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        pose_film_enabled=True, pose_film_hidden=args.pose_film_hidden, seed=args.seed,
    )
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
