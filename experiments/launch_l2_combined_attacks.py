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

Lever 1 (the differentiable brotli-rate surrogate, ``tac/losses/rate_surrogate.py``)
is DESIGN-ONLY — that module was NOT built (it does not exist on disk), so it is
SKIPPED here per the task brief (do NOT build it from scratch in this arm). This
arm is seg+pose, the two wired-in attacks. See the design memo
``.omx/research/incurriculum_levers_design_floor_chasing_20260612.md`` §"Lever 1".

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
    # -- Lever 2 (seg surrogate) --
    p.add_argument("--seg-surrogate", default="soft_cosine", choices=list(_VALID_SURROGATES),
                   help="score-domain argmax-flip d_seg surrogate on the active stage.")
    p.add_argument("--seg-temperature", type=float, default=1.0,
                   help="prediction softmax temperature for the surrogate (1.0=unit; "
                        "the design memo's anneal toward hard (T->0.05) needs a "
                        "per-epoch driver hook that does NOT exist yet, so this is a "
                        "STATIC per-stage T — start at 1.0, the lowest-risk option-(b) "
                        "value per the memo's anneal-instability note).")
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
    seg_surrogate: str,
    seg_temperature: float,
):
    """Build the faithful PR95 curriculum, enable the seg surrogate on the active
    stage, and (when ``total_epoch_budget`` is None) CAP the active stage at
    fork+extra-epochs + truncate so the arm exits cleanly with a DONE.

    When ``total_epoch_budget`` is given (e.g. the basin's faithful 29650), the
    FULL 8-stage curriculum is kept (stage boundaries line up with the baseline
    for matched-epoch diffing across stages) and the seg surrogate is enabled on
    the active stage AND every later stage (so the attack stays on as the
    curriculum advances)."""
    from tac.torch_vehicle.curriculum import build_curriculum

    curriculum = build_curriculum(
        total_epoch_budget=total_epoch_budget,
        ema_decay=ema_decay,
        eval_every=eval_every,
    )

    def _enable_seg(spec):
        return replace(spec, seg_surrogate=seg_surrogate, seg_temperature=seg_temperature)

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
    """Tiny SYNTHETIC-scorer end-to-end: 3-epoch BOTH-ATTACKS train → byte-close
    WITH the pose section → archive parses → pose-section round-trips. Proves the
    combined seg+pose driver/train/export path runs end-to-end WITHOUT loading the
    heavy real scorer (no basin contention). Starts from a fresh 6-pair init (fast,
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

    # BOTH attacks active: seg_surrogate set + pose_film_enabled in the cfg.
    spec = StageSpec(
        name="self_test", epochs=3, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
        seg_surrogate=args.seg_surrogate, seg_temperature=args.seg_temperature,
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
    if archive_ok:
        pose = parse_pose_section(arch_path.read_bytes(), driver.v.parse_archive)
        pose_ok = pose is not None and tuple(pose.shape) == (n_pairs, 6)
    passed = archive_ok and pose_ok and seg_attack_active and pose_attack_active
    result = {
        "self_test": "PASS" if passed else "FAIL",
        "both_attacks_active": {
            "seg_surrogate": spec.seg_surrogate,
            "seg_attack_active": seg_attack_active,
            "pose_film_enabled": pose_attack_active,
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

    # Build the curriculum with the seg surrogate enabled (Lever 2) — pose-FiLM
    # (Lever 3) is enabled via the cfg below.
    curriculum, capped_to = _build_combined_curriculum(
        fork_stage=fork_stage,
        fork_epoch_in_stage=fork_epoch_in_stage,
        extra_epochs=args.extra_epochs,
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        seg_surrogate=args.seg_surrogate,
        seg_temperature=args.seg_temperature,
    )
    print(json.dumps({
        "combined_attacks": {
            "lever2_seg_surrogate": args.seg_surrogate,
            "lever2_seg_temperature": args.seg_temperature,
            "lever3_pose_film_enabled": True,
            "lever1_rate_surrogate": "SKIPPED (tac/losses/rate_surrogate.py not built)",
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
