#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Launch the Lever-3 pose-FiLM A/B arm — resume from the IMMUTABLE basin forkpoint
into a NEW out-dir, enable ONLY pose-FiLM (all else default-OFF), train forward.

Lever 3 (the Quantizr STORE-pose lesson; design memo
``.omx/research/incurriculum_levers_design_floor_chasing_20260612.md`` §"Lever 3"):
store the 6 GT pose scalars per pair as side information (Wyner-Ziv) and
FiLM-condition the decoder on them, so ``d_pose`` collapses toward the stored-pose
quant floor and the decoder capacity is freed for ``d_seg`` + rate. The stored pose
is range-coded into an ADDITIVE archive section (~1 KB charged); the vendored codec
stays pristine.

A/B discipline (clean delta attribution):
  * The arm resumes from the FROZEN forkpoint snapshot
    ``experiments/results/forkpoints/basin_bc20_<stamp>/`` (the immutable init shared
    by every lever arm) into a NEW out-dir — NEVER the live control run dir.
  * The forkpoint was saved from a VENDORED (no-FiLM) decoder; this launcher REMAPS
    its decoder + EMA weights into the FiLM-wrapper key layout (``decoder.*`` +
    identity-init ``pose_film.*`` + a ``stored_pose`` buffer seeded from the GT pose)
    and writes a FORK-SEED checkpoint (optimizer state intentionally ``None`` — the
    AdamW/Muon momentum cannot transfer across the FiLM architecture change; fresh
    optimizers start from these weights at the SAME curriculum position). FiLM is
    IDENTITY at init, so the first FiLM step is bit-equal to the baseline
    continuation — the A/B delta is FiLM-only.
  * The control arm continues the baseline from the SAME forkpoint with FiLM OFF;
    diff the two trajectories for the clean pose-axis delta.

GATING: this launcher does NOT start a full training arm by itself — the GO/NO-GO is
the SEPARATE CPU disambiguator smoke
(``experiments/smoke_pose_film_cpu_disambiguator.py``), which decides whether the
base_ch=20 d_pose has headroom above the stored-pose quant floor (else the pose
section is pure byte cost). Run it FIRST. ``--self-test`` runs a tiny ~3-epoch
SYNTHETIC-scorer end-to-end check (fork-seed remap → train → byte-close → pose
parse) and exits — no heavy real-scorer load, no contention with the live basin
daemon. ``--go`` is REQUIRED to launch the full real-scorer arm.

Authority: per-step SegNet gradient on MPS (the 90x lever; bit-identical on d_seg);
per-step PoseNet gradient on the CPU AUTHORITY (zero pose drift); the EXACT
d_seg/d_pose that pick BEST run on the CPU authority. ``[macOS-CPU advisory]``
NON-PROMOTABLE — a sub-frontier basin GATES, never IS, a paired contest-CPU+CUDA
exact eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_CONTROL_RUN_DIR = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600")
_DEFAULT_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forkpoint", type=Path, default=_DEFAULT_FORKPOINT,
                   help="immutable basin forkpoint dir (read-only seed; NEVER mutated).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="NEW FiLM A/B out-dir (default: "
                        "experiments/results/lever3_pose_film_ab_<stamp>). MUST NOT be "
                        "the live control run dir.")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--total-epoch-budget", type=int, default=None,
                   help="proportional epoch budget across the 8 PR95 stages "
                        "(None=full faithful 29,650).")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"],
                   help="SegNet-path gradient backend (mps=Apple GPU 90x lever).")
    p.add_argument("--split-by-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pose-film-hidden", type=int, default=8,
                   help="FiLM MLP bottleneck width.")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"),
                   help="dir holding gt_targets_n<N>.pt (reused to skip the precompute).")
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", action="store_true",
                   help="tiny ~3-epoch SYNTHETIC-scorer end-to-end check then exit; "
                        "does NOT load the real scorer or launch the full arm.")
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the FULL real-scorer arm (the CPU "
                        "disambiguator smoke is the GO/NO-GO gate).")
    p.add_argument("--dashboard", action="store_true",
                   help="print the dashboard for an existing run and exit.")
    return p


def _default_out_dir() -> Path:
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(f"experiments/results/lever3_pose_film_ab_{stamp}")


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
    position_override=None,
) -> dict:
    """Remap the VENDORED forkpoint checkpoint into a FiLM-wrapper FORK-SEED
    checkpoint in ``out_dir`` (optimizer state ``None`` → fresh optimizers).

    The curriculum position (stage/epoch) + RNG are preserved (unless
    ``position_override`` is given) so the FiLM arm starts at the EXACT basin point
    the control continues from. FiLM is identity at init."""
    from tac.torch_vehicle.checkpoint import (
        TorchCheckpointPosition,
        load_checkpoint,
        read_manifest,
        save_checkpoint,
    )

    man = read_manifest(forkpoint)
    merged = load_checkpoint(forkpoint, map_location="cpu")
    if int(man["base_channels"]) != base_channels:
        raise ValueError(
            f"forkpoint base_channels={man['base_channels']} != --base-channels={base_channels}"
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
        "best_score": float("inf"),  # the FiLM arm tracks its OWN best fresh
        "best_ep": int(man.get("best_ep", 0)),
        "best_stage": int(man.get("best_stage", 0)),
    }
    position = position_override or TorchCheckpointPosition(
        stage_index=int(man["stage_index"]),
        epoch_in_stage=int(man["epoch_in_stage"]),
    )
    save_checkpoint(seed_state, out_dir, position)
    return {
        "seeded_position": {
            "stage_index": position.stage_index,
            "epoch_in_stage": position.epoch_in_stage,
            "stage_name": man.get("stage_name", ""),
        },
        "film_param_keys": sorted(k for k in film_init if k.startswith("pose_film.")),
        "forkpoint_best_score": float(man.get("best_score", float("inf"))),
    }


def _run_self_test(args, out_dir: Path) -> int:
    """Tiny SYNTHETIC-scorer end-to-end: 3-epoch FiLM train → byte-close →
    pose-section parse. Proves the FiLM driver/train/export path runs end-to-end
    WITHOUT loading the heavy real scorer (no basin contention). This validates the
    FiLM WIRE-IN mechanics (the full arm exercises the real-forkpoint resume); it
    starts from a fresh 6-pair init, not the 600-pair forkpoint, so it is fast and
    self-contained."""
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
        return torch.nn.functional.cross_entropy(s, t)

    spec = StageSpec(
        name="self_test", epochs=3, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
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
    ok = arch_path.exists()
    pose_ok = False
    if ok:
        pose = parse_pose_section(arch_path.read_bytes(), driver.v.parse_archive)
        pose_ok = pose is not None and tuple(pose.shape) == (n_pairs, 6)
    result = {
        "self_test": "PASS" if (ok and pose_ok) else "FAIL",
        "best_score": summary.get("best_score"),
        "best_archive_exists": ok,
        "pose_section_parses": pose_ok,
        "out_dir": str(out_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (ok and pose_ok) else 1


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
            "The FiLM A/B MUST use a NEW out-dir (clean delta attribution + safety)."
        )
    if not args.forkpoint.exists():
        raise SystemExit(f"forkpoint not found: {args.forkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        return _run_self_test(args, out_dir)

    if not args.go:
        raise SystemExit(
            "REFUSED to launch the full arm without --go. The GO/NO-GO is the SEPARATE "
            "CPU disambiguator smoke (experiments/smoke_pose_film_cpu_disambiguator.py): "
            "run it FIRST; it decides whether base_ch=20 d_pose has headroom above the "
            "stored-pose quant floor. Re-run with --go once the smoke says GO, or "
            "--self-test for a tiny end-to-end check."
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
    seed_info = _seed_film_checkpoint_from_forkpoint(
        args.forkpoint, out_dir,
        base_channels=args.base_channels, latent_dim=args.latent_dim,
        n_pairs=int(scorer.n_pairs), pose_targets=scorer.pose_targets,
        pose_film_hidden=args.pose_film_hidden,
    )
    print(json.dumps({"fork_seed": seed_info}, indent=2, sort_keys=True), flush=True)

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        pose_film_enabled=True, pose_film_hidden=args.pose_film_hidden, seed=args.seed,
    )
    driver = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle())
    summary = driver.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
