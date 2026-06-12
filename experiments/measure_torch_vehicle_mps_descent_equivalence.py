#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Descent-equivalence A/B for the torch-vehicle MPS gradient: does training the
P2 PR95 vehicle with the per-step forward+backward on torch-MPS (the 104x Apple
GPU lever) reach the SAME exact-(torch-CPU)-d_seg AND d_pose trajectory as
training with the torch-CPU authority gradient?

This is the DECISIVE gate before the base_ch=20 basin run is launched on MPS.
The 90-104x bench + ~1.0 per-step gradient cosine are a FIRST sanity — NOT
descent-equivalence (the n600 lesson: a ~1.0 cosine gradient can still compound
into a POSE blow-up that a d_seg-only check never sees). This harness runs the
bounded both-terms A/B and adjudicates through the canonical reusable acceptance
gate (``tac.mlx_pr95_port.speedup_acceptance_gate.evaluate_descent_equivalence``),
which REFUSES a d_seg-only pass and detects the pose-divergence signature.

Method (apples-to-apples, NO-FAKE):
  * Build TWO ``TorchVehicleDriver`` runs from the SAME seed (identical CPU init).
  * Arm A: device='cpu', train_device='cpu'  — the AUTHORITY gradient (baseline).
  * Arm B: device='cpu', train_device='mps'  — the FAST gradient (candidate).
  * Same curriculum, same n, same per-epoch CPU-drawn pair permutation (the driver
    builds randperm on CPU then moves, so the permutation is device-independent).
  * BOTH arms' BEST-tracker exact_eval runs on the torch-CPU authority (the driver
    always evals on ``device``=cpu regardless of train_device) — so the REPORTED
    d_seg/d_pose are the SAME authority metric on both arms; only the GRADIENT
    that drove the steps differs.
  * Read each arm's per-eval (d_seg, d_pose) from the telemetry JSONL and feed the
    two trajectories to the acceptance gate.

If BOTH d_seg AND d_pose track within tolerance with no divergence -> the MPS
gradient is descent-equivalent -> the basin run can be launched on MPS (the
104x). If EITHER diverges -> REJECT; recommend the torch-CPU basin (pid 42035)
or the Modal CUDA run, and report which term failed.

Authority: the exact d_seg/d_pose are torch-CPU (the only authority for BOTH
arms). The MPS arm's GRADIENT is research-signal; its REPORTED metric is
torch-CPU. $0, local. The PASS at n<600 is PROVISIONAL (gate flags it) — but a
PASS at a feasible n + the ~1.0 cosine is the evidence that licenses the n600
basin run (which itself re-evals on CPU authority every eval epoch, so a late
divergence would still be CAUGHT live, not hidden).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tac.torch_vehicle.curriculum import StageSpec


def _ce_seg_loss(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _muon_basin_curriculum(epochs: int, eval_every: int, batch_size: int) -> list[StageSpec]:
    """A single-stage Muon curriculum — the stage-8 recipe that drives the basin
    (Muon on the decoder weights + AdamW on the latents). Exercises the FULL
    gradient path (Muon Newton-Schulz orthogonalization + AdamW) so the A/B tests
    the real descent gradient, not a toy AdamW-only path.

    init_latents_random=True so arm A and arm B start from the SAME CPU-drawn
    latents (the driver draws on CPU then moves to the train device).
    """
    return [
        StageSpec(
            name="ab_muon_basin",
            epochs=epochs,
            seg_loss_fn=_ce_seg_loss,
            eval_every=eval_every,
            batch_size=batch_size,
            ema_decay=0.999,
            use_muon=True,
            adamw_lr=1e-3,
            muon_lr=0.02,
            muon_weight_decay=5e-4,
            latent_lr_mult=10.0,
            grad_clip=50.0,
            grad_clip_muon=50.0,
            lr_floor_ratio=5e-6,
            seg_weight=100.0,
            pose_weight=1.0,
            cat_lambda=0.0,
            cat_sigma=0.1,
            use_qat=False,
            init_latents_random=True,
        )
    ]


def _run_arm(out_dir, *, train_device, video_path, n_pairs, base_channels,
             curriculum, seed, label, targets_cache, split_by_head=False):
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    cfg = TorchVehicleConfig(
        base_channels=base_channels,
        latent_dim=28,
        out_dir=Path(out_dir),
        checkpoint_every_epochs=10_000,  # the A/B is short; skip mid-run checkpoints
        device="cpu",                    # AUTHORITY = CPU (the exact metric)
        train_device=train_device,       # GRADIENT backend (cpu or mps)
        split_by_head=split_by_head,     # SegNet on train_device, PoseNet on CPU auth
        seed=seed,
    )
    # Capped + cached CPU-authority targets (the per-step target precompute is
    # bounded to n_pairs; the cache is reused across both arms + re-runs).
    scorer = RealScorerContext(
        video_path, device="cpu", train_device=train_device,
        split_by_head=split_by_head,
        max_pairs=int(n_pairs), targets_cache=targets_cache,
    )
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    t0 = time.time()
    summary = driver.run()
    dt = time.time() - t0
    # Read the per-eval (d_seg, d_pose) trajectory from the telemetry JSONL.
    traj_path = Path(out_dir) / "torch_vehicle_trajectory.jsonl"
    rows = [json.loads(line) for line in traj_path.read_text().strip().splitlines()]
    evald = [r for r in rows if r.get("evaluated") and r.get("d_seg") is not None]
    traj = [
        {"epoch": r["global_epoch"], "d_seg": float(r["d_seg"]),
         "d_pose": float(r["d_pose"])}
        for r in evald
    ]
    n_epochs = max((r["global_epoch"] for r in rows), default=0)
    s_per_epoch = dt / max(n_epochs, 1)
    print(f"[{label}] done: {n_epochs} epochs in {dt:.1f}s ({s_per_epoch:.2f}s/epoch); "
          f"best_score={summary.get('best_score'):.5f}", flush=True)
    for t in traj:
        print(f"  [{label}] ep {t['epoch']:>4}: d_seg={t['d_seg']:.6f} d_pose={t['d_pose']:.6f}",
              flush=True)
    return traj, s_per_epoch, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--out-root",
                    default="experiments/results/torch_vehicle_mps_descent_ab")
    ap.add_argument("--out-json",
                    default="experiments/results/torch_vehicle_mps_descent_ab/verdict.json")
    ap.add_argument("--candidate-mode", default="full_mps",
                    choices=["full_mps", "split_by_head"],
                    help="full_mps: candidate runs the WHOLE forward on MPS (the prior "
                         "REJECT). split_by_head: SegNet on MPS, PoseNet on the CPU "
                         "authority, cotangents summed (the pose-axis salvage).")
    args = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    if not torch.backends.mps.is_available():
        print("[FATAL] torch-MPS unavailable — cannot run the MPS arm. Use a Mac.",
              flush=True)
        return 2

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = str(import_vendored("data").get_default_video_path())

    out_root = Path(args.out_root)
    targets_cache = out_root / "gt_targets_cache"
    curriculum = _muon_basin_curriculum(args.epochs, args.eval_every, args.batch_size)

    print("=== ARM A: train_device=cpu (AUTHORITY gradient, baseline) ===", flush=True)
    traj_a, spe_a, sum_a = _run_arm(
        out_root / "arm_cpu", train_device="cpu", video_path=video_path,
        n_pairs=args.n_pairs, base_channels=args.base_channels,
        curriculum=curriculum, seed=args.seed, label="cpu_grad",
        targets_cache=targets_cache)

    split_by_head = args.candidate_mode == "split_by_head"
    arm_b_label = "split_by_head_grad" if split_by_head else "mps_grad"
    arm_b_dir = "arm_split_by_head" if split_by_head else "arm_mps"
    print(f"\n=== ARM B: train_device=mps candidate_mode={args.candidate_mode} "
          f"({'SegNet=MPS / PoseNet=CPU-authority' if split_by_head else 'WHOLE forward MPS'}) ===",
          flush=True)
    traj_b, spe_b, sum_b = _run_arm(
        out_root / arm_b_dir, train_device="mps", video_path=video_path,
        n_pairs=args.n_pairs, base_channels=args.base_channels,
        curriculum=curriculum, seed=args.seed, label=arm_b_label,
        targets_cache=targets_cache, split_by_head=split_by_head)

    # Adjudicate through the CANONICAL both-terms acceptance gate.
    from tac.mlx_pr95_port.speedup_acceptance_gate import evaluate_descent_equivalence

    verdict = evaluate_descent_equivalence(traj_a, traj_b, n_pairs=args.n_pairs)

    print("\n=== BOTH-TERMS ACCEPTANCE GATE (exact torch-CPU d_seg AND d_pose) ===")
    by_a = {r["epoch"]: r for r in traj_a}
    by_b = {r["epoch"]: r for r in traj_b}
    print(f"{'epoch':>6} {'cpu_seg':>10} {'mps_seg':>10} {'cpu_pose':>11} {'mps_pose':>11}")
    for ep in sorted(set(by_a) & set(by_b)):
        a, b = by_a[ep], by_b[ep]
        print(f"{ep:>6} {a['d_seg']:>10.6f} {b['d_seg']:>10.6f} "
              f"{a['d_pose']:>11.6f} {b['d_pose']:>11.6f}")
    print(f"\nseg:  {verdict.seg.reason}")
    print(f"pose: {verdict.pose.reason}")
    print(f"\nVERDICT: {'PASS' if verdict.passed else 'REJECT'} "
          f"(n={verdict.n_pairs}, epochs_compared={verdict.epochs_compared})")
    for r in verdict.reasons:
        print(f"  - {r}")
    print(f"\nthroughput: cpu_grad {spe_a:.2f}s/epoch  mps_grad {spe_b:.2f}s/epoch  "
          f"speedup={spe_a / max(spe_b, 1e-9):.1f}x")

    out = {
        "config": vars(args),
        "candidate_mode": args.candidate_mode,
        "arm_cpu_grad": traj_a, "arm_candidate": traj_b,
        "cpu_grad_s_per_epoch": spe_a, "candidate_s_per_epoch": spe_b,
        "epoch_speedup": spe_a / max(spe_b, 1e-9),
        "gate_passed": verdict.passed,
        "gate_generalization_warning": verdict.generalization_warning,
        "gate_reasons": list(verdict.reasons),
        "seg_verdict": {
            "tracks": verdict.seg.tracks_within_tol, "diverged": verdict.seg.diverged,
            "final_abs_gap": verdict.seg.final_abs_gap, "reason": verdict.seg.reason,
        },
        "pose_verdict": {
            "tracks": verdict.pose.tracks_within_tol, "diverged": verdict.pose.diverged,
            "diverged_at_epoch": verdict.pose.diverged_at_epoch,
            "final_abs_gap": verdict.pose.final_abs_gap, "reason": verdict.pose.reason,
        },
        "authority": "[macOS-CPU advisory] exact d_seg/d_pose torch-CPU for BOTH arms; "
                     "MPS arm GRADIENT is research-signal. NON-PROMOTABLE.",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
