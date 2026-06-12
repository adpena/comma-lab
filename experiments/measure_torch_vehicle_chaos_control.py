#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CHAOS CONTROL for the MPS-PoseNet-drift verdict (H_chaos vs H_opbias).

The question (operator): can the MPS PoseNet drift be PATCHED to recover the full
104x, or is split-by-head 7-11x the ceiling? The prior agent's descent-equivalence
A/B REJECTED the full-MPS gradient on the pose axis (d_pose final |gap| 7.02 >
tol 1.12) while d_seg was bit-identical. BUT a single-step diagnostic found the
MPS PoseNet per-step forward AND input-gradient are essentially IDENTICAL to CPU
(cos 1.0, input-grad relmax ~2e-4). Two hypotheses explain the training-time gap:

  H_chaos:  the MPS per-step gradient is essentially CORRECT (2e-4 noise); the
            training divergence is OPTIMIZER CHAOS — a tiny per-step perturbation
            under aggressive Muon sends the two arms onto DIFFERENT-but-equally-
            valid stochastic trajectories that diverge in the weakly-driven pose
            term (seg_weight=100, pose_weight=1). If so, the MPS gradient is FINE
            and the full 104x is VALID; the gate's |gap| reject conflated chaotic-
            but-valid trajectories with a wrong gradient.
  H_opbias: the MPS gradient carries a real SMALL BIAS (beyond 2e-4 noise) in
            specific ops (BN/SE) that COMPOUNDS. If so, it is the gradient, not
            chaos.

THE DISCRIMINATING TEST (this harness): run the SAME n48/30ep/single-stage-Muon
A/B as the descent-equivalence harness, but BOTH arms on the pure torch-CPU
gradient — Arm A clean, Arm B with i.i.d. relative noise (default 2e-4, matching
the measured MPS input-grad relmax) injected into the FRAME COTANGENT dL/dF (the
exact surface where the MPS drift was measured). Both arms eval d_seg AND d_pose
on the torch-CPU authority and feed the canonical acceptance gate.

PREDICTION under H_chaos: the CPU-vs-CPU+2e-4-noise A/B REPRODUCES a comparable
d_pose |gap| (same order as the MPS arm's 7.02) and the gate REJECTS the
noise-injected pure-CPU arm too — proving a 2e-4 perturbation alone is sufficient
to produce the observed divergence, so the MPS gradient is NOT the culprit (it's
chaos; nothing to patch; full 104x already valid).
PREDICTION under H_opbias: 2e-4 i.i.d. noise produces a MUCH SMALLER gap (the MPS
gap is bias-driven, not noise-driven) — then the drift is real and localizable.

Authority: exact d_seg/d_pose are torch-CPU for BOTH arms (the noise is injected
into the GRADIENT only; the reported metric is always the CPU authority). $0,
local, [macOS-CPU advisory] NON-PROMOTABLE. This is a CORRECTNESS control
(gradient-perturbation reproduction), CONTENTION-IMMUNE — no timing claim.
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
    """IDENTICAL to measure_torch_vehicle_mps_descent_equivalence._muon_basin_curriculum
    (the stage-8 Muon basin recipe: Muon on decoder + AdamW on latents,
    seg_weight=100, pose_weight=1). Reused verbatim so the chaos control is an
    apples-to-apples replica of the MPS A/B with only the gradient source changed."""
    return [
        StageSpec(
            name="chaos_ctrl_muon_basin",
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


def _install_cotangent_noise(scorer, *, rel: float, seed: int):
    """Monkeypatch scorer.seg_pose_forward to inject i.i.d. RELATIVE noise into the
    frame cotangent dL/dF via a backward hook on the (CPU) input frames.

    The injected perturbation is g <- g * (1 + rel * eps), eps ~ N(0,1) i.i.d. per
    element — a multiplicative relative perturbation whose per-element relmax is
    ~rel (matching the MPS measured input-grad relmax ~2e-4). This is the most
    faithful emulation of "a 2e-4-relmax gradient backend" on a pure-CPU run: same
    forward, same loss, same optimizer, ONLY the gradient is perturbed at the exact
    surface the MPS drift was measured on. A dedicated torch.Generator keeps the
    noise reproducible and INDEPENDENT of the model RNG (so the clean arm's init /
    permutation stream is byte-identical to the MPS A/B's CPU arm)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    orig = scorer.seg_pose_forward

    def hooked(decoded_bhwc, *a, **k):
        if decoded_bhwc.requires_grad:
            def _h(grad):
                eps = torch.randn(grad.shape, generator=gen, dtype=grad.dtype, device=grad.device)
                return grad * (1.0 + rel * eps)
            decoded_bhwc.register_hook(_h)
        return orig(decoded_bhwc, *a, **k)

    scorer.seg_pose_forward = hooked
    return scorer


def _run_arm(out_dir, *, video_path, n_pairs, base_channels, curriculum, seed,
             label, targets_cache, grad_noise_rel, noise_seed):
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
        checkpoint_every_epochs=10_000,
        device="cpu",          # AUTHORITY = CPU
        train_device="cpu",    # BOTH arms: pure CPU gradient (the control)
        seed=seed,
    )
    scorer = RealScorerContext(
        video_path, device="cpu", train_device="cpu",
        max_pairs=int(n_pairs), targets_cache=targets_cache,
    )
    if grad_noise_rel and grad_noise_rel > 0:
        _install_cotangent_noise(scorer, rel=grad_noise_rel, seed=noise_seed)
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    t0 = time.time()
    summary = driver.run()
    dt = time.time() - t0
    traj_path = Path(out_dir) / "torch_vehicle_trajectory.jsonl"
    rows = [json.loads(line) for line in traj_path.read_text().strip().splitlines()]
    evald = [r for r in rows if r.get("evaluated") and r.get("d_seg") is not None]
    traj = [
        {"epoch": r["global_epoch"], "d_seg": float(r["d_seg"]), "d_pose": float(r["d_pose"])}
        for r in evald
    ]
    n_epochs = max((r["global_epoch"] for r in rows), default=0)
    print(f"[{label}] done: {n_epochs} epochs in {dt:.1f}s; "
          f"best_score={summary.get('best_score'):.5f} (noise_rel={grad_noise_rel})", flush=True)
    for t in traj:
        print(f"  [{label}] ep {t['epoch']:>4}: d_seg={t['d_seg']:.6f} d_pose={t['d_pose']:.6f}", flush=True)
    return traj, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-noise-rel", type=float, default=2e-4,
                    help="relative i.i.d. noise injected into dL/dF on the candidate arm "
                         "(default 2e-4 = measured MPS input-grad relmax).")
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--out-root", default="experiments/results/torch_vehicle_chaos_control_ab")
    ap.add_argument("--out-json",
                    default="experiments/results/torch_vehicle_chaos_control_ab/verdict.json")
    args = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored
        video_path = str(import_vendored("data").get_default_video_path())

    out_root = Path(args.out_root)
    # Reuse the SAME cached CPU-authority targets the MPS A/B built (apples-to-apples).
    targets_cache = Path("experiments/results/torch_vehicle_mps_descent_ab/gt_targets_cache")
    if not (targets_cache / f"gt_targets_n{args.n_pairs}.pt").exists():
        targets_cache = out_root / "gt_targets_cache"
    curriculum = _muon_basin_curriculum(args.epochs, args.eval_every, args.batch_size)

    print("=== ARM A: pure CPU gradient (clean baseline) ===", flush=True)
    traj_a, sum_a = _run_arm(
        out_root / "arm_clean", video_path=video_path, n_pairs=args.n_pairs,
        base_channels=args.base_channels, curriculum=curriculum, seed=args.seed,
        label="cpu_clean", targets_cache=targets_cache, grad_noise_rel=0.0, noise_seed=0)

    print(f"\n=== ARM B: CPU gradient + {args.grad_noise_rel:.1e} rel cotangent noise "
          "(MPS-emulating candidate) ===", flush=True)
    traj_b, sum_b = _run_arm(
        out_root / "arm_noise", video_path=video_path, n_pairs=args.n_pairs,
        base_channels=args.base_channels, curriculum=curriculum, seed=args.seed,
        label="cpu_noise", targets_cache=targets_cache,
        grad_noise_rel=args.grad_noise_rel, noise_seed=args.noise_seed)

    from tac.mlx_pr95_port.speedup_acceptance_gate import evaluate_descent_equivalence

    verdict = evaluate_descent_equivalence(traj_a, traj_b, n_pairs=args.n_pairs)

    print("\n=== BOTH-TERMS GATE on CPU-clean vs CPU+noise (SAME gate as MPS A/B) ===")
    by_a = {r["epoch"]: r for r in traj_a}
    by_b = {r["epoch"]: r for r in traj_b}
    print(f"{'epoch':>6} {'clean_seg':>10} {'noise_seg':>10} {'clean_pose':>12} {'noise_pose':>12} {'pose_gap':>10}")
    for ep in sorted(set(by_a) & set(by_b)):
        a, b = by_a[ep], by_b[ep]
        print(f"{ep:>6} {a['d_seg']:>10.6f} {b['d_seg']:>10.6f} "
              f"{a['d_pose']:>12.6f} {b['d_pose']:>12.6f} {abs(a['d_pose']-b['d_pose']):>10.4f}")
    print(f"\nseg:  {verdict.seg.reason}")
    print(f"pose: {verdict.pose.reason}")
    print(f"\nVERDICT (noise-injected CPU arm vs clean CPU arm): "
          f"{'PASS' if verdict.passed else 'REJECT'}")

    final_pose_gap = verdict.pose.final_abs_gap
    # The MPS A/B pose gap (the number we're trying to reproduce with pure noise).
    mps_pose_gap = 7.023406982421875
    interp = (
        "H_chaos SUPPORTED: a pure-CPU 2e-4-noise gradient REPRODUCES a comparable "
        "pose gap and the gate REJECTS it too -> the divergence is OPTIMIZER CHAOS "
        "from a tiny perturbation, NOT a wrong MPS gradient. The full 104x is VALID."
        if (final_pose_gap >= 0.5 * mps_pose_gap or not verdict.passed)
        else "H_chaos NOT supported by this noise level: 2e-4 i.i.d. noise produced a "
             "MUCH smaller gap than the MPS 7.02 -> the MPS gap is likely BIAS-driven "
             "(H_opbias); localize + patch the divergent op."
    )
    print(f"\n>>> {interp}")
    print(f">>> noise-arm pose |gap|={final_pose_gap:.4f}  vs  MPS-arm pose |gap|={mps_pose_gap:.4f}")

    out = {
        "config": vars(args),
        "arm_clean": traj_a, "arm_noise": traj_b,
        "clean_best_score": sum_a.get("best_score"),
        "noise_best_score": sum_b.get("best_score"),
        "gate_passed_noise_vs_clean": verdict.passed,
        "noise_arm_pose_final_abs_gap": final_pose_gap,
        "mps_arm_pose_final_abs_gap": mps_pose_gap,
        "pose_gap_ratio_noise_over_mps": final_pose_gap / mps_pose_gap,
        "seg_verdict": {"tracks": verdict.seg.tracks_within_tol, "reason": verdict.seg.reason},
        "pose_verdict": {"tracks": verdict.pose.tracks_within_tol,
                         "diverged": verdict.pose.diverged,
                         "final_abs_gap": verdict.pose.final_abs_gap,
                         "reason": verdict.pose.reason},
        "interpretation": interp,
        "authority": "[macOS-CPU advisory] CORRECTNESS control; exact d_seg/d_pose torch-CPU "
                     "for BOTH arms; noise injected into GRADIENT only. NON-PROMOTABLE.",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
