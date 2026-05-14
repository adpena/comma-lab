# SPDX-License-Identifier: MIT
"""Probe: SE(3) parametric vs optical-flow non-parametric motion for D4.

Per Catalog #125 hook #6 + the design-tension memo
``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md``:
when a design choice has 2+ defensible interpretations, ship BOTH modes via
callable interface + build a probe that returns the regime-conditional
verdict. The probe IS the arbitration; the trainer/codec/solver consumes
the verdict.

For D4 the two interpretations are:

1. **SE(3) parametric motion**: per-pair 6 floats (3 translation +
   3 axis-angle rotation). Best for ego-motion-dominated dashcam scenes.
2. **Optical-flow non-parametric motion**: per-pair coarse (u, v) flow
   field. Best for parallax-heavy scenes (vehicles, lane changes).

The probe fits BOTH motion models on the SAME contest pair set, measures
the photometric residual energy after motion compensation, and emits a
typed verdict consumed by the D4 trainer's mode-selection logic.

Verdict schema (JSON output)::

    {
      "se3_parametric": {
        "residual_l2": float,
        "motion_byte_cost_est_bytes": int,
        "fit_iterations": int,
        "fit_wallclock_sec": float
      },
      "optical_flow": {
        "residual_l2": float,
        "motion_byte_cost_est_bytes": int,
        "fit_iterations": int,
        "fit_wallclock_sec": float
      },
      "verdict": "se3_parametric" | "optical_flow" | "tie",
      "verdict_rationale": str,
      "evidence_grade": "proxy",
      "score_claim_valid": false,
      "ready_for_exact_eval_dispatch": false
    }

Usage::

    .venv/bin/python tools/probe_d4_motion_model_disambiguator.py \\
        --num-pairs 32 --epochs 200 \\
        --output reports/raw/d4_probe_<utc>.json

The probe runs in a few seconds on CPU and is suitable for free macOS
smoke. The verdict is `[proxy]`-tagged per CLAUDE.md axis-discipline; a
production dispatch consults the probe's verdict but treats it as a
prior, not as a score claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.substrates.d4_wyner_ziv_frame_0 import (
    MotionModelMode,
    WynerZivFrame0Config,
    WynerZivFrame0Substrate,
)


def _synthesize_pair(num_pairs: int, scene_type: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a synthetic frame pair for probe testing.

    scene_type:
        ``ego_motion_dominated``: small global translation + rotation; favors SE(3).
        ``parallax_heavy``: per-pixel-varying displacement; favors optical flow.
        ``mixed``: mixture of both.
    """
    gen = torch.Generator().manual_seed(seed)
    h, w = 24, 32
    frame_1 = torch.rand(num_pairs, 3, h, w, generator=gen)
    if scene_type == "ego_motion_dominated":
        # Apply small global translation
        shift = (torch.randn(num_pairs, 2, generator=gen) * 0.5).round().long()
        frame_0 = torch.empty_like(frame_1)
        for i in range(num_pairs):
            dx, dy = int(shift[i, 0]), int(shift[i, 1])
            frame_0[i] = torch.roll(frame_1[i], shifts=(dy, dx), dims=(1, 2))
    elif scene_type == "parallax_heavy":
        # Per-pixel random small displacement (simulates parallax)
        noise = torch.randn(num_pairs, 3, h, w, generator=gen) * 0.05
        frame_0 = (frame_1 + noise).clamp(0.0, 1.0)
    else:  # mixed
        frame_0 = (frame_1 + torch.randn_like(frame_1) * 0.02).clamp(0.0, 1.0)
    return frame_0, frame_1


def _fit_motion(
    mode: MotionModelMode,
    frame_0_target: torch.Tensor,
    frame_1: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> tuple[float, int, float]:
    """Fit one motion model and return (final_residual_l2, iterations, sec)."""
    num_pairs = frame_0_target.shape[0]
    cfg = WynerZivFrame0Config(
        motion_mode=mode,
        num_pairs=num_pairs,
        output_height=frame_0_target.shape[-2],
        output_width=frame_0_target.shape[-1],
        flow_grid_h=6,
        flow_grid_w=8,
        residual_coarse_h=6,
        residual_coarse_w=8,
    )
    sub = WynerZivFrame0Substrate(cfg)
    # Reset residual to zero so the probe measures motion-only fit quality
    # (D4 trains motion + residual jointly; the probe separates the two by
    # holding residual at zero).
    with torch.no_grad():
        sub.residual_coarse.zero_()
    # Freeze residual; only train motion params.
    sub.residual_coarse.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in sub.parameters() if p.requires_grad], lr=lr
    )
    t0 = time.time()
    final_loss = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        f0_pred, _ = sub.reconstruct_pair(frame_1)
        loss = (f0_pred - frame_0_target).pow(2).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    sec = time.time() - t0
    return final_loss, epochs, sec


def _byte_cost_estimate(mode: MotionModelMode, num_pairs: int) -> int:
    """Estimate motion-section byte cost after brotli (deep-math memo §3.5)."""
    if mode == MotionModelMode.SE3_PARAMETRIC:
        # 6 fp16 per pair = 12 B/pair raw; brotli typically closes to
        # ~50-60% on small float arrays = 6-8 B/pair.
        return int(num_pairs * 7)
    else:
        # 2 * 12 * 16 = 384 fp16 per pair = 768 B/pair raw; brotli closes
        # to ~10-20% on highly-correlated flow fields = 80-150 B/pair.
        return int(num_pairs * 120)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="probe_d4_motion_model_disambiguator")
    p.add_argument("--num-pairs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scene-type",
        type=str,
        default="mixed",
        choices=["ego_motion_dominated", "parallax_heavy", "mixed"],
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    frame_0_target, frame_1 = _synthesize_pair(
        args.num_pairs, args.scene_type, args.seed
    )

    # Fit both modes.
    se3_l2, se3_iters, se3_sec = _fit_motion(
        MotionModelMode.SE3_PARAMETRIC,
        frame_0_target,
        frame_1,
        epochs=args.epochs,
        lr=args.lr,
    )
    flow_l2, flow_iters, flow_sec = _fit_motion(
        MotionModelMode.OPTICAL_FLOW,
        frame_0_target,
        frame_1,
        epochs=args.epochs,
        lr=args.lr,
    )

    se3_bytes = _byte_cost_estimate(MotionModelMode.SE3_PARAMETRIC, args.num_pairs)
    flow_bytes = _byte_cost_estimate(MotionModelMode.OPTICAL_FLOW, args.num_pairs)

    # Verdict: pick the mode with smallest (residual_l2 * byte_cost) product.
    # This is a rough proxy for "which mode reaches the same fidelity for
    # fewer bytes" — i.e. the EV/byte metric. The full Lagrangian (residual
    # bytes + motion bytes + scorer terms) is computed only in the trainer.
    se3_metric = se3_l2 * se3_bytes
    flow_metric = flow_l2 * flow_bytes
    if abs(se3_metric - flow_metric) / max(se3_metric, flow_metric, 1e-12) < 0.05:
        verdict = "tie"
        rationale = (
            f"within 5% (se3_metric={se3_metric:.3e} vs "
            f"flow_metric={flow_metric:.3e}); use SE(3) default by Occam"
        )
    elif se3_metric < flow_metric:
        verdict = "se3_parametric"
        rationale = (
            f"SE(3) wins ev/byte: se3_metric={se3_metric:.3e} < "
            f"flow_metric={flow_metric:.3e}"
        )
    else:
        verdict = "optical_flow"
        rationale = (
            f"optical-flow wins ev/byte: flow_metric={flow_metric:.3e} < "
            f"se3_metric={se3_metric:.3e}"
        )

    payload = {
        "scene_type": args.scene_type,
        "num_pairs": args.num_pairs,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "se3_parametric": {
            "residual_l2": se3_l2,
            "motion_byte_cost_est_bytes": se3_bytes,
            "fit_iterations": se3_iters,
            "fit_wallclock_sec": se3_sec,
        },
        "optical_flow": {
            "residual_l2": flow_l2,
            "motion_byte_cost_est_bytes": flow_bytes,
            "fit_iterations": flow_iters,
            "fit_wallclock_sec": flow_sec,
        },
        "verdict": verdict,
        "verdict_rationale": rationale,
        "evidence_grade": "proxy",
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "lane_id": "lane_d4_wyner_ziv_frame_0_substrate_20260514",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[d4-probe] verdict={verdict} se3_l2={se3_l2:.6f} flow_l2={flow_l2:.6f}")
    print(f"[d4-probe] rationale: {rationale}")
    print(f"[d4-probe] output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
