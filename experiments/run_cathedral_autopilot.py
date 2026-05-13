"""Cathedral autopilot — beautiful end-to-end Pareto sweep on synthetic substrate.

Demonstrates that the cathedral actually OPTIMIZES (not just measures):

    contest_rate_distortion_system  →  the objective S = 100·d + sqrt(10·p) + 25·B/N
    shannon_h2_loss                 →  the differentiable rate proxy R(θ) = H₀(θ)·n/8
    soft-Lagrangian (rate ≤ R_max)  →  gradient descent on θ + dual ascent on λ
    importance-flip threshold       →  pose-marginal divergence at d_pose < 2.5e-4

The autopilot runs a Pareto sweep over R_max, solving each inner problem
to near-KKT, and reports the (rate, distortion, score) trajectory.

Strict-scorer-rule: pure CPU + torch. No scorer load. The "distortion"
function is a closed-form MSE-to-reference; this is the canonical CPU
sanity-loop that validates the optimization machinery BEFORE wiring the
actual contest scorer (which lives only on the contest-CUDA dispatch
path).

Cross-references:

- :mod:`tac.contest_rate_distortion_system` — contest formula + marginals
- :mod:`tac.shannon_h2_loss` — differentiable H₀
- :mod:`tac.joint_admm_coordinator` — Boyd 2011 §3.4 (production ADMM)
- ``feedback_automate_and_densify_intelligence_20260507`` — the operating
  preference behind this module
"""

import json
import pathlib
from dataclasses import asdict, dataclass

import torch

from tac.contest_rate_distortion_system import (
    contest_score,
    contest_score_decomposition,
    contest_score_marginals,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
from tac.shannon_h2_loss import shannon_h0_loss


# ---------------------------------------------------------------------------
# Beautiful dataclass for one Pareto point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierPoint:
    """One (rate, distortion, score) point on the empirical Pareto frontier."""
    R_target_bytes: float
    rate_bytes: float
    distortion_seg: float
    distortion_pose: float
    score: float
    lambda_rate: float
    n_inner_steps: int
    converged: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The autopilot
# ---------------------------------------------------------------------------

def synthetic_substrate(seed: int = 0, scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Build a leaf state_dict shaped per FIXED_STATE_SCHEMA, requires_grad=True."""
    g = torch.Generator().manual_seed(seed)
    return {
        name: (torch.randn(*shape, generator=g) * scale).requires_grad_(True)
        for name, shape in FIXED_STATE_SCHEMA
    }


def reference_substrate(seed: int = 1, scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Same shape, different seed — the "ideal" weights the optimizer should approach."""
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


def total_rate_h0(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Differentiable rate proxy in BYTES via per-tensor H₀."""
    bits = torch.tensor(0.0)
    for w in weights.values():
        bits = bits + shannon_h0_loss(w) * w.numel()
    return bits / 8.0


def mse_distortion(
    weights: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (seg_proxy, pose_proxy) — both MSE proxies for sanity-loop math.

    First half of tensors → seg-proxy; second half → pose-proxy. This is a
    pure didactic split; production wires real SegNet / PoseNet at the
    contest-CUDA dispatch path (forbidden in this CPU module per
    strict-scorer-rule).
    """
    names = list(weights.keys())
    half = len(names) // 2
    seg = torch.tensor(0.0)
    pose = torch.tensor(0.0)
    for name in names[:half]:
        seg = seg + (weights[name] - reference[name]).pow(2).mean()
    for name in names[half:]:
        pose = pose + (weights[name] - reference[name]).pow(2).mean()
    return seg / max(half, 1), pose / max(len(names) - half, 1)


def solve_rate_constrained_inner(
    weights: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    R_target_bytes: float,
    n_steps: int = 80,
    primal_lr: float = 1e-2,
    dual_lr: float = 1e-3,
    kkt_eps: float = 1e-3,
) -> FrontierPoint:
    """Solve min D(θ) s.t. R(θ) ≤ R_target via soft-Lagrangian + dual ascent.

    Returns a FrontierPoint with the final (rate, seg_distortion,
    pose_distortion, score, λ, n_steps, converged) values.
    """
    optim = torch.optim.SGD(weights.values(), lr=primal_lr)
    lam = torch.tensor(0.0)
    converged = False
    steps_taken = 0
    for step in range(n_steps):
        optim.zero_grad()
        seg, pose = mse_distortion(weights, reference)
        rate = total_rate_h0(weights)
        violation = (rate - R_target_bytes).clamp_min(0.0)
        # The Lagrangian: distortion (sum of seg + sqrt(pose) per contest formula)
        # + λ · rate violation. We use the actual contest_score on (seg, pose, R)
        # for the distortion side so the optimizer learns the sqrt-coupled regime.
        L = contest_score(
            seg_distortion=seg,
            pose_distortion=pose,
            archive_bytes=rate,
        ) + lam * violation
        L.backward()
        optim.step()
        # Dual ascent
        with torch.no_grad():
            r = float(total_rate_h0(weights))
            lam_new = float(lam) + dual_lr * (r - R_target_bytes)
            lam = torch.tensor(max(0.0, lam_new))
        steps_taken = step + 1
        # KKT check
        with torch.no_grad():
            primal_violation = max(0.0, float(total_rate_h0(weights)) - R_target_bytes)
            comp = float(lam) * primal_violation
        if primal_violation < kkt_eps and comp < kkt_eps:
            converged = True
            break
    with torch.no_grad():
        seg, pose = mse_distortion(weights, reference)
        rate = float(total_rate_h0(weights))
        score = float(contest_score(
            seg_distortion=seg,
            pose_distortion=pose,
            archive_bytes=rate,
        ))
    return FrontierPoint(
        R_target_bytes=float(R_target_bytes),
        rate_bytes=rate,
        distortion_seg=float(seg),
        distortion_pose=float(pose),
        score=score,
        lambda_rate=float(lam),
        n_inner_steps=steps_taken,
        converged=converged,
    )


def trace_pareto_frontier(
    *,
    R_targets: list[float],
    seed: int = 0,
    n_inner_steps: int = 80,
) -> list[FrontierPoint]:
    """Solve the rate-distortion problem at each R_target and collect points."""
    reference = reference_substrate(seed=seed + 1)
    points: list[FrontierPoint] = []
    for R_t in R_targets:
        # Fresh weight init for each R_target — avoids bias from prior solve.
        weights = synthetic_substrate(seed=seed)
        pt = solve_rate_constrained_inner(
            weights, reference, R_target_bytes=R_t, n_steps=n_inner_steps,
        )
        points.append(pt)
    return points


def summarize_frontier(points: list[FrontierPoint]) -> str:
    """Render a beautiful markdown table of the swept frontier."""
    lines = [
        "| R_target (B) | rate (B) | seg | pose | score | λ | steps | KKT |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in points:
        lines.append(
            f"| {p.R_target_bytes:,.0f} | {p.rate_bytes:,.0f} | "
            f"{p.distortion_seg:.6f} | {p.distortion_pose:.6f} | "
            f"{p.score:.6f} | {p.lambda_rate:.4f} | {p.n_inner_steps} | "
            f"{'GREEN' if p.converged else 'budget'} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Cathedral autopilot: end-to-end Pareto sweep")
    p.add_argument("--R-min", type=int, default=20_000, help="smallest rate target")
    p.add_argument("--R-max", type=int, default=200_000, help="largest rate target")
    p.add_argument("--n-points", type=int, default=8, help="grid size for Pareto sweep")
    p.add_argument("--n-inner-steps", type=int, default=80, help="primal optimization steps per λ")
    p.add_argument("--seed", type=int, default=0, help="substrate seed")
    p.add_argument("--output-dir", default=None, help="output dir; defaults to lane_cathedral_autopilot_<UTC>/")
    args = p.parse_args(argv)

    targets = [
        args.R_min + (args.R_max - args.R_min) * k / max(1, args.n_points - 1)
        for k in range(args.n_points)
    ]
    points = trace_pareto_frontier(
        R_targets=targets, seed=args.seed, n_inner_steps=args.n_inner_steps,
    )

    print("Cathedral autopilot — Pareto frontier sweep on synthetic substrate")
    print()
    print(summarize_frontier(points))

    # Persist manifest
    import datetime as _dt
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else (
        pathlib.Path(f"experiments/results/lane_cathedral_autopilot_{ts}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "experiments/run_cathedral_autopilot",
        "evidence_grade": "[empirical]",
        "score_claim": False,
        "seed": args.seed,
        "n_inner_steps": args.n_inner_steps,
        "R_targets": targets,
        "frontier": [p.as_dict() for p in points],
    }
    (out_dir / "frontier.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {out_dir / 'frontier.json'}")

    # Best valid (R ≤ R_max) point by score
    valid = [p for p in points if p.rate_bytes <= args.R_max]
    if valid:
        best = min(valid, key=lambda p: p.score)
        print(
            f"\nbest valid point: R={best.rate_bytes:,.0f} B, "
            f"score={best.score:.6f}, R_target={best.R_target_bytes:,.0f}, "
            f"converged={best.converged}"
        )
        marg = contest_score_marginals(
            seg_distortion=best.distortion_seg,
            pose_distortion=best.distortion_pose,
            archive_bytes=best.rate_bytes,
        )
        dec = contest_score_decomposition(
            seg_distortion=best.distortion_seg,
            pose_distortion=best.distortion_pose,
            archive_bytes=best.rate_bytes,
        )
        print(f"  marginals: ∂S/∂seg={marg['dS_dseg']:.2f}  ∂S/∂pose={marg['dS_dpose']:.2f}  ∂S/∂B={marg['dS_dbytes']:.4e}")
        print(f"  decomposition: seg={dec['seg_share']*100:.1f}%  pose={dec['pose_share']*100:.1f}%  rate={dec['rate_share']*100:.1f}%")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
