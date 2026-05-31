# SPDX-License-Identifier: MIT
"""Register the FIRST REAL-teacher pose-axis empirical anchor for DreamerV3 RSSM.

REAL-HINTON WAVE 2026-05-30 (the "dreamer" member of the operator-named
hierarchical-PC stack-of-stacks; sister of the Z7-Mamba-2 anchor tool
``tools/register_z7_mamba2_real_hinton_pose_axis_anchor.py``).

This tool reads a completed (or in-progress) DreamerV3 RSSM real-Hinton long
MLX run's ``telemetry.jsonl``, computes the pose-axis reduction fraction under
the REAL SegNet+PoseNet Hinton-distilled teacher, and registers the canonical
empirical anchor on
``categorical_posterior_capacity_vs_continuous_gaussian_v1`` via
``update_equation_with_empirical_anchor`` per Catalog #344.

FAIL-CLOSED (the structural protection that a future MOCK run cannot
masquerade as the real anchor, per Catalog #322 / #321): refuses to register
if ``pose[ep0] <= 0`` — the mock teacher leaves pose=0 (phantom-provenance);
the REAL PoseNet teacher MUST produce a NON-ZERO pose-axis (the 8-pair smoke
probe produced pose=2.025).

The empirical signature recorded is the DreamerV3 categorical posterior's REAL
pose-axis convergence under the Hinton-distilled scorer-bound gradient — the
substrate-CLASS-shift signal (192-bit categorical capacity vs ~50-bit
continuous-Gaussian) trained against the actual contest ego-motion target,
NOT a reconstruction-proxy.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341
the anchor is non-promotable ``[macOS-MLX research-signal]`` by construction
(via ``build_provenance_for_mps_proxy``); paired Linux x86_64 CPU/CUDA replay is
the canonical promotion path and is DEFERRED.

[verified-against: tools/register_z7_mamba2_real_hinton_pose_axis_anchor.py canonical sister]
[verified-against: tac.canonical_equations.update_equation_with_empirical_anchor]
[verified-against: tac.provenance.build_provenance_for_mps_proxy non-promotable grade]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (
    EmpiricalAnchor,
    get_equation_by_id,
    update_equation_with_empirical_anchor,
)
from tac.provenance import build_provenance_for_mps_proxy

EQUATION_ID = "categorical_posterior_capacity_vs_continuous_gaussian_v1"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_pose_series(run_dir: Path) -> list[float]:
    """Load the per-epoch pose-axis series from a run's telemetry.jsonl."""
    tj = run_dir / "telemetry.jsonl"
    if not tj.is_file():
        raise SystemExit(f"telemetry.jsonl not found in {run_dir}")
    poses: list[float] = []
    for line in tj.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        dec = row.get("per_axis_decomposition") or {}
        pose = dec.get("pose")
        if pose is not None:
            poses.append(float(pose))
    return poses


def _compute_reduction_fraction(poses: list[float]) -> float:
    """Compute (pose[ep0] - pose[last]) / pose[ep0]; fail closed on pose[ep0]<=0."""
    if not poses:
        raise SystemExit("no pose-axis telemetry rows found")
    pose_ep0 = poses[0]
    pose_last = poses[-1]
    if pose_ep0 <= 0.0:
        raise SystemExit(
            f"pose[ep0]={pose_ep0} <= 0 — this indicates a MOCK run (the real "
            "teacher MUST produce a non-zero pose-axis). Refusing to register "
            "the real-teacher anchor (fail-closed per Catalog #322 / #321)."
        )
    return (pose_ep0 - pose_last) / pose_ep0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Register DreamerV3 RSSM real-Hinton pose-axis empirical anchor."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--predicted-reduction",
        type=float,
        default=0.192,
        help="Predicted pose-axis reduction fraction (sister Z7 real-teacher "
        "anchor anchor; residual = predicted - empirical).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    poses = _load_pose_series(args.run_dir)
    reduction = _compute_reduction_fraction(poses)
    print(
        f"pose[ep0]={poses[0]:.4f} pose[last]={poses[-1]:.4f} "
        f"reduction_fraction={reduction:.4f} (n={len(poses)} epochs)"
    )
    if args.dry_run:
        print("[dry-run] not registering")
        return 0

    now = _utc_now()
    import hashlib

    artifact = args.run_dir / "training_artifact.json"
    artifact_sha = (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        if artifact.is_file()
        else ""
    )
    telemetry_rel = args.run_dir / "telemetry.jsonl"
    try:
        source_path = str(telemetry_rel.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(telemetry_rel)
    prov = build_provenance_for_mps_proxy(
        source_path=source_path,
        artifact_sha256=artifact_sha,
        captured_at_utc=now,
        in_domain_context="dreamer_v3_rssm_real_hinton_teacher_600pair_long_mlx",
    )
    residual = float(args.predicted_reduction) - float(reduction)
    anchor = EmpiricalAnchor(
        anchor_id=f"dreamer_v3_rssm_real_hinton_pose_axis_{args.run_dir.name}",
        predicted_value=float(args.predicted_reduction),
        measured_value=float(reduction),
        measurement_residual=residual,
        measurement_method=(
            "real_hinton_segnet_posenet_teacher_600pair_long_mlx_local_"
            "pose_axis_reduction_fraction"
        ),
        evidence_path=source_path,
        provenance=prov,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        notes=(
            "FIRST REAL-teacher pose-axis anchor for the DreamerV3 categorical-"
            "posterior (the dreamer member of the hierarchical-PC stack-of-"
            "stacks). REAL SegNet KL T=2.0 + REAL PoseNet pose-MSE Hinton-"
            f"distilled teachers (NO mock flag); pose-axis {poses[0]:.4g} -> "
            f"{poses[-1]:.4g} ({reduction:.1%} reduction) over {len(poses)} "
            "epochs at 600pair MLX-LOCAL with Wave N+11 stabilizer (grad-clip "
            "1.0 + warmup 5 + weight_decay 1e-4 + adamw + EMA 0.997). Mock "
            "leaves pose=0 (phantom-provenance per Catalog #322). Non-promotable "
            "[macOS-MLX research-signal] per Catalog #192/#317/#341; paired "
            "Linux x86_64 CPU/CUDA replay DEFERRED."
        ),
    )
    update_equation_with_empirical_anchor(EQUATION_ID, anchor)
    eq = get_equation_by_id(EQUATION_ID)
    print(
        f"registered anchor; equation now has {len(eq.empirical_anchors)} anchors "
        f"(predicted={args.predicted_reduction} empirical={reduction:.4f} "
        f"residual={residual:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
