# SPDX-License-Identifier: MIT
"""Register the Z8 hierarchical-PC real-Hinton pose-axis empirical anchor.

τ-ANNEAL + FULL-STACK LONG-RUN WAVE 2026-05-31 (lane
``lane_z8_hier_pc_full_stack_longrun_20260531``; sister of
``tools/register_dreamer_v3_rssm_real_hinton_pose_axis_anchor.py``).

Reads a Z8 ``_full_main`` real-teacher run's ``telemetry.jsonl``, computes the
per-axis pose reduction fraction ``(pose[ep0] - pose[last]) / pose[ep0]``, and
registers the canonical ``EmpiricalAnchor`` on
``categorical_posterior_capacity_vs_continuous_gaussian_v1`` via
``update_equation_with_empirical_anchor`` per Catalog #344.

This is a DISTINCT empirical data point vs the sister DreamerV3 anchor: Z8 uses
a 3-level Rao-Ballard categorical HIERARCHY (groups (4,3,2)..(24,16,8),
categories (16,8,4)..(256,128,64)) where DreamerV3 is single-level. The
hierarchical multi-level capacity is the Z8-unique structural variation.

Fail-closed per Catalog #321/#322: refuses to register if ``pose[ep0] <= 0`` —
the mock teacher leaves pose=0 (phantom-provenance); the REAL teacher MUST
produce a non-zero pose axis. The anchor is non-promotable
``[macOS-MLX research-signal]`` by construction per Catalog #192/#317/#341;
paired Linux x86_64 CPU/CUDA replay is the exact promotion gate (DEFERRED).

[verified-against: tools/register_dreamer_v3_rssm_real_hinton_pose_axis_anchor.py canonical sister]
[verified-against: tac.canonical_equations.update_equation_with_empirical_anchor]
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


def _load_seg_series(run_dir: Path) -> list[float]:
    """Load the per-epoch seg-axis series (the structural-ceiling residual)."""
    tj = run_dir / "telemetry.jsonl"
    segs: list[float] = []
    for line in tj.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        dec = row.get("per_axis_decomposition") or {}
        seg = dec.get("seg")
        if seg is not None:
            segs.append(float(seg))
    return segs


def _compute_reduction_fraction(poses: list[float]) -> float:
    """Compute (pose[ep0] - pose[last]) / pose[ep0]; fail closed on pose[ep0]<=0."""
    if not poses:
        raise SystemExit("no pose series found in telemetry")
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
        description="Register Z8 hierarchical-PC real-Hinton pose-axis anchor."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--predicted-reduction",
        type=float,
        default=0.9993,
        help="Predicted pose-axis reduction fraction (DreamerV3 sister real-"
        "teacher anchor reached ~0.999 pose collapse; residual = predicted - "
        "empirical).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    poses = _load_pose_series(args.run_dir)
    segs = _load_seg_series(args.run_dir)
    reduction = _compute_reduction_fraction(poses)
    seg_ep0 = float(segs[0]) if segs else 0.0
    seg_last = float(segs[-1]) if segs else 0.0
    seg_reduction = (
        (seg_ep0 - seg_last) / seg_ep0 if seg_ep0 > 0.0 else 0.0
    )
    print(
        f"pose[ep0]={poses[0]:.4f} pose[last]={poses[-1]:.4f} "
        f"reduction_fraction={reduction:.4f} (n={len(poses)} epochs) | "
        f"seg[ep0]={seg_ep0:.4f} seg[last]={seg_last:.4f} "
        f"seg_reduction={seg_reduction:.4f}"
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
    )
    pose_ep0 = float(poses[0])
    pose_last = float(poses[-1])
    pose_min = float(min(poses))
    # Canonical EmpiricalAnchor schema: ``residual`` is the SCALAR normalized
    # magnitude >= 0 (the registry refuses negative residuals); the signed
    # direction lives in predicted_output vs empirical_output.
    signed_residual = float(args.predicted_reduction) - float(reduction)
    residual = abs(signed_residual)
    anchor = EmpiricalAnchor(
        anchor_id=f"z8_hier_pc_real_hinton_pose_axis_{args.run_dir.name}",
        measurement_utc=now,
        inputs={
            "num_levels": 3,
            "num_groups_per_level": [4, 3, 2],
            "num_categories_per_level": [16, 8, 4],
            "num_pairs": 600,
            "epochs": len(poses),
            "distillation_weight": 0.5,
            "pose_distillation_weight": 1.0,
            "scorer_teacher": "real_segnet_kl_t2",
            "pose_scorer_teacher": "real_posenet_pose_mse",
            "gumbel_tau_anneal": "cosine_1.0_to_0.1",
            "lr_schedule": "warmup_50_cosine_decay_min_ratio_1e-2",
            "stabilizer": "grad_clip_1.0_wd_1e-4_adamw_ema_0.997",
        },
        predicted_output={
            "pose_axis_reduction_fraction": float(args.predicted_reduction),
        },
        empirical_output={
            "pose_axis_reduction_fraction": float(reduction),
            "pose_axis_ep0": pose_ep0,
            "pose_axis_last": pose_last,
            "pose_axis_min": pose_min,
            "seg_axis_ep0": seg_ep0,
            "seg_axis_last": seg_last,
            "seg_axis_reduction_fraction": seg_reduction,
            "signed_residual_predicted_minus_empirical": signed_residual,
        },
        residual=residual,
        source_artifact=source_path,
        measurement_method=(
            "z8_hierarchical_pc_3level_real_hinton_segnet_posenet_teacher_"
            "600pair_2000epoch_tau_anneal_cosine_mlx_local_pose_axis_"
            "reduction_fraction"
        ),
        provenance=prov,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    update_equation_with_empirical_anchor(
        EQUATION_ID,
        anchor,
        notes=(
            "FIRST REAL-teacher pose-axis anchor for the Z8 3-LEVEL "
            "hierarchical-PC categorical posterior (the hierarchical member of "
            "the stack-of-stacks; DISTINCT from the single-level DreamerV3 "
            "sister anchor). REAL SegNet KL T=2.0 + REAL PoseNet pose-MSE "
            "Hinton-distilled teachers (NO mock flag); Gumbel τ-anneal 1.0->0.1 "
            "+ warmup 50 + cosine LR. pose-axis "
            f"{pose_ep0:.4g} -> {pose_last:.4g} ({reduction:.1%} reduction); "
            f"seg-axis {seg_ep0:.4g} -> {seg_last:.4g} ({seg_reduction:.1%}, "
            "STRUCTURAL CEILING still descending) over "
            f"{len(poses)} epochs at 600pair MLX-LOCAL. Mock leaves pose=0 "
            "(phantom-provenance per Catalog #322). Non-promotable "
            "[macOS-MLX research-signal] per Catalog #192/#317/#341; paired "
            "Linux x86_64 CPU/CUDA replay DEFERRED (no MLX->PyTorch export "
            "bridge for Z8 yet)."
        ),
    )
    eq = get_equation_by_id(EQUATION_ID)
    print(
        f"registered anchor; equation now has {len(eq.empirical_anchors)} "
        f"anchors (predicted={args.predicted_reduction} "
        f"empirical={reduction:.4f} residual={residual:.4f})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI
    raise SystemExit(main())
