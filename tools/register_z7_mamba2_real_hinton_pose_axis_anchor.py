# SPDX-License-Identifier: MIT
"""Register the Z7-Mamba-2 REAL-Hinton-teacher pose-axis EmpiricalAnchor.

The Wave N+11 QUAD composition HALTed partly because Z7-Mamba-2's only landed
canonical-equation anchor used the MOCK scorer teacher (pose-axis = 0). This
tool registers the FIRST REAL-teacher pose-axis anchor on
``z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`` from a real
``--full`` MLX-LOCAL run that wired the REAL SegNet + PoseNet Hinton-distilled
teachers (NO ``--allow-mock-scorer-teacher``).

The anchor's empirical value is the pose-axis REDUCTION observed across the
run: ``pose_axis_reduction = pose[ep0] - pose[ep_last]`` from the run's
``telemetry.jsonl`` ``per_axis_decomposition.pose`` channel. The predicted
value is the equation's standing prediction (the prior real-teacher anchor #3's
19.2% pose-axis reduction at the reduced-lr Wave N+10 RESUME config); the
residual is ``predicted - empirical`` per the canonical equation contract.

Usage::

    .venv/bin/python tools/register_z7_mamba2_real_hinton_pose_axis_anchor.py \\
        --run-dir .omx/research/z7_mamba2_real_hinton_long_mlx_<utc>

Per Catalog #344 (canonical equations registry) + Catalog #371 (auto-
recalibration trigger) + Catalog #323 (canonical Provenance) + Catalog #192/
#317/#341 (non-promotable [macOS-MLX research-signal] markers).

[verified-against: tac.canonical_equations.update_equation_with_empirical_anchor]
[verified-against: tac.provenance.build_provenance_for_mps_proxy macOS-MLX non-promotable]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_EQUATION_ID = "z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1"
# Prior real-teacher anchor #3 (Wave N+10 RESUME): 19.2% pose-axis reduction at
# reduced lr=1e-4 / 600pair / 50ep. The standing prediction for the canonical
# equation's real-teacher pose-axis reduction.
_PREDICTED_POSE_AXIS_REDUCTION_FRACTION = 0.192


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_pose_axis_series(telemetry_path: Path) -> list[float]:
    """Read the per-epoch ``per_axis_decomposition.pose`` series."""
    series: list[float] = []
    if not telemetry_path.is_file():
        return series
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        decomp = row.get("per_axis_decomposition") or {}
        pose = decomp.get("pose")
        if pose is not None:
            series.append(float(pose))
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Real-Hinton long MLX run output dir (contains telemetry.jsonl).",
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=None,
        help="UTC timestamp for the anchor provenance (default: now).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed anchor without registering it.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()

    telemetry = run_dir / "telemetry.jsonl"
    artifact = run_dir / "training_artifact.json"
    pose_series = _read_pose_axis_series(telemetry)
    if len(pose_series) < 2:
        print(
            f"ERROR: telemetry {telemetry} has < 2 pose-axis rows "
            f"({len(pose_series)}); cannot compute a pose-axis reduction. The "
            "real-teacher run must have completed at least 2 epochs.",
            file=sys.stderr,
        )
        return 2

    pose_ep0 = pose_series[0]
    pose_last = pose_series[-1]
    # Verify the REAL teacher fired: a mock-teacher run has pose-axis == 0.
    if pose_ep0 <= 0.0:
        print(
            f"ERROR: pose-axis at ep0 is {pose_ep0} (<= 0). This indicates a "
            "MOCK teacher run (pose=0) — the REAL teacher path was not "
            "exercised. Re-run WITHOUT --allow-mock-scorer-teacher.",
            file=sys.stderr,
        )
        return 3

    pose_reduction_abs = pose_ep0 - pose_last
    pose_reduction_frac = (
        pose_reduction_abs / pose_ep0 if pose_ep0 != 0.0 else 0.0
    )
    residual = (
        _PREDICTED_POSE_AXIS_REDUCTION_FRACTION - pose_reduction_frac
    )

    captured = args.captured_at_utc
    if captured is None:
        from datetime import datetime, timezone

        captured = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(
        f"[real-hinton pose-axis anchor] epochs={len(pose_series)} "
        f"pose_ep0={pose_ep0:.6g} pose_last={pose_last:.6g} "
        f"reduction_abs={pose_reduction_abs:.6g} "
        f"reduction_frac={pose_reduction_frac:.4f} "
        f"predicted_frac={_PREDICTED_POSE_AXIS_REDUCTION_FRACTION} "
        f"residual={residual:.4f}"
    )

    if args.dry_run:
        print("[dry-run] anchor NOT registered.")
        return 0

    from tac.canonical_equations import (
        EmpiricalAnchor,
        get_equation_by_id,
        update_equation_with_empirical_anchor,
    )
    from tac.provenance import build_provenance_for_mps_proxy

    artifact_sha = _sha256(artifact)
    prov = build_provenance_for_mps_proxy(
        source_path=str(
            artifact.relative_to(repo_root)
            if str(artifact).startswith(str(repo_root))
            else artifact
        ),
        artifact_sha256=artifact_sha,
        captured_at_utc=captured,
        in_domain_context="z7_mamba2_real_hinton_teacher_600pair_long_mlx",
    )
    anchor = EmpiricalAnchor(
        predicted_value=_PREDICTED_POSE_AXIS_REDUCTION_FRACTION,
        empirical_value=pose_reduction_frac,
        predicted_vs_empirical_residual=residual,
        provenance=prov,
        measurement_method=(
            "real_hinton_segnet_posenet_teacher_600pair_long_mlx_local_"
            "pose_axis_reduction_fraction"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        notes=(
            "FIRST REAL-teacher pose-axis anchor (extincts the Wave N+11 QUAD "
            "HALT mock-teacher pose=0 gap). Real SegNet KL T=2.0 + real "
            "PoseNet pose-MSE Hinton-distilled teachers wired (NO mock flag); "
            f"pose-axis {pose_ep0:.4g} -> {pose_last:.4g} "
            f"({pose_reduction_frac:.1%} reduction) over {len(pose_series)} "
            "epochs at 600pair MLX-LOCAL with Wave N+11 stabilizer (grad-clip "
            "1.0 + warmup 5 + weight_decay 1e-4 + d_state=8 + EMA 0.997). "
            "Non-promotable [macOS-MLX research-signal] per Catalog "
            "#192/#317/#341; paired Linux x86_64 CPU/CUDA replay DEFERRED."
        ),
    )
    update_equation_with_empirical_anchor(_EQUATION_ID, anchor)
    eq = get_equation_by_id(_EQUATION_ID)
    print(
        f"anchor registered; count={len(eq.empirical_anchors)} "
        f"last_calibration_utc={eq.last_calibration_utc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
