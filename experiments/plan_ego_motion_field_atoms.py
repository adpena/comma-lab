#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan low-dimensional ego-motion field atoms from optimized poses.

This is a deterministic planning artifact. It does not build an archive,
dispatch jobs, warp pixels, or claim score. The purpose is to expose the full
pose/ego-motion basis as charged allocation signals that downstream mask,
pose, foveation, and multimask planners can consume without repeating brittle
radial-zoom geometry bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.submission_archive import load_optimized_poses  # noqa: E402


SCHEMA = "ego_motion_field_atom_plan_v1"
TOOL = "experiments/plan_ego_motion_field_atoms.py"
EVIDENCE_GRADE = "planning_only"
SCORE_CLAIM = False
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_FRAME_HEIGHT = 384
DEFAULT_FRAME_WIDTH = 512
DEFAULT_FOE_X = 256.0
DEFAULT_FOE_Y = 174.0
DEFAULT_SIGMA = 72.0
DEFAULT_CENTER_GAIN_X = 4.0
DEFAULT_CENTER_GAIN_Y = 2.0


class EgoMotionPlanError(ValueError):
    """Raised when ego-motion planning inputs are invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EgoMotionPlanError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise EgoMotionPlanError(f"{field} must be finite")
    return out


def _parse_center(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("center must be 'x,y'")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center values must be numeric") from exc


def _diff_same_length(values: np.ndarray, *, order: int) -> np.ndarray:
    if order <= 0:
        return values.astype(np.float64, copy=True)
    out = values.astype(np.float64, copy=True)
    for _ in range(order):
        if out.size <= 1:
            return np.zeros_like(values, dtype=np.float64)
        d = np.diff(out, prepend=out[0])
        out = d
    return out


def _robust_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(values))
    return max(scale, 1e-9)


def _normalised_abs(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - float(np.median(values))
    return np.abs(centered) / _robust_scale(values.astype(np.float64))


def _pose_dimension_names(pose_dim: int) -> list[str]:
    base = ["velocity", "pose_dim1", "pose_dim2", "pose_dim3", "pose_dim4", "pose_dim5"]
    if pose_dim <= len(base):
        return base[:pose_dim]
    return base + [f"pose_dim{idx}" for idx in range(len(base), pose_dim)]


def _frame_centers_from_pose(
    poses: np.ndarray,
    *,
    base_center: tuple[float, float],
    gain_x: float,
    gain_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    centers = np.zeros((poses.shape[0] * 2, 2), dtype=np.float64)
    velocity = poses[:, 0]
    accel = _diff_same_length(velocity, order=1)
    yaw_like = poses[:, 1] if poses.shape[1] > 1 else np.zeros_like(velocity)
    pitch_like = poses[:, 2] if poses.shape[1] > 2 else np.zeros_like(velocity)
    yaw_norm = np.clip(yaw_like / _robust_scale(yaw_like), -4.0, 4.0)
    pitch_norm = np.clip(pitch_like / _robust_scale(pitch_like), -4.0, 4.0)
    accel_norm = np.clip(accel / _robust_scale(accel), -4.0, 4.0)
    pair_centers_x = np.clip(base_center[0] + gain_x * yaw_norm, 0.0, float(width - 1))
    pair_centers_y = np.clip(base_center[1] + gain_y * (pitch_norm + 0.25 * accel_norm), 0.0, float(height - 1))
    centers[0::2, 0] = pair_centers_x
    centers[1::2, 0] = pair_centers_x
    centers[0::2, 1] = pair_centers_y
    centers[1::2, 1] = pair_centers_y
    return centers


def _atom_record(
    *,
    family: str,
    pair_index: int,
    frame_indices: list[int],
    value: float,
    strength: float,
    byte_proxy: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    value = float(value)
    strength = float(strength)
    byte_proxy = int(byte_proxy)
    raw_id = json.dumps(
        {"family": family, "pair_index": pair_index, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    score_proxy = strength / 1_000_000.0
    rate_cost = RATE_SCORE_PER_BYTE * byte_proxy
    return {
        "atom_id": f"ego_{family}_{_sha256_bytes(raw_id)[:16]}",
        "family": family,
        "pair_index": int(pair_index),
        "frame_indices": [int(v) for v in frame_indices],
        "identity": identity,
        "value": round(value, 12),
        "strength_proxy": round(strength, 12),
        "estimated_charged_bytes_proxy": byte_proxy,
        "estimated_score_saved_proxy": round(score_proxy, 12),
        "estimated_rate_score_cost": round(rate_cost, 12),
        "estimated_lagrangian_net_proxy": round(score_proxy - rate_cost, 12),
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
    }


def _top_motion_atoms(
    poses: np.ndarray,
    *,
    max_atoms: int,
) -> list[dict[str, Any]]:
    names = _pose_dimension_names(poses.shape[1])
    atoms: list[dict[str, Any]] = []
    velocity = poses[:, 0]
    derivatives = {
        "velocity_level": velocity,
        "velocity_accel": _diff_same_length(velocity, order=1),
        "velocity_jerk": _diff_same_length(velocity, order=2),
    }
    for dim in range(1, poses.shape[1]):
        derivatives[f"{names[dim]}_level"] = poses[:, dim]
        derivatives[f"{names[dim]}_delta"] = _diff_same_length(poses[:, dim], order=1)
    if poses.shape[1] > 1:
        derivatives["curvature_velocity_yaw_product"] = velocity * poses[:, 1]
    if poses.shape[1] > 2:
        derivatives["pitch_velocity_product"] = velocity * poses[:, 2]

    for family, values in derivatives.items():
        strengths = _normalised_abs(values)
        for pair_index, strength in enumerate(strengths):
            atoms.append(
                _atom_record(
                    family=family,
                    pair_index=pair_index,
                    frame_indices=[2 * pair_index, 2 * pair_index + 1],
                    value=float(values[pair_index]),
                    strength=float(strength),
                    byte_proxy=4,
                    identity={"basis": family, "pose_source": "optimized_poses"},
                )
            )
    atoms.sort(
        key=lambda atom: (
            -float(atom["estimated_lagrangian_net_proxy"]),
            -float(atom["strength_proxy"]),
            atom["family"],
            atom["pair_index"],
        )
    )
    return atoms[:max_atoms]


def build_plan(
    *,
    pose_path: Path,
    output_json: Path,
    expected_pairs: int | None = 600,
    max_atoms: int = 512,
    foveal_center: tuple[float, float] = (DEFAULT_FOE_X, DEFAULT_FOE_Y),
    foveal_sigma: float = DEFAULT_SIGMA,
    center_gain_x: float = DEFAULT_CENTER_GAIN_X,
    center_gain_y: float = DEFAULT_CENTER_GAIN_Y,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
) -> dict[str, Any]:
    if max_atoms <= 0:
        raise EgoMotionPlanError("max_atoms must be positive")
    if foveal_sigma <= 0:
        raise EgoMotionPlanError("foveal_sigma must be positive")
    if frame_width <= 0 or frame_height <= 0:
        raise EgoMotionPlanError("frame dimensions must be positive")
    pose_path = pose_path.resolve()
    poses_t = load_optimized_poses(pose_path, pose_dim=6, expected_n_pairs=expected_pairs)
    poses = poses_t.detach().cpu().numpy().astype(np.float64, copy=False)
    if poses.ndim != 2 or poses.shape[1] < 1:
        raise EgoMotionPlanError(f"pose tensor must be rank-2, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise EgoMotionPlanError("pose tensor contains NaN or inf")
    if expected_pairs is not None and poses.shape[0] != expected_pairs:
        raise EgoMotionPlanError(f"expected {expected_pairs} pose pairs, got {poses.shape[0]}")

    centers = _frame_centers_from_pose(
        poses,
        base_center=foveal_center,
        gain_x=float(center_gain_x),
        gain_y=float(center_gain_y),
        width=frame_width,
        height=frame_height,
    )
    center_payload = {
        "frame_centers": [[round(float(x), 12), round(float(y), 12)] for x, y in centers.tolist()]
    }
    atoms = _top_motion_atoms(poses, max_atoms=max_atoms)
    names = _pose_dimension_names(poses.shape[1])
    pose_stats = {}
    for dim, name in enumerate(names):
        values = poses[:, dim]
        pose_stats[name] = {
            "mean": round(float(np.mean(values)), 12),
            "std": round(float(np.std(values)), 12),
            "min": round(float(np.min(values)), 12),
            "max": round(float(np.max(values)), 12),
            "robust_scale": round(_robust_scale(values), 12),
        }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": SCORE_CLAIM,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "non_promotable_warning": (
            "Planning only. Use this to weight allocation, foveation, or learned decoder "
            "hypotheses; no score/rank/promotion claim is valid until a closed archive "
            "charges every bit and passes exact CUDA auth eval."
        ),
        "inputs": {
            "pose_path": {
                "path": str(pose_path),
                "sha256": _sha256_file(pose_path),
                "size_bytes": pose_path.stat().st_size,
            },
            "expected_pairs": expected_pairs,
            "frame_width": frame_width,
            "frame_height": frame_height,
        },
        "ego_motion_basis": {
            "pose_dim_names": names,
            "interpretation": {
                "velocity": "dominant longitudinal scalar axis used by QP1",
                "pose_dim1_to_5": (
                    "non-velocity PoseNet/control dimensions; useful as allocation "
                    "signals before any charged residual codec is promoted"
                ),
                "products": "low-order curvature/pitch interactions, not pixel warps",
            },
            "radial_zoom_guard": (
                "This plan deliberately emits control-field atoms only. It does not "
                "authorize radial zoom or image warp runtime changes; tiny geometry "
                "errors must be exact-evaluated before promotion."
            ),
        },
        "pose_stats": pose_stats,
        "dynamic_foveation": {
            "schema": "ego_motion_dynamic_foveation_manifest_v1",
            "tool": TOOL,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": EVIDENCE_GRADE,
            "mode": "dynamic_per_frame_from_pose_field",
            "base_center": [round(float(foveal_center[0]), 12), round(float(foveal_center[1]), 12)],
            "sigma": round(float(foveal_sigma), 12),
            "center_gain_x": round(float(center_gain_x), 12),
            "center_gain_y": round(float(center_gain_y), 12),
            "frame_center_count": int(centers.shape[0]),
            "frame_centers_sha256": _sha256_bytes(_json_bytes(center_payload)),
            "first_frame_center": [round(float(centers[0, 0]), 12), round(float(centers[0, 1]), 12)],
            "last_frame_center": [round(float(centers[-1, 0]), 12), round(float(centers[-1, 1]), 12)],
            "frame_centers": center_payload["frame_centers"],
            "provenance": {
                "pose_path": str(pose_path),
                "pose_sha256": _sha256_file(pose_path),
                "pose_size_bytes": pose_path.stat().st_size,
                "source_plan_schema": SCHEMA,
                "source_pose_contract": "optimized_poses.bin loaded via tac.submission_archive.load_optimized_poses",
            },
            "charged_byte_policy": (
                "If shipped, these centers or their generator parameters must be charged "
                "inside archive.zip. As planner input they are empirical only."
            ),
        },
        "atoms": atoms,
        "atom_count_emitted": len(atoms),
        "formulas": {
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "dynamic_center_x": "clip(base_x + center_gain_x * robust_z(pose_dim1), 0, width-1)",
            "dynamic_center_y": (
                "clip(base_y + center_gain_y * (robust_z(pose_dim2) + "
                "0.25*robust_z(delta_velocity)), 0, height-1)"
            ),
            "atom_strength_proxy": "robust absolute z-score of pose/derivative/interaction basis",
        },
    }
    payload["output_sha256_without_self"] = _sha256_bytes(_json_bytes(payload))
    _write_json(output_json, payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=600)
    parser.add_argument("--max-atoms", type=int, default=512)
    parser.add_argument("--foveal-center", type=_parse_center, default=(DEFAULT_FOE_X, DEFAULT_FOE_Y))
    parser.add_argument("--foveal-sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--center-gain-x", type=float, default=DEFAULT_CENTER_GAIN_X)
    parser.add_argument("--center-gain-y", type=float, default=DEFAULT_CENTER_GAIN_Y)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = build_plan(
        pose_path=args.pose_path,
        output_json=args.output_json,
        expected_pairs=args.expected_pairs,
        max_atoms=args.max_atoms,
        foveal_center=args.foveal_center,
        foveal_sigma=args.foveal_sigma,
        center_gain_x=args.center_gain_x,
        center_gain_y=args.center_gain_y,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "score_claim": payload["score_claim"],
                "evidence_grade": payload["evidence_grade"],
                "atom_count_emitted": payload["atom_count_emitted"],
                "frame_centers_sha256": payload["dynamic_foveation"]["frame_centers_sha256"],
                "output_json": str(args.output_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
