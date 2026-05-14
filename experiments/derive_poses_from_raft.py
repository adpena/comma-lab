#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tac.raft_pose import (
    build_pose_tensor_from_flow,
    calibrate_pose_dim0,
    compute_raft_flow,
    flow_to_pose_dim0,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(cpu_tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "tensor_sha256": _tensor_sha256(cpu_tensor),
    }


def _tensor_stats_record(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu().float()
    count = int(cpu_tensor.numel())
    record = _tensor_record(cpu_tensor)
    record["count"] = count
    if count:
        record["mean"] = float(cpu_tensor.mean().item())
        record["std_unbiased_false"] = float(cpu_tensor.std(unbiased=False).item())
        record["min"] = float(cpu_tensor.min().item())
        record["max"] = float(cpu_tensor.max().item())
    else:
        record["mean"] = None
        record["std_unbiased_false"] = None
        record["min"] = None
        record["max"] = None
    return record


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_pose_tensor(path: Path, device: str) -> torch.Tensor:
    data = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(data, dict):
        for key in ("poses", "optimized_poses", "gt_poses"):
            value = data.get(key)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32, device=device)
        raise ValueError(f"{path} dict has no poses, optimized_poses, or gt_poses key")
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def _default_manifest_output(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".manifest.json")


def build_manifest(
    *,
    video_path: Path,
    baseline_poses_path: Path,
    output_path: Path,
    device: str,
    n_frames: int,
    baseline_poses: torch.Tensor,
    output_poses: torch.Tensor,
    raw_dim0: torch.Tensor,
    calibrated_dim0: torch.Tensor,
    calibration_a: float,
    calibration_b: float,
    per_dim_rmse: torch.Tensor,
) -> dict[str, Any]:
    output_record = _file_record(output_path)
    output_record.update(_tensor_record(output_poses))

    baseline_record = _file_record(baseline_poses_path)
    baseline_record.update(_tensor_record(baseline_poses))

    return {
        "schema_version": 1,
        "tool": "experiments.derive_poses_from_raft",
        "codec_surface": "la_pose_raft_pose_derivation",
        "evidence_grade": "empirical_pose_derivation_non_score",
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "poses_are_training_or_planning_inputs_only",
            "exact_archive_rebuild_and_cuda_auth_eval_required_before_score_claim",
        ],
        "device": device,
        "n_frames_requested": int(n_frames),
        "frame_limit_pushed_into_decoder": True,
        "video": _file_record(video_path),
        "baseline_poses": baseline_record,
        "output_poses": output_record,
        "raw_dim0": _tensor_stats_record(raw_dim0),
        "calibrated_dim0": _tensor_stats_record(calibrated_dim0),
        "calibration": {
            "method": "least_squares_dim0_to_baseline_dim0",
            "a": float(calibration_a),
            "b": float(calibration_b),
        },
        "per_dim_rmse": {
            "values": [float(value) for value in per_dim_rmse.detach().cpu().tolist()],
            **_tensor_record(per_dim_rmse),
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive FiLM poses from RAFT flow")
    parser.add_argument("--video", required=True, help="Path to upstream/videos/0.mkv")
    parser.add_argument("--baseline-poses", required=True, help="Lane A optimized_poses.pt for calibration")
    parser.add_argument("--output", required=True, help="Output raft_poses.pt")
    parser.add_argument(
        "--manifest-output",
        help="Output custody manifest JSON; defaults to <output>.manifest.json",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--n-frames", type=int, default=1200)
    args = parser.parse_args()

    video_path = Path(args.video)
    baseline_path = Path(args.baseline_poses)
    out_path = Path(args.output)
    manifest_path = Path(args.manifest_output) if args.manifest_output else _default_manifest_output(out_path)

    baseline = _load_pose_tensor(baseline_path, args.device)
    if baseline.ndim != 2 or baseline.shape[1] != 6:
        raise ValueError(f"--baseline-poses must have shape (N, 6), got {tuple(baseline.shape)}")

    flow = compute_raft_flow(str(video_path), n_frames=int(args.n_frames), device=args.device)
    raw_dim0_all = flow_to_pose_dim0(flow)
    raw_dim0 = raw_dim0_all[::2][: baseline.shape[0]]
    if raw_dim0.numel() != baseline.shape[0]:
        raise ValueError(
            f"RAFT produced {raw_dim0.numel()} renderer-pair flow values, "
            f"but baseline has {baseline.shape[0]} poses"
        )

    calibrated_dim0, a, b = calibrate_pose_dim0(raw_dim0, baseline[:, 0])
    poses = build_pose_tensor_from_flow(calibrated_dim0, baseline_poses=baseline)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(poses.detach(), out_path)

    delta = poses - baseline[: poses.shape[0]]
    per_dim_rmse = torch.sqrt(torch.mean(delta.square(), dim=0))
    manifest = build_manifest(
        video_path=video_path,
        baseline_poses_path=baseline_path,
        output_path=out_path,
        device=args.device,
        n_frames=int(args.n_frames),
        baseline_poses=baseline,
        output_poses=poses,
        raw_dim0=raw_dim0,
        calibrated_dim0=calibrated_dim0,
        calibration_a=a,
        calibration_b=b,
        per_dim_rmse=per_dim_rmse,
    )
    write_manifest(manifest_path, manifest)

    print(f"raft_pose: wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"raft_pose: wrote manifest {manifest_path} ({manifest_path.stat().st_size} bytes)")
    print(f"raft_pose: manifest_sha256={_sha256_file(manifest_path)}")
    print(f"raft_pose: calibration a={a:.8g} b={b:.8g}")
    print(f"raft_pose: raw_dim0 mean={raw_dim0.mean().item():.6f} std={raw_dim0.std(unbiased=False).item():.6f}")
    print(
        "raft_pose: calibrated_dim0 "
        f"mean={calibrated_dim0.mean().item():.6f} "
        f"std={calibrated_dim0.std(unbiased=False).item():.6f}"
    )
    print("raft_pose: per_dim_rmse=[" + ", ".join(f"{v:.6f}" for v in per_dim_rmse.tolist()) + "]")


if __name__ == "__main__":
    main()
