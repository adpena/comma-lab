from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.derive_poses_from_raft import (
    _default_manifest_output,
    _sha256_file,
    _tensor_sha256,
    build_manifest,
    write_manifest,
)


def test_default_manifest_output_keeps_pose_suffix() -> None:
    path = _default_manifest_output(Path("runs/raft_poses.pt"))

    assert path == Path("runs/raft_poses.pt.manifest.json")


def test_tensor_sha256_includes_shape_and_dtype() -> None:
    flat = torch.arange(4, dtype=torch.float32)
    matrix = flat.reshape(2, 2)
    int_tensor = torch.arange(4, dtype=torch.int32)

    assert _tensor_sha256(flat) != _tensor_sha256(matrix)
    assert _tensor_sha256(flat) != _tensor_sha256(int_tensor)
    assert _tensor_sha256(matrix) == _tensor_sha256(matrix.clone())


def test_build_manifest_records_custody_and_non_score_status(tmp_path: Path) -> None:
    video = tmp_path / "0.mkv"
    video.write_bytes(b"video-bytes")
    baseline_path = tmp_path / "baseline.pt"
    output_path = tmp_path / "raft_poses.pt"
    baseline = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    output = baseline.clone()
    output[:, 0] = torch.tensor([0.1, 0.2, 0.3])
    torch.save(baseline, baseline_path)
    torch.save(output, output_path)

    manifest = build_manifest(
        video_path=video,
        baseline_poses_path=baseline_path,
        output_path=output_path,
        device="cuda",
        n_frames=1200,
        baseline_poses=baseline,
        output_poses=output,
        raw_dim0=torch.tensor([1.0, 2.0, 3.0]),
        calibrated_dim0=output[:, 0],
        calibration_a=0.5,
        calibration_b=-1.0,
        per_dim_rmse=torch.sqrt(torch.mean((output - baseline).square(), dim=0)),
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["frame_limit_pushed_into_decoder"] is True
    assert manifest["video"]["sha256"] == _sha256_file(video)
    assert manifest["baseline_poses"]["shape"] == [3, 6]
    assert manifest["output_poses"]["shape"] == [3, 6]
    assert manifest["raw_dim0"]["count"] == 3
    assert manifest["raw_dim0"]["mean"] == 2.0
    assert manifest["calibration"] == {
        "method": "least_squares_dim0_to_baseline_dim0",
        "a": 0.5,
        "b": -1.0,
    }
    assert len(manifest["per_dim_rmse"]["values"]) == 6
    assert "exact_archive_rebuild_and_cuda_auth_eval_required_before_score_claim" in manifest["blockers"]


def test_write_manifest_is_deterministic_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_manifest(path, {"z": 1, "a": {"b": False}})

    assert path.read_text().startswith('{\n  "a":')
    assert json.loads(path.read_text()) == {"a": {"b": False}, "z": 1}
