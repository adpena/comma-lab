# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient import PER_PAIR_GRADIENT_TENSOR_KIND
from tac.master_gradient_mlx_extractor import (
    AXIS_ORDER,
    EVIDENCE_TAG_MLX,
    HARDWARE_SUBSTRATE_MLX,
    HEURISTIC_GRADIENT_BYTE_DOMAIN,
    HEURISTIC_GRADIENT_TENSOR_KIND,
    MLXMasterGradientError,
    MLXMasterGradientResult,
    TensorByteSpan,
    build_mlx_master_gradient_anchor,
    mlx_master_gradient_anchor_blockers,
    project_per_tensor_sensitivity_to_per_byte,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _result(arr: np.ndarray, *, anchor_eligible: bool = False) -> MLXMasterGradientResult:
    metadata = {
        "measurement_method": "per_tensor_central_fd_via_mlx_scorer_oracle_projected_per_byte",
        "codec_grammar": "fec6_pr101_fixed_section",
        "gradient_tensor_kind": HEURISTIC_GRADIENT_TENSOR_KIND,
        "gradient_byte_domain": HEURISTIC_GRADIENT_BYTE_DOMAIN,
        "master_gradient_anchor_eligible": False,
        "master_gradient_anchor_blockers": ["source_runtime_full_frame_parity_missing"],
    }
    if anchor_eligible:
        metadata.update(
            {
                "gradient_tensor_kind": PER_PAIR_GRADIENT_TENSOR_KIND,
                "gradient_byte_domain": "zip_inner_member_payload",
                "master_gradient_anchor_eligible": True,
                "master_gradient_anchor_blockers": [],
            }
        )
    return MLXMasterGradientResult(
        per_pair_per_byte=arr,
        archive_sha256="a" * 64,
        archive_bytes_count=int(arr.shape[0]),
        n_pairs_used=int(arr.shape[1]),
        n_pairs_total=600,
        axes=AXIS_ORDER,
        operating_point={
            "d_seg": 0.1,
            "d_pose": 0.2,
            "rate": float(arr.shape[0]) / 37_545_489.0,
            "score": 100.0 * 0.1 + (10.0 * 0.2) ** 0.5 + 25.0 * float(arr.shape[0]) / 37_545_489.0,
        },
        fd_rel_eps=1e-2,
        n_decoder_tensors=1,
        decompressed_decoder_len=4,
        decoder_blob_offset=0,
        metadata=metadata,
    )


def test_project_per_tensor_sensitivity_to_per_byte_preserves_pair_axis() -> None:
    span = TensorByteSpan(
        name="decoder.weight",
        storage_index=0,
        shape=(2,),
        numel=2,
        mantissa_byte_offset=1,
        fp16_scale=0.5,
    )

    projected = project_per_tensor_sensitivity_to_per_byte(
        [span],
        {"decoder.weight": np.array([2.0, 4.0], dtype=np.float64)},
        {"decoder.weight": np.array([6.0, 8.0], dtype=np.float64)},
        archive_bytes_count=6,
        decoder_blob_offset=2,
        n_pairs_used=2,
    )

    assert projected.shape == (6, 2, 3)
    assert np.all(projected[:3] == 0.0)
    assert np.all(projected[5] == 0.0)
    np.testing.assert_allclose(projected[3:5, :, 0], [[0.5, 1.0], [0.5, 1.0]])
    np.testing.assert_allclose(projected[3:5, :, 1], [[1.5, 2.0], [1.5, 2.0]])
    assert np.all(projected[:, :, 2] == 0.0)


def test_mlx_tensor_fd_heuristic_refuses_master_gradient_anchor_authority() -> None:
    arr = np.zeros((4, 2, 3), dtype=np.float64)
    arr[1, :, 0] = [0.1, 0.2]
    arr[2, :, 1] = [0.3, 0.4]

    result = _result(arr)
    blockers = mlx_master_gradient_anchor_blockers(result)
    assert "source_runtime_full_frame_parity_missing" in blockers
    assert "gradient_tensor_kind_not_canonical_per_pair_per_byte" in blockers
    with pytest.raises(MLXMasterGradientError, match="not eligible"):
        build_mlx_master_gradient_anchor(
            result,
            gradient_array_path=Path("/tmp/master_gradient_mlx.npy"),
            measurement_call_id="local_mlx_smoke",
            measurement_utc="2026-05-27T00:00:00+00:00",
        )


def test_anchor_helper_accepts_future_canonical_per_pair_mlx_result() -> None:
    arr = np.zeros((4, 2, 3), dtype=np.float64)
    arr[1, :, 0] = [0.1, 0.2]
    arr[2, :, 1] = [0.3, 0.4]

    anchor = build_mlx_master_gradient_anchor(
        _result(arr, anchor_eligible=True),
        gradient_array_path=Path("/tmp/master_gradient_mlx.npy"),
        measurement_call_id="local_mlx_smoke",
        measurement_utc="2026-05-27T00:00:00+00:00",
    )
    payload = asdict(anchor)

    assert payload["measurement_axis"] == EVIDENCE_TAG_MLX
    assert payload["measurement_hardware"] == HARDWARE_SUBSTRATE_MLX
    assert payload["gradient_tensor_kind"] == PER_PAIR_GRADIENT_TENSOR_KIND
    assert payload["n_pairs"] == 2
    assert payload["n_pairs_used"] == 2
    assert payload["n_pairs_total"] == 600
    assert payload["score_axis_dominance"]["promotion_authority"] is False
    assert payload["score_axis_dominance"]["raw_archive_byte_authority"] is False


def test_cli_preserves_heuristic_without_master_gradient_anchor(monkeypatch, tmp_path: Path) -> None:
    tool_path = REPO_ROOT / "tools" / "extract_master_gradient_mlx.py"
    spec = importlib.util.spec_from_file_location("extract_master_gradient_mlx_under_test", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    arr = np.zeros((5, 2, 3), dtype=np.float64)
    arr[1, :, 0] = [0.25, 0.5]
    arr[2, :, 1] = [0.75, 1.0]

    def fake_extract(*_args: object, **_kwargs: object) -> MLXMasterGradientResult:
        assert _kwargs["pair_batch_size"] == 16
        return _result(arr)

    monkeypatch.setattr(tool, "extract_mlx_per_pair_master_gradient", fake_extract)

    out = tmp_path / "gradient.npy"
    anchors = tmp_path / "master_gradient_anchors.jsonl"
    rc = tool.main(
        [
            "--archive",
            str(tmp_path / "archive.zip"),
            "--out",
            str(out),
            "--anchor-jsonl",
            str(anchors),
            "--no-manifest",
            "--call-id",
            "unit_test_mlx_anchor",
        ]
    )

    assert rc == 0
    assert np.load(out).shape == (5, 2, 3)
    sidecar = json.loads(out.with_suffix(".npy.meta.json").read_text(encoding="utf-8"))
    assert sidecar["master_gradient_anchor_written"] is False
    assert sidecar["npy_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert sidecar["determinism"]["seed"] == 20260527
    assert sidecar["determinism"]["numpy_seeded"] is True
    assert "--call-id" in sidecar["argv"]
    replay = json.loads(out.with_suffix(".npy.replay_bundle.json").read_text(encoding="utf-8"))
    assert replay["schema"] == "mlx_master_gradient_replay_bundle.v1"
    assert replay["output"]["npy_sha256"] == sidecar["npy_sha256"]
    assert replay["environment"]["schema"] == "safe_replay_environment_capture.v1"
    assert replay["calibration_gate"]["ready_for_exact_eval_dispatch"] is False
    assert "contest_cpu_or_cuda_auth_axis_payload_missing" in replay["calibration_gate"]["blockers"]
    assert "source_runtime_full_frame_parity_missing" in sidecar["master_gradient_anchor_blockers"]
    assert sidecar["score_claim"] is False
    assert sidecar["ready_for_exact_eval_dispatch"] is False
    assert not anchors.exists()


def test_cli_refuses_anchor_write_for_heuristic_result(monkeypatch, tmp_path: Path) -> None:
    tool_path = REPO_ROOT / "tools" / "extract_master_gradient_mlx.py"
    spec = importlib.util.spec_from_file_location("extract_master_gradient_mlx_under_test_refuse", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    arr = np.zeros((5, 2, 3), dtype=np.float64)

    def fake_extract(*_args: object, **_kwargs: object) -> MLXMasterGradientResult:
        return _result(arr)

    monkeypatch.setattr(tool, "extract_mlx_per_pair_master_gradient", fake_extract)

    rc = tool.main(
        [
            "--archive",
            str(tmp_path / "archive.zip"),
            "--out",
            str(tmp_path / "gradient.npy"),
            "--anchor-jsonl",
            str(tmp_path / "master_gradient_anchors.jsonl"),
            "--no-manifest",
            "--write-anchor",
        ]
    )

    assert rc == 1


def test_cli_writes_mlx_manifest_as_false_authority(monkeypatch, tmp_path: Path) -> None:
    tool_path = REPO_ROOT / "tools" / "extract_master_gradient_mlx.py"
    spec = importlib.util.spec_from_file_location("extract_master_gradient_mlx_under_test_manifest", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    arr = np.zeros((5, 2, 3), dtype=np.float64)

    def fake_extract(*_args: object, **_kwargs: object) -> MLXMasterGradientResult:
        return _result(arr)

    monkeypatch.setattr(tool, "extract_mlx_per_pair_master_gradient", fake_extract)

    manifest = tmp_path / "mlx_manifest.jsonl"
    rc = tool.main(
        [
            "--archive",
            str(tmp_path / "archive.zip"),
            "--out",
            str(tmp_path / "gradient.npy"),
            "--manifest-jsonl",
            str(manifest),
        ]
    )

    assert rc == 0
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[-1])
    assert row["kind"] == "mlx_per_pair_master_gradient"
    assert row["evidence_tag"] == EVIDENCE_TAG_MLX
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["master_gradient_anchor_written"] is False
    assert row["npy_sha256"] == hashlib.sha256((tmp_path / "gradient.npy").read_bytes()).hexdigest()
    assert row["determinism"]["seed"] == 20260527
    assert row["replay_bundle_sha256"]
    assert row["calibration_gate"]["ready_for_exact_eval_dispatch"] is False


def test_mlx_replay_diff_reports_hash_and_env_mismatch(tmp_path: Path) -> None:
    tool_path = REPO_ROOT / "tools" / "diff_mlx_master_gradient_replay.py"
    spec = importlib.util.spec_from_file_location("diff_mlx_master_gradient_replay_under_test", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    left = {
        "schema": "mlx_master_gradient_replay_bundle.v1",
        "archive": {"sha256": "a" * 64},
        "output": {"npy_sha256": "b" * 64, "npy_shape": [5, 2, 3]},
        "determinism": {"seed": 20260527},
        "git_head": "abc123",
        "environment": {"env": {"PYTHONHASHSEED": "0"}},
        "calibration_gate": {"ready_for_exact_eval_dispatch": False},
    }
    right = json.loads(json.dumps(left))

    matched = tool.diff_replay_payloads(left, right)
    assert matched["matched"] is True
    assert matched["score_claim"] is False

    right["output"]["npy_sha256"] = "c" * 64
    right["environment"]["env"]["PYTHONHASHSEED"] = "1"
    diff = tool.diff_replay_payloads(left, right)

    assert diff["matched"] is False
    assert "output.npy_sha256" in diff["mismatches"]
    assert diff["changed_environment_keys"] == ["PYTHONHASHSEED"]


def test_mlx_replay_rerun_command_rewrites_outputs_and_strips_side_effects(
    tmp_path: Path,
) -> None:
    tool_path = REPO_ROOT / "tools" / "rerun_mlx_master_gradient_replay.py"
    spec = importlib.util.spec_from_file_location("rerun_mlx_master_gradient_replay_under_test", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    bundle_path = tmp_path / "source.replay_bundle.json"
    bundle = {
        "schema": "mlx_master_gradient_replay_bundle.v1",
        "tool": "tools/extract_master_gradient_mlx.py",
        "argv": [
            "--archive",
            "archive.zip",
            "--out",
            "original.npy",
            "--manifest-jsonl",
            ".omx/state/mlx_research_signal_manifest.jsonl",
            "--replay-bundle-path=original.replay_bundle.json",
            "--write-anchor",
            "--no-replay-bundle",
            "--n-pairs",
            "2",
        ],
        "archive": {"sha256": "a" * 64},
        "calibration_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }

    record = tool.build_rerun_command(
        bundle,
        bundle_path=bundle_path,
        output_dir=tmp_path / "reruns",
        python_executable="/usr/bin/python3",
        run_id="stable_replay",
    )
    command = record["command"]

    assert command[:2] == [
        "/usr/bin/python3",
        str(REPO_ROOT / "tools" / "extract_master_gradient_mlx.py"),
    ]
    assert "--archive" in command
    assert "--n-pairs" in command
    assert "--write-anchor" not in command
    assert "--no-replay-bundle" not in command
    assert "--manifest-jsonl" not in command
    assert not any(str(part).startswith("--replay-bundle-path=") for part in command)
    assert command.count("--out") == 1
    assert command.count("--replay-bundle-path") == 1
    assert command[-1] == "--no-manifest"
    assert record["run_dir"] == str(tmp_path / "reruns" / "stable_replay")
    assert record["score_claim"] is False
    assert record["ready_for_exact_eval_dispatch"] is False


def test_mlx_replay_rerun_rejects_truthy_calibration_authority(tmp_path: Path) -> None:
    tool_path = REPO_ROOT / "tools" / "rerun_mlx_master_gradient_replay.py"
    spec = importlib.util.spec_from_file_location("rerun_mlx_master_gradient_replay_bad_gate", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    bundle = {
        "schema": "mlx_master_gradient_replay_bundle.v1",
        "tool": "tools/extract_master_gradient_mlx.py",
        "argv": ["--archive", "archive.zip", "--out", "original.npy"],
        "calibration_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": True,
        },
    }

    with pytest.raises(tool.MLXReplayRerunError, match="ready_for_exact_eval_dispatch"):
        tool.build_rerun_command(
            bundle,
            bundle_path=tmp_path / "bad.replay_bundle.json",
            output_dir=tmp_path / "reruns",
            python_executable="/usr/bin/python3",
        )
