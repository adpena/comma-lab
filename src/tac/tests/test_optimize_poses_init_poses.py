# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "optimize_poses.py"


def _load_optimize_poses_module():
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("optimize_poses_for_init_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_init_poses_flag_visible_in_help() -> None:
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr[:500]
    assert "--init-poses" in result.stdout


def test_init_poses_is_mutually_exclusive_with_legacy_warm_start_flags() -> None:
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            "checkpoint.pt",
            "--masks",
            "masks.pt",
            "--device",
            "cpu",
            "--init-poses",
            "raft_poses.pt",
            "--seed-poses-path",
            "seed_poses.pt",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode != 0
    assert "--init-poses is mutually exclusive" in result.stderr


def test_load_init_pose_tensor_accepts_tensor_and_dict(tmp_path: Path) -> None:
    mod = _load_optimize_poses_module()

    tensor_path = tmp_path / "poses.pt"
    torch.save(torch.arange(18, dtype=torch.float32).reshape(3, 6), tensor_path)
    loaded = mod._load_init_pose_tensor(
        tensor_path,
        expected_n_pairs=2,
        source_label="--init-poses",
    )
    assert loaded.shape == (2, 6)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, torch.arange(12, dtype=torch.float32).reshape(2, 6))

    dict_path = tmp_path / "poses_dict.pt"
    torch.save({"poses": torch.ones(4, 6)}, dict_path)
    loaded_dict = mod._load_init_pose_tensor(
        dict_path,
        expected_n_pairs=4,
        source_label="--init-poses",
    )
    assert loaded_dict.shape == (4, 6)


def test_load_init_pose_tensor_rejects_bad_shape_and_nonfinite(tmp_path: Path) -> None:
    mod = _load_optimize_poses_module()

    bad_shape = tmp_path / "bad_shape.pt"
    torch.save(torch.zeros(4, 5), bad_shape)
    with pytest.raises(ValueError, match="shape"):
        mod._load_init_pose_tensor(
            bad_shape,
            expected_n_pairs=4,
            source_label="--init-poses",
        )

    bad_value = tmp_path / "bad_value.pt"
    poses = torch.zeros(4, 6)
    poses[0, 0] = float("nan")
    torch.save(poses, bad_value)
    with pytest.raises(ValueError, match="NaN or Inf"):
        mod._load_init_pose_tensor(
            bad_value,
            expected_n_pairs=4,
            source_label="--init-poses",
        )
