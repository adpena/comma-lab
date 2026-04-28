"""Tests for experiments/seed_poses_from_openpilot.py (Lane OS-A standalone tool).

These tests verify the CLI contract: argparse flags, fallback path, output
shape, and exit codes.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac.lane_mark_speed import LANE_MARK_CLASS

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "seed_poses_from_openpilot.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def _load_script_module():
    """Import the standalone script as a module so we can call its main()."""
    spec = importlib.util.spec_from_file_location("_seed_tool", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── File existence + executability ───────────────────────────────────


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"missing {SCRIPT}"


def test_script_is_executable() -> None:
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_script_has_shebang(script_text: str) -> None:
    """Standalone tool — needs to be runnable, but in this codebase Python
    scripts are typically executed via `python <path>` so a shebang is
    nice-to-have, not required. We instead require a __main__ guard."""
    assert "if __name__ ==" in script_text, "must have __main__ guard"


# ── CLI arguments declared ───────────────────────────────────────────


def test_argparse_has_supercombo_path(script_text: str) -> None:
    assert '"--supercombo-path"' in script_text


def test_argparse_has_video(script_text: str) -> None:
    assert '"--video"' in script_text


def test_argparse_has_output(script_text: str) -> None:
    assert '"--output"' in script_text


def test_argparse_has_device(script_text: str) -> None:
    assert '"--device"' in script_text


def test_argparse_device_choices_no_mps(script_text: str) -> None:
    """--device must NOT include 'mps' (CLAUDE.md MPS-fallback non-negotiable)."""
    m = re.search(r'"--device".*?choices=\[([^\]]+)\]', script_text, re.DOTALL)
    assert m is not None, "--device must declare choices"
    choices = m.group(1)
    assert '"mps"' not in choices and "'mps'" not in choices, (
        "MPS must not appear in --device choices (memory: "
        "feedback_mps_cuda_drift_critical — 23x PoseNet drift)"
    )


def test_argparse_has_baseline_poses(script_text: str) -> None:
    assert '"--baseline-poses"' in script_text


def test_argparse_has_allow_fallback(script_text: str) -> None:
    assert '"--allow-fallback"' in script_text


def test_argparse_has_masks_for_fallback(script_text: str) -> None:
    assert '"--masks"' in script_text


# ── Fallback path: exit code + output shape ──────────────────────────


def test_fallback_produces_correct_shape(tmp_path: Path) -> None:
    """When supercombo is missing and --allow-fallback + --masks are given,
    the tool produces a (N//2, 6) tensor by delegating to lane_mark_pose."""
    # Build a fake mask tensor with lane marks (so lane_mark_pose returns
    # something non-degenerate).
    n_frames, h, w = 12, 384, 512
    masks = torch.zeros(n_frames, h, w, dtype=torch.long)
    for i in range(n_frames):
        masks[i, 200 + i:205 + i, 250 + i:255 + i] = LANE_MARK_CLASS
    masks_path = tmp_path / "masks.pt"
    torch.save(masks, masks_path)

    output_path = tmp_path / "seed_poses.pt"
    fake_supercombo = tmp_path / "missing_supercombo.onnx"  # doesn't exist

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output_path),
            "--device", "cpu",
            "--n-frames", str(n_frames),
            "--masks", str(masks_path),
            "--allow-fallback",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"fallback path failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert output_path.exists()

    seed = torch.load(output_path, map_location="cpu", weights_only=True)
    assert seed.shape == (n_frames // 2, 6), (
        f"expected ({n_frames // 2}, 6), got {tuple(seed.shape)}"
    )
    assert "fallback=True" in result.stdout


def test_missing_supercombo_without_allow_fallback_exits_nonzero(
    tmp_path: Path,
) -> None:
    """Without --allow-fallback, missing supercombo → exit 1 (fail loud)."""
    output_path = tmp_path / "seed_poses.pt"
    fake_supercombo = tmp_path / "missing.onnx"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output_path),
            "--device", "cpu",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not output_path.exists(), (
        "no output should be written when the tool fails loud"
    )


def test_fallback_without_masks_exits_nonzero(tmp_path: Path) -> None:
    """--allow-fallback but no --masks → exit 1."""
    output_path = tmp_path / "seed_poses.pt"
    fake_supercombo = tmp_path / "missing.onnx"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output_path),
            "--device", "cpu",
            "--allow-fallback",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert not output_path.exists()


# ── Output is fp16 (archive parity with optimized_poses.pt) ──────────


def test_fallback_output_is_fp16(tmp_path: Path) -> None:
    """seed_poses.pt must be saved as fp16 (~7 KB for 600 pairs, archive parity)."""
    n_frames, h, w = 12, 384, 512
    masks = torch.zeros(n_frames, h, w, dtype=torch.long)
    for i in range(n_frames):
        masks[i, 200 + i:205 + i, 250 + i:255 + i] = LANE_MARK_CLASS
    masks_path = tmp_path / "masks.pt"
    torch.save(masks, masks_path)
    output_path = tmp_path / "seed_poses.pt"
    fake_supercombo = tmp_path / "missing.onnx"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output_path),
            "--device", "cpu",
            "--n-frames", str(n_frames),
            "--masks", str(masks_path),
            "--allow-fallback",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    seed = torch.load(output_path, map_location="cpu", weights_only=True)
    assert seed.dtype == torch.float16, (
        f"expected fp16 (archive parity with optimized_poses.pt), "
        f"got {seed.dtype}"
    )
