# SPDX-License-Identifier: MIT
"""Regression test for the Lane PD pose-delta wiring in compress_archive.py.

The encode-side wiring landed in commit 2d913687 added:
  * ``--pose-delta`` CLI flag (mutually exclusive with ``--binary-poses``)
  * ``_convert_poses_to_pose_delta()`` helper

Without this regression test, a future refactor could silently drop the
flag from the parser or break the helper — Lane 2 would then quietly stop
saving 49% on the pose payload, the same KL-distill bug class as
silent-default overrides (memory:
``feedback_silent_default_bug_class_findings_20260429.md``).

Tests:
  1. The argparse parser exposes ``--pose-delta`` with the expected help text.
  2. The mutual-exclusion guard fires when both ``--pose-delta`` and
     ``--binary-poses`` are passed.
  3. ``_convert_poses_to_pose_delta`` produces a torch-saved dict whose
     ``format`` sentinel is ``pose_delta_v1`` and whose
     ``submission_archive.load_optimized_poses`` round-trip lands within
     the codec's documented per-channel quantisation bound.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[3]
_COMPRESS_ARCHIVE_PATH = (
    _REPO / "submissions" / "robust_current" / "compress_archive.py"
)


def _import_compress_archive():
    """Import compress_archive.py by absolute path (the file lives outside
    the importable tac package so we can't ``import`` it directly)."""
    spec = importlib.util.spec_from_file_location(
        "compress_archive_module",
        _COMPRESS_ARCHIVE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_COMPRESS_ARCHIVE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smooth_poses(n_pairs: int = 600) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.cumsum(
        torch.randn(n_pairs, 6) * 0.001, dim=0
    ) + torch.randn(6) * 0.01


def test_parser_exposes_pose_delta_flag():
    """The new --pose-delta flag must be in the parser. Without this
    regression check, a refactor could silently drop the flag and Lane 2
    would stop saving 49% on poses without anyone noticing."""
    mod = _import_compress_archive()
    saved_argv = sys.argv
    try:
        sys.argv = [
            "compress_archive.py",
            "--renderer-bin", "/tmp/nope",
            "--masks-path", "/tmp/nope",
            "--poses-path", "/tmp/nope",
            "--pose-delta",
            "--dry-run",
        ]
        args = mod._parse_args()
        assert getattr(args, "pose_delta", None) is True
    finally:
        sys.argv = saved_argv


def test_mutual_exclusion_with_binary_poses():
    """--binary-poses and --pose-delta both encode poses; passing both must
    raise loudly, not silently pick one (CLAUDE.md non-negotiable: no
    silent defaults / no silent picks)."""
    mod = _import_compress_archive()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        renderer = td / "renderer.bin"
        renderer.write_bytes(b"x" * 64)
        masks = td / "masks.mkv"
        masks.write_bytes(b"x" * 64)
        poses_path = td / "poses.pt"
        torch.save(_smooth_poses(60), poses_path)
        out = td / "archive.zip"

        saved_argv = sys.argv
        try:
            sys.argv = [
                "compress_archive.py",
                "--renderer-bin", str(renderer),
                "--masks-path", str(masks),
                "--poses-path", str(poses_path),
                "--output", str(out),
                "--binary-poses",
                "--pose-delta",
            ]
            with pytest.raises(SystemExit, match=r"mutually exclusive"):
                mod.main()
        finally:
            sys.argv = saved_argv


def test_convert_poses_to_pose_delta_roundtrip():
    """The new helper must produce a sentinel dict the canonical loader
    accepts, and the round-trip must land within Lane PD's documented
    per-channel quantisation bound."""
    mod = _import_compress_archive()
    from tac.submission_archive import load_optimized_poses

    poses = _smooth_poses(600)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_path = td / "poses.pt"
        torch.save(poses, src_path)
        out_path = mod._convert_poses_to_pose_delta(src_path, td)
        assert out_path.exists()
        # The sentinel-detecting loader returns a vanilla fp32 tensor.
        recovered = load_optimized_poses(
            out_path, pose_dim=6, expected_n_pairs=600
        )
        assert recovered.shape == poses.shape
        assert recovered.dtype == torch.float32
        err = (poses - recovered).abs().max().item()
        # Smooth trajectory + int8 deltas → < 1e-3 max-abs reconstruction.
        assert err < 1e-3, f"max-abs error {err:.6e} too large"
        # The encoded torch.save'd dict should still beat raw fp16 storage.
        # Empirical (2026-04-29): 18-20% savings vs raw-fp16-bytes once torch
        # pickle overhead is amortised on both sides. The docstring's "49%"
        # number compared core encoded (3618 B) to raw fp16 numel*2 (7200 B)
        # without accounting for torch.save dict pickle overhead (~2KB on
        # both sides). The empirical floor is ~15% and we pin the test there.
        fp16_bytes = poses.numel() * 2  # fp16 is 2 bytes per scalar
        encoded_bytes = out_path.stat().st_size
        savings = 1 - encoded_bytes / fp16_bytes
        assert savings > 0.15, (
            f"expected >15% savings vs naive fp16 ({fp16_bytes}B); "
            f"got {savings*100:.1f}% ({encoded_bytes}B)"
        )
