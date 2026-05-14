# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.archive_diet`.

Covers:

* Pass-through (no techniques) preserves input archive byte-for-byte
  modulo deterministic re-zip.
* ``pose_delta`` reduces optimized_poses.pt member size on a synthetic
  smooth pose trajectory.
* ``mkv_passthrough`` reduces archive size when masks.mkv is highly
  compressed already (DEFLATE on AV1 wastes a few hundred bytes).
* :func:`verify_diet_archive` returns ok=True for the lossless / nearly
  lossless techniques.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.archive_diet import (
    DietResult,
    diet_archive,
    verify_diet_archive,
)


def _make_synthetic_archive(tmp_path: Path) -> Path:
    """Build a tiny submission-shaped archive in tmp_path/in.zip.

    Members:
        renderer.bin              random bytes (32 KB)
        masks.mkv                 random bytes (16 KB)  (incompressible-ish)
        optimized_poses.pt        smooth (n=600, 6) trajectory tensor
    """
    rng = np.random.default_rng(0)
    renderer_bytes = rng.bytes(32 * 1024)
    # AV1-like incompressible payload: random + a tiny non-zero header.
    mkv_bytes = b"\x1a\x45\xdf\xa3" + rng.bytes(16 * 1024 - 4)
    # Smooth pose trajectory: cumulative gaussian noise → very small deltas
    # so Lane PD compresses well.
    deltas = rng.normal(loc=0.0, scale=0.001, size=(599, 6)).astype(np.float32)
    anchor = np.array([0.5, -0.3, 1.7, 0.01, -0.02, 0.005], dtype=np.float32)
    poses = np.cumsum(np.concatenate([anchor[None], deltas], axis=0), axis=0)
    poses_t = torch.from_numpy(poses)
    poses_buf = io.BytesIO()
    torch.save(poses_t, poses_buf)
    poses_bytes = poses_buf.getvalue()

    in_zip = tmp_path / "in.zip"
    with zipfile.ZipFile(in_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("renderer.bin", renderer_bytes)
        zf.writestr("masks.mkv", mkv_bytes)
        zf.writestr("optimized_poses.pt", poses_bytes)
    return in_zip


def test_diet_archive_passthrough_no_techniques(tmp_path: Path) -> None:
    """With techniques=[], output should be a valid deterministic archive
    that decodes to identical members. Output size may differ from input
    by a few bytes due to deterministic timestamp normalization.
    """
    in_zip = _make_synthetic_archive(tmp_path)
    out_zip = tmp_path / "out.zip"
    res = diet_archive(in_zip, out_zip, techniques=[])
    assert isinstance(res, DietResult)
    assert res.input_bytes > 0
    assert res.output_bytes > 0
    # Pass-through must not lose more than 5% to re-zip overhead.
    assert abs(res.input_bytes - res.output_bytes) / res.input_bytes < 0.05
    # Verify identity (raw bytes equal for opaque members).
    rep = verify_diet_archive(in_zip, out_zip)
    assert rep["ok"], rep


def test_diet_archive_pose_delta_saves_bytes(tmp_path: Path) -> None:
    """pose_delta on a smooth trajectory should shrink optimized_poses.pt.

    Smooth fp32 (600,6) is 14_400 B raw + torch.save overhead. Lane PD:
    anchor (12 B) + scale (12 B) + deltas_q (3594 B) + dict overhead ≈ 3700 B.
    """
    in_zip = _make_synthetic_archive(tmp_path)
    out_zip = tmp_path / "out_pose.zip"
    res = diet_archive(in_zip, out_zip, techniques=["pose_delta"])

    assert "pose_delta" in res.techniques_applied
    in_pose = res.per_member_in["optimized_poses.pt"]
    out_pose = res.per_member_out["optimized_poses.pt"]
    assert out_pose < in_pose, (
        f"expected pose_delta to shrink optimized_poses.pt; "
        f"in={in_pose} out={out_pose}"
    )
    # Concrete savings >= 30% on this smooth trajectory.
    assert out_pose / in_pose < 0.7

    rep = verify_diet_archive(in_zip, out_zip)
    assert rep["ok"], rep
    pose_check = rep["per_member"]["optimized_poses.pt"]
    assert pose_check["ok"]
    # Lane PD spec: per-dim error bounded by delta_scale/127 — should be
    # well under 0.5 on a smooth trajectory.
    assert pose_check["max_abs_diff"] < 0.5


def test_diet_archive_mkv_passthrough_shrinks_or_equal(tmp_path: Path) -> None:
    """ZIP_STORED on the (already-compressed-shaped) mkv should not grow
    the archive and is usually a small win for genuinely random AV1 bytes.
    """
    in_zip = _make_synthetic_archive(tmp_path)
    out_zip = tmp_path / "out_mkv.zip"
    res = diet_archive(in_zip, out_zip, techniques=["mkv_passthrough"])
    assert "mkv_passthrough" in res.techniques_applied
    assert res.output_bytes <= res.input_bytes + 64  # tolerance for header diff
    rep = verify_diet_archive(in_zip, out_zip)
    assert rep["ok"], rep


def test_diet_archive_arithmetic_renderer_noop_on_asym_magic(tmp_path: Path) -> None:
    """ASYM-magic renderer.bin must NOT be touched by arithmetic_renderer.
    It should fall through to the zip pass and end up byte-identical inside.
    """
    rng = np.random.default_rng(3)
    poses_buf = io.BytesIO()
    torch.save(torch.zeros(2, 6), poses_buf)
    in_zip = tmp_path / "asym.zip"
    asym_bytes = b"ASYM" + rng.bytes(2048)
    with zipfile.ZipFile(in_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("renderer.bin", asym_bytes)
        zf.writestr("masks.mkv", b"\x1a\x45\xdf\xa3" + rng.bytes(512))
        zf.writestr("optimized_poses.pt", poses_buf.getvalue())

    out_zip = tmp_path / "asym_diet.zip"
    res = diet_archive(in_zip, out_zip, techniques=["arithmetic_renderer"])
    # arithmetic_renderer must be classified noop because the magic isn't xz.
    assert "arithmetic_renderer" not in res.techniques_applied
    assert "arithmetic_renderer" in res.techniques_noop
    # Renderer bytes survive unchanged.
    rep = verify_diet_archive(in_zip, out_zip)
    assert rep["ok"], rep


def test_diet_archive_unknown_technique_classified_noop(tmp_path: Path) -> None:
    in_zip = _make_synthetic_archive(tmp_path)
    out_zip = tmp_path / "out_unk.zip"
    res = diet_archive(in_zip, out_zip, techniques=["does_not_exist"])
    assert "does_not_exist" in res.techniques_noop
    assert "does_not_exist" not in res.techniques_applied


def test_diet_archive_combined_techniques(tmp_path: Path) -> None:
    """pose_delta + mkv_passthrough should compose without surprise."""
    in_zip = _make_synthetic_archive(tmp_path)
    out_zip = tmp_path / "out_combo.zip"
    res = diet_archive(
        in_zip, out_zip, techniques=["pose_delta", "mkv_passthrough"]
    )
    assert "pose_delta" in res.techniques_applied
    assert "mkv_passthrough" in res.techniques_applied
    assert res.output_bytes < res.input_bytes
    rep = verify_diet_archive(in_zip, out_zip)
    assert rep["ok"], rep


def test_diet_archive_idempotent_on_pose_delta(tmp_path: Path) -> None:
    """Running pose_delta twice should NOT shrink the archive further."""
    in_zip = _make_synthetic_archive(tmp_path)
    pass1 = tmp_path / "p1.zip"
    pass2 = tmp_path / "p2.zip"
    diet_archive(in_zip, pass1, techniques=["pose_delta"])
    res2 = diet_archive(pass1, pass2, techniques=["pose_delta"])
    # Lane PD format dict is detected and re-encoding is skipped.
    assert "pose_delta" not in res2.techniques_applied
    assert pass1.stat().st_size == pass2.stat().st_size


def test_diet_archive_deterministic(tmp_path: Path) -> None:
    """Same input + same techniques → byte-identical output across runs."""
    in_zip = _make_synthetic_archive(tmp_path)
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"
    diet_archive(in_zip, out_a, techniques=["pose_delta", "mkv_passthrough"])
    diet_archive(in_zip, out_b, techniques=["pose_delta", "mkv_passthrough"])
    assert out_a.read_bytes() == out_b.read_bytes()


@pytest.mark.skipif(
    not Path("/Users/adpena/Projects/pact/experiments/results/lane_a_landed/archive_lane_a.zip").is_file(),
    reason="Lane A anchor archive not present (CI / clean checkout).",
)
def test_diet_archive_on_real_lane_a_archive(tmp_path: Path) -> None:
    """Real-archive smoke test: run on the Lane G v3 anchor and confirm
    pose_delta saves bytes. Skipped automatically if anchor is missing.
    """
    src = Path("/Users/adpena/Projects/pact/experiments/results/lane_a_landed/archive_lane_a.zip")
    out_zip = tmp_path / "lane_a_diet.zip"
    res = diet_archive(
        src, out_zip, techniques=["pose_delta", "mkv_passthrough"]
    )
    # On the Lane G v3 anchor the pose_delta technique must apply.
    assert "pose_delta" in res.techniques_applied
    assert res.savings_bytes > 0, res.as_dict()
    rep = verify_diet_archive(src, out_zip)
    assert rep["ok"], rep
