"""Regression tests for Check 42: pose-projection train/inference parity.

Catches the BUG-1 class from Lane M-V2 audit (memory:
project_lane_m_v2_audit_council_findings_20260428): a pose-projection
helper used at optimization time but never called from inflate produces
a train/inference distribution mismatch.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_pose_projection_train_inference_parity,
)


def _setup_fake_repo(root: Path) -> None:
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "submissions" / "robust_current").mkdir(parents=True, exist_ok=True)


def test_strict_passes_on_real_codebase() -> None:
    """Live repo must pass — every pose-projection helper has parity or waiver."""
    violations = check_pose_projection_train_inference_parity(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"Check 42 found {len(violations)} violation(s). Add inflate-side "
        f"call OR `# PROJECT_PARITY_WAIVED:` marker. Violations:\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_detects_optimizer_only_helper(tmp_path: Path) -> None:
    """Helper defined in optimize script with no inflate call → flag."""
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text(textwrap.dedent('''\
        import torch
        def _project_to_renderer_pose(cond):
            return torch.zeros_like(cond)
    '''))
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# does not call _project_to_renderer_pose\n")
    violations = check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "_project_to_renderer_pose" in violations[0]


def test_passes_when_inflate_calls_helper(tmp_path: Path) -> None:
    """Helper called from inflate → no flag."""
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text(textwrap.dedent('''\
        def _project_to_renderer_pose(cond):
            return cond
    '''))
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text(textwrap.dedent('''\
        from optimize_poses import _project_to_renderer_pose
        result = _project_to_renderer_pose(x)
    '''))
    assert check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_passes_with_waiver_marker(tmp_path: Path) -> None:
    """`# PROJECT_PARITY_WAIVED:` near def → no flag."""
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text(textwrap.dedent('''\
        # PROJECT_PARITY_WAIVED: this is a tto-time only helper
        def _project_to_renderer_pose(cond):
            return cond
    '''))
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# nothing\n")
    assert check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_waiver_window_extends_back_15_lines(tmp_path: Path) -> None:
    """Multi-line waiver comments work (up to 15 lines before def)."""
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text(textwrap.dedent('''\
        # Some unrelated lines
        # ...
        # PROJECT_PARITY_WAIVED: BUG-1 fix queued in V3
        # Multi-line context follows here:
        # the optimizer currently zero-pads dims 1-5
        # while inflate uses raw frozen-baseline saved tensor
        # creating a train/inference mismatch (Lane M-V2)
        # V3-clean ($0.30, 2h) fixes this by passing
        # init_poses[:, 1:6] through this projection.
        # Once V3 lands the waiver should be removed.
        def _project_to_renderer_pose(cond):
            return cond
    '''))
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# nothing\n")
    assert check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_pose_pad_helper_also_detected(tmp_path: Path) -> None:
    """`*_pose_pad*` helpers also flagged."""
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text(textwrap.dedent('''\
        def baseline_pose_pad(cond):
            return cond
    '''))
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# does not call\n")
    violations = check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "baseline_pose_pad" in violations[0]


def test_strict_raises_metabugviolation(tmp_path: Path) -> None:
    _setup_fake_repo(tmp_path)
    opt = tmp_path / "experiments" / "optimize_poses.py"
    opt.write_text("def _project_to_renderer_pose(cond):\n    return cond\n")
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# nothing\n")
    with pytest.raises(MetaBugViolation, match="POSE-PROJECTION"):
        check_pose_projection_train_inference_parity(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_test_files_excluded(tmp_path: Path) -> None:
    """Test files don't trigger Check 42."""
    _setup_fake_repo(tmp_path)
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
    test_file = tmp_path / "src" / "tac" / "tests" / "test_thing.py"
    test_file.write_text("def _project_to_renderer_pose():\n    pass\n")
    inflate = tmp_path / "submissions" / "robust_current" / "inflate_renderer.py"
    inflate.write_text("# nothing\n")
    assert check_pose_projection_train_inference_parity(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []
