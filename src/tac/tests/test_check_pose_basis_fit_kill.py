# SPDX-License-Identifier: MIT
"""Regression tests for Check 91 (pose-basis-fit kill).

Lane GP v3 (89.67 [Modal-T4-CPU]) was killed 2026-04-30 per Council #271
+ Lane GP v4 design verdict (.omx/research/council_lane_gp_v4_design_20260430.md).
The killed lane class is "any smooth-basis pose-fit at K < raw fp16 size".

This check forbids new experiments/fit_pose_*.py files from importing
np.polyfit / scipy.interpolate.{BSpline,splrep,CubicSpline} / scipy.fft.dct
WITHOUT the `# LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED:` marker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_pose_basis_fit_kill_acknowledged,
)


def _make_repo_with_fit_file(tmp_path: Path, filename: str, content: str) -> Path:
    """Create a tmp 'repo' with experiments/<filename> containing `content`."""
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / filename).write_text(content)
    return tmp_path


def test_clean_fit_file_with_kill_marker_passes(tmp_path: Path) -> None:
    """A fit_pose_*.py file with the kill marker is exempt from the check."""
    content = (
        "#!/usr/bin/env python3\n"
        "# LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED: kill memo at design_20260430.md\n"
        "import numpy as np\n"
        "coeffs = np.polyfit([0,1,2], [1,2,3], 1)\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_legacy.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_polyfit_without_marker_violates(tmp_path: Path) -> None:
    """np.polyfit usage WITHOUT the kill marker is a violation."""
    content = (
        "import numpy as np\n"
        "x = np.linspace(0, 1, 100)\n"
        "y = np.sin(x)\n"
        "coeffs = np.polyfit(x, y, deg=10)\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_v5_polynomial.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "fit_pose_v5_polynomial.py" in violations[0]
    assert "polyfit" in violations[0]


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """strict=True must raise PreflightError on any violation."""
    content = (
        "import numpy as np\n"
        "coeffs = np.polyfit([0,1,2], [1,2,3], 1)\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_v6.py", content)
    with pytest.raises(PreflightError) as exc_info:
        check_pose_basis_fit_kill_acknowledged(
            repo_root=repo, strict=True, verbose=False
        )
    assert "POSE-BASIS-FIT KILL" in str(exc_info.value)
    assert "council_lane_gp_v4_design_20260430.md" in str(exc_info.value)


def test_bspline_without_marker_violates(tmp_path: Path) -> None:
    """scipy.interpolate.BSpline usage WITHOUT marker violates."""
    content = (
        "from scipy.interpolate import BSpline\n"
        "import numpy as np\n"
        "knots = np.linspace(0, 1, 10)\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_bspline.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "fit_pose_bspline.py" in violations[0]


def test_dct_without_marker_violates(tmp_path: Path) -> None:
    """scipy.fft.dct usage WITHOUT marker violates."""
    content = (
        "from scipy.fft import dct\n"
        "import numpy as np\n"
        "x = np.zeros(100)\n"
        "coeffs = dct(x, type=2, norm='ortho')\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_dct.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "scipy.fft.dct" in violations[0] or "dct" in violations[0]


def test_cubic_spline_without_marker_violates(tmp_path: Path) -> None:
    """scipy.interpolate.CubicSpline usage WITHOUT marker violates."""
    content = (
        "from scipy.interpolate import CubicSpline\n"
        "import numpy as np\n"
        "cs = CubicSpline([0, 1], [0, 1])\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "fit_pose_natural_cubic.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_non_fit_pose_file_ignored(tmp_path: Path) -> None:
    """Files NOT matching fit_pose_*.py glob are ignored."""
    # An experiment file that uses polyfit but doesn't match the glob.
    content = (
        "import numpy as np\n"
        "coeffs = np.polyfit([0,1,2], [1,2,3], 1)\n"
    )
    repo = _make_repo_with_fit_file(tmp_path, "train_renderer.py", content)
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_real_repo_fit_pose_gp_passes() -> None:
    """The actual experiments/fit_pose_gp.py in the repo MUST have the kill marker."""
    # Resolve the actual repo root from preflight.REPO_ROOT.
    from tac.preflight import REPO_ROOT

    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert violations == [], (
        f"Real repo fit_pose_*.py files violated Check 91: {violations}. "
        f"Add `# LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED:` marker."
    )


def test_check_91_strict_mode_real_repo_passes() -> None:
    """Strict mode against the real repo must not raise (0 violations)."""
    from tac.preflight import REPO_ROOT

    # Should NOT raise.
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=REPO_ROOT, strict=True, verbose=False
    )
    assert violations == []


def test_no_fit_pose_files_returns_empty(tmp_path: Path) -> None:
    """Edge case: experiments/ exists but has no fit_pose_*.py — returns []."""
    (tmp_path / "experiments").mkdir()
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_no_experiments_dir_returns_empty(tmp_path: Path) -> None:
    """Edge case: no experiments/ dir at all — returns []."""
    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_round1_fix_module_evasion_blocked(tmp_path: Path) -> None:
    """Round 1 council finding: a smooth-basis fit placed in src/tac/ instead
    of experiments/ MUST also trigger the gate.

    Tested module-name globs: pose_*_fit.py, pose_*_basis.py, pose_*_polynomial.py,
    pose_*_spline.py, pose_*_dct.py, pose_*_wavelet.py, pose_gaussian_process.py.
    """
    src_tac = tmp_path / "src" / "tac"
    src_tac.mkdir(parents=True)
    # An evasion attempt: place a smooth-basis module under src/tac/ (NOT
    # under experiments/) without the kill marker.
    content = (
        "import numpy as np\n"
        "def fit_dim0(poses):\n"
        "    return np.polyfit(np.arange(len(poses)), poses, deg=10)\n"
    )
    (src_tac / "pose_bspline_fit.py").write_text(content)

    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "pose_bspline_fit.py" in violations[0]


def test_round1_fix_marker_in_src_tac_passes(tmp_path: Path) -> None:
    """A src/tac/pose_*_fit.py file WITH the kill marker is exempt."""
    src_tac = tmp_path / "src" / "tac"
    src_tac.mkdir(parents=True)
    content = (
        "# LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED: archival module\n"
        "import numpy as np\n"
        "coeffs = np.polyfit([0,1], [0,1], 1)\n"
    )
    (src_tac / "pose_legacy_fit.py").write_text(content)

    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_round1_fix_test_files_skipped(tmp_path: Path) -> None:
    """Files inside src/tac/tests/ are SKIPPED (legitimate exercise of patterns)."""
    tests_dir = tmp_path / "src" / "tac" / "tests"
    tests_dir.mkdir(parents=True)
    content = (
        "import numpy as np\n"
        "def test_polyfit_works():\n"
        "    coeffs = np.polyfit([0,1,2], [1,2,3], 1)\n"
        "    assert len(coeffs) == 2\n"
    )
    # Even though the filename matches pose_*_fit.py, /tests/ skip should fire.
    (tests_dir / "pose_demo_fit.py").write_text(content)

    violations = check_pose_basis_fit_kill_acknowledged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []
