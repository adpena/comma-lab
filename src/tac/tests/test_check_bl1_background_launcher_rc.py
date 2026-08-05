# SPDX-License-Identifier: MIT
"""BL1 / task #937 guard tests for detached-launcher rc laundering."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_bl1_launcher_rationale_is_placeholder,
    check_background_launcher_rc_not_job_verdict,
)


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_catches_returncode_zero_success_verdict(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/bad.py",
        "import subprocess\n"
        "def launch():\n"
        "    proc = subprocess.run(['python3', 'tools/launch_detached_process.py'], check=False)\n"
        "    if proc.returncode == 0:\n"
        "        return 'job succeeded'\n",
    )

    violations = check_background_launcher_rc_not_job_verdict(repo_root=tmp_path)
    assert len(violations) == 1
    assert "tools/bad.py:4" in violations[0]
    assert "done-receipt" in violations[0]


def test_catches_check_true_on_detached_launcher(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/bad_check_true.py",
        "import subprocess\n"
        "def launch():\n"
        "    subprocess.run(['python3', 'tools/launch_detached_process.py'], check=True)\n",
    )

    violations = check_background_launcher_rc_not_job_verdict(repo_root=tmp_path)
    assert len(violations) == 1
    assert "check=True" in violations[0]


def test_allows_nonzero_launcher_start_failure_guard(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/good.py",
        "import subprocess\n"
        "def launch():\n"
        "    proc = subprocess.run(['python3', 'tools/launch_detached_process.py'], check=False)\n"
        "    if proc.returncode != 0:\n"
        "        raise RuntimeError('detached launcher failed to start')\n"
        "    return 'launched, not completed'\n",
    )

    assert check_background_launcher_rc_not_job_verdict(repo_root=tmp_path) == []


def test_same_line_waiver_allows_explicit_launcher_start_health(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/waived.py",
        "import subprocess\n"
        "def launch():\n"
        "    proc = subprocess.run(['python3', 'tools/launch_detached_process.py'], check=False)\n"
        "    if proc.returncode == 0:  # LAUNCHER_RC_OK: launch-start health only, receipt read elsewhere\n"
        "        return 'launcher started'\n",
    )

    assert check_background_launcher_rc_not_job_verdict(repo_root=tmp_path) == []


def test_placeholder_waiver_is_rejected(tmp_path: Path) -> None:
    assert _check_bl1_launcher_rationale_is_placeholder("<reason>") is True
    _write(
        tmp_path,
        "tools/placeholder.py",
        "import subprocess\n"
        "def launch():\n"
        "    proc = subprocess.run(['python3', 'tools/launch_detached_process.py'], check=False)\n"
        "    if proc.returncode == 0:  # LAUNCHER_RC_OK:<reason>\n"
        "        return 'job succeeded'\n",
    )

    violations = check_background_launcher_rc_not_job_verdict(repo_root=tmp_path)
    assert len(violations) == 1


def test_shell_success_chain_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/bad.sh",
        "python3 tools/launch_detached_process.py --output-dir d -- false && echo job-ok\n",
    )

    violations = check_background_launcher_rc_not_job_verdict(repo_root=tmp_path)
    assert len(violations) == 1
    assert "scripts/bad.sh:1" in violations[0]


def test_strict_raises(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/bad.py",
        "import subprocess\n"
        "proc = subprocess.run(['python3', 'tools/launch_detached_process.py'], check=False)\n"
        "if not proc.returncode:\n"
        "    print('job passed')\n",
    )

    with pytest.raises(PreflightError):
        check_background_launcher_rc_not_job_verdict(repo_root=tmp_path, strict=True)


def test_live_repo_has_no_launcher_rc_job_success_violations() -> None:
    assert check_background_launcher_rc_not_job_verdict() == []
