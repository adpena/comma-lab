from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import CodebaseDriftError, check_codebase_drift


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _stub_repo(root: Path) -> Path:
    for rel in (
        "scripts",
        "experiments",
        "src/tac/contrib",
        "src/tac/deploy",
        "src/tac/experiments",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


_NOHUP_PY = """
    import subprocess

    subprocess.run("nohup python experiments/train_renderer.py", shell=True)
"""


def test_codebase_drift_skips_experiments_results_python_artifacts(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/results/public_clone/source/bad.py", _NOHUP_PY)

    violations = check_codebase_drift(
        strict=False,
        repo_root=repo,
        verbose=False,
    )

    assert violations == []


def test_codebase_drift_still_scans_experiments_source_python(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/source_bad.py", _NOHUP_PY)

    violations = check_codebase_drift(
        strict=False,
        repo_root=repo,
        verbose=False,
    )

    assert any("nohup" in violation for violation in violations)
    with pytest.raises(CodebaseDriftError, match="CODEBASE DRIFT DETECTED"):
        check_codebase_drift(strict=True, repo_root=repo, verbose=False)


def test_codebase_drift_verbose_reports_scope_before_scan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _stub_repo(tmp_path)

    check_codebase_drift(strict=False, repo_root=repo, verbose=True)

    out = capsys.readouterr().out
    assert "[codebase-drift] scanning source launch surfaces" in out
    assert "skipping experiments/results artifacts" in out
    assert "[codebase-drift] OK:" in out
