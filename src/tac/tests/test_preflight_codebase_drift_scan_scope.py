# SPDX-License-Identifier: MIT
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import CodebaseDriftError, check_codebase_drift
import tac.preflight as preflight_mod


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


def test_codebase_drift_prefilter_skips_harmless_python_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/harmless.py", "VALUE = 1\n")
    original_parse = preflight_mod.ast.parse
    calls = 0

    def counting_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(preflight_mod.ast, "parse", counting_parse)

    violations = check_codebase_drift(
        strict=False,
        repo_root=repo,
        verbose=False,
    )

    assert violations == []
    assert calls == 0


def test_codebase_drift_historical_glob_exempt_file_not_flagged(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    # A named member of the frozen historical exemption set: the launch_*.py
    # NAME rule yields, and harmless content produces no scan violation.
    _write(repo / "experiments/launch_taper_ab.py", "VALUE = 1\n")

    violations = check_codebase_drift(strict=False, repo_root=repo, verbose=False)

    assert violations == []


def test_codebase_drift_new_launcher_still_refused(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    # Any NEW launch_*.py outside the frozen exemption set still refuses —
    # the exemption cannot grow without a reviewed preflight.py edit.
    _write(repo / "experiments/launch_brand_new_thing.py", "VALUE = 1\n")

    violations = check_codebase_drift(strict=False, repo_root=repo, verbose=False)

    assert any("launch_brand_new_thing.py" in v for v in violations)


def test_codebase_drift_historical_bash_exempt_skips_allowlist_rule(
    tmp_path: Path,
) -> None:
    repo = _stub_repo(tmp_path)
    _write(
        repo / "experiments/stage_wr1_realized_gate.sh",
        "#!/bin/bash\necho ok\n",
    )

    violations = check_codebase_drift(strict=False, repo_root=repo, verbose=False)

    assert violations == []


def test_codebase_drift_bash_exempt_still_content_scanned(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    # The exemption yields only the name/location rule; forbidden CONTENT
    # (the nohup+pgrep watcher pattern) inside an exempt file still refuses.
    _write(
        repo / "experiments/stage_wr1_realized_gate.sh",
        "#!/bin/bash\nnohup python train.py &\nwhile pgrep -f train.py; do sleep 5; done\n",
    )

    violations = check_codebase_drift(strict=False, repo_root=repo, verbose=False)

    assert any("watcher pattern" in v for v in violations)


def test_codebase_drift_non_exempt_bash_still_refused(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/random_helper.sh", "#!/bin/bash\necho hi\n")

    violations = check_codebase_drift(strict=False, repo_root=repo, verbose=False)

    assert any("random_helper.sh" in v for v in violations)


def test_codebase_drift_uses_rg_prefilter_for_python_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo(tmp_path)
    harmless = repo / "experiments" / "harmless.py"
    bad = repo / "experiments" / "source_bad.py"
    _write(harmless, "VALUE = 1\n")
    _write(bad, _NOHUP_PY)
    scanned: list[Path] = []
    original_scan = preflight_mod._scan_python_for_forbidden

    def fake_rg(root, dirs, regex, **kwargs):
        assert regex == preflight_mod._CODEBASE_DRIFT_PY_PREFILTER_RE
        return (bad.resolve(),)

    def recording_scan(path, *, source_index=None):
        scanned.append(path.resolve())
        return original_scan(path, source_index=source_index)

    monkeypatch.setattr(preflight_mod, "_rg_python_files_matching_regex", fake_rg)
    monkeypatch.setattr(preflight_mod, "_scan_python_for_forbidden", recording_scan)

    violations = check_codebase_drift(
        strict=False,
        repo_root=repo,
        verbose=False,
    )

    assert any("nohup" in violation for violation in violations)
    assert scanned == [bad.resolve()]
