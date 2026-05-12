"""Checks 67/68/69 — proactive bug-detection regression tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tac.preflight as preflight
from tac.preflight import (
    MetaBugViolation,
    PreflightTimeoutError,
    check_lane_anchor_files_exist_locally,
    check_python_files_compile,
    check_shell_scripts_syntax_clean,
)


# ---------------- Check 67 (python-files-compile) ----------------

def _py_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "tac").mkdir(parents=True)
    return repo


def test_check_67_clean_python_passes(tmp_path):
    repo = _py_repo(tmp_path)
    (repo / "src" / "tac" / "good.py").write_text("def f(x): return x + 1\n")
    v = check_python_files_compile(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_67_syntax_error_detected(tmp_path):
    repo = _py_repo(tmp_path)
    (repo / "src" / "tac" / "broken.py").write_text("def f(x: return x\n")
    v = check_python_files_compile(repo_root=repo, strict=False, verbose=False)
    assert len(v) == 1
    assert "broken.py" in v[0]


def test_check_67_incremental_cache_invalidates_changed_file(tmp_path):
    repo = _py_repo(tmp_path)
    target = repo / "src" / "tac" / "cached.py"
    target.write_text("def f(x):\n    return x + 1\n")
    assert check_python_files_compile(repo_root=repo, strict=False, verbose=False) == []
    assert (repo / ".omx" / "cache" / "python_compile_success.json").exists()

    target.write_text("def f(x: return x\n# changed size invalidates cache\n")
    v = check_python_files_compile(repo_root=repo, strict=False, verbose=False)
    assert len(v) == 1
    assert "cached.py" in v[0]


def test_check_67_strict_raises(tmp_path):
    repo = _py_repo(tmp_path)
    (repo / "src" / "tac" / "broken.py").write_text("def f(x: return x\n")
    with pytest.raises(MetaBugViolation, match="FAIL TO COMPILE"):
        check_python_files_compile(repo_root=repo, strict=True, verbose=False)


def test_check_67_timeout_propagates_without_fake_compile_violation(tmp_path, monkeypatch):
    repo = _py_repo(tmp_path)
    (repo / "src" / "tac" / "slow.py").write_text("VALUE = 1\n")

    def timeout_compile(*_args, **_kwargs):
        raise PreflightTimeoutError("budget exhausted")

    monkeypatch.setattr(preflight, "compile", timeout_compile, raising=False)

    with pytest.raises(PreflightTimeoutError, match="budget exhausted"):
        check_python_files_compile(repo_root=repo, strict=True, verbose=False)


def test_preflight_parallel_runner_enabled_by_default(monkeypatch):
    monkeypatch.delenv("PACT_PREFLIGHT_ENABLE_PARALLEL", raising=False)
    assert preflight._preflight_parallel_enabled() is True


def test_preflight_parallel_runner_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PACT_PREFLIGHT_ENABLE_PARALLEL", "0")
    assert preflight._preflight_parallel_enabled() is False


def test_preflight_parallel_worker_count_is_bounded(monkeypatch):
    monkeypatch.delenv("PACT_PREFLIGHT_PARALLEL_WORKERS", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert preflight._preflight_parallel_worker_count() == 8

    monkeypatch.setenv("PACT_PREFLIGHT_PARALLEL_WORKERS", "2")
    assert preflight._preflight_parallel_worker_count() == 2

    monkeypatch.setenv("PACT_PREFLIGHT_PARALLEL_WORKERS", "999")
    assert preflight._preflight_parallel_worker_count() == 16

    monkeypatch.setenv("PACT_PREFLIGHT_PARALLEL_WORKERS", "bad")
    assert preflight._preflight_parallel_worker_count() == 8

    monkeypatch.delenv("PACT_PREFLIGHT_PARALLEL_WORKERS", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert preflight._preflight_parallel_worker_count() == 2

    monkeypatch.setenv("PACT_PREFLIGHT_PARALLEL_WORKERS", "bad")
    assert preflight._preflight_parallel_worker_count() == 2


def test_preflight_source_index_prewarm_is_opt_in(monkeypatch):
    monkeypatch.delenv("PACT_PREFLIGHT_PREWARM_SOURCE_INDEX", raising=False)
    assert preflight._preflight_source_index_prewarm_enabled() is False

    monkeypatch.setenv("PACT_PREFLIGHT_PREWARM_SOURCE_INDEX", "1")
    assert preflight._preflight_source_index_prewarm_enabled() is True


def test_gate5_runtime_closure_runs_in_fast_fail_block():
    source = inspect.getsource(preflight.preflight_all)
    gate5 = "check_gate5_runtime_closure(strict=True"
    assert source.count(gate5) == 1
    assert source.index(gate5) < source.index(
        "check_remote_archive_only_eval_custody_closure"
    )


def test_check_67_skip_pycache(tmp_path):
    repo = _py_repo(tmp_path)
    cache = repo / "src" / "tac" / "__pycache__"
    cache.mkdir()
    (cache / "garbage.py").write_text("invalid !!! python")
    v = check_python_files_compile(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_67_real_codebase_clean():
    v = check_python_files_compile(strict=False, verbose=False)
    assert v == [], f"unexpected: {v}"


# ---------------- Check 68 (shell-scripts-syntax) ----------------

def _sh_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    return repo


def test_check_68_clean_shell_passes(tmp_path):
    repo = _sh_repo(tmp_path)
    (repo / "scripts" / "good.sh").write_text("#!/bin/bash\necho hello\n")
    v = check_shell_scripts_syntax_clean(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_68_incremental_cache_invalidates_changed_script(tmp_path):
    repo = _sh_repo(tmp_path)
    target = repo / "scripts" / "cached.sh"
    target.write_text("#!/bin/bash\necho hello\n")
    assert check_shell_scripts_syntax_clean(
        repo_root=repo,
        strict=True,
        verbose=False,
    ) == []
    assert (repo / ".omx" / "cache" / "shell_syntax_clean.json").exists()

    target.write_text("#!/bin/bash\nif [ x = y ]; then\necho missing fi\n")
    v = check_shell_scripts_syntax_clean(
        repo_root=repo,
        strict=False,
        verbose=False,
    )
    assert len(v) == 1
    assert "cached.sh" in v[0]


def test_check_68_syntax_error_detected(tmp_path):
    repo = _sh_repo(tmp_path)
    (repo / "scripts" / "broken.sh").write_text(
        "#!/bin/bash\nif [ x = y ]; then\necho missing fi\n"
    )
    v = check_shell_scripts_syntax_clean(repo_root=repo, strict=False, verbose=False)
    assert len(v) == 1
    assert "broken.sh" in v[0]


def test_check_68_strict_raises(tmp_path):
    repo = _sh_repo(tmp_path)
    (repo / "scripts" / "broken.sh").write_text("if [ a = b ]; then\necho x\n")
    with pytest.raises(MetaBugViolation, match="bash -n"):
        check_shell_scripts_syntax_clean(repo_root=repo, strict=True, verbose=False)


def test_check_68_directory_named_dot_sh_ignored(tmp_path):
    repo = _sh_repo(tmp_path)
    (repo / "scripts" / "recovered_x.sh").mkdir()
    v = check_shell_scripts_syntax_clean(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_68_real_codebase_clean():
    v = check_shell_scripts_syntax_clean(strict=False, verbose=False)
    assert v == [], f"unexpected: {v}"


# ---------------- Check 69 (anchor files exist locally) ----------------

def test_check_69_anchor_present_passes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "experiments" / "results" / "lane_a_landed").mkdir(parents=True)
    (repo / "experiments" / "results" / "lane_a_landed" / "archive_lane_a.zip").write_bytes(b"x")
    (repo / "scripts" / "remote_lane_x.sh").write_text(
        'ANCHOR_LANE_A_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"\n'
    )
    v = check_lane_anchor_files_exist_locally(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_69_missing_anchor_detected(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "remote_lane_x.sh").write_text(
        'ANCHOR_RENDERER="experiments/results/nope/renderer.bin"\n'
    )
    v = check_lane_anchor_files_exist_locally(repo_root=repo, strict=False, verbose=False)
    assert len(v) == 1
    assert "nope/renderer.bin" in v[0]


def test_check_69_strict_raises(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "remote_lane_x.sh").write_text(
        'LANE_A_ARCHIVE="experiments/results/missing.zip"\n'
    )
    with pytest.raises(MetaBugViolation, match="DO NOT EXIST LOCALLY"):
        check_lane_anchor_files_exist_locally(repo_root=repo, strict=True, verbose=False)


def test_check_69_env_default_resolved(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    # FAKE_LANE_OK:test fixture path for anchor-resolution guard.
    (repo / "experiments" / "results" / "lane_a").mkdir(parents=True)
    (repo / "experiments" / "results" / "lane_a" / "x.bin").write_bytes(b"x")
    (repo / "scripts" / "remote_lane_x.sh").write_text(
        'ANCHOR_X="${ANCHOR_X:-experiments/results/lane_a/x.bin}"\n'
    )
    v = check_lane_anchor_files_exist_locally(repo_root=repo, strict=False, verbose=False)
    assert v == []


def test_check_69_real_codebase_clean():
    v = check_lane_anchor_files_exist_locally(strict=False, verbose=False)
    assert v == [], f"unexpected: {v}"


# ---------------- Check 70 (pytest --collect-only) ----------------

def test_check_70_real_codebase_collects_clean():
    """The actual pact test suite must collect without errors."""
    from tac.preflight import check_pytest_collection_clean
    v = check_pytest_collection_clean(strict=False, verbose=False)
    assert v == [], f"unexpected: {v}"
