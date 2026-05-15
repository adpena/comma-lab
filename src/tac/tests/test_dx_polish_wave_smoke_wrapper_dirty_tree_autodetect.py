# SPDX-License-Identifier: MIT
"""Tests for DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-4).

Verifies the auto-detect-dirty-tree behavior in
``tools/run_modal_smoke_before_full.py``:

  * SMOKE phase auto-activates Catalog #202 paired-env bypass when the
    working tree is dirty.
  * SMOKE phase does NOT activate the bypass on a clean tree.
  * FULL phase NEVER auto-activates the bypass (only the smoke phase does).
  * `_count_dirty_paths` returns the porcelain row count.
  * Operator-set Catalog #202 paired-env vars are preserved through both
    phases (no regression of the existing manual bypass surface).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.run_modal_smoke_before_full import (
    _count_dirty_paths,
    _spawn_full_dispatch,
    _spawn_smoke_dispatch,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# _count_dirty_paths
# ---------------------------------------------------------------------------


def test_count_dirty_paths_returns_int() -> None:
    """Live-repo regression: helper returns a plain int."""

    n = _count_dirty_paths(REPO_ROOT)
    assert isinstance(n, int)
    assert n >= 0


def test_count_dirty_paths_zero_on_non_git(tmp_path: Path) -> None:
    """A non-git directory returns 0 (fail-safe)."""

    n = _count_dirty_paths(tmp_path)
    assert n == 0


# ---------------------------------------------------------------------------
# _spawn_smoke_dispatch — dirty-tree auto-relax
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_subprocess(monkeypatch):
    captured: list[dict] = []

    def fake_run(cmd, **kwargs):
        captured.append({"cmd": list(cmd), "env": dict(kwargs.get("env", {}))})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="✓ DISPATCHED via .spawn() - call_id=fc-test\ninstance_job_id=job-test\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_smoke_dispatch_dirty_tree_activates_catalog_202_bypass(
    fake_subprocess, monkeypatch, tmp_path
):
    """DX-4: dirty tree at smoke time -> Catalog #202 paired-env bypass active."""
    # Force `_count_dirty_paths` to report a dirty tree.
    import tools.run_modal_smoke_before_full as wrapper

    monkeypatch.setattr(wrapper, "_count_dirty_paths", lambda _root: 5)
    _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert len(fake_subprocess) == 1
    env = fake_subprocess[0]["env"]
    assert env.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK") == "1"
    assert env.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED") == "1"


def test_smoke_dispatch_clean_tree_does_not_activate_bypass(
    fake_subprocess, monkeypatch, tmp_path
):
    """DX-4: clean tree at smoke time -> no bypass injected."""
    import tools.run_modal_smoke_before_full as wrapper

    monkeypatch.setattr(wrapper, "_count_dirty_paths", lambda _root: 0)
    # Ensure no inherited env var leakage from the test runner.
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", raising=False
    )
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert len(fake_subprocess) == 1
    env = fake_subprocess[0]["env"]
    assert "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK" not in env
    assert "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED" not in env


def test_smoke_dispatch_preserves_operator_set_catalog_202_envs(
    fake_subprocess, monkeypatch, tmp_path
):
    """If operator already exported the paired-env bypass, smoke phase
    inherits both env vars regardless of dirty-tree count."""
    import tools.run_modal_smoke_before_full as wrapper

    monkeypatch.setattr(wrapper, "_count_dirty_paths", lambda _root: 0)
    monkeypatch.setenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "1")
    monkeypatch.setenv("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "1")
    _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    env = fake_subprocess[0]["env"]
    assert env.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK") == "1"
    assert env.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED") == "1"


# ---------------------------------------------------------------------------
# _spawn_full_dispatch — never auto-relax
# ---------------------------------------------------------------------------


def test_full_dispatch_dirty_tree_does_not_inject_bypass_env(
    monkeypatch, capsys
):
    """DX-4: FULL phase NEVER auto-injects the paired-env bypass into
    its subprocess env. The full canary's $5-15 spend justifies the
    strict clean-head check."""
    import tools.run_modal_smoke_before_full as wrapper

    monkeypatch.setattr(wrapper, "_count_dirty_paths", lambda _root: 7)
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", raising=False
    )
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    captured: list[dict] = []

    def fake_run(cmd, **kwargs):
        captured.append({"cmd": list(cmd), "env": dict(kwargs.get("env", {}) or {})})
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = _spawn_full_dispatch(
        Path("dummy_recipe.yaml"),
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert rc == 0
    # Capture the warning printed to stderr.
    err = capsys.readouterr().err
    assert "FULL phase does NOT auto-relax" in err
    # The subprocess env (when not explicitly passed) inherits from the
    # parent. We assert that the wrapper itself did NOT add bypass envs.
    # subprocess.run was called without env=; the captured env dict is
    # empty because we passed env={} default-falsy to dict().
    if captured and captured[0].get("env"):
        env = captured[0]["env"]
        # If env was inherited and the OPERATOR var is set in parent,
        # that's an operator override (OK). But the wrapper itself
        # MUST NOT have injected it.
        # Verify by checking the wrapper source contract via the
        # Catalog #238 strict gate.
        from tac.preflight_dx_polish_gates import (
            check_smoke_path_default_relaxes_clean_head,
        )

        violations = check_smoke_path_default_relaxes_clean_head(
            repo_root=REPO_ROOT
        )
        assert violations == [], "; ".join(violations)


def test_full_dispatch_clean_tree_does_not_print_warning(
    monkeypatch, capsys
):
    import tools.run_modal_smoke_before_full as wrapper

    monkeypatch.setattr(wrapper, "_count_dirty_paths", lambda _root: 0)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    _spawn_full_dispatch(
        Path("dummy_recipe.yaml"),
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    err = capsys.readouterr().err
    assert "FULL phase does NOT auto-relax" not in err
