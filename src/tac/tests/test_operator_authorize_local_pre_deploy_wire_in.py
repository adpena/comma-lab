# SPDX-License-Identifier: MIT
"""Tests for the WIRE-AND-INTEGRATE-ALL local pre-deploy harness wire-in.

The 2026-05-15 wire-in adds `_run_local_pre_deploy_check` between
Catalog #152's `_validate_required_input_files` and `_native_dispatch_preflight`
so every operator-authorize dispatch runs the 30s harness BEFORE GPU spending.

Sister of Catalog #243 (the structural sister gate refusing dispatch wrappers
that bypass the canonical helper).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_operator_authorize():
    """Side-effect-free import of operator_authorize module for symbol checks."""
    import sys

    spec = importlib.util.spec_from_file_location(
        "_pact_oauth_under_test",
        REPO_ROOT / "tools" / "operator_authorize.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so dataclass machinery
    # can resolve cls.__module__ via sys.modules lookup.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_local_pre_deploy_check_helper_exists() -> None:
    """The canonical helper must be defined in operator_authorize.py."""
    mod = _import_operator_authorize()
    assert callable(getattr(mod, "_run_local_pre_deploy_check", None)), (
        "_run_local_pre_deploy_check helper missing from operator_authorize.py"
    )


def test_run_local_pre_deploy_check_invoked_in_main() -> None:
    """The helper must be invoked between Catalog #152 and _native_dispatch_preflight."""
    text = (REPO_ROOT / "tools" / "operator_authorize.py").read_text()
    assert "_run_local_pre_deploy_check(" in text, (
        "_run_local_pre_deploy_check must be invoked in operator_authorize.py main()"
    )
    # Verify the invocation comes before _native_dispatch_preflight to gate
    # the dispatch BEFORE provider-specific preflight (which may incur cost).
    invoke_idx = text.index("_run_local_pre_deploy_check(")
    native_idx = text.index("_native_dispatch_preflight(recipe)")
    assert invoke_idx < native_idx, (
        "_run_local_pre_deploy_check must be called BEFORE _native_dispatch_preflight "
        "so the harness fires before any provider-specific cost is incurred"
    )


def test_run_local_pre_deploy_check_invoked_after_required_input_files() -> None:
    """The wire-in must come AFTER Catalog #152's required-input-files check."""
    text = (REPO_ROOT / "tools" / "operator_authorize.py").read_text()
    # Look at the LAST occurrence of each (the main() body, not the def).
    cat_152_idx = text.rindex("_validate_required_input_files(str(trainer), recipe)")
    # _run_local_pre_deploy_check has both a def site and a call site; we
    # want the CALL site (the one in main()).
    wire_in_def_idx = text.index("def _run_local_pre_deploy_check")
    wire_in_call_idx = text.index("_run_local_pre_deploy_check(", wire_in_def_idx + 1)
    # find the ACTUAL call in main(), which is after the def end
    # (skip the recursive `_run_local_pre_deploy_check(` mention inside def)
    # The def is followed by the helper body; the main()-side call comes
    # after 'def _normalize_declared_local_path' (the next function).
    next_def_idx = text.index("def _normalize_declared_local_path", wire_in_def_idx)
    wire_in_call_idx_in_main = text.index(
        "_run_local_pre_deploy_check(", next_def_idx
    )
    assert cat_152_idx < wire_in_call_idx_in_main, (
        "_run_local_pre_deploy_check must be called AFTER _validate_required_input_files "
        "so missing-required-input is reported first (cheaper feedback loop)"
    )


def test_helper_passes_strict_flag() -> None:
    """The helper must invoke local_pre_deploy_check.py with --strict."""
    text = (REPO_ROOT / "tools" / "operator_authorize.py").read_text()
    helper_start = text.index("def _run_local_pre_deploy_check")
    helper_end = text.index("\n\n\n", helper_start)
    helper_body = text[helper_start:helper_end]
    assert '"--strict"' in helper_body, (
        "_run_local_pre_deploy_check must pass --strict to local_pre_deploy_check.py "
        "so harness violations abort the dispatch (vs warn-only)"
    )


def test_helper_supports_paired_env_bypass() -> None:
    """Bypass requires paired OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON."""
    text = (REPO_ROOT / "tools" / "operator_authorize.py").read_text()
    assert "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK" in text
    assert "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON" in text
    # Verify the paired-discipline guard
    helper_start = text.index("def _run_local_pre_deploy_check")
    helper_end = text.index("\n\n\n", helper_start)
    helper_body = text[helper_start:helper_end]
    assert "requires paired" in helper_body.lower(), (
        "Bypass must require paired-env discipline per CLAUDE.md "
        "(comment-only contracts are FORBIDDEN; the helper must enforce at runtime)"
    )


def test_helper_systemexit_on_bare_skip_without_reason(tmp_path: Path) -> None:
    """SKIP=1 without reason must raise SystemExit (paired-env discipline)."""
    mod = _import_operator_authorize()
    env = os.environ.copy()
    env["OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK"] = "1"
    env.pop("OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON", None)
    # Patch os.environ for the subprocess call below
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text("# fake\n")
    # Direct call to the helper with env mocked
    saved = {}
    for k in (
        "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK",
        "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON",
    ):
        if k in os.environ:
            saved[k] = os.environ[k]
            del os.environ[k]
    os.environ["OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK"] = "1"
    try:
        with pytest.raises(SystemExit) as ei:
            mod._run_local_pre_deploy_check(str(fake_trainer), "test_recipe")
        assert "paired" in str(ei.value).lower()
    finally:
        for k in (
            "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK",
            "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON",
        ):
            if k in os.environ:
                del os.environ[k]
        for k, v in saved.items():
            os.environ[k] = v


def test_helper_silent_on_paired_bypass(tmp_path: Path) -> None:
    """SKIP=1 with valid reason must return cleanly without invoking harness."""
    mod = _import_operator_authorize()
    fake_trainer = tmp_path / "fake_trainer.py"
    # Note: never created; if helper didn't bypass, it would either crash
    # or call the missing harness (which would WARN but still return). The
    # bypass path returns BEFORE the harness invocation.
    saved = {}
    for k in (
        "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK",
        "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON",
    ):
        if k in os.environ:
            saved[k] = os.environ[k]
            del os.environ[k]
    os.environ["OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK"] = "1"
    os.environ["OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON"] = (
        "test_paired_bypass_acceptance"
    )
    try:
        # Should NOT raise; should NOT invoke harness
        result = mod._run_local_pre_deploy_check(str(fake_trainer), "test_recipe")
        assert result is None
    finally:
        for k in (
            "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK",
            "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON",
        ):
            if k in os.environ:
                del os.environ[k]
        for k, v in saved.items():
            os.environ[k] = v


def test_helper_missing_harness_warns_does_not_crash(tmp_path: Path, monkeypatch) -> None:
    """If local_pre_deploy_check.py is missing, helper should WARN not crash."""
    mod = _import_operator_authorize()
    # Point REPO_ROOT into an empty tmp dir so the harness is not found
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text("# noop\n")
    # Should NOT raise (warn-only when harness missing)
    result = mod._run_local_pre_deploy_check(str(fake_trainer), "test_recipe")
    assert result is None


def test_live_repo_z3_trainer_passes_harness() -> None:
    """Live integration: Z3 v2 trainer (post-fix) passes the harness."""
    z3_trainer = REPO_ROOT / "experiments/train_substrate_z3_balle_hyperprior_bolton.py"
    if not z3_trainer.exists():
        pytest.skip(f"Z3 trainer not found at {z3_trainer}")
    harness = REPO_ROOT / "tools/local_pre_deploy_check.py"
    assert harness.exists(), f"harness missing: {harness}"
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable,
            str(harness),
            "--trainer",
            str(z3_trainer),
            "--recipe",
            "substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch",
            "--strict",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    # Z3 v2 trainer was the empirical anchor - post-fix should pass cleanly
    assert result.returncode == 0, (
        f"Z3 trainer harness failed with rc={result.returncode}\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
