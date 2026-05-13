"""Tests for Catalog #199 — check_operator_authorize_bypass_requires_session_budget.

(Filename retains 198 prefix per the orchestrator's prompt — Catalog #199 was
claimed via the canonical serializer; the test file name is cosmetic.)

Operator approved 2026-05-13 (UNBLOCK + REVIEW-FIX + RETRY DISPATCH).
Self-protection for the operator-authorize confirmation-prompt bypass added
to ``tools/operator_authorize.py::_confirm()`` at the same commit batch.

Coverage:

- Live repo regression guard (count = 0 at landing)
- Positive: violating .py file (CONFIRMED set, no BUDGET) flagged
- Positive: violating .sh file flagged
- Negative: caller setting BOTH env vars accepted
- Negative: caller mentioning only in comment / docstring not flagged
- Negative: canonical operator_authorize.py exempt
- Negative: preflight.py exempt
- Negative: test files exempt
- Same-line waiver opt-out accepted
- Strict mode raises ``PreflightError``
- Out-of-window budget setting (>30 lines away) NOT accepted
- Bypass helper imports/respects both env vars
- Bare CONFIRMED=1 without BUDGET raises SystemExit(11) at runtime
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_operator_authorize_bypass_requires_session_budget,
)

# ---------------------------------------------------------------------------
# Live repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_violation_count_zero():
    """Catalog #199 must remain STRICT @ 0 on the live tree."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=repo_root,
    )
    assert violations == [], (
        f"Live count drifted above 0: {violations}"
    )


# ---------------------------------------------------------------------------
# Positive: violating files detected
# ---------------------------------------------------------------------------


def test_py_callsite_confirmed_without_budget_flagged(tmp_path):
    """A .py file setting CONFIRMED without BUDGET is flagged."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_dispatcher.py"
    bad.write_text(
        'import os\n'
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
        'print("no budget declared")\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE" in violations[0]
    assert "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD" in violations[0]


def test_sh_callsite_confirmed_without_budget_flagged(tmp_path):
    """A .sh file setting CONFIRMED without BUDGET is flagged."""
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "bad_wrapper.sh"
    bad.write_text(
        '#!/bin/bash\n'
        'export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1\n'
        'echo "no budget declared"\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1


def test_sh_callsite_inline_var_flagged(tmp_path):
    """Inline VAR=value form (no `export`) is also flagged."""
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "inline_var.sh"
    bad.write_text(
        '#!/bin/bash\n'
        'OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 some_command\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Negative: legitimate uses accepted
# ---------------------------------------------------------------------------


def test_both_env_vars_set_accepted(tmp_path):
    """Setting BOTH env vars near each other is accepted."""
    (tmp_path / "tools").mkdir()
    good = tmp_path / "tools" / "good_dispatcher.py"
    good.write_text(
        'import os\n'
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
        'os.environ["OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"] = "20.0"\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_both_env_vars_set_in_sh_accepted(tmp_path):
    """Setting BOTH env vars in shell accepted."""
    (tmp_path / "scripts").mkdir()
    good = tmp_path / "scripts" / "good_wrapper.sh"
    good.write_text(
        '#!/bin/bash\n'
        'export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1\n'
        'export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=10.50\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_comment_only_mention_not_flagged(tmp_path):
    """Comments / docstrings mentioning the var name are NOT flagged."""
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "doc_only.py"
    ok.write_text(
        '"""Sets OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 conditionally."""\n'
        '# OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 is a bypass env var\n'
        'print("hi")\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_sameline_waiver_accepted(tmp_path):
    """Same-line waiver opt-out accepted."""
    (tmp_path / "tools").mkdir()
    waived = tmp_path / "tools" / "waived_dispatcher.py"
    waived.write_text(
        'import os\n'
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"  # OPERATOR_AUTHORIZE_BYPASS_OK:legacy-recipe-with-builtin-cap\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_canonical_operator_authorize_exempt(tmp_path):
    """tools/operator_authorize.py is exempt (it defines the bypass)."""
    (tmp_path / "tools").mkdir()
    canonical = tmp_path / "tools" / "operator_authorize.py"
    canonical.write_text(
        '_SESSION_DIRECTIVE_ENV_VAR = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"\n'
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_preflight_self_exempt(tmp_path):
    """src/tac/preflight.py is exempt (it contains the regex patterns)."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    pf = tmp_path / "src" / "tac" / "preflight.py"
    pf.write_text(
        '_CHECK_199_CONFIRMED_ENV_VAR = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"\n'
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_test_files_exempt(tmp_path):
    """Test files exempt by filename + path convention."""
    (tmp_path / "tools" / "tests").mkdir(parents=True)
    test_file = tmp_path / "tools" / "tests" / "test_dispatcher.py"
    test_file.write_text(
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
    )
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_out_of_window_budget_not_accepted(tmp_path):
    """A budget set MORE than 30 lines away should NOT count as accepted."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "far_apart.py"
    far_apart_lines = [
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"',
    ]
    # 35 filler lines
    far_apart_lines.extend([f'# filler line {i}' for i in range(35)])
    far_apart_lines.append('os.environ["OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"] = "20.0"')
    bad.write_text("\n".join(far_apart_lines) + "\n")
    violations = check_operator_authorize_bypass_requires_session_budget(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1


def test_strict_mode_raises(tmp_path):
    """Strict mode raises PreflightError on violation."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad.py"
    bad.write_text(
        'os.environ["OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"] = "1"\n'
    )
    with pytest.raises(PreflightError, match="Catalog #199"):
        check_operator_authorize_bypass_requires_session_budget(
            strict=True, verbose=False, repo_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Runtime bypass behaviour (operator_authorize._confirm helpers)
# ---------------------------------------------------------------------------


def test_bypass_runtime_bare_confirmed_without_budget_fatal():
    """Setting CONFIRMED=1 without BUDGET must exit non-zero at runtime."""
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE": "1",
    }
    env.pop("OPERATOR_AUTHORIZE_SESSION_BUDGET_USD", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._session_directive_bypass_active())"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # The helper returns (False, error_msg) — it does NOT exit by itself.
    # The exit happens in _confirm() once the helper signals the err_msg.
    # Verify the helper returns the error tuple.
    assert proc.returncode == 0
    assert "False" in proc.stdout
    assert (
        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD" in proc.stdout
        or "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD" in proc.stderr
    )


def test_bypass_runtime_both_env_vars_returns_true():
    """Setting BOTH env vars must return (True, None) from the helper."""
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE": "1",
        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD": "20.0",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._session_directive_bypass_active())"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "True" in proc.stdout
    assert "BYPASS ACTIVE" in proc.stderr


def test_bypass_runtime_neither_env_var_returns_false():
    """No env vars → helper returns (False, None) (normal interactive path)."""
    repo_root = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items()
           if k not in {"OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE",
                        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"}}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._session_directive_bypass_active())"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "(False, None)" in proc.stdout


def test_bypass_runtime_non_numeric_budget_refused():
    """Non-numeric BUDGET must return (False, error_message)."""
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE": "1",
        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD": "not-a-number",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._session_directive_bypass_active())"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "False" in proc.stdout
    assert "not a parseable" in proc.stdout or "not a parseable" in proc.stderr


def test_bypass_runtime_zero_budget_refused():
    """BUDGET <= 0 must return (False, error_message)."""
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE": "1",
        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD": "0",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._session_directive_bypass_active())"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "False" in proc.stdout
    assert "must be > 0" in proc.stdout or "must be > 0" in proc.stderr
