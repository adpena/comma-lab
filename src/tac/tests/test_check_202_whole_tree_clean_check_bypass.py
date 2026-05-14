"""Tests for Catalog #202 — check_catalog_202_bypass_requires_paired_env_attestation.

STRICT preflight gate that refuses any state of ``tools/operator_authorize.py``
that drops the canonical paired-env-var bypass contract for
``--require-clean-head`` in the Modal dispatch path.

Empirical anchor: 2026-05-13 5-smoke wave session — the dirty-tree dispatch
blocker recurred 3 times because sister-subagent unrelated infrastructure
work touched files outside the explicit sentinel set, but
``tools/operator_authorize.py::_dispatch_modal`` hardcoded
``--require-clean-head`` so ``experiments/modal_train_lane.py`` refused
dispatch on ANY whole-tree dirt. Catalog #166's worker-side sentinel hash
check would have passed cleanly but the dispatch never fired.

Sister of Catalog #199 (paired-env-var pattern, parent template),
Catalog #166 (worker-side sentinel hash check that runs independently),
Catalog #201 (sentinel-vs-mount parity).

Coverage:
- Live repo regression guard (count = 0 at landing)
- AST gate: missing helper -> violation
- AST gate: helper missing intent env-var check -> violation
- AST gate: helper missing attestation env-var check -> violation
- AST gate: helper missing SystemExit -> violation
- AST gate: dispatch fn missing helper invocation -> violation
- AST gate: dispatch fn missing --require-clean-head literal -> violation
- AST gate: helper accepts module-level constant binding (canonical pattern)
- AST gate: helper accepts inline string literal pattern
- Same-line waiver opt-out accepted on def line
- Placeholder waiver `<reason>` rejected (no self-waiver)
- Strict mode raises ``PreflightError``
- Runtime helper: both env vars set -> returns True + LOUD banner to stderr
- Runtime helper: neither env var set -> returns False
- Runtime helper: bare intent without attestation -> SystemExit(12)
- _dispatch_modal: when bypass inactive, --require-clean-head appears in cmd
- _dispatch_modal: when bypass active, --require-clean-head omitted from cmd
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_catalog_202_bypass_requires_paired_env_attestation,
)


# ---------------------------------------------------------------------------
# Live repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_violation_count_zero():
    """Catalog #202 must remain STRICT @ 0 on the live tree."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=repo_root,
    )
    assert violations == [], (
        f"Live count drifted above 0: {violations}"
    )


# ---------------------------------------------------------------------------
# AST gate: helper completeness
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fakerepo"
    (root / "tools").mkdir(parents=True)
    return root


def _write_target(root: Path, body: str) -> None:
    (root / "tools" / "operator_authorize.py").write_text(body)


def test_helper_missing_flagged(tmp_path):
    """Missing helper triggers a violation."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run', '--require-clean-head']
                # helper missing!
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("missing" in s for s in v)


def test_helper_missing_intent_env_var_flagged(tmp_path):
    """Helper that doesn't read intent env var is flagged."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                # missing intent env var check entirely
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if not attest:
                    raise SystemExit(12)
                return True

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("intent env var" in s for s in v)


def test_helper_missing_attestation_env_var_flagged(tmp_path):
    """Helper that doesn't read attestation env var is flagged."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "")
                if not intent:
                    return False
                # missing attestation env var check entirely
                raise SystemExit(12)

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("attestation env var" in s for s in v)


def test_helper_missing_systemexit_flagged(tmp_path):
    """Helper that doesn't raise SystemExit on partial-config is flagged."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "")
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if intent and attest:
                    return True
                # silently returns False on partial-config — NOT allowed
                return False

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("SystemExit" in s for s in v)


def test_dispatch_fn_missing_helper_invocation_flagged(tmp_path):
    """`_dispatch_modal` that doesn't invoke helper is flagged."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "")
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if intent and not attest:
                    raise SystemExit(12)
                return bool(intent and attest)

            def _dispatch_modal(recipe):
                # helper not called — flag always passed
                cmd = ['modal', 'run', '--require-clean-head']
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("does NOT invoke" in s for s in v)


def test_dispatch_fn_missing_flag_literal_flagged(tmp_path):
    """`_dispatch_modal` that drops --require-clean-head literal is flagged."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "")
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if intent and not attest:
                    raise SystemExit(12)
                return bool(intent and attest)

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                # helper called, but flag literal dropped — bypass always active
                if not _whole_tree_clean_check_bypass_active():
                    pass
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert any("no longer references" in s for s in v)


def test_helper_accepts_module_constant_pattern(tmp_path):
    """Canonical pattern: ``os.environ.get(_INTENT_ENV_VAR, "")`` accepted."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            _INTENT = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
            _ATTEST = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get(_INTENT, "")
                if not intent:
                    return False
                attest = os.environ.get(_ATTEST, "")
                if not attest:
                    raise SystemExit(12)
                return True

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert v == []


def test_helper_accepts_inline_string_literal_pattern(tmp_path):
    """Inline literal pattern also accepted."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                intent = os.environ.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "")
                if not intent:
                    return False
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if not attest:
                    raise SystemExit(12)
                return True

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert v == []


def test_helper_accepts_subscript_pattern(tmp_path):
    """``os.environ["VAR"]`` subscript pattern also accepted."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:
                if "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK" not in os.environ:
                    return False
                intent = os.environ["OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"]
                attest = os.environ.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", "")
                if intent and not attest:
                    raise SystemExit(12)
                return bool(intent and attest)

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver mechanism
# ---------------------------------------------------------------------------


def test_sameline_waiver_on_helper_def_accepted(tmp_path):
    """Same-line waiver on helper def line opts out of the helper checks."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:  # CATALOG_202_BYPASS_LOGIC_OK:legacy-test-stub
                # missing all required logic
                return False

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run']
                if not _whole_tree_clean_check_bypass_active():
                    cmd.append('--require-clean-head')
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert v == []


def test_placeholder_waiver_rejected(tmp_path):
    """The literal ``<reason>`` placeholder must not self-waive."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _whole_tree_clean_check_bypass_active() -> bool:  # CATALOG_202_BYPASS_LOGIC_OK:<reason>
                return False

            def _dispatch_modal(recipe):
                cmd = ['modal', 'run', '--require-clean-head']
                return cmd
            """
        ).strip(),
    )
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    # Placeholder waiver does NOT opt out — gate still flags missing logic.
    assert len(v) >= 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_strict_mode_raises(tmp_path):
    """Strict mode raises PreflightError on violation."""
    root = _make_repo(tmp_path)
    _write_target(
        root,
        textwrap.dedent(
            """
            import os
            import sys

            def _dispatch_modal(recipe):
                return ['modal', 'run', '--require-clean-head']
            """
        ).strip(),
    )
    with pytest.raises(PreflightError, match="Catalog #202"):
        check_catalog_202_bypass_requires_paired_env_attestation(
            strict=True, verbose=False, repo_root=root,
        )


def test_missing_target_file_returns_empty(tmp_path):
    """If the target file is absent, gate returns [] (skip silently)."""
    root = tmp_path / "emptyrepo"
    root.mkdir()
    v = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False, verbose=False, repo_root=root,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Runtime helper behaviour
# ---------------------------------------------------------------------------


_INTENT_VAR = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
_ATTEST_VAR = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"

_SUBPROCESS_HELPER_PROBE = (
    "import sys; sys.path.insert(0, 'tools'); "
    "import operator_authorize as oa; "
    "print('RESULT:', oa._whole_tree_clean_check_bypass_active())"
)


def _clean_env(extra: dict | None = None) -> dict:
    env = {
        k: v for k, v in os.environ.items()
        if k not in {_INTENT_VAR, _ATTEST_VAR}
    }
    if extra:
        env.update(extra)
    return env


def test_runtime_neither_env_var_returns_false():
    """No env vars set -> helper returns False (normal default path)."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_HELPER_PROBE],
        cwd=repo_root,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "RESULT: False" in proc.stdout


def test_runtime_both_env_vars_returns_true_with_loud_banner():
    """Both env vars set -> True + LOUD banner to stderr."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_HELPER_PROBE],
        cwd=repo_root,
        env=_clean_env({_INTENT_VAR: "1", _ATTEST_VAR: "1"}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "RESULT: True" in proc.stdout
    assert "[OPERATOR-AUTHORIZE BYPASS]" in proc.stderr
    assert "Catalog #202" in proc.stderr


def test_runtime_bare_intent_without_attestation_systemexit_12():
    """Bare intent without attestation -> SystemExit(12)."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_HELPER_PROBE],
        cwd=repo_root,
        env=_clean_env({_INTENT_VAR: "1"}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 12
    assert "FATAL" in proc.stderr
    assert "Catalog #202" in proc.stderr


def test_runtime_intent_falsy_returns_false():
    """Intent flag set to '0'/'false'/'no' -> False (no SystemExit)."""
    repo_root = Path(__file__).resolve().parents[3]
    for falsy in ("0", "false", "no", ""):
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_HELPER_PROBE],
            cwd=repo_root,
            env=_clean_env({_INTENT_VAR: falsy}),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"Falsy intent {falsy!r} should NOT trigger SystemExit; "
            f"stderr={proc.stderr!r}"
        )
        assert "RESULT: False" in proc.stdout


def test_runtime_attestation_falsy_with_intent_true_systemexit_12():
    """Intent truthy + attestation falsy -> SystemExit(12)."""
    repo_root = Path(__file__).resolve().parents[3]
    for falsy in ("0", "false", "no"):
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_HELPER_PROBE],
            cwd=repo_root,
            env=_clean_env({_INTENT_VAR: "1", _ATTEST_VAR: falsy}),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 12, (
            f"Truthy intent + falsy attestation {falsy!r} must SystemExit(12); "
            f"stderr={proc.stderr!r}"
        )


# ---------------------------------------------------------------------------
# _dispatch_modal command construction
# ---------------------------------------------------------------------------


_DISPATCH_PROBE_TEMPLATE = """
import os, sys, subprocess
sys.path.insert(0, 'tools')
import operator_authorize as oa

# Monkey-patch subprocess.call to capture the cmd instead of executing modal.
captured = {}
def fake_call(cmd, **kw):
    captured['cmd'] = list(cmd)
    return 0
oa.subprocess.call = fake_call

# Build a minimal Recipe stub. _dispatch_modal reads recipe.raw["modal"]
# (lane_script), recipe.remote_driver, recipe.lane_id, recipe.timeout_hours,
# recipe.gpu — all read-only attribute access, no Recipe class needed.
class _R:
    def __init__(self):
        self.raw = {
            'modal': {'lane_script': 'experiments/modal_train_lane.py'},
            'sentinel_files': [],
        }
        self.remote_driver = 'experiments/modal_train_lane.py'
        self.timeout_hours = 1.0
        self.gpu = 'T4'
    @property
    def lane_id(self):
        return 'lane_test_dummy'

oa._dispatch_modal(_R(), 'job-test', '')
print('CMD:', ' '.join(captured.get('cmd', [])))
"""


def test_dispatch_modal_includes_flag_when_bypass_inactive(tmp_path, monkeypatch):
    """When bypass is INACTIVE, --require-clean-head appears in the cmd."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _DISPATCH_PROBE_TEMPLATE],
        cwd=repo_root,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}; stdout={proc.stdout!r}; "
        f"stderr={proc.stderr!r}"
    )
    cmd_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("CMD:")]
    assert len(cmd_lines) == 1, f"unexpected stdout: {proc.stdout!r}"
    assert "--require-clean-head" in cmd_lines[0]


def test_dispatch_modal_omits_flag_when_bypass_active(tmp_path):
    """When bypass is ACTIVE, --require-clean-head is OMITTED from the cmd."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _DISPATCH_PROBE_TEMPLATE],
        cwd=repo_root,
        env=_clean_env({_INTENT_VAR: "1", _ATTEST_VAR: "1"}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}; stdout={proc.stdout!r}; "
        f"stderr={proc.stderr!r}"
    )
    cmd_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("CMD:")]
    assert len(cmd_lines) == 1, f"unexpected stdout: {proc.stdout!r}"
    assert "--require-clean-head" not in cmd_lines[0]
    # And the bypass banner fired
    assert "[OPERATOR-AUTHORIZE BYPASS]" in proc.stderr
    assert "Catalog #202" in proc.stderr
