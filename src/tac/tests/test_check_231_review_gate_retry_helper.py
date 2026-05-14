# SPDX-License-Identifier: MIT
"""Tests for Catalog #231 — review-gate retry-helper gate.

Per `feedback_duckdb_lock_fix_review_gate_hook_landed_20260514.md` operator
decision #1 (RECOMMENDED YES).

Refuses any future ``tools/review_gate_hook.py`` revision that re-introduces
a bare ``duckdb.connect()`` call outside the canonical retry helper.

Same-line waiver: ``# REVIEW_GATE_BARE_DUCKDB_OK:<reason>`` on the bare
duckdb.connect line.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_231_CANONICAL_HELPER_TOKENS,
    _CHECK_231_WAIVER_RE,
    _check_231_collect_bare_duckdb_connects,
    check_review_gate_hook_uses_retry_helper,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_repo(tmp_path):
    """Build a fake repo with a tools/ subdir."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    return repo


def _write_hook(repo, body):
    p = repo / "tools" / "review_gate_hook.py"
    p.write_text(body)
    return p


# ─── Live repo regression guard ──────────────────────────────────────


def test_live_repo_zero_violations():
    """The canonical fix is already present; live count = 0 STRICT @ 0."""
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert v == []


# ─── Positive: bare duckdb.connect outside helper is flagged ─────────


def test_bare_duckdb_connect_flagged(fake_repo):
    body = (
        "import duckdb\n"
        "def main():\n"
        "    conn = duckdb.connect('.omx/state/review_tracker.duckdb')\n"
        "    return conn\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    # Two violations: no canonical helper token AND bare connect call
    assert len(v) >= 1
    assert any("bare `duckdb.connect" in line for line in v)


def test_bare_connect_with_canonical_token_still_flagged(fake_repo):
    """If the file has a token reference BUT also a bare connect, the bare
    connect is still flagged (the gate checks both requirements)."""
    body = (
        "import duckdb\n"
        "# This file uses _connect_duckdb in some other context.\n"
        "def main():\n"
        "    conn = duckdb.connect('.omx/state/review_tracker.duckdb')\n"
        "    return conn\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("bare `duckdb.connect" in line for line in v)


# ─── Negative: canonical helper used, no bare connect ────────────────


def test_canonical_helper_no_bare_connect_clean(fake_repo):
    body = (
        "from review_tracker import _connect_duckdb\n"
        "def main():\n"
        "    conn = _connect_duckdb(read_only=True)\n"
        "    return conn\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert v == []


def test_canonical_helper_internal_connect_exempt(fake_repo):
    """A duckdb.connect INSIDE the canonical helper function body itself
    is exempt (the helper legitimately wraps the bare call)."""
    body = (
        "import duckdb\n"
        "from review_tracker import _connect_duckdb\n"
        "def _connect_duckdb_local():\n"
        "    # Wrapped retry helper - bare call IS the helper internals.\n"
        "    return duckdb.connect()\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    # The function name carries the canonical token so the internal call
    # is exempt. Plus a canonical helper token is imported.
    assert v == []


# ─── Docstring exclusion ──────────────────────────────────────────────


def test_docstring_mention_not_flagged(fake_repo):
    """A duckdb.connect string mention in a docstring is NOT a Call node."""
    body = (
        '"""Hook docstring.\n\n'
        "Historically a bare ``duckdb.connect(read_only=True)`` call here\n"
        "caused 80-300s stalls under sister-subagent writers.\n"
        '"""\n'
        "from review_tracker import _connect_duckdb\n"
        "def main():\n"
        "    return _connect_duckdb(read_only=True)\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert v == []


# ─── Waiver acceptance ────────────────────────────────────────────────


def test_same_line_waiver_accepted(fake_repo):
    body = (
        "import duckdb\n"
        "from review_tracker import _connect_duckdb\n"
        "def admin_only():\n"
        "    return duckdb.connect()  # REVIEW_GATE_BARE_DUCKDB_OK:admin-init-only\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert v == []


def test_placeholder_waiver_rejected(fake_repo):
    body = (
        "import duckdb\n"
        "from review_tracker import _connect_duckdb\n"
        "def admin_only():\n"
        "    return duckdb.connect()  # REVIEW_GATE_BARE_DUCKDB_OK:<reason>\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("bare `duckdb.connect" in line for line in v)


# ─── Strict mode ──────────────────────────────────────────────────────


def test_strict_raises_on_violation(fake_repo):
    body = (
        "import duckdb\n"
        "def main():\n"
        "    return duckdb.connect()\n"
    )
    _write_hook(fake_repo, body)
    with pytest.raises(PreflightError) as exc_info:
        check_review_gate_hook_uses_retry_helper(
            repo_root=fake_repo, strict=True, verbose=False
        )
    assert "Catalog #231" in str(exc_info.value)


def test_strict_silent_on_clean(fake_repo):
    body = (
        "from review_tracker import _connect_duckdb\n"
        "def main():\n"
        "    return _connect_duckdb()\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=True, verbose=False
    )
    assert v == []


# ─── AST helper unit tests ────────────────────────────────────────────


def test_ast_helper_collects_bare_call():
    src = (
        "import duckdb\n"
        "def f():\n"
        "    return duckdb.connect()\n"
    )
    calls = _check_231_collect_bare_duckdb_connects(src)
    assert len(calls) == 1


def test_ast_helper_exempts_canonical_function():
    src = (
        "import duckdb\n"
        "def _connect_duckdb():\n"
        "    return duckdb.connect()\n"
    )
    calls = _check_231_collect_bare_duckdb_connects(src)
    assert calls == []


def test_ast_helper_handles_syntax_error():
    src = "this is not python code {{{"
    calls = _check_231_collect_bare_duckdb_connects(src)
    assert calls == []


def test_ast_helper_ignores_unrelated_calls():
    src = (
        "import sqlite3\n"
        "def f():\n"
        "    return sqlite3.connect('x.db')\n"
    )
    calls = _check_231_collect_bare_duckdb_connects(src)
    assert calls == []


def test_ast_helper_multiple_bare_calls():
    src = (
        "import duckdb\n"
        "def a():\n"
        "    return duckdb.connect()\n"
        "def b():\n"
        "    return duckdb.connect('x.db')\n"
    )
    calls = _check_231_collect_bare_duckdb_connects(src)
    assert len(calls) == 2


# ─── Edge cases ───────────────────────────────────────────────────────


def test_no_hook_file_no_op(tmp_path):
    """If tools/review_gate_hook.py absent, gate is no-op."""
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_string_repo_root_accepted(fake_repo):
    body = (
        "from review_tracker import _connect_duckdb\n"
        "def main(): return _connect_duckdb()\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=str(fake_repo), strict=False, verbose=False
    )
    assert v == []


def test_canonical_tokens_constant():
    """Canonical tokens include the source-memo references."""
    assert "_connect_duckdb" in _CHECK_231_CANONICAL_HELPER_TOKENS
    assert "load_entities_from_json_snapshot" in _CHECK_231_CANONICAL_HELPER_TOKENS
    assert "_is_duckdb_lock_error" in _CHECK_231_CANONICAL_HELPER_TOKENS


def test_waiver_regex_well_formed():
    assert _CHECK_231_WAIVER_RE.search(
        "# REVIEW_GATE_BARE_DUCKDB_OK:admin-init-only"
    )
    assert not _CHECK_231_WAIVER_RE.search(
        "# REVIEW_GATE_BARE_DUCKDB_OK:<reason>"
    )


def test_verbose_output(fake_repo, capsys):
    body = (
        "from review_tracker import _connect_duckdb\n"
        "def main(): return _connect_duckdb()\n"
    )
    _write_hook(fake_repo, body)
    check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "review-gate-retry-helper" in captured.out


def test_missing_canonical_token_flagged(fake_repo):
    """Hook file with no canonical helper token gets a violation row."""
    body = (
        "def main():\n"
        "    print('hello')\n"
        "    return None\n"
    )
    _write_hook(fake_repo, body)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("does not import or reference" in line for line in v)


def test_preflight_all_wiring_strict():
    """Wired into preflight_all() with strict=True (STRICT-from-byte-one)."""
    from tac import preflight as pf
    source = Path(pf.__file__).read_text(encoding="utf-8")
    callsite_idx = source.find(
        "check_review_gate_hook_uses_retry_helper(strict=True"
    )
    assert callsite_idx > 0


def test_unreadable_hook_file_handled(fake_repo, monkeypatch):
    """A read error on the hook file returns a read-error violation."""
    _write_hook(fake_repo, "import duckdb\n")
    original_read = Path.read_text

    def _broken_read(self, *args, **kwargs):
        if "review_gate_hook" in str(self):
            raise OSError("simulated read failure")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read)
    v = check_review_gate_hook_uses_retry_helper(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("read error" in line for line in v)
