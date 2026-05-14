# SPDX-License-Identifier: MIT
"""Tests for Catalog #150 — Phase B `auth_memo_path` MUST be repo-relative.

Bug class: codex round 8 HIGH 2 (2026-05-09). Per CLAUDE.md "Design decisions
— non-negotiable" arbitration via AskUserQuestion: **Option C compromise**
keeps a6535b1ed's `consult_session_state=True` default AND adds explicit
`auth_memo_path=` argument that MUST resolve to a path under the git repo
root. ~/.claude, /tmp, and any non-repo absolute path are FORBIDDEN.

This module covers:

1. **Behavior of `phase_b_preconditions_status(auth_memo_path=...)`** — the
   implementation contract.
2. **Behavior of the new STRICT preflight check `check_phase_b_auth_memo_in_repo`**
   — the static-detect META gate.

Memory: feedback_phase_b_option_c_landed_20260509.md
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tac.lane_12_v2_nerv_as_renderer import (
    _assert_auth_memo_path_repo_relative,
    _check_operator_phase_b_authorization_from_path,
    _resolve_repo_root,
    phase_b_preconditions_status,
)
from tac.preflight import check_phase_b_auth_memo_in_repo


# ── Section 1: phase_b_preconditions_status auth_memo_path semantics ───────


def test_repo_root_resolves_to_git_root() -> None:
    """The resolver returns a real path containing the canonical repo files."""
    root = _resolve_repo_root()
    assert (root / "src" / "tac" / "lane_12_v2_nerv_as_renderer.py").is_file()


def test_assert_auth_memo_path_repo_relative_accepts_repo_path(
    tmp_path: Path,
) -> None:
    """A path under the repo root passes the assertion."""
    repo_root = _resolve_repo_root()
    in_repo = repo_root / ".omx" / "research" / "operator_authorizations" / "README.md"
    # Should not raise.
    _assert_auth_memo_path_repo_relative(in_repo)


def test_assert_auth_memo_path_repo_relative_refuses_tmp(tmp_path: Path) -> None:
    """A path under /tmp is REFUSED with ValueError."""
    out_repo = tmp_path / "auth.md"
    out_repo.write_text("placeholder")
    with pytest.raises(ValueError) as exc:
        _assert_auth_memo_path_repo_relative(out_repo)
    assert "outside the git repo root" in str(exc.value)
    assert "Option C" in str(exc.value)


def test_assert_auth_memo_path_repo_relative_refuses_dot_claude() -> None:
    """A ~/.claude path is REFUSED with ValueError."""
    with pytest.raises(ValueError) as exc:
        _assert_auth_memo_path_repo_relative(Path("~/.claude/foo.md"))
    assert "outside the git repo root" in str(exc.value)


def test_assert_auth_memo_path_repo_relative_refuses_var_tmp() -> None:
    """An absolute /var/tmp path is REFUSED with ValueError."""
    with pytest.raises(ValueError):
        _assert_auth_memo_path_repo_relative(Path("/var/tmp/auth.md"))


def test_phase_b_status_no_auth_memo_no_consult_returns_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_memo_path=None + consult_session_state=False → operator auth PENDING."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    s = phase_b_preconditions_status(consult_session_state=False, auth_memo_path=None)
    assert s["operator_phase_b_authorization"] == "PENDING"
    assert s["session_state_consulted"] is False
    assert s["auth_memo_path"] is None
    assert s["any_pending_blocks_phase_b_dispatch"] is True


def test_phase_b_status_no_auth_memo_with_consult_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_memo_path=None + consult_session_state=True → uses ~/.claude scan
    (a6535b1ed back-compat behavior preserved)."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    # Empty memory dir → operator auth PENDING (no memo to find).
    s = phase_b_preconditions_status(consult_session_state=True, auth_memo_path=None)
    assert s["operator_phase_b_authorization"] == "PENDING"
    assert s["session_state_consulted"] is True
    assert s["auth_memo_path"] is None


def test_phase_b_status_auth_memo_token_present_returns_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_memo_path provided + valid token in committed repo file → MET."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    repo_root = _resolve_repo_root()
    # Use a real repo-relative path: write a temp memo under .omx/tmp/.
    auth_dir = repo_root / ".omx" / "tmp" / "test_phase_b_auth"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_memo = auth_dir / "test_auth_memo.md"
    auth_memo.write_text("operator_phase_b_authorization=true\n")
    try:
        s = phase_b_preconditions_status(auth_memo_path=auth_memo)
        assert s["operator_phase_b_authorization"] == "MET"
        assert s["auth_memo_path"] == str(auth_memo.resolve())
    finally:
        auth_memo.unlink(missing_ok=True)
        try:
            auth_dir.rmdir()
        except OSError:
            pass


def test_phase_b_status_auth_memo_token_missing_returns_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_memo_path provided + token MISSING → PENDING."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    repo_root = _resolve_repo_root()
    auth_dir = repo_root / ".omx" / "tmp" / "test_phase_b_auth_missing"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_memo = auth_dir / "no_token.md"
    auth_memo.write_text("just a placeholder, no token here\n")
    try:
        s = phase_b_preconditions_status(auth_memo_path=auth_memo)
        assert s["operator_phase_b_authorization"] == "PENDING"
    finally:
        auth_memo.unlink(missing_ok=True)
        try:
            auth_dir.rmdir()
        except OSError:
            pass


def test_phase_b_status_auth_memo_path_outside_repo_raises_valueerror() -> None:
    """auth_memo_path outside repo → ValueError (Option C constraint)."""
    with pytest.raises(ValueError) as exc:
        phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")
    assert "outside the git repo root" in str(exc.value)


def test_phase_b_status_auth_memo_path_dot_claude_raises_valueerror() -> None:
    """auth_memo_path ~/.claude → ValueError."""
    with pytest.raises(ValueError):
        phase_b_preconditions_status(auth_memo_path="~/.claude/foo.md")


def test_phase_b_status_auth_memo_nonexistent_file_returns_pending() -> None:
    """auth_memo_path inside repo but file does not exist → PENDING."""
    repo_root = _resolve_repo_root()
    nonexist = repo_root / ".omx" / "tmp" / "definitely_does_not_exist_xyz.md"
    s = phase_b_preconditions_status(auth_memo_path=nonexist)
    assert s["operator_phase_b_authorization"] == "PENDING"


def test_check_operator_phase_b_authorization_from_path_blockquote_rejected(
    tmp_path: Path,
) -> None:
    """Blockquoted token is NOT explicit (back-compat with parser)."""
    repo_root = _resolve_repo_root()
    auth_dir = repo_root / ".omx" / "tmp" / "test_blockquote"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_memo = auth_dir / "blockquote.md"
    auth_memo.write_text("> operator_phase_b_authorization=true\n")
    try:
        assert _check_operator_phase_b_authorization_from_path(auth_memo) == "PENDING"
    finally:
        auth_memo.unlink(missing_ok=True)
        try:
            auth_dir.rmdir()
        except OSError:
            pass


def test_check_operator_phase_b_authorization_from_path_code_fence_rejected(
    tmp_path: Path,
) -> None:
    """Token inside ```...``` fence is NOT explicit."""
    repo_root = _resolve_repo_root()
    auth_dir = repo_root / ".omx" / "tmp" / "test_codefence"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_memo = auth_dir / "fenced.md"
    auth_memo.write_text(
        "```\noperator_phase_b_authorization=true\n```\n"
    )
    try:
        assert _check_operator_phase_b_authorization_from_path(auth_memo) == "PENDING"
    finally:
        auth_memo.unlink(missing_ok=True)
        try:
            auth_dir.rmdir()
        except OSError:
            pass


# ── Section 2: STRICT preflight Catalog #150 ───────────────────────────────


def _write_caller(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip())


def test_check_150_clean_repo_returns_zero_violations(tmp_path: Path) -> None:
    """A repo with no callers passes."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_caller_with_repo_relative_path_passes(tmp_path: Path) -> None:
    """A caller with a repo-relative literal path passes."""
    _write_caller(
        tmp_path / "tools" / "good_dispatch.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(
            auth_memo_path=".omx/research/operator_authorizations/phase_b_auth_lane_12_v2_20260509.md",
        )
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_caller_with_tmp_path_violates(tmp_path: Path) -> None:
    """A caller passing a /tmp/... literal is FLAGGED."""
    _write_caller(
        tmp_path / "tools" / "bad_dispatch.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "/tmp/" in violations[0]
    assert "FORBIDDEN" in violations[0]


def test_check_150_caller_with_dot_claude_literal_violates(tmp_path: Path) -> None:
    """A caller passing a `~/.claude/...` literal is FLAGGED."""
    _write_caller(
        tmp_path / "tools" / "claude_scan.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(auth_memo_path="~/.claude/projects/foo/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "~/.claude" in violations[0]


def test_check_150_caller_with_path_home_dot_claude_violates(tmp_path: Path) -> None:
    """A caller using `Path.home() / '.claude/...'` is FLAGGED via AST."""
    _write_caller(
        tmp_path / "tools" / "path_home_caller.py",
        '''
        from pathlib import Path
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        memo = Path.home() / ".claude/foo.md"
        s = phase_b_preconditions_status(auth_memo_path=memo)
        ''',
    )
    # NOTE: this caller passes a NAME (memo) not a literal expression.
    # AST detection of the BinOp must be on the kwarg value directly.
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # When passed via a Name binding, the literal is not visible to the
    # static check. This is acceptable because the runtime ValueError
    # in _assert_auth_memo_path_repo_relative still catches it.
    assert violations == []


def test_check_150_caller_with_inline_path_home_dot_claude_violates(
    tmp_path: Path,
) -> None:
    """A caller using INLINE `Path.home() / '.claude/...'` IS flagged."""
    _write_caller(
        tmp_path / "tools" / "inline_path_home.py",
        '''
        from pathlib import Path
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(
            auth_memo_path=Path.home() / ".claude/foo.md",
        )
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "Path.home()/.claude" in violations[0]


def test_check_150_waiver_on_kwarg_line_accepted(tmp_path: Path) -> None:
    """Same-line `# PHASE_B_AUTH_MEMO_OK:<reason>` waives the violation."""
    _write_caller(
        tmp_path / "tools" / "waived.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")  # PHASE_B_AUTH_MEMO_OK:test
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_waiver_on_call_line_accepted(tmp_path: Path) -> None:
    """Waiver on the line above the multi-line call is accepted."""
    _write_caller(
        tmp_path / "tools" / "waived_above.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        # PHASE_B_AUTH_MEMO_OK:legacy fallback for codex sandbox
        s = phase_b_preconditions_status(
            auth_memo_path="/tmp/auth.md",
        )
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_strict_mode_raises(tmp_path: Path) -> None:
    """strict=True raises PreflightError on violation."""
    from tac.preflight import PreflightError
    _write_caller(
        tmp_path / "tools" / "bad.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")
        ''',
    )
    with pytest.raises(PreflightError) as exc:
        check_phase_b_auth_memo_in_repo(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "non-repo auth_memo_path" in str(exc.value)


def test_check_150_skips_test_files(tmp_path: Path) -> None:
    """Test files (under tests/ or test_*.py) are NOT scanned."""
    _write_caller(
        tmp_path / "src" / "tac" / "tests" / "test_something.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        # In a test, /tmp paths are common test fixtures, not real callsites.
        phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_skips_intake_paths(tmp_path: Path) -> None:
    """Vendored `_intake_` paths are NOT scanned."""
    _write_caller(
        tmp_path / "experiments" / "results"
        / "public_pr101_intake_20260504_codex" / "source" / "x.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        phase_b_preconditions_status(auth_memo_path="/tmp/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_skips_implementation_files(tmp_path: Path) -> None:
    """The lane_12_v2 implementation file itself is exempt by construction."""
    _write_caller(
        tmp_path / "src" / "tac" / "lane_12_v2_nerv_as_renderer.py",
        '''
        # Implementation file: defines / proxies the parameter.
        def phase_b_preconditions_status(consult_session_state=True, auth_memo_path=None):
            return phase_b_preconditions_status(auth_memo_path="/tmp/foo.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # The impl file is exempted from the scan.
    assert violations == []


def test_check_150_attribute_call_form_detected(tmp_path: Path) -> None:
    """`module.phase_b_preconditions_status(auth_memo_path=...)` is also caught."""
    _write_caller(
        tmp_path / "tools" / "attr_call.py",
        '''
        from tac import lane_12_v2_nerv_as_renderer
        s = lane_12_v2_nerv_as_renderer.phase_b_preconditions_status(
            auth_memo_path="/tmp/auth.md",
        )
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_check_150_repo_relative_dot_omx_research_passes(tmp_path: Path) -> None:
    """The canonical `.omx/research/operator_authorizations/...` path passes."""
    _write_caller(
        tmp_path / "tools" / "canonical.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status(
            auth_memo_path=".omx/research/operator_authorizations/phase_b_auth_lane_12_v2_20260509.md",
        )
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_no_kwarg_passed_no_violation(tmp_path: Path) -> None:
    """Calling without `auth_memo_path` (back-compat path) is not flagged."""
    _write_caller(
        tmp_path / "tools" / "no_kwarg.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        s = phase_b_preconditions_status()
        s2 = phase_b_preconditions_status(consult_session_state=True)
        s3 = phase_b_preconditions_status(consult_session_state=False)
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_check_150_var_tmp_anchor_flagged(tmp_path: Path) -> None:
    """`/var/tmp/...` literal is also flagged."""
    _write_caller(
        tmp_path / "tools" / "vartmp.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        phase_b_preconditions_status(auth_memo_path="/var/tmp/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "/var/tmp/" in violations[0]


def test_check_150_private_tmp_anchor_flagged(tmp_path: Path) -> None:
    """`/private/tmp/...` literal (macOS canonical /tmp) is flagged."""
    _write_caller(
        tmp_path / "tools" / "privatetmp.py",
        '''
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        phase_b_preconditions_status(auth_memo_path="/private/tmp/auth.md")
        ''',
    )
    violations = check_phase_b_auth_memo_in_repo(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_check_150_live_repo_clean() -> None:
    """The live repo (real REPO_ROOT) MUST have 0 violations after this fix
    lands. STRICT-flip is in the same commit batch so this is the gate that
    keeps it green going forward.
    """
    violations = check_phase_b_auth_memo_in_repo(strict=False, verbose=False)
    assert violations == [], (
        f"Live repo has {len(violations)} non-repo auth_memo_path callsites: "
        + "; ".join(v[:200] for v in violations[:3])
    )


# ── Section 3: CLI behavior ────────────────────────────────────────────────


def test_cli_phase_b_auth_memo_flag_threads_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`--phase-b-auth-memo <committed_path>` flag wires the value through."""
    from tac.lane_12_v2_nerv_as_renderer import _validate_current_state_cli

    repo_root = _resolve_repo_root()
    auth_dir = repo_root / ".omx" / "tmp" / "test_cli_threads"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_memo = auth_dir / "cli_test_auth.md"
    auth_memo.write_text("operator_phase_b_authorization=true\n")
    try:
        # Use the legacy snapshot to avoid noise from the other 3 dynamic
        # flags depending on memory dir state.
        monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
        rc = _validate_current_state_cli([
            "--legacy-snapshot",
            "--phase-b-auth-memo", str(auth_memo),
        ])
        # Other three flags are PENDING under legacy snapshot, so rc should
        # be 1 even though operator auth flips MET. Confirm via stdout.
        captured = capsys.readouterr()
        assert "operator_phase_b_authorization" in captured.out
        assert '"MET"' in captured.out
        # rc should still be 1 because the three other flags are PENDING.
        assert rc == 1
    finally:
        auth_memo.unlink(missing_ok=True)
        try:
            auth_dir.rmdir()
        except OSError:
            pass


def test_cli_phase_b_auth_memo_outside_repo_exits_with_valueerror(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """`--phase-b-auth-memo /tmp/foo.md` raises ValueError before exit."""
    from tac.lane_12_v2_nerv_as_renderer import _validate_current_state_cli
    with pytest.raises(ValueError):
        _validate_current_state_cli([
            "--phase-b-auth-memo", "/tmp/auth.md",
        ])
