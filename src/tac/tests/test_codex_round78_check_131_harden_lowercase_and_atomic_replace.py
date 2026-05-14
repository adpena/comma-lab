# SPDX-License-Identifier: MIT
"""Tests for the codex round-8 MEDIUM in-place hardening of Catalog #131.

Bug class: codex round 8 MEDIUM (2026-05-09). The previous Catalog #131
implementation collected shared-state path bindings via an all-caps regex
only, missing lowercase variables (e.g. `state_path = Path('.omx/state/foo.json')`).
Separately, presence of any lock token in the 20-line lookback waived the
write — even without an atomic-replace pattern, leaving partial-read races
unguarded.

Fix: AST-based path binding for lowercase names, attribute paths
(`self.foo_path`), and Path-joins. Lock-token alone is no longer sufficient
to waive a direct `write_text/write_bytes` on a shared-state path; the
canonical helper invocation OR explicit transactional pattern (write to
`<path>.tmp` + `os.replace`) is required.

Memory: feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from tac.preflight import (
    _bare_write_collect_shared_vars,
    _bare_write_collect_shared_vars_ast,
    check_no_bare_writes_to_shared_state,
)


def _write(tmp_path: Path, rel: str, source: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(source))
    return p


# ── AST collector unit tests ──────────────────────────────────────────────


def test_ast_collector_finds_lowercase_binding() -> None:
    text = dedent("""
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert "state_path" in out


def test_ast_collector_finds_attribute_binding() -> None:
    text = dedent("""
        class Foo:
            def __init__(self):
                self.tracker_path = Path('.omx/state/active.json')
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert "self.tracker_path" in out


def test_ast_collector_finds_annotated_binding() -> None:
    text = dedent("""
        from pathlib import Path
        cache: Path = Path('.omx/state/foo.json')
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert "cache" in out


def test_ast_collector_finds_path_join_binding() -> None:
    text = dedent("""
        from pathlib import Path
        REPO_ROOT = Path('/x')
        base = REPO_ROOT / '.omx/state/foo.json'
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert "base" in out


def test_ast_collector_finds_fstring_binding() -> None:
    text = dedent("""
        name = 'foo'
        path_var = f'.omx/state/{name}.json'
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert "path_var" in out


def test_ast_collector_ignores_non_shared_paths() -> None:
    text = dedent("""
        path = Path('/tmp/random.json')
        cache_path = Path('/var/log/foo.log')
    """)
    out = _bare_write_collect_shared_vars_ast(text)
    assert out == set()


def test_combined_collector_finds_both_caps_and_lowercase() -> None:
    text = dedent("""
        ACTIVE_JOBS_PATH = Path('.omx/state/lightning_active_jobs.json')
        state_path = Path('.omx/state/foo.json')
    """)
    out = _bare_write_collect_shared_vars(text)
    assert "ACTIVE_JOBS_PATH" in out
    assert "state_path" in out


# ── Lowercase binding → write detection (the codex finding) ────────────────


def test_lowercase_binding_with_bare_write_is_violation(tmp_path: Path) -> None:
    """The codex finding: lowercase shared-state binding + direct write
    was previously invisible to #131."""
    _write(tmp_path, "src/tac/foo_lowercase.py", """
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')

        def main():
            data = "{}"
            state_path.write_text(data)
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) >= 1, v
    assert any("foo_lowercase.py" in vv for vv in v), v


def test_attribute_binding_with_bare_write_is_violation(tmp_path: Path) -> None:
    """`self.foo_path = ...` followed by `self.foo_path.write_text(...)`."""
    _write(tmp_path, "src/tac/foo_attribute.py", """
        from pathlib import Path

        class Tracker:
            def __init__(self):
                self.tracker_path = Path('.omx/state/lightning_active_jobs.json')

            def save(self, data):
                self.tracker_path.write_text(data)
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) >= 1, v


# ── Atomic-replace requirement (codex round 8 MEDIUM) ─────────────────────


def test_lock_alone_no_atomic_replace_is_violation(tmp_path: Path) -> None:
    """Locked direct write_text WITHOUT atomic replace pattern is now a
    violation — partial reads are visible even under lock."""
    _write(tmp_path, "src/tac/foo_lock_no_replace.py", """
        import fcntl
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')

        def save(data):
            with open(state_path.with_suffix('.lock'), 'w') as fd:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
                # Direct write under lock — partial-read race remains
                state_path.write_text(data)
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) >= 1, v
    msg = "\n".join(v)
    assert "atomic-replace" in msg or "atomic replace" in msg


def test_lock_with_tmp_then_os_replace_is_allowed(tmp_path: Path) -> None:
    """The canonical transactional pattern is allowed."""
    _write(tmp_path, "src/tac/foo_atomic.py", """
        import fcntl
        import os
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')

        def save(data):
            with open(state_path.with_suffix('.lock'), 'w') as fd:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
                tmp = state_path.with_suffix('.tmp')
                tmp.write_text(data)
                os.replace(tmp, state_path)
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # The .tmp.write_text is NOT a shared-state path (`.tmp` doesn't match
    # any shared-state marker) — but if it did, the os.replace pattern
    # exempts it. The os.replace itself targets state_path which is shared
    # but the line_is_atomic_replace check exempts.
    assert v == [], v


def test_canonical_helper_call_in_lookback_is_allowed(tmp_path: Path) -> None:
    """A canonical-helper call in the lookback exempts the broader region."""
    _write(tmp_path, "src/tac/foo_helper.py", """
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')

        def save(data):
            update_active_jobs_locked(lambda r: r + [{"x": 1}])
            # Other writes that follow are OK because the helper owns
            # the transactional contract.
            state_path.write_text(data)
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


# ── Same-line waiver still respected ──────────────────────────────────────


def test_same_line_waiver_still_works(tmp_path: Path) -> None:
    _write(tmp_path, "src/tac/foo_waived.py", """
        from pathlib import Path
        state_path = Path('.omx/state/foo.json')

        def save(data):
            state_path.write_text(data)  # BARE_WRITE_OK:single-writer-test-fixture
    """)
    v = check_no_bare_writes_to_shared_state(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


# ── Real-repo smoke ───────────────────────────────────────────────────────


def test_real_repo_live_count_zero_after_harden() -> None:
    """The hardened #131 must remain at 0 violations on the real repo.

    If this fails, either:
    (1) the harden surfaced legitimate new violations that need fixing
        per CLAUDE.md (DO NOT skip — fix them), OR
    (2) the harden over-detected (false-positive) — narrow detection or
        add canonical-helper tokens / waivers.
    """
    v = check_no_bare_writes_to_shared_state(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #131 hardened must be at 0 violations on the real repo; "
        f"got {len(v)}:\n" + "\n".join(v[:10])
    )


def test_real_repo_strict_mode_does_not_raise() -> None:
    """If this fails, see test_real_repo_live_count_zero_after_harden above."""
    check_no_bare_writes_to_shared_state(strict=True, verbose=False)
