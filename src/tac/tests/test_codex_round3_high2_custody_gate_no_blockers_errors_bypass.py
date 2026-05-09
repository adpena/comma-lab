"""Tests for codex round-3 HIGH 2 fix — Check #130 no longer accepts bare `blockers`/`errors`.

Bug class (codex round-3 HIGH 2, 2026-05-09): the previous
``_TAG_GRADE_LOCAL_VALIDATOR_TOKENS`` accept-list included the bare
identifiers ``"blockers"`` and ``"errors"``. Any unrelated
``blockers = []`` / ``errors = []`` variable in the +/-6 line window
silently waived the tag-only custody gate. The strict gate failed open.

Fix: remove ``"blockers"`` and ``"errors"`` from the accept-list.
Require a CONCRETE validator call OR an explicit
``archive_sha256`` / ``sha256_file(`` reference OR a same-line
``# CUSTODY_VALIDATOR_OK:<reason>`` waiver.

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_tag_only_custody_validation,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Negative tests: unrelated blockers/errors must NOT satisfy the gate ──


def test_unrelated_blockers_var_does_not_satisfy_custody_gate(tmp_path):
    """A predicate with `blockers = []` nearby but NO concrete validator
    call MUST fail strict.

    Pre-fix bug: the accept-list contained ``"blockers"``, so this exact
    pattern silently passed even though it's a tag-only promotion path
    with no real custody check.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "bad_blockers.py").write_text(
        "def maybe_promote(payload):\n"
        "    blockers = []  # unrelated upstream collection\n"
        "    if not payload.get('archive_path'):\n"
        "        blockers.append('missing_path')\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    # No real SHA, no real validator call\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True  # PROMOTE without custody — the bug\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_blockers.py" in x]
    assert len(matches) == 1, (
        "post-fix: bare `blockers = []` in the window MUST NOT satisfy "
        f"the custody gate. Got {len(matches)} match(es): {v}"
    )


def test_unrelated_errors_var_does_not_satisfy_custody_gate(tmp_path):
    """A predicate with `errors = []` nearby but NO concrete validator
    call MUST fail strict.

    Same META as `blockers` — bare `errors` was an accept-list entry
    that silently let tag-only paths through.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "bad_errors.py").write_text(
        "def maybe_promote(payload):\n"
        "    errors = []  # unrelated parser collection\n"
        "    if not isinstance(payload, dict):\n"
        "        errors.append('not_dict')\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    # No real SHA, no real validator call\n"
        "    if evidence_grade in {'contest-cuda', 'A++'}:\n"
        "        return True  # PROMOTE without custody — the bug\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_errors.py" in x]
    assert len(matches) == 1, (
        "post-fix: bare `errors = []` in the window MUST NOT satisfy "
        f"the custody gate. Got {len(matches)} match(es): {v}"
    )


# ── Positive tests: real validator calls still pass ──


def test_concrete_validate_custody_call_satisfies_gate(tmp_path):
    """A predicate with `validate_custody(...)` in the +/-6 window MUST pass.

    Concrete validator calls are the canonical accept signal; this is
    untouched by the HIGH 2 fix.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_validate.py").write_text(
        "def maybe_promote(payload):\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        validate_custody(payload)  # concrete validator call\n"
        "        return True\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_validate.py" not in x for x in v)


def test_archive_sha256_in_window_satisfies_gate(tmp_path):
    """An `archive_sha256` reference in the +/-6 window MUST pass.

    `archive_sha256` is the canonical promotion field; every promotion
    path routes through it.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_sha.py").write_text(
        "def maybe_promote(payload):\n"
        "    archive_sha256 = payload.get('archive_sha256')\n"
        "    if not archive_sha256:\n"
        "        return False\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_sha.py" not in x for x in v)


def test_sha256_file_call_satisfies_gate(tmp_path):
    """A `sha256_file(...)` call in the +/-6 window MUST pass.

    The trailing `(` requires the call form; a stray `sha256_file`
    docstring would NOT silently waive the gate.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_sha_file.py").write_text(
        "def maybe_promote(payload, archive_path):\n"
        "    actual = sha256_file(archive_path)\n"
        "    if actual != payload.get('expected_sha'):\n"
        "        return False\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_sha_file.py" not in x for x in v)


# ── Strict-mode round-trip ──


def test_strict_raises_on_unrelated_blockers_var(tmp_path):
    """Strict mode must raise PreflightError for the bare-blockers pattern."""
    root = _make_repo(tmp_path)
    (root / "tools" / "bad.py").write_text(
        "def f(payload):\n"
        "    blockers = []\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True\n"
    )
    with pytest.raises(PreflightError, match="check_no_tag_only_custody_validation"):
        check_no_tag_only_custody_validation(
            repo_root=root, strict=True, verbose=False
        )


def test_same_line_waiver_still_accepts(tmp_path):
    """The `# CUSTODY_VALIDATOR_OK:` waiver is the prescribed escape valve.

    This is the correct way for blocker-accumulator gates that fail-close
    on `sys.exit(...)` to opt out of the gate when the SHA verification
    is not in the +/-6 line window.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_waived.py").write_text(
        "def f(payload):\n"
        "    blockers = []\n"
        "    evidence_grade = payload.get('evidence_grade', '')\n"
        "    if evidence_grade in {'A', 'A++'}:  # CUSTODY_VALIDATOR_OK: blocker-accumulator with sys.exit on failure\n"
        "        blockers.append('non-promotable')\n"
        "    if blockers:\n"
        "        raise SystemExit('FATAL')\n"
    )
    v = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_waived.py" not in x for x in v)


# ── Live-repo sanity: the fix landed at 0 violations ──


def test_130_live_repo_clean_after_high2_fix():
    """Live-repo sanity: catalog #130 must remain at 0 violations after the
    HIGH 2 tightening, with the two real blocker-accumulator sites
    (lightning_dispatch_pr106_stack.py + meta_lagrangian_atom_ledger_adapter.py)
    annotated with same-line CUSTODY_VALIDATOR_OK waivers per the gate's
    prescribed escape valve.
    """
    v = check_no_tag_only_custody_validation(strict=False, verbose=False)
    assert v == [], (
        f"HIGH 2 fix: catalog #130 has {len(v)} violations:\n"
        + "\n".join(v[:5])
    )
