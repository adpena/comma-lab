"""Tests for Catalog #136 — custody-validator accept lists must be CONCRETE only.

Defense-in-depth on codex round-3 HIGH 2 (2026-05-09): the round-3 fix patched
ONE accept-list (`_TAG_GRADE_LOCAL_VALIDATOR_TOKENS`). This META gate refuses
ANY future accept-list addition that re-introduces a generic-token bypass at
ANY ``_*VALIDATOR_TOKENS`` / ``_*ACCEPT_TOKENS`` / ``_*GUARD_TOKENS`` /
``_*GATE_TOKENS`` / ``_*CUSTODY_TOKENS`` / ``_*VALIDATOR_PATTERNS`` /
``_*VALIDATOR_FNS`` constant.

Memory: feedback_production_hardening_polish_defense_in_depth_landed_20260509.md.
Cross-ref Catalog #130 (call-site missing-validator) + #133 (#131 exempt-list audit).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_custody_gate_accept_tokens_concrete_only,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Catches every forbidden generic identifier ───────────────────────────


def test_136_catches_blockers(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "blockers",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad.py" in x]
    assert len(matches) == 1
    assert "blockers" in matches[0]


def test_136_catches_errors(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        '_X_ACCEPT_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "errors",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad.py" in x]
    assert len(matches) == 1
    assert "errors" in matches[0]


def test_136_catches_failures(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "bad.py").write_text(
        '_X_GUARD_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "failures",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert any("bad.py" in x and "failures" in x for x in v)


def test_136_catches_validations(tmp_path):
    root = _make_repo(tmp_path)
    (root / "experiments" / "bad.py").write_text(
        '_X_GATE_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "validations",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert any("bad.py" in x and "validations" in x for x in v)


def test_136_catches_warnings(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        '_X_CUSTODY_TOKENS = (\n'
        '    "warnings",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert any("bad.py" in x and "warnings" in x for x in v)


def test_136_catches_issues_problems_results_checks_messages_verdicts_reasons(tmp_path):
    """All remaining forbidden generics in one go."""
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        '_X_VALIDATOR_PATTERNS = (\n'
        '    "issues",\n'
        '    "problems",\n'
        '    "results",\n'
        '    "checks",\n'
        '    "messages",\n'
        '    "verdicts",\n'
        '    "reasons",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    bad_violations = [x for x in v if "bad.py" in x]
    assert len(bad_violations) == 7


# ── Accept clean concrete validator tokens ───────────────────────────────


def test_136_accepts_concrete_validator_tokens(tmp_path):
    """Concrete validator tokens (function-call patterns, sha256 references)
    are accepted."""
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "ok.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "validate_custody_verdict",\n'
        '    "posterior_update_locked",\n'
        '    "posterior_update(",\n'
        '    "is_promotable_exact_cuda_evidence",\n'
        '    "promotable_exact_cuda_evidence_blockers",\n'
        '    "sha256_file(",\n'
        '    "archive_sha256",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_136_accepts_function_call_pattern_with_blockers(tmp_path):
    """`promotable_exact_cuda_evidence_blockers` (a CONCRETE function name
    that happens to contain 'blockers') is accepted; only the BARE token
    `blockers` is forbidden."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "promotable_exact_cuda_evidence_blockers",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


# ── Out-of-scope patterns must NOT be flagged ────────────────────────────


def test_136_ignores_non_validator_constant(tmp_path):
    """A constant whose name doesn't match the accept-list pattern is
    out-of-scope; the gate must NOT flag it."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        '_GENERIC_TOKEN_BLOCKLIST = (\n'
        '    "blockers",\n'
        '    "errors",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_136_ignores_dict_with_blockers_key(tmp_path):
    """`{'blockers': True}` in a dict literal is not an accept-list entry."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        '_VALIDATOR_TOKENS_CONFIG = {\n'
        '    "blockers": True,\n'
        '    "errors": False,\n'
        '}\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_136_ignores_comments_and_docstrings(tmp_path):
    """Forbidden tokens in comments or docstrings are not flagged."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        '"""This module has VALIDATOR_TOKENS but only mentions blockers in docs."""\n'
        '# _VALIDATOR_TOKENS = ("blockers",)  # commented out\n'
        '_X_VALIDATOR_TOKENS = (\n'
        '    "validate_custody",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_136_ignores_tuple_with_string_containing_paren(tmp_path):
    """`"sha256_file("` contains a `(` but is a CONCRETE entry — must accept."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "sha256_file(",\n'
        '    "archive_sha256",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


# ── Waiver works ─────────────────────────────────────────────────────────


def test_136_per_entry_waiver_accepts(tmp_path):
    """`# ACCEPT_TOKENS_CONCRETE_OK:` per-entry waiver lets through a
    bare token if it really IS the contract."""
    root = _make_repo(tmp_path)
    (root / "tools" / "waived.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "validate_custody",\n'
        '    "blockers",  # ACCEPT_TOKENS_CONCRETE_OK: typed accessor name\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("waived.py" not in x for x in v)


def test_136_def_line_waiver_accepts(tmp_path):
    """`# ACCEPT_TOKENS_CONCRETE_OK:` on the assignment-target line waives
    the entire tuple."""
    root = _make_repo(tmp_path)
    (root / "tools" / "waived.py").write_text(
        '_X_VALIDATOR_TOKENS = (  # ACCEPT_TOKENS_CONCRETE_OK: legacy contract\n'
        '    "validate_custody",\n'
        '    "blockers",\n'
        '    "errors",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("waived.py" not in x for x in v)


# ── Strict-mode round-trip ───────────────────────────────────────────────


def test_136_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "bad.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "blockers",\n'
        ')\n'
    )
    with pytest.raises(
        PreflightError, match="check_custody_gate_accept_tokens_concrete_only"
    ):
        check_custody_gate_accept_tokens_concrete_only(
            repo_root=root, strict=True, verbose=False
        )


def test_136_test_files_excluded(tmp_path):
    """Test files (which legitimately mock the bug pattern) MUST be excluded."""
    root = _make_repo(tmp_path)
    (root / "tools" / "test_foo.py").write_text(
        '_X_VALIDATOR_TOKENS = (\n'
        '    "blockers",\n'
        '    "errors",\n'
        ')\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("test_foo.py" not in x for x in v)


def test_136_intake_clones_excluded(tmp_path):
    """Vendored public-PR intake clones MUST be excluded (per Check 109)."""
    root = _make_repo(tmp_path)
    (root / "experiments" / "results").mkdir()
    (root / "experiments" / "results" / "public_pr_intake_test").mkdir()
    (root / "experiments" / "results" / "public_pr_intake_test" / "src").mkdir()
    (root / "experiments" / "results" / "public_pr_intake_test" / "src" / "bad.py").write_text(
        '_X_VALIDATOR_TOKENS = ("blockers",)\n'
    )
    v = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root, strict=False, verbose=False
    )
    assert all("public_pr_intake_test" not in x for x in v)


# ── Live-repo sanity ─────────────────────────────────────────────────────


def test_136_live_repo_clean():
    """Live-repo sanity: catalog #136 must land at 0 violations.

    Round-3's fix already removed the only known instance (`blockers`/`errors`
    from `_TAG_GRADE_LOCAL_VALIDATOR_TOKENS`).
    """
    v = check_custody_gate_accept_tokens_concrete_only(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #136 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )
