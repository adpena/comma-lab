"""Regression tests for PCC2 — `check_no_comment_only_contracts`.

Bug class: a placeholder/stub function carries a comment promising the
wrapper/deploy/caller will swap in the real implementation, but the wrapper
never actually performs the swap. The IMP cycle 0 = 1.98 metabug (38×
regression vs anchor 0.052) was rooted in this exact class.

Council deliberation:
  feedback_grand_council_pcc2_comment_only_contracts_20260430.md
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    _COMMENT_ONLY_CONTRACT_PATTERNS_AUDIT,
    _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    _find_enclosing_function_body,
    _has_backing_assertion,
    _scan_file_for_comment_only_contracts,
    check_no_comment_only_contracts,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


def _scaffold(tmp_path: Path, rel: str, content: str) -> Path:
    """Write `content` to tmp_path/scripts/rel so the scan-dirs match
    the repo's `scripts`/`experiments`/`src/tac`/`submissions/robust_current`
    structure."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


# ── Tight-pattern detection ─────────────────────────────────────────────────


def test_strict_pattern_deploy_swaps_in_detected(tmp_path: Path) -> None:
    p = _write(
        tmp_path, "stub.py",
        '''
        def f():
            # deploy script swaps in train_distill
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert len(findings) == 1
    rel_s, lineno, snippet, backed = findings[0]
    assert "deploy script swaps in" in snippet
    assert backed is False  # no assert/raise in body


def test_strict_pattern_overrides_this_stub_detected(tmp_path: Path) -> None:
    p = _write(
        tmp_path, "stub.py",
        '''
        def f():
            """Doc.

            The deploy script OVERRIDES this stub by calling X.
            """
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert any("OVERRIDES this stub" in s for _, _, s, _ in findings)


def test_strict_pattern_the_deploy_script_does_detected(
    tmp_path: Path,
) -> None:
    p = _write(
        tmp_path, "stub.py",
        '''
        def f():
            # we don't claim score progress here — the deploy script does the work
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert len(findings) >= 1


def test_strict_pattern_wrapper_script_handles_detected(
    tmp_path: Path,
) -> None:
    p = _write(
        tmp_path, "stub.py",
        '''
        def f():
            # wrapper script handles the dispatch
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert len(findings) >= 1


def test_strict_pattern_caller_is_responsible_NOT_in_strict(
    tmp_path: Path,
) -> None:
    """Q2 verdict: 'caller is responsible for X' is NOT in STRICT — too many
    legitimate API-docstring false positives in this codebase."""
    p = _write(
        tmp_path, "lib.py",
        '''
        def helper():
            """The caller is responsible for thresholding the result."""
            return tensor
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert findings == []


def test_audit_pattern_caller_is_responsible_IS_in_audit(
    tmp_path: Path,
) -> None:
    """Same content in audit mode — broader pattern set catches it."""
    p = _write(
        tmp_path, "lib.py",
        '''
        def helper():
            # caller is responsible for thresholding
            return tensor
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_AUDIT,
    )
    assert any("caller is responsible for" in s for _, _, s, _ in findings)


# ── Backing-assertion logic (Q3 verdict liberal scope) ──────────────────────


def test_backing_assertion_assert_in_function_body_satisfies(
    tmp_path: Path,
) -> None:
    """Q3 verdict (a): assert inside the enclosing function body."""
    p = _write(
        tmp_path, "stub.py",
        '''
        def f(x):
            # deploy script swaps in real_impl
            assert x is not None, "wrapper contract requires x"
            return x
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert len(findings) == 1
    assert findings[0][3] is True  # backed


def test_backing_assertion_raise_in_function_body_satisfies(
    tmp_path: Path,
) -> None:
    """Q3 verdict (a): raise inside the enclosing function body."""
    p = _write(
        tmp_path, "stub.py",
        '''
        def f(x):
            # deploy script swaps in real_impl
            if x is None:
                raise RuntimeError("wrapper contract violated")
            return x
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert findings[0][3] is True  # backed


def test_backing_assertion_within_50_lines_satisfies(
    tmp_path: Path,
) -> None:
    """Q3 verdict (b): backing assertion within ±50 lines (different
    function body, but close enough that a reviewer would see it)."""
    body_lines = "\n".join(
        ["    pass  # filler line {}".format(i) for i in range(20)]
    )
    p = _write(
        tmp_path, "stub.py",
        f'''
        def helper(x):
            assert x is not None  # the contract assertion
            return x

        def f():
            # deploy script swaps in real_impl
        {body_lines}
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert findings[0][3] is True  # backed (within ±50 lines)


def test_backing_assertion_check_star_anywhere_in_file_satisfies(
    tmp_path: Path,
) -> None:
    """Q3 verdict (c): check_<name>(...) call anywhere in the file."""
    p = _write(
        tmp_path, "stub.py",
        '''
        def setup():
            check_wrapper_contract_satisfied()

        def f():
            # deploy script swaps in real_impl
            return 0

        def check_wrapper_contract_satisfied():
            return True
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert findings[0][3] is True  # backed via check_* sibling


def test_backing_assertion_truly_absent_unbacked(
    tmp_path: Path,
) -> None:
    """Negative case: no assert, no raise, no check_*. Must be unbacked."""
    p = _write(
        tmp_path, "stub.py",
        '''
        def f():
            # deploy script swaps in real_impl
            return 0
        ''',
    )
    findings = _scan_file_for_comment_only_contracts(
        p, tmp_path, _COMMENT_ONLY_CONTRACT_PATTERNS_STRICT,
    )
    assert len(findings) == 1
    assert findings[0][3] is False  # unbacked


# ── _find_enclosing_function_body helper ────────────────────────────────────


def test_find_enclosing_function_body_innermost_wins() -> None:
    import ast
    src = textwrap.dedent(
        '''
        def outer():
            def inner():
                x = 1  # line 4
                return x
            return inner

        '''
    )
    tree = ast.parse(src)
    rng = _find_enclosing_function_body(tree, lineno=4)
    assert rng is not None
    start, end = rng
    # Innermost is `inner` (lines 3-5 in the dedented source).
    assert start <= 4 <= end
    # `inner` body is smaller than `outer` body.
    outer_rng = _find_enclosing_function_body(tree, lineno=2)
    assert outer_rng is not None
    assert (end - start) < (outer_rng[1] - outer_rng[0])


def test_find_enclosing_function_body_returns_none_at_module_level() -> None:
    import ast
    src = "x = 1\ny = 2\n"
    tree = ast.parse(src)
    assert _find_enclosing_function_body(tree, lineno=1) is None


# ── Strict-mode integration ──────────────────────────────────────────────────


def test_strict_mode_raises_on_unbacked_violation(tmp_path: Path) -> None:
    """When strict=True and an unbacked finding exists, raise MetaBugViolation."""
    # Scaffold a fake repo layout.
    _scaffold(
        tmp_path, "experiments/bad.py",
        '''
        def f():
            # deploy script swaps in real_impl
            return 0
        ''',
    )
    with pytest.raises(MetaBugViolation, match="COMMENT-ONLY CONTRACT"):
        check_no_comment_only_contracts(
            strict=True, verbose=False, repo_root=tmp_path,
        )


def test_strict_mode_passes_when_all_findings_backed(tmp_path: Path) -> None:
    _scaffold(
        tmp_path, "experiments/good.py",
        '''
        def f(x):
            # deploy script swaps in real_impl
            assert x is not None
            return x
        ''',
    )
    violations = check_no_comment_only_contracts(
        strict=True, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_strict_mode_passes_on_empty_repo(tmp_path: Path) -> None:
    """No scan dirs exist — gracefully returns 0 violations."""
    violations = check_no_comment_only_contracts(
        strict=True, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


# ── Audit mode ──────────────────────────────────────────────────────────────


def test_audit_mode_does_not_raise_even_unbacked(tmp_path: Path) -> None:
    """Audit mode is informational only — never raises (audit broadens
    patterns; STRICT mode is the gate)."""
    _scaffold(
        tmp_path, "experiments/bad.py",
        '''
        def f():
            # caller is responsible for X
            return 0
        ''',
    )
    # Audit mode: even with strict=True (an operator misuse), still doesn't
    # raise because audit IS the broader check; the STRICT gate uses the
    # tight set. Document this property.
    violations = check_no_comment_only_contracts(
        strict=False, verbose=False, repo_root=tmp_path, audit=True,
    )
    # Audit mode returns the unbacked subset (same as non-audit returns
    # against the strict pattern set), but doesn't raise.
    # The 'caller is responsible for X' line gets caught by the audit
    # pattern set; whether it's backed depends on backing rules.
    assert isinstance(violations, list)


# ── Real-codebase live-count check ──────────────────────────────────────────


def test_real_codebase_live_count_zero() -> None:
    """The actual live codebase MUST be at 0 unbacked findings (the
    reason this check is being landed). If this fails, a new comment-only
    contract has been introduced and a backing assertion is required.

    This is the regression test for the IMP cycle 0 = 1.98 metabug class.
    """
    violations = check_no_comment_only_contracts(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"NEW COMMENT-ONLY CONTRACT INTRODUCED:\n"
        + "\n".join(f"  • {v}" for v in violations)
        + "\n\nFix: add a backing assertion (assert/raise/check_*) within "
        "±50 lines of the comment OR inside the enclosing function body. "
        "See feedback_grand_council_pcc2_comment_only_contracts_20260430.md."
    )


# ── _has_backing_assertion direct tests ─────────────────────────────────────


def test_has_backing_assertion_with_decorator() -> None:
    text = textwrap.dedent(
        '''
        @requires_wrapper_contract
        def f():
            # deploy script swaps in X
            return 0
        '''
    )
    lines = text.splitlines()
    import ast
    tree = ast.parse(text)
    # Comment is on line 4 (1-indexed after dedent + leading newline)
    assert _has_backing_assertion(text, lines, comment_lineno=4, tree=tree) is True


def test_has_backing_assertion_no_assertion_returns_false() -> None:
    text = textwrap.dedent(
        '''
        def f():
            # deploy script swaps in X
            x = 1
            y = 2
            return x + y
        '''
    )
    lines = text.splitlines()
    import ast
    tree = ast.parse(text)
    assert _has_backing_assertion(text, lines, comment_lineno=3, tree=tree) is False
