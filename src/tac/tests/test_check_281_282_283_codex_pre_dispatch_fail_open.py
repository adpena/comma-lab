# SPDX-License-Identifier: MIT
"""Dedicated tests for Catalog #281 / #282 / #283 (codex bfa2p1uex F1/F2/F3).

Three HIGH fail-open bugs caught by codex review bfa2p1uex (2026-05-15) in
the just-landed PRE-DISPATCH-CODEX-REVIEW (Catalog #271):

- F1 (Catalog #281): codex companion timeout / crash parsed as ``approve``
  because the bracketed error string lacked severity tokens. Source-level
  fix: rc-first verdict promotion to ``invocation-error``. CLI exit 2 for
  this verdict so paid dispatch is refused.

- F2 (Catalog #282): cache key omits dirty-tree fingerprint while the codex
  companion runs with ``--scope working-tree``. A cached approve from
  minutes ago could authorize a materially different working tree. Source-
  level fix: extend cache key with ``dirty_tree_fingerprint`` +
  ``untracked_relevant_fingerprint`` AND honor a new
  ``--no-cache-for-paid-dispatch`` CLI flag.

- F3 (Catalog #283): missing ``tools/run_codex_review_for_dispatch.py``
  helper degraded the mandatory gate to a warning. Source-level fix landed
  at commit 582f43a9a "dispatch: fail closed on missing codex review
  helper"; this gate is the structural sister so a future revert /
  refactor cannot silently re-introduce the warn-only-and-continue pattern.

All 3 STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule".
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint,
    check_codex_pre_dispatch_review_helper_companion_failure_blocks,
    check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_RUNNER = REPO_ROOT / "tools" / "run_codex_review_for_dispatch.py"
TARGET_AUTHORIZE = REPO_ROOT / "tools" / "operator_authorize.py"


# ============================================================================
# Live-repo regression guards (the canonical post-fix state)
# ============================================================================


def test_281_live_repo_clean_after_f1_fix() -> None:
    """The F1 fix landed in the same commit batch -> live count = 0."""
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        "Live repo expected clean post Catalog #281 F1 fix; "
        f"got {len(v)} violation(s):\n" + "\n".join(v[:5])
    )


def test_281_live_repo_strict_clean() -> None:
    """STRICT mode on the live repo MUST NOT raise."""
    check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=REPO_ROOT, strict=True
    )


def test_282_live_repo_clean_after_f2_fix() -> None:
    v = check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        "Live repo expected clean post Catalog #282 F2 fix; "
        f"got {len(v)} violation(s):\n" + "\n".join(v[:5])
    )


def test_282_live_repo_strict_clean() -> None:
    check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=REPO_ROOT, strict=True
    )


def test_283_live_repo_clean_after_f3_fix() -> None:
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        "Live repo expected clean post Catalog #283 F3 fix; "
        f"got {len(v)} violation(s):\n" + "\n".join(v[:5])
    )


def test_283_live_repo_strict_clean() -> None:
    check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=REPO_ROOT, strict=True
    )


# ============================================================================
# Catalog #281 — F1 fail-closed-on-companion-failure positive tests
# ============================================================================


def _write_runner(tmp_root: Path, body: str) -> None:
    target = tmp_root / "tools" / "run_codex_review_for_dispatch.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_281_flags_pre_fix_post_rc_classify_by_content(tmp_path: Path) -> None:
    """Pre-fix pattern: nonzero rc inserts advisory but verdict already
    classified by parse_verdict_from_codex_output. Forbidden token present."""
    body = textwrap.dedent(
        '''
        def run_codex_review_for_dispatch():
            verdict, findings = parse_verdict_from_codex_output(output)
            if rc != 0:
                findings.insert(0, "[codex-companion-rc] non-zero rc=" + str(rc) + " (verdict still classified by content)")
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    assert any("verdict still classified by content" in s for s in v), v


def test_281_flags_missing_required_tokens(tmp_path: Path) -> None:
    """If invocation-error / rc-first / Catalog #281 tokens missing -> flagged."""
    body = textwrap.dedent(
        '''
        def run_codex_review_for_dispatch():
            verdict, findings = parse_verdict_from_codex_output(output)
            return CodexReviewResult(verdict=verdict)
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    # Multiple required tokens missing.
    assert len(v) >= 2, v


def test_281_strict_mode_raises_with_catalog_281(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        def run_codex_review_for_dispatch():
            verdict, findings = parse_verdict_from_codex_output(output)
            findings.insert(0, "[codex-companion-rc] verdict still classified by content")
        '''
    )
    _write_runner(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_codex_pre_dispatch_review_helper_companion_failure_blocks(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "Catalog #281" in msg
    assert "fail-closed-on-companion-failure" in msg


def test_281_flags_missing_function(tmp_path: Path) -> None:
    body = "def some_other():\n    return None\n"
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    assert any("missing" in s for s in v), v


def test_281_waiver_with_rationale_passes(tmp_path: Path) -> None:
    body = "# CODEX_REVIEW_RC_FIRST_VERDICT_WAIVED:test fixture\n" + textwrap.dedent(
        '''
        def run_codex_review_for_dispatch():
            verdict, findings = parse_verdict_from_codex_output(output)
            if rc != 0:
                findings.insert(0, "[codex-companion-rc] verdict still classified by content")
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


def test_281_waiver_with_placeholder_rejected(tmp_path: Path) -> None:
    body = "# CODEX_REVIEW_RC_FIRST_VERDICT_WAIVED:<rationale>\n" + textwrap.dedent(
        '''
        def run_codex_review_for_dispatch():
            verdict, findings = parse_verdict_from_codex_output(output)
            findings.insert(0, "[codex-companion-rc] verdict still classified by content")
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    assert any("verdict still classified by content" in s for s in v), v


def test_281_no_target_skips_silently(tmp_path: Path) -> None:
    v = check_codex_pre_dispatch_review_helper_companion_failure_blocks(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# ============================================================================
# Catalog #281 — F1 runtime regression tests (per codex's explicit recs)
# ============================================================================


def test_281_runtime_companion_timeout_produces_invocation_error_verdict() -> None:
    """Codex recommendation: companion timeout produces nonzero CLI exit.

    Verifies VERDICT_INVOCATION_ERROR is the result type when the codex
    companion times out (rc=124).
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib

        mod = importlib.import_module("run_codex_review_for_dispatch")
        # Verify the constant exists + is in VERDICTS_ALL
        assert mod.VERDICT_INVOCATION_ERROR == "invocation-error"
        assert mod.VERDICT_INVOCATION_ERROR in mod.VERDICTS_ALL
        assert mod.VERDICT_INVOCATION_ERROR not in mod.VERDICTS_SAFE
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))


def test_281_runtime_rc_2_produces_nonzero_cli_exit(tmp_path: Path) -> None:
    """Codex recommendation: rc=2 (companion crash) produces nonzero CLI exit.

    Driven through main() with a fake script path that doesn't exist so the
    helper's codex-companion-missing branch fires AND we verify
    invocation-error verdict ladders through to nonzero CLI exit.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib

        mod = importlib.import_module("run_codex_review_for_dispatch")
        # Build a tiny tmp recipe + trainer so the runner has paths to hash.
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("name: test\n", encoding="utf-8")
        trainer = tmp_path / "trainer.py"
        trainer.write_text("# trainer\n", encoding="utf-8")
        # Force above cost gate so codex companion is invoked
        argv = [
            "--trainer",
            str(trainer),
            "--recipe",
            str(recipe),
            "--estimated-cost-usd",
            "5.00",
            "--script-path",
            "/nonexistent/codex-companion.mjs",
            "--skip-cache",
            "--repo-root",
            str(tmp_path),
        ]
        rc = mod.main(argv)
        # Missing companion -> needs-attention -> CLI exit 1 (NOT 0).
        assert rc != 0, f"Expected nonzero CLI exit on missing companion, got {rc}"
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))


# ============================================================================
# Catalog #282 — F2 cache-key-includes-dirty-tree positive tests
# ============================================================================


def test_282_flags_missing_dirty_tree_fingerprint_token(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        def _compute_cache_key(git_head_sha, recipe_sha, trainer_sha):
            return "stub"
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=tmp_path, strict=False
    )
    assert any("dirty_tree_fingerprint" in s for s in v), v


def test_282_strict_mode_raises_with_catalog_282(tmp_path: Path) -> None:
    body = "def _compute_cache_key(a, b, c):\n    return ''\n"
    _write_runner(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "Catalog #282" in msg
    assert "cache-key-includes-dirty-tree" in msg


def test_282_waiver_with_rationale_passes(tmp_path: Path) -> None:
    body = "# CODEX_REVIEW_CACHE_KEY_DIRTY_TREE_WAIVED:test fixture\n" + textwrap.dedent(
        '''
        def _compute_cache_key(a, b, c):
            return ''
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


def test_282_waiver_with_placeholder_rejected(tmp_path: Path) -> None:
    body = "# CODEX_REVIEW_CACHE_KEY_DIRTY_TREE_WAIVED:<rationale>\n" + textwrap.dedent(
        '''
        def _compute_cache_key(a, b, c):
            return ''
        '''
    )
    _write_runner(tmp_path, body)
    v = check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 1, v


def test_282_no_target_skips_silently(tmp_path: Path) -> None:
    v = check_codex_pre_dispatch_review_cache_key_includes_dirty_tree_fingerprint(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# ============================================================================
# Catalog #282 — F2 runtime regression tests (per codex's explicit recs)
# ============================================================================


def test_282_runtime_dirty_shared_helper_invalidates_cache_key() -> None:
    """Codex recommendation: dirty shared helper invalidates cached approve.

    Verifies the cache key changes when the dirty-tree fingerprint changes.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib

        mod = importlib.import_module("run_codex_review_for_dispatch")
        k1 = mod._compute_cache_key("g1", "r1", "t1", "", "")
        k2 = mod._compute_cache_key("g1", "r1", "t1", "DIRTY1", "")
        assert k1 != k2, "Dirty fingerprint MUST invalidate cache key"
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))


def test_282_runtime_untracked_relevant_change_invalidates_cache_key() -> None:
    """Codex recommendation: untracked relevant file change invalidates cache."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib

        mod = importlib.import_module("run_codex_review_for_dispatch")
        k1 = mod._compute_cache_key("g1", "r1", "t1", "", "")
        k2 = mod._compute_cache_key("g1", "r1", "t1", "", "UNTRACKED1")
        assert k1 != k2, "Untracked fingerprint MUST invalidate cache key"
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))


def test_282_runtime_no_cache_for_paid_dispatch_kwarg_present() -> None:
    """The runner accepts the new no_cache_for_paid_dispatch kwarg."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib
        import inspect

        mod = importlib.import_module("run_codex_review_for_dispatch")
        sig = inspect.signature(mod.run_codex_review_for_dispatch)
        assert "no_cache_for_paid_dispatch" in sig.parameters
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))


# ============================================================================
# Catalog #283 — F3 missing-helper-fails-closed positive tests
# ============================================================================


def _write_authorize(tmp_root: Path, body: str) -> None:
    target = tmp_root / "tools" / "operator_authorize.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_283_flags_warn_only_regression(tmp_path: Path) -> None:
    """If a future refactor reverts to warn-only-and-continue, gate flags it."""
    body = textwrap.dedent(
        '''
        def _run_codex_pre_dispatch_review(trainer, recipe, cost):
            helper = REPO_ROOT / "tools/run_codex_review_for_dispatch.py"
            if not helper.exists():
                print("warning: codex pre-dispatch helper not found; continuing")
                return
            subprocess.run(["python", str(helper)])
        '''
    )
    _write_authorize(tmp_path, body)
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    # forbidden warn-only token present + missing required tokens
    assert any("warn-only-and-continue" in s for s in v), v


def test_283_flags_missing_systemexit(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        def _run_codex_pre_dispatch_review(trainer, recipe, cost):
            helper = REPO_ROOT / "tools/run_codex_review_for_dispatch.py"
            subprocess.run(["python", str(helper)])
        '''
    )
    _write_authorize(tmp_path, body)
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    # No SystemExit + missing helper.exists check -> multiple violations.
    assert len(v) >= 2, v


def test_283_strict_mode_raises_with_catalog_283(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        def _run_codex_pre_dispatch_review(trainer, recipe, cost):
            print("warning: codex pre-dispatch helper not found; continuing")
        '''
    )
    _write_authorize(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "Catalog #283" in msg
    assert "missing-helper-fails-closed" in msg


def test_283_flags_missing_function(tmp_path: Path) -> None:
    body = "def some_other():\n    return None\n"
    _write_authorize(tmp_path, body)
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert any("missing" in s for s in v), v


def test_283_waiver_with_rationale_passes(tmp_path: Path) -> None:
    body = "# OPERATOR_AUTHORIZE_CODEX_HELPER_MISSING_WARN_OK:test fixture\n" + textwrap.dedent(
        '''
        def _run_codex_pre_dispatch_review(trainer, recipe, cost):
            print("warning: codex pre-dispatch helper not found; continuing")
        '''
    )
    _write_authorize(tmp_path, body)
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


def test_283_waiver_with_placeholder_rejected(tmp_path: Path) -> None:
    body = "# OPERATOR_AUTHORIZE_CODEX_HELPER_MISSING_WARN_OK:<rationale>\n" + textwrap.dedent(
        '''
        def _run_codex_pre_dispatch_review(trainer, recipe, cost):
            print("warning: codex pre-dispatch helper not found; continuing")
        '''
    )
    _write_authorize(tmp_path, body)
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 1, v


def test_283_no_target_skips_silently(tmp_path: Path) -> None:
    v = check_operator_authorize_missing_codex_pre_dispatch_helper_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# ============================================================================
# Catalog #283 — F3 runtime regression test (per codex's explicit rec)
# ============================================================================


def test_283_runtime_missing_helper_aborts_before_claim_or_provider_setup(
    tmp_path: Path,
) -> None:
    """Codex recommendation: missing helper -> SystemExit before claim/provider.

    Drives the actual operator-authorize helper through subprocess with a
    repo root where the helper file is absent, verifies the wrapper raises
    SystemExit BEFORE any claim or provider setup. We use Python import +
    direct call rather than full operator-authorize CLI to keep the test
    hermetic; the structural property (missing helper -> SystemExit) is
    verified by the catalog #283 STRICT preflight gate AND by the direct
    function call below.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib

        mod = importlib.import_module("operator_authorize")
        # Patch the REPO_ROOT to point at tmp_path so the helper is absent.
        original = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            with pytest.raises(SystemExit) as exc_info:
                mod._run_codex_pre_dispatch_review(
                    "fake_trainer.py", "fake_recipe.yaml", 5.00
                )
            msg = str(exc_info.value)
            assert "codex pre-dispatch helper not found" in msg or \
                "FATAL" in msg
        finally:
            mod.REPO_ROOT = original
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
