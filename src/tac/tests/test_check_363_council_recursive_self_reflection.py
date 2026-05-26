# SPDX-License-Identifier: MIT
"""Tests for Catalog #363 STRICT preflight gate (Surface 3 of protocol landing).

Covers ``check_council_deliberation_has_empirical_verification_status``:

* live-repo regression guard (≤2 ceiling pre-backfill)
* synthetic post-cutoff council memo flagged when missing canonical tokens
* pre-cutoff council memo exempt by date filter
* 4 canonical taxonomy tokens accepted
* `empirical_verification_status` field reference accepted
* `recursive_self_reflection` / `Round 2/3` discipline indicator accepted
* `PROVISIONAL-PENDING-VERIFICATION` marker accepted
* `AssumptionEmpiricalVerification` canonical helper reference accepted
* same-line waiver semantics (rationale accepted / placeholder rejected /
  short rationale rejected)
* strict-mode raises with Catalog #363 message
* string repo_root accepted
* orchestrator wire-in warn-only regression guard
* Catalog #185 sister-callable regression guard
* Catalog #176 sister CLAUDE.md row regression guard

Mirrors the test pattern of
:mod:`src.tac.tests.test_check_292_grand_council_assumption_statements`.

Verified-against: canonical landing memo
council_recursive_self_reflection_protocol_landed_20260526T134200Z.md §6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_council_deliberation_has_empirical_verification_status,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


_T3_MEMO_FIXTURE_HEADER = """# T3 Grand Council — Test Fixture

**Date:** 2026-05-26T00:00:00Z

"""


def _write_council_memo(
    research_dir: Path,
    name: str,
    body: str,
) -> Path:
    """Write a synthetic council memo into research_dir."""
    research_dir.mkdir(parents=True, exist_ok=True)
    f = research_dir / name
    f.write_text(_T3_MEMO_FIXTURE_HEADER + body, encoding="utf-8")
    return f


# ──────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ──────────────────────────────────────────────────────────────────────


def test_live_repo_regression_guard():
    """Live repo violation count is bounded pre-backfill.

    Per the canonical landing memo, live count at landing is 1 (T3 council
    7d04474cb is the canonical bug-class anchor). Allow ≤5 ceiling for
    future sister landings that may briefly accumulate before backfill.
    """
    violations = check_council_deliberation_has_empirical_verification_status(
        strict=False, verbose=False
    )
    assert len(violations) <= 5, (
        f"live-repo Catalog #363 violations exceeded warn-only ceiling: "
        f"{len(violations)} > 5"
    )


# ──────────────────────────────────────────────────────────────────────
# Synthetic council memo scope + cutoff
# ──────────────────────────────────────────────────────────────────────


def test_synthetic_post_cutoff_memo_without_tokens_flagged(tmp_path: Path):
    """Post-cutoff memo lacking ALL canonical tokens is flagged."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Some unrelated body content with no taxonomy tokens.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "grand_council_test_20260526.md" in violations[0]
    assert "Catalog #363" in violations[0] or "empirical-verification-status" in violations[0]


def test_pre_cutoff_memo_exempt(tmp_path: Path):
    """Pre-2026-05-26 memo is exempt by date filter."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260525.md",
        "Pre-cutoff body content with no taxonomy tokens.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_post_cutoff_with_far_future_date_in_scope(tmp_path: Path):
    """A 2027-MM-DD memo is in scope (post-cutoff)."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20270101.md",
        "Body without canonical tokens.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ──────────────────────────────────────────────────────────────────────
# 4 canonical taxonomy tokens accepted
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("token", [
    "VERIFIED_VIA_SOURCE_INSPECTION",
    "VERIFIED_VIA_EMPIRICAL_ANCHOR",
    "INFERRED_FROM_DOMAIN_LITERATURE",
    "ASSUMED_AWAITING_VERIFICATION",
])
def test_canonical_taxonomy_token_accepted(tmp_path: Path, token: str):
    """Each of 4 canonical taxonomy tokens accepts the memo."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        f"Body mentioning {token}.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_taxonomy_token_case_insensitive(tmp_path: Path):
    """Lowercase variant of taxonomy token also accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body mentioning verified_via_source_inspection in lowercase.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────
# Sister discipline tokens
# ──────────────────────────────────────────────────────────────────────


def test_empirical_verification_status_field_token_accepted(tmp_path: Path):
    """Field name reference accepts the memo."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "We discuss `empirical_verification_status` per Catalog #363.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_recursive_self_reflection_token_accepted(tmp_path: Path):
    """Discipline-indicator token `recursive_self_reflection` accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "The recursive_self_reflection protocol applies.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_round_2_self_reflection_token_accepted(tmp_path: Path):
    """`Round 2 self-reflection` discipline indicator accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Round 2 self-reflection identified the gap.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_provisional_pending_verification_marker_accepted(tmp_path: Path):
    """Round 3 downgrade marker `PROVISIONAL-PENDING-VERIFICATION` accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Verdict status: PROVISIONAL-PENDING-VERIFICATION per Round 3 downgrade.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_assumption_empirical_verification_helper_token_accepted(tmp_path: Path):
    """Canonical helper class name reference accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "We route through AssumptionEmpiricalVerification dataclass.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────
# Waiver semantics
# ──────────────────────────────────────────────────────────────────────


def test_waiver_with_rationale_accepted(tmp_path: Path):
    """Non-placeholder waiver rationale accepts."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body without tokens. "
        "# COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:deferred_to_round_2_protocol_op_routable_per_catalog_363_landing",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_waiver_placeholder_rejected(tmp_path: Path):
    """`<rationale>` placeholder literal is rejected per Catalog #287 sister."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body without tokens. "
        "# COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:<rationale>",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_waiver_reason_placeholder_rejected(tmp_path: Path):
    """`<reason>` placeholder literal also rejected."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body without tokens. "
        "# COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:<reason>",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_waiver_short_rationale_rejected(tmp_path: Path):
    """Rationale <4 chars is rejected."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body without tokens. # COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:abc",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ──────────────────────────────────────────────────────────────────────
# Strict mode
# ──────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_with_catalog_363_message(tmp_path: Path):
    """Strict mode raises PreflightError citing Catalog #363."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body without tokens or waiver.",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_council_deliberation_has_empirical_verification_status(
            research_dir=research, repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #363" in str(exc_info.value)
    assert "empirical-verification-status" in str(exc_info.value)


def test_strict_mode_silent_on_clean(tmp_path: Path):
    """Strict mode does not raise when no violations."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "Body mentions VERIFIED_VIA_SOURCE_INSPECTION.",
    )
    # Should not raise.
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=True, verbose=False,
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────
# Other surface coverage
# ──────────────────────────────────────────────────────────────────────


def test_no_research_dir_silent(tmp_path: Path):
    """Missing research dir does not raise."""
    research = tmp_path / "nonexistent"
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_string_repo_root_accepted(tmp_path: Path):
    """Path-or-str repo_root signature accepts both."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "grand_council_test_20260526.md",
        "VERIFIED_VIA_SOURCE_INSPECTION token here.",
    )
    # Pass str (not Path) repo_root.
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=str(tmp_path), strict=False, verbose=False,
    )
    assert violations == []


def test_multi_violation_aggregation(tmp_path: Path):
    """Multiple flagged memos return multiple violation entries."""
    research = tmp_path / "research"
    _write_council_memo(research, "grand_council_a_20260526.md", "no tokens A")
    _write_council_memo(research, "grand_council_b_20260526.md", "no tokens B")
    _write_council_memo(research, "skunkworks_council_c_20260526.md", "no tokens C")
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 3


def test_filename_must_match_council_pattern(tmp_path: Path):
    """Unrelated .md files in research dir are NOT scanned."""
    research = tmp_path / "research"
    research.mkdir()
    f = research / "unrelated_report_20260526.md"
    f.write_text("body without council keyword", encoding="utf-8")
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_t3_grand_council_prefix_in_repo_research_in_scope(tmp_path: Path):
    """In-repo `.omx/research/t3_grand_council_*.md` (no feedback_ prefix) is in scope."""
    research = tmp_path / "research"
    _write_council_memo(
        research,
        "t3_grand_council_test_topic_20260526.md",
        "Body without canonical tokens.",
    )
    violations = check_council_deliberation_has_empirical_verification_status(
        research_dir=research, repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ──────────────────────────────────────────────────────────────────────
# META-meta sister regression guards
# ──────────────────────────────────────────────────────────────────────


def test_catalog_185_sister_callable():
    """Gate function callable via tac.preflight module globals (Catalog #185)."""
    import tac.preflight as preflight_mod
    assert callable(
        getattr(
            preflight_mod,
            "check_council_deliberation_has_empirical_verification_status",
            None,
        )
    )


def test_catalog_176_sister_claude_md_row_present():
    """CLAUDE.md catalog table contains the Catalog #363 row (Catalog #176)."""
    repo_root = Path(__file__).resolve().parents[3]
    claude_md = repo_root / "CLAUDE.md"
    assert claude_md.exists()
    body = claude_md.read_text(encoding="utf-8")
    # Per Catalog #176, the CLAUDE.md catalog table must reference the strict
    # gate function name. The Recursive self-reflection protocol amendment
    # subsection IS the canonical reference per Surface 4 design.
    assert "check_council_deliberation_has_empirical_verification_status" in body
    assert "Catalog #363" in body


def test_orchestrator_wire_in_warn_only_regression_guard():
    """preflight_all wires the gate at strict=False (WARN-ONLY).

    Per CLAUDE.md "Strict-flip atomicity rule" + the canonical landing memo
    Surface 3 initial-warn-only-rationale: strict-flip is planned AFTER
    backfill drives live count to 0; until then the orchestrator callsite
    MUST remain strict=False.
    """
    repo_root = Path(__file__).resolve().parents[3]
    preflight_py = repo_root / "src" / "tac" / "preflight.py"
    body = preflight_py.read_text(encoding="utf-8")
    # Find the orchestrator callsite. We expect "strict=False" within the
    # same function call. Use a substring locator pattern.
    target = "check_council_deliberation_has_empirical_verification_status("
    idx = body.find(target)
    assert idx >= 0, "orchestrator callsite missing"
    # Read up to 200 chars past the callsite start.
    window = body[idx:idx + 200]
    assert "strict=False" in window, (
        "orchestrator callsite must be strict=False per Strict-flip "
        "atomicity rule + canonical landing memo Surface 3"
    )
