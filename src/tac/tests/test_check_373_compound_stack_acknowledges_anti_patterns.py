# SPDX-License-Identifier: MIT
"""Catalog #373 STRICT preflight gate — compound stack proposal acknowledges anti-patterns.

CANONICAL-ANTI-PATTERNS REGISTRY 2026-05-28 Layer 3 self-protection tests.
Verifies:

  * Live-repo regression guard (no in-scope memo violates at landing)
  * Synthetic post-cutoff memo with compound-stack trigger + registered
    anti-pattern match WITHOUT acknowledgment is flagged
  * Synthetic post-cutoff memo WITH same-line waiver is accepted
  * Placeholder rationale `<rationale>` / `<reason>` / short / empty
    rejected per Catalog #287 sister discipline
  * Pre-cutoff memos exempt (Strict-flip atomicity rule)
  * Acknowledgment via citing anti_pattern_id + canonical_unwind_path
    accepted
  * Strict-mode raises with PreflightError mentioning Catalog #373
  * Orchestrator wire-in `strict=True` regression guard
  * Catalog #185 sister-callable regression guard
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_compound_stack_proposal_acknowledges_known_anti_patterns,
)


# Cutoff per the gate's contract.
_GATE_CUTOFF = "20260529"


# ──────────────────────────────────────────────────────────────────────
# Helper unit tests
# ──────────────────────────────────────────────────────────────────────


def test_extract_yyyymmdd_helper():
    from tac.preflight import _check_373_extract_yyyymmdd

    assert (
        _check_373_extract_yyyymmdd(Path("/foo/something_landed_20260530.md"))
        == "20260530"
    )
    assert (
        _check_373_extract_yyyymmdd(Path("/foo/no_landed_marker.md"))
        is None
    )


def test_post_cutoff_helper():
    from tac.preflight import _check_373_post_cutoff

    assert _check_373_post_cutoff("20260529") is True
    assert _check_373_post_cutoff("20260530") is True
    assert _check_373_post_cutoff("20260528") is False
    assert _check_373_post_cutoff("20250101") is False


def test_has_trigger_helper_canonical_tokens():
    from tac.preflight import _check_373_has_trigger

    assert _check_373_has_trigger("we propose a compound stack of foo + bar")
    assert _check_373_has_trigger("the compounding order matters here")
    assert _check_373_has_trigger("our stacking approach is...")
    assert _check_373_has_trigger("a stack-of-stacks composition")
    assert _check_373_has_trigger("stack of stacks (sister form)")
    assert _check_373_has_trigger("evaluate the combined codec output")
    # Non-trigger body
    assert not _check_373_has_trigger(
        "this memo discusses sensitivity analysis with no compound work"
    )


def test_valid_waiver_helper_rejects_placeholders():
    from tac.preflight import _check_373_has_valid_waiver

    assert _check_373_has_valid_waiver(
        "Some body... # ANTI_PATTERN_MATCH_INTENTIONAL_OK:"
        "deliberate test of canonical alternative\nmore body"
    )
    # Placeholder rejected
    assert not _check_373_has_valid_waiver(
        "# ANTI_PATTERN_MATCH_INTENTIONAL_OK:<rationale>"
    )
    assert not _check_373_has_valid_waiver(
        "# ANTI_PATTERN_MATCH_INTENTIONAL_OK:<reason>"
    )
    # Short rationale rejected
    assert not _check_373_has_valid_waiver(
        "# ANTI_PATTERN_MATCH_INTENTIONAL_OK:no"
    )
    # No waiver
    assert not _check_373_has_valid_waiver("no waiver present in this body")


def test_acknowledgment_helper():
    from tac.preflight import _check_373_has_acknowledgment

    # No matched ids = vacuous accept
    assert _check_373_has_acknowledgment("any body", [])
    # ID cited + canonical_unwind_path token
    assert _check_373_has_acknowledgment(
        "we cite anti-pattern lzma_on_already_brotli_saturated_compounding_v1 "
        "and apply its canonical_unwind_path: choose ONE entropy coder",
        ["lzma_on_already_brotli_saturated_compounding_v1"],
    )
    # ID cited but no acknowledgment token
    assert not _check_373_has_acknowledgment(
        "we cite anti-pattern lzma_on_already_brotli_saturated_compounding_v1 "
        "but ignore it",
        ["lzma_on_already_brotli_saturated_compounding_v1"],
    )
    # ID NOT cited
    assert not _check_373_has_acknowledgment(
        "we apply canonical_unwind_path but never cite the id",
        ["lzma_on_already_brotli_saturated_compounding_v1"],
    )


# ──────────────────────────────────────────────────────────────────────
# End-to-end gate behavior (synthetic memos in tmp_path)
# ──────────────────────────────────────────────────────────────────────


def _write_memo(research_dir: Path, name: str, body: str) -> Path:
    research_dir.mkdir(parents=True, exist_ok=True)
    p = research_dir / name
    p.write_text(body, encoding="utf-8")
    return p


def test_live_repo_regression_guard():
    """Live repo at landing must report 0 violations (cutoff exempts pre-2026-05-29)."""
    violations = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"Catalog #373 expected 0 violations at landing; got {len(violations)}: "
        + "\n".join(violations[:3])
    )


def test_no_research_dir(tmp_path):
    # Empty repo with no .omx/research/ -> 0 violations
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert out == []


def test_pre_cutoff_memo_exempt(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "foo_landed_20260520.md",
        "We propose a compound stack of brotli + lzma chained together "
        "with no acknowledgment.",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    # Pre-cutoff (20260520 < 20260529): exempt
    assert out == []


def test_post_cutoff_no_trigger_no_violation(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "foo_landed_20260530.md",
        "Sensitivity-map analysis with no compound work mentioned.",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert out == []


def test_post_cutoff_trigger_no_match_with_waiver_clean(tmp_path):
    """Body has a trigger but uses the canonical waiver to opt out.

    The Layer 1+2 matcher's token-overlap heuristic is intentionally
    BROAD (default min_confidence=0.5) so most compound-stack
    proposals will trigger SOME match per the Layer 1+2 builtins.
    The canonical opt-out is the waiver per Catalog #287 (or the
    acknowledgment cite per Layer 3 path (a) tested separately).
    """
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "waived_no_match_landed_20260530.md",
        "We propose a compound stack of fresh canonical primitives. "
        "# ANTI_PATTERN_MATCH_INTENTIONAL_OK:design memo only no actual "
        "compound stack work proposed; trigger phrase is incidental",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    # Waiver accepted -> no violation
    assert out == []


def test_post_cutoff_with_anti_pattern_match_flagged(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "bad_landed_20260530.md",
        # Body strongly references the canonical brotli+lzma anti-pattern
        # whose recurrence_condition phrase is "compounding entropy coders
        # that operate on similar redundancy domains"
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains. The expected ratio gain is small.",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    # Should match brotli_plus_lzma anti-pattern AND lzma_on_already_brotli
    # AND have no acknowledgment -> 1 violation
    assert len(out) >= 1
    assert "compound-stack proposal matches registered anti-pattern" in out[0]


def test_post_cutoff_with_waiver_accepted(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "waived_landed_20260530.md",
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains. # ANTI_PATTERN_MATCH_INTENTIONAL_OK:exploring "
        "regression in our specific substrate to ratify the anti-pattern "
        "with new empirical evidence per Layer 4 auto-recalibrator",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert out == []


def test_post_cutoff_placeholder_waiver_rejected(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "fake_waiver_landed_20260530.md",
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains. # ANTI_PATTERN_MATCH_INTENTIONAL_OK:<rationale>",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    # Placeholder rejected per Catalog #287 -> violation
    assert len(out) >= 1


def test_post_cutoff_short_rationale_rejected(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "short_landed_20260530.md",
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains. # ANTI_PATTERN_MATCH_INTENTIONAL_OK:no",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert len(out) >= 1


def test_post_cutoff_acknowledgment_accepted(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "ack_landed_20260530.md",
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains. We acknowledge anti_pattern_id=brotli_plus_lzma_chained_anti_pattern_v1 "
        "and apply its canonical_unwind_path: choose ONE high-quality "
        "entropy coder (brotli q=11) standalone. matched anti-pattern: "
        "brotli_plus_lzma_chained_anti_pattern_v1",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert out == []


def test_post_cutoff_strict_mode_raises(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "raise_landed_20260530.md",
        "We propose a compound stack chaining brotli + lzma — "
        "compounding entropy coders that operate on similar redundancy "
        "domains.",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_compound_stack_proposal_acknowledges_known_anti_patterns(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #373" in str(exc_info.value)


def test_strict_silent_on_clean(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "clean_landed_20260530.md",
        "Sensitivity-map analysis with no compound work.",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=True
    )
    assert out == []


def test_multi_violation_aggregation(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    for i in range(3):
        _write_memo(
            research_dir,
            f"multi_{i}_landed_20260530.md",
            "We propose a compound stack chaining brotli + lzma — "
            "compounding entropy coders that operate on similar redundancy "
            "domains.",
        )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=tmp_path, strict=False
    )
    assert len(out) >= 3


def test_string_repo_root_accepted(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    _write_memo(
        research_dir,
        "str_landed_20260530.md",
        "Sensitivity-map analysis no compound work.",
    )
    out = check_compound_stack_proposal_acknowledges_known_anti_patterns(
        repo_root=str(tmp_path), strict=False
    )
    assert out == []


def test_orchestrator_wire_in_strict_true():
    """Wave N+2 mandate: gate IS wired strict=True in preflight_all()."""
    pf = Path(__file__).resolve().parents[3] / "src" / "tac" / "preflight.py"
    if not pf.is_file():
        pytest.skip("preflight.py not found in expected layout")
    source = pf.read_text(encoding="utf-8")
    # Match the orchestrator callsite with strict=True (multiline tolerant)
    pattern = re.compile(
        r"check_compound_stack_proposal_acknowledges_known_anti_patterns\(\s*strict=True",
        re.MULTILINE,
    )
    assert pattern.search(source), (
        "Catalog #373 orchestrator callsite must wire `strict=True` per "
        "Wave N+2 mandate. Expected `check_compound_stack_proposal_acknowledges_known_anti_patterns(\\n strict=True, ...)`"
    )


def test_catalog_185_sister_callable_regression():
    """Catalog #185 sister: the gate function MUST be importable via globals.

    Catalog #185 META-meta-meta gate enforces that every CLAUDE.md
    catalog row entry's `check_<name>` function is callable via
    `tac.preflight` globals.
    """
    import tac.preflight as preflight_mod

    fn = getattr(
        preflight_mod,
        "check_compound_stack_proposal_acknowledges_known_anti_patterns",
        None,
    )
    assert callable(fn), (
        "Catalog #373 gate must be importable via tac.preflight globals per "
        "Catalog #185 sister regression"
    )


def test_canonical_helper_module_token_constants_pinned():
    from tac.preflight import (
        _CHECK_373_CUTOFF_YYYYMMDD,
        _CHECK_373_TRIGGER_TOKENS,
        _CHECK_373_WAIVER_TOKEN,
    )

    assert _CHECK_373_CUTOFF_YYYYMMDD == "20260529"
    assert _CHECK_373_WAIVER_TOKEN == "ANTI_PATTERN_MATCH_INTENTIONAL_OK"
    assert "compound stack" in _CHECK_373_TRIGGER_TOKENS
    assert "stacking" in _CHECK_373_TRIGGER_TOKENS
    assert "stack-of-stacks" in _CHECK_373_TRIGGER_TOKENS
