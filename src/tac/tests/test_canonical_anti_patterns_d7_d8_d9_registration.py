# SPDX-License-Identifier: MIT
"""Dedicated tests for canonical rename wave D7+D8+D9 anti-patterns.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable + Slot EEE Class 2
forbidden pattern (`Tests-verify-constants-not-behavior`): every test
verifies BEHAVIOR — actual schema validation, actual register-then-retrieve
round-trip, actual canonical_unwind_path non-placeholder content, actual
ledger persistence + APPEND-ONLY semantics. If every test would still pass
when the builder body is replaced by `return canonical_markers`, the test
suite would be verifying constants not behavior; the tests below
deliberately exercise registry persistence + retrieval + invariants so a
no-op builder would fail.

Cross-references:
  * `tac.canonical_anti_patterns.d7_d8_d9_builders` — module under test
  * `tac.canonical_anti_patterns.registry` — canonical persistence layer
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * Catalog #344 — canonical anti-patterns + equations registry
  * Catalog #110 / #113 — APPEND-ONLY HISTORICAL_PROVENANCE
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_anti_patterns import (
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_DISCIPLINE,
    PARADIGM_RIGOR_LOSS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    AntiPattern,
    EmpiricalFalsification,
    build_all_d7_d8_d9_anti_patterns,
    build_canonical_default_plateau_substrate_disguised_as_class_shift_v1,
    build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1,
    build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1,
    get_anti_pattern_by_id,
    load_anti_patterns_strict,
    populate_d7_d8_d9_anti_patterns,
    query_anti_patterns,
)


# Canonical anti-pattern IDs under test (mirrors d7_d8_d9_builders module).
_D7_ID = "canonical_default_plateau_substrate_disguised_as_class_shift_v1"
_D8_ID = "micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1"
_D9_ID = "hnerv_pr95_language_anchoring_local_minima_perpetuation_v1"
_ALL_IDS = (_D7_ID, _D8_ID, _D9_ID)


# -----------------------------------------------------------------------------
# Builder behavior tests (per-anti-pattern construction)
# -----------------------------------------------------------------------------


def test_d7_builder_returns_valid_anti_pattern_with_canonical_id() -> None:
    """D7 builder constructs AntiPattern that passes ALL __post_init__ invariants."""
    ap = build_canonical_default_plateau_substrate_disguised_as_class_shift_v1()
    assert isinstance(ap, AntiPattern)
    assert ap.anti_pattern_id == _D7_ID
    # Severity per memo: HIGH (canonical default → 0.196-0.199 plateau is structural)
    assert ap.severity == SEVERITY_HIGH
    # Paradigm: discipline (substrate-engineering discipline failure per memo D7)
    assert ap.paradigm_class == PARADIGM_DISCIPLINE


def test_d8_builder_returns_valid_anti_pattern_with_canonical_id() -> None:
    """D8 builder constructs AntiPattern that passes ALL __post_init__ invariants."""
    ap = build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1()
    assert isinstance(ap, AntiPattern)
    assert ap.anti_pattern_id == _D8_ID
    # Severity per memo: HIGH (micro-optimization budget wasted is structural EV loss)
    assert ap.severity == SEVERITY_HIGH
    # Paradigm: discipline (optimization-budget discipline failure per memo D8)
    assert ap.paradigm_class == PARADIGM_DISCIPLINE


def test_d9_builder_returns_valid_anti_pattern_with_canonical_id() -> None:
    """D9 builder constructs AntiPattern that passes ALL __post_init__ invariants."""
    ap = build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1()
    assert isinstance(ap, AntiPattern)
    assert ap.anti_pattern_id == _D9_ID
    # Severity per memo: MEDIUM (language anchoring is documentation-discipline)
    assert ap.severity == SEVERITY_MEDIUM
    # Paradigm: rigor_loss (documentation-discipline rigor loss per memo D9)
    assert ap.paradigm_class == PARADIGM_RIGOR_LOSS


def test_build_all_returns_exactly_three_canonical_anti_patterns_in_order() -> None:
    """Aggregator returns the canonical 3-anti-pattern tuple D7+D8+D9 in order."""
    aps = build_all_d7_d8_d9_anti_patterns()
    assert len(aps) == 3
    assert [ap.anti_pattern_id for ap in aps] == [_D7_ID, _D8_ID, _D9_ID]


# -----------------------------------------------------------------------------
# Canonical schema invariant tests (BEHAVIOR — validates each anti-pattern
# actually passes the canonical AntiPattern.__post_init__ contract)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("anti_pattern_id,builder", [
    (_D7_ID, build_canonical_default_plateau_substrate_disguised_as_class_shift_v1),
    (_D8_ID, build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1),
    (_D9_ID, build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1),
])
def test_each_anti_pattern_has_non_empty_canonical_unwind_path(
    anti_pattern_id: str, builder
) -> None:
    """Per memo: canonical_unwind_path MUST be substantive operator-routable
    text, not a placeholder."""
    ap = builder()
    assert isinstance(ap.canonical_unwind_path, str)
    assert len(ap.canonical_unwind_path.strip()) >= 100, (
        f"{anti_pattern_id} canonical_unwind_path too short: "
        f"{len(ap.canonical_unwind_path)} chars; expected >=100"
    )
    # Per Catalog #287 placeholder rejection sister discipline: refuse
    # placeholder literals as the canonical unwind path content.
    lower = ap.canonical_unwind_path.lower()
    for placeholder in ("<rationale>", "<reason>", "tbd", "todo"):
        assert placeholder not in lower, (
            f"{anti_pattern_id} canonical_unwind_path contains placeholder "
            f"literal {placeholder!r}"
        )


@pytest.mark.parametrize("anti_pattern_id,builder", [
    (_D7_ID, build_canonical_default_plateau_substrate_disguised_as_class_shift_v1),
    (_D8_ID, build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1),
    (_D9_ID, build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1),
])
def test_each_anti_pattern_has_canonical_producers_or_consumers(
    anti_pattern_id: str, builder
) -> None:
    """Per AntiPattern invariant: orphan anti-patterns (zero producers AND
    zero consumers) are refused. Verifies each anti-pattern declares
    machine-readable consumer module-paths so the cathedral autopilot
    ranker can actually route to them."""
    ap = builder()
    assert ap.canonical_consumers, (
        f"{anti_pattern_id} has no canonical_consumers (orphan anti-pattern)"
    )
    # All three should declare cathedral_consumers/anti_pattern_lookup_consumer/
    # as a consumer per memo specifications (this is the canonical auto-discovery
    # consumer per Catalog #335).
    consumers_text = " ".join(ap.canonical_consumers)
    assert "anti_pattern" in consumers_text.lower() or "preflight" in consumers_text.lower(), (
        f"{anti_pattern_id} canonical_consumers list lacks anti-pattern "
        f"lookup consumer / preflight gate reference: {ap.canonical_consumers}"
    )


def test_d7_has_substantive_empirical_falsification_anchored_to_audit_memo() -> None:
    """D7 anchors the 18-shared-assumption audit empirically per memo specs.

    Verifies the empirical anchor is NOT fake (per CLAUDE.md NO FAKE
    IMPLEMENTATIONS) — checks that audit memo path is cited AND the
    empirical_output dict contains substantive measurements.
    """
    ap = build_canonical_default_plateau_substrate_disguised_as_class_shift_v1()
    assert ap.recurrence_count == 1, (
        "D7 should have exactly 1 honest empirical falsification anchor "
        "(the 18-assumption audit; per HONESTY DISCIPLINE we do NOT fake "
        "30 separate substrate-confirmation anchors)"
    )
    fals = ap.empirical_falsifications[0]
    # Memo specifies audit-memo anchor path
    assert "assumptions_challenge_audit" in fals.empirical_artifact_path.lower(), (
        f"D7 empirical_artifact_path should reference assumptions audit memo; "
        f"got {fals.empirical_artifact_path!r}"
    )
    # Substantive empirical output: must have audit-measurable fields
    assert fals.empirical_output["assumptions_audited"] == 18
    assert 0 <= fals.empirical_output["shared_assumption_prevalence_max"] <= 1.0
    # IMPLEMENTATION_LEVEL_CONFIRMATION per Catalog #307 paradigm-vs-implementation
    assert fals.incident_classification == INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION


def test_d8_has_zero_empirical_falsifications_as_design_only_at_landing() -> None:
    """D8 is design-only at landing per memo HONESTY DISCIPLINE — the
    empirical falsifications will accumulate from future audits.

    Verifies we honestly registered 0 anchors rather than faking anchors
    per CLAUDE.md NO FAKE IMPLEMENTATIONS forbidden class #5
    (Returns-canonical-markers-without-doing-work)."""
    ap = build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1()
    assert ap.recurrence_count == 0, (
        "D8 should have 0 empirical falsifications at landing per memo "
        "HONESTY DISCIPLINE; future audits will append via "
        "append_empirical_falsification helper"
    )
    # Even with 0 falsifications, the anti-pattern is valid (design-only)
    # — verifies the registry can hold design-only anti-patterns per the
    # canonical schema (sister of canonical equations with FORMALIZATION_PENDING).
    assert ap.is_actively_recurring is False  # property under test


def test_d9_canonical_extinction_anchor_cites_d1_d2_commit_6d3c42635() -> None:
    """D9 EmpiricalFalsification ratifies today's D1+D2 canonical rename
    wave as the extinction event per memo.

    Verifies the empirical anchor is REAL (cites today's commit SHA) and
    quantifies the canonical structural-protection-surfaces signature
    documented in the memo."""
    ap = build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1()
    assert ap.recurrence_count == 1
    fals = ap.empirical_falsifications[0]
    # Per memo: D1+D2 inline canonical rename wave at commit 6d3c42635
    assert "6d3c42635" in fals.empirical_artifact_path, (
        f"D9 empirical_artifact_path should cite commit 6d3c42635 per memo; "
        f"got {fals.empirical_artifact_path!r}"
    )
    # Canonical structural-protection-surfaces signature: 2 surfaces renamed
    assert fals.empirical_output["forward_looking_section_headers_renamed"] == 2
    assert fals.empirical_output["historical_provenance_references_preserved"] is True
    # Per Catalog #307: ratification at new substrate (canonical extinction event)
    assert fals.incident_classification == INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE


# -----------------------------------------------------------------------------
# Registry persistence round-trip tests (verifies populate_d7_d8_d9
# actually writes via canonical helper + retrieval works)
# -----------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path: Path) -> tuple[Path, Path]:
    """Provide an isolated ledger + lock path for write-side tests so the
    canonical production registry is not mutated by the test."""
    p = tmp_path / "test_canonical_anti_patterns_registry.jsonl"
    lock = tmp_path / "test_canonical_anti_patterns_registry.lock"
    return p, lock


def test_populate_d7_d8_d9_writes_three_rows_to_isolated_ledger(
    isolated_registry: tuple[Path, Path]
) -> None:
    """populate_d7_d8_d9_anti_patterns appends 3 anti-pattern-registered
    events to the canonical fcntl-locked JSONL ledger."""
    p_path, lock_path = isolated_registry
    assert not p_path.exists()
    registered = populate_d7_d8_d9_anti_patterns(
        path=p_path,
        lock_path=lock_path,
        agent="test_d7_d8_d9_registration",
    )
    assert len(registered) == 3
    assert p_path.exists()
    # Verify ledger row count = 3 (one per anti-pattern; APPEND-ONLY)
    lines = [
        ln for ln in p_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) == 3, (
        f"Expected 3 ledger rows; got {len(lines)}"
    )
    # Verify ledger payload schema: each row is JSON, has event_type +
    # anti_pattern_payload
    for ln in lines:
        row = json.loads(ln)
        assert row["event_type"] == "anti_pattern_registered"
        assert "anti_pattern_payload" in row
        assert row["anti_pattern_payload"]["anti_pattern_id"] in _ALL_IDS


def test_populate_d7_d8_d9_is_idempotent_via_append_only(
    isolated_registry: tuple[Path, Path]
) -> None:
    """Per CLAUDE.md Catalog #110/#113 APPEND-ONLY: re-running populate
    appends NEW rows (not mutates existing). Latest-row-wins query
    semantics in `get_anti_pattern_by_id` ensure consumers see the
    most recent payload."""
    p_path, lock_path = isolated_registry
    populate_d7_d8_d9_anti_patterns(
        path=p_path, lock_path=lock_path, agent="test_first_run"
    )
    populate_d7_d8_d9_anti_patterns(
        path=p_path, lock_path=lock_path, agent="test_second_run"
    )
    # Expect 6 rows total (3 from each invocation; APPEND-ONLY)
    lines = [
        ln for ln in p_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) == 6
    # But the per-id retrieval should still work (latest-row-wins)
    rows = load_anti_patterns_strict(p_path)
    anti_pattern_ids_seen = [r["anti_pattern_payload"]["anti_pattern_id"] for r in rows]
    # Each id appears exactly 2x (once per populate invocation)
    for anti_pattern_id in _ALL_IDS:
        assert anti_pattern_ids_seen.count(anti_pattern_id) == 2


def test_register_then_retrieve_round_trip_preserves_canonical_fields(
    isolated_registry: tuple[Path, Path]
) -> None:
    """Round-trip BEHAVIOR test: register an anti-pattern via canonical
    helper, retrieve via canonical query helper, verify fields preserved.

    This is what makes the test BEHAVIORAL (not constant-checking): if
    the registration helper silently fails or the query helper silently
    returns stale data, the test fails."""
    p_path, lock_path = isolated_registry
    registered = populate_d7_d8_d9_anti_patterns(
        path=p_path, lock_path=lock_path, agent="test_round_trip"
    )
    # Query each registered anti-pattern back via canonical helper
    for ap_original in registered:
        ap_retrieved = get_anti_pattern_by_id(ap_original.anti_pattern_id, path=p_path)
        assert ap_retrieved is not None, (
            f"{ap_original.anti_pattern_id} not retrievable from registry "
            f"after registration"
        )
        # Verify canonical fields preserved (the actual behavior under test)
        assert ap_retrieved.anti_pattern_id == ap_original.anti_pattern_id
        assert ap_retrieved.severity == ap_original.severity
        assert ap_retrieved.paradigm_class == ap_original.paradigm_class
        assert ap_retrieved.canonical_unwind_path == ap_original.canonical_unwind_path
        assert ap_retrieved.recurrence_count == ap_original.recurrence_count


def test_query_anti_patterns_returns_d7_d8_d9_when_registered(
    isolated_registry: tuple[Path, Path]
) -> None:
    """query_anti_patterns canonical helper returns the 3 newly-registered
    anti-patterns from the isolated ledger."""
    p_path, lock_path = isolated_registry
    populate_d7_d8_d9_anti_patterns(
        path=p_path, lock_path=lock_path, agent="test_query"
    )
    all_anti_patterns = query_anti_patterns(path=p_path)
    queried_ids = {ap.anti_pattern_id for ap in all_anti_patterns}
    for expected_id in _ALL_IDS:
        assert expected_id in queried_ids, (
            f"Expected anti-pattern {expected_id} not found in canonical "
            f"query result; found {sorted(queried_ids)}"
        )


# -----------------------------------------------------------------------------
# Live canonical registry regression guard (verifies the production
# registry actually contains the 3 anti-patterns this landing registered)
# -----------------------------------------------------------------------------


def test_live_canonical_registry_contains_d7_d8_d9_after_landing() -> None:
    """Live-repo regression guard: the canonical production registry at
    `.omx/state/canonical_anti_patterns_registry.jsonl` must contain the
    3 anti-patterns this landing registered.

    This is the canonical "did the landing actually land?" test per
    Slot EEE 6-axis audit Axis F (cite-vs-impl mismatch detection)."""
    for expected_id in _ALL_IDS:
        ap = get_anti_pattern_by_id(expected_id)
        assert ap is not None, (
            f"Live canonical registry missing {expected_id} — landing failed "
            f"or registration was rolled back; re-run populate_d7_d8_d9_anti_patterns"
        )
        assert isinstance(ap, AntiPattern)


# -----------------------------------------------------------------------------
# Public API export regression guard
# -----------------------------------------------------------------------------


def test_canonical_init_exports_d7_d8_d9_builders_in_public_api() -> None:
    """The canonical `tac.canonical_anti_patterns` __init__ MUST re-export
    the d7_d8_d9 builders so future agents can discover them via the
    canonical package API (not by hunting through internal submodules)."""
    import tac.canonical_anti_patterns as ap_pkg
    # Each builder + the aggregator + the populate helper must be in __all__
    expected_exports = (
        "build_all_d7_d8_d9_anti_patterns",
        "build_canonical_default_plateau_substrate_disguised_as_class_shift_v1",
        "build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1",
        "build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1",
        "populate_d7_d8_d9_anti_patterns",
    )
    for export_name in expected_exports:
        assert export_name in ap_pkg.__all__, (
            f"Public API regression: {export_name!r} missing from "
            f"tac.canonical_anti_patterns.__all__"
        )
        assert hasattr(ap_pkg, export_name), (
            f"Public API regression: {export_name!r} not bound on "
            f"tac.canonical_anti_patterns module"
        )


# -----------------------------------------------------------------------------
# Catalog #340 sister-checkpoint guard regression guard
# -----------------------------------------------------------------------------


def test_d7_d8_d9_builder_module_imports_via_canonical_package() -> None:
    """Verify the d7_d8_d9_builders module can be imported via canonical
    package path (regression guard against future package-layout drift)."""
    from tac.canonical_anti_patterns.d7_d8_d9_builders import (
        build_all_d7_d8_d9_anti_patterns as builder_aggregator,
    )
    # Builder aggregator returns the canonical 3-anti-pattern tuple
    aps = builder_aggregator()
    assert len(aps) == 3
    assert all(isinstance(ap, AntiPattern) for ap in aps)
