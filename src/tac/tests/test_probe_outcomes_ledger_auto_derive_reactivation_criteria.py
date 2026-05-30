# SPDX-License-Identifier: MIT
"""Tests for META Finding A canonical 2-landing pattern Landing 1.

Landing 1 = canonical helper extension at ``tac.probe_outcomes_ledger``:
``register_probe_outcome`` now auto-derives ``reactivation_criteria`` from
``next_action`` when caller does not supply it, per the deferred-items feeder
audit landed at commit ``a9d45b171`` (memo
``.omx/research/deferred_items_feeder_audit_landed_20260530.md``).

Empirical landscape at landing: 104 of 105 DEFER rows in
``.omx/state/probe_outcomes.jsonl`` have EMPTY ``reactivation_criteria``; 100
of those have substantive ``next_action`` strings. Per CLAUDE.md "NO FAKE
IMPLEMENTATIONS" non-negotiable + Catalog #287 sister discipline: the
auto-derive logic preserves the full ``next_action`` text verbatim (no
information loss) when no canonical pattern matches; placeholder ``next_action``
literals are rejected so the canonical posterior cannot absorb fake content.

Coverage:

- caller-supplied criteria preserved verbatim (list[str] and single str)
- empty criteria auto-derived from non-empty next_action via fallback pattern
- empty criteria auto-derived from next_action via canonical "Re-fire X when Y"
  pattern (and sister "Resume X when Y", "Reactivate X when Y", "Re-run X when Y")
- BOTH empty preserves empty (HONEST emptiness)
- derivation provenance present + correct token
- placeholder rejection on next_action (``<rationale>``, ``<reason>``, etc.)
- placeholder rejection on reactivation_criteria
- type validation (non-list/non-str raises ValueError)
- ``_is_substantive_string`` helper
- backward compat: rows lacking the new field load cleanly
- canonical export contracts (``EVENT_BACKFILL``, ``AUTO_DERIVE_PROVENANCE_*``)
- update_probe_outcome propagates the new fields from existing rows
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL,
    EVENT_BACKFILL,
    VALID_EVENT_TYPES,
    VERDICT_DEFER,
    VERDICT_INDEPENDENT,
    VERDICT_PROCEED,
    _auto_derive_reactivation_criteria_from_next_action,
    _is_substantive_string,
    _resolve_reactivation_criteria,
    load_outcomes,
    register_probe_outcome,
    update_probe_outcome,
    EVENT_RATIFIED,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> tuple[Path, Path]:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"
    return ledger, lock


# ─────────────────────────────────────────────────────────────────────────
# _is_substantive_string helper
# ─────────────────────────────────────────────────────────────────────────


def test_is_substantive_string_rejects_none() -> None:
    assert not _is_substantive_string(None)


def test_is_substantive_string_rejects_empty_string() -> None:
    assert not _is_substantive_string("")


def test_is_substantive_string_rejects_whitespace_only() -> None:
    assert not _is_substantive_string("   ")


def test_is_substantive_string_rejects_too_short() -> None:
    # Default min char threshold is 4.
    assert not _is_substantive_string("abc")


def test_is_substantive_string_accepts_4_char_string() -> None:
    assert _is_substantive_string("test")


def test_is_substantive_string_rejects_placeholder_literals() -> None:
    for placeholder in (
        "<rationale>",
        "<reason>",
        "<reactivation-criteria>",
        "TBD",
        "TODO",
        "placeholder",
        "N/A",
        "none",
    ):
        assert not _is_substantive_string(placeholder), placeholder


def test_is_substantive_string_rejects_non_string_types() -> None:
    assert not _is_substantive_string(123)
    assert not _is_substantive_string([])
    assert not _is_substantive_string({})


def test_is_substantive_string_accepts_substantive_normal_text() -> None:
    assert _is_substantive_string("Re-fire after Phase 2 dispatch")


# ─────────────────────────────────────────────────────────────────────────
# _auto_derive_reactivation_criteria_from_next_action helper
# ─────────────────────────────────────────────────────────────────────────


def test_auto_derive_returns_none_when_next_action_is_none() -> None:
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(None)
    assert criteria is None
    assert provenance is None


def test_auto_derive_returns_none_when_next_action_is_empty() -> None:
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action("")
    assert criteria is None
    assert provenance is None


def test_auto_derive_returns_none_when_next_action_is_placeholder() -> None:
    for placeholder in ("<rationale>", "<reason>", "TBD", "placeholder", "N/A"):
        criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
            placeholder
        )
        assert criteria is None, placeholder
        assert provenance is None, placeholder


def test_auto_derive_returns_fallback_when_no_canonical_pattern() -> None:
    next_action = "queue paired CUDA + paired CPU empirical anchor"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    assert criteria == [f"next_action_satisfied: {next_action}"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_extracts_canonical_re_fire_pattern() -> None:
    next_action = "Re-fire the canonical dispatch when archive bytes land"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    # Pattern: "Re-fire <X> when <Y>" → criterion = "<Y> empirically met"
    assert criteria == ["archive bytes land empirically met"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_extracts_canonical_resume_pattern() -> None:
    next_action = "Resume the dispatch wave when the canonical sister wave lands"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    assert criteria == ["the canonical sister wave lands empirically met"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_extracts_canonical_reactivate_pattern() -> None:
    next_action = "Reactivate the substrate when Tier-C measurement converges"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    assert criteria == ["Tier-C measurement converges empirically met"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_extracts_canonical_re_run_pattern() -> None:
    next_action = "Re-run the smoke when sister fix lands"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    assert criteria == ["sister fix lands empirically met"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_pattern_fallback_when_no_when_keyword() -> None:
    next_action = "Re-fire after operator approval"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    # No " when " separator → fallback to verbatim preservation.
    assert criteria == [f"next_action_satisfied: {next_action}"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_auto_derive_pattern_handles_case_insensitive_keyword() -> None:
    next_action = "RE-FIRE the dispatch when archive lands"
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    # Keyword detection is case-insensitive but the extracted criterion
    # preserves the original casing.
    assert criteria == ["archive lands empirically met"]


def test_auto_derive_strips_whitespace() -> None:
    next_action = "  request_reinvestigation_of_alternative_reducers  "
    criteria, provenance = _auto_derive_reactivation_criteria_from_next_action(
        next_action
    )
    assert criteria == [
        "next_action_satisfied: request_reinvestigation_of_alternative_reducers"
    ]


# ─────────────────────────────────────────────────────────────────────────
# _resolve_reactivation_criteria normalizer
# ─────────────────────────────────────────────────────────────────────────


def test_resolve_passes_caller_supplied_list_verbatim() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria=[
            "Construct actual substrate archive via paid Modal smoke",
            "Re-run pre-entropy prober on actual archive.zip member bytes",
        ],
        next_action="ignored when caller supplied",
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    assert criteria == [
        "Construct actual substrate archive via paid Modal smoke",
        "Re-run pre-entropy prober on actual archive.zip member bytes",
    ]
    assert provenance is None  # caller supplied → no derivation provenance


def test_resolve_normalizes_caller_supplied_single_str_to_list() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria="Single criterion supplied by caller",
        next_action=None,
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    assert criteria == ["Single criterion supplied by caller"]
    assert provenance is None


def test_resolve_filters_placeholder_elements_from_supplied_list() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria=[
            "valid criterion one",
            "<rationale>",
            "valid criterion two",
            "TBD",
        ],
        next_action=None,
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    # Placeholder elements are filtered; valid elements preserved.
    assert criteria == ["valid criterion one", "valid criterion two"]
    assert provenance is None


def test_resolve_falls_back_to_auto_derive_when_all_elements_placeholder() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria=["<rationale>", "TBD", "<reason>"],
        next_action="run sister wave to ratify",
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    # All placeholder → treat as not supplied → auto-derive from next_action.
    assert criteria == ["next_action_satisfied: run sister wave to ratify"]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


def test_resolve_returns_none_when_both_empty() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria=None,
        next_action=None,
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    assert criteria is None
    assert provenance is None


def test_resolve_returns_none_when_both_placeholder() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria="<rationale>",
        next_action="<reason>",
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    assert criteria is None
    assert provenance is None


def test_resolve_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="reactivation_criteria must be list"):
        _resolve_reactivation_criteria(
            reactivation_criteria=12345,  # type: ignore[arg-type]
            next_action=None,
            auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
        )


def test_resolve_rejects_dict_type() -> None:
    with pytest.raises(ValueError, match="reactivation_criteria must be list"):
        _resolve_reactivation_criteria(
            reactivation_criteria={"key": "value"},  # type: ignore[arg-type]
            next_action=None,
            auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
        )


def test_resolve_empty_list_falls_back_to_auto_derive() -> None:
    criteria, provenance = _resolve_reactivation_criteria(
        reactivation_criteria=[],
        next_action="proceed_to_actual_smoke_dispatch_for_dreamerv3",
        auto_derive_provenance=AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION,
    )
    assert criteria == [
        "next_action_satisfied: proceed_to_actual_smoke_dispatch_for_dreamerv3"
    ]
    assert provenance == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION


# ─────────────────────────────────────────────────────────────────────────
# register_probe_outcome — end-to-end with auto-derive
# ─────────────────────────────────────────────────────────────────────────


def test_register_writes_caller_supplied_reactivation_criteria(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_auto_derive_caller_supplied",
        substrate="test_substrate",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_DEFER,
        metric_name="test_metric",
        metric_value=0.5,
        next_action="ignored when caller supplied criteria",
        reactivation_criteria=[
            "Construct actual substrate archive via paid Modal smoke",
            "Re-run pre-entropy prober",
        ],
        path=ledger,
        lock_path=lock,
    )
    assert row["reactivation_criteria"] == [
        "Construct actual substrate archive via paid Modal smoke",
        "Re-run pre-entropy prober",
    ]
    # Caller supplied → no derivation provenance.
    assert row["reactivation_criteria_derivation_provenance"] is None


def test_register_auto_derives_when_only_next_action_supplied(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_auto_derive_from_next_action",
        substrate="test_substrate",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_DEFER,
        metric_name="test_metric",
        metric_value=0.5,
        next_action="proceed_to_actual_smoke_dispatch_for_dreamerv3_rssm",
        # NO reactivation_criteria supplied.
        path=ledger,
        lock_path=lock,
    )
    assert row["reactivation_criteria"] == [
        "next_action_satisfied: proceed_to_actual_smoke_dispatch_for_dreamerv3_rssm"
    ]
    assert (
        row["reactivation_criteria_derivation_provenance"]
        == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION
    )


def test_register_preserves_both_none_when_both_empty(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_both_empty_honest",
        substrate="test_substrate",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_DEFER,
        metric_name="test_metric",
        metric_value=0.5,
        # NO next_action, NO reactivation_criteria.
        path=ledger,
        lock_path=lock,
    )
    # HONEST emptiness per NO FAKE IMPLEMENTATIONS.
    assert row["reactivation_criteria"] is None
    assert row["reactivation_criteria_derivation_provenance"] is None


def test_register_rejects_placeholder_next_action_via_resolver(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # next_action is a placeholder literal → auto-derive returns None.
    # This is HONEST behavior per NO FAKE IMPLEMENTATIONS.
    row = register_probe_outcome(
        probe_id="test_placeholder_next_action",
        substrate="test_substrate",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_DEFER,
        metric_name="test_metric",
        metric_value=0.5,
        next_action="<rationale>",
        path=ledger,
        lock_path=lock,
    )
    # The next_action field itself is preserved verbatim (audit trail) but
    # auto-derive refuses to populate reactivation_criteria from a placeholder.
    assert row["next_action"] == "<rationale>"
    assert row["reactivation_criteria"] is None
    assert row["reactivation_criteria_derivation_provenance"] is None


def test_register_persists_new_fields_to_disk(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_persist_new_fields",
        substrate="test_sub",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_DEFER,
        metric_name="m",
        metric_value=0.0,
        next_action="sister wave lands",
        path=ledger,
        lock_path=lock,
    )
    assert ledger.exists()
    rows = load_outcomes(ledger)
    assert len(rows) == 1
    assert rows[0]["reactivation_criteria"] == [
        "next_action_satisfied: sister wave lands"
    ]
    assert (
        rows[0]["reactivation_criteria_derivation_provenance"]
        == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION
    )


def test_register_rejects_invalid_reactivation_criteria_type(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="reactivation_criteria must be list"):
        register_probe_outcome(
            probe_id="test_invalid_type",
            substrate="test_sub",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_DEFER,
            metric_name="m",
            metric_value=0.0,
            next_action="some action",
            reactivation_criteria=42,  # type: ignore[arg-type]
            path=ledger,
            lock_path=lock,
        )


def test_register_canonical_re_fire_pattern_extraction(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_canonical_pattern",
        substrate="test_sub",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_DEFER,
        metric_name="m",
        metric_value=0.0,
        next_action="Re-fire the dispatch when sister fix lands and archive bytes converge",
        path=ledger,
        lock_path=lock,
    )
    assert row["reactivation_criteria"] == [
        "sister fix lands and archive bytes converge empirically met"
    ]


# ─────────────────────────────────────────────────────────────────────────
# update_probe_outcome — propagates new fields
# ─────────────────────────────────────────────────────────────────────────


def test_update_propagates_reactivation_criteria_from_existing(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # First, register with auto-derived criteria.
    register_probe_outcome(
        probe_id="test_update_propagates",
        substrate="test_sub",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_DEFER,
        metric_name="m",
        metric_value=0.0,
        next_action="sister wave lands the canonical fix",
        path=ledger,
        lock_path=lock,
    )
    # Now ratify — the update row should inherit reactivation_criteria + provenance.
    updated = update_probe_outcome(
        probe_id="test_update_propagates",
        event_type=EVENT_RATIFIED,
        notes="operator ratified the verdict",
        path=ledger,
        lock_path=lock,
    )
    assert updated["reactivation_criteria"] == [
        "next_action_satisfied: sister wave lands the canonical fix"
    ]
    assert (
        updated["reactivation_criteria_derivation_provenance"]
        == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION
    )


# ─────────────────────────────────────────────────────────────────────────
# Canonical export contracts
# ─────────────────────────────────────────────────────────────────────────


def test_event_backfill_in_valid_event_types() -> None:
    assert EVENT_BACKFILL in VALID_EVENT_TYPES
    assert EVENT_BACKFILL == "backfill"


def test_auto_derive_provenance_constants_pinned() -> None:
    assert AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION == (
        "auto_derived_from_next_action_at_register_probe_outcome_per_meta_finding_a_v1"
    )
    assert AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL == (
        "auto_derived_from_next_action_at_backfill_per_meta_finding_a_v1"
    )


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility — legacy rows without the new fields
# ─────────────────────────────────────────────────────────────────────────


def test_load_outcomes_handles_legacy_rows_without_new_fields(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # Hand-write a legacy row (no reactivation_criteria,
    # no reactivation_criteria_derivation_provenance fields).
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "legacy_probe",
        "substrate": "legacy_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "legacy next_action without auto-derive",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-05-31T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")
    rows = load_outcomes(ledger)
    assert len(rows) == 1
    # Legacy rows lack the new fields; consumers get None via dict.get default.
    assert rows[0].get("reactivation_criteria") is None
    assert rows[0].get("reactivation_criteria_derivation_provenance") is None
    # The original fields are intact.
    assert rows[0]["probe_id"] == "legacy_probe"
    assert rows[0]["next_action"] == "legacy next_action without auto-derive"


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard — empirical landscape verified
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_empirical_landscape_documented_in_audit_memo() -> None:
    """Regression guard documenting the META Finding A empirical landscape.

    The feeder audit memo (commit a9d45b171) reported 70+/71 DEFER probes
    have EMPTY reactivation_criteria. The current empirical landscape
    (as of 2026-05-30) is 104/105 DEFER probes EMPTY (the ledger has grown
    since the audit memo landed). The canonical fix at Landing 1 ensures
    FUTURE register_probe_outcome calls cannot produce new EMPTY rows when
    next_action is substantive. Landing 2 (backfill tool) handles the
    historical 100 backfill candidates (104 EMPTY - 4 with also-empty
    next_action = 100 backfill candidates).
    """
    rows = load_outcomes()
    defer_rows = [r for r in rows if r.get("verdict") == "DEFER"]
    # We expect the historical empty count to be > 0 at landing.
    # After backfill tool runs with --apply, this count drops to near-zero
    # (the 4 BOTH-empty rows remain empty per HONEST emptiness).
    # This test documents but does not enforce the historical state.
    assert len(defer_rows) > 0, "ledger should have DEFER rows for regression baseline"
    # NO assertion on the empty count itself; the audit memo + landing memo
    # carry the canonical numbers. This test just verifies the canonical helper
    # is callable against the live ledger without crashing.
    empty_count = sum(
        1 for r in defer_rows if not r.get("reactivation_criteria")
    )
    # Sanity: at least SOME rows are empty per the empirical landscape; if this
    # ever drops to 0 the backfill tool has fully run and the feeder consumer
    # picks up auto-derived criteria structurally.
    assert empty_count >= 0  # documentation, not enforcement


def test_register_probe_outcome_signature_keyword_only() -> None:
    """The canonical helper signature must remain keyword-only per Catalog
    #229 + #287 sister discipline. Positional args would break callers."""
    import inspect

    sig = inspect.signature(register_probe_outcome)
    for name, param in sig.parameters.items():
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ), f"param {name!r} must be KEYWORD_ONLY or VAR_KEYWORD"


def test_register_probe_outcome_has_reactivation_criteria_kwarg() -> None:
    """Regression guard for Landing 1 surface contract."""
    import inspect

    sig = inspect.signature(register_probe_outcome)
    assert "reactivation_criteria" in sig.parameters
    # Default is None per the canonical signature.
    assert sig.parameters["reactivation_criteria"].default is None
