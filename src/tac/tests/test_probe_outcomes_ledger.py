# SPDX-License-Identifier: MIT
"""Tests for tac.probe_outcomes_ledger (Catalog #313 canonical helper).

Mirrors the depth of src/tac/tests/test_modal_call_id_ledger.py (Catalog #245
exemplar). Covers:

- ProbeOutcomeRecord schema validation
- register_probe_outcome happy path + invalid input rejection
- update_probe_outcome event-type transitions + denormalization
- load_outcomes lenient (skips malformed)
- load_outcomes_strict raises on corrupt
- quarantine on corrupt
- atomic write (no .tmp leakage)
- query helpers (by_probe_id / by_substrate / by_recipe / latest_blocking / blocking_outcomes)
- expires_at_utc staleness window auto-transition
- 4-process spawn-pool concurrent-append stress (mirror Catalog #245 4-proc test)
- full lifecycle (adjudicated → ratified → operator_override)
- backfill correctness on ATW v2 D4 anchor
"""

from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    BLOCKER_STATUS_ADVISORY,
    BLOCKER_STATUS_BLOCKING,
    BLOCKER_STATUS_EXPIRED,
    BLOCKING_VERDICTS,
    DEFAULT_STALENESS_WINDOW_DAYS,
    EVENT_ADJUDICATED,
    EVENT_BACKFILL,
    EVENT_EXPIRED,
    EVENT_OPERATOR_OVERRIDE,
    EVENT_RATIFIED,
    EVENT_SUPERSEDED,
    LOCK_TIMEOUT_SECONDS,
    PROBE_OUTCOMES_LEDGER_PATH,
    SCHEMA_VERSION,
    VALID_BLOCKER_STATUSES,
    VALID_EVENT_TYPES,
    VALID_VERDICTS,
    VERDICT_DEFER,
    VERDICT_INDEPENDENT,
    VERDICT_PROCEED,
    VERDICT_PROMOTE,
    ProbeOutcomesLedgerCorruptError,
    ProbeOutcomeView,
    latest_blocking_outcome_by_recipe,
    latest_blocking_outcome_by_substrate,
    latest_outcome_by_probe_id,
    load_outcomes,
    load_outcomes_strict,
    query_all_post_utc,
    query_blocking_outcomes,
    query_by_probe_id,
    query_by_recipe,
    query_by_substrate,
    register_probe_outcome,
    update_probe_outcome,
)

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> tuple[Path, Path]:
    """Return (ledger_path, lock_path) inside a tmpdir for isolated tests."""
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"
    return ledger, lock


# ─────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────


def test_schema_version_pinned() -> None:
    assert SCHEMA_VERSION == 1


def test_valid_event_types_canonical_set() -> None:
    # META Finding A canonical 2-landing pattern (2026-05-30): EVENT_BACKFILL
    # added to the canonical event taxonomy so the backfill tool can append
    # APPEND-ONLY rows with auto-derived reactivation_criteria from next_action
    # WITHOUT mutating the original adjudicated row.
    assert frozenset(
        {
            EVENT_ADJUDICATED,
            EVENT_RATIFIED,
            EVENT_SUPERSEDED,
            EVENT_EXPIRED,
            EVENT_OPERATOR_OVERRIDE,
            EVENT_BACKFILL,
        }
    ) == VALID_EVENT_TYPES


def test_valid_verdicts_includes_all_canonical_tokens() -> None:
    assert "INDEPENDENT" in VALID_VERDICTS
    assert "KILL" in VALID_VERDICTS
    assert "DEFER" in VALID_VERDICTS
    assert "PROMOTE" in VALID_VERDICTS
    assert "PROCEED" in VALID_VERDICTS
    assert "PARTIAL" in VALID_VERDICTS
    assert "OPERATOR_REVIEW_REQUIRED" in VALID_VERDICTS
    # Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O3 (2026-05-28): INFRASTRUCTURE_FAILURE
    # is the canonical verdict semantically distinct from INDEPENDENT for the
    # F19 segfault class. See VERDICT_INFRASTRUCTURE_FAILURE docstring.
    assert "INFRASTRUCTURE_FAILURE" in VALID_VERDICTS


def test_blocking_verdicts_canonical_set() -> None:
    """Per CLAUDE.md 'Forbidden premature KILL': blocking verdicts are
    research-deferrals, NOT kills.

    Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O3 (2026-05-28): INFRASTRUCTURE_FAILURE
    added to BLOCKING_VERDICTS so re-running the exact same infrastructure-broken
    probe is refused at the gate (would just re-crash and waste paid GPU spend).
    Resolution requires sister probe with corrected infrastructure OR operator
    override per Catalog #313 paired-env bypass.
    """
    assert frozenset({"INDEPENDENT", "KILL", "DEFER", "INFRASTRUCTURE_FAILURE"}) == BLOCKING_VERDICTS


def test_valid_blocker_statuses_canonical_set() -> None:
    assert frozenset(
        {BLOCKER_STATUS_BLOCKING, BLOCKER_STATUS_ADVISORY, BLOCKER_STATUS_EXPIRED}
    ) == VALID_BLOCKER_STATUSES


# ─────────────────────────────────────────────────────────────────────────
# register_probe_outcome happy path + validation
# ─────────────────────────────────────────────────────────────────────────


def test_register_probe_outcome_writes_row(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_probe_1",
        substrate="test_substrate",
        recipe_path=".omx/operator_authorize_recipes/test_recipe.yaml",
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/test_probe_verdict.md",
        next_action="do_not_dispatch",
        path=ledger,
        lock_path=lock,
    )
    assert row["probe_id"] == "test_probe_1"
    assert row["verdict"] == "INDEPENDENT"
    assert row["blocker_status"] == BLOCKER_STATUS_BLOCKING  # auto from INDEPENDENT
    assert row["schema_version"] == 1
    assert row["event_type"] == EVENT_ADJUDICATED
    assert "expires_at_utc" in row


def test_register_probe_outcome_persists_to_disk(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_probe_2",
        substrate="test_sub",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_PROMOTE,
        metric_name="mse_delta",
        metric_value=0.01,
        path=ledger,
        lock_path=lock,
    )
    assert ledger.exists()
    rows = load_outcomes(ledger)
    assert len(rows) == 1
    assert rows[0]["probe_id"] == "test_probe_2"
    assert rows[0]["verdict"] == "PROMOTE"
    assert rows[0]["blocker_status"] == BLOCKER_STATUS_ADVISORY  # auto from PROMOTE


def test_register_rejects_empty_probe_id(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="probe_id"):
        register_probe_outcome(
            probe_id="",
            substrate="s",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
        )


def test_register_rejects_empty_substrate(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="substrate"):
        register_probe_outcome(
            probe_id="p",
            substrate="",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
        )


def test_register_rejects_invalid_verdict(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="verdict"):
        register_probe_outcome(
            probe_id="p",
            substrate="s",
            recipe_path=None,
            probe_kind="k",
            verdict="BOGUS_VERDICT",
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
        )


def test_register_rejects_newline_in_probe_id(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="probe_id"):
        register_probe_outcome(
            probe_id="bad\nid",
            substrate="s",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
        )


def test_register_extra_metadata_attached(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROCEED,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
        custom_field="some_value",
    )
    assert row["custom_field"] == "some_value"


def test_register_extra_cannot_overwrite_reserved(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="collides"):
        register_probe_outcome(
            probe_id="p",
            substrate="s",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
            **{"event_type": "BAD"},  # collides with reserved
        )


def test_register_blocker_status_auto_blocking_for_independent(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    assert row["blocker_status"] == BLOCKER_STATUS_BLOCKING


def test_register_blocker_status_auto_advisory_for_promote(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROMOTE,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    assert row["blocker_status"] == BLOCKER_STATUS_ADVISORY


def test_register_blocker_status_explicit_override(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        blocker_status=BLOCKER_STATUS_ADVISORY,  # operator opted in to advisory-only
        path=ledger,
        lock_path=lock,
    )
    assert row["blocker_status"] == BLOCKER_STATUS_ADVISORY


def test_register_expires_at_utc_computed(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        adjudicated_at_utc="2026-05-16T00:00:00.000000Z",
        staleness_window_days=30,
        path=ledger,
        lock_path=lock,
    )
    assert row["expires_at_utc"].startswith("2026-06-15")


# ─────────────────────────────────────────────────────────────────────────
# update_probe_outcome event-type transitions
# ─────────────────────────────────────────────────────────────────────────


def test_update_ratified_appends_new_row(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.006,
        path=ledger,
        lock_path=lock,
    )
    update_probe_outcome(
        probe_id="p",
        event_type=EVENT_RATIFIED,
        notes="council ratified verdict",
        path=ledger,
        lock_path=lock,
    )
    rows = load_outcomes(ledger)
    assert len(rows) == 2
    assert rows[0]["event_type"] == EVENT_ADJUDICATED
    assert rows[1]["event_type"] == EVENT_RATIFIED
    # Per HISTORICAL_PROVENANCE: original row is byte-preserved.
    assert rows[0]["verdict"] == "INDEPENDENT"


def test_update_expired_auto_blocker_transition(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    row = update_probe_outcome(
        probe_id="p",
        event_type=EVENT_EXPIRED,
        notes="aged past 30-day window",
        path=ledger,
        lock_path=lock,
    )
    assert row["blocker_status"] == BLOCKER_STATUS_EXPIRED


def test_update_operator_override_auto_advisory(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    row = update_probe_outcome(
        probe_id="p",
        event_type=EVENT_OPERATOR_OVERRIDE,
        notes="operator override: fresh evidence; re-dispatch authorized",
        path=ledger,
        lock_path=lock,
    )
    assert row["blocker_status"] == BLOCKER_STATUS_ADVISORY


def test_update_rejects_unknown_probe_id(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    with pytest.raises(ValueError, match="no prior adjudicated event"):
        update_probe_outcome(
            probe_id="ghost_probe",
            event_type=EVENT_RATIFIED,
            path=ledger,
            lock_path=lock,
        )


def test_update_rejects_invalid_event_type(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    with pytest.raises(ValueError, match="event_type"):
        update_probe_outcome(
            probe_id="p",
            event_type="BOGUS",
            path=ledger,
            lock_path=lock,
        )


# ─────────────────────────────────────────────────────────────────────────
# load_outcomes lenient/strict semantics
# ─────────────────────────────────────────────────────────────────────────


def test_load_outcomes_empty_when_missing(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, _ = tmp_ledger
    assert load_outcomes(ledger) == []


def test_load_outcomes_lenient_skips_malformed(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, _ = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"valid": "row1"}\nnot valid json\n{"valid": "row2"}\n', encoding="utf-8")
    rows = load_outcomes(ledger)
    assert len(rows) == 2  # the malformed line is skipped


def test_load_outcomes_lenient_skips_non_dict_root(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, _ = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"ok": 1}\n["array", "root"]\n', encoding="utf-8")
    rows = load_outcomes(ledger)
    assert len(rows) == 1


def test_load_outcomes_strict_empty_when_missing(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, _ = tmp_ledger
    assert load_outcomes_strict(ledger) == []


def test_load_outcomes_strict_raises_on_malformed_json(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"valid": 1}\nbad json\n', encoding="utf-8")
    with pytest.raises(ProbeOutcomesLedgerCorruptError):
        load_outcomes_strict(ledger)


def test_load_outcomes_strict_raises_on_non_dict_root(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('["bad", "root"]\n', encoding="utf-8")
    with pytest.raises(ProbeOutcomesLedgerCorruptError):
        load_outcomes_strict(ledger)


# ─────────────────────────────────────────────────────────────────────────
# Quarantine on corrupt
# ─────────────────────────────────────────────────────────────────────────


def test_quarantine_on_corrupt_append(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('bad json line\n', encoding="utf-8")
    with pytest.raises(ProbeOutcomesLedgerCorruptError, match="quarantined"):
        register_probe_outcome(
            probe_id="p",
            substrate="s",
            recipe_path=None,
            probe_kind="k",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=0.0,
            path=ledger,
            lock_path=lock,
        )
    # Original corrupt file moved aside; .corrupt.<utc> sibling exists.
    siblings = list(ledger.parent.iterdir())
    corrupt_files = [p for p in siblings if ".corrupt." in p.name]
    assert len(corrupt_files) == 1


# ─────────────────────────────────────────────────────────────────────────
# Atomic write — no .tmp leakage
# ─────────────────────────────────────────────────────────────────────────


def test_atomic_write_no_tmp_leakage(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROCEED,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    siblings = list(ledger.parent.iterdir())
    tmp_files = [p for p in siblings if ".tmp." in p.name]
    assert len(tmp_files) == 0


# ─────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────


def test_query_by_probe_id_chronological(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    update_probe_outcome(
        probe_id="p",
        event_type=EVENT_RATIFIED,
        path=ledger,
        lock_path=lock,
    )
    rows = query_by_probe_id("p", path=ledger)
    assert len(rows) == 2
    assert rows[0]["event_type"] == EVENT_ADJUDICATED
    assert rows[1]["event_type"] == EVENT_RATIFIED


def test_query_by_substrate(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p1",
        substrate="atw",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    register_probe_outcome(
        probe_id="p2",
        substrate="atw",
        recipe_path=None,
        probe_kind="k2",
        verdict=VERDICT_PROMOTE,
        metric_name="m",
        metric_value=1.0,
        path=ledger,
        lock_path=lock,
    )
    register_probe_outcome(
        probe_id="p3",
        substrate="other",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROCEED,
        metric_name="m",
        metric_value=0.5,
        path=ledger,
        lock_path=lock,
    )
    atw_rows = query_by_substrate("atw", path=ledger)
    assert len(atw_rows) == 2


def test_query_by_recipe_normalizes_paths(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=".omx/operator_authorize_recipes/test.yaml",
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    rows = query_by_recipe(".omx/operator_authorize_recipes/test.yaml", path=ledger)
    assert len(rows) == 1


def test_latest_outcome_by_probe_id(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    latest = latest_outcome_by_probe_id("p", path=ledger)
    assert latest is not None
    assert latest["event_type"] == EVENT_ADJUDICATED


def test_latest_outcome_by_probe_id_none_when_absent(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    assert latest_outcome_by_probe_id("ghost", path=ledger) is None


def test_query_blocking_outcomes_filters_advisory(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="blocking_one",
        substrate="s1",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    register_probe_outcome(
        probe_id="advisory_one",
        substrate="s2",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROMOTE,  # auto-advisory
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(path=ledger)
    assert len(blockers) == 1
    assert blockers[0]["probe_id"] == "blocking_one"


def test_query_blocking_outcomes_filters_expired(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # Register with adjudicated_at_utc 60 days ago + window=30 → expires_at < now.
    register_probe_outcome(
        probe_id="old_blocking",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        adjudicated_at_utc="2026-03-16T00:00:00.000000Z",
        staleness_window_days=30,
        path=ledger,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(
        now_utc=_dt.datetime(2026, 5, 16, tzinfo=_dt.UTC),
        path=ledger,
    )
    assert len(blockers) == 0  # expired


def test_query_blocking_outcomes_filters_via_expired_event(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    update_probe_outcome(
        probe_id="p",
        event_type=EVENT_EXPIRED,
        path=ledger,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(path=ledger)
    assert len(blockers) == 0


def test_latest_blocking_outcome_by_recipe_returns_view(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=".omx/operator_authorize_recipes/sub_a.yaml",
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    view = latest_blocking_outcome_by_recipe(
        ".omx/operator_authorize_recipes/sub_a.yaml",
        path=ledger,
    )
    assert view is not None
    assert isinstance(view, ProbeOutcomeView)
    assert view.verdict == "INDEPENDENT"
    assert view.blocker_status == BLOCKER_STATUS_BLOCKING


def test_latest_blocking_outcome_by_recipe_none_when_absent(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    assert latest_blocking_outcome_by_recipe(
        ".omx/operator_authorize_recipes/nope.yaml",
        path=ledger,
    ) is None


def test_latest_blocking_outcome_by_substrate(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="atw_codec_v2",
        recipe_path=".omx/operator_authorize_recipes/some_recipe.yaml",
        probe_kind="k",
        verdict=VERDICT_INDEPENDENT,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    view = latest_blocking_outcome_by_substrate("atw_codec_v2", path=ledger)
    assert view is not None
    assert view.substrate == "atw_codec_v2"


def test_query_all_post_utc(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROCEED,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    rows = query_all_post_utc("2020-01-01T00:00:00Z", path=ledger)
    assert len(rows) == 1


# ─────────────────────────────────────────────────────────────────────────
# Full lifecycle
# ─────────────────────────────────────────────────────────────────────────


def test_full_lifecycle_adjudicated_ratified_override(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="lifecycle_probe",
        substrate="atw_codec_v2",
        recipe_path=".omx/operator_authorize_recipes/atw_v2.yaml",
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mi_bits",
        metric_value=0.006385,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/atw_v2_d4_probe_verdict.md",
        next_action="do_not_dispatch_atw_v2_phase2_from_this_signal",
        path=ledger,
        lock_path=lock,
    )
    update_probe_outcome(
        probe_id="lifecycle_probe",
        event_type=EVENT_RATIFIED,
        notes="T3 council ratified INDEPENDENT verdict",
        path=ledger,
        lock_path=lock,
    )
    # Operator override clears the blocker.
    update_probe_outcome(
        probe_id="lifecycle_probe",
        event_type=EVENT_OPERATOR_OVERRIDE,
        notes="operator override: G2-PARTIAL alternative ready; re-dispatch authorized",
        path=ledger,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(path=ledger)
    assert len(blockers) == 0  # final state is advisory


# ─────────────────────────────────────────────────────────────────────────
# Concurrent-append stress (4-proc spawn pool)
# ─────────────────────────────────────────────────────────────────────────


def _spawn_writer(args: tuple[Path, Path, int, int]) -> int:
    """Worker that registers `count` probe outcomes with unique probe_ids."""
    ledger, lock, worker_id, count = args
    # Re-import inside subprocess since spawn-pool does not share Python state.
    from tac.probe_outcomes_ledger import (
        VERDICT_PROCEED,
        register_probe_outcome,
    )

    for i in range(count):
        register_probe_outcome(
            probe_id=f"w{worker_id}_p{i}",
            substrate=f"sub_{worker_id}",
            recipe_path=None,
            probe_kind="stress",
            verdict=VERDICT_PROCEED,
            metric_name="m",
            metric_value=float(i),
            path=ledger,
            lock_path=lock,
        )
    return worker_id


@pytest.mark.timeout(120)
def test_concurrent_append_4proc_spawn_pool(tmp_ledger: tuple[Path, Path]) -> None:
    """4-proc spawn-pool concurrent-append stress — mirror Catalog #245.

    Each of 4 processes appends 5 rows; final ledger must have 20 rows total
    with no torn writes (every line parses as a dict).
    """
    ledger, lock = tmp_ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        worker_ids = pool.map(
            _spawn_writer,
            [(ledger, lock, wid, 5) for wid in range(4)],
        )
    assert sorted(worker_ids) == [0, 1, 2, 3]

    rows = load_outcomes_strict(ledger)
    assert len(rows) == 20  # 4 workers x 5 rows each
    probe_ids = {r["probe_id"] for r in rows}
    assert len(probe_ids) == 20  # unique


# ─────────────────────────────────────────────────────────────────────────
# Backfill correctness — ATW v2 D4 anchor
# ─────────────────────────────────────────────────────────────────────────


def test_backfill_atw_v2_d4_anchor_correctness(tmp_ledger: tuple[Path, Path]) -> None:
    """Backfill the ATW v2 D4 H(latent|scorer_class) probe verdict and verify
    the gate-time query returns the INDEPENDENT blocker."""
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        substrate="atw_codec_v2",
        recipe_path=".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385502752,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
        next_action="do_not_dispatch_atw_v2_phase2_from_this_signal",
        adjudicated_at_utc="2026-05-16T22:47:41.000000Z",
        agent="codex",
        notes="probe found I(latent;scorer_class)=0.006385 bits/symbol, 2 orders of magnitude below threshold",
        path=ledger,
        lock_path=lock,
    )
    view = latest_blocking_outcome_by_recipe(
        ".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
        path=ledger,
    )
    assert view is not None
    assert view.verdict == "INDEPENDENT"
    assert view.probe_id == "atw_v2_d4_h_latent_given_scorer_class_20260516"
    assert view.metric_value == pytest.approx(0.006385502752)


# ─────────────────────────────────────────────────────────────────────────
# Sister discipline regression guard — exports
# ─────────────────────────────────────────────────────────────────────────


def test_lock_timeout_constant_pinned() -> None:
    assert LOCK_TIMEOUT_SECONDS == 30


def test_default_staleness_window_pinned() -> None:
    assert DEFAULT_STALENESS_WINDOW_DAYS == 30


def test_canonical_path_is_omx_state() -> None:
    assert ".omx/state/" in str(PROBE_OUTCOMES_LEDGER_PATH)
    assert PROBE_OUTCOMES_LEDGER_PATH.name == "probe_outcomes.jsonl"


def test_probe_outcome_view_round_trip(tmp_ledger: tuple[Path, Path]) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=".omx/operator_authorize_recipes/x.yaml",
        probe_kind="k",
        verdict=VERDICT_DEFER,
        metric_name="m",
        metric_value=0.5,
        path=ledger,
        lock_path=lock,
    )
    view = latest_blocking_outcome_by_recipe(
        ".omx/operator_authorize_recipes/x.yaml",
        path=ledger,
    )
    assert view is not None
    assert view.verdict == "DEFER"


def test_jsonl_byte_stable_sort_keys(tmp_ledger: tuple[Path, Path]) -> None:
    """Per Catalog #245 sister discipline: payload uses json.dumps(sort_keys=True)
    so byte-level diff is meaningful across runs."""
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="p",
        substrate="s",
        recipe_path=None,
        probe_kind="k",
        verdict=VERDICT_PROCEED,
        metric_name="m",
        metric_value=0.0,
        path=ledger,
        lock_path=lock,
    )
    text = ledger.read_text(encoding="utf-8")
    # First key should be alphabetically first ("adjudicated_at_utc").
    first_line = text.splitlines()[0]
    parsed = json.loads(first_line)
    keys = list(parsed.keys())
    assert keys == sorted(keys)
