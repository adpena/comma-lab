"""Tests for the harness-engineering crosswalk ADOPT items A1-A4
(.omx/research/harness_engineering_crosswalk_20260719_codex.md).

A1: FailureEventV2 typed lifecycle + migration (the bulk — 15+ cases).
A2: watchdog cadence (durable receipt, idempotent dedup, buffered-log ≠ death).
A3: hard self-review round cap (monotonic cap vs resettable clean-pass streak).
A4: provider error classifier + pre-dispatch lexical lint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tac import harness_failure_ledger as L
from tac import provider_error_classifier as P
from tac import self_review_cap as R


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "harness_failure_ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------- A1: schema
def test_v2_event_roundtrips_and_validates():
    ev = L.FailureEventV2(class_id="c1", event_kind="OBSERVED", ts="2026-07-19T00:00:00Z",
                          resolution_state="OPEN")
    assert ev.schema == "harness_failure.v2"
    assert ev.to_dict()["class_id"] == "c1"


def test_v2_rejects_unknown_event_kind():
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="c", event_kind="BOGUS", ts="t"))


def test_v2_rejects_missing_class_id():
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="   ", event_kind="OBSERVED", ts="t"))


def test_v2_rejects_bad_resolution_state():
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="c", event_kind="OBSERVED", ts="t",
                                        resolution_state="MOSTLY_OK"))


def test_v2_rejects_bad_handoff_and_shape():
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="c", event_kind="OBSERVED", ts="t",
                                        earliest_failed_handoff="nowhere"))
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="c", event_kind="OBSERVED", ts="t",
                                        failure_shape="vibes"))


def test_v2_recurrence_requires_parent_class_id():
    with pytest.raises(L.FailureLedgerV2Error):
        L._validate_v2(L.FailureEventV2(class_id="c", event_kind="RECURRENCE", ts="t"))
    # with a parent it validates
    L._validate_v2(L.FailureEventV2(class_id="c", event_kind="RECURRENCE", ts="t",
                                    parent_class_id="c"))


# ---------------------------------------------------------------------------- A1: aliases
def test_normalize_class_id_collapses_both_known_aliases():
    assert L.normalize_class_id("dashboard_false_FAIL_at_init") == \
        "dashboard_hardcoded_gate_boundary_false_fail_at_init"
    assert L.normalize_class_id("codex_probe_token_limit_death_incomplete_wip") == \
        "codex_probe_token_limit_death_incomplete_wip_20260712"
    assert L.normalize_class_id("some_other_id") == "some_other_id"  # non-alias untouched


# ------------------------------------------------------------ A1: prose-vs-structured closure
def test_resolution_state_never_inferred_from_prose():
    prose = {"failure_class": "x", "resolution": "PERMANENT FIX LANDED aab928: did the thing"}
    # a non-enum prose resolution → VERIFY_PENDING, NOT CLOSED
    assert L.legacy_row_resolution_state(prose) == "VERIFY_PENDING"


def test_resolution_state_enum_and_status_and_bool():
    assert L.legacy_row_resolution_state({"resolution": "class-fixed"}) == "FIX_ONLY"
    assert L.legacy_row_resolution_state({"resolution": "worked-around"}) == "FIX_ONLY"
    assert L.legacy_row_resolution_state({"resolution": "gate-landed"}) == "CLOSED"
    assert L.legacy_row_resolution_state({"status": "open_prevention_owed"}) == "OPEN"
    assert L.legacy_row_resolution_state({"status": "resolved_with_prevention_owed"}) == "FIX_ONLY"
    assert L.legacy_row_resolution_state({"status": "recurrence_resolved_cure_confirmed"}) == "CLOSED"
    assert L.legacy_row_resolution_state({"resolved": True}) == "CLOSED"
    assert L.legacy_row_resolution_state({"terminal_cause": "nothing structured"}) == "OPEN"


def test_status_beats_resolution_prose_precedence():
    # provider row shape: prose in resolution + structured status → status wins (FIX_ONLY)
    row = {"failure_class": "provider", "status": "resolved_with_prevention_owed",
           "resolution": "salvage-commit to branch; PREVENTION owed: avoid phrasing"}
    assert L.legacy_row_resolution_state(row) == "FIX_ONLY"


# ---------------------------------------------------------------------------- A1: projection
def _legacy_fixture() -> list[dict]:
    return [
        {"failure_id": "sig", "event": "opened", "surface": "gate", "ts": "2026-07-01T00:00:00Z",
         "terminal_cause": "x", "schema": "harness_failure.v1"},
        {"failure_id": "sig", "event": "resolution", "resolution": "class-fixed",
         "ts": "2026-07-02T00:00:00Z", "schema": "harness_failure.v1"},
        {"failure_id": "sig", "event": "recurrence", "ts": "2026-07-03T00:00:00Z",
         "status": "recurrence_resolved_cure_confirmed", "schema": "harness_failure.v1"},
        {"failure_class": "phantom", "ts_utc": "2026-07-04T00:00:00Z",
         "owed_fix": "cron the watchdog"},
        {"class_id": "merge", "first_seen_utc": "2026-07-05", "resolved": True,
         "written_at_utc": "2026-07-05T00:00:00Z"},
        {"failure_id": "codex_probe_token_limit_death_incomplete_wip",
         "event": "opened", "surface": "subagent", "terminal_cause": "x",
         "ts": "2026-07-06T00:00:00Z", "schema": "harness_failure.v1"},
        {"failure_id": "codex_probe_token_limit_death_incomplete_wip_20260712",
         "event": "resolution", "resolution": "worked-around", "ts": "2026-07-07T00:00:00Z",
         "schema": "harness_failure.v1"},
    ]


def test_projection_one_row_per_semantic_class_with_alias_collapse():
    proj = L.project_legacy_rows_to_v2(_legacy_fixture())
    ids = {p.class_id for p in proj}
    # 4 semantic classes: sig, phantom, merge, and the collapsed token-limit alias pair
    assert ids == {"sig", "phantom", "merge",
                   "codex_probe_token_limit_death_incomplete_wip_20260712"}
    assert len(proj) == 4  # alias pair folded into ONE row
    by = {p.class_id: p for p in proj}
    assert by["sig"].resolution_state == "CLOSED"      # latest status = cure_confirmed
    assert by["sig"].recurrence_count == 1
    assert by["phantom"].resolution_state == "OPEN"    # owed fix, no structured closure
    assert by["merge"].resolution_state == "CLOSED"    # resolved: True
    tok = by["codex_probe_token_limit_death_incomplete_wip_20260712"]
    assert tok.resolution_state == "FIX_ONLY"          # worked-around
    assert "codex_probe_token_limit_death_incomplete_wip" in tok.legacy_alias


def test_projection_never_emits_phantom_question_mark_class():
    rows = [*_legacy_fixture(), {"note": "row with no class key at all"}]
    proj = L.project_legacy_rows_to_v2(rows)
    assert "?" not in {p.class_id for p in proj}
    assert "" not in {p.class_id for p in proj}


def test_projection_is_idempotent_skips_existing_v2(tmp_path):
    path = _write_rows(tmp_path, _legacy_fixture())
    first = L.project_legacy_rows_to_v2(L.load_raw_rows(path))
    for p in first:
        L.append_failure_event_v2(p, path=path)
    # projecting again over the now-migrated file yields NO new projections
    again = L.project_legacy_rows_to_v2(L.load_raw_rows(path))
    assert again == []


# ---------------------------------------------------------------------------- A1: migration IO
def test_migration_appends_only_and_preserves_originals(tmp_path):
    path = _write_rows(tmp_path, _legacy_fixture())
    before = path.read_text(encoding="utf-8")
    proj = L.project_legacy_rows_to_v2(L.load_raw_rows(path))
    for p in proj:
        L.append_failure_event_v2(p, path=path)
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)  # originals byte-preserved as a prefix (append-only)
    assert len(L.load_failure_events_v2(path)) == 4


def test_summarize_v2_open_notclosed_recurrent(tmp_path):
    path = _write_rows(tmp_path, _legacy_fixture())
    for p in L.project_legacy_rows_to_v2(L.load_raw_rows(path)):
        L.append_failure_event_v2(p, path=path)
    s = L.summarize_v2(path)
    assert s["classes"] == 4
    assert s["unresolved"] == ["phantom"]                      # only OPEN
    assert set(s["not_closed"]) == {"phantom",
        "codex_probe_token_limit_death_incomplete_wip_20260712"}  # OPEN + FIX_ONLY
    assert s["recurrent"] == ["sig"]


def test_failure_states_v2_recurrence_and_reopen(tmp_path):
    path = tmp_path / "l.jsonl"
    L.append_failure_event_v2(L.FailureEventV2("c", "OBSERVED", "2026-07-01T00:00:00Z"), path=path)
    L.append_failure_event_v2(L.FailureEventV2("c", "FIX_LANDED", "2026-07-02T00:00:00Z"), path=path)
    L.append_failure_event_v2(L.FailureEventV2("c", "VERIFIED_CLOSED", "2026-07-03T00:00:00Z"), path=path)
    st = L.failure_states_v2(path)["c"]
    assert st.resolution_state == "CLOSED" and st.is_resolved
    # a reopen falsifies the closure
    L.append_failure_event_v2(L.FailureEventV2("c", "REOPENED", "2026-07-04T00:00:00Z",
                                               parent_class_id="c", resolution_state="OPEN"), path=path)
    st = L.failure_states_v2(path)["c"]
    assert st.resolution_state == "OPEN" and not st.is_resolved


def test_load_failure_events_v2_ignores_legacy_rows(tmp_path):
    path = _write_rows(tmp_path, _legacy_fixture())
    assert L.load_failure_events_v2(path) == []          # no V2 rows yet
    assert len(L.load_raw_rows(path)) == len(_legacy_fixture())


# ---------------------------------------------------------------------------- A1: real ledger
def test_real_ledger_projects_to_20_classes_with_correct_states():
    """The crosswalk's A1 falsifiable gate against the LIVE ledger's LEGACY rows.

    Projects the legacy (non-V2) subset directly so the assertion is stable regardless of any
    later V2 lifecycle events (e.g. FIX_LANDED advancement) appended after migration — it tests
    the migration DERIVATION on real legacy data, not the evolving live state."""
    legacy = [r for r in L.load_raw_rows() if r.get("schema") != L.SCHEMA_VERSION_V2]
    proj = L.project_legacy_rows_to_v2(legacy)
    states = {p.class_id: p.resolution_state for p in proj}
    assert len(states) == 20, f"expected 20 semantic classes, got {len(states)}"
    assert "?" not in states and "" not in states
    assert states["phantom_death_buffered_log_plus_misfired_grep_liveness"] == "OPEN"
    assert states["provider_content_filter_false_positive_kills_arm"] == "FIX_ONLY"
    assert states["arm_review_spiral_unbounded_seal_loop"] == "OPEN"
    # 2026-07-19: a legacy recurrence row (resolved=false, MAIN-shell rc=144 x2) was appended
    # AFTER the agent's worktree forked; the legacy derivation now correctly yields OPEN.
    assert states["sigurg_144_harness_kills_bg_bash_process_group"] == "OPEN"


# ---------------------------------------------------------------------------- A2: watchdog
def _load_cadence():
    spec = importlib.util.spec_from_file_location(
        "wcadence",
        str(Path(__file__).resolve().parents[3] / "tools" / "witness_chain_watchdog_cadence.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_watchdog_quiet_alive_never_actions(tmp_path):
    W = _load_cadence()
    W._wcw.scan = lambda **k: [{"label": "c2", "pid": 1, "verdict": "RUNNING_QUIET", "run_dir": "/x"}]
    r = W.run_cadence(receipts_path=tmp_path / "rec.jsonl")
    assert r["new_action_count"] == 0
    assert "RUNNING_QUIET" in r["benign_verdicts"]


def test_watchdog_dead_actions_once_then_dedups(tmp_path):
    W = _load_cadence()
    rec = tmp_path / "rec.jsonl"
    W._wcw.scan = lambda **k: [{"label": "c2", "pid": 1, "verdict": "CHAIN_DEAD_NO_RECEIPT",
                                "run_dir": "/x"}]
    r1 = W.run_cadence(receipts_path=rec)
    r2 = W.run_cadence(receipts_path=rec)  # identical observation replayed
    assert r1["new_action_count"] == 1
    assert r2["new_action_count"] == 0 and r2["suppressed_duplicate_count"] == 1
    # a durable receipt was written per run
    assert len(rec.read_text().splitlines()) == 2


def test_watchdog_unreadable_registry_flags(tmp_path):
    W = _load_cadence()
    W._wcw.scan = lambda **k: [{"verdict": "REGISTRY_UNREADABLE", "ts": "t"}]
    r = W.run_cadence(receipts_path=tmp_path / "rec.jsonl")
    assert r["registry_unreadable"] is True and r["new_action_count"] == 0


# ---------------------------------------------------------------------------- A3: review cap
def test_five_non_clean_rounds_escalate(tmp_path):
    path = tmp_path / "rounds.jsonl"
    for _ in range(5):
        R.record_round("arm", clean=False, path=path)
    allowed, verdict = R.may_start_round("arm", path=path)
    assert allowed is False and verdict == "ESCALATE_MAIN"


def test_finding_resets_only_clean_streak_not_hard_cap(tmp_path):
    path = tmp_path / "rounds.jsonl"
    R.record_round("arm", clean=True, path=path)
    R.record_round("arm", clean=True, path=path)
    st = R.record_round("arm", clean=False, path=path)  # finding in round 3
    assert st.rounds_completed == 3        # hard cap counter NOT reset
    assert st.clean_pass_streak == 0       # only the seal streak reset
    assert st.verdict == "PROCEED"


def test_three_clean_passes_seal_before_cap(tmp_path):
    path = tmp_path / "rounds.jsonl"
    for _ in range(3):
        R.record_round("arm", clean=True, path=path)
    allowed, verdict = R.may_start_round("arm", path=path)
    assert verdict == "SEALED" and allowed is False
    assert R.self_review_verdict("arm", path=path) == "SEALED"


def test_record_round_requires_arm_id(tmp_path):
    with pytest.raises(R.SelfReviewCapError):
        R.record_round("  ", clean=True, path=tmp_path / "r.jsonl")


# ---------------------------------------------------------------------------- A4: provider
def test_classify_crosswalk_named_live_examples():
    assert P.classify_provider_error("The 'sol' model is not supported").error_class == "provider_model"
    v = P.classify_provider_error("You've hit your usage limit, try again at Jul 24th")
    assert v.error_class == "provider_quota" and v.is_provider_fault


def test_classify_content_filter_and_arm_and_unknown():
    assert P.classify_provider_error("blocked by the safety classifier").error_class == \
        "provider_content_filter"
    assert P.classify_provider_error("Traceback ... AssertionError").error_class == "arm_failure"
    assert P.is_provider_fault("Traceback ...") is False
    assert P.classify_provider_error("").error_class == "unknown"


def test_provider_fault_checked_before_arm_failure():
    # a quota message that also contains 'rc=1' style noise still classifies as provider
    v = P.classify_provider_error("rc=1: You've hit your usage limit; try again at Jul 24th")
    assert v.error_class == "provider_quota" and v.is_provider_fault


def test_lint_detects_security_scanner_phrasing_with_suggestions():
    findings = P.lint_provider_trigger_phrasing(
        "The gate runs a destructive-operation scan; the strict-prefix guard refuses bad input."
    )
    triggers = {f.trigger for f in findings}
    assert "destructive-operation scan" in triggers
    assert "strict-prefix … refuses" in triggers
    assert all(f.suggestion for f in findings)  # every finding carries neutral wording


def test_lint_clean_prose_and_allowlist():
    assert P.lint_provider_trigger_phrasing("Compress the archive and measure d_seg.") == []
    # allowlisted trigger is not reported (explicit reviewed exception)
    assert P.lint_provider_trigger_phrasing(
        "destructive-operation scan", allow=("destructive-operation scan",)) == []


# ------------------------------------------------------------------ A1: preflight gate
def _mk_ledger(tmp_path: Path, rows: list[dict]) -> Path:
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    p = tmp_path / ".omx" / "state" / "harness_failure_ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_preflight_gate_warns_on_unmigrated_and_ok_when_migrated(tmp_path):
    from tac import preflight
    _mk_ledger(tmp_path, _legacy_fixture())
    warn = preflight.check_harness_failure_ledger_v2_hygiene(repo_root=tmp_path, strict=False)
    assert any("lack a canonical V2 projection" in v for v in warn)
    with pytest.raises(preflight.PreflightError):
        preflight.check_harness_failure_ledger_v2_hygiene(repo_root=tmp_path, strict=True)
    # migrate → gate clean
    path = tmp_path / ".omx" / "state" / "harness_failure_ledger.jsonl"
    for p in L.project_legacy_rows_to_v2(L.load_raw_rows(path)):
        L.append_failure_event_v2(p, path=path)
    assert preflight.check_harness_failure_ledger_v2_hygiene(repo_root=tmp_path, strict=True) == []


def test_preflight_gate_flags_unknown_lifecycle_shape(tmp_path):
    from tac import preflight
    _mk_ledger(tmp_path, [{"schema": "harness_failure.v2", "class_id": "c",
                           "event_kind": "TELEPORTED", "ts": "t"}])
    warn = preflight.check_harness_failure_ledger_v2_hygiene(repo_root=tmp_path, strict=False)
    assert any("unknown event_kind" in v for v in warn)


def test_preflight_gate_ok_when_no_ledger(tmp_path):
    from tac import preflight
    assert preflight.check_harness_failure_ledger_v2_hygiene(repo_root=tmp_path, strict=True) == []
