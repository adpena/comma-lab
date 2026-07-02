# SPDX-License-Identifier: MIT
"""Catalog #185 scope extension — strict-gate per-violation-count DRIFT ALARM.

Tests `check_strict_gate_violation_counts_within_declared_baseline` and its
shared helpers (`load_strict_gate_violation_baseline`,
`snapshot_strict_gate_violation_count`, `classify_strict_gate_drift`,
`evaluate_strict_gate_violation_drift`).

The alarm closes the #185 blind spot: #185's row-self-claim scan only
invokes a gate whose CLAUDE.md row literally says "live count: 0". A strict
gate whose row omits that phrase (canonical anchor: Catalog #344, which
silently drifted 0 -> 480) is never invoked and its drift is invisible. The
alarm reads a committed baseline manifest and flags any watched strict gate
whose LIVE violation count exceeds its DECLARED baseline_max (a NEW
regression), or a declared gate that is no longer a callable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tac.preflight as preflight_mod
from tac.preflight import (
    STRICT_GATE_DRIFT_AT_BASELINE,
    STRICT_GATE_DRIFT_MISSING_CALLABLE,
    STRICT_GATE_DRIFT_NOT_INVOKABLE,
    STRICT_GATE_DRIFT_OVER_BASELINE,
    STRICT_GATE_DRIFT_UNDER_BASELINE,
    PreflightError,
    check_strict_gate_violation_counts_within_declared_baseline,
    classify_strict_gate_drift,
    evaluate_strict_gate_violation_drift,
    load_strict_gate_violation_baseline,
    snapshot_strict_gate_violation_count,
)

_MANIFEST_RELPATH = ".omx/state/strict_gate_violation_baseline.json"


def _write_manifest(tmp_path: Path, gates: dict) -> Path:
    """Write a synthetic baseline manifest under a tmp repo root."""
    path = tmp_path / _MANIFEST_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gates": gates}), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────
# Phase 0: verdict constants
# ─────────────────────────────────────────────────────────────────────────


def test_verdict_constants_are_distinct():
    vals = {
        STRICT_GATE_DRIFT_OVER_BASELINE,
        STRICT_GATE_DRIFT_AT_BASELINE,
        STRICT_GATE_DRIFT_UNDER_BASELINE,
        STRICT_GATE_DRIFT_MISSING_CALLABLE,
        STRICT_GATE_DRIFT_NOT_INVOKABLE,
    }
    assert len(vals) == 5


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: load_strict_gate_violation_baseline (defensive parsing)
# ─────────────────────────────────────────────────────────────────────────


def test_load_missing_manifest_returns_empty(tmp_path):
    assert load_strict_gate_violation_baseline(tmp_path) == {}


def test_load_malformed_json_returns_empty(tmp_path):
    path = tmp_path / _MANIFEST_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_strict_gate_violation_baseline(tmp_path) == {}


def test_load_missing_gates_key_returns_empty(tmp_path):
    path = tmp_path / _MANIFEST_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_doc": "no gates"}), encoding="utf-8")
    assert load_strict_gate_violation_baseline(tmp_path) == {}


def test_load_valid_manifest_parses_entries(tmp_path):
    _write_manifest(tmp_path, {
        "check_x": {"catalog": 344, "baseline_max": 480, "reason": "backlog"},
    })
    b = load_strict_gate_violation_baseline(tmp_path)
    assert b["check_x"]["baseline_max"] == 480
    assert b["check_x"]["catalog"] == 344
    assert b["check_x"]["reason"] == "backlog"


def test_load_drops_bad_baseline_max_values(tmp_path):
    _write_manifest(tmp_path, {
        "check_neg": {"baseline_max": -1},
        "check_bool": {"baseline_max": True},  # bool is not a valid int here
        "check_str": {"baseline_max": "5"},
        "check_ok": {"baseline_max": 0},
    })
    b = load_strict_gate_violation_baseline(tmp_path)
    assert set(b) == {"check_ok"}


def test_load_non_dict_gates_dropped(tmp_path):
    _write_manifest(tmp_path, {
        "check_ok": {"baseline_max": 3},
        "check_bad": ["not", "a", "dict"],
    })
    b = load_strict_gate_violation_baseline(tmp_path)
    assert set(b) == {"check_ok"}


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: snapshot_strict_gate_violation_count
# ─────────────────────────────────────────────────────────────────────────


def test_snapshot_missing_callable_returns_none(tmp_path):
    assert snapshot_strict_gate_violation_count(
        "check_does_not_exist_anywhere", repo_root=tmp_path
    ) is None


def test_snapshot_counts_list_length(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["a", "b", "c"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_count", fake)
    assert snapshot_strict_gate_violation_count(
        "check_fake_count", repo_root=tmp_path
    ) == 3


def test_snapshot_non_list_return_is_zero(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return None
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_none", fake)
    assert snapshot_strict_gate_violation_count(
        "check_fake_none", repo_root=tmp_path
    ) == 0


def test_snapshot_falls_back_to_bare_signature(tmp_path, monkeypatch):
    """A gate that does NOT accept repo_root must still snapshot via the
    bare (strict, verbose) signature."""
    def fake(*, strict=False, verbose=False):
        return ["x"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_bare", fake)
    assert snapshot_strict_gate_violation_count(
        "check_fake_bare", repo_root=tmp_path
    ) == 1


def test_snapshot_incompatible_signature_returns_none(tmp_path, monkeypatch):
    def fake(*, unrelated_kwarg=None):
        return ["should not appear"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_badsig", fake)
    assert snapshot_strict_gate_violation_count(
        "check_fake_badsig", repo_root=tmp_path
    ) is None


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: classify_strict_gate_drift
# ─────────────────────────────────────────────────────────────────────────


def test_classify_over_baseline():
    assert classify_strict_gate_drift(500, 480) == STRICT_GATE_DRIFT_OVER_BASELINE


def test_classify_at_baseline():
    assert classify_strict_gate_drift(480, 480) == STRICT_GATE_DRIFT_AT_BASELINE


def test_classify_under_baseline():
    assert classify_strict_gate_drift(3, 480) == STRICT_GATE_DRIFT_UNDER_BASELINE


def test_classify_none_is_not_invokable():
    assert classify_strict_gate_drift(None, 0) == STRICT_GATE_DRIFT_NOT_INVOKABLE


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: evaluate_strict_gate_violation_drift
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_missing_callable_verdict(tmp_path):
    _write_manifest(tmp_path, {
        "check_vanished_gate": {"catalog": 999, "baseline_max": 0},
    })
    recs = evaluate_strict_gate_violation_drift(tmp_path)
    assert len(recs) == 1
    assert recs[0]["verdict"] == STRICT_GATE_DRIFT_MISSING_CALLABLE
    assert recs[0]["live_count"] is None


def test_evaluate_records_shape_and_over_baseline(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["v1", "v2", "v3"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_eval", fake)
    _write_manifest(tmp_path, {
        "check_fake_eval": {"catalog": 344, "baseline_max": 1},
    })
    recs = evaluate_strict_gate_violation_drift(tmp_path)
    assert recs[0]["verdict"] == STRICT_GATE_DRIFT_OVER_BASELINE
    assert recs[0]["live_count"] == 3
    assert recs[0]["baseline_max"] == 1
    assert recs[0]["catalog"] == 344


# ─────────────────────────────────────────────────────────────────────────
# Phase 5: check_strict_gate_violation_counts_within_declared_baseline
# ─────────────────────────────────────────────────────────────────────────


def test_check_no_manifest_returns_empty(tmp_path):
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_at_baseline_is_clean(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["a", "b"]  # count 2 == baseline
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_at", fake)
    _write_manifest(tmp_path, {
        "check_fake_at": {"catalog": 1, "baseline_max": 2},
    })
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_over_baseline_alarms(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["a", "b", "c", "d"]  # count 4 > baseline 2
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_over", fake)
    _write_manifest(tmp_path, {
        "check_fake_over": {"catalog": 344, "baseline_max": 2},
    })
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "check_fake_over" in v[0]
    assert "344" in v[0]
    assert "4" in v[0] and "2" in v[0]  # live count + baseline both named


def test_check_missing_callable_alarms(tmp_path):
    _write_manifest(tmp_path, {
        "check_gate_that_was_renamed": {"catalog": 7, "baseline_max": 0},
    })
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "check_gate_that_was_renamed" in v[0]
    assert "STALE" in v[0] or "stale" in v[0]


def test_check_under_baseline_is_advisory_not_violation(tmp_path, monkeypatch, capsys):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["only one"]  # count 1 < baseline 10
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_under", fake)
    _write_manifest(tmp_path, {
        "check_fake_under": {"catalog": 5, "baseline_max": 10},
    })
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=True
    )
    assert v == []  # UNDER_BASELINE is NOT a violation
    out = capsys.readouterr().out
    assert "advisory" in out.lower() or "tighten" in out.lower()


def test_check_strict_mode_raises_on_over_baseline(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["a", "b", "c"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_raise", fake)
    _write_manifest(tmp_path, {
        "check_fake_raise": {"catalog": 344, "baseline_max": 0},
    })
    with pytest.raises(PreflightError, match="drifted above"):
        check_strict_gate_violation_counts_within_declared_baseline(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_strict_mode_clean_does_not_raise(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return []
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_clean", fake)
    _write_manifest(tmp_path, {
        "check_fake_clean": {"catalog": 1, "baseline_max": 0},
    })
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []


def test_check_verbose_alarm_output(tmp_path, monkeypatch, capsys):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return ["a", "b"]
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_vout", fake)
    _write_manifest(tmp_path, {
        "check_fake_vout": {"catalog": 344, "baseline_max": 0},
    })
    check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=tmp_path, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "ALARM" in out


def test_check_str_repo_root_accepted(tmp_path, monkeypatch):
    def fake(*, strict=False, verbose=False, repo_root=None):
        return []
    monkeypatch.setitem(preflight_mod.__dict__, "check_fake_str", fake)
    _write_manifest(tmp_path, {"check_fake_str": {"baseline_max": 0}})
    v = check_strict_gate_violation_counts_within_declared_baseline(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert v == []


# ─────────────────────────────────────────────────────────────────────────
# Phase 6: LIVE repo — the alarm must be GREEN (known backlog declared)
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_alarm_is_green():
    """The seeded manifest declares the known backlog (#344 -> 480 etc.), so
    the alarm must be clean on the live repo. It fires only on a NEW drift
    PAST a declared baseline — the whole point of the baseline manifest."""
    v = check_strict_gate_violation_counts_within_declared_baseline(
        strict=False, verbose=False
    )
    assert v == [], (
        "Strict-gate drift alarm fired on the live repo — a watched strict "
        "gate drifted PAST its declared baseline:\n"
        + "\n".join(f"  - {x[:240]}" for x in v)
    )


def test_live_repo_watches_catalog_344():
    """Regression guard: #344 MUST be in the watched set (it is the canonical
    anchor — the gate #185's row-self-claim scan is blind to)."""
    recs = evaluate_strict_gate_violation_drift()
    names = {r["check_name"] for r in recs}
    assert "check_empirical_finding_memo_references_canonical_equation" in names
