# SPDX-License-Identifier: MIT
"""Tests for tac.verdicts.emit — the canonical verdict-emission helper.

Covers the REFUSAL surface (missing scope / missing rows for measured claim /
missing composition / missing constraint_carved / negative-without-reformulation)
and the atomic-write roundtrip.
"""
from __future__ import annotations

import json

import pytest

from tac.verdicts import (
    AxisTag,
    Composition,
    InteractionSign,
    MeasurementRow,
    Provenance,
    ReviewStatus,
    ScopeLevel,
    VerdictEmitError,
    VerdictScope,
    emit_verdict,
    verdict_payload,
)


@pytest.fixture(autouse=True)
def _isolate_bulletin(tmp_path, monkeypatch):
    """#389: emit_verdict now posts a fail-open verdict_landed bulletin. Redirect the
    bulletin store to a tmp path so these unit tests never pollute the live
    ``.omx/state/session_events.jsonl`` feed (a concurrent seal agent reads it)."""
    from tac.session_bus import bulletin as _bull

    monkeypatch.setattr(_bull, "DEFAULT_BULLETIN_PATH", tmp_path / "ev.jsonl")
    monkeypatch.setattr(_bull, "DEFAULT_BULLETIN_LOCK_PATH", tmp_path / ".ev.lock")


def _row() -> MeasurementRow:
    return MeasurementRow(
        value=0.0031,
        units="d_seg",
        axis_tag=AxisTag.THROUGH_R,
        provenance=Provenance(git_sha="abc", tool="t"),
        n_samples=600,
        review_status=ReviewStatus.REVIEWED,
    )


def _scope() -> VerdictScope:
    return VerdictScope(level=ScopeLevel.FORMULATION, scoped_to="ground-frame xi transport")


def _comp() -> Composition:
    return Composition(sign=InteractionSign.ANTAGONISTIC, active_levers=("basis-match",), measured=True)


def _emit(tmp_path, **over):
    kw = {
        "path": tmp_path / "v.json",
        "verdict": "NO-GO",
        "scope": _scope(),
        "rows": [_row()],
        "composition": _comp(),
        "constraint_carved": "removes the ground-frame ego-freeze chart region",
    }
    kw.update(over)
    return emit_verdict(**kw)


# ------------------------------- VerdictScope ----------------------------------
def test_scope_requires_scoped_to():
    with pytest.raises(VerdictEmitError):
        VerdictScope(level=ScopeLevel.INSTANCE, scoped_to="")


def test_scope_family_requires_evidence():
    with pytest.raises(VerdictEmitError):
        VerdictScope(level=ScopeLevel.FAMILY, scoped_to="the whole carrier family")
    ok = VerdictScope(
        level=ScopeLevel.FAMILY,
        scoped_to="cheap-carrier family",
        family_evidence="killed across 5 structurally distinct formulations",
    )
    assert ok.level is ScopeLevel.FAMILY


def test_scope_level_coerce_string():
    s = VerdictScope(level="instance", scoped_to="one config")
    assert s.level is ScopeLevel.INSTANCE
    assert s.level.rank == 0


# ------------------------------- Composition (P12) -----------------------------
def test_composition_requires_sign_or_deferred():
    with pytest.raises(VerdictEmitError):
        Composition()  # neither sign nor deferred


def test_composition_deferred_cannot_carry_sign():
    with pytest.raises(VerdictEmitError):
        Composition(sign=InteractionSign.ADDITIVE, deferred_to_ab_protocol=True)


def test_composition_deferred_factory():
    c = Composition.deferred()
    assert c.deferred_to_ab_protocol
    assert c.to_json_dict()["sign"] is None


# ------------------------------- emit refusals ---------------------------------
def test_emit_refuses_missing_scope(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, scope=None)
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, scope="formulation")  # a bare string is not a VerdictScope


def test_emit_refuses_measured_claim_without_rows(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, rows=[])


def test_emit_allows_no_rows_when_not_measured(tmp_path):
    # A pure design/DEFER verdict legitimately carries no measurement rows.
    p = _emit(tmp_path, verdict="DEFER", rows=[], measured=False, is_negative=False)
    assert json.loads(p.read_text())["measured"] is False


def test_emit_refuses_missing_composition_P12(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, composition=None)


def test_emit_accepts_deferred_composition_sentinel(tmp_path):
    p = _emit(tmp_path, composition="deferred_to_ab_protocol")
    assert json.loads(p.read_text())["composition"]["deferred_to_ab_protocol"] is True


def test_emit_refuses_missing_constraint_carved_P10(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, constraint_carved="")


def test_emit_negative_requires_reformulation_queue(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, is_negative=True, reformulation_queue=None)


def test_emit_negative_empty_queue_requires_reason(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, is_negative=True, reformulation_queue=[])
    # empty WITH reason is allowed (genuine exhaustion)
    p = _emit(
        tmp_path,
        is_negative=True,
        reformulation_queue=[],
        reformulation_empty_reason="all 5 formulations killed at n600",
    )
    assert json.loads(p.read_text())["reformulation_queue"] == []


def test_emit_refuses_bad_row_type(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, rows=[{"value": 1.0}])


def test_emit_refuses_empty_verdict(tmp_path):
    with pytest.raises(VerdictEmitError):
        _emit(tmp_path, verdict="  ")


# ------------------------------- roundtrip / atomic ----------------------------
def test_emit_roundtrip_full_payload(tmp_path):
    p = _emit(
        tmp_path,
        is_negative=True,
        reformulation_queue=["flip-weighted median", "per-edge fresh field"],
        stores_consulted=["DAG FEED-07c", "MEMORY L68"],
    )
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == "verdict.v1"
    assert payload["verdict"] == "NO-GO"
    assert payload["scope"]["level"] == "formulation"
    assert payload["composition"]["sign"] == "antagonistic"
    assert payload["constraint_carved"].startswith("removes")
    assert payload["reformulation_queue"] == ["flip-weighted median", "per-edge fresh field"]
    assert payload["rows"][0]["axis_tag"] == "[through-R]"
    assert payload["stores_consulted"] == ["DAG FEED-07c", "MEMORY L68"]


def test_emit_atomic_no_tmp_left_behind(tmp_path):
    p = _emit(tmp_path)
    assert p.exists()
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []


def test_emit_overwrite_is_atomic_replace(tmp_path):
    p1 = _emit(tmp_path, verdict="NO-GO")
    p2 = _emit(tmp_path, verdict="PROCEED", is_negative=False)
    assert p1 == p2
    assert json.loads(p2.read_text())["verdict"] == "PROCEED"


# --- verdict_payload: the EMBEDDABLE surface (#936, ddm_vw1) -----------------------------


def test_verdict_payload_returns_validated_dict_without_writing(tmp_path):
    """The embeddable half: same validation, no file. Motivated by a MEASURED adoption
    failure — 474 of 10,728 tracked .py files hand-roll a "verdict_scope" key while
    emit_verdict had 0 production callers, because embedding had no typed path."""
    payload = verdict_payload(
        verdict="NO-GO",
        scope=VerdictScope(level=ScopeLevel.FORMULATION, scoped_to="this coder on this residual"),
        composition=Composition.deferred(),
        constraint_carved="pins the axis-aligned offset coder as non-viable",
        measured=False,
        is_negative=True,
        reformulation_queue=["true Euclidean-normal offset", "joint 2-D context model"],
    )
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "verdict.v1"
    assert payload["scope"]["level"] == "formulation"
    assert payload["is_negative"] is True
    assert len(payload["reformulation_queue"]) == 2
    # It writes NOTHING: the whole point is embedding in a host artifact.
    assert list(tmp_path.iterdir()) == []
    # And it is embeddable + serializable as-is.
    import json as _json
    host = {"equation_id": "v8_geometric", "verdict": payload}
    assert _json.loads(_json.dumps(host))["verdict"]["scope"]["level"] == "formulation"


def test_verdict_payload_enforces_the_same_refusals_as_emit():
    """The embeddable path must not be a validation bypass — that would make it a
    laundering channel for exactly the claims emit_verdict refuses."""
    # FAMILY scope without family_evidence refuses at VerdictScope construction.
    with pytest.raises(VerdictEmitError):
        VerdictScope(level=ScopeLevel.FAMILY, scoped_to="the whole family")
    # A negative with no reformulation queue refuses.
    with pytest.raises(VerdictEmitError):
        verdict_payload(
            verdict="KILL",
            scope=VerdictScope(level=ScopeLevel.INSTANCE, scoped_to="one config"),
            composition=Composition.deferred(),
            constraint_carved="x",
            measured=False,
            is_negative=True,
        )
    # A measured claim with no rows refuses.
    with pytest.raises(VerdictEmitError):
        verdict_payload(
            verdict="PROCEED",
            scope=VerdictScope(level=ScopeLevel.INSTANCE, scoped_to="one config"),
            composition=Composition.deferred(),
            constraint_carved="x",
            measured=True,
        )


def test_emit_verdict_and_verdict_payload_share_one_validation_path(tmp_path):
    """emit_verdict must be a thin write-wrapper: identical payload modulo timestamp."""
    import json as _json

    kw = dict(
        verdict="DEFER",
        scope=VerdictScope(level=ScopeLevel.INSTANCE, scoped_to="one config"),
        composition=Composition.deferred(),
        constraint_carved="pins nothing yet",
        measured=False,
    )
    direct = verdict_payload(**kw)
    written = _json.loads(emit_verdict(tmp_path / "v.json", **kw).read_text())
    direct.pop("emitted_at_utc"), written.pop("emitted_at_utc")
    assert direct == written
