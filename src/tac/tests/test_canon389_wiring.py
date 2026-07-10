# SPDX-License-Identifier: MIT
"""#389 canonicalization wiring: producer post_event fail-open paths.

Covers the two new bulletin producers wired in the #389 sweep:
* ``tac.verdicts.emit_verdict`` -> ``verdict_landed`` (after the durable JSON write);
* ``tac.session_bus.recovery_manifest.register_inflight / complete`` ->
  ``agent_spawned`` / ``agent_completed`` (after the durable checkpoint write).

Both are FAIL-OPEN: a raising bulletin must never break the authoritative write. Each
test isolates the bulletin store to a tmp path (monkeypatching the module defaults) so
the real ``.omx/state`` feed is untouched.
"""
from __future__ import annotations

import pytest

from tac import session_bus  # noqa: F401  (ensure package import path is live)
from tac.session_bus import bulletin as bull
from tac.session_bus import recovery_manifest as rm
from tac.verdicts import Composition, ScopeLevel, VerdictScope, emit_verdict
from tac.verdicts.measurement_row import (
    AxisTag,
    MeasurementRow,
    Provenance,
    ReviewStatus,
)


@pytest.fixture
def tmp_bulletin(tmp_path, monkeypatch):
    """Redirect the default bulletin store + lock to a tmp path for one test."""
    path = tmp_path / "session_events.jsonl"
    lock = tmp_path / ".session_events.jsonl.lock"
    monkeypatch.setattr(bull, "DEFAULT_BULLETIN_PATH", path)
    monkeypatch.setattr(bull, "DEFAULT_BULLETIN_LOCK_PATH", lock)
    return path


def _a_row():
    return MeasurementRow(
        value=0.0042,
        units="fraction",
        axis_tag=AxisTag.THROUGH_R,
        provenance=Provenance(git_sha="deadbeef", tool="test"),
        n_samples=600,
        review_status=ReviewStatus.PROVISIONAL,
        quantity="d_seg",
    )


def test_emit_verdict_posts_verdict_landed_bulletin(tmp_path, tmp_bulletin):
    out = tmp_path / "verdict.json"
    emit_verdict(
        out,
        verdict="PROCEED",
        scope=VerdictScope(ScopeLevel.INSTANCE, scoped_to="lane_band_probe"),
        rows=[_a_row()],
        composition=Composition(sign="orthogonal"),
        constraint_carved="pins the lane-band coverage>=0.5 region",
    )
    assert out.exists()  # durable write happened
    events = bull.read_events(bulletin_path=tmp_bulletin)
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == bull.EVENT_VERDICT_LANDED
    assert ev["subject"] == "lane_band_probe"
    assert ev["agent_label"] == "emit_verdict"
    assert ev["payload"]["verdict"] == "PROCEED"
    assert ev["payload"]["scope_level"] == "instance"
    assert ev["payload"]["n_rows"] == 1
    assert ev["payload"]["composition_sign"] == "orthogonal"
    # a seal-round staleness_check on that surface picks the landing up.
    report = bull.staleness_check(["lane_band"], bulletin_path=tmp_bulletin)
    assert report.stale is True


def test_emit_verdict_is_fail_open_if_bulletin_raises(tmp_path, monkeypatch):
    """A raising post_event must NOT break the (already durable) verdict write."""
    out = tmp_path / "verdict.json"

    def _boom(*a, **k):
        raise RuntimeError("bulletin store exploded")

    monkeypatch.setattr(bull, "post_event", _boom)
    # Must not raise.
    emit_verdict(
        out,
        verdict="NO-GO",
        scope=VerdictScope(ScopeLevel.INSTANCE, scoped_to="x"),
        rows=[_a_row()],
        composition=Composition.deferred(),
        constraint_carved="removes config X",
        is_negative=True,
        reformulation_queue=["try Y", "try Z"],
    )
    assert out.exists()  # the durable verdict still landed


def test_recovery_lifecycle_posts_are_fail_open(monkeypatch):
    """register_inflight / complete post lifecycle events but swallow bulletin errors."""

    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(bull, "post_event", _boom)
    # _post_lifecycle must swallow; a direct call must not raise.
    rm._post_lifecycle("agent_spawned", "sid-1", "lane_x", "parent-1")


def test_post_lifecycle_writes_event(tmp_bulletin):
    rm._post_lifecycle("agent_completed", "sid-xyz", None, None)
    events = bull.read_events(bulletin_path=tmp_bulletin)
    assert len(events) == 1
    assert events[0]["kind"] == bull.EVENT_AGENT_COMPLETED
    assert events[0]["subject"] == "sid-xyz"
    assert events[0]["agent_label"] == "recovery_manifest"


def test_register_and_complete_call_lifecycle(monkeypatch):
    """register_inflight -> agent_spawned, complete -> agent_completed (wiring check).

    The checkpoint tool is stubbed so no real store is touched; the lifecycle post is
    spied so we assert the semantic transition fires the right kind.
    """
    calls = []
    monkeypatch.setattr(
        rm, "_post_lifecycle", lambda kind, sid, lane, parent: calls.append((kind, sid))
    )

    class _StubTool:
        def append_checkpoint(self, **kw):
            return dict(kw)

    monkeypatch.setattr(rm, "_checkpoint_tool", lambda: _StubTool())

    rm.register_inflight("sid-A", "resume ctx here", lane_id="lane_a")
    rm.complete("sid-A")
    assert ("agent_spawned", "sid-A") in calls
    assert ("agent_completed", "sid-A") in calls
