"""Unit tests for the codex landing review gate (tools/codex_landing_review_gate.py).

Pure-logic tests over ``compute_pending`` + disposition semantics — no live fleet,
no pgrep dependency (alive_fn is injected). Mirrors the triality-drift-detector
testing style: the load-bearing classification lives in module functions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "codex_landing_review_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("codex_landing_review_gate", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codex_landing_review_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load()


def _deleg(label, stamp, *, done=True, tmp=None):
    """A delegation dict; ``done`` writes a real .done marker so _landing_done sees it."""
    d = {"label": label, "stamp": stamp}
    if done and tmp is not None:
        marker = tmp / f"{label}_{stamp}.done"
        marker.write_text("rc=0\n")
        d["done_marker"] = str(marker)
        d["last"] = str(marker)  # reuse for mtime
    return d


# ---- compute_pending: the core classifier ----------------------------------
def test_landed_dead_undispositioned_is_pending(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: False)
    assert len(pending) == 1
    assert pending[0]["label"] == "arm_a"
    assert pending[0]["reason"] == "undispositioned"


def test_landed_but_alive_is_not_pending(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    # process still alive ⇒ held-by-design, NOT a landing to review
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: True)
    assert pending == []


def test_not_done_is_not_pending(tmp_path):
    d = {"label": "arm_a", "stamp": "S1"}  # no marker ⇒ not landed
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: False)
    assert pending == []


@pytest.mark.parametrize("status", ["reviewed_committed", "respawned", "closed"])
def test_terminal_disposition_clears(tmp_path, status):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    disp = {("arm_a", "S1"): {"label": "arm_a", "stamp": "S1", "status": status}}
    pending = G.compute_pending([d], disp, alive_fn=lambda arm, s: False)
    assert pending == []


def test_held_entangled_with_live_blocker_is_not_pending(tmp_path):
    landed = _deleg("arm_a", "S1", tmp=tmp_path)
    blocker = {"label": "live_b", "stamp": "S2"}  # a delegation for the blocker
    disp = {("arm_a", "S1"): {
        "label": "arm_a", "stamp": "S1", "status": "held_entangled",
        "blocked_by": ["live_b"]}}
    # live_b is alive ⇒ legitimately held
    pending = G.compute_pending(
        [landed, blocker], disp, alive_fn=lambda arm, s: arm == "live_b")
    assert pending == []


def test_held_entangled_resurfaces_when_blockers_die(tmp_path):
    landed = _deleg("arm_a", "S1", tmp=tmp_path)
    blocker = {"label": "live_b", "stamp": "S2"}
    disp = {("arm_a", "S1"): {
        "label": "arm_a", "stamp": "S1", "status": "held_entangled",
        "blocked_by": ["live_b"]}}
    # everything dead ⇒ the held landing MUST resurface (held is not a grave)
    pending = G.compute_pending([landed, blocker], disp, alive_fn=lambda arm, s: False)
    assert len(pending) == 1
    assert pending[0]["reason"] == "held_entangled_unblocked"


def test_unknown_status_is_pending_failsafe(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    disp = {("arm_a", "S1"): {"label": "arm_a", "stamp": "S1", "status": "weird"}}
    pending = G.compute_pending([d], disp, alive_fn=lambda arm, s: False)
    assert len(pending) == 1  # fail-safe toward review


# ---- pending_signature: loop-safety key ------------------------------------
def test_signature_stable_and_order_independent():
    a = [{"label": "x", "stamp": "1"}, {"label": "y", "stamp": "2"}]
    b = [{"label": "y", "stamp": "2"}, {"label": "x", "stamp": "1"}]
    assert G.pending_signature(a) == G.pending_signature(b)
    c = [{"label": "x", "stamp": "1"}]
    assert G.pending_signature(a) != G.pending_signature(c)


def test_empty_signature_is_stable():
    assert G.pending_signature([]) == G.pending_signature([])


# ---- latest_dispositions: latest-row-wins ----------------------------------
def test_latest_row_wins():
    rows = [
        {"label": "a", "stamp": "1", "status": "held_entangled"},
        {"label": "a", "stamp": "1", "status": "reviewed_committed"},
    ]
    latest = G.latest_dispositions(rows)
    assert latest[("a", "1")]["status"] == "reviewed_committed"


def test_rows_missing_keys_skipped():
    rows = [{"status": "closed"}, {"label": "a", "status": "closed"},
            {"label": "a", "stamp": "1", "status": "closed"}]
    latest = G.latest_dispositions(rows)
    assert list(latest.keys()) == [("a", "1")]


# ---- constants sanity ------------------------------------------------------
def test_state_partitions():
    assert {"reviewed_committed", "respawned", "closed"} == G.TERMINAL_STATES
    assert "held_entangled" in G.NONTERMINAL_STATES
    assert G.VALID_STATES == G.TERMINAL_STATES | G.NONTERMINAL_STATES
