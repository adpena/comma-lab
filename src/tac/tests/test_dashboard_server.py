"""Tests for tools/dashboard_server.py — the ASGI+WS level-set dashboard.

Pure/unit coverage of: the access gate decision (local-open / public-gated /
constant-time key match), and the log-tail -> new-point detection that drives the
WebSocket push. No network, no GPU, no uvicorn process.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(_TOOLS))

ds = pytest.importorskip("dashboard_server")


# ───────────────────────── access gate ─────────────────────────
def test_gate_no_key_configured_allows_everything():
    assert ds.gate_decision({}, None, None, "") == "allow"
    assert ds.gate_decision({"cf-ray": "x"}, None, None, "") == "allow"


def test_gate_local_request_allowed_even_with_key():
    # no Cf-Ray / Cf-Connecting-Ip -> trusted local -> allowed without key
    assert ds.gate_decision({}, None, None, "secret") == "allow"


def test_gate_public_without_key_denied():
    h = {"cf-ray": "abc123", "cf-connecting-ip": "1.2.3.4"}
    assert ds.gate_decision(h, None, None, "secret") == "deny"


def test_gate_public_with_correct_query_key_allowed():
    h = {"cf-ray": "abc"}
    assert ds.gate_decision(h, "secret", None, "secret") == "allow"


def test_gate_public_with_correct_cookie_or_header_allowed():
    h = {"cf-connecting-ip": "9.9.9.9"}
    assert ds.gate_decision(h, None, "secret", "secret") == "allow"
    assert ds.gate_decision({"cf-ray": "z", "x-dash-key": "secret"}, None, None, "secret") == "allow"


def test_gate_public_with_wrong_key_denied():
    h = {"cf-ray": "abc"}
    assert ds.gate_decision(h, "nope", None, "secret") == "deny"


def test_gate_header_case_insensitive():
    h = {"Cf-Ray": "abc"}  # mixed case as servers send it
    assert ds.gate_decision(h, None, None, "secret") == "deny"


def test_gate_strict_local_ws_requires_key_even_without_cf_headers():
    # cloudflared omits Cf-Ray on the WS upgrade -> strict mode must NOT local-bypass.
    assert ds.gate_decision({}, None, None, "secret", strict_local=True) == "deny"
    assert ds.gate_decision({}, "secret", None, "secret", strict_local=True) == "allow"
    assert ds.gate_decision({}, None, "secret", "secret", strict_local=True) == "allow"
    # no key configured -> open even in strict mode (local-only / no-tunnel)
    assert ds.gate_decision({}, None, None, "", strict_local=True) == "allow"


# ───────────────────────── tail -> new-point detection ─────────────────────────
def _verdict(ep, dseg):
    return {"stage": "verdict", "epoch": ep, "d_seg": dseg, "d_pose": 1.0,
            "blob_bytes": 1000, "implied_S": 1.0, "ts": "2026-06-30T00:00:00Z"}


def _write(log, rows):
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_refresh_returns_only_new_points():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run.log"
        _write(log, [_verdict(0, 0.7), _verdict(25, 0.3)])
        cfg = ds.Config(run_dir=d, poll=1.0, auto_latest=False,
                        cadence_state=str(Path(d) / "cad.json"))
        state = ds.LiveState(cfg)
        first = state.refresh()
        assert {p["epoch"] for p in first} == {0, 25}
        assert len(state.trajectory) == 2
        # no change -> no new points
        assert state.refresh() == []
        # append a verdict -> only the new one is returned (the WS push delta)
        _write(log, [_verdict(0, 0.7), _verdict(25, 0.3), _verdict(50, 0.04)])
        delta = state.refresh()
        assert [p["epoch"] for p in delta] == [50]
        assert delta[0]["d_seg"] == 0.04
        assert len(state.trajectory) == 3


def test_refresh_slims_to_traj_keys_only():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run.log"
        row = _verdict(0, 0.5)
        row["axis"] = "[macOS-CPU advisory] NON-PROMOTABLE"
        row["secret_internal"] = "should-not-ship"
        _write(log, [row])
        cfg = ds.Config(run_dir=d, auto_latest=False, cadence_state=str(Path(d) / "cad.json"))
        state = ds.LiveState(cfg)
        pts = state.refresh()
        assert "secret_internal" not in pts[0]
        assert set(pts[0].keys()) == set(ds._TRAJ_KEYS)


def test_snapshot_and_meta_shape():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run.log"
        _write(log, [_verdict(0, 0.7)])
        cfg = ds.Config(run_dir=d, tau=300, l7=600, auto_latest=False, cadence_state=str(Path(d) / "cad.json"))
        state = ds.LiveState(cfg)
        state.refresh()
        snap = state.snapshot()
        assert snap["type"] == "snapshot"
        assert len(snap["trajectory"]) == 1
        assert snap["meta"]["tau"] == 300 and snap["meta"]["l7"] == 600
        assert snap["meta"]["pointer"] == ds.POINTER
        assert "kind" in snap["liveness"]


def test_missing_log_is_graceful():
    with tempfile.TemporaryDirectory() as d:
        cfg = ds.Config(run_dir=d, auto_latest=False, cadence_state=str(Path(d) / "cad.json"))
        state = ds.LiveState(cfg)
        assert state.refresh() == []
        assert state.snapshot()["liveness"]["kind"] == "missing"


def test_page_html_has_ws_and_tabs_and_authority():
    html = ds._page_html(ds.Config())
    assert "new WebSocket" in html
    assert 'data-tab="tri"' in html and 'data-tab="live"' in html
    assert "NON-PROMOTABLE" in html and "0.19110" in html
    assert "/api/state" in html  # polling fallback present
    assert "__BOOT__" not in html  # template fully rendered


def test_login_html_is_401_body():
    body = ds._login_html()
    assert "Access key required" in body
