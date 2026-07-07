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
        # pointer is READ from the canonical file (never a module constant):
        # score when the file resolves, None otherwise — both honest states.
        ptr = ds.frontier_pointer()
        assert snap["meta"]["pointer"] == (ptr.get("score") if ptr.get("ok") else None)
        assert "pointer_info" in snap["meta"]
        # conditional-rendering payloads always present (values may be None)
        assert "goal_src" in snap["meta"] and "costate" in snap["meta"]
        assert "pose_blind" in snap["meta"]
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
    assert "NON-PROMOTABLE" in html
    # CLAUDE.md "Frontier scores are pointer-only": the page carries NO baked
    # pointer literal — prose spans (.ptrv) are filled from the pointer file.
    assert "0.19110" not in html
    assert 'class="ptrv"' in html
    assert "/api/state" in html  # polling fallback present
    assert "__BOOT__" not in html  # template fully rendered


# ───────────────────────── pointer / goals / costate (conditional rendering) ─────────────────────────
def test_frontier_pointer_reads_canonical_file_or_fails_open():
    ptr = ds.frontier_pointer()
    assert isinstance(ptr, dict) and "ok" in ptr
    if ptr["ok"]:
        assert isinstance(ptr["score"], float) and ptr["axis"] == "contest-CPU"
    else:
        assert "reason" in ptr


def test_derive_goal_info_conditional_sources():
    rows = [{"epoch": 1, "d_seg": 0.004, "d_pose": 114.0, "blob_bytes": 82859}]
    # explicit override wins, labelled as such
    g = ds._derive_goal_info(rows, False, 0.00092, ds._TARGET_S_T1, ds._ARCHIVE_NORM_BYTES)
    assert g == {"value": 0.00092, "source": "override(env/cli)"}
    # pose-blind arm: pose term dropped, value derived from measured rate
    g = ds._derive_goal_info(rows, True, None, ds._TARGET_S_T1, ds._ARCHIVE_NORM_BYTES)
    assert g["value"] is not None and g["value"] > 0 and "derived" in g["source"]
    # pose held but hugely off-target: value withheld with an explicit reason
    g = ds._derive_goal_info(rows, False, None, ds._TARGET_S_T1, ds._ARCHIVE_NORM_BYTES)
    assert g["value"] is None and "unreachable" in g["source"]
    # nothing measured -> no source at all (rendered "—")
    g = ds._derive_goal_info([], None, None, ds._TARGET_S_T1, ds._ARCHIVE_NORM_BYTES)
    assert g == {"value": None, "source": None}


def test_read_costate_absent_and_present():
    with tempfile.TemporaryDirectory() as d:
        # no shadow file -> None -> the panel is conditionally ABSENT
        assert ds._read_costate(d) is None
        assert ds._read_costate(None) is None
        row = {"epoch": 925,
               "classification": {"classification": "converging"},
               "recommendations": [{"action": "CONTINUE_STAGE",
                                    "predicted_dS": -0.25, "horizon_epochs": 25}],
               "duty_to_measure": [{"lever": "A", "state": "never-fired"},
                                   {"lever": "B", "state": "measured"}]}
        (Path(d) / "costate_shadow.jsonl").write_text(json.dumps(row) + "\n")
        c = ds._read_costate(d)
        assert c["ok"] and c["classification"] == "CONVERGING"
        assert c["rec"]["action"] == "CONTINUE_STAGE"
        assert c["duty_owed"] == 2 and c["duty_never_fired"] == 1


def test_run_identity_conditional_and_provenance():
    # no run dir -> None -> the header row is conditionally ABSENT
    assert ds._derive_run_identity(None, {}, None, None) is None
    with tempfile.TemporaryDirectory() as d:
        # DECLARED: launch.sh identity header wins, verbatim, provenance "declared"
        (Path(d) / "launch.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            "# tac-config-family: sealed_205\n"
            "# tac-run-purpose: A/B arm: eikonal 0.07 vs mod32cap parent\ncd /repo\n")
        r = ds._derive_run_identity(d, {"w-pose": "1.0"}, False, None)
        assert r["purpose"]["provenance"] == "declared"
        assert r["purpose"]["label"] == "A/B arm: eikonal 0.07 vs mod32cap parent"
        assert r["scope"]["label"] == "seg+pose (w_pose=1.0)"
    with tempfile.TemporaryDirectory() as d:
        # DERIVED clean baseline: islands off + capacity-cap name; labelled "derived"
        d2 = Path(d) / "levelset_n600_witness_mod32cap_x"
        d2.mkdir()
        r = ds._derive_run_identity(str(d2), {"w-pose": "0", "mod-dim": "32",
                                              "eikonal-weight": "0"}, True, None)
        assert r["purpose"]["label"] == "clean baseline / control"
        # a guess is never rendered as a declaration
        assert r["purpose"]["provenance"] == "derived"
        assert any("mod32cap" in e for e in r["purpose"]["evidence"])
        assert r["scope"]["label"].startswith("seg-only")
    # DERIVED A/B arm: FOREIGN stage-checkpoint resume; same-dir resume is NOT an A/B arm
    r = ds._derive_run_identity("/x/runB", {"w-pose": "0"}, True,
                                "/x/runA/levelset_ckpt_stageCE_ep299.npz")
    assert r["purpose"]["label"].startswith("A/B arm")
    r = ds._derive_run_identity("/x/runB", {"w-pose": "0"}, True,
                                "/x/runB/levelset_resume_state.npz")
    assert r["purpose"]["label"] == "clean baseline / control"
    # DERIVED frontier candidate: island levers on
    r = ds._derive_run_identity("/x/runC", {"seed-islands": True, "eikonal-weight": "0.07"},
                                False, None)
    assert r["purpose"]["label"] == "frontier candidate"


def test_login_html_is_401_body():
    body = ds._login_html()
    assert "Access key required" in body
