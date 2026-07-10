"""Unit tests for tools/dashboard_ctl.py — the idempotent ensure-up decision logic.

Covers the three properties the incident (2026-07-10 dashboard-down) demands:
  * idempotent-when-healthy (fresh code -> NO-OP, never disrupts a working server),
  * restart-when-down / duplicate,
  * code-staleness detection (a code edit since server start -> reload; old server -> adopt).

Process / port / HTTP are all mocked — no live server is required (hermetic).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dashboard_ctl as dc  # noqa: E402


# ───────────────────────── is_code_stale ─────────────────────────
def test_code_stale_when_disk_newer_than_server():
    health = {"code_mtime": 1000.0}
    assert dc.is_code_stale(health, disk_mtime=1010.0, grace_s=2.0) is True


def test_code_fresh_when_disk_equals_server():
    health = {"code_mtime": 1000.0}
    assert dc.is_code_stale(health, disk_mtime=1000.0, grace_s=2.0) is False


def test_code_fresh_within_grace_window():
    # a same-second save (disk 1s newer) is absorbed by the 2s grace -> not stale (no flap)
    health = {"code_mtime": 1000.0}
    assert dc.is_code_stale(health, disk_mtime=1001.0, grace_s=2.0) is False


def test_old_server_without_code_mtime_is_stale_for_adoption():
    # a server that predates the code_mtime field must be reloaded once to adopt new code
    assert dc.is_code_stale({"ok": True}, disk_mtime=1234.0) is True


def test_no_health_is_not_stale():
    # down server: staleness is irrelevant (decide_action returns restart on health None)
    assert dc.is_code_stale(None, disk_mtime=9999.0) is False


def test_invalid_code_mtime_is_not_stale():
    assert dc.is_code_stale({"code_mtime": "bogus"}, disk_mtime=9999.0) is False
    assert dc.is_code_stale({"code_mtime": 0.0}, disk_mtime=9999.0) is False


# ───────────────────────── decide_action ─────────────────────────
def test_decide_noop_when_healthy_fresh_single():
    assert dc.decide_action({"ok": True, "code_mtime": 5.0}, code_stale=False, n_server_procs=1) == "noop"


def test_decide_reload_when_healthy_but_stale():
    assert dc.decide_action({"ok": True, "code_mtime": 5.0}, code_stale=True, n_server_procs=1) == "reload"


def test_decide_restart_when_down():
    assert dc.decide_action(None, code_stale=False, n_server_procs=0) == "restart"


def test_decide_restart_when_duplicate_servers():
    # two servers on the port -> collapse to one fresh instance even if healthy+fresh
    assert dc.decide_action({"ok": True, "code_mtime": 5.0}, code_stale=False, n_server_procs=2) == "restart"


def test_decide_down_precedence_over_stale():
    # a down server restarts regardless of the (irrelevant) staleness flag
    assert dc.decide_action(None, code_stale=True, n_server_procs=1) == "restart"


# ───────────────────────── ensure_up orchestration (mocked actions) ─────────────────────────
def _patch(monkeypatch, *, health, procs=1, stale_disk_mtime=0.0, supervisor=False):
    monkeypatch.setattr(dc, "probe_health", lambda port, timeout=4.0: health)
    monkeypatch.setattr(dc, "server_procs", lambda port: [(1, 1, "x")] * procs)
    monkeypatch.setattr(dc, "disk_code_mtime", lambda: stale_disk_mtime)
    monkeypatch.setattr(dc, "supervisor_alive", lambda: supervisor)
    calls = {"restart": 0, "reload": 0}
    monkeypatch.setattr(dc, "do_restart", lambda port, **k: calls.__setitem__("restart", calls["restart"] + 1) or 0)
    monkeypatch.setattr(dc, "do_reload", lambda port, **k: calls.__setitem__("reload", calls["reload"] + 1) or 0)
    return calls


def test_ensure_up_healthy_fresh_is_noop(monkeypatch):
    calls = _patch(monkeypatch, health={"ok": True, "code_mtime": 100.0}, procs=1, stale_disk_mtime=100.0)
    rc = dc.ensure_up(8790, quiet=True)
    assert rc == 0 and calls == {"restart": 0, "reload": 0}


def test_ensure_up_stale_code_triggers_reload(monkeypatch):
    calls = _patch(monkeypatch, health={"ok": True, "code_mtime": 100.0}, procs=1, stale_disk_mtime=200.0)
    dc.ensure_up(8790, quiet=True)
    assert calls["reload"] == 1 and calls["restart"] == 0


def test_ensure_up_down_triggers_restart(monkeypatch):
    calls = _patch(monkeypatch, health=None, procs=0, supervisor=False)
    dc.ensure_up(8790, quiet=True)
    assert calls["restart"] == 1 and calls["reload"] == 0


def test_ensure_up_old_server_adopts_via_reload(monkeypatch):
    # old server (no code_mtime) is healthy -> reload once to adopt new code
    calls = _patch(monkeypatch, health={"ok": True}, procs=1, stale_disk_mtime=100.0)
    dc.ensure_up(8790, quiet=True)
    assert calls["reload"] == 1 and calls["restart"] == 0


def test_ensure_up_defers_to_supervisor_when_down_but_supervisor_heals(monkeypatch):
    # down + supervisor alive: the grace-wait sees it come back healthy -> no bare restart
    states = [None, {"ok": True, "code_mtime": 100.0}]  # first probe down, second (post-grace) up
    monkeypatch.setattr(dc, "probe_health", lambda port, timeout=4.0: states.pop(0) if states else {"ok": True, "code_mtime": 100.0})
    monkeypatch.setattr(dc, "_wait_healthy", lambda port, t: True)
    monkeypatch.setattr(dc, "server_procs", lambda port: [(1, 1, "x")])
    monkeypatch.setattr(dc, "disk_code_mtime", lambda: 100.0)
    monkeypatch.setattr(dc, "supervisor_alive", lambda: True)
    calls = {"restart": 0, "reload": 0}
    monkeypatch.setattr(dc, "do_restart", lambda port, **k: calls.__setitem__("restart", calls["restart"] + 1) or 0)
    monkeypatch.setattr(dc, "do_reload", lambda port, **k: calls.__setitem__("reload", calls["reload"] + 1) or 0)
    dc.ensure_up(8790, quiet=True, supervisor_grace_s=0.1)
    assert calls["restart"] == 0  # supervisor healed it; we did not fight it


def test_ensure_up_down_supervisor_gone_restarts(monkeypatch):
    calls = _patch(monkeypatch, health=None, procs=0, supervisor=False)
    dc.ensure_up(8790, quiet=True, supervisor_grace_s=0.1)
    assert calls["restart"] == 1


# ───────────────────────── status smoke (mocked) ─────────────────────────
def test_cmd_status_down(monkeypatch, capsys):
    monkeypatch.setattr(dc, "probe_health", lambda port, timeout=4.0: None)
    monkeypatch.setattr(dc, "server_procs", lambda port: [])
    monkeypatch.setattr(dc, "supervisor_alive", lambda: False)
    monkeypatch.setattr(dc, "_ps_rows", lambda: [])
    rc = dc.cmd_status(8790)
    out = capsys.readouterr().out
    assert rc == 0 and "DOWN" in out and "ensure-up" in out


def test_cmd_status_up(monkeypatch, capsys):
    health = {"ok": True, "watched": "daemon.log", "watched_dir": "experiments/results/levelset_x",
              "code_mtime": 100.0, "last_update_age_s": 42, "n_points": 7, "last_epoch": 50,
              "next_epoch": 75, "started_utc": "2026-07-10T15:00:00Z"}
    monkeypatch.setattr(dc, "probe_health", lambda port, timeout=4.0: health)
    monkeypatch.setattr(dc, "server_procs", lambda port: [(1, 1, "x")])
    monkeypatch.setattr(dc, "supervisor_alive", lambda: False)
    monkeypatch.setattr(dc, "_ps_rows", lambda: [])
    monkeypatch.setattr(dc, "disk_code_mtime", lambda: 100.0)
    rc = dc.cmd_status(8790)
    out = capsys.readouterr().out
    assert rc == 0 and "UP" in out and "levelset_x" in out and "code fresh" in out


def test_fmt_age():
    assert dc._fmt_age(None) == "?"
    assert dc._fmt_age(30) == "30s"
    assert dc._fmt_age(120).endswith("m")
    assert dc._fmt_age(7200).endswith("h")
