"""Tests for tools/dashboard_supervisor.py — singleton + auto-teardown lifecycle.

Pure coverage of: proc classification (never the training daemon), kill-selection
with process-group self-exclusion, the inactivity-teardown decision, the quick-URL
parser, and the access-key persistence helper. No real process kills, no network.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(_TOOLS))

sup = pytest.importorskip("dashboard_supervisor")


# ───────────────────────── classify ─────────────────────────
def test_classify_never_kills_training_daemon():
    cmd = ".venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py --epochs 600"
    assert sup.classify_proc(cmd, 8790) is None


def test_classify_dashboard_kinds():
    assert sup.classify_proc("python tools/dashboard_server.py --port 8790", 8790) == "server"
    assert sup.classify_proc("python tools/render_levelset_dashboard.py --watch", 8790) == "render"
    assert sup.classify_proc("python tools/dashboard_supervisor.py --run", 8790) == "supervisor"
    assert sup.classify_proc("cloudflared tunnel --url http://127.0.0.1:8790", 8790) == "tunnel"
    assert sup.classify_proc("cloudflared tunnel run", 8790) == "tunnel"
    assert sup.classify_proc("python -m http.server 8790 --directory x", 8790) == "httpd"
    # stale dash httpd on a different port still matched via the dir signature
    assert sup.classify_proc("python -m http.server 8789 --directory .omx/tmp/dash_levelset_deploy", 8790) == "httpd"
    assert sup.classify_proc("python some_other_thing.py", 8790) is None


# ───────────────────────── select_kill_pids ─────────────────────────
def test_select_kill_excludes_own_pgid_and_training():
    rows = [
        (100, 100, "python tools/dashboard_server.py --port 8790"),   # server, kill
        (200, 200, "cloudflared tunnel run"),                          # tunnel, kill
        (300, 300, ".venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py"),  # training, NEVER
        (400, 999, "python tools/dashboard_supervisor.py --run"),      # supervisor in OWN group -> excluded
        (401, 999, "python tools/dashboard_supervisor.py --run"),      # same own group -> excluded
    ]
    got = sup.select_kill_pids(rows, 8790, exclude_pgids={999},
                               kinds={"server", "render", "httpd", "tunnel", "supervisor"})
    pids = {p for p, _, _ in got}
    assert pids == {100, 200}  # never 300 (training), never 400/401 (own group)


def test_select_kill_kind_filter():
    rows = [
        (100, 100, "python tools/dashboard_server.py --port 8790"),
        (200, 200, "cloudflared tunnel run"),
    ]
    got = sup.select_kill_pids(rows, 8790, exclude_pgids=set(), kinds={"server"})
    assert {p for p, _, _ in got} == {100}  # tunnel excluded by kind filter


# ───────────────────────── inactivity teardown ─────────────────────────
def test_teardown_when_training_gone_and_logs_stale():
    assert sup.should_teardown_inactive(newest_age_s=3600, training_alive=False, threshold_s=1800) is True


def test_no_teardown_while_training_alive():
    # even very stale logs do not tear down while training is alive
    assert sup.should_teardown_inactive(newest_age_s=99999, training_alive=True, threshold_s=1800) is False


def test_no_teardown_when_logs_fresh():
    assert sup.should_teardown_inactive(newest_age_s=60, training_alive=False, threshold_s=1800) is False


def test_teardown_when_no_logs_and_training_gone():
    assert sup.should_teardown_inactive(newest_age_s=None, training_alive=False, threshold_s=1800) is True


def test_no_teardown_no_logs_but_training_alive():
    assert sup.should_teardown_inactive(newest_age_s=None, training_alive=True, threshold_s=1800) is False


# ───────────────────────── training liveness (signature-robust) ─────────────────────────
def test_training_alive_via_pid():
    assert sup.training_alive(os.getpid(), "zzz_nonexistent_sig_123") is True


def test_training_alive_false_when_pid_dead_and_sig_absent():
    assert sup.training_alive(0, "zzz_nonexistent_sig_123_qwerty") is False


# ───────────────────────── quick-url parser ─────────────────────────
def test_parse_quick_url():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "tunnel.log"
        log.write_text("2026-... INF +-----+\n| https://abc-def-ghi.trycloudflare.com |\n+-----+\n")
        assert sup._parse_quick_url(str(log)) == "https://abc-def-ghi.trycloudflare.com"


def test_parse_quick_url_absent():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "tunnel.log"
        log.write_text("starting connection...\n")
        assert sup._parse_quick_url(str(log)) is None


# ───────────────────────── newest_log_age ─────────────────────────
def test_newest_log_age_none_when_empty():
    with tempfile.TemporaryDirectory() as d:
        assert sup.newest_log_age_s(d) is None


def test_newest_log_age_positive():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.log").write_text("x")
        age = sup.newest_log_age_s(d)
        assert age is not None and age >= 0


# ───────────────────────── access key persistence ─────────────────────────
def test_access_key_stable_across_calls():
    with tempfile.TemporaryDirectory() as d:
        k1 = sup.load_or_make_access_key(d)
        k2 = sup.load_or_make_access_key(d)
        assert k1 == k2 and len(k1) > 16
        # persisted file is chmod 600
        mode = os.stat(os.path.join(d, ".access_key")).st_mode & 0o777
        assert mode == 0o600
