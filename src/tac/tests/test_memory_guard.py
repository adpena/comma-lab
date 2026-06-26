"""NO-FAKE tests for the unified-memory OOM guard (tools/memory_guard.py).

The #1 correctness property is CONTROL-PLANE SAFETY: the watchdog kill-selector
must NEVER return a control-plane process (claude / codex / node / ssh / shells
/ the guard / its ancestors) as a victim — EVEN when those are the largest-RSS
processes AND (worst case) coincidentally match a training pattern. These tests
exercise that adversarially against synthetic ``ps`` tables.

Custody model: the selector only kills processes under EXPLICIT CUSTODY (the
durable-daemon registry), identity-gated against PID recycling. A process NOT in
custody is never killed, even if it matches a training pattern.

Provenance: the protection core is vendored from molt 3b1e49b18 (the
custody-gated, control-plane-safe HEAD).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import tools/memory_guard.py directly (it lives under tools/, not a package).
_MG_PATH = Path(__file__).resolve().parents[3] / "tools" / "memory_guard.py"
_spec = importlib.util.spec_from_file_location("tac_memory_guard_under_test", _MG_PATH)
mg = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec so dataclass forward-ref resolution (PEP 563 string
# annotations from ``from __future__ import annotations``) can find the module.
sys.modules["tac_memory_guard_under_test"] = mg
_spec.loader.exec_module(mg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sample(pid, ppid, rss_kb, command, pgid=None):
    return mg.ProcessSample(pid=pid, ppid=ppid, rss_kb=rss_kb, command=command, pgid=pgid if pgid is not None else pid)


def _table(*samples):
    return {s.pid: s for s in samples}


def _custody(*recs):
    """recs: tuples of (label, pid, pgid, cmd)."""
    return [mg.CustodyRecord(label=l, pid=p, pgid=g, cmd=c) for (l, p, g, c) in recs]


TRAIN_CMD = ".venv/bin/python experiments/train_witness_realized_through_R_mlx.py --n 16"


# ---------------------------------------------------------------------------
# 1. ps parsing
# ---------------------------------------------------------------------------
def test_parse_ps_table_5col():
    text = "  500   1 500 1048576 python train_witness.py --n 16\n  600 500 500 2048 child\n"
    samples = mg.parse_ps_table(text)
    assert samples[500].pid == 500 and samples[500].ppid == 1 and samples[500].pgid == 500
    assert samples[500].rss_kb == 1048576
    assert "train_witness" in samples[500].command


def test_parse_ps_table_skips_garbage():
    assert mg.parse_ps_table("\n   \nnot a row\n") == {}


# ---------------------------------------------------------------------------
# 2. control-plane classification (vendored)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", [
    "/Applications/Claude.app/Contents/MacOS/Claude",
    "node /Users/x/.nvm/versions/node/v20/lib/node_modules/@anthropic-ai/claude-code/cli.js",
    "codex app-server",
    "/Applications/Codex.app/Contents/MacOS/Codex Helper (Renderer)",
    "codex-macos-sandbox run",
    "/cua_node/bin/node_repl",
])
def test_is_host_control_plane_process_true(cmd):
    assert mg.is_host_control_plane_process(_sample(1234, 1, 1000, cmd)) is True


@pytest.mark.parametrize("cmd", [
    ".venv/bin/python experiments/train_witness_realized_through_R_mlx.py",
    "/usr/bin/python3 train_renderer.py",
    "vm_stat",
])
def test_is_host_control_plane_process_false_for_training(cmd):
    assert mg.is_host_control_plane_process(_sample(1234, 1, 1000, cmd)) is False


def test_launcher_command_running_control_plane_arg_is_protected():
    # `node <path>/claude-code/cli.js` — launcher executing a control-plane arg.
    s = _sample(900, 1, 5_000_000, "node /opt/@anthropic-ai/claude-code/cli.js")
    assert mg.is_host_control_plane_process(s) is True


# ---------------------------------------------------------------------------
# 3. THE MAKE-OR-BREAK: control-plane processes are NEVER selected,
#    even as largest RSS and even when coincidentally matching a training token.
# ---------------------------------------------------------------------------
def test_codex_largest_rss_matching_training_token_is_refused():
    """Worst case (the exact bug class molt 3b1e49b18 fixed): a Codex process is
    the largest-RSS, IS (wrongly) registered in custody, IS detached, AND its
    command coincidentally contains a training token. It MUST still be refused
    because it is a host control-plane app."""
    codex = _sample(200, 1, 90_000_000, "codex app-server --foo train_witness", pgid=200)
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"), codex)
    custody = _custody(("rogue", 200, 200, "codex app-server"))  # adversarially in custody
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=1, self_pgid=1)
    assert victim is None, f"MUST refuse codex; got {victim}"


def test_claude_codex_node_ssh_never_selected_even_as_largest():
    self_pid, self_pgid = 400, 400
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(100, 1, 80_000_000, "/Applications/Claude.app/Contents/MacOS/Claude"),
        _sample(200, 1, 70_000_000, "codex app-server"),
        _sample(300, 100, 40_000_000, "/Applications/Claude.app/.../Code Helper (Renderer)"),
        _sample(700, 1, 30_000_000, "ssh -N tunnel@host"),
        _sample(800, 1, 25_000_000, "tmux new-session"),
        _sample(self_pid, 1, 2000, ".venv/bin/python tools/memory_guard.py --watch", pgid=self_pgid),
    )
    # Even if (adversarially) every control-plane pid were "in custody", refuse.
    custody = _custody(
        ("c1", 100, 100, "claude"), ("c2", 200, 200, "codex"),
        ("c3", 300, 300, "node"), ("c4", 700, 700, "ssh"), ("c5", 800, 800, "tmux"),
    )
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=self_pid, self_pgid=self_pgid)
    assert victim is None


def test_guard_and_ancestors_never_selected():
    # The guard's parent chain (launchd -> shell -> guard) must be protected even
    # if (impossibly) registered in custody and matching the pattern.
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(50, 1, 9_000_000, "-zsh"),
        _sample(400, 50, 9_000_000, ".venv/bin/python tools/memory_guard.py --watch train_witness", pgid=400),
    )
    custody = _custody(("self", 400, 400, "tools/memory_guard.py"), ("anc", 50, 50, "-zsh"))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None


# ---------------------------------------------------------------------------
# 4. POSITIVE: a real custody training arm IS selected.
# ---------------------------------------------------------------------------
def test_selects_legit_custody_training_arm():
    self_pid, self_pgid = 400, 400
    arm = _sample(500, 1, 10_000_000, TRAIN_CMD, pgid=500)
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(100, 1, 80_000_000, "/Applications/Claude.app/Contents/MacOS/Claude"),
        _sample(self_pid, 1, 2000, "tools/memory_guard.py --watch", pgid=self_pgid),
        arm,
    )
    custody = _custody(("witness_n16", 500, 500, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=self_pid, self_pgid=self_pgid)
    assert victim is not None
    assert victim.pid == 500 and victim.pgid == 500 and victim.label == "witness_n16"
    assert victim.rss_gb == pytest.approx(10.0, rel=1e-3)


def test_selects_largest_of_multiple_custody_arms():
    self_pid = 400
    a = _sample(500, 1, 5_000_000, TRAIN_CMD, pgid=500)
    b = _sample(501, 1, 12_000_000, TRAIN_CMD.replace("--n 16", "--n 600"), pgid=501)
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"),
                     _sample(self_pid, 1, 2000, "memory_guard.py --watch", pgid=self_pid), a, b)
    custody = _custody(("small", 500, 500, TRAIN_CMD), ("big", 501, 501, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=self_pid, self_pgid=400)
    assert victim.pid == 501 and victim.label == "big"


# ---------------------------------------------------------------------------
# 5. CUSTODY gate: a training arm NOT in custody is NEVER killed.
# ---------------------------------------------------------------------------
def test_training_arm_not_in_custody_is_refused():
    self_pid = 400
    arm = _sample(500, 1, 50_000_000, TRAIN_CMD, pgid=500)
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"),
                     _sample(self_pid, 1, 2000, "memory_guard.py --watch", pgid=self_pid), arm)
    # custody empty → nothing killable even though arm is huge + matches pattern.
    victim = mg.select_kill_victim(samples, custody_records=[], self_pid=self_pid, self_pgid=400)
    assert victim is None


def test_empty_table_returns_none():
    assert mg.select_kill_victim({}, custody_records=[], self_pid=400, self_pgid=400) is None


# ---------------------------------------------------------------------------
# 6. IDENTITY gate (PID recycling defense).
# ---------------------------------------------------------------------------
def test_identity_gate_rejects_recycled_pid_different_command():
    # custody says pid 500 was our train arm; live pid 500 is now an unrelated tool.
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"),
                     _sample(500, 1, 50_000_000, "/bin/some_unrelated_daemon", pgid=500))
    custody = _custody(("witness", 500, 500, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None


def test_identity_gate_rejects_pgid_mismatch():
    # live pgid differs from recorded pgid → not our daemon.
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"),
                     _sample(500, 1, 50_000_000, TRAIN_CMD, pgid=999))
    custody = _custody(("witness", 500, 500, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None


# ---------------------------------------------------------------------------
# 7. control-plane LINEAGE: a custody arm descended from claude is protected.
# ---------------------------------------------------------------------------
def test_custody_arm_with_control_plane_ancestor_is_refused():
    # arm 550 is a child of claude (100) — protected by lineage even in custody.
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(100, 1, 80_000_000, "/Applications/Claude.app/Contents/MacOS/Claude"),
        _sample(550, 100, 20_000_000, TRAIN_CMD, pgid=550),
        _sample(400, 1, 2000, "memory_guard.py --watch", pgid=400),
    )
    custody = _custody(("witness", 550, 550, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None


# ---------------------------------------------------------------------------
# 8. group-leader gate: a custody arm that shares a process group (pgid != pid)
#    is refused (group-kill could reach a co-tenant / shell job).
# ---------------------------------------------------------------------------
def test_non_group_leader_custody_arm_is_refused():
    samples = _table(_sample(1, 0, 5000, "/sbin/launchd"),
                     _sample(400, 1, 2000, "memory_guard.py --watch", pgid=400),
                     _sample(560, 1, 30_000_000, TRAIN_CMD, pgid=999))  # not its own leader
    custody = _custody(("witness", 560, 999, TRAIN_CMD))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None


# ---------------------------------------------------------------------------
# 9. launch preflight
# ---------------------------------------------------------------------------
def test_check_launch_ok_refuses_when_below_floor():
    d = mg.check_launch_ok(min_free_gb=30.0, projected_gb=25.0, available=40.0)
    assert d.ok is False and d.headroom_after_gb == pytest.approx(15.0)


def test_check_launch_ok_allows_with_headroom():
    d = mg.check_launch_ok(min_free_gb=30.0, projected_gb=25.0, available=100.0)
    assert d.ok is True and d.headroom_after_gb == pytest.approx(75.0)


def test_check_launch_ok_exact_boundary_is_ok():
    d = mg.check_launch_ok(min_free_gb=30.0, projected_gb=0.0, available=30.0)
    assert d.ok is True  # >= floor


def test_check_launch_ok_just_below_boundary_refused():
    d = mg.check_launch_ok(min_free_gb=30.0, projected_gb=0.0, available=29.99)
    assert d.ok is False


# ---------------------------------------------------------------------------
# 10. custody loader
# ---------------------------------------------------------------------------
def test_load_custody_records_filters_non_running(tmp_path):
    reg = tmp_path / "durable_daemons.json"
    reg.write_text(json.dumps([
        {"label": "alive", "pid": 500, "pgid": 500, "cmd": ["python", "train_witness.py"], "status": "running"},
        {"label": "dead", "pid": 600, "pgid": 600, "cmd": "x", "status": "stopped"},
        {"label": "bad", "pid": 0, "pgid": 0, "cmd": "x", "status": "running"},
    ]))
    recs = mg.load_custody_records(reg)
    assert len(recs) == 1 and recs[0].label == "alive" and recs[0].pid == 500
    assert recs[0].cmd == "python train_witness.py"


def test_load_custody_records_missing_file(tmp_path):
    assert mg.load_custody_records(tmp_path / "nope.json") == []


# ---------------------------------------------------------------------------
# 11. denylist direct refusal (belt-and-suspenders)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["ssh host", "tmux a", "-zsh", "/bin/bash", "/sbin/launchd"])
def test_extra_protected_direct_denylist(cmd):
    assert mg._matches_extra_protected(cmd) is True


def test_python_not_in_any_protected_set():
    # critical: a bare training python must NOT be protected by the denylists.
    assert mg._matches_extra_protected(TRAIN_CMD) is False
    assert mg.is_host_control_plane_process(_sample(1, 1, 1, TRAIN_CMD)) is False
