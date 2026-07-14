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
    assert samples[500].start_identity is None  # legacy fixtures cannot authorize history relaxation


def test_parse_ps_table_lstart_binds_kernel_process_identity():
    text = (
        "500 1 500 1048576 Tue Jul 14 11:23:45 2026 "
        "python train_witness.py --n 16\n"
    )
    sample = mg.parse_ps_table(text)[500]
    assert sample.start_identity == "Tue Jul 14 11:23:45 2026"
    assert sample.command == "python train_witness.py --n 16"


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


# ---------------------------------------------------------------------------
# 12. HIGH review finding 1 — the REAL installed codex cask binary must be
#     classified control-plane (basename codex-aarch64-apple-darwin + path
#     /Caskroom/codex/ both previously escaped).
# ---------------------------------------------------------------------------
CODEX_CASK = "/opt/homebrew/Caskroom/codex/0.139.0/codex-aarch64-apple-darwin"


def test_real_codex_cask_binary_classified_control_plane():
    assert mg.is_host_control_plane_process(_sample(200, 1, 90_000_000, CODEX_CASK + " app-server")) is True


def test_real_codex_cask_exec_name_matched():
    # basename alone (no helpful path) must still classify.
    assert mg._is_control_plane_exec_name("codex-aarch64-apple-darwin") is True
    assert mg._is_control_plane_exec_name("codex-x86_64-apple-darwin") is True
    assert mg._is_control_plane_exec_name("claude-code") is True
    # and a training python basename must NOT.
    assert mg._is_control_plane_exec_name("python3") is False
    assert mg._is_control_plane_exec_name("python") is False


def test_node_cli_js_under_codex_or_claude_path_is_control_plane():
    # the real entry is cli.js (not claude.js/codex.js) — match the PATH.
    assert mg.is_host_control_plane_process(
        _sample(300, 1, 5_000_000, "node /opt/tools/claude-code/cli.js")
    ) is True
    assert mg.is_host_control_plane_process(
        _sample(301, 1, 5_000_000, "node /opt/homebrew/Caskroom/codex/0.139.0/cli.js")
    ) is True


def test_MAKE_OR_BREAK_real_codex_cask_custody_poisoned_with_training_token_refused():
    """THE make-or-break (HIGH review finding 1d): the EXACT resolved codex cask
    binary, ADVERSARIALLY in custody, own group leader, largest RSS, AND a
    training token (train_witness_realized_through_R_mlx.py) in its argv. The
    selector MUST return victim=None — the operator's #1 rule ("NEVER kill
    codex") holds even under custody poisoning + a coincidental training token."""
    cmd = CODEX_CASK + " --foo experiments/train_witness_realized_through_R_mlx.py"
    codex = _sample(200, 1, 99_000_000, cmd, pgid=200)
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        codex,
        _sample(400, 1, 2000, ".venv/bin/python tools/memory_guard.py --watch", pgid=400),
    )
    custody = _custody(("rogue_codex", 200, 200, cmd))  # adversarial custody poison
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is None, f"MUST refuse the real codex cask binary; got {victim}"


# ---------------------------------------------------------------------------
# 13. M3/HIGH-2 — SHED debounce + near-boundary ALERT, on the CONSERVATIVE metric.
# ---------------------------------------------------------------------------
def test_shed_decision_ok_above_floor():
    assert mg.shed_decision(consecutive_subfloor=9, avail=40.0, min_free_gb=30.0) == "ok"


def test_shed_decision_alert_when_debounce_not_met():
    assert mg.shed_decision(consecutive_subfloor=1, avail=25.0, min_free_gb=30.0,
                            shed_consecutive=3, shed_margin_gb=2.0) == "alert"


def test_shed_decision_alert_near_boundary_even_if_sustained():
    # sustained sub-floor but only WITHIN the margin → ALERT, never kill.
    assert mg.shed_decision(consecutive_subfloor=5, avail=29.0, min_free_gb=30.0,
                            shed_consecutive=3, shed_margin_gb=2.0) == "alert"


def test_shed_decision_sheds_when_sustained_and_clearly_below():
    assert mg.shed_decision(consecutive_subfloor=3, avail=25.0, min_free_gb=30.0,
                            shed_consecutive=3, shed_margin_gb=2.0) == "shed"


# ---------------------------------------------------------------------------
# 13b. HIGH-2 — the shed metric must NEVER overcount (was: generous overcount
#      that could read 'ok' under pressure and skip a needed shed → OOM).
# ---------------------------------------------------------------------------
def test_shed_metric_never_overcounts_live():
    # conservative shed metric must be <= the broadest conceivable estimate.
    assert mg.available_gb() <= mg._broadest_available_estimate_gb() + 1e-6


def test_available_excludes_file_backed_overcount():
    # PROOF the OOM-dangerous overcount is gone: huge ACTIVE file-backed cache +
    # tiny free → the shed metric stays tiny (would shed), NOT inflated by
    # file_backed. (file_backed=100000 pages would have read ~1.6GB+ on the old
    # generous metric; the conservative metric ignores it entirely.)
    fields = {"page_size": 16384, "free": 100, "inactive": 100,
              "speculative": 0, "purgeable": 10, "file_backed": 100_000}
    shed_metric = mg._available_gb_from_fields(fields)
    assert shed_metric == pytest.approx((100 + 100) * 16384 / 1e9)  # ONLY free+inactive
    assert shed_metric < 0.01  # well below any sane floor → would correctly shed
    # invariant: shed metric never exceeds the broadest estimate.
    assert shed_metric <= mg._broadest_available_estimate_gb_from_fields(fields)


def test_parse_vm_stat_reads_speculative_purgeable_file_backed():
    sample_out = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                   100.\n"
        "Pages inactive:               200.\n"
        "Pages speculative:            40.\n"
        "Pages purgeable:              50.\n"
        "File-backed pages:            300.\n"
    )
    f = mg._parse_vm_stat(sample_out)
    assert f["page_size"] == 16384
    assert f["free"] == 100 and f["inactive"] == 200
    assert f["speculative"] == 40 and f["purgeable"] == 50 and f["file_backed"] == 300


# ---------------------------------------------------------------------------
# 13c. HIGH-1 corollary — RSS attribution = the whole process group/descendants
#      (was: the ~10MB safe_run wrapper).
# ---------------------------------------------------------------------------
def test_group_rss_sums_descendants():
    samples = _table(
        _sample(500, 1, 10_000, "python safe_run.py -- python train.py", pgid=500),  # ~10MB wrapper
        _sample(501, 500, 2_000_000, "python train.py", pgid=501),                    # ~2GB inner child
        _sample(502, 501, 500_000, "python train.py worker", pgid=501),               # 0.5GB grandchild
    )
    assert mg.group_rss_gb(samples, 500) == pytest.approx((10_000 + 2_000_000 + 500_000) / 1e6, rel=1e-6)


def test_selector_attributes_group_rss_for_wrapped_arm():
    # A safe_run wrapper (custody) whose inner trainer is the real footprint:
    # victim.rss_gb must be the GROUP rss (~2.5GB), not the ~10MB wrapper.
    wrapper_cmd = "python tools/safe_run.py --rss-mb 30000 --timeout 60 -- python experiments/train_witness_realized_through_R_mlx.py"
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 10_000, wrapper_cmd, pgid=700),                                 # wrapper (custody)
        _sample(701, 700, 2_500_000, "python experiments/train_witness_realized_through_R_mlx.py", pgid=701),  # inner (own session)
    )
    custody = _custody(("witness_n600", 700, 700, wrapper_cmd))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is not None, "wrapped training arm MUST be sheddable (safe_run not denylisted)"
    assert victim.pid == 700 and victim.pgid == 700  # kills the wrapper pgid → cascades to inner
    assert victim.rss_gb == pytest.approx((10_000 + 2_500_000) / 1e6, rel=1e-6)  # GROUP rss, not wrapper


def test_safe_run_no_longer_in_extra_protected_denylist():
    # removing safe_run.py from the denylist is what makes wrapped arms sheddable.
    assert mg._matches_extra_protected("python tools/safe_run.py -- python train_witness.py") is False
    # but memory_guard.py + spawn_durable_daemon.py stay protected.
    assert mg._matches_extra_protected("python tools/memory_guard.py --watch") is True


# ---------------------------------------------------------------------------
# 14. SIGTERM → grace → SIGKILL escalation cascade (task #409).
#     The escalation is the last-resort shed of a VETTED custody training arm.
#     Its #1 invariant is UNCHANGED control-plane safety: the exempt-set re-check
#     (pgid_is_killable_custody_arm) must gate the SIGKILL exactly as the selector
#     gated the SIGTERM — a pgid recycled onto the control plane during the grace
#     window is the molt CP-kill class and must NEVER be SIGKILLed.
# ---------------------------------------------------------------------------
class _FakeSignals:
    """Records os.killpg / os.kill calls and models per-pgid liveness.

    ``sig == 0`` is the existence probe (POSIX): raises ProcessLookupError when
    the group is gone, PermissionError when in ``perm_pgids``. A real SIGKILL (or
    a pgid in ``term_kills``) removes the group from ``alive``.
    """

    def __init__(self, *, alive_pgids=(), term_kills=(), missing_pgids=(),
                 perm_pgids=(), missing_pids=()):
        self.calls: list[tuple[str, int, int]] = []
        self.alive = set(alive_pgids)
        self.term_kills = set(term_kills)
        self.missing_pgids = set(missing_pgids)
        self.perm_pgids = set(perm_pgids)
        self.missing_pids = set(missing_pids)

    def killpg(self, pgid, sig):
        self.calls.append(("killpg", pgid, sig))
        if sig == 0:  # existence probe
            if pgid in self.perm_pgids:
                raise PermissionError
            if pgid in self.alive:
                return None
            raise ProcessLookupError
        if pgid in self.perm_pgids:
            raise PermissionError
        if pgid in self.missing_pgids or pgid not in self.alive:
            raise ProcessLookupError
        import signal as _sig
        if sig == _sig.SIGKILL or pgid in self.term_kills:
            self.alive.discard(pgid)
        return None

    def kill(self, pid, sig):
        self.calls.append(("kill", pid, sig))
        if pid in self.missing_pids:
            raise ProcessLookupError
        return None

    def signals_to(self, target):
        import signal as _sig
        names = {_sig.SIGTERM: "TERM", _sig.SIGKILL: "KILL", 0: "PROBE"}
        return [names.get(s, s) for (_k, t, s) in self.calls if t == target]


class _Clock:
    """Deterministic monotonic clock; ``sleep`` advances it (no real waiting)."""

    def __init__(self, start=0.0):
        self.t = start

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


def _install(monkeypatch, fake, clock=None):
    monkeypatch.setattr(mg.os, "killpg", fake.killpg)
    monkeypatch.setattr(mg.os, "kill", fake.kill)
    if clock is not None:
        monkeypatch.setattr(mg, "time", clock)


import signal  # noqa: E402  (used by the escalation tests below)


# --- _signal_pgrp ----------------------------------------------------------
def test_signal_pgrp_delivers_to_group(monkeypatch):
    fake = _FakeSignals(alive_pgids={500})
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(500, 500, signal.SIGTERM) is True
    assert fake.calls == [("killpg", 500, signal.SIGTERM)]  # no pid fallback


def test_signal_pgrp_falls_back_to_pid_when_group_gone(monkeypatch):
    fake = _FakeSignals(alive_pgids=set(), missing_pgids={500})
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(500, 501, signal.SIGTERM) is True
    assert ("kill", 501, signal.SIGTERM) in fake.calls


def test_signal_pgrp_falls_back_on_permission_error(monkeypatch):
    fake = _FakeSignals(alive_pgids={500}, perm_pgids={500})
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(500, 501, signal.SIGTERM) is True
    assert ("kill", 501, signal.SIGTERM) in fake.calls


def test_signal_pgrp_noop_when_both_gone(monkeypatch):
    fake = _FakeSignals(missing_pgids={500}, missing_pids={501})
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(500, 501, signal.SIGTERM) is False


def test_signal_pgrp_nonpositive_pgid_never_broadcasts(monkeypatch):
    # pgid<=0 must NOT call killpg (killpg(0,...) broadcasts to the caller group!).
    fake = _FakeSignals(missing_pids=set())
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(0, 777, signal.SIGKILL) is True
    assert all(kind != "killpg" for (kind, _t, _s) in fake.calls)
    assert ("kill", 777, signal.SIGKILL) in fake.calls


def test_signal_pgrp_no_pid_fallback_when_disabled(monkeypatch):
    # SIGKILL escalation is group-only: a gone group must NOT fall back to
    # os.kill(pid) (the pid may be RECYCLED onto a control-plane process).
    fake = _FakeSignals(missing_pgids={500})  # group already gone (killpg ESRCH)
    _install(monkeypatch, fake)
    assert mg._signal_pgrp(500, 500, signal.SIGKILL, pid_fallback=False) is False
    assert all(kind != "kill" for (kind, _t, _s) in fake.calls), "no os.kill(pid) fallback"


# --- _pgrp_alive -----------------------------------------------------------
def test_pgrp_alive_true(monkeypatch):
    fake = _FakeSignals(alive_pgids={500})
    _install(monkeypatch, fake)
    assert mg._pgrp_alive(500) is True
    assert fake.calls == [("killpg", 500, 0)]  # probes with signal 0


def test_pgrp_alive_false_when_gone(monkeypatch):
    fake = _FakeSignals(alive_pgids=set())
    _install(monkeypatch, fake)
    assert mg._pgrp_alive(500) is False


def test_pgrp_alive_permission_means_alive(monkeypatch):
    fake = _FakeSignals(perm_pgids={500})
    _install(monkeypatch, fake)
    assert mg._pgrp_alive(500) is True


def test_pgrp_alive_nonpositive_never_probes(monkeypatch):
    fake = _FakeSignals()
    _install(monkeypatch, fake)
    assert mg._pgrp_alive(0) is False
    assert fake.calls == []  # never broadcast-probe pgid 0


# --- _kill_pgrp escalation cascade -----------------------------------------
def test_kill_pgrp_sigterm_first_then_sigkill_when_survives(monkeypatch):
    # (a) SIGTERM tried first; (b) SIGKILL only after grace + still alive.
    fake = _FakeSignals(alive_pgids={500})  # SIGTERM does NOT kill it
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=lambda pg, pi: True)
    seq = fake.signals_to(500)
    assert seq[0] == "TERM", "SIGTERM must be sent FIRST"
    assert "KILL" in seq, "must escalate to SIGKILL after grace"
    assert seq.index("TERM") < seq.index("KILL")
    assert clock.t >= 1.0, "SIGKILL only after the full grace window elapsed"


def test_kill_pgrp_no_sigkill_when_exits_on_sigterm(monkeypatch):
    # Clean path: arm dies on SIGTERM during grace → NO SIGKILL.
    fake = _FakeSignals(alive_pgids={500}, term_kills={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    called = {"recheck": 0}

    def _recheck(pg, pi):
        called["recheck"] += 1
        return True

    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=_recheck)
    assert "KILL" not in fake.signals_to(500)
    assert called["recheck"] == 0, "re-check only runs on the escalation path"


def test_kill_pgrp_no_sigkill_when_recheck_fails(monkeypatch):
    # THE molt CP-kill defense: arm survives grace but the exempt-set re-check
    # says the pgid is no longer a safe custody arm (recycled / now protected) →
    # SIGKILL is ABORTED. SIGTERM was the only signal.
    fake = _FakeSignals(alive_pgids={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=lambda pg, pi: False)
    seq = fake.signals_to(500)
    assert seq[0] == "TERM"
    assert "KILL" not in seq  # SIGKILL aborted by the exempt-set re-check
    assert set(seq) <= {"TERM", "PROBE"}  # only SIGTERM + grace probes, never SIGKILL


def test_kill_pgrp_no_sigkill_without_recheck(monkeypatch):
    # Fail-safe default: no safety re-check supplied → SIGTERM-only (no SIGKILL).
    fake = _FakeSignals(alive_pgids={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=None)
    assert "KILL" not in fake.signals_to(500)
    assert fake.signals_to(500)[0] == "TERM"


def test_kill_pgrp_already_dead_is_noop(monkeypatch):
    # (e) already-dead pgrp → SIGTERM no-op → returns, no SIGKILL, no ESRCH crash.
    fake = _FakeSignals(missing_pgids={500}, missing_pids={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=lambda pg, pi: True)
    assert all(s != signal.SIGKILL for (_k, _t, s) in fake.calls)


def test_kill_pgrp_no_sigkill_when_dies_at_deadline(monkeypatch):
    # Grace-window edge: group survives the poll loop but is gone at the final
    # post-grace liveness check → NO SIGKILL, even though still_killable is True.
    fake = _FakeSignals(alive_pgids={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    # Alive throughout the grace poll loop (clock.t < grace), gone exactly at the
    # post-grace liveness check (clock.t == grace after the loop advances it).
    monkeypatch.setattr(mg, "_pgrp_alive", lambda pgid: clock.t < 1.0)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=lambda pg, pi: True)
    assert "KILL" not in fake.signals_to(500)


def test_kill_pgrp_no_escalation_when_sigterm_noop(monkeypatch):
    # SIGTERM was a pure no-op (nothing to signal) → never enters grace/escalation.
    fake = _FakeSignals(missing_pgids={500}, missing_pids={500})
    clock = _Clock()
    _install(monkeypatch, fake, clock)
    mg._kill_pgrp(500, 500, grace_sec=1.0, poll_sec=0.2, still_killable=lambda pg, pi: True)
    # only the single SIGTERM attempt (killpg TERM + pid TERM fallback); no probes.
    assert all(s != 0 for (_k, _t, s) in fake.calls), "no grace probes after a no-op SIGTERM"


# --- pgid_is_killable_custody_arm (the SIGKILL exempt-set re-check) ---------
def test_recheck_true_for_live_custody_arm():
    cmd = TRAIN_CMD
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 2_500_000, cmd, pgid=700),
    )
    custody = _custody(("witness", 700, 700, cmd))
    assert mg.pgid_is_killable_custody_arm(
        700, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is True


def test_recheck_false_when_pgid_recycled_to_control_plane():
    # THE molt class: the training pid/pgid was recycled onto a codex process
    # during the grace window. The re-check MUST reject it (→ no SIGKILL).
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 90_000_000, "codex app-server", pgid=700),  # pid 700 recycled → codex
    )
    custody = _custody(("witness", 700, 700, TRAIN_CMD))
    assert mg.pgid_is_killable_custody_arm(
        700, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is False


def test_recheck_false_when_control_plane_shares_pgid():
    # A different control-plane pid now lives in pgid 700 (arm exited); the custody
    # pid 700 is gone. No custody arm remains in the group → not killable.
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(900, 1, 90_000_000, "codex app-server", pgid=700),
    )
    custody = _custody(("witness", 700, 700, TRAIN_CMD))
    assert mg.pgid_is_killable_custody_arm(
        700, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is False


def test_recheck_false_when_arm_exited():
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
    )  # pid 700 gone
    custody = _custody(("witness", 700, 700, TRAIN_CMD))
    assert mg.pgid_is_killable_custody_arm(
        700, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is False


def test_recheck_false_for_nonpositive_pgid():
    assert mg.pgid_is_killable_custody_arm(0, samples={}, custody_records=[]) is False
    assert mg.pgid_is_killable_custody_arm(-1, samples={}, custody_records=[]) is False


def test_recheck_false_when_samples_empty():
    # Cannot verify the live table → fail safe (never SIGKILL blind).
    custody = _custody(("witness", 700, 700, TRAIN_CMD))
    assert mg.pgid_is_killable_custody_arm(
        700, samples={}, custody_records=custody, self_pid=400, self_pgid=400
    ) is False


def test_recheck_false_when_not_in_custody():
    # A training arm in pgid 700 that is NOT under custody → not killable.
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 2_500_000, TRAIN_CMD, pgid=700),
    )
    assert mg.pgid_is_killable_custody_arm(
        700, samples=samples, custody_records=[], self_pid=400, self_pgid=400
    ) is False


def test_recheck_wrong_pgid_returns_false():
    # The custody arm lives in pgid 700; asking about a different pgid → False.
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 2_500_000, TRAIN_CMD, pgid=700),
    )
    custody = _custody(("witness", 700, 700, TRAIN_CMD))
    assert mg.pgid_is_killable_custody_arm(
        999, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is False


def test_recheck_samples_fresh_by_default(monkeypatch):
    # samples=None → the re-check samples the LIVE table itself (recycling defense
    # relies on a FRESH read, not the stale sample from the shed decision).
    cmd = TRAIN_CMD
    live = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 2_500_000, cmd, pgid=700),
    )
    monkeypatch.setattr(mg, "sample_processes", lambda: live)
    monkeypatch.setattr(mg, "load_custody_records", lambda: _custody(("witness", 700, 700, cmd)))
    assert mg.pgid_is_killable_custody_arm(700, self_pid=400, self_pgid=400) is True


# --- refactor regression: shared gate parity with the selector -------------
def test_shared_gate_parity_selector_and_recheck():
    # A clean custody arm: selector picks it AND the re-check confirms it.
    cmd = TRAIN_CMD
    samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 2_500_000, cmd, pgid=700),
    )
    custody = _custody(("witness", 700, 700, cmd))
    victim = mg.select_kill_victim(samples, custody_records=custody, self_pid=400, self_pgid=400)
    assert victim is not None and victim.pgid == 700
    assert mg.pgid_is_killable_custody_arm(
        victim.pgid, samples=samples, custody_records=custody, self_pid=400, self_pgid=400
    ) is True
    # A control-plane custody arm: selector refuses (None) AND the re-check rejects.
    cp_samples = _table(
        _sample(1, 0, 5000, "/sbin/launchd"),
        _sample(400, 1, 2000, "python tools/memory_guard.py --watch", pgid=400),
        _sample(700, 1, 90_000_000, "codex app-server", pgid=700),
    )
    cp_custody = _custody(("rogue", 700, 700, "codex app-server"))
    assert mg.select_kill_victim(cp_samples, custody_records=cp_custody, self_pid=400, self_pgid=400) is None
    assert mg.pgid_is_killable_custody_arm(
        700, samples=cp_samples, custody_records=cp_custody, self_pid=400, self_pgid=400
    ) is False


def test_watch_loop_default_grace_derived_from_interval(monkeypatch):
    # DERIVED, not magic: default grace = max(GRACE_MIN, interval). Assert via a
    # one-shot watch loop that never sheds (avail high) — it must not raise and the
    # derivation constant is exposed.
    assert mg.DEFAULT_SIGKILL_GRACE_MIN_SEC == 3.0
    monkeypatch.setattr(mg, "available_gb", lambda: 999.0)
    monkeypatch.setattr(mg.time, "sleep", lambda s: None)
    rc = mg.watch_loop(
        min_free_gb=30.0, kill_pattern=mg.DEFAULT_TRAINING_KILL_PATTERN,
        interval=7.0, warn_margin=8.0, max_iterations=1,
    )
    assert rc == 0
    assert mg._matches_extra_protected("python tools/spawn_durable_daemon.py --status") is True
