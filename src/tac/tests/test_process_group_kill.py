# SPDX-License-Identifier: MIT
"""A timeout must reach the GRANDCHILD, not just the shell.

POSITIVE CONTROL FIRST (`test_plain_subprocess_run_leaves_the_grandchild_alive`): the
hazard is reproduced with stock `subprocess.run(..., timeout=...)` before anything is
asserted about the cure, so the cure's assertions are not measuring an absent hazard.
That control IS the ddm_cpu1 shape in miniature — `bash` under a wall-clock cap with the
real worker underneath — the shape that let a decoder run 4,369.6 s past a 1,800 s
timeout and write a full report for a run the harness had recorded as failed.

Every test spawns a REAL process tree and reads liveness from the OS. Nothing here
mocks the kill.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tac.process_group_kill import (
    GroupKillReceipt,
    ProcessGroupTimeout,
    group_alive,
    kill_process_group,
    run_in_process_group,
)

# bash spawns a background grandchild that outlives it, records the grandchild's pid,
# then blocks. Killing only `bash` leaves the grandchild running — that is the defect.
_TREE = (
    'sleep 120 & echo $! > "{pidfile}"; '
    'while [ ! -s "{pidfile}" ]; do sleep 0.02; done; '
    "sleep 120"
)


def _script(pidfile: Path) -> list[str]:
    return ["bash", "-c", _TREE.format(pidfile=pidfile)]


def _await_pidfile(pidfile: Path, *, timeout_s: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pidfile.exists():
            raw = pidfile.read_text().strip()
            if raw.isdigit():
                return int(raw)
        time.sleep(0.02)
    raise AssertionError("the grandchild never recorded its pid — the fixture is broken")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _await_death(pid: int, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# --------------------------------------------------------------------------------------
# POSITIVE CONTROL — the hazard is real.
# --------------------------------------------------------------------------------------

def test_plain_subprocess_run_leaves_the_grandchild_alive(tmp_path: Path) -> None:
    """Stock subprocess.run kills the shell and orphans the worker. The ddm_cpu1 shape."""

    pidfile = tmp_path / "grandchild.pid"
    grandchild = -1
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(_script(pidfile), timeout=1.5, capture_output=True, text=True)
        grandchild = _await_pidfile(pidfile)
        # The kill reached bash. It did not reach this.
        assert _alive(grandchild), (
            "the control did not reproduce the hazard — without it, the cure's "
            "assertions below would be vacuous"
        )
    finally:
        if grandchild > 0:
            _reap(grandchild)


# --------------------------------------------------------------------------------------
# THE CURE — the same tree, killed to the root.
# --------------------------------------------------------------------------------------

def test_run_in_process_group_kills_the_grandchild(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild.pid"
    grandchild = -1
    try:
        with pytest.raises(ProcessGroupTimeout) as caught:
            run_in_process_group(
                _script(pidfile), timeout=1.5, capture_output=True, text=True, term_grace_s=0.5
            )
        grandchild = _await_pidfile(pidfile)
        assert _await_death(grandchild), f"grandchild {grandchild} survived the group kill"
        receipt = caught.value.kill_receipt
        assert isinstance(receipt, GroupKillReceipt)
        assert receipt.term_delivered is True
        assert receipt.survivors_after is False
        assert caught.value.group_survivors_after_kill is False
    finally:
        if grandchild > 0:
            _reap(grandchild)


def test_the_timeout_is_still_a_TimeoutExpired_for_existing_handlers(tmp_path: Path) -> None:
    """Migration must not break `except subprocess.TimeoutExpired` at the call sites."""

    pidfile = tmp_path / "g.pid"
    grandchild = -1
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_in_process_group(_script(pidfile), timeout=1.0, term_grace_s=0.3)
        if pidfile.exists() and pidfile.read_text().strip().isdigit():
            grandchild = int(pidfile.read_text().strip())
    finally:
        if grandchild > 0:
            _reap(grandchild)


def test_a_process_that_ignores_sigterm_is_still_killed(tmp_path: Path) -> None:
    """The escalation leg: SIGTERM alone is not a kill."""

    stubborn = ["bash", "-c", "trap '' TERM; sleep 120"]
    with pytest.raises(ProcessGroupTimeout) as caught:
        run_in_process_group(stubborn, timeout=1.0, term_grace_s=0.4)
    assert caught.value.kill_receipt is not None
    assert caught.value.kill_receipt.kill_delivered is True
    assert caught.value.kill_receipt.survivors_after is False


def test_a_ctrl_c_while_waiting_also_takes_the_tree_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`start_new_session` detaches the child from our terminal's signals.

    Without the BaseException leg, the cure would ORPHAN on Ctrl-C — introducing the
    exact defect it exists to remove, on the interactive path.
    """

    pidfile = tmp_path / "grandchild.pid"
    real = subprocess.Popen.communicate
    state = {"first": True}

    def _interrupted(self, *args, **kwargs):
        if state["first"]:
            state["first"] = False
            _await_pidfile(pidfile)  # let the tree exist before we abort
            raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", _interrupted)
    grandchild = -1
    try:
        with pytest.raises(KeyboardInterrupt):
            run_in_process_group(_script(pidfile), timeout=60, term_grace_s=0.5)
        grandchild = _await_pidfile(pidfile)
        assert _await_death(grandchild), f"grandchild {grandchild} survived a Ctrl-C"
    finally:
        if grandchild > 0:
            _reap(grandchild)


# --------------------------------------------------------------------------------------
# The cure must not change the ordinary path.
# --------------------------------------------------------------------------------------

def test_a_normal_run_returns_the_same_completedprocess_shape() -> None:
    done = run_in_process_group(
        ["bash", "-c", "printf hello; printf oops >&2; exit 0"],
        timeout=30, capture_output=True, text=True,
    )
    assert done.returncode == 0
    assert done.stdout == "hello"
    assert done.stderr == "oops"


def test_a_nonzero_rc_is_returned_not_raised_unless_check(tmp_path: Path) -> None:
    done = run_in_process_group(["bash", "-c", "exit 7"], timeout=30)
    assert done.returncode == 7
    with pytest.raises(subprocess.CalledProcessError):
        run_in_process_group(["bash", "-c", "exit 7"], timeout=30, check=True)


def test_cwd_and_env_are_honoured(tmp_path: Path) -> None:
    (tmp_path / "marker").write_text("x")
    done = run_in_process_group(
        ["bash", "-c", 'ls marker; printf "%s" "$TAC_PROBE"'],
        cwd=tmp_path, env={**os.environ, "TAC_PROBE": "seen"},
        timeout=30, capture_output=True, text=True,
    )
    assert done.returncode == 0
    assert "seen" in done.stdout


# --------------------------------------------------------------------------------------
# Refusals — a group kill aimed wrong is worse than no kill.
# --------------------------------------------------------------------------------------

def test_killing_our_own_group_is_refused() -> None:
    with pytest.raises(ValueError, match="OWN process group"):
        kill_process_group(os.getpgrp())


@pytest.mark.parametrize("pgid", [0, -1, -42])
def test_non_positive_pgid_is_refused_because_killpg_broadcasts(pgid: int) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        kill_process_group(pgid)


def test_group_alive_is_false_for_a_dead_group_and_never_broadcasts() -> None:
    proc = subprocess.Popen(["bash", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    assert group_alive(proc.pid) is False
    assert group_alive(0) is False
    assert group_alive(-1) is False


def test_killing_an_already_dead_group_is_a_clean_no_op() -> None:
    proc = subprocess.Popen(["bash", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    receipt = kill_process_group(proc.pid, term_grace_s=0.1)
    assert receipt.survivors_after is False
    assert receipt.to_dict()["pgid"] == proc.pid


# --------------------------------------------------------------------------------------
# ANTI-TWIN — sister of test_process_liveness_no_new_copies.py.
# --------------------------------------------------------------------------------------

_MIGRATED = (
    "src/tac/submission_chain.py",
    "src/tac/deploy/modal/modal_asymmetric_warp_deploy.py",
)


def test_the_migrated_sites_route_through_the_canonical_helper() -> None:
    """A helper nobody imports is a twin waiting to happen."""

    repo = Path(__file__).resolve().parents[3]
    for rel in _MIGRATED:
        text = (repo / rel).read_text(encoding="utf-8")
        assert "run_in_process_group" in text, f"{rel} no longer routes through the helper"


def test_the_migrated_shell_timeouts_no_longer_use_bare_subprocess_run() -> None:
    """The exact defect signature: `subprocess.run` + `timeout=` in a migrated file.

    `subprocess.run` without a timeout cannot orphan on a timeout it does not have, so
    only the timed calls are in scope. A deliberate exception carries a same-line
    `# GROUP_KILL_OK:<rationale>` waiver.
    """

    repo = Path(__file__).resolve().parents[3]
    offenders: list[str] = []
    for rel in _MIGRATED:
        lines = (repo / rel).read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "subprocess.run(" not in line.split("#", 1)[0]:
                continue
            window = "\n".join(lines[index : index + 12])
            if "timeout=" in window and "GROUP_KILL_OK:" not in window:
                offenders.append(f"{rel}:{index + 1}: {line.strip()}")
    assert not offenders, (
        "these timed subprocess.run calls kill only the direct child; a shell or "
        "wrapper underneath orphans its worker (ddm_cpu1: 4,369 s past a 1,800 s "
        "timeout). Use tac.process_group_kill.run_in_process_group:\n  "
        + "\n  ".join(offenders)
    )
