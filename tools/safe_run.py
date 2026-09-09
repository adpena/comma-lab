#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""safe_run.py — run a command under HARD wall-time AND process-group RSS caps,
killing the whole process group cleanly *before* a runaway loop can OOM the host.

VENDORED from molt (``/Users/adpena/Projects/molt/tools/safe_run.py``,
origin/main 353784a1a) — borrowed-substrate accounting: this is molt's, copied
near-verbatim. It is the DEFENSE-IN-DEPTH layer-3 of this repo's OOM protection:
the whole-machine watchdog (``tools/memory_guard.py --watch``) sheds the largest
training arm when the machine approaches the 30 GB floor; this per-arm wrapper
makes EACH arm bound its OWN process-group RSS so it SIGKILLs itself before it
can balloon — bounding each arm, not just the whole machine.

CONTROL-PLANE SAFE BY CONSTRUCTION: this wrapper only ever kills the process
GROUP IT SPAWNED (``start_new_session=True`` → the child is its own session/
group leader, pgid == child pid). It can NEVER reach the control plane (claude /
codex / shell) because those are not in the spawned group. No allowlist/denylist
is needed here — the kill scope is structurally the spawned subtree only.

NOTE on molt provenance: safe_run is cited from molt 353784a1a, but its
group-only-kill design PREDATES / SIDESTEPS the control-plane-kill bug class
that the *selector* side had at that HEAD (the selector picks arbitrary
host processes; safe_run only ever kills the one group it created). The
selector-side fix (custody-gating + control-plane lineage) is vendored
separately in tools/memory_guard.py from molt's FIXED HEAD 3b1e49b18.

Usage:
    python3 tools/safe_run.py [options] -- CMD [ARGS...]
    python3 tools/safe_run.py [options] CMD [ARGS...]     # `--` optional

Options:
    --rss-mb N      kill if the process-group RSS exceeds N MiB   (default 2048)
    --timeout S     kill if wall-clock exceeds S seconds          (default 30)
    --poll S        RSS/timeout poll interval in seconds          (default 0.2)
    --label TEXT    label for the SAFE_RUN status line            (default cmd[0])
    --quiet         suppress the SAFE_RUN status line on success
    --json          emit a one-line JSON status to stderr (for tooling)

Exit codes:
    0..127   the child's own exit code (forwarded) when it exits on its own
    124      TIMEOUT  — killed for exceeding --timeout
    137      OOM      — killed for exceeding --rss-mb (128 + SIGKILL)
    125      could not start the command
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXIT_TIMEOUT = 124
EXIT_OOM = 137
EXIT_SPAWN = 125
STATUS_RECEIPT_SCHEMA = "safe_run_status_receipt.v1"
# verified-at-source: .omx/research/ddm_gov3_directive_cooperative_pause_and_stale_instance_20260905.md:88
PROJECTION_ATTRIBUTION_ERROR_GIB = 5.0


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        sep = argv.index("--")
        ours, cmd = argv[:sep], argv[sep + 1 :]
    else:
        ours, cmd = [], []
        known_val = {
            "--rss-mb",
            "--timeout",
            "--far-timeout",
            "--progress-stall-minutes",
            "--progress-history-jsonl",
            "--poll",
            "--label",
            "--status-receipt",
            "--child-pidfile",
            "--projected-gib",
            "--admission-override-rationale",
            "--admit-over-projection",
        }
        known_flag = {"--quiet", "--json", "--skip-admission-gate"}
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok in known_val:
                ours += [tok, argv[i + 1]] if i + 1 < len(argv) else [tok]
                i += 2
                continue
            if tok.split("=", 1)[0] in known_val and "=" in tok:
                ours.append(tok)
                i += 1
                continue
            if tok in known_flag:
                ours.append(tok)
                i += 1
                continue
            cmd = argv[i:]
            break
        else:
            cmd = []

    p = argparse.ArgumentParser(prog="safe_run.py", add_help=True)
    p.add_argument("--rss-mb", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--far-timeout",
        type=float,
        default=None,
        help="absolute far ceiling when progress heartbeats are present (at least 3x serial estimate)",
    )
    p.add_argument(
        "--progress-stall-minutes",
        type=float,
        default=None,
        help="kill only after history.jsonl completed_steps stops advancing for this many minutes",
    )
    p.add_argument("--progress-history-jsonl", type=Path, default=None)
    p.add_argument("--poll", type=float, default=0.2)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--status-receipt",
        type=Path,
        default=None,
        help=(
            "Durable JSON receipt updated atomically on every RSS sample tick "
            "with peak_rss_observed, last_sample_ts, and kill_action."
        ),
    )
    p.add_argument(
        "--child-pidfile",
        type=Path,
        default=None,
        help=(
            "Durable plain-text pidfile for the spawned child process. Operators "
            "must kill by this pidfile, not by pattern-matching the wrapped argv."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    # (review-fix CRITICAL) safe_run stamps the child TAC_GOVERNED_ADMISSION so the heavy
    # entrypoint's admission guard passes — but the per-process --rss-mb cap is exactly the
    # SYSTEM-BLIND mechanism that FAILED to prevent the 2026-07-02 >128GB crash. So before
    # stamping governed, safe_run now runs the SAME system-total admission gate the durable-daemon
    # path runs. These flags parameterize it (parity with spawn_durable_daemon).
    p.add_argument("--projected-gib", type=float, default=None,
                   help="projected peak footprint (GiB) for the system admission gate; "
                        "defaults to --rss-mb/1024.")
    p.add_argument("--admission-override-rationale", default=None,
                   help="operator-verbatim rationale to override a system-admission REFUSE.")
    p.add_argument(
        "--admit-over-projection",
        default=None,
        metavar="RATIONALE",
        help="explicit substantive escape for a projection more than 5 GiB above the ceiling",
    )
    p.add_argument("--skip-admission-gate", action="store_true",
                   help="skip the system admission gate (infra/protection commands only).")
    ns = p.parse_args(ours)
    if (ns.progress_stall_minutes is None) != (ns.progress_history_jsonl is None):
        p.error("--progress-stall-minutes and --progress-history-jsonl must be supplied together")
    if ns.progress_stall_minutes is not None and ns.progress_stall_minutes <= 0:
        p.error("--progress-stall-minutes must be positive")
    if ns.far_timeout is not None and ns.far_timeout < ns.timeout:
        p.error("--far-timeout must be at least --timeout")
    if ns.admit_over_projection is not None and not _rationale_is_real(ns.admit_over_projection):
        p.error("--admit-over-projection requires a substantive non-placeholder rationale")
    ns.projection_override_used = False
    return ns, cmd


def _group_rss_kib(pgid: int) -> int:
    """Sum RSS (KiB) of every process in the given process group via `ps`.

    Dependency-free; works on macOS + Linux. Returns 0 if `ps` reports nothing
    (e.g. the group already exited)."""
    try:
        out = subprocess.run(  # subprocess-no-check-OK: ps -g empty/failure means the group already exited (documented: returns 0)
            ["ps", "-o", "rss=", "-g", str(pgid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return 0
    total = 0
    for line in out.split():
        try:
            total += int(line)
        except ValueError:
            pass
    return total


def _kill_group(pgid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except Exception:
            pass
        time.sleep(0.15)


def _spawn_debug(cmd: list[str]) -> str:
    """Detailed debug context for a spawn failure — NO silent failures.

    Surfaces the collapse-to-one-arg class explicitly: an unquoted shell
    expansion that folds a whole command line into a single argv[0] shows up as
    ``len(cmd)=1`` with a cmd[0] full of spaces. Printing len(cmd), the first few
    tokens, and cwd makes that root cause obvious instead of a bare 'not found'.
    """
    cmd0 = cmd[0] if cmd else "<none>"
    detail = (
        f"len(cmd)={len(cmd)} cmd[:3]={cmd[:3]!r} cwd={os.getcwd()!r}"
    )
    if len(cmd) == 1 and (" " in cmd0):
        detail += (
            "  HINT: cmd[0] contains spaces AND len(cmd)==1 -> the command was "
            "likely word-split-collapsed into a single argv[0] (an unquoted shell "
            "expansion). Pass the command as separate argv tokens or via a script."
        )
    return detail


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_receipt_path(ns: argparse.Namespace) -> Path | None:
    if ns.status_receipt is not None:
        return ns.status_receipt.expanduser().resolve(strict=False)
    env_path = os.environ.get("SAFE_RUN_STATUS_RECEIPT")
    if env_path:
        return Path(env_path).expanduser().resolve(strict=False)
    return None


def _child_pidfile_path(ns: argparse.Namespace, status_receipt: Path | None) -> Path | None:
    if ns.child_pidfile is not None:
        return ns.child_pidfile.expanduser().resolve(strict=False)
    env_path = os.environ.get("SAFE_RUN_CHILD_PIDFILE")
    if env_path:
        return Path(env_path).expanduser().resolve(strict=False)
    if status_receipt is not None:
        return status_receipt.with_name(f"{status_receipt.name}.child.pid")
    return None


def _write_status_receipt(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _write_child_pidfile(path: Path | None, pid: int | None) -> None:
    if path is None or pid is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(f"{int(pid)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _child_only_kill_command(path: Path | None) -> str | None:
    if path is None:
        return None
    quoted = shlex.quote(str(path))
    return f"kill -TERM \"$(cat {quoted})\""


def _rationale_is_real(s: str | None) -> bool:
    """A real operator override rationale: non-placeholder, >= 8 chars."""
    if not s or not isinstance(s, str):
        return False
    low = s.strip().lower()
    return len(low) >= 8 and low not in ("<reason>", "<rationale>", "placeholder", "tbd", "n/a")


def projection_refusal_policy(
    *,
    admits: bool,
    projected_system_used_gib: float | None,
    adaptive_ceiling_gib: float | None,
    enforcing: bool,
    admit_over_projection: str | None,
) -> dict[str, Any]:
    """Pure SUM-over-RAM policy, including the hard >5 GiB refusal boundary."""
    overage = (
        float(projected_system_used_gib) - float(adaptive_ceiling_gib)
        if isinstance(projected_system_used_gib, (int, float))
        and isinstance(adaptive_ceiling_gib, (int, float))
        else None
    )
    hard = bool(not admits and overage is not None and overage > PROJECTION_ATTRIBUTION_ERROR_GIB)
    supplied = admit_over_projection is not None
    real_escape = _rationale_is_real(admit_over_projection)
    if admits:
        return {"action": "ADMIT", "hard": False, "overage_gib": overage, "override_used": False}
    if supplied and not real_escape:
        return {
            "action": "REFUSE",
            "hard": hard,
            "overage_gib": overage,
            "override_used": False,
            "reason": "--admit-over-projection rationale is missing or a placeholder",
        }
    if real_escape:
        return {"action": "ADMIT", "hard": hard, "overage_gib": overage, "override_used": True}
    return {
        "action": "REFUSE" if (hard or enforcing) else "ADVISORY",
        "hard": hard,
        "overage_gib": overage,
        "override_used": False,
    }


def _latest_completed_steps(path: Path | None) -> int | None:
    """Last valid completed_steps heartbeat in a possibly-partial JSONL file."""
    if path is None or not path.is_file():
        return None
    latest: int | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("completed_steps") if isinstance(row, dict) else None
                if isinstance(value, int) and not isinstance(value, bool):
                    latest = value
    except OSError:
        return None
    return latest


def _system_admission_gate(ns: argparse.Namespace, cmd: list[str]) -> int | None:
    """(review-fix CRITICAL) The SAME system-total admission gate the durable-daemon path runs, so
    that stamping the child as GOVERNED is TRUTHFUL (a real SUM-over-RAM check ran), not a per-
    process-cap fiction. Returns an exit code to ABORT with (refuse), or None to proceed.

    Parity with spawn_durable_daemon._system_admission_gate: REFUSE (rc=5) only when ENFORCE is
    armed AND the live system decision denies AND no real operator override; ADVISORY/unavailable/
    admit all proceed. Fail-safe: a governor import/read hiccup proceeds (the daemon path does the
    same) — never silently BLOCK, and never falsely refuse.

    AUTO-RECONCILE FIRST (ddm_gb1 D4, 2026-08-15) — mirrors
    ``witness_memory_preflight.system_aware_admission``. The other two admission paths converge the
    durable-daemon registry to GROUND TRUTH before projecting: witness_memory_preflight since the
    2026-07-09 phantom-growth fix, spawn_durable_daemon._do_start under its registry lock since
    2026-07-11. safe_run did NOT, and that asymmetry is measured: a daemon killed out-of-band
    (SIGKILL/OOM/jetsam/machine-sleep/SIGURG-144) can never write ``recorded=stopped``, and every
    phantom ``running`` row charges UNKNOWN_GROWTH_HEADROOM_GIB = 25 GiB of active growth. On
    2026-08-15 THREE dead rows (pids 7506, 8997, 31881) summed to "active-growth 100.0 GiB" and
    REFUSED a real relaunch twice; a manual ``reconcile_dead_daemons()`` converged them and the
    same launch was admitted at projected 81.6 < ceiling 116.0. Fail-OPEN on the reconcile leg (a
    self-clean must never crash a launch); the admission decision below stays fail-CLOSED."""
    if ns.skip_admission_gate:
        return None
    try:
        import system_memory_governor as gov  # tools/ is sys.path[0] for this script
    except Exception as exc:  # noqa: BLE001 — never let a governor hiccup block a launch
        print(f"safe_run.py: WARNING system admission gate unavailable ({exc!r}); proceeding "
              f"(per-process --rss-mb cap remains).", file=sys.stderr)
        return None
    try:
        import spawn_durable_daemon as _sdd  # tools/ is sys.path[0] for this script
        _sdd.reconcile_dead_daemons(verbose=False)
    except Exception:
        pass  # reconcile is a best-effort self-clean; the admission decision below is the real gate
    projected = ns.projected_gib if ns.projected_gib is not None else float(ns.rss_mb) / 1024.0
    try:
        ctx = gov.live_admission_decision(projected_new_gib=float(projected))
    except Exception as exc:  # noqa: BLE001
        print(f"safe_run.py: WARNING admission read failed ({exc!r}); proceeding.", file=sys.stderr)
        return None
    d = ctx.decision
    enforcing = gov.admission_enforcing()
    policy = projection_refusal_policy(
        admits=bool(d.admit),
        projected_system_used_gib=getattr(d, "projected_system_used_gib", None),
        adaptive_ceiling_gib=getattr(d, "adaptive_ceiling_gib", None),
        enforcing=enforcing,
        admit_over_projection=ns.admit_over_projection,
    )
    if policy["action"] == "ADMIT":
        if policy["override_used"]:
            ns.projection_override_used = True
            print(
                "safe_run.py: OVER-PROJECTION ADMISSION ESCAPE "
                f"(operator rationale): {ns.admit_over_projection!r} — proceeding despite: {d.reason}",
                file=sys.stderr,
            )
        return None
    if policy.get("reason"):
        print(f"safe_run.py: REFUSED: {policy['reason']}; {d.reason}", file=sys.stderr)
        return 5
    if _rationale_is_real(ns.admission_override_rationale) and not policy["hard"]:
        print(f"safe_run.py: ADMISSION OVERRIDE (operator rationale): "
              f"{ns.admission_override_rationale!r} — proceeding despite: {d.reason}", file=sys.stderr)
        return None
    action = policy["action"]
    overage = policy.get("overage_gib")
    hard_chain = (
        f"; overage {overage:.1f} GiB > attribution error {PROJECTION_ATTRIBUTION_ERROR_GIB:.1f} GiB"
        if policy["hard"] and isinstance(overage, (int, float))
        else ""
    )
    print(f"safe_run.py: {'REFUSED' if action == 'REFUSE' else 'WOULD-REFUSE (ADVISORY)'} "
          f"(system admission gate — SUM-over-RAM crash guard): {d.reason} "
          f"[projected={projected:.1f}GiB]{hard_chain}", file=sys.stderr)
    return 5 if action == "REFUSE" else None


# ── governed-admission registry visibility (review-fix HIGH: bare safe_run was invisible) ────────
# A BARE safe_run job (not wrapped by spawn_durable_daemon) never appeared in the durable-daemon
# registry: the governor's only view of it was the OUR_JOBS_PATTERN ps regex, with NO projected
# peak — so its remaining growth-to-peak was invisible to every other launch's admission decision.
# Fix: after safe_run's OWN admission gate admits, write the SAME pending-reservation row the
# daemon path writes (inside the SAME fcntl registry lock, closing the same TOCTOU), promote it to
# a running row (with the real child pid + projected peak) after Popen, and mark it stopped on
# exit. The helpers are the daemon's own (lazy same-dir import: single implementation of the
# locked/atomic registry mutation); registration is best-effort VISIBILITY — a registry hiccup
# never blocks or kills the run (the admission decision itself already stood).


def _sdd_module():
    """Lazy import of tools/spawn_durable_daemon (same directory) for the shared registry helpers.
    Returns the module or None — safe_run stays dependency-light and never fails on a missing
    sibling (the gate still ran; only the growth-accounting visibility degrades to the ps regex)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import spawn_durable_daemon as sdd  # noqa: PLC0415
        return sdd
    except Exception:
        return None


def _gate_and_reserve(ns: argparse.Namespace, cmd: list[str]):
    """Admission gate + (on ADMIT) pending-reservation write, in ONE locked critical section —
    mirrors spawn_durable_daemon._do_start's TOCTOU fix for the bare-safe_run path.

    Returns ``(abort_rc | None, reservation | None)``; ``reservation = (sdd, label, projected)``
    is later promoted by ``_activate_reservation`` / released by ``_release_reservation``.
    Skipped entirely when ``--skip-admission-gate`` (the daemon-wrapped path: the daemon already
    gated AND registered this job) and under pytest (hermetic — never write the LIVE registry
    from a test; the reservation helpers are unit-tested directly with tmp paths)."""
    if ns.skip_admission_gate:
        return None, None
    sdd = _sdd_module()
    if sdd is None or os.environ.get("PYTEST_CURRENT_TEST"):
        return _system_admission_gate(ns, cmd), None
    projected = ns.projected_gib if ns.projected_gib is not None else float(ns.rss_mb) / 1024.0
    label = f"saferun_{ns.label or os.path.basename(cmd[0])}_pid{os.getpid()}"
    try:
        with sdd._registry_lock():
            sdd._update_registry_locked(sdd._sweep_stale_pending_rows)
            rc = _system_admission_gate(ns, cmd)
            if rc is not None:
                return rc, None
            sdd._write_pending_reservation(label, projected)
            return None, (sdd, label, float(projected))
    except Exception as exc:  # noqa: BLE001 — visibility is best-effort, never blocks the run
        print(f"safe_run.py: WARNING registry reservation failed ({exc!r}); proceeding with the "
              f"un-reserved gate (growth accounting falls back to the ps-pattern view).",
              file=sys.stderr)
        return _system_admission_gate(ns, cmd), None


def _activate_reservation(reservation, pid: int, pgid: int, cmd: list[str]) -> None:
    """Promote the pending reservation to a live 'running' registry row carrying the real child
    pid + projected peak, so other launches' admission decisions count this job's remaining
    growth-to-peak (not just its ps-visible current RSS). Best-effort."""
    if not reservation:
        return
    sdd, label, projected = reservation
    try:
        import datetime as _dt
        sdd._register_daemon({
            "label": label, "pid": int(pid), "pgid": int(pgid), "cmd": list(cmd), "log": "",
            "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "cwd": os.getcwd(), "status": "running", "projected_peak_gib": float(projected),
        })
    except Exception as exc:  # noqa: BLE001
        print(f"safe_run.py: WARNING registry promote failed ({exc!r}); job runs unregistered "
              f"(ps-pattern fallback covers it).", file=sys.stderr)


def _release_reservation(reservation, *, reason: str) -> None:
    """Release the registry footprint on exit: drop a still-pending reservation (spawn failed) or
    mark the promoted running row stopped. Best-effort. NOTE: a hard SIGKILL of safe_run itself
    skips this — the leaked 'running' row is harmless (its dead pid drops out of the governor's
    live join) and spawn_durable_daemon --reconcile cleans it up."""
    if not reservation:
        return
    sdd, label, _projected = reservation
    try:
        sdd._clear_pending_reservation(label)
        sdd._mark_stopped(label, reason=reason)
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str]) -> int:
    ns, cmd = _parse_args(argv)
    if not cmd:
        print(
            "safe_run.py: no command given (use `-- CMD ARGS`); "
            f"received argv={argv[:6]!r} (len={len(argv)}) cwd={os.getcwd()!r}",
            file=sys.stderr,
        )
        return EXIT_SPAWN

    label = ns.label or os.path.basename(cmd[0])
    rss_limit_kib = ns.rss_mb * 1024
    start = time.monotonic()
    start_utc = _utc_now()
    status_receipt = _status_receipt_path(ns)
    child_pidfile = _child_pidfile_path(ns, status_receipt)
    progress_history = (
        None
        if ns.progress_history_jsonl is None
        else ns.progress_history_jsonl.expanduser().resolve(strict=False)
    )
    far_timeout = (
        float(ns.far_timeout)
        if ns.far_timeout is not None
        else (float(ns.timeout) * 3.0 if progress_history is not None else float(ns.timeout))
    )
    peak_kib = 0
    peak_observed = False
    last_sample_ts: str | None = None
    child_pid: int | None = None
    pgid: int | None = None
    kill_action: dict[str, Any] | None = None
    heartbeat_seen = False
    last_completed_steps: int | None = None
    last_progress_monotonic = start
    last_progress_utc: str | None = None

    def _emit_status(status: str, *, exit_code: int | None = None) -> None:
        _write_status_receipt(
            status_receipt,
            {
                "schema": STATUS_RECEIPT_SCHEMA,
                "generated_utc": _utc_now(),
                "start_utc": start_utc,
                "label": label,
                "argv": list(cmd),
                "child_pid": child_pid,
                "pgid": pgid,
                "child_pidfile": None if child_pidfile is None else str(child_pidfile),
                "child_only_kill_command": _child_only_kill_command(child_pidfile),
                "operator_kill_rule": (
                    "Kill the child pid from child_pidfile; never pkill/pgrep by the wrapped argv."
                ),
                "status": status,
                "exit": exit_code,
                # (ddm_av3 F2) `status` is "ok" for ANY child exit that was not a
                # timeout/oom/kill/interrupt -- so a CRASHED child reads
                # `status=ok, exit=1`.  ddm_lr1/A2 produced exactly that receipt
                # after losing its whole result to an IsADirectoryError.  The
                # exit-code passthrough contract is unchanged (consumers that
                # already check `exit` keep working); these two derived fields
                # are additive, so a consumer keying on `status` alone can see
                # the disagreement instead of reading a crash as a success.
                "child_exit_nonzero": exit_code is not None and exit_code != 0,
                "receipt_status_disagrees_with_exit": (
                    status == "ok" and exit_code is not None and exit_code != 0
                ),
                "elapsed_s": round(time.monotonic() - start, 3),
                "rss_limit_mib": ns.rss_mb,
                "timeout_s": ns.timeout,
                "far_timeout_s": far_timeout,
                "progress_stall_minutes": ns.progress_stall_minutes,
                "progress_history_jsonl": None if progress_history is None else str(progress_history),
                "heartbeat_seen": heartbeat_seen,
                "last_completed_steps": last_completed_steps,
                "last_progress_utc": last_progress_utc,
                "admit_over_projection_rationale": ns.admit_over_projection,
                "projection_override_used": bool(ns.projection_override_used),
                "projection_attribution_error_gib": PROJECTION_ATTRIBUTION_ERROR_GIB,
                "poll_s": ns.poll,
                "peak_rss_observed": peak_observed,
                "peak_rss_kib": int(peak_kib),
                "peak_rss_mib": round(peak_kib / 1024.0, 3),
                "last_sample_ts": last_sample_ts,
                "kill_action": kill_action,
            },
        )

    # (review-fix CRITICAL) run the SYSTEM-TOTAL admission gate BEFORE stamping governed. The old
    # code stamped governed off the per-process --rss-mb cap alone — the exact SYSTEM-BLIND
    # mechanism that failed to stop the 2026-07-02 >128GB crash. Now the stamp is truthful: a real
    # SUM-over-RAM decision ran. REFUSE (rc=5) aborts before spawn when enforce is armed.
    # (review-fix HIGH) gate + pending-reservation in ONE locked section (bare-safe_run visibility
    # + the same TOCTOU close as the daemon path); no-op for --skip-admission-gate (daemon-wrapped).
    _adm_rc, _reservation = _gate_and_reserve(ns, cmd)
    if _adm_rc is not None:
        _emit_status("admission_refused", exit_code=_adm_rc)
        return _adm_rc

    # (#254) stamp the child env as GOVERNED so the heavy entrypoint's admission guard passes —
    # safe_run IS a governed path (it now runs the system admission gate above + RSS-caps +
    # shed-cascades the child). Set the marker DIRECTLY by its stable name (==
    # tac.admission_guard.GOVERNED_MARKER_ENV) so it works even when src/ is not importable from
    # this tools/ launcher. A raw launch that skips safe_run/launch_witness_run lacks this marker
    # and is refused when enforce is armed.
    _child_env = dict(os.environ)
    _child_env["TAC_GOVERNED_ADMISSION"] = "1"

    try:
        # start_new_session=True -> child is a session+group leader (pgid == pid),
        # so we can SIGKILL the whole tree and not just the immediate child.
        proc = subprocess.Popen(cmd, start_new_session=True, env=_child_env)
    except FileNotFoundError as exc:
        print(
            f"safe_run.py: failed to start {cmd[0]!r}: command not found "
            f"({exc}) [{_spawn_debug(cmd)}]",
            file=sys.stderr,
        )
        _release_reservation(_reservation, reason="safe_run_spawn_failed")
        _emit_status("spawn_failed", exit_code=EXIT_SPAWN)
        return EXIT_SPAWN
    except Exception as exc:  # noqa: BLE001
        print(
            f"safe_run.py: failed to start {cmd[0]!r}: {exc!r} [{_spawn_debug(cmd)}]",
            file=sys.stderr,
        )
        _release_reservation(_reservation, reason="safe_run_spawn_failed")
        _emit_status("spawn_failed", exit_code=EXIT_SPAWN)
        return EXIT_SPAWN

    child_pid = proc.pid
    pgid = proc.pid  # equals the new session/group id
    _write_child_pidfile(child_pidfile, child_pid)
    # promote the pending reservation to a live running row (real pid + projected peak).
    _activate_reservation(_reservation, proc.pid, pgid, cmd)
    _emit_status("running", exit_code=None)
    status = "ok"
    rc = 0

    # HIGH-1 fix (cascade): the inner child is its OWN session/group leader, so an
    # EXTERNAL killpg(THIS wrapper's group) reaches ONLY the wrapper, not the
    # child — and spawn_durable_daemon records THIS wrapper as the daemon's
    # custody pgid, so the OOM watchdog sheds the arm by killpg(wrapper). Without
    # cascading, the wrapper would die and the inner trainer would reparent to
    # launchd and keep running UNCAPPED (layer-2 kill would destroy layer-3 cap).
    # Install SIGTERM/SIGINT handlers that kill the child's group, then exit, so
    # an external shed CASCADES to the arm.
    def _cascade_kill(signum, _frame):  # noqa: ANN001
        nonlocal kill_action
        kill_action = {
            "reason": "external_signal",
            "signal": int(signum),
            "action": "SIGTERM_then_SIGKILL_process_group",
        }
        _emit_status("killed", exit_code=128 + signum)
        _kill_group(pgid)
        os._exit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _cascade_kill)
        signal.signal(signal.SIGINT, _cascade_kill)
    except (ValueError, OSError):
        pass  # not main thread / unsupported — fall back to default behavior

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            elapsed = time.monotonic() - start
            now = time.monotonic()
            observed_steps = _latest_completed_steps(progress_history)
            if observed_steps is not None and (
                not heartbeat_seen or last_completed_steps is None or observed_steps > last_completed_steps
            ):
                heartbeat_seen = True
                last_completed_steps = observed_steps
                last_progress_monotonic = now
                last_progress_utc = _utc_now()
            timeout_reason: str | None = None
            if heartbeat_seen and ns.progress_stall_minutes is not None:
                if now - last_progress_monotonic > ns.progress_stall_minutes * 60.0:
                    timeout_reason = "progress_stall"
                elif elapsed > far_timeout:
                    timeout_reason = "far_timeout"
            elif elapsed > ns.timeout:
                # No heartbeat ever appeared: preserve the pre-gov3 wall-clock behavior exactly.
                timeout_reason = "timeout"
            if timeout_reason is not None:
                status = timeout_reason
                kill_action = {
                    "reason": timeout_reason,
                    "elapsed_s": round(elapsed, 3),
                    "timeout_s": ns.timeout,
                    "far_timeout_s": far_timeout,
                    "progress_stall_minutes": ns.progress_stall_minutes,
                    "last_completed_steps": last_completed_steps,
                    "action": "SIGTERM_then_SIGKILL_process_group",
                }
                _emit_status(status, exit_code=EXIT_TIMEOUT)
                _kill_group(pgid)
                proc.wait()
                rc = EXIT_TIMEOUT
                break
            rss = _group_rss_kib(pgid)
            peak_observed = True
            last_sample_ts = _utc_now()
            if rss > peak_kib:
                peak_kib = rss
            _emit_status("running", exit_code=None)
            if rss > rss_limit_kib:
                status = "oom"
                kill_action = {
                    "reason": "rss_limit",
                    "rss_kib": int(rss),
                    "rss_limit_kib": int(rss_limit_kib),
                    "action": "SIGTERM_then_SIGKILL_process_group",
                }
                _emit_status(status, exit_code=EXIT_OOM)
                _kill_group(pgid)
                proc.wait()
                rc = EXIT_OOM
                break
            time.sleep(ns.poll)
    except KeyboardInterrupt:
        status = "interrupted"
        kill_action = {
            "reason": "keyboard_interrupt",
            "action": "SIGTERM_then_SIGKILL_process_group",
        }
        _emit_status(status, exit_code=130)
        _kill_group(pgid)
        proc.wait()
        rc = 130

    if rc is None:
        rc = proc.returncode if proc.returncode is not None else 0

    elapsed = time.monotonic() - start
    peak_mib = peak_kib / 1024.0
    detail = (
        f"status={status} exit={rc} peak_rss={peak_mib:.0f}MiB "
        f"elapsed={elapsed:.2f}s limit_rss={ns.rss_mb}MiB limit_t={ns.timeout:g}s"
    )
    if ns.json:
        print(
            "SAFE_RUN "
            + json.dumps(
                {
                    "label": label,
                    "status": status,
                    "exit": rc,
                    "peak_rss_mib": round(peak_mib),
                    "elapsed_s": round(elapsed, 3),
                    "rss_limit_mib": ns.rss_mb,
                    "timeout_s": ns.timeout,
                }
            ),
            file=sys.stderr,
        )
    elif not (ns.quiet and status == "ok"):
        print(f"SAFE_RUN [{label}] {detail}", file=sys.stderr)

    _release_reservation(_reservation, reason=f"safe_run_exit_{status}")
    _emit_status(status, exit_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
