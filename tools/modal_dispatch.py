#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reliable, detached Modal dispatch wrapper — fire / status / logs / stop / harvest.

WHY (2026-07-12): the witness cloud launcher's preflight->H100 orchestration takes
>2 min (image build/push) and its dispatch identity (app_id / call_id) was only
recoverable by grepping raw logs. The Claude Code bash harness kills foreground
commands at ~2 min and background commands at ~5 min (SIGURG-144), so the launcher
kept dying mid-orchestration and the app_id had to be hand-hunted every time.

This wrapper fixes both:
  * ``fire`` launches the target command FULLY DETACHED (``start_new_session``; own
    process group; stdio -> /dev/null) so it survives the harness entirely, tees
    its combined output to a durable log, and a watcher extracts the app_id +
    call_id + terminal state into a durable JSON state file.
  * ``status`` / ``logs`` / ``stop`` / ``harvest`` then operate off that durable
    state — no forensics, no lost dispatches, one reliable answer.

It is provider-command-agnostic: ``fire`` runs whatever argv follows ``--`` (the
witness cloud launcher, an eval, anything), so it is not coupled to one recipe.

Subcommands:
  fire    -- detach-launch a command; prints dispatch_id immediately and returns
  status  -- durable state (+ optional live modal query) for one/all dispatches
  logs    -- tail the durably-captured launcher log
  stop    -- ``modal app stop <app_id>`` money kill-switch
  harvest -- delegate to tools/harvest_modal_calls.py for the captured call_id
  _run    -- (internal) the detached worker; do not call directly

Durable state lives under ``.omx/state/modal_dispatch/`` (per the no-/tmp rule):
  <dispatch_id>.json  -- machine-readable dispatch state
  <dispatch_id>.log   -- combined stdout+stderr of the launched command
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".omx" / "state" / "modal_dispatch"

# Modal resource ids: ap-<22 base62> (app), fc-<26 crockford> (FunctionCall).
_APP_RE = re.compile(r"\bap-[A-Za-z0-9]{10,}\b")
_CALL_RE = re.compile(r"\bfc-[A-Za-z0-9]{10,}\b")
_REFUSED_RE = re.compile(r"\bREFUSED\b")
_TRACEBACK_RE = re.compile(r"\bTraceback \(most recent call last\)")

TERMINAL_STATES = {"done", "failed", "refused"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_path(dispatch_id: str) -> Path:
    return STATE_DIR / f"{dispatch_id}.json"


def _log_path(dispatch_id: str) -> Path:
    return STATE_DIR / f"{dispatch_id}.log"


def _write_state(path: Path, state: dict) -> None:
    """Atomic state write (tmp + os.replace) so a reader never sees a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- fire
def cmd_fire(args: argparse.Namespace) -> int:
    argv = args.argv
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("REFUSED: fire needs a command after '--'", file=sys.stderr)
        return 2
    dispatch_id = f"{args.label}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    state_path = _state_path(dispatch_id)
    log_path = _log_path(dispatch_id)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _write_state(
        state_path,
        {
            "dispatch_id": dispatch_id,
            "label": args.label,
            "argv": argv,
            "state": "spawning",
            "app_id": None,
            "call_id": None,
            "launcher_pid": None,
            "launcher_rc": None,
            "started_at_utc": _now(),
            "log_path": str(log_path),
        },
    )
    # Detached worker: new session (own pgid), stdio -> /dev/null, so it is fully
    # immune to the harness foreground-2min and background-SIGURG kills.
    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run",
        "--dispatch-id",
        dispatch_id,
        "--",
        *argv,
    ]
    with open(os.devnull, "wb") as devnull:
        proc = subprocess.Popen(
            worker,
            cwd=str(REPO),
            stdin=devnull,
            stdout=devnull,
            stderr=devnull,
            start_new_session=True,
        )
    print(
        json.dumps(
            {
                "dispatch_id": dispatch_id,
                "detached_worker_pid": proc.pid,
                "state_path": str(state_path),
                "log_path": str(log_path),
                "status_cmd": f".venv/bin/python tools/modal_dispatch.py status --dispatch-id {dispatch_id}",
                "note": "detached (start_new_session); survives the harness. Poll status; do not block.",
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------- _run
def cmd_run(args: argparse.Namespace) -> int:
    """Detached worker: run argv, tee to log, extract ids, persist terminal state."""
    argv = args.argv
    if argv and argv[0] == "--":
        argv = argv[1:]
    dispatch_id = args.dispatch_id
    state_path = _state_path(dispatch_id)
    log_path = _log_path(dispatch_id)
    state = _read_state(state_path) or {"dispatch_id": dispatch_id, "argv": argv}
    state.setdefault("app_id", None)
    state.setdefault("call_id", None)
    state["launcher_pid"] = os.getpid()
    state["state"] = "running"
    _write_state(state_path, state)

    refused = False
    traceback_seen = False
    with open(log_path, "w", buffering=1) as log:
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            dirty = False
            if state.get("app_id") is None and (m := _APP_RE.search(line)):
                state["app_id"] = m.group(0)
                dirty = True
            if state.get("call_id") is None and (m := _CALL_RE.search(line)):
                state["call_id"] = m.group(0)
                state["state"] = "dispatched"
                dirty = True
            if _REFUSED_RE.search(line):
                refused = True
            if _TRACEBACK_RE.search(line):
                traceback_seen = True
            if dirty:
                _write_state(state_path, state)
        rc = proc.wait()

    state["launcher_rc"] = rc
    if refused and state.get("call_id") is None:
        state["state"] = "refused"
    elif rc != 0 or traceback_seen:
        # rc!=0 but a call_id was captured => the dispatch still spawned; flag it
        # as failed-launcher for review, but the call_id remains harvestable.
        state["state"] = "failed"
    elif state.get("call_id") is not None:
        state["state"] = "done"
    else:
        state["state"] = "failed"
    _write_state(state_path, state)
    return 0


# -------------------------------------------------------------------------- status
def _live_call_state(call_id: str, state: dict | None = None) -> str:
    """Bounded liveness probe of a FunctionCall; never raises.

    Terminal observations are mirrored into the canonical call-id ledger per
    Catalog #330 before the status string is returned.
    """
    try:
        import modal

        try:
            result = modal.functions.FunctionCall.from_id(call_id).get(timeout=2)
            # The probe already paid for the payload; persist its structured
            # signal instead of discarding it for a bare status string.
            _mirror_terminal_call_state(
                call_id,
                harvested=_ledger_signal_fields(result),
                fallback_claim={"status": "completed_modal_result_returned"},
                state=state,
            )
            return "completed"
        except TimeoutError:
            # Plain poll timeout is NONTERMINAL — leave the ledger in-flight.
            return "running"
        except Exception as exc:
            name = type(exc).__name__
            if "OutputExpired" in name:
                _mirror_terminal_call_state(
                    call_id, harvested={"status": "expired"}, state=state
                )
                return "expired(>24h)"
            return f"unknown({name})"
    except Exception:
        return "unqueryable"


# The exact fields the canonical ledger helper reads. `fire` wraps an ARBITRARY
# launcher command, so a returned result may carry anything (including a large
# artifact blob). The ledger is a small shared append-only JSONL under a size
# policy, so the probe mirrors this bounded projection rather than dumping a
# stranger's payload into it. Nothing the helper consumes is lost; persisting the
# ARTIFACTS is `cmd_harvest`'s job (tools/harvest_modal_calls.py --execute),
# which writes them to their own durable path.
_LEDGER_SIGNAL_FIELDS: tuple[str, ...] = (
    "status",
    "crash_kind",
    "rc",
    "returncode",
    "elapsed_seconds",
    "score",
    "score_axis",
    "archive_sha256",
    "archive_bytes",
    "evidence_grade",
)


def _ledger_signal_fields(result: object) -> dict | None:
    """Project a Modal result onto the fields the call-id ledger actually reads."""
    if not isinstance(result, dict):
        return None
    projected = {k: result[k] for k in _LEDGER_SIGNAL_FIELDS if k in result}
    return projected or None


def _mirror_terminal_call_state(
    call_id: str,
    *,
    harvested: dict | None,
    fallback_claim: dict | None = None,
    state: dict | None = None,
) -> None:
    """Catalog #330 — mirror a TERMINAL Modal observation into the call-id ledger.

    ``status --live`` polls ``FunctionCall.get``, which is exactly the surface
    Catalog #330 governs: a probe that observes terminal provider state and does
    not mirror it leaves the canonical ledger stuck at ``dispatched`` forever
    (the harvest-side sister of the Catalog #245 dispatch-registration gap).

    Delegates to the canonical helper — the same thin-wrapper shape
    ``tools/harvest_modal_calls.py`` and ``tools/parallel_harvest_actuator.py``
    use — so classification stays in one place. Best-effort: a read-only status
    command must not fail because the ledger hiccuped, but the failure is
    printed rather than swallowed so the silence never passes for success.
    """
    try:
        from tac.deploy.modal.harvest_outcomes import (
            append_terminal_call_id_ledger_event,
            call_ledger_status_from_terminal_harvest,
        )

        # rc-derived classification wins whenever the returned payload carries
        # one (rc=0 -> harvested, rc!=0 -> failed). Only when the payload is
        # unclassifiable do we fall back to the weaker — but still observed —
        # claim that the provider returned a terminal result. Never fabricate an
        # rc we did not see.
        terminal_claim = None
        if (
            fallback_claim is not None
            and call_ledger_status_from_terminal_harvest(harvested=harvested) is None
        ):
            terminal_claim = fallback_claim

        metadata: dict = {"call_id": call_id, "platform": "modal"}
        if isinstance(state, dict):
            metadata["label"] = state.get("label")
            metadata["dispatched_at_utc"] = state.get("started_at_utc")

        outcome = append_terminal_call_id_ledger_event(
            repo_root=REPO,
            metadata=metadata,
            harvested=harvested,
            terminal_claim=terminal_claim,
            agent="modal_dispatch_status_live",
        )
        reason = outcome.get("reason")
        if not outcome.get("appended") and not outcome.get("already_terminal") and reason:
            print(
                f"WARN: call_id ledger not updated for {call_id}: {reason}",
                file=sys.stderr,
            )
    except Exception as exc:  # never break a read-only status command
        print(
            f"WARN: call_id ledger mirror failed for {call_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def cmd_status(args: argparse.Namespace) -> int:
    if args.dispatch_id:
        paths = [_state_path(args.dispatch_id)]
    else:
        paths = sorted(STATE_DIR.glob("*.json"))
        if args.label:
            paths = [p for p in paths if _read_state(p) and _read_state(p).get("label") == args.label]
        if not args.all:
            paths = paths[-args.limit :]
    rows = []
    for p in paths:
        st = _read_state(p)
        if not st:
            continue
        live = ""
        if args.live and st.get("call_id"):
            live = _live_call_state(st["call_id"], st)
        rows.append(
            {
                "dispatch_id": st.get("dispatch_id"),
                "state": st.get("state"),
                "live_call": live,
                "app_id": st.get("app_id"),
                "call_id": st.get("call_id"),
                "launcher_rc": st.get("launcher_rc"),
                "started_at_utc": st.get("started_at_utc"),
                "updated_at_utc": st.get("updated_at_utc"),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("(no dispatches found under .omx/state/modal_dispatch/)")
        return 0
    for r in rows:
        live = f"  live={r['live_call']}" if r["live_call"] else ""
        print(
            f"{r['dispatch_id']}  [{r['state']}]{live}\n"
            f"    app={r['app_id']}  call={r['call_id']}  rc={r['launcher_rc']}  "
            f"upd={r['updated_at_utc']}"
        )
    return 0


# ---------------------------------------------------------------------------- logs
def cmd_logs(args: argparse.Namespace) -> int:
    log_path = _log_path(args.dispatch_id)
    if not log_path.exists():
        print(f"(no log at {log_path})", file=sys.stderr)
        return 1
    lines = log_path.read_text().splitlines()
    for line in lines[-args.tail :]:
        print(line)
    return 0


# ---------------------------------------------------------------------------- stop
def cmd_stop(args: argparse.Namespace) -> int:
    st = _read_state(_state_path(args.dispatch_id))
    if not st or not st.get("app_id"):
        print(f"REFUSED: no app_id captured for {args.dispatch_id}", file=sys.stderr)
        return 2
    app_id = st["app_id"]
    print(f"stopping app {app_id} (money kill-switch) ...")
    r = subprocess.run(
        [".venv/bin/modal", "app", "stop", app_id], cwd=str(REPO), text=True
    )
    return r.returncode


# ------------------------------------------------------------------------- harvest
def cmd_harvest(args: argparse.Namespace) -> int:
    st = _read_state(_state_path(args.dispatch_id))
    if not st or not st.get("call_id"):
        print(f"REFUSED: no call_id captured for {args.dispatch_id}", file=sys.stderr)
        return 2
    cmd = [
        sys.executable,
        "tools/harvest_modal_calls.py",
        "--from-ledger",
        "--call-id",
        st["call_id"],
    ]
    if args.execute:
        cmd.append("--execute")
    r = subprocess.run(cmd, cwd=str(REPO), text=True)
    return r.returncode


# ----------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fire = sub.add_parser("fire", help="detach-launch a modal dispatch command")
    p_fire.add_argument("--label", required=True, help="short slug for this dispatch")
    p_fire.add_argument("argv", nargs=argparse.REMAINDER, help="-- <command to run>")
    p_fire.set_defaults(func=cmd_fire)

    p_run = sub.add_parser("_run", help="(internal) detached worker")
    p_run.add_argument("--dispatch-id", required=True)
    p_run.add_argument("argv", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="durable dispatch state (+ optional live query)")
    p_status.add_argument("--dispatch-id", default=None)
    p_status.add_argument("--label", default=None)
    p_status.add_argument("--all", action="store_true", help="show every dispatch")
    p_status.add_argument("--limit", type=int, default=5, help="most-recent N (default 5)")
    p_status.add_argument("--live", action="store_true", help="probe modal FunctionCall liveness")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="tail the captured launcher log")
    p_logs.add_argument("--dispatch-id", required=True)
    p_logs.add_argument("--tail", type=int, default=40)
    p_logs.set_defaults(func=cmd_logs)

    p_stop = sub.add_parser("stop", help="modal app stop <app_id> (money kill-switch)")
    p_stop.add_argument("--dispatch-id", required=True)
    p_stop.set_defaults(func=cmd_stop)

    p_harvest = sub.add_parser("harvest", help="harvest the captured call_id")
    p_harvest.add_argument("--dispatch-id", required=True)
    p_harvest.add_argument("--execute", action="store_true")
    p_harvest.set_defaults(func=cmd_harvest)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
