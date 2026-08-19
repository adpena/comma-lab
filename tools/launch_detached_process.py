#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch a long local command as a watched detached session with provenance."""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "detached_local_process_launch.v2"
DONE_RECEIPT_SCHEMA = "detached_local_process_done.v2"
CONSUMED_RECEIPT_SCHEMA = "detached_local_process_done_consumed.v1"

# See the v1 history for the measured 300-360s fleet-reaper incidents.  A
# detached argv containing either standalone word is not safe without the
# explicit fleet carve-out.
_REAPER_NAME_PREDICATE = re.compile(r"\b(claude|codex)\b")
_RECEIPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class LaunchRefusal(RuntimeError):
    def __init__(self, message: str, *, rc: int = 2, **detail: Any) -> None:
        super().__init__(message)
        self.rc = rc
        self.detail = detail


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _git_sha(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(_canonical_json(payload), encoding="utf-8")
    tmp.replace(path)


def _tail_lines(path: Path, limit: int = 20) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs_dir() -> Path:
    # Preserve the original CLI contract: receipt custody is relative to the
    # launcher's invocation directory, which is the repository in production
    # and a hermetic temporary tree in controls.
    return Path.cwd() / ".omx" / "tmp" / "codex_runs"


def _next_launch_counter(runs: Path) -> int:
    runs.mkdir(parents=True, exist_ok=True)
    counter_path = runs / "_detached_launch_counter"
    lock_path = runs / "_detached_launch_counter.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            current = int(counter_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            current = 0
        value = current + 1
        # Keep a plain integer for bounded shell inspection.
        tmp = counter_path.with_name(f".{counter_path.name}.plain.{os.getpid()}")
        tmp.write_text(f"{value}\n", encoding="utf-8")
        tmp.replace(counter_path)
        return value


def _consumed_path(done_path: Path) -> Path:
    return done_path.with_name(done_path.name + ".consumed.json")


def _armed_path(done_path: Path) -> Path:
    return done_path.with_name(done_path.name + ".armed.json")


def _receipt_is_consumed(done_path: Path) -> bool:
    marker = _consumed_path(done_path)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return (
            isinstance(payload, dict)
            and payload.get("schema") == CONSUMED_RECEIPT_SCHEMA
            and payload.get("receipt_sha256") == _sha256(done_path)
        )
    except (OSError, json.JSONDecodeError):
        return False


def _tombstone_receipt(done_path: Path, *, reason: str) -> dict[str, Any]:
    stamp = _utc_compact()
    digest = _sha256(done_path)
    target = done_path.with_name(f"{done_path.name}.superseded.{stamp}.{digest[:12]}")
    ordinal = 1
    while target.exists():
        target = done_path.with_name(
            f"{done_path.name}.superseded.{stamp}.{digest[:12]}.{ordinal}"
        )
        ordinal += 1
    done_path.replace(target)
    consumed = _consumed_path(done_path)
    consumed_target: str | None = None
    if consumed.exists():
        moved = target.with_name(target.name + ".consumed.json")
        consumed.replace(moved)
        consumed_target = str(moved)
    record = {
        "schema": "detached_local_process_receipt_tombstone.v1",
        "generated_utc": _utc_now(),
        "reason": reason,
        "original_path": str(done_path),
        "tombstone_path": str(target),
        "receipt_sha256": digest,
        "consumed_marker_tombstone": consumed_target,
    }
    _write_json(target.with_name(target.name + ".tombstone.json"), record)
    return record


def _reserve_receipt(path: Path, *, receipt_name: str, counter: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "detached_local_process_receipt_arm.v1",
        "generated_utc": _utc_now(),
        "receipt_name": receipt_name,
        "monotonic_launch_counter": counter,
        "owner_launcher_pid": os.getpid(),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise LaunchRefusal(
            "done receipt name already has an active launch reservation",
            rc=6,
            receipt_arm_path=str(path),
        ) from exc


def _update_receipt_reservation(
    path: Path | None,
    *,
    identity: Mapping[str, Any],
) -> None:
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LaunchRefusal("receipt reservation is malformed", rc=6, receipt_arm_path=str(path))
    payload["launch_id"] = dict(identity)
    _write_json(path, payload)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _receipt_reservation_active(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        owner_pid = int(payload["owner_launcher_pid"])
        launch_id = payload.get("launch_id")
        launch_pid = int(launch_id["pid"]) if isinstance(launch_id, dict) else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        # A malformed reservation cannot certify that no launch still owns it.
        return True
    return _pid_exists(owner_pid) or (launch_pid is not None and _pid_exists(launch_pid))


def _tombstone_reservation(path: Path, *, dry_run: bool) -> dict[str, Any]:
    digest = _sha256(path)
    if dry_run:
        return {
            "dry_run_would_tombstone": str(path),
            "reason": "explicit_stale_reservation_supersede",
            "receipt_arm_sha256": digest,
        }
    stamp = _utc_compact()
    target = path.with_name(f"{path.name}.superseded.{stamp}.{digest[:12]}")
    ordinal = 1
    while target.exists():
        target = path.with_name(f"{path.name}.superseded.{stamp}.{digest[:12]}.{ordinal}")
        ordinal += 1
    path.replace(target)
    record = {
        "schema": "detached_local_process_receipt_arm_tombstone.v1",
        "generated_utc": _utc_now(),
        "reason": "explicit_stale_reservation_supersede",
        "original_path": str(path),
        "tombstone_path": str(target),
        "receipt_arm_sha256": digest,
    }
    _write_json(target.with_name(target.name + ".tombstone.json"), record)
    return record


def _release_receipt_reservation(path: Path | None, counter: int) -> None:
    if path is None:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("monotonic_launch_counter")) != int(counter):
            return
        path.unlink()
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return


def _path_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        # If contents cannot be inspected, the path cannot be certified fresh.
        return True
    return True


def _resolve_fresh_roots(raw_roots: list[Path], suffix: bool) -> tuple[dict[str, str], list[dict[str, Any]]]:
    roots = [path.expanduser().resolve(strict=False) for path in raw_roots]
    if len(set(roots)) != len(roots):
        raise LaunchRefusal("duplicate --fresh-root path")
    for left in roots:
        for right in roots:
            if left != right and (left in right.parents or right in left.parents):
                raise LaunchRefusal("nested --fresh-root paths are ambiguous", left=str(left), right=str(right))
    mapping: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    stamp = _utc_compact()
    for raw_root, root in zip(raw_roots, roots, strict=True):
        dirty = _path_nonempty(root)
        effective = root
        state = "absent" if not root.exists() else "clean_existing"
        if dirty:
            if not suffix:
                raise LaunchRefusal(
                    "fresh root exists and is non-empty",
                    rc=7,
                    fresh_root=str(root),
                    cure="choose a new root or pass --fresh-root-suffix; existing bytes are never deleted",
                )
            effective = root.with_name(f"{root.name}_{stamp}")
            ordinal = 1
            while effective.exists():
                effective = root.with_name(f"{root.name}_{stamp}_{ordinal:02d}")
                ordinal += 1
            state = "minted_from_nonempty"
        mapping[str(root)] = str(effective)
        raw_text = str(raw_root.expanduser())
        if raw_text != str(root):
            mapping[raw_text] = str(effective)
        rows.append(
            {"requested_path": str(root), "effective_path": str(effective), "state": state}
        )
    return mapping, rows


def _rewrite_value(value: str, mapping: Mapping[str, str]) -> str:
    result = value
    # Longest path first if a future caller relaxes the nested-root refusal.
    for old in sorted(mapping, key=len, reverse=True):
        new = mapping[old]
        if old != new and old in result:
            if old.startswith(os.sep):
                pattern = re.escape(old) + r"(?=$|/)"
                result = re.sub(pattern, lambda _match, replacement=new: replacement, result)
            else:
                pattern = r"(^|=)" + re.escape(old) + r"(?=$|/)"
                result = re.sub(
                    pattern,
                    lambda match, replacement=new: match.group(1) + replacement,
                    result,
                )
    return result


def _rewrite_path(path: Path, mapping: Mapping[str, str]) -> Path:
    return Path(_rewrite_value(str(path.expanduser().resolve(strict=False)), mapping))


def _launch_identity(manifest_path: Path, pid: int, counter: int) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "pid": int(pid),
        "monotonic_launch_counter": int(counter),
    }


def _done_receipt(
    *,
    identity: Mapping[str, Any],
    receipt_name: str,
    rc: int,
    elapsed_s: float,
    adjudicated_at_launch: bool,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "schema": DONE_RECEIPT_SCHEMA,
        "generated_utc": _utc_now(),
        "receipt_name": receipt_name,
        "launch_id": dict(identity),
        "rc": int(rc),
        "elapsed_s": round(float(elapsed_s), 6),
        "detail": detail,
        "adjudicated_at_launch": bool(adjudicated_at_launch),
    }


def _supervisor_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start-gate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--counter", required=True, type=int)
    parser.add_argument("--done", type=Path)
    parser.add_argument("--armed", type=Path)
    parser.add_argument("--receipt-name", default="")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        return 126

    started = time.time()
    caught_signal: int | None = None
    child: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame: Any) -> None:
        nonlocal caught_signal
        caught_signal = signum
        if child is not None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        try:
            signal.signal(getattr(signal, name), forward)
        except (AttributeError, OSError, ValueError):
            pass
    for name in ("SIGURG", "SIGPIPE"):
        try:
            signal.signal(getattr(signal, name), signal.SIG_IGN)
        except (AttributeError, OSError, ValueError):
            pass
    deadline = time.monotonic() + 30.0
    while not args.start_gate.exists() and caught_signal is None:
        if time.monotonic() >= deadline:
            caught_signal = signal.SIGTERM
            break
        time.sleep(0.02)

    detail = ""
    if caught_signal is not None:
        rc = 128 + int(caught_signal)
        detail = f"signal_before_exec={caught_signal}"
    else:
        try:
            child = subprocess.Popen(args.cmd)
            rc = int(child.wait())
        except FileNotFoundError as exc:
            rc = 127
            detail = f"exec_error=FileNotFoundError:{exc.filename}"
        except OSError as exc:
            rc = 126
            detail = f"exec_error={type(exc).__name__}:{exc.errno}"
    if args.done is not None:
        identity = _launch_identity(args.manifest, os.getpid(), args.counter)
        _write_json(
            args.done,
            _done_receipt(
                identity=identity,
                receipt_name=args.receipt_name,
                rc=rc,
                elapsed_s=time.time() - started,
                adjudicated_at_launch=False,
                detail=detail,
            ),
        )
        _release_receipt_reservation(args.armed, args.counter)
    return rc


def _apply_and_verify_nice(pid: int, requested: int) -> int:
    try:
        os.setpriority(os.PRIO_PROCESS, pid, requested)
        actual = os.getpriority(os.PRIO_PROCESS, pid)
    except (OSError, PermissionError) as exc:
        raise LaunchRefusal(
            "requested niceness could not be applied",
            rc=8,
            pid=pid,
            requested_nice=requested,
            error=f"{type(exc).__name__}: {exc}",
        ) from exc
    if actual != requested:
        raise LaunchRefusal(
            "requested niceness did not verify",
            rc=8,
            pid=pid,
            requested_nice=requested,
            actual_nice=actual,
        )
    return int(actual)


def _stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _adjudicate_receipt(
    done_path: Path | None,
    *,
    identity: Mapping[str, Any],
    receipt_name: str,
    rc: int,
    elapsed_s: float,
    detail: str,
) -> None:
    if done_path is None:
        return
    payload: dict[str, Any]
    try:
        loaded = json.loads(done_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload.update(
        _done_receipt(
            identity=identity,
            receipt_name=receipt_name,
            rc=rc,
            elapsed_s=elapsed_s,
            adjudicated_at_launch=True,
            detail=detail,
        )
    )
    _write_json(done_path, payload)


def _validate_watcher(tool: Path, config: Path, cwd: Path, env: Mapping[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, str(tool), "--config", str(config), "--validate-only"],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise LaunchRefusal(
            "watcher config validation failed",
            rc=9,
            tool=str(tool),
            config=str(config),
            stderr=result.stderr[-2000:],
        )


def _safe_run_wrapper_flag(effective_cmd: Sequence[str], flag: str) -> str | None:
    """Read one launcher-injected safe_run flag from the argv the child receives.

    Scans ONLY the wrapper's own flag region — everything before the first bare
    ``--`` separator — so a same-named flag inside the WRAPPED user command can
    never be mistaken for a value the launcher owns.
    """
    value: str | None = None
    for index, part in enumerate(effective_cmd):
        if part == "--":
            break
        if part == flag and index + 1 < len(effective_cmd):
            value = effective_cmd[index + 1]
    return value


def _sweep_superseded_values(
    node: Any,
    corrections: Mapping[str, str],
    trail: str,
    found: list[dict[str, str]],
) -> Any:
    """Apply an already-proven path correction to EVERY occurrence of that value.

    The key-name corrections above fix ``pid_file`` and ``log_path`` because those
    are the names the launcher knows.  A config is free to carry the SAME drifted
    path under any other key -- ``success_receipts[].path`` is the live one, and
    it is read by the watcher exactly like the keys that were just corrected.
    Fixing by key name alone therefore repairs one holder of a wrong value and
    leaves its twin untouched, in the same file, while announcing success.

    So once a value is PROVEN wrong (the launcher derived the right one), sweep
    it by VALUE across the whole document.  Every replacement is recorded with
    its JSON path so the supersession stays as loud as a key-name one.
    """
    if isinstance(node, dict):
        return {
            key: _sweep_superseded_values(value, corrections, f"{trail}.{key}", found)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            _sweep_superseded_values(item, corrections, f"{trail}[{index}]", found)
            for index, item in enumerate(node)
        ]
    if isinstance(node, str) and node in corrections:
        found.append({"key": trail.lstrip("."), "declared": node, "derived": corrections[node]})
        return corrections[node]
    return node


def _derive_watcher_config(
    config_path: Path,
    *,
    kind: str,
    effective_cmd: Sequence[str],
    resource_budget: Mapping[str, Any],
    out: Path,
    log_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Derive a watcher's run-specific paths from the launch itself.

    Both watcher configs carry values that MUST agree with the launch:
    ``pid_file`` (the pid the watcher polls, BOTH kinds), ``success_receipts``
    (liveness: how a clean rc=0 exit is told apart from a silent death), and
    ``log_path`` (quality: the stdout log the poller parses).  All were
    hand-typed into the config while the launcher independently passed or
    created the same paths, so the pair could drift.  It did, on 2026-08-16:

      * ``ddm_ra2c_rank4`` — the config's ``pid_file`` carried a ``launcher/``
        path component that run's layout did not have, so the watcher published
        ``pidfile_missing_or_invalid`` 303 s into a run that finished rc=0.
      * ``ddm_ra2c_alpha_ladder_a0`` — the liveness config declared no
        ``success_receipts``, so the watcher published ``child_dead`` 9 s after
        a clean rc=0 exit, with the rc=0 safe_run receipt sitting unread in the
        same directory as the pidfile.
      * BOTH runs' QUALITY configs pointed ``log_path`` at a ``stdout.log`` that
        never existed (the launcher writes ``run.log``), and rank4's quality
        ``pid_file`` carried the same ``launcher/`` drift.  A quality poller
        whose pidfile is missing reads ``_pid_alive`` false and returns 0 after
        its startup grace — a SILENT clean exit that writes no event receipt.
        It watched nothing, and looked exactly like health.

    The launcher already holds every one of these values: it wrote them into
    the argv the child receives, or it created the file itself.  Derive them,
    write an EFFECTIVE config beside the watcher logs (never mutate the
    caller's file), and return it with a record of what was derived.

    Fail-closed policy, stated explicitly because silence here IS the defect:

      * derived value equals the declared one -> ``argv_confirmed``.
      * derived value differs                 -> ``argv_superseded``; the argv
        wins (it is what safe_run will actually honor), the declared value is
        preserved in the record, and the supersession is announced on stderr AT
        LAUNCH rather than discovered at review.
      * no safe_run wrapper, or the flag absent from it -> ``config_declared``;
        fall back to the hand-typed value.  The launcher does NOT invent a path
        it does not own.  ``pid_file`` remains mandatory in the watcher's own
        ``load_config``, so an absent value still fails closed there.
      * a value proven wrong under one key is swept by VALUE across the whole
        document -> ``value_swept``.  Correcting by key NAME alone repairs the
        holder the launcher happens to know and leaves an identical wrong value
        under any other key -- ``success_receipts[].path`` above is handed back
        as ``config_declared`` without its path ever being inspected, so the
        same drifted ``stdout.log`` survived in the very file where ``log_path``
        had just been corrected for it.  One wrong path, one correction,
        everywhere it appears.

    Deriving from the argv rather than from ``resource_budget`` is deliberate:
    the argv is the thing the child obeys.  The budget record is cross-checked
    against it and a disagreement REFUSES the launch, because that would mean
    the launcher's own two accounts of the same path had diverged.
    """
    record: dict[str, Any] = {
        "schema": "detached_local_process_watcher_derivation.v1",
        "kind": kind,
        "declared_config": str(config_path),
        "effective_config": None,
        "pid_file": {"source": "config_declared", "value": None, "declared": None},
        "success_receipts": {"source": "config_declared", "value": None},
        "log_path": {"source": "config_declared", "value": None},
        "supersessions": [],
    }
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Not the launcher's file to adjudicate; the watcher's own
        # --validate-only pass refuses an unreadable config loudly.
        record["error"] = f"{type(exc).__name__}: {exc}"
        return config_path, record
    if not isinstance(config, dict):
        record["error"] = "config is not a JSON object"
        return config_path, record

    wrapped = str(resource_budget.get("mode")) == "derived_and_enforced"
    derived_pidfile = _safe_run_wrapper_flag(effective_cmd, "--child-pidfile") if wrapped else None
    derived_receipt = _safe_run_wrapper_flag(effective_cmd, "--status-receipt") if wrapped else None
    for flag, argv_value, budget_key in (
        ("--child-pidfile", derived_pidfile, "child_pidfile"),
        ("--status-receipt", derived_receipt, "status_receipt"),
    ):
        budget_value = resource_budget.get(budget_key)
        if wrapped and budget_value is not None and argv_value != str(budget_value):
            raise LaunchRefusal(
                "launcher argv and resource budget disagree about a safe_run path",
                rc=10,
                flag=flag,
                argv_value=argv_value,
                resource_budget_value=str(budget_value),
            )

    changed = False
    declared_pidfile = config.get("pid_file")
    record["pid_file"]["declared"] = declared_pidfile
    record["pid_file"]["value"] = declared_pidfile
    if derived_pidfile is not None:
        record["pid_file"]["value"] = derived_pidfile
        if declared_pidfile == derived_pidfile:
            record["pid_file"]["source"] = "argv_confirmed"
        else:
            record["pid_file"]["source"] = "argv_superseded"
            record["supersessions"].append(
                {"key": "pid_file", "declared": declared_pidfile, "derived": derived_pidfile}
            )
            config["pid_file"] = derived_pidfile
            changed = True

    if "success_receipts" in config or kind == "liveness":
        if config.get("success_receipts"):
            record["success_receipts"]["source"] = "config_declared"
            record["success_receipts"]["value"] = config["success_receipts"]
        elif derived_receipt is not None:
            config["success_receipts"] = [{"label": "safe_run_status", "path": derived_receipt}]
            config.setdefault("success_settle_s", 90)
            record["success_receipts"]["source"] = "argv_derived"
            record["success_receipts"]["value"] = config["success_receipts"]
            changed = True

    # The quality poller parses the run's stdout; the launcher CREATED that file,
    # so a declared path that disagrees is drift by construction. A poller aimed
    # at a log that does not exist never alarms and never says so.
    if "log_path" in config:
        declared_log = config.get("log_path")
        record["log_path"]["value"] = str(log_path)
        if declared_log == str(log_path):
            record["log_path"]["source"] = "launch_confirmed"
        else:
            record["log_path"]["source"] = "launch_superseded"
            record["supersessions"].append(
                {"key": "log_path", "declared": declared_log, "derived": str(log_path)}
            )
            config["log_path"] = str(log_path)
            changed = True

    # Correct by VALUE, not only by key name.  Anything the launcher just proved
    # wrong under a known key is equally wrong wherever else it appears -- most
    # concretely inside ``success_receipts[].path``, which the branch above hands
    # back as ``config_declared`` without ever looking at the path it contains.
    corrections = {
        str(row["declared"]): str(row["derived"])
        for row in record["supersessions"]
        if isinstance(row.get("declared"), str) and row["declared"] != row["derived"]
    }
    if corrections:
        swept: list[dict[str, str]] = []
        config = _sweep_superseded_values(config, corrections, "", swept)
        for row in swept:
            record["supersessions"].append({**row, "source": "value_swept"})
            changed = True
        if swept:
            record["value_sweep"] = swept
            # The record snapshots taken above predate the sweep; refresh them so
            # the manifest reports what the watcher will actually read.
            if config.get("success_receipts"):
                record["success_receipts"]["value"] = config["success_receipts"]
            if "log_path" in config:
                record["log_path"]["value"] = config["log_path"]
            if config.get("pid_file") is not None:
                record["pid_file"]["value"] = config["pid_file"]

    if not changed:
        return config_path, record

    config["derived_by"] = record["schema"]
    effective = out / "watchers" / f"{kind}_config_effective.json"
    effective.parent.mkdir(parents=True, exist_ok=True)
    effective.write_text(json.dumps(config, indent=1), encoding="utf-8")
    record["effective_config"] = str(effective)
    for row in record["supersessions"]:
        # Loud AT LAUNCH: a drifted hand-typed path is exactly the silence this
        # cure exists to end.  Do not downgrade this to a manifest-only note.
        print(
            json.dumps(
                {
                    "watcher_config_superseded": {"kind": kind, **row},
                    "effective_config": str(effective),
                }
            ),
            file=sys.stderr,
        )
    return effective, record


def _arm_watchers(
    *,
    out: Path,
    cwd: Path,
    env: Mapping[str, str],
    liveness_config: Path,
    quality_config: Path,
    launch_manifest: Path,
    watcher_start_gate: Path,
    launch_counter: int,
    runs: Path,
) -> list[dict[str, Any]]:
    repo = Path(__file__).resolve().parents[1]
    specs = (
        ("liveness", repo / "tools" / "run_liveness_watcher.py", liveness_config),
        ("quality", repo / "tools" / "run_quality_poller.py", quality_config),
    )
    watcher_dir = out / "watchers"
    watcher_dir.mkdir(parents=True, exist_ok=True)
    armed: list[dict[str, Any]] = []
    processes: list[tuple[str, subprocess.Popen[Any], Path]] = []
    for label, tool, config in specs:
        log_path = watcher_dir / f"{label}.log"
        event_receipt = runs / f"watched_launch_{launch_counter}_{label}.done"
        with log_path.open("ab", buffering=0) as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(tool),
                    "--config",
                    str(config),
                    "--event-receipt",
                    str(event_receipt),
                    "--launch-manifest",
                    str(launch_manifest),
                    "--start-gate",
                    str(watcher_start_gate),
                ],
                cwd=cwd,
                env=dict(env),
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        time.sleep(0.05)
        rc = proc.poll()
        if rc is not None:
            for _armed_label, armed_proc, _armed_log in processes:
                try:
                    armed_proc.terminate()
                except ProcessLookupError:
                    continue
            raise LaunchRefusal(
                "watcher exited while arming",
                rc=9,
                watcher=label,
                watcher_rc=rc,
                log_path=str(log_path),
                last_log_lines=_tail_lines(log_path),
            )
        processes.append((label, proc, log_path))
        armed.append(
            {
                "kind": label,
                "pid": int(proc.pid),
                "config_path": str(config),
                "log_path": str(log_path),
                "tool_path": str(tool),
                "event_receipt_path": str(event_receipt),
            }
        )
    time.sleep(0.1)
    for label, proc, log_path in processes:
        rc = proc.poll()
        if rc is None:
            continue
        for _armed_label, armed_proc, _armed_log in processes:
            if armed_proc.poll() is None:
                armed_proc.terminate()
        raise LaunchRefusal(
            "watcher exited while arming",
            rc=9,
            watcher=label,
            watcher_rc=rc,
            log_path=str(log_path),
            last_log_lines=_tail_lines(log_path),
        )
    _write_json(watcher_dir / "watchers_manifest.json", {"schema": "watched_launch.watchers.v1", "watchers": armed})
    return armed


def _stop_watchers(rows: list[dict[str, Any]]) -> None:
    """Stop newly armed watchers when launch adjudication stays synchronous."""

    for row in rows:
        try:
            os.kill(int(row["pid"]), signal.SIGTERM)
            row["stopped_at_launch"] = True
        except (KeyError, TypeError, ValueError, ProcessLookupError):
            row["stopped_at_launch"] = True
        except PermissionError:
            row["stopped_at_launch"] = False


def _derive_resource_budget(
    args: argparse.Namespace,
    *,
    out: Path,
    cmd: list[str],
    env: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    """Build a real safe_run envelope from measured demand and host policy."""

    if not args.derive_resource_budgets:
        return cmd, {
            "mode": "child_owned",
            "note": "launcher did not invent an RSS or thread limit without measured demand",
        }
    if "safe_run.py" in " ".join(cmd):
        raise LaunchRefusal(
            "--derive-resource-budgets cannot wrap an argv that already owns safe_run",
            rc=10,
        )
    try:
        import system_memory_governor as memory_governor

        operator_ceiling_gib = float(memory_governor.operator_ceiling_gib())
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise LaunchRefusal(
            "cannot resolve the canonical operator memory ceiling",
            rc=10,
            error=f"{type(exc).__name__}: {exc}",
        ) from exc
    projected_peak_gib = float(args.measured_peak_rss_gib)
    if operator_ceiling_gib <= 0 or projected_peak_gib > operator_ceiling_gib:
        raise LaunchRefusal(
            "measured peak does not fit the canonical operator ceiling",
            rc=10,
            measured_peak_rss_gib=projected_peak_gib,
            operator_ceiling_gib=operator_ceiling_gib,
        )
    logical_cpus = max(1, os.cpu_count() or 1)
    thread_budget = min(int(args.measured_thread_need), logical_cpus)
    for key in _THREAD_ENV_KEYS:
        env[key] = str(thread_budget)
    rss_cap_mib = int(operator_ceiling_gib * 1024)
    safe_run = Path(__file__).resolve().with_name("safe_run.py")
    status_receipt = out / "resource_safe_run_status.json"
    child_pidfile = out / "resource_safe_run_child.pid"
    wrapped = [
        sys.executable,
        str(safe_run),
        "--rss-mb",
        str(rss_cap_mib),
        "--projected-gib",
        str(projected_peak_gib),
        "--timeout",
        str(float(args.walltime_cap_s)),
        "--status-receipt",
        str(status_receipt),
        "--child-pidfile",
        str(child_pidfile),
        "--quiet",
        "--",
        *cmd,
    ]
    return wrapped, {
        "mode": "derived_and_enforced",
        "measured_peak_rss_gib": projected_peak_gib,
        "operator_ceiling_gib": operator_ceiling_gib,
        "rss_cap_mib": rss_cap_mib,
        "measured_thread_need": int(args.measured_thread_need),
        "logical_cpus": logical_cpus,
        "thread_budget": thread_budget,
        "thread_environment": {key: env[key] for key in _THREAD_ENV_KEYS},
        "walltime_cap_s": float(args.walltime_cap_s),
        "status_receipt": str(status_receipt),
        "child_pidfile": str(child_pidfile),
        "concurrency_policy": "niceness plus system admission; no artificial abstinence",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Resource policy: new heavy jobs should use --derive-resource-budgets with a "
            "measured peak, measured thread need, and wall cap. The launcher then enforces "
            "the canonical host ceiling (116 GiB by current governor policy) through safe_run "
            "and derives threads from measured need and available CPUs. It never copies the "
            "old 16384 MiB wrapper literal. Concurrency is handled by --nice plus the system "
            "admission gate, not by leaving the machine idle."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cwd", default=".", type=Path)
    parser.add_argument("--purpose", default="detached local long run")
    parser.add_argument(
        "--authority", default="local detached execution; downstream artifacts decide authority"
    )
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--fresh-root", action="append", default=[], type=Path, metavar="PATH")
    parser.add_argument(
        "--fresh-root-suffix",
        action="store_true",
        help="If a declared fresh root is nonempty, mint PATH_<utc> and rewrite declared paths.",
    )
    parser.add_argument("--nice", type=int, default=None, metavar="N")
    parser.add_argument("--derive-resource-budgets", action="store_true")
    parser.add_argument("--measured-peak-rss-gib", type=float)
    parser.add_argument("--measured-thread-need", type=int)
    parser.add_argument("--walltime-cap-s", type=float)
    parser.add_argument("--done-receipt", default=None, metavar="NAME")
    parser.add_argument(
        "--receipt-supersede",
        action="store_true",
        help="Tombstone an unconsumed prior receipt before arming this launch.",
    )
    parser.add_argument("--arm-watchers", action="store_true")
    parser.add_argument("--liveness-config", type=Path)
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--verify-alive-secs", type=float, default=3.0)
    parser.add_argument("--allow-reaper-name-match", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("command argv is required after --")
    for item in args.env:
        if "=" not in item:
            parser.error(f"--env must be KEY=VALUE, got {item!r}")
    if args.fresh_root_suffix and not args.fresh_root:
        parser.error("--fresh-root-suffix requires at least one --fresh-root")
    if args.receipt_supersede and not args.done_receipt:
        parser.error("--receipt-supersede requires --done-receipt")
    if args.done_receipt and not _RECEIPT_NAME.fullmatch(args.done_receipt):
        parser.error("--done-receipt must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    if args.nice is not None and not -20 <= args.nice <= 20:
        parser.error("--nice must be between -20 and 20")
    resource_values = (
        args.measured_peak_rss_gib,
        args.measured_thread_need,
        args.walltime_cap_s,
    )
    if args.derive_resource_budgets and any(value is None for value in resource_values):
        parser.error(
            "--derive-resource-budgets requires --measured-peak-rss-gib, "
            "--measured-thread-need, and --walltime-cap-s"
        )
    if not args.derive_resource_budgets and any(value is not None for value in resource_values):
        parser.error("resource measurement flags require --derive-resource-budgets")
    if args.derive_resource_budgets and (
        args.measured_peak_rss_gib <= 0
        or args.measured_thread_need <= 0
        or args.walltime_cap_s <= 0
    ):
        parser.error("resource measurements and caps must be positive")
    if args.arm_watchers and (args.liveness_config is None or args.quality_config is None):
        parser.error("--arm-watchers requires --liveness-config and --quality-config")
    if args.verify_alive_secs < 0:
        parser.error("--verify-alive-secs must be nonnegative")
    return args


def _print_refusal(exc: LaunchRefusal, **context: Any) -> int:
    print(json.dumps({"error": str(exc), **exc.detail, **context}, sort_keys=True), file=sys.stderr)
    return exc.rc


def main() -> int:
    args = parse_args()
    try:
        mapping, fresh_rows = _resolve_fresh_roots(args.fresh_root, args.fresh_root_suffix)
    except LaunchRefusal as exc:
        return _print_refusal(exc)
    out = _rewrite_path(args.output_dir, mapping)
    cwd = _rewrite_path(args.cwd, mapping)
    cmd = [_rewrite_value(str(part), mapping) for part in args.cmd]
    env_items = [_rewrite_value(item, mapping) for item in args.env]

    exe = cmd[0]
    if "/" in exe and not Path(exe).expanduser().exists():
        return _print_refusal(LaunchRefusal(f"executable not found: {exe}"))
    for part in cmd[1:]:
        if part.endswith(".py"):
            candidate = Path(part) if Path(part).is_absolute() else cwd / part
            if not candidate.exists():
                return _print_refusal(LaunchRefusal(f"script not found at cwd: {candidate}"))
            break
    reaper_hits = sorted({m.group(0) for part in cmd for m in _REAPER_NAME_PREDICATE.finditer(part)})
    if reaper_hits and not args.allow_reaper_name_match:
        return _print_refusal(
            LaunchRefusal(
                "argv matches the fleet reaper predicate; child would be SIGTERMed at ~300-360s",
                rc=5,
                matched_tokens=reaper_hits,
                matching_argv_parts=[part for part in cmd if _REAPER_NAME_PREDICATE.search(part)],
                cure="move script/output paths outside standalone claude/codex path components",
            )
        )

    env = os.environ.copy()
    env.update(dict(item.split("=", 1) for item in env_items))
    try:
        effective_cmd, resource_budget = _derive_resource_budget(
            args,
            out=out,
            cmd=cmd,
            env=env,
        )
    except LaunchRefusal as exc:
        return _print_refusal(exc)
    liveness_config = _rewrite_path(args.liveness_config, mapping) if args.liveness_config else None
    quality_config = _rewrite_path(args.quality_config, mapping) if args.quality_config else None
    watcher_derivations: dict[str, Any] = {}
    # effective_cmd, NOT cmd: --child-pidfile and --status-receipt are injected
    # by _derive_resource_budget and appear ONLY in the wrapped argv.  Passing
    # the raw cmd made this derivation silently inert.
    try:
        if liveness_config is not None:
            liveness_config, watcher_derivations["liveness"] = _derive_watcher_config(
                liveness_config,
                kind="liveness",
                effective_cmd=effective_cmd,
                resource_budget=resource_budget,
                out=out,
                log_path=out / "run.log",
            )
        if quality_config is not None:
            quality_config, watcher_derivations["quality"] = _derive_watcher_config(
                quality_config,
                kind="quality",
                effective_cmd=effective_cmd,
                resource_budget=resource_budget,
                out=out,
                log_path=out / "run.log",
            )
    except LaunchRefusal as exc:
        return _print_refusal(exc)
    if args.arm_watchers:
        repo = Path(__file__).resolve().parents[1]
        try:
            _validate_watcher(repo / "tools" / "run_liveness_watcher.py", liveness_config, cwd, env)
            _validate_watcher(repo / "tools" / "run_quality_poller.py", quality_config, cwd, env)
        except LaunchRefusal as exc:
            return _print_refusal(exc)

    runs = _runs_dir()
    done_path = runs / f"{args.done_receipt}.done" if args.done_receipt else None
    receipt_arm_path = _armed_path(done_path) if done_path is not None else None
    receipt_tombstone: dict[str, Any] | None = None
    receipt_arm_tombstone: dict[str, Any] | None = None
    if receipt_arm_path is not None and receipt_arm_path.exists():
        active = _receipt_reservation_active(receipt_arm_path)
        if active or not args.receipt_supersede:
            return _print_refusal(
                LaunchRefusal(
                    "done receipt name already has an active launch reservation"
                    if active
                    else "done receipt name has a stale launch reservation",
                    rc=6,
                    receipt_arm_path=str(receipt_arm_path),
                    cure=(
                        "wait for the active launch to finish"
                        if active
                        else "pass --receipt-supersede to preserve and replace the stale reservation"
                    ),
                )
            )
        receipt_arm_tombstone = _tombstone_reservation(
            receipt_arm_path,
            dry_run=bool(args.dry_run),
        )
    if done_path is not None and done_path.exists():
        consumed = _receipt_is_consumed(done_path)
        if not consumed and not args.receipt_supersede:
            return _print_refusal(
                LaunchRefusal(
                    "done receipt name is already armed and unconsumed",
                    rc=6,
                    done_receipt_path=str(done_path),
                    cure="let the fleet monitor consume it or pass --receipt-supersede to preserve and replace it",
                )
            )
        if args.dry_run:
            receipt_tombstone = {
                "dry_run_would_tombstone": str(done_path),
                "reason": "explicit_unconsumed_supersede" if not consumed else "consumed_name_reuse",
                "receipt_sha256": _sha256(done_path),
            }
        else:
            receipt_tombstone = _tombstone_receipt(
                done_path,
                reason="explicit_unconsumed_supersede" if not consumed else "consumed_name_reuse",
            )

    out.mkdir(parents=True, exist_ok=True)
    for row in fresh_rows:
        Path(row["effective_path"]).mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    pid_path = out / "run.pid"
    manifest_path = out / "launch_manifest.json"
    start_gate = out / ".launch_start_gate"
    watcher_start_gate = out / ".watchers_start_gate"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": _utc_now(),
        "cwd": str(cwd),
        "git_sha": _git_sha(cwd),
        "purpose": str(args.purpose),
        "authority": str(args.authority),
        "detach_method": "gated supervisor + subprocess.Popen(start_new_session=True)",
        "output_dir": str(out),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "argv": cmd,
        "effective_argv": effective_cmd,
        "dry_run": bool(args.dry_run),
        "reaper_predicate_hits": reaper_hits,
        "reaper_name_match_allowed": bool(args.allow_reaper_name_match),
        "fresh_roots": fresh_rows,
        "requested_nice": args.nice,
        "actual_nice": None,
        "resource_budget": resource_budget,
        "done_receipt_path": str(done_path) if done_path else None,
        "receipt_tombstone": receipt_tombstone,
        "receipt_arm_tombstone": receipt_arm_tombstone,
        "watchers_requested": bool(args.arm_watchers),
        "watcher_config_derivation": watcher_derivations or None,
        "watchers": [],
    }
    if args.dry_run:
        _write_json(manifest_path, payload)
        print(json.dumps({"dry_run": True, "manifest_path": str(manifest_path), "fresh_roots": fresh_rows}))
        return 0

    counter = _next_launch_counter(runs)
    if receipt_arm_path is not None:
        try:
            _reserve_receipt(
                receipt_arm_path,
                receipt_name=args.done_receipt,
                counter=counter,
            )
        except LaunchRefusal as exc:
            return _print_refusal(exc)
    supervisor_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_supervise",
        "--start-gate",
        str(start_gate),
        "--manifest",
        str(manifest_path),
        "--counter",
        str(counter),
    ]
    if done_path is not None:
        supervisor_argv.extend(
            [
                "--done",
                str(done_path),
                "--armed",
                str(receipt_arm_path),
                "--receipt-name",
                args.done_receipt,
            ]
        )
    supervisor_argv.extend(["--", *effective_cmd])
    started = time.time()
    try:
        with log_path.open("ab", buffering=0) as log:
            proc = subprocess.Popen(
                supervisor_argv,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        _release_receipt_reservation(receipt_arm_path, counter)
        payload["adjudicated_at_launch"] = True
        payload["launch_error"] = {
            "error": "supervisor spawn failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        _write_json(manifest_path, payload)
        return _print_refusal(
            LaunchRefusal(
                "supervisor spawn failed",
                rc=2,
                error=f"{type(exc).__name__}: {exc}",
            ),
            manifest_path=str(manifest_path),
        )
    identity = _launch_identity(manifest_path, proc.pid, counter)
    try:
        _update_receipt_reservation(receipt_arm_path, identity=identity)
    except (OSError, json.JSONDecodeError, LaunchRefusal) as exc:
        _stop_process(proc)
        _adjudicate_receipt(
            done_path,
            identity=identity,
            receipt_name=args.done_receipt or "",
            rc=6,
            elapsed_s=time.time() - started,
            detail="receipt reservation identity update failed",
        )
        _release_receipt_reservation(receipt_arm_path, counter)
        payload["adjudicated_at_launch"] = True
        payload["launch_error"] = {
            "error": "receipt reservation identity update failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        _write_json(manifest_path, payload)
        return _print_refusal(
            LaunchRefusal(
                "receipt reservation identity update failed",
                rc=6,
                error=f"{type(exc).__name__}: {exc}",
            ),
            manifest_path=str(manifest_path),
            pid=proc.pid,
        )
    payload["pid"] = int(proc.pid)
    payload["launch_id"] = identity
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    try:
        if args.nice is not None:
            payload["actual_nice"] = _apply_and_verify_nice(proc.pid, args.nice)
        else:
            try:
                payload["actual_nice"] = int(os.getpriority(os.PRIO_PROCESS, proc.pid))
            except OSError:
                payload["actual_nice"] = None
        # Watchers receive this manifest as their launch-identity authority, so
        # publish it before they are allowed to start.
        _write_json(manifest_path, payload)
        if args.arm_watchers:
            payload["watchers"] = _arm_watchers(
                out=out,
                cwd=cwd,
                env=env,
                liveness_config=liveness_config,
                quality_config=quality_config,
                launch_manifest=manifest_path,
                watcher_start_gate=watcher_start_gate,
                launch_counter=counter,
                runs=runs,
            )
        _write_json(manifest_path, payload)
        _write_json(start_gate, {"schema": "detached_local_process_start_gate.v1", "launch_id": identity})
    except LaunchRefusal as exc:
        _stop_process(proc)
        payload["adjudicated_at_launch"] = True
        payload["launch_error"] = {"error": str(exc), **exc.detail}
        _write_json(manifest_path, payload)
        _adjudicate_receipt(
            done_path,
            identity=identity,
            receipt_name=args.done_receipt or "",
            rc=exc.rc,
            elapsed_s=time.time() - started,
            detail=str(exc),
        )
        _release_receipt_reservation(receipt_arm_path, counter)
        return _print_refusal(exc, manifest_path=str(manifest_path), pid=proc.pid)

    if args.verify_alive_secs > 0:
        deadline = time.monotonic() + float(args.verify_alive_secs)
        rc: int | None = None
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if rc is not None:
            _stop_watchers(payload["watchers"])
            payload["adjudicated_at_launch"] = True
            payload["verify_alive_rc"] = int(rc)
            _write_json(manifest_path, payload)
            _adjudicate_receipt(
                done_path,
                identity=identity,
                receipt_name=args.done_receipt or "",
                rc=int(rc),
                elapsed_s=time.time() - started,
                detail="detached child exited during verify-alive window",
            )
            _release_receipt_reservation(receipt_arm_path, counter)
            error = {
                "error": "detached child exited during verify-alive window",
                "pid": int(proc.pid),
                "rc": int(rc),
                "verify_alive_secs": float(args.verify_alive_secs),
                "output_dir": str(out),
                "manifest_path": str(manifest_path),
                "log_path": str(log_path),
                "last_log_lines": _tail_lines(log_path),
            }
            print(json.dumps(error, sort_keys=True), file=sys.stderr)
            return int(rc) if 1 <= int(rc) <= 125 else 4
    _write_json(
        watcher_start_gate,
        {"schema": "detached_local_process_watchers_start_gate.v1", "launch_id": identity},
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "pid": int(proc.pid),
                "launch_id": identity,
                "output_dir": str(out),
                "manifest_path": str(manifest_path),
                "log_path": str(log_path),
                "fresh_roots": fresh_rows,
                "watchers": payload["watchers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_supervise":
        raise SystemExit(_supervisor_main(sys.argv[2:]))
    raise SystemExit(main())
