#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Watch one detached run for silent death or stalled durable evidence.

The watcher is intentionally scorer-free.  It reads a PID file plus configured
mtime surfaces and publishes exactly one atomic JSON alert before exiting.
Configuration is JSON so the same executable can watch trainers, profilers, and
archive builders without embedding run-specific paths in source.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "pact.run_liveness_watcher.v1"
CONFIG_SCHEMA = "pact.run_liveness_watcher.config.v1"
ALERT_SCHEMA = "pact.run_liveness_watcher.alert.v1"


class ConfigError(ValueError):
    """The watcher config cannot support a real liveness decision."""


def _ignore_harness_signals() -> None:
    for name in ("SIGURG", "SIGPIPE", "SIGHUP"):
        try:
            signal.signal(getattr(signal, name), signal.SIG_IGN)
        except (AttributeError, OSError, ValueError):
            pass


def _positive_number(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be {relation}")
    return number


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a nonempty path string")
    return Path(value).expanduser().resolve(strict=False)


def _check_list(config: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = config.get(key, [])
    if not isinstance(rows, list):
        raise ConfigError(f"{key} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigError(f"{key}[{index}] must be an object")
        normalized.append(
            {
                "path": _path(row.get("path"), f"{key}[{index}].path"),
                "max_age_s": _positive_number(
                    row.get("max_age_s"), f"{key}[{index}].max_age_s"
                ),
                "grace_s": _positive_number(
                    row.get("grace_s", 0),
                    f"{key}[{index}].grace_s",
                    allow_zero=True,
                ),
                "label": str(row.get("label") or f"{key}[{index}]"),
            }
        )
    return normalized


def _success_receipt_list(config: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = config.get(key, [])
    if not isinstance(rows, list):
        raise ConfigError(f"{key} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigError(f"{key}[{index}] must be an object")
        normalized.append(
            {
                "path": _path(row.get("path"), f"{key}[{index}].path"),
                "label": str(row.get("label") or f"{key}[{index}]"),
            }
        )
    return normalized


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"config schema must be {CONFIG_SCHEMA}")
    config = {
        "schema": CONFIG_SCHEMA,
        "pid_file": _path(raw.get("pid_file"), "pid_file"),
        "alert_path": _path(raw.get("alert_path"), "alert_path"),
        "poll_s": _positive_number(raw.get("poll_s", 30), "poll_s"),
        "initial_delay_s": _positive_number(
            raw.get("initial_delay_s", 0), "initial_delay_s", allow_zero=True
        ),
        "warmup_s": _positive_number(raw.get("warmup_s", 0), "warmup_s", allow_zero=True),
        "receipt_checks": _check_list(raw, "receipt_checks"),
        "heartbeat_checks": _check_list(raw, "heartbeat_checks"),
        "artifact_checks": _check_list(raw, "artifact_checks"),
        "success_receipts": _success_receipt_list(raw, "success_receipts"),
        "success_settle_s": _positive_number(
            raw.get("success_settle_s", 30), "success_settle_s", allow_zero=True
        ),
    }
    if not any(
        config[key] for key in ("receipt_checks", "heartbeat_checks", "artifact_checks")
    ):
        raise ConfigError("at least one receipt, heartbeat, or artifact mtime check is required")
    return config


def _read_pid(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    pid = int(text)
    if pid <= 1:
        raise ValueError(f"unsafe pid {pid}")
    return pid


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists even if this process cannot signal it.
        return True
    return not _pid_is_zombie(pid)


def _pid_is_zombie(pid: int) -> bool:
    # kill(pid, 0) succeeds on zombies: an exited child stays in the process
    # table until its (possibly stopped) parent reaps it, so signal-based
    # liveness alone reports it alive forever. A zombie can never run again —
    # for liveness purposes it is dead.
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip().startswith("Z")


def _json_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in ("status", "exit", "rc", "error") if key in payload}


_FAILURE_STATUSES = frozenset({"failed", "error", "killed", "timeout", "timed_out", "oom"})
_SUCCESS_STATUSES = frozenset({"completed", "complete", "success", "succeeded", "ok", "done"})


def _summary_is_success(summary: Mapping[str, Any]) -> bool:
    """A receipt proves clean completion only when nothing in it contradicts rc=0."""
    rc = summary.get("rc")
    exit_code = summary.get("exit")
    if isinstance(rc, (int, float)) and not isinstance(rc, bool) and int(rc) != 0:
        return False
    if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and int(exit_code) != 0:
        return False
    status = str(summary.get("status", "")).strip().lower()
    if status in _FAILURE_STATUSES:
        return False
    if isinstance(rc, (int, float)) and not isinstance(rc, bool) and int(rc) == 0:
        return True
    if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and int(exit_code) == 0:
        return True
    return status in _SUCCESS_STATUSES


def evaluate(config: Mapping[str, Any], *, started_at: float, now: float) -> dict[str, Any] | None:
    """Return one typed alert body, or ``None`` while all configured checks hold."""

    pid_file = config["pid_file"]
    try:
        pid = _read_pid(pid_file)
    except (OSError, ValueError) as exc:
        if now - started_at <= config["warmup_s"]:
            return None
        return {
            "reason": "pidfile_missing_or_invalid",
            "pid_file": str(pid_file),
            "detail": str(exc),
        }
    if not _pid_alive(pid):
        success_summaries = {
            row["label"]: _json_summary(row["path"])
            for row in config.get("success_receipts", [])
            if row["path"].exists()
        }
        for label, summary in success_summaries.items():
            if _summary_is_success(summary):
                return {
                    "reason": "clean_exit",
                    "pid": pid,
                    "success_receipt": label,
                    "summary": summary,
                }
        detail: dict[str, Any] = {"reason": "child_dead", "pid": pid}
        summaries = {
            row["label"]: _json_summary(row["path"])
            for row in config["receipt_checks"]
            if row["path"].exists()
        }
        if success_summaries:
            detail["success_receipt_summaries"] = success_summaries
        if summaries:
            detail["receipt_summaries"] = summaries
        return detail

    for family, reason in (
        ("receipt_checks", "receipt_stale"),
        ("heartbeat_checks", "heartbeat_stale"),
        ("artifact_checks", "artifact_stalled"),
    ):
        for row in config[family]:
            if now - started_at <= row["grace_s"]:
                continue
            try:
                age_s = max(0.0, now - row["path"].stat().st_mtime)
                missing = False
            except OSError:
                age_s = None
                missing = True
            if missing or (age_s is not None and age_s > row["max_age_s"]):
                return {
                    "reason": reason,
                    "pid": pid,
                    "label": row["label"],
                    "path": str(row["path"]),
                    "age_s": age_s,
                    "missing": missing,
                    "max_age_s": row["max_age_s"],
                }
    return None


def atomic_write_once(path: Path, payload: Mapping[str, Any]) -> bool:
    """Publish one durable alert without requiring hard-link support."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        return False
    claim = path.with_name(f".{path.name}.publish-lock")
    try:
        claim.mkdir()
    except FileExistsError as exc:
        if path.exists():
            return False
        raise RuntimeError(f"alert publish lock exists without alert: {claim}") from exc
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        if path.exists():
            return False
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        return True
    finally:
        tmp.unlink(missing_ok=True)
        claim.rmdir()


def run(config: Mapping[str, Any], *, once: bool = False) -> int:
    started_at = time.time()
    if not once and config["initial_delay_s"] > 0:
        time.sleep(config["initial_delay_s"])
    first_dead_at: float | None = None
    while True:
        now = time.time()
        alert = evaluate(config, started_at=started_at, now=now)
        if alert is not None and alert.get("reason") == "clean_exit":
            print(json.dumps({"schema": SCHEMA, **alert}, sort_keys=True, default=str))
            return 0
        if (
            alert is not None
            and alert.get("reason") == "child_dead"
            and config.get("success_receipts")
            and not once
        ):
            # The launcher writes the success receipt moments after the child
            # exits; give it success_settle_s before adjudicating death.
            if first_dead_at is None:
                first_dead_at = now
            if now - first_dead_at < config.get("success_settle_s", 0):
                time.sleep(min(config["poll_s"], 5.0))
                continue
        if alert is not None:
            body = {
                "schema": ALERT_SCHEMA,
                "watcher_schema": SCHEMA,
                "generated_unix_s": now,
                **alert,
            }
            atomic_write_once(config["alert_path"], body)
            return 1
        first_dead_at = None
        if once:
            return 0
        time.sleep(config["poll_s"])


def _write_event_receipt(
    event_receipt: Path, launch_manifest: Path, alert_path: Path, elapsed_s: float
) -> None:
    manifest = json.loads(launch_manifest.read_text(encoding="utf-8"))
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, dict):
        raise ConfigError("launch manifest has no structured launch_id")
    _write_json_payload = {
        "schema": "detached_local_process_done.v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt_name": event_receipt.name.removesuffix(".done"),
        "launch_id": launch_id,
        "rc": 1,
        "elapsed_s": round(elapsed_s, 6),
        "detail": f"liveness_watcher_alert={alert_path}",
        "adjudicated_at_launch": False,
        "watcher_kind": "liveness",
    }
    event_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = event_receipt.with_name(f".{event_receipt.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_write_json_payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(event_receipt)


def _wait_for_start_gate(path: Path | None) -> None:
    while path is not None and not path.exists():
        time.sleep(0.02)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Evaluate immediately once, then exit.")
    parser.add_argument("--event-receipt", type=Path)
    parser.add_argument("--launch-manifest", type=Path)
    parser.add_argument("--start-gate", type=Path)
    args = parser.parse_args(argv)
    if (args.event_receipt is None) != (args.launch_manifest is None):
        parser.error("--event-receipt and --launch-manifest must be supplied together")
    try:
        config = load_config(args.config.expanduser().resolve(strict=False))
    except ConfigError as exc:
        print(json.dumps({"error": str(exc), "schema": SCHEMA}), file=os.sys.stderr)
        return 2
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "config_valid": True}, sort_keys=True))
        return 0
    _ignore_harness_signals()
    started_at = time.time()
    _wait_for_start_gate(args.start_gate)
    rc = run(config, once=args.once)
    if rc != 0 and args.event_receipt is not None:
        try:
            _write_event_receipt(
                args.event_receipt.expanduser().resolve(strict=False),
                args.launch_manifest.expanduser().resolve(strict=False),
                config["alert_path"],
                time.time() - started_at,
            )
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            print(json.dumps({"error": f"event receipt failed: {exc}"}), file=os.sys.stderr)
            return 3
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
