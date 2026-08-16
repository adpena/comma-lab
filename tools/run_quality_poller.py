#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Poll JSON-line training telemetry for configurable quality regressions.

This is the canonical promotion of the rx2 quality poller.  Field names,
thresholds, phase knee, and enabled alert conditions live in JSON config; the
executable contains no run-specific model or path assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac import process_liveness

SCHEMA = "pact.run_quality_poller.v1"
CONFIG_SCHEMA = "pact.run_quality_poller.config.v1"
ALERT_SCHEMA = "pact.run_quality_poller.alert.v1"


class ConfigError(ValueError):
    """The quality-poller config is incomplete or internally inconsistent."""


def _ignore_harness_signals() -> None:
    for name in ("SIGURG", "SIGPIPE", "SIGHUP"):
        try:
            signal.signal(getattr(signal, name), signal.SIG_IGN)
        except (AttributeError, OSError, ValueError):
            pass


def _path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a nonempty path string")
    return Path(value).expanduser().resolve(strict=False)


def _number(value: Any, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ConfigError(f"{field_name} must be finite{' and positive' if positive else ''}")
    return result


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a nonempty string")
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be boolean")
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise ConfigError(f"{field_name} must be an integer")
    return result


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"config schema must be {CONFIG_SCHEMA}")
    fields = raw.get("fields")
    conditions = raw.get("alert_conditions")
    knee = raw.get("phase_knee")
    best = raw.get("best_not_latest")
    if not isinstance(fields, dict) or not isinstance(conditions, dict):
        raise ConfigError("fields and alert_conditions must be objects")
    if not isinstance(knee, dict) or not isinstance(best, dict):
        raise ConfigError("phase_knee and best_not_latest must be objects")
    finite_fields = fields.get("finite", [])
    if not isinstance(finite_fields, list) or not all(
        isinstance(item, str) and item for item in finite_fields
    ):
        raise ConfigError("fields.finite must be a list of field names")
    enabled = {
        name: _boolean(conditions.get(name, False), f"alert_conditions.{name}")
        for name in ("joint_regression", "qat_knee_shock", "nan_or_garbage", "stale_telemetry")
    }
    raw_bands = raw.get("regression_bands", [])
    if not isinstance(raw_bands, list):
        raise ConfigError("regression_bands must be a list")
    regression_bands: list[dict[str, Any]] = []
    for index, band in enumerate(raw_bands):
        if not isinstance(band, dict):
            raise ConfigError(f"regression_bands[{index}] must be an object")
        field_name = _string(band.get("field"), f"regression_bands[{index}].field")
        if field_name not in finite_fields and field_name != fields.get("value"):
            raise ConfigError(
                f"regression_bands[{index}].field must also be checked by fields.finite"
            )
        regression_bands.append(
            {
                "field": field_name,
                "label": _string(
                    band.get("label", field_name), f"regression_bands[{index}].label"
                ),
                "upper": _number(band.get("upper"), f"regression_bands[{index}].upper"),
                "start_epoch": _integer(
                    band.get("start_epoch"), f"regression_bands[{index}].start_epoch"
                ),
            }
        )
    config = {
        "schema": CONFIG_SCHEMA,
        "log_path": _path(raw.get("log_path"), "log_path"),
        "pid_file": _path(raw.get("pid_file"), "pid_file"),
        "telemetry_path": _path(raw.get("telemetry_path"), "telemetry_path"),
        "alert_path": _path(raw.get("alert_path"), "alert_path"),
        "poll_s": _number(raw.get("poll_s", 60), "poll_s", positive=True),
        "eval_period_s": _number(raw.get("eval_period_s"), "eval_period_s", positive=True),
        "stale_periods": _number(raw.get("stale_periods", 3), "stale_periods", positive=True),
        "startup_grace_s": _number(raw.get("startup_grace_s", 0), "startup_grace_s"),
        "json_marker": _string(raw.get("json_marker"), "json_marker"),
        "epoch_field": _string(fields.get("epoch"), "fields.epoch"),
        "value_field": _string(fields.get("value"), "fields.value"),
        "phase_field": _string(fields.get("phase"), "fields.phase"),
        "finite_fields": finite_fields,
        "bar_value": _number(raw.get("bar_value"), "bar_value"),
        "bar_start_epoch": _integer(raw.get("bar_start_epoch"), "bar_start_epoch"),
        "knee_epoch": _integer(knee.get("epoch"), "phase_knee.epoch"),
        "knee_window_epochs": _integer(
            knee.get("window_epochs", 3), "phase_knee.window_epochs"
        ),
        "knee_shock_multiplier": _number(
            knee.get("shock_multiplier", 1.25), "phase_knee.shock_multiplier", positive=True
        ),
        "continuous_phase": _string(knee.get("continuous_phase"), "phase_knee.continuous_phase"),
        "best_phase": _string(best.get("phase"), "best_not_latest.phase"),
        "best_min_rows": _integer(best.get("min_rows", 4), "best_not_latest.min_rows"),
        "best_lag_epochs": _integer(best.get("lag_epochs", 6), "best_not_latest.lag_epochs"),
        "conditions": enabled,
        "regression_bands": regression_bands,
    }
    if config["startup_grace_s"] < 0:
        raise ConfigError("startup_grace_s must be nonnegative")
    for name in ("bar_start_epoch", "knee_epoch", "knee_window_epochs", "best_min_rows", "best_lag_epochs"):
        if config[name] < 0:
            raise ConfigError(f"{name} must be nonnegative")
    if any(band["start_epoch"] < 0 for band in regression_bands):
        raise ConfigError("regression-band start epochs must be nonnegative")
    if not any(enabled.values()) and not regression_bands:
        raise ConfigError("at least one alert condition or regression band must be enabled")
    return config


# Liveness is read through the CANONICAL surface, never re-implemented here.
# ``_pid_alive`` was hand-rolled 11x across the tree with DIVERGENT semantics
# (PermissionError reads ALIVE at three sites, DEAD at two); a 12th copy would
# have deepened that drift.  ``tac.process_liveness`` is the single source.
PID_ALIVE = process_liveness.ALIVE
PID_DEAD = process_liveness.DEAD
PID_UNREADABLE = process_liveness.UNREADABLE


def _pid_state(pid_file: Path) -> str:
    """Tri-state pid read for this poller's pid FILE.

    The bool this replaced collapsed three states into one, and that collapse
    WAS the defect (#1064, measured by ddm_lw2): a MISSING or garbage
    ``pid_file`` -- the poller is BLIND -- read identically to a child that
    exited -- genuine death, correctly owned by the liveness watcher.  So a
    drifted ``pid_file`` made this poller return 0 silently, write no event
    receipt, and look exactly like health.  ``ddm_ra2c_rank4``'s quality poller
    watched nothing for a whole run: 0-byte log, no ``.done``, rc=0.

    A watcher that fails silently is strictly worse than no watcher: it spends
    a watcher's trust budget and pays out nothing.
    """
    return process_liveness.pid_file_state(pid_file)


def _pid_alive(pid_file: Path) -> bool:
    return _pid_state(pid_file) == PID_ALIVE


def blindness_alert(
    config: Mapping[str, Any], *, now: float, started_at: float
) -> dict[str, Any] | None:
    """Can this poller observe its target at all?  Loud past the startup grace.

    Deliberately NOT gated by ``config["conditions"]``: an operator may disable
    a *quality* condition, but "am I actually watching anything" is never a
    tunable.  Within the startup grace this stays silent, because a launcher
    legitimately has not written the pidfile or the log yet.
    """
    if now - started_at < config["startup_grace_s"]:
        return None
    if _pid_state(config["pid_file"]) == PID_UNREADABLE:
        return {
            "reason": "watcher_blind_pid_file_unreadable",
            "pid_file": str(config["pid_file"]),
            "startup_grace_s": config["startup_grace_s"],
        }
    if not config["log_path"].exists():
        return {
            "reason": "watcher_blind_log_absent",
            "log_path": str(config["log_path"]),
            "startup_grace_s": config["startup_grace_s"],
        }
    return None


def read_eval_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = config["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        if config["json_marker"] not in line or "{" not in line:
            continue
        try:
            row = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        if config["epoch_field"] in row and config["value_field"] in row:
            rows.append(row)
    return rows


def _atomic_write_once(path: Path, payload: Mapping[str, Any]) -> bool:
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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class PollState:
    seen_epochs: set[int] = field(default_factory=set)
    last_new_row_ts: float = field(default_factory=time.time)
    last_continuous_value: float | None = None
    best_not_latest_reported: bool = False


def _finite_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    names = [config["value_field"], *config["finite_fields"]]
    for name in names:
        try:
            if not math.isfinite(float(row[name])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def poll_once(
    config: Mapping[str, Any], state: PollState, *, now: float
) -> tuple[dict[str, Any] | None, bool]:
    """Evaluate new log rows; return (alert, child_alive)."""

    rows = read_eval_rows(config)
    epoch_field = config["epoch_field"]
    value_field = config["value_field"]
    phase_field = config["phase_field"]
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            epoch = int(row[epoch_field])
        except (KeyError, TypeError, ValueError):
            if config["conditions"]["nan_or_garbage"]:
                return {"reason": "nan_or_garbage", "row": row}, _pid_alive(
                    config["pid_file"]
                )
            continue
        if epoch not in state.seen_epochs:
            new_rows.append(row)
    if new_rows:
        state.last_new_row_ts = now
        for row in new_rows:
            epoch = int(row[epoch_field])
            state.seen_epochs.add(epoch)
            _append_jsonl(config["telemetry_path"], {"ts": now, **row})
            if config["conditions"]["nan_or_garbage"] and not _finite_row(row, config):
                return {"reason": "nan_or_garbage", "row": row}, _pid_alive(config["pid_file"])
            value = float(row[value_field])
            phase = row.get(phase_field)
            if phase == config["continuous_phase"] or epoch < config["knee_epoch"]:
                state.last_continuous_value = value
            elif (
                config["conditions"]["qat_knee_shock"]
                and state.last_continuous_value is not None
                and epoch < config["knee_epoch"] + config["knee_window_epochs"]
                and value > config["knee_shock_multiplier"] * state.last_continuous_value
            ):
                return {
                    "reason": "qat_knee_shock",
                    "epoch": epoch,
                    "value": value,
                    "last_continuous": state.last_continuous_value,
                }, _pid_alive(config["pid_file"])
            if (
                config["conditions"]["joint_regression"]
                and epoch >= config["bar_start_epoch"]
                and value > config["bar_value"]
            ):
                return {
                    "reason": "joint_regression",
                    "epoch": epoch,
                    "value": value,
                    "bar": config["bar_value"],
                }, _pid_alive(config["pid_file"])
            for band in config["regression_bands"]:
                if epoch < band["start_epoch"]:
                    continue
                try:
                    band_value = float(row[band["field"]])
                except (KeyError, TypeError, ValueError):
                    return {
                        "reason": "regression_band_unreadable",
                        "label": band["label"],
                        "field": band["field"],
                        "epoch": epoch,
                    }, _pid_alive(config["pid_file"])
                if not math.isfinite(band_value):
                    return {
                        "reason": "regression_band_unreadable",
                        "label": band["label"],
                        "field": band["field"],
                        "epoch": epoch,
                    }, _pid_alive(config["pid_file"])
                if band_value > band["upper"]:
                    return {
                        "reason": "regression_band",
                        "label": band["label"],
                        "field": band["field"],
                        "epoch": epoch,
                        "value": band_value,
                        "upper": band["upper"],
                    }, _pid_alive(config["pid_file"])

    phase_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get(phase_field) != config["best_phase"]:
            continue
        try:
            int(row[epoch_field])
            float(row[value_field])
        except (KeyError, TypeError, ValueError):
            continue
        phase_rows.append(row)
    if len(phase_rows) >= config["best_min_rows"] and not state.best_not_latest_reported:
        best_row = min(phase_rows, key=lambda row: float(row[value_field]))
        latest_epoch = max(int(row[epoch_field]) for row in phase_rows)
        best_epoch = int(best_row[epoch_field])
        if latest_epoch - best_epoch >= config["best_lag_epochs"]:
            _append_jsonl(
                config["telemetry_path"],
                {
                    "ts": now,
                    "info": "best_not_latest",
                    "best_epoch": best_epoch,
                    "latest_epoch": latest_epoch,
                },
            )
            state.best_not_latest_reported = True

    child_alive = _pid_alive(config["pid_file"])
    if (
        config["conditions"]["stale_telemetry"]
        and child_alive
        and now - state.last_new_row_ts > config["stale_periods"] * config["eval_period_s"]
    ):
        return {
            "reason": "stale_telemetry",
            "seconds_since_new_row": now - state.last_new_row_ts,
        }, child_alive
    return None, child_alive


def run(config: Mapping[str, Any], *, once: bool = False) -> int:
    state = PollState()
    started_at = time.time()
    while True:
        now = time.time()
        # Blindness is checked FIRST: if the poller cannot see its target, every
        # downstream quality verdict is drawn from an empty read and would be
        # reported under the wrong reason (or, before the cure, not at all).
        blind = blindness_alert(config, now=now, started_at=started_at)
        if blind is not None:
            _atomic_write_once(
                config["alert_path"],
                {
                    "schema": ALERT_SCHEMA,
                    "watcher_schema": SCHEMA,
                    "generated_unix_s": now,
                    **blind,
                },
            )
            return 1
        alert, alive = poll_once(config, state, now=now)
        if alert is not None:
            _atomic_write_once(
                config["alert_path"],
                {
                    "schema": ALERT_SCHEMA,
                    "watcher_schema": SCHEMA,
                    "generated_unix_s": now,
                    **alert,
                },
            )
            return 1
        if once:
            return 0
        if not alive:
            if now - started_at < config["startup_grace_s"]:
                time.sleep(config["poll_s"])
                continue
            # Process death is intentionally owned by the liveness watcher.
            return 0
        time.sleep(config["poll_s"])


def _write_event_receipt(
    event_receipt: Path, launch_manifest: Path, alert_path: Path, elapsed_s: float
) -> None:
    manifest = json.loads(launch_manifest.read_text(encoding="utf-8"))
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, dict):
        raise ConfigError("launch manifest has no structured launch_id")
    payload = {
        "schema": "detached_local_process_done.v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt_name": event_receipt.name.removesuffix(".done"),
        "launch_id": launch_id,
        "rc": 1,
        "elapsed_s": round(elapsed_s, 6),
        "detail": f"quality_watcher_alert={alert_path}",
        "adjudicated_at_launch": False,
        "watcher_kind": "quality",
    }
    event_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = event_receipt.with_name(f".{event_receipt.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
