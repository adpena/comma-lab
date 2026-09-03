#!/usr/bin/env python3
"""Run the sealed QBR1 cells serially, then invoke sealed adjudication.

The driver is deliberately narrow: the fire order owns every child argv and
every scientific config value.  This file only verifies the sealed inputs,
applies the five authorized claim fields, sequences one detached cell at a
time, and records durable control-plane receipts.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # script-invoked: repo root for `tools`
from tools import claim_lane_dispatch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FIRE_ORDER = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/SEALED_MAIN_FIRE_ORDER.json"
)
DEFAULT_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
DEFAULT_SCORER_CLAIM = "ddm_qbr1_scorer_20260903"
DEFAULT_METAL_CLAIM = "ddm_qbr1_metal_20260903"
DEFAULT_RESERVE_BYTES = 8 * 1024**3
FIRE_ORDER_SCHEMA = "ddm_qbr1_sealed_main_fire_order.v2"
DONE_RECEIPT_SCHEMA = "detached_local_process_done.v2"
CONSUMED_RECEIPT_SCHEMA = "detached_local_process_done_consumed.v1"
CHAIN_LEDGER_SCHEMA = "ddm_qbr1_cell_chain_ledger.v1"


class ChainRefusal(RuntimeError):
    """A typed fail-closed chain stop."""

    def __init__(self, reason: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ddm_qbr1_cell_chain_refusal.v1",
            "status": "REFUSED",
            "reason": self.reason,
            "message": str(self),
            **self.detail,
        }


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def utc_text(value: dt.datetime | None = None) -> str:
    return (value or utc_now()).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChainRefusal("INVALID_JSON", f"{label} is not readable JSON", path=str(path)) from exc
    if not isinstance(value, dict):
        raise ChainRefusal("INVALID_JSON", f"{label} must be a JSON object", path=str(path))
    return value


def verify_fact(label: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    try:
        path = Path(str(fact["path"]))
        expected_bytes = int(fact["bytes"])
        expected_sha = str(fact["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainRefusal("INVALID_FILE_FACT", f"{label} has an invalid file fact") from exc
    if not path.is_file():
        raise ChainRefusal("SOURCE_PIN_MISSING", f"{label} is missing", path=str(path))
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise ChainRefusal(
            "SOURCE_PIN_MISMATCH",
            f"{label} bytes or SHA-256 changed",
            path=str(path),
            expected_bytes=expected_bytes,
            actual_bytes=actual_bytes,
            expected_sha256=expected_sha,
            actual_sha256=actual_sha,
        )
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}


def validate_fire_order(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    if order.get("schema") != FIRE_ORDER_SCHEMA:
        raise ChainRefusal("FIRE_ORDER_SCHEMA", "unexpected sealed fire-order schema")
    raw_cells = order.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 6:
        raise ChainRefusal("FIRE_ORDER_CELL_COUNT", "sealed fire order must contain six cells")
    if not all(isinstance(row, dict) for row in raw_cells):
        raise ChainRefusal("FIRE_ORDER_CELL_SHAPE", "every fire-order cell must be an object")
    cells = sorted(raw_cells, key=lambda row: int(row.get("order", -1)))
    if [row.get("order") for row in cells] != list(range(1, 7)):
        raise ChainRefusal("FIRE_ORDER_SEQUENCE", "cell order must be exactly 1 through 6")
    ids = [str(row.get("cell_id", "")) for row in cells]
    if any(not cell_id for cell_id in ids) or len(set(ids)) != 6:
        raise ChainRefusal("FIRE_ORDER_CELL_IDS", "cell IDs must be six unique non-empty strings")
    for row in cells:
        argv = row.get("launcher_argv")
        if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
            raise ChainRefusal("LAUNCHER_ARGV", "launcher_argv must be a string array")
        if argv.count("AUTHORIZED_CONFIG_PATH") != 1:
            raise ChainRefusal(
                "LAUNCHER_PLACEHOLDER",
                "launcher_argv must contain exactly one AUTHORIZED_CONFIG_PATH token",
                cell_id=row["cell_id"],
            )
        if argv.count("--") != 1 or argv.count("--output-dir") != 1:
            raise ChainRefusal(
                "LAUNCHER_SHAPE",
                "launcher_argv must contain one child delimiter and one output directory",
                cell_id=row["cell_id"],
            )
        if not isinstance(row.get("config"), dict):
            raise ChainRefusal("CONFIG_FACT", "cell config fact is missing", cell_id=row["cell_id"])
    adjudication = order.get("adjudication_argv")
    if not isinstance(adjudication, list) or not adjudication or not all(
        isinstance(token, str) for token in adjudication
    ):
        raise ChainRefusal("ADJUDICATION_ARGV", "adjudication_argv must be a non-empty string array")
    return cells


def launcher_output_dir(argv: Sequence[str]) -> Path:
    index = argv.index("--output-dir")
    if index + 1 >= len(argv):
        raise ChainRefusal("LAUNCHER_SHAPE", "--output-dir has no value")
    return Path(argv[index + 1])


def bind_argv(argv: Sequence[str], authorized_path: Path) -> list[str]:
    return [str(authorized_path) if token == "AUTHORIZED_CONFIG_PATH" else token for token in argv]


def child_argv(launcher_argv: Sequence[str]) -> list[str]:
    delimiter = launcher_argv.index("--")
    return list(launcher_argv[delimiter + 1 :])


def authorized_path(sealed_path: Path, *, root: Path | None = None) -> Path:
    parent = root if root is not None else sealed_path.parent.parent / "authorized_configs"
    return parent / sealed_path.name


def authorized_config(
    sealed: Mapping[str, Any], scorer_claim_id: str, metal_claim_id: str
) -> dict[str, Any]:
    expected = copy.deepcopy(dict(sealed))
    expected["launch_authorized"] = True
    for field, claim_id in (
        ("scorer_lane", scorer_claim_id),
        ("metal_lane", metal_claim_id),
    ):
        lane = expected.get(field)
        if not isinstance(lane, dict):
            raise ChainRefusal("CLAIM_MUTATION_SHAPE", f"{field} must be an object")
        lane["claimed"] = True
        lane["claim_id"] = claim_id
    return expected


def write_or_verify_authorized(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        actual = load_json(path, label="authorized config")
        if actual != expected:
            raise ChainRefusal(
                "AUTHORIZED_CONFIG_CONFLICT",
                "existing authorized config differs from the exact claim mutation",
                path=str(path),
            )
    else:
        atomic_write(path, canonical_json_bytes(expected))
    actual = load_json(path, label="authorized config")
    if actual != expected:
        raise ChainRefusal(
            "AUTHORIZED_CONFIG_ROUNDTRIP",
            "authorized config changed during its atomic write",
            path=str(path),
        )
    return file_fact(path)


def verify_source_pins(order: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    pins = config.get("source_pins")
    if not isinstance(pins, dict) or pins != order.get("inputs"):
        raise ChainRefusal(
            "SOURCE_PIN_SET_MISMATCH",
            "cell source_pins differ from the sealed fire-order inputs",
        )
    return [verify_fact(f"source_pins.{name}", pins[name]) for name in sorted(pins)]


def parse_now(value: str) -> dt.datetime:
    if not value:
        return utc_now()
    parsed = claim_lane_dispatch._parse_utc(value)
    if parsed is None:
        raise ChainRefusal("NOW_UTC", "--now-utc must be an ISO-8601 timestamp")
    return parsed


def verify_claim(
    claims_path: Path,
    claim_id: str,
    expected_platform: str,
    *,
    now: dt.datetime,
    ttl_hours: float,
) -> dict[str, Any]:
    try:
        claims = claim_lane_dispatch._parse_claims(claims_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChainRefusal("CLAIMS_UNREADABLE", "claim ledger is unreadable", path=str(claims_path)) from exc
    latest = claim_lane_dispatch._latest_claims_by_job(claims)
    matches = []
    for (lane_id, _job), claim in latest.items():
        stamp = claim_lane_dispatch._parse_utc(claim.timestamp_utc)
        if (
            lane_id == claim_id
            and claim.platform == expected_platform
            and not claim_lane_dispatch._is_terminal(claim.status)
            and stamp is not None
            and now - stamp <= dt.timedelta(hours=ttl_hours)
        ):
            matches.append(claim)
    if not matches:
        raise ChainRefusal(
            "CLAIM_NOT_LIVE",
            "required lane claim is absent, terminal, stale, or on the wrong platform",
            claim_id=claim_id,
            expected_platform=expected_platform,
            ttl_hours=ttl_hours,
        )
    selected = max(matches, key=lambda claim: claim_lane_dispatch._parse_utc(claim.timestamp_utc))
    return {
        "timestamp_utc": selected.timestamp_utc,
        "agent": selected.agent,
        "lane_id": selected.lane_id,
        "platform": selected.platform,
        "instance_job_id": selected.instance_job_id,
        "status": selected.status,
    }


def storage_preflight(path: Path, reserve_bytes: int) -> dict[str, Any]:
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if not anchor.exists():
        raise ChainRefusal(
            "STORAGE_ANCHOR_MISSING",
            "no existing ancestor is available for the storage preflight",
            path=str(path),
        )
    usage = os.statvfs(anchor)
    available = int(usage.f_bavail * usage.f_frsize)
    if available < reserve_bytes:
        raise ChainRefusal(
            "AP_RESERVE",
            "APDataStore free space is below the required reserve",
            path=str(anchor),
            available_bytes=available,
            reserve_bytes=reserve_bytes,
        )
    return {"path": str(anchor), "available_bytes": available, "reserve_bytes": reserve_bytes}


def read_pid(path: Path) -> int:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise ChainRefusal("PIDFILE_INVALID", "cell pidfile is unreadable", path=str(path)) from exc
    if pid <= 1:
        raise ChainRefusal("PIDFILE_INVALID", "cell pidfile contains an unsafe PID", path=str(path))
    return pid


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_complete_result(path: Path, config: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = load_json(path, label="cell result")
    if result.get("complete") is not True:
        return None
    if result.get("cell_id") != config.get("cell_id"):
        raise ChainRefusal("RESULT_CELL_ID", "complete result belongs to another cell", path=str(path))
    if int(result.get("completed_steps", -1)) != int(config.get("total_steps", -2)):
        raise ChainRefusal("RESULT_STEPS", "complete result does not end at total_steps", path=str(path))
    return result


def validate_manifest(
    manifest_path: Path,
    *,
    launch_dir: Path,
    expected_child_argv: Sequence[str],
) -> dict[str, Any]:
    manifest = load_json(manifest_path, label="launch manifest")
    pid_path = launch_dir / "run.pid"
    pid = read_pid(pid_path)
    if (
        manifest.get("schema") != "detached_local_process_launch.v2"
        or manifest.get("dry_run") is not False
        or manifest.get("output_dir") != str(launch_dir)
        or manifest.get("pid") != pid
        or manifest.get("argv") != list(expected_child_argv)
        or manifest.get("launch_id", {}).get("pid") != pid
        or manifest.get("launch_id", {}).get("manifest_path") != str(manifest_path)
    ):
        raise ChainRefusal(
            "LAUNCH_MANIFEST_IDENTITY",
            "launch manifest does not match the sealed cell invocation",
            path=str(manifest_path),
        )
    done_path = manifest.get("done_receipt_path")
    if not isinstance(done_path, str) or not done_path:
        raise ChainRefusal("DONE_RECEIPT_PATH", "launch manifest has no done receipt path")
    return manifest


def live_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    live = []
    for cell in cells:
        launch_dir = launcher_output_dir(cell["launcher_argv"])
        pid_path = launch_dir / "run.pid"
        if pid_path.is_file():
            pid = read_pid(pid_path)
            if pid_alive(pid):
                live.append({"cell_id": cell["cell_id"], "pid": pid, "launch_dir": str(launch_dir)})
    return live


def append_ledger(path: Path, row: Mapping[str, Any]) -> None:
    existing = b""
    if path.exists():
        existing = path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            raise ChainRefusal("CHAIN_LEDGER_CORRUPT", "chain ledger lacks a trailing newline")
        for line in existing.splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChainRefusal("CHAIN_LEDGER_CORRUPT", "chain ledger has invalid JSONL") from exc
            if not isinstance(parsed, dict) or parsed.get("schema") != CHAIN_LEDGER_SCHEMA:
                raise ChainRefusal("CHAIN_LEDGER_CORRUPT", "chain ledger has an unexpected row")
    atomic_write(path, existing + json.dumps(dict(row), sort_keys=True, allow_nan=False).encode() + b"\n")


def consume_done_receipt(done_path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    digest = sha256_file(done_path)
    marker_path = done_path.with_name(done_path.name + ".consumed.json")
    marker = {
        "schema": CONSUMED_RECEIPT_SCHEMA,
        "consumed_utc": utc_text(),
        "consumer": "experiments/ddm_qbr1_cell_chain.py",
        "receipt_path": str(done_path),
        "receipt_sha256": digest,
        "suppressed_adjudicated_at_launch": bool(receipt.get("adjudicated_at_launch")),
    }
    if marker_path.exists():
        existing = load_json(marker_path, label="done receipt consumption marker")
        if existing.get("schema") == CONSUMED_RECEIPT_SCHEMA and existing.get(
            "receipt_sha256"
        ) == digest:
            return file_fact(marker_path)
    atomic_write(marker_path, canonical_json_bytes(marker))
    return file_fact(marker_path)


def wait_for_terminal(
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    poll_seconds: float,
    terminal_grace_seconds: float,
) -> dict[str, Any]:
    launch_dir = launcher_output_dir(cell["launcher_argv"])
    manifest_path = launch_dir / "launch_manifest.json"
    pid = int(manifest["pid"])
    done_path = Path(str(manifest["done_receipt_path"]))
    result_path = Path(str(config["output"])) / "RESULT.json"
    dead_since: float | None = None
    while True:
        if done_path.is_file():
            receipt = load_json(done_path, label="done receipt")
            if (
                receipt.get("schema") == DONE_RECEIPT_SCHEMA
                and receipt.get("launch_id") == manifest.get("launch_id")
            ):
                if int(receipt.get("rc", -1)) != 0:
                    raise ChainRefusal(
                        "CELL_TERMINAL_NONZERO",
                        "cell terminal receipt reports failure",
                        cell_id=cell["cell_id"],
                        receipt=receipt,
                    )
                result = load_complete_result(result_path, config)
                if result is None:
                    raise ChainRefusal(
                        "CELL_RESULT_INCOMPLETE",
                        "successful terminal receipt lacks a complete cell result",
                        cell_id=cell["cell_id"],
                        result_path=str(result_path),
                    )
                consumed = consume_done_receipt(done_path, receipt)
                return {
                    "manifest": file_fact(manifest_path),
                    "done_receipt": file_fact(done_path),
                    "done_receipt_consumed": consumed,
                    "result": file_fact(result_path),
                }
        if pid_alive(pid):
            dead_since = None
        else:
            dead_since = dead_since or time.monotonic()
            if time.monotonic() - dead_since >= terminal_grace_seconds:
                raise ChainRefusal(
                    "CELL_DIED_WITHOUT_RECEIPT",
                    "cell supervisor died without its matching terminal receipt",
                    cell_id=cell["cell_id"],
                    pid=pid,
                    done_receipt_path=str(done_path),
                )
        time.sleep(poll_seconds)


@contextlib.contextmanager
def chain_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ChainRefusal("CHAIN_ALREADY_RUNNING", "another chain driver holds the lock") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield


def preconditions(
    order: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    claims_path: Path,
    scorer_claim_id: str,
    metal_claim_id: str,
    now: dt.datetime | None,
    ttl_hours: float,
    reserve_bytes: int,
) -> dict[str, Any]:
    source_pins = verify_source_pins(order, config)
    storage = storage_preflight(Path(str(config["output"])), reserve_bytes)
    claim_now = now or utc_now()
    return {
        "source_pins": source_pins,
        "storage": storage,
        "scorer_claim": verify_claim(
            claims_path,
            scorer_claim_id,
            "local_macos_cpu",
            now=claim_now,
            ttl_hours=ttl_hours,
        ),
        "metal_claim": verify_claim(
            claims_path,
            metal_claim_id,
            "local_mlx_metal",
            now=claim_now,
            ttl_hours=ttl_hours,
        ),
    }


def consume_completed_cell_receipt(launch_dir: Path) -> dict[str, Any] | None:
    """Consume only a success receipt whose identity matches this cell manifest."""
    manifest_path = launch_dir / "launch_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path, label="completed-cell launch manifest")
    done_value = manifest.get("done_receipt_path")
    if not isinstance(done_value, str) or not done_value:
        return None
    done_path = Path(done_value)
    if not done_path.is_file():
        return None
    receipt = load_json(done_path, label="completed-cell done receipt")
    if receipt.get("schema") != DONE_RECEIPT_SCHEMA or receipt.get("launch_id") != manifest.get(
        "launch_id"
    ):
        return None
    if int(receipt.get("rc", -1)) != 0:
        raise ChainRefusal(
            "RECEIPT_RESULT_CONFLICT",
            "complete result conflicts with a nonzero matching terminal receipt",
            launch_dir=str(launch_dir),
        )
    return consume_done_receipt(done_path, receipt)


def append_complete_skip(
    ledger_path: Path,
    cell: Mapping[str, Any],
    result_path: Path,
) -> None:
    consumed = consume_completed_cell_receipt(launcher_output_dir(cell["launcher_argv"]))
    append_ledger(
        ledger_path,
        {
            "schema": CHAIN_LEDGER_SCHEMA,
            "written_at_utc": utc_text(),
            "cell_id": cell["cell_id"],
            "order": cell["order"],
            "action": "SKIPPED_COMPLETE",
            "result": file_fact(result_path),
            "matching_done_receipt_consumed": consumed,
        },
    )


def dry_run_plan(args: argparse.Namespace, order: Mapping[str, Any], cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    now = parse_now(args.now_utc)
    live = live_cells(cells)
    if len(live) > 1:
        raise ChainRefusal("MULTIPLE_LIVE_CELLS", "more than one QBR1 cell is live", live=live)
    source_set_verified = False
    sequence = []
    with tempfile.TemporaryDirectory(prefix="ddm_bc1_dry_run_") as temporary:
        dry_authorized_root = Path(temporary) / "authorized_configs"
        for cell in cells:
            sealed_fact = verify_fact(f"cells.{cell['cell_id']}.config", cell["config"])
            sealed = load_json(Path(sealed_fact["path"]), label="sealed config")
            if sealed.get("cell_id") != cell["cell_id"]:
                raise ChainRefusal("CONFIG_CELL_ID", "sealed config cell_id differs from fire order")
            if not source_set_verified:
                evidence = preconditions(
                    order,
                    sealed,
                    claims_path=args.claims,
                    scorer_claim_id=args.scorer_claim_id,
                    metal_claim_id=args.metal_claim_id,
                    now=now,
                    ttl_hours=args.claim_ttl_hours,
                    reserve_bytes=args.reserve_bytes,
                )
                source_set_verified = True
            expected = authorized_config(sealed, args.scorer_claim_id, args.metal_claim_id)
            dry_path = authorized_path(Path(sealed_fact["path"]), root=dry_authorized_root)
            dry_fact = write_or_verify_authorized(dry_path, expected)
            live_path = authorized_path(Path(sealed_fact["path"]))
            result = load_complete_result(Path(str(sealed["output"])) / "RESULT.json", sealed)
            live_ids = {row["cell_id"] for row in live}
            action = (
                "SKIP_COMPLETE"
                if result is not None
                else "ATTACH_LIVE"
                if cell["cell_id"] in live_ids
                else "WOULD_AUTHORIZE_AND_LAUNCH"
            )
            sequence.append(
                {
                    "order": cell["order"],
                    "cell_id": cell["cell_id"],
                    "action": action,
                    "sealed_config": sealed_fact,
                    "dry_run_authorized_config": dry_fact,
                    "live_authorized_config_path": str(live_path),
                    "launcher_argv": bind_argv(cell["launcher_argv"], live_path),
                    "argv_difference_from_seal": ["AUTHORIZED_CONFIG_PATH"],
                }
            )
    return {
        "schema": "ddm_qbr1_cell_chain_dry_run.v1",
        "status": "DRY_RUN_ONLY_NO_LAUNCH",
        "fire_order": file_fact(args.fire_order),
        "preconditions": evidence,
        "live_cells": live,
        "sequence": sequence,
        "adjudication_argv": order["adjudication_argv"],
        "chain_done_path": str(args.fire_order.parent / "CHAIN_DONE.json"),
        "temporary_authorized_configs_removed": True,
    }


def run_chain(args: argparse.Namespace, order: Mapping[str, Any], cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    root = args.fire_order.parent
    ledger_path = root / "CHAIN_LEDGER.jsonl"
    done_path = root / "CHAIN_DONE.json"
    fixed_now = parse_now(args.now_utc) if args.now_utc else None
    if done_path.is_file():
        done = load_json(done_path, label="chain completion receipt")
        current_fire_order = file_fact(args.fire_order)
        if (
            done.get("schema") == "ddm_qbr1_cell_chain_done.v1"
            and done.get("status") == "COMPLETE"
            and done.get("fire_order") == current_fire_order
        ):
            return done
        raise ChainRefusal(
            "CHAIN_DONE_CONFLICT",
            "existing CHAIN_DONE.json does not bind this sealed fire order",
            path=str(done_path),
        )
    for cell in cells:
        sealed_fact = verify_fact(f"cells.{cell['cell_id']}.config", cell["config"])
        sealed_path = Path(sealed_fact["path"])
        sealed = load_json(sealed_path, label="sealed config")
        if sealed.get("cell_id") != cell["cell_id"]:
            raise ChainRefusal("CONFIG_CELL_ID", "sealed config cell_id differs from fire order")
        result_path = Path(str(sealed["output"])) / "RESULT.json"
        live = live_cells(cells)
        if len(live) > 1:
            raise ChainRefusal("MULTIPLE_LIVE_CELLS", "more than one QBR1 cell is live", live=live)
        complete = load_complete_result(result_path, sealed)
        if complete is not None and (not live or live[0]["cell_id"] != cell["cell_id"]):
            append_complete_skip(ledger_path, cell, result_path)
            continue
        if live and live[0]["cell_id"] != cell["cell_id"]:
            raise ChainRefusal(
                "OTHER_CELL_LIVE",
                "another QBR1 cell is live; refusing concurrent launch",
                current_cell=cell["cell_id"],
                live=live,
            )

        evidence = preconditions(
            order,
            sealed,
            claims_path=args.claims,
            scorer_claim_id=args.scorer_claim_id,
            metal_claim_id=args.metal_claim_id,
            now=fixed_now,
            ttl_hours=args.claim_ttl_hours,
            reserve_bytes=args.reserve_bytes,
        )
        live = live_cells(cells)
        if len(live) > 1:
            raise ChainRefusal("MULTIPLE_LIVE_CELLS", "more than one QBR1 cell is live", live=live)
        complete = load_complete_result(result_path, sealed)
        if complete is not None and (not live or live[0]["cell_id"] != cell["cell_id"]):
            append_complete_skip(ledger_path, cell, result_path)
            continue
        if live and live[0]["cell_id"] != cell["cell_id"]:
            raise ChainRefusal(
                "OTHER_CELL_LIVE",
                "another QBR1 cell is live; refusing concurrent launch",
                current_cell=cell["cell_id"],
                live=live,
            )

        auth_path = authorized_path(sealed_path)
        expected = authorized_config(sealed, args.scorer_claim_id, args.metal_claim_id)
        bound_launcher = bind_argv(cell["launcher_argv"], auth_path)
        expected_child = child_argv(bound_launcher)
        launch_dir = launcher_output_dir(cell["launcher_argv"])
        manifest_path = launch_dir / "launch_manifest.json"
        action = "ATTACHED_LIVE"
        launcher_receipt: dict[str, Any] | None = None
        if live:
            if not auth_path.is_file():
                raise ChainRefusal(
                    "LIVE_CELL_AUTH_CONFIG_MISSING",
                    "live cell lacks its authorized config",
                    cell_id=cell["cell_id"],
                )
            authorized_fact = write_or_verify_authorized(auth_path, expected)
            manifest = validate_manifest(
                manifest_path, launch_dir=launch_dir, expected_child_argv=expected_child
            )
        else:
            if manifest_path.exists() or (launch_dir / "run.pid").exists() or result_path.exists():
                raise ChainRefusal(
                    "INCOMPLETE_PRIOR_ATTEMPT",
                    "non-live cell has prior launch or result evidence; refusing to re-fire",
                    cell_id=cell["cell_id"],
                    launch_dir=str(launch_dir),
                    result_path=str(result_path),
                )
            authorized_fact = write_or_verify_authorized(auth_path, expected)
            launched = subprocess.run(bound_launcher, check=False, capture_output=True, text=True)
            launcher_receipt = {
                "returncode": launched.returncode,
                "stdout": launched.stdout,
                "stderr": launched.stderr,
                "argv": bound_launcher,
            }
            if launched.returncode != 0:
                raise ChainRefusal(
                    "LAUNCHER_REFUSED",
                    "sealed launcher argv returned nonzero",
                    cell_id=cell["cell_id"],
                    launcher=launcher_receipt,
                )
            manifest = validate_manifest(
                manifest_path, launch_dir=launch_dir, expected_child_argv=expected_child
            )
            action = "LAUNCHED"

        terminal = wait_for_terminal(
            cell,
            sealed,
            manifest,
            poll_seconds=args.poll_seconds,
            terminal_grace_seconds=args.terminal_grace_seconds,
        )
        append_ledger(
            ledger_path,
            {
                "schema": CHAIN_LEDGER_SCHEMA,
                "written_at_utc": utc_text(),
                "cell_id": cell["cell_id"],
                "order": cell["order"],
                "action": action,
                "preconditions": evidence,
                "authorized_config": authorized_fact,
                "launcher": launcher_receipt,
                "terminal": terminal,
            },
        )

    for cell in cells:
        config = load_json(Path(str(cell["config"]["path"])), label="sealed config")
        if load_complete_result(Path(str(config["output"])) / "RESULT.json", config) is None:
            raise ChainRefusal("ADJUDICATION_INPUTS", "not all six cells are complete")
    adjudication_argv = list(order["adjudication_argv"])
    adjudicated = subprocess.run(adjudication_argv, check=False, capture_output=True, text=True)
    if adjudicated.returncode != 0:
        raise ChainRefusal(
            "ADJUDICATION_NONZERO",
            "sealed adjudication argv returned nonzero",
            returncode=adjudicated.returncode,
            stdout=adjudicated.stdout,
            stderr=adjudicated.stderr,
        )
    output_index = adjudication_argv.index("--output")
    adjudication_output = Path(adjudication_argv[output_index + 1])
    adjudication = load_json(adjudication_output, label="adjudication result")
    if adjudication.get("schema") != "ddm_qbr1_adjudication_result.v1":
        raise ChainRefusal("ADJUDICATION_RESULT", "adjudication result schema is unexpected")
    append_ledger(
        ledger_path,
        {
            "schema": CHAIN_LEDGER_SCHEMA,
            "written_at_utc": utc_text(),
            "action": "ADJUDICATION_COMPLETE",
            "argv": adjudication_argv,
            "result": file_fact(adjudication_output),
        },
    )
    done = {
        "schema": "ddm_qbr1_cell_chain_done.v1",
        "status": "COMPLETE",
        "written_at_utc": utc_text(),
        "fire_order": file_fact(args.fire_order),
        "chain_ledger": file_fact(ledger_path),
        "adjudication": file_fact(adjudication_output),
        "cells_complete": [cell["cell_id"] for cell in cells],
    }
    atomic_write(done_path, canonical_json_bytes(done))
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fire-order", type=Path, default=DEFAULT_FIRE_ORDER)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--scorer-claim-id", default=DEFAULT_SCORER_CLAIM)
    parser.add_argument("--metal-claim-id", default=DEFAULT_METAL_CLAIM)
    parser.add_argument("--claim-ttl-hours", type=float, default=24.0)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--terminal-grace-seconds", type=float, default=30.0)
    parser.add_argument("--now-utc", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.claim_ttl_hours <= 0
        or args.reserve_bytes < 0
        or args.poll_seconds <= 0
        or args.terminal_grace_seconds < 0
    ):
        print(json.dumps(ChainRefusal("INVALID_ARGUMENT", "numeric limits are invalid").as_dict()))
        return 2
    try:
        order = load_json(args.fire_order, label="sealed fire order")
        cells = validate_fire_order(order)
        if args.dry_run:
            result = dry_run_plan(args, order, cells)
        else:
            root = args.fire_order.parent
            with chain_lock(root / "CHAIN_DRIVER.lock"):
                result = run_chain(args, order, cells)
    except ChainRefusal as exc:
        if not args.dry_run and exc.reason != "CHAIN_ALREADY_RUNNING":
            try:
                append_ledger(
                    args.fire_order.parent / "CHAIN_LEDGER.jsonl",
                    {
                        "schema": CHAIN_LEDGER_SCHEMA,
                        "written_at_utc": utc_text(),
                        "action": "CHAIN_REFUSED",
                        "refusal": exc.as_dict(),
                    },
                )
            except (OSError, ChainRefusal):
                pass
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
