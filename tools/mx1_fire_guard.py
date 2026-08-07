#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed fire guard for ddm_mx1 Row-1 Metal launches.

The guard consumes the launch ticket and one argv key. It refuses unless the
ticket-required mem-probe receipt exists, passed, contains MLX/load telemetry,
and matches the fire argv. The mlx-train entrypoint then consumes the verdict as
its last line of defense.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MEM_PROBE_RECEIPT_SCHEMA = "ddm_mx1_load_phase_peak_receipt.v1"
FIRE_GUARD_VERDICT_SCHEMA = "ddm_mx1_fire_guard_verdict.v1"
EXIT_REFUSED = 4
RECEIPT_FRESHNESS_WINDOW_SECONDS = 6 * 60 * 60


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _host_fingerprint() -> dict[str, str]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ticket_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("launch_ticket"), dict):
        return payload["launch_ticket"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("ticket JSON must be an object")


def _default_verdict_path(ticket_path: Path, argv_key: str) -> Path:
    safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in argv_key)
    return ticket_path.parent / f"{ticket_path.stem}.{safe_key}.fire_guard_verdict.json"


def _unwrap_safe_run(argv: list[str]) -> list[str]:
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv


def _flag_value_map(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if "=" in key:
                k, v = key.split("=", 1)
                out[k] = v
                i += 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[key] = argv[i + 1]
                i += 2
                continue
            out[key] = "true"
        i += 1
    return out


def _norm_path(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (REPO / path).resolve()
    return str(path)


def _norm_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _effective_microbatch_pairs(device: Any, pairs: int | None, explicit: int | None) -> int | None:
    if pairs is None:
        return None
    total_pairs = max(1, int(pairs))
    if explicit is not None and explicit > 0:
        return max(1, min(int(explicit), total_pairs))
    if str(device).lower() == "gpu":
        return max(1, min(4, total_pairs))
    return total_pairs


def _parsed_fire_config(argv: list[str]) -> dict[str, Any]:
    raw = _unwrap_safe_run(argv)
    flags = _flag_value_map(raw)
    device = flags.get("device")
    pairs = _norm_int(flags.get("pairs"))
    explicit_microbatch_pairs = _norm_int(flags.get("microbatch_pairs", 0))
    return {
        "mode": flags.get("mode"),
        "device": device,
        "pairs": pairs,
        "lr": _norm_float(flags.get("lr")),
        "ce_fraction": _norm_float(flags.get("ce_fraction")),
        "softplus_fraction": _norm_float(flags.get("softplus_fraction")),
        "bits": _norm_int(flags.get("bits")),
        "microbatch_pairs": _effective_microbatch_pairs(device, pairs, explicit_microbatch_pairs),
        "mem_budget_gb": _norm_float(flags.get("mem_budget_gb")),
        "allow_soft_mem_limit": "allow_soft_mem_limit" in flags,
        "input_cache": _norm_path(flags.get("input_cache")),
        "target_cache": _norm_path(flags.get("target_cache")),
        "init": _norm_path(flags.get("init")),
        "fire_guard_verdict": _norm_path(flags.get("fire_guard_verdict")),
        "launch_ticket_path": _norm_path(flags.get("launch_ticket_path")),
        "fire_argv_key": flags.get("fire_argv_key"),
    }


def _receipt_config(receipt: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(receipt.get("argv_config") or {})
    device = cfg.get("device", receipt.get("device_request"))
    pairs = _norm_int(cfg.get("pairs", receipt.get("pairs")))
    microbatch_plan = (receipt.get("train_result_summary") or {}).get("microbatch_plan")
    if not isinstance(microbatch_plan, dict):
        microbatch_plan = receipt.get("microbatch_plan")
    microbatch_pairs = (
        _norm_int(microbatch_plan.get("microbatch_pairs"))
        if isinstance(microbatch_plan, dict)
        else None
    )
    if microbatch_pairs is None:
        microbatch_pairs = _effective_microbatch_pairs(device, pairs, _norm_int(cfg.get("microbatch_pairs", 0)))
    return {
        "mode": "mlx-train",
        "device": device,
        "pairs": pairs,
        "lr": _norm_float(cfg.get("lr")),
        "ce_fraction": _norm_float(cfg.get("ce_fraction")),
        "softplus_fraction": _norm_float(cfg.get("softplus_fraction")),
        "bits": _norm_int(cfg.get("bits")),
        "microbatch_pairs": microbatch_pairs,
        "mem_budget_gb": _norm_float(cfg.get("mem_budget_gb", receipt.get("mem_budget_gb_arg"))),
        "allow_soft_mem_limit": bool(cfg.get("allow_soft_mem_limit", False)),
        "input_cache": _norm_path(cfg.get("input_cache", receipt.get("input_cache"))),
        "target_cache": _norm_path(cfg.get("target_cache", receipt.get("target_cache"))),
        "init": _norm_path(cfg.get("init", receipt.get("init_checkpoint"))),
    }


def _sample_has_mlx(sample: dict[str, Any]) -> bool:
    return any(sample.get(key) is not None for key in ("mlx_active_gib", "mlx_cache_gib", "mlx_peak_gib"))


def _validate_samples(receipt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    samples = receipt.get("samples")
    if not isinstance(samples, list) or not samples:
        return False, "receipt_samples_missing", {"sample_count": 0}
    required_stage = (receipt.get("clearance_checks") or {}).get("required_stage")
    has_limit_stage = any(
        isinstance(row, dict) and str(row.get("stage", "")).startswith("after_require_mlx")
        for row in samples
    )
    train_step_samples = [
        row for row in samples
        if isinstance(row, dict) and str(row.get("stage", "")).startswith("after_train_step_")
    ]
    final_sample = next(
        (row for row in samples if isinstance(row, dict) and row.get("stage") == required_stage),
        None,
    )
    final_has_mlx = isinstance(final_sample, dict) and _sample_has_mlx(final_sample)
    detail = {
        "sample_count": len(samples),
        "required_stage": required_stage,
        "has_limit_stage": has_limit_stage,
        "train_step_sample_count": len(train_step_samples),
        "final_has_mlx": final_has_mlx,
    }
    if not has_limit_stage:
        return False, "receipt_load_stage_sample_missing", detail
    if not train_step_samples:
        return False, "receipt_train_step_sample_missing", detail
    if not final_has_mlx:
        return False, "receipt_final_step_mlx_telemetry_missing", detail
    return True, "samples_ok", detail


def _validate_memory_limits(receipt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    limits = receipt.get("memory_limits") or (receipt.get("train_result_summary") or {}).get("memory_limits")
    if not isinstance(limits, dict):
        return False, "receipt_memory_limits_missing", {}
    if limits.get("enforcement") == "software_stage_step_cap":
        software = receipt.get("software_budget") or (receipt.get("train_result_summary") or {}).get("software_budget")
        detail = {"memory_limits": limits, "software_budget": software}
        if not limits.get("software_cap_installed") or not limits.get("software_budget_bytes"):
            return False, "receipt_software_cap_not_installed", detail
        if not isinstance(software, dict):
            return False, "receipt_software_budget_summary_missing", detail
        if software.get("enforcement") != "software_stage_step_cap":
            return False, "receipt_software_budget_enforcement_mismatch", detail
        if int(software.get("check_count") or 0) <= 0:
            return False, "receipt_software_budget_checks_missing", detail
        last_check = software.get("last_check")
        if not isinstance(last_check, dict) or last_check.get("within_budget") is not True:
            return False, "receipt_software_budget_not_clear", detail
        return True, "software_memory_cap_ok", detail
    hard_required = bool(limits.get("hard_limit_required"))
    hard_satisfied = bool(limits.get("hard_limit_satisfied"))
    soft_allowed = bool(limits.get("soft_limit_allowed_by_cli"))
    if hard_required and not hard_satisfied and not soft_allowed:
        return False, "receipt_hard_mlx_limit_not_satisfied", limits
    return True, "memory_limits_ok", limits


def _validate_receipt_freshness(receipt_path: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        stat = receipt_path.stat()
    except OSError as exc:
        return False, "mem_probe_receipt_stat_error", {"error": f"{type(exc).__name__}: {exc}"}
    age_seconds = max(0.0, time.time() - stat.st_mtime)
    detail = {
        "receipt_mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace("+00:00", "Z"),
        "age_seconds": age_seconds,
        "freshness_window_seconds": RECEIPT_FRESHNESS_WINDOW_SECONDS,
        "freshness_rule": "receipt mtime must be <= 6h old because host memory state drifts across reboots",
    }
    if age_seconds > RECEIPT_FRESHNESS_WINDOW_SECONDS:
        return False, "mem_probe_receipt_stale", detail
    return True, "freshness_ok", detail


def _same_float(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-12


def _validate_config_match(fire: dict[str, Any], receipt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for key in (
        "mode",
        "device",
        "pairs",
        "bits",
        "microbatch_pairs",
        "allow_soft_mem_limit",
        "input_cache",
        "target_cache",
        "init",
    ):
        comparisons[key] = {"fire": fire.get(key), "receipt": receipt.get(key), "match": fire.get(key) == receipt.get(key)}
    for key in ("lr", "ce_fraction", "softplus_fraction", "mem_budget_gb"):
        comparisons[key] = {
            "fire": fire.get(key),
            "receipt": receipt.get(key),
            "match": _same_float(fire.get(key), receipt.get(key)),
        }
    mismatches = [key for key, row in comparisons.items() if not row["match"]]
    if mismatches:
        return False, "receipt_config_mismatch", {"mismatches": mismatches, "comparisons": comparisons}
    return True, "config_ok", {"comparisons": comparisons}


def _validate_host(receipt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    receipt_host = receipt.get("host")
    current = _host_fingerprint()
    if not isinstance(receipt_host, dict):
        return False, "receipt_host_missing", {"current_host": current}
    if receipt_host.get("node") != current["node"] or receipt_host.get("machine") != current["machine"]:
        return False, "receipt_host_mismatch", {"receipt_host": receipt_host, "current_host": current}
    return True, "host_ok", {"receipt_host": receipt_host, "current_host": current}


def evaluate_guard(ticket_path: Path, argv_key: str) -> dict[str, Any]:
    ticket = _ticket_payload(_load_json(ticket_path))
    checks: list[dict[str, Any]] = []
    argv = ticket.get(argv_key)
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return _verdict("failed", "argv_key_missing_or_not_string_list", ticket_path, argv_key, checks)
    if not ticket.get("mem_probe_receipt_required"):
        return _verdict("failed", "ticket_mem_probe_not_required", ticket_path, argv_key, checks)

    keyed_receipts = ticket.get("mem_probe_receipt_paths")
    if isinstance(keyed_receipts, dict) and argv_key in keyed_receipts:
        receipt_path_raw = keyed_receipts[argv_key]
    else:
        receipt_path_raw = ticket.get("mem_probe_receipt_path")
    receipt_path = Path(str(receipt_path_raw)) if receipt_path_raw else None
    if receipt_path is not None and not receipt_path.is_absolute():
        receipt_path = (REPO / receipt_path).resolve()
    if receipt_path is None:
        return _verdict("failed", "ticket_mem_probe_receipt_path_missing", ticket_path, argv_key, checks)
    if not receipt_path.exists():
        return _verdict(
            "failed",
            "mem_probe_receipt_missing",
            ticket_path,
            argv_key,
            checks,
            receipt_path=receipt_path,
        )
    ok, reason, detail = _validate_receipt_freshness(receipt_path)
    checks.append({"name": "receipt_freshness", "status": "passed" if ok else "failed", "reason": reason, "detail": detail})
    if not ok:
        return _verdict("failed", reason, ticket_path, argv_key, checks, receipt_path=receipt_path)

    try:
        receipt = _load_json(receipt_path)
    except Exception as exc:
        return _verdict(
            "failed",
            "mem_probe_receipt_parse_error",
            ticket_path,
            argv_key,
            checks,
            receipt_path=receipt_path,
            error=f"{type(exc).__name__}: {exc}",
        )
    checks.append({"name": "receipt_parse", "status": "passed"})

    if receipt.get("schema") != MEM_PROBE_RECEIPT_SCHEMA:
        return _verdict("failed", "receipt_schema_mismatch", ticket_path, argv_key, checks, receipt_path=receipt_path)
    checks.append({"name": "receipt_schema", "status": "passed"})
    if receipt.get("status") != "passed" or receipt.get("metal_fire_clearance") is not True:
        return _verdict("failed", "receipt_status_not_clearance", ticket_path, argv_key, checks, receipt_path=receipt_path)
    checks.append({"name": "receipt_status", "status": "passed"})

    for name, validator in (
        ("host", _validate_host),
        ("samples", _validate_samples),
        ("memory_limits", _validate_memory_limits),
    ):
        ok, reason, detail = validator(receipt)
        checks.append({"name": name, "status": "passed" if ok else "failed", "reason": reason, "detail": detail})
        if not ok:
            return _verdict("failed", reason, ticket_path, argv_key, checks, receipt_path=receipt_path)

    fire_config = _parsed_fire_config(argv)
    if fire_config.get("fire_argv_key") not in (None, argv_key):
        return _verdict(
            "failed",
            "fire_argv_key_mismatch",
            ticket_path,
            argv_key,
            checks,
            receipt_path=receipt_path,
            fire_config=fire_config,
            receipt_config=_receipt_config(receipt),
        )
    ticket_from_argv = fire_config.get("launch_ticket_path")
    if ticket_from_argv is not None and _norm_path(ticket_from_argv) != _norm_path(ticket_path):
        return _verdict(
            "failed",
            "fire_launch_ticket_path_mismatch",
            ticket_path,
            argv_key,
            checks,
            receipt_path=receipt_path,
            fire_config=fire_config,
            receipt_config=_receipt_config(receipt),
        )
    receipt_config = _receipt_config(receipt)
    ok, reason, detail = _validate_config_match(fire_config, receipt_config)
    checks.append({
        "name": "config_match",
        "status": "passed" if ok else "failed",
        "reason": reason,
        "detail": detail,
    })
    if not ok:
        return _verdict(
            "failed",
            reason,
            ticket_path,
            argv_key,
            checks,
            receipt_path=receipt_path,
            fire_config=fire_config,
            receipt_config=receipt_config,
        )

    return _verdict(
        "passed",
        "fire_guard_passed",
        ticket_path,
        argv_key,
        checks,
        receipt_path=receipt_path,
        fire_config=fire_config,
        receipt_config=receipt_config,
    )


def _verdict(
    status: str,
    reason_code: str,
    ticket_path: Path,
    argv_key: str,
    checks: list[dict[str, Any]],
    *,
    receipt_path: Path | None = None,
    fire_config: dict[str, Any] | None = None,
    receipt_config: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": FIRE_GUARD_VERDICT_SCHEMA,
        "status": status,
        "reason_code": reason_code,
        "reason": reason_code,
        "axis": "[apparatus / scorer-free]",
        "score_claim": False,
        "timestamp_utc": _utc_now_iso(),
        "ticket_path": str(ticket_path),
        "argv_key": argv_key,
        "receipt_path": None if receipt_path is None else str(receipt_path),
        "checks": checks,
        "fire_config": fire_config,
        "receipt_config": receipt_config,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--argv-key", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    out = args.out or _default_verdict_path(args.ticket, args.argv_key)
    try:
        verdict = evaluate_guard(args.ticket, args.argv_key)
    except Exception as exc:
        verdict = _verdict(
            "failed",
            "guard_internal_error",
            args.ticket,
            args.argv_key,
            [],
            error=f"{type(exc).__name__}: {exc}",
        )
    verdict["verdict_path"] = str(out)
    _write_json_atomic(out, verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    return 0 if verdict["status"] == "passed" else EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
