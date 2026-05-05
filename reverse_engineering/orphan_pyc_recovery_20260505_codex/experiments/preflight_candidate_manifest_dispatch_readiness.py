#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob 314f65501f759b128d6b575ec17aec4703e14b5e via `git fsck --lost-found`
# original path: experiments/preflight_candidate_manifest_dispatch_readiness.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Fail-closed preflight for candidate manifest dispatch readiness.

This is a cheap local guard for the gap between builder-specific readiness
fields and the narrower Lightning submit-time ``exact_eval_dispatch_gate``.
Run it on a candidate ``manifest.json`` before any lane claim or exact-eval
queue command.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "candidate_manifest_dispatch_readiness_preflight_v1"
BLOCKING_TEXT_MARKERS = (
    "blocked",
    "fail_closed",
    "failed",
    "missing",
    "non_dispatchable",
    "not_run",
    "planning_only",
    "no_remote_dispatch",
    "local_only",
    "invalid",
    "negative",
)
READY_TEXT_MARKERS = (
    "eligible_for_exact_eval_after_lane_claim",
    "eligible_for_cuda_auth_eval_after_lane_claim",
    "ready",
    "passed",
)
FALSE_READY_FIELDS = (
    "dispatch_unlocked",
    "ready_for_exact_eval_after_lane_claim",
    "ready_for_exact_eval_dispatch_claim",
    "ready_for_exact_eval_dispatch",
    "ready_for_fixed_runtime_exact_eval",
    "safe_for_exact_eval_dispatch",
    "safe_for_remote_dispatch",
    "dispatchable",
    "dispatch_ready_now",
)
RUNTIME_CHANGING_MARKERS = (
    "stbm1br",
    "hpm1",
    "hpac",
    "qfq4",
    "qma9",
    "qh0",
    "qps1",
    "qzs3",
    "qrgb",
)
FORMULA_ONLY_KEY_MARKERS = (
    "formula_only",
    "if_components_identical",
    "if_archive_is_valid",
)
RUNTIME_SUPPORT_FALSE_KEYS = (
    "runtime_can_decode_without_edits",
    "runtime_can_decode",
    "public_pr85_replay_qfq4_model_loader",
    "robust_current_qfq4_renderer_loader",
)
DECODE_PARITY_FALSE_KEYS = (
    "decoded_tensor_parity",
    "local_decode_byte_parity_proven",
    "full_decode_byte_parity_proven",
    "byte_parity_achieved",
    "pr91_ready_for_exact_eval",
)
TERMINAL_CLAIM_STATUS_PREFIXES = (
    "completed_",
    "completed_score=",
    "completed_no_frontier",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"manifest must be a JSON object: {path}")
    return payload


def _stringify_items(items: Any) -> list[str]:
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return [str(item) for item in items]
    if items in (None, ""):
        return []
    return [str(items)]


def _has_blocking_text(value: str) -> bool:
    text = value.lower()
    return any(marker in text for marker in BLOCKING_TEXT_MARKERS)


def _has_ready_text(value: str) -> bool:
    text = value.lower()
    return any(marker in text for marker in READY_TEXT_MARKERS)


def _append_blocker(blockers: list[dict[str, Any]], code: str, detail: str) -> None:
    blockers.append({"code": code, "severity": "blocking", "detail": detail})


def _contains_runtime_changing_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_runtime_changing_marker(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_runtime_changing_marker(item) for item in value)
    text = str(value).lower()
    return any(marker in text for marker in RUNTIME_CHANGING_MARKERS)


def _requires_exact_runtime_contract(manifest: Mapping[str, Any]) -> bool:
    fail_closed = manifest.get("fail_closed_preflight")
    if isinstance(fail_closed, Mapping):
        if fail_closed.get("exact_eval_requires_explicit_pr85_replay_runtime") is True:
            return True
        if fail_closed.get("exact_eval_requires_submitted_runtime_contract") is True:
            return True
    for key in (
        "candidate_id",
        "policy_id",
        "tool",
        "runtime_support",
        "source_policy",
        "segments",
        "candidate_bundle",
    ):
        if key in manifest and _contains_runtime_changing_marker(manifest.get(key)):
            return True
    return False


def _iter_leaf_values(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_iter_leaf_values(item, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.extend(_iter_leaf_values(item, prefix=f"{prefix}[{index}]"))
    else:
        rows.append((prefix, value))
    return rows


def _leaf_key(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _is_nonzero_formula_value(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _is_explanatory_rate_formula_path(path: str, manifest: Mapping[str, Any]) -> bool:
    lower_path = path.lower()
    if not lower_path.startswith("formula_only_rate_delta_"):
        return False
    if manifest.get("score_claim") is not False:
        return False
    readiness = manifest.get("dispatch_readiness")
    if not isinstance(readiness, Mapping):
        return False
    checks = readiness.get("checks")
    if not isinstance(checks, Mapping):
        return False
    exact_runtime = manifest.get("exact_eval_runtime_contract")
    if not isinstance(exact_runtime, Mapping):
        return False
    if exact_runtime.get("ready_for_exact_eval_runtime") is not True:
        return False
    return any(
        checks.get(key) is True
        for key in (
            "strict_zip_single_member_x",
            "candidate_changes_only_randmulti_vs_stbm",
            "randmulti_decoded_rows_match_stbm",
            "candidate_score_claim_false",
        )
    )


def _audit_recursive_fail_closed_fields(
    manifest: Mapping[str, Any],
    *,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    for path, value in _iter_leaf_values(manifest):
        lower_path = path.lower()
        key = _leaf_key(lower_path)
        if any(marker in lower_path for marker in FORMULA_ONLY_KEY_MARKERS) and _is_nonzero_formula_value(value):
            if _is_explanatory_rate_formula_path(path, manifest):
                warnings.append(
                    {
                        "code": "formula_only_rate_summary_non_dispatching",
                        "detail": f"{path} is recorded as a non-score explanatory rate projection",
                    }
                )
                continue
            _append_blocker(
                blockers,
                "formula_only_dispatch_evidence",
                f"{path} is formula-only and cannot unlock exact-eval dispatch",
            )
            continue
        if key in RUNTIME_SUPPORT_FALSE_KEYS and value is False:
            _append_blocker(
                blockers,
                "runtime_loader_support_false",
                f"{path} is false",
            )
            continue
        if key in DECODE_PARITY_FALSE_KEYS and value is False:
            _append_blocker(
                blockers,
                "decode_or_exact_parity_false",
                f"{path} is false",
            )
            continue
        if key in {"status", "readiness_status", "failure_reason", "blocker_class"}:
            if isinstance(value, str) and _has_blocking_text(value):
                _append_blocker(
                    blockers,
                    "nested_blocking_status",
                    f"{path}={value!r}",
                )


def _audit_gate_mapping(
    gate: Mapping[str, Any],
    *,
    path: str,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    required = gate.get("required")
    if required is True and gate.get("safe_for_exact_eval_dispatch") is not True:
        _append_blocker(
            blockers,
            f"{path}:unsafe_required_exact_eval_gate",
            f"{path} requires safe_for_exact_eval_dispatch=true",
        )
    for key in FALSE_READY_FIELDS:
        if key in gate and gate[key] is False:
            _append_blocker(
                blockers,
                f"{path}:{key}_false",
                f"{path}.{key} is false",
            )
    for key in ("status", "dispatch_gate", "readiness_status", "next_dispatch_gate"):
        value = gate.get(key)
        if isinstance(value, str) and _has_blocking_text(value):
            _append_blocker(
                blockers,
                f"{path}:{key}_blocked",
                f"{path}.{key}={value!r}",
            )
    nested_blockers = _stringify_items(gate.get("blockers") or gate.get("remaining_blockers"))
    if nested_blockers:
        _append_blocker(
            blockers,
            f"{path}:reported_blockers",
            f"{path} reports blockers: {', '.join(nested_blockers[:8])}",
        )
    if (
        not nested_blockers
        and gate.get("required") is not True
        and not any(key in gate for key in FALSE_READY_FIELDS)
    ):
        warnings.append(
            {
                "code": f"{path}:nonstandard_gate_shape",
                "detail": f"{path} has no recognized readiness boolean",
            }
        )


def _parse_utc(value: str) -> dt.datetime | None:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _is_terminal_claim_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_STATUS_PREFIXES)


def _parse_claim_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[0] in {"timestamp_utc", "---"}:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _manifest_lane_id(manifest: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [manifest.get("lane_id")]
    for key in ("dispatch_readiness", "exact_eval_dispatch_gate", "dispatch_gate", "dispatch"):
        section = manifest.get(key)
        if isinstance(section, Mapping):
            candidates.append(section.get("lane_id"))
            claim = section.get("claim")
            if isinstance(claim, Mapping):
                candidates.append(claim.get("lane_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _active_lane_claims(
    claims: Sequence[Mapping[str, str]],
    *,
    lane_id: str,
    now: dt.datetime,
    ttl_hours: float,
) -> list[dict[str, str]]:
    cutoff = now - dt.timedelta(hours=ttl_hours)
    conflicts: list[dict[str, str]] = []
    closed_instance_job_ids: set[str] = set()
    for claim in claims:
        if claim.get("lane_id") != lane_id:
            continue
        status = str(claim.get("status", ""))
        instance_job_id = str(claim.get("instance_job_id", ""))
        if _is_terminal_claim_status(status):
            closed_instance_job_ids.add(instance_job_id)
            continue
        if instance_job_id in closed_instance_job_ids:
            continue
        timestamp = _parse_utc(str(claim.get("timestamp_utc", "")))
        if timestamp is None or timestamp >= cutoff:
            conflicts.append(dict(claim))
    return conflicts


def _lane_claim_report(
    manifest: Mapping[str, Any],
    *,
    claims_path: Path | None,
    now_utc: str | None,
    ttl_hours: float,
) -> dict[str, Any]:
    lane_id = _manifest_lane_id(manifest)
    if claims_path is None:
        return {
            "checked": False,
            "lane_id": lane_id,
            "claims_path": None,
            "active_conflicts": [],
        }
    now = _parse_utc(now_utc) if now_utc else dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if now is None:
        raise SystemExit(f"invalid --now-utc: {now_utc}")
    claims = _parse_claim_rows(claims_path)
    conflicts = _active_lane_claims(claims, lane_id=lane_id, now=now, ttl_hours=ttl_hours) if lane_id else []
    return {
        "checked": True,
        "lane_id": lane_id,
        "claims_path": str(claims_path),
        "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": ttl_hours,
        "active_conflicts": conflicts,
        "claim_count": len(claims),
    }


def build_preflight(
    manifest_path: Path,
    *,
    claims_path: Path | None = None,
    now_utc: str | None = None,
    ttl_hours: float = 24.0,
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    gate = manifest.get("exact_eval_dispatch_gate")
    if isinstance(gate, Mapping):
        _audit_gate_mapping(
            gate,
            path="exact_eval_dispatch_gate",
            blockers=blockers,
            warnings=warnings,
        )

    dispatch_gate = manifest.get("dispatch_gate")
    if isinstance(dispatch_gate, str):
        if _has_blocking_text(dispatch_gate):
            _append_blocker(
                blockers,
                "dispatch_gate_blocked",
                f"dispatch_gate={dispatch_gate!r}",
            )
        elif _has_ready_text(dispatch_gate):
            warnings.append(
                {
                    "code": "lane_claim_still_required",
                    "detail": f"dispatch_gate={dispatch_gate!r}; verify active Level-2 lane claim before GPU dispatch",
                }
            )
        else:
            warnings.append(
                {
                    "code": "unrecognized_dispatch_gate_text",
                    "detail": f"dispatch_gate={dispatch_gate!r}",
                }
            )
    elif isinstance(dispatch_gate, Mapping):
        _audit_gate_mapping(
            dispatch_gate,
            path="dispatch_gate",
            blockers=blockers,
            warnings=warnings,
        )

    for key in FALSE_READY_FIELDS:
        if key in manifest and manifest[key] is False:
            _append_blocker(blockers, f"{key}_false", f"{key} is false")

    for key in ("exact_eval_readiness_status", "readiness_status", "build_status"):
        value = manifest.get(key)
        if isinstance(value, str) and _has_blocking_text(value):
            _append_blocker(blockers, f"{key}_blocked", f"{key}={value!r}")

    _audit_recursive_fail_closed_fields(manifest, blockers=blockers, warnings=warnings)

    for path, key in (
        ("fail_closed_preflight", "remaining_exact_eval_blockers"),
        ("exact_eval_runtime_contract", "remaining_blockers"),
        ("runtime_gate", "remaining_blockers"),
        ("fixed_runtime_bridge", "remaining_blockers"),
    ):
        section = manifest.get(path)
        if isinstance(section, Mapping):
            nested = _stringify_items(section.get(key))
            if nested:
                _append_blocker(
                    blockers,
                    f"{path}:{key}",
                    f"{path}.{key}: {', '.join(nested[:8])}",
                )

    exact_runtime = manifest.get("exact_eval_runtime_contract")
    if _requires_exact_runtime_contract(manifest) and not isinstance(exact_runtime, Mapping):
        _append_blocker(
            blockers,
            "exact_eval_runtime_contract:missing_for_runtime_changing_candidate",
            "runtime-changing candidate must record the exact submitted inflate runtime contract",
        )
    if isinstance(exact_runtime, Mapping) and exact_runtime.get("ready_for_exact_eval_runtime") is False:
        _append_blocker(
            blockers,
            "exact_eval_runtime_contract:not_ready",
            "exact eval runtime contract is not ready",
        )

    fixed_runtime = manifest.get("fixed_runtime_preflight")
    if isinstance(fixed_runtime, Mapping):
        ready = fixed_runtime.get("ready_for_fixed_runtime_exact_eval")
        ready = fixed_runtime.get("ready_for_fixed_runtime_exact_eval_readiness", ready)
        if ready is False:
            _append_blocker(
                blockers,
                "fixed_runtime_preflight:not_ready",
                "fixed runtime preflight is not ready",
            )

    if str(manifest.get("evidence_grade", "")).lower().startswith("a-negative"):
        _append_blocker(
            blockers,
            "evidence_grade_exact_negative",
            "manifest evidence_grade is exact-negative",
        )
    if manifest.get("exact_negative") is True:
        _append_blocker(blockers, "exact_negative_true", "manifest is marked exact_negative=true")

    lane_claim = _lane_claim_report(
        manifest,
        claims_path=claims_path,
        now_utc=now_utc,
        ttl_hours=ttl_hours,
    )
    if lane_claim["checked"] and not lane_claim.get("lane_id"):
        _append_blocker(
            blockers,
            "lane_claim_missing_lane_id",
            "claims-path was supplied but manifest did not expose a lane_id",
        )
    for conflict in lane_claim.get("active_conflicts", []):
        _append_blocker(
            blockers,
            "active_lane_claim_conflict",
            (
                "active same-lane claim exists: "
                f"{conflict.get('timestamp_utc')} {conflict.get('agent')} "
                f"{conflict.get('platform')} {conflict.get('instance_job_id')} "
                f"status={conflict.get('status')}"
            ),
        )

    return {
        "schema": SCHEMA,
        "manifest_path": str(manifest_path),
        "candidate_id": manifest.get("candidate_id"),
        "ready_for_exact_eval_dispatch": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "lane_claim": lane_claim,
        "score_claim": bool(manifest.get("score_claim", False)),
        "dispatch_performed": bool(manifest.get("dispatch_performed", False)),
        "remote_jobs_dispatched": bool(manifest.get("remote_jobs_dispatched", False)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claims-path", type=Path)
    parser.add_argument("--now-utc")
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    payload = build_preflight(
        args.manifest,
        claims_path=args.claims_path,
        now_utc=args.now_utc,
        ttl_hours=args.ttl_hours,
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.fail_if_not_ready and not payload["ready_for_exact_eval_dispatch"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
