#!/usr/bin/env python3
"""Harvest paired contest-CPU/CUDA auth-eval artifacts into the drift registry.

The per-architecture CUDA/CPU profile registry is a learning system, but it
must only learn from paired, custody-compatible artifacts. This tool accepts
either:

* one or more combined JSON payloads that already contain ``cpu`` and ``cuda``
  sections; or
* explicit ``--pair CPU_JSON CUDA_JSON`` paths; or
* a ``--scan-root`` tree where individual auth-eval JSONs are grouped by
  archive SHA, archive bytes, and runtime-tree SHA.

It never promotes, ranks, kills, or dispatches. It writes a planning-only
registry update report and delegates the actual posterior update to
``tac.optimization.cuda_cpu_axis_profile_registry``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.cuda_cpu_axis_profile_registry import (
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_REGISTRY_PATH,
    ArchitectureProfile,
    harvest_new_anchor_and_update,
    read_registry,
    serialize_registry,
    write_registry,
)

try:
    from tools.auth_eval_records import AuthEvalRecord, parse_auth_eval_payload
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from auth_eval_records import AuthEvalRecord, parse_auth_eval_payload


AUTH_EVAL_GLOB = "contest_auth_eval*.json"


@dataclass(frozen=True)
class AxisArtifact:
    """Parsed single-axis auth-eval artifact."""

    path: Path
    payload: dict[str, Any]
    record: AuthEvalRecord


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _runtime_tree_sha256(payload: dict[str, Any]) -> str | None:
    manifest = payload.get("inflate_runtime_manifest")
    candidates = (
        payload.get("runtime_content_tree_sha256"),
        payload.get("inflate_runtime_content_tree_sha256"),
        _nested_get(payload, "provenance", "runtime_content_tree_sha256"),
        _nested_get(
            payload,
            "provenance",
            "inflate_runtime_manifest",
            "runtime_content_tree_sha256",
        ),
        manifest.get("runtime_content_tree_sha256") if isinstance(manifest, dict) else None,
        payload.get("runtime_tree_sha256"),
        payload.get("inflate_runtime_tree_sha256"),
        _nested_get(payload, "provenance", "runtime_tree_sha256"),
        _nested_get(payload, "provenance", "inflate_runtime_manifest", "runtime_tree_sha256"),
        manifest.get("runtime_tree_sha256") if isinstance(manifest, dict) else None,
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _hardware_label(payload: dict[str, Any], record: AuthEvalRecord) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    value = (
        payload.get("hardware")
        or provenance.get("hardware")
        or payload.get("runner_label")
        or provenance.get("runner_label")
        or record.evidence_grade
    )
    return str(value or record.device or record.score_axis)


def _parse_axis_artifact(path: Path) -> tuple[AxisArtifact | None, str | None]:
    try:
        payload = _load_json(path)
    except Exception as exc:
        return None, f"{path}: unreadable_json:{type(exc).__name__}"
    record = parse_auth_eval_payload(payload)
    if record is None:
        return None, f"{path}: unparseable_auth_eval"
    return AxisArtifact(path=path, payload=payload, record=record), None


def _axis_blockers(axis: AxisArtifact, *, expected_axis: str) -> list[str]:
    record = axis.record
    blockers: list[str] = []
    if record.score_axis != expected_axis:
        blockers.append(
            f"{axis.path}: expected {expected_axis}, got {record.score_axis}"
        )
    if record.score is None:
        blockers.append(f"{axis.path}: missing_score")
    if record.archive_sha256 is None:
        blockers.append(f"{axis.path}: missing_archive_sha256")
    if record.archive_bytes is None:
        blockers.append(f"{axis.path}: missing_archive_bytes")
    if record.avg_segnet_dist is None:
        blockers.append(f"{axis.path}: missing_avg_segnet_dist")
    if record.avg_posenet_dist is None:
        blockers.append(f"{axis.path}: missing_avg_posenet_dist")
    if record.samples != 600:
        blockers.append(f"{axis.path}: expected_600_samples_got_{record.samples}")
    if _runtime_tree_sha256(axis.payload) is None:
        blockers.append(f"{axis.path}: missing_runtime_tree_sha256")
    if record.hardware_compliance_blocker:
        blockers.append(
            f"{axis.path}: hardware_compliance_blocker={record.hardware_compliance_blocker}"
        )
    return blockers


def build_combined_payload_from_pair(
    *,
    cpu_json: Path,
    cuda_json: Path,
    architecture_class: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a combined payload accepted by the profile registry.

    The pair must be full-sample ``contest_cpu`` + ``contest_cuda`` on the
    same archive SHA, archive bytes, and runtime-tree SHA. Mismatches are
    returned as blockers instead of raising so batch harvests can preserve
    partial signal.
    """

    cpu_axis, cpu_error = _parse_axis_artifact(cpu_json)
    cuda_axis, cuda_error = _parse_axis_artifact(cuda_json)
    blockers = [err for err in (cpu_error, cuda_error) if err]
    if cpu_axis is None or cuda_axis is None:
        return None, blockers

    blockers.extend(_axis_blockers(cpu_axis, expected_axis="contest_cpu"))
    blockers.extend(_axis_blockers(cuda_axis, expected_axis="contest_cuda"))
    if blockers:
        return None, blockers

    cpu = cpu_axis.record
    cuda = cuda_axis.record
    cpu_runtime = _runtime_tree_sha256(cpu_axis.payload)
    cuda_runtime = _runtime_tree_sha256(cuda_axis.payload)
    if cpu.archive_sha256 != cuda.archive_sha256:
        blockers.append("cpu_cuda_archive_sha256_mismatch")
    if cpu.archive_bytes != cuda.archive_bytes:
        blockers.append("cpu_cuda_archive_bytes_mismatch")
    if cpu_runtime != cuda_runtime:
        blockers.append("cpu_cuda_runtime_tree_sha256_mismatch")
    if blockers:
        return None, blockers

    payload: dict[str, Any] = {
        "schema": "paired_cuda_cpu_axis_profile_anchor.v1",
        "source": "paired_contest_auth_eval_artifacts",
        "source_paths": {
            "cpu": str(cpu_json),
            "cuda": str(cuda_json),
        },
        "architecture_class": architecture_class,
        "archive_bytes": int(cpu.archive_bytes or 0),
        "archive_sha256": str(cpu.archive_sha256 or ""),
        "runtime_tree_sha256": str(cpu_runtime or ""),
        "sample_count": int(cpu.samples or 0),
        "cpu": {
            "score": float(cpu.score or 0.0),
            "pose": float(cpu.avg_posenet_dist or 0.0),
            "seg": float(cpu.avg_segnet_dist or 0.0),
            "hardware": _hardware_label(cpu_axis.payload, cpu),
            "score_axis": cpu.score_axis,
            "evidence_grade": cpu.evidence_grade,
        },
        "cuda": {
            "score": float(cuda.score or 0.0),
            "pose": float(cuda.avg_posenet_dist or 0.0),
            "seg": float(cuda.avg_segnet_dist or 0.0),
            "hardware": _hardware_label(cuda_axis.payload, cuda),
            "score_axis": cuda.score_axis,
            "evidence_grade": cuda.evidence_grade,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return payload, []


def _scan_axis_artifacts(root: Path) -> tuple[list[AxisArtifact], list[str]]:
    artifacts: list[AxisArtifact] = []
    blockers: list[str] = []
    for path in sorted(root.rglob(AUTH_EVAL_GLOB)):
        parsed, error = _parse_axis_artifact(path)
        if parsed is None:
            if error:
                blockers.append(error)
            continue
        if parsed.record.score_axis in {"contest_cpu", "contest_cuda"}:
            artifacts.append(parsed)
    return artifacts, blockers


def _pairing_key(axis: AxisArtifact) -> tuple[str, int, str] | None:
    record = axis.record
    runtime_tree = _runtime_tree_sha256(axis.payload)
    if record.archive_sha256 and record.archive_bytes is not None and runtime_tree:
        return (record.archive_sha256, int(record.archive_bytes), runtime_tree)
    return None


def discover_pairs(scan_roots: Iterable[Path]) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Find same-archive contest-CPU/CUDA pairs under ``scan_roots``."""

    grouped: dict[tuple[str, int, str], dict[str, AxisArtifact]] = {}
    blockers: list[str] = []
    for root in scan_roots:
        artifacts, scan_blockers = _scan_axis_artifacts(root)
        blockers.extend(scan_blockers)
        for axis in artifacts:
            key = _pairing_key(axis)
            if key is None:
                blockers.append(f"{axis.path}: missing_pairing_key")
                continue
            bucket = grouped.setdefault(key, {})
            current = bucket.get(axis.record.score_axis)
            if current is None or axis.path.stat().st_mtime >= current.path.stat().st_mtime:
                bucket[axis.record.score_axis] = axis

    pairs: list[tuple[Path, Path]] = []
    for bucket in grouped.values():
        cpu = bucket.get("contest_cpu")
        cuda = bucket.get("contest_cuda")
        if cpu is not None and cuda is not None:
            pairs.append((cpu.path, cuda.path))
    return pairs, blockers


def harvest_registry(
    *,
    combined_payloads: Iterable[dict[str, Any]],
    pairs: Iterable[tuple[Path, Path]],
    registry_path: Path,
    audit_log_path: Path | None,
    architecture_class: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Harvest payloads and return a machine-readable non-promotable report."""

    registry: dict[str, ArchitectureProfile] = read_registry(registry_path)
    updates: list[dict[str, Any]] = []
    blockers: list[str] = []

    for payload in combined_payloads:
        update = harvest_new_anchor_and_update(
            payload,
            registry=registry,
            architecture_class=architecture_class or payload.get("architecture_class"),
            audit_log_path=None if dry_run else audit_log_path,
        )
        if update is None:
            blockers.append(
                f"{payload.get('source', 'combined_payload')}: incomplete_pair_or_custody"
            )
            continue
        updates.append(update.to_dict())

    for cpu_json, cuda_json in pairs:
        payload, pair_blockers = build_combined_payload_from_pair(
            cpu_json=cpu_json,
            cuda_json=cuda_json,
            architecture_class=architecture_class,
        )
        if payload is None:
            blockers.extend(pair_blockers)
            continue
        update = harvest_new_anchor_and_update(
            payload,
            registry=registry,
            architecture_class=architecture_class or payload.get("architecture_class"),
            audit_log_path=None if dry_run else audit_log_path,
        )
        if update is None:
            blockers.append(f"{cpu_json} + {cuda_json}: incomplete_pair_or_custody")
            continue
        updates.append(update.to_dict())

    if not dry_run and updates:
        write_registry(registry, registry_path)

    return {
        "schema": "cuda_cpu_axis_profile_registry_harvest_report.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "registry_path": str(registry_path),
        "audit_log_path": str(audit_log_path) if audit_log_path else None,
        "dry_run": dry_run,
        "update_count": len(updates),
        "updates": updates,
        "blockers": blockers,
        "registry_snapshot": serialize_registry(registry),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "notes": [
            "This tool only updates a CPU/CUDA drift learning registry.",
            "Pairs must be full-sample contest-CPU and contest-CUDA artifacts for the same archive/runtime custody.",
            "Outlier anchors stay in the audit trail and do not auto-promote into the posterior.",
        ],
    }


def _load_combined_payloads(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [_load_json(path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG_PATH)
    parser.add_argument("--combined-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--pair",
        type=Path,
        nargs=2,
        action="append",
        metavar=("CPU_JSON", "CUDA_JSON"),
        default=[],
        help="Explicit paired contest-CPU and contest-CUDA auth-eval JSONs.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[],
        help="Search for contest_auth_eval*.json files and pair by archive/runtime SHA.",
    )
    parser.add_argument("--architecture-class")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = [tuple(pair) for pair in args.pair]
    scan_blockers: list[str] = []
    if args.scan_root:
        discovered, scan_blockers = discover_pairs(args.scan_root)
        pairs.extend(discovered)

    report = harvest_registry(
        combined_payloads=_load_combined_payloads(args.combined_json),
        pairs=pairs,
        registry_path=args.registry,
        audit_log_path=args.audit_log,
        architecture_class=args.architecture_class,
        dry_run=args.dry_run,
    )
    report["scan_blockers"] = scan_blockers
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
