# SPDX-License-Identifier: MIT
"""Exact-gate bridge for HPRC incremental runner evidence."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_INCREMENTAL_EXACT_GATE_BRIDGE_SCHEMA = (
    "hprc_incremental_exact_gate_bridge.v1"
)


def build_hprc_incremental_exact_gate_bridge(
    *,
    execution_report_path: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return an exact-gate bridge for one HPRC incremental execution report."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    report_path = _resolve(execution_report_path, base=root)
    report = _load_json_object(report_path)
    archive = report.get("archive", {}) if isinstance(report.get("archive"), dict) else {}
    archive_path = _resolve_required_path(archive.get("path"), base=root)
    hprc_0bin_path = _resolve_required_path(archive.get("hprc_0bin_path"), base=root)
    archive_checks = _file_hash_checks(
        path=archive_path,
        expected_sha256=str(archive.get("sha256") or ""),
        expected_bytes=archive.get("bytes"),
        label="archive_zip",
    )
    hprc_0bin_checks = _file_hash_checks(
        path=hprc_0bin_path,
        expected_sha256=str(archive.get("hprc_0bin_sha256") or ""),
        expected_bytes=None,
        label="hprc_0bin",
    )
    proof_binding = (
        report.get("receiver_proof_binding")
        if isinstance(report.get("receiver_proof_binding"), dict)
        else {}
    )
    proof_checks = _receiver_proof_checks(
        proof_binding=proof_binding,
        expected_archive_sha256=str(archive.get("sha256") or ""),
        repo_root=root,
    )
    cleanup = report.get("cleanup") if isinstance(report.get("cleanup"), dict) else {}
    cleanup_checks = _cleanup_checks(cleanup)
    source_blockers = _source_exact_blockers(report)
    custody_blockers = [
        *archive_checks["blockers"],
        *hprc_0bin_checks["blockers"],
        *proof_checks["blockers"],
        *cleanup_checks["blockers"],
    ]
    preclaim_blockers = list(custody_blockers)
    preclaim_ready = not preclaim_blockers
    score_authority_blockers = _dedupe(
        [
            "contest_cpu_cuda_exact_eval_not_executed",
            "mlx_local_response_is_advisory_not_score_authority",
            *source_blockers,
        ]
    )
    ready_for_exact_eval_dispatch = bool(preclaim_ready)
    return {
        "schema": HPRC_INCREMENTAL_EXACT_GATE_BRIDGE_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "execution_report_path": report_path.as_posix(),
        "execution_report_sha256": _sha256_file(report_path),
        "candidate_id": report.get("candidate_id"),
        "candidate_variant_id": report.get("candidate_variant_id"),
        "archive": {
            "path": archive_path.as_posix() if archive_path is not None else None,
            "bytes": archive.get("bytes"),
            "sha256": archive.get("sha256"),
            "hprc_0bin_path": hprc_0bin_path.as_posix() if hprc_0bin_path is not None else None,
            "hprc_0bin_sha256": archive.get("hprc_0bin_sha256"),
        },
        "archive_custody": archive_checks,
        "hprc_0bin_custody": hprc_0bin_checks,
        "receiver_proof_custody": proof_checks,
        "cleanup_custody": cleanup_checks,
        "mlx_advisory_summary": report.get("incremental_summary"),
        "source_exact_axis_gate": report.get("exact_axis_gate"),
        "exact_dispatch_plan": {
            "schema": "hprc_incremental_exact_dispatch_plan.v1",
            "target_modes": ["contest_exact_eval"],
            "lane_id": "hprc_hierarchical_predictive_receiver_codec",
            "dispatchable_after_lane_claim": ready_for_exact_eval_dispatch,
            "preclaim_ready": preclaim_ready,
            "preclaim_blockers": preclaim_blockers,
            "score_authority_blockers_before_promotion": score_authority_blockers,
            "archive_zip_path": archive_path.as_posix() if archive_path is not None else None,
            "archive_sha256": archive.get("sha256"),
            "requires_lane_dispatch_claim": True,
            "requires_contest_cpu_cuda_axis_payload": True,
            "requires_posterior_update_after_result": True,
        },
        "exact_axis_gate": {
            "ready_for_exact_eval_dispatch": ready_for_exact_eval_dispatch,
            "blockers": _dedupe([*preclaim_blockers, *score_authority_blockers]),
        },
        "posterior_learning_hook": {
            "schema": "hprc_incremental_exact_gate_posterior_hook.v1",
            "candidate_family": "hprc_pair_scoped_residual",
            "stage": "exact_gate",
            "scope": "pair",
            "archive_sha256": archive.get("sha256"),
            "record_positive_after_exact_axis_only": True,
            "record_blocker_now": not preclaim_ready,
            "blocker_codes": preclaim_blockers,
        },
        **FALSE_AUTHORITY,
        "ready_for_exact_eval_dispatch": ready_for_exact_eval_dispatch,
    }


def write_hprc_incremental_exact_gate_bridge(
    *,
    output_path: str | Path,
    bridge: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic HPRC exact-gate bridge artifact."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bridge, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _file_hash_checks(
    *,
    path: Path | None,
    expected_sha256: str,
    expected_bytes: Any,
    label: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    exists = path is not None and path.is_file()
    actual_sha = None
    actual_bytes = None
    if exists and path is not None:
        actual_sha = _sha256_file(path)
        actual_bytes = path.stat().st_size
    else:
        blockers.append(f"{label}_missing")
    if exists and expected_sha256 and actual_sha != expected_sha256:
        blockers.append(f"{label}_sha256_mismatch")
    if exists and expected_bytes is not None and actual_bytes != int(expected_bytes):
        blockers.append(f"{label}_bytes_mismatch")
    return {
        "schema": "hprc_exact_gate_file_custody.v1",
        "label": label,
        "path": None if path is None else path.as_posix(),
        "exists": exists,
        "expected_sha256": expected_sha256 or None,
        "actual_sha256": actual_sha,
        "expected_bytes": None if expected_bytes is None else int(expected_bytes),
        "actual_bytes": actual_bytes,
        "verified": not blockers,
        "blockers": blockers,
    }


def _receiver_proof_checks(
    *,
    proof_binding: dict[str, Any],
    expected_archive_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    proof_path = _resolve_optional_path(proof_binding.get("proof_path"), base=repo_root)
    if proof_binding.get("receiver_contract_satisfied") is not True:
        blockers.append("receiver_contract_not_satisfied")
    if proof_binding.get("runtime_consumption_proof_ready") is not True:
        blockers.append("runtime_consumption_proof_not_ready")
    if proof_binding.get("archive_sha256") != expected_archive_sha256:
        blockers.append("receiver_proof_binding_archive_sha256_mismatch")
    proof_payload: dict[str, Any] = {}
    if proof_path is None or not proof_path.is_file():
        blockers.append("receiver_proof_path_missing")
    else:
        proof_payload = _load_json_object(proof_path)
        if proof_payload.get("archive_sha256") != expected_archive_sha256:
            blockers.append("receiver_proof_payload_archive_sha256_mismatch")
        if proof_payload.get("receiver_contract_satisfied") is not True:
            blockers.append("receiver_proof_payload_contract_not_satisfied")
        if proof_payload.get("runtime_consumption_proof_ready") is not True:
            blockers.append("receiver_proof_payload_runtime_not_ready")
    return {
        "schema": "hprc_exact_gate_receiver_proof_custody.v1",
        "binding_status": proof_binding.get("status"),
        "proof_path": None if proof_path is None else proof_path.as_posix(),
        "proof_sha256": None
        if proof_path is None or not proof_path.is_file()
        else _sha256_file(proof_path),
        "archive_sha256": proof_binding.get("archive_sha256"),
        "receiver_contract_satisfied": proof_binding.get("receiver_contract_satisfied") is True,
        "runtime_consumption_proof_ready": proof_binding.get("runtime_consumption_proof_ready") is True,
        "receiver_output_sha256": proof_binding.get("receiver_output_sha256"),
        "receiver_output_bytes": proof_binding.get("receiver_output_bytes"),
        "verified": not blockers,
        "blockers": _dedupe([*blockers, *proof_binding.get("blockers", [])]),
    }


def _cleanup_checks(cleanup: dict[str, Any]) -> dict[str, Any]:
    blockers = list(cleanup.get("blockers", []))
    if cleanup.get("status") != "planned":
        blockers.append(f"cleanup_status_{cleanup.get('status') or 'missing'}")
    return {
        "schema": "hprc_exact_gate_cleanup_custody.v1",
        "source_status": cleanup.get("status"),
        "plan_path": cleanup.get("plan_path"),
        "blocked_bytes_retained": int(cleanup.get("blocked_bytes_retained") or 0),
        "reclaimable_bytes": int(cleanup.get("reclaimable_bytes") or 0),
        "verified": not blockers,
        "blockers": _dedupe(blockers),
    }


def _source_exact_blockers(report: dict[str, Any]) -> list[str]:
    gate = report.get("exact_axis_gate")
    if not isinstance(gate, dict):
        return ["source_exact_axis_gate_missing"]
    blockers = gate.get("blockers")
    if not isinstance(blockers, list):
        return ["source_exact_axis_gate_blockers_missing"]
    return [str(blocker) for blocker in blockers]


def _resolve_required_path(value: Any, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve(value, base=base)


def _resolve_optional_path(value: Any, *, base: Path) -> Path | None:
    return _resolve_required_path(value, base=base)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_INCREMENTAL_EXACT_GATE_BRIDGE_SCHEMA",
    "build_hprc_incremental_exact_gate_bridge",
    "write_hprc_incremental_exact_gate_bridge",
]
