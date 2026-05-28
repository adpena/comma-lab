# SPDX-License-Identifier: MIT
"""Deterministic byte-transform executors for repair-family campaigns.

The executor emits concrete, hash-bound transform packets for the queue-owned
repair families. These packets are encoder-side candidate deltas and MLX-local
planning evidence only; they are not score, promotion, rank/kill, or exact-eval
authority.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_replay_bundle import (
    capture_safe_replay_environment,
    stable_json_sha256,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
)
from tac.repo_io import (
    ArtifactWriteError,
    json_text,
    sha256_bytes,
    sha256_file,
    write_bytes_artifact,
)

REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA = (
    "repair_family_byte_transform_execution_report.v1"
)
REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA = (
    "repair_family_byte_transform_payload.v1"
)
REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA = (
    "repair_family_byte_transform_delta.v1"
)
REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA = (
    "repair_family_byte_transform_replay_bundle.v1"
)
REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA = (
    "repair_family_exact_eval_handoff_gate.v1"
)

SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES: frozenset[str] = frozenset(
    {
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "palette_frame_asymmetry_prior",
        "frame0_k16_palette_asymmetry",
        "entropy_boundary_probe",
    }
)

_FAMILY_TRANSFORM_KINDS: Mapping[str, str] = {
    "posenet_null_bottom_decile": "posenet_null_bottom_decile_frame0_repair_packet",
    "segnet_class_region_waterfill": "segnet_class_region_waterfill_mask_packet",
    "per_region_selector_codec": "per_region_selector_codec_delta_packet",
    "palette_frame_asymmetry_prior": "frame0_k16_palette_asymmetry_transform_packet",
    "frame0_k16_palette_asymmetry": "frame0_k16_palette_asymmetry_transform_packet",
    "entropy_boundary_probe": "entropy_boundary_probe_transform_packet",
}

_FAMILY_SIGNAL_KEYS: Mapping[str, tuple[str, ...]] = {
    "posenet_null_bottom_decile": ("posenet_null_bottom_decile_pair_ids",),
    "segnet_class_region_waterfill": ("segnet_class_region_mask_ids",),
    "per_region_selector_codec": ("selector_payload_bits_per_region",),
    "palette_frame_asymmetry_prior": (
        "palette_dynamics_context",
        "repair_dynamics_palette_prior",
    ),
    "frame0_k16_palette_asymmetry": (
        "palette_dynamics_context",
        "repair_dynamics_palette_prior",
    ),
    "entropy_boundary_probe": ("entropy_boundary_probe_manifest",),
}


class RepairFamilyByteTransformExecutorError(ValueError):
    """Raised when a repair-family byte-transform executor cannot run."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _git_text(args: Sequence[str], *, repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _file_record(
    *,
    label: str,
    path: str | Path,
    repo_root: str | Path,
    required: bool = True,
) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    present = resolved.is_file()
    if required and not present:
        raise RepairFamilyByteTransformExecutorError(
            f"required artifact missing: {label}={path}"
        )
    record = {
        "label": label,
        "path": _repo_rel(resolved, repo_root),
        "present": present,
        "required": required,
    }
    if present:
        record.update({"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size})
    return record


def _artifact_path_from_statuses(manifest: Mapping[str, Any], key: str) -> str:
    replay = _mapping(manifest.get("component_response_replay"))
    for item in replay.get("local_mlx_custody_paths") or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key:
            return str(item.get("path") or "").strip()
    for item in _mapping(manifest.get("receiver_verification")).get(
        "local_mlx_custody_paths"
    ) or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key:
            return str(item.get("path") or "").strip()
    return ""


def _component_terms(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = _mapping(manifest.get("component_response_replay"))
    return _mapping(replay.get("component_response_terms"))


def _component_probe_delta(manifest: Mapping[str, Any]) -> dict[str, Any]:
    terms = _component_terms(manifest)
    segnet_delta = _safe_float(terms.get("segnet_delta_score_units"))
    posenet_delta = _safe_float(terms.get("posenet_delta_score_units"))
    combined = _safe_float(terms.get("combined_delta_score_units"))
    if combined == 0.0:
        combined = segnet_delta + posenet_delta
    replay = _mapping(manifest.get("component_response_replay"))
    axis = str(replay.get("axis_tag") or replay.get("response_axis") or "").strip()
    return {
        "schema": "repair_family_byte_transform_mlx_probe_delta.v1",
        "component_response_axis": axis or "[macOS-MLX research-signal]",
        "advisory_delta_ready": manifest.get("component_response_replayed") is True,
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "combined_delta_score_units": combined,
        "evidence_grade": replay.get("evidence_grade")
        or "local_mlx_component_response_replay_only",
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _nested_first(manifest: Mapping[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest.get(key)
    replay_terms = _component_terms(manifest)
    if key in replay_terms:
        return replay_terms.get(key)
    for container_key in (
        "fractal_optimization_scope",
        "component_response_replay",
        "receiver_verification",
        "palette_dynamics_context",
    ):
        container = _mapping(manifest.get(container_key))
        if key in container:
            return container.get(key)
    return None


def _family_signal_payload(manifest: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in _FAMILY_SIGNAL_KEYS.get(family_id, ()):
        value = _nested_first(manifest, key)
        if value is not None:
            payload[key] = value
    if family_id == "frame0_k16_palette_asymmetry" and "palette_dynamics_context" not in payload:
        value = _nested_first(manifest, "palette_dynamics_context")
        if value is not None:
            payload["palette_dynamics_context"] = value
    return payload


def _active_levels(manifest: Mapping[str, Any]) -> list[str]:
    scope = _mapping(manifest.get("fractal_optimization_scope"))
    active = _string_list(scope.get("active_levels"))
    return active or _string_list(scope.get("declared_levels"))


def _build_transform_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    family_id: str,
) -> dict[str, Any]:
    return {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_manifest_path": str(manifest_path),
        "source_manifest_schema": manifest.get("schema"),
        "materializer_id": manifest.get("materializer_id"),
        "family_id": family_id,
        "target_kind": manifest.get("target_kind"),
        "transform_kind": _FAMILY_TRANSFORM_KINDS.get(
            family_id,
            "unclassified_repair_family_transform_packet",
        ),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(manifest.get("candidate_chain_ids")),
        "entropy_position_label": manifest.get("entropy_position_label"),
        "active_entropy_stage": dict(_mapping(manifest.get("active_entropy_stage"))),
        "fractal_levels": _active_levels(manifest),
        "allocated_repair_bytes": _safe_int(
            manifest.get("allocated_repair_bytes")
            or _component_terms(manifest).get("allocated_repair_bytes")
        ),
        "family_signal_payload": _family_signal_payload(manifest, family_id),
        "mlx_local_probe_delta": _component_probe_delta(manifest),
        "receiver_contract_kind": manifest.get("receiver_contract_kind"),
        "encoder_side_only": True,
        "receiver_must_not_optimize": True,
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "deterministic_encoder_side_repair_transform_delta_only",
        "forbidden_use": "score_claim_or_budget_spend_or_receiver_optimization",
        **FALSE_AUTHORITY,
    }


def _write_transform_payload(
    *,
    payload: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    typed_response_id: str,
    allow_overwrite: bool,
) -> dict[str, Any]:
    output = _resolve(output_dir, repo_root)
    output.mkdir(parents=True, exist_ok=True)
    stem_parts = [family_id, typed_response_id or "unknown"]
    stem = "_".join(_slug(item) for item in stem_parts)
    target = output / f"{stem}_byte_transform_payload.json"
    payload_bytes = json_text(payload).encode("utf-8")
    expected_existing_sha256 = None
    skipped = False
    if target.exists() and allow_overwrite:
        existing = target.read_bytes()
        if existing == payload_bytes:
            skipped = True
        else:
            expected_existing_sha256 = sha256_file(target)
    write_result = None
    if not skipped:
        write_result = write_bytes_artifact(
            target,
            payload_bytes,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_sha256,
        )
    return {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA,
        "path": _repo_rel(target, repo_root),
        "sha256": sha256_bytes(payload_bytes),
        "bytes": len(payload_bytes),
        "skipped_identical_existing_artifact": skipped,
        "bytes_written": 0 if write_result is None else write_result.bytes_written,
        "transform_payload_schema": payload.get("schema"),
        "transform_kind": payload.get("transform_kind"),
        "family_id": family_id,
        "typed_response_id": typed_response_id or None,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _optional_archive_copy(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    archive = _mapping(manifest.get("candidate_archive"))
    path_text = str(archive.get("path") or "").strip()
    blockers: list[str] = []
    if not path_text:
        blockers.append("candidate_archive_path_missing")
        return {"materialized": False, "path": None, "sha256": None, "bytes": None}, blockers
    source = _resolve(path_text, repo_root)
    if not source.is_file():
        blockers.append("candidate_archive_file_missing")
        return {"materialized": False, "path": path_text, "sha256": None, "bytes": None}, blockers
    expected_sha = str(archive.get("sha256") or "").strip()
    actual_sha = sha256_file(source)
    if expected_sha and expected_sha != actual_sha:
        blockers.append("candidate_archive_sha256_mismatch")
        return {"materialized": False, "path": path_text, "sha256": actual_sha, "bytes": source.stat().st_size}, blockers
    output = _resolve(output_dir, repo_root) / "candidate_archive_passthrough.zip"
    if output.exists() and not allow_overwrite:
        raise ArtifactWriteError(f"refusing to overwrite existing artifact: {output}")
    if output.exists() and allow_overwrite and sha256_file(output) == actual_sha:
        skipped = True
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        skipped = False
    return {
        "materialized": True,
        "path": _repo_rel(output, repo_root),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "passthrough_copy_of_manifest_candidate_archive": True,
        "skipped_identical_existing_artifact": skipped,
    }, blockers


def _exact_eval_handoff_gate(
    *,
    manifest: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    receiver = _mapping(manifest.get("receiver_verification"))
    exact_blockers = ordered_unique(
        [
            *list(blockers),
            *(
                []
                if candidate_archive.get("materialized") is True
                else ["byte_closed_candidate_archive_missing"]
            ),
            *(
                []
                if manifest.get("receiver_contract_satisfied") is True
                and receiver.get("runtime_consumption_proof_passed") is True
                else ["archive_bound_receiver_runtime_proof_missing"]
            ),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    return {
        "schema": REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA,
        "eligible_for_exact_eval_handoff": False,
        "candidate_archive_materialized": candidate_archive.get("materialized") is True,
        "archive_bound_runtime_consumption_proof_ready": (
            manifest.get("receiver_contract_satisfied") is True
            and receiver.get("runtime_consumption_proof_passed") is True
        ),
        "component_response_axis": "[macOS-MLX research-signal]",
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "blockers": exact_blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _source_records(
    *,
    manifest_path: str | Path,
    delta: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    records = [
        _file_record(
            label="repair_family_materializer_manifest",
            path=manifest_path,
            repo_root=repo_root,
        ),
        _file_record(
            label="repair_family_byte_transform_payload",
            path=str(delta.get("path") or ""),
            repo_root=repo_root,
        ),
    ]
    replay = _mapping(manifest.get("component_response_replay"))
    for label, key in (
        ("local_mlx_response", "local_mlx_response_path"),
        ("reference_local_mlx_response", "reference_local_mlx_response_path"),
    ):
        path = str(replay.get(key) or "").strip() or _artifact_path_from_statuses(
            manifest,
            key,
        )
        if path:
            records.append(
                _file_record(label=label, path=path, repo_root=repo_root, required=False)
            )
    return records


def _build_replay_bundle(
    *,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    delta: Mapping[str, Any],
    replay_argv: Sequence[str],
    invocation_argv: Sequence[str],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root)
    source_records = _source_records(
        manifest_path=manifest_path,
        delta=delta,
        manifest=manifest,
        repo_root=repo,
    )
    hash_manifest = {
        "schema": "repair_family_byte_transform_replay_hash_manifest.v1",
        "source_records": source_records,
        "family_id": manifest.get("family_id") or manifest.get("target_kind"),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "delta_sha256": delta.get("sha256"),
        "delta_bytes": delta.get("bytes"),
        "replay_argv": list(replay_argv),
    }
    environment = capture_safe_replay_environment()
    python_context = {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
    }
    execution_context = {
        "schema": "repair_family_byte_transform_replay_execution_context.v1",
        "invocation_argv": list(invocation_argv),
        "python": python_context,
        "environment": environment,
        "git": {
            "head": _git_text(["rev-parse", "HEAD"], repo_root=repo),
            "status_short": _git_text(["status", "--short"], repo_root=repo),
        },
    }
    bundle = {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "tool": "tools/run_repair_family_byte_transform_executor.py",
        "replay_target_tool": "tools/run_repair_family_byte_transform_executor.py",
        "source_manifest_path": str(manifest_path),
        "source_manifest_schema": manifest.get("schema"),
        "family_id": manifest.get("family_id") or manifest.get("target_kind"),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "component_response_axis": "[macOS-MLX research-signal]",
        "source_records": source_records,
        "hash_manifest": hash_manifest,
        "hash_manifest_sha256": stable_json_sha256(hash_manifest),
        "source_records_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_replay_source_records.v1",
                "source_records": source_records,
            }
        ),
        "replay_argv": list(replay_argv),
        "replay_argv_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_replay_argv.v1",
                "replay_argv": list(replay_argv),
            }
        ),
        "invocation_argv": list(invocation_argv),
        "invocation_argv_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_invocation_argv.v1",
                "invocation_argv": list(invocation_argv),
            }
        ),
        "execution_context_manifest": execution_context,
        "execution_context_sha256": stable_json_sha256(execution_context),
        "environment_sha256": stable_json_sha256(environment),
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bundle,
        context="repair_family_byte_transform_replay_bundle",
    )
    return bundle


def build_repair_family_byte_transform_execution_report(
    *,
    family_materializer_manifest: Mapping[str, Any],
    family_materializer_manifest_path: str | Path,
    output_dir: str | Path,
    replay_argv: Sequence[str],
    invocation_argv: Sequence[str],
    repo_root: str | Path,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a concrete repair-family byte-transform executor."""

    if (
        family_materializer_manifest.get("schema")
        != REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    ):
        raise RepairFamilyByteTransformExecutorError(
            "repair family byte transform requires repair family materializer manifest"
        )
    require_no_truthy_authority_fields(
        family_materializer_manifest,
        context="repair_family_byte_transform_manifest",
    )
    raw_family_id = str(
        family_materializer_manifest.get("family_id")
        or family_materializer_manifest.get("target_kind")
        or ""
    ).strip()
    family_id = raw_family_id or "unclassified_repair_family"
    supported = family_id in SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES
    blockers: list[str] = [
        "repair_family_byte_transform_is_mlx_advisory_only",
        "exact_auth_eval_required_before_score_or_promotion_claim",
    ]
    if not supported:
        blockers.append(f"unsupported_repair_family_byte_transform:{family_id}")

    transform_payload = _build_transform_payload(
        manifest=family_materializer_manifest,
        manifest_path=family_materializer_manifest_path,
        family_id=family_id,
    )
    delta = _write_transform_payload(
        payload=transform_payload,
        output_dir=output_dir,
        repo_root=repo_root,
        family_id=family_id,
        typed_response_id=str(family_materializer_manifest.get("typed_response_id") or ""),
        allow_overwrite=allow_overwrite,
    )
    candidate_archive, archive_blockers = _optional_archive_copy(
        manifest=family_materializer_manifest,
        output_dir=output_dir,
        repo_root=repo_root,
        allow_overwrite=allow_overwrite,
    )
    blockers.extend(archive_blockers)
    replay_bundle = _build_replay_bundle(
        manifest_path=family_materializer_manifest_path,
        manifest=family_materializer_manifest,
        delta=delta,
        replay_argv=replay_argv,
        invocation_argv=invocation_argv,
        repo_root=repo_root,
    )
    exact_gate = _exact_eval_handoff_gate(
        manifest=family_materializer_manifest,
        candidate_archive=candidate_archive,
        blockers=blockers,
    )
    component_delta = _component_probe_delta(family_materializer_manifest)
    report = {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "materializer_id": (
            "repair_family_byte_transform_executor:"
            f"{family_id}"
        ),
        "manifest_kind": "repair_family_byte_transform_execution_report",
        "source_family_materializer_manifest_path": str(
            family_materializer_manifest_path
        ),
        "source_family_materializer_manifest_schema": (
            family_materializer_manifest.get("schema")
        ),
        "target_kind": family_materializer_manifest.get("target_kind"),
        "family_id": family_id,
        "typed_response_id": family_materializer_manifest.get("typed_response_id"),
        "candidate_chain_id": family_materializer_manifest.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(
            family_materializer_manifest.get("candidate_chain_ids")
        ),
        "repair_budget_candidate_chain_id": family_materializer_manifest.get(
            "repair_budget_candidate_chain_id"
        ),
        "repair_budget_candidate_chain_ids": _string_list(
            family_materializer_manifest.get("repair_budget_candidate_chain_ids")
        ),
        "entropy_position_label": family_materializer_manifest.get(
            "entropy_position_label"
        ),
        "active_entropy_stage": dict(
            _mapping(family_materializer_manifest.get("active_entropy_stage"))
        ),
        "fractal_optimization_scope": dict(
            _mapping(family_materializer_manifest.get("fractal_optimization_scope"))
        ),
        "byte_transform_supported": supported,
        "byte_transform_delta_emitted": True,
        "byte_transform_delta": delta,
        "candidate_delta": delta,
        "candidate_archive": candidate_archive,
        "byte_closed_candidate_emitted": candidate_archive.get("materialized") is True,
        "candidate_archive_materialized": candidate_archive.get("materialized") is True,
        "runtime_consumption_proof_path": family_materializer_manifest.get(
            "runtime_consumption_proof_path"
        ),
        "receiver_contract_kind": family_materializer_manifest.get(
            "receiver_contract_kind"
        ),
        "receiver_contract_satisfied": (
            family_materializer_manifest.get("receiver_contract_satisfied") is True
        ),
        "component_response_replayed": (
            family_materializer_manifest.get("component_response_replayed") is True
        ),
        "component_response_replay": dict(
            _mapping(family_materializer_manifest.get("component_response_replay"))
        ),
        "mlx_local_probe_delta": component_delta,
        "component_response_replay_axis_tag": component_delta["component_response_axis"],
        "component_response_replay_path": _mapping(
            family_materializer_manifest.get("component_response_replay")
        ).get("artifact_path"),
        "exact_eval_handoff_gate": exact_gate,
        "exact_eval_handoff_eligible": False,
        "replay_bundle_schema": REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
        "replay_bundle_hash_manifest_sha256": replay_bundle.get(
            "hash_manifest_sha256"
        ),
        "readiness_blockers": ordered_unique(blockers),
        "blockers": ordered_unique(blockers),
        "local_mlx_rows_are_advisory_only": True,
        "encoder_side_only": True,
        "receiver_must_remain_decode_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_byte_transform_executor_local_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context=f"repair_family_byte_transform_execution_report:{family_id}",
    )
    return report, replay_bundle


__all__ = [
    "REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA",
    "REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA",
    "SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES",
    "RepairFamilyByteTransformExecutorError",
    "build_repair_family_byte_transform_execution_report",
]
