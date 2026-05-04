#!/usr/bin/env python3
"""Plan bounded Alpha mask residual alternatives from a candidate manifest.

This tool consumes an ``alpha_mask_candidate_builder`` manifest plus its
grayscale and AMR1 repair artifacts.  It validates archive custody, then emits
deterministic byte/agreement estimates for compressed and selective repair
payload policies.  It does not run scorer networks, does not build a contest
archive, and does not make score evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import lzma
import os
import platform
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
DEFAULT_CANDIDATE_MANIFEST = (
    REPO_ROOT
    / "experiments/results/alpha_mask_candidate_builder_pfp16_20260501_full/"
    / "alpha_mask_candidate_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/alpha_mask_residual_planner"

SCHEMA = "alpha_mask_residual_planner_v1"
REPORT_NAME = "alpha_mask_residual_planner_manifest.json"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this planner. These are byte/agreement design "
    "estimates over Alpha mask residual artifacts; a deterministic finalist "
    "archive and exact CUDA auth eval are required before any score claim."
)


@dataclass(frozen=True)
class PlannerConfig:
    frame_group_size: int = 50
    max_policies: int = 40
    max_repair_runs: int = 1_000_000
    max_payload_bytes: int = 64_000_000
    min_agreement: float = 0.998
    byte_target: int | None = None


@dataclass(frozen=True)
class ResidualPolicy:
    name: str
    kind: str
    selected_run_indices: tuple[int, ...]
    selected_pixels: int
    details: dict[str, Any]


def _load_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("alpha_mask_candidate_builder_for_planner", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load builder module from {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder_module()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _validate_config(config: PlannerConfig) -> None:
    if config.frame_group_size <= 0:
        raise ValueError(f"frame_group_size must be positive, got {config.frame_group_size}")
    if config.max_policies <= 0:
        raise ValueError(f"max_policies must be positive, got {config.max_policies}")
    if config.max_repair_runs <= 0:
        raise ValueError(f"max_repair_runs must be positive, got {config.max_repair_runs}")
    if config.max_payload_bytes <= 0:
        raise ValueError(f"max_payload_bytes must be positive, got {config.max_payload_bytes}")
    if not (0.0 <= config.min_agreement <= 1.0):
        raise ValueError(f"min_agreement must be in [0,1], got {config.min_agreement}")
    if config.byte_target is not None and config.byte_target <= 0:
        raise ValueError(f"byte_target must be positive when provided, got {config.byte_target}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_existing_path(raw: str, *, manifest_dir: Path) -> Path:
    if not raw or "\x00" in raw:
        raise ValueError(f"unsafe empty or NUL path in manifest: {raw!r}")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, manifest_dir / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"manifest path {raw!r} does not exist; searched {searched}")


def _artifact_by_role(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    artifacts = manifest.get("candidate", {}).get("artifacts", [])
    matches = [item for item in artifacts if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one candidate artifact with role {role!r}, got {len(matches)}")
    return dict(matches[0])


def _require_false(manifest: dict[str, Any], path: tuple[str, ...]) -> None:
    cursor: Any = manifest
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(f"manifest missing {'.'.join(path)}")
        cursor = cursor[key]
    if cursor is not False:
        raise ValueError(f"manifest {'.'.join(path)} must be false for non-promotable planning")


def _validate_non_promotable_manifest(manifest: dict[str, Any]) -> None:
    _require_false(manifest, ("score_claim",))
    _require_false(manifest, ("promotion_eligible",))
    _require_false(manifest, ("scorer_network_loaded",))
    _require_false(manifest, ("candidate", "score_claim"))
    _require_false(manifest, ("candidate", "promotion_eligible"))
    if manifest.get("evidence_grade") != EVIDENCE_GRADE:
        raise ValueError(f"manifest evidence_grade must be {EVIDENCE_GRADE!r}")
    if "contest_auth_eval.py --device cuda" not in manifest.get("canonical_score_source_required", ""):
        raise ValueError("manifest must preserve exact CUDA auth eval as the required score source")


def _validate_file_record(path: Path, record: dict[str, Any], *, label: str) -> dict[str, Any]:
    actual_size = int(path.stat().st_size)
    actual_sha = _sha256_file(path)
    expected_size = record.get("size_bytes")
    expected_sha = record.get("sha256")
    if expected_size is not None and int(expected_size) != actual_size:
        raise ValueError(f"{label} size mismatch: manifest={expected_size} actual={actual_size}")
    if expected_sha is not None and str(expected_sha) != actual_sha:
        raise ValueError(f"{label} sha256 mismatch: manifest={expected_sha} actual={actual_sha}")
    return {
        "path": str(path),
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "validated_against_manifest": True,
    }


def _validate_source_archive(manifest: dict[str, Any], *, manifest_dir: Path) -> dict[str, Any]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("manifest missing source object")
    archive = _resolve_existing_path(str(source.get("archive_path", "")), manifest_dir=manifest_dir)
    mask_member = source.get("mask_member")
    if not isinstance(mask_member, dict):
        raise ValueError("manifest missing source.mask_member object")
    member_name = str(mask_member.get("name", ""))
    member_data, source_meta = builder._read_archive_member(archive, member_name)
    actual_member = source_meta["mask_member"]
    if int(mask_member.get("size_bytes")) != int(actual_member["size_bytes"]):
        raise ValueError("source mask member size mismatch against archive")
    if str(mask_member.get("sha256")) != str(actual_member["sha256"]):
        raise ValueError("source mask member sha256 mismatch against archive")
    if str(mask_member.get("sha256")) != _sha256_bytes(member_data):
        raise ValueError("source mask member sha256 mismatch against decoded bytes")
    manifest_archive_sha = source.get("archive_sha256")
    if manifest_archive_sha is not None and str(manifest_archive_sha) != source_meta["archive_sha256"]:
        raise ValueError("source archive sha256 mismatch against archive")
    manifest_archive_size = source.get("archive_size_bytes")
    if manifest_archive_size is not None and int(manifest_archive_size) != int(source_meta["archive_size_bytes"]):
        raise ValueError("source archive size mismatch against archive")
    return {
        "archive": {
            "path": str(archive),
            "size_bytes": int(source_meta["archive_size_bytes"]),
            "sha256": source_meta["archive_sha256"],
        },
        "mask_member": actual_member,
        "member_inventory": source_meta["member_inventory"],
        "validated_zip_safety": True,
    }


def _load_and_validate_inputs(candidate_manifest: Path, config: PlannerConfig) -> dict[str, Any]:
    manifest_path = candidate_manifest.resolve()
    manifest = _read_json(manifest_path)
    _validate_non_promotable_manifest(manifest)
    manifest_dir = manifest_path.parent
    source_archive = _validate_source_archive(manifest, manifest_dir=manifest_dir)

    grayscale_record = _artifact_by_role(manifest, "alpha4_grayscale_lut_video")
    repair_record = _artifact_by_role(manifest, "alpha4_residual_repair_payload")
    grayscale_path = _resolve_existing_path(str(grayscale_record.get("path", "")), manifest_dir=manifest_dir)
    repair_path = _resolve_existing_path(str(repair_record.get("path", "")), manifest_dir=manifest_dir)
    grayscale_validation = _validate_file_record(grayscale_path, grayscale_record, label="grayscale artifact")
    repair_validation = _validate_file_record(repair_path, repair_record, label="repair artifact")

    repair_payload = repair_path.read_bytes()
    if len(repair_payload) > config.max_payload_bytes:
        raise ValueError(
            f"repair payload exceeds planner bound: {len(repair_payload)} > {config.max_payload_bytes}"
        )
    repair_header, runs = builder._decode_repair_payload(repair_payload)
    if str(repair_header.get("schema")) != builder.REPAIR_SCHEMA:
        raise ValueError(f"unsupported repair schema {repair_header.get('schema')!r}")
    if len(runs) > config.max_repair_runs:
        raise ValueError(f"repair run count exceeds planner bound: {len(runs)} > {config.max_repair_runs}")

    source_sha = manifest["source"]["decoded_masks"]["class_id_u8_sha256"]
    candidate_sha = manifest["candidate"]["alpha4"]["decoded_candidate_masks"]["class_id_u8_sha256"]
    if repair_header.get("source_mask_u8_sha256") != source_sha:
        raise ValueError("repair header source mask sha does not match candidate manifest")
    if repair_header.get("candidate_mask_u8_sha256") != candidate_sha:
        raise ValueError("repair header candidate mask sha does not match candidate manifest")

    before = manifest["candidate"]["alpha4"]["agreement_before_repair"]
    different_pixels = int(before["different_pixels"])
    selection = repair_header.get("selection", {})
    if int(selection.get("total_residual_pixels", different_pixels)) != different_pixels:
        raise ValueError("repair header residual count does not match Alpha4 disagreement")
    if sum(int(run.length) for run in runs) != int(selection.get("selected_repair_pixels", 0)):
        raise ValueError("repair run lengths do not match repair selection selected pixels")

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "source_archive": source_archive,
        "grayscale": grayscale_validation,
        "repair": repair_validation,
        "repair_header": repair_header,
        "runs": runs,
        "repair_payload": repair_payload,
    }


def _bounded_steps(total: int, max_steps: int) -> list[int]:
    if total <= 0 or max_steps <= 0:
        return []
    if total <= max_steps:
        return list(range(1, total + 1))
    steps = {1, total}
    power = 1
    while power < total and len(steps) < max_steps:
        steps.add(power)
        power *= 2
    for idx in range(1, max_steps + 1):
        steps.add(max(1, min(total, round(idx * total / max_steps))))
    return sorted(steps)[:max_steps]


def _source_class_priority(manifest: dict[str, Any], repair_header: dict[str, Any]) -> list[int]:
    raw = repair_header.get("selection", {}).get("class_priority")
    if raw is None:
        raw = manifest.get("builder_config", {}).get("class_priority", builder.DEFAULT_CLASS_PRIORITY)
    priority = [int(value) for value in raw]
    if sorted(priority) != list(builder.CLASS_IDS):
        raise ValueError(f"class priority must be a permutation of {list(builder.CLASS_IDS)}, got {priority}")
    return priority


def _policy_from_indices(
    *,
    name: str,
    kind: str,
    indices: list[int],
    runs: list[Any],
    details: dict[str, Any],
) -> ResidualPolicy:
    unique_sorted = tuple(sorted(set(int(index) for index in indices)))
    selected_pixels = sum(int(runs[index].length) for index in unique_sorted)
    return ResidualPolicy(
        name=name,
        kind=kind,
        selected_run_indices=unique_sorted,
        selected_pixels=int(selected_pixels),
        details=details,
    )


def _build_policies(
    *,
    runs: list[Any],
    manifest: dict[str, Any],
    repair_header: dict[str, Any],
    config: PlannerConfig,
) -> list[ResidualPolicy]:
    policies: list[ResidualPolicy] = [
        ResidualPolicy(
            name="omit_repair_payload",
            kind="no_repair",
            selected_run_indices=(),
            selected_pixels=0,
            details={"description": "use Alpha4 grayscale payload without a residual repair member"},
        )
    ]

    priority = _source_class_priority(manifest, repair_header)
    for count in range(1, len(priority) + 1):
        selected_classes = set(priority[:count])
        indices = [index for index, run in enumerate(runs) if int(run.class_id) in selected_classes]
        policies.append(
            _policy_from_indices(
                name="class_prefix_" + "_".join(str(value) for value in priority[:count]),
                kind="class_priority_prefix",
                indices=indices,
                runs=runs,
                details={"source_classes": [int(value) for value in priority[:count]]},
            )
        )

    remaining_slots = max(0, config.max_policies - len(policies))
    if remaining_slots:
        groups: dict[int, list[int]] = {}
        for index, run in enumerate(runs):
            group_id = int(run.frame_index) // config.frame_group_size
            groups.setdefault(group_id, []).append(index)
        group_rank = sorted(
            groups,
            key=lambda group_id: (
                -sum(int(runs[index].length) for index in groups[group_id]),
                group_id,
            ),
        )
        steps = _bounded_steps(len(group_rank), remaining_slots)
        for step in steps:
            selected_groups = group_rank[:step]
            indices: list[int] = []
            for group_id in selected_groups:
                indices.extend(groups[group_id])
            group_ranges = [
                {
                    "start_frame": int(group_id * config.frame_group_size),
                    "end_frame_exclusive": int((group_id + 1) * config.frame_group_size),
                }
                for group_id in selected_groups
            ]
            policies.append(
                _policy_from_indices(
                    name=f"top_residual_frame_groups_{step:04d}_of_{len(group_rank):04d}",
                    kind="top_residual_frame_groups",
                    indices=indices,
                    runs=runs,
                    details={
                        "frame_group_size": int(config.frame_group_size),
                        "selected_group_count": int(step),
                        "total_group_count": int(len(group_rank)),
                        "selected_group_ranges": group_ranges,
                    },
                )
            )

    deduped: list[ResidualPolicy] = []
    seen: set[tuple[int, ...]] = set()
    for policy in policies:
        if policy.selected_run_indices in seen:
            continue
        seen.add(policy.selected_run_indices)
        deduped.append(policy)
        if len(deduped) >= config.max_policies:
            break
    return deduped


def _compress_payloads(payload: bytes, *, omit_payload: bool) -> list[dict[str, Any]]:
    if omit_payload:
        return [
            {
                "compressor": "omit_repair_payload",
                "payload_format": "none",
                "payload_bytes": b"",
                "runtime_integration_required": True,
                "available": True,
            }
        ]

    results = [
        {
            "compressor": "raw_amr1",
            "payload_format": builder.REPAIR_SCHEMA,
            "payload_bytes": payload,
            "runtime_integration_required": False,
            "available": True,
        },
        {
            "compressor": "zlib9",
            "payload_format": f"{builder.REPAIR_SCHEMA}+zlib",
            "payload_bytes": zlib.compress(payload, level=9),
            "runtime_integration_required": True,
            "available": True,
        },
        {
            "compressor": "lzma_xz9e",
            "payload_format": f"{builder.REPAIR_SCHEMA}+lzma_xz",
            "payload_bytes": lzma.compress(payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME),
            "runtime_integration_required": True,
            "available": True,
        },
    ]
    try:
        import brotli  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        results.append(
            {
                "compressor": "brotli11",
                "payload_format": f"{builder.REPAIR_SCHEMA}+brotli",
                "payload_bytes": b"",
                "runtime_integration_required": True,
                "available": False,
                "unavailable_reason": str(exc),
            }
        )
    else:
        results.append(
            {
                "compressor": "brotli11",
                "payload_format": f"{builder.REPAIR_SCHEMA}+brotli",
                "payload_bytes": brotli.compress(payload, quality=11, lgwin=24),
                "runtime_integration_required": True,
                "available": True,
            }
        )
    return results


def _selection_meta_for_policy(
    *,
    policy: ResidualPolicy,
    total_residual_pixels: int,
    total_residual_runs: int,
    config: PlannerConfig,
) -> dict[str, Any]:
    coverage = 1.0 if total_residual_pixels == 0 else policy.selected_pixels / total_residual_pixels
    return {
        "strategy": "alpha_mask_residual_planner_selective_policy_v1",
        "policy_name": policy.name,
        "policy_kind": policy.kind,
        "policy_details": policy.details,
        "total_residual_pixels": int(total_residual_pixels),
        "total_residual_runs": int(total_residual_runs),
        "selected_repair_pixels": int(policy.selected_pixels),
        "selected_repair_runs": int(len(policy.selected_run_indices)),
        "residual_pixel_coverage": _round_float(coverage),
        "partial_repair": bool(policy.selected_pixels != total_residual_pixels),
        "fail_on_partial_repair": False,
        "planner_non_promotable": True,
        "max_repair_runs": int(config.max_repair_runs),
        "max_payload_bytes": int(config.max_payload_bytes),
    }


def _candidate_records_for_policy(
    *,
    policy: ResidualPolicy,
    runs: list[Any],
    repair_header: dict[str, Any],
    total_residual_pixels: int,
    total_residual_runs: int,
    grayscale_bytes: int,
    original_mask_bytes: int,
    byte_target: int,
    base_equal_pixels: int,
    num_pixels: int,
    config: PlannerConfig,
) -> list[dict[str, Any]]:
    selected_runs = [runs[index] for index in policy.selected_run_indices]
    selection_meta = _selection_meta_for_policy(
        policy=policy,
        total_residual_pixels=total_residual_pixels,
        total_residual_runs=total_residual_runs,
        config=config,
    )
    omit_payload = len(selected_runs) == 0
    payload = b""
    if not omit_payload:
        payload = builder._encode_repair_payload(
            selected_runs,
            shape=tuple(int(value) for value in repair_header["shape"]),
            source_mask_sha256=str(repair_header["source_mask_u8_sha256"]),
            candidate_mask_sha256=str(repair_header["candidate_mask_u8_sha256"]),
            selection_meta=selection_meta,
        )
    repaired_equal_pixels = base_equal_pixels + int(policy.selected_pixels)
    agreement_estimate = repaired_equal_pixels / num_pixels if num_pixels else 0.0

    records = []
    for compressed in _compress_payloads(payload, omit_payload=omit_payload):
        if not compressed["available"]:
            records.append(
                {
                    "policy_name": policy.name,
                    "policy_kind": policy.kind,
                    "compressor": compressed["compressor"],
                    "available": False,
                    "unavailable_reason": compressed.get("unavailable_reason"),
                    "score_claim": False,
                    "promotion_eligible": False,
                }
            )
            continue
        payload_bytes = compressed["payload_bytes"]
        residual_size = int(len(payload_bytes))
        total_size = int(grayscale_bytes + residual_size)
        records.append(
            {
                "policy_name": policy.name,
                "policy_kind": policy.kind,
                "policy_details": policy.details,
                "compressor": compressed["compressor"],
                "payload_format": compressed["payload_format"],
                "available": True,
                "runtime_integration_required": bool(compressed["runtime_integration_required"]),
                "repair_full_coverage": bool(policy.selected_pixels == total_residual_pixels),
                "selected_repair_runs": int(len(policy.selected_run_indices)),
                "selected_repair_pixels": int(policy.selected_pixels),
                "total_residual_runs": int(total_residual_runs),
                "total_residual_pixels": int(total_residual_pixels),
                "residual_pixel_coverage": _round_float(
                    1.0 if total_residual_pixels == 0 else policy.selected_pixels / total_residual_pixels
                ),
                "agreement_estimate_after_selected_repair": _round_float(agreement_estimate),
                "meets_min_agreement": bool(agreement_estimate >= config.min_agreement),
                "grayscale_size_bytes": int(grayscale_bytes),
                "residual_payload_size_bytes": residual_size,
                "residual_payload_sha256": _sha256_bytes(payload_bytes),
                "candidate_total_payload_bytes": total_size,
                "original_mask_member_size_bytes": int(original_mask_bytes),
                "delta_vs_original_mask_member_bytes": int(total_size - original_mask_bytes),
                "under_original_mask_member_bytes": bool(total_size <= original_mask_bytes),
                "byte_target_size_bytes": int(byte_target),
                "byte_target_pass": bool(total_size <= byte_target),
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    return records


def _compressor_availability(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    availability: dict[str, dict[str, Any]] = {}
    for record in records:
        compressor = str(record["compressor"])
        availability.setdefault(
            compressor,
            {
                "available": bool(record.get("available", False)),
                "unavailable_reason": record.get("unavailable_reason"),
            },
        )
    return availability


def _best_records(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    available = [record for record in records if record.get("available") is True]
    ordered = sorted(
        available,
        key=lambda record: (
            int(record["candidate_total_payload_bytes"]),
            -float(record["agreement_estimate_after_selected_repair"]),
            str(record["policy_name"]),
            str(record["compressor"]),
        ),
    )
    return ordered[:limit]


def _best_under_target(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    viable = [
        record
        for record in records
        if record.get("available") is True
        and record.get("byte_target_pass") is True
        and record.get("meets_min_agreement") is True
    ]
    ordered = sorted(
        viable,
        key=lambda record: (
            -float(record["agreement_estimate_after_selected_repair"]),
            int(record["candidate_total_payload_bytes"]),
            str(record["policy_name"]),
            str(record["compressor"]),
        ),
    )
    return ordered[:limit]


def _byte_agreement_frontier(records: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    available = [record for record in records if record.get("available") is True]
    ordered = sorted(
        available,
        key=lambda record: (
            int(record["candidate_total_payload_bytes"]),
            -float(record["agreement_estimate_after_selected_repair"]),
            str(record["policy_name"]),
            str(record["compressor"]),
        ),
    )
    frontier: list[dict[str, Any]] = []
    best_agreement = -1.0
    for record in ordered:
        agreement = float(record["agreement_estimate_after_selected_repair"])
        if agreement > best_agreement:
            frontier.append(record)
            best_agreement = agreement
        if len(frontier) >= limit:
            break
    return frontier


def _config_record(config: PlannerConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def _provenance(command: list[str] | None) -> dict[str, Any]:
    return {
        "tool": "experiments/alpha_mask_residual_planner.py",
        "command": list(command) if command is not None else list(sys.argv),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": {
            key: os.environ[key]
            for key in (
                "PYTHONHASHSEED",
                "UV_PROJECT_ENVIRONMENT",
                "TAC_FFMPEG",
                "TAC_FFPROBE",
                "TAC_UPSTREAM_DIR",
            )
            if key in os.environ
        },
    }


def _assert_non_promotable_report(report: dict[str, Any]) -> None:
    if report.get("score_claim") is not False:
        raise AssertionError("planner report score_claim must be false")
    if report.get("promotion_eligible") is not False:
        raise AssertionError("planner report promotion_eligible must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError("planner report evidence_grade must be empirical")
    if report.get("scorer_network_loaded") is not False:
        raise AssertionError("planner must not load scorer networks")
    for record in report.get("candidate_records", []):
        if record.get("available") is True and (
            "residual_payload_sha256" not in record or "residual_payload_size_bytes" not in record
        ):
            raise AssertionError(f"candidate record missing byte custody: {record}")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_NAME
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; use --force to overwrite")


def plan_residual_alternatives(
    *,
    candidate_manifest: Path,
    output_dir: Path,
    config: PlannerConfig,
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    inputs = _load_and_validate_inputs(candidate_manifest, config)
    _prepare_output_dir(output_dir, force=force)

    manifest = inputs["manifest"]
    repair_header = inputs["repair_header"]
    runs = inputs["runs"]
    before = manifest["candidate"]["alpha4"]["agreement_before_repair"]
    num_pixels = int(before["num_pixels"])
    total_residual_pixels = int(before["different_pixels"])
    base_equal_pixels = int(before["equal_pixels"])
    grayscale_bytes = int(inputs["grayscale"]["size_bytes"])
    original_mask_bytes = int(inputs["source_archive"]["mask_member"]["size_bytes"])
    byte_target = int(config.byte_target or original_mask_bytes)
    residual_budget_after_grayscale = int(byte_target - grayscale_bytes)

    policies = _build_policies(
        runs=runs,
        manifest=manifest,
        repair_header=repair_header,
        config=config,
    )
    candidate_records: list[dict[str, Any]] = []
    for policy in policies:
        candidate_records.extend(
            _candidate_records_for_policy(
                policy=policy,
                runs=runs,
                repair_header=repair_header,
                total_residual_pixels=total_residual_pixels,
                total_residual_runs=len(runs),
                grayscale_bytes=grayscale_bytes,
                original_mask_bytes=original_mask_bytes,
                byte_target=byte_target,
                base_equal_pixels=base_equal_pixels,
                num_pixels=num_pixels,
                config=config,
            )
        )

    best_under_target = _best_under_target(candidate_records)
    if residual_budget_after_grayscale < 0:
        next_step = (
            "alpha4 grayscale bytes already exceed the byte target before residuals; "
            "rerun the candidate builder/planner at higher CRF or with a smaller base representation."
        )
    elif best_under_target:
        next_step = (
            "review best_under_target_meeting_min_agreement and build a deterministic runtime "
            "integration for the selected residual format before any exact eval."
        )
    else:
        next_step = (
            "no evaluated policy met both byte target and min-agreement; try stricter selective "
            "repair policies, higher CRF, or a different Alpha base representation."
        )

    source_manifest_record = {
        "path": str(inputs["manifest_path"]),
        "size_bytes": int(inputs["manifest_path"].stat().st_size),
        "sha256": _sha256_file(inputs["manifest_path"]),
    }
    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_planner_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "identify bounded Alpha residual compression and selective-repair alternatives "
            "that may reduce mask payload bytes without producing score evidence"
        ),
        "config": _config_record(config),
        "source_candidate_manifest": source_manifest_record,
        "custody": {
            "source_archive": inputs["source_archive"],
            "grayscale_artifact": inputs["grayscale"],
            "repair_artifact": inputs["repair"],
            "repair_header_sha256": _sha256_bytes(
                json.dumps(repair_header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "repair_payload_sha256": _sha256_bytes(inputs["repair_payload"]),
            "repair_payload_size_bytes": int(len(inputs["repair_payload"])),
        },
        "byte_target": {
            "target_size_bytes": byte_target,
            "original_mask_member_size_bytes": original_mask_bytes,
            "grayscale_size_bytes": grayscale_bytes,
            "residual_budget_after_grayscale_bytes": residual_budget_after_grayscale,
            "grayscale_alone_under_target": bool(grayscale_bytes <= byte_target),
        },
        "agreement_context": {
            "num_pixels": num_pixels,
            "alpha4_equal_pixels_before_repair": base_equal_pixels,
            "alpha4_different_pixels_before_repair": total_residual_pixels,
            "alpha4_agreement_before_repair": _round_float(base_equal_pixels / num_pixels),
            "min_agreement": _round_float(config.min_agreement),
        },
        "planner_summary": {
            "policies_evaluated": int(len(policies)),
            "candidate_records_evaluated": int(len(candidate_records)),
            "compressor_availability": _compressor_availability(candidate_records),
            "best_by_total_bytes": _best_records(candidate_records),
            "byte_agreement_pareto_frontier": _byte_agreement_frontier(candidate_records),
            "best_under_target_meeting_min_agreement": best_under_target,
            "next_step": next_step,
        },
        "candidate_records": candidate_records,
        "provenance": _provenance(command),
    }
    _assert_non_promotable_report(report)
    report_path = output_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-group-size", type=int, default=PlannerConfig.frame_group_size)
    parser.add_argument("--max-policies", type=int, default=PlannerConfig.max_policies)
    parser.add_argument("--max-repair-runs", type=int, default=PlannerConfig.max_repair_runs)
    parser.add_argument("--max-payload-bytes", type=int, default=PlannerConfig.max_payload_bytes)
    parser.add_argument("--min-agreement", type=float, default=PlannerConfig.min_agreement)
    parser.add_argument(
        "--byte-target",
        type=int,
        default=None,
        help="Payload byte target. Defaults to the source masks.mkv member size from the candidate manifest.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing planner report in output-dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = PlannerConfig(
        frame_group_size=args.frame_group_size,
        max_policies=args.max_policies,
        max_repair_runs=args.max_repair_runs,
        max_payload_bytes=args.max_payload_bytes,
        min_agreement=args.min_agreement,
        byte_target=args.byte_target,
    )
    report = plan_residual_alternatives(
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
        config=config,
        command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        force=args.force,
    )
    summary = report["planner_summary"]
    target = report["byte_target"]
    print(
        f"[empirical:{args.output_dir / REPORT_NAME}] Alpha residual planner evaluated "
        f"{summary['candidate_records_evaluated']} records across {summary['policies_evaluated']} policies; "
        f"target={target['target_size_bytes']}B best_under_target="
        f"{len(summary['best_under_target_meeting_min_agreement'])}. "
        "No score claim; CUDA auth eval required for any finalist archive.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
