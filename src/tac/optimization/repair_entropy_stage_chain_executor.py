# SPDX-License-Identifier: MIT
"""Execute entropy-stage repair chains as composed archive candidates.

The chain executor is intentionally encoder-side. It reuses the existing
family byte-transform executor, but feeds each selected stage the archive
emitted by the previous stage so the output is a real composed ZIP candidate,
not just a bundle of independent leaf reports.
"""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_family_byte_transform_executor import (
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
    RepairFamilyByteTransformExecutorError,
    build_repair_family_byte_transform_execution_report,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact

REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA = (
    "repair_entropy_stage_chain_execution_bundle.v1"
)
REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_REPORT_SCHEMA = (
    "repair_entropy_stage_chain_execution_report.v1"
)
REPAIR_ENTROPY_STAGE_CHAIN_STAGE_REPORT_SCHEMA = (
    "repair_entropy_stage_chain_stage_report.v1"
)


class RepairEntropyStageChainExecutorError(ValueError):
    """Raised when the entropy-stage chain executor input is structurally invalid."""


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


def _safe_int(value: Any, *, default: int = 999) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairEntropyStageChainExecutorError(f"{path} must contain a JSON object")
    return payload


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool,
) -> None:
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=sha256_file(path) if path.exists() and allow_overwrite else None,
    )


def _archive_record(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    return {
        "path": _repo_rel(resolved, repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _source_archive_record_from_report(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = _mapping(report.get("candidate_archive"))
    path_text = str(candidate.get("source_archive_path") or "").strip()
    expected_sha = str(candidate.get("source_archive_sha256") or "").strip()
    blockers: list[str] = []
    if not path_text:
        path_text = str(candidate.get("path") or "").strip()
        expected_sha = str(candidate.get("sha256") or "").strip()
        blockers.append("source_archive_path_missing_using_candidate_archive_as_chain_base")
    if not path_text:
        return None, ordered_unique([*blockers, "chain_source_archive_path_missing"])
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        return None, ordered_unique([*blockers, "chain_source_archive_file_missing"])
    actual_sha = sha256_file(path)
    if expected_sha and expected_sha != actual_sha:
        blockers.append("chain_source_archive_sha256_mismatch")
    return {
        "path": _repo_rel(path, repo_root),
        "sha256": actual_sha,
        "bytes": path.stat().st_size,
    }, ordered_unique(blockers)


def _report_sort_key(report: Mapping[str, Any]) -> tuple[int, str, str]:
    stage = _mapping(report.get("active_entropy_stage"))
    return (
        _safe_int(stage.get("order") or report.get("entropy_stage_order")),
        str(report.get("family_id") or ""),
        str(report.get("typed_response_id") or ""),
    )


def _ordered_report_refs(
    *,
    execution_reports: Sequence[Mapping[str, Any]],
    execution_report_paths: Sequence[str | Path],
    work_order_bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, report in enumerate(execution_reports):
        if report.get("schema") != REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA:
            continue
        path = str(execution_report_paths[index]) if index < len(execution_report_paths) else ""
        refs.append(
            {
                "report": report,
                "path": path,
                "family_id": str(report.get("family_id") or ""),
                "typed_response_id": str(report.get("typed_response_id") or ""),
                "sort_key": _report_sort_key(report),
            }
        )
    if not work_order_bundle:
        return sorted(refs, key=lambda item: item["sort_key"])

    ordered: list[dict[str, Any]] = []
    used: set[int] = set()
    for work_order in work_order_bundle.get("work_orders") or []:
        if not isinstance(work_order, Mapping):
            continue
        typed_order = _string_list(work_order.get("typed_response_order"))
        family_order = _string_list(work_order.get("family_order"))
        for position, typed_response_id in enumerate(typed_order):
            family_id = family_order[position] if position < len(family_order) else ""
            for ref_index, ref in enumerate(refs):
                if ref_index in used:
                    continue
                if typed_response_id and ref["typed_response_id"] == typed_response_id:
                    ordered.append(ref)
                    used.add(ref_index)
                    break
                if family_id and ref["family_id"] == family_id:
                    ordered.append(ref)
                    used.add(ref_index)
                    break
    seen_ids = {id(ref) for ref in ordered}
    ordered.extend(ref for ref in sorted(refs, key=lambda item: item["sort_key"]) if id(ref) not in seen_ids)
    return ordered


def _stage_manifest(
    *,
    source_manifest: Mapping[str, Any],
    input_archive: Mapping[str, Any],
    chain_id: str,
    stage_index: int,
    source_report_path: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(source_manifest))
    manifest["candidate_archive"] = {
        "path": input_archive.get("path"),
        "sha256": input_archive.get("sha256"),
        "bytes": input_archive.get("bytes"),
    }
    manifest["byte_closed_candidate_emitted"] = True
    manifest["receiver_contract_satisfied"] = False
    manifest["runtime_consumption_proof_path"] = None
    manifest["entropy_stage_chain_compiler_context"] = {
        "schema": "repair_entropy_stage_chain_compiler_context.v1",
        "chain_id": chain_id,
        "stage_index": stage_index,
        "source_execution_report_path": source_report_path,
        "stage_input_archive": dict(input_archive),
        "encoder_side_only": True,
        "receiver_must_remain_decode_only": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    manifest.update(FALSE_AUTHORITY)
    return manifest


def _candidate_output_record(
    candidate_archive: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    path_text = str(candidate_archive.get("path") or "").strip()
    if not path_text:
        return None, ["stage_candidate_archive_path_missing"]
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        return None, ["stage_candidate_archive_file_missing"]
    actual = _archive_record(path, repo_root=repo_root)
    expected_sha = str(candidate_archive.get("sha256") or "").strip()
    blockers: list[str] = []
    if expected_sha and expected_sha != actual["sha256"]:
        blockers.append("stage_candidate_archive_sha256_mismatch")
    return actual, blockers


def _build_one_chain(
    *,
    chain_index: int,
    base_archive: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> dict[str, Any]:
    chain_id = f"entropy_stage_chain_{chain_index:03d}_{str(base_archive.get('sha256') or '')[:12]}"
    chain_dir = output_dir / _slug(chain_id)
    chain_dir.mkdir(parents=True, exist_ok=True)
    current_archive = dict(base_archive)
    stages: list[dict[str, Any]] = []
    execution_blockers: list[str] = []
    for stage_index, ref in enumerate(refs, start=1):
        report = _mapping(ref.get("report"))
        manifest_path_text = str(report.get("source_family_materializer_manifest_path") or "").strip()
        family_id = str(report.get("family_id") or "")
        typed_response_id = str(report.get("typed_response_id") or "")
        stage_dir = chain_dir / f"stage_{stage_index:03d}_{_slug(family_id)}_{_slug(typed_response_id)}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_record: dict[str, Any] = {
            "schema": REPAIR_ENTROPY_STAGE_CHAIN_STAGE_REPORT_SCHEMA,
            "stage_index": stage_index,
            "family_id": family_id,
            "typed_response_id": typed_response_id,
            "source_execution_report_path": ref.get("path") or None,
            "entropy_position_label": report.get("entropy_position_label"),
            "active_entropy_stage": dict(_mapping(report.get("active_entropy_stage"))),
            "fractal_optimization_scope": dict(
                _mapping(report.get("fractal_optimization_scope"))
            ),
            "allocated_repair_bytes": report.get("allocated_repair_bytes"),
            "byte_transform_delta": dict(_mapping(report.get("byte_transform_delta"))),
            "mlx_local_probe_delta": dict(_mapping(report.get("mlx_local_probe_delta"))),
            "stage_input_archive": dict(current_archive),
            "stage_output_archive": None,
            "stage_execution_report_path": None,
            "stage_replay_bundle_path": None,
            "stage_materialized": False,
            "stage_receiver_proof_ready": False,
            "stage_blockers": [],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            **FALSE_AUTHORITY,
        }
        if not manifest_path_text:
            stage_record["stage_blockers"] = ["source_family_materializer_manifest_path_missing"]
            execution_blockers.extend(stage_record["stage_blockers"])
            stages.append(stage_record)
            break
        manifest_path = _resolve(manifest_path_text, repo_root)
        if not manifest_path.is_file():
            stage_record["stage_blockers"] = ["source_family_materializer_manifest_file_missing"]
            execution_blockers.extend(stage_record["stage_blockers"])
            stages.append(stage_record)
            break
        try:
            source_manifest = _load_json(manifest_path)
            if source_manifest.get("schema") != REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA:
                raise RepairEntropyStageChainExecutorError(
                    "source family materializer manifest schema unsupported"
                )
            manifest = _stage_manifest(
                source_manifest=source_manifest,
                input_archive=current_archive,
                chain_id=chain_id,
                stage_index=stage_index,
                source_report_path=str(ref.get("path") or ""),
            )
            chain_manifest_path = stage_dir / "chain_input_family_materializer_manifest.json"
            _write_json(chain_manifest_path, manifest, allow_overwrite=allow_overwrite)
            stage_report, stage_bundle = build_repair_family_byte_transform_execution_report(
                family_materializer_manifest=manifest,
                family_materializer_manifest_path=_repo_rel(chain_manifest_path, repo_root),
                output_dir=stage_dir / "byte_transform",
                replay_argv=[
                    "python",
                    "tools/run_repair_entropy_stage_chain_executor.py",
                    "--chain-input-manifest",
                    _repo_rel(chain_manifest_path, repo_root),
                ],
                invocation_argv=["repair_entropy_stage_chain_executor"],
                repo_root=repo_root,
                allow_overwrite=allow_overwrite,
            )
            stage_report_path = stage_dir / "stage_execution_report.json"
            stage_bundle_path = stage_dir / "stage_replay_bundle.json"
            _write_json(stage_report_path, stage_report, allow_overwrite=allow_overwrite)
            _write_json(stage_bundle_path, stage_bundle, allow_overwrite=allow_overwrite)
            candidate = _mapping(stage_report.get("candidate_archive"))
            proof_path_text = str(
                candidate.get("runtime_consumption_proof_path")
                or stage_report.get("runtime_consumption_proof_path")
                or ""
            ).strip()
            proof_path = _resolve(proof_path_text, repo_root) if proof_path_text else None
            proof_present = bool(proof_path and proof_path.is_file())
            next_archive, output_blockers = _candidate_output_record(
                candidate,
                repo_root=repo_root,
            )
            if output_blockers:
                execution_blockers.extend(output_blockers)
            if next_archive is None:
                stage_record["stage_blockers"] = ordered_unique(output_blockers)
                stages.append(stage_record)
                break
            current_archive = next_archive
            stage_record.update(
                {
                    "stage_output_archive": dict(current_archive),
                    "stage_execution_report_path": _repo_rel(stage_report_path, repo_root),
                    "stage_replay_bundle_path": _repo_rel(stage_bundle_path, repo_root),
                    "stage_execution_report_schema": stage_report.get("schema"),
                    "stage_replay_bundle_schema": stage_bundle.get("schema"),
                    "stage_archive_transform_kind": stage_report.get("selected_archive_transform_kind"),
                    "stage_materialized": True,
                    "stage_receiver_proof_ready": (
                        candidate.get("runtime_consumption_proof_ready") is True
                    ),
                    "stage_runtime_consumption_proof_path": None
                    if proof_path is None
                    else _repo_rel(proof_path, repo_root),
                    "stage_runtime_consumption_proof_sha256": (
                        sha256_file(proof_path) if proof_present and proof_path else None
                    ),
                    "stage_runtime_consumption_proof_bytes": (
                        proof_path.stat().st_size
                        if proof_present and proof_path
                        else None
                    ),
                    "stage_receiver_contract_kind": (
                        stage_report.get("receiver_contract_kind")
                        or candidate.get("receiver_contract_kind")
                    ),
                    "stage_receiver_contract_satisfied": (
                        stage_report.get("receiver_contract_satisfied") is True
                        or candidate.get("receiver_contract_satisfied") is True
                    ),
                    "stage_report_blockers": _string_list(stage_report.get("blockers")),
                    "stage_blockers": ordered_unique(output_blockers),
                }
            )
        except (
            ArtifactWriteError,
            OSError,
            RepairEntropyStageChainExecutorError,
            RepairFamilyByteTransformExecutorError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            blocker = f"entropy_stage_chain_stage_execution_failed:{family_id}:{typed_response_id}:{exc}"
            stage_record["stage_blockers"] = [blocker]
            execution_blockers.append(blocker)
            stages.append(stage_record)
            break
        require_no_truthy_authority_fields(
            stage_record,
            context=f"repair_entropy_stage_chain_stage_report:{chain_id}:{stage_index}",
        )
        stages.append(stage_record)
    complete = len(stages) == len(refs) and all(stage.get("stage_materialized") is True for stage in stages)
    final_archive = dict(current_archive) if complete else None
    source_bytes = _safe_int(base_archive.get("bytes"), default=0)
    final_bytes = _safe_int((final_archive or {}).get("bytes"), default=0)
    report = {
        "schema": REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "chain_id": chain_id,
        "chain_index": chain_index,
        "source_archive": dict(base_archive),
        "stage_count": len(stages),
        "planned_stage_count": len(refs),
        "stages": stages,
        "archive_bound_candidate_emitted": complete and final_archive is not None,
        "candidate_archive": final_archive,
        "candidate_archive_materialized": complete and final_archive is not None,
        "runtime_consumption_proof_ready": bool(stages)
        and stages[-1].get("stage_receiver_proof_ready") is True,
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": final_bytes if final_archive is not None else None,
        "cumulative_saved_bytes": source_bytes - final_bytes if final_archive is not None else None,
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "blockers": ordered_unique(
            [
                *execution_blockers,
                "contest_cpu_or_cuda_exact_axis_payload_required",
                "lane_dispatch_claim_required_before_exact_eval",
            ]
        ),
        "allowed_use": "encoder_side_composed_archive_candidate_for_exact_axis_handoff",
        "forbidden_use": "score_claim_or_budget_spend_or_receiver_optimization",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context=f"repair_entropy_stage_chain_execution_report:{chain_id}",
    )
    return report


def build_repair_entropy_stage_chain_execution_bundle(
    *,
    execution_reports: Sequence[Mapping[str, Any]],
    execution_report_paths: Sequence[str | Path],
    work_order_bundle: Mapping[str, Any] | None,
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Execute selected entropy-stage reports as composed archive chains."""

    output = _resolve(output_dir, repo_root)
    output.mkdir(parents=True, exist_ok=True)
    ordered_refs = _ordered_report_refs(
        execution_reports=execution_reports,
        execution_report_paths=execution_report_paths,
        work_order_bundle=work_order_bundle,
    )
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    blockers: list[str] = []
    for ref in ordered_refs:
        source_archive, source_blockers = _source_archive_record_from_report(
            _mapping(ref.get("report")),
            repo_root=repo_root,
        )
        blockers.extend(source_blockers)
        if source_archive is None:
            continue
        key = str(source_archive.get("sha256") or source_archive.get("path") or "")
        if key not in groups:
            groups[key] = {"base_archive": source_archive, "refs": []}
        groups[key]["refs"].append(ref)
    chain_reports: list[dict[str, Any]] = []
    for chain_index, group in enumerate(groups.values(), start=1):
        chain_report = _build_one_chain(
            chain_index=chain_index,
            base_archive=group["base_archive"],
            refs=group["refs"],
            output_dir=output,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        )
        chain_reports.append(chain_report)
        chain_report_path = output / f"{chain_report['chain_id']}.json"
        _write_json(chain_report_path, chain_report, allow_overwrite=allow_overwrite)
    if len(groups) > 1:
        blockers.append("selected_entropy_stage_reports_span_multiple_source_archives")
    materialized = [
        report for report in chain_reports if report.get("archive_bound_candidate_emitted") is True
    ]
    bundle = {
        "schema": REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_work_order_bundle_schema": None
        if work_order_bundle is None
        else work_order_bundle.get("schema"),
        "source_execution_report_count": len(execution_reports),
        "ordered_stage_count": len(ordered_refs),
        "source_archive_group_count": len(groups),
        "chain_count": len(chain_reports),
        "materialized_chain_candidate_count": len(materialized),
        "runtime_consumption_proof_ready_count": sum(
            1 for report in materialized if report.get("runtime_consumption_proof_ready") is True
        ),
        "chain_reports": chain_reports,
        "candidate_archives": [
            report.get("candidate_archive")
            for report in materialized
            if isinstance(report.get("candidate_archive"), Mapping)
        ],
        "composed_archive_candidate_default": True,
        "encoder_side_only": True,
        "receiver_must_remain_decode_only": True,
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "blockers": ordered_unique(
            [
                *blockers,
                *[
                    blocker
                    for report in chain_reports
                    for blocker in _string_list(report.get("blockers"))
                ],
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "local_archive_bound_chain_candidate_generation_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bundle,
        context="repair_entropy_stage_chain_execution_bundle",
    )
    return bundle


__all__ = [
    "REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA",
    "REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_REPORT_SCHEMA",
    "REPAIR_ENTROPY_STAGE_CHAIN_STAGE_REPORT_SCHEMA",
    "RepairEntropyStageChainExecutorError",
    "build_repair_entropy_stage_chain_execution_bundle",
]
