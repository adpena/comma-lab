# SPDX-License-Identifier: MIT
"""Fail-closed exact-ready bridge inputs for repair-family handoff plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ArchiveBoundCandidateContractError,
    archive_bound_candidate_contract_fields_for_row,
    archive_bound_candidate_contracts_from_payload,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_family_stack_search import (
    REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.optimizer.exact_readiness import (
    QUEUE_SCHEMA as EXACT_READY_QUEUE_SCHEMA,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.repo_io import read_json, sha256_file

REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA = "repair_family_exact_ready_bridge_report.v1"
REPAIR_FAMILY_EXACT_READY_BRIDGE_ROW_SCHEMA = "repair_family_exact_ready_bridge_row.v1"
REPAIR_FAMILY_EXACT_READY_SOURCE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"


class RepairFamilyExactReadyBridgeError(ValueError):
    """Raised when a repair-family exact-ready bridge cannot be built."""


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


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _file_custody(
    *,
    path_text: str,
    repo_root: str | Path,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not path_text:
        blockers.append(f"{label}_path_missing")
        return {
            "schema": "repair_family_exact_ready_bridge_file_custody.v1",
            "label": label,
            "path": None,
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "sha256_matches": False,
            "bytes_match": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        blockers.append(f"{label}_file_missing")
        return {
            "schema": "repair_family_exact_ready_bridge_file_custody.v1",
            "label": label,
            "path": _repo_rel(path, repo_root),
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "sha256_matches": False,
            "bytes_match": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    sha_matches = expected_sha256 is None or actual_sha == expected_sha256
    bytes_match = expected_bytes is None or actual_bytes == expected_bytes
    if not sha_matches:
        blockers.append(f"{label}_sha256_mismatch")
    if not bytes_match:
        blockers.append(f"{label}_bytes_mismatch")
    return {
        "schema": "repair_family_exact_ready_bridge_file_custody.v1",
        "label": label,
        "path": _repo_rel(path, repo_root),
        "present": True,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "sha256_matches": sha_matches,
        "bytes_match": bytes_match,
        "custody_complete": not blockers,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }, blockers


def _runtime_proof_payload(
    *,
    proof_path_text: str,
    repo_root: str | Path,
) -> tuple[Mapping[str, Any], list[str]]:
    if not proof_path_text:
        return {}, ["runtime_consumption_proof_path_missing"]
    path = _resolve(proof_path_text, repo_root)
    if not path.is_file():
        return {}, ["runtime_consumption_proof_file_missing"]
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        return {}, [f"runtime_consumption_proof_json_invalid:{exc}"]
    if not isinstance(payload, Mapping):
        return {}, ["runtime_consumption_proof_not_object"]
    require_no_truthy_authority_fields(
        payload,
        context=f"repair_family_exact_ready_bridge_runtime_proof:{proof_path_text}",
        fields=(
            "score_claim",
            "score_claim_valid",
            "score_claim_eligible",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
            "field_selection_ready_for_exact_eval_dispatch",
            "dispatch_attempted",
            "gpu_launched",
            "dispatch_packet_ready",
            "exact_cuda_auth_eval",
            "contest_cuda_auth_eval",
            "promotable",
        ),
    )
    return payload, []


def _candidate_source_archive(proof_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = proof_payload.get("source_archive")
    return source if isinstance(source, Mapping) else {}


def _candidate_archive_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    archive = row.get("candidate_archive")
    return archive if isinstance(archive, Mapping) else {}


def _runtime_proof_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = row.get("runtime_consumption_proof")
    return proof if isinstance(proof, Mapping) else {}


def _archive_contracts_from_handoff_row(
    handoff_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contract = handoff_row.get("archive_bound_candidate_contract")
    surface = handoff_row.get("archive_bound_candidate_contract_surface")
    if not contract and not surface:
        return []
    try:
        return archive_bound_candidate_contracts_from_payload(
            handoff_row,
            label="repair_family_exact_ready_bridge_handoff_row",
        )
    except ArchiveBoundCandidateContractError as exc:
        raise RepairFamilyExactReadyBridgeError(str(exc)) from exc


def _contract_candidate_sha256(contract: Mapping[str, Any]) -> str | None:
    archive = contract.get("candidate_archive")
    value = archive.get("sha256") if isinstance(archive, Mapping) else None
    return value if isinstance(value, str) and value.strip() else None


def _contract_candidate_archive(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    archive = contract.get("candidate_archive")
    return archive if isinstance(archive, Mapping) else {}


def _contract_source_archive(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    archive = contract.get("source_archive")
    return archive if isinstance(archive, Mapping) else {}


def _contract_runtime_proof_row(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    proof_path = contract.get("runtime_consumption_proof_path")
    if isinstance(proof_path, str) and proof_path.strip():
        return {"path": proof_path.strip()}
    return {}


def _merged_mapping(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in primary.items() if value not in ("", None)})
    return merged


def _submission_runtime_custody(
    *,
    candidate_id: str | None = None,
    candidate_sha256: str | None,
    candidate_bytes: int | None,
    submission_dirs: Sequence[str | Path],
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not submission_dirs:
        blockers.append("submission_dir_missing_for_runtime_content_tree_custody")
        return {
            "schema": "repair_family_exact_ready_bridge_submission_runtime_custody.v1",
            "submission_dir": None,
            "archive_path": None,
            "archive_manifest_path": None,
            "report_txt_path": None,
            "inflate_sh_path": None,
            "runtime_manifest": None,
            "runtime_tree_sha256": None,
            "runtime_content_tree_sha256": None,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    repo = Path(repo_root)
    best_blockers: list[str] = []
    for raw_dir in submission_dirs:
        submission_dir = _resolve(raw_dir, repo)
        local_blockers: list[str] = []
        archive_path = submission_dir / "archive.zip"
        archive_manifest_path = submission_dir / "archive_manifest.json"
        report_txt = submission_dir / "report.txt"
        inflate_sh = submission_dir / "inflate.sh"
        if not submission_dir.is_dir():
            local_blockers.append("submission_dir_not_directory")
        if not archive_path.is_file():
            local_blockers.append("submission_archive_zip_missing")
        else:
            archive_sha = sha256_file(archive_path)
            archive_bytes = archive_path.stat().st_size
            if candidate_sha256 and archive_sha != candidate_sha256:
                local_blockers.append("submission_archive_sha256_mismatch")
            if candidate_bytes is not None and archive_bytes != candidate_bytes:
                local_blockers.append("submission_archive_bytes_mismatch")
        if not archive_manifest_path.is_file():
            local_blockers.append("submission_archive_manifest_missing")
        else:
            try:
                archive_manifest = read_json(archive_manifest_path)
            except (OSError, ValueError) as exc:
                archive_manifest = {}
                local_blockers.append(f"submission_archive_manifest_read_failed:{exc}")
            if candidate_id and isinstance(archive_manifest, Mapping):
                manifest_candidate_id = str(archive_manifest.get("candidate_id") or "")
                if manifest_candidate_id and manifest_candidate_id != candidate_id:
                    local_blockers.append("submission_archive_manifest_candidate_id_mismatch")
        if not report_txt.is_file():
            local_blockers.append("submission_report_txt_missing")
        if not inflate_sh.is_file():
            local_blockers.append("submission_inflate_sh_missing")
        runtime_manifest = None
        if not local_blockers:
            try:
                runtime_manifest = runtime_dependency_manifest(submission_dir, repo)
            except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
                local_blockers.append(f"runtime_dependency_manifest_failed:{exc}")
        runtime_tree_sha = (
            runtime_manifest.get("runtime_tree_sha256") if isinstance(runtime_manifest, Mapping) else None
        )
        runtime_content_sha = (
            runtime_manifest.get("runtime_content_tree_sha256") if isinstance(runtime_manifest, Mapping) else None
        )
        if not isinstance(runtime_tree_sha, str) or len(runtime_tree_sha) != 64:
            local_blockers.append("runtime_tree_sha256_missing")
        if not isinstance(runtime_content_sha, str) or len(runtime_content_sha) != 64:
            local_blockers.append("runtime_content_tree_sha256_missing")
        if not local_blockers:
            return {
                "schema": ("repair_family_exact_ready_bridge_submission_runtime_custody.v1"),
                "submission_dir": _repo_rel(submission_dir, repo),
                "archive_path": _repo_rel(archive_path, repo),
                "archive_manifest_path": _repo_rel(archive_manifest_path, repo),
                "report_txt_path": _repo_rel(report_txt, repo),
                "inflate_sh_path": _repo_rel(inflate_sh, repo),
                "runtime_manifest": runtime_manifest,
                "runtime_tree_sha256": runtime_tree_sha,
                "runtime_content_tree_sha256": runtime_content_sha,
                "custody_complete": True,
                "blockers": [],
                **FALSE_AUTHORITY,
            }, []
        best_blockers = ordered_unique([*best_blockers, *local_blockers])
    return {
        "schema": "repair_family_exact_ready_bridge_submission_runtime_custody.v1",
        "submission_dir": None,
        "archive_path": None,
        "archive_manifest_path": None,
        "report_txt_path": None,
        "inflate_sh_path": None,
        "runtime_manifest": None,
        "runtime_tree_sha256": None,
        "runtime_content_tree_sha256": None,
        "custody_complete": False,
        "blockers": best_blockers or ["submission_runtime_custody_missing"],
        **FALSE_AUTHORITY,
    }, best_blockers or ["submission_runtime_custody_missing"]


def _serialized_delta(
    *,
    source_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_sha = source_archive.get("sha256")
    candidate_sha = candidate_archive.get("sha256")
    source_bytes = source_archive.get("bytes")
    candidate_bytes = candidate_archive.get("bytes")
    if not (
        isinstance(source_sha, str)
        and isinstance(candidate_sha, str)
        and isinstance(source_bytes, int)
        and not isinstance(source_bytes, bool)
        and isinstance(candidate_bytes, int)
        and not isinstance(candidate_bytes, bool)
    ):
        return None
    return build_serialized_archive_delta_contract(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
        modeled_saved_bytes=source_bytes - candidate_bytes,
        require_realized_saving=False,
    )


def _bridge_row(
    *,
    handoff_row: Mapping[str, Any],
    submission_dirs: Sequence[str | Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    family_id = str(handoff_row.get("family_id") or "repair_family")
    typed_response_id = str(handoff_row.get("typed_response_id") or "typed_response")
    candidate_chain_id = str(handoff_row.get("candidate_chain_id") or typed_response_id)
    handoff_contracts = _archive_contracts_from_handoff_row(handoff_row)
    handoff_archive_bound_contract = handoff_contracts[0] if handoff_contracts else {}
    candidate_archive_row = _merged_mapping(
        _contract_candidate_archive(handoff_archive_bound_contract),
        _candidate_archive_from_row(handoff_row),
    )
    runtime_proof_row = _merged_mapping(
        _contract_runtime_proof_row(handoff_archive_bound_contract),
        _runtime_proof_from_row(handoff_row),
    )
    expected_sha = (
        str(candidate_archive_row.get("expected_sha256") or candidate_archive_row.get("sha256") or "").strip() or None
    )
    expected_bytes = candidate_archive_row.get("expected_bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        expected_bytes = candidate_archive_row.get("bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        expected_bytes = None
    candidate_id = f"repair_family_exact_handoff__{_slug(family_id)}__{_slug(typed_response_id)}"
    archive_custody, archive_blockers = _file_custody(
        path_text=str(candidate_archive_row.get("path") or ""),
        repo_root=repo_root,
        expected_sha256=expected_sha,
        expected_bytes=expected_bytes,
        label="candidate_archive",
    )
    proof_custody, proof_file_blockers = _file_custody(
        path_text=str(runtime_proof_row.get("path") or ""),
        repo_root=repo_root,
        expected_sha256=str(runtime_proof_row.get("sha256") or "").strip() or None,
        expected_bytes=runtime_proof_row.get("bytes")
        if isinstance(runtime_proof_row.get("bytes"), int) and not isinstance(runtime_proof_row.get("bytes"), bool)
        else None,
        label="runtime_consumption_proof",
    )
    proof_payload, proof_payload_blockers = _runtime_proof_payload(
        proof_path_text=str(runtime_proof_row.get("path") or ""),
        repo_root=repo_root,
    )
    receiver_contract_satisfied = (
        proof_custody.get("custody_complete") is True
        and not proof_payload_blockers
        and proof_payload.get("receiver_contract_satisfied") is True
    )
    source_archive = _merged_mapping(
        _candidate_source_archive(proof_payload),
        _contract_source_archive(handoff_archive_bound_contract),
    )
    candidate_archive = {
        "path": archive_custody.get("path"),
        "sha256": archive_custody.get("sha256"),
        "bytes": archive_custody.get("bytes"),
    }
    serialized_delta = _serialized_delta(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
    )
    submission_custody, submission_blockers = _submission_runtime_custody(
        candidate_id=candidate_id,
        candidate_sha256=archive_custody.get("sha256")
        if isinstance(archive_custody.get("sha256"), str)
        else expected_sha,
        candidate_bytes=archive_custody.get("bytes")
        if isinstance(archive_custody.get("bytes"), int)
        else expected_bytes,
        submission_dirs=submission_dirs,
        repo_root=repo_root,
    )
    archive_path_for_queue = (
        submission_custody.get("archive_path")
        if submission_custody.get("custody_complete") is True
        else archive_custody.get("path")
    )
    row_blockers = ordered_unique(
        [
            *archive_blockers,
            *proof_file_blockers,
            *proof_payload_blockers,
            *submission_blockers,
            *(["archive_bound_candidate_contract_missing"] if not handoff_archive_bound_contract else []),
            *_string_list(handoff_archive_bound_contract.get("blockers")),
            *(
                ["archive_bound_candidate_contract_stale_candidate_sha256"]
                if handoff_archive_bound_contract
                and _contract_candidate_sha256(handoff_archive_bound_contract)
                not in (None, archive_custody.get("sha256"))
                else []
            ),
            *_string_list(handoff_row.get("blockers")),
            "repair_family_exact_handoff_bridge_requires_exact_readiness_gate",
            "contest_cpu_or_cuda_auth_axis_required_before_score_or_dispatch",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    score_affecting_payload_changed = bool(
        serialized_delta
        and (
            serialized_delta.get("source_archive_sha256") != serialized_delta.get("candidate_archive_sha256")
            or serialized_delta.get("source_archive_bytes") != serialized_delta.get("candidate_archive_bytes")
        )
    )
    bridge_source_row = {
        "candidate_id": candidate_id,
        "source_candidate_id": candidate_chain_id,
        "candidate_family": "repair_family_exact_handoff",
        "family_id": family_id,
        "typed_response_id": typed_response_id,
        "candidate_chain_id": candidate_chain_id,
        "candidate_chain_ids": _string_list(handoff_row.get("candidate_chain_ids")),
        "entropy_position_label": handoff_row.get("entropy_position_label"),
        "entropy_stage_order": handoff_row.get("entropy_stage_order"),
        "lane_id": f"repair_family_exact_handoff::{_slug(family_id)}",
        "target_modes": ["contest_exact_eval"],
        "target_score_axes_required": ["contest_cpu", "contest_cuda"],
        "score_axis": "contest_cuda",
        "target_score_axis": "contest_cuda",
        "archive_path": archive_path_for_queue,
        "candidate_archive_path": archive_path_for_queue,
        "candidate_archive_sha256": archive_custody.get("sha256"),
        "archive_sha256": archive_custody.get("sha256"),
        "candidate_archive_bytes": archive_custody.get("bytes"),
        "archive_bytes": archive_custody.get("bytes"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_archive_bytes": source_archive.get("bytes"),
        "runtime_consumption_proof_path": proof_custody.get("path"),
        "runtime_consumption_proof_sha256": proof_custody.get("sha256"),
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_schema": proof_payload.get("schema"),
        "runtime_consumption_proof_passed": (
            proof_payload.get("runtime_consumption_proof_passed") is True or proof_payload.get("passed") is True
        ),
        "runtime_consumption_proof_status": (
            "archive_bound_proof_custody_present"
            if proof_custody.get("custody_complete") is True
            else "runtime_proof_custody_incomplete"
        ),
        "receiver_contract_id": proof_payload.get("receiver_contract_id"),
        "receiver_contract_kind": proof_payload.get("receiver_contract_kind"),
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "runtime_tree_sha256": submission_custody.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": submission_custody.get("runtime_content_tree_sha256"),
        "submission_dir": submission_custody.get("submission_dir"),
        "archive_manifest_path": submission_custody.get("archive_manifest_path"),
        "inflate_sh_path": submission_custody.get("inflate_sh_path"),
        "serialized_archive_delta": serialized_delta,
        "score_affecting_payload_changed": score_affecting_payload_changed,
        "charged_bits_changed": score_affecting_payload_changed,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        "local_mlx_rows_are_advisory_only": True,
        "evidence_semantics": ("repair_family_archive_bound_handoff_bridge_input_fail_closed"),
        "dispatch_blockers": row_blockers,
    }
    bridge_source_row.update(FALSE_AUTHORITY)
    bridge_source_row["score_affecting_payload_changed"] = score_affecting_payload_changed
    bridge_source_row["charged_bits_changed"] = score_affecting_payload_changed
    bridge_source_row.update(
        archive_bound_candidate_contract_fields_for_row(
            bridge_source_row,
            repo_root=repo_root,
            selected_transform_kind=str(
                handoff_row.get("archive_native_transform_kind") or handoff_row.get("target_kind") or family_id
            ),
            family_id=family_id,
            typed_response_id=typed_response_id,
            candidate_chain_id=candidate_chain_id,
            entropy_position_label=handoff_row.get("entropy_position_label")
            if isinstance(handoff_row.get("entropy_position_label"), str)
            else None,
            entropy_stage_order=handoff_row.get("entropy_stage_order")
            if isinstance(handoff_row.get("entropy_stage_order"), int)
            and not isinstance(handoff_row.get("entropy_stage_order"), bool)
            else None,
        )
    )
    archive_bound_contract = bridge_source_row["archive_bound_candidate_contract"]
    bridge_source_row["archive_bound_contract_substrate_tags"] = _string_list(
        archive_bound_contract.get("archive_substrate_tags")
    )
    bridge_source_row["archive_bound_contract_acquisition_penalty"] = archive_bound_contract.get("acquisition_penalty")
    require_no_truthy_authority_fields(
        bridge_source_row,
        context=f"repair_family_exact_ready_bridge_source_row:{candidate_id}",
    )
    bridge = {
        "schema": REPAIR_FAMILY_EXACT_READY_BRIDGE_ROW_SCHEMA,
        "candidate_id": candidate_id,
        "family_id": family_id,
        "typed_response_id": typed_response_id,
        "candidate_chain_id": candidate_chain_id,
        "candidate_chain_ids": _string_list(handoff_row.get("candidate_chain_ids")),
        "entropy_position_label": handoff_row.get("entropy_position_label"),
        "entropy_stage_order": handoff_row.get("entropy_stage_order"),
        "chain_stage_identities": [
            dict(item) for item in handoff_row.get("chain_stage_identities") or [] if isinstance(item, Mapping)
        ],
        "archive_custody": archive_custody,
        "runtime_consumption_proof_custody": proof_custody,
        "runtime_consumption_proof_schema": proof_payload.get("schema"),
        "submission_runtime_custody": submission_custody,
        "archive_bound_candidate_contract": archive_bound_contract,
        "archive_custody_complete": archive_custody.get("custody_complete") is True,
        "runtime_proof_custody_complete": proof_custody.get("custody_complete") is True and not proof_payload_blockers,
        "runtime_content_tree_custody_complete": (submission_custody.get("custody_complete") is True),
        "bridge_source_queue_row": bridge_source_row,
        "failure_rebudgeting_identity": {
            "schema": "repair_family_exact_failure_rebudgeting_identity.v1",
            "candidate_id": candidate_id,
            "family_id": family_id,
            "typed_response_id": typed_response_id,
            "candidate_chain_id": candidate_chain_id,
            "candidate_chain_ids": _string_list(handoff_row.get("candidate_chain_ids")),
            "entropy_position_label": handoff_row.get("entropy_position_label"),
            "entropy_stage_order": handoff_row.get("entropy_stage_order"),
            "chain_stage_identities": [
                dict(item) for item in handoff_row.get("chain_stage_identities") or [] if isinstance(item, Mapping)
            ],
            "source_archive_sha256": source_archive.get("sha256"),
            "source_archive_bytes": source_archive.get("bytes"),
            "candidate_archive_sha256": archive_custody.get("sha256"),
            "candidate_archive_bytes": archive_custody.get("bytes"),
            "runtime_consumption_proof_sha256": proof_custody.get("sha256"),
            "runtime_consumption_proof_bytes": proof_custody.get("bytes"),
            "runtime_content_tree_sha256": submission_custody.get("runtime_content_tree_sha256"),
            "runtime_tree_sha256": submission_custody.get("runtime_tree_sha256"),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": row_blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_exact_ready_bridge_input_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bridge,
        context=f"repair_family_exact_ready_bridge_row:{candidate_id}",
    )
    return bridge


def build_repair_family_exact_ready_bridge(
    *,
    exact_handoff_plan: Mapping[str, Any],
    exact_handoff_plan_path: str | Path | None = None,
    submission_dirs: Sequence[str | Path] = (),
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build fail-closed exact-ready bridge inputs from a repair handoff plan."""

    if exact_handoff_plan.get("schema") != REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA:
        raise RepairFamilyExactReadyBridgeError(
            "repair exact-ready bridge requires repair_family_exact_handoff_plan.v1"
        )
    require_no_truthy_authority_fields(
        exact_handoff_plan,
        context="repair_family_exact_ready_bridge_handoff_plan",
    )
    source_rows = [
        row
        for row in exact_handoff_plan.get("archive_bound_rows") or exact_handoff_plan.get("rows") or []
        if isinstance(row, Mapping)
    ]
    bridge_rows = [
        _bridge_row(
            handoff_row=row,
            submission_dirs=submission_dirs,
            repo_root=repo_root,
        )
        for row in source_rows
    ]
    source_queue_rows = [dict(row["bridge_source_queue_row"]) for row in bridge_rows]
    source_queue = {
        "schema": REPAIR_FAMILY_EXACT_READY_SOURCE_QUEUE_SCHEMA,
        "tool": "tac.optimization.repair_family_exact_ready_bridge",
        "source_exact_handoff_plan_path": (None if exact_handoff_plan_path is None else str(exact_handoff_plan_path)),
        "top_k": source_queue_rows,
        "dispatch_ready": [],
        "dispatch_ready_count": 0,
        "n_candidates": len(source_queue_rows),
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    blocked_exact_ready_queue = {
        "schema": EXACT_READY_QUEUE_SCHEMA,
        "tool": "tac.optimization.repair_family_exact_ready_bridge",
        "source_exact_handoff_plan_path": (None if exact_handoff_plan_path is None else str(exact_handoff_plan_path)),
        "n_candidates": len(source_queue_rows),
        "top_k_count": len(source_queue_rows),
        "dispatch_ready_count": 0,
        "dispatch_ready": [],
        "top_k": source_queue_rows,
        "top_k_forensic": source_queue_rows,
        "evidence_boundary": {
            "schema": "repair_family_blocked_exact_ready_queue_boundary.v1",
            "score_claim": False,
            "dispatch_ready_rows_forbidden_until_exact_readiness_gate": True,
            "contest_cpu_or_cuda_auth_eval_required_before_score_claim": True,
            "lane_dispatch_claim_required_before_gpu_or_remote_eval": True,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
    }
    archive_count = sum(1 for row in bridge_rows if row.get("archive_custody_complete") is True)
    proof_count = sum(1 for row in bridge_rows if row.get("runtime_proof_custody_complete") is True)
    runtime_content_count = sum(1 for row in bridge_rows if row.get("runtime_content_tree_custody_complete") is True)
    blockers = ordered_unique(
        [
            *([] if bridge_rows else ["repair_family_exact_handoff_rows_missing"]),
            *[blocker for row in bridge_rows for blocker in _string_list(row.get("blockers"))],
            "promote_optimizer_candidate_for_exact_eval_required_before_dispatch_ready",
            "contest_cpu_or_cuda_auth_axis_required_before_score_or_dispatch",
        ]
    )
    report = {
        "schema": REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA,
        "source_exact_handoff_plan_path": (None if exact_handoff_plan_path is None else str(exact_handoff_plan_path)),
        "source_exact_handoff_plan_schema": exact_handoff_plan.get("schema"),
        "candidate_count": len(bridge_rows),
        "archive_custody_proven_count": archive_count,
        "runtime_proof_custody_proven_count": proof_count,
        "runtime_content_tree_custody_proven_count": runtime_content_count,
        "source_optimizer_queue_schema": source_queue["schema"],
        "blocked_exact_ready_queue_schema": blocked_exact_ready_queue["schema"],
        "blocked_exact_ready_dispatch_ready_count": 0,
        "rows": bridge_rows,
        "blockers": blockers,
        "ready_for_exact_eval_dispatch": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_family_exact_ready_bridge_fail_closed_inputs",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    for payload, context in (
        (source_queue, "repair_family_exact_ready_source_queue"),
        (blocked_exact_ready_queue, "repair_family_blocked_exact_ready_queue"),
        (report, "repair_family_exact_ready_bridge_report"),
    ):
        require_no_truthy_authority_fields(payload, context=context)
    return {
        "source_optimizer_queue": source_queue,
        "blocked_exact_ready_queue": blocked_exact_ready_queue,
        "bridge_report": report,
    }


__all__ = [
    "REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA",
    "REPAIR_FAMILY_EXACT_READY_BRIDGE_ROW_SCHEMA",
    "REPAIR_FAMILY_EXACT_READY_SOURCE_QUEUE_SCHEMA",
    "RepairFamilyExactReadyBridgeError",
    "build_repair_family_exact_ready_bridge",
]
