# SPDX-License-Identifier: MIT
"""Deterministic inverse-scorer cell candidate materializer.

The materializer emits a byte-closed archive by appending a compact IAS1
descriptor to the template archive member. The descriptor is a candidate
generation packet only; exact-eval authority remains blocked until a runtime
adapter proves the descriptor bytes are consumed by inflate/runtime code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import build_candidate_diff_manifest
from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.optimization.inverse_steganalysis_acquisition import ACTION_FUNCTIONAL_SCHEMA
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import read_json, repo_relative, sha256_bytes, sha256_file

PLAN_SCHEMA = "inverse_scorer_cell_candidate_plan_v1"
CANDIDATE_SCHEMA = "inverse_scorer_cell_candidate_v1"
VERIFIED_CANDIDATE_SCHEMA = "inverse_scorer_cell_candidate_receiver_verified_v1"
RECEIVER_PROOF_SCHEMA = "inverse_scorer_cell_receiver_proof_v1"
RECEIVER_VERIFICATION_SCHEMA = "inverse_scorer_cell_receiver_verification_v1"
RUNTIME_ADAPTER_SCHEMA = "inverse_scorer_cell_runtime_adapter_v1"
DESCRIPTOR_SCHEMA = "inverse_scorer_cell_descriptor_v1"
MATERIALIZER_ID = "inverse_scorer_cell_candidate_adapter"
TARGET_KIND = "inverse_scorer_cell_candidate_v1"
RECEIVER_CONTRACT_ID = "inverse_scorer_cell_receiver.v1"
RECEIVER_CONTRACT_KIND = "inverse_scorer_coordinate_candidate"
IAS1_MAGIC = b"IAS1"
REQUIRED_CONTEXT_FIELDS = (
    "raw_contest_video_digest",
    "candidate_archive_template",
    "inverse_action_functional",
    "output_archive",
    "manifest_out",
    "runtime_consumption_proof",
)
FALSE_AUTHORITY = dict(PROXY_FALSE_AUTHORITY_FIELDS)


class InverseScorerCellMaterializerError(ValueError):
    """Raised when inverse-scorer cell materialization inputs are malformed."""


def build_inverse_scorer_cell_candidate_plan(
    *,
    raw_contest_video_digest: str | Mapping[str, Any] | None,
    candidate_archive_template: str | Path | Mapping[str, Any] | None,
    inverse_action_functional: str | Path | Mapping[str, Any] | None,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    atom_ids: Sequence[str] = (),
    selected_limit: int | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed plan for an inverse-scorer cell candidate."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    blockers: list[str] = []
    digest = _normalize_raw_digest(raw_contest_video_digest)
    if not digest:
        blockers.append("raw_contest_video_digest_missing")
    template_record = _optional_archive_template_record(candidate_archive_template, repo)
    if not template_record:
        blockers.append("candidate_archive_template_missing")
    action, action_record = _optional_action_functional_record(
        inverse_action_functional,
        repo,
    )
    selected_cells = _selected_action_cells(
        action,
        atom_ids=atom_ids,
        selected_limit=selected_limit,
    ) if action is not None else []
    if action is None:
        blockers.append("inverse_action_functional_missing")
    elif not selected_cells:
        blockers.append("inverse_action_functional_selected_cells_missing")
    receiver_verification = verify_inverse_scorer_cell_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_raw_contest_video_digest=digest or None,
    )
    blockers.extend(receiver_verification["blockers"])
    return apply_proxy_evidence_boundary(
        {
            "schema": PLAN_SCHEMA,
            "materializer_id": MATERIALIZER_ID,
            "target_kind": TARGET_KIND,
            "receiver_contract_id": RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
            "required_context_fields": list(REQUIRED_CONTEXT_FIELDS),
            "raw_contest_video_digest": digest,
            "candidate_archive_template": template_record,
            "inverse_action_functional": action_record,
            "selected_cell_count": len(selected_cells),
            "selected_atom_ids": [str(row["atom_id"]) for row in selected_cells],
            "receiver_contract_satisfied": not blockers,
            "receiver_verification": receiver_verification,
            "readiness_blockers": ordered_unique(blockers),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=blockers,
    )


def materialize_inverse_scorer_cell_candidate(
    *,
    raw_contest_video_digest: str | Mapping[str, Any],
    candidate_archive_template: str | Path,
    inverse_action_functional: str | Path | Mapping[str, Any],
    output_archive: str | Path,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    atom_ids: Sequence[str] = (),
    selected_limit: int | None = None,
    repo_root: str | Path | None = None,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
) -> dict[str, Any]:
    """Append an IAS1 descriptor packet to a template single-member archive."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    template_path = _resolve_existing_path(candidate_archive_template, repo)
    output_path = _resolve_output_path(output_archive, repo)
    _refuse_overwrite_unless_expected(
        output_path,
        allow_overwrite=allow_overwrite,
        expected_sha256=expected_existing_output_sha256,
    )
    digest = _normalize_raw_digest(raw_contest_video_digest)
    if not digest:
        raise InverseScorerCellMaterializerError(
            "raw_contest_video_digest must be non-empty"
        )
    action, action_record = _load_action_functional_record(
        inverse_action_functional,
        repo,
    )
    selected_cells = _selected_action_cells(
        action,
        atom_ids=atom_ids,
        selected_limit=selected_limit,
    )
    if not selected_cells:
        raise InverseScorerCellMaterializerError(
            "inverse action functional contains no selected cells"
        )
    try:
        template = read_strict_single_member_zip(template_path)
    except HnervLowlevelPackError as exc:
        raise InverseScorerCellMaterializerError(str(exc)) from exc

    descriptor = _descriptor_payload(
        raw_contest_video_digest=digest,
        action_record=action_record,
        template_archive={
            "path": repo_relative(template_path, repo),
            "bytes": template.archive_bytes,
            "sha256": template.archive_sha256,
            "member_name": template.member_name,
            "member_bytes": template.member_bytes,
            "member_sha256": sha256_bytes(template.payload),
        },
        selected_cells=selected_cells,
    )
    descriptor_json = _canonical_json_bytes(descriptor)
    packet = pack_inverse_scorer_cell_descriptor(descriptor)
    candidate_payload = template.payload + packet
    write_stored_single_member_zip(
        output_path,
        member_name=template.member_name,
        payload=candidate_payload,
    )
    candidate_archive = {
        "path": repo_relative(output_path, repo),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "member_name": template.member_name,
        "member_bytes": len(candidate_payload),
        "member_sha256": sha256_bytes(candidate_payload),
    }
    descriptor_record = {
        "schema": DESCRIPTOR_SCHEMA,
        "magic": IAS1_MAGIC.decode("ascii"),
        "packet_offset": len(template.payload),
        "packet_bytes": len(packet),
        "packet_sha256": sha256_bytes(packet),
        "json_bytes": len(descriptor_json),
        "json_sha256": sha256_bytes(descriptor_json),
        "selected_atom_ids": [str(row["atom_id"]) for row in selected_cells],
        "selected_cell_count": len(selected_cells),
    }
    receiver_verification = verify_inverse_scorer_cell_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_candidate_archive_sha256=candidate_archive["sha256"],
        required_candidate_member_sha256=candidate_archive["member_sha256"],
        required_descriptor_packet_sha256=descriptor_record["packet_sha256"],
        required_raw_contest_video_digest=digest,
        required_selected_atom_ids=descriptor_record["selected_atom_ids"],
    )
    readiness_blockers = ordered_unique(
        [
            *receiver_verification["blockers"],
            *(
                []
                if receiver_verification["receiver_contract_satisfied"] is True
                else ["inverse_scorer_cell_receiver_contract_not_satisfied"]
            ),
            "candidate_inflate_output_parity_missing",
            "exact_auth_eval_required_before_score_claim",
        ]
    )
    diff_manifest = build_candidate_diff_manifest(
        source_archive=template_path,
        candidate_archive=output_path,
        source_label="inverse_scorer_cell_template_archive",
        candidate_label="inverse_scorer_cell_candidate_archive",
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": CANDIDATE_SCHEMA,
            "materializer_id": MATERIALIZER_ID,
            "target_kind": TARGET_KIND,
            "receiver_contract_id": RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
            "required_context_fields": list(REQUIRED_CONTEXT_FIELDS),
            "byte_closed_candidate_emitted": True,
            "candidate_generation_only": True,
            "raw_contest_video_digest": digest,
            "template_archive": descriptor["template_archive"],
            "candidate_archive": candidate_archive,
            "archive_diff_manifest": diff_manifest,
            "inverse_action_functional": action_record,
            "inverse_scorer_cell_descriptor": descriptor_record,
            "selected_cells": selected_cells,
            "receiver_verification": receiver_verification,
            "receiver_contract_satisfied": (
                receiver_verification["receiver_contract_satisfied"] is True
            ),
            "readiness_blockers": readiness_blockers,
            "ready_for_archive_preflight": False,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=readiness_blockers,
    )


def build_inverse_scorer_cell_receiver_proof(
    *,
    runtime_adapter_manifest: str | Path | Mapping[str, Any],
    candidate_manifest: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a runtime-adapter manifest into an inverse-scorer receiver proof."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    adapter, adapter_record = _load_required_mapping_with_record(
        runtime_adapter_manifest,
        repo=repo,
        label="runtime adapter manifest",
    )
    if adapter.get("schema") != RUNTIME_ADAPTER_SCHEMA:
        raise InverseScorerCellMaterializerError(
            f"runtime adapter manifest must have schema {RUNTIME_ADAPTER_SCHEMA}"
        )
    candidate, candidate_record = _candidate_manifest_for_adapter(
        adapter,
        candidate_manifest=candidate_manifest,
        repo=repo,
        adapter_record=adapter_record,
    )
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    descriptor = _mapping(candidate.get("inverse_scorer_cell_descriptor"))
    descriptor_consumption = _mapping(adapter.get("descriptor_consumption"))
    runtime_probe = _mapping(adapter.get("runtime_consumption_probe"))
    selected_atom_ids = [
        str(item) for item in descriptor.get("selected_atom_ids") or [] if str(item)
    ]
    blockers = ordered_unique(
        [
            *(
                []
                if adapter.get("score_claim") is False
                else ["runtime_adapter_manifest_must_not_claim_score"]
            ),
            *(
                []
                if adapter.get("dispatch_attempted") is False
                else ["runtime_adapter_manifest_must_not_dispatch"]
            ),
            *(
                []
                if runtime_probe.get("passed") is True
                else ["runtime_consumption_probe_not_passed"]
            ),
            *(
                []
                if descriptor_consumption.get("passed") is True
                else ["descriptor_consumption_not_passed"]
            ),
            *(
                []
                if _clean_str(descriptor_consumption.get("descriptor_packet_sha256"))
                == _clean_str(descriptor.get("packet_sha256"))
                else ["descriptor_consumption_sha_mismatch"]
            ),
            *(
                []
                if _clean_str(_mapping(adapter.get("candidate_archive")).get("sha256"))
                == _clean_str(candidate_archive.get("sha256"))
                else ["runtime_adapter_candidate_archive_sha_mismatch"]
            ),
            *(["descriptor_selected_atom_ids_missing"] if not selected_atom_ids else []),
        ]
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": RECEIVER_PROOF_SCHEMA,
            "receiver_contract_id": RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
            "ready_for_exact_eval_runtime": not blockers,
            "runtime_adapter_manifest": adapter_record,
            "candidate_manifest": candidate_record,
            "candidate_archive_sha256": _clean_str(candidate_archive.get("sha256")),
            "candidate_member_sha256": _clean_str(
                candidate_archive.get("member_sha256")
            ),
            "descriptor_packet_sha256": _clean_str(descriptor.get("packet_sha256")),
            "raw_contest_video_digest": candidate.get("raw_contest_video_digest"),
            "selected_atom_ids": selected_atom_ids,
            "runtime_tree_sha256": _clean_str(adapter.get("runtime_tree_sha256")),
            "runtime_consumption_probe": runtime_probe,
            "descriptor_consumption": descriptor_consumption,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=blockers,
    )


def verify_inverse_scorer_cell_candidate_manifest(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    runtime_consumption_proof: str | Path | Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an existing inverse-scorer cell candidate manifest."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate, candidate_record = _load_required_mapping_with_record(
        candidate_manifest,
        repo=repo,
        label="inverse-scorer cell candidate manifest",
    )
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise InverseScorerCellMaterializerError(
            f"candidate manifest must have schema {CANDIDATE_SCHEMA}"
        )
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    descriptor = _mapping(candidate.get("inverse_scorer_cell_descriptor"))
    custody = _candidate_archive_custody(candidate_archive, repo=repo)
    receiver_verification = verify_inverse_scorer_cell_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_candidate_archive_sha256=_clean_str(candidate_archive.get("sha256")),
        required_candidate_member_sha256=_clean_str(candidate_archive.get("member_sha256")),
        required_descriptor_packet_sha256=_clean_str(descriptor.get("packet_sha256")),
        required_raw_contest_video_digest=candidate.get("raw_contest_video_digest"),
        required_selected_atom_ids=[
            str(item) for item in descriptor.get("selected_atom_ids") or []
        ],
    )
    readiness_blockers = ordered_unique(
        [
            *[str(item) for item in candidate.get("readiness_blockers") or []],
            *custody["blockers"],
            *receiver_verification["blockers"],
            *(
                []
                if receiver_verification["receiver_contract_satisfied"] is True
                and not custody["blockers"]
                else ["inverse_scorer_cell_receiver_contract_not_satisfied"]
            ),
        ]
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": VERIFIED_CANDIDATE_SCHEMA,
            "source_candidate_schema": candidate.get("schema"),
            "materializer_id": MATERIALIZER_ID,
            "target_kind": TARGET_KIND,
            "receiver_contract_id": RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
            "source_candidate_manifest": candidate_record,
            "candidate_archive": candidate_archive,
            "candidate_archive_custody": custody,
            "inverse_scorer_cell_descriptor": descriptor,
            "receiver_verification": receiver_verification,
            "receiver_contract_satisfied": (
                receiver_verification["receiver_contract_satisfied"] is True
                and not custody["blockers"]
            ),
            "readiness_blockers": readiness_blockers,
            "ready_for_archive_preflight": False,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=readiness_blockers,
    )


def verify_inverse_scorer_cell_receiver_contract(
    *,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None,
    required_candidate_archive_sha256: str | None = None,
    required_candidate_member_sha256: str | None = None,
    required_descriptor_packet_sha256: str | None = None,
    required_raw_contest_video_digest: str | Mapping[str, Any] | None = None,
    required_selected_atom_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate a receiver/runtime proof for the IAS1 descriptor."""

    blockers: list[str] = []
    proof = _load_optional_mapping(runtime_consumption_proof)
    if proof is None:
        blockers.append("runtime_consumption_proof_missing")
    else:
        if proof.get("schema") != RECEIVER_PROOF_SCHEMA:
            blockers.append("runtime_consumption_proof_schema_mismatch")
        for key, blocker in (
            ("score_claim", "runtime_consumption_proof_must_not_claim_score"),
            ("promotion_eligible", "runtime_consumption_proof_must_not_promote"),
            ("rank_or_kill_eligible", "runtime_consumption_proof_must_not_rank_or_kill"),
        ):
            if proof.get(key) is not False:
                blockers.append(blocker)
        if proof.get("ready_for_exact_eval_runtime") is not True:
            blockers.append("runtime_consumption_proof_not_ready")
        _match_text(
            blockers,
            proof.get("candidate_archive_sha256"),
            required_candidate_archive_sha256,
            "runtime_consumption_proof_archive_sha_mismatch",
        )
        _match_text(
            blockers,
            proof.get("candidate_member_sha256"),
            required_candidate_member_sha256,
            "runtime_consumption_proof_member_sha_mismatch",
        )
        _match_text(
            blockers,
            proof.get("descriptor_packet_sha256"),
            required_descriptor_packet_sha256,
            "runtime_consumption_proof_descriptor_sha_mismatch",
        )
        required_digest = _normalize_raw_digest(required_raw_contest_video_digest)
        if required_digest and _normalize_raw_digest(
            proof.get("raw_contest_video_digest")
        ) != required_digest:
            blockers.append("runtime_consumption_proof_raw_digest_mismatch")
        proof_atoms = [str(item) for item in proof.get("selected_atom_ids") or []]
        if not proof_atoms:
            blockers.append("runtime_consumption_proof_selected_atom_ids_missing")
        missing_atoms = [
            atom for atom in required_selected_atom_ids if str(atom) not in proof_atoms
        ]
        if missing_atoms:
            blockers.append("runtime_consumption_proof_selected_atom_ids_mismatch")

    return apply_proxy_evidence_boundary(
        {
            "schema": RECEIVER_VERIFICATION_SCHEMA,
            "receiver_contract_id": RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
            "receiver_contract_satisfied": not blockers,
            "proof_schema": proof.get("schema") if proof is not None else None,
            "proof_candidate_archive_sha256": (
                proof.get("candidate_archive_sha256") if proof is not None else ""
            ),
            "proof_candidate_member_sha256": (
                proof.get("candidate_member_sha256") if proof is not None else ""
            ),
            "proof_descriptor_packet_sha256": (
                proof.get("descriptor_packet_sha256") if proof is not None else ""
            ),
            "blockers": ordered_unique(blockers),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=blockers,
    )


def pack_inverse_scorer_cell_descriptor(descriptor: Mapping[str, Any]) -> bytes:
    """Return deterministic IAS1 bytes for a descriptor mapping."""

    payload = _canonical_json_bytes(descriptor)
    if len(payload) > 0xFFFFFFFF:
        raise InverseScorerCellMaterializerError("descriptor too large for IAS1")
    return IAS1_MAGIC + len(payload).to_bytes(4, "little") + payload


def unpack_inverse_scorer_cell_descriptor(packet: bytes) -> dict[str, Any]:
    """Parse deterministic IAS1 bytes and return the descriptor mapping."""

    if len(packet) < 8 or packet[:4] != IAS1_MAGIC:
        raise InverseScorerCellMaterializerError("IAS1 descriptor magic missing")
    size = int.from_bytes(packet[4:8], "little")
    payload = packet[8 : 8 + size]
    if len(payload) != size or len(packet) != 8 + size:
        raise InverseScorerCellMaterializerError("IAS1 descriptor length mismatch")
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("schema") != DESCRIPTOR_SCHEMA:
        raise InverseScorerCellMaterializerError("IAS1 descriptor schema mismatch")
    return parsed


def _descriptor_payload(
    *,
    raw_contest_video_digest: str,
    action_record: Mapping[str, Any],
    template_archive: Mapping[str, Any],
    selected_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": DESCRIPTOR_SCHEMA,
        "materializer_id": MATERIALIZER_ID,
        "target_kind": TARGET_KIND,
        "raw_contest_video_digest": raw_contest_video_digest,
        "inverse_action_functional": dict(action_record),
        "template_archive": dict(template_archive),
        "selected_cell_count": len(selected_cells),
        "selected_cells": [_normalize_selected_cell(row) for row in selected_cells],
        **FALSE_AUTHORITY,
    }


def _selected_action_cells(
    action: Mapping[str, Any],
    *,
    atom_ids: Sequence[str],
    selected_limit: int | None,
) -> list[dict[str, Any]]:
    if action.get("schema") != ACTION_FUNCTIONAL_SCHEMA:
        raise InverseScorerCellMaterializerError(
            f"inverse action functional must have schema {ACTION_FUNCTIONAL_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(
            action,
            context="inverse_action_functional",
        )
    except ValueError as exc:
        raise InverseScorerCellMaterializerError(str(exc)) from exc
    if selected_limit is not None and (
        isinstance(selected_limit, bool) or selected_limit < 1
    ):
        raise InverseScorerCellMaterializerError("selected_limit must be >= 1")
    wanted = {str(atom_id) for atom_id in atom_ids if str(atom_id)}
    selected = [
        _normalize_selected_cell(row)
        for row in _mapping(action.get("water_bucket")).get("selected_cells") or []
        if isinstance(row, Mapping)
        and (not wanted or str(row.get("atom_id")) in wanted)
    ]
    if selected_limit is not None:
        selected = selected[:selected_limit]
    return selected


def _normalize_selected_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    atom_id = _clean_str(row.get("atom_id"))
    if not atom_id:
        raise InverseScorerCellMaterializerError("selected cell atom_id missing")
    return {
        "atom_id": atom_id,
        "candidate_id": _clean_str(row.get("candidate_id")),
        "scope_axis": _clean_str(row.get("scope_axis")),
        "component": _clean_str(row.get("component")),
        "water_fill_cost_bytes": _int(row.get("water_fill_cost_bytes"), "water_fill_cost_bytes", minimum=1),
        "expected_score_gain": _float(row.get("expected_score_gain"), "expected_score_gain", minimum=0.0),
        "euler_lagrange_residual": _float(row.get("euler_lagrange_residual"), "euler_lagrange_residual"),
    }


def _candidate_archive_custody(
    record: Mapping[str, Any],
    *,
    repo: Path,
) -> dict[str, Any]:
    path = _resolve_optional_path(record.get("path"), repo)
    blockers: list[str] = []
    if path is None:
        blockers.append("candidate_archive_path_missing_or_unreadable")
        return {"path": _clean_str(record.get("path")), "exists": False, "blockers": blockers}
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    expected_sha = _clean_str(record.get("sha256"))
    if expected_sha and actual_sha != expected_sha:
        blockers.append("candidate_archive_sha_mismatch")
    if record.get("bytes") is not None and int(record["bytes"]) != actual_bytes:
        blockers.append("candidate_archive_bytes_mismatch")
    try:
        member = read_strict_single_member_zip(path)
    except Exception:
        member = None
        blockers.append("candidate_archive_single_member_read_failed")
    expected_member_sha = _clean_str(record.get("member_sha256"))
    if member is not None and expected_member_sha and sha256_bytes(member.payload) != expected_member_sha:
        blockers.append("candidate_archive_member_sha_mismatch")
    return {
        "path": repo_relative(path, repo),
        "exists": True,
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "member_name": "" if member is None else member.member_name,
        "member_sha256": "" if member is None else sha256_bytes(member.payload),
        "blockers": ordered_unique(blockers),
    }


def _candidate_manifest_for_adapter(
    adapter: Mapping[str, Any],
    *,
    candidate_manifest: str | Path | Mapping[str, Any] | None,
    repo: Path,
    adapter_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_manifest is not None:
        payload, record = _load_required_mapping_with_record(
            candidate_manifest,
            repo=repo,
            label="candidate manifest",
        )
    else:
        manifest_ref = _mapping(adapter.get("candidate_manifest"))
        path_text = _clean_str(manifest_ref.get("path"))
        if not path_text:
            raise InverseScorerCellMaterializerError(
                "runtime adapter manifest missing candidate_manifest.path"
            )
        manifest_path = _candidate_manifest_path_from_adapter_record(
            path_text,
            adapter_record=adapter_record,
            repo=repo,
        )
        payload, record = _load_required_mapping_with_record(
            manifest_path,
            repo=repo,
            label="candidate manifest",
        )
        expected_sha = _clean_str(manifest_ref.get("sha256"))
        if expected_sha and record.get("sha256") != expected_sha:
            raise InverseScorerCellMaterializerError(
                "candidate manifest sha256 does not match runtime adapter record"
            )
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise InverseScorerCellMaterializerError(
            f"candidate manifest must have schema {CANDIDATE_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(payload, context="candidate manifest")
    except ValueError as exc:
        raise InverseScorerCellMaterializerError(str(exc)) from exc
    return payload, record


def _optional_archive_template_record(
    value: str | Path | Mapping[str, Any] | None,
    repo: Path,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    path = _resolve_optional_path(value, repo)
    if path is None:
        return {"path": str(value)}
    return {
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _optional_action_functional_record(
    value: str | Path | Mapping[str, Any] | None,
    repo: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if value is None:
        return None, {}
    try:
        return _load_action_functional_record(value, repo)
    except InverseScorerCellMaterializerError:
        return None, {"path": str(value)}


def _load_action_functional_record(
    value: str | Path | Mapping[str, Any],
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _load_required_mapping_with_record(
        value,
        repo=repo,
        label="inverse action functional",
    )
    if payload.get("schema") != ACTION_FUNCTIONAL_SCHEMA:
        raise InverseScorerCellMaterializerError(
            f"inverse action functional must have schema {ACTION_FUNCTIONAL_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(payload, context="inverse action functional")
    except ValueError as exc:
        raise InverseScorerCellMaterializerError(str(exc)) from exc
    return payload, record


def _load_required_mapping_with_record(
    value: str | Path | Mapping[str, Any],
    *,
    repo: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value), {"provided_inline": True, "path": "", "sha256": ""}
    path = _resolve_existing_path(value, repo)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseScorerCellMaterializerError(f"{label} unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise InverseScorerCellMaterializerError(f"{label} is not a JSON object: {path}")
    return payload, {
        "provided_inline": False,
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_optional_mapping(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseScorerCellMaterializerError(
            f"runtime consumption proof unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise InverseScorerCellMaterializerError(
            f"runtime consumption proof is not a JSON object: {path}"
        )
    return payload


def _candidate_manifest_path_from_adapter_record(
    path_text: str,
    *,
    adapter_record: Mapping[str, Any],
    repo: Path,
) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    repo_path = repo / path
    if repo_path.exists():
        return repo_path
    adapter_path_text = _clean_str(adapter_record.get("path"))
    if adapter_path_text:
        adapter_path = Path(adapter_path_text)
        if not adapter_path.is_absolute():
            adapter_path = repo / adapter_path
        sibling_path = adapter_path.parent / path
        if sibling_path.exists():
            return sibling_path
    return repo_path


def _resolve_existing_path(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise InverseScorerCellMaterializerError(f"path does not exist: {path}")
    return path


def _resolve_optional_path(value: Any, repo: Path) -> Path | None:
    text = _clean_str(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = repo / path
    return path if path.exists() else None


def _resolve_output_path(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _refuse_overwrite_unless_expected(
    path: Path,
    *,
    allow_overwrite: bool,
    expected_sha256: str | None,
) -> None:
    if not path.exists():
        return
    if not allow_overwrite:
        raise InverseScorerCellMaterializerError(
            f"{path}: already exists; pass allow_overwrite with expected sha"
        )
    if not expected_sha256:
        raise InverseScorerCellMaterializerError(
            f"{path}: expected_existing_output_sha256 is required before overwrite"
        )
    if sha256_file(path) != expected_sha256:
        raise InverseScorerCellMaterializerError(
            f"{path}: existing sha256 does not match expected"
        )


def _normalize_raw_digest(value: str | Mapping[str, Any] | Any) -> str:
    if isinstance(value, Mapping):
        try:
            require_no_truthy_authority_fields(
                value,
                context="raw_contest_video_digest",
            )
        except ValueError as exc:
            raise InverseScorerCellMaterializerError(str(exc)) from exc
        for key in (
            "raw_contest_video_sha256",
            "raw_video_sha256",
            "raw_sha256",
            "sha256",
            "digest",
        ):
            text = _clean_str(value.get(key))
            if text:
                return text
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _clean_str(value)


def _match_text(
    blockers: list[str],
    actual: Any,
    expected: str | None,
    blocker: str,
) -> None:
    if expected and _clean_str(actual) != _clean_str(expected):
        blockers.append(blocker)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise InverseScorerCellMaterializerError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InverseScorerCellMaterializerError(f"{label} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise InverseScorerCellMaterializerError(f"{label} must be >= {minimum}")
    return result


def _float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise InverseScorerCellMaterializerError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InverseScorerCellMaterializerError(f"{label} must be numeric") from exc
    if minimum is not None and result < minimum:
        raise InverseScorerCellMaterializerError(f"{label} must be >= {minimum}")
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


__all__ = [
    "CANDIDATE_SCHEMA",
    "DESCRIPTOR_SCHEMA",
    "FALSE_AUTHORITY",
    "IAS1_MAGIC",
    "MATERIALIZER_ID",
    "PLAN_SCHEMA",
    "RECEIVER_CONTRACT_ID",
    "RECEIVER_CONTRACT_KIND",
    "RECEIVER_PROOF_SCHEMA",
    "RECEIVER_VERIFICATION_SCHEMA",
    "REQUIRED_CONTEXT_FIELDS",
    "RUNTIME_ADAPTER_SCHEMA",
    "TARGET_KIND",
    "VERIFIED_CANDIDATE_SCHEMA",
    "InverseScorerCellMaterializerError",
    "build_inverse_scorer_cell_candidate_plan",
    "build_inverse_scorer_cell_receiver_proof",
    "materialize_inverse_scorer_cell_candidate",
    "pack_inverse_scorer_cell_descriptor",
    "unpack_inverse_scorer_cell_descriptor",
    "verify_inverse_scorer_cell_candidate_manifest",
    "verify_inverse_scorer_cell_receiver_contract",
]
