# SPDX-License-Identifier: MIT
"""Harvest completed materializer chains into optimizer candidate rows.

Materializer chain manifests are custody evidence, not dispatch authority. This
adapter validates the live archive/artifact surface and emits the planning-row
shape consumed by ``tools/build_optimizer_candidate_queue.py``.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    PACKET_MEMBER_MERGE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
    RENDERER_PAYLOAD_DFL1_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.optimizer.exact_readiness import validate_serialized_archive_delta_contract

SUPPORTED_CHAIN_SCHEMAS = frozenset(
    {
        "byte_range_entropy_recode_chain_v1",
        "inverse_scorer_cell_candidate_chain_v1",
    }
)
SUPPORTED_FAMILY_AGNOSTIC_MATERIALIZER_SCHEMAS = frozenset(
    {
        ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
        PACKET_MEMBER_MERGE_SCHEMA,
        PACKET_MEMBER_RECOMPRESS_SCHEMA,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
        RENDERER_PAYLOAD_DFL1_SCHEMA,
        TENSOR_FACTORIZE_SCHEMA,
    }
)
SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS = (
    SUPPORTED_CHAIN_SCHEMAS | SUPPORTED_FAMILY_AGNOSTIC_MATERIALIZER_SCHEMAS
)
TOOL_NAME = "tools/build_optimizer_candidate_queue.py"
LOCAL_ADVISORY_AXIS_TOKENS = (
    "macos-cpu-advisory",
    "macos-cpu",
    "cpu-advisory",
    "macos-mlx",
    "mlx-research-signal",
    "locality-control",
)


class MaterializerChainHarvestError(ValueError):
    """Raised when a materializer chain cannot be harvested safely."""


def adapt_materializer_chain_manifest_to_candidate(
    chain: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one non-authoritative optimizer candidate row for ``chain``."""

    schema = str(chain.get("schema") or "")
    if schema not in SUPPORTED_CHAIN_SCHEMAS:
        raise MaterializerChainHarvestError(f"unsupported_chain_schema:{schema!r}")
    _require_false_authority(chain, label="chain")
    _require_chain_complete(chain)
    _require_serialized_archive_delta(chain)

    candidate_archive = _archive_record(chain, "candidate_archive", repo_root=repo_root)
    source_archive = _archive_record(chain, "source_archive", repo_root=repo_root)
    delta_blockers, delta_facts = validate_serialized_archive_delta_contract(
        chain,
        actual_candidate_archive_bytes=candidate_archive["bytes"],
    )
    delta_blockers.extend(
        _serialized_delta_archive_custody_blockers(
            delta_facts,
            source_archive=source_archive,
            candidate_archive=candidate_archive,
        )
    )
    if delta_blockers:
        raise MaterializerChainHarvestError(
            "serialized_archive_delta_blocked:" + ",".join(delta_blockers)
        )
    _validate_chain_artifacts(chain, repo_root=repo_root)

    source_sha = _string_or_none(source_archive.get("sha256"))
    source_bytes = _positive_int(source_archive.get("bytes"))
    archive_changed = (
        source_sha is not None and candidate_archive["sha256"] != source_sha
    )
    byte_changed = (
        source_bytes is not None and candidate_archive["bytes"] != source_bytes
    )
    delta_status = str(delta_facts.get("expected_status") or "").strip()
    realized_saved_bytes = delta_facts.get("computed_realized_saved_bytes")
    rate_positive = (
        delta_status == "realized_saving"
        and delta_facts.get("expected_savings_realized") is True
    )
    rate_semantics = (
        "realized_archive_saving"
        if rate_positive
        else "successful_quality_spend_not_byte_saving_progress"
    )
    candidate_id = _candidate_id(
        chain, schema=schema, archive_sha=candidate_archive["sha256"]
    )
    row = {
        "candidate_id": candidate_id,
        "lane_id": str(chain.get("lane_id") or f"materializer_harvest::{schema}"),
        "lane_class": "materializer_chain_harvest",
        "candidate_family": _candidate_family(schema),
        "optimizer_tool": TOOL_NAME,
        "schema": schema,
        "target_kind": chain.get("target_kind"),
        "materializer_id": chain.get("materializer_id"),
        "receiver_contract_kind": chain.get("receiver_contract_kind"),
        "source_manifest_path": _repo_rel(source_path, repo_root),
        "source_paths": [_repo_rel(source_path, repo_root)],
        "candidate_archive_path": candidate_archive["path"],
        "archive_path": candidate_archive["path"],
        "candidate_archive_sha256": candidate_archive["sha256"],
        "archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_bytes": candidate_archive["bytes"],
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_bytes,
        "source_archive_path": source_archive.get("path"),
        "serialized_archive_delta": dict(chain.get("serialized_archive_delta") or {}),
        "serialized_archive_delta_validated": delta_facts,
        "materializer_rate_outcome": chain.get("materializer_rate_outcome")
        or delta_status,
        "rate_positive": rate_positive,
        "realized_saved_bytes": realized_saved_bytes,
        "signal_semantics": chain.get("signal_semantics") or rate_semantics,
        "quality_spend_allowed": chain.get("quality_spend_allowed") is True,
        "score_affecting_payload_changed": archive_changed,
        "charged_bits_changed": byte_changed,
        "score_affecting_change_proof": _score_affecting_change_proof(
            source_sha=source_sha,
            source_bytes=source_bytes,
            candidate_archive=candidate_archive,
            archive_changed=archive_changed,
            byte_changed=byte_changed,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": _string_list(chain.get("readiness_blockers")),
        "next_required_gates": _string_list(chain.get("next_required_gates")),
        "chain_artifact_count": len(chain.get("artifacts") or {}),
        "chain_step_count": len(chain.get("chain_steps") or []),
        **_runtime_consumption_proof_fields(chain),
        **_chain_runtime_context_fields(chain),
        "local_advisory_axes": _local_advisory_axes(chain),
        "local_advisory_axes_semantics": (
            "non_authoritative_planning_signal_only_not_score_claim"
        ),
        "evidence_semantics": (
            "materializer_chain_harvest_candidate_pending_exact_readiness:"
            f"{rate_semantics}"
        ),
        "evidence_grade": "[materializer-chain-harvest-no-score]",
        "harvested_at_utc": _utc_now(),
    }
    out = apply_proxy_evidence_boundary(
        row,
        dispatch_blockers=[
            "materializer_chain_is_not_dispatch_authorization",
            "materialized_archive_runtime_custody_required",
            "exact_readiness_promotion_required",
            "exact_auth_eval_result_required_before_score_claim",
            *_string_list(chain.get("readiness_blockers")),
            *_string_list(chain.get("dispatch_blockers")),
        ],
    )
    out["score_affecting_payload_changed"] = archive_changed
    out["charged_bits_changed"] = byte_changed
    return out


def adapt_materializer_manifest_to_candidate(
    manifest: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one non-authoritative optimizer row for any materializer manifest."""

    schema = str(manifest.get("schema") or "")
    if schema in SUPPORTED_CHAIN_SCHEMAS:
        return adapt_materializer_chain_manifest_to_candidate(
            manifest,
            source_path=source_path,
            repo_root=repo_root,
        )
    if schema in SUPPORTED_FAMILY_AGNOSTIC_MATERIALIZER_SCHEMAS:
        return adapt_family_agnostic_materializer_manifest_to_candidate(
            manifest,
            source_path=source_path,
            repo_root=repo_root,
        )
    raise MaterializerChainHarvestError(f"unsupported_materializer_schema:{schema!r}")


def adapt_family_agnostic_materializer_manifest_to_candidate(
    manifest: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Adapt one family-agnostic byte-closed candidate manifest for readiness."""

    schema = str(manifest.get("schema") or "")
    if schema not in SUPPORTED_FAMILY_AGNOSTIC_MATERIALIZER_SCHEMAS:
        raise MaterializerChainHarvestError(
            f"unsupported_family_agnostic_materializer_schema:{schema!r}"
        )
    _require_false_authority(manifest, label="family_agnostic_materializer_manifest")
    if manifest.get("byte_closed_candidate_emitted") is not True:
        raise MaterializerChainHarvestError("byte_closed_candidate_emitted_not_true")

    candidate_archive = _archive_record(manifest, "candidate_archive", repo_root=repo_root)
    source_archive = _archive_record(manifest, "source_archive", repo_root=repo_root)
    source_sha = _string_or_none(source_archive.get("sha256"))
    source_bytes = _positive_int(source_archive.get("bytes"))
    archive_changed = (
        source_sha is not None and candidate_archive["sha256"] != source_sha
    )
    byte_changed = (
        source_bytes is not None and candidate_archive["bytes"] != source_bytes
    )
    serialized_delta = build_serialized_archive_delta_contract(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
    )
    delta_status = str(serialized_delta.get("status") or "").strip()
    realized_saved_bytes = serialized_delta.get("realized_saved_bytes")
    rate_positive = (
        delta_status == "realized_saving"
        and serialized_delta.get("savings_realized") is True
    )
    receiver_verification = manifest.get("receiver_verification")
    receiver_map = (
        receiver_verification if isinstance(receiver_verification, Mapping) else {}
    )
    proof_present = receiver_map.get("proof_present") is True
    receiver_satisfied = manifest.get("receiver_contract_satisfied") is True
    # At this queue boundary the flag means "no runtime-adapter blocker remains",
    # not necessarily "a separate adapter file exists".  Identity-style
    # materializers can have no generated adapter while still being ready once
    # the runtime-consumption proof is verified.
    runtime_adapter_ready = receiver_satisfied
    raw_proof_path = (
        receiver_map.get("proof_path") or manifest.get("runtime_consumption_proof_path")
    )
    proof_path = raw_proof_path.strip() if isinstance(raw_proof_path, str) else None
    candidate_member = _zip_member_record_from_manifest(
        manifest,
        "candidate_member",
        archive_path=_record_path(
            manifest["candidate_archive"],
            repo_root=repo_root,
            label="candidate_archive",
        ),
    )
    source_member = _zip_member_record_from_manifest(
        manifest,
        "source_member",
        archive_path=_record_path(
            manifest["source_archive"],
            repo_root=repo_root,
            label="source_archive",
        ),
    )
    candidate_id = _candidate_id(
        manifest,
        schema=schema,
        archive_sha=candidate_archive["sha256"],
    )
    row = {
        "candidate_id": candidate_id,
        "lane_id": str(manifest.get("lane_id") or f"materializer_harvest::{schema}"),
        "lane_class": "family_agnostic_materializer_harvest",
        "candidate_family": _candidate_family(schema),
        "optimizer_tool": TOOL_NAME,
        "schema": schema,
        "target_kind": manifest.get("target_kind"),
        "materializer_id": manifest.get("materializer_id"),
        "receiver_contract_kind": manifest.get("receiver_contract_kind"),
        "source_manifest_path": _repo_rel(source_path, repo_root),
        "source_paths": [_repo_rel(source_path, repo_root)],
        "candidate_archive_path": candidate_archive["path"],
        "archive_path": candidate_archive["path"],
        "candidate_archive_sha256": candidate_archive["sha256"],
        "archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_bytes": candidate_archive["bytes"],
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_bytes,
        "source_archive_path": source_archive.get("path"),
        **_member_candidate_fields("candidate", candidate_member),
        **_member_candidate_fields("source", source_member),
        "serialized_archive_delta": serialized_delta,
        "materializer_rate_outcome": manifest.get("materializer_rate_outcome")
        or delta_status,
        "rate_positive": rate_positive,
        "realized_saved_bytes": realized_saved_bytes,
        "signal_semantics": (
            "realized_archive_saving"
            if rate_positive
            else "successful_quality_spend_not_byte_saving_progress"
        ),
        "score_affecting_payload_changed": archive_changed,
        "charged_bits_changed": byte_changed,
        "score_affecting_change_proof": _score_affecting_change_proof(
            source_sha=source_sha,
            source_bytes=source_bytes,
            candidate_archive=candidate_archive,
            archive_changed=archive_changed,
            byte_changed=byte_changed,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": runtime_adapter_ready,
        "receiver_contract_satisfied": receiver_satisfied,
        "candidate_runtime_adapter_blocker_cleared": runtime_adapter_ready,
        "readiness_blockers": _string_list(manifest.get("readiness_blockers")),
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present" if proof_present else "missing",
        "runtime_consumption_proof_path": proof_path,
        **_submission_runtime_harvest_fields(manifest),
        **_packet_member_merge_harvest_fields(manifest),
        **_renderer_payload_dfl1_harvest_fields(
            manifest,
            runtime_proof=_load_optional_runtime_proof(proof_path, repo_root=repo_root),
            repo_root=repo_root,
        ),
        "local_advisory_axes": _local_advisory_axes(manifest),
        "local_advisory_axes_semantics": (
            "non_authoritative_planning_signal_only_not_score_claim"
        ),
        "evidence_semantics": (
            "family_agnostic_materializer_candidate_pending_exact_readiness"
        ),
        "evidence_grade": "[family-agnostic-materializer-no-score]",
        "harvested_at_utc": _utc_now(),
    }
    out = apply_proxy_evidence_boundary(
        row,
        dispatch_blockers=[
            "materializer_candidate_is_not_dispatch_authorization",
            "materialized_archive_runtime_custody_required",
            "exact_readiness_promotion_required",
            "exact_auth_eval_result_required_before_score_claim",
            *(
                []
                if receiver_satisfied
                else ["family_agnostic_receiver_contract_not_satisfied"]
            ),
            *_string_list(manifest.get("readiness_blockers")),
            *_string_list(manifest.get("dispatch_blockers")),
        ],
    )
    out["score_affecting_payload_changed"] = archive_changed
    out["charged_bits_changed"] = byte_changed
    out["runtime_adapter_ready"] = runtime_adapter_ready
    out["receiver_contract_satisfied"] = receiver_satisfied
    out["candidate_runtime_adapter_blocker_cleared"] = runtime_adapter_ready
    return out


def _submission_runtime_harvest_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "source_runtime_dir",
        "source_submission_dir",
        "submission_dir",
        "candidate_runtime_dir",
    ):
        value = _nonempty_string(manifest.get(key))
        if value is not None:
            fields[key] = value
    inflate_runtime_dir = _nonempty_string(manifest.get("inflate_runtime_dir"))
    if inflate_runtime_dir is not None and "source_runtime_dir" not in fields:
        fields["source_runtime_dir"] = inflate_runtime_dir
    return fields


def _packet_member_merge_harvest_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != PACKET_MEMBER_MERGE_SCHEMA:
        return {}
    fields: dict[str, Any] = {
        "selected_member_names": _string_list(manifest.get("selected_member_names")),
    }
    runtime = manifest.get("packet_member_merge_receiver_runtime")
    if not isinstance(runtime, Mapping):
        return fields
    fields["packet_member_merge_receiver_runtime"] = dict(runtime)
    runtime_dir = _nonempty_string(runtime.get("runtime_dir"))
    if runtime_dir is not None:
        fields["candidate_runtime_dir"] = runtime_dir
        fields["packet_member_merge_runtime_dir"] = runtime_dir
    runtime_manifest_path = _nonempty_string(runtime.get("runtime_manifest_path"))
    if runtime_manifest_path is not None:
        fields["packet_member_merge_runtime_manifest_path"] = runtime_manifest_path
    source_runtime_dir = _nonempty_string(runtime.get("source_runtime_dir"))
    if source_runtime_dir is not None:
        fields["packet_member_merge_source_runtime_dir"] = source_runtime_dir
    runtime_tree_sha = _string_or_none(runtime.get("runtime_tree_sha256"))
    if runtime_tree_sha is not None:
        fields["candidate_runtime_tree_sha256"] = runtime_tree_sha
        fields["packet_member_merge_receiver_runtime_tree_sha256"] = runtime_tree_sha
    return fields


def _renderer_payload_dfl1_harvest_fields(
    manifest: Mapping[str, Any],
    *,
    runtime_proof: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if manifest.get("schema") != RENDERER_PAYLOAD_DFL1_SCHEMA:
        return {}
    fields: dict[str, Any] = {
        "renderer_payload_dfl1_anatomy_semantics": (
            "non_authoritative_planning_signal_only"
        ),
        "selected_member_names": _string_list(manifest.get("selected_member_names")),
    }
    raw_payload_member_name = manifest.get("payload_member_name")
    if isinstance(raw_payload_member_name, str) and raw_payload_member_name.strip():
        fields["payload_member_name"] = raw_payload_member_name.strip()
    selected_payload = manifest.get("selected_payload")
    if isinstance(selected_payload, Mapping):
        fields["selected_payload"] = dict(selected_payload)
    parity_verification = manifest.get("full_frame_inflate_parity_verification")
    if isinstance(parity_verification, Mapping):
        fields["full_frame_inflate_parity_verification"] = dict(parity_verification)
        fields["full_frame_inflate_parity_proven"] = (
            parity_verification.get("full_frame_inflate_parity_satisfied") is True
        )
        proof_path = _nonempty_string(parity_verification.get("proof_path"))
        if proof_path is not None:
            fields["renderer_payload_dfl1_inflate_parity_proof_path"] = proof_path
        proof_sha = _string_or_none(parity_verification.get("proof_sha256"))
        if proof_sha is not None:
            fields["renderer_payload_dfl1_inflate_parity_proof_sha256"] = proof_sha
        fields["renderer_payload_dfl1_inflate_parity_satisfied"] = (
            parity_verification.get("full_frame_inflate_parity_satisfied") is True
        )
        proof_path = _nonempty_string(parity_verification.get("proof_path"))
        if proof_path is not None:
            fields["renderer_payload_dfl1_full_frame_inflate_parity_proof_path"] = (
                proof_path
            )
            resolved_proof = Path(proof_path)
            if not resolved_proof.is_absolute():
                resolved_proof = repo_root / resolved_proof
            if resolved_proof.is_file() and not resolved_proof.is_symlink():
                fields[
                    "renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256"
                ] = _sha256_file(resolved_proof)
        fields["renderer_payload_dfl1_full_frame_inflate_parity_satisfied"] = (
            parity_verification.get("full_frame_inflate_parity_satisfied") is True
        )
    payload_table = runtime_proof.get("payload_table")
    if isinstance(payload_table, Mapping):
        fields["payload_table"] = dict(payload_table)
    runtime_parity_verification = runtime_proof.get(
        "full_frame_inflate_parity_verification"
    )
    if isinstance(runtime_parity_verification, Mapping):
        fields["runtime_full_frame_inflate_parity_verification"] = dict(
            runtime_parity_verification
        )
    if runtime_proof.get("source_runtime_unpacker_parse_satisfied") is True:
        fields["source_runtime_unpacker_parse_satisfied"] = True
    reconstructed = runtime_proof.get("reconstructed_member_sha256s")
    if isinstance(reconstructed, Mapping):
        fields["reconstructed_member_sha256s"] = {
            str(name): str(sha)
            for name, sha in reconstructed.items()
            if str(name) and str(sha)
        }
    runtime_probe = runtime_proof.get("runtime_consumption_probe")
    native_probe = (
        runtime_probe.get("native_unpacker_probe")
        if isinstance(runtime_probe, Mapping)
        else None
    )
    if isinstance(native_probe, Mapping):
        native_member_sha256s = native_probe.get("member_sha256s")
        if isinstance(native_member_sha256s, Mapping):
            fields["native_unpacker_member_sha256s"] = {
                str(name): str(sha)
                for name, sha in native_member_sha256s.items()
                if str(name) and str(sha)
            }
    return fields


def _load_optional_runtime_proof(
    proof_path: str | None,
    *,
    repo_root: Path,
) -> Mapping[str, Any]:
    if proof_path is None:
        return {}
    path = Path(proof_path)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file() or resolved.is_symlink():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _zip_member_record_from_manifest(
    manifest: Mapping[str, Any],
    key: str,
    *,
    archive_path: Path,
) -> dict[str, Any]:
    raw = manifest.get(key)
    if not isinstance(raw, Mapping):
        return {}
    name = _string_or_none(raw.get("name") or raw.get("member_name"))
    if name is None:
        return {}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            try:
                payload = archive.read(name)
            except KeyError as exc:
                raise MaterializerChainHarvestError(f"{key}_missing_in_archive:{name}") from exc
    except zipfile.BadZipFile as exc:
        raise MaterializerChainHarvestError(f"{key}_archive_not_zip") from exc
    observed_sha = hashlib.sha256(payload).hexdigest()
    observed_bytes = len(payload)
    declared_sha = _string_or_none(raw.get("sha256") or raw.get("member_sha256"))
    if declared_sha is not None and declared_sha != observed_sha:
        raise MaterializerChainHarvestError(f"{key}_sha256_mismatch")
    declared_bytes_value = raw.get("bytes")
    if declared_bytes_value is None:
        declared_bytes_value = raw.get("member_bytes")
    declared_bytes = _non_negative_int(declared_bytes_value)
    if declared_bytes is not None and declared_bytes != observed_bytes:
        raise MaterializerChainHarvestError(f"{key}_bytes_mismatch")
    return {
        **dict(raw),
        "name": name,
        "sha256": observed_sha,
        "bytes": observed_bytes,
    }


def _member_candidate_fields(prefix: str, member: Mapping[str, Any]) -> dict[str, Any]:
    if not member:
        return {}
    return {
        f"{prefix}_member_name": member.get("name"),
        f"{prefix}_member_sha256": member.get("sha256"),
        f"{prefix}_member_bytes": member.get("bytes"),
        f"{prefix}_member": dict(member),
    }


def _runtime_consumption_proof_fields(chain: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "runtime_consumption_proof_required",
        "runtime_consumption_proof_status",
        "runtime_consumption_proof_path",
    ):
        if key in chain:
            out[key] = chain[key]
    if "runtime_consumption_proof_path" not in out:
        artifacts = chain.get("artifacts")
        receiver = (
            artifacts.get("receiver_proof")
            if isinstance(artifacts, Mapping)
            else None
        )
        proof_path = (
            _nonempty_string(receiver.get("path"))
            if isinstance(receiver, Mapping)
            else None
        )
        if proof_path is not None:
            out["runtime_consumption_proof_required"] = True
            out["runtime_consumption_proof_status"] = "present"
            out["runtime_consumption_proof_path"] = proof_path
    return out


def _chain_runtime_context_fields(chain: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "source_runtime_dir",
        "inflate_runtime_dir",
        "candidate_runtime_dir",
        "runtime_dir",
        "source_submission_dir",
        "submission_dir",
    ):
        value = _nonempty_string(chain.get(key))
        if value is not None:
            out[key] = value
    return out


def _candidate_id(chain: Mapping[str, Any], *, schema: str, archive_sha: str) -> str:
    value = chain.get("candidate_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"{_candidate_family(schema)}_{archive_sha[:12]}"


def _candidate_family(schema: str) -> str:
    if schema == "byte_range_entropy_recode_chain_v1":
        return "byte_range_entropy_recode"
    if schema == "inverse_scorer_cell_candidate_chain_v1":
        return "inverse_scorer_cell"
    if schema == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA:
        return "archive_section_entropy_recode"
    if schema == PACKET_MEMBER_MERGE_SCHEMA:
        return "packet_member_merge"
    if schema == PACKET_MEMBER_RECOMPRESS_SCHEMA:
        return "packet_member_recompress"
    if schema == PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA:
        return "packet_member_zip_header_elide"
    if schema == RENDERER_PAYLOAD_DFL1_SCHEMA:
        return "renderer_payload_dfl1"
    if schema == TENSOR_FACTORIZE_SCHEMA:
        return "tensor_factorize"
    return "materializer_chain"


def _require_chain_complete(chain: Mapping[str, Any]) -> None:
    required_true = (
        "byte_closed_candidate_emitted",
        "runtime_adapter_ready",
        "receiver_contract_satisfied",
        "candidate_runtime_adapter_blocker_cleared",
    )
    for key in required_true:
        if chain.get(key) is not True:
            raise MaterializerChainHarvestError(f"{key}_not_true")


def _require_serialized_archive_delta(chain: Mapping[str, Any]) -> None:
    raw = chain.get("serialized_archive_delta")
    if raw is None:
        raise MaterializerChainHarvestError("serialized_archive_delta_missing")
    if not isinstance(raw, Mapping):
        raise MaterializerChainHarvestError("serialized_archive_delta_not_object")


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MaterializerChainHarvestError(str(exc)) from exc


def _archive_record(
    chain: Mapping[str, Any],
    key: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    raw = chain.get(key)
    if not isinstance(raw, Mapping):
        raise MaterializerChainHarvestError(f"{key}_missing")
    path = _record_path(raw, repo_root=repo_root, label=key)
    observed_sha = _sha256_file(path)
    observed_bytes = path.stat().st_size
    declared_sha = _string_or_none(raw.get("sha256") or raw.get("archive_sha256"))
    declared_bytes = _positive_int(raw.get("bytes") or raw.get("archive_bytes"))
    if declared_sha is None:
        raise MaterializerChainHarvestError(f"{key}_sha256_missing")
    if declared_sha != observed_sha:
        raise MaterializerChainHarvestError(f"{key}_sha256_mismatch")
    if declared_bytes is None:
        raise MaterializerChainHarvestError(f"{key}_bytes_missing")
    if declared_bytes != observed_bytes:
        raise MaterializerChainHarvestError(f"{key}_bytes_mismatch")
    return {
        **dict(raw),
        "path": _repo_rel(path, repo_root),
        "sha256": observed_sha,
        "bytes": observed_bytes,
    }


def _serialized_delta_archive_custody_blockers(
    delta_facts: Mapping[str, Any],
    *,
    source_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    source_bytes = _positive_int(delta_facts.get("source_archive_bytes"))
    candidate_bytes = _positive_int(delta_facts.get("candidate_archive_bytes"))
    if source_bytes != source_archive["bytes"]:
        blockers.append(
            "serialized_archive_delta_source_bytes_mismatch:"
            f"{source_bytes}!={source_archive['bytes']}"
        )
    if candidate_bytes != candidate_archive["bytes"]:
        blockers.append(
            "serialized_archive_delta_candidate_bytes_mismatch:"
            f"{candidate_bytes}!={candidate_archive['bytes']}"
        )
    return blockers


def _record_path(record: Mapping[str, Any], *, repo_root: Path, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value.strip():
        raise MaterializerChainHarvestError(f"{label}_path_missing")
    path = Path(value)
    raw_path = path if path.is_absolute() else repo_root / path
    if raw_path.is_symlink():
        raise MaterializerChainHarvestError(f"{label}_file_is_symlink:{path}")
    resolved = raw_path.resolve(strict=False)
    if not path.is_absolute():
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            raise MaterializerChainHarvestError(f"{label}_path_outside_repo") from None
    if not resolved.is_file():
        raise MaterializerChainHarvestError(f"{label}_file_missing:{path}")
    return resolved


def _validate_chain_artifacts(chain: Mapping[str, Any], *, repo_root: Path) -> None:
    artifacts = chain.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise MaterializerChainHarvestError("chain_artifacts_missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise MaterializerChainHarvestError(f"artifact_record_not_object:{name}")
        _validate_artifact_record(record, repo_root=repo_root, label=f"artifact:{name}")
    steps = chain.get("chain_steps")
    if not isinstance(steps, list) or not steps:
        raise MaterializerChainHarvestError("chain_steps_missing")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise MaterializerChainHarvestError(f"chain_step_not_object:{index}")
        if step.get("status") != "succeeded":
            raise MaterializerChainHarvestError(f"chain_step_not_succeeded:{index}")
        artifact = step.get("artifact")
        if isinstance(artifact, Mapping):
            _validate_artifact_record(
                artifact,
                repo_root=repo_root,
                label=f"chain_step_artifact:{index}",
            )


def _validate_artifact_record(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> None:
    path = _record_path(record, repo_root=repo_root, label=label)
    declared_sha = _string_or_none(record.get("sha256"))
    if declared_sha is None:
        raise MaterializerChainHarvestError(f"{label}_sha256_missing")
    if _sha256_file(path) != declared_sha:
        raise MaterializerChainHarvestError(f"{label}_sha256_mismatch")
    declared_bytes = _positive_int(record.get("bytes"))
    if declared_bytes is None:
        raise MaterializerChainHarvestError(f"{label}_bytes_missing")
    if path.stat().st_size != declared_bytes:
        raise MaterializerChainHarvestError(f"{label}_bytes_mismatch")


def _score_affecting_change_proof(
    *,
    source_sha: str | None,
    source_bytes: int | None,
    candidate_archive: Mapping[str, Any],
    archive_changed: bool,
    byte_changed: bool,
) -> dict[str, Any]:
    return {
        "source_archive_sha256": source_sha,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_changed": archive_changed,
        "byte_different": byte_changed,
    }


def _local_advisory_axes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mapping in _iter_mappings(payload):
        axis = mapping.get("score_axis") or mapping.get("axis")
        if not isinstance(axis, str):
            continue
        token = _axis_token(axis)
        if not any(item in token for item in LOCAL_ADVISORY_AXIS_TOKENS):
            continue
        out.append(
            {
                "score_axis": axis,
                "score": mapping.get("score") or mapping.get("score_recomputed"),
                "evidence_grade": mapping.get("evidence_grade"),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        )
    return _dedupe_advisory_axes(out)


def _dedupe_advisory_axes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = repr(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for inner in value.values():
            yield from _iter_mappings(inner)
    elif isinstance(value, list | tuple):
        for inner in value:
            yield from _iter_mappings(inner)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(
        value, Mapping | bytes | bytearray
    ):
        return ordered_unique(str(item) for item in value if str(item))
    return [str(value)]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _axis_token(value: str) -> str:
    return value.strip().strip("[]").lower().replace("_", "-").replace(" ", "-")


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "SUPPORTED_CHAIN_SCHEMAS",
    "SUPPORTED_FAMILY_AGNOSTIC_MATERIALIZER_SCHEMAS",
    "SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS",
    "MaterializerChainHarvestError",
    "adapt_family_agnostic_materializer_manifest_to_candidate",
    "adapt_materializer_chain_manifest_to_candidate",
    "adapt_materializer_manifest_to_candidate",
]
