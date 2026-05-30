# SPDX-License-Identifier: MIT
"""Reusable adapter spine for archive-bound candidate producers.

New substrate and archive/entropy materializers should implement one small
emitter interface and let this module produce the shared queue package:
contract surface, deterministic replay metadata, MLX advisory request,
receiver-proof gate, exact-axis blocker, and posterior-update hook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    archive_bound_candidate_contract_fields_for_row,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA = (
    "tac_archive_bound_candidate_replay_bundle.v1"
)
ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA = (
    "tac_archive_bound_candidate_mlx_triage_request.v1"
)
ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA = (
    "tac_archive_bound_candidate_receiver_proof_gate.v1"
)
ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA = (
    "tac_archive_bound_candidate_exact_axis_blocker.v1"
)
ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA = (
    "tac_archive_bound_candidate_posterior_update_hook.v1"
)


class ArchiveBoundCandidateAdapterError(ValueError):
    """Raised when an archive-bound adapter emits invalid rows."""


@runtime_checkable
class ArchiveBoundCandidateAdapter(Protocol):
    """Protocol for substrate/archive candidate emitters."""

    adapter_id: str
    candidate_family: str

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """Return false-authority candidate rows for contract normalization."""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contract(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("archive_bound_candidate_contract"))


def _candidate_archive(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_contract(row).get("candidate_archive"))


def _runtime_proof_path(row: Mapping[str, Any]) -> str:
    value = row.get("runtime_consumption_proof_path") or _contract(row).get(
        "runtime_consumption_proof_path"
    )
    return value.strip() if isinstance(value, str) else ""


def _row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("candidate_id", "candidate_chain_id", "typed_response_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    contract_key = _contract(row).get("contract_key")
    if isinstance(contract_key, str) and contract_key.strip():
        return contract_key[:16]
    return f"archive_bound_candidate_{index}"


def _replay_bundle(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    contract = _contract(row)
    candidate_archive = _candidate_archive(row)
    source_archive = _mapping(contract.get("source_archive"))
    blockers = [
        *([] if candidate_archive.get("path") else ["candidate_archive_path_missing"]),
        *([] if candidate_archive.get("sha256") else ["candidate_archive_sha256_missing"]),
    ]
    return {
        "schema": ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA,
        "candidate_id": _row_id(row, index),
        "contract_key": contract.get("contract_key"),
        "candidate_archive": dict(candidate_archive),
        "source_archive": dict(source_archive),
        "runtime_consumption_proof_path": _runtime_proof_path(row),
        "runtime_adapter_manifest": dict(
            _mapping(contract.get("runtime_adapter_manifest"))
        ),
        "replay_argv": _string_list(row.get("replay_argv")),
        "replay_env": dict(_mapping(row.get("replay_env"))),
        "input_artifacts": _string_list(row.get("input_artifacts")),
        "blockers": ordered_unique(blockers),
        "replay_bundle_ready": not blockers,
        "allowed_use": "deterministic_local_replay_metadata_only",
        "forbidden_use": "score_claim_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _mlx_triage_request(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    command = _string_list(row.get("mlx_triage_argv") or row.get("mlx_probe_argv"))
    blockers = [] if command else ["mlx_local_triage_command_missing"]
    return {
        "schema": ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA,
        "candidate_id": _row_id(row, index),
        "contract_key": _contract(row).get("contract_key"),
        "mlx_triage_argv": command,
        "blockers": blockers,
        "ready_for_mlx_local_triage": bool(command),
        "score_authority": False,
        "allowed_use": "macos_mlx_research_signal_for_budget_routing_only",
        "forbidden_use": "score_claim_promotion_rank_or_kill_authority",
        **FALSE_AUTHORITY,
    }


def _receiver_proof_gate(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    contract = _contract(row)
    proof_ready = contract.get("runtime_consumption_proof_ready") is True
    receiver_ok = contract.get("receiver_contract_satisfied") is True
    blockers = [
        *([] if proof_ready else ["receiver_runtime_proof_missing"]),
        *([] if receiver_ok else ["receiver_contract_not_satisfied"]),
    ]
    return {
        "schema": ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA,
        "candidate_id": _row_id(row, index),
        "contract_key": contract.get("contract_key"),
        "runtime_consumption_proof_path": _runtime_proof_path(row),
        "receiver_contract_kind": contract.get("receiver_contract_kind"),
        "receiver_contract_satisfied": receiver_ok,
        "runtime_consumption_proof_ready": proof_ready,
        "blockers": ordered_unique(blockers),
        "receiver_proof_gate_passed": proof_ready and receiver_ok,
        "allowed_use": "receiver_runtime_custody_gate_only",
        "forbidden_use": "score_claim_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _exact_axis_blocker(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    contract = _contract(row)
    blockers = list(_string_list(contract.get("blockers")))
    if contract.get("archive_bound_candidate_ready_for_exact_handoff") is not True:
        blockers.append("archive_bound_candidate_not_ready_for_exact_handoff")
    blockers.extend(
        [
            "contest_cpu_or_cuda_authority_required",
            "lane_preclaim_required_before_exact_eval",
        ]
    )
    return {
        "schema": ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA,
        "candidate_id": _row_id(row, index),
        "contract_key": contract.get("contract_key"),
        "blockers": ordered_unique(blockers),
        "ready_for_exact_eval_dispatch": False,
        "exact_axis_dispatch_allowed": False,
        "allowed_use": "exact_axis_handoff_blocker_until_auth_custody_complete",
        "forbidden_use": "direct_dispatch_or_score_claim",
        **FALSE_AUTHORITY,
    }


def _posterior_hook(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    contract = _contract(row)
    return {
        "schema": ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA,
        "candidate_id": _row_id(row, index),
        "contract_key": contract.get("contract_key"),
        "family_id": contract.get("family_id"),
        "entropy_position_label": contract.get("entropy_position_label"),
        "archive_substrate_tags": _string_list(contract.get("archive_substrate_tags")),
        "posterior_update_trigger": (
            "append_after_mlx_receiver_preclaim_or_exact_axis_result"
        ),
        "negative_result_demotes_family_stage_scope": True,
        "allowed_use": "continual_learning_append_hook_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_archive_bound_candidate_adapter_package(
    adapter: ArchiveBoundCandidateAdapter,
    *,
    context: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one adapter and emit the complete reusable candidate package."""

    if not isinstance(adapter, ArchiveBoundCandidateAdapter):
        raise ArchiveBoundCandidateAdapterError(
            "adapter must satisfy ArchiveBoundCandidateAdapter Protocol"
        )
    adapter_context = dict(context or {})
    raw_rows = adapter.emit_archive_bound_candidate_rows(adapter_context)
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, bytes | bytearray):
        raise ArchiveBoundCandidateAdapterError(
            "emit_archive_bound_candidate_rows must return a sequence of mappings"
        )
    rows: list[dict[str, Any]] = []
    surfaces: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ArchiveBoundCandidateAdapterError(f"row {index} must be a mapping")
        require_no_truthy_authority_fields(
            raw_row,
            context=f"archive_bound_candidate_adapter_row:{index}",
        )
        row = dict(raw_row)
        row.setdefault("candidate_family", adapter.candidate_family)
        row.setdefault("adapter_id", adapter.adapter_id)
        row.update(
            archive_bound_candidate_contract_fields_for_row(
                row,
                repo_root=repo_root,
                family_id=str(row.get("candidate_family") or adapter.candidate_family),
                candidate_chain_id=str(row.get("candidate_id") or _row_id(row, index)),
            )
        )
        surface = row["archive_bound_candidate_contract_surface"]
        if surface.get("schema") != ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA:
            raise ArchiveBoundCandidateAdapterError(
                f"row {index} emitted invalid archive-bound contract surface"
            )
        rows.append(row)
        surfaces.append(surface)
    replay_bundles = [_replay_bundle(row, index=index) for index, row in enumerate(rows)]
    mlx_triage_requests = [
        _mlx_triage_request(row, index=index) for index, row in enumerate(rows)
    ]
    receiver_proof_gates = [
        _receiver_proof_gate(row, index=index) for index, row in enumerate(rows)
    ]
    exact_axis_blockers = [
        _exact_axis_blocker(row, index=index) for index, row in enumerate(rows)
    ]
    posterior_hooks = [_posterior_hook(row, index=index) for index, row in enumerate(rows)]
    package = {
        "schema": ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
        "adapter_id": adapter.adapter_id,
        "candidate_family": adapter.candidate_family,
        "candidate_row_count": len(rows),
        "candidate_rows": rows,
        "archive_bound_candidate_contract_surfaces": surfaces,
        "deterministic_replay_bundles": replay_bundles,
        "mlx_triage_requests": mlx_triage_requests,
        "receiver_proof_gates": receiver_proof_gates,
        "exact_axis_blockers": exact_axis_blockers,
        "posterior_update_hooks": posterior_hooks,
        "ready_contract_count": sum(
            1
            for row in rows
            if _contract(row).get("archive_bound_candidate_ready_for_exact_handoff")
            is True
        ),
        "mlx_triage_ready_count": sum(
            1
            for row in mlx_triage_requests
            if row.get("ready_for_mlx_local_triage") is True
        ),
        "receiver_proof_gate_passed_count": sum(
            1
            for row in receiver_proof_gates
            if row.get("receiver_proof_gate_passed") is True
        ),
        "allowed_use": "archive_bound_candidate_pipeline_package_only",
        "forbidden_use": "score_claim_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        package,
        context=f"archive_bound_candidate_adapter_package:{adapter.adapter_id}",
    )
    return package


__all__ = [
    "ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA",
    "ArchiveBoundCandidateAdapter",
    "ArchiveBoundCandidateAdapterError",
    "build_archive_bound_candidate_adapter_package",
]
