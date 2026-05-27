# SPDX-License-Identifier: MIT
"""Harvest completed materializer chains into optimizer source queues.

The experiment queue tells us what ran; the chain manifest tells us what was
actually produced. This module treats queue state as a filter only, then
revalidates live chain manifests through ``tac.optimizer.materializer_chain_harvest``
before emitting planning-only optimizer queue rows.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME as BYTE_RANGE_CHAIN_MANIFEST_NAME,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_SCHEMA as BYTE_RANGE_CHAIN_SCHEMA,
)
from tac.optimization.family_agnostic_materializers import (
    RENDERER_PAYLOAD_DFL1_SCHEMA,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    verify_renderer_payload_dfl1_full_frame_inflate_parity_proof,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_SCORER_CELL_CHAIN_MANIFEST_NAME,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_SCHEMA as INVERSE_SCORER_CELL_CHAIN_SCHEMA,
)
from tac.optimization.materializer_feedback import (
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.runtime_adapter_identity import RUNTIME_TREE_SHA_FIELDS
from tac.optimizer.candidate_queue import build_candidate_queue
from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ExactReadinessError,
    promote_candidate_for_exact_eval,
)
from tac.optimizer.exact_readiness import (
    json_dumps as exact_readiness_json_dumps,
)
from tac.optimizer.materializer_chain_harvest import (
    SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS,
    MaterializerChainHarvestError,
    adapt_materializer_manifest_to_candidate,
    revalidate_runtime_consumption_proof_for_candidate,
)
from tac.repo_io import sha256_file

from .byte_shaving_campaign_queue import (
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
)
from .experiment_queue import ExperimentQueueError, connect_state_readonly

HARVEST_SCHEMA = "materializer_chain_harvest_report.v1"
EXACT_READINESS_BRIDGE_SCHEMA = "materializer_chain_exact_readiness_bridge_report.v1"
TOOL_NAME = "comma_lab.scheduler.materializer_chain_harvest"
EXACT_READINESS_BRIDGE_TOOL = (
    "comma_lab.scheduler.materializer_chain_harvest.exact_readiness_bridge"
)
MATERIALIZER_HARVEST_CLEARABLE_SOURCE_BLOCKERS = (
    "materializer_chain_is_not_dispatch_authorization",
    "materialized_archive_runtime_custody_required",
    "materializer_chain_harvest_candidate_pending_exact_readiness",
    "materializer_candidate_is_not_dispatch_authorization",
    "family_agnostic_materializer_candidate_pending_exact_readiness",
    "exact_readiness_promotion_required",
    "exact_auth_eval_result_required_before_score_claim",
    "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
    "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
)
MATERIALIZER_BRIDGE_OPERATOR_CLEARABLE_SOURCE_BLOCKER_ALLOWLIST: frozenset[str] = (
    frozenset()
)
CHAIN_MANIFEST_NAME_BY_SCHEMA = {
    BYTE_RANGE_CHAIN_SCHEMA: BYTE_RANGE_CHAIN_MANIFEST_NAME,
    INVERSE_SCORER_CELL_CHAIN_SCHEMA: INVERSE_SCORER_CELL_CHAIN_MANIFEST_NAME,
}
KNOWN_CHAIN_MANIFEST_NAMES = frozenset(CHAIN_MANIFEST_NAME_BY_SCHEMA.values())
RUNTIME_CONTEXT_PATH_FIELDS = (
    "source_runtime_dir",
    "source_submission_dir",
    "source_inflate_sh_path",
    "candidate_submission_dir",
    "candidate_inflate_sh_path",
    "candidate_archive_path",
    "inflate_runtime_dir",
    "packet_member_merge_source_runtime_dir",
    "renderer_payload_dfl1_source_runtime_dir",
    "tensor_factorize_source_runtime_dir",
    "runtime_consumption_proof_path",
    "runtime_consumption_proof_out",
)
RUNTIME_CONTEXT_NESTED_IDENTITY_FIELDS = (
    "runtime_manifest",
    "candidate_runtime",
    "receiver_verification",
    "packet_member_merge_receiver_runtime",
    "tensor_factorize_receiver_runtime",
    "renderer_payload_dfl1_receiver_runtime",
)
RUNTIME_CONTEXT_IDENTITY_FIELDS = tuple(
    ordered_unique(
        [
            *RUNTIME_TREE_SHA_FIELDS,
            "runtime_content_tree_sha256",
            "candidate_runtime_content_tree_sha256",
            "submission_runtime_tree_sha256",
            "submission_runtime_content_tree_sha256",
            "expected_runtime_tree_sha256",
            "expected_inflate_runtime_tree_sha256",
            "expected_candidate_runtime_tree_sha256",
            "byte_range_entropy_recode_runtime_tree_sha256",
            "packet_member_merge_receiver_runtime_tree_sha256",
            "tensor_factorize_receiver_runtime_tree_sha256",
            "renderer_payload_dfl1_runtime_tree_sha256",
        ]
    )
)
_SHA256_HEX = frozenset("0123456789abcdef")
RUNTIME_PROOF_REVALIDATION_STALE_BLOCKERS = frozenset(
    {
        "family_agnostic_receiver_contract_not_satisfied",
        "materializer_chain_receiver_contract_not_satisfied",
        "packet_member_recompress_receiver_contract_not_satisfied",
        "packet_member_zip_header_elide_receiver_contract_not_satisfied",
        "runtime_consumption_proof_missing",
        "runtime_consumption_proof_not_passed",
        "runtime_consumption_proof_path_missing",
        "runtime_consumption_proof_file_missing",
    }
)


def harvest_materializer_chain_manifests(
    *,
    repo_root: str | Path,
    work_queue_path: str | Path | None = None,
    experiment_queue_state_path: str | Path | None = None,
    experiment_queue_id: str | None = None,
    chain_manifest_paths: Sequence[str | Path] = (),
    chain_roots: Sequence[str | Path] = (),
    sweep_manifest_specs: Sequence[str | Path] = (),
    renderer_payload_dfl1_inflate_parity_proofs: Sequence[str | Path] = (),
    allowed_artifact_roots: Sequence[str | Path] = (),
    require_succeeded_state: bool = True,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return a harvest report and planning-only optimizer source queue."""

    if top_k is not None and (isinstance(top_k, bool) or top_k < 1):
        raise ExperimentQueueError("top_k must be >= 1 when provided")
    if experiment_queue_state_path is not None and not str(experiment_queue_id or "").strip():
        raise ExperimentQueueError(
            "experiment_queue_id is required when experiment_queue_state_path is provided"
        )
    repo = Path(repo_root)
    state_rows = _load_state_rows(
        experiment_queue_state_path,
        experiment_queue_id=experiment_queue_id,
    )
    discoveries = _discover_chain_manifest_candidates(
        repo_root=repo,
        work_queue_path=work_queue_path,
        chain_manifest_paths=chain_manifest_paths,
        chain_roots=chain_roots,
        sweep_manifest_specs=sweep_manifest_specs,
    )

    accepted_paths: list[Path] = []
    accepted_discoveries: list[dict[str, Any]] = []
    inspected_rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for discovery in discoveries:
        path = _resolve_path(discovery["path"], repo_root=repo)
        path_key = path.resolve(strict=False)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        row = {
            "path": _repo_rel(path, repo),
            "source": discovery.get("source"),
            "work_id": discovery.get("work_id"),
            "declared_schema": discovery.get("schema"),
            "state_rows": _state_rows_for_discovery(discovery, state_rows),
            "accepted": False,
            "blockers": [],
        }
        for key in (
            "sweep_manifest_path",
            "sweep_observation_id",
            "sweep_rate_positive",
            "sweep_receiver_contract_satisfied",
        ):
            if key in discovery:
                row[key] = discovery.get(key)
        state_blockers = _state_blockers(
            row,
            require_succeeded_state=require_succeeded_state,
            state_filter_active=experiment_queue_state_path is not None,
        )
        if state_blockers:
            row["blockers"] = state_blockers
            inspected_rows.append(row)
            continue
        blockers, observed_schema = _validate_materializer_manifest(path, repo_root=repo)
        row["observed_schema"] = observed_schema
        if blockers:
            row["blockers"] = blockers
            inspected_rows.append(row)
            continue
        row["accepted"] = True
        accepted_paths.append(path)
        accepted_discoveries.append(dict(discovery))
        inspected_rows.append(row)

    source_queue = apply_proxy_evidence_boundary(
        build_candidate_queue(accepted_paths, repo_root=repo, top_k=top_k),
        dispatch_blockers=[
            "materializer_chain_harvest_source_queue_is_planning_only",
            "submission_runtime_closure_required_before_dispatch",
            "exact_readiness_promotion_required_before_dispatch",
        ],
    )
    _apply_discovery_runtime_context(source_queue, accepted_discoveries, repo_root=repo)
    runtime_proof_revalidation_report = (
        _apply_runtime_consumption_proof_revalidation_for_runtime_context(
            source_queue,
            repo_root=repo,
        )
    )
    dfl1_sidecar_report = _apply_renderer_payload_dfl1_sidecar_parity_proofs(
        source_queue,
        renderer_payload_dfl1_inflate_parity_proofs,
        repo_root=repo,
        allowed_artifact_roots=allowed_artifact_roots,
    )
    accepted_rows = [row for row in inspected_rows if row["accepted"] is True]
    rejected_rows = [row for row in inspected_rows if row["accepted"] is not True]
    report = apply_proxy_evidence_boundary(
        {
            "schema": HARVEST_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "work_queue_path": _repo_rel(_resolve_path(work_queue_path, repo_root=repo), repo)
            if work_queue_path is not None
            else None,
            "experiment_queue_state_path": _repo_rel(
                _resolve_path(experiment_queue_state_path, repo_root=repo),
                repo,
            )
            if experiment_queue_state_path is not None
            else None,
            "experiment_queue_id": experiment_queue_id,
            "require_succeeded_state": require_succeeded_state,
            "discovered_manifest_count": len(discoveries),
            "sweep_manifest_count": len(sweep_manifest_specs),
            "unique_manifest_count": len(inspected_rows),
            "accepted_manifest_count": len(accepted_rows),
            "rejected_manifest_count": len(rejected_rows),
            "accepted_manifest_paths": [row["path"] for row in accepted_rows],
            "rows": inspected_rows,
            "source_queue_schema": source_queue["schema"],
            "source_queue_candidate_count": source_queue["n_candidates"],
            "source_queue_dispatch_ready_count": source_queue["dispatch_ready_count"],
            "runtime_consumption_proof_revalidation": (
                runtime_proof_revalidation_report
            ),
            "renderer_payload_dfl1_sidecar_parity": dfl1_sidecar_report,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=[
            "materializer_chain_harvest_is_source_queue_only",
            "exact_readiness_promotion_required_before_dispatch",
            "lane_claim_required_before_gpu_or_remote_eval",
        ],
    )
    return {"report": report, "source_queue": source_queue}


def run_exact_readiness_bridge_for_harvested_queue(
    *,
    repo_root: str | Path,
    source_queue_path: str | Path,
    exact_readiness_out_dir: str | Path,
    candidate_ids: Sequence[str] = (),
    allow_source_blockers: Sequence[str] = (),
    dispatch_claims_path: str | Path | None = None,
    claim_ttl_hours: float = 24.0,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> dict[str, Any]:
    """Run the exact-readiness gate for harvested materializer source rows.

    The returned report is an observation artifact. Only per-candidate
    ``*_exact_ready_queue.json`` outputs from the existing promoter are dispatch
    packets, and those still require a lane claim before provider launch.
    """

    repo = Path(repo_root)
    queue_path = _resolve_path(source_queue_path, repo_root=repo)
    out_dir = _resolve_path(exact_readiness_out_dir, repo_root=repo)
    if allow_above_active_floor_dispatch and not operator_override_reason:
        raise ExperimentQueueError(
            "allow_above_active_floor_dispatch requires operator_override_reason"
        )
    queue_payload = _load_json(queue_path)
    if not isinstance(queue_payload, Mapping):
        raise ExperimentQueueError("source queue must be an object")
    if queue_payload.get("schema") != "optimizer_candidate_queue_v1":
        raise ExperimentQueueError(
            f"expected optimizer_candidate_queue_v1, got {queue_payload.get('schema')!r}"
        )
    dispatch_ready_rows = queue_payload.get("dispatch_ready")
    if isinstance(dispatch_ready_rows, list) and dispatch_ready_rows:
        raise ExperimentQueueError(
            "exact_readiness_bridge_source_queue_must_not_have_dispatch_ready_rows"
        )
    candidate_filter = {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
    rows = [
        row
        for row in queue_payload.get("top_k") or []
        if isinstance(row, Mapping)
        and (
            not candidate_filter
            or str(row.get("candidate_id") or "") in candidate_filter
        )
    ]
    if candidate_filter:
        found = {str(row.get("candidate_id") or "") for row in rows}
        missing = sorted(candidate_filter - found)
        if missing:
            raise ExperimentQueueError(
                "exact_readiness_candidate_id_missing:" + ",".join(missing)
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    extra_clearable_source_blockers = _validated_bridge_extra_clearable_source_blockers(
        allow_source_blockers,
        operator_override_reason=operator_override_reason,
    )
    clearable_source_blockers = ordered_unique(
        [
            *MATERIALIZER_HARVEST_CLEARABLE_SOURCE_BLOCKERS,
            *extra_clearable_source_blockers,
        ]
    )
    _require_bridge_source_queue_identity(rows, candidate_filter=candidate_filter)
    resolved_dispatch_claims_path = (
        _resolve_path(dispatch_claims_path, repo_root=repo)
        if dispatch_claims_path is not None
        else repo / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    report_rows: list[dict[str, Any]] = []
    ready_count = 0
    skipped_count = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        slug = _safe_slug(candidate_id)
        per_candidate_report_path = out_dir / f"{slug}.exact_readiness_report.json"
        exact_ready_queue_path = out_dir / f"{slug}.exact_ready_queue.json"
        skip_reason = _exact_readiness_skip_reason_for_harvested_row(row)
        if skip_reason is not None:
            readiness_report = {
                "schema": "optimizer_candidate_exact_eval_readiness_report_v1",
                "tool": "tools/promote_optimizer_candidate_for_exact_eval.py",
                "generated_at_utc": _utc_now(),
                "source_queue_path": _repo_rel(queue_path, repo),
                "candidate_id": candidate_id,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [skip_reason],
                "facts": {
                    "materializer_exact_readiness_skipped": True,
                    "materializer_exact_readiness_skip_reason": skip_reason,
                    "rate_positive": row.get("rate_positive"),
                    "realized_saved_bytes": row.get("realized_saved_bytes"),
                    "serialized_archive_delta": row.get("serialized_archive_delta"),
                    **_exact_readiness_skip_custody_facts(row),
                },
            }
            per_candidate_report_path.write_text(
                exact_readiness_json_dumps(readiness_report),
                encoding="utf-8",
            )
            skipped_count += 1
            report_rows.append(
                {
                    "candidate_id": candidate_id,
                    "readiness_verdict": "skipped_non_rate_positive_materializer",
                    "exact_ready_queue_written": False,
                    "exact_readiness_report_path": _repo_rel(
                        per_candidate_report_path,
                        repo,
                    ),
                    "exact_ready_queue_path": None,
                    "blockers": [skip_reason],
                }
            )
            continue
        try:
            result = promote_candidate_for_exact_eval(
                queue_path,
                candidate_id,
                repo_root=repo,
                active_floor_archive_bytes=active_floor_archive_bytes,
                active_floor_score=active_floor_score,
                allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
                operator_override_reason=operator_override_reason,
                extra_clearable_source_blockers=clearable_source_blockers,
                dispatch_claims_path=resolved_dispatch_claims_path,
                claim_ttl_hours=claim_ttl_hours,
            )
            readiness_report = result["report"]
            promoted_queue = result["promoted_queue"]
        except ExactReadinessError as exc:
            readiness_report = {
                "schema": "optimizer_candidate_exact_eval_readiness_report_v1",
                "tool": "tools/promote_optimizer_candidate_for_exact_eval.py",
                "generated_at_utc": _utc_now(),
                "source_queue_path": _repo_rel(queue_path, repo),
                "candidate_id": candidate_id,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [str(exc)],
                "facts": {},
            }
            promoted_queue = None
        per_candidate_report_path.write_text(
            exact_readiness_json_dumps(readiness_report),
            encoding="utf-8",
        )
        ready = promoted_queue is not None
        if ready:
            exact_ready_queue_path.write_text(
                exact_readiness_json_dumps(promoted_queue),
                encoding="utf-8",
            )
            ready_count += 1
        report_rows.append(
            {
                "candidate_id": candidate_id,
                "readiness_verdict": "exact_ready_queue_written" if ready else "blocked",
                "exact_ready_queue_written": ready,
                "exact_readiness_report_path": _repo_rel(per_candidate_report_path, repo),
                "exact_ready_queue_path": _repo_rel(exact_ready_queue_path, repo)
                if ready
                else None,
                "blockers": list(readiness_report.get("blockers") or []),
            }
        )

    report = apply_proxy_evidence_boundary(
        {
            "schema": EXACT_READINESS_BRIDGE_SCHEMA,
            "tool": EXACT_READINESS_BRIDGE_TOOL,
            "generated_at_utc": _utc_now(),
            "source_queue_path": _repo_rel(queue_path, repo),
            "exact_readiness_out_dir": _repo_rel(out_dir, repo),
            "candidate_count": len(rows),
            "ready_candidate_count": ready_count,
            "skipped_candidate_count": skipped_count,
            "blocked_candidate_count": len(rows) - ready_count - skipped_count,
            "clearable_source_blockers": clearable_source_blockers,
            "operator_clearable_source_blockers": extra_clearable_source_blockers,
            "dispatch_claims_path": _repo_rel(resolved_dispatch_claims_path, repo),
            "rows": report_rows,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=[
            "bridge_report_is_not_dispatch_authority",
            "use_per_candidate_exact_ready_queue_only_when_present",
            "lane_claim_required_before_gpu_or_remote_eval",
        ],
    )
    require_no_truthy_authority_fields(
        report,
        context="materializer_chain_exact_readiness_bridge_report",
    )
    return report


def _validated_bridge_extra_clearable_source_blockers(
    blockers: Sequence[str],
    *,
    operator_override_reason: str | None,
) -> list[str]:
    extras = ordered_unique(str(item) for item in blockers if str(item))
    if not extras:
        return []
    if not operator_override_reason:
        raise ExperimentQueueError(
            "exact_readiness_extra_source_blocker_requires_operator_override_reason"
        )
    not_allowed = [
        blocker
        for blocker in extras
        if blocker not in MATERIALIZER_BRIDGE_OPERATOR_CLEARABLE_SOURCE_BLOCKER_ALLOWLIST
    ]
    if not_allowed:
        raise ExperimentQueueError(
            "exact_readiness_extra_source_blocker_not_allowlisted:"
            + ",".join(sorted(not_allowed))
        )
    return extras


def _exact_readiness_skip_reason_for_harvested_row(
    row: Mapping[str, Any],
) -> str | None:
    if not _is_materializer_candidate_row(row):
        return None
    if row.get("rate_positive") is False:
        return "materializer_candidate_not_rate_positive_for_exact_readiness"
    saved_bytes = row.get("realized_saved_bytes")
    if isinstance(saved_bytes, bool):
        return None
    if isinstance(saved_bytes, int | float) and saved_bytes <= 0:
        return "materializer_candidate_not_rate_positive_for_exact_readiness"
    delta = row.get("serialized_archive_delta")
    if isinstance(delta, Mapping):
        if delta.get("rate_positive") is False:
            return "materializer_candidate_not_rate_positive_for_exact_readiness"
        delta_saved = delta.get("realized_saved_bytes")
        if (
            isinstance(delta_saved, int | float)
            and not isinstance(delta_saved, bool)
            and delta_saved <= 0
        ):
            return "materializer_candidate_not_rate_positive_for_exact_readiness"
        for key in ("status", "materializer_rate_outcome", "expected_status"):
            value = str(delta.get(key) or "").strip().lower()
            if value in {"zero_delta", "size_regression", "not_rate_positive"}:
                return "materializer_candidate_not_rate_positive_for_exact_readiness"
    blockers = [
        *_text_values(row.get("dispatch_blockers")),
        *_text_values(row.get("readiness_blockers")),
        *_text_values(row.get("blockers")),
    ]
    if any(
        blocker == "candidate_not_rate_positive"
        or blocker.endswith(":candidate_not_rate_positive")
        for blocker in blockers
    ):
        return "materializer_candidate_not_rate_positive_for_exact_readiness"
    return None


def _exact_readiness_skip_custody_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "archive_bytes",
        "archive_path",
        "archive_sha256",
        "candidate_archive_bytes",
        "candidate_archive_path",
        "candidate_archive_sha256",
        "candidate_member_bytes",
        "candidate_member_name",
        "candidate_member_sha256",
        "charged_bits_changed",
        "materializer_rate_outcome",
        "receiver_contract_satisfied",
        "runtime_consumption_proof_path",
        "runtime_consumption_proof_required",
        "runtime_consumption_proof_status",
        "runtime_content_tree_sha256",
        "runtime_tree_sha256",
        "score_affecting_change_proof",
        "score_affecting_payload_changed",
        "source_archive_bytes",
        "source_archive_path",
        "source_archive_sha256",
        "source_inflate_sh_path",
        "source_member_bytes",
        "source_member_name",
        "source_member_sha256",
        "source_runtime_dir",
        "source_submission_dir",
        "submission_dir",
        "submission_runtime_content_tree_sha256",
        "submission_runtime_tree_sha256",
    )
    return {key: row.get(key) for key in keys if key in row}


def _is_materializer_candidate_row(row: Mapping[str, Any]) -> bool:
    schema = str(row.get("schema") or "")
    materializer_id = str(row.get("materializer_id") or "")
    target_kind = str(row.get("target_kind") or "")
    candidate_family = str(row.get("candidate_family") or "")
    return (
        schema in CHAIN_MANIFEST_NAME_BY_SCHEMA
        or schema in SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS
        or "materializer" in schema
        or bool(materializer_id)
        or (
            target_kind.endswith("_v1")
            and (
                "materializer" in candidate_family
                or "serialized_archive_delta" in row
            )
        )
    )


def _apply_renderer_payload_dfl1_sidecar_parity_proofs(
    source_queue: dict[str, Any],
    proof_refs: Sequence[str | Path],
    *,
    repo_root: Path,
    allowed_artifact_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    proofs = [_resolve_path(ref, repo_root=repo_root) for ref in proof_refs]
    allowed_roots = [
        _resolve_path(root, repo_root=repo_root) for root in allowed_artifact_roots
    ]
    rows = [
        row
        for row in source_queue.get("top_k") or []
        if isinstance(row, dict) and _is_renderer_payload_dfl1_row(row)
    ]
    report_rows: list[dict[str, Any]] = []
    applied_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate_sha = str(
            row.get("candidate_archive_sha256") or row.get("archive_sha256") or ""
        ).lower()
        source_sha = str(row.get("source_archive_sha256") or "").lower()
        row_report = {
            "candidate_id": candidate_id,
            "applied": False,
            "proof_path": None,
            "blockers": [],
        }
        if not proofs:
            row_report["blockers"] = [
                "renderer_payload_dfl1_parity_proof_missing"
            ]
            report_rows.append(row_report)
            continue
        for proof_path in proofs:
            proof_report = _renderer_payload_dfl1_sidecar_parity_overlay(
                row,
                proof_path=proof_path,
                repo_root=repo_root,
                allowed_artifact_roots=allowed_roots,
                required_source_archive_sha256=source_sha,
                required_candidate_archive_sha256=candidate_sha,
            )
            if proof_report["applied"] is True:
                row_report.update(proof_report)
                applied_ids.add(candidate_id)
                break
            row_report["blockers"].extend(proof_report["blockers"])
        row_report["blockers"] = ordered_unique(row_report["blockers"])
        report_rows.append(row_report)
    if applied_ids:
        _mirror_dfl1_sidecar_overlay_to_queue_lists(source_queue, applied_ids)
    return {
        "proof_count": len(proofs),
        "candidate_count": len(rows),
        "applied_candidate_count": len(applied_ids),
        "rows": report_rows,
    }


def _renderer_payload_dfl1_sidecar_parity_overlay(
    row: dict[str, Any],
    *,
    proof_path: Path,
    repo_root: Path,
    allowed_artifact_roots: Sequence[Path] = (),
    required_source_archive_sha256: str,
    required_candidate_archive_sha256: str,
) -> dict[str, Any]:
    rel_proof = _repo_rel_no_resolve(proof_path, repo_root)
    blockers: list[str] = []
    if not proof_path.is_file():
        blockers.append(f"renderer_payload_dfl1_parity_proof_missing:{rel_proof}")
    if proof_path.is_symlink():
        blockers.append(f"renderer_payload_dfl1_parity_proof_is_symlink:{rel_proof}")
    if not _path_is_repo_confined(proof_path, repo_root) and not _path_is_under_any_root(
        proof_path,
        allowed_artifact_roots,
    ):
        blockers.append(
            f"renderer_payload_dfl1_parity_proof_outside_allowed_roots:{rel_proof}"
        )
    if not required_source_archive_sha256 or not required_candidate_archive_sha256:
        blockers.append("renderer_payload_dfl1_parity_archive_sha_missing")
    if blockers:
        return {
            "applied": False,
            "proof_path": rel_proof,
            "blockers": blockers,
        }
    verification = verify_renderer_payload_dfl1_full_frame_inflate_parity_proof(
        full_frame_inflate_parity_proof=proof_path,
        required_source_archive_sha256=required_source_archive_sha256,
        required_candidate_archive_sha256=required_candidate_archive_sha256,
        repo_root=repo_root,
    )
    if verification.get("full_frame_inflate_parity_satisfied") is not True:
        return {
            "applied": False,
            "proof_path": rel_proof,
            "blockers": [
                "renderer_payload_dfl1_parity_proof_not_satisfied",
                *_text_values(verification.get("blockers")),
            ],
        }
    proof_sha = sha256_file(proof_path)
    verification = dict(verification)
    verification["proof_path"] = rel_proof
    verification["proof_sha256"] = proof_sha
    stale_blockers = {
        "family_agnostic_receiver_contract_not_satisfied",
        "renderer_payload_dfl1_full_frame_inflate_parity_missing",
        "renderer_payload_dfl1_full_frame_inflate_parity_not_satisfied",
        "renderer_payload_dfl1_full_frame_inflate_parity_proof_missing",
        "renderer_payload_dfl1_receiver_contract_not_satisfied",
        "renderer_payload_dfl1_v1_runtime_adapter_not_ready",
        "runtime_consumption_proof_not_passed",
    }
    row.update(
        {
            "full_frame_inflate_parity_verification": verification,
            "full_frame_inflate_parity_proven": True,
            "receiver_contract_satisfied": True,
            "renderer_payload_dfl1_inflate_parity_satisfied": True,
            "renderer_payload_dfl1_inflate_parity_proof_path": rel_proof,
            "renderer_payload_dfl1_inflate_parity_proof_sha256": proof_sha,
            "renderer_payload_dfl1_full_frame_inflate_parity_satisfied": True,
            "renderer_payload_dfl1_full_frame_inflate_parity_proof_path": rel_proof,
            "renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256": proof_sha,
            "runtime_consumption_proof_status": "sidecar_full_frame_parity_applied",
        }
    )
    for key in ("readiness_blockers", "dispatch_blockers", "blockers"):
        if key in row:
            row[key] = [
                blocker
                for blocker in _text_values(row.get(key))
                if blocker not in stale_blockers
            ]
    source_paths = [
        str(item) for item in row.get("source_paths") or [] if isinstance(item, str)
    ]
    if rel_proof not in source_paths:
        row["source_paths"] = [*source_paths, rel_proof]
    return {
        "applied": True,
        "proof_path": rel_proof,
        "proof_sha256": proof_sha,
        "blockers": [],
    }


def _mirror_dfl1_sidecar_overlay_to_queue_lists(
    source_queue: dict[str, Any],
    applied_ids: set[str],
) -> None:
    source_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in source_queue.get("top_k") or []
        if isinstance(row, dict)
    }
    for list_name in ("top_k_forensic", "dispatch_ready"):
        for row in source_queue.get(list_name) or []:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id not in applied_ids:
                continue
            source = source_by_id.get(candidate_id)
            if source is None or source is row:
                continue
            for key in (
                "full_frame_inflate_parity_verification",
                "full_frame_inflate_parity_proven",
                "receiver_contract_satisfied",
                "renderer_payload_dfl1_inflate_parity_satisfied",
                "renderer_payload_dfl1_inflate_parity_proof_path",
                "renderer_payload_dfl1_inflate_parity_proof_sha256",
                "renderer_payload_dfl1_full_frame_inflate_parity_satisfied",
                "renderer_payload_dfl1_full_frame_inflate_parity_proof_path",
                "renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256",
                "runtime_consumption_proof_status",
                "readiness_blockers",
                "dispatch_blockers",
                "blockers",
                "source_paths",
            ):
                if key in source:
                    row[key] = source[key]


def _apply_discovery_runtime_context(
    source_queue: dict[str, Any],
    discoveries: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> None:
    runtime_context_by_path: dict[str, dict[str, Any]] = {}
    for discovery in discoveries:
        context = _runtime_context_from_discovery(discovery)
        if not context:
            continue
        manifest_path = _resolve_path(discovery["path"], repo_root=repo_root)
        runtime_context_by_path[_repo_rel(manifest_path, repo_root)] = context
    if not runtime_context_by_path:
        return

    for list_name in ("top_k", "top_k_forensic", "dispatch_ready"):
        for row in source_queue.get(list_name) or []:
            if not isinstance(row, dict):
                continue
            context = _runtime_context_for_queue_row(
                row,
                runtime_context_by_path,
                repo_root=repo_root,
            )
            if context is None:
                continue
            _overlay_runtime_context(row, context)


def _runtime_context_for_queue_row(
    row: Mapping[str, Any],
    runtime_context_by_path: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
) -> Mapping[str, Any] | None:
    candidate_paths: list[str] = []
    for key in ("source_manifest_path", "source_path", "manifest_path"):
        value = _nonempty_text(row.get(key))
        if value is not None:
            candidate_paths.append(
                _repo_rel(_resolve_path(value, repo_root=repo_root), repo_root)
            )
    for value in row.get("source_paths") or []:
        text = _nonempty_text(value)
        if text is not None:
            candidate_paths.append(
                _repo_rel(_resolve_path(text, repo_root=repo_root), repo_root)
            )
    for path in ordered_unique(candidate_paths):
        context = runtime_context_by_path.get(path)
        if context is not None:
            return context
    return None


def _overlay_runtime_context(row: dict[str, Any], context: Mapping[str, Any]) -> None:
    applied: list[str] = []
    identity_conflicts: list[str] = []
    for key in RUNTIME_CONTEXT_PATH_FIELDS:
        value = _nonempty_text(context.get(key))
        if value is None:
            continue
        if _nonempty_text(row.get(key)) is None:
            row[key] = value
            applied.append(key)
    for key in RUNTIME_CONTEXT_IDENTITY_FIELDS:
        value = _sha256_text(context.get(key))
        if value is None:
            continue
        existing = _sha256_text(row.get(key))
        if existing is None:
            row[key] = value
            applied.append(key)
            continue
        if existing == value:
            applied.append(key)
            continue
        if existing != value:
            identity_conflicts.append(
                f"materializer_harvest_runtime_context_tree_sha256_conflict:{key}"
            )
    sources = [
        source
        for source in context.get("materializer_harvest_runtime_context_sources") or []
        if isinstance(source, Mapping)
    ]
    if sources:
        existing = [
            source
            for source in row.get("materializer_harvest_runtime_context_sources") or []
            if isinstance(source, Mapping)
        ]
        seen = {
            json.dumps(source, sort_keys=True, separators=(",", ":"))
            for source in existing
        }
        merged_sources = list(existing)
        for source in sources:
            key = json.dumps(source, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            merged_sources.append(dict(source))
        row["materializer_harvest_runtime_context_sources"] = merged_sources
    if applied:
        row["materializer_harvest_runtime_context_applied_fields"] = ordered_unique(
            [
                *[
                    str(field)
                    for field in row.get(
                        "materializer_harvest_runtime_context_applied_fields"
                    )
                    or []
                    if str(field)
                ],
                *applied,
            ]
        )
    if identity_conflicts:
        row["materializer_harvest_runtime_context_identity_blockers"] = (
            ordered_unique(
                [
                    *[
                        str(blocker)
                        for blocker in row.get(
                            "materializer_harvest_runtime_context_identity_blockers"
                        )
                        or []
                        if str(blocker)
                    ],
                    *identity_conflicts,
                ]
            )
        )
        row["runtime_adapter_ready"] = False
        row["candidate_runtime_adapter_blocker_cleared"] = False
        row["receiver_contract_satisfied"] = False
        for key in ("readiness_blockers", "dispatch_blockers", "blockers"):
            row[key] = ordered_unique(
                [
                    *_text_values(row.get(key)),
                    *identity_conflicts,
                ]
            )
        row["ready_for_exact_eval_dispatch"] = False


def _apply_runtime_consumption_proof_revalidation_for_runtime_context(
    source_queue: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    applied_ids: set[str] = set()
    blocked_ids: set[str] = set()
    for list_name in ("top_k", "top_k_forensic", "dispatch_ready"):
        for row in source_queue.get(list_name) or []:
            if not isinstance(row, dict) or not _is_materializer_candidate_row(row):
                continue
            if not _runtime_context_proof_revalidation_required(row):
                continue
            revalidation = revalidate_runtime_consumption_proof_for_candidate(
                row,
                repo_root=repo_root,
            )
            _overlay_runtime_proof_revalidation(row, revalidation)
            candidate_id = str(row.get("candidate_id") or "")
            if revalidation.get("receiver_contract_satisfied") is True:
                applied_ids.add(candidate_id)
            else:
                blocked_ids.add(candidate_id)
            rows.append(
                {
                    "list_name": list_name,
                    "candidate_id": candidate_id,
                    "runtime_consumption_proof_path": revalidation.get(
                        "runtime_consumption_proof_path"
                    ),
                    "runtime_consumption_proof_status": revalidation.get(
                        "runtime_consumption_proof_status"
                    ),
                    "receiver_contract_satisfied": revalidation.get(
                        "receiver_contract_satisfied"
                    )
                    is True,
                    "blockers": list(revalidation.get("blockers") or []),
                }
            )
    return {
        "candidate_count": len(applied_ids | blocked_ids),
        "row_count": len(rows),
        "revalidated_candidate_count": len(applied_ids),
        "blocked_candidate_count": len(blocked_ids),
        "rows": rows,
    }


def _overlay_runtime_proof_revalidation(
    row: dict[str, Any],
    revalidation: Mapping[str, Any],
) -> None:
    blockers = _text_values(revalidation.get("blockers"))
    proof_status = _nonempty_text(revalidation.get("runtime_consumption_proof_status"))
    proof_path = _nonempty_text(revalidation.get("runtime_consumption_proof_path"))
    row["runtime_consumption_proof_revalidation"] = dict(revalidation)
    if proof_status is not None:
        row["runtime_consumption_proof_status"] = proof_status
    if proof_path is not None:
        row["runtime_consumption_proof_path"] = proof_path
    proof_sha = _sha256_text(revalidation.get("runtime_consumption_proof_sha256"))
    if proof_sha is not None:
        row["runtime_consumption_proof_sha256"] = proof_sha
    proof_schema = _nonempty_text(revalidation.get("runtime_consumption_proof_schema"))
    if proof_schema is not None:
        row["runtime_consumption_proof_schema"] = proof_schema
    if revalidation.get("receiver_contract_satisfied") is True:
        row["receiver_contract_satisfied"] = True
        row["runtime_adapter_ready"] = True
        row["candidate_runtime_adapter_blocker_cleared"] = True
        for key in ("readiness_blockers", "dispatch_blockers", "blockers"):
            if key in row:
                row[key] = [
                    blocker
                    for blocker in _text_values(row.get(key))
                    if not _runtime_proof_revalidation_stale_blocker(blocker)
                ]
        return

    row["receiver_contract_satisfied"] = False
    row["runtime_adapter_ready"] = False
    row["candidate_runtime_adapter_blocker_cleared"] = False
    for key in ("readiness_blockers", "dispatch_blockers", "blockers"):
        row[key] = ordered_unique(
            [
                *[
                    blocker
                    for blocker in _text_values(row.get(key))
                    if blocker != "runtime_consumption_proof_path_missing"
                ],
                *blockers,
            ]
        )


def _runtime_context_proof_revalidation_required(row: Mapping[str, Any]) -> bool:
    if _nonempty_text(
        row.get("runtime_consumption_proof_path")
        or row.get("runtime_consumption_proof_out")
    ) is None:
        return False
    applied_fields = {
        str(field)
        for field in row.get("materializer_harvest_runtime_context_applied_fields")
        or []
        if str(field)
    }
    return (
        "runtime_consumption_proof_path" in applied_fields
        or "runtime_consumption_proof_out" in applied_fields
    )


def _runtime_proof_revalidation_stale_blocker(blocker: str) -> bool:
    if blocker in RUNTIME_PROOF_REVALIDATION_STALE_BLOCKERS:
        return True
    if blocker.startswith("runtime_consumption_proof"):
        return True
    if blocker.startswith("receiver_verification:runtime_consumption_proof"):
        return True
    return blocker.endswith("_receiver_contract_not_satisfied")


def _is_renderer_payload_dfl1_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("schema") == RENDERER_PAYLOAD_DFL1_SCHEMA
        or row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
        or row.get("candidate_family") == "renderer_payload_dfl1"
    )


def _path_is_repo_confined(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(repo_root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _path_is_under_any_root(path: Path, roots: Sequence[Path]) -> bool:
    resolved_path = path.resolve(strict=False)
    for root in roots:
        try:
            resolved_path.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        return True
    return False


def _text_values(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item)]


def _require_bridge_source_queue_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_filter: set[str],
) -> None:
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    missing_ids = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            missing_ids += 1
            continue
        if candidate_id in seen:
            duplicate_ids.add(candidate_id)
        seen.add(candidate_id)
    if missing_ids:
        raise ExperimentQueueError("exact_readiness_candidate_id_missing_in_source_row")
    if duplicate_ids:
        raise ExperimentQueueError(
            "exact_readiness_duplicate_candidate_id:"
            + ",".join(sorted(duplicate_ids))
        )
    if candidate_filter and not rows:
        raise ExperimentQueueError("exact_readiness_candidate_filter_matched_no_rows")


def _discover_chain_manifest_candidates(
    *,
    repo_root: Path,
    work_queue_path: str | Path | None,
    chain_manifest_paths: Sequence[str | Path],
    chain_roots: Sequence[str | Path],
    sweep_manifest_specs: Sequence[str | Path],
) -> list[dict[str, Any]]:
    discoveries: list[dict[str, Any]] = []
    explicit_chain_paths = [
        _resolve_path(raw_path, repo_root=repo_root).resolve(strict=False)
        for raw_path in chain_manifest_paths
    ]
    explicit_chain_path_set = set(explicit_chain_paths)
    matched_explicit_paths: set[Path] = set()
    for raw_spec in sweep_manifest_specs:
        work_id, path = _parse_sweep_manifest_spec(raw_spec)
        discoveries.extend(
            _sweep_manifest_candidates(
                path,
                repo_root=repo_root,
                work_id=work_id,
                source="explicit_sweep_manifest",
            )
        )
    if work_queue_path is not None:
        work_queue_discoveries = _work_queue_manifest_candidates(
            _load_json(_resolve_path(work_queue_path, repo_root=repo_root)),
            work_queue_path=_resolve_path(work_queue_path, repo_root=repo_root),
            repo_root=repo_root,
        )
        for discovery in work_queue_discoveries:
            if explicit_chain_path_set:
                discovery_path = _resolve_path(
                    discovery["path"],
                    repo_root=repo_root,
                ).resolve(strict=False)
                if discovery_path not in explicit_chain_path_set:
                    continue
                matched_explicit_paths.add(discovery_path)
            discoveries.append(discovery)
    for raw_path in chain_manifest_paths:
        resolved = _resolve_path(raw_path, repo_root=repo_root).resolve(strict=False)
        if resolved in matched_explicit_paths:
            continue
        discoveries.append(
            {
                "source": "explicit_chain_manifest",
                "path": str(raw_path),
                "schema": None,
                "work_id": None,
            }
        )
    for raw_root in chain_roots:
        discoveries.extend(_chain_root_manifest_candidates(raw_root, repo_root=repo_root))
    return discoveries


def _work_queue_manifest_candidates(
    payload: Any,
    *,
    work_queue_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ExperimentQueueError("materializer work queue must be an object")
    if payload.get("schema") != MATERIALIZER_WORK_QUEUE_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_WORK_QUEUE_SCHEMA}")
    discoveries: list[dict[str, Any]] = []
    for index, raw_row in enumerate(payload.get("rows") or []):
        if not isinstance(raw_row, Mapping):
            raise ExperimentQueueError(f"materializer work queue row {index} must be an object")
        try:
            require_no_truthy_authority_fields(
                raw_row,
                context=f"materializer_work_queue.rows.{index}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        work_id = str(raw_row.get("work_id") or "")
        runtime_context = _runtime_context_from_work_queue_row(
            raw_row,
            work_id=work_id or None,
            work_queue_path=work_queue_path,
            repo_root=repo_root,
        )
        for condition in raw_row.get("postconditions") or []:
            if not isinstance(condition, Mapping):
                continue
            condition_type = condition.get("type")
            if condition_type == "materializer_chain_complete":
                schema = str(condition.get("schema") or "")
                path = condition.get("path")
            elif condition_type == "json_completion_contract":
                required_equals = condition.get("required_equals")
                schema = str(
                    required_equals.get("schema")
                    if isinstance(required_equals, Mapping)
                    else condition.get("schema") or ""
                )
                path = condition.get("path")
            else:
                continue
            if (
                schema == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA
                and isinstance(path, str)
                and path.strip()
            ):
                discoveries.extend(
                    _sweep_manifest_candidates(
                        path,
                        repo_root=repo_root,
                        work_id=work_id or None,
                        source="materializer_work_queue_sweep_postcondition",
                        work_queue_path=work_queue_path,
                        backlog_key=raw_row.get("backlog_key"),
                        runtime_context=runtime_context,
                    )
                )
                continue
            if schema not in SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS:
                continue
            if not isinstance(path, str) or not path.strip():
                continue
            discoveries.append(
                {
                    "source": "materializer_work_queue_manifest_postcondition",
                    "work_queue_path": _repo_rel(work_queue_path, repo_root),
                    "path": path,
                    "schema": schema,
                    "work_id": work_id or None,
                    "backlog_key": raw_row.get("backlog_key"),
                    "runtime_context": dict(runtime_context),
                }
            )
    return discoveries


def _runtime_context_from_work_queue_row(
    row: Mapping[str, Any],
    *,
    work_id: str | None,
    work_queue_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    closure = _mapping(row.get("materializer_context_closure_plan"))
    proof = _mapping(closure.get("receiver_proof_request"))
    closure_binding = _mapping(closure.get("receiver_runtime_binding_context"))
    top_binding = _mapping(row.get("receiver_runtime_binding_context"))
    renderer_context = _mapping(row.get("renderer_payload_dfl1_parity_context"))

    context: dict[str, Any] = {}
    for key in RUNTIME_CONTEXT_PATH_FIELDS:
        if key.startswith("source_") or key.endswith("_source_runtime_dir"):
            value = _first_text(
                row.get(key),
                closure.get(key),
                proof.get(key),
                renderer_context.get(key),
            )
        else:
            value = _first_text(
                row.get(key),
                closure.get(key),
                proof.get(key),
                closure_binding.get(key),
                top_binding.get(key),
                renderer_context.get(key),
            )
        if value is not None:
            context[key] = value

    identity_sources = list(
        _runtime_identity_source_mappings(
            row,
            closure,
            proof,
            closure_binding,
            top_binding,
            renderer_context,
        )
    )
    for key in RUNTIME_CONTEXT_IDENTITY_FIELDS:
        value = _first_sha256_text(*(source.get(key) for source in identity_sources))
        if value is not None:
            context[key] = value
    for key, nested_key in (
        ("candidate_runtime_tree_sha256", "candidate_runtime"),
        ("runtime_tree_sha256", "runtime_manifest"),
        (
            "packet_member_merge_receiver_runtime_tree_sha256",
            "packet_member_merge_receiver_runtime",
        ),
        (
            "tensor_factorize_receiver_runtime_tree_sha256",
            "tensor_factorize_receiver_runtime",
        ),
        (
            "renderer_payload_dfl1_runtime_tree_sha256",
            "renderer_payload_dfl1_receiver_runtime",
        ),
    ):
        if key in context:
            continue
        value = _first_sha256_text(
            *(
                _mapping(source.get(nested_key)).get("runtime_tree_sha256")
                for source in identity_sources
            )
        )
        if value is not None:
            context[key] = value

    source_runtime = _first_text(
        proof.get("source_runtime_dir"),
        closure.get("source_runtime_dir"),
        row.get("source_runtime_dir"),
        renderer_context.get("source_runtime_dir"),
        renderer_context.get("renderer_payload_dfl1_source_runtime_dir"),
        closure_binding.get("source_runtime_dir"),
        closure_binding.get("candidate_submission_dir"),
        top_binding.get("source_runtime_dir"),
        top_binding.get("candidate_submission_dir"),
        row.get("candidate_submission_dir"),
        row.get("source_submission_dir"),
    )
    if source_runtime is not None:
        context["source_runtime_dir"] = source_runtime
        context["source_submission_dir"] = source_runtime
        context["packet_member_merge_source_runtime_dir"] = source_runtime
        context["renderer_payload_dfl1_source_runtime_dir"] = source_runtime
        context["tensor_factorize_source_runtime_dir"] = source_runtime
    source_inflate = _first_text(
        proof.get("source_inflate_sh_path"),
        closure.get("source_inflate_sh_path"),
        row.get("source_inflate_sh_path"),
        renderer_context.get("source_inflate_sh_path"),
        closure_binding.get("candidate_inflate_sh_path"),
        top_binding.get("candidate_inflate_sh_path"),
        row.get("candidate_inflate_sh_path"),
    )
    if source_inflate is None and source_runtime is not None:
        source_inflate = (Path(source_runtime) / "inflate.sh").as_posix()
    if source_inflate is None:
        source_inflate = _first_text(
            closure_binding.get("source_inflate_sh_path"),
            top_binding.get("source_inflate_sh_path"),
        )
    if source_inflate is not None:
        context["source_inflate_sh_path"] = source_inflate

    candidate_submission = _first_text(
        closure_binding.get("candidate_submission_dir"),
        top_binding.get("candidate_submission_dir"),
        row.get("candidate_submission_dir"),
        closure.get("candidate_submission_dir"),
    )
    if candidate_submission is not None:
        context["candidate_submission_dir"] = candidate_submission
    candidate_inflate = _first_text(
        closure_binding.get("candidate_inflate_sh_path"),
        top_binding.get("candidate_inflate_sh_path"),
        row.get("candidate_inflate_sh_path"),
        closure.get("candidate_inflate_sh_path"),
    )
    if candidate_inflate is not None:
        context["candidate_inflate_sh_path"] = candidate_inflate
    candidate_archive = _first_text(
        closure_binding.get("candidate_archive_path"),
        top_binding.get("candidate_archive_path"),
        row.get("candidate_archive_path"),
        closure.get("candidate_archive_path"),
    )
    if candidate_archive is not None:
        context["candidate_archive_path"] = candidate_archive

    if context:
        context["materializer_harvest_runtime_context_sources"] = [
            {
                "source": "materializer_work_queue_row",
                "work_queue_path": _repo_rel(work_queue_path, repo_root),
                "work_id": work_id,
            }
        ]
    return context


def _runtime_context_from_discovery(discovery: Mapping[str, Any]) -> dict[str, Any]:
    raw = discovery.get("runtime_context")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _runtime_identity_source_mappings(
    *mappings: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        out.append(mapping)
        for key in RUNTIME_CONTEXT_NESTED_IDENTITY_FIELDS:
            nested = mapping.get(key)
            if isinstance(nested, Mapping):
                out.append(nested)
    return out


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, str):
        text = value
    else:
        return None
    text = text.strip()
    return text or None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _nonempty_text(value)
        if text is not None:
            return text
    return None


def _sha256_text(value: Any) -> str | None:
    text = _nonempty_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if len(lowered) == 64 and all(ch in _SHA256_HEX for ch in lowered):
        return lowered
    return None


def _first_sha256_text(*values: Any) -> str | None:
    for value in values:
        text = _sha256_text(value)
        if text is not None:
            return text
    return None


def _parse_sweep_manifest_spec(raw_spec: str | Path) -> tuple[str | None, str | Path]:
    if isinstance(raw_spec, Path):
        return None, raw_spec
    text = str(raw_spec)
    if "=" not in text:
        return None, text
    work_id, path = text.split("=", 1)
    work_id = work_id.strip()
    path = path.strip()
    if not work_id or not path:
        raise ExperimentQueueError(
            "sweep manifest specs must be PATH or work_id=PATH"
        )
    return work_id, path


def _sweep_manifest_candidates(
    sweep_manifest_path: str | Path,
    *,
    repo_root: Path,
    work_id: str | None,
    source: str,
    work_queue_path: Path | None = None,
    backlog_key: Any = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = _resolve_path(sweep_manifest_path, repo_root=repo_root)
    if not path.is_file():
        raise ExperimentQueueError(f"materializer sweep manifest missing: {path}")
    if path.is_symlink():
        raise ExperimentQueueError(f"materializer sweep manifest is symlink: {path}")
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ExperimentQueueError("materializer sweep manifest must be an object")
    if payload.get("schema") != FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA:
        raise ExperimentQueueError(
            "expected "
            f"{FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA}, "
            f"got {payload.get('schema')!r}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="family_agnostic_materializer_sweep",
        )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    discoveries: list[dict[str, Any]] = []
    for index, observation in enumerate(payload.get("observations") or []):
        if not isinstance(observation, Mapping):
            continue
        try:
            require_no_truthy_authority_fields(
                observation,
                context=f"family_agnostic_materializer_sweep.observations.{index}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        manifest_path = observation.get("manifest_path") or observation.get("source_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            continue
        discoveries.append(
            {
                "source": source,
                "work_queue_path": _repo_rel(work_queue_path, repo_root)
                if work_queue_path is not None
                else None,
                "sweep_manifest_path": _repo_rel(path, repo_root),
                "sweep_observation_id": observation.get("observation_id"),
                "sweep_rate_positive": observation.get("rate_positive") is True,
                "sweep_receiver_contract_satisfied": (
                    observation.get("receiver_contract_satisfied") is True
                ),
                "path": manifest_path,
                "schema": None,
                "work_id": work_id,
                "backlog_key": backlog_key,
                "runtime_context": dict(runtime_context or {}),
            }
        )
    return discoveries


def _chain_root_manifest_candidates(
    root: str | Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    root_path = _resolve_path(root, repo_root=repo_root)
    if root_path.is_file():
        return [
            {
                "source": "chain_root_file",
                "path": str(root_path),
                "schema": None,
                "work_id": None,
            }
        ]
    if not root_path.is_dir():
        return [
            {
                "source": "chain_root_missing",
                "path": str(root_path),
                "schema": None,
                "work_id": None,
            }
        ]
    discoveries: list[dict[str, Any]] = []
    for name in sorted(KNOWN_CHAIN_MANIFEST_NAMES):
        for path in sorted(root_path.rglob(name)):
            discoveries.append(
                {
                    "source": "chain_root_scan",
                    "path": str(path),
                    "schema": None,
                    "work_id": None,
                }
            )
    return discoveries


def _load_state_rows(
    state_path: str | Path | None,
    *,
    experiment_queue_id: str | None,
) -> dict[str, list[dict[str, Any]]]:
    if state_path is None:
        return {}
    query = """
        SELECT queue_id, experiment_id, step_id, status, attempts,
               updated_at_utc, last_event_json
        FROM step_state
        WHERE step_id = ?
    """
    params: list[Any] = [MATERIALIZER_EXECUTION_STEP_ID]
    if experiment_queue_id is not None:
        query += " AND queue_id = ?"
        params.append(experiment_queue_id)
    query += " ORDER BY queue_id, experiment_id, step_id"
    with connect_state_readonly(state_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event = _json_or_empty(row["last_event_json"])
        item = {
            "queue_id": row["queue_id"],
            "experiment_id": row["experiment_id"],
            "step_id": row["step_id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "updated_at_utc": row["updated_at_utc"],
            "last_event": event,
        }
        out.setdefault(str(row["experiment_id"]), []).append(item)
    return out


def _state_rows_for_discovery(
    discovery: Mapping[str, Any],
    state_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    work_id = discovery.get("work_id")
    if not isinstance(work_id, str) or not work_id:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ordered_unique([work_id, _materializer_execution_experiment_id(work_id)]):
        for row in state_rows.get(key, []):
            identity = (
                str(row.get("queue_id") or ""),
                str(row.get("experiment_id") or ""),
                str(row.get("step_id") or ""),
            )
            if identity in seen:
                continue
            seen.add(identity)
            out.append(dict(row))
    return out


def _materializer_execution_experiment_id(work_id: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", work_id.lower()).strip("_") or "row"


def _state_blockers(
    row: Mapping[str, Any],
    *,
    require_succeeded_state: bool,
    state_filter_active: bool,
) -> list[str]:
    state_rows = row.get("state_rows")
    work_id = row.get("work_id")
    if not isinstance(work_id, str) or not work_id:
        if state_filter_active:
            return ["experiment_queue_state_work_id_missing_for_manifest"]
        return []
    if not isinstance(state_rows, list) or not state_rows:
        if state_filter_active:
            return [f"experiment_queue_state_missing:{work_id}"]
        return []
    if not require_succeeded_state:
        return []
    statuses = ordered_unique(
        str(item.get("status") or "") for item in state_rows if isinstance(item, Mapping)
    )
    if "succeeded" not in statuses:
        return [f"experiment_queue_state_not_succeeded:{work_id}:{','.join(statuses)}"]
    return []


def _validate_materializer_manifest(path: Path, *, repo_root: Path) -> tuple[list[str], str | None]:
    if not path.is_file():
        return [f"materializer_manifest_missing:{path}"], None
    if path.is_symlink():
        return [f"materializer_manifest_is_symlink:{path}"], None
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"materializer_manifest_json_invalid:{exc}"], None
    if not isinstance(payload, Mapping):
        return ["materializer_manifest_not_object"], None
    schema = str(payload.get("schema") or "")
    if schema not in SUPPORTED_MATERIALIZER_MANIFEST_SCHEMAS:
        return [f"unsupported_materializer_schema:{schema!r}"], schema or None
    if payload.get("status") == "failed":
        return ["materializer_manifest_status_failed"], schema
    try:
        adapt_materializer_manifest_to_candidate(
            payload,
            source_path=path,
            repo_root=repo_root,
        )
    except MaterializerChainHarvestError as exc:
        return [str(exc)], schema
    return [], schema


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.absolute()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _repo_rel_no_resolve(path: Path, repo_root: Path) -> str:
    try:
        return path.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:120] or "candidate"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_or_empty(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise ExperimentQueueError(f"refusing_to_overwrite_json:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)


__all__ = [
    "EXACT_READINESS_BRIDGE_SCHEMA",
    "HARVEST_SCHEMA",
    "harvest_materializer_chain_manifests",
    "run_exact_readiness_bridge_for_harvested_queue",
    "write_json",
]
