# SPDX-License-Identifier: MIT
"""Queue-owned P18/P19 scorer-region -> P11 selector -> P15 repack chains."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from comma_lab.scheduler.repair_cascade_mlx_probe_queue import (
    REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA,
    repair_cascade_rows_from_payload,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_ZIP_REPACK_SCHEMA,
    ARCHIVE_ZIP_REPACK_TARGET_KIND,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
)
from tac.repo_io import sha256_file

SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA = (
    "scorer_region_selector_chain_context.v1"
)
SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA = "scorer_region_selector_chain_report.v1"
SCORER_REGION_SELECTOR_CHAIN_QUEUE_METADATA_SCHEMA = (
    "scorer_region_selector_chain_queue_metadata.v1"
)


class ScorerRegionSelectorChainQueueError(ValueError):
    """Raised when a scorer-region selector chain cannot be built."""


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


def _file_status(key: str, path: str | Path | None, *, repo_root: str | Path) -> dict[str, Any]:
    text = str(path or "").strip()
    resolved = _resolve(text, repo_root) if text else None
    exists = bool(resolved is not None and resolved.is_file())
    out: dict[str, Any] = {
        "key": key,
        "path": _repo_rel(resolved, repo_root) if resolved is not None else None,
        "exists": exists,
    }
    if exists and resolved is not None:
        out.update({"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size})
    return out


def _read_json_file(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionSelectorChainQueueError(f"JSON artifact missing: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerRegionSelectorChainQueueError(f"JSON artifact must be an object: {path}")
    return payload


def _archive_record(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionSelectorChainQueueError(f"archive missing: {path}")
    return {
        "path": _repo_rel(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _source_archive_from_submission_dir(
    source_submission_dir: str | Path,
    *,
    repo_root: str | Path,
) -> Path:
    source_dir = _resolve(source_submission_dir, repo_root)
    archive = source_dir / "archive.zip"
    if not archive.is_file():
        raise ScorerRegionSelectorChainQueueError(
            f"source submission archive missing: {archive}"
        )
    return archive


def _cascade_row_from_work_order(
    source_waterfill_work_order: str | Path | None,
    *,
    repo_root: str | Path,
) -> dict[str, Any] | None:
    if source_waterfill_work_order is None:
        return None
    payload = _read_json_file(source_waterfill_work_order, repo_root=repo_root)
    require_no_truthy_authority_fields(
        payload,
        context="scorer_region_selector_chain_source_waterfill_work_order",
    )
    rows = repair_cascade_rows_from_payload(payload)
    if not rows:
        return None
    preferred = [
        row
        for row in rows
        if "P19" in json.dumps(row, sort_keys=True)
        and "P18" in json.dumps(row, sort_keys=True)
        and "P11" in json.dumps(row, sort_keys=True)
    ]
    row = dict((preferred or rows)[0])
    row.setdefault("schema", REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA)
    return row


def build_scorer_region_selector_chain_context(
    *,
    repo_root: str | Path,
    source_submission_dir: str | Path,
    source_waterfill_work_order: str | Path | None = None,
    full_frame_inflate_parity_proof: str | Path | None = None,
    posenet_null_pairs: str | Path | None = None,
    segnet_region_masks: str | Path | None = None,
    selector_region_bits: str | Path | None = None,
    chain_label: str = "cascade_c_p19_p18_to_p11_selector_context_then_p15_repack",
) -> dict[str, Any]:
    """Build the explicit upstream custody context for a composed chain."""

    archive = _source_archive_from_submission_dir(source_submission_dir, repo_root=repo_root)
    cascade = _cascade_row_from_work_order(
        source_waterfill_work_order,
        repo_root=repo_root,
    )
    artifact_status = [
        _file_status("full_frame_inflate_parity_proof", full_frame_inflate_parity_proof, repo_root=repo_root),
        _file_status("posenet_null_bottom_decile_pair_ids", posenet_null_pairs, repo_root=repo_root),
        _file_status("segnet_class_region_mask_ids", segnet_region_masks, repo_root=repo_root),
        _file_status("selector_payload_bits_per_region", selector_region_bits, repo_root=repo_root),
    ]
    status_by_key = {str(row["key"]): row for row in artifact_status}
    upstream_ready = (
        status_by_key["posenet_null_bottom_decile_pair_ids"]["exists"] is True
        and status_by_key["segnet_class_region_mask_ids"]["exists"] is True
    )
    blockers = ordered_unique(
        [
            *(
                []
                if source_waterfill_work_order is not None and cascade is not None
                else ["source_repair_waterfill_cascade_row_missing"]
            ),
            *(
                []
                if status_by_key["full_frame_inflate_parity_proof"]["exists"] is True
                else ["full_frame_inflate_parity_proof_missing"]
            ),
            *(
                []
                if upstream_ready
                else [
                    "p19_posenet_null_pairs_missing",
                    "p18_segnet_region_masks_missing",
                ]
            ),
        ]
    )
    context = {
        "schema": SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
        "chain_label": chain_label,
        "chain_position_order": ["P19", "P18", "P11", "P15"],
        "source_submission_dir": _repo_rel(_resolve(source_submission_dir, repo_root), repo_root),
        "source_archive": _archive_record(archive, repo_root=repo_root),
        "source_waterfill_work_order": (
            _file_status("source_waterfill_work_order", source_waterfill_work_order, repo_root=repo_root)
            if source_waterfill_work_order is not None
            else None
        ),
        "cascade_row": cascade,
        "artifact_status": artifact_status,
        "p18_p19_upstream_ready": upstream_ready,
        "p11_rate_anchor_can_run": True,
        "p15_repack_can_run_after_p11": True,
        "chain_execution_policy": (
            "run_p11_p15_rate_anchor_even_when_p18_p19_artifacts_are_missing; "
            "block_distortion_budget_spend_until_upstream_artifacts_exist"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_chain_context_for_local_materialization_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        context,
        context="scorer_region_selector_chain_context",
    )
    return context


def _saved_bytes(manifest: Mapping[str, Any]) -> int:
    for container_key in ("selected_recode", "selected_repack"):
        container = _mapping(manifest.get(container_key))
        if "saved_bytes" in container:
            try:
                return int(container.get("saved_bytes") or 0)
            except (TypeError, ValueError):
                return 0
    source = _mapping(manifest.get("source_archive"))
    candidate = _mapping(manifest.get("candidate_archive"))
    try:
        return int(source.get("bytes") or 0) - int(candidate.get("bytes") or 0)
    except (TypeError, ValueError):
        return 0


def build_scorer_region_selector_chain_report(
    *,
    repo_root: str | Path,
    chain_context: Mapping[str, Any],
    chain_context_path: str | Path,
    selector_manifest: Mapping[str, Any],
    selector_manifest_path: str | Path,
    repack_manifest: Mapping[str, Any],
    repack_manifest_path: str | Path,
) -> dict[str, Any]:
    """Summarize the composed chain and select the next local survivor archive."""

    if chain_context.get("schema") != SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA:
        raise ScorerRegionSelectorChainQueueError("chain context schema mismatch")
    if selector_manifest.get("schema") != FECA_REPARAMETERIZATION_MANIFEST_SCHEMA:
        raise ScorerRegionSelectorChainQueueError("selector manifest schema mismatch")
    if repack_manifest.get("schema") != ARCHIVE_ZIP_REPACK_SCHEMA:
        raise ScorerRegionSelectorChainQueueError("repack manifest schema mismatch")
    require_no_truthy_authority_fields(
        chain_context,
        context="scorer_region_selector_chain_report_context",
    )
    require_no_truthy_authority_fields(
        selector_manifest,
        context="scorer_region_selector_chain_report_selector_manifest",
    )
    require_no_truthy_authority_fields(
        repack_manifest,
        context="scorer_region_selector_chain_report_repack_manifest",
    )

    selector_saved = _saved_bytes(selector_manifest)
    repack_saved = _saved_bytes(repack_manifest)
    selector_archive = _mapping(selector_manifest.get("candidate_archive"))
    repack_archive = _mapping(repack_manifest.get("candidate_archive"))
    repack_positive = (
        repack_saved > 0
        and repack_manifest.get("receiver_contract_satisfied") is True
        and repack_archive.get("path")
    )
    selected_archive = repack_archive if repack_positive else selector_archive
    selected_stage = "P15_archive_zip_repack" if repack_positive else "P11_selector_context_recode"
    readiness_blockers = ordered_unique(
        [
            *[str(item) for item in chain_context.get("blockers") or [] if str(item)],
            *[str(item) for item in selector_manifest.get("readiness_blockers") or [] if str(item)],
            *[str(item) for item in repack_manifest.get("readiness_blockers") or [] if str(item)],
            "contest_auth_eval_required_before_score_or_promotion_claim",
        ]
    )
    report = {
        "schema": SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
        "chain_label": chain_context.get("chain_label"),
        "chain_position_order": ["P19", "P18", "P11", "P15"],
        "chain_context_artifact": _file_status("chain_context", chain_context_path, repo_root=repo_root),
        "selector_manifest_artifact": _file_status("selector_manifest", selector_manifest_path, repo_root=repo_root),
        "repack_manifest_artifact": _file_status("repack_manifest", repack_manifest_path, repo_root=repo_root),
        "p18_p19_upstream_ready": chain_context.get("p18_p19_upstream_ready") is True,
        "selector_saved_bytes": selector_saved,
        "repack_saved_bytes_after_selector": repack_saved,
        "cumulative_rate_saved_bytes_vs_source": selector_saved + max(0, repack_saved),
        "repack_positive": repack_positive,
        "selected_local_survivor_stage": selected_stage,
        "selected_local_survivor_archive": dict(selected_archive),
        "selector_candidate_archive": dict(selector_archive),
        "repack_candidate_archive": dict(repack_archive),
        "selected_selector_codec": (
            selector_manifest.get("selected_codec_family")
            or _mapping(selector_manifest.get("selected_recode")).get("codec_family")
        ),
        "selected_repack_strategy": _mapping(repack_manifest.get("selected_repack")).get("strategy"),
        "selected_repack_plan_key": _mapping(repack_manifest.get("selected_repack")).get("plan_key"),
        "receiver_contracts_satisfied": {
            "selector": selector_manifest.get("receiver_contract_satisfied") is True,
            "repack": repack_manifest.get("receiver_contract_satisfied") is True,
        },
        "blockers": readiness_blockers,
        "readiness_blockers": readiness_blockers,
        "recommended_next_action": (
            "materialize_p18_p19_upstream_repairs_then_rerun_chain"
            if chain_context.get("p18_p19_upstream_ready") is not True
            else "run_local_cpu_or_mlx_component_spot_check_before_exact_anchor"
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_local_chain_report_for_planning_and_harvest",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="scorer_region_selector_chain_report",
    )
    return report


def build_scorer_region_selector_chain_queue(
    *,
    repo_root: str | Path,
    queue_id: str,
    source_submission_dir: str | Path,
    output_root: str | Path,
    source_waterfill_work_order: str | Path | None = None,
    full_frame_inflate_parity_proof: str | Path | None = None,
    posenet_null_pairs: str | Path | None = None,
    segnet_region_masks: str | Path | None = None,
    selector_region_bits: str | Path | None = None,
    chain_label: str = "cascade_c_p19_p18_to_p11_selector_context_then_p15_repack",
    codec_families: Sequence[str] = (
        "fec10_adaptive_blend",
        "fec8_markov_static_order1",
        "fec8_markov_adaptive_order1",
        "fec8_markov_static_order2",
    ),
    scales: Sequence[int] = (32, 64, 128, 256),
    alphas: Sequence[int] = (1, 2, 4),
    max_concurrency_local_cpu: int = 1,
) -> dict[str, Any]:
    """Return an experiment_queue.v1 definition for the composed chain."""

    source_archive = _source_archive_from_submission_dir(
        source_submission_dir,
        repo_root=repo_root,
    )
    root = _resolve(output_root, repo_root)
    context_path = root / "chain_context.json"
    selector_dir = root / "p11_selector_context_recode"
    selector_manifest = selector_dir / "feca_selector_reparameterization_manifest.json"
    selector_archive = selector_dir / "submission_dir" / "archive.zip"
    repack_dir = root / "p15_archive_zip_repack"
    repack_archive = repack_dir / "archive.zip"
    repack_manifest = repack_dir / "archive_zip_repack_manifest.json"
    repack_proof = repack_dir / "archive_zip_repack.runtime_consumption_proof.json"
    chain_report = root / "scorer_region_selector_chain_report.json"

    context_ref = _repo_rel(context_path, repo_root)
    selector_manifest_ref = _repo_rel(selector_manifest, repo_root)
    selector_archive_ref = _repo_rel(selector_archive, repo_root)
    repack_archive_ref = _repo_rel(repack_archive, repo_root)
    repack_manifest_ref = _repo_rel(repack_manifest, repo_root)
    repack_proof_ref = _repo_rel(repack_proof, repo_root)
    chain_report_ref = _repo_rel(chain_report, repo_root)
    source_submission_ref = _repo_rel(_resolve(source_submission_dir, repo_root), repo_root)
    output_root_ref = _repo_rel(root, repo_root)

    context_cmd = [
        ".venv/bin/python",
        "tools/build_scorer_region_selector_chain_context.py",
        "--source-submission-dir",
        source_submission_ref,
        "--output",
        context_ref,
        "--chain-label",
        chain_label,
        "--overwrite",
    ]
    optional_flags = (
        ("--source-waterfill-work-order", source_waterfill_work_order),
        ("--full-frame-inflate-parity-proof", full_frame_inflate_parity_proof),
        ("--posenet-null-pairs", posenet_null_pairs),
        ("--segnet-region-masks", segnet_region_masks),
        ("--selector-region-bits", selector_region_bits),
    )
    for flag, value in optional_flags:
        if value is not None and str(value).strip():
            context_cmd.extend([flag, _repo_rel(_resolve(value, repo_root), repo_root)])

    selector_cmd = [
        ".venv/bin/python",
        "tools/build_feca_selector_reparameterized_candidate.py",
        "--source-submission-dir",
        source_submission_ref,
        "--output-dir",
        _repo_rel(selector_dir, repo_root),
        "--upstream-entropy-position",
        "P19",
        "--upstream-entropy-position",
        "P18",
        "--downstream-materializer-target",
        ARCHIVE_ZIP_REPACK_TARGET_KIND,
        "--chain-parent-artifact",
        context_ref,
        "--chain-label",
        chain_label,
        "--overwrite",
    ]
    for scale in scales:
        selector_cmd.extend(["--scale", str(int(scale))])
    for alpha in alphas:
        selector_cmd.extend(["--alpha", str(int(alpha))])
    for family in codec_families:
        selector_cmd.extend(["--codec-family", str(family)])
    if full_frame_inflate_parity_proof is not None and str(full_frame_inflate_parity_proof).strip():
        selector_cmd.extend(
            [
                "--full-frame-inflate-parity-proof",
                _repo_rel(_resolve(full_frame_inflate_parity_proof, repo_root), repo_root),
            ]
        )

    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {"local_cpu": int(max_concurrency_local_cpu)},
        },
        "metadata": {
            "schema": SCORER_REGION_SELECTOR_CHAIN_QUEUE_METADATA_SCHEMA,
            "queue_id": queue_id,
            "chain_label": chain_label,
            "chain_position_order": ["P19", "P18", "P11", "P15"],
            "source_submission_dir": source_submission_ref,
            "source_archive": _archive_record(source_archive, repo_root=repo_root),
            "output_root": output_root_ref,
            "chain_context_path": context_ref,
            "selector_manifest_path": selector_manifest_ref,
            "repack_manifest_path": repack_manifest_ref,
            "chain_report_path": chain_report_ref,
            "local_mlx_or_cpu_first": True,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "queue_owned_p18_p19_p11_p15_local_chain",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        "experiments": [
            {
                "id": "scorer_region_selector_repack_chain",
                "priority": 1,
                "status": "queued",
                "tags": [
                    "frontier-rate-attack",
                    "cascade-c",
                    "p18-p19-p11-p15",
                    "local-proof-chain",
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": "scorer_region_selector_chain_experiment_metadata.v1",
                    "chain_label": chain_label,
                    "output_root": output_root_ref,
                    "budget_spend_allowed": False,
                    "ready_for_budget_spend": False,
                    "ready_for_exact_eval_dispatch": False,
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": "build_p18_p19_chain_context",
                        "kind": "command",
                        "command": context_cmd,
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": context_ref,
                                "key": "schema",
                                "equals": SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": context_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [context_ref],
                            "input_artifact_paths": [source_submission_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "materialize_p11_selector_context_recode",
                        "kind": "command",
                        "requires": ["build_p18_p19_chain_context"],
                        "command": selector_cmd,
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 300,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": selector_manifest_ref,
                                "key": "schema",
                                "equals": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": selector_manifest_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [_repo_rel(selector_dir, repo_root)],
                            "input_artifact_paths": [source_submission_ref, context_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "materialize_p15_archive_zip_repack",
                        "kind": "command",
                        "requires": ["materialize_p11_selector_context_recode"],
                        "command": [
                            ".venv/bin/python",
                            "tools/run_family_agnostic_materializer.py",
                            "--target-kind",
                            ARCHIVE_ZIP_REPACK_TARGET_KIND,
                            "--archive-path",
                            selector_archive_ref,
                            "--output-archive",
                            repack_archive_ref,
                            "--output-manifest",
                            repack_manifest_ref,
                            "--runtime-consumption-proof-out",
                            repack_proof_ref,
                            "--allow-overwrite",
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": repack_manifest_ref,
                                "key": "schema",
                                "equals": ARCHIVE_ZIP_REPACK_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": repack_manifest_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [repack_archive_ref, repack_manifest_ref, repack_proof_ref],
                            "input_artifact_paths": [selector_archive_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "emit_composed_chain_report",
                        "kind": "command",
                        "requires": ["materialize_p15_archive_zip_repack"],
                        "command": [
                            ".venv/bin/python",
                            "tools/build_scorer_region_selector_chain_report.py",
                            "--chain-context",
                            context_ref,
                            "--selector-manifest",
                            selector_manifest_ref,
                            "--repack-manifest",
                            repack_manifest_ref,
                            "--output",
                            chain_report_ref,
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": chain_report_ref,
                                "key": "schema",
                                "equals": SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
                            },
                            {"type": "json_false_authority", "path": chain_report_ref},
                        ],
                        "telemetry": {
                            "artifact_paths": [chain_report_ref],
                            "input_artifact_paths": [context_ref, selector_manifest_ref, repack_manifest_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                ],
            }
        ],
    }
    return normalize_queue_definition(queue)


__all__ = [
    "SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA",
    "SCORER_REGION_SELECTOR_CHAIN_QUEUE_METADATA_SCHEMA",
    "SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA",
    "ScorerRegionSelectorChainQueueError",
    "build_scorer_region_selector_chain_context",
    "build_scorer_region_selector_chain_queue",
    "build_scorer_region_selector_chain_report",
]
