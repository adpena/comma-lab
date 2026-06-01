# SPDX-License-Identifier: MIT
"""Campaign helpers for HPRC archive materialization.

This module stays substrate-local: it packages bytes it is handed, emits typed
exact-readiness refusals, and leaves a small durable manifest that the next
planner can consume without treating the scaffold as score authority. Lab
placement policy, SSD waterfalls, and queue orchestration live outside ``tac``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file, write_json_artifact
from tac.substrates.hprc.archive import HprcPacketConfig
from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY,
    HPRC_ARCHIVE_CANDIDATE_FAMILY,
    HPRC_ARCHIVE_TRANSFORM_KIND,
    build_minimal_hprc_v0_packet,
    export_hprc_archive_bytes,
)
from tac.substrates.hprc.lineage import (
    hprc_campaign_manifest,
    primary_rate_collapse_candidates,
    residual_sidecar_candidates,
)
from tac.substrates.hprc.resolution_contract import hprc_resolution_contract

HPRC_CAMPAIGN_MANIFEST_SCHEMA = "hprc_campaign_manifest.v1"
HPRC_EXACT_READINESS_REFUSAL_SCHEMA = "hprc_exact_readiness_refusal.v1"
HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA = "hprc_queue_followup_report.v1"

HPRC_V0_EXACT_READINESS_BLOCKERS: tuple[str, ...] = (
    "hprc_v0_receiver_scaffold_not_trained_renderer",
    "contest_resolution_contract_not_proven_by_full_frame_inflate",
    "trained_receiver_export_missing",
    "z8_scorer_weighted_residual_sidecar_missing",
    "full_video_p18_p19_allocator_not_bound_to_candidate",
    "local_cpu_full_video_replay_not_executed",
    "contest_cpu_cuda_exact_eval_not_executed",
)


@dataclass(frozen=True)
class HprcCampaignRunResult:
    """Durable output pointers for one HPRC materialization campaign run."""

    schema: str
    run_id: str
    output_dir: str
    archive_zip_path: str
    archive_zip_sha256: str
    archive_zip_bytes: int
    campaign_manifest_path: str
    exact_readiness_refusal_path: str
    storage_plan_path: str | None
    archive_bound_package_path: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def materialize_minimal_hprc_campaign(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    decoder_family_id: int = 95,
    retain_receiver_output: bool = False,
    storage_plan_path: str | Path | None = None,
    mlx_triage_argv: Sequence[str] | None = None,
) -> HprcCampaignRunResult:
    """Materialize the runnable HPRC V0 scaffold in an already selected directory."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    selected_output_dir = Path(output_dir).expanduser()
    if not selected_output_dir.is_absolute():
        selected_output_dir = root / selected_output_dir
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    run_label = run_id or selected_output_dir.name

    storage_plan = None if storage_plan_path is None else Path(storage_plan_path)

    packet = build_minimal_hprc_v0_packet(
        config=HprcPacketConfig(decoder_family_id=int(decoder_family_id)),
        decoder_family_id=int(decoder_family_id),
    )
    archive_zip_path, archive_sha256, archive_bytes = export_hprc_archive_bytes(
        packet,
        selected_output_dir,
        repo_root=root,
        retain_receiver_proof_output=retain_receiver_output,
        mlx_triage_argv=mlx_triage_argv,
    )

    archive_bound_package_path = (
        selected_output_dir / "archive_bound_candidate_adapter_package.json"
    )
    exact_refusal_path = selected_output_dir / "hprc_exact_readiness_refusal.json"
    exact_refusal = build_hprc_exact_readiness_refusal(
        run_id=run_label,
        archive_zip_path=archive_zip_path,
        archive_zip_sha256=archive_sha256,
        archive_zip_bytes=archive_bytes,
    )
    _write_json_maybe_overwrite(exact_refusal_path, exact_refusal)

    campaign_manifest_path = selected_output_dir / "hprc_campaign_manifest.json"
    campaign_manifest = build_hprc_campaign_manifest(
        run_id=run_label,
        output_dir=selected_output_dir,
        archive_zip_path=archive_zip_path,
        archive_zip_sha256=archive_sha256,
        archive_zip_bytes=archive_bytes,
        storage_plan_path=storage_plan,
        archive_bound_package_path=archive_bound_package_path,
        exact_readiness_refusal_path=exact_refusal_path,
    )
    _write_json_maybe_overwrite(campaign_manifest_path, campaign_manifest)

    return HprcCampaignRunResult(
        schema="hprc_campaign_run_result.v1",
        run_id=run_label,
        output_dir=selected_output_dir.as_posix(),
        archive_zip_path=archive_zip_path.as_posix(),
        archive_zip_sha256=archive_sha256,
        archive_zip_bytes=int(archive_bytes),
        campaign_manifest_path=campaign_manifest_path.as_posix(),
        exact_readiness_refusal_path=exact_refusal_path.as_posix(),
        storage_plan_path=None if storage_plan is None else storage_plan.as_posix(),
        archive_bound_package_path=archive_bound_package_path.as_posix(),
    )


def build_hprc_exact_readiness_refusal(
    *,
    run_id: str,
    archive_zip_path: Path,
    archive_zip_sha256: str,
    archive_zip_bytes: int,
) -> dict[str, Any]:
    """Return the typed refusal that prevents V0 scaffold promotion."""

    return {
        "schema": HPRC_EXACT_READINESS_REFUSAL_SCHEMA,
        "run_id": run_id,
        "archive_zip_path": archive_zip_path.as_posix(),
        "archive_zip_sha256": archive_zip_sha256,
        "archive_zip_bytes": int(archive_zip_bytes),
        "ready": False,
        "blockers": list(HPRC_V0_EXACT_READINESS_BLOCKERS),
        "next_required_proofs": [
            "train_or_import_counted_receiver_weights_and_latents",
            "prove_inflate_outputs_1200_native_1164x874_uint8_rgb_frames",
            "prove_full_frame_inflate_output_change_from_valid_section_mutations",
            "run_full_video_local_cpu_replay_after_mlx_triage",
            "attach_z8_residual_token_sidecar_with_p18_p19_allocator",
            "dispatch_contest_cpu_cuda_only_after_local_winner",
        ],
        **FALSE_AUTHORITY,
    }


def build_hprc_campaign_manifest(
    *,
    run_id: str,
    output_dir: Path,
    archive_zip_path: Path,
    archive_zip_sha256: str,
    archive_zip_bytes: int,
    storage_plan_path: Path | None,
    archive_bound_package_path: Path,
    exact_readiness_refusal_path: Path,
) -> dict[str, Any]:
    """Return the planner-facing HPRC campaign manifest."""

    return {
        "schema": HPRC_CAMPAIGN_MANIFEST_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "run_id": run_id,
        "lane_id": "hprc_hierarchical_predictive_receiver_codec",
        "candidate_family": HPRC_ARCHIVE_CANDIDATE_FAMILY,
        "transform_kind": HPRC_ARCHIVE_TRANSFORM_KIND,
        "output_dir": output_dir.as_posix(),
        "archive_zip_path": archive_zip_path.as_posix(),
        "archive_zip_sha256": archive_zip_sha256,
        "archive_zip_bytes": int(archive_zip_bytes),
        "storage_plan_path": None
        if storage_plan_path is None
        else storage_plan_path.as_posix(),
        "archive_bound_package_path": archive_bound_package_path.as_posix(),
        "exact_readiness_refusal_path": exact_readiness_refusal_path.as_posix(),
        "phase_status": {
            "archive_spine": "landed",
            "storage_waterfall": "selected_or_explicit",
            "resolution_contract": "declared_not_proven",
            "receiver_scaffold": "runnable_non_promotable",
            "trained_receiver": "missing",
            "z8_residual_sidecar": "missing",
            "full_video_p18_p19_allocator": "missing",
            "local_cpu_replay": "missing",
            "contest_exact_auth": "refused",
        },
        "resolution_contract": hprc_resolution_contract(),
        "primary_rate_collapse_candidates": list(primary_rate_collapse_candidates()),
        "residual_sidecar_candidates": list(residual_sidecar_candidates()),
        "campaign_taxonomy": hprc_campaign_manifest(),
        "queue_next_actions": [
            {
                "id": "hprc_v1_train_export_archive",
                "resource_kind": "local_mlx",
                "status": "ready_via_hprc_compact_receiver_long_training_adapter",
                "adapter": "tac.substrates.hprc.training_adapter.HprcCompactReceiverLongTrainingAdapter",
            },
            {
                "id": "pr95_control_full_frame_inflate_parity",
                "resource_kind": "local_cpu",
                "status": "ready_to_wire",
            },
            {
                "id": "z8_residual_token_sidecar_pack",
                "resource_kind": "local_mlx",
                "status": "blocked_until_base_receiver_output_exists",
            },
            {
                "id": "full_video_p18_p19_allocator_relinearized",
                "resource_kind": "local_mlx",
                "status": "blocked_until_candidate_frames_exist",
            },
        ],
        **FALSE_AUTHORITY,
    }


def build_hprc_queue_followup_report(
    *,
    training_result_path: str | Path,
    decode_pairs: int,
    full_replay_min_pairs: int = 600,
    local_replay_summary_path: str | Path | None = None,
    exact_auth_gate_path: str | Path | None = None,
    z8_archive_bin_path: str | Path | None = None,
    z8_surface_path: str | Path | None = None,
    z8_reference_pairs_npy_path: str | Path | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Build the queue-owned follow-up contract for one HPRC campaign output.

    This report is intentionally not a score authority. It turns the training
    queue result into planner-readable next work: full local replay when the
    candidate covers all 600 pairs, exact-auth gating after local replay, and
    the Z8 residual/P18-P19 allocator requirements that must be satisfied before
    HPRC can become a score-moving archive lane.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    result_path = _resolve(training_result_path, base=root)
    result = _load_json_object(result_path)
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    archive_zip_path = artifact.get("archive_path")
    archive_zip_sha256 = artifact.get("archive_sha256")
    archive_zip_bytes = artifact.get("archive_bytes")
    export_dir = result_path.parent / "hprc_compact_receiver_archive_export"
    export_manifest_path = result_path.parent / "hprc_compact_receiver_training_export.json"
    archive_bound_package_path = export_dir / "archive_bound_candidate_adapter_package.json"
    receiver_proof_path = export_dir / "receiver_proof" / "hprc_receiver_proof.json"
    replay_summary = _optional_json(local_replay_summary_path, base=root)
    exact_gate = _optional_json(exact_auth_gate_path, base=root)
    local_replay_required = int(decode_pairs) >= int(full_replay_min_pairs)
    local_replay_passed = (
        isinstance(replay_summary, dict)
        and replay_summary.get("evaluation_passed") is True
    )
    exact_gate_recommended = (
        isinstance(exact_gate, dict)
        and exact_gate.get("exact_auth_dispatch_recommended") is True
    )
    z8_archive_path = None if z8_archive_bin_path is None else _resolve(z8_archive_bin_path, base=root)
    z8_surface = None if z8_surface_path is None else _resolve(z8_surface_path, base=root)
    z8_reference_pairs = (
        None
        if z8_reference_pairs_npy_path is None
        else _resolve(z8_reference_pairs_npy_path, base=root)
    )

    replay_blockers: list[str] = []
    if local_replay_required and not local_replay_passed:
        replay_blockers.append("full_video_local_cpu_replay_not_passed_or_missing")
    if not local_replay_required:
        replay_blockers.append("partial_pair_campaign_not_full_video_replay_candidate")

    z8_sidecar_blockers: list[str] = []
    if not archive_zip_path:
        z8_sidecar_blockers.append("hprc_archive_zip_missing_from_training_result")
    if local_replay_required and not local_replay_passed:
        z8_sidecar_blockers.append("hprc_full_video_local_replay_gate_not_passed")
    if z8_archive_path is None or not z8_archive_path.is_file():
        z8_sidecar_blockers.append("z8_source_archive_bin_not_provided_or_missing")

    allocator_blockers: list[str] = []
    if z8_archive_path is None or not z8_archive_path.is_file():
        allocator_blockers.append("z8_source_archive_bin_not_provided_or_missing")
    if z8_surface is None and z8_reference_pairs is None:
        allocator_blockers.append("z8_full_video_p18_p19_surface_or_reference_pairs_missing")
    if z8_surface is not None and not z8_surface.is_file():
        allocator_blockers.append("z8_full_video_p18_p19_surface_missing")
    if z8_reference_pairs is not None and not z8_reference_pairs.is_file():
        allocator_blockers.append("z8_reference_pairs_npy_missing")

    promotion_blockers = [
        *replay_blockers,
        *z8_sidecar_blockers,
        *allocator_blockers,
        *(
            []
            if exact_gate_recommended
            else ["local_gate_did_not_recommend_exact_cpu_auth_dispatch"]
        ),
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    exact_gate_blockers = (
        ["exact_auth_gate_not_executed_or_missing"]
        if exact_gate is None
        else list(exact_gate.get("blockers") or [])
    )
    return {
        "schema": HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "training_result_path": _repo_relative(result_path, root),
        "decode_pairs": int(decode_pairs),
        "full_replay_min_pairs": int(full_replay_min_pairs),
        "archive": {
            "archive_zip_path": archive_zip_path,
            "archive_zip_sha256": archive_zip_sha256,
            "archive_zip_bytes": archive_zip_bytes,
            "export_manifest_path": _repo_relative(export_manifest_path, root),
            "archive_bound_package_path": _repo_relative(archive_bound_package_path, root),
            "receiver_proof_path": _repo_relative(receiver_proof_path, root),
            "receiver_proof_present": receiver_proof_path.is_file(),
        },
        "local_replay_gate": {
            "required": local_replay_required,
            "summary_path": (
                None
                if local_replay_summary_path is None
                else _repo_relative(_resolve(local_replay_summary_path, base=root), root)
            ),
            "evaluation_passed": local_replay_passed,
            "axis_tag": None if replay_summary is None else replay_summary.get("axis_tag"),
            "local_score_estimate": None
            if replay_summary is None
            else replay_summary.get("local_score_estimate"),
            "blockers": replay_blockers,
        },
        "exact_auth_gate": {
            "gate_path": (
                None
                if exact_auth_gate_path is None
                else _repo_relative(_resolve(exact_auth_gate_path, base=root), root)
            ),
            "exact_auth_dispatch_recommended": exact_gate_recommended,
            "next_required_action": None
            if exact_gate is None
            else exact_gate.get("next_required_action"),
            "blockers": exact_gate_blockers,
        },
        "z8_residual_sidecar_followup": {
            "status": "ready_for_queue_planning" if not z8_sidecar_blockers else "blocked",
            "z8_archive_bin_path": None
            if z8_archive_path is None
            else _repo_relative(z8_archive_path, root),
            "required_role": "attach_z8_scorer_weighted_residual_sidecar_to_hprc_receiver",
            "blockers": z8_sidecar_blockers,
        },
        "full_video_p18_p19_allocator_followup": {
            "status": "ready_for_queue_execution" if not allocator_blockers else "blocked",
            "z8_surface_path": None if z8_surface is None else _repo_relative(z8_surface, root),
            "z8_reference_pairs_npy_path": None
            if z8_reference_pairs is None
            else _repo_relative(z8_reference_pairs, root),
            "allocator_contract": (
                "full_video_exact_chunked_p18_p19_surface_then_hard_archive_projection"
            ),
            "blockers": allocator_blockers,
        },
        "promotion_gate": {
            "ready_for_exact_eval_dispatch": False,
            "cpu_then_cuda_order": ["[contest-CPU]", "[contest-CUDA]"],
            "blockers": sorted(dict.fromkeys(promotion_blockers)),
        },
        **FALSE_AUTHORITY,
    }


def write_hprc_queue_followup_report(
    *,
    output_path: str | Path,
    report: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        report,
        allow_overwrite=allow_overwrite or expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )
    return path


def _write_json_maybe_overwrite(path: Path, payload: Any) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _optional_json(path: str | Path | None, *, base: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _resolve(path, base=base)
    if not resolved.is_file():
        return None
    return _load_json_object(resolved)


def _resolve(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve(strict=False)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_CAMPAIGN_MANIFEST_SCHEMA",
    "HPRC_EXACT_READINESS_REFUSAL_SCHEMA",
    "HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA",
    "HPRC_V0_EXACT_READINESS_BLOCKERS",
    "HprcCampaignRunResult",
    "build_hprc_campaign_manifest",
    "build_hprc_exact_readiness_refusal",
    "build_hprc_queue_followup_report",
    "hprc_resolution_contract",
    "materialize_minimal_hprc_campaign",
    "write_hprc_queue_followup_report",
]
