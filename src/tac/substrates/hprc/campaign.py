# SPDX-License-Identifier: MIT
"""Campaign helpers for HPRC archive materialization.

This module stays substrate-local: it packages bytes it is handed, emits typed
exact-readiness refusals, and leaves a small durable manifest that the next
planner can consume without treating the scaffold as score authority. Lab
placement policy, SSD waterfalls, and queue orchestration live outside ``tac``.
"""

from __future__ import annotations

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


def _write_json_maybe_overwrite(path: Path, payload: Any) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_CAMPAIGN_MANIFEST_SCHEMA",
    "HPRC_EXACT_READINESS_REFUSAL_SCHEMA",
    "HPRC_V0_EXACT_READINESS_BLOCKERS",
    "HprcCampaignRunResult",
    "build_hprc_campaign_manifest",
    "build_hprc_exact_readiness_refusal",
    "hprc_resolution_contract",
    "materialize_minimal_hprc_campaign",
]
