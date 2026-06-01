#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run MLX-first compact renderer spine acquisition.

The PR95/HNeRV PyTorch continuation lane is useful as a control, but the
production path for new compact bases is MLX/Metal first with portable NumPy
artifacts. This tool turns MLX long-training reports into the shared HPRC
representation spine, acquisition report, and bounded-runner plan. It never
grants score authority; exact CPU/CUDA still owns promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.hprc.spine_acquisition import (  # noqa: E402
    DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    build_spine_acquisition_report,
)
from tac.substrates.hprc.spine_bounded_runner import (  # noqa: E402
    build_spine_bounded_runner_plan,
    write_spine_bounded_runner_plan,
)
from tools.emit_compact_renderer_spine_adapter import (  # noqa: E402
    emit_compact_renderer_spine_adapter,
)

COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA = "compact_renderer_mlx_spine_runner.v1"
DEFAULT_SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
TARGET_FAMILIES = (
    "pr95_hnerv",
    "rnerv",
    "sr_nerv",
    "boostnerv",
    "pvq_nerv",
    "rt_vq_nerv",
    "pact_nerv_selector_v4",
    "pact_nerv_vq",
)
EXECUTABLE_FAMILIES = ("pact_nerv_vq",)
COMPACT_FAMILY_BACKENDS: dict[str, dict[str, Any]] = {
    "pr95_hnerv": {
        "canonical_family": "pr95_hnerv",
        "backend_status": "checkpoint_adapter_available",
        "trainer_kind": "pr95_mlx_8stage_continuation_control",
        "trainer_entrypoint": "tools/run_pr95_mlx_long_training.py",
        "archive_exporter": None,
        "receiver_proof": "blocked_until_pr95_archive_runtime_adapter_emits_byte_closed_packet",
        "next_action": "continue_or_import_pr95_mlx_checkpoint_then_emit_spine_projection",
    },
    "pact_nerv_vq": {
        "canonical_family": "pact_nerv_vq",
        "backend_status": "executable_mlx_backend_available",
        "trainer_kind": "canonical_mlx_score_aware_harness",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_vq",
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": "train_mlx_export_archive_spine_receiver_proof_then_full_video_replay",
    },
    "pvq_nerv": {
        "canonical_family": "pact_nerv_vq",
        "backend_status": "executable_via_pact_nerv_vq_adapter",
        "trainer_kind": "canonical_mlx_score_aware_harness",
        "trainer_entrypoint": "tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_vq",
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": "route_to_pact_nerv_vq_until_pvq_specific_adapter_diverges",
    },
    "pact_nerv_selector_v4": {
        "canonical_family": "pact_nerv",
        "backend_status": "archive_exporter_available_trainer_actuator_pending",
        "trainer_kind": "selector_v4_mlx_renderer_available",
        "trainer_entrypoint": "pending_runner_execute_family_pact_nerv_selector_v4",
        "archive_exporter": (
            "tac.substrates.pact_nerv_selector_v4.archive_candidate."
            "export_pact_nerv_selector_v4_mlx_archive"
        ),
        "receiver_proof": "generated_inflate_sh_receiver_proof_from_archive_exporter",
        "next_action": "wire_selector_v4_bundle_into_this_runner_or_import_existing_training_artifact",
    },
    "rnerv": {
        "canonical_family": "rnerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_mlx_compact_base_trainer",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": "implement_rnerv_mlx_renderer_exporter_under_spine_contract",
    },
    "sr_nerv": {
        "canonical_family": "sr_nerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_mlx_compact_base_trainer",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": "implement_lowres_base_plus_charged_upsampler_mlx_adapter",
    },
    "boostnerv": {
        "canonical_family": "boostnerv",
        "backend_status": "migration_required",
        "trainer_kind": "pytorch_or_l0_scaffold_not_mlx_first_runner_ready",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_mlx_or_portable_runtime_adapter_implemented",
        "next_action": "migrate_boost_residual_to_mlx_or_mark_as_non_primary_sidecar",
    },
    "rt_vq_nerv": {
        "canonical_family": "rt_vq_nerv",
        "backend_status": "migration_required",
        "trainer_kind": "missing_residual_token_vq_mlx_adapter",
        "trainer_entrypoint": None,
        "archive_exporter": None,
        "receiver_proof": "missing_until_adapter_implemented",
        "next_action": "implement_residual_token_vq_as_charged_section_not_hidden_sidecar",
    },
}


class CompactRendererMlxSpineRunnerError(ValueError):
    """Raised when an MLX compact renderer row cannot enter the spine."""


def adapt_pr95_mlx_report_to_spine(
    *,
    pr95_mlx_report_path: str | Path,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Adapt the latest exported PR95 MLX checkpoint into the shared spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    report_path = _resolve(pr95_mlx_report_path, base=root)
    pr95_report = _load_json(report_path)
    checkpoint = _select_latest_exported_checkpoint(pr95_report, base=root)
    pt_path = _resolve_existing(checkpoint["pytorch_state_dict_path"], base=root)
    latents_path = _resolve_existing(checkpoint["latents_path"], base=root)
    export_manifest_path = _optional_existing(
        checkpoint.get("pytorch_export_manifest_path"),
        base=root,
    )
    projection_dir = out / "pr95_hnerv_spine"
    adapter_report = emit_compact_renderer_spine_adapter(
        family="pr95_hnerv",
        output_dir=projection_dir,
        decoder_blob=pt_path,
        latents_blob=latents_path,
        trained_weights_provenance=_trained_provenance(
            report_path=report_path,
            checkpoint=checkpoint,
            role="weights",
        ),
        trained_latents_provenance=_trained_provenance(
            report_path=report_path,
            checkpoint=checkpoint,
            role="latents",
        ),
        manifest_extra=_coverage_manifest_extra(pr95_report),
    )
    projection_manifest = Path(adapter_report["projection"]["manifest_path"])
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection_manifest],
        hard_byte_ceilings=hard_byte_ceilings,
    )
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    _write_json(acquisition_path, acquisition)
    runner_plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        repo_root=root,
    )
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    write_spine_bounded_runner_plan(
        output_path=runner_plan_path,
        plan=runner_plan,
        allow_overwrite=True,
    )
    final_report = _base_report(
        output_dir=out,
        mode="adapted_pr95_mlx_report",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final_report.update(
        {
            "pr95_mlx_report_path": report_path.as_posix(),
            "pr95_mlx_report_sha256": _sha256_file(report_path),
            "selected_checkpoint": _checkpoint_summary(checkpoint, base=root),
            "pytorch_export_manifest_path": (
                None if export_manifest_path is None else export_manifest_path.as_posix()
            ),
            "spine_adapter_report_path": adapter_report["report_path"],
            "projection_manifest_paths": [projection_manifest.as_posix()],
            "acquisition_report_path": acquisition_path.as_posix(),
            "bounded_runner_plan_path": runner_plan_path.as_posix(),
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "selected_runner_rows": runner_plan["selected_runner_rows"],
            "blockers": _dedupe(
                [
                    *adapter_report["exact_gate"]["blockers"],
                    *runner_plan["blockers"],
                    "mlx_local_report_is_advisory_not_score_authority",
                ]
            ),
        }
    )
    final_report_path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(final_report_path, final_report)
    return {**final_report, "report_path": final_report_path.as_posix()}


def build_plan_only_report(
    *,
    output_dir: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    repo_root: str | Path = REPO_ROOT,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write the executable backlog when trained MLX artifacts are absent."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    report = _base_report(
        output_dir=out,
        mode="plan_only",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    report["blockers"] = [
        "trained_mlx_checkpoint_report_or_execute_pr95_mlx_smoke_required",
        "receiver_proof_not_yet_emitted",
        "contest_cpu_cuda_exact_eval_missing",
    ]
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, report)
    return {**report, "report_path": path.as_posix()}


def execute_pr95_mlx_smoke_and_adapt(
    *,
    output_dir: str | Path,
    max_frames: int,
    smoke_epochs_per_stage: int,
    training_loss_surface: str,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int | None = None,
    base_channels: int | None = None,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the existing MLX smoke path, then adapt its checkpoint output."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    pr95_report_path = out / "pr95_mlx_long_training_smoke_report.json"
    checkpoint_root = out / "pr95_mlx_checkpoints"
    telemetry_path = out / "pr95_mlx_telemetry.jsonl"
    command = [
        sys.executable,
        str(root / "tools/run_pr95_mlx_long_training.py"),
        "--output-report",
        pr95_report_path.as_posix(),
        "--checkpoint-root",
        checkpoint_root.as_posix(),
        "--telemetry-path",
        telemetry_path.as_posix(),
        "--source-video-path",
        Path(source_video_path).as_posix(),
        "--max-frames",
        str(int(max_frames)),
        "--smoke-mode",
        "--execute-smoke",
        "--smoke-epochs-per-stage",
        str(int(smoke_epochs_per_stage)),
        "--checkpoint-every-epochs",
        str(int(smoke_epochs_per_stage)),
        "--training-loss-surface",
        training_loss_surface,
        "--random-seed",
        str(int(random_seed)),
        "--operator-run-label",
        "compact_renderer_mlx_spine_runner",
    ]
    if latent_dim is not None:
        command.extend(["--latent-dim", str(int(latent_dim))])
    if base_channels is not None:
        command.extend(["--base-channels", str(int(base_channels))])
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        blocker_report = _base_report(
            output_dir=out,
            mode="pr95_mlx_smoke_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "pr95_mlx_smoke_command": command,
                "pr95_mlx_smoke_returncode": completed.returncode,
                "blockers": ["pr95_mlx_smoke_command_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}
    adapted = adapt_pr95_mlx_report_to_spine(
        pr95_mlx_report_path=pr95_report_path,
        output_dir=out / "spine_from_pr95_mlx_smoke",
        hard_byte_ceilings=hard_byte_ceilings,
        hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
        allow_overwrite=allow_overwrite,
        repo_root=root,
    )
    final = _base_report(
        output_dir=out,
        mode="executed_pr95_mlx_smoke_and_adapted",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "pr95_mlx_smoke_command": command,
            "pr95_mlx_smoke_report_path": pr95_report_path.as_posix(),
            "pr95_mlx_smoke_report_sha256": _sha256_file(pr95_report_path),
            "adapted_report_path": adapted["report_path"],
            "adapted_blockers": adapted["blockers"],
            "blockers": adapted["blockers"],
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def execute_pact_nerv_vq_mlx_smoke_and_adapt(
    *,
    output_dir: str | Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    hard_byte_ceilings: tuple[int, ...] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    hprc_queue_followup_report_paths: tuple[str | Path, ...] = (),
    latent_dim: int = 8,
    embed_dim: int = 8,
    codebook_size: int = 16,
    decoder_channel: int = 8,
    ema_decay: float = 0.9,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    random_seed: int = 0,
    allow_overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train/export a tiny real-video PACT-NeRV-VQ candidate through the spine."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactRendererMlxSpineRunnerError(
            f"output dir is non-empty; pass --overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _run_pact_nerv_vq_mlx_smoke(
            output_dir=out / "pact_nerv_vq_mlx_training",
            num_pairs=num_pairs,
            epochs=epochs,
            batch_pair_indices_per_step=batch_pair_indices_per_step,
            learning_rate=learning_rate,
            source_video_path=source_video_path,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            codebook_size=codebook_size,
            decoder_channel=decoder_channel,
            ema_decay=ema_decay,
            segnet_distillation_weight=segnet_distillation_weight,
            pose_distillation_weight=pose_distillation_weight,
            segnet_distillation_objective=segnet_distillation_objective,
            distillation_temperature=distillation_temperature,
            segnet_tau_boundary=segnet_tau_boundary,
            segnet_hinge_margin=segnet_hinge_margin,
            distillation_device=distillation_device,
            allow_segnet_only_research=allow_segnet_only_research,
            random_seed=random_seed,
            repo_root=root,
        )
    except Exception as exc:
        blocker_report = _base_report(
            output_dir=out,
            mode="pact_nerv_vq_mlx_smoke_failed",
            hard_byte_ceilings=hard_byte_ceilings,
            repo_root=root,
        )
        blocker_report.update(
            {
                "execute_family": "pact_nerv_vq",
                "failure": repr(exc),
                "blockers": ["pact_nerv_vq_mlx_smoke_or_export_failed"],
            }
        )
        path = out / "compact_renderer_mlx_spine_runner_report.json"
        _write_json(path, blocker_report)
        return {**blocker_report, "report_path": path.as_posix()}

    artifact_dict = artifact.as_dict() if hasattr(artifact, "as_dict") else dict(artifact)
    archive_path = artifact_dict.get("archive_path")
    training_dir = out / "pact_nerv_vq_mlx_training"
    spine_manifest = training_dir / "hprc_representation_spine_pact_nerv_vq_manifest.json"
    receiver_proof_path = (
        training_dir / "receiver_proof" / "pact_nerv_vq_mlx_receiver_proof.json"
    )
    projection_paths = [spine_manifest] if spine_manifest.is_file() else []
    receiver_proof_paths = [receiver_proof_path] if receiver_proof_path.is_file() else []
    acquisition_path = out / "hprc_spine_acquisition_report.json"
    runner_plan_path = out / "hprc_spine_bounded_runner_plan.json"
    selected_runner_rows: list[dict[str, Any]] = []
    blockers: list[Any] = [
        "full_video_mlx_scorer_replay_not_attached",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if projection_paths:
        acquisition = build_spine_acquisition_report(
            projection_manifest_paths=projection_paths,
            hard_byte_ceilings=hard_byte_ceilings,
        )
        _write_json(acquisition_path, acquisition)
        runner_plan = build_spine_bounded_runner_plan(
            acquisition_report_path=acquisition_path,
            receiver_proof_report_paths=receiver_proof_paths,
            hprc_queue_followup_report_paths=hprc_queue_followup_report_paths,
            repo_root=root,
        )
        write_spine_bounded_runner_plan(
            output_path=runner_plan_path,
            plan=runner_plan,
            allow_overwrite=True,
        )
        selected_runner_rows = list(runner_plan.get("selected_runner_rows") or [])
        blockers.extend(runner_plan.get("blockers") or [])
    else:
        blockers.append("pact_nerv_vq_spine_projection_manifest_missing")
    if archive_path is None:
        blockers.append("pact_nerv_vq_archive_export_missing")

    final = _base_report(
        output_dir=out,
        mode="executed_pact_nerv_vq_mlx_smoke_and_exported",
        hard_byte_ceilings=hard_byte_ceilings,
        repo_root=root,
    )
    final.update(
        {
            "execute_family": "pact_nerv_vq",
            "num_pairs": int(num_pairs),
            "coverage_valid_for_base_comparison": int(num_pairs) >= 600,
            "training_artifact": artifact_dict,
            "archive_path": archive_path,
            "archive_bytes": artifact_dict.get("archive_bytes"),
            "archive_sha256": artifact_dict.get("archive_sha256"),
            "ema_decay": float(ema_decay),
            "score_aware_training": {
                "schema": "compact_pact_nerv_vq_score_aware_training.v1",
                "segnet_distillation_weight": float(segnet_distillation_weight),
                "pose_distillation_weight": float(pose_distillation_weight),
                "segnet_distillation_objective": segnet_distillation_objective,
                "distillation_temperature": float(distillation_temperature),
                "segnet_tau_boundary": float(segnet_tau_boundary),
                "segnet_hinge_margin": float(segnet_hinge_margin),
                "distillation_device": distillation_device,
                "allow_segnet_only_research": bool(allow_segnet_only_research),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "projection_manifest_paths": [path.as_posix() for path in projection_paths],
            "receiver_proof_report_paths": [
                path.as_posix() for path in receiver_proof_paths
            ],
            "hprc_queue_followup_report_paths": [
                _resolve(path, base=root).as_posix()
                for path in hprc_queue_followup_report_paths
            ],
            "acquisition_report_path": (
                acquisition_path.as_posix() if acquisition_path.is_file() else None
            ),
            "bounded_runner_plan_path": (
                runner_plan_path.as_posix() if runner_plan_path.is_file() else None
            ),
            "selected_runner_rows": selected_runner_rows,
            "blockers": _dedupe(blockers),
        }
    )
    path = out / "compact_renderer_mlx_spine_runner_report.json"
    _write_json(path, final)
    return {**final, "report_path": path.as_posix()}


def _base_report(
    *,
    output_dir: Path,
    mode: str,
    hard_byte_ceilings: tuple[int, ...],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "repo_root": repo_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "hard_byte_ceilings": [int(value) for value in hard_byte_ceilings],
        "execution_contract": {
            "schema": "compact_renderer_mlx_first_execution_contract.v1",
            "primary_accelerator": "MLX/Metal",
            "portable_reference": "NumPy-compatible serialized weights/latents/tokens",
            "pytorch_role": "control, calibration, and optional export bridge only",
            "score_authority": "contest CPU/CUDA exact eval only",
            "promotion_surface": "archive.zip bytes plus receiver proof plus exact gate",
        },
        "target_family_rows": _target_family_rows(),
        "compact_base_campaign_rows": _compact_base_campaign_rows(
            hard_byte_ceilings=hard_byte_ceilings
        ),
        "cleanup_contract": {
            "schema": "ssd_first_cleanup_contract.v1",
            "artifact_tier": "ssd_preferred",
            "large_artifacts_under_output_dir": True,
            "delete_policy": "success_scratch_only; evidence manifests retained",
        },
        **FALSE_AUTHORITY,
    }


def _target_family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        backend = COMPACT_FAMILY_BACKENDS[family]
        rows.append(
            {
                "schema": "compact_renderer_mlx_target_family_row.v1",
                "family": family,
                "canonical_family": backend["canonical_family"],
                "status": backend["backend_status"],
                "trainer_kind": backend["trainer_kind"],
                "trainer_entrypoint": backend["trainer_entrypoint"],
                "archive_exporter": backend["archive_exporter"],
                "receiver_proof": backend["receiver_proof"],
                "next_action": backend["next_action"],
                "required_inputs": [
                    "trained_decoder_weights_or_program",
                    "trained_latents_or_tokens",
                    "archive_charged_runtime_config",
                ],
                "automatic_outputs": [
                    "hprc_representation_spine_projection",
                    "spine_acquisition_row",
                    "bounded_runner_row",
                    "receiver_proof_gate",
                    "exact_axis_blocker_or_dispatch_packet",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _compact_base_campaign_rows(
    *,
    hard_byte_ceilings: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        backend = COMPACT_FAMILY_BACKENDS[family]
        for ceiling in hard_byte_ceilings:
            status = str(backend["backend_status"])
            if status in {
                "executable_mlx_backend_available",
                "executable_via_pact_nerv_vq_adapter",
            }:
                route_status = "queued_for_mlx_training_archive_export_receiver_proof"
            elif status == "checkpoint_adapter_available":
                route_status = "queued_for_checkpoint_import_or_long_continuation"
            elif status == "archive_exporter_available_trainer_actuator_pending":
                route_status = "trainer_actuator_migration_required"
            else:
                route_status = "migration_required_before_runner_execution"
            rows.append(
                {
                    "schema": "compact_renderer_mlx_campaign_row.v1",
                    "family": family,
                    "canonical_family": backend["canonical_family"],
                    "hard_byte_ceiling": int(ceiling),
                    "backend_status": backend["backend_status"],
                    "route_status": route_status,
                    "trainer_kind": backend["trainer_kind"],
                    "trainer_entrypoint": backend["trainer_entrypoint"],
                    "archive_exporter": backend["archive_exporter"],
                    "receiver_proof": backend["receiver_proof"],
                    "byte_policy": (
                        "train/export only charged weights, latents, selectors, "
                        "codebooks, and residual tokens; no hidden sidecars"
                    ),
                    "value_policy": (
                        "full-video MLX scorer replay prices section value; "
                        "residuals admitted only when delta_nonrate + rate_cost < 0"
                    ),
                    "exact_gate_policy": (
                        "receiver-proven byte-closed archive must be plausible "
                        "under the hard ceiling before contest CPU/CUDA spend"
                    ),
                    "next_action": backend["next_action"],
                    **FALSE_AUTHORITY,
                }
            )
    return rows


def _run_pact_nerv_vq_mlx_smoke(
    *,
    output_dir: Path,
    num_pairs: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    source_video_path: str | Path,
    latent_dim: int,
    embed_dim: int,
    codebook_size: int,
    decoder_channel: int,
    ema_decay: float,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    allow_segnet_only_research: bool,
    random_seed: int,
    repo_root: Path,
) -> Any:
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )
    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig
    from tac.substrates.pact_nerv_vq.archive_candidate import (
        export_pact_nerv_vq_mlx_archive,
    )
    from tac.substrates.pact_nerv_vq.mlx_renderer import PactNervVqSubstrateMLX

    pairs = int(num_pairs)
    if pairs < 1:
        raise CompactRendererMlxSpineRunnerError("num_pairs must be >= 1")
    if segnet_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "segnet_distillation_weight must be >= 0"
        )
    if pose_distillation_weight < 0.0:
        raise CompactRendererMlxSpineRunnerError(
            "pose_distillation_weight must be >= 0"
        )
    if (
        segnet_distillation_weight > 0.0
        and pose_distillation_weight <= 0.0
        and not allow_segnet_only_research
    ):
        raise CompactRendererMlxSpineRunnerError(
            "SegNet-bound compact training must also bind PoseNet. Pass "
            "--pose-distillation-weight > 0, or explicitly pass "
            "--allow-segnet-only-research for a false-authority SegNet-axis probe."
        )
    cfg = PactNervVqConfig(
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([int(decoder_channel)] * 7),
        num_upsample_blocks=7,
        codebook_size=int(codebook_size),
        num_pairs=pairs,
        output_height=384,
        output_width=512,
    )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=pairs,
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )
    model = PactNervVqSubstrateMLX(cfg)

    def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
        return {"vq_commitment": model_obj.last_commitment_loss}

    def _export_archive(model_obj: Any, archive_output_dir: Path) -> tuple[Path, str, int]:
        return export_pact_nerv_vq_mlx_archive(
            model_obj,
            archive_output_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=True,
            retain_receiver_proof_output=False,
            mlx_triage_argv=[
                "tools/run_compact_renderer_mlx_spine_runner.py",
                "--execute-family",
                "pact_nerv_vq",
            ],
        )

    artifact_metadata = {
        "schema": "compact_renderer_pact_nerv_vq_mlx_runner_metadata.v1",
        "family": "pact_nerv_vq",
        "num_pairs": pairs,
        "full_video_pairs_required_for_promotion": 600,
        "archive_exporter": (
            "tac.substrates.pact_nerv_vq.archive_candidate."
            "export_pact_nerv_vq_mlx_archive"
        ),
        "score_aware_training": {
            "schema": "compact_pact_nerv_vq_score_aware_training.v1",
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "segnet_distillation_objective": segnet_distillation_objective,
            "distillation_temperature": float(distillation_temperature),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "distillation_device": distillation_device,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
        },
        "score_authority": "false_macos_mlx_research_signal",
    }
    bundle_kwargs: dict[str, Any] = {
        "model": model,
        "target_rgb_0": target_rgb_0,
        "target_rgb_1": target_rgb_1,
        "num_pairs": pairs,
        "forward_convention": "call_b2chw_255",
        "extra_loss_terms": _extra_loss_terms,
        "extra_loss_weights": {"vq_commitment": float(cfg.commitment_weight)},
        "export_archive_fn": _export_archive,
        "substrate_artifact_metadata": artifact_metadata,
    }
    teacher_probe_bundle = RendererBundle(**bundle_kwargs)
    scorer_teacher = None
    learnable_student_head = None
    pose_scorer_teacher = None
    learnable_pose_student_head = None
    if segnet_distillation_weight > 0.0:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=repo_root / "upstream",
            device=distillation_device,
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=int(scorer_teacher.num_classes),
            seed=int(random_seed),
        )
    if pose_distillation_weight > 0.0:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacher_probe_bundle,
            upstream_dir=repo_root / "upstream",
            device=distillation_device,
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=int(pose_scorer_teacher.pose_dims),
            seed=int(random_seed) + 1,
        )
    bundle = RendererBundle(
        **bundle_kwargs,
        distillation_weight=float(segnet_distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        distillation_temperature=float(distillation_temperature),
        segnet_distillation_objective=segnet_distillation_objective,
        segnet_tau_boundary=float(segnet_tau_boundary),
        segnet_hinge_margin=float(segnet_hinge_margin),
        distillation_num_classes=(
            int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
        ),
        pose_distillation_weight=float(pose_distillation_weight),
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=int(pose_scorer_teacher.pose_dims)
        if pose_scorer_teacher is not None
        else 6,
        allow_segnet_only_research=bool(allow_segnet_only_research),
    )
    return run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="compact_runner_pact_nerv_vq_mlx",
        lane_id="lane_compact_renderer_mlx_spine_runner_pact_nerv_vq_20260601",
        output_dir=output_dir,
        epochs=int(epochs),
        batch_pair_indices_per_step=max(1, int(batch_pair_indices_per_step)),
        learning_rate=float(learning_rate),
        ema_decay=float(ema_decay),
        seed=int(random_seed),
        checkpoint_interval_epochs=max(1, int(epochs)),
        notes=(
            "Compact renderer MLX spine runner PACT-NeRV-VQ smoke using real "
            "contest video targets, byte-closed archive export, receiver proof, "
            "and false-authority MLX evidence only."
        ),
    )


def _coverage_manifest_extra(report: dict[str, Any]) -> dict[str, Any]:
    frames = _positive_int(report.get("source_video_frame_count"))
    if frames is None:
        frames = _positive_int(report.get("max_frames"))
    num_pairs = None if frames is None else frames // 2
    return {
        "num_frames": frames,
        "num_pairs": num_pairs,
        "coverage_source": "pr95_mlx_long_training_report",
        "coverage_note": (
            "max_frames smoke coverage is not full-video comparable; "
            "bounded runner must scale before base-byte promotion"
        ),
    }


def _select_latest_exported_checkpoint(
    report: dict[str, Any],
    *,
    base: Path,
) -> dict[str, Any]:
    rows = report.get("checkpoint_artifacts")
    if not isinstance(rows, list):
        raise CompactRendererMlxSpineRunnerError("checkpoint_artifacts_missing")
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        if raw.get("pytorch_export_succeeded") is not True:
            continue
        if raw.get("trained_latents_exported") is not True:
            continue
        if not raw.get("pytorch_state_dict_path") or not raw.get("latents_path"):
            continue
        _resolve_existing(raw["pytorch_state_dict_path"], base=base)
        _resolve_existing(raw["latents_path"], base=base)
        candidates.append(raw)
    if not candidates:
        raise CompactRendererMlxSpineRunnerError(
            "no_checkpoint_with_exported_weights_and_latents"
        )
    return max(
        candidates,
        key=lambda row: (int(row.get("global_epoch") or 0), int(row.get("stage_index") or 0)),
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _checkpoint_summary(row: dict[str, Any], *, base: Path) -> dict[str, Any]:
    summary = dict(row)
    for key in (
        "mlx_checkpoint_path",
        "pytorch_state_dict_path",
        "latents_path",
        "pytorch_export_manifest_path",
    ):
        value = row.get(key)
        if isinstance(value, str) and value:
            path = _resolve(value, base=base)
            summary[f"{key}_resolved"] = path.as_posix()
            if path.is_file():
                summary[f"{key}_bytes"] = path.stat().st_size
                summary[f"{key}_sha256"] = _sha256_file(path)
    return summary


def _trained_provenance(
    *,
    report_path: Path,
    checkpoint: dict[str, Any],
    role: str,
) -> str:
    return (
        f"role={role}; source=tools/run_pr95_mlx_long_training.py; "
        f"report={report_path.as_posix()}; report_sha256={_sha256_file(report_path)}; "
        f"stage={checkpoint.get('stage_index')}; global_epoch={checkpoint.get('global_epoch')}; "
        f"evidence_grade={checkpoint.get('evidence_grade')}; "
        "authority=macos_mlx_research_signal_false_authority"
    )


def _default_output_dir() -> Path:
    for root in DEFAULT_SSD_ROOTS:
        if root.exists():
            return root / "compact_renderer_mlx_spine_runner" / _stamp()
    raise CompactRendererMlxSpineRunnerError(
        "no SSD artifact root found; pass --output-dir explicitly"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-pr95-mlx-report", type=Path)
    parser.add_argument("--execute-pr95-mlx-smoke", action="store_true")
    parser.add_argument(
        "--execute-family",
        choices=EXECUTABLE_FAMILIES,
        help="Execute a real MLX compact-family training/export row.",
    )
    parser.add_argument("--source-video-path", default=Path("upstream/videos/0.mkv"), type=Path)
    parser.add_argument("--max-frames", default=4, type=int)
    parser.add_argument("--num-pairs", default=2, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-pairs", default=1, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--compact-latent-dim", default=8, type=int)
    parser.add_argument("--compact-embed-dim", default=8, type=int)
    parser.add_argument("--compact-codebook-size", default=16, type=int)
    parser.add_argument("--compact-decoder-channel", default=8, type=int)
    parser.add_argument(
        "--compact-ema-decay",
        default=0.9,
        type=float,
        help=(
            "Weight EMA decay for compact MLX archive export. Short byte-ceiling "
            "sweeps use a faster shadow than the long-run canonical 0.997 so "
            "receiver-proven archives do not export near-initial gray frames."
        ),
    )
    parser.add_argument(
        "--segnet-distillation-weight",
        default=0.0,
        type=float,
        help=(
            "Bind compact training to the real MLX SegNet teacher through the "
            "learnable student-head loss. False-authority MLX research signal only."
        ),
    )
    parser.add_argument(
        "--pose-distillation-weight",
        default=0.0,
        type=float,
        help=(
            "Bind compact training to the real PoseNet teacher through the "
            "learnable pose-head loss. Required for frontier-targeting "
            "score-aware compact runs."
        ),
    )
    parser.add_argument(
        "--segnet-distillation-objective",
        choices=(
            "kl_t2",
            "boundary_tckd",
            "boundary_decision_tckd",
            "boundary_argmax_hinge",
        ),
        default="kl_t2",
    )
    parser.add_argument("--distillation-temperature", default=2.0, type=float)
    parser.add_argument("--segnet-tau-boundary", default=1.0, type=float)
    parser.add_argument("--segnet-hinge-margin", default=1.0, type=float)
    parser.add_argument(
        "--distillation-device",
        default="cpu",
        help="Device used to build real scorer teacher caches; default CPU.",
    )
    parser.add_argument(
        "--allow-segnet-only-research",
        action="store_true",
        help=(
            "Explicitly allow SegNet-only compact training without PoseNet. "
            "This remains false-authority and is never promotion-ready by itself."
        ),
    )
    parser.add_argument("--smoke-epochs-per-stage", default=1, type=int)
    parser.add_argument(
        "--training-loss-surface",
        choices=("rgb_mse", "rgb_yuv6_mse"),
        default="rgb_yuv6_mse",
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--random-seed", default=0, type=int)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int)
    parser.add_argument(
        "--hprc-queue-followup-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "Existing HPRC queue followup report to feed into the spine "
            "bounded-runner posterior. Repeatable."
        ),
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = [
        args.execute_pr95_mlx_smoke,
        args.from_pr95_mlx_report is not None,
        args.execute_family is not None,
    ]
    if sum(1 for item in modes if item) > 1:
        raise SystemExit(
            "pass only one of --execute-pr95-mlx-smoke, --from-pr95-mlx-report, "
            "or --execute-family"
        )
    ceilings = tuple(args.hard_byte_ceiling or DEFAULT_BASE_RENDERER_BYTE_CEILINGS)
    output_dir = args.output_dir or _default_output_dir()
    if args.execute_pr95_mlx_smoke:
        report = execute_pr95_mlx_smoke_and_adapt(
            output_dir=output_dir,
            max_frames=args.max_frames,
            smoke_epochs_per_stage=args.smoke_epochs_per_stage,
            training_loss_surface=args.training_loss_surface,
            source_video_path=args.source_video_path,
            hard_byte_ceilings=ceilings,
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.from_pr95_mlx_report is not None:
        report = adapt_pr95_mlx_report_to_spine(
            pr95_mlx_report_path=args.from_pr95_mlx_report,
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    elif args.execute_family == "pact_nerv_vq":
        report = execute_pact_nerv_vq_mlx_smoke_and_adapt(
            output_dir=output_dir,
            num_pairs=args.num_pairs,
            epochs=args.epochs,
            batch_pair_indices_per_step=args.batch_pairs,
            learning_rate=args.learning_rate,
            source_video_path=args.source_video_path,
            hard_byte_ceilings=ceilings,
            hprc_queue_followup_report_paths=tuple(args.hprc_queue_followup_report),
            latent_dim=args.compact_latent_dim,
            embed_dim=args.compact_embed_dim,
            codebook_size=args.compact_codebook_size,
            decoder_channel=args.compact_decoder_channel,
            ema_decay=args.compact_ema_decay,
            segnet_distillation_weight=args.segnet_distillation_weight,
            pose_distillation_weight=args.pose_distillation_weight,
            segnet_distillation_objective=args.segnet_distillation_objective,
            distillation_temperature=args.distillation_temperature,
            segnet_tau_boundary=args.segnet_tau_boundary,
            segnet_hinge_margin=args.segnet_hinge_margin,
            distillation_device=args.distillation_device,
            allow_segnet_only_research=args.allow_segnet_only_research,
            random_seed=args.random_seed,
            allow_overwrite=args.overwrite,
            repo_root=args.repo_root,
        )
    else:
        report = build_plan_only_report(
            output_dir=output_dir,
            hard_byte_ceilings=ceilings,
            repo_root=args.repo_root,
            allow_overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
                "mode": report["mode"],
                "report_path": report["report_path"],
                "blockers": report.get("blockers", []),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _resolve_existing(path: str | Path, *, base: Path) -> Path:
    resolved = _resolve(path, base=base)
    if not resolved.is_file():
        raise CompactRendererMlxSpineRunnerError(f"required file missing: {resolved}")
    return resolved


def _optional_existing(path: Any, *, base: Path) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    resolved = _resolve(path, base=base)
    return resolved if resolved.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompactRendererMlxSpineRunnerError(f"JSON root must be object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        json.JSONDecodeError,
        CompactRendererMlxSpineRunnerError,
    ) as exc:
        print(f"run_compact_renderer_mlx_spine_runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
