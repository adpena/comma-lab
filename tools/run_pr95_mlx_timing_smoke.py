#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run native MLX timing smokes for the public PR95 HNeRV reproduction lane."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    EXACT_READINESS_REFUSAL_BLOCKERS,
    FALSE_AUTHORITY,
    LANE_ID,
    PR95_STAGE_MODULES,
    SMOKE_MANIFEST_SCHEMA,
    pr95_default_optimizer_descriptor_id,
    pr95_mlx_optimizer_descriptor_row,
    run_pr95_mlx_synthetic_timing_smoke,
    write_pr95_mlx_byte_closed_smoke_archive,
)
from tac.local_acceleration.pr95_hnerv_mlx_training import (  # noqa: E402
    SOURCE_FAITHFUL_PREPROCESS_SCHEMA,
    SOURCE_VIDEO_PREPROCESS_SCHEMA,
    run_pr95_mlx_source_faithful_smoke,
    run_pr95_mlx_source_video_preprocess_smoke,
)
from tac.optimization.local_training_runtime_profile import (  # noqa: E402
    validate_runtime_profile_observation,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    write_representation_training_probe_manifest,
)

SOURCE_PREPROCESS_CONFLATED_BLOCKER = (
    "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx"
)
SOURCE_PREPROCESS_PORTED_LOSS_UNWIRED_BLOCKER = (
    "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx"
)
SOURCE_VIDEO_LOADER_UNWIRED_BLOCKER = (
    "pr95_source_video_loader_ported_but_training_loop_not_source_video_backed"
)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _parse_int_tuple(value: str, *, field: str, expected_len: int | None = None) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be comma-separated integers") from exc
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(f"{field} must have {expected_len} comma-separated integers")
    if not parsed or any(dim < 1 for dim in parsed):
        raise ValueError(f"{field} dimensions must all be positive")
    return parsed


def _source_preprocess_gradient_ready(
    source_faithful_preprocess_smoke: dict[str, Any] | None,
) -> bool:
    if not isinstance(source_faithful_preprocess_smoke, dict):
        return False
    grad_probe = source_faithful_preprocess_smoke.get("gradient_probe")
    return (
        source_faithful_preprocess_smoke.get("source_faithful_preprocess_ready")
        is True
        and isinstance(grad_probe, dict)
        and grad_probe.get("gradient_reachable") is True
    )


def _source_video_preprocess_ready(
    source_video_preprocess_smoke: dict[str, Any] | None,
) -> bool:
    if not isinstance(source_video_preprocess_smoke, dict):
        return False
    grad_probe = source_video_preprocess_smoke.get("gradient_probe")
    return (
        source_video_preprocess_smoke.get("source_video_loader_ready") is True
        and source_video_preprocess_smoke.get("source_video_preprocess_ready") is True
        and isinstance(grad_probe, dict)
        and grad_probe.get("gradient_reachable") is True
    )


def _exact_readiness_blockers(
    *,
    runtime_consumption_proven: bool = False,
    source_faithful_preprocess_smoke: dict[str, Any] | None = None,
    source_video_preprocess_smoke: dict[str, Any] | None = None,
) -> list[str]:
    blockers = [
        blocker
        for blocker in EXACT_READINESS_REFUSAL_BLOCKERS
        if not (
            runtime_consumption_proven
            and blocker
            in {
                "byte_closed_smoke_archive_not_consumed_by_pr95_runtime",
                "runtime_consumption_proof_missing",
            }
        )
    ]
    if _source_preprocess_gradient_ready(source_faithful_preprocess_smoke):
        blockers = [
            blocker
            for blocker in blockers
            if blocker != SOURCE_PREPROCESS_CONFLATED_BLOCKER
        ]
        blockers.append(SOURCE_PREPROCESS_PORTED_LOSS_UNWIRED_BLOCKER)
    if _source_video_preprocess_ready(source_video_preprocess_smoke):
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "pr95_source_video_loader_not_ported_to_mlx"
        ]
        blockers.append(SOURCE_VIDEO_LOADER_UNWIRED_BLOCKER)
        source_video_refusal = source_video_preprocess_smoke.get(
            "exact_readiness_refusal",
            {},
        )
        if isinstance(source_video_refusal, dict):
            blockers.extend(
                str(item) for item in source_video_refusal.get("blockers", [])
            )
    if runtime_consumption_proven:
        blockers.extend(
            [
                "runtime_consumption_smoke_is_not_score_authority",
                "full_frame_inflate_parity_against_source_runtime_not_run",
            ]
        )
    return list(dict.fromkeys(blockers))


def _candidate_id(
    *,
    stage: int,
    seed: int,
    steps: int,
    base_channels: int,
    optimizer_descriptor_id: str,
) -> str:
    descriptor = _slug(optimizer_descriptor_id)
    return (
        f"pr95_hnerv_mlx_stage{stage}_{descriptor}"
        f"_seed{seed}_steps{steps}_c{base_channels}"
    )


def _stage_module(stage: int) -> str:
    try:
        return PR95_STAGE_MODULES[int(stage)]
    except KeyError as exc:
        raise ValueError(f"unsupported PR95 stage {stage!r}") from exc


def _recommended_execution_command(
    *,
    stage: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    seed: int,
    base_channels: int,
    latent_dim: int,
    output_dir: Path,
    write_byte_closed_smoke: bool,
    write_pr95_public_archive_export: bool,
    prove_pr95_runtime_consumption: bool,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
    source_preprocess_shape: str,
    source_preprocess_camera_hw: str,
    source_preprocess_gradient_shape: str,
    source_video_path: Path,
    source_video_upstream_dir: Path,
    source_video_pair_indices: list[int],
    source_video_output_hw: str,
    source_video_gradient_shape: str,
    runtime_proof_timeout_seconds: float,
    runtime_proof_max_output_bytes: int,
    optimizer_descriptor_id: str,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/run_pr95_mlx_timing_smoke.py",
        "--stage",
        str(stage),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--synthetic-pairs",
        str(synthetic_pairs),
        "--seed",
        str(seed),
        "--base-channels",
        str(base_channels),
        "--latent-dim",
        str(latent_dim),
        "--optimizer-descriptor-id",
        optimizer_descriptor_id,
        "--output-dir",
        _rel(output_dir),
        "--allow-existing-output-dir",
    ]
    if write_byte_closed_smoke:
        command.append("--write-byte-closed-smoke")
    if write_pr95_public_archive_export:
        command.append("--write-pr95-public-archive-export")
    if prove_pr95_runtime_consumption:
        command.append("--prove-pr95-runtime-consumption")
        command.extend(
            [
                "--runtime-proof-timeout-seconds",
                str(runtime_proof_timeout_seconds),
                "--runtime-proof-max-output-bytes",
                str(runtime_proof_max_output_bytes),
            ]
        )
    if write_source_faithful_preprocess_smoke:
        command.extend(
            [
                "--write-source-faithful-preprocess-smoke",
                "--source-preprocess-shape",
                source_preprocess_shape,
                "--source-preprocess-camera-hw",
                source_preprocess_camera_hw,
                "--source-preprocess-gradient-shape",
                source_preprocess_gradient_shape,
            ]
        )
    if write_source_video_preprocess_smoke:
        command.extend(
            [
                "--write-source-video-preprocess-smoke",
                "--source-video-path",
                _rel(source_video_path),
                "--source-video-upstream-dir",
                _rel(source_video_upstream_dir),
                "--source-video-output-hw",
                source_video_output_hw,
                "--source-video-gradient-shape",
                source_video_gradient_shape,
            ]
        )
        for pair_index in source_video_pair_indices:
            command.extend(["--source-video-pair-index", str(pair_index)])
    return command


def _json_equals_postcondition(path: str, key: str, value: Any) -> dict[str, Any]:
    return {"type": "json_equals", "path": path, "key": key, "equals": value}


def _json_array_contains_postcondition(path: str, key: str, value: Any) -> dict[str, Any]:
    return {"type": "json_array_contains", "path": path, "key": key, "contains": value}


def _json_false_authority_postcondition(path: str) -> dict[str, Any]:
    return {"type": "json_false_authority", "path": path}


def _extra_artifact_postconditions(
    *,
    output_dir: Path,
    write_pr95_public_archive_export: bool,
    prove_pr95_runtime_consumption: bool,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
) -> list[dict[str, Any]]:
    postconditions: list[dict[str, Any]] = []
    if write_source_faithful_preprocess_smoke:
        smoke_path = _rel(output_dir / "source_faithful_preprocess_smoke.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": smoke_path},
                _json_equals_postcondition(
                    smoke_path,
                    "schema",
                    SOURCE_FAITHFUL_PREPROCESS_SCHEMA,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "source_faithful_preprocess_ready",
                    True,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "gradient_probe.gradient_reachable",
                    True,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "exact_readiness_refusal.ready",
                    False,
                ),
                _json_array_contains_postcondition(
                    smoke_path,
                    "exact_readiness_refusal.blockers",
                    "pr95_training_loop_not_yet_source_faithful",
                ),
                _json_false_authority_postcondition(smoke_path),
            ]
        )
    if write_source_video_preprocess_smoke:
        smoke_path = _rel(output_dir / "source_video_preprocess_smoke.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": smoke_path},
                _json_equals_postcondition(
                    smoke_path,
                    "schema",
                    SOURCE_VIDEO_PREPROCESS_SCHEMA,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "source_video_loader_ready",
                    True,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "source_video_preprocess_ready",
                    True,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "gradient_probe.gradient_reachable",
                    True,
                ),
                _json_equals_postcondition(
                    smoke_path,
                    "exact_readiness_refusal.ready",
                    False,
                ),
                _json_array_contains_postcondition(
                    smoke_path,
                    "exact_readiness_refusal.blockers",
                    "pr95_training_loop_not_yet_source_faithful",
                ),
                _json_false_authority_postcondition(smoke_path),
            ]
        )
    if write_pr95_public_archive_export:
        export_path = _rel(output_dir / "pr95_public_archive_export.json")
        archive_path = _rel(output_dir / "pr95_public_archive.zip")
        postconditions.extend(
            [
                {"type": "path_exists", "path": archive_path},
                {"type": "path_exists", "path": export_path},
                _json_equals_postcondition(
                    export_path,
                    "schema",
                    "pr95_hnerv_archive_export.v1",
                ),
                _json_equals_postcondition(
                    export_path,
                    "runtime_consumption_proof_present",
                    prove_pr95_runtime_consumption,
                ),
                _json_false_authority_postcondition(export_path),
            ]
        )
    if prove_pr95_runtime_consumption:
        proof_path = _rel(output_dir / "runtime_consumption_proof.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": proof_path},
                _json_equals_postcondition(
                    proof_path,
                    "schema",
                    "pr95_hnerv_public_runtime_consumption_proof.v1",
                ),
                _json_equals_postcondition(
                    proof_path,
                    "runtime_consumption_proven",
                    True,
                ),
                _json_false_authority_postcondition(proof_path),
            ]
        )
    return postconditions


def _build_representation_training_plan(
    *,
    stage: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    seed: int,
    base_channels: int,
    latent_dim: int,
    output_dir: Path,
    write_byte_closed_smoke: bool,
    write_pr95_public_archive_export: bool,
    prove_pr95_runtime_consumption: bool,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
    source_preprocess_shape: str,
    source_preprocess_camera_hw: str,
    source_preprocess_gradient_shape: str,
    source_video_path: Path,
    source_video_upstream_dir: Path,
    source_video_pair_indices: list[int],
    source_video_output_hw: str,
    source_video_gradient_shape: str,
    recommended_execution: dict[str, Any],
    optimizer_descriptor: dict[str, Any],
) -> dict[str, Any]:
    stage_module = _stage_module(stage)
    optimizer_descriptor_id = str(optimizer_descriptor["descriptor_id"])
    optimizer_training_config = optimizer_descriptor.get("training_config", {})
    optimizer_backend_status = str(
        optimizer_training_config.get("backend_status")
        if isinstance(optimizer_training_config, dict)
        else ""
    )
    optimizer_blockers = (
        optimizer_training_config.get("dispatch_blockers", [])
        if isinstance(optimizer_training_config, dict)
        else []
    )
    return write_representation_training_probe_manifest(
        output_dir / "representation_training_plan.json",
        schema="representation_training_probe_plan_v1",
        candidate_id=_candidate_id(
            stage=stage,
            seed=seed,
            steps=steps,
            base_channels=base_channels,
            optimizer_descriptor_id=optimizer_descriptor_id,
        ),
        lane_id=LANE_ID,
        lane_class="pr95_hnerv_mlx_reproduction_local_training_proxy",
        candidate_family="pr95_hnerv_mlx_reproduction_timing_smoke",
        representation_family="hnerv",
        substrate_family="nerv_family",
        profile="pr95_hnerv_mlx_stage_timing_smoke",
        param_schema="pr95_hnerv_mlx_timing_smoke_params_v1",
        training_signal_kind="local_mlx_representation_training_runtime_probe",
        seed=seed,
        device_requested="mlx",
        device_selected="mlx",
        output_dir=_rel(output_dir),
        stage_count=1,
        stages=[{"index": stage, "module": stage_module}],
        training_recipe={
            "id": "pr95_hnerv_mlx_synthetic_stage_timing_smoke",
            "source_pr": 95,
            "stage_module": stage_module,
            "steps": steps,
            "batch_size": batch_size,
            "synthetic_pairs": synthetic_pairs,
            "quality_comparable": False,
            "full_scorer_gradient_path": False,
        },
        optimizer_recipe={
            "id": optimizer_descriptor_id,
            "optimizer_descriptor_id": optimizer_descriptor_id,
            "optimizer_config_sha256": optimizer_descriptor["config_sha256"],
            "optimizer_backend_status": optimizer_backend_status,
            "parameter_group_lr_policy_id": optimizer_descriptor[
                "parameter_group_lr_policy_id"
            ],
            "parameter_group_lr_policy_sha256": optimizer_descriptor[
                "parameter_group_lr_policy_sha256"
            ],
            "parameter_group_lr_policy": optimizer_descriptor[
                "parameter_group_lr_policy"
            ],
            "stage_module": stage_module,
            "stage_uses_muon": stage == 8,
            "hidden_2d_plus_weights": "Muon" if stage == 8 else "AdamW",
            "bias_norm_scalar_stem_rgb_head": "AdamW",
        },
        scheduler_recipe={
            "id": "single_process_mlx_local_smoke",
            "resource_kind": "local_mlx",
            "queue_authority": "experiment_queue.v1",
        },
        candidate_params={
            "stage_index": stage,
            "stage_module": stage_module,
            "stage_count": 1,
            "training_backend": "mlx",
            "base_channels": base_channels,
            "latent_dim": latent_dim,
            "optimizer_descriptor_id": optimizer_descriptor_id,
            "optimizer_config_sha256": optimizer_descriptor["config_sha256"],
            "optimizer_backend_status": optimizer_backend_status,
            "parameter_group_lr_policy_id": optimizer_descriptor[
                "parameter_group_lr_policy_id"
            ],
            "parameter_group_lr_policy_sha256": optimizer_descriptor[
                "parameter_group_lr_policy_sha256"
            ],
            "byte_closed_smoke_archive_requested": write_byte_closed_smoke,
            "pr95_public_archive_export_requested": write_pr95_public_archive_export,
            "pr95_runtime_consumption_proof_requested": prove_pr95_runtime_consumption,
            "source_faithful_preprocess_smoke_requested": (
                write_source_faithful_preprocess_smoke
            ),
            "source_preprocess_shape": source_preprocess_shape,
            "source_preprocess_camera_hw": source_preprocess_camera_hw,
            "source_preprocess_gradient_shape": source_preprocess_gradient_shape,
            "source_video_preprocess_smoke_requested": (
                write_source_video_preprocess_smoke
            ),
            "source_video_path": _rel(source_video_path),
            "source_video_upstream_dir": _rel(source_video_upstream_dir),
            "source_video_pair_indices": source_video_pair_indices,
            "source_video_output_hw": source_video_output_hw,
            "source_video_gradient_shape": source_video_gradient_shape,
            "quality_comparable": False,
            "score_claim": False,
        },
        dispatch_blockers=[
            "pr95_hnerv_mlx_timing_smoke_is_proxy_signal",
            *[str(item) for item in optimizer_blockers],
            *EXACT_READINESS_REFUSAL_BLOCKERS,
        ],
        evidence_grade="[macOS-MLX research-signal]",
        source_anchor="PR95 HNeRV Muon native MLX timing smoke plan",
        score_lowering_hypothesis=(
            "Queue PR95/HNeRV decoder and optimizer timing on local MLX so "
            "representation-training throughput can be measured before "
            "byte-closed export and exact auth anchoring."
        ),
        variant_axes=[
            "stage_curriculum",
            "optimizer_recipe",
            "training_backend",
            "representation_family",
            "byte_closed_archive_export",
        ],
        paired_modes=[
            "stage1_adamw",
            "stage5_adamw_qat_loss_path_future",
            "stage8_muon_adamw_partition",
        ],
        extra_fields={
            "source_schema": "pr95_hnerv_mlx_timing_smoke_plan.v1",
            "representation_family_class": "hnerv_variant",
            "recommended_execution": recommended_execution,
            "authority_contract": {
                "score_claim": False,
                "quality_authority": False,
                "promotion_authority": False,
                "dispatch_authority": False,
                "score_authority_requires": [
                    "runtime_consumption_proof",
                    "receiver_proof",
                    "pytorch_export_forward_parity",
                    "byte_closed_contest_archive_export",
                    "exact_cpu_cuda_auth_eval",
                ],
            },
            **FALSE_AUTHORITY,
        },
    )


def _build_plan(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    stage = int(args.stage)
    stage_module = _stage_module(stage)
    source_video_pair_indices = list(args.source_video_pair_index or [0])
    optimizer_descriptor_id = (
        args.optimizer_descriptor_id or pr95_default_optimizer_descriptor_id(stage)
    )
    optimizer_descriptor = pr95_mlx_optimizer_descriptor_row(optimizer_descriptor_id)
    optimizer_training_config = optimizer_descriptor.get("training_config", {})
    optimizer_backend_status = str(
        optimizer_training_config.get("backend_status")
        if isinstance(optimizer_training_config, dict)
        else ""
    )
    optimizer_blockers = (
        optimizer_training_config.get("dispatch_blockers", [])
        if isinstance(optimizer_training_config, dict)
        else []
    )
    recommended_execution = {
        "schema": "local_training_recommended_execution.v1",
        "tool": "tools/run_pr95_mlx_timing_smoke.py",
        "training_backend": "mlx",
        "device": "mlx",
        "resource_kind": "local_mlx",
        "output_manifest": _rel(output_dir / "manifest.json"),
        "representation_manifest": _rel(
            output_dir / "representation_training_manifest.json"
        ),
        "plan_manifest": _rel(output_dir / "plan.json"),
        "archive_export_manifest": _rel(output_dir / "pr95_public_archive_export.json")
        if args.write_pr95_public_archive_export or args.prove_pr95_runtime_consumption
        else None,
        "archive_export_zip": _rel(output_dir / "pr95_public_archive.zip")
        if args.write_pr95_public_archive_export or args.prove_pr95_runtime_consumption
        else None,
        "runtime_consumption_proof": _rel(output_dir / "runtime_consumption_proof.json")
        if args.prove_pr95_runtime_consumption
        else None,
        "source_faithful_preprocess_smoke": _rel(
            output_dir / "source_faithful_preprocess_smoke.json"
        )
        if args.write_source_faithful_preprocess_smoke
        else None,
        "source_preprocess_shape": args.source_preprocess_shape
        if args.write_source_faithful_preprocess_smoke
        else None,
        "source_preprocess_camera_hw": args.source_preprocess_camera_hw
        if args.write_source_faithful_preprocess_smoke
        else None,
        "source_preprocess_gradient_shape": args.source_preprocess_gradient_shape
        if args.write_source_faithful_preprocess_smoke
        else None,
        "source_video_preprocess_smoke": _rel(
            output_dir / "source_video_preprocess_smoke.json"
        )
        if args.write_source_video_preprocess_smoke
        else None,
        "source_video_path": _rel(args.source_video_path)
        if args.write_source_video_preprocess_smoke
        else None,
        "source_video_upstream_dir": _rel(args.source_video_upstream_dir)
        if args.write_source_video_preprocess_smoke
        else None,
        "source_video_pair_indices": source_video_pair_indices
        if args.write_source_video_preprocess_smoke
        else None,
        "source_video_output_hw": args.source_video_output_hw
        if args.write_source_video_preprocess_smoke
        else None,
        "source_video_gradient_shape": args.source_video_gradient_shape
        if args.write_source_video_preprocess_smoke
        else None,
        "extra_artifact_postconditions": _extra_artifact_postconditions(
            output_dir=output_dir,
            write_pr95_public_archive_export=(
                args.write_pr95_public_archive_export
                or args.prove_pr95_runtime_consumption
            ),
            prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
            write_source_faithful_preprocess_smoke=(
                args.write_source_faithful_preprocess_smoke
            ),
            write_source_video_preprocess_smoke=(
                args.write_source_video_preprocess_smoke
            ),
        ),
        "optimizer_descriptor_id": optimizer_descriptor_id,
        "optimizer_config_sha256": optimizer_descriptor["config_sha256"],
        "optimizer_backend_status": optimizer_backend_status,
        "parameter_group_lr_policy_id": optimizer_descriptor[
            "parameter_group_lr_policy_id"
        ],
        "parameter_group_lr_policy_sha256": optimizer_descriptor[
            "parameter_group_lr_policy_sha256"
        ],
        "python_command_args": _recommended_execution_command(
            stage=stage,
            steps=args.steps,
            batch_size=args.batch_size,
            synthetic_pairs=args.synthetic_pairs,
            seed=args.seed,
            base_channels=args.base_channels,
            latent_dim=args.latent_dim,
            output_dir=output_dir,
            write_byte_closed_smoke=args.write_byte_closed_smoke,
            write_pr95_public_archive_export=(
                args.write_pr95_public_archive_export
                or args.prove_pr95_runtime_consumption
            ),
            prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
            write_source_faithful_preprocess_smoke=(
                args.write_source_faithful_preprocess_smoke
            ),
            write_source_video_preprocess_smoke=(
                args.write_source_video_preprocess_smoke
            ),
            source_preprocess_shape=args.source_preprocess_shape,
            source_preprocess_camera_hw=args.source_preprocess_camera_hw,
            source_preprocess_gradient_shape=args.source_preprocess_gradient_shape,
            source_video_path=args.source_video_path,
            source_video_upstream_dir=args.source_video_upstream_dir,
            source_video_pair_indices=source_video_pair_indices,
            source_video_output_hw=args.source_video_output_hw,
            source_video_gradient_shape=args.source_video_gradient_shape,
            runtime_proof_timeout_seconds=args.runtime_proof_timeout_seconds,
            runtime_proof_max_output_bytes=args.runtime_proof_max_output_bytes,
            optimizer_descriptor_id=optimizer_descriptor_id,
        ),
        "candidate_generation_only": True,
        **FALSE_AUTHORITY,
    }
    recommended_execution = {
        key: value for key, value in recommended_execution.items() if value is not None
    }
    representation_plan = _build_representation_training_plan(
        stage=stage,
        steps=args.steps,
        batch_size=args.batch_size,
        synthetic_pairs=args.synthetic_pairs,
        seed=args.seed,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        output_dir=output_dir,
        write_byte_closed_smoke=args.write_byte_closed_smoke,
        write_pr95_public_archive_export=(
            args.write_pr95_public_archive_export
            or args.prove_pr95_runtime_consumption
        ),
        prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
        write_source_faithful_preprocess_smoke=(
            args.write_source_faithful_preprocess_smoke
        ),
        write_source_video_preprocess_smoke=args.write_source_video_preprocess_smoke,
        source_preprocess_shape=args.source_preprocess_shape,
        source_preprocess_camera_hw=args.source_preprocess_camera_hw,
        source_preprocess_gradient_shape=args.source_preprocess_gradient_shape,
        source_video_path=args.source_video_path,
        source_video_upstream_dir=args.source_video_upstream_dir,
        source_video_pair_indices=source_video_pair_indices,
        source_video_output_hw=args.source_video_output_hw,
        source_video_gradient_shape=args.source_video_gradient_shape,
        recommended_execution=recommended_execution,
        optimizer_descriptor=optimizer_descriptor,
    )
    return {
        "schema": "representation_training_probe_plan_v1",
        "lane_id": LANE_ID,
        "candidate_id": _candidate_id(
            stage=stage,
            seed=args.seed,
            steps=args.steps,
            base_channels=args.base_channels,
            optimizer_descriptor_id=optimizer_descriptor_id,
        ),
        "candidate_family": "pr95_hnerv_mlx_reproduction_timing_smoke",
        "representation_family": "hnerv",
        "representation_family_class": "hnerv_variant",
        "substrate_family": "nerv_family",
        "generated_utc": datetime.now(UTC).isoformat(),
        "stage_index": stage,
        "stage_module": stage_module,
        "stages": [{"index": stage, "module": stage_module}],
        "stage_count": 1,
        "training_backend": "mlx",
        "device_requested": "mlx",
        "device_selected": "mlx",
        "seed": args.seed,
        "optimizer_descriptor_id": optimizer_descriptor_id,
        "optimizer_config_sha256": optimizer_descriptor["config_sha256"],
        "optimizer_backend_status": optimizer_backend_status,
        "parameter_group_lr_policy_id": optimizer_descriptor[
            "parameter_group_lr_policy_id"
        ],
        "parameter_group_lr_policy_sha256": optimizer_descriptor[
            "parameter_group_lr_policy_sha256"
        ],
        "output_dir": _rel(output_dir),
        "write_byte_closed_smoke": bool(args.write_byte_closed_smoke),
        "write_pr95_public_archive_export": bool(
            args.write_pr95_public_archive_export or args.prove_pr95_runtime_consumption
        ),
        "prove_pr95_runtime_consumption": bool(args.prove_pr95_runtime_consumption),
        "write_source_faithful_preprocess_smoke": bool(
            args.write_source_faithful_preprocess_smoke
        ),
        "source_preprocess_shape": args.source_preprocess_shape,
        "source_preprocess_camera_hw": args.source_preprocess_camera_hw,
        "source_preprocess_gradient_shape": args.source_preprocess_gradient_shape,
        "write_source_video_preprocess_smoke": bool(
            args.write_source_video_preprocess_smoke
        ),
        "source_video_path": _rel(args.source_video_path),
        "source_video_upstream_dir": _rel(args.source_video_upstream_dir),
        "source_video_pair_indices": source_video_pair_indices,
        "source_video_output_hw": args.source_video_output_hw,
        "source_video_gradient_shape": args.source_video_gradient_shape,
        "recommended_execution": recommended_execution,
        "representation_training_plan": _rel(
            output_dir / "representation_training_plan.json"
        ),
        "representation_training_plan_schema": representation_plan["schema"],
        "dispatch_blockers": [
            "pr95_hnerv_mlx_timing_smoke_plan_is_proxy_signal",
            *[str(item) for item in optimizer_blockers],
            *EXACT_READINESS_REFUSAL_BLOCKERS,
        ],
        "evidence_grade": "[macOS-MLX research-signal]",
        **FALSE_AUTHORITY,
    }


def _representation_manifest(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    archive_summary: dict[str, Any] | None,
    runtime_consumption_proof: dict[str, Any] | None,
    source_faithful_preprocess_smoke: dict[str, Any] | None,
    source_video_preprocess_smoke: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_index = int(manifest["stage_index"])
    stage_module = str(manifest["stage_module"])
    candidate_id = str(manifest["candidate_id"])
    optimizer_recipe = (
        manifest["optimizer_recipe"]
        if isinstance(manifest.get("optimizer_recipe"), dict)
        else {}
    )
    sidecar_path = output_dir / "representation_training_manifest.json"
    public_archive_export = (
        manifest["pr95_public_archive_export"]
        if isinstance(manifest.get("pr95_public_archive_export"), dict)
        else None
    )
    archive_zip = public_archive_export or archive_summary
    runtime_consumption_proven = bool(
        runtime_consumption_proof
        and runtime_consumption_proof.get("runtime_consumption_proven") is True
    )
    exact_blockers = _exact_readiness_blockers(
        runtime_consumption_proven=runtime_consumption_proven,
        source_faithful_preprocess_smoke=source_faithful_preprocess_smoke,
        source_video_preprocess_smoke=source_video_preprocess_smoke,
    )
    return write_representation_training_probe_manifest(
        sidecar_path,
        candidate_id=candidate_id,
        lane_id=LANE_ID,
        lane_class="pr95_hnerv_mlx_reproduction_local_training_proxy",
        candidate_family="pr95_hnerv_mlx_reproduction_timing_smoke",
        representation_family="hnerv",
        substrate_family="nerv_family",
        profile="pr95_hnerv_mlx_stage_timing_smoke",
        param_schema="pr95_hnerv_mlx_timing_smoke_params_v1",
        training_signal_kind="local_mlx_representation_training_runtime_probe",
        seed=manifest.get("seed"),
        device_requested="mlx",
        device_selected="mlx",
        output_dir=_rel(output_dir),
        stage_count=1,
        stages=[{"index": stage_index, "module": stage_module}],
        results=[
            {
                "stage_index": stage_index,
                "stage_module": stage_module,
                "best_score": None,
                "training_loss": manifest.get("last_loss"),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        training_recipe={
            "id": "pr95_hnerv_mlx_synthetic_stage_timing_smoke",
            "source_pr": 95,
            "stage_module": stage_module,
            "steps": manifest.get("steps"),
            "batch_size": manifest.get("batch_size"),
            "synthetic_pairs": manifest.get("synthetic_pairs"),
            "quality_comparable": False,
            "full_scorer_gradient_path": False,
        },
        optimizer_recipe=optimizer_recipe,
        scheduler_recipe={
            "id": "single_process_mlx_local_smoke",
            "resource_kind": "local_mlx",
            "queue_authority": "experiment_queue.v1",
        },
        candidate_params={
            "stage_index": stage_index,
            "stage_module": stage_module,
            "stage_count": 1,
            "training_backend": "mlx",
            "seed": manifest.get("seed"),
            "base_channels": manifest.get("architecture", {}).get("base_channels"),
            "latent_dim": manifest.get("architecture", {}).get("latent_dim"),
            "optimizer_descriptor_id": optimizer_recipe.get(
                "optimizer_descriptor_id"
            ),
            "optimizer_config_sha256": optimizer_recipe.get("optimizer_config_sha256"),
            "optimizer_backend_status": optimizer_recipe.get(
                "optimizer_backend_status"
            ),
            "parameter_group_lr_policy_id": optimizer_recipe.get(
                "parameter_group_lr_policy_id"
            ),
            "parameter_group_lr_policy_sha256": optimizer_recipe.get(
                "parameter_group_lr_policy_sha256"
            ),
            "parameter_group_fingerprint_sha256": optimizer_recipe.get(
                "parameter_group_fingerprint_sha256"
            ),
            "byte_closed_smoke_archive_emitted": archive_summary is not None,
            "pr95_public_archive_export_emitted": public_archive_export is not None,
            "pr95_runtime_consumption_proof_present": runtime_consumption_proven,
            "source_faithful_preprocess_smoke_present": (
                source_faithful_preprocess_smoke is not None
            ),
            "source_preprocess_shape": manifest.get("source_preprocess_shape"),
            "source_preprocess_camera_hw": manifest.get("source_preprocess_camera_hw"),
            "source_preprocess_gradient_shape": manifest.get(
                "source_preprocess_gradient_shape"
            ),
            "source_video_preprocess_smoke_present": (
                source_video_preprocess_smoke is not None
            ),
            "source_video_path": manifest.get("source_video_path"),
            "source_video_upstream_dir": manifest.get("source_video_upstream_dir"),
            "source_video_pair_indices": manifest.get("source_video_pair_indices"),
            "source_video_output_hw": manifest.get("source_video_output_hw"),
            "source_video_gradient_shape": manifest.get(
                "source_video_gradient_shape"
            ),
            "quality_comparable": False,
            "score_claim": False,
        },
        archive_zip=archive_zip,
        dispatch_blockers=[
            "pr95_hnerv_mlx_timing_smoke_is_proxy_signal",
            *exact_blockers,
        ],
        evidence_grade="[macOS-MLX research-signal]",
        source_anchor="PR95 HNeRV Muon native MLX timing smoke",
        score_lowering_hypothesis=(
            "Measure PR95/HNeRV decoder and optimizer timing on MLX so local "
            "substrate training can replace expensive cloud iteration after "
            "byte-closed export and exact auth anchoring."
        ),
        variant_axes=[
            "stage_curriculum",
            "optimizer_recipe",
            "training_backend",
            "representation_family",
            "byte_closed_archive_export",
        ],
        paired_modes=[
            "stage1_adamw",
            "stage5_adamw_qat_loss_path_future",
            "stage8_muon_adamw_partition",
        ],
        extra_fields={
            "source_schema": SMOKE_MANIFEST_SCHEMA,
            "runtime_profile": manifest["runtime_profile"],
            "runtime_profiles": [manifest["runtime_profile"]],
            "exact_readiness_refusal": manifest["exact_readiness_refusal"],
            "pytorch_export_parity": manifest["pytorch_export_parity"],
            "byte_closed_smoke_archive": archive_summary,
            "pr95_public_archive_export": public_archive_export,
            "runtime_consumption_proof": runtime_consumption_proof or {},
            "source_faithful_preprocess_smoke": (
                source_faithful_preprocess_smoke or {}
            ),
            "source_video_preprocess_smoke": source_video_preprocess_smoke or {},
            "authority_contract": {
                "score_claim": False,
                "quality_authority": False,
                "promotion_authority": False,
                "dispatch_authority": False,
                "score_authority_requires": [
                    "runtime_consumption_proof",
                    "receiver_proof",
                    "pytorch_export_forward_parity",
                    "byte_closed_contest_archive_export",
                    "exact_cpu_cuda_auth_eval",
                ],
            },
            **FALSE_AUTHORITY,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=int,
        choices=sorted(PR95_STAGE_MODULES),
        required=True,
        help="PR95 stage to time-smoke: 1, 5, or 8.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--synthetic-pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-channels", type=int, default=36)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument(
        "--optimizer-descriptor-id",
        help=(
            "Optimizer scheduler descriptor ID. Defaults to the source-faithful "
            "PR95 descriptor for the selected stage."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for manifest/runtime-profile/optional smoke archive.",
    )
    parser.add_argument(
        "--write-byte-closed-smoke",
        action="store_true",
        help="Emit a deterministic single-member ZIP smoke archive for queue plumbing.",
    )
    parser.add_argument(
        "--write-pr95-public-archive-export",
        action="store_true",
        help="Emit a deterministic PR95-compatible archive.zip consumed by the public runtime.",
    )
    parser.add_argument(
        "--prove-pr95-runtime-consumption",
        action="store_true",
        help="Run public PR95 inflate.sh against the emitted PR95 archive export.",
    )
    parser.add_argument("--runtime-proof-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--runtime-proof-max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument(
        "--write-source-faithful-preprocess-smoke",
        action="store_true",
        help=(
            "Emit native MLX eval-roundtrip/YUV6 preprocessing smoke with "
            "gradient reachability proof. Local training signal only."
        ),
    )
    parser.add_argument(
        "--source-preprocess-shape",
        default="1,2,384,512,3",
        help="Comma-separated RGB NHWC-ish smoke input shape.",
    )
    parser.add_argument(
        "--source-preprocess-camera-hw",
        default="874,1164",
        help="Comma-separated camera H,W for preprocessing smoke.",
    )
    parser.add_argument(
        "--source-preprocess-gradient-shape",
        default="1,2,16,20,3",
        help="Comma-separated RGB shape for preprocessing gradient probe.",
    )
    parser.add_argument(
        "--write-source-video-preprocess-smoke",
        action="store_true",
        help=(
            "Decode real PR95 source-video pairs through upstream CPU decode, "
            "then emit native MLX scorer-resolution/YUV6 preprocess smoke."
        ),
    )
    parser.add_argument(
        "--source-video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Path to PR95 source video 0.mkv.",
    )
    parser.add_argument(
        "--source-video-upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="Path to upstream PR95 runtime directory containing frame_utils.py.",
    )
    parser.add_argument(
        "--source-video-pair-index",
        action="append",
        type=int,
        help="PR95 pair index to decode for source-video preprocess smoke.",
    )
    parser.add_argument(
        "--source-video-output-hw",
        default="384,512",
        help="Comma-separated scorer output H,W for source-video preprocess smoke.",
    )
    parser.add_argument(
        "--source-video-gradient-shape",
        default="1,2,16,20,3",
        help="Comma-separated RGB shape for source-video gradient probe.",
    )
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Write plan.json and representation_training_plan.json for "
            "experiment_queue execution without requiring MLX at planning time."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.allow_existing_output_dir:
        raise SystemExit(
            f"output directory already exists: {_rel(output_dir)} "
            "(pass --allow-existing-output-dir to append/overwrite manifests)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        plan = _build_plan(args, output_dir=output_dir)
        _write_json(output_dir / "plan.json", plan)
        summary = {
            "schema": "pr95_hnerv_mlx_timing_smoke_plan_summary_v1",
            "ok": True,
            "plan": _rel(output_dir / "plan.json"),
            "representation_training_plan": _rel(
                output_dir / "representation_training_plan.json"
            ),
            "candidate_id": plan["candidate_id"],
            "stage_index": plan["stage_index"],
            "stage_module": plan["stage_module"],
            "queue_schema": "experiment_queue.v1",
            **FALSE_AUTHORITY,
        }
        _write_json(output_dir / "plan_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    manifest = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=args.stage,
        steps=args.steps,
        batch_size=args.batch_size,
        synthetic_pairs=args.synthetic_pairs,
        seed=args.seed,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        optimizer_descriptor_id=args.optimizer_descriptor_id,
        pr95_public_archive_export_path=(
            output_dir / "pr95_public_archive.zip"
            if args.write_pr95_public_archive_export
            or args.prove_pr95_runtime_consumption
            else None
        ),
    )
    validate_runtime_profile_observation(manifest["runtime_profile"])
    archive_summary = (
        write_pr95_mlx_byte_closed_smoke_archive(manifest, output_dir=output_dir)
        if args.write_byte_closed_smoke
        else None
    )
    if archive_summary is not None:
        manifest["byte_closed_smoke_archive"] = archive_summary
    public_archive_export = (
        manifest["pr95_public_archive_export"]
        if isinstance(manifest.get("pr95_public_archive_export"), dict)
        else None
    )
    if public_archive_export is not None:
        _write_json(output_dir / "pr95_public_archive_export.json", public_archive_export)

    runtime_consumption_proof: dict[str, Any] | None = None
    if args.prove_pr95_runtime_consumption:
        if public_archive_export is None:
            raise SystemExit("--prove-pr95-runtime-consumption requires PR95 archive export")
        proof_path = output_dir / "runtime_consumption_proof.json"
        proof_result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "prove_pr95_public_archive_runtime_consumption.py"),
                "--archive-zip",
                str(output_dir / "pr95_public_archive.zip"),
                "--output-json",
                str(proof_path),
                "--timeout-seconds",
                str(args.runtime_proof_timeout_seconds),
                "--max-output-bytes",
                str(args.runtime_proof_max_output_bytes),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.runtime_proof_timeout_seconds + 30,
        )
        if proof_result.returncode != 0:
            raise SystemExit(
                "PR95 public runtime-consumption proof failed: "
                f"{proof_result.stderr or proof_result.stdout}"
            )
        runtime_consumption_proof = json.loads(proof_path.read_text(encoding="utf-8"))
        manifest["runtime_consumption_proof"] = runtime_consumption_proof
        manifest["runtime_consumption_proof_path"] = _rel(proof_path)
        public_archive_export["runtime_consumption_proof_present"] = True
        public_archive_export["runtime_consumption_proof_path"] = _rel(proof_path)
        _write_json(output_dir / "pr95_public_archive_export.json", public_archive_export)
        manifest["runtime_profile"]["packet_compiler_bridge"][
            "runtime_consumption_proof_present"
        ] = runtime_consumption_proof.get("runtime_consumption_proven") is True
        manifest["runtime_profile"]["packet_compiler_bridge"][
            "runtime_consumption_proof_path"
        ] = _rel(proof_path)
        manifest["runtime_profile"]["packet_compiler_bridge"]["blockers"] = [
            blocker
            for blocker in manifest["runtime_profile"]["packet_compiler_bridge"].get(
                "blockers",
                [],
            )
            if blocker != "runtime_consumption_proof_missing"
        ]

    source_faithful_preprocess_smoke: dict[str, Any] | None = None
    if args.write_source_faithful_preprocess_smoke:
        camera_hw = _parse_int_tuple(
            args.source_preprocess_camera_hw,
            field="source-preprocess-camera-hw",
            expected_len=2,
        )
        source_faithful_preprocess_smoke = run_pr95_mlx_source_faithful_smoke(
            input_shape=_parse_int_tuple(
                args.source_preprocess_shape,
                field="source-preprocess-shape",
            ),
            camera_hw=(camera_hw[0], camera_hw[1]),
            seed=args.seed,
            gradient_probe_shape=_parse_int_tuple(
                args.source_preprocess_gradient_shape,
                field="source-preprocess-gradient-shape",
            ),
        )
        _write_json(
            output_dir / "source_faithful_preprocess_smoke.json",
            source_faithful_preprocess_smoke,
        )
        manifest["source_faithful_preprocess_smoke"] = source_faithful_preprocess_smoke
        manifest["source_faithful_preprocess_smoke_path"] = _rel(
            output_dir / "source_faithful_preprocess_smoke.json"
        )
        manifest["source_preprocess_shape"] = args.source_preprocess_shape
        manifest["source_preprocess_camera_hw"] = args.source_preprocess_camera_hw
        manifest["source_preprocess_gradient_shape"] = (
            args.source_preprocess_gradient_shape
        )

    source_video_preprocess_smoke: dict[str, Any] | None = None
    if args.write_source_video_preprocess_smoke:
        output_hw = _parse_int_tuple(
            args.source_video_output_hw,
            field="source-video-output-hw",
            expected_len=2,
        )
        source_video_pair_indices = list(args.source_video_pair_index or [0])
        source_video_preprocess_smoke = run_pr95_mlx_source_video_preprocess_smoke(
            video_path=args.source_video_path,
            upstream_dir=args.source_video_upstream_dir,
            pair_indices=source_video_pair_indices,
            output_hw=(output_hw[0], output_hw[1]),
            gradient_probe_shape=_parse_int_tuple(
                args.source_video_gradient_shape,
                field="source-video-gradient-shape",
            ),
        )
        _write_json(
            output_dir / "source_video_preprocess_smoke.json",
            source_video_preprocess_smoke,
        )
        manifest["source_video_preprocess_smoke"] = source_video_preprocess_smoke
        manifest["source_video_preprocess_smoke_path"] = _rel(
            output_dir / "source_video_preprocess_smoke.json"
        )
        manifest["source_video_path"] = _rel(args.source_video_path)
        manifest["source_video_upstream_dir"] = _rel(args.source_video_upstream_dir)
        manifest["source_video_pair_indices"] = source_video_pair_indices
        manifest["source_video_output_hw"] = args.source_video_output_hw
        manifest["source_video_gradient_shape"] = args.source_video_gradient_shape

    manifest["exact_readiness_refusal"]["blockers"] = _exact_readiness_blockers(
        runtime_consumption_proven=(
            runtime_consumption_proof is not None
            and runtime_consumption_proof.get("runtime_consumption_proven") is True
        ),
        source_faithful_preprocess_smoke=source_faithful_preprocess_smoke,
        source_video_preprocess_smoke=source_video_preprocess_smoke,
    )

    _write_json(output_dir / "runtime_profile.json", manifest["runtime_profile"])
    _write_json(output_dir / "manifest.json", manifest)
    representation = _representation_manifest(
        manifest=manifest,
        output_dir=output_dir,
        archive_summary=archive_summary,
        runtime_consumption_proof=runtime_consumption_proof,
        source_faithful_preprocess_smoke=source_faithful_preprocess_smoke,
        source_video_preprocess_smoke=source_video_preprocess_smoke,
    )
    summary = {
        "schema": "pr95_hnerv_mlx_timing_smoke_run_summary_v1",
        "ok": True,
        "manifest": _rel(output_dir / "manifest.json"),
        "runtime_profile": _rel(output_dir / "runtime_profile.json"),
        "representation_training_manifest": _rel(
            output_dir / "representation_training_manifest.json"
        ),
        "byte_closed_smoke_archive": archive_summary,
        "pr95_public_archive_export": public_archive_export,
        "runtime_consumption_proof": runtime_consumption_proof,
        "source_faithful_preprocess_smoke": source_faithful_preprocess_smoke,
        "source_video_preprocess_smoke": source_video_preprocess_smoke,
        "candidate_id": manifest["candidate_id"],
        "seconds_per_step": manifest["timing"]["seconds_per_step"],
        "representation_manifest_schema": representation["schema"],
        **FALSE_AUTHORITY,
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
