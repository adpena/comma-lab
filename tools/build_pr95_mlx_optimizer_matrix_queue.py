#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build queue-owned PR95/HNeRV MLX optimizer timing-smoke matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.local_training_queue import (  # noqa: E402
    build_local_training_execution_queue,
)
from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    EXACT_READINESS_REFUSAL_BLOCKERS,
    FALSE_AUTHORITY,
    LANE_ID,
    PR95_MLX_BACKEND_STATUS_LOCAL_TIMING_PROXY,
    PR95_MLX_LOSS_SURFACE_RGB_MSE,
    PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE,
    PR95_MLX_LOSS_SURFACES,
    PR95_STAGE_MODULES,
    Pr95HNeRVMlxError,
    pr95_mlx_optimizer_config_from_descriptor,
)
from tac.optimization.optimizer_scheduler_registry import (  # noqa: E402
    default_optimizer_scheduler_registry,
)
from tac.repo_io import ArtifactWriteError, read_json, write_json_artifact  # noqa: E402

MATRIX_SCHEMA = "pr95_hnerv_mlx_optimizer_matrix_queue.v1"
CONTROL_PROFILE_FULL_SOURCE_VIDEO_RUNTIME = "full_pr95_source_video_runtime"
CONTROL_PROFILE_CHOICES = (CONTROL_PROFILE_FULL_SOURCE_VIDEO_RUNTIME,)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _stage_indices(values: list[int] | None) -> list[int]:
    stages = values or sorted(PR95_STAGE_MODULES)
    unknown = sorted(set(stages) - set(PR95_STAGE_MODULES))
    if unknown:
        raise ExperimentQueueError(
            f"unsupported PR95 stage(s): {unknown}; supported={sorted(PR95_STAGE_MODULES)}"
        )
    return list(dict.fromkeys(int(stage) for stage in stages))


def _descriptor_stage_indices(row: dict[str, Any]) -> list[int]:
    training_config = row.get("training_config")
    if not isinstance(training_config, dict):
        return []
    indices = training_config.get("pr95_stage_indices")
    if not isinstance(indices, list):
        return []
    out: list[int] = []
    for value in indices:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _executable_descriptors_for_stage(stage: int) -> list[str]:
    rows = default_optimizer_scheduler_registry().to_dict()["descriptors"]
    out: list[str] = []
    for row in rows:
        training_config = row.get("training_config")
        if not isinstance(training_config, dict):
            continue
        if (
            training_config.get("backend_status")
            != PR95_MLX_BACKEND_STATUS_LOCAL_TIMING_PROXY
        ):
            continue
        if int(stage) not in _descriptor_stage_indices(row):
            continue
        out.append(str(row["descriptor_id"]))
    if not out:
        raise ExperimentQueueError(f"no executable PR95 MLX descriptors for stage {stage}")
    return out


def _candidate_output_dir(
    *,
    output_root: Path,
    matrix_cell_id: str,
    stage: int,
    descriptor_id: str,
    seed: int,
    base_channels: int,
) -> Path:
    return (
        output_root
        / f"stage{stage}"
        / _slug(descriptor_id)
        / f"seed{seed}_c{base_channels}_{matrix_cell_id[:12]}"
    )


def _matrix_cell_id(
    *,
    stage: int,
    descriptor_id: str,
    seed: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    base_channels: int,
    latent_dim: int,
    write_source_video_preprocess_smoke: bool,
    train_on_source_video_pairs: bool,
    source_video_loss_surface: str,
    source_video_path: Path,
    source_video_upstream_dir: Path,
    source_video_pair_indices: list[int],
    source_video_output_hw: str,
    source_video_gradient_shape: str,
) -> str:
    payload = {
        "stage": stage,
        "optimizer_descriptor_id": descriptor_id,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "synthetic_pairs": synthetic_pairs,
        "base_channels": base_channels,
        "latent_dim": latent_dim,
    }
    if write_source_video_preprocess_smoke or train_on_source_video_pairs:
        payload.update(
            {
                "write_source_video_preprocess_smoke": write_source_video_preprocess_smoke,
                "train_on_source_video_pairs": train_on_source_video_pairs,
                "source_video_loss_surface": source_video_loss_surface,
                "source_video_path": str(source_video_path),
                "source_video_upstream_dir": str(source_video_upstream_dir),
                "source_video_pair_indices": source_video_pair_indices,
                "source_video_output_hw": source_video_output_hw,
                "source_video_gradient_shape": source_video_gradient_shape,
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_executable_descriptor(stage: int, descriptor_id: str) -> str | None:
    try:
        pr95_mlx_optimizer_config_from_descriptor(
            descriptor_id,
            stage_index=stage,
        )
    except (Pr95HNeRVMlxError, ValueError) as exc:
        return str(exc)
    return None


def _emit_plan(
    *,
    repo_root: Path,
    stage: int,
    descriptor_id: str,
    seed: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    base_channels: int,
    latent_dim: int,
    output_dir: Path,
    write_byte_closed_smoke: bool,
    write_pr95_public_archive_export: bool,
    prove_pr95_runtime_consumption: bool,
    write_pytorch_export_parity: bool,
    pytorch_export_sample_indices: list[int],
    pytorch_export_mlx_device: str,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
    train_on_source_video_pairs: bool,
    source_video_loss_surface: str,
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
    allow_existing_plan_dirs: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "tools" / "run_pr95_mlx_timing_smoke.py"),
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
        descriptor_id,
        "--output-dir",
        str(output_dir),
        "--plan-only",
    ]
    if write_byte_closed_smoke:
        command.append("--write-byte-closed-smoke")
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
    if write_source_video_preprocess_smoke or train_on_source_video_pairs:
        if write_source_video_preprocess_smoke:
            command.append("--write-source-video-preprocess-smoke")
        command.extend(
            [
                "--source-video-path",
                str(source_video_path),
                "--source-video-upstream-dir",
                str(source_video_upstream_dir),
                "--source-video-output-hw",
                source_video_output_hw,
                "--source-video-gradient-shape",
                source_video_gradient_shape,
            ]
        )
        for pair_index in source_video_pair_indices:
            command.extend(["--source-video-pair-index", str(pair_index)])
    if train_on_source_video_pairs:
        command.append("--train-on-source-video-pairs")
        command.extend(["--source-video-loss-surface", source_video_loss_surface])
    if write_pr95_public_archive_export or prove_pr95_runtime_consumption:
        command.append("--write-pr95-public-archive-export")
    if prove_pr95_runtime_consumption:
        command.extend(
            [
                "--prove-pr95-runtime-consumption",
                "--runtime-proof-timeout-seconds",
                str(runtime_proof_timeout_seconds),
                "--runtime-proof-max-output-bytes",
                str(runtime_proof_max_output_bytes),
            ]
        )
    if write_pytorch_export_parity:
        command.extend(
            [
                "--write-pytorch-export-parity",
                "--pytorch-export-mlx-device",
                pytorch_export_mlx_device,
            ]
        )
        for sample_index in pytorch_export_sample_indices:
            command.extend(["--pytorch-export-sample-index", str(sample_index)])
    if allow_existing_plan_dirs:
        command.append("--allow-existing-output-dir")
    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise ExperimentQueueError(
            "PR95 MLX timing-smoke plan emission failed for "
            f"stage={stage} descriptor={descriptor_id} seed={seed}: {result.stderr}"
        )
    plan_path = output_dir / "plan.json"
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ExperimentQueueError(f"{plan_path}: expected JSON object")
    return plan


def build_pr95_mlx_optimizer_matrix_queue(
    *,
    repo_root: Path,
    output_root: Path,
    queue_id: str,
    stages: list[int],
    seeds: list[int],
    descriptor_ids: list[str] | None,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    base_channels: int,
    latent_dim: int,
    write_byte_closed_smoke: bool,
    write_pr95_public_archive_export: bool,
    prove_pr95_runtime_consumption: bool,
    write_pytorch_export_parity: bool,
    pytorch_export_sample_indices: list[int],
    pytorch_export_mlx_device: str,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
    train_on_source_video_pairs: bool,
    source_video_loss_surface: str,
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
    local_cpu_concurrency: int,
    local_mlx_concurrency: int,
    local_cuda_concurrency: int,
    local_mps_concurrency: int,
    timeout_seconds: int,
    control_profile: str | None = None,
    allow_existing_plan_dirs: bool = False,
) -> dict[str, Any]:
    """Emit per-candidate plans and return matrix manifest plus queue payload."""

    selected_stages = _stage_indices(stages)
    selected_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    source_video_config_requested = (
        write_source_video_preprocess_smoke or train_on_source_video_pairs
    )
    selected_source_video_pair_indices = (
        list(dict.fromkeys(int(index) for index in source_video_pair_indices))
        if source_video_config_requested
        else []
    )
    if source_video_config_requested and not selected_source_video_pair_indices:
        selected_source_video_pair_indices = [0]
    if (
        not train_on_source_video_pairs
        and source_video_loss_surface != PR95_MLX_LOSS_SURFACE_RGB_MSE
    ):
        raise ExperimentQueueError(
            "--source-video-loss-surface other than rgb_mse requires "
            "--train-on-source-video-pairs"
        )
    if not selected_seeds:
        raise ExperimentQueueError("at least one seed is required")
    output_root.mkdir(parents=True, exist_ok=True)

    plans: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    seen_cell_ids: set[str] = set()
    for stage in selected_stages:
        stage_descriptors = (
            list(dict.fromkeys(descriptor_ids))
            if descriptor_ids
            else _executable_descriptors_for_stage(stage)
        )
        for descriptor_id in stage_descriptors:
            refusal = _validate_executable_descriptor(stage, descriptor_id)
            if refusal is not None:
                refusals.append(
                    {
                        "stage_index": stage,
                        "optimizer_descriptor_id": descriptor_id,
                        "reason": refusal,
                        "queued": False,
                        **FALSE_AUTHORITY,
                    }
                )
                continue
            for seed in selected_seeds:
                matrix_cell_id = _matrix_cell_id(
                    stage=stage,
                    descriptor_id=descriptor_id,
                    seed=seed,
                    steps=steps,
                    batch_size=batch_size,
                    synthetic_pairs=synthetic_pairs,
                    base_channels=base_channels,
                    latent_dim=latent_dim,
                    write_source_video_preprocess_smoke=(
                        write_source_video_preprocess_smoke
                    ),
                    train_on_source_video_pairs=train_on_source_video_pairs,
                    source_video_loss_surface=source_video_loss_surface,
                    source_video_path=source_video_path,
                    source_video_upstream_dir=source_video_upstream_dir,
                    source_video_pair_indices=selected_source_video_pair_indices,
                    source_video_output_hw=source_video_output_hw,
                    source_video_gradient_shape=source_video_gradient_shape,
                )
                if matrix_cell_id in seen_cell_ids:
                    raise ExperimentQueueError(
                        f"duplicate PR95 MLX matrix cell id: {matrix_cell_id}"
                    )
                seen_cell_ids.add(matrix_cell_id)
                plan_dir = _candidate_output_dir(
                    output_root=output_root,
                    matrix_cell_id=matrix_cell_id,
                    stage=stage,
                    descriptor_id=descriptor_id,
                    seed=seed,
                    base_channels=base_channels,
                )
                plan = _emit_plan(
                    repo_root=repo_root,
                    stage=stage,
                    descriptor_id=descriptor_id,
                    seed=seed,
                    steps=steps,
                    batch_size=batch_size,
                    synthetic_pairs=synthetic_pairs,
                    base_channels=base_channels,
                    latent_dim=latent_dim,
                    output_dir=plan_dir,
                    write_byte_closed_smoke=write_byte_closed_smoke,
                    write_pr95_public_archive_export=write_pr95_public_archive_export,
                    prove_pr95_runtime_consumption=prove_pr95_runtime_consumption,
                    write_pytorch_export_parity=write_pytorch_export_parity,
                    pytorch_export_sample_indices=pytorch_export_sample_indices,
                    pytorch_export_mlx_device=pytorch_export_mlx_device,
                    write_source_faithful_preprocess_smoke=(
                        write_source_faithful_preprocess_smoke
                    ),
                    write_source_video_preprocess_smoke=(
                        write_source_video_preprocess_smoke
                    ),
                    train_on_source_video_pairs=train_on_source_video_pairs,
                    source_video_loss_surface=source_video_loss_surface,
                    source_preprocess_shape=source_preprocess_shape,
                    source_preprocess_camera_hw=source_preprocess_camera_hw,
                    source_preprocess_gradient_shape=source_preprocess_gradient_shape,
                    source_video_path=source_video_path,
                    source_video_upstream_dir=source_video_upstream_dir,
                    source_video_pair_indices=selected_source_video_pair_indices,
                    source_video_output_hw=source_video_output_hw,
                    source_video_gradient_shape=source_video_gradient_shape,
                    runtime_proof_timeout_seconds=runtime_proof_timeout_seconds,
                    runtime_proof_max_output_bytes=runtime_proof_max_output_bytes,
                    allow_existing_plan_dirs=allow_existing_plan_dirs,
                )
                plans.append(plan)
                plan_records.append(
                    {
                        "matrix_cell_id": matrix_cell_id,
                        "candidate_id": plan["candidate_id"],
                        "stage_index": stage,
                        "stage_module": plan["stage_module"],
                        "seed": seed,
                        "optimizer_descriptor_id": descriptor_id,
                        "optimizer_config_sha256": plan["optimizer_config_sha256"],
                        "plan": _rel(plan_dir / "plan.json", repo_root),
                        "representation_training_plan": _rel(
                            plan_dir / "representation_training_plan.json",
                            repo_root,
                        ),
                        "run_manifest": _rel(plan_dir / "manifest.json", repo_root),
                        "queued": True,
                        **FALSE_AUTHORITY,
                    }
                )

    if not plans:
        refusal_summary = "; ".join(
            f"stage={row['stage_index']} descriptor={row['optimizer_descriptor_id']}: "
            f"{row['reason']}"
            for row in refusals[:5]
        )
        raise ExperimentQueueError(
            "no executable PR95 MLX matrix plans selected"
            + (f" ({refusal_summary})" if refusal_summary else "")
        )

    queue = build_local_training_execution_queue(
        plans,
        queue_id=queue_id,
        repo_root=repo_root,
        lane_id=LANE_ID,
        local_cpu_concurrency=local_cpu_concurrency,
        local_mlx_concurrency=local_mlx_concurrency,
        local_cuda_concurrency=local_cuda_concurrency,
        local_mps_concurrency=local_mps_concurrency,
        timeout_seconds=timeout_seconds,
    )
    manifest = {
        "schema": MATRIX_SCHEMA,
        "generated_at_utc": _utc_now(),
        "lane_id": LANE_ID,
        "queue_id": queue_id,
        "control_profile": control_profile,
        "output_root": _rel(output_root, repo_root),
        "stage_indices": selected_stages,
        "seeds": selected_seeds,
        "steps": steps,
        "batch_size": batch_size,
        "synthetic_pairs": synthetic_pairs,
        "base_channels": base_channels,
        "latent_dim": latent_dim,
        "write_byte_closed_smoke": write_byte_closed_smoke,
        "write_pr95_public_archive_export": bool(
            write_pr95_public_archive_export
            or prove_pr95_runtime_consumption
            or write_pytorch_export_parity
        ),
        "prove_pr95_runtime_consumption": prove_pr95_runtime_consumption,
        "write_pytorch_export_parity": write_pytorch_export_parity,
        "pytorch_export_sample_indices": pytorch_export_sample_indices,
        "pytorch_export_mlx_device": pytorch_export_mlx_device,
        "write_source_faithful_preprocess_smoke": (
            write_source_faithful_preprocess_smoke
        ),
        "source_preprocess_shape": source_preprocess_shape,
        "source_preprocess_camera_hw": source_preprocess_camera_hw,
        "source_preprocess_gradient_shape": source_preprocess_gradient_shape,
        "write_source_video_preprocess_smoke": (
            write_source_video_preprocess_smoke
        ),
        "train_on_source_video_pairs": train_on_source_video_pairs,
        "source_video_loss_surface": source_video_loss_surface,
        "source_video_path": _rel(source_video_path, repo_root),
        "source_video_upstream_dir": _rel(source_video_upstream_dir, repo_root),
        "source_video_pair_indices": selected_source_video_pair_indices,
        "source_video_output_hw": source_video_output_hw,
        "source_video_gradient_shape": source_video_gradient_shape,
        "runtime_proof_timeout_seconds": runtime_proof_timeout_seconds,
        "runtime_proof_max_output_bytes": runtime_proof_max_output_bytes,
        "plan_count": len(plan_records),
        "refusal_count": len(refusals),
        "plans": plan_records,
        "refusals": refusals,
        "queue_schema": queue["schema"],
        "queue_controls": queue["controls"],
        "execution_commands": {
            "validate": [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                "<queue-output>",
                "validate",
            ],
            "init": [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                "<queue-output>",
                "init",
            ],
            "run_worker_execute": [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                "<queue-output>",
                "run-worker",
                "--execute",
                "--max-parallel",
                "0",
            ],
        },
        "authority_contract": {
            "queue_authority": "experiment_queue.v1",
            "local_signal_axis": "[macOS-MLX research-signal]",
            "score_authority_requires": [
                *EXACT_READINESS_REFUSAL_BLOCKERS,
            ],
            "score_claim": False,
            "quality_authority": False,
            "promotion_authority": False,
            "rank_or_kill_authority": False,
            "ready_for_exact_eval_dispatch": False,
        },
        **FALSE_AUTHORITY,
    }
    return {"manifest": manifest, "queue": queue}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-profile",
        choices=CONTROL_PROFILE_CHOICES,
        default=None,
        help=(
            "Apply a canonical PR95 MLX control profile. "
            "full_pr95_source_video_runtime emits stage 1/2/3/4/5/8, full PR95 "
            "dimensions, source-video RGB+YUV6 timing loss, and PR95 runtime "
            "consumption proof requests."
        ),
    )
    parser.add_argument("--stage", action="append", type=int, dest="stages")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--optimizer-descriptor-id", action="append")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--synthetic-pairs", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=36)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--write-byte-closed-smoke", action="store_true")
    parser.add_argument(
        "--write-source-faithful-preprocess-smoke",
        action="store_true",
        help=(
            "Ask each timing-smoke plan to emit the native MLX "
            "source-faithful PR95 eval-roundtrip/YUV6 preprocess smoke."
        ),
    )
    parser.add_argument(
        "--source-preprocess-shape",
        default="1,2,384,512,3",
        help="Synthetic RGB NHWC fixture shape for source preprocess smoke.",
    )
    parser.add_argument(
        "--source-preprocess-camera-hw",
        default="874,1164",
        help="Camera-space H,W for source preprocess eval-roundtrip smoke.",
    )
    parser.add_argument(
        "--source-preprocess-gradient-shape",
        default="1,2,16,20,3",
        help="Small synthetic RGB NHWC shape for source preprocess gradient probe.",
    )
    parser.add_argument(
        "--write-source-video-preprocess-smoke",
        action="store_true",
        help=(
            "Ask each timing-smoke plan to decode real PR95 source-video "
            "pairs and emit native MLX scorer/YUV6 preprocess smoke."
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
        "--train-on-source-video-pairs",
        action="store_true",
        help=(
            "Ask each timing-smoke plan to train against decoded PR95 "
            "source-video RGB pairs. Local MLX research signal only."
        ),
    )
    parser.add_argument(
        "--source-video-loss-surface",
        choices=PR95_MLX_LOSS_SURFACES,
        default=PR95_MLX_LOSS_SURFACE_RGB_MSE,
        help=(
            "Loss surface for generated --train-on-source-video-pairs plans. "
            "rgb_yuv6_mse adds scorer-preprocess YUV6 MSE to RGB MSE."
        ),
    )
    parser.add_argument("--write-pr95-public-archive-export", action="store_true")
    parser.add_argument("--prove-pr95-runtime-consumption", action="store_true")
    parser.add_argument("--write-pytorch-export-parity", action="store_true")
    parser.add_argument(
        "--pytorch-export-sample-index",
        action="append",
        type=int,
        help="Latent row index for PyTorch export forward parity proof.",
    )
    parser.add_argument(
        "--pytorch-export-mlx-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="MLX device for generated PyTorch export forward parity proofs.",
    )
    parser.add_argument("--runtime-proof-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--runtime-proof-max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-mlx-concurrency", type=int, default=4)
    parser.add_argument("--local-cuda-concurrency", type=int, default=1)
    parser.add_argument("--local-mps-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--allow-existing-plan-dirs", action="store_true")
    return parser.parse_args(argv)


def _apply_control_profile(args: argparse.Namespace) -> None:
    if args.control_profile is None:
        return
    if args.control_profile == CONTROL_PROFILE_FULL_SOURCE_VIDEO_RUNTIME:
        # PR 95 published 8-stage curriculum: current MLX control covers the
        # source-faithful Stage 1/2/3/4/5/8 timing spine per the PR95-STAGE-4-MLX-BUILD
        # landing (`.omx/research/pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525.md`)
        # + MLX-PARADIGM-T3 commit `916c43d89` Op #3 cascade (Stage 4 = QAT
        # bridge continuing Stage 3 smooth-disagreement cosine + applying
        # smooth_disagreement_seg_loss(tau=0.3) with QAT fake-quant).
        args.stages = [1, 2, 3, 4, 5, 8]
        args.batch_size = 1
        args.synthetic_pairs = 1
        args.base_channels = 36
        args.latent_dim = 28
        args.write_pr95_public_archive_export = True
        args.prove_pr95_runtime_consumption = True
        args.write_pytorch_export_parity = True
        if args.pytorch_export_sample_index is None:
            args.pytorch_export_sample_index = [0]
        args.pytorch_export_mlx_device = "cpu"
        args.write_source_video_preprocess_smoke = True
        args.train_on_source_video_pairs = True
        args.source_video_loss_surface = PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
        if args.source_video_pair_index is None:
            args.source_video_pair_index = [0]
        args.source_video_output_hw = "384,512"
        args.local_mlx_concurrency = 1
        if args.timeout_seconds == 0:
            args.timeout_seconds = 3600
        return
    raise ExperimentQueueError(f"unsupported control profile: {args.control_profile}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_control_profile(args)
    try:
        payload = build_pr95_mlx_optimizer_matrix_queue(
            repo_root=args.repo_root,
            output_root=args.output_root,
            queue_id=args.queue_id,
            stages=args.stages,
            seeds=args.seed or [17],
            descriptor_ids=args.optimizer_descriptor_id,
            steps=args.steps,
            batch_size=args.batch_size,
            synthetic_pairs=args.synthetic_pairs,
            base_channels=args.base_channels,
            latent_dim=args.latent_dim,
            write_byte_closed_smoke=args.write_byte_closed_smoke,
            write_pr95_public_archive_export=(
                args.write_pr95_public_archive_export
                or args.prove_pr95_runtime_consumption
                or args.write_pytorch_export_parity
            ),
            prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
            write_pytorch_export_parity=args.write_pytorch_export_parity,
            pytorch_export_sample_indices=args.pytorch_export_sample_index or [0],
            pytorch_export_mlx_device=args.pytorch_export_mlx_device,
            write_source_faithful_preprocess_smoke=(
                args.write_source_faithful_preprocess_smoke
            ),
            write_source_video_preprocess_smoke=(
                args.write_source_video_preprocess_smoke
            ),
            train_on_source_video_pairs=args.train_on_source_video_pairs,
            source_video_loss_surface=args.source_video_loss_surface,
            source_preprocess_shape=args.source_preprocess_shape,
            source_preprocess_camera_hw=args.source_preprocess_camera_hw,
            source_preprocess_gradient_shape=args.source_preprocess_gradient_shape,
            source_video_path=args.source_video_path,
            source_video_upstream_dir=args.source_video_upstream_dir,
            source_video_pair_indices=args.source_video_pair_index or [0],
            source_video_output_hw=args.source_video_output_hw,
            source_video_gradient_shape=args.source_video_gradient_shape,
            runtime_proof_timeout_seconds=args.runtime_proof_timeout_seconds,
            runtime_proof_max_output_bytes=args.runtime_proof_max_output_bytes,
            local_cpu_concurrency=args.local_cpu_concurrency,
            local_mlx_concurrency=args.local_mlx_concurrency,
            local_cuda_concurrency=args.local_cuda_concurrency,
            local_mps_concurrency=args.local_mps_concurrency,
            timeout_seconds=args.timeout_seconds,
            control_profile=args.control_profile,
            allow_existing_plan_dirs=args.allow_existing_plan_dirs,
        )
        queue_artifact = write_json_artifact(args.queue_output, payload["queue"])
        manifest = {
            **payload["manifest"],
            "queue_output": _rel(args.queue_output, args.repo_root),
            "queue_output_sha256": queue_artifact.sha256,
        }
        manifest_artifact = write_json_artifact(args.manifest_output, manifest)
    except (ExperimentQueueError, ArtifactWriteError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "schema": "pr95_hnerv_mlx_optimizer_matrix_queue_summary_v1",
                "manifest": _rel(args.manifest_output, args.repo_root),
                "manifest_sha256": manifest_artifact.sha256,
                "queue": _rel(args.queue_output, args.repo_root),
                "queue_sha256": queue_artifact.sha256,
                "queue_id": payload["queue"]["queue_id"],
                "plan_count": payload["manifest"]["plan_count"],
                "refusal_count": payload["manifest"]["refusal_count"],
                **FALSE_AUTHORITY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
