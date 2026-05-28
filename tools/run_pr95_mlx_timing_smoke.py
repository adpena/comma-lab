#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run native MLX timing smokes for the public PR95 HNeRV reproduction lane."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_PR95_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    EXACT_READINESS_REFUSAL_BLOCKERS,
    FALSE_AUTHORITY,
    LANE_ID,
    PR95_MLX_LOSS_SURFACE_RGB_MSE,
    PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE,
    PR95_MLX_LOSS_SURFACES,
    PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_TIMING_ONLY,
    PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY,
    PR95_STAGE_MODULES,
    SMOKE_MANIFEST_SCHEMA,
    compare_pr95_public_archive_forward_with_pytorch,
    parse_pr95_public_archive_zip,
    pr95_default_optimizer_descriptor_id,
    pr95_mlx_optimizer_descriptor_row,
    run_pr95_mlx_synthetic_timing_smoke,
    write_pr95_mlx_byte_closed_smoke_archive,
    write_pr95_public_archive_pytorch_export_forward_parity,
)
from tac.local_acceleration.pr95_hnerv_mlx_contract import (  # noqa: E402
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
    PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
    PR95_LEGACY_TRAINING_LOOP_NOT_SOURCE_FAITHFUL_BLOCKER,
    PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER,
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER,
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
)
from tac.local_acceleration.pr95_hnerv_mlx_training import (  # noqa: E402
    SOURCE_FAITHFUL_PREPROCESS_SCHEMA,
    SOURCE_VIDEO_PREPROCESS_SCHEMA,
    load_pr95_source_pairs_nhwc,
    pr95_pair_frame_indices,
    pr95_source_pairs_to_scorer_targets_mlx,
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
SOURCE_PREPROCESS_PORTED_LOSS_UNWIRED_BLOCKER = PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER
SOURCE_VIDEO_LOADER_UNWIRED_BLOCKER = (
    "pr95_source_video_loader_ported_but_training_loop_not_source_video_backed"
)
PR95_ARCHIVE_EXPORT_NOT_RUNTIME_CONSUMED_BLOCKER = (
    "pr95_archive_export_is_byte_closed_but_not_runtime_consumed"
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _embedded_runtime_consumption_proof(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep embedded proof summaries from being parsed as extra proof records."""

    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "archive_path",
            "archive_bytes",
            "archive_sha256",
            "candidate_archive",
            "path",
            "bytes",
            "sha256",
            "proof_path",
            "proof_bytes",
            "proof_sha256",
        }
    }


def _load_public_pr95_decoder_cls(model_path: Path) -> Any:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"public PR95 source model.py not found: {model_path}")
    spec = importlib.util.spec_from_file_location(
        f"public_pr95_hnerv_model_{abs(hash(model_path.resolve()))}",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import public PR95 model.py: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.HNeRVDecoder
    except AttributeError as exc:
        raise RuntimeError(f"{model_path} does not define HNeRVDecoder") from exc


def _mark_public_archive_export_runtime_consumed(
    public_archive_export: dict[str, Any],
    *,
    proof_path: Path,
    runtime_consumption_proof: dict[str, Any],
) -> None:
    proven = runtime_consumption_proof.get("runtime_consumption_proven") is True
    public_archive_export["runtime_consumption_proof_present"] = True
    public_archive_export["runtime_consumption_proven"] = proven
    public_archive_export["runtime_consumption_proof_path"] = _rel(proof_path)
    public_archive_export["runtime_consumption_proof_bytes"] = (
        proof_path.stat().st_size
    )
    public_archive_export["runtime_consumption_proof_sha256"] = _sha256_file(
        proof_path
    )
    if not proven:
        return
    public_archive_export["receiver_contract_satisfied"] = True
    public_archive_export["receiver_proof_present"] = True
    refusal = public_archive_export.get("exact_readiness_refusal")
    if not isinstance(refusal, dict):
        return
    blockers = refusal.get("blockers")
    if not isinstance(blockers, list):
        return
    refusal["blockers"] = [
        blocker
        for blocker in blockers
        if blocker != PR95_ARCHIVE_EXPORT_NOT_RUNTIME_CONSUMED_BLOCKER
    ]


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


def _source_video_pair_indices(args: argparse.Namespace) -> list[int]:
    return list(args.source_video_pair_index or [0])


def _source_video_output_hw(args: argparse.Namespace) -> tuple[int, int]:
    parsed = _parse_int_tuple(
        args.source_video_output_hw,
        field="source-video-output-hw",
        expected_len=2,
    )
    return parsed[0], parsed[1]


def _load_source_video_training_targets_n2chw(
    args: argparse.Namespace,
    *,
    pair_indices: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    output_hw = _source_video_output_hw(args)
    if output_hw != (384, 512):
        raise ValueError(
            "--train-on-source-video-pairs requires --source-video-output-hw 384,512 "
            "because the current PR95 MLX decoder emits scorer-resolution RGB"
        )
    import mlx.core as mx

    source_pairs = load_pr95_source_pairs_nhwc(
        args.source_video_path,
        pair_indices=pair_indices,
        upstream_dir=args.source_video_upstream_dir,
    )
    source_pairs_mlx = mx.array(source_pairs)
    scorer_rgb, _yuv6 = pr95_source_pairs_to_scorer_targets_mlx(
        source_pairs_mlx,
        output_hw=output_hw,
    )
    target_n2chw = mx.transpose(scorer_rgb, (0, 1, 4, 2, 3))
    mx.eval(target_n2chw)
    return np.asarray(target_n2chw).astype(np.float32, copy=False), {
        "kind": "pr95_source_video_rgb_pairs",
        "video_path": _rel(args.source_video_path),
        "upstream_dir": _rel(args.source_video_upstream_dir),
        "pair_indices": pair_indices,
        "frame_indices": pr95_pair_frame_indices(pair_indices),
        "source_frame_pair_shape": [int(dim) for dim in source_pairs.shape],
        "target_shape_n2chw": [int(dim) for dim in target_n2chw.shape],
        "output_hw": list(output_hw),
        "training_loss_surface": args.source_video_loss_surface,
    }


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
    pytorch_export_forward_parity_proven: bool = False,
    source_faithful_preprocess_smoke: dict[str, Any] | None = None,
    source_video_preprocess_smoke: dict[str, Any] | None = None,
    source_video_training: bool = False,
    training_loss_surface: str = PR95_MLX_LOSS_SURFACE_RGB_MSE,
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
            if blocker != PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER
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
    if source_video_training:
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER,
                "synthetic_targets_do_not_establish_contest_quality",
                "pr95_hnerv_mlx_training_is_synthetic_timing_only_not_source_faithful",
                SOURCE_VIDEO_LOADER_UNWIRED_BLOCKER,
                PR95_LEGACY_TRAINING_LOOP_NOT_SOURCE_FAITHFUL_BLOCKER,
            }
        ]
        if training_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE:
            blockers = [
                blocker
                for blocker in blockers
                if blocker != SOURCE_PREPROCESS_PORTED_LOSS_UNWIRED_BLOCKER
            ]
            blockers.extend(
                [
                    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
                    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
                    "source_video_rgb_yuv6_preprocess_loss_is_not_score_authority",
                ]
            )
        else:
            blockers.extend(
                [
                    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
                    "source_video_rgb_timing_smoke_is_not_score_authority",
                ]
            )
    if runtime_consumption_proven:
        blockers.extend(
            [
                "runtime_consumption_smoke_is_not_score_authority",
                PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
            ]
        )
    if pytorch_export_forward_parity_proven:
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                PR95_EXPORT_FORWARD_PARITY_BLOCKER,
                "requires_pytorch_export_forward_parity_on_source_checkpoint",
            }
        ]
        blockers.append("pytorch_export_forward_parity_is_local_probe_not_score_authority")
    return list(dict.fromkeys(blockers))


def _apply_pytorch_export_parity_to_exact_readiness(
    payload: dict[str, Any],
    *,
    pytorch_export_forward_parity_proven: bool,
) -> None:
    """Reconcile sidecar exact-readiness blockers with archive-level export proof."""

    if not pytorch_export_forward_parity_proven:
        return
    exact = payload.get("exact_readiness_refusal")
    if not isinstance(exact, dict):
        return
    blockers = exact.get("blockers")
    if not isinstance(blockers, list):
        return
    exact["blockers"] = [
        blocker
        for blocker in blockers
        if blocker
        not in {
            PR95_EXPORT_FORWARD_PARITY_BLOCKER,
            "requires_pytorch_export_forward_parity_on_source_checkpoint",
        }
    ]
    if "pytorch_export_forward_parity_is_local_probe_not_score_authority" not in exact[
        "blockers"
    ]:
        exact["blockers"].append(
            "pytorch_export_forward_parity_is_local_probe_not_score_authority"
        )


def _without_pytorch_export_parity_blockers(blockers: Any) -> list[str]:
    if not isinstance(blockers, list):
        return []
    return [
        str(blocker)
        for blocker in blockers
        if blocker
        not in {
            PR95_EXPORT_FORWARD_PARITY_BLOCKER,
            "requires_pytorch_export_forward_parity_on_source_checkpoint",
        }
    ]


def _apply_pytorch_export_parity_to_source_faithfulness(
    manifest: dict[str, Any],
    *,
    pytorch_export_forward_parity_proven: bool,
) -> None:
    """Remove stale source-faithfulness blockers after local export parity proves."""

    if not pytorch_export_forward_parity_proven:
        return
    for payload in (
        manifest,
        manifest.get("runtime_profile"),
        manifest.get("optimizer_recipe"),
    ):
        if not isinstance(payload, dict):
            continue
        blockers = payload.get("source_faithfulness_blockers")
        if isinstance(blockers, list):
            payload["source_faithfulness_blockers"] = (
                _without_pytorch_export_parity_blockers(blockers)
            )


def _candidate_id(
    *,
    stage: int,
    seed: int,
    steps: int,
    base_channels: int,
    optimizer_descriptor_id: str,
    train_on_source_video_pairs: bool = False,
    source_video_loss_surface: str = PR95_MLX_LOSS_SURFACE_RGB_MSE,
) -> str:
    descriptor = _slug(optimizer_descriptor_id)
    candidate = (
        f"pr95_hnerv_mlx_stage{stage}_{descriptor}"
        f"_seed{seed}_steps{steps}_c{base_channels}"
    )
    if not train_on_source_video_pairs:
        return candidate
    suffix = (
        "_source_video_rgb_yuv6"
        if source_video_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
        else "_source_video_rgb"
    )
    return candidate + suffix


def _stage_module(stage: int) -> str:
    try:
        return PR95_STAGE_MODULES[int(stage)]
    except KeyError as exc:
        raise ValueError(f"unsupported PR95 stage {stage!r}") from exc


def _effective_optimizer_blockers_for_target(
    blockers: list[Any],
    *,
    train_on_source_video_pairs: bool,
    source_video_loss_surface: str,
) -> list[str]:
    effective = [str(item) for item in blockers]
    if train_on_source_video_pairs:
        effective = [
            blocker
            for blocker in effective
            if blocker != PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER
        ]
    if (
        train_on_source_video_pairs
        and source_video_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    ):
        effective = [
            blocker
            for blocker in effective
            if blocker != PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER
        ]
    return effective


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
    write_pytorch_export_parity: bool,
    write_mlx_gpu_drift_attestation: bool,
    public_pr95_source_model: Path,
    pytorch_export_sample_indices: list[int],
    pytorch_export_mlx_device: str,
    pytorch_export_atol_max: float,
    pytorch_export_atol_mean: float,
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
    if write_pytorch_export_parity:
        command.append("--write-pytorch-export-parity")
        command.extend(
            [
                "--public-pr95-source-model",
                _rel(public_pr95_source_model),
                "--pytorch-export-mlx-device",
                pytorch_export_mlx_device,
                "--pytorch-export-atol-max",
                str(pytorch_export_atol_max),
                "--pytorch-export-atol-mean",
                str(pytorch_export_atol_mean),
            ]
        )
        for sample_index in pytorch_export_sample_indices:
            command.extend(["--pytorch-export-sample-index", str(sample_index)])
    if write_mlx_gpu_drift_attestation:
        command.append("--write-mlx-gpu-drift-attestation")
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
    if train_on_source_video_pairs:
        command.append("--train-on-source-video-pairs")
        command.extend(["--source-video-loss-surface", source_video_loss_surface])
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
    write_pytorch_export_parity: bool,
    write_mlx_gpu_drift_attestation: bool,
    write_source_faithful_preprocess_smoke: bool,
    write_source_video_preprocess_smoke: bool,
    train_on_source_video_pairs: bool,
    source_video_loss_surface: str,
) -> list[dict[str, Any]]:
    postconditions: list[dict[str, Any]] = []
    if train_on_source_video_pairs:
        target_path = _rel(output_dir / "source_video_training_target.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": target_path},
                _json_equals_postcondition(
                    target_path,
                    "schema",
                    "pr95_hnerv_mlx_source_video_training_target.v1",
                ),
                _json_equals_postcondition(
                    target_path,
                    "source_video_training_ready",
                    True,
                ),
                _json_equals_postcondition(
                    target_path,
                    "target_source.kind",
                    "pr95_source_video_rgb_pairs",
                ),
                _json_equals_postcondition(
                    target_path,
                    "training_loss_surface",
                    source_video_loss_surface,
                ),
                _json_false_authority_postcondition(target_path),
            ]
        )
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
                    PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER,
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
                    PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER,
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
    if write_pytorch_export_parity:
        pt_path = _rel(output_dir / "pr95_pytorch_state_dict.pt")
        parity_path = _rel(output_dir / "pytorch_export_forward_parity.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": pt_path},
                {"type": "path_exists", "path": parity_path},
                _json_equals_postcondition(
                    parity_path,
                    "schema",
                    "pr95_hnerv_mlx_pytorch_export_forward_parity.v1",
                ),
                _json_equals_postcondition(
                    parity_path,
                    "pytorch_export_forward_parity_established",
                    True,
                ),
                _json_equals_postcondition(
                    parity_path,
                    "exact_readiness_refusal.ready",
                    False,
                ),
                _json_false_authority_postcondition(parity_path),
            ]
        )
    if write_mlx_gpu_drift_attestation:
        drift_path = _rel(output_dir / "mlx_gpu_forward_drift_attestation.json")
        postconditions.extend(
            [
                {"type": "path_exists", "path": drift_path},
                _json_equals_postcondition(
                    drift_path,
                    "schema",
                    "pr95_hnerv_public_archive_mlx_forward_parity.v1",
                ),
                _json_equals_postcondition(drift_path, "mlx_device", "gpu"),
                _json_equals_postcondition(
                    drift_path,
                    "exact_readiness_refusal.ready",
                    False,
                ),
                _json_false_authority_postcondition(drift_path),
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
    write_pytorch_export_parity: bool,
    write_mlx_gpu_drift_attestation: bool,
    public_pr95_source_model: Path,
    pytorch_export_sample_indices: list[int],
    pytorch_export_mlx_device: str,
    pytorch_export_atol_max: float,
    pytorch_export_atol_mean: float,
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
    effective_optimizer_blockers = _effective_optimizer_blockers_for_target(
        optimizer_blockers,
        train_on_source_video_pairs=train_on_source_video_pairs,
        source_video_loss_surface=source_video_loss_surface,
    )
    exact_blockers = _exact_readiness_blockers(
        source_video_training=train_on_source_video_pairs,
        training_loss_surface=source_video_loss_surface,
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
            train_on_source_video_pairs=train_on_source_video_pairs,
            source_video_loss_surface=source_video_loss_surface,
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
            "id": (
                "pr95_hnerv_mlx_source_video_rgb_yuv6_timing_smoke"
                if train_on_source_video_pairs
                and source_video_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
                else (
                    "pr95_hnerv_mlx_source_video_rgb_timing_smoke"
                    if train_on_source_video_pairs
                    else "pr95_hnerv_mlx_synthetic_stage_timing_smoke"
                )
            ),
            "source_pr": 95,
            "stage_module": stage_module,
            "steps": steps,
            "batch_size": batch_size,
            "synthetic_pairs": synthetic_pairs,
            "source_video_training_requested": train_on_source_video_pairs,
            "training_loss_surface": source_video_loss_surface,
            "target_source_kind": (
                "pr95_source_video_rgb_pairs"
                if train_on_source_video_pairs
                else "synthetic_rgb_pairs"
            ),
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
            "pytorch_export_forward_parity_requested": write_pytorch_export_parity,
            "mlx_gpu_forward_drift_attestation_requested": (
                write_mlx_gpu_drift_attestation
            ),
            "public_pr95_source_model": _rel(public_pr95_source_model),
            "pytorch_export_sample_indices": pytorch_export_sample_indices,
            "pytorch_export_mlx_device": pytorch_export_mlx_device,
            "pytorch_export_atol_max": pytorch_export_atol_max,
            "pytorch_export_atol_mean": pytorch_export_atol_mean,
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
            "train_on_source_video_pairs": train_on_source_video_pairs,
            "training_loss_surface": source_video_loss_surface,
            "target_source_kind": (
                "pr95_source_video_rgb_pairs"
                if train_on_source_video_pairs
                else "synthetic_rgb_pairs"
            ),
            "quality_comparable": False,
            "score_claim": False,
        },
        dispatch_blockers=[
            "pr95_hnerv_mlx_timing_smoke_is_proxy_signal",
            *effective_optimizer_blockers,
            *exact_blockers,
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
    source_video_pair_indices = _source_video_pair_indices(args)
    needs_pr95_archive_export = (
        args.write_pr95_public_archive_export
        or args.prove_pr95_runtime_consumption
        or args.write_pytorch_export_parity
        or args.write_mlx_gpu_drift_attestation
    )
    if args.train_on_source_video_pairs and _source_video_output_hw(args) != (384, 512):
        raise ValueError(
            "--train-on-source-video-pairs requires --source-video-output-hw 384,512"
        )
    if (
        not args.train_on_source_video_pairs
        and args.source_video_loss_surface != PR95_MLX_LOSS_SURFACE_RGB_MSE
    ):
        raise ValueError(
            "--source-video-loss-surface other than rgb_mse requires "
            "--train-on-source-video-pairs"
        )
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
    effective_optimizer_blockers = _effective_optimizer_blockers_for_target(
        optimizer_blockers,
        train_on_source_video_pairs=bool(args.train_on_source_video_pairs),
        source_video_loss_surface=args.source_video_loss_surface,
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
        if needs_pr95_archive_export
        else None,
        "archive_export_zip": _rel(output_dir / "pr95_public_archive.zip")
        if needs_pr95_archive_export
        else None,
        "runtime_consumption_proof": _rel(output_dir / "runtime_consumption_proof.json")
        if args.prove_pr95_runtime_consumption
        else None,
        "pytorch_export_state_dict": _rel(output_dir / "pr95_pytorch_state_dict.pt")
        if args.write_pytorch_export_parity
        else None,
        "pytorch_export_forward_parity": _rel(
            output_dir / "pytorch_export_forward_parity.json"
        )
        if args.write_pytorch_export_parity
        else None,
        "public_pr95_source_model": _rel(args.public_pr95_source_model)
        if args.write_pytorch_export_parity
        else None,
        "pytorch_export_sample_indices": args.pytorch_export_sample_index or [0]
        if args.write_pytorch_export_parity
        else None,
        "pytorch_export_mlx_device": args.pytorch_export_mlx_device
        if args.write_pytorch_export_parity
        else None,
        "mlx_gpu_forward_drift_attestation": _rel(
            output_dir / "mlx_gpu_forward_drift_attestation.json"
        )
        if args.write_mlx_gpu_drift_attestation
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
        if args.write_source_video_preprocess_smoke or args.train_on_source_video_pairs
        else None,
        "source_video_upstream_dir": _rel(args.source_video_upstream_dir)
        if args.write_source_video_preprocess_smoke or args.train_on_source_video_pairs
        else None,
        "source_video_pair_indices": source_video_pair_indices
        if args.write_source_video_preprocess_smoke or args.train_on_source_video_pairs
        else None,
        "source_video_output_hw": args.source_video_output_hw
        if args.write_source_video_preprocess_smoke or args.train_on_source_video_pairs
        else None,
        "source_video_gradient_shape": args.source_video_gradient_shape
        if args.write_source_video_preprocess_smoke or args.train_on_source_video_pairs
        else None,
        "source_video_training_target": _rel(
            output_dir / "source_video_training_target.json"
        )
        if args.train_on_source_video_pairs
        else None,
        "train_on_source_video_pairs": args.train_on_source_video_pairs
        if args.train_on_source_video_pairs
        else None,
        "source_video_loss_surface": args.source_video_loss_surface
        if args.train_on_source_video_pairs
        else None,
        "extra_artifact_postconditions": _extra_artifact_postconditions(
            output_dir=output_dir,
            write_pr95_public_archive_export=(
                needs_pr95_archive_export
            ),
            prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
            write_pytorch_export_parity=args.write_pytorch_export_parity,
            write_mlx_gpu_drift_attestation=args.write_mlx_gpu_drift_attestation,
            write_source_faithful_preprocess_smoke=(
                args.write_source_faithful_preprocess_smoke
            ),
            write_source_video_preprocess_smoke=(
                args.write_source_video_preprocess_smoke
            ),
            train_on_source_video_pairs=args.train_on_source_video_pairs,
            source_video_loss_surface=args.source_video_loss_surface,
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
                needs_pr95_archive_export
            ),
            prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
            write_pytorch_export_parity=args.write_pytorch_export_parity,
            write_mlx_gpu_drift_attestation=args.write_mlx_gpu_drift_attestation,
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
            source_video_pair_indices=source_video_pair_indices,
            source_video_output_hw=args.source_video_output_hw,
            source_video_gradient_shape=args.source_video_gradient_shape,
            runtime_proof_timeout_seconds=args.runtime_proof_timeout_seconds,
            runtime_proof_max_output_bytes=args.runtime_proof_max_output_bytes,
            public_pr95_source_model=args.public_pr95_source_model,
            pytorch_export_sample_indices=args.pytorch_export_sample_index or [0],
            pytorch_export_mlx_device=args.pytorch_export_mlx_device,
            pytorch_export_atol_max=args.pytorch_export_atol_max,
            pytorch_export_atol_mean=args.pytorch_export_atol_mean,
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
            needs_pr95_archive_export
        ),
        prove_pr95_runtime_consumption=args.prove_pr95_runtime_consumption,
        write_pytorch_export_parity=args.write_pytorch_export_parity,
        write_mlx_gpu_drift_attestation=args.write_mlx_gpu_drift_attestation,
        public_pr95_source_model=args.public_pr95_source_model,
        pytorch_export_sample_indices=args.pytorch_export_sample_index or [0],
        pytorch_export_mlx_device=args.pytorch_export_mlx_device,
        pytorch_export_atol_max=args.pytorch_export_atol_max,
        pytorch_export_atol_mean=args.pytorch_export_atol_mean,
        write_source_faithful_preprocess_smoke=(
            args.write_source_faithful_preprocess_smoke
        ),
        write_source_video_preprocess_smoke=args.write_source_video_preprocess_smoke,
        train_on_source_video_pairs=args.train_on_source_video_pairs,
        source_video_loss_surface=args.source_video_loss_surface,
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
    exact_blockers = _exact_readiness_blockers(
        source_video_training=bool(args.train_on_source_video_pairs),
        training_loss_surface=args.source_video_loss_surface,
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
            train_on_source_video_pairs=args.train_on_source_video_pairs,
            source_video_loss_surface=args.source_video_loss_surface,
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
            needs_pr95_archive_export
        ),
        "prove_pr95_runtime_consumption": bool(args.prove_pr95_runtime_consumption),
        "write_pytorch_export_parity": bool(args.write_pytorch_export_parity),
        "write_mlx_gpu_drift_attestation": bool(
            args.write_mlx_gpu_drift_attestation
        ),
        "public_pr95_source_model": _rel(args.public_pr95_source_model),
        "pytorch_export_sample_indices": args.pytorch_export_sample_index or [0],
        "pytorch_export_mlx_device": args.pytorch_export_mlx_device,
        "pytorch_export_atol_max": args.pytorch_export_atol_max,
        "pytorch_export_atol_mean": args.pytorch_export_atol_mean,
        "write_source_faithful_preprocess_smoke": bool(
            args.write_source_faithful_preprocess_smoke
        ),
        "source_preprocess_shape": args.source_preprocess_shape,
        "source_preprocess_camera_hw": args.source_preprocess_camera_hw,
        "source_preprocess_gradient_shape": args.source_preprocess_gradient_shape,
        "write_source_video_preprocess_smoke": bool(
            args.write_source_video_preprocess_smoke
        ),
        "train_on_source_video_pairs": bool(args.train_on_source_video_pairs),
        "source_video_loss_surface": args.source_video_loss_surface,
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
            *effective_optimizer_blockers,
            *exact_blockers,
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
    pytorch_export_forward_parity: dict[str, Any] | None,
    mlx_gpu_forward_drift_attestation: dict[str, Any] | None,
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
    pytorch_export_forward_parity_proven = bool(
        pytorch_export_forward_parity
        and pytorch_export_forward_parity.get(
            "pytorch_export_forward_parity_established"
        )
        is True
    )
    source_video_training = manifest.get("source_video_training") is True
    training_loss_surface = str(
        manifest.get("training_loss_surface") or PR95_MLX_LOSS_SURFACE_RGB_MSE
    )
    training_recipe_id = (
        "pr95_hnerv_mlx_source_video_rgb_yuv6_timing_smoke"
        if source_video_training
        and training_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
        else (
            "pr95_hnerv_mlx_source_video_rgb_timing_smoke"
            if source_video_training
            else "pr95_hnerv_mlx_synthetic_stage_timing_smoke"
        )
    )
    exact_blockers = _exact_readiness_blockers(
        runtime_consumption_proven=runtime_consumption_proven,
        pytorch_export_forward_parity_proven=(
            pytorch_export_forward_parity_proven
        ),
        source_faithful_preprocess_smoke=source_faithful_preprocess_smoke,
        source_video_preprocess_smoke=source_video_preprocess_smoke,
        source_video_training=source_video_training,
        training_loss_surface=training_loss_surface,
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
            "id": training_recipe_id,
            "source_pr": 95,
            "stage_module": stage_module,
            "steps": manifest.get("steps"),
            "batch_size": manifest.get("batch_size"),
            "synthetic_pairs": manifest.get("synthetic_pairs"),
            "training_pair_count": manifest.get("training_pair_count"),
            "source_video_training": source_video_training,
            "training_loss_surface": training_loss_surface,
            "loss_surface_weights": manifest.get("loss_surface_weights"),
            "target_yuv6_shape": manifest.get("target_yuv6_shape"),
            "yuv6_preprocess_kind": manifest.get("yuv6_preprocess_kind"),
            "target_source_kind": manifest.get("target_source_kind"),
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
            "pytorch_export_forward_parity_requested": (
                manifest.get("write_pytorch_export_parity") is True
            ),
            "pytorch_export_forward_parity_present": (
                pytorch_export_forward_parity is not None
            ),
            "pytorch_export_forward_parity_established": (
                pytorch_export_forward_parity_proven
            ),
            "mlx_gpu_forward_drift_attestation_requested": (
                manifest.get("write_mlx_gpu_drift_attestation") is True
            ),
            "mlx_gpu_forward_drift_attestation_present": (
                mlx_gpu_forward_drift_attestation is not None
            ),
            "public_pr95_source_model": manifest.get("public_pr95_source_model"),
            "pytorch_export_sample_indices": manifest.get(
                "pytorch_export_sample_indices"
            ),
            "pytorch_export_mlx_device": manifest.get("pytorch_export_mlx_device"),
            "pytorch_export_atol_max": manifest.get("pytorch_export_atol_max"),
            "pytorch_export_atol_mean": manifest.get("pytorch_export_atol_mean"),
            "source_video_training": source_video_training,
            "training_loss_surface": training_loss_surface,
            "loss_surface_weights": manifest.get("loss_surface_weights"),
            "target_yuv6_shape": manifest.get("target_yuv6_shape"),
            "yuv6_preprocess_kind": manifest.get("yuv6_preprocess_kind"),
            "target_source_kind": manifest.get("target_source_kind"),
            "target_source": manifest.get("target_source"),
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
            "pytorch_export_forward_parity": pytorch_export_forward_parity or {},
            "mlx_gpu_forward_drift_attestation": (
                mlx_gpu_forward_drift_attestation or {}
            ),
            "byte_closed_smoke_archive": archive_summary,
            "pr95_public_archive_export": public_archive_export,
            "runtime_consumption_proof": manifest.get("runtime_consumption_proof", {}),
            "source_video_training_target": manifest.get(
                "source_video_training_target",
                {},
            ),
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
        help="PR95 stage to time-smoke.",
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
    parser.add_argument(
        "--write-pytorch-export-parity",
        action="store_true",
        help=(
            "Export the PR95 archive state dict as .pt and prove PyTorch/MLX "
            "forward parity on selected latent rows. Local implementation "
            "evidence only."
        ),
    )
    parser.add_argument(
        "--write-mlx-gpu-drift-attestation",
        action="store_true",
        help=(
            "Emit a separate MLX GPU-vs-PyTorch forward-drift attestation for "
            "Metal training calibration. This is not the PyTorch export parity "
            "gate and never grants score or promotion authority."
        ),
    )
    parser.add_argument(
        "--public-pr95-source-model",
        type=Path,
        default=DEFAULT_PUBLIC_PR95_SOURCE_MODEL,
        help="Path to public PR95 source model.py defining HNeRVDecoder.",
    )
    parser.add_argument(
        "--pytorch-export-sample-index",
        action="append",
        type=int,
        help="Latent row index to include in PyTorch export forward parity proof.",
    )
    parser.add_argument(
        "--pytorch-export-mlx-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="MLX device for PyTorch export forward parity proof.",
    )
    parser.add_argument("--pytorch-export-atol-max", type=float, default=2e-3)
    parser.add_argument("--pytorch-export-atol-mean", type=float, default=1e-4)
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
        "--train-on-source-video-pairs",
        action="store_true",
        help=(
            "Use decoded PR95 source-video RGB pairs as the timing-smoke "
            "training targets. This is still local MLX research signal, not "
            "score authority."
        ),
    )
    parser.add_argument(
        "--source-video-loss-surface",
        choices=PR95_MLX_LOSS_SURFACES,
        default=PR95_MLX_LOSS_SURFACE_RGB_MSE,
        help=(
            "Loss surface for --train-on-source-video-pairs. rgb_yuv6_mse "
            "adds native MLX scorer-preprocess YUV6 MSE to RGB MSE but is "
            "still not full scorer-loss authority."
        ),
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
    needs_pr95_archive_export = (
        args.write_pr95_public_archive_export
        or args.prove_pr95_runtime_consumption
        or args.write_pytorch_export_parity
        or args.write_mlx_gpu_drift_attestation
    )
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

    source_video_pair_indices = _source_video_pair_indices(args)
    target_pairs_n2chw = None
    target_source = None
    source_video_training_target: dict[str, Any] | None = None
    if args.train_on_source_video_pairs:
        target_pairs_n2chw, target_source = _load_source_video_training_targets_n2chw(
            args,
            pair_indices=source_video_pair_indices,
        )
        source_video_training_target = {
            "schema": "pr95_hnerv_mlx_source_video_training_target.v1",
            "lane_id": LANE_ID,
            "source_video_training_ready": True,
            "training_fidelity": (
                PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY
                if args.source_video_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
                else PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_TIMING_ONLY
            ),
            "training_loss_surface": args.source_video_loss_surface,
            "target_source": target_source,
            "target_shape_n2chw": [int(dim) for dim in target_pairs_n2chw.shape],
            "exact_readiness_refusal": {
                "ready": False,
                "blockers": (
                    [
                        PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
                        PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
                        "source_video_rgb_yuv6_preprocess_loss_is_not_score_authority",
                        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
                    ]
                    if args.source_video_loss_surface
                    == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
                    else [
                        PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
                        "source_video_rgb_timing_smoke_is_not_score_authority",
                        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
                    ]
                ),
            },
            **FALSE_AUTHORITY,
        }
        _write_json(
            output_dir / "source_video_training_target.json",
            source_video_training_target,
        )

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
            if needs_pr95_archive_export
            else None
        ),
        target_pairs_n2chw=target_pairs_n2chw,
        target_source=target_source,
        training_loss_surface=args.source_video_loss_surface,
    )
    validate_runtime_profile_observation(manifest["runtime_profile"])
    manifest["write_pytorch_export_parity"] = bool(args.write_pytorch_export_parity)
    manifest["write_mlx_gpu_drift_attestation"] = bool(
        args.write_mlx_gpu_drift_attestation
    )
    manifest["public_pr95_source_model"] = _rel(args.public_pr95_source_model)
    manifest["pytorch_export_sample_indices"] = args.pytorch_export_sample_index or [0]
    manifest["pytorch_export_mlx_device"] = args.pytorch_export_mlx_device
    manifest["pytorch_export_atol_max"] = args.pytorch_export_atol_max
    manifest["pytorch_export_atol_mean"] = args.pytorch_export_atol_mean
    archive_summary = (
        write_pr95_mlx_byte_closed_smoke_archive(manifest, output_dir=output_dir)
        if args.write_byte_closed_smoke
        else None
    )
    if archive_summary is not None:
        manifest["byte_closed_smoke_archive"] = archive_summary
    if source_video_training_target is not None:
        manifest["source_video_training_target"] = source_video_training_target
        manifest["source_video_training_target_path"] = _rel(
            output_dir / "source_video_training_target.json"
        )
    public_archive_export = (
        manifest["pr95_public_archive_export"]
        if isinstance(manifest.get("pr95_public_archive_export"), dict)
        else None
    )
    if public_archive_export is not None:
        manifest["candidate_archive"] = public_archive_export["candidate_archive"]
        manifest["archive_path"] = public_archive_export["archive_path"]
        manifest["archive_bytes"] = public_archive_export["archive_bytes"]
        manifest["archive_sha256"] = public_archive_export["archive_sha256"]
        _write_json(output_dir / "pr95_public_archive_export.json", public_archive_export)

    torch_decoder_cls: Any | None = None
    packet = None
    if args.write_pytorch_export_parity or args.write_mlx_gpu_drift_attestation:
        if public_archive_export is None:
            raise SystemExit(
                "PR95 archive export is required for export parity or MLX GPU drift "
                "attestation"
            )
        torch_decoder_cls = _load_public_pr95_decoder_cls(args.public_pr95_source_model)
        packet = parse_pr95_public_archive_zip(output_dir / "pr95_public_archive.zip")

    pytorch_export_forward_parity: dict[str, Any] | None = None
    if args.write_pytorch_export_parity:
        assert torch_decoder_cls is not None
        assert packet is not None
        pytorch_export_forward_parity = (
            write_pr95_public_archive_pytorch_export_forward_parity(
                packet,
                torch_decoder_cls,
                output_pt_path=output_dir / "pr95_pytorch_state_dict.pt",
                run_id=str(manifest["candidate_id"]),
                sample_indices=args.pytorch_export_sample_index or [0],
                mlx_device=args.pytorch_export_mlx_device,
                atol_max=args.pytorch_export_atol_max,
                atol_mean=args.pytorch_export_atol_mean,
                overwrite=True,
            )
        )
        manifest["pytorch_export_forward_parity"] = pytorch_export_forward_parity
        manifest["pytorch_export_forward_parity_path"] = _rel(
            output_dir / "pytorch_export_forward_parity.json"
        )
        manifest["pytorch_export_state_dict_pt_path"] = _rel(
            output_dir / "pr95_pytorch_state_dict.pt"
        )
        manifest["pytorch_export_parity"] = {
            **manifest["pytorch_export_parity"],
            "forward_parity_on_source_checkpoint": (
                pytorch_export_forward_parity[
                    "pytorch_export_forward_parity_established"
                ]
            ),
            "pytorch_export_forward_parity_path": _rel(
                output_dir / "pytorch_export_forward_parity.json"
            ),
            "pytorch_export_state_dict_pt_path": _rel(
                output_dir / "pr95_pytorch_state_dict.pt"
            ),
            "pt_sha256": pytorch_export_forward_parity["pt_sha256"],
        }
        if pytorch_export_forward_parity[
            "pytorch_export_forward_parity_established"
        ]:
            manifest["pytorch_export_parity"].pop("blocker", None)
            manifest["pytorch_export_parity"]["authority_note"] = (
                "local_mlx_pytorch_export_parity_probe_is_not_contest_auth_eval"
            )
        _apply_pytorch_export_parity_to_source_faithfulness(
            manifest,
            pytorch_export_forward_parity_proven=(
                pytorch_export_forward_parity[
                    "pytorch_export_forward_parity_established"
                ]
            ),
        )
        _write_json(
            output_dir / "pytorch_export_forward_parity.json",
            pytorch_export_forward_parity,
        )

    mlx_gpu_forward_drift_attestation: dict[str, Any] | None = None
    if args.write_mlx_gpu_drift_attestation:
        assert torch_decoder_cls is not None
        assert packet is not None
        mlx_gpu_forward_drift_attestation = (
            compare_pr95_public_archive_forward_with_pytorch(
                packet,
                torch_decoder_cls,
                sample_indices=args.pytorch_export_sample_index or [0],
                mlx_device="gpu",
                atol_max=args.pytorch_export_atol_max,
                atol_mean=args.pytorch_export_atol_mean,
            )
        )
        manifest["mlx_gpu_forward_drift_attestation"] = (
            mlx_gpu_forward_drift_attestation
        )
        manifest["mlx_gpu_forward_drift_attestation_path"] = _rel(
            output_dir / "mlx_gpu_forward_drift_attestation.json"
        )
        _write_json(
            output_dir / "mlx_gpu_forward_drift_attestation.json",
            mlx_gpu_forward_drift_attestation,
        )

    runtime_consumption_proof: dict[str, Any] | None = None
    if args.prove_pr95_runtime_consumption:
        if public_archive_export is None:
            raise SystemExit("--prove-pr95-runtime-consumption requires PR95 archive export")
        proof_path = output_dir / "runtime_consumption_proof.json"
        proof_command = [
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
        ]
        if args.allow_existing_output_dir and proof_path.exists():
            proof_command.extend(
                [
                    "--allow-overwrite",
                    "--expected-existing-sha256",
                    _sha256_file(proof_path),
                ]
            )
        proof_result = subprocess.run(
            proof_command,
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
        manifest["runtime_consumption_proof"] = _embedded_runtime_consumption_proof(
            runtime_consumption_proof
        )
        manifest["runtime_consumption_proof_path"] = _rel(proof_path)
        manifest["runtime_consumption_proof_bytes"] = proof_path.stat().st_size
        manifest["runtime_consumption_proof_sha256"] = _sha256_file(proof_path)
        if runtime_consumption_proof.get("runtime_consumption_proven") is True:
            manifest["receiver_contract_satisfied"] = True
            manifest["receiver_proof_present"] = True
        _mark_public_archive_export_runtime_consumed(
            public_archive_export,
            proof_path=proof_path,
            runtime_consumption_proof=runtime_consumption_proof,
        )
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
        _apply_pytorch_export_parity_to_exact_readiness(
            source_faithful_preprocess_smoke,
            pytorch_export_forward_parity_proven=(
                pytorch_export_forward_parity is not None
                and pytorch_export_forward_parity.get(
                    "pytorch_export_forward_parity_established"
                )
                is True
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
        _apply_pytorch_export_parity_to_exact_readiness(
            source_video_preprocess_smoke,
            pytorch_export_forward_parity_proven=(
                pytorch_export_forward_parity is not None
                and pytorch_export_forward_parity.get(
                    "pytorch_export_forward_parity_established"
                )
                is True
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
        pytorch_export_forward_parity_proven=(
            pytorch_export_forward_parity is not None
            and pytorch_export_forward_parity.get(
                "pytorch_export_forward_parity_established"
            )
            is True
        ),
        source_faithful_preprocess_smoke=source_faithful_preprocess_smoke,
        source_video_preprocess_smoke=source_video_preprocess_smoke,
        source_video_training=manifest.get("source_video_training") is True,
        training_loss_surface=str(
            manifest.get("training_loss_surface") or PR95_MLX_LOSS_SURFACE_RGB_MSE
        ),
    )

    _write_json(output_dir / "runtime_profile.json", manifest["runtime_profile"])
    _write_json(output_dir / "manifest.json", manifest)
    representation = _representation_manifest(
        manifest=manifest,
        output_dir=output_dir,
        archive_summary=archive_summary,
        runtime_consumption_proof=runtime_consumption_proof,
        pytorch_export_forward_parity=pytorch_export_forward_parity,
        mlx_gpu_forward_drift_attestation=mlx_gpu_forward_drift_attestation,
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
        "pytorch_export_forward_parity": pytorch_export_forward_parity,
        "mlx_gpu_forward_drift_attestation": mlx_gpu_forward_drift_attestation,
        "source_faithful_preprocess_smoke": source_faithful_preprocess_smoke,
        "source_video_preprocess_smoke": source_video_preprocess_smoke,
        "source_video_training": manifest.get("source_video_training"),
        "source_video_training_target": source_video_training_target,
        "target_source": manifest.get("target_source"),
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
