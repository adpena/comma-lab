#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the HPRC compact receiver train/export adapter on low-res RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact
from tac.substrates._shared.trainer_skeleton import decode_real_pairs
from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY,
    HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
)
from tac.substrates.hprc.training_adapter import (
    HPRC_LONG_TRAINING_SUBSTRATE_ID,
    HprcCompactReceiverLongTrainingAdapter,
)
from tac.training.long_training_canonical import (
    CurriculumStage,
    LongTrainingConfig,
    run_long_training,
)

HPRC_LONG_TRAINING_STORAGE_PLAN_SCHEMA = "hprc_compact_receiver_training_storage_plan.v1"
HPRC_LONG_TRAINING_RESULT_SCHEMA = "hprc_compact_receiver_training_run_result.v1"
DEFAULT_HPRC_LONG_TRAINING_WORKLOAD_SUBDIR = "experiments/results/hprc_long_training"
DEFAULT_HPRC_LONG_TRAINING_EXPECTED_BYTES = HPRC_RECEIVER_PROOF_SCRATCH_BYTES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-npy", type=Path)
    source.add_argument("--video-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--decode-pairs", type=int, default=8)
    parser.add_argument("--decode-max-pairs", type=int)
    parser.add_argument("--decode-height", type=int, default=96)
    parser.add_argument("--decode-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-pair-indices-per-step", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--curriculum-preset",
        choices=(
            "single_stage",
            "hprc_native_rate_ramp_v1",
            "hprc_pr95_pose_guard_rate_v1",
        ),
        default="single_stage",
        help=(
            "Training schedule. hprc_native_rate_ramp_v1 adds residual-recon "
            "warmup, protected rate ramp, and low-LR polish before archive export. "
            "hprc_pr95_pose_guard_rate_v1 is the 8-stage PR95-style scaffold: "
            "fit, protected scorer-surface repair, native rate ramp, and final "
            "byte-polish while preserving P18/P19 protected residual cells."
        ),
    )
    parser.add_argument("--basis-count", type=int, default=3)
    parser.add_argument("--residual-grid-h", type=int, default=24)
    parser.add_argument("--residual-grid-w", type=int, default=32)
    parser.add_argument(
        "--enable-protected-residual-pathway",
        action="store_true",
        help=(
            "Store an explicit high-resolution protected residual sidecar in "
            "RESIDUAL_RC v2 so PoseNet/SegNet-sensitive geometry does not have "
            "to be represented by the coarse interior residual grid."
        ),
    )
    parser.add_argument(
        "--protected-residual-grid-h",
        type=int,
        help="Grid height for the protected residual sidecar; defaults to residual-grid-h.",
    )
    parser.add_argument(
        "--protected-residual-grid-w",
        type=int,
        help="Grid width for the protected residual sidecar; defaults to residual-grid-w.",
    )
    parser.add_argument(
        "--protected-residual-mask-threshold",
        type=float,
        help=(
            "When a P18/P19 residual-protection surface is present, emit "
            "high-res protected residual cells only where protection weight is "
            "at least this value. The dense surface still guides the loss."
        ),
    )
    parser.add_argument(
        "--protected-residual-mask-top-fraction",
        type=float,
        help=(
            "When a P18/P19 residual-protection surface is present, emit only "
            "the highest-priority fraction of protected residual cells. Ties "
            "are broken by a deterministic hash of cell index."
        ),
    )
    parser.add_argument(
        "--training-backend",
        choices=("auto", "mlx", "numpy"),
        default="auto",
        help=(
            "Train with MLX/Metal when available, but always export a numpy-only "
            "receiver archive for contest CPU/T4 auth eval."
        ),
    )
    parser.add_argument("--initial-latent-gain", type=float, default=1.0)
    parser.add_argument("--initial-residual-gain", type=float, default=1.0)
    parser.add_argument("--initial-receiver-state-gain", type=float, default=0.25)
    parser.add_argument(
        "--native-rate-aware",
        action="store_true",
        help="Optimize HPRC residual tokens with a train-time rate proxy before archive export.",
    )
    parser.add_argument("--rate-aware-residual-l1-weight", type=float, default=0.0)
    parser.add_argument("--rate-aware-residual-prox-weight", type=float, default=0.0)
    parser.add_argument(
        "--rate-aware-residual-protection-npy",
        type=Path,
        help=(
            "Optional residual-token protection surface. Shape must broadcast to "
            "frames x residual_grid_h x residual_grid_w x 3; 1 protects, 0 shrinks."
        ),
    )
    parser.add_argument(
        "--skip-runtime-consumption-proof",
        action="store_true",
        help="Emit archive bytes without running generated inflate receiver proof.",
    )
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path storage tier override")
    parser.add_argument(
        "--storage-workload-subdir",
        default=DEFAULT_HPRC_LONG_TRAINING_WORKLOAD_SUBDIR,
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=DEFAULT_HPRC_LONG_TRAINING_EXPECTED_BYTES,
    )
    parser.add_argument(
        "--allow-local-output-dir",
        action="store_true",
        help="Allow local-disk fallback only by explicit opt-in.",
    )
    parser.add_argument(
        "--storage-plan-path",
        type=Path,
        help="Optional external storage-waterfall plan path to preserve in the result.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    output_dir, storage_plan_path = _resolve_output_dir(args, repo_root=repo_root)
    frames, source_manifest = _load_source_frames(args, repo_root=repo_root)
    residual_protection, protection_manifest = _load_residual_protection(
        args,
        repo_root=repo_root,
        expected_residual_shape=(
            int(frames.shape[0]),
            int(args.residual_grid_h),
            int(args.residual_grid_w),
            3,
        ),
    )
    protected_residual_mask, protected_residual_mask_manifest = _build_protected_residual_mask(
        residual_protection,
        args=args,
    )
    external_storage_plan = _resolve_optional_path(args.storage_plan_path, repo_root=repo_root)
    storage_plan_for_result = storage_plan_path or external_storage_plan
    adapter = HprcCompactReceiverLongTrainingAdapter(
        frames,
        basis_count=int(args.basis_count),
        residual_grid_h=int(args.residual_grid_h),
        residual_grid_w=int(args.residual_grid_w),
        source_manifest=source_manifest,
        initial_latent_gain=float(args.initial_latent_gain),
        initial_residual_gain=float(args.initial_residual_gain),
        initial_receiver_state_gain=float(args.initial_receiver_state_gain),
        repo_root=repo_root,
        retain_receiver_proof_output=bool(args.retain_receiver_output),
        emit_archive_bound_candidate_package=not bool(args.skip_runtime_consumption_proof),
        native_rate_aware=bool(args.native_rate_aware),
        rate_aware_residual_l1_weight=float(args.rate_aware_residual_l1_weight),
        rate_aware_residual_prox_weight=float(args.rate_aware_residual_prox_weight),
        residual_protection=residual_protection,
        protected_residual_mask=protected_residual_mask,
        enable_protected_residual_pathway=bool(args.enable_protected_residual_pathway),
        protected_residual_grid_h=args.protected_residual_grid_h,
        protected_residual_grid_w=args.protected_residual_grid_w,
        training_backend=str(args.training_backend),
    )
    curriculum_stages = _build_curriculum_stages(args)
    config = LongTrainingConfig(
        substrate_id=HPRC_LONG_TRAINING_SUBSTRATE_ID,
        lane_id="lane_hprc_compact_receiver_training",
        epochs=int(args.epochs),
        batch_pair_indices_per_step=int(args.batch_pair_indices_per_step),
        curriculum_stages=curriculum_stages,
        checkpoint_interval_epochs=max(1, min(int(args.epochs), 10)),
        early_stopping_patience=max(2, int(args.epochs) * 2),
        learning_rate=float(args.learning_rate),
        output_dir=output_dir,
        notes="HPRC compact receiver train/export adapter run with storage-waterfall custody.",
    )
    artifact = run_long_training(adapter, config)
    result = {
        "schema": HPRC_LONG_TRAINING_RESULT_SCHEMA,
        "run_id": output_dir.name,
        "storage_plan_path": None
        if storage_plan_for_result is None
        else storage_plan_for_result.as_posix(),
        "source_manifest": source_manifest,
        "residual_protection_manifest": protection_manifest,
        "protected_residual_mask_manifest": protected_residual_mask_manifest,
        "training_backend": {
            "requested": str(args.training_backend),
            "effective": adapter.effective_training_backend,
            "portable_runtime": "numpy",
            "contest_runtime_requires_mlx": False,
            "contest_runtime_requires_torch": False,
        },
        "curriculum_preset": str(args.curriculum_preset),
        "curriculum_stages": [stage.as_dict() for stage in curriculum_stages],
        "protected_highres_residual_pathway": adapter.artifact_metadata()[
            "protected_highres_residual_pathway"
        ],
        "artifact": artifact.as_dict(),
        "runtime_consumption_proof_requested": not bool(args.skip_runtime_consumption_proof),
        "exact_axis_blocker": "contest_cpu_cuda_exact_eval_not_executed",
        **FALSE_AUTHORITY,
    }
    result_path = (
        output_dir / "hprc_compact_receiver_training_run_result.json"
        if args.output_manifest is None
        else _resolve_optional_path(args.output_manifest, repo_root=repo_root)
    )
    if result_path is None:
        raise ValueError("failed to resolve output manifest")
    _write_json_maybe_overwrite(result_path, result)
    print(json.dumps({**result, "result_path": result_path.as_posix()}, sort_keys=True))
    return 0


def _build_curriculum_stages(args: argparse.Namespace) -> tuple[CurriculumStage, ...]:
    epochs = int(args.epochs)
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")
    final_l1 = float(args.rate_aware_residual_l1_weight) if bool(args.native_rate_aware) else 0.0
    final_prox = float(args.rate_aware_residual_prox_weight) if bool(args.native_rate_aware) else 0.0
    if str(args.curriculum_preset) == "single_stage":
        weights = _hprc_loss_weights(
            residual_l1=final_l1,
            residual_prox=final_prox,
            residual_recon_update=1.0 if bool(args.native_rate_aware) else 0.0,
        )
        return (
            CurriculumStage(
                name="compact_receiver_gain_fit",
                start_epoch=0,
                end_epoch=epochs,
                loss_weights=weights,
                notes=(
                    "Fit HPRC compact receiver decode RDO gains and, when enabled, "
                    "native residual-token rate pressure before archive export."
                ),
            ),
        )
    if str(args.curriculum_preset) == "hprc_pr95_pose_guard_rate_v1":
        return _build_pr95_pose_guard_rate_curriculum(
            epochs,
            final_l1=final_l1,
            final_prox=final_prox,
        )
    if str(args.curriculum_preset) != "hprc_native_rate_ramp_v1":
        raise ValueError(f"unknown curriculum preset: {args.curriculum_preset!r}")

    stages = [
        (
            "rdo_gain_and_residual_recon_warmup",
            0.20,
            1.0,
            _hprc_loss_weights(
                residual_l1=0.0,
                residual_prox=0.0,
                residual_recon_update=1.0,
                gain_l2=1e-4,
            ),
            "Improve residual fidelity before asking the archive-rate proxy to shrink tokens.",
        ),
        (
            "protected_residual_recon_fit",
            0.30,
            0.8,
            _hprc_loss_weights(
                residual_l1=0.10 * final_l1,
                residual_prox=0.10 * final_prox,
                residual_recon_update=1.0,
                gain_l2=1e-4,
            ),
            "Start gentle rate pressure while preserving P18/P19-protected cells.",
        ),
        (
            "native_rate_ramp",
            0.30,
            0.5,
            _hprc_loss_weights(
                residual_l1=0.60 * final_l1,
                residual_prox=0.60 * final_prox,
                residual_recon_update=0.75,
                gain_l2=1e-4,
            ),
            "Move toward compact residual tokens under the native HPRC rate proxy.",
        ),
        (
            "byte_closed_polish",
            0.20,
            0.25,
            _hprc_loss_weights(
                residual_l1=final_l1,
                residual_prox=final_prox,
                residual_recon_update=0.35,
                gain_l2=1e-4,
            ),
            "Low-LR polish at the final archive-rate operating point.",
        ),
    ]
    spans = _epoch_spans(epochs, [stage[1] for stage in stages])
    return tuple(
        CurriculumStage(
            name=name,
            start_epoch=start,
            end_epoch=end,
            loss_weights=weights,
            lr_scale=lr_scale,
            notes=notes,
        )
        for (name, _fraction, lr_scale, weights, notes), (start, end) in zip(
            stages,
            spans,
            strict=True,
        )
        if end > start
    )


def _build_pr95_pose_guard_rate_curriculum(
    epochs: int,
    *,
    final_l1: float,
    final_prox: float,
) -> tuple[CurriculumStage, ...]:
    stages = [
        (
            "rdo_gain_anchor",
            0.08,
            0.80,
            _hprc_loss_weights(
                residual_l1=0.0,
                residual_prox=0.0,
                residual_recon_update=0.0,
                gain_l2=1e-4,
            ),
            "Anchor compact receiver gains before mutating residual tokens.",
        ),
        (
            "residual_warm_start",
            0.12,
            1.00,
            _hprc_loss_weights(
                residual_l1=0.0,
                residual_prox=0.0,
                residual_recon_update=0.45,
                score_protection_recon=1.0,
                gain_l2=1e-4,
            ),
            "Start residual fitting with scorer-protected cells weighted above interiors.",
        ),
        (
            "pose_guard_fit",
            0.15,
            1.00,
            _hprc_loss_weights(
                residual_l1=0.0,
                residual_prox=0.0,
                residual_recon_update=1.0,
                score_protection_recon=3.0,
                gain_l2=1e-4,
            ),
            "Spend training capacity on P18/P19 protected cells before rate pressure.",
        ),
        (
            "gentle_native_rate_probe",
            0.12,
            0.80,
            _hprc_loss_weights(
                residual_l1=0.05 * final_l1,
                residual_prox=0.05 * final_prox,
                residual_recon_update=0.85,
                score_protection_recon=3.0,
                gain_l2=1e-4,
            ),
            "Introduce a small rate proxy while measuring whether protected cells move.",
        ),
        (
            "native_rate_ramp",
            0.18,
            0.60,
            _hprc_loss_weights(
                residual_l1=0.35 * final_l1,
                residual_prox=0.35 * final_prox,
                residual_recon_update=0.65,
                score_protection_recon=2.5,
                gain_l2=1e-4,
            ),
            "Move toward compact tokens after the protected residual pathway is fit.",
        ),
        (
            "protected_token_compaction",
            0.15,
            0.45,
            _hprc_loss_weights(
                residual_l1=0.70 * final_l1,
                residual_prox=0.70 * final_prox,
                residual_recon_update=0.45,
                score_protection_recon=2.0,
                gain_l2=1e-4,
            ),
            "Apply strong pressure to unprotected cells while retaining scorer-critical ones.",
        ),
        (
            "byte_closed_polish",
            0.12,
            0.25,
            _hprc_loss_weights(
                residual_l1=final_l1,
                residual_prox=final_prox,
                residual_recon_update=0.30,
                score_protection_recon=1.5,
                gain_l2=1e-4,
            ),
            "Polish at the final archive-rate point before byte-closed export.",
        ),
        (
            "pose_guard_repair_polish",
            0.08,
            0.15,
            _hprc_loss_weights(
                residual_l1=0.25 * final_l1,
                residual_prox=0.25 * final_prox,
                residual_recon_update=0.80,
                score_protection_recon=4.0,
                gain_l2=1e-4,
            ),
            "Final protected repair pass to avoid the high-rate-low-pose failure mode.",
        ),
    ]
    spans = _epoch_spans(epochs, [stage[1] for stage in stages])
    return tuple(
        CurriculumStage(
            name=name,
            start_epoch=start,
            end_epoch=end,
            loss_weights=weights,
            lr_scale=lr_scale,
            notes=notes,
        )
        for (name, _fraction, lr_scale, weights, notes), (start, end) in zip(
            stages,
            spans,
            strict=True,
        )
        if end > start
    )


def _hprc_loss_weights(
    *,
    residual_l1: float,
    residual_prox: float,
    residual_recon_update: float,
    score_protection_recon: float = 0.0,
    gain_l2: float = 0.0,
) -> dict[str, float]:
    weights = {
        "recon": 1.0,
        "residual_recon_update": max(0.0, float(residual_recon_update)),
        "residual_rate_l1": max(0.0, float(residual_l1)),
        "residual_rate_prox": max(0.0, float(residual_prox)),
    }
    if score_protection_recon > 0.0:
        weights["score_protection_recon"] = float(score_protection_recon)
    if gain_l2 > 0.0:
        weights["gain_l2"] = float(gain_l2)
    return weights


def _epoch_spans(epochs: int, fractions: list[float]) -> list[tuple[int, int]]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if len(fractions) < 1:
        raise ValueError("fractions must be non-empty")
    active = min(epochs, len(fractions))
    selected = fractions[:active]
    total = sum(float(v) for v in selected)
    raw = [float(v) / total * epochs for v in selected]
    lengths = [max(1, int(v)) for v in raw]
    while sum(lengths) > epochs:
        index = max(range(len(lengths)), key=lambda i: lengths[i])
        if lengths[index] == 1:
            break
        lengths[index] -= 1
    while sum(lengths) < epochs:
        deficits = [raw[i] - lengths[i] for i in range(len(lengths))]
        index = max(range(len(lengths)), key=lambda i: deficits[i])
        lengths[index] += 1
    spans: list[tuple[int, int]] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        spans.append((start, end))
        start = end
    return spans


def _load_source_frames(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    if args.frames_npy is not None:
        frames_path = Path(args.frames_npy).expanduser().resolve(strict=False)
        frames = np.load(frames_path, mmap_mode="r")
        return frames, {
            "schema": "hprc_compact_receiver_training_source_frames.v1",
            "source_kind": "frames_npy",
            "frames_npy_path": frames_path.as_posix(),
            "frames_npy_bytes": frames_path.stat().st_size,
            "frames_npy_sha256": sha256_file(frames_path),
            "frames_shape": [int(v) for v in frames.shape],
            "frames_dtype": str(frames.dtype),
            "score_claim": False,
            "promotion_eligible": False,
        }

    if int(args.decode_pairs) < 1:
        raise ValueError("--decode-pairs must be >= 1")
    if int(args.decode_height) < 1 or int(args.decode_width) < 1:
        raise ValueError("--decode-height and --decode-width must be >= 1")
    video_path = Path(args.video_path).expanduser()
    if not video_path.is_absolute():
        video_path = repo_root / video_path
    video_path = video_path.resolve(strict=False)
    pairs = decode_real_pairs(
        video_path,
        n_pairs=int(args.decode_pairs),
        max_pairs=args.decode_max_pairs,
        substrate_tag="hprc_compact_receiver",
        repo_root=repo_root,
    )
    import torch.nn.functional as F

    if pairs.ndim != 5 or int(pairs.shape[1]) != 2 or int(pairs.shape[2]) != 3:
        raise ValueError("decode_real_pairs returned an unexpected tensor shape")
    flat = pairs.reshape((-1, int(pairs.shape[2]), int(pairs.shape[3]), int(pairs.shape[4])))
    if (int(flat.shape[2]), int(flat.shape[3])) != (
        int(args.decode_height),
        int(args.decode_width),
    ):
        flat = F.interpolate(
            flat.float(),
            size=(int(args.decode_height), int(args.decode_width)),
            mode="bilinear",
            align_corners=False,
        )
    frames = flat.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
    frames_sha = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
    return frames, {
        "schema": "hprc_compact_receiver_training_source_frames.v1",
        "source_kind": "contest_video_decode",
        "video_path": video_path.as_posix(),
        "video_bytes": video_path.stat().st_size,
        "video_sha256": sha256_file(video_path),
        "decode_pairs_requested": int(args.decode_pairs),
        "decode_max_pairs": None if args.decode_max_pairs is None else int(args.decode_max_pairs),
        "decoded_pairs": int(pairs.shape[0]),
        "decoded_frame_count": int(frames.shape[0]),
        "decoded_source_shape": [int(v) for v in pairs.shape],
        "frames_shape": [int(v) for v in frames.shape],
        "frames_dtype": str(frames.dtype),
        "frames_sha256": frames_sha,
        "resize_mode": "bilinear_align_corners_false",
        "score_claim": False,
        "promotion_eligible": False,
    }


def _load_residual_protection(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    expected_residual_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if args.rate_aware_residual_protection_npy is None:
        return None, None
    path = Path(args.rate_aware_residual_protection_npy).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise ValueError(f"rate-aware residual protection missing: {path}")
    arr = np.load(path, mmap_mode="r")
    projected, projection_manifest = _project_residual_protection_to_expected_shape(
        arr,
        expected_residual_shape=expected_residual_shape,
        path=path,
    )
    return projected, {
        "schema": "hprc_native_rate_residual_protection_input.v1",
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": [int(v) for v in projected.shape],
        "source_shape": [int(v) for v in arr.shape],
        "dtype": str(projected.dtype),
        "semantics": "1=protect_from_rate_pressure,0=safest_to_shrink",
        **projection_manifest,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _project_residual_protection_to_expected_shape(
    arr: np.ndarray,
    *,
    expected_residual_shape: tuple[int, int, int, int],
    path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    frames, grid_h, grid_w, channels = expected_residual_shape
    shape = tuple(int(v) for v in arr.shape)
    allowed = {
        (frames, grid_h, grid_w, channels),
        (frames, grid_h, grid_w),
        (frames, 1, 1, 1),
        (frames, 1, 1),
        (1, grid_h, grid_w, channels),
        (1, grid_h, grid_w),
    }
    if shape in allowed:
        return np.asarray(arr, dtype=np.float32), {
            "prefix_projected_from_full_video_surface": False,
            "projection_kind": "shape_exact_or_broadcastable",
        }
    if (
        len(shape) in {3, 4}
        and shape[0] >= frames
        and shape[1] == grid_h
        and shape[2] == grid_w
    ):
        if len(shape) == 4 and shape[3] not in {1, channels}:
            raise ValueError(
                "rate-aware residual protection channel mismatch: "
                f"{path} has shape {shape}, expected channel count 1 or {channels}"
            )
        return np.asarray(arr[:frames], dtype=np.float32), {
            "prefix_projected_from_full_video_surface": True,
            "projection_kind": "first_n_frames_prefix",
            "projected_frame_count": int(frames),
            "source_frame_count": int(shape[0]),
        }
    raise ValueError(
        "rate-aware residual protection shape mismatch: "
        f"{path} has shape {tuple(shape)}, but decode/residual config expects one of "
        f"{sorted(allowed)} or a longer full-video prefix-compatible surface with "
        f"shape (N>={frames}, {grid_h}, {grid_w}[, {channels}]). Match "
        "--decode-pairs/--decode-max-pairs and --residual-grid-h/--residual-grid-w, "
        "or rebuild the P18/P19 protection surface."
    )


def _build_protected_residual_mask(
    residual_protection: np.ndarray | None,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not bool(args.enable_protected_residual_pathway):
        return None, None
    threshold = args.protected_residual_mask_threshold
    top_fraction = args.protected_residual_mask_top_fraction
    if threshold is None and top_fraction is None:
        return None, {
            "schema": "hprc_protected_residual_mask.v1",
            "enabled": False,
            "reason": "protected_pathway_unmasked",
            "score_claim": False,
            "promotion_eligible": False,
        }
    if residual_protection is None:
        raise ValueError(
            "protected residual mask sparsification requires "
            "--rate-aware-residual-protection-npy"
        )
    arr = np.asarray(residual_protection, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("protected residual mask source must be FxHxWxC after projection")
    if not np.all(np.isfinite(arr)):
        raise ValueError("protected residual mask source contains non-finite values")

    candidate_mask = np.ones(arr.size, dtype=bool)
    threshold_value: float | None = None
    if threshold is not None:
        threshold_value = float(threshold)
        if threshold_value < 0.0:
            raise ValueError("--protected-residual-mask-threshold must be non-negative")
        candidate_mask &= arr.reshape(-1) >= threshold_value

    if top_fraction is not None:
        fraction = float(top_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("--protected-residual-mask-top-fraction must be in (0, 1]")
        candidates = np.flatnonzero(candidate_mask)
        keep = min(len(candidates), max(1, int(np.ceil(arr.size * fraction))))
        selected = np.array([], dtype=np.int64)
        if keep:
            scores = arr.reshape(-1)[candidates]
            # Deterministic tie spreading prevents top-k ties from collapsing
            # into the first frames/cells when coarse P18/P19 surfaces are binary.
            tie = (
                candidates.astype(np.uint64) * np.uint64(11400714819323198485)
            ).astype(np.uint64)
            order = np.lexsort((tie, -scores))
            selected = candidates[order[:keep]]
            threshold_value = float(np.min(scores[order[:keep]]))
        sparse_mask = np.zeros(arr.size, dtype=bool)
        sparse_mask[selected] = True
        candidate_mask = sparse_mask

    mask = candidate_mask.reshape(arr.shape).astype(np.float32)
    active = int(np.count_nonzero(mask))
    if active <= 0:
        raise ValueError("protected residual mask sparsification selected no cells")
    return mask, {
        "schema": "hprc_protected_residual_mask.v1",
        "enabled": True,
        "source": "rate_aware_residual_protection",
        "shape": [int(v) for v in mask.shape],
        "active_cell_count": active,
        "total_cell_count": int(mask.size),
        "active_fraction": float(active / mask.size) if mask.size else 0.0,
        "threshold": threshold_value,
        "top_fraction": None if top_fraction is None else float(top_fraction),
        "tie_breaker": "uint64_golden_ratio_hash_of_flat_cell_index",
        "semantics": (
            "1=emit_highres_protected_residual_cell,"
            "0=let_coarse_receiver_or_fill_handle_cell"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _resolve_output_dir(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> tuple[Path, Path | None]:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, None

    tiers = parse_storage_tier_specs(
        list(args.storage_tier),
        repo_root=repo_root,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=bool(args.allow_local_output_dir),
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=str(args.storage_workload_subdir),
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    workload_root = require_selected_storage(plan)
    run_id = args.run_id or _utc_stamp()
    output_dir = workload_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_plan_path = output_dir / "hprc_compact_receiver_training_storage_plan.json"
    _write_json_maybe_overwrite(
        storage_plan_path,
        {
            "schema": HPRC_LONG_TRAINING_STORAGE_PLAN_SCHEMA,
            "storage_plan": plan.to_dict(),
            "selected_training_output_dir": output_dir.as_posix(),
            **FALSE_AUTHORITY,
        },
    )
    return output_dir, storage_plan_path


def _resolve_optional_path(path: Path | None, *, repo_root: Path) -> Path | None:
    if path is None:
        return None
    out = Path(path).expanduser()
    return out if out.is_absolute() else repo_root / out


def _write_json_maybe_overwrite(path: Path, payload: object) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _utc_stamp() -> str:
    return time.strftime("hprc_compact_receiver_training_%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, StorageTierError, ValueError) as exc:
        print(f"run_hprc_compact_receiver_training failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
