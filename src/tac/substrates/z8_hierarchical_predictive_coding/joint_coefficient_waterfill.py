# SPDX-License-Identifier: MIT
"""Executable joint P18/P19 water-fill rate attack for Z8 wavelet coefficients.

The current faithful Z8 archive is rate-bound: the Mallat wavelet payload keeps
distortion low but dominates bytes. This module turns the joint SegNet/PoseNet
surface into a deterministic coefficient-level transform:

1. project pixel/pair joint weights onto each Mallat detail subband;
2. select low-joint-weight, pose-null atoms as rate-attack candidates;
3. dead-zone and quantize only those detail coefficients;
4. rebuild a byte-closed Z8HPC1 archive and measure rate/distortion deltas.

All outputs are local/advisory until exact contest CPU/CUDA eval signs the
archive/runtime pair.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.repo_io import write_json
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    pack_archive,
    parse_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
    export_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    pack_pair_pyramids_to_wavelet_blob,
    parse_pair_blobs_from_wavelet_blob,
    reconstruct_pair_rgb_from_pyramid,
)
from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
    WaveletDetail2D,
)
from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
    projected_pair_pyramids_from_archive_bytes,
)

Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA = "z8_joint_p18_p19_coefficient_waterfill_rate_attack.v1"
Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA = "z8_joint_p18_p19_coefficient_deadzone_candidate.v1"
Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA = "z8_joint_p18_p19_coefficient_relinearized_search.v1"
Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE = (
    "project_joint_p18_p19_pixel_pair_surface_to_mallat_detail_subbands_then_"
    "dead_zone_low_joint_weight_pose_null_coefficients"
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


@dataclass(frozen=True)
class Z8JointCoefficientWaterfillConfig:
    """Controls deterministic Z8 coefficient dead-zone quantization."""

    joint_weight_quantile: float = 0.35
    coefficient_deadzone_quantile: float = 0.50
    quantization_step: float = 1.0 / 255.0
    pose_null_required: bool = True
    max_pairs: int | None = None
    emit_archive_zip: bool = True
    emit_receiver_proof: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("joint_weight_quantile", self.joint_weight_quantile),
            ("coefficient_deadzone_quantile", self.coefficient_deadzone_quantile),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.quantization_step < 0.0:
            raise ValueError("quantization_step must be >= 0")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive when provided")


@dataclass(frozen=True)
class Z8JointCoefficientRelinearizationSearchConfig:
    """Bounded iterative search over fresh joint P18/P19 coefficient surfaces.

    Each iteration consumes a fresh surface from the MLX scorer-VJP/pose-null
    lane, evaluates a small deterministic dead-zone grid on the current
    archive, accepts the lowest proxy objective that satisfies the distortion
    guard, and remeasures cumulative rate/distortion versus the original
    archive. Reusing the same surface while claiming relinearization is refused
    by default.
    """

    joint_weight_quantiles: tuple[float, ...] = (0.20, 0.35, 0.50)
    coefficient_deadzone_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    quantization_steps: tuple[float, ...] = (1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0)
    max_iterations: int = 3
    max_cumulative_mse: float | None = None
    rate_weight: float = 25.0
    distortion_weight: float = 10_000.0
    interaction_penalty_weight: float = 10_000.0
    require_fresh_surface_per_iteration: bool = True
    pose_null_required: bool = True
    max_pairs: int | None = None
    emit_archive_zip: bool = True
    emit_receiver_proof: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        for name, values in (
            ("joint_weight_quantiles", self.joint_weight_quantiles),
            ("coefficient_deadzone_quantiles", self.coefficient_deadzone_quantiles),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            for value in values:
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"{name} entries must be in [0, 1]")
        if not self.quantization_steps:
            raise ValueError("quantization_steps must not be empty")
        if any(float(value) < 0.0 for value in self.quantization_steps):
            raise ValueError("quantization_steps entries must be >= 0")
        if self.max_cumulative_mse is not None and self.max_cumulative_mse < 0.0:
            raise ValueError("max_cumulative_mse must be >= 0 when provided")
        if self.rate_weight < 0.0:
            raise ValueError("rate_weight must be >= 0")
        if self.distortion_weight < 0.0:
            raise ValueError("distortion_weight must be >= 0")
        if self.interaction_penalty_weight < 0.0:
            raise ValueError("interaction_penalty_weight must be >= 0")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive when provided")


def _as_bool_mask(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=bool)


def _normalize_surface_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.ndim > 5:
        raise ValueError(f"{name} must have at most 5 dimensions; got {arr.shape}")
    return arr


def _surface_digest(joint_weight: Any, safe_mask: Any | None) -> str:
    h = hashlib.sha256()
    joint = np.ascontiguousarray(np.asarray(joint_weight, dtype=np.float64))
    h.update(str(joint.shape).encode("utf-8"))
    h.update(joint.tobytes())
    if safe_mask is not None:
        mask = np.ascontiguousarray(np.asarray(safe_mask, dtype=bool))
        h.update(str(mask.shape).encode("utf-8"))
        h.update(mask.tobytes())
    return h.hexdigest()


def _surface_pair(surface: Any) -> tuple[Any, Any | None]:
    if isinstance(surface, dict):
        if "joint_weight" not in surface:
            raise ValueError("surface dict must contain joint_weight")
        return surface["joint_weight"], surface.get("rate_attack_deadzone_mask")
    if isinstance(surface, tuple) and len(surface) == 2:
        return surface[0], surface[1]
    return surface, None


def load_joint_p18_p19_surface_file(path: str | Path) -> tuple[Any, Any | None]:
    """Load a joint surface file consumed by Z8 materializer/search CLIs."""

    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        if "joint_weight" not in data:
            raise ValueError(f"{p} must contain joint_weight")
        mask = data.get("rate_attack_deadzone_mask", None)
        return data["joint_weight"], mask
    if p.suffix == ".npy":
        return np.load(p), None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "joint_weight" not in payload:
        raise ValueError(f"{p} JSON must contain joint_weight")
    return payload["joint_weight"], payload.get("rate_attack_deadzone_mask")


def _surface_slice_for_pair_frame(
    arr: np.ndarray,
    *,
    pair_idx: int,
    frame_idx: int,
) -> np.ndarray:
    """Return a 2D/3D surface for one pair/frame from common surface layouts."""

    if arr.ndim == 0:
        return np.asarray([[float(arr)]], dtype=np.float64)
    if arr.ndim == 1:
        return np.asarray([[float(arr[pair_idx % arr.shape[0]])]], dtype=np.float64)
    if arr.ndim in {2, 3}:
        return arr
    if arr.ndim == 4:
        return arr[pair_idx % arr.shape[0]]
    # arr.ndim == 5: (pairs, frames, H, W, C)
    return arr[pair_idx % arr.shape[0], frame_idx % arr.shape[1]]


def _resize_hw_mean(
    surface: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Area-pool or repeat a 2D/3D surface to ``(H, W, C)`` deterministically."""

    target_h, target_w, target_c = (int(v) for v in target_shape)
    src = np.asarray(surface, dtype=np.float64)
    if src.ndim == 0:
        src = src.reshape(1, 1)
    if src.ndim == 1:
        src = src.reshape(src.shape[0], 1)
    if src.ndim == 2:
        src = src[:, :, None]
    if src.ndim != 3:
        raise ValueError(f"surface slice must be 2D or 3D; got {src.shape}")
    src_h, src_w, src_c = src.shape
    out = np.zeros((target_h, target_w, target_c), dtype=np.float64)
    for y in range(target_h):
        y0 = int(np.floor(y * src_h / target_h))
        y1 = int(np.ceil((y + 1) * src_h / target_h))
        y1 = max(y1, y0 + 1)
        for x in range(target_w):
            x0 = int(np.floor(x * src_w / target_w))
            x1 = int(np.ceil((x + 1) * src_w / target_w))
            x1 = max(x1, x0 + 1)
            pooled = src[y0:y1, x0:x1].mean(axis=(0, 1))
            if src_c == target_c:
                out[y, x] = pooled
            elif src_c == 1:
                out[y, x] = float(pooled[0])
            else:
                out[y, x] = float(np.mean(pooled))
    return out


def _project_surface_to_subband(
    arr: np.ndarray,
    *,
    pair_idx: int,
    frame_idx: int,
    subband_shape: tuple[int, int, int],
) -> np.ndarray:
    return _resize_hw_mean(
        _surface_slice_for_pair_frame(arr, pair_idx=pair_idx, frame_idx=frame_idx),
        target_shape=subband_shape,
    )


def _quantize_selected(
    coeff: np.ndarray,
    *,
    projected_joint: np.ndarray,
    projected_safe: np.ndarray | None,
    config: Z8JointCoefficientWaterfillConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    coeff64 = np.asarray(coeff, dtype=np.float64)
    if projected_joint.shape != coeff64.shape:
        raise ValueError(f"projected_joint shape {projected_joint.shape} != coeff shape {coeff64.shape}")
    water_level = float(np.quantile(projected_joint, config.joint_weight_quantile))
    eligible = projected_joint <= water_level
    if config.pose_null_required:
        eligible = np.zeros_like(eligible, dtype=bool) if projected_safe is None else eligible & (projected_safe >= 0.5)
    abs_coeff = np.abs(coeff64)
    if np.any(eligible):
        deadzone_threshold = float(np.quantile(abs_coeff[eligible], config.coefficient_deadzone_quantile))
    else:
        deadzone_threshold = 0.0
    changed = eligible & (abs_coeff <= deadzone_threshold)
    out = coeff64.copy()
    if config.quantization_step > 0.0:
        coarse = np.round(out[eligible] / config.quantization_step) * (config.quantization_step)
        out[eligible] = coarse
    out[changed] = 0.0
    out32 = out.astype(np.float32)
    delta = out32.astype(np.float64) - coeff64
    return out32, {
        "water_level": water_level,
        "deadzone_abs_threshold": deadzone_threshold,
        "eligible_coefficients": int(np.count_nonzero(eligible)),
        "dead_zoned_coefficients": int(np.count_nonzero(changed)),
        "total_coefficients": int(coeff64.size),
        "max_abs_coeff_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_coeff_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
    }


def _mutate_detail(
    detail: WaveletDetail2D,
    *,
    joint_surface: np.ndarray,
    safe_mask: np.ndarray | None,
    pair_idx: int,
    frame_idx: int,
    level_idx: int,
    config: Z8JointCoefficientWaterfillConfig,
) -> tuple[WaveletDetail2D, list[dict[str, Any]]]:
    mutated: dict[str, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    for subband_name in ("lh", "hl", "hh"):
        coeff = np.asarray(getattr(detail, subband_name), dtype=np.float32)
        projected_joint = _project_surface_to_subband(
            joint_surface,
            pair_idx=pair_idx,
            frame_idx=frame_idx,
            subband_shape=coeff.shape,
        )
        projected_safe = (
            _project_surface_to_subband(
                safe_mask.astype(np.float64),
                pair_idx=pair_idx,
                frame_idx=frame_idx,
                subband_shape=coeff.shape,
            )
            if safe_mask is not None
            else None
        )
        next_coeff, sub_stats = _quantize_selected(
            coeff,
            projected_joint=projected_joint,
            projected_safe=projected_safe,
            config=config,
        )
        mutated[subband_name] = next_coeff
        sub_stats.update(
            {
                "pair_index": int(pair_idx),
                "frame_index": int(frame_idx),
                "level_index": int(level_idx),
                "subband": subband_name,
            }
        )
        stats.append(sub_stats)
    return (
        WaveletDetail2D(
            lh=mutated["lh"],
            hl=mutated["hl"],
            hh=mutated["hh"],
        ),
        stats,
    )


def _mutate_pair_pyramids(
    pair_pyramids: list[dict[str, Any]],
    *,
    joint_surface: np.ndarray,
    safe_mask: np.ndarray | None,
    config: Z8JointCoefficientWaterfillConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_limit = len(pair_pyramids) if config.max_pairs is None else min(len(pair_pyramids), int(config.max_pairs))
    out: list[dict[str, Any]] = []
    subband_stats: list[dict[str, Any]] = []
    for pair_idx, pyramid in enumerate(pair_pyramids):
        if pair_idx >= pair_limit:
            out.append(pyramid)
            continue
        next_pyramid = dict(pyramid)
        for frame_idx, details_key in enumerate(("frame_0_details", "frame_1_details")):
            next_details: list[WaveletDetail2D] = []
            for level_idx, detail in enumerate(pyramid[details_key]):
                mutated, stats = _mutate_detail(
                    detail,
                    joint_surface=joint_surface,
                    safe_mask=safe_mask,
                    pair_idx=pair_idx,
                    frame_idx=frame_idx,
                    level_idx=level_idx,
                    config=config,
                )
                next_details.append(mutated)
                subband_stats.extend(stats)
            next_pyramid[details_key] = next_details
        out.append(next_pyramid)

    total = sum(row["total_coefficients"] for row in subband_stats)
    eligible = sum(row["eligible_coefficients"] for row in subband_stats)
    zeroed = sum(row["dead_zoned_coefficients"] for row in subband_stats)
    return out, {
        "pair_count": len(pair_pyramids),
        "mutated_pair_count": pair_limit,
        "total_detail_coefficients": total,
        "eligible_coefficients": eligible,
        "dead_zoned_coefficients": zeroed,
        "eligible_fraction": float(eligible / total) if total else 0.0,
        "dead_zoned_fraction": float(zeroed / total) if total else 0.0,
        "subband_stats": subband_stats,
    }


def _small_receiver_tensor(archive_bytes: bytes) -> np.ndarray:
    binding, pair_pyramids, _stats = projected_pair_pyramids_from_archive_bytes(archive_bytes)
    frames: list[np.ndarray] = []
    for pyramid in pair_pyramids:
        frame_0, frame_1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        frames.append(frame_0.reshape(-1))
        frames.append(frame_1.reshape(-1))
    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(frames).astype(np.float32, copy=False)


def _distortion_report(before_archive: bytes, after_archive: bytes) -> dict[str, Any]:
    before = _small_receiver_tensor(before_archive)
    after = _small_receiver_tensor(after_archive)
    if before.shape != after.shape:
        return {
            "small_receiver_distortion_measured": False,
            "blocker": "receiver_tensor_shape_mismatch",
            "before_shape": list(before.shape),
            "after_shape": list(after.shape),
        }
    delta = after.astype(np.float64) - before.astype(np.float64)
    return {
        "small_receiver_distortion_measured": True,
        "receiver_tensor_values": int(before.size),
        "mse": float(np.mean(delta * delta)) if delta.size else 0.0,
        "mae": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _candidate_proxy_objective(
    *,
    result: dict[str, Any],
    cumulative_distortion: dict[str, Any],
    previous_cumulative_mse: float,
    config: Z8JointCoefficientRelinearizationSearchConfig,
) -> float:
    rate = result["rate_report"]["archive_rate_ratio"]
    mse = float(cumulative_distortion.get("mse", float("inf")))
    interaction = max(0.0, mse - float(previous_cumulative_mse))
    return (
        float(config.rate_weight) * float(rate)
        + float(config.distortion_weight) * mse
        + float(config.interaction_penalty_weight) * interaction
    )


def apply_joint_p18_p19_deadzone_to_z8_archive(
    archive_bytes: bytes,
    *,
    joint_weight: Any,
    rate_attack_deadzone_mask: Any | None = None,
    config: Z8JointCoefficientWaterfillConfig | None = None,
) -> dict[str, Any]:
    """Return mutated Z8HPC1 bytes plus rate/distortion measurement metadata."""

    cfg = config or Z8JointCoefficientWaterfillConfig()
    joint_surface = _normalize_surface_array(joint_weight, name="joint_weight")
    safe_mask = _as_bool_mask(rate_attack_deadzone_mask)
    if safe_mask is not None and safe_mask.shape != joint_surface.shape and safe_mask.size != 1:
        raise ValueError("rate_attack_deadzone_mask must match joint_weight shape or be scalar")
    arc = parse_archive(archive_bytes)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    mutated_pyramids, coeff_report = _mutate_pair_pyramids(
        pair_pyramids,
        joint_surface=joint_surface,
        safe_mask=safe_mask,
        config=cfg,
    )
    mutated_wavelet_blob = pack_pair_pyramids_to_wavelet_blob(mutated_pyramids)
    mutated_archive = pack_archive(
        arc.decoder_state_dict,
        arc.per_level_category_indices,
        mutated_wavelet_blob,
        arc.wyner_ziv_top_blob,
        arc.dreamer_state_blob,
        {
            **arc.meta,
            "joint_p18_p19_rate_attack": {
                "schema": Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA,
                "role": Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
                "joint_weight_quantile": float(cfg.joint_weight_quantile),
                "coefficient_deadzone_quantile": float(cfg.coefficient_deadzone_quantile),
                "quantization_step": float(cfg.quantization_step),
                "pose_null_required": bool(cfg.pose_null_required),
            },
        },
        num_levels=arc.num_levels,
        num_groups_per_level=arc.num_groups_per_level,
        num_categories_per_level=arc.num_categories_per_level,
        num_pairs=arc.num_pairs,
        decoder_latent_dim=arc.decoder_latent_dim,
        base_channels=arc.base_channels,
        wavelet_basis_id=arc.wavelet_basis_id,
        schema_version=arc.schema_version,
    )
    before_wavelet_len = len(arc.wavelet_coeffs_blob)
    after_arc = parse_archive(mutated_archive)
    after_wavelet_len = len(after_arc.wavelet_coeffs_blob)
    before_archive_len = len(archive_bytes)
    after_archive_len = len(mutated_archive)
    rate_report = {
        "before_archive_bytes": before_archive_len,
        "after_archive_bytes": after_archive_len,
        "archive_byte_delta": after_archive_len - before_archive_len,
        "archive_rate_ratio": (float(after_archive_len / before_archive_len) if before_archive_len else 1.0),
        "before_wavelet_blob_bytes": before_wavelet_len,
        "after_wavelet_blob_bytes": after_wavelet_len,
        "wavelet_blob_byte_delta": after_wavelet_len - before_wavelet_len,
        "wavelet_blob_rate_ratio": (float(after_wavelet_len / before_wavelet_len) if before_wavelet_len else 1.0),
    }
    distortion = _distortion_report(archive_bytes, mutated_archive)
    return {
        "schema": Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA,
        "role": Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
        "mutated_archive_bytes": mutated_archive,
        "original_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "mutated_archive_sha256": hashlib.sha256(mutated_archive).hexdigest(),
        "coefficient_report": coeff_report,
        "rate_report": rate_report,
        "distortion_report": distortion,
        "axis_tag": "[macOS-CPU advisory]",
        "allowed_use": "local_z8_rate_attack_materialization_and_acquisition_signal",
        "forbidden_use": "score_claim_or_exact_promotion_without_contest_eval",
        **FALSE_AUTHORITY,
    }


def materialize_joint_p18_p19_deadzone_candidate(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    joint_weight: Any,
    rate_attack_deadzone_mask: Any | None = None,
    config: Z8JointCoefficientWaterfillConfig | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a byte-closed Z8 archive variant plus advisory manifest."""

    cfg = config or Z8JointCoefficientWaterfillConfig()
    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=rate_attack_deadzone_mask,
        config=cfg,
    )
    mutated_archive = bytes(result["mutated_archive_bytes"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_bin = out_dir / "0.bin"
    candidate_bin.write_bytes(mutated_archive)
    archive_zip_path: Path | None = None
    archive_sha256: str | None = None
    archive_zip_bytes: int | None = None
    if cfg.emit_archive_zip:
        archive_zip_path, archive_sha256, archive_zip_bytes = export_z8hpc1_archive_bytes(
            mutated_archive,
            out_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=cfg.emit_receiver_proof,
            emit_byte_mutation_proof=False,
            retain_receiver_proof_output=False,
        )
    manifest = {
        "schema": Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA,
        "candidate_bin_path": candidate_bin.as_posix(),
        "candidate_bin_sha256": hashlib.sha256(mutated_archive).hexdigest(),
        "candidate_bin_bytes": len(mutated_archive),
        "archive_zip_path": archive_zip_path.as_posix() if archive_zip_path else None,
        "archive_zip_sha256": archive_sha256,
        "archive_zip_bytes": archive_zip_bytes,
        "receiver_proof_executed": bool(cfg.emit_receiver_proof),
        "exact_axis_blocker": (
            None if cfg.emit_receiver_proof else "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
        ),
        "waterfill_result": {key: value for key, value in result.items() if key != "mutated_archive_bytes"},
        **FALSE_AUTHORITY,
    }
    manifest_path = out_dir / "z8_joint_p18_p19_deadzone_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    write_json(manifest_path, manifest)
    return manifest


def run_joint_p18_p19_relinearized_deadzone_search(
    archive_bytes: bytes,
    *,
    surfaces: Sequence[Any],
    config: Z8JointCoefficientRelinearizationSearchConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded iterative Z8 dead-zone search over fresh joint surfaces."""

    cfg = config or Z8JointCoefficientRelinearizationSearchConfig()
    if not surfaces:
        raise ValueError("surfaces must not be empty")
    original = bytes(archive_bytes)
    current = original
    seen_surface_digests: set[str] = set()
    accepted: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    previous_cumulative_mse = 0.0
    final_blocker: str | None = None

    for iteration_index, raw_surface in enumerate(surfaces[: cfg.max_iterations]):
        joint_weight, safe_mask = _surface_pair(raw_surface)
        digest = _surface_digest(joint_weight, safe_mask)
        if cfg.require_fresh_surface_per_iteration and digest in seen_surface_digests:
            raise ValueError(
                "fresh surface required for iterative relinearization; duplicate "
                f"surface digest at iteration {iteration_index}: {digest}"
            )
        seen_surface_digests.add(digest)

        iteration_candidates: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for joint_q in cfg.joint_weight_quantiles:
            for deadzone_q in cfg.coefficient_deadzone_quantiles:
                for q_step in cfg.quantization_steps:
                    waterfill_cfg = Z8JointCoefficientWaterfillConfig(
                        joint_weight_quantile=float(joint_q),
                        coefficient_deadzone_quantile=float(deadzone_q),
                        quantization_step=float(q_step),
                        pose_null_required=bool(cfg.pose_null_required),
                        max_pairs=cfg.max_pairs,
                        emit_archive_zip=False,
                        emit_receiver_proof=False,
                    )
                    result = apply_joint_p18_p19_deadzone_to_z8_archive(
                        current,
                        joint_weight=joint_weight,
                        rate_attack_deadzone_mask=safe_mask,
                        config=waterfill_cfg,
                    )
                    mutated = bytes(result["mutated_archive_bytes"])
                    cumulative_distortion = _distortion_report(original, mutated)
                    cumulative_mse = float(cumulative_distortion.get("mse", float("inf")))
                    guard_ok = cfg.max_cumulative_mse is None or cumulative_mse <= float(cfg.max_cumulative_mse)
                    objective = _candidate_proxy_objective(
                        result=result,
                        cumulative_distortion=cumulative_distortion,
                        previous_cumulative_mse=previous_cumulative_mse,
                        config=cfg,
                    )
                    row = {
                        "schema": "z8_joint_p18_p19_relinearized_candidate.v1",
                        "iteration_index": int(iteration_index),
                        "surface_digest": digest,
                        "joint_weight_quantile": float(joint_q),
                        "coefficient_deadzone_quantile": float(deadzone_q),
                        "quantization_step": float(q_step),
                        "proxy_objective": float(objective),
                        "guard_ok": bool(guard_ok),
                        "rate_report": result["rate_report"],
                        "incremental_distortion_report": result["distortion_report"],
                        "cumulative_distortion_report": cumulative_distortion,
                        "coefficient_summary": {
                            key: value for key, value in result["coefficient_report"].items() if key != "subband_stats"
                        },
                        "mutated_archive_sha256": result["mutated_archive_sha256"],
                        **FALSE_AUTHORITY,
                    }
                    iteration_candidates.append(row)
                    all_candidates.append(row)
                    if guard_ok and (best is None or objective < best["proxy_objective"]):
                        best = {**row, "_mutated_archive_bytes": mutated}
        if best is None:
            final_blocker = f"all_candidates_failed_cumulative_distortion_guard_at_iteration_{iteration_index}"
            break
        current = bytes(best.pop("_mutated_archive_bytes"))
        previous_cumulative_mse = float(best["cumulative_distortion_report"].get("mse", previous_cumulative_mse))
        accepted.append(best)

    cumulative_rate_report = {
        "original_archive_bytes": len(original),
        "final_archive_bytes": len(current),
        "archive_byte_delta": len(current) - len(original),
        "archive_rate_ratio": float(len(current) / len(original)) if original else 1.0,
    }
    final_distortion = _distortion_report(original, current)
    return {
        "schema": Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA,
        "role": Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
        "ste_boundary": "straight_through_deadzone_quantization_proxy",
        "surface_refresh_contract": ("fresh_joint_p18_p19_surface_per_iteration_from_mlx_scorer_vjp"),
        "pose_guard": "pose_null_mask_and_mahalanobis_ail_weights_consumed",
        "interaction_penalty": ("penalize_cumulative_mse_increase_between_relinearization_steps"),
        "iterations_requested": int(cfg.max_iterations),
        "iterations_accepted": len(accepted),
        "candidate_count": len(all_candidates),
        "accepted_candidates": accepted,
        "candidate_grid": all_candidates,
        "cumulative_rate_report": cumulative_rate_report,
        "final_distortion_report": final_distortion,
        "final_blocker": final_blocker,
        "final_archive_sha256": hashlib.sha256(current).hexdigest(),
        "final_archive_bytes_payload": current,
        "axis_tag": "[macOS-CPU advisory]",
        "allowed_use": "local_z8_rate_attack_acquisition_and_materialization_signal",
        "forbidden_use": "score_claim_or_exact_promotion_without_contest_eval",
        **FALSE_AUTHORITY,
    }


def materialize_joint_p18_p19_relinearized_deadzone_search(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    surfaces: Sequence[Any],
    config: Z8JointCoefficientRelinearizationSearchConfig | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write the accepted iterative Z8 dead-zone candidate and manifest."""

    cfg = config or Z8JointCoefficientRelinearizationSearchConfig()
    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=surfaces,
        config=cfg,
    )
    final_archive = bytes(result["final_archive_bytes_payload"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_bin = out_dir / "0.bin"
    final_bin.write_bytes(final_archive)

    archive_zip_path: Path | None = None
    archive_sha256: str | None = None
    archive_zip_bytes: int | None = None
    if cfg.emit_archive_zip:
        archive_zip_path, archive_sha256, archive_zip_bytes = export_z8hpc1_archive_bytes(
            final_archive,
            out_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=cfg.emit_receiver_proof,
            emit_byte_mutation_proof=False,
            retain_receiver_proof_output=False,
        )

    manifest = {key: value for key, value in result.items() if key != "final_archive_bytes_payload"}
    manifest.update(
        {
            "candidate_bin_path": final_bin.as_posix(),
            "candidate_bin_sha256": hashlib.sha256(final_archive).hexdigest(),
            "candidate_bin_bytes": len(final_archive),
            "archive_zip_path": (archive_zip_path.as_posix() if archive_zip_path else None),
            "archive_zip_sha256": archive_sha256,
            "archive_zip_bytes": archive_zip_bytes,
            "receiver_proof_executed": bool(cfg.emit_receiver_proof),
            "exact_axis_blocker": (
                None if cfg.emit_receiver_proof else "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
            ),
            **FALSE_AUTHORITY,
        }
    )
    manifest_path = out_dir / "z8_joint_p18_p19_relinearized_search_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE",
    "Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA",
    "Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA",
    "Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA",
    "Z8JointCoefficientRelinearizationSearchConfig",
    "Z8JointCoefficientWaterfillConfig",
    "apply_joint_p18_p19_deadzone_to_z8_archive",
    "materialize_joint_p18_p19_deadzone_candidate",
    "materialize_joint_p18_p19_relinearized_deadzone_search",
    "run_joint_p18_p19_relinearized_deadzone_search",
]
