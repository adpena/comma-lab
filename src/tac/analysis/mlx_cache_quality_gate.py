# SPDX-License-Identifier: MIT
"""False-authority quality gate for MLX scorer input caches."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from tac.repo_io import write_json
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

MLX_CACHE_QUALITY_GATE_SCHEMA = "mlx_cache_quality_gate.v1"


class MLXCacheQualityGateError(RuntimeError):
    """Raised for malformed cache-quality gate inputs."""


@dataclass(frozen=True)
class CacheArrayStats:
    name: str
    shape: tuple[int, ...]
    dtype: str
    min_value: float
    max_value: float
    mean: float
    std: float
    dynamic_range: float

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "std": self.std,
            "dynamic_range": self.dynamic_range,
        }


def build_mlx_cache_quality_gate(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    sample_pairs: int = 16,
    min_segnet_std: float = 1.0,
    min_segnet_dynamic_range: float = 16.0,
    max_segnet_mae_vs_reference_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    """Inspect candidate scorer-input cache health against a reference cache."""

    if sample_pairs < 1:
        raise MLXCacheQualityGateError("sample_pairs must be >= 1")
    candidate = Path(candidate_cache_dir).expanduser().resolve(strict=False)
    reference = Path(reference_cache_dir).expanduser().resolve(strict=False)
    if not candidate.is_dir():
        raise MLXCacheQualityGateError(f"candidate cache dir missing: {candidate}")
    if not reference.is_dir():
        raise MLXCacheQualityGateError(f"reference cache dir missing: {reference}")

    cand_seg = _load_cache_array(candidate, "segnet_last_rgb.npy")
    ref_seg = _load_cache_array(reference, "segnet_last_rgb.npy")
    cand_pose = _load_cache_array(candidate, "posenet_yuv6_pair.npy")
    ref_pose = _load_cache_array(reference, "posenet_yuv6_pair.npy")
    n_seg = _sample_count(cand_seg, ref_seg, sample_pairs)
    n_pose = _sample_count(cand_pose, ref_pose, sample_pairs)
    cand_seg_sample = np.asarray(cand_seg[:n_seg], dtype=np.float32)
    ref_seg_sample = np.asarray(ref_seg[:n_seg], dtype=np.float32)
    cand_pose_sample = np.asarray(cand_pose[:n_pose], dtype=np.float32)
    ref_pose_sample = np.asarray(ref_pose[:n_pose], dtype=np.float32)

    cand_seg_stats = _stats("candidate_segnet_last_rgb", cand_seg_sample)
    ref_seg_stats = _stats("reference_segnet_last_rgb", ref_seg_sample)
    cand_pose_stats = _stats("candidate_posenet_yuv6_pair", cand_pose_sample)
    ref_pose_stats = _stats("reference_posenet_yuv6_pair", ref_pose_sample)

    seg_mae = _mean_abs(cand_seg_sample, ref_seg_sample)
    seg_rmse = _rmse(cand_seg_sample, ref_seg_sample)
    pose_mae = _mean_abs(cand_pose_sample, ref_pose_sample)
    pose_rmse = _rmse(cand_pose_sample, ref_pose_sample)
    blockers: list[str] = ["mlx_cache_quality_gate_is_false_authority"]
    if cand_seg_stats.std < min_segnet_std:
        blockers.append("candidate_segnet_last_rgb_degenerate_constant_or_flat")
    if cand_seg_stats.dynamic_range < min_segnet_dynamic_range:
        blockers.append("candidate_segnet_last_rgb_dynamic_range_too_low")
    if seg_mae > max_segnet_mae_vs_reference_for_fit_gate:
        blockers.append("candidate_segnet_last_rgb_far_from_reference_fit_gate")
    if cand_seg_sample.shape[1:] != ref_seg_sample.shape[1:]:
        blockers.append("segnet_cache_shape_mismatch")
    if cand_pose_sample.shape[1:] != ref_pose_sample.shape[1:]:
        blockers.append("posenet_cache_shape_mismatch")
    blockers = _ordered_unique(blockers)

    if "candidate_segnet_last_rgb_degenerate_constant_or_flat" in blockers:
        verdict = "FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE"
    elif "candidate_segnet_last_rgb_far_from_reference_fit_gate" in blockers:
        verdict = "FIT_OR_SCALE_FAILURE"
    else:
        verdict = "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"

    return {
        "schema": MLX_CACHE_QUALITY_GATE_SCHEMA,
        "candidate_cache_dir": candidate.as_posix(),
        "reference_cache_dir": reference.as_posix(),
        "sample_pairs": int(sample_pairs),
        "verdict": verdict,
        "candidate_cache_nondegenerate": verdict == "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY",
        "fit_gate_passed": verdict == "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY",
        "stats": {
            "candidate_segnet_last_rgb": cand_seg_stats.as_jsonable(),
            "reference_segnet_last_rgb": ref_seg_stats.as_jsonable(),
            "candidate_posenet_yuv6_pair": cand_pose_stats.as_jsonable(),
            "reference_posenet_yuv6_pair": ref_pose_stats.as_jsonable(),
        },
        "distance_to_reference": {
            "segnet_last_rgb_mae": seg_mae,
            "segnet_last_rgb_rmse": seg_rmse,
            "posenet_yuv6_pair_mae": pose_mae,
            "posenet_yuv6_pair_rmse": pose_rmse,
        },
        "thresholds": {
            "min_segnet_std": float(min_segnet_std),
            "min_segnet_dynamic_range": float(min_segnet_dynamic_range),
            "max_segnet_mae_vs_reference_for_fit_gate": float(
                max_segnet_mae_vs_reference_for_fit_gate
            ),
        },
        "blockers": blockers,
        "recommended_next_actions": _recommended_next_actions(verdict),
        **FALSE_AUTHORITY,
    }


def write_mlx_cache_quality_gate(
    *,
    output_json: str | Path,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    sample_pairs: int = 16,
    min_segnet_std: float = 1.0,
    min_segnet_dynamic_range: float = 16.0,
    max_segnet_mae_vs_reference_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    report = build_mlx_cache_quality_gate(
        candidate_cache_dir=candidate_cache_dir,
        reference_cache_dir=reference_cache_dir,
        sample_pairs=sample_pairs,
        min_segnet_std=min_segnet_std,
        min_segnet_dynamic_range=min_segnet_dynamic_range,
        max_segnet_mae_vs_reference_for_fit_gate=max_segnet_mae_vs_reference_for_fit_gate,
    )
    out = Path(output_json).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = out.as_posix()
    write_json(out, report)
    return report


def _load_cache_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.is_file():
        raise MLXCacheQualityGateError(f"cache array missing: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim < 2:
        raise MLXCacheQualityGateError(f"cache array has invalid rank: {path}")
    return arr


def _sample_count(a: np.ndarray, b: np.ndarray, sample_pairs: int) -> int:
    n = min(int(a.shape[0]), int(b.shape[0]), int(sample_pairs))
    if n < 1:
        raise MLXCacheQualityGateError("cache arrays have no sample rows")
    return n


def _stats(name: str, value: np.ndarray) -> CacheArrayStats:
    min_value = _finite_float(np.min(value), f"{name}.min")
    max_value = _finite_float(np.max(value), f"{name}.max")
    return CacheArrayStats(
        name=name,
        shape=tuple(int(dim) for dim in value.shape),
        dtype=str(value.dtype),
        min_value=min_value,
        max_value=max_value,
        mean=_finite_float(np.mean(value), f"{name}.mean"),
        std=_finite_float(np.std(value), f"{name}.std"),
        dynamic_range=max_value - min_value,
    )


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return _finite_float(np.mean(np.abs(a - b)), "mean_abs")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return _finite_float(np.sqrt(np.mean((a - b) * (a - b))), "rmse")


def _finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise MLXCacheQualityGateError(f"{label} is not finite")
    return out


def _recommended_next_actions(verdict: str) -> list[str]:
    if verdict == "FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE":
        return [
            "block_exact_eval_and_score_claims",
            "inspect_renderer_initialization_training_step_and_archive_export",
            "require_nonconstant_cache_gate_before_section_value_or_spend",
        ]
    if verdict == "FIT_OR_SCALE_FAILURE":
        return [
            "block_exact_eval_and_score_claims",
            "inspect_training_target_scaling_rgb_order_and_eval_roundtrip",
            "run_small_reference_fit_before_long_training",
        ]
    return [
        "local_cache_quality_gate_passed_still_false_authority",
        "continue_with_full_video_distortion_and_receiver_proof_gates",
    ]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "MLX_CACHE_QUALITY_GATE_SCHEMA",
    "MLXCacheQualityGateError",
    "build_mlx_cache_quality_gate",
    "write_mlx_cache_quality_gate",
]
