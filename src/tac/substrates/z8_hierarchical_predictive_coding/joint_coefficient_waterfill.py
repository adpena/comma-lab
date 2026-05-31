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
from collections.abc import Callable, Mapping, Sequence
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
    summarize_wavelet_blob_detail_codecs,
)
from tac.substrates.z8_hierarchical_predictive_coding.inflate_runtime_benchmark import (
    benchmark_z8_submission_inflate_runtime,
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
FULL_VIDEO_EVIDENCE_SCOPE = "full_video"
FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION = "full_video_exact_accumulation"
SINGLE_UPDATE_AFTER_FULL_REDUCTION = "single_update_after_all_pair_shards_reduce"
TRUE_P19_POSE_SURFACE_KIND = "per_axis_posenet_jacobian_mahalanobis_v1"
P19_POSE_SURFACE_BLOCKER = "p19_pose_surface_not_true_per_axis_jacobian"


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
    require_full_video_surface_coverage: bool = True
    require_surface_archive_freshness: bool = True
    require_exact_full_video_gradient_reduction: bool = True
    require_true_p19_pose_surface: bool = True
    measure_receiver_distortion: bool = True
    mutate_coefficients: bool = True
    entropy_code_quantized_details: bool = False
    entropy_detail_quantization_step: float | None = None
    entropy_detail_quantization_steps: Mapping[str, float] | None = None
    lossless_brotli_precondition_details: bool = False
    emit_inflate_runtime_benchmark_work_order: bool = True
    run_inflate_runtime_benchmark: bool = False
    inflate_runtime_benchmark_timeout_seconds: float = 1800.0
    inflate_runtime_benchmark_auth_window_seconds: float = 1800.0
    inflate_runtime_benchmark_device: str = "cpu"

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
        if self.entropy_detail_quantization_step is not None and self.entropy_detail_quantization_step <= 0.0:
            raise ValueError("entropy_detail_quantization_step must be positive when provided")
        if self.entropy_detail_quantization_steps is not None:
            for key, value in self.entropy_detail_quantization_steps.items():
                if value <= 0.0:
                    raise ValueError(f"entropy_detail_quantization_steps[{key!r}] must be positive")
            if self.entropy_detail_quantization_step is not None:
                raise ValueError("entropy_detail_quantization_step and entropy_detail_quantization_steps are exclusive")
        if self.entropy_code_quantized_details and self.lossless_brotli_precondition_details:
            raise ValueError("entropy_code_quantized_details and lossless_brotli_precondition_details are exclusive")
        if self.run_inflate_runtime_benchmark and not self.emit_archive_zip:
            raise ValueError("run_inflate_runtime_benchmark requires emit_archive_zip")
        if self.inflate_runtime_benchmark_timeout_seconds <= 0.0:
            raise ValueError("inflate_runtime_benchmark_timeout_seconds must be positive")
        if self.inflate_runtime_benchmark_auth_window_seconds <= 0.0:
            raise ValueError("inflate_runtime_benchmark_auth_window_seconds must be positive")


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
    require_full_video_surface_coverage: bool = True
    require_surface_archive_freshness: bool = True
    require_exact_full_video_gradient_reduction: bool = True
    require_true_p19_pose_surface: bool = True
    local_replay_prefilter_top_k: int | None = None
    skip_receiver_mse_proxy_when_full_video_replay: bool = True
    mutate_coefficients: bool = True
    entropy_code_quantized_details: bool = False
    entropy_detail_quantization_step: float | None = None
    entropy_detail_quantization_steps: Mapping[str, float] | None = None
    measure_receiver_mse_proxy: bool = True
    lossless_brotli_precondition_details: bool = False
    emit_inflate_runtime_benchmark_work_order: bool = True
    run_inflate_runtime_benchmark: bool = False
    inflate_runtime_benchmark_timeout_seconds: float = 1800.0
    inflate_runtime_benchmark_auth_window_seconds: float = 1800.0
    inflate_runtime_benchmark_device: str = "cpu"

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
        if self.local_replay_prefilter_top_k is not None and self.local_replay_prefilter_top_k <= 0:
            raise ValueError("local_replay_prefilter_top_k must be positive when provided")
        if self.entropy_detail_quantization_step is not None and self.entropy_detail_quantization_step <= 0.0:
            raise ValueError("entropy_detail_quantization_step must be positive when provided")
        if self.entropy_detail_quantization_steps is not None:
            for key, value in self.entropy_detail_quantization_steps.items():
                if value <= 0.0:
                    raise ValueError(f"entropy_detail_quantization_steps[{key!r}] must be positive")
            if self.entropy_detail_quantization_step is not None:
                raise ValueError("entropy_detail_quantization_step and entropy_detail_quantization_steps are exclusive")
        if self.entropy_code_quantized_details and self.lossless_brotli_precondition_details:
            raise ValueError("entropy_code_quantized_details and lossless_brotli_precondition_details are exclusive")
        if self.run_inflate_runtime_benchmark and not self.emit_archive_zip:
            raise ValueError("run_inflate_runtime_benchmark requires emit_archive_zip")
        if self.inflate_runtime_benchmark_timeout_seconds <= 0.0:
            raise ValueError("inflate_runtime_benchmark_timeout_seconds must be positive")
        if self.inflate_runtime_benchmark_auth_window_seconds <= 0.0:
            raise ValueError("inflate_runtime_benchmark_auth_window_seconds must be positive")


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
    if not np.all(np.isfinite(arr)):
        nonfinite = int(arr.size - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"{name} contains non-finite values ({nonfinite}/{arr.size})")
    return arr


def _surface_digest(
    joint_weight: Any,
    safe_mask: Any | None,
    *,
    linearization_archive_sha: str | None = None,
) -> str:
    h = hashlib.sha256()
    joint = np.ascontiguousarray(np.asarray(joint_weight))
    h.update(str(joint.shape).encode("utf-8"))
    h.update(str(joint.dtype).encode("utf-8"))
    h.update(joint.tobytes())
    if safe_mask is not None:
        mask = np.ascontiguousarray(np.asarray(safe_mask, dtype=bool))
        h.update(str(mask.shape).encode("utf-8"))
        h.update(mask.tobytes())
    if linearization_archive_sha:
        h.update(str(linearization_archive_sha).strip().lower().encode("ascii"))
    return h.hexdigest()


def _surface_string(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item()).strip() or None
        if value.size == 1:
            return str(value.reshape(-1)[0]).strip() or None
        return None
    text = str(value).strip()
    return text or None


def _surface_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return bool(value.item())
        if value.size == 1:
            return bool(value.reshape(-1)[0])
        return None
    return bool(value)


def _surface_payload(
    surface: Any,
    rate_attack_deadzone_mask: Any | None = None,
) -> dict[str, Any]:
    if isinstance(surface, Mapping):
        if "joint_weight" not in surface:
            raise ValueError("surface dict must contain joint_weight")
        return {
            "joint_weight": surface["joint_weight"],
            "rate_attack_deadzone_mask": surface.get(
                "rate_attack_deadzone_mask",
                rate_attack_deadzone_mask,
            ),
            "linearization_archive_sha": _surface_string(
                surface.get("linearization_archive_sha") or surface.get("archive_sha256")
            ),
            "evidence_scope": _surface_string(surface.get("evidence_scope")),
            "target_mode": _surface_string(surface.get("target_mode")),
            "gradient_reduction_semantics": _surface_string(surface.get("gradient_reduction_semantics")),
            "gradient_reduction_authority": _surface_bool(surface.get("gradient_reduction_authority")),
            "optimizer_update_authority": _surface_bool(surface.get("optimizer_update_authority")),
            "optimizer_update_semantics": _surface_string(surface.get("optimizer_update_semantics")),
            "full_video_reduction_complete": _surface_bool(surface.get("full_video_reduction_complete")),
            "budget_spend_authority": _surface_bool(surface.get("budget_spend_authority")),
            "pose_surface_kind": _surface_string(surface.get("pose_surface_kind")),
            "pose_surface_authority": _surface_bool(surface.get("pose_surface_authority")),
            "pose_axis_count": surface.get("pose_axis_count"),
            "pose_inverse_variance": surface.get("pose_inverse_variance"),
            "pose_surface_blockers": list(surface.get("pose_surface_blockers") or []),
        }
    if isinstance(surface, tuple) and len(surface) == 2:
        return _surface_payload(
            {"joint_weight": surface[0], "rate_attack_deadzone_mask": surface[1]},
            rate_attack_deadzone_mask=rate_attack_deadzone_mask,
        )
    if isinstance(surface, tuple) and len(surface) == 3:
        meta = dict(surface[2] or {})
        meta.update({"joint_weight": surface[0], "rate_attack_deadzone_mask": surface[1]})
        return _surface_payload(meta, rate_attack_deadzone_mask=rate_attack_deadzone_mask)
    return {
        "joint_weight": surface,
        "rate_attack_deadzone_mask": rate_attack_deadzone_mask,
        "linearization_archive_sha": None,
        "evidence_scope": None,
        "target_mode": None,
        "gradient_reduction_semantics": None,
        "gradient_reduction_authority": None,
        "optimizer_update_authority": None,
        "optimizer_update_semantics": None,
        "full_video_reduction_complete": None,
        "budget_spend_authority": None,
        "pose_surface_kind": None,
        "pose_surface_authority": None,
        "pose_axis_count": None,
        "pose_inverse_variance": None,
        "pose_surface_blockers": [],
    }


def _surface_pair(surface: Any) -> tuple[Any, Any | None]:
    payload = _surface_payload(surface)
    return payload["joint_weight"], payload.get("rate_attack_deadzone_mask")


def load_joint_p18_p19_surface_file(path: str | Path) -> dict[str, Any]:
    """Load a joint surface file consumed by Z8 materializer/search CLIs."""

    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        if "joint_weight" not in data:
            raise ValueError(f"{p} must contain joint_weight")
        mask = data.get("rate_attack_deadzone_mask", None)
        return _surface_payload(
            {
                "joint_weight": data["joint_weight"],
                "rate_attack_deadzone_mask": mask,
                "linearization_archive_sha": data.get("linearization_archive_sha", None),
                "evidence_scope": data.get("evidence_scope", None),
                "target_mode": data.get("target_mode", None),
                "gradient_reduction_semantics": data.get(
                    "gradient_reduction_semantics",
                    None,
                ),
                "gradient_reduction_authority": data.get(
                    "gradient_reduction_authority",
                    None,
                ),
                "optimizer_update_authority": data.get("optimizer_update_authority", None),
                "optimizer_update_semantics": data.get("optimizer_update_semantics", None),
                "full_video_reduction_complete": data.get(
                    "full_video_reduction_complete",
                    None,
                ),
                "budget_spend_authority": data.get("budget_spend_authority", None),
                "pose_surface_kind": data.get("pose_surface_kind", None),
                "pose_surface_authority": data.get("pose_surface_authority", None),
                "pose_axis_count": data.get("pose_axis_count", None),
                "pose_inverse_variance": data.get("pose_inverse_variance", None),
            }
        )
    if p.suffix == ".npy":
        return _surface_payload(np.load(p))
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "joint_weight" not in payload:
        raise ValueError(f"{p} JSON must contain joint_weight")
    return _surface_payload(payload)


def _declared_pair_count(arr: np.ndarray | None) -> int | None:
    if arr is None:
        return None
    # Full-video surfaces are pair-indexed. Plain (H,W[,C]) maps are useful for
    # exploratory smokes but are broadcast surfaces, not full-video gradients.
    if arr.ndim in {4, 5}:
        return int(arr.shape[0])
    return None


def _full_video_surface_coverage_report(
    *,
    joint_surface: np.ndarray,
    safe_mask: np.ndarray | None,
    archive_num_pairs: int,
    max_pairs: int | None,
) -> dict[str, Any]:
    required_pair_count = int(archive_num_pairs) if max_pairs is None else min(int(archive_num_pairs), int(max_pairs))
    joint_pair_count = _declared_pair_count(joint_surface)
    safe_pair_count = _declared_pair_count(safe_mask)
    joint_ok = joint_pair_count is not None and joint_pair_count >= required_pair_count
    mask_ok = safe_mask is None or (safe_pair_count is not None and safe_pair_count >= required_pair_count)
    return {
        "schema": "z8_joint_p18_p19_full_video_surface_coverage.v1",
        "required_pair_count": required_pair_count,
        "archive_num_pairs": int(archive_num_pairs),
        "max_pairs": max_pairs,
        "joint_surface_shape": list(joint_surface.shape),
        "joint_surface_declared_pair_count": joint_pair_count,
        "rate_attack_deadzone_mask_shape": (list(safe_mask.shape) if safe_mask is not None else None),
        "rate_attack_deadzone_mask_declared_pair_count": safe_pair_count,
        "full_video_surface_coverage": bool(joint_ok and mask_ok),
        "blocker": (None if joint_ok and mask_ok else "joint_p18_p19_surface_does_not_cover_full_archive_pair_grid"),
    }


def _looks_like_sha256(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _surface_freshness_report(
    *,
    surface_payload: Mapping[str, Any],
    current_archive_sha256: str,
) -> dict[str, Any]:
    linearized_at = _surface_string(surface_payload.get("linearization_archive_sha"))
    evidence_scope = _surface_string(surface_payload.get("evidence_scope"))
    blockers: list[str] = []
    if not _looks_like_sha256(linearized_at):
        blockers.append("z8_joint_surface_linearization_archive_sha_missing")
    elif str(linearized_at).lower() != str(current_archive_sha256).lower():
        blockers.append(
            "z8_joint_surface_stale_tangent_plane:"
            f"linearized_at={linearized_at}:current_archive={current_archive_sha256}"
        )
    if evidence_scope is not None and evidence_scope != FULL_VIDEO_EVIDENCE_SCOPE:
        blockers.append(f"z8_joint_surface_not_full_video_scope:{evidence_scope}")
    return {
        "schema": "z8_joint_p18_p19_surface_freshness_report.v1",
        "current_archive_sha256": current_archive_sha256,
        "linearization_archive_sha": linearized_at,
        "evidence_scope": evidence_scope,
        "fresh_for_current_archive": not blockers,
        "blockers": blockers,
    }


def _surface_gradient_reduction_report(
    *,
    surface_payload: Mapping[str, Any],
) -> dict[str, Any]:
    semantics = _surface_string(surface_payload.get("gradient_reduction_semantics"))
    gradient_authority = _surface_bool(surface_payload.get("gradient_reduction_authority"))
    optimizer_authority = _surface_bool(surface_payload.get("optimizer_update_authority"))
    optimizer_semantics = _surface_string(surface_payload.get("optimizer_update_semantics"))
    full_video_reduction_complete = _surface_bool(surface_payload.get("full_video_reduction_complete"))
    budget_spend_authority = _surface_bool(surface_payload.get("budget_spend_authority"))
    blockers: list[str] = []
    if semantics != FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION:
        blockers.append("z8_joint_surface_gradient_reduction_not_exact_full_video_accumulation")
    if gradient_authority is not True:
        blockers.append("z8_joint_surface_gradient_reduction_authority_missing")
    if optimizer_authority is not True:
        blockers.append("z8_joint_surface_optimizer_update_authority_missing")
    if optimizer_semantics != SINGLE_UPDATE_AFTER_FULL_REDUCTION:
        blockers.append("z8_joint_surface_optimizer_update_semantics_not_single_full_reduction")
    if full_video_reduction_complete is not True:
        blockers.append("z8_joint_surface_full_video_reduction_not_complete")
    if budget_spend_authority is not True:
        blockers.append("z8_joint_surface_budget_spend_authority_missing")
    return {
        "schema": "z8_joint_p18_p19_surface_gradient_reduction_report.v1",
        "gradient_reduction_semantics": semantics,
        "gradient_reduction_authority": gradient_authority,
        "optimizer_update_authority": optimizer_authority,
        "optimizer_update_semantics": optimizer_semantics,
        "full_video_reduction_complete": full_video_reduction_complete,
        "budget_spend_authority": budget_spend_authority,
        "exact_full_video_gradient_reduction": not blockers,
        "blockers": blockers,
    }


def _surface_true_p19_report(
    *,
    surface_payload: Mapping[str, Any],
) -> dict[str, Any]:
    pose_kind = _surface_string(surface_payload.get("pose_surface_kind"))
    pose_authority = _surface_bool(surface_payload.get("pose_surface_authority"))
    pose_blockers = [str(item) for item in surface_payload.get("pose_surface_blockers") or []]
    raw_axis_count = surface_payload.get("pose_axis_count")
    raw_inverse_variance = surface_payload.get("pose_inverse_variance")
    try:
        axis_count = int(np.asarray(raw_axis_count).reshape(-1)[0]) if raw_axis_count is not None else None
    except (TypeError, ValueError, IndexError):
        axis_count = None
    inverse_variance = (
        np.asarray(raw_inverse_variance, dtype=np.float64).reshape(-1)
        if raw_inverse_variance is not None
        else np.zeros((0,), dtype=np.float64)
    )
    blockers: list[str] = []
    if pose_kind != TRUE_P19_POSE_SURFACE_KIND:
        blockers.append(f"z8_joint_surface_pose_kind_not_true_p19:{pose_kind}")
    if pose_authority is not True:
        blockers.append("z8_joint_surface_pose_authority_missing")
    if axis_count != 6:
        blockers.append(f"z8_joint_surface_pose_axis_count_not_six:{axis_count}")
    if inverse_variance.size != 6:
        blockers.append("z8_joint_surface_pose_inverse_variance_not_six_axis")
    elif not np.all(np.isfinite(inverse_variance)):
        blockers.append("z8_joint_surface_pose_inverse_variance_nonfinite")
    elif np.any(inverse_variance <= 0.0):
        blockers.append("z8_joint_surface_pose_inverse_variance_nonpositive")
    blockers.extend(pose_blockers)
    return {
        "schema": "z8_joint_p18_p19_true_pose_surface_report.v1",
        "required_pose_surface_kind": TRUE_P19_POSE_SURFACE_KIND,
        "pose_surface_kind": pose_kind,
        "pose_surface_authority": pose_authority,
        "pose_axis_count": axis_count,
        "pose_inverse_variance": [float(v) for v in inverse_variance.tolist()],
        "true_p19_pose_surface": not blockers,
        "blockers": blockers,
    }


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
    coeff32 = np.asarray(coeff, dtype=np.float32)
    if projected_joint.shape != coeff32.shape:
        raise ValueError(f"projected_joint shape {projected_joint.shape} != coeff shape {coeff32.shape}")
    water_level = float(np.quantile(projected_joint, config.joint_weight_quantile))
    eligible = projected_joint <= water_level
    if config.pose_null_required:
        eligible = np.zeros_like(eligible, dtype=bool) if projected_safe is None else eligible & (projected_safe >= 0.5)
    abs_coeff = np.abs(coeff32)
    if np.any(eligible):
        deadzone_threshold = float(np.quantile(abs_coeff[eligible], config.coefficient_deadzone_quantile))
    else:
        deadzone_threshold = 0.0
    changed = eligible & (abs_coeff <= deadzone_threshold)
    out = coeff32.copy()
    if config.quantization_step > 0.0:
        coarse = np.round(out[eligible] / config.quantization_step) * config.quantization_step
        out[eligible] = coarse
    out[changed] = 0.0
    delta = out - coeff32
    return out.astype(np.float32, copy=False), {
        "water_level": water_level,
        "deadzone_abs_threshold": deadzone_threshold,
        "eligible_coefficients": int(np.count_nonzero(eligible)),
        "dead_zoned_coefficients": int(np.count_nonzero(changed)),
        "total_coefficients": int(coeff32.size),
        "max_abs_coeff_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_coeff_delta": float(np.mean(np.abs(delta), dtype=np.float64)) if delta.size else 0.0,
    }


def _mutate_detail(
    detail: WaveletDetail2D,
    *,
    projected_joint: np.ndarray,
    projected_safe: np.ndarray | None,
    level_idx: int,
    config: Z8JointCoefficientWaterfillConfig,
) -> tuple[WaveletDetail2D, list[dict[str, Any]]]:
    mutated: dict[str, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    for subband_name in ("lh", "hl", "hh"):
        coeff = np.asarray(getattr(detail, subband_name), dtype=np.float32)
        if projected_joint.shape != coeff.shape:
            raise ValueError(f"projected_joint shape {projected_joint.shape} != coeff shape {coeff.shape}")
        if projected_safe is not None and projected_safe.shape != coeff.shape:
            raise ValueError(f"projected_safe shape {projected_safe.shape} != coeff shape {coeff.shape}")
        next_coeff, sub_stats = _quantize_selected(
            coeff,
            projected_joint=projected_joint,
            projected_safe=projected_safe,
            config=config,
        )
        mutated[subband_name] = next_coeff
        sub_stats.update(
            {
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
                first_subband = np.asarray(detail.lh, dtype=np.float32)
                projected_joint = _project_surface_to_subband(
                    joint_surface,
                    pair_idx=pair_idx,
                    frame_idx=frame_idx,
                    subband_shape=first_subband.shape,
                )
                projected_safe = (
                    _project_surface_to_subband(
                        safe_mask,
                        pair_idx=pair_idx,
                        frame_idx=frame_idx,
                        subband_shape=first_subband.shape,
                    )
                    if safe_mask is not None
                    else None
                )
                mutated, stats = _mutate_detail(
                    detail,
                    projected_joint=projected_joint,
                    projected_safe=projected_safe,
                    level_idx=level_idx,
                    config=config,
                )
                next_details.append(mutated)
                for row in stats:
                    row["pair_index"] = int(pair_idx)
                    row["frame_index"] = int(frame_idx)
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


def _identity_pair_pyramid_report(
    pair_pyramids: list[dict[str, Any]],
    *,
    max_pairs: int | None,
) -> dict[str, Any]:
    pair_limit = len(pair_pyramids) if max_pairs is None else min(len(pair_pyramids), int(max_pairs))
    subband_stats: list[dict[str, Any]] = []
    total = 0
    for pair_idx, pyramid in enumerate(pair_pyramids[:pair_limit]):
        for frame_idx, details_key in enumerate(("frame_0_details", "frame_1_details")):
            for level_idx, detail in enumerate(pyramid[details_key]):
                for subband_name in ("lh", "hl", "hh"):
                    coeff = np.asarray(getattr(detail, subband_name), dtype=np.float32)
                    count = int(coeff.size)
                    total += count
                    subband_stats.append(
                        {
                            "pair_index": int(pair_idx),
                            "frame_index": int(frame_idx),
                            "level_index": int(level_idx),
                            "subband": subband_name,
                            "water_level": None,
                            "deadzone_abs_threshold": None,
                            "eligible_coefficients": 0,
                            "dead_zoned_coefficients": 0,
                            "total_coefficients": count,
                            "max_abs_coeff_delta": 0.0,
                            "mean_abs_coeff_delta": 0.0,
                        }
                    )
    return {
        "pair_count": len(pair_pyramids),
        "mutated_pair_count": 0,
        "total_detail_coefficients": int(total),
        "eligible_coefficients": 0,
        "dead_zoned_coefficients": 0,
        "eligible_fraction": 0.0,
        "dead_zoned_fraction": 0.0,
        "subband_stats": subband_stats,
    }


def _parse_entropy_detail_step_key(key: str) -> tuple[str, int, str]:
    parts = str(key).split(":")
    if len(parts) != 3:
        raise ValueError(
            "entropy detail step keys must be frame_0_details|frame_1_details:level:lh|hl|hh"
        )
    details_key, level_text, subband = parts
    if details_key not in {"frame_0_details", "frame_1_details"}:
        raise ValueError(f"unsupported entropy detail frame key: {details_key}")
    if subband not in {"lh", "hl", "hh"}:
        raise ValueError(f"unsupported entropy detail subband key: {subband}")
    try:
        level_idx = int(level_text)
    except ValueError as exc:
        raise ValueError(f"entropy detail level must be an integer: {level_text}") from exc
    if level_idx < 0:
        raise ValueError("entropy detail level must be non-negative")
    return (details_key, level_idx, subband)


def _normalize_entropy_detail_steps(
    raw: Mapping[str, float] | None,
) -> dict[tuple[str, int, str], float] | None:
    if raw is None:
        return None
    out: dict[tuple[str, int, str], float] = {}
    for key, value in raw.items():
        step = float(value)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError(f"entropy detail step for {key!r} must be finite and positive")
        out[_parse_entropy_detail_step_key(str(key))] = step
    return out


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
    delta = after - before
    return {
        "small_receiver_distortion_measured": True,
        "receiver_tensor_values": int(before.size),
        "mse": float(np.mean(delta * delta, dtype=np.float64)) if delta.size else 0.0,
        "mae": float(np.mean(np.abs(delta), dtype=np.float64)) if delta.size else 0.0,
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _skipped_distortion_report(reason: str) -> dict[str, Any]:
    return {
        "small_receiver_distortion_measured": False,
        "receiver_tensor_values": None,
        "mse": 0.0,
        "mae": None,
        "max_abs_delta": None,
        "blocker": reason,
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


def _local_replay_objective(report: Mapping[str, Any]) -> float | None:
    for key in ("contest_action_proxy", "local_action_proxy", "score_proxy"):
        if key in report:
            try:
                value = float(report[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return None


def _local_replay_blockers(report: Mapping[str, Any]) -> list[str]:
    blockers = [str(item) for item in (report.get("blockers") or []) if str(item)]
    if report.get("replay_ok", True) is not True:
        blockers.append("full_video_local_replay_not_ok")
    if report.get("full_video_local_replay_executed") is not True:
        blockers.append("full_video_local_replay_not_executed")
    if report.get("full_video_local_replay_scope") not in {None, "full_video"}:
        blockers.append(f"full_video_local_replay_wrong_scope:{report.get('full_video_local_replay_scope')}")
    if _local_replay_objective(report) is None:
        blockers.append("full_video_local_replay_objective_missing")
    return blockers


def _inflate_runtime_benchmark_work_order(
    *,
    out_dir: Path,
    submission_dir: Path,
    config: Z8JointCoefficientWaterfillConfig | Z8JointCoefficientRelinearizationSearchConfig,
    run_label: str,
) -> dict[str, Any]:
    file_list = out_dir / f"{run_label}_inflate_benchmark_file_list.txt"
    if not file_list.exists():
        file_list.write_text("0\n", encoding="utf-8")
    output_dir = out_dir / f"{run_label}_inflate_runtime_benchmark_output"
    report_path = out_dir / f"{run_label}_inflate_runtime_benchmark.json"
    command = [
        ".venv/bin/python",
        "tools/benchmark_z8_submission_inflate_runtime.py",
        "--inflate-sh",
        (submission_dir / "inflate.sh").as_posix(),
        "--archive-dir",
        submission_dir.as_posix(),
        "--file-list",
        file_list.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--timeout-seconds",
        str(float(config.inflate_runtime_benchmark_timeout_seconds)),
        "--auth-eval-window-seconds",
        str(float(config.inflate_runtime_benchmark_auth_window_seconds)),
        "--inflate-device",
        str(config.inflate_runtime_benchmark_device),
        "--out-json",
        report_path.as_posix(),
    ]
    return {
        "schema": "z8_inflate_runtime_benchmark_work_order.v1",
        "purpose": (
            "Measure candidate receiver decode pressure through the real "
            "inflate.sh contract before exact-axis promotion."
        ),
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "run_label": run_label,
        "inflate_sh": (submission_dir / "inflate.sh").as_posix(),
        "archive_dir": submission_dir.as_posix(),
        "file_list": file_list.as_posix(),
        "output_dir": output_dir.as_posix(),
        "report_path": report_path.as_posix(),
        "command": command,
        "run_requested": bool(config.run_inflate_runtime_benchmark),
        "timeout_seconds": float(config.inflate_runtime_benchmark_timeout_seconds),
        "auth_eval_window_seconds": float(config.inflate_runtime_benchmark_auth_window_seconds),
        "inflate_device": str(config.inflate_runtime_benchmark_device),
    }


def _maybe_run_inflate_runtime_benchmark(
    work_order: Mapping[str, Any],
    *,
    config: Z8JointCoefficientWaterfillConfig | Z8JointCoefficientRelinearizationSearchConfig,
) -> dict[str, Any] | None:
    if not config.run_inflate_runtime_benchmark:
        return None
    report = benchmark_z8_submission_inflate_runtime(
        inflate_sh=Path(str(work_order["inflate_sh"])),
        archive_dir=Path(str(work_order["archive_dir"])),
        file_list=Path(str(work_order["file_list"])),
        output_dir=Path(str(work_order["output_dir"])),
        timeout_seconds=float(config.inflate_runtime_benchmark_timeout_seconds),
        auth_eval_window_seconds=float(config.inflate_runtime_benchmark_auth_window_seconds),
        inflate_device=str(config.inflate_runtime_benchmark_device),
    )
    report_path = Path(str(work_order["report_path"]))
    write_json(report_path, report)
    report["report_path"] = report_path.as_posix()
    write_json(report_path, report)
    return report


def apply_joint_p18_p19_deadzone_to_z8_archive(
    archive_bytes: bytes,
    *,
    joint_weight: Any,
    rate_attack_deadzone_mask: Any | None = None,
    config: Z8JointCoefficientWaterfillConfig | None = None,
) -> dict[str, Any]:
    """Return mutated Z8HPC1 bytes plus rate/distortion measurement metadata."""

    cfg = config or Z8JointCoefficientWaterfillConfig()
    arc = parse_archive(archive_bytes)
    current_archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if cfg.mutate_coefficients:
        surface = _surface_payload(
            joint_weight,
            rate_attack_deadzone_mask=rate_attack_deadzone_mask,
        )
        joint_surface = _normalize_surface_array(surface["joint_weight"], name="joint_weight")
        safe_mask = _as_bool_mask(surface.get("rate_attack_deadzone_mask"))
        if safe_mask is not None and safe_mask.shape != joint_surface.shape and safe_mask.size != 1:
            raise ValueError("rate_attack_deadzone_mask must match joint_weight shape or be scalar")
        coverage_report = _full_video_surface_coverage_report(
            joint_surface=joint_surface,
            safe_mask=safe_mask,
            archive_num_pairs=arc.num_pairs,
            max_pairs=cfg.max_pairs,
        )
        if cfg.require_full_video_surface_coverage and not coverage_report["full_video_surface_coverage"]:
            raise ValueError(str(coverage_report["blocker"]))
        freshness_report = _surface_freshness_report(
            surface_payload=surface,
            current_archive_sha256=current_archive_sha256,
        )
        if cfg.require_surface_archive_freshness and not freshness_report["fresh_for_current_archive"]:
            raise ValueError(",".join(str(item) for item in freshness_report["blockers"]))
        gradient_reduction_report = _surface_gradient_reduction_report(
            surface_payload=surface,
        )
        if (
            cfg.require_exact_full_video_gradient_reduction
            and not gradient_reduction_report["exact_full_video_gradient_reduction"]
        ):
            raise ValueError(",".join(str(item) for item in gradient_reduction_report["blockers"]))
        true_p19_report = _surface_true_p19_report(surface_payload=surface)
        if cfg.require_true_p19_pose_surface and not true_p19_report["true_p19_pose_surface"]:
            raise ValueError(",".join(str(item) for item in true_p19_report["blockers"]))
    else:
        joint_surface = np.zeros((), dtype=np.float64)
        safe_mask = None
        coverage_report = {
            "schema": "z8_joint_p18_p19_full_video_surface_coverage.v1",
            "required_pair_count": int(arc.num_pairs) if cfg.max_pairs is None else min(int(arc.num_pairs), int(cfg.max_pairs)),
            "archive_num_pairs": int(arc.num_pairs),
            "max_pairs": cfg.max_pairs,
            "joint_surface_shape": None,
            "joint_surface_declared_pair_count": None,
            "rate_attack_deadzone_mask_shape": None,
            "rate_attack_deadzone_mask_declared_pair_count": None,
            "full_video_surface_coverage": True,
            "blocker": None,
            "storage_only_no_surface_required": True,
        }
        freshness_report = {
            "schema": "z8_joint_p18_p19_surface_freshness_report.v1",
            "current_archive_sha256": current_archive_sha256,
            "linearization_archive_sha": None,
            "evidence_scope": "storage_only_no_surface_required",
            "fresh_for_current_archive": True,
            "blockers": [],
            "storage_only_no_surface_required": True,
        }
        true_p19_report = {
            "schema": "z8_joint_p18_p19_true_pose_surface_report.v1",
            "required_pose_surface_kind": TRUE_P19_POSE_SURFACE_KIND,
            "pose_surface_kind": "storage_only_no_surface_required",
            "pose_surface_authority": None,
            "pose_axis_count": None,
            "pose_inverse_variance": [],
            "true_p19_pose_surface": True,
            "blockers": [],
            "storage_only_no_surface_required": True,
        }
        gradient_reduction_report = {
            "schema": "z8_joint_p18_p19_surface_gradient_reduction_report.v1",
            "gradient_reduction_semantics": "storage_only_no_surface_required",
            "gradient_reduction_authority": None,
            "optimizer_update_authority": None,
            "optimizer_update_semantics": "storage_only_no_surface_required",
            "full_video_reduction_complete": None,
            "budget_spend_authority": None,
            "exact_full_video_gradient_reduction": True,
            "blockers": [],
            "storage_only_no_surface_required": True,
        }
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    if cfg.mutate_coefficients:
        mutated_pyramids, coeff_report = _mutate_pair_pyramids(
            pair_pyramids,
            joint_surface=joint_surface,
            safe_mask=safe_mask,
            config=cfg,
        )
    else:
        mutated_pyramids = pair_pyramids
        coeff_report = _identity_pair_pyramid_report(
            pair_pyramids,
            max_pairs=cfg.max_pairs,
        )
    entropy_detail_steps = _normalize_entropy_detail_steps(cfg.entropy_detail_quantization_steps)
    entropy_detail_step = (
        None
        if entropy_detail_steps is not None
        else (
            float(cfg.entropy_detail_quantization_step)
            if cfg.entropy_detail_quantization_step is not None
            else float(cfg.quantization_step)
        )
    )
    if not cfg.entropy_code_quantized_details:
        entropy_detail_step = None
        entropy_detail_steps = None
    mutated_wavelet_blob = pack_pair_pyramids_to_wavelet_blob(
        mutated_pyramids,
        detail_quantization_step=entropy_detail_step,
        detail_quantization_steps=entropy_detail_steps,
        detail_lossless_preconditioner=bool(cfg.lossless_brotli_precondition_details),
    )
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
                "mutate_coefficients": bool(cfg.mutate_coefficients),
                "require_full_video_surface_coverage": bool(cfg.require_full_video_surface_coverage),
                "entropy_code_quantized_details": bool(cfg.entropy_code_quantized_details),
                "entropy_detail_quantization_step": entropy_detail_step,
                "entropy_detail_quantization_steps": dict(cfg.entropy_detail_quantization_steps or {}),
                "lossless_brotli_precondition_details": bool(cfg.lossless_brotli_precondition_details),
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
    before_detail_codec_summary = summarize_wavelet_blob_detail_codecs(arc.wavelet_coeffs_blob)
    after_detail_codec_summary = summarize_wavelet_blob_detail_codecs(after_arc.wavelet_coeffs_blob)
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
        "entropy_code_quantized_details": bool(cfg.entropy_code_quantized_details),
        "entropy_detail_quantization_step": entropy_detail_step,
        "entropy_detail_quantization_steps": dict(cfg.entropy_detail_quantization_steps or {}),
        "lossless_brotli_precondition_details": bool(cfg.lossless_brotli_precondition_details),
        "mutate_coefficients": bool(cfg.mutate_coefficients),
        "before_detail_codec_summary": before_detail_codec_summary,
        "after_detail_codec_summary": after_detail_codec_summary,
    }
    if cfg.measure_receiver_distortion:
        distortion = _distortion_report(archive_bytes, mutated_archive)
    else:
        distortion = _skipped_distortion_report("receiver_mse_proxy_skipped_by_full_video_local_replay")
    return {
        "schema": Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA,
        "role": Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
        "mutated_archive_bytes": mutated_archive,
        "original_archive_sha256": current_archive_sha256,
        "mutated_archive_sha256": hashlib.sha256(mutated_archive).hexdigest(),
        "coefficient_report": coeff_report,
        "rate_report": rate_report,
        "distortion_report": distortion,
        "full_video_surface_coverage_report": coverage_report,
        "surface_freshness_report": freshness_report,
        "surface_gradient_reduction_report": gradient_reduction_report,
        "surface_true_p19_report": true_p19_report,
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
    inflate_runtime_benchmark_work_order: dict[str, Any] | None = None
    inflate_runtime_benchmark_report: dict[str, Any] | None = None
    if cfg.emit_archive_zip:
        archive_zip_path, archive_sha256, archive_zip_bytes = export_z8hpc1_archive_bytes(
            mutated_archive,
            out_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=cfg.emit_receiver_proof,
            emit_byte_mutation_proof=False,
            emit_runtime_payload_bridge_report=cfg.emit_receiver_proof,
            retain_receiver_proof_output=False,
        )
        if cfg.emit_inflate_runtime_benchmark_work_order:
            inflate_runtime_benchmark_work_order = _inflate_runtime_benchmark_work_order(
                out_dir=out_dir,
                submission_dir=out_dir / "submission",
                config=cfg,
                run_label="z8_joint_p18_p19_deadzone",
            )
            inflate_runtime_benchmark_report = _maybe_run_inflate_runtime_benchmark(
                inflate_runtime_benchmark_work_order,
                config=cfg,
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
        "inflate_runtime_benchmark_work_order": inflate_runtime_benchmark_work_order,
        "inflate_runtime_benchmark_report": inflate_runtime_benchmark_report,
        "inflate_runtime_benchmark_executed": inflate_runtime_benchmark_report is not None,
        "exact_axis_blocker": (
            "contest_cpu_cuda_eval_not_executed"
            if cfg.emit_receiver_proof
            else "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
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
    surfaces: Sequence[Any] | None = None,
    surface_provider: Callable[[int, bytes], Any] | None = None,
    local_replay_evaluator: Callable[..., Mapping[str, Any]] | None = None,
    config: Z8JointCoefficientRelinearizationSearchConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded iterative Z8 dead-zone search over fresh joint surfaces."""

    cfg = config or Z8JointCoefficientRelinearizationSearchConfig()
    if surface_provider is None and not surfaces:
        raise ValueError("surfaces must not be empty unless surface_provider is provided")
    original = bytes(archive_bytes)
    current = original
    seen_surface_digests: set[str] = set()
    accepted: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    previous_cumulative_mse = 0.0
    final_blocker: str | None = None

    for iteration_index in range(cfg.max_iterations):
        if surface_provider is not None:
            raw_surface = surface_provider(iteration_index, current)
        else:
            assert surfaces is not None
            if iteration_index >= len(surfaces):
                final_blocker = f"surface_sequence_exhausted_before_max_iterations:iteration={iteration_index}"
                break
            raw_surface = surfaces[iteration_index]
        surface = _surface_payload(raw_surface)
        digest = _surface_digest(
            surface["joint_weight"],
            surface.get("rate_attack_deadzone_mask"),
            linearization_archive_sha=surface.get("linearization_archive_sha"),
        )
        if cfg.require_fresh_surface_per_iteration and digest in seen_surface_digests:
            raise ValueError(
                "fresh surface required for iterative relinearization; duplicate "
                f"surface digest at iteration {iteration_index}: {digest}"
            )
        seen_surface_digests.add(digest)

        iteration_candidates: list[dict[str, Any]] = []
        measure_receiver_proxy = bool(cfg.measure_receiver_mse_proxy) and not (
            local_replay_evaluator is not None
            and cfg.skip_receiver_mse_proxy_when_full_video_replay
            and cfg.max_cumulative_mse is None
        )
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
                        require_full_video_surface_coverage=bool(cfg.require_full_video_surface_coverage),
                        require_surface_archive_freshness=bool(cfg.require_surface_archive_freshness),
                        require_exact_full_video_gradient_reduction=bool(
                            cfg.require_exact_full_video_gradient_reduction
                        ),
                        measure_receiver_distortion=measure_receiver_proxy,
                        mutate_coefficients=bool(cfg.mutate_coefficients),
                        entropy_code_quantized_details=bool(cfg.entropy_code_quantized_details),
                        entropy_detail_quantization_step=cfg.entropy_detail_quantization_step,
                        entropy_detail_quantization_steps=cfg.entropy_detail_quantization_steps,
                        lossless_brotli_precondition_details=bool(cfg.lossless_brotli_precondition_details),
                    )
                    result = apply_joint_p18_p19_deadzone_to_z8_archive(
                        current,
                        joint_weight=surface,
                        config=waterfill_cfg,
                    )
                    mutated = bytes(result["mutated_archive_bytes"])
                    if measure_receiver_proxy:
                        cumulative_distortion = _distortion_report(original, mutated)
                    else:
                        cumulative_distortion = _skipped_distortion_report(
                            "cumulative_receiver_mse_proxy_skipped_by_full_video_local_replay"
                        )
                    cumulative_mse = float(cumulative_distortion.get("mse", float("inf")))
                    acceptance_blockers: list[str] = []
                    if measure_receiver_proxy:
                        guard_ok = cfg.max_cumulative_mse is None or (
                            cumulative_distortion.get("small_receiver_distortion_measured") is True
                            and cumulative_mse <= float(cfg.max_cumulative_mse)
                        )
                    elif local_replay_evaluator is not None:
                        guard_ok = True
                    elif not cfg.mutate_coefficients:
                        guard_ok = True
                        acceptance_blockers.append("rate_only_storage_probe_no_receiver_proxy")
                    else:
                        guard_ok = False
                        acceptance_blockers.append(
                            "mutating_candidate_requires_receiver_proxy_or_full_video_replay"
                        )
                    candidate_proxy_objective = _candidate_proxy_objective(
                        result=result,
                        cumulative_distortion=cumulative_distortion,
                        previous_cumulative_mse=previous_cumulative_mse,
                        config=cfg,
                    )
                    objective = candidate_proxy_objective
                    objective_source = "receiver_mse_rate_proxy"
                    if local_replay_evaluator is not None and not measure_receiver_proxy:
                        objective_source = "rate_proxy_prefilter_before_full_video_local_replay"
                    elif not measure_receiver_proxy and not cfg.mutate_coefficients:
                        objective_source = "rate_only_storage_probe"
                    elif not measure_receiver_proxy:
                        objective_source = "rate_only_mutation_probe_blocked"
                    candidate_metadata = {
                        "joint_weight_quantile": float(joint_q),
                        "coefficient_deadzone_quantile": float(deadzone_q),
                        "quantization_step": float(q_step),
                        "mutated_archive_sha256": result["mutated_archive_sha256"],
                    }
                    row = {
                        "schema": "z8_joint_p18_p19_relinearized_candidate.v1",
                        "iteration_index": int(iteration_index),
                        "surface_digest": digest,
                        "joint_weight_quantile": float(joint_q),
                        "coefficient_deadzone_quantile": float(deadzone_q),
                        "quantization_step": float(q_step),
                        "proxy_objective": float(objective),
                        "receiver_proxy_objective": float(candidate_proxy_objective),
                        "objective_source": objective_source,
                        "guard_ok": bool(guard_ok),
                        "acceptance_blockers": acceptance_blockers,
                        "full_video_local_replay_required": local_replay_evaluator is not None,
                        "full_video_local_replay_prefilter_top_k": cfg.local_replay_prefilter_top_k,
                        "full_video_local_replay_report": None,
                        "full_video_local_replay_blockers": [],
                        "hard_archive_projection_replay_executed": False,
                        "receiver_mse_proxy_measured": bool(measure_receiver_proxy),
                        "rate_report": result["rate_report"],
                        "incremental_distortion_report": result["distortion_report"],
                        "cumulative_distortion_report": cumulative_distortion,
                        "full_video_surface_coverage_report": result["full_video_surface_coverage_report"],
                        "surface_freshness_report": result["surface_freshness_report"],
                        "surface_gradient_reduction_report": result["surface_gradient_reduction_report"],
                        "surface_true_p19_report": result["surface_true_p19_report"],
                        "coefficient_summary": {
                            key: value for key, value in result["coefficient_report"].items() if key != "subband_stats"
                        },
                        "mutated_archive_sha256": result["mutated_archive_sha256"],
                        "_candidate_metadata": candidate_metadata,
                        "_mutated_archive_bytes": mutated,
                        **FALSE_AUTHORITY,
                    }
                    iteration_candidates.append(row)
        if local_replay_evaluator is not None:
            replay_candidates = [
                (idx, row)
                for idx, row in enumerate(iteration_candidates)
                if row.get("guard_ok") is True
            ]
            if cfg.local_replay_prefilter_top_k is None:
                replay_indices = {idx for idx, _row in replay_candidates}
            else:
                replay_indices = {
                    idx
                    for idx, _row in sorted(
                        replay_candidates,
                        key=lambda item: float(item[1].get("receiver_proxy_objective", float("inf"))),
                    )[: int(cfg.local_replay_prefilter_top_k)]
                }
            for idx, row in enumerate(iteration_candidates):
                if idx not in replay_indices:
                    row["full_video_local_replay_blockers"] = [
                        "full_video_local_replay_skipped_by_prefilter"
                    ]
                    row["guard_ok"] = False
                    continue
                local_replay_report = dict(
                    local_replay_evaluator(
                        candidate_archive_bytes=bytes(row["_mutated_archive_bytes"]),
                        source_archive_bytes=original,
                        current_archive_bytes=current,
                        iteration_index=iteration_index,
                        candidate_index=idx,
                        candidate_metadata=dict(row["_candidate_metadata"]),
                    )
                )
                local_replay_blockers = _local_replay_blockers(local_replay_report)
                local_objective = _local_replay_objective(local_replay_report)
                if local_objective is not None:
                    row["proxy_objective"] = float(local_objective)
                    row["objective_source"] = "full_video_local_replay"
                    row["full_video_local_replay_report"] = local_replay_report
                    row["full_video_local_replay_blockers"] = local_replay_blockers
                    row["acceptance_blockers"] = list(row.get("acceptance_blockers") or []) + local_replay_blockers
                    row["hard_archive_projection_replay_executed"] = True
                    row["guard_ok"] = bool(row["guard_ok"] and not local_replay_blockers)
        for row in iteration_candidates:
            mutated = bytes(row.pop("_mutated_archive_bytes"))
            row.pop("_candidate_metadata", None)
            all_candidates.append(row)
            if row.get("guard_ok") is True and (best is None or row["proxy_objective"] < best["proxy_objective"]):
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
        "full_video_surface_contract": ("joint_and_pose_null_surfaces_must_cover_the_full_archive_pair_grid"),
        "pose_guard": "pose_null_mask_and_mahalanobis_ail_weights_consumed",
        "interaction_penalty": ("penalize_cumulative_mse_increase_between_relinearization_steps"),
        "accept_reject_authority": (
            "full_video_local_replay_when_local_replay_evaluator_is_provided_else_receiver_mse_proxy"
        ),
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
    surfaces: Sequence[Any] | None = None,
    surface_provider: Callable[[int, bytes], Any] | None = None,
    local_replay_evaluator: Callable[..., Mapping[str, Any]] | None = None,
    config: Z8JointCoefficientRelinearizationSearchConfig | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write the accepted iterative Z8 dead-zone candidate and manifest."""

    cfg = config or Z8JointCoefficientRelinearizationSearchConfig()
    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=surfaces,
        surface_provider=surface_provider,
        local_replay_evaluator=local_replay_evaluator,
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
    inflate_runtime_benchmark_work_order: dict[str, Any] | None = None
    inflate_runtime_benchmark_report: dict[str, Any] | None = None
    if cfg.emit_archive_zip:
        archive_zip_path, archive_sha256, archive_zip_bytes = export_z8hpc1_archive_bytes(
            final_archive,
            out_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=cfg.emit_receiver_proof,
            emit_byte_mutation_proof=False,
            emit_runtime_payload_bridge_report=cfg.emit_receiver_proof,
            retain_receiver_proof_output=False,
        )
        if cfg.emit_inflate_runtime_benchmark_work_order:
            inflate_runtime_benchmark_work_order = _inflate_runtime_benchmark_work_order(
                out_dir=out_dir,
                submission_dir=out_dir / "submission",
                config=cfg,
                run_label="z8_joint_p18_p19_relinearized_search",
            )
            inflate_runtime_benchmark_report = _maybe_run_inflate_runtime_benchmark(
                inflate_runtime_benchmark_work_order,
                config=cfg,
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
            "inflate_runtime_benchmark_work_order": inflate_runtime_benchmark_work_order,
            "inflate_runtime_benchmark_report": inflate_runtime_benchmark_report,
            "inflate_runtime_benchmark_executed": inflate_runtime_benchmark_report is not None,
            "exact_axis_blocker": (
                "contest_cpu_cuda_eval_not_executed"
                if cfg.emit_receiver_proof
                else "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
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
    "FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION",
    "SINGLE_UPDATE_AFTER_FULL_REDUCTION",
    "Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE",
    "Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA",
    "Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA",
    "Z8_JOINT_COEFFICIENT_WATERFILL_SCHEMA",
    "Z8JointCoefficientRelinearizationSearchConfig",
    "Z8JointCoefficientWaterfillConfig",
    "apply_joint_p18_p19_deadzone_to_z8_archive",
    "load_joint_p18_p19_surface_file",
    "materialize_joint_p18_p19_deadzone_candidate",
    "materialize_joint_p18_p19_relinearized_deadzone_search",
    "run_joint_p18_p19_relinearized_deadzone_search",
]
