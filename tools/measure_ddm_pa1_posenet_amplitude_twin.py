#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the PoseNet amplitude twin on the strict DDM E2 n600 endpoint.

This runner is deliberately local-only.  It:

1. opens the SHA-bound E2 raw camera pairs produced by the strict runtime;
2. measures official post-resize, post-``rgb_to_yuv6`` input moments against
   the canonical GT cache;
3. compares a COUNTED GT-stat target with a candidate-FREE target derived only
   from frozen scorer weights and the existing candidate at decode time;
4. starts with frame 0, which is structurally Seg-free, and measures both
   frozen PoseNet and SegNet on all 600 pairs; and
5. conditionally measures one joint-frame placement rung when frame 0 lowers
   d_pose and the pre-registered amplitude falsifier has not fired.

The output is ``[macOS-CPU frozen-scorer advisory]`` research evidence only.
It is not a contest score and cannot move the frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    open_stored_npy_memmap,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    SOURCE_BYTES,
    DirectDescriptionError,
    _read_regular_file_once,
)
from tools.measure_ddm_v14_realization_fidelity import _load_models  # noqa: E402
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    DDMV15ScorerSolvedTemplateConfigV1,
)

SCHEMA = "ddm_pa1_posenet_amplitude_twin_receipt.v1"
CONFIG_SCHEMA = "DDMPA1PoseNetAmplitudeTwinConfigV1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
LANE_ID = "lane_ddm_pa1_posenet_amplitude_twin_20260723"
POINTER = "0.1910828242 [contest-CPU]"
PAIR_COUNT = 600
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
CHANNEL_NAMES = (
    "t0_y00",
    "t0_y10",
    "t0_y01",
    "t0_y11",
    "t0_u",
    "t0_v",
    "t1_y00",
    "t1_y10",
    "t1_y01",
    "t1_y11",
    "t1_u",
    "t1_v",
)
ATLAS_STEM_LAYERS = (
    "vision.stem.0.conv_kxk.0.bn",
    "vision.stem.0.conv_scale.bn",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256(_read_regular_file_once(path))


def _stream_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = rfc8785_canonicalize(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


class DDMPA1PoseNetAmplitudeTwinConfigV1(BaseModel):
    """SHA-bound typed configuration for the local PA1 measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMPA1PoseNetAmplitudeTwinConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    e2_receipt_path: str = Field(min_length=1)
    e2_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e2_raw_path: str = Field(min_length=1)
    e2_raw_bytes: Literal[3_662_409_600] = 3_662_409_600
    e2_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e2_archive_path: str = Field(min_length=1)
    e2_archive_bytes: Literal[343_466] = 343_466
    e2_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_config_path: str = Field(min_length=1)
    scorer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    atlas_manifest_path: str = Field(min_length=1)
    atlas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_modules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_evaluate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_frame_utils_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    posenet_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segnet_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_count: Literal[600] = PAIR_COUNT
    batch_size: Literal[16] = 16
    frame0_gt_payload_bytes: Literal[24] = 24
    joint_gt_payload_bytes: Literal[48] = 48
    scorer_target_payload_bytes: Literal[0] = 0
    amplitude_equivalence_margin: Literal[0.1] = 0.1
    required_free_bytes: int = Field(default=134_217_728, ge=1)
    output_dir: str = Field(min_length=1)
    research_only: Literal[True] = True
    execution_allowed: Literal[True] = True
    paid_or_remote_allowed: Literal[False] = False
    contest_eval_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed(self) -> DDMPA1PoseNetAmplitudeTwinConfigV1:
        if self.frame0_gt_payload_bytes != 6 * 2 * 2:
            raise ValueError("frame-0 GT payload must be 6 gain/bias fp16 pairs")
        if self.joint_gt_payload_bytes != len(CHANNEL_NAMES) * 2 * 2:
            raise ValueError("joint GT payload must be 12 gain/bias fp16 pairs")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _bound(path_text: str, expected: str, label: str) -> bytes:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = _read_regular_file_once(path)
    observed = _sha256(payload)
    if observed != expected:
        raise DirectDescriptionError(f"{label} SHA-256 differs: {observed} != {expected}")
    return payload


def _moment_state(channels: int) -> dict[str, Any]:
    return {
        "count": 0,
        "sum": np.zeros(channels, dtype=np.float64),
        "sum_sq": np.zeros(channels, dtype=np.float64),
    }


def _moment_update(state: dict[str, Any], tensor: Any) -> None:
    import torch

    if tensor.ndim != 4:
        raise DirectDescriptionError("moment tensor must be [N,C,H,W]")
    state["count"] += int(tensor.shape[0] * tensor.shape[2] * tensor.shape[3])
    state["sum"] += tensor.sum(dim=(0, 2, 3), dtype=torch.float64).cpu().numpy()
    state["sum_sq"] += tensor.square().sum(dim=(0, 2, 3), dtype=torch.float64).cpu().numpy()


def _moment_merge(state: dict[str, Any], payload: Mapping[str, Any]) -> None:
    state["count"] += int(payload["count"])
    state["sum"] += np.asarray(payload["sum"], dtype=np.float64)
    state["sum_sq"] += np.asarray(payload["sum_sq"], dtype=np.float64)


def _moment_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": int(state["count"]),
        "sum": np.asarray(state["sum"], dtype=np.float64).tolist(),
        "sum_sq": np.asarray(state["sum_sq"], dtype=np.float64).tolist(),
    }


def _moment_finish(state: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["count"])
    if count <= 0:
        raise DirectDescriptionError("cannot finish empty moments")
    mean = np.asarray(state["sum"], dtype=np.float64) / count
    variance = np.maximum(
        np.asarray(state["sum_sq"], dtype=np.float64) / count - mean * mean,
        0.0,
    )
    return {
        "count_per_channel": count,
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "std": np.sqrt(variance).tolist(),
    }


def _fit_affine(
    source_moments: Mapping[str, Any],
    target_moments: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit moment matching and quantize the exact stored parameters to fp16."""

    target = _moment_finish(target_moments)
    return _fit_affine_to_target(
        source_moments,
        np.asarray(target["mean"], dtype=np.float64),
        np.asarray(target["variance"], dtype=np.float64),
        quantize_fp16=True,
    )


def _fit_affine_to_target(
    source_moments: Mapping[str, Any],
    target_mean: np.ndarray,
    target_variance: np.ndarray,
    *,
    quantize_fp16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a channel affine to explicit target moments."""

    source = _moment_finish(source_moments)
    source_mean = np.asarray(source["mean"], dtype=np.float64)
    source_var = np.asarray(source["variance"], dtype=np.float64)
    target_mean = np.asarray(target_mean, dtype=np.float64)
    target_var = np.asarray(target_variance, dtype=np.float64)
    if target_mean.shape != source_mean.shape or target_var.shape != source_var.shape:
        raise DirectDescriptionError("affine target channel geometry differs")
    if np.any(source_var <= 1e-12):
        raise DirectDescriptionError("source YUV6 channel variance is degenerate")
    gain = np.sqrt(target_var / source_var)
    bias = target_mean - gain * source_mean
    if quantize_fp16:
        gain_q = gain.astype("<f2").astype(np.float32)
        bias_q = bias.astype("<f2").astype(np.float32)
    else:
        gain_q = gain.astype(np.float32)
        bias_q = bias.astype(np.float32)
    if not np.all(np.isfinite(gain_q)) or not np.all(np.isfinite(bias_q)):
        raise DirectDescriptionError("affine is non-finite")
    return gain_q, bias_q


def _encode_affine(gain: np.ndarray, bias: np.ndarray) -> bytes:
    gain = np.asarray(gain)
    bias = np.asarray(bias)
    if gain.shape != bias.shape or gain.shape[-1] != len(CHANNEL_NAMES):
        raise DirectDescriptionError("affine shape must end in 12 channels")
    return gain.astype("<f2").tobytes() + bias.astype("<f2").tobytes()


def _encode_frame0_affine(gain: np.ndarray, bias: np.ndarray) -> bytes:
    gain = np.asarray(gain)
    bias = np.asarray(bias)
    if gain.shape != (12,) or bias.shape != (12,):
        raise DirectDescriptionError("frame-0 affine source must have 12 channels")
    return gain[:6].astype("<f2").tobytes() + bias[:6].astype("<f2").tobytes()


def _normalized_moments(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _moment_finish(raw)
    mean = (np.asarray(value["mean"]) - 127.5) / 63.75
    variance = np.asarray(value["variance"]) / (63.75 * 63.75)
    return {
        **value,
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "std": np.sqrt(variance).tolist(),
        "normalization": "(raw_yuv6 - 127.5) / 63.75",
    }


def _inverse_yuv6(yuv: Any) -> Any:
    """Deterministic block-constant inverse used by the camera-side realizer."""

    import torch

    if yuv.ndim != 5 or yuv.shape[2] != 6:
        raise DirectDescriptionError("inverse YUV6 expects [B,T,6,H,W]")
    b, t, _c, h, w = yuv.shape
    y = torch.empty((b, t, h * 2, w * 2), dtype=yuv.dtype, device=yuv.device)
    y[:, :, 0::2, 0::2] = yuv[:, :, 0]
    y[:, :, 1::2, 0::2] = yuv[:, :, 1]
    y[:, :, 0::2, 1::2] = yuv[:, :, 2]
    y[:, :, 1::2, 1::2] = yuv[:, :, 3]
    u = yuv[:, :, 4].repeat_interleave(2, -2).repeat_interleave(2, -1)
    v = yuv[:, :, 5].repeat_interleave(2, -2).repeat_interleave(2, -1)
    red = y + 1.402 * (v - 128.0)
    blue = y + 1.772 * (u - 128.0)
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    return torch.stack((red, green, blue), dim=2).clamp(0.0, 255.0)


def _realize_affine(
    camera_pairs: np.ndarray,
    pair_ids: Sequence[int],
    gain: np.ndarray,
    bias: np.ndarray,
    posenet: Any,
    *,
    placement: Literal["frame0", "joint"],
) -> np.ndarray:
    """Realize scorer-coordinate affine as a shared camera-RGB residual."""

    import torch
    import torch.nn.functional as functional

    value = np.asarray(camera_pairs)
    if value.dtype != np.uint8 or value.shape[1:] != (2, *CAMERA_HW, 3):
        raise DirectDescriptionError("camera affine requires uint8 [B,2,874,1164,3]")
    if len(pair_ids) != value.shape[0]:
        raise DirectDescriptionError("pair ids and camera batch differ")
    gain = np.asarray(gain, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if gain.shape != (1, 12) or bias.shape != gain.shape:
        raise DirectDescriptionError("affine geometry differs")

    tensor = torch.from_numpy(np.array(value, copy=True, order="C")).permute(0, 1, 4, 2, 3).contiguous().float()
    raw_yuv = _official_pose_preprocess(tensor).reshape(value.shape[0], 2, 6, SCORER_HW[0] // 2, SCORER_HW[1] // 2)
    gain_batch = torch.from_numpy(np.repeat(gain, value.shape[0], axis=0)).reshape(value.shape[0], 2, 6, 1, 1)
    bias_batch = torch.from_numpy(np.repeat(bias, value.shape[0], axis=0)).reshape(value.shape[0], 2, 6, 1, 1)
    corrected_yuv = (raw_yuv * gain_batch + bias_batch).clamp(0.0, 255.0)
    corrected_low = _inverse_yuv6(corrected_yuv).reshape(-1, 3, *SCORER_HW)
    baseline_low = _inverse_yuv6(raw_yuv).reshape(-1, 3, *SCORER_HW)
    residual_low = (corrected_low - baseline_low).reshape(value.shape[0], 2, 3, *SCORER_HW)
    if placement == "frame0":
        residual_low[:, 1].zero_()
    residual_low = residual_low.reshape(-1, 3, *SCORER_HW)
    residual_camera = functional.interpolate(residual_low, size=CAMERA_HW, mode="bilinear").reshape(
        value.shape[0], 2, 3, *CAMERA_HW
    )
    corrected_camera = (tensor + residual_camera).clamp(0.0, 255.0).round()
    return np.ascontiguousarray(corrected_camera.to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy())


def _official_pose_preprocess(tensor: Any) -> Any:
    """Call the pinned upstream resize + ``rgb_to_yuv6`` path literally."""

    import einops
    import torch.nn.functional as functional
    from frame_utils import rgb_to_yuv6
    from modules import segnet_model_input_size

    batch_size, sequence_length = tensor.shape[:2]
    flat = einops.rearrange(
        tensor,
        "b t c h w -> (b t) c h w",
        b=batch_size,
        t=sequence_length,
        c=3,
    )
    resized = functional.interpolate(
        flat,
        size=(segnet_model_input_size[1], segnet_model_input_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    yuv = rgb_to_yuv6(resized)
    return einops.rearrange(
        yuv,
        "(b t) c h w -> b (t c) h w",
        b=batch_size,
        t=sequence_length,
        c=6,
    ).contiguous()


def _official_pose_input(camera_pairs: np.ndarray, posenet: Any) -> tuple[Any, Any]:
    import torch

    value = np.asarray(camera_pairs)
    tensor = torch.from_numpy(np.array(value, copy=True, order="C")).permute(0, 1, 4, 2, 3).contiguous().float()
    raw = _official_pose_preprocess(tensor)
    normalized = (raw - posenet._mean) / posenet._std
    return raw, normalized


def _stem_branch_modules(posenet: Any) -> dict[str, Any]:
    modules = dict(posenet.named_modules())
    result = {}
    for layer_id in ATLAS_STEM_LAYERS:
        branch = layer_id.removesuffix(".bn")
        module = modules.get(f"{branch}.conv")
        if module is None:
            raise DirectDescriptionError(f"PoseNet stem branch absent under pinned inventory: {branch}.conv")
        result[layer_id] = module
    return result


def _derive_scorer_only_input_target(
    posenet: Any,
    atlas: Mapping[str, Any],
) -> dict[str, Any]:
    """Invert first-stem BN moments to a scorer-only 12-channel target.

    The least-squares system contains only frozen convolution weights, frozen
    BN running statistics, scorer geometry, and fixed clipping bounds.  No GT
    or candidate-video statistic enters the target.  At decode time the
    existing candidate's moments may be used to compute the affine that reaches
    this fixed target; that computation does not introduce a new video fact.
    """

    import torch
    import torch.nn.functional as functional
    from scipy.optimize import lsq_linear

    mean_rows = []
    mean_rhs = []
    variance_rows = []
    variance_rhs = []
    derivation_rows = []
    height, width = SCORER_HW[0] // 2, SCORER_HW[1] // 2
    for layer_id, convolution in _stem_branch_modules(posenet).items():
        factor = atlas["stem_factors"][layer_id]["payload"]
        running_mean = np.asarray(factor["running_mean"], dtype=np.float64)
        running_variance = np.asarray(factor["running_variance"], dtype=np.float64)
        weight = convolution.weight.detach().to(dtype=torch.float64)
        bias = (
            np.zeros(weight.shape[0], dtype=np.float64)
            if convolution.bias is None
            else convolution.bias.detach().cpu().numpy().astype(np.float64)
        )
        coefficient_mean = np.empty((weight.shape[0], weight.shape[1]))
        coefficient_variance = np.empty_like(coefficient_mean)
        for channel in range(weight.shape[1]):
            unit = torch.zeros((1, weight.shape[1], height, width), dtype=torch.float64)
            unit[:, channel] = 1.0
            with torch.inference_mode():
                response = functional.conv2d(
                    unit,
                    weight,
                    bias=None,
                    stride=convolution.stride,
                    padding=convolution.padding,
                    dilation=convolution.dilation,
                    groups=convolution.groups,
                )
                variance_response = functional.conv2d(
                    unit,
                    weight.square(),
                    bias=None,
                    stride=convolution.stride,
                    padding=convolution.padding,
                    dilation=convolution.dilation,
                    groups=convolution.groups,
                )
            coefficient_mean[:, channel] = response.mean(dim=(0, 2, 3)).cpu().numpy()
            coefficient_variance[:, channel] = variance_response.mean(dim=(0, 2, 3)).cpu().numpy()
        scale = np.sqrt(running_variance + float(factor["epsilon"]))
        mean_rows.append(coefficient_mean / scale[:, None])
        mean_rhs.append((running_mean - bias) / scale)
        variance_rows.append(coefficient_variance / running_variance[:, None])
        variance_rhs.append(np.ones_like(running_variance))
        derivation_rows.append(
            {
                "layer_id": layer_id,
                "factor_id": atlas["stem_factors"][layer_id]["factor_id"],
                "mean_matrix_shape": list(coefficient_mean.shape),
                "variance_matrix_shape": list(coefficient_variance.shape),
            }
        )
    mean_matrix = np.concatenate(mean_rows, axis=0)
    variance_matrix = np.concatenate(variance_rows, axis=0)
    normalized_mean = np.linalg.lstsq(mean_matrix, np.concatenate(mean_rhs), rcond=None)[0]
    normalized_mean = np.clip(normalized_mean, -2.0, 2.0)
    variance_solution = lsq_linear(
        variance_matrix,
        np.concatenate(variance_rhs),
        bounds=(1e-4, 16.0),
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=1000,
    )
    if not variance_solution.success:
        raise DirectDescriptionError(f"scorer-only variance inverse failed: {variance_solution.message}")
    normalized_variance = variance_solution.x
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_residual = mean_matrix @ normalized_mean - np.concatenate(mean_rhs)
        variance_residual = variance_matrix @ normalized_variance - np.concatenate(variance_rhs)
    if not np.all(np.isfinite(mean_residual)) or not np.all(np.isfinite(variance_residual)):
        raise DirectDescriptionError("scorer-only inverse residual is non-finite")
    return {
        "normalized_mean": normalized_mean.tolist(),
        "normalized_variance": normalized_variance.tolist(),
        "raw_mean": (normalized_mean * 63.75 + 127.5).tolist(),
        "raw_variance": (normalized_variance * (63.75**2)).tolist(),
        "mean_weighted_residual_rms": float(np.sqrt(np.mean(mean_residual * mean_residual))),
        "variance_relative_residual_rms": float(np.sqrt(np.mean(variance_residual * variance_residual))),
        "variance_solver": {
            "name": "scipy.optimize.lsq_linear",
            "bounds": [1e-4, 16.0],
            "status": int(variance_solution.status),
            "optimality": float(variance_solution.optimality),
        },
        "derivation": derivation_rows,
        "rate_partition": "FREE",
        "video_derived_target_values": False,
        "decode_rule": (
            "derive candidate moments from already-counted decoded frames; "
            "map them to these scorer-only constants; store no affine payload"
        ),
    }


def _input_and_stem_moments(
    camera_pairs: np.ndarray,
    posenet: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    raw, normalized = _official_pose_input(camera_pairs, posenet)
    input_state = _moment_state(12)
    _moment_update(input_state, raw)
    branches: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for layer_id, convolution in _stem_branch_modules(posenet).items():
            state = _moment_state(int(convolution.out_channels))
            _moment_update(state, convolution(normalized))
            branches[layer_id] = state
    return input_state, branches


def _score_batch(
    camera_pairs: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    import torch

    value = np.asarray(camera_pairs)
    tensor = torch.from_numpy(np.array(value, copy=True, order="C")).permute(0, 1, 4, 2, 3).contiguous().float()
    raw = _official_pose_preprocess(tensor)
    normalized = (raw - posenet._mean) / posenet._std
    input_state = _moment_state(12)
    _moment_update(input_state, raw)
    branch_states: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for layer_id, convolution in _stem_branch_modules(posenet).items():
            state = _moment_state(int(convolution.out_channels))
            _moment_update(state, convolution(normalized))
            branch_states[layer_id] = state
        pose_output = posenet(raw)
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
        cells = segnet(segnet.preprocess_input(tensor)).argmax(dim=1).cpu().numpy().astype(np.uint8)
    labels = np.asarray(labels)
    poses = np.asarray(poses, dtype=np.float64)
    return {
        "errors": int(np.count_nonzero(cells != labels)),
        "sites": int(cells.size),
        "pose_squared_error_sum": float(np.square(pose6 - poses).sum(dtype=np.float64)),
        "pose_coordinates": int(pose6.size),
        "cells_sha256": _sha256_array(cells),
        "pose6_sha256": _sha256_array(pose6),
        "input_moments": _moment_payload(input_state),
        "stem_pre_bn_moments": {key: _moment_payload(state) for key, state in branch_states.items()},
    }


def _load_atlas(config: DDMPA1PoseNetAmplitudeTwinConfigV1) -> dict[str, Any]:
    manifest_path = Path(config.atlas_manifest_path)
    manifest = json.loads(
        _bound(
            str(manifest_path),
            config.atlas_manifest_sha256,
            "AT1x atlas manifest",
        )
    )
    if manifest.get("schema") != "ddm_at1x_atlas_materialization.v1":
        raise DirectDescriptionError("AT1x atlas manifest schema differs")
    closed_path = manifest_path.parent / "stage_closed_forms.json"
    closed = json.loads(_read_regular_file_once(closed_path))
    rows = {}
    for row in closed["factors"]:
        if (
            row.get("network") == "posenet"
            and row.get("factor_kind") == "closed_form.batchnorm_expected_stats"
            and row.get("layer_id") in ATLAS_STEM_LAYERS
        ):
            shard = Path(row["shard"]["path"])
            if _sha256_path(shard) != row["shard"]["sha256"]:
                raise DirectDescriptionError("AT1x factor shard SHA differs")
            payload = json.loads(_read_regular_file_once(shard))
            if payload.get("content_sha256") != row["content_sha256"]:
                raise DirectDescriptionError("AT1x factor content identity differs")
            rows[row["layer_id"]] = payload
    if set(rows) != set(ATLAS_STEM_LAYERS):
        raise DirectDescriptionError("AT1x first-stem BN factor coverage differs")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "closed_forms_path": str(closed_path),
        "stem_factors": rows,
    }


def _atlas_comparison(
    observed: Mapping[str, Mapping[str, Any]],
    atlas: Mapping[str, Any],
    *,
    surface: str,
) -> list[dict[str, Any]]:
    rows = []
    for layer_id in ATLAS_STEM_LAYERS:
        factor = atlas["stem_factors"][layer_id]
        current = _moment_finish(observed[layer_id])
        mean = np.asarray(current["mean"], dtype=np.float64)
        variance = np.asarray(current["variance"], dtype=np.float64)
        running_mean = np.asarray(factor["payload"]["running_mean"], dtype=np.float64)
        running_variance = np.asarray(factor["payload"]["running_variance"], dtype=np.float64)
        epsilon = float(factor["payload"]["epsilon"])
        if mean.shape != running_mean.shape:
            raise DirectDescriptionError("AT1x BN channel geometry differs")
        mean_z = (mean - running_mean) / np.sqrt(running_variance + epsilon)
        log_var = np.log((variance + epsilon) / (running_variance + epsilon))
        rows.append(
            {
                "surface": surface,
                "layer_id": layer_id,
                "factor_id": factor["factor_id"],
                "factor_content_sha256": factor["content_sha256"],
                "atlas_consumption_status": factor["consumption_status"],
                "mean_z_rms": float(np.sqrt(np.mean(mean_z * mean_z))),
                "mean_z_max_abs": float(np.max(np.abs(mean_z))),
                "log_variance_ratio_rms": float(np.sqrt(np.mean(log_var * log_var))),
                "channels": int(mean.size),
                "comparison": (
                    "observed pre-BN moments versus frozen running moments; diagnostic, non-causal, non-additive"
                ),
            }
        )
    return rows


def _load_source(
    config: DDMPA1PoseNetAmplitudeTwinConfigV1,
) -> dict[str, Any]:
    receipt = json.loads(
        _bound(
            config.e2_receipt_path,
            config.e2_receipt_sha256,
            "E2 receipt",
        )
    )
    archive = _bound(
        config.e2_archive_path,
        config.e2_archive_sha256,
        "E2 strict archive",
    )
    if (
        len(archive) != config.e2_archive_bytes
        or receipt.get("score_claim") is not False
        or receipt.get("pointer_moved") is not False
        or receipt.get("pose_root_cause", {}).get("classification") != "ABSENT_FROM_COMPOSED_PACKET"
    ):
        raise DirectDescriptionError("E2 endpoint authority differs")
    independent = next(row for row in receipt["score_rows"] if row["id"] == "e2_independent_frozen_scorer")
    if (
        int(independent["archive_bytes"]) != config.e2_archive_bytes
        or independent["d_pose"] != "162.580958694146"
        or independent["d_seg"] != "0.028614807129"
    ):
        raise DirectDescriptionError("E2 independent baseline differs")
    raw_path = Path(config.e2_raw_path)
    if _stream_identity(raw_path) != (
        config.e2_raw_bytes,
        config.e2_raw_sha256,
    ):
        raise DirectDescriptionError("E2 strict raw identity differs")
    scorer_payload = _bound(
        config.scorer_config_path,
        config.scorer_config_sha256,
        "frozen scorer config",
    )
    scorer_config = DDMV15ScorerSolvedTemplateConfigV1.model_validate_json(scorer_payload)
    if (
        scorer_config.pair_start != 0
        or scorer_config.pair_count != PAIR_COUNT
        or scorer_config.scorer_batch_size != config.batch_size
        or scorer_config.score_claim is not False
    ):
        raise DirectDescriptionError("frozen scorer geometry differs")
    cache_path = Path(scorer_config.target_cache_path)
    if _stream_identity(cache_path) != (
        scorer_config.target_cache_bytes,
        scorer_config.target_cache_sha256,
    ):
        raise DirectDescriptionError("GT scorer cache identity differs")
    segnet, posenet, scorer = _load_models(scorer_config)
    expected = {
        "modules_sha256": config.upstream_modules_sha256,
        "posenet_weights_sha256": config.posenet_weights_sha256,
        "segnet_weights_sha256": config.segnet_weights_sha256,
    }
    for key, value in expected.items():
        if scorer.get(key) != value:
            raise DirectDescriptionError(f"frozen scorer custody differs for {key}")
    upstream = Path(scorer["modules_path"]).parent
    for name, digest in (
        ("evaluate.py", config.upstream_evaluate_sha256),
        ("frame_utils.py", config.upstream_frame_utils_sha256),
    ):
        if _sha256_path(upstream / name) != digest:
            raise DirectDescriptionError(f"upstream transitive source differs: {name}")
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, 2, *CAMERA_HW, 3),
    )
    ctx = {
        "posenet": posenet,
        "segnet": segnet,
        "scorer_custody": scorer,
        "scorer_config": scorer_config,
        "gt_f0": open_stored_npy_memmap(cache_path, "gt_f0"),
        "gt_f1": open_stored_npy_memmap(cache_path, "gt_f1"),
        "labels_all": open_stored_npy_memmap(cache_path, "lstars"),
        "poses_all": open_stored_npy_memmap(cache_path, "gt_poses"),
    }
    if (
        ctx["gt_f0"].shape[0] != PAIR_COUNT
        or ctx["gt_f1"].shape[0] != PAIR_COUNT
        or ctx["labels_all"].shape[0] != PAIR_COUNT
        or ctx["poses_all"].shape != (PAIR_COUNT, 6)
    ):
        raise DirectDescriptionError("GT cache geometry differs")
    return {
        "receipt": receipt,
        "baseline_row": independent,
        "archive": archive,
        "raw": raw,
        "raw_path": raw_path,
        "ctx": ctx,
        "scorer_config": scorer_config,
        "target_cache_path": cache_path,
    }


def _validate_runtime_source(
    config: DDMPA1PoseNetAmplitudeTwinConfigV1,
) -> dict[str, Any]:
    import av
    import numpy
    import safetensors
    import segmentation_models_pytorch
    import timm
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "av": av.__version__,
            "numpy": numpy.__version__,
            "safetensors": safetensors.__version__,
            "segmentation_models_pytorch": segmentation_models_pytorch.__version__,
            "timm": timm.__version__,
            "torch": torch.__version__,
        },
        "upstream_sources": {
            "modules.py": config.upstream_modules_sha256,
            "evaluate.py": config.upstream_evaluate_sha256,
            "frame_utils.py": config.upstream_frame_utils_sha256,
        },
        "weights": {
            "posenet": config.posenet_weights_sha256,
            "segnet": config.segnet_weights_sha256,
        },
        "device": "cpu",
        "deterministic_algorithms": True,
        "seed": config.seed,
        "score_claim": False,
        "evidence_axis": AXIS,
    }


def _stage_stats(
    config: DDMPA1PoseNetAmplitudeTwinConfigV1,
    source: Mapping[str, Any],
    atlas: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    final_path = root / "stage_checkpoints" / "01_input_statistics.json"
    if final_path.exists():
        value = json.loads(_read_regular_file_once(final_path))
        if value.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("statistics resume config differs")
        return value
    ctx = source["ctx"]
    posenet = ctx["posenet"]
    global_states = {
        "candidate": _moment_state(12),
        "gt": _moment_state(12),
    }
    stem_states = {
        name: {
            layer_id: _moment_state(int(_stem_branch_modules(posenet)[layer_id].out_channels))
            for layer_id in ATLAS_STEM_LAYERS
        }
        for name in ("candidate", "gt")
    }
    batch_root = root / "stage_checkpoints" / "01_input_statistics_batches"
    for start in range(0, PAIR_COUNT, config.batch_size):
        stop = min(start + config.batch_size, PAIR_COUNT)
        checkpoint = batch_root / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if row.get("typed_config_sha256") != config.typed_config_hash():
                raise DirectDescriptionError("statistics batch resume config differs")
        else:
            pair_ids = list(range(start, stop))
            camera = np.asarray(source["raw"][start:stop])
            gt = np.stack(
                (
                    np.asarray(ctx["gt_f0"][start:stop], dtype=np.uint8),
                    np.asarray(ctx["gt_f1"][start:stop], dtype=np.uint8),
                ),
                axis=1,
            )
            candidate_input, candidate_stem = _input_and_stem_moments(camera, posenet)
            gt_input, gt_stem = _input_and_stem_moments(gt, posenet)
            row = {
                "schema": "ddm_pa1_input_statistics_batch.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "pair_start": start,
                "pair_stop": stop,
                "pair_ids": list(pair_ids),
                "candidate_camera_sha256": _sha256_array(camera),
                "strict_e2_raw_sha256": config.e2_raw_sha256,
                "gt_camera_sha256": _sha256_array(gt),
                "input_moments": {
                    "candidate": _moment_payload(candidate_input),
                    "gt": _moment_payload(gt_input),
                },
                "stem_pre_bn_moments": {
                    "candidate": {key: _moment_payload(value) for key, value in candidate_stem.items()},
                    "gt": {key: _moment_payload(value) for key, value in gt_stem.items()},
                },
                "score_claim": False,
                "evidence_axis": AXIS,
            }
            _atomic_json(checkpoint, row)
        for name in ("candidate", "gt"):
            _moment_merge(global_states[name], row["input_moments"][name])
            for layer_id, payload in row["stem_pre_bn_moments"][name].items():
                _moment_merge(stem_states[name][layer_id], payload)

    gt_gain, gt_bias = _fit_affine(global_states["candidate"], global_states["gt"])
    gt_payload = _encode_affine(gt_gain[None], gt_bias[None])
    gt_frame0_payload = _encode_frame0_affine(gt_gain, gt_bias)
    if len(gt_payload) != config.joint_gt_payload_bytes:
        raise DirectDescriptionError("joint GT affine payload byte count differs")
    if len(gt_frame0_payload) != config.frame0_gt_payload_bytes:
        raise DirectDescriptionError("frame-0 GT affine payload byte count differs")
    scorer_target = _derive_scorer_only_input_target(posenet, atlas)
    scorer_gain, scorer_bias = _fit_affine_to_target(
        global_states["candidate"],
        np.asarray(scorer_target["raw_mean"], dtype=np.float64),
        np.asarray(scorer_target["raw_variance"], dtype=np.float64),
        quantize_fp16=False,
    )
    candidate_norm = _normalized_moments(global_states["candidate"])
    gt_norm = _normalized_moments(global_states["gt"])
    mean_gap = np.asarray(candidate_norm["mean"]) - np.asarray(gt_norm["mean"])
    candidate_variance = np.asarray(candidate_norm["variance"], dtype=np.float64)
    gt_variance = np.asarray(gt_norm["variance"], dtype=np.float64)
    gt_std = np.sqrt(np.maximum(gt_variance, 1e-12))
    standardized_mean_gap_rms = float(np.sqrt(np.mean(np.square(mean_gap / gt_std))))
    log_std_ratio_rms = float(
        np.sqrt(
            np.mean(np.square(0.5 * np.log(np.maximum(candidate_variance, 1e-12) / np.maximum(gt_variance, 1e-12))))
        )
    )
    amplitude_gap_small = (
        standardized_mean_gap_rms <= config.amplitude_equivalence_margin
        and log_std_ratio_rms <= config.amplitude_equivalence_margin
    )
    result = {
        "schema": "ddm_pa1_input_statistics.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "status": "MEASURED_N600_ADVISORY",
        "pair_count": PAIR_COUNT,
        "batch_count": math.ceil(PAIR_COUNT / config.batch_size),
        "all_batches_checkpointed_and_preserved": True,
        "official_preprocess": (
            "camera RGB -> torch bilinear 384x512 -> rgb_to_yuv6 -> "
            "two frames concatenated to 12 channels -> (x-127.5)/63.75"
        ),
        "channel_names": list(CHANNEL_NAMES),
        "candidate_normalized_moments": candidate_norm,
        "gt_normalized_moments": gt_norm,
        "candidate_minus_gt": {
            "mean": mean_gap.tolist(),
            "mean_gap_rms": float(np.sqrt(np.mean(mean_gap * mean_gap))),
            "standardized_mean_gap_rms": standardized_mean_gap_rms,
            "log_std_ratio_rms": log_std_ratio_rms,
            "maximum_absolute_mean_gap": float(np.max(np.abs(mean_gap))),
        },
        "amplitude_falsifier": {
            "pre_registered_equivalence_margin": (config.amplitude_equivalence_margin),
            "threshold_status": "SPECULATIVE_PRE_REGISTERED",
            "small_requires": ("standardized_mean_gap_rms<=margin and log_std_ratio_rms<=margin"),
            "amplitude_gap_small": amplitude_gap_small,
            "action_if_small": (
                "classify pose-value information absence, stop before affine "
                "ladder, and scope the negative to this formulation"
            ),
        },
        "gt_stat_target": {
            "rate_partition": "COUNTED",
            "target_source": "GT_VIDEO_DERIVED",
            "gain_fp16_roundtripped": gt_gain.tolist(),
            "bias_fp16_roundtripped": gt_bias.tolist(),
            "frame0_payload_bytes": config.frame0_gt_payload_bytes,
            "frame0_payload_sha256": _sha256(gt_frame0_payload),
            "joint_payload_bytes": len(gt_payload),
            "joint_payload_sha256": _sha256(gt_payload),
            "payload_layout": "12 little-endian fp16 gains then 12 fp16 biases",
        },
        "scorer_stat_target": {
            **scorer_target,
            "gain_decode_derived_float32": scorer_gain.tolist(),
            "bias_decode_derived_float32": scorer_bias.tolist(),
            "payload_bytes": config.scorer_target_payload_bytes,
        },
        "atlas_bn_comparison": [
            *_atlas_comparison(stem_states["candidate"], atlas, surface="e2_candidate"),
            *_atlas_comparison(stem_states["gt"], atlas, surface="gt"),
        ],
        "atlas_limit": {
            "direct_input_bn_exists": False,
            "atlas_amplitude_factor_count": atlas["manifest"]["amplitude_factors"]["count"],
            "interpretation": (
                "The atlas has no direct 12-channel input BN. The scorer-only "
                "target is the weighted least-squares inverse of both frozen "
                "first-stem BN running-stat tables through their frozen convs."
            ),
        },
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _atomic_json(final_path, result)
    return result


def _baseline_measurement(
    config: DDMPA1PoseNetAmplitudeTwinConfigV1,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint = source["baseline_row"]
    sites = PAIR_COUNT * SCORER_HW[0] * SCORER_HW[1]
    errors = 3_375_540
    pose_coordinates = PAIR_COUNT * 6
    d_seg = errors / sites
    d_pose = float(endpoint["d_pose"])
    pose_sse = d_pose * pose_coordinates
    if f"{d_seg:.12f}" != endpoint["d_seg"]:
        raise DirectDescriptionError("E2 strict Seg aggregate differs from receipt")
    return {
        "arm": "e2_independent_frozen_scorer_control",
        "archive_bytes": int(endpoint["archive_bytes"]),
        "counted_delta_bytes": 0,
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "d_pose": d_pose,
        "pose_term": math.sqrt(10.0 * d_pose),
        "score_claim": False,
        "evidence_axis": AXIS,
    }


def _arm_parameters(
    stats: Mapping[str, Any],
    arm: Literal["frame0_gt", "frame0_scorer", "joint_gt", "joint_scorer"],
) -> tuple[np.ndarray, np.ndarray, int, Literal["frame0", "joint"]]:
    placement: Literal["frame0", "joint"] = "frame0" if arm.startswith("frame0_") else "joint"
    if arm.endswith("_gt"):
        row = stats["gt_stat_target"]
        payload_bytes = int(row["frame0_payload_bytes"]) if placement == "frame0" else int(row["joint_payload_bytes"])
        return (
            np.asarray(row["gain_fp16_roundtripped"], dtype=np.float32)[None],
            np.asarray(row["bias_fp16_roundtripped"], dtype=np.float32)[None],
            payload_bytes,
            placement,
        )
    row = stats["scorer_stat_target"]
    return (
        np.asarray(row["gain_decode_derived_float32"], dtype=np.float32)[None],
        np.asarray(row["bias_decode_derived_float32"], dtype=np.float32)[None],
        int(row["payload_bytes"]),
        placement,
    )


def _stage_arm(
    config: DDMPA1PoseNetAmplitudeTwinConfigV1,
    source: Mapping[str, Any],
    atlas: Mapping[str, Any],
    stats: Mapping[str, Any],
    root: Path,
    arm: Literal["frame0_gt", "frame0_scorer", "joint_gt", "joint_scorer"],
) -> dict[str, Any]:
    stage_number = {
        "frame0_gt": "02",
        "frame0_scorer": "03",
        "joint_gt": "04",
        "joint_scorer": "05",
    }[arm]
    final_path = root / "stage_checkpoints" / f"{stage_number}_{arm}_measurement.json"
    if final_path.exists():
        value = json.loads(_read_regular_file_once(final_path))
        if value.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError(f"{arm} resume config differs")
        return value
    gain, bias, payload_bytes, placement = _arm_parameters(stats, arm)
    ctx = source["ctx"]
    input_state = _moment_state(12)
    stem_states = {
        layer_id: _moment_state(int(_stem_branch_modules(ctx["posenet"])[layer_id].out_channels))
        for layer_id in ATLAS_STEM_LAYERS
    }
    rows = []
    batch_root = root / "stage_checkpoints" / f"{stage_number}_{arm}_batches"
    for start in range(0, PAIR_COUNT, config.batch_size):
        stop = min(start + config.batch_size, PAIR_COUNT)
        checkpoint = batch_root / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if row.get("typed_config_sha256") != config.typed_config_hash():
                raise DirectDescriptionError(f"{arm} batch resume config differs")
        else:
            pair_ids = tuple(range(start, stop))
            camera = np.asarray(source["raw"][start:stop])
            corrected = _realize_affine(
                camera,
                pair_ids,
                gain,
                bias,
                ctx["posenet"],
                placement=placement,
            )
            labels = np.asarray(ctx["labels_all"][start:stop])
            poses = np.asarray(ctx["poses_all"][start:stop])
            scored = _score_batch(corrected, labels, poses, ctx["segnet"], ctx["posenet"])
            row = {
                "schema": "ddm_pa1_affine_batch.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "arm": arm,
                "pair_start": start,
                "pair_stop": stop,
                "pair_ids": list(pair_ids),
                "source_camera_sha256": _sha256_array(camera),
                "corrected_camera_sha256": _sha256_array(corrected),
                "changed_channel_values": int(np.count_nonzero(camera != corrected)),
                "changed_rgb_pixels": int(np.count_nonzero(np.any(camera != corrected, axis=-1))),
                "changed_frame1_channel_values": int(np.count_nonzero(camera[:, 1] != corrected[:, 1])),
                **scored,
                "score_claim": False,
                "evidence_axis": AXIS,
            }
            _atomic_json(checkpoint, row)
        rows.append(row)
        _moment_merge(input_state, row["input_moments"])
        for layer_id, payload in row["stem_pre_bn_moments"].items():
            _moment_merge(stem_states[layer_id], payload)
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    if placement == "frame0" and (
        errors != 3_375_540 or any(int(row["changed_frame1_channel_values"]) != 0 for row in rows)
    ):
        raise DirectDescriptionError("frame-0 placement violated structural Seg-free frame ownership")
    result = {
        "schema": "ddm_pa1_affine_measurement.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "arm": arm,
        "status": "MEASURED_N600_ADVISORY",
        "pair_count": PAIR_COUNT,
        "batch_count": len(rows),
        "all_batches_checkpointed_and_preserved": True,
        "counted_delta_bytes": payload_bytes,
        "archive_bytes_if_composed": len(source["archive"]) + payload_bytes,
        "rate_partition": "COUNTED" if arm.endswith("_gt") else "FREE",
        "placement": placement,
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "d_pose": d_pose,
        "pose_term": math.sqrt(10.0 * d_pose),
        "actual_post_realization_normalized_moments": _normalized_moments(input_state),
        "atlas_bn_comparison": _atlas_comparison(stem_states, atlas, surface=arm),
        "changed_channel_values": sum(int(row["changed_channel_values"]) for row in rows),
        "changed_rgb_pixels": sum(int(row["changed_rgb_pixels"]) for row in rows),
        "changed_frame1_channel_values": sum(int(row["changed_frame1_channel_values"]) for row in rows),
        "batch_digest_chain_sha256": _sha256(
            "".join(row["corrected_camera_sha256"] + row["cells_sha256"] + row["pose6_sha256"] for row in rows).encode()
        ),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _atomic_json(final_path, result)
    return result


def _delta(
    baseline: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    delta_d_pose = float(arm["d_pose"]) - float(baseline["d_pose"])
    delta_d_seg = float(arm["d_seg"]) - float(baseline["d_seg"])
    delta_bytes = int(arm["counted_delta_bytes"])
    delta_pose_term = float(arm["pose_term"]) - float(baseline["pose_term"])
    delta_seg_term = 100.0 * delta_d_seg
    delta_rate_term = 25.0 * delta_bytes / SOURCE_BYTES
    joint = delta_seg_term + delta_pose_term + delta_rate_term
    return {
        "delta_d_pose": delta_d_pose,
        "delta_pose_term_sqrt_10_d_pose": delta_pose_term,
        "delta_d_seg": delta_d_seg,
        "delta_seg_term_100_d_seg": delta_seg_term,
        "delta_bytes": delta_bytes,
        "delta_rate_term": delta_rate_term,
        "joint_delta_s": joint,
        "pose_positive": delta_d_pose < 0.0,
        "joint_positive": joint < 0.0,
        "rate_price_per_byte": 25.0 / SOURCE_BYTES,
    }


def _run(config: DDMPA1PoseNetAmplitudeTwinConfigV1) -> dict[str, Any]:
    import torch

    output = REPO_ROOT / config.output_dir
    free = shutil.disk_usage(output.parent if output.parent.exists() else REPO_ROOT).free
    if free < config.required_free_bytes:
        raise DirectDescriptionError(f"storage preflight failed: {free} < {config.required_free_bytes}")
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    source = _load_source(config)
    atlas = _load_atlas(config)
    baseline = _baseline_measurement(config, source)
    stats = _stage_stats(config, source, atlas, output)
    arms: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    menu_rows: list[dict[str, Any]] = []
    falsifier_fired = bool(stats["amplitude_falsifier"]["amplitude_gap_small"])
    selected_joint_arm = None
    if falsifier_fired:
        verdict = "AMPLITUDE_GAP_EQUIVALENT_POSE_VALUE_ABSENCE_FORMULATION_STOP"
        verdict_scope = (
            "FORMULATION:E2 global YUV6 first-two-moment gap under the "
            "pre-registered 0.1 equivalence margin; pose-amplitude and compact "
            "pose families remain open under other statistics or inverses"
        )
    else:
        for arm_name in ("frame0_gt", "frame0_scorer"):
            arm = _stage_arm(config, source, atlas, stats, output, arm_name)
            arms[arm_name] = arm
            deltas[arm_name] = _delta(baseline, arm)
        positive = [name for name in arms if deltas[name]["pose_positive"]]
        if positive:
            selected_frame0 = min(positive, key=lambda name: arms[name]["d_pose"])
            selected_joint_arm = "joint_gt" if selected_frame0.endswith("_gt") else "joint_scorer"
            joint = _stage_arm(config, source, atlas, stats, output, selected_joint_arm)
            arms[selected_joint_arm] = joint
            deltas[selected_joint_arm] = _delta(baseline, joint)
            verdict = "FRAME0_POSE_AMPLITUDE_POSITIVE_N600_ADVISORY"
            verdict_scope = (
                "INSTANCE:E2 strict raw x selected frame-0 global moment target "
                "x camera-residual realizer; joint placement is one bounded "
                "follow-up rung, not promotion authority"
            )
        else:
            verdict = "FRAME0_POSE_AMPLITUDE_NONPOSITIVE_N600_ADVISORY"
            verdict_scope = (
                "FORMULATION:E2 x frame-0 global first-two-moment matching x "
                "GT-stat and first-stem-BN-derived targets; broader amplitude "
                "statistics and compact pose inverses remain open"
            )
        for name, arm in arms.items():
            first_rung = name.startswith("frame0_")
            menu_rows.append(
                {
                    "pool_id": "pose_amplitude",
                    "rung": name,
                    "FIRST-RUNG": first_rung,
                    "rate_partition": arm["rate_partition"],
                    "measurement": {
                        "d_pose": arm["d_pose"],
                        "d_seg": arm["d_seg"],
                        "delta_bytes": arm["counted_delta_bytes"],
                    },
                    "delta": deltas[name],
                    "authority_surface": (
                        "shared camera RGB -> literal upstream YUV6 -> frozen "
                        "PoseNet; mandatory frozen SegNet collateral"
                    ),
                    "next_measurement": (
                        f"one joint-frame placement rung: {selected_joint_arm}"
                        if first_rung and deltas[name]["pose_positive"] and selected_joint_arm is not None
                        else "no further ladder in this arm"
                    ),
                }
            )
    target_delta = None
    if {"frame0_gt", "frame0_scorer"} <= set(arms):
        pose_target_gap = arms["frame0_scorer"]["d_pose"] - arms["frame0_gt"]["d_pose"]
        target_delta = {
            "d_pose_scorer_minus_gt": pose_target_gap,
            "d_seg_scorer_minus_gt": (arms["frame0_scorer"]["d_seg"] - arms["frame0_gt"]["d_seg"]),
            "approximately_equal_margin_d_pose": 0.01,
            "approximately_equal": abs(pose_target_gap) <= 0.01,
            "margin_status": "SPECULATIVE_PRE_REGISTERED",
        }
    receipt = {
        "schema": SCHEMA,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "verdict": verdict,
        "verdict_scope": verdict_scope,
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "source": {
            "e2_receipt_path": config.e2_receipt_path,
            "e2_receipt_sha256": config.e2_receipt_sha256,
            "e2_archive_path": config.e2_archive_path,
            "e2_archive_sha256": config.e2_archive_sha256,
            "e2_archive_bytes": len(source["archive"]),
            "e2_raw_path": config.e2_raw_path,
            "e2_raw_bytes": config.e2_raw_bytes,
            "e2_raw_sha256": config.e2_raw_sha256,
            "gt_cache_path": str(source["target_cache_path"]),
            "gt_cache_sha256": source["scorer_config"].target_cache_sha256,
            "gt_decode": "frame_utils.yuv420_to_rgb",
        },
        "runtime_custody": _validate_runtime_source(config),
        "atlas": {
            "manifest_path": atlas["manifest_path"],
            "manifest_sha256": config.atlas_manifest_sha256,
            "closed_forms_path": atlas["closed_forms_path"],
            "stem_factor_ids": [atlas["stem_factors"][layer_id]["factor_id"] for layer_id in ATLAS_STEM_LAYERS],
            "amplitude_factors_consumed_as_authority": 0,
            "reason": (
                "AT1x declared amplitude factor count zero; only source-bound "
                "BN running-stat tables are compared diagnostically"
            ),
        },
        "input_statistics": stats,
        "baseline": baseline,
        "arms": arms,
        "deltas": deltas,
        "target_source_comparison": target_delta,
        "selected_joint_arm": selected_joint_arm,
        "menu1_rows": menu_rows,
        "mechanism_boundary": {
            "prior": ("post_hoc_stored_corrections_dead_joint_descent_required_law_20260718"),
            "prior_application": (
                "post-hoc pose-value storage without a compact "
                "code-to-photometry inverse is dead on this witness vehicle"
            ),
            "this_mechanism": (
                "statistics matching changes realized photometry so frozen "
                "PoseNet sees matched global amplitude statistics; it does not "
                "store target Pose6 values or claim an inverse"
            ),
            "falsifier_fired": falsifier_fired,
        },
        "rate_partition": {
            "gt_video_derived_target": {
                "partition": "COUNTED",
                "frame0_bytes": config.frame0_gt_payload_bytes,
                "joint_bytes": config.joint_gt_payload_bytes,
            },
            "scorer_weight_derived_target": {
                "partition": "FREE",
                "bytes": config.scorer_target_payload_bytes,
                "composition_status": "NOT_COMPOSED_IN_GOVERNED_E2_INFLATE",
                "boundary": (
                    "hard-coded constants derive only from frozen scorer "
                    "weights/BN tables; candidate moments derive at decode from "
                    "already-counted E2 content"
                ),
                "promotion_blocker": (
                    "compose the generic two-pass moment transform into the "
                    "governed E2 inflate path and remeasure exact archive bytes"
                ),
            },
            "seg_side_30byte_row": (
                "the same FREE/NULL/COUNTED law applies: GT-video-derived "
                "amplitude facts remain COUNTED; scorer-only expected-stat "
                "constants are candidate FREE, subject to receiver survival"
            ),
        },
        "storage": {
            "preflight_status": "PASS",
            "required_free_bytes": config.required_free_bytes,
            "observed_free_bytes": free,
            "bulk_artifacts_created": False,
            "batch_checkpoints": "small JSON, immutable, local durable",
            "destructive_cleanup_performed": False,
        },
        "triality": {
            "dsl": (
                "typed config seals E2 raw/archive/scorer/atlas hashes, exact "
                "n600, FREE/COUNTED payload sizes, and the falsifier margin"
            ),
            "dag": (
                "E2 raw + canonical GT + frozen BN tables -> official YUV6 "
                "moments -> target-partitioned affine -> frame0-first shared "
                "camera RGB R -> PoseNet+SegNet -> conditional joint rung"
            ),
            "equations": [
                "ddm_pa1_pose_amplitude_moment_match_v1",
                "ddm_pa1_scorer_only_bn_inverse_target_v1",
                "ddm_pa1_free_null_counted_target_partition_v1",
                "ddm_pa1_shared_rgb_joint_price_v1",
            ],
        },
        "directive_consumption": [
            {
                "directive": "closure axiom / scorer-native coordinates",
                "status": "CONSUMED",
                "application": "exact official 12-channel YUV6 and frozen scorer path",
            },
            {
                "directive": "non-linguistic amplitude axis",
                "status": "CONSUMED",
                "application": "GT and scorer-only moment targets priced separately",
            },
            {
                "directive": "derive before measure / AT1x atlas",
                "status": "CONSUMED_WITH_LIMIT",
                "application": (
                    "both first-stem BN tables are inverted through frozen conv "
                    "weights; atlas amplitude count zero remains explicit"
                ),
            },
            {
                "directive": "total influence / relay solve",
                "status": "DEFERRED",
                "application": "menu1/rs1 owns non-additive relay composition",
            },
            {
                "directive": "surgical EV / luma-chroma asymmetry",
                "status": "CONSUMED",
                "application": ("channel rows remain separate; no blanket spatial correction"),
            },
            {
                "directive": "Fisher metric / curvelet residual basis",
                "status": "NOT_APPLICABLE_WITH_RATIONALE",
                "application": (
                    "this rung is a global prosody coordinate, not a residual basis or flip-ranking actuator"
                ),
            },
            {
                "directive": "MAIN charter corrections 2026-07-23T22:32:18Z",
                "status": "CONSUMED",
                "application": (
                    "E2 baseline; post-hoc-dead prior; amplitude falsifier; "
                    "FREE/COUNTED target partition; frame0-first placement"
                ),
            },
        ],
        "main_landing_review_required": True,
        "research_only": True,
    }
    _atomic_json(output / "ddm_pa1_posenet_amplitude_twin_receipt.json", receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="typed DDMPA1PoseNetAmplitudeTwinConfigV1 JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = DDMPA1PoseNetAmplitudeTwinConfigV1.model_validate_json(_read_regular_file_once(args.config))
    receipt = _run(config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "verdict_scope": receipt["verdict_scope"],
                "deltas": receipt["deltas"],
                "target_source_comparison": receipt["target_source_comparison"],
                "receipt": str(REPO_ROOT / config.output_dir / "ddm_pa1_posenet_amplitude_twin_receipt.json"),
                "score_claim": False,
                "pointer": POINTER,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
