# SPDX-License-Identifier: MIT
"""Native MLX reproduction primitives for the public PR95 HNeRV lane.

This module is deliberately narrow: it ports the PR95 decoder topology, archive
grammar, and optimizer partition into MLX so local Apple Silicon timing and
parity work can become queueable evidence. The training loop now supports both
synthetic timing targets and real PR95 source-video RGB/YUV6 target losses, but
it is still not a full PR95 reproduction until scorer-network loss,
stage/QAT/resume parity, export parity, full-frame inflate parity, and exact
CPU/CUDA auth eval anchor it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import struct
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration.mlx_scorer_adapters import (
    mlx_reference_conv2d_nhwc,
)
from tac.local_acceleration.pr95_hnerv_mlx_contract import (
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
    PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
    PR95_QAT_RESUME_UNPORTED_BLOCKER,
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
    PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER,
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
)
from tac.local_acceleration.pr95_hnerv_mlx_stage_losses import (
    pr95_mlx_stage_loss_contract_from_training_config,
)
from tac.optimization.optimizer_scheduler_registry import (
    OptimizerSchedulerRegistryError,
    default_optimizer_scheduler_registry,
)
from tac.optimization.parameter_group_lr_policy import (
    build_parameter_group_lr_policy_fingerprint,
)

try:  # pragma: no cover - exercised in environments with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    tree_unflatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


LANE_ID = "lane_pr95_hnerv_mlx_reproduction"
SMOKE_MANIFEST_SCHEMA = "pr95_hnerv_mlx_timing_smoke_manifest_v1"
SMOKE_ARCHIVE_SCHEMA = "pr95_hnerv_mlx_byte_closed_smoke_archive_v1"
PUBLIC_ARCHIVE_PACKET_SCHEMA = "pr95_hnerv_public_archive_packet.v1"
PUBLIC_ARCHIVE_FORWARD_PARITY_SCHEMA = "pr95_hnerv_public_archive_mlx_forward_parity.v1"
PUBLIC_ARCHIVE_DECODER_TRACE_SCHEMA = "pr95_hnerv_public_archive_mlx_decoder_trace.v1"
PR95_MLX_PYTORCH_EXPORT_FORWARD_PARITY_SCHEMA = (
    "pr95_hnerv_mlx_pytorch_export_forward_parity.v1"
)
PR95_ARCHIVE_EXPORT_SCHEMA = "pr95_hnerv_archive_export.v1"
PR95_ARCHIVE_N_QUANT = 127
PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE = "optimized"
PR95_MLX_CONV2D_ACCUMULATION_MODES: tuple[str, ...] = (
    PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    "fixed_fp32",
    "kahan_fp32",
    "fixed_fp64",
)
_PR95_MLX_RGB_HEAD_CONV_NAMES: tuple[str, ...] = ("rgb_0", "rgb_1")
_PR95_MLX_REFINE_CONV_NAMES: tuple[str, ...] = ("refine0", "refine1")
_PR95_MLX_BLOCK_CONV_NAMES: tuple[str, ...] = tuple(
    name for i in range(6) for name in (f"blocks.{i}.conv", f"blocks.{i}.skip_conv")
)
PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS: dict[str, dict[str, str]] = {
    "none": {},
    "rgb_heads_kahan_fp32": dict.fromkeys(
        _PR95_MLX_RGB_HEAD_CONV_NAMES,
        "kahan_fp32",
    ),
    "refine_rgb_heads_kahan_fp32": dict.fromkeys(
        (*_PR95_MLX_REFINE_CONV_NAMES, *_PR95_MLX_RGB_HEAD_CONV_NAMES),
        "kahan_fp32",
    ),
    "blocks_kahan_fp32": dict.fromkeys(_PR95_MLX_BLOCK_CONV_NAMES, "kahan_fp32"),
    "blocks01_kahan_fp32": dict.fromkeys(
        tuple(
            name
            for i in range(2)
            for name in (f"blocks.{i}.conv", f"blocks.{i}.skip_conv")
        ),
        "kahan_fp32",
    ),
    "blocks_refine_kahan_fp32": dict.fromkeys(
        (*_PR95_MLX_BLOCK_CONV_NAMES, *_PR95_MLX_REFINE_CONV_NAMES),
        "kahan_fp32",
    ),
}

PR95_STAGE_MODULES: dict[int, str] = {
    1: "stage1_v328_ce",
    2: "stage2_v331_softplus",
    3: "stage3_v332_smooth",
    4: "stage4_v332_qat",
    5: "stage5_c1a_l7",
    6: "stage6_lambda_sweep",
    7: "stage7_sigma_sweep",
    8: "stage8_muon_finetune",
}
PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS: dict[int, str] = {
    1: "pr95_stage1_adamw_baseline_mlx",
    2: "pr95_stage2_adamw_baseline_mlx",
    3: "pr95_stage3_adamw_baseline_mlx",
    4: "pr95_stage4_adamw_qat_mlx",
    5: "pr95_stage5_adamw_baseline_mlx",
    6: "pr95_stage6_adamw_lambda_sweep_mlx",
    7: "pr95_stage7_adamw_sigma_sweep_mlx",
    8: "pr95_stage8_muon_adamw_mlx",
}
PR95_MLX_BACKEND_STATUS_LOCAL_TIMING_PROXY = "implemented_mlx_local_timing_proxy"
PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY = "synthetic_timing_only"
PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_TIMING_ONLY = (
    "source_video_rgb_timing_only"
)
PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY = (
    "source_video_rgb_yuv6_preprocess_coupled_timing_only"
)
PR95_MLX_LOSS_SURFACE_RGB_MSE = "rgb_mse"
PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE = "rgb_yuv6_mse"
PR95_MLX_LOSS_SURFACES = (
    PR95_MLX_LOSS_SURFACE_RGB_MSE,
    PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE,
)
PR95_MLX_SOURCE_FAITHFUL_BLOCKERS: tuple[str, ...] = (
    PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER,
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
    PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER,
    PR95_QAT_RESUME_UNPORTED_BLOCKER,
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
)
PR95_MLX_SOURCE_VIDEO_RGB_BLOCKERS: tuple[str, ...] = (
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
    PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER,
    PR95_QAT_RESUME_UNPORTED_BLOCKER,
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
)
PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS: tuple[str, ...] = (
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER,
    PR95_QAT_RESUME_UNPORTED_BLOCKER,
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
    "reproduction_claim": False,
    "pr95_1to1_reproduction_claim": False,
    "reproduction_equivalence": False,
}

EXACT_READINESS_REFUSAL_BLOCKERS: tuple[str, ...] = (
    "pr95_hnerv_mlx_timing_smoke_is_local_training_signal_not_score",
    "pr95_hnerv_mlx_training_is_synthetic_timing_only_not_source_faithful",
    *PR95_MLX_SOURCE_FAITHFUL_BLOCKERS,
    "synthetic_targets_do_not_establish_contest_quality",
    "byte_closed_smoke_archive_not_consumed_by_pr95_runtime",
    "runtime_consumption_proof_missing",
    "receiver_proof_missing",
    "requires_pytorch_export_forward_parity_on_source_checkpoint",
    "requires_byte_closed_contest_archive_export",
    "requires_exact_cpu_cuda_auth_eval_before_score_claim",
)


class Pr95HNeRVMlxError(RuntimeError):
    """Raised when the PR95 MLX lane cannot execute faithfully."""


@dataclass(frozen=True)
class Pr95PublicArchivePacket:
    """Decoded public PR95 archive packet plus byte custody metadata."""

    archive_zip_path: Path
    archive_zip_sha256: str
    member_name: str
    member_bytes: int
    member_sha256: str
    member_compress_type: int
    state_dict: dict[str, np.ndarray]
    latents: np.ndarray
    meta: dict[str, Any]

    def custody_manifest(self) -> dict[str, Any]:
        return {
            "schema": PUBLIC_ARCHIVE_PACKET_SCHEMA,
            "archive_zip_path": self.archive_zip_path.as_posix(),
            "archive_zip_sha256": self.archive_zip_sha256,
            "member_name": self.member_name,
            "member_bytes": self.member_bytes,
            "member_sha256": self.member_sha256,
            "member_compress_type": self.member_compress_type,
            "meta": dict(self.meta),
            "latent_shape": [int(dim) for dim in self.latents.shape],
            "state_dict_tensor_count": len(self.state_dict),
            "source_pr": 95,
            "submission": "hnerv_muon",
            **FALSE_AUTHORITY,
        }


def require_mlx() -> None:
    """Fail clearly when imported on a machine without MLX."""

    if mx is None or nn is None or tree_flatten is None or tree_unflatten is None:
        raise Pr95HNeRVMlxError(
            "MLX is required for PR95/HNeRV local reproduction"
        ) from _MLX_IMPORT_ERROR


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _numpy_float32_from_any(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _brotli_decompress(data: bytes) -> bytes:
    try:
        import brotli
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise Pr95HNeRVMlxError("brotli is required to parse PR95 archives") from exc
    return brotli.decompress(data)


def _read_exact(buffer: io.BytesIO, size: int, *, field: str) -> bytes:
    data = buffer.read(size)
    if len(data) != size:
        raise Pr95HNeRVMlxError(
            f"truncated PR95 archive while reading {field}: expected {size} bytes, "
            f"got {len(data)}"
        )
    return data


def _read_u32(buffer: io.BytesIO, *, field: str) -> int:
    return struct.unpack("<I", _read_exact(buffer, 4, field=field))[0]


def _decode_pr95_zigzag_u8(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _encode_pr95_zigzag_i8(values: np.ndarray) -> bytes:
    arr = values.astype(np.int32, copy=False)
    encoded = np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)
    return encoded.tobytes()


def _brotli_compress(data: bytes, *, quality: int = 11) -> bytes:
    try:
        import brotli
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise Pr95HNeRVMlxError("brotli is required to build PR95 archives") from exc
    return brotli.compress(data, quality=quality)


def _decode_pr95_decoder_blob(data: bytes) -> dict[str, np.ndarray]:
    raw = _brotli_decompress(data)
    buffer = io.BytesIO(raw)
    tensor_count = _read_u32(buffer, field="decoder.tensor_count")
    state_dict: dict[str, np.ndarray] = {}
    for index in range(tensor_count):
        name_len = _read_u32(buffer, field=f"decoder.{index}.name_len")
        name = _read_exact(buffer, name_len, field=f"decoder.{index}.name").decode(
            "utf-8"
        )
        ndim = _read_u32(buffer, field=f"decoder.{name}.ndim")
        shape = tuple(
            _read_u32(buffer, field=f"decoder.{name}.shape.{axis}")
            for axis in range(ndim)
        )
        scale = struct.unpack(
            "<f", _read_exact(buffer, 4, field=f"decoder.{name}.scale")
        )[0]
        flat_size = _read_u32(buffer, field=f"decoder.{name}.flat_size")
        flat = _decode_pr95_zigzag_u8(
            _read_exact(buffer, flat_size, field=f"decoder.{name}.quantized")
        )
        expected = math.prod(shape) if shape else 1
        if flat.size != expected:
            raise Pr95HNeRVMlxError(
                f"decoder tensor {name!r} has {flat.size} values but shape {shape} "
                f"requires {expected}"
            )
        state_dict[name] = flat.astype(np.float32).reshape(shape) * float(scale)
    trailing = buffer.read()
    if trailing:
        raise Pr95HNeRVMlxError(
            f"decoder blob has {len(trailing)} trailing byte(s) after state dict"
        )
    return state_dict


def _encode_pr95_decoder_blob(
    state_dict: Mapping[str, Any],
    *,
    n_quant: int = PR95_ARCHIVE_N_QUANT,
    brotli_quality: int = 11,
) -> bytes:
    buffer = io.BytesIO()
    buffer.write(struct.pack("<I", len(state_dict)))
    for name, value in state_dict.items():
        tensor = _numpy_float32_from_any(value)
        if not np.isfinite(tensor).all():
            raise Pr95HNeRVMlxError(f"decoder tensor {name!r} contains non-finite values")
        max_abs = float(np.max(np.abs(tensor))) if tensor.size else 0.0
        scale = max_abs / n_quant if max_abs > 0 else 1.0
        quantized = (
            np.rint(tensor / scale)
            .clip(-n_quant, n_quant)
            .astype(np.int8)
            .reshape(-1)
        )
        name_bytes = str(name).encode("utf-8")
        buffer.write(struct.pack("<I", len(name_bytes)))
        buffer.write(name_bytes)
        buffer.write(struct.pack("<I", tensor.ndim))
        for dim in tensor.shape:
            buffer.write(struct.pack("<I", int(dim)))
        buffer.write(struct.pack("<f", float(scale)))
        buffer.write(struct.pack("<I", int(quantized.size)))
        buffer.write(_encode_pr95_zigzag_i8(quantized))
    return _brotli_compress(buffer.getvalue(), quality=brotli_quality)


def _decode_pr95_latents_payload(raw: bytes) -> np.ndarray:
    buffer = io.BytesIO(raw)
    n_pairs = _read_u32(buffer, field="latents.n_pairs")
    latent_dim = _read_u32(buffer, field="latents.latent_dim")
    mins = np.frombuffer(
        _read_exact(buffer, latent_dim * 2, field="latents.mins_fp16"),
        dtype=np.float16,
    ).astype(np.float32)
    scales = np.frombuffer(
        _read_exact(buffer, latent_dim * 2, field="latents.scales_fp16"),
        dtype=np.float16,
    ).astype(np.float32)
    total = n_pairs * latent_dim
    lo = np.frombuffer(
        _read_exact(buffer, total, field="latents.delta_lo"),
        dtype=np.uint8,
    ).astype(np.uint16)
    hi = np.frombuffer(
        _read_exact(buffer, total, field="latents.delta_hi"),
        dtype=np.uint8,
    ).astype(np.uint16)
    trailing = buffer.read()
    if trailing:
        raise Pr95HNeRVMlxError(
            f"latent payload has {len(trailing)} trailing byte(s)"
        )
    delta_zz = ((hi << 8) | lo).reshape(n_pairs, latent_dim)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    quantized = np.empty_like(delta, dtype=np.int32)
    quantized[0] = delta[0]
    for index in range(1, n_pairs):
        quantized[index] = quantized[index - 1] + delta[index]
    return quantized.astype(np.uint8).astype(np.float32) * scales[None, :] + mins[None, :]


def _encode_pr95_latents_payload(latents: Any) -> bytes:
    tensor = _numpy_float32_from_any(latents)
    if tensor.ndim != 2:
        raise Pr95HNeRVMlxError(
            f"PR95 latents must be rank-2 (n_pairs, latent_dim), got {tensor.shape}"
        )
    n_pairs, latent_dim = (int(dim) for dim in tensor.shape)
    mins = tensor.min(axis=0)
    maxs = tensor.max(axis=0)
    scales = np.maximum((maxs - mins) / 254.0, 1e-10).astype(np.float32)
    quantized = np.rint((tensor - mins[None, :]) / scales[None, :]).clip(0, 254)
    quantized_u8 = quantized.astype(np.uint8)
    delta = np.empty_like(quantized_u8, dtype=np.int16)
    delta[0] = quantized_u8[0]
    delta[1:] = quantized_u8[1:].astype(np.int16) - quantized_u8[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8)
    hi = (delta_zz >> 8).astype(np.uint8)
    return b"".join(
        [
            struct.pack("<II", n_pairs, latent_dim),
            mins.astype(np.float16).tobytes(),
            scales.astype(np.float16).tobytes(),
            lo.tobytes(),
            hi.tobytes(),
        ]
    )


def _normalize_pr95_archive_meta(
    meta: Mapping[str, Any] | None,
    *,
    latents: np.ndarray,
) -> dict[str, Any]:
    n_pairs, latent_dim = (int(dim) for dim in latents.shape)
    source = dict(meta or {})
    normalized = {
        "n_pairs": int(source.get("n_pairs", n_pairs)),
        "latent_dim": int(source.get("latent_dim", latent_dim)),
        "base_channels": int(source.get("base_channels", 36)),
        "eval_size": [int(dim) for dim in source.get("eval_size", [384, 512])],
    }
    if normalized["n_pairs"] != n_pairs or normalized["latent_dim"] != latent_dim:
        raise Pr95HNeRVMlxError(
            f"PR95 meta {normalized!r} does not match latent shape {latents.shape}"
        )
    if normalized["eval_size"] != [384, 512]:
        raise Pr95HNeRVMlxError(
            f"PR95 export currently supports eval_size [384, 512], got "
            f"{normalized['eval_size']}"
        )
    return normalized


def _expected_pr95_state_shapes(
    *,
    latent_dim: int,
    base_channels: int,
) -> dict[str, tuple[int, ...]]:
    channels = [
        base_channels,
        base_channels,
        base_channels,
        int(base_channels * 0.75),
        int(base_channels * 0.58),
        int(base_channels * 0.5),
        int(base_channels * 0.5),
    ]
    shapes: dict[str, tuple[int, ...]] = {
        "stem.weight": (channels[0] * 6 * 8, latent_dim),
        "stem.bias": (channels[0] * 6 * 8,),
    }
    for index in range(6):
        in_ch = channels[index]
        out_ch = channels[index + 1]
        shapes[f"blocks.{index}.weight"] = (out_ch * 4, in_ch, 3, 3)
        shapes[f"blocks.{index}.bias"] = (out_ch * 4,)
        if in_ch != out_ch:
            shapes[f"skips.{index}.weight"] = (out_ch, in_ch, 1, 1)
            shapes[f"skips.{index}.bias"] = (out_ch,)
    final_ch = channels[-1]
    shapes.update(
        {
            "refine.0.weight": (final_ch // 2, final_ch, 3, 3),
            "refine.0.bias": (final_ch // 2,),
            "refine.1.weight": (final_ch, final_ch // 2, 3, 3),
            "refine.1.bias": (final_ch,),
            "rgb_0.weight": (3, final_ch, 3, 3),
            "rgb_0.bias": (3,),
            "rgb_1.weight": (3, final_ch, 3, 3),
            "rgb_1.bias": (3,),
        }
    )
    return shapes


def _validate_pr95_state_dict_shapes(
    state_dict: Mapping[str, Any],
    *,
    latent_dim: int,
    base_channels: int,
) -> None:
    expected = _expected_pr95_state_shapes(
        latent_dim=latent_dim,
        base_channels=base_channels,
    )
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise Pr95HNeRVMlxError(
            "PR95 state_dict key mismatch"
            f"; missing={missing or []}; extra={extra or []}"
        )
    for name, expected_shape in expected.items():
        actual_shape = tuple(
            int(dim) for dim in _numpy_float32_from_any(state_dict[name]).shape
        )
        if actual_shape != expected_shape:
            raise Pr95HNeRVMlxError(
                f"PR95 tensor {name!r} shape {actual_shape} does not match "
                f"expected {expected_shape}"
            )


def build_pr95_public_archive_member(
    state_dict: Mapping[str, Any],
    latents: Any,
    *,
    meta: Mapping[str, Any] | None = None,
    brotli_quality: int = 11,
) -> bytes:
    """Build a source-compatible PR95 ``0.bin`` archive member."""

    latents_np = _numpy_float32_from_any(latents)
    if latents_np.ndim != 2:
        raise Pr95HNeRVMlxError(
            f"PR95 latents must be rank-2 (n_pairs, latent_dim), got {latents_np.shape}"
        )
    if min(latents_np.shape) < 1:
        raise Pr95HNeRVMlxError(f"PR95 latents must be non-empty, got {latents_np.shape}")
    if not np.isfinite(latents_np).all():
        raise Pr95HNeRVMlxError("PR95 latents contain non-finite values")
    normalized_meta = _normalize_pr95_archive_meta(meta, latents=latents_np)
    _validate_pr95_state_dict_shapes(
        state_dict,
        latent_dim=int(normalized_meta["latent_dim"]),
        base_channels=int(normalized_meta["base_channels"]),
    )
    meta_blob = _brotli_compress(
        json.dumps(normalized_meta).encode("utf-8"),
        quality=brotli_quality,
    )
    decoder_blob = _encode_pr95_decoder_blob(
        state_dict,
        brotli_quality=brotli_quality,
    )
    latents_blob = _brotli_compress(
        _encode_pr95_latents_payload(latents_np),
        quality=brotli_quality,
    )
    output = io.BytesIO()
    output.write(struct.pack("<I", len(meta_blob)))
    output.write(meta_blob)
    output.write(struct.pack("<I", len(decoder_blob)))
    output.write(decoder_blob)
    output.write(struct.pack("<I", len(latents_blob)))
    output.write(latents_blob)
    return output.getvalue()


def parse_pr95_public_archive_member(
    archive_bytes: bytes,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Parse the public PR95 ``0.bin`` member without importing the source tree."""

    buffer = io.BytesIO(archive_bytes)
    meta_len = _read_u32(buffer, field="archive.meta_len")
    meta = json.loads(_brotli_decompress(_read_exact(buffer, meta_len, field="archive.meta")))
    decoder_len = _read_u32(buffer, field="archive.decoder_len")
    state_dict = _decode_pr95_decoder_blob(
        _read_exact(buffer, decoder_len, field="archive.decoder_blob")
    )
    latent_len = _read_u32(buffer, field="archive.latents_len")
    latents = _decode_pr95_latents_payload(
        _brotli_decompress(_read_exact(buffer, latent_len, field="archive.latents_blob"))
    )
    trailing = buffer.read()
    if trailing:
        raise Pr95HNeRVMlxError(
            f"PR95 archive member has {len(trailing)} trailing byte(s)"
        )
    return state_dict, latents.astype(np.float32, copy=False), meta


def parse_pr95_public_archive_zip(
    archive_zip_path: Path,
    *,
    member_name: str = "0.bin",
) -> Pr95PublicArchivePacket:
    """Load the public PR95 ZIP and decode its HNeRV state/latent packet."""

    archive_zip_path = Path(archive_zip_path)
    if not archive_zip_path.is_file():
        raise Pr95HNeRVMlxError(f"PR95 archive ZIP not found: {archive_zip_path}")
    archive_zip_sha256 = _sha256_file(archive_zip_path)
    with zipfile.ZipFile(archive_zip_path) as zf:
        try:
            info = zf.getinfo(member_name)
        except KeyError as exc:
            names = ", ".join(zf.namelist())
            raise Pr95HNeRVMlxError(
                f"PR95 archive ZIP is missing member {member_name!r}; members: {names}"
            ) from exc
        archive_bytes = zf.read(member_name)
    state_dict, latents, meta = parse_pr95_public_archive_member(archive_bytes)
    if tuple(latents.shape) != (
        int(meta.get("n_pairs", -1)),
        int(meta.get("latent_dim", -1)),
    ):
        raise Pr95HNeRVMlxError(
            f"latent shape {latents.shape} does not match PR95 meta {meta!r}"
        )
    return Pr95PublicArchivePacket(
        archive_zip_path=archive_zip_path,
        archive_zip_sha256=archive_zip_sha256,
        member_name=member_name,
        member_bytes=len(archive_bytes),
        member_sha256=_sha256_bytes(archive_bytes),
        member_compress_type=int(info.compress_type),
        state_dict=state_dict,
        latents=latents,
        meta=meta,
    )


def write_pr95_public_archive_zip(
    state_dict: Mapping[str, Any],
    latents: Any,
    *,
    meta: Mapping[str, Any] | None,
    output_zip_path: Path,
    member_name: str = "0.bin",
    brotli_quality: int = 11,
) -> dict[str, Any]:
    """Write a deterministic PR95-compatible single-member ZIP archive."""

    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    member_bytes = build_pr95_public_archive_member(
        state_dict,
        latents,
        meta=meta,
        brotli_quality=brotli_quality,
    )
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.extra = b""
    with zipfile.ZipFile(output_zip_path, "w") as zf:
        zf.comment = b""
        zf.writestr(info, member_bytes)
    reparsed = parse_pr95_public_archive_zip(output_zip_path, member_name=member_name)
    archive_zip_bytes = output_zip_path.stat().st_size
    archive_zip_sha256 = _sha256_file(output_zip_path)
    return {
        "schema": PR95_ARCHIVE_EXPORT_SCHEMA,
        "archive_zip_path": output_zip_path.as_posix(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha256,
        "archive_path": output_zip_path.as_posix(),
        "archive_bytes": archive_zip_bytes,
        "archive_sha256": archive_zip_sha256,
        "candidate_archive": {
            "path": output_zip_path.as_posix(),
            "bytes": archive_zip_bytes,
            "sha256": archive_zip_sha256,
        },
        "path": output_zip_path.as_posix(),
        "bytes": archive_zip_bytes,
        "sha256": archive_zip_sha256,
        "member_name": member_name,
        "member": member_name,
        "member_bytes": len(member_bytes),
        "member_sha256": _sha256_bytes(member_bytes),
        "member_compress_type": int(zipfile.ZIP_STORED),
        "parsed_meta": reparsed.meta,
        "parsed_latent_shape": [int(dim) for dim in reparsed.latents.shape],
        "parsed_state_dict_tensor_count": len(reparsed.state_dict),
        "runtime_consumption_proof_present": False,
        "receiver_proof_present": False,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "pr95_archive_export_is_byte_closed_but_not_runtime_consumed",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }


def _param_count_from_tree(tree: Any) -> int:
    require_mlx()
    count = 0
    for _name, value in tree_flatten(tree):  # type: ignore[misc]
        shape = getattr(value, "shape", ())
        if shape:
            count += math.prod(int(dim) for dim in shape)
    return count


def _mlx_array_from_any(value: Any) -> Any:
    require_mlx()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return mx.array(np.asarray(value, dtype=np.float32))  # type: ignore[union-attr]


def _torch_conv_to_mlx(value: Any) -> Any:
    arr = np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value,
        dtype=np.float32,
    )
    if arr.ndim != 4:
        raise ValueError(f"expected torch conv weight with 4 dims, got {arr.shape}")
    return _mlx_array_from_any(np.transpose(arr, (0, 2, 3, 1)))


def _mlx_conv_to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 4:
        raise ValueError(f"expected MLX conv weight with 4 dims, got {arr.shape}")
    return np.transpose(arr, (0, 3, 1, 2))


def _pair_int(value: int | tuple[int, int], *, label: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{label} must be an int or pair, got {value!r}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def validate_pr95_mlx_conv2d_accumulation_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in PR95_MLX_CONV2D_ACCUMULATION_MODES:
        raise ValueError(
            "conv2d_accumulation_mode must be one of "
            f"{PR95_MLX_CONV2D_ACCUMULATION_MODES}, got {mode!r}"
        )
    return normalized


def validate_pr95_mlx_conv2d_accumulation_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    if overrides is None:
        return {}
    validated: dict[str, str] = {}
    for raw_name, raw_mode in overrides.items():
        name = str(raw_name)
        if not name:
            raise ValueError("conv2d accumulation override names must be non-empty")
        validated[name] = validate_pr95_mlx_conv2d_accumulation_mode(str(raw_mode))
    return validated


def pr95_mlx_conv2d_accumulation_overrides_from_preset(preset: str) -> dict[str, str]:
    normalized = str(preset)
    if normalized not in PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS:
        raise ValueError(
            "conv2d accumulation override preset must be one of "
            f"{tuple(PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS)}, got {preset!r}"
        )
    return dict(PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS[normalized])


def pr95_mlx_conv2d_accumulation_overrides_from_items(
    items: Sequence[str] | None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    overrides = validate_pr95_mlx_conv2d_accumulation_overrides(base)
    for raw_item in items or ():
        item = str(raw_item).strip()
        name, sep, mode = item.partition("=")
        if not item or sep != "=" or not name.strip() or not mode.strip():
            raise ValueError(
                "conv2d accumulation override items must use '<module>=<mode>', "
                f"got {raw_item!r}"
            )
        overrides[name.strip()] = validate_pr95_mlx_conv2d_accumulation_mode(
            mode.strip()
        )
    return overrides


def pixel_shuffle_2x_nhwc(x: Any, *, upscale_factor: int = 2) -> Any:
    """CANONICAL PixelShuffle for NHWC tensors using native MLX reshape/transpose.

    Canonical PR95 helper for use by every Path 3 Apple-Silicon MLX substrate
    (and any future MLX renderer) that needs PyTorch-byte-stable pixel-shuffle
    upsampling. Sister substrates MUST import and delegate to this helper
    rather than re-implement local copies (per CONSOLIDATE-OP-1 2026-05-26
    extraction wave; prevents the FIX-WAVE-R1 + FIX-WAVE-R1' channel-LAST
    convention drift class from recurring at future Path 3 substrate launches).

    Convention: channel-FIRST reshape ``(B, H, W, out_C, 2, 2)`` + transpose
    ``(0, 1, 4, 2, 5, 3)``. This matches PyTorch ``nn.PixelShuffle(2)``
    byte-for-byte; empirically PyTorch-byte-stable (0.0 absolute drift per
    sister D=Z6 anchor 2026-05-26).

    Empirical drift bounds vs PyTorch ``nn.PixelShuffle(2)``:
    - CANONICAL channel-FIRST: 0.0 absolute drift (this implementation)
    - FORBIDDEN channel-LAST ``(B, H, W, 2, 2, out_C)`` + ``(0, 1, 3, 2, 4, 5)``:
      2.40 absolute drift (A=DreamerV3 pre-FIX-WAVE-R1 anchor)
    - FORBIDDEN channel-LAST same as above: 3.77 absolute drift
      (F=Z8 pre-FIX-WAVE-R1' anchor 2026-05-26T08:03Z)

    Canonical invocation pattern for substrates (see also Catalog #295 inflate
    self-containment scope — this helper is MLX training-time only; the
    substrate inflate.py runtime is PyTorch-only and uses native
    ``F.pixel_shuffle(x, upscale_factor=2)``)::

        from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc

        decoded = pixel_shuffle_2x_nhwc(self.conv(x))

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9
    runtime closure: MLX-trained-PyTorch-inflated model MUST be the same
    runtime the MLX trainer observes at convergence.

    Args:
        x: NHWC tensor with shape ``(B, H, W, C)`` and ``C`` divisible by 4.
        upscale_factor: Only ``2`` is supported (PR95 + Path 3 substrates use
            only the 2x case; raises ``ValueError`` otherwise).

    Returns:
        NHWC tensor with shape ``(B, 2*H, 2*W, C/4)``.

    Raises:
        ValueError: ``upscale_factor != 2`` or ``len(x.shape) != 4`` or
            ``C % 4 != 0``.
    """

    require_mlx()
    if upscale_factor != 2:
        raise ValueError("PR95 uses only 2x pixel shuffle")
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC tensor, got shape {x.shape}")
    batch, height, width, channels = (int(dim) for dim in x.shape)
    block = upscale_factor * upscale_factor
    if channels % block:
        raise ValueError(
            f"channels must be divisible by {block} for 2x pixel shuffle; got {channels}"
        )
    out_channels = channels // block
    y = mx.reshape(  # type: ignore[union-attr]
        x,
        (batch, height, width, out_channels, upscale_factor, upscale_factor),
    )
    y = mx.transpose(y, (0, 1, 4, 2, 5, 3))  # type: ignore[union-attr]
    return mx.reshape(  # type: ignore[union-attr]
        y,
        (batch, height * upscale_factor, width * upscale_factor, out_channels),
    )


def bilinear_resize2x_align_corners_false_nhwc(x: Any) -> Any:
    """CANONICAL 2x bilinear resize for NHWC tensors matching PyTorch align_corners=False.

    Canonical PR95 helper for use by every Path 3 Apple-Silicon MLX substrate
    that needs PyTorch-byte-stable 2x bilinear upsampling. Sister substrates
    MUST import and delegate to this helper rather than re-implement local
    copies (per CONSOLIDATE-OP-1 2026-05-26 extraction wave; prevents the
    FIX-WAVE-R1 + FIX-WAVE-R1' ``mx.repeat`` 2x-approximation drift class
    from recurring at future Path 3 substrate launches).

    Empirical drift bounds vs PyTorch ``F.interpolate(scale_factor=2,
    mode='bilinear', align_corners=False)``:
    - CANONICAL specialized 2x implementation: 0.0 absolute drift
      (this implementation)
    - FORBIDDEN ``mx.repeat`` 2x approximation: 0.99 absolute drift
      (A=DreamerV3 pre-FIX-WAVE-R1 anchor)
    - FORBIDDEN ``mx.repeat`` 2x approximation: 1.51 absolute drift
      (F=Z8 pre-FIX-WAVE-R1' anchor 2026-05-26T08:03Z)

    Implementation: specialized closed-form 2x bilinear that exploits the
    align_corners=False formula at scale=2 producing weights ``0.75, 0.25``
    per neighbor. Width pass first, then height pass; chained reshape +
    stack rebuilds the upsampled tensor in NHWC layout.

    Canonical invocation pattern for substrates (Catalog #295 inflate
    self-containment scope — MLX training-time only; substrate inflate.py
    runtime is PyTorch-only and uses native
    ``F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)``)::

        from tac.local_acceleration.pr95_hnerv_mlx import (
            bilinear_resize2x_align_corners_false_nhwc,
        )

        identity = bilinear_resize2x_align_corners_false_nhwc(x)

    For arbitrary target sizes (NOT 2x), use the sister general-form
    helper :func:`bilinear_resize_nhwc` instead (added 2026-05-26 in
    same CONSOLIDATE-OP-1 wave to cover D=Z6's general-resize signature).

    Args:
        x: NHWC tensor with shape ``(B, H, W, C)``.

    Returns:
        NHWC tensor with shape ``(B, 2*H, 2*W, C)``.

    Raises:
        ValueError: ``len(x.shape) != 4``.
    """

    require_mlx()
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC tensor, got shape {x.shape}")

    left = mx.concatenate([x[:, :, :1, :], x[:, :, :-1, :]], axis=2)  # type: ignore[union-attr]
    right = mx.concatenate([x[:, :, 1:, :], x[:, :, -1:, :]], axis=2)  # type: ignore[union-attr]
    even_w = x * 0.75 + left * 0.25
    odd_w = x * 0.75 + right * 0.25
    width_up = mx.reshape(  # type: ignore[union-attr]
        mx.stack([even_w, odd_w], axis=3),  # type: ignore[union-attr]
        (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]) * 2, int(x.shape[3])),
    )

    top = mx.concatenate([width_up[:, :1, :, :], width_up[:, :-1, :, :]], axis=1)  # type: ignore[union-attr]
    bottom = mx.concatenate([width_up[:, 1:, :, :], width_up[:, -1:, :, :]], axis=1)  # type: ignore[union-attr]
    even_h = width_up * 0.75 + top * 0.25
    odd_h = width_up * 0.75 + bottom * 0.25
    return mx.reshape(  # type: ignore[union-attr]
        mx.stack([even_h, odd_h], axis=2),  # type: ignore[union-attr]
        (
            int(width_up.shape[0]),
            int(width_up.shape[1]) * 2,
            int(width_up.shape[2]),
            int(width_up.shape[3]),
        ),
    )


def bilinear_resize_nhwc(
    x: Any,
    *,
    target_h: int,
    target_w: int,
    align_corners: bool = False,
) -> Any:
    """CANONICAL generalized bilinear resize for NHWC tensors (arbitrary target shape).

    Canonical PR95 helper for use by Path 3 MLX substrates that need
    PyTorch-byte-stable bilinear resize to arbitrary ``(target_h, target_w)``.
    Sister substrates MUST import and delegate to this helper rather than
    re-implement local copies (per CONSOLIDATE-OP-1 2026-05-26 extraction
    wave; canonical pattern matches D=Z6
    ``time_traveler_l5_z6.mlx_renderer::_bilinear_resize_nhwc`` signature
    so the migration is signature-compatible).

    Use this for the GENERAL case (target sizes that are not 2x of the
    input). For the specialized 2x case use :func:`bilinear_resize2x_align_corners_false_nhwc`
    (faster + closed-form weights ``0.75, 0.25``).

    Empirical drift bound vs PyTorch ``F.interpolate(size=(target_h,
    target_w), mode='bilinear', align_corners=False)``: ``≤ 1e-5`` absolute
    drift for fp32 inputs (anchor: sister D=Z6 conditional-resize call site
    at ``mlx_renderer.py:351`` — pixel boundaries computed via
    ``(dst + 0.5) / scale - 0.5`` mapping then clamped to ``[0, src - 1]``).

    Implementation: align_corners=False formula
    ``src_y = (dst_y + 0.5) * (src_h / target_h) - 0.5``
    ``src_x = (dst_x + 0.5) * (src_w / target_w) - 0.5``
    clamped to valid range; floor + 1 for bilinear corners; fractional
    interpolation weights.

    Identity short-circuit: when ``(target_h, target_w) == (H, W)`` the
    helper returns ``x`` unchanged (no reshape, no work).

    Canonical invocation pattern for substrates (Catalog #295 inflate
    self-containment scope — MLX training-time only; substrate inflate.py
    runtime is PyTorch-only and uses native
    ``F.interpolate(size=..., mode='bilinear', align_corners=False)``)::

        from tac.local_acceleration.pr95_hnerv_mlx import bilinear_resize_nhwc

        if int(x.shape[1]) != target_h or int(x.shape[2]) != target_w:
            x = bilinear_resize_nhwc(x, target_h=target_h, target_w=target_w)

    Args:
        x: NHWC tensor with shape ``(B, H, W, C)``.
        target_h: Target output height (positive int).
        target_w: Target output width (positive int).
        align_corners: PyTorch ``F.interpolate(..., align_corners=...)``
            convention. Only ``False`` is supported in this canonical
            helper (matches PyTorch default + sister D=Z6 + sister A=DreamerV3
            + sister F=Z8 canonical convention). ``True`` raises ``ValueError``.

    Returns:
        NHWC tensor with shape ``(B, target_h, target_w, C)``.

    Raises:
        ValueError: ``len(x.shape) != 4``, ``target_h <= 0``,
            ``target_w <= 0``, or ``align_corners != False``.
    """

    require_mlx()
    if align_corners is not False:
        raise ValueError(
            "bilinear_resize_nhwc: only align_corners=False is supported "
            "(canonical PyTorch default; matches sister Path 3 substrate "
            "convention). Got align_corners="
            f"{align_corners!r}"
        )
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC; got shape {x.shape}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"target_h and target_w must be positive; got "
            f"({target_h}, {target_w})"
        )
    batch, h_in, w_in, channels = (int(d) for d in x.shape)
    if h_in == target_h and w_in == target_w:
        return x
    # PyTorch align_corners=False mapping: src = (dst + 0.5) * (src/target) - 0.5
    h_scale = h_in / target_h
    w_scale = w_in / target_w
    h_idx_f = (mx.arange(target_h, dtype=mx.float32) + 0.5) * h_scale - 0.5  # type: ignore[union-attr]
    w_idx_f = (mx.arange(target_w, dtype=mx.float32) + 0.5) * w_scale - 0.5  # type: ignore[union-attr]
    h_idx_f = mx.clip(h_idx_f, 0.0, h_in - 1.0)  # type: ignore[union-attr]
    w_idx_f = mx.clip(w_idx_f, 0.0, w_in - 1.0)  # type: ignore[union-attr]
    h_lo = mx.floor(h_idx_f).astype(mx.int32)  # type: ignore[union-attr]
    h_hi = mx.minimum(h_lo + 1, h_in - 1)  # type: ignore[union-attr]
    w_lo = mx.floor(w_idx_f).astype(mx.int32)  # type: ignore[union-attr]
    w_hi = mx.minimum(w_lo + 1, w_in - 1)  # type: ignore[union-attr]
    h_frac = h_idx_f - mx.floor(h_idx_f)  # type: ignore[union-attr]
    w_frac = w_idx_f - mx.floor(w_idx_f)  # type: ignore[union-attr]
    h_lo_b = h_lo[:, None]
    h_hi_b = h_hi[:, None]
    w_lo_b = w_lo[None, :]
    w_hi_b = w_hi[None, :]
    tl = x[:, h_lo_b, w_lo_b, :]
    tr = x[:, h_lo_b, w_hi_b, :]
    bl = x[:, h_hi_b, w_lo_b, :]
    br = x[:, h_hi_b, w_hi_b, :]
    h_frac_b = mx.reshape(h_frac, (1, target_h, 1, 1))  # type: ignore[union-attr]
    w_frac_b = mx.reshape(w_frac, (1, 1, target_w, 1))  # type: ignore[union-attr]
    top = tl * (1.0 - w_frac_b) + tr * w_frac_b
    bot = bl * (1.0 - w_frac_b) + br * w_frac_b
    return top * (1.0 - h_frac_b) + bot * h_frac_b


class _PR95Conv2dMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PR95 Conv2d with explicit accumulation-mode selection.

    ``optimized`` uses native ``mx.conv2d`` for training throughput. The fixed
    modes delegate to the shared reference Conv2d used by scorer drift probes,
    so PR95 parity/debug paths do not grow a second implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    ) -> None:
        require_mlx()
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair_int(kernel_size, label="kernel_size")
        self.stride = _pair_int(stride, label="stride")
        self.padding = _pair_int(padding, label="padding")
        self.dilation = _pair_int(dilation, label="dilation")
        self.groups = int(groups)
        self.conv2d_accumulation_mode = validate_pr95_mlx_conv2d_accumulation_mode(
            conv2d_accumulation_mode
        )
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if self.in_channels % self.groups != 0:
            raise ValueError(
                f"in_channels {self.in_channels} not divisible by groups {self.groups}"
            )
        native = nn.Conv2d(  # type: ignore[union-attr]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bool(bias),
        )
        self.weight = native.weight
        self.bias = native.bias if bool(bias) else None

    def __call__(self, x: Any) -> Any:
        if self.conv2d_accumulation_mode == PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE:
            out = mx.conv2d(  # type: ignore[union-attr]
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            return out if self.bias is None else out + self.bias
        return mlx_reference_conv2d_nhwc(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            accumulation_mode=self.conv2d_accumulation_mode,
        )


class _HNeRVUpsampleBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
        skip_conv2d_accumulation_mode: str | None = None,
    ) -> None:
        require_mlx()
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv2d_accumulation_mode = validate_pr95_mlx_conv2d_accumulation_mode(
            conv2d_accumulation_mode
        )
        self.skip_conv2d_accumulation_mode = validate_pr95_mlx_conv2d_accumulation_mode(
            skip_conv2d_accumulation_mode or self.conv2d_accumulation_mode
        )
        self.conv = _PR95Conv2dMLX(
            in_channels,
            out_channels * 4,
            3,
            padding=1,
            conv2d_accumulation_mode=self.conv2d_accumulation_mode,
        )
        self.skip_conv = (
            _PR95Conv2dMLX(
                in_channels,
                out_channels,
                1,
                conv2d_accumulation_mode=self.skip_conv2d_accumulation_mode,
            )
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: Any) -> Any:
        identity = bilinear_resize2x_align_corners_false_nhwc(x)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        decoded = pixel_shuffle_2x_nhwc(self.conv(x))
        return mx.sin(decoded + identity)  # type: ignore[union-attr]


class HNeRVDecoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PR95 HNeRV decoder in native MLX/NHWC form.

    The public PR95 PyTorch model returns ``(B, 2, 3, 384, 512)``.  This MLX
    port keeps MLX's NHWC kernels internally and returns that same N2CHW layout
    by default for export/parity tests.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 28,
        base_channels: int = 36,
        eval_size: tuple[int, int] = (384, 512),
        output_layout: str = "n2chw",
        conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
        conv2d_accumulation_overrides: Mapping[str, str] | None = None,
    ) -> None:
        require_mlx()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.eval_size = tuple(int(dim) for dim in eval_size)
        self.base_h = 6
        self.base_w = 8
        self.output_layout = output_layout
        self.conv2d_accumulation_mode = validate_pr95_mlx_conv2d_accumulation_mode(
            conv2d_accumulation_mode
        )
        self.conv2d_accumulation_overrides = (
            validate_pr95_mlx_conv2d_accumulation_overrides(
                conv2d_accumulation_overrides
            )
        )
        if self.eval_size != (self.base_h * 64, self.base_w * 64):
            raise ValueError(
                "PR95 decoder topology fixes eval_size to "
                f"{(self.base_h * 64, self.base_w * 64)}, got {self.eval_size}"
            )
        if output_layout not in {"n2chw", "n2hwc"}:
            raise ValueError("output_layout must be 'n2chw' or 'n2hwc'")
        channels = [
            self.base_channels,
            self.base_channels,
            self.base_channels,
            int(self.base_channels * 0.75),
            int(self.base_channels * 0.58),
            int(self.base_channels * 0.5),
            int(self.base_channels * 0.5),
        ]
        if min(channels) < 1:
            raise ValueError("base_channels too small for PR95 channel taper")
        self.channels = channels

        def conv_mode_for(name: str) -> str:
            return self.conv2d_accumulation_overrides.get(
                name,
                self.conv2d_accumulation_mode,
            )

        self.stem = nn.Linear(  # type: ignore[union-attr]
            self.latent_dim,
            channels[0] * self.base_h * self.base_w,
        )
        self.blocks = [
            _HNeRVUpsampleBlockMLX(
                channels[i],
                channels[i + 1],
                conv2d_accumulation_mode=conv_mode_for(f"blocks.{i}.conv"),
                skip_conv2d_accumulation_mode=conv_mode_for(
                    f"blocks.{i}.skip_conv"
                ),
            )
            for i in range(6)
        ]
        final_ch = channels[-1]
        self.refine0 = _PR95Conv2dMLX(
            final_ch,
            final_ch // 2,
            3,
            padding=2,
            dilation=2,
            conv2d_accumulation_mode=conv_mode_for("refine0"),
        )
        self.refine1 = _PR95Conv2dMLX(
            final_ch // 2,
            final_ch,
            3,
            padding=1,
            conv2d_accumulation_mode=conv_mode_for("refine1"),
        )
        self.rgb_0 = _PR95Conv2dMLX(
            final_ch,
            3,
            3,
            padding=1,
            conv2d_accumulation_mode=conv_mode_for("rgb_0"),
        )
        self.rgb_1 = _PR95Conv2dMLX(
            final_ch,
            3,
            3,
            padding=1,
            conv2d_accumulation_mode=conv_mode_for("rgb_1"),
        )

    def features_nhwc(self, z: Any) -> Any:
        batch = int(z.shape[0])
        x = self.stem(z)
        x = mx.reshape(  # type: ignore[union-attr]
            x,
            (batch, self.channels[0], self.base_h, self.base_w),
        )
        x = mx.transpose(x, (0, 2, 3, 1))  # type: ignore[union-attr]
        x = mx.sin(x)  # type: ignore[union-attr]
        for block in self.blocks:
            x = block(x)
        refined = self.refine1(self.refine0(x))
        return x + 0.1 * mx.sin(refined)  # type: ignore[union-attr]

    def decode_pair_nhwc(self, z: Any) -> Any:
        x = self.features_nhwc(z)
        f0 = mx.sigmoid(self.rgb_0(x)) * 255.0  # type: ignore[union-attr]
        f1 = mx.sigmoid(self.rgb_1(x)) * 255.0  # type: ignore[union-attr]
        return mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]

    def __call__(self, z: Any) -> Any:
        pair = self.decode_pair_nhwc(z)
        if self.output_layout == "n2hwc":
            return pair
        return mx.transpose(pair, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": "pr95_hnerv_mlx_architecture_v1",
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "eval_size": list(self.eval_size),
            "base_grid": [self.base_h, self.base_w],
            "channels": list(self.channels),
            "upsample_blocks": 6,
            "internal_layout": "NHWC",
            "default_output_layout": self.output_layout,
            "conv2d_accumulation_mode": self.conv2d_accumulation_mode,
            "conv2d_accumulation_overrides": dict(
                self.conv2d_accumulation_overrides
            ),
            "decoder_param_count": _param_count_from_tree(self.parameters()),
            "source_pr": 95,
            "source_architecture": "submissions/hnerv_muon/src/model.py::HNeRVDecoder",
        }


class HNeRVSyntheticTrainingBundleMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Decoder plus trainable per-pair latents for timing-only MLX smokes."""

    def __init__(
        self,
        *,
        latent_count: int,
        latent_dim: int = 28,
        base_channels: int = 36,
        seed: int = 0,
        output_layout: str = "n2chw",
        conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
        conv2d_accumulation_overrides: Mapping[str, str] | None = None,
    ) -> None:
        require_mlx()
        super().__init__()
        key = mx.random.key(seed)  # type: ignore[union-attr]
        self.latents = mx.random.normal((latent_count, latent_dim), key=key) * 0.1  # type: ignore[union-attr]
        self.decoder = HNeRVDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_layout=output_layout,
            conv2d_accumulation_mode=conv2d_accumulation_mode,
            conv2d_accumulation_overrides=conv2d_accumulation_overrides,
        )

    def __call__(self, indices: Any) -> Any:
        return self.decoder(mx.take(self.latents, indices, axis=0))  # type: ignore[union-attr]


def load_pytorch_state_dict_into_mlx(
    model: HNeRVDecoderMLX,
    state_dict: dict[str, Any],
) -> None:
    """Load a PR95 PyTorch ``state_dict`` into an MLX decoder."""

    require_mlx()
    flat = dict(tree_flatten(model.parameters()))  # type: ignore[misc]

    def set_param(path: str, value: Any) -> None:
        if path not in flat:
            raise KeyError(f"MLX model has no parameter path {path!r}")
        flat[path] = value

    set_param("stem.weight", _mlx_array_from_any(state_dict["stem.weight"]))
    set_param("stem.bias", _mlx_array_from_any(state_dict["stem.bias"]))
    for index in range(6):
        set_param(
            f"blocks.{index}.conv.weight",
            _torch_conv_to_mlx(state_dict[f"blocks.{index}.weight"]),
        )
        set_param(
            f"blocks.{index}.conv.bias",
            _mlx_array_from_any(state_dict[f"blocks.{index}.bias"]),
        )
        skip_weight = f"skips.{index}.weight"
        if skip_weight in state_dict:
            set_param(
                f"blocks.{index}.skip_conv.weight",
                _torch_conv_to_mlx(state_dict[skip_weight]),
            )
            set_param(
                f"blocks.{index}.skip_conv.bias",
                _mlx_array_from_any(state_dict[f"skips.{index}.bias"]),
            )

    set_param("refine0.weight", _torch_conv_to_mlx(state_dict["refine.0.weight"]))
    set_param("refine0.bias", _mlx_array_from_any(state_dict["refine.0.bias"]))
    set_param("refine1.weight", _torch_conv_to_mlx(state_dict["refine.1.weight"]))
    set_param("refine1.bias", _mlx_array_from_any(state_dict["refine.1.bias"]))
    for head in ("rgb_0", "rgb_1"):
        set_param(f"{head}.weight", _torch_conv_to_mlx(state_dict[f"{head}.weight"]))
        set_param(f"{head}.bias", _mlx_array_from_any(state_dict[f"{head}.bias"]))

    model.update(tree_unflatten(list(flat.items())))  # type: ignore[misc]


def pytorch_state_dict_from_mlx(
    model: HNeRVDecoderMLX,
    *,
    as_torch: bool = False,
) -> dict[str, Any]:
    """Export MLX decoder parameters using the public PR95 PyTorch names."""

    require_mlx()
    flat = dict(tree_flatten(model.parameters()))  # type: ignore[misc]
    exported: dict[str, np.ndarray] = {
        "stem.weight": np.asarray(flat["stem.weight"]),
        "stem.bias": np.asarray(flat["stem.bias"]),
    }
    for index in range(6):
        exported[f"blocks.{index}.weight"] = _mlx_conv_to_numpy(
            flat[f"blocks.{index}.conv.weight"]
        )
        exported[f"blocks.{index}.bias"] = np.asarray(flat[f"blocks.{index}.conv.bias"])
        skip_weight = f"blocks.{index}.skip_conv.weight"
        if skip_weight in flat:
            exported[f"skips.{index}.weight"] = _mlx_conv_to_numpy(flat[skip_weight])
            exported[f"skips.{index}.bias"] = np.asarray(
                flat[f"blocks.{index}.skip_conv.bias"]
            )
    exported["refine.0.weight"] = _mlx_conv_to_numpy(flat["refine0.weight"])
    exported["refine.0.bias"] = np.asarray(flat["refine0.bias"])
    exported["refine.1.weight"] = _mlx_conv_to_numpy(flat["refine1.weight"])
    exported["refine.1.bias"] = np.asarray(flat["refine1.bias"])
    for head in ("rgb_0", "rgb_1"):
        exported[f"{head}.weight"] = _mlx_conv_to_numpy(flat[f"{head}.weight"])
        exported[f"{head}.bias"] = np.asarray(flat[f"{head}.bias"])
    if not as_torch:
        return exported
    import torch

    return {name: torch.from_numpy(value.copy()) for name, value in exported.items()}


def _sample_indices_for_pr95_packet(
    total: int,
    sample_indices: Sequence[int] | None,
) -> list[int]:
    if total < 1:
        raise Pr95HNeRVMlxError("PR95 packet has no latent rows")
    if sample_indices is None:
        sample_indices = (0, total // 2, total - 1)
    out: list[int] = []
    for raw_index in sample_indices:
        index = int(raw_index)
        if index < 0 or index >= total:
            raise Pr95HNeRVMlxError(
                f"sample index {index} out of range for {total} PR95 latent rows"
            )
        if index not in out:
            out.append(index)
    return out


def _mlx_device_from_name(device: str) -> Any:
    require_mlx()
    normalized = device.lower()
    if normalized == "cpu":
        return mx.cpu  # type: ignore[union-attr]
    if normalized == "gpu":
        return mx.gpu  # type: ignore[union-attr]
    raise ValueError("mlx_device must be 'cpu' or 'gpu'")


def compare_pr95_public_archive_forward_with_pytorch(
    packet: Pr95PublicArchivePacket,
    torch_decoder_cls: Any,
    *,
    sample_indices: Sequence[int] | None = None,
    mlx_device: str = "cpu",
    atol_max: float = 2e-3,
    atol_mean: float = 1e-4,
    conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    conv2d_accumulation_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare MLX against PyTorch on decoded public PR95 packet state.

    The result is a local implementation-parity probe.  It deliberately remains
    non-promotable and cannot claim contest score authority.
    """

    require_mlx()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise Pr95HNeRVMlxError("torch is required for PR95 parity probes") from exc

    meta = packet.meta
    conv_mode = validate_pr95_mlx_conv2d_accumulation_mode(conv2d_accumulation_mode)
    conv_overrides = validate_pr95_mlx_conv2d_accumulation_overrides(
        conv2d_accumulation_overrides
    )
    indices = _sample_indices_for_pr95_packet(int(packet.latents.shape[0]), sample_indices)
    z_np = packet.latents[indices].astype(np.float32, copy=False)
    torch_state_dict = {
        name: torch.from_numpy(value.astype(np.float32, copy=True))
        for name, value in packet.state_dict.items()
    }
    torch_model = torch_decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(int(dim) for dim in meta["eval_size"]),
    ).eval()
    torch_model.load_state_dict(torch_state_dict)

    previous_device = mx.default_device()  # type: ignore[union-attr]
    mx.set_default_device(_mlx_device_from_name(mlx_device))  # type: ignore[union-attr]
    try:
        mlx_model = HNeRVDecoderMLX(
            latent_dim=int(meta["latent_dim"]),
            base_channels=int(meta["base_channels"]),
            eval_size=tuple(int(dim) for dim in meta["eval_size"]),
            conv2d_accumulation_mode=conv_mode,
            conv2d_accumulation_overrides=conv_overrides,
        )
        load_pytorch_state_dict_into_mlx(mlx_model, packet.state_dict)
        started = time.perf_counter()
        with torch.no_grad():
            torch_output = torch_model(torch.from_numpy(z_np)).detach().cpu().numpy()
        mlx_output = mlx_model(mx.array(z_np))  # type: ignore[union-attr]
        mx.eval(mlx_output)  # type: ignore[union-attr]
        elapsed = time.perf_counter() - started
        mlx_output_np = np.asarray(mlx_output)
    finally:
        mx.set_default_device(previous_device)  # type: ignore[union-attr]

    diff = np.abs(torch_output - mlx_output_np)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    p99_abs = float(np.quantile(diff, 0.99)) if diff.size else 0.0
    p999_abs = float(np.quantile(diff, 0.999)) if diff.size else 0.0
    passed = max_abs <= float(atol_max) and mean_abs <= float(atol_mean)
    blockers = [
        "local_mlx_forward_parity_probe_is_not_contest_auth_eval",
        "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    ]
    if not passed:
        blockers.append("pytorch_mlx_forward_drift_exceeds_configured_tolerance")
    return {
        "schema": PUBLIC_ARCHIVE_FORWARD_PARITY_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "source_pr": 95,
        "submission": "hnerv_muon",
        "evidence_grade": "[macOS-MLX research-signal]",
        "mlx_device": mlx_device,
        "conv2d_accumulation_mode": conv_mode,
        "conv2d_accumulation_overrides": conv_overrides,
        "sample_indices": indices,
        "sample_count": len(indices),
        "elapsed_seconds": elapsed,
        "torch_output_shape": [int(dim) for dim in torch_output.shape],
        "mlx_output_shape": [int(dim) for dim in mlx_output_np.shape],
        "parity": {
            "passed": passed,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "p99_abs": p99_abs,
            "p999_abs": p999_abs,
            "atol_max": float(atol_max),
            "atol_mean": float(atol_mean),
        },
        "archive_packet": packet.custody_manifest(),
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }


def trace_pr95_public_archive_decoder_with_pytorch(
    packet: Pr95PublicArchivePacket,
    torch_decoder_cls: Any,
    *,
    sample_indices: Sequence[int] | None = None,
    mlx_device: str = "cpu",
    cliff_threshold: float = 1.0e-5,
    conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    conv2d_accumulation_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Trace PR95 decoder boundary drift between PyTorch and MLX.

    This is a diagnostic implementation-parity manifest.  It intentionally
    carries the same false-authority contract as the final-output parity probe:
    boundary traces localize MLX drift, but never replace full-frame inflate
    parity or contest CPU/CUDA auth eval.
    """

    require_mlx()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise Pr95HNeRVMlxError("torch is required for PR95 trace probes") from exc

    if float(cliff_threshold) < 0.0:
        raise Pr95HNeRVMlxError(
            f"cliff_threshold must be >= 0, got {cliff_threshold}"
        )

    meta = packet.meta
    conv_mode = validate_pr95_mlx_conv2d_accumulation_mode(conv2d_accumulation_mode)
    conv_overrides = validate_pr95_mlx_conv2d_accumulation_overrides(
        conv2d_accumulation_overrides
    )
    indices = _sample_indices_for_pr95_packet(int(packet.latents.shape[0]), sample_indices)
    z_np = packet.latents[indices].astype(np.float32, copy=False)
    torch_state_dict = {
        name: torch.from_numpy(value.astype(np.float32, copy=True))
        for name, value in packet.state_dict.items()
    }
    torch_model = torch_decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(int(dim) for dim in meta["eval_size"]),
    ).eval()
    torch_model.load_state_dict(torch_state_dict)

    previous_device = mx.default_device()  # type: ignore[union-attr]
    mx.set_default_device(_mlx_device_from_name(mlx_device))  # type: ignore[union-attr]
    try:
        mlx_model = HNeRVDecoderMLX(
            latent_dim=int(meta["latent_dim"]),
            base_channels=int(meta["base_channels"]),
            eval_size=tuple(int(dim) for dim in meta["eval_size"]),
            conv2d_accumulation_mode=conv_mode,
            conv2d_accumulation_overrides=conv_overrides,
        )
        load_pytorch_state_dict_into_mlx(mlx_model, packet.state_dict)
        started = time.perf_counter()
        with torch.no_grad():
            torch_trace = _trace_torch_pr95_decoder(torch_model, z_np)
        mlx_trace = _trace_mlx_pr95_decoder(mlx_model, z_np)
        elapsed = time.perf_counter() - started
    finally:
        mx.set_default_device(previous_device)  # type: ignore[union-attr]

    rows = _decoder_trace_rows(
        torch_trace=torch_trace,
        mlx_trace=mlx_trace,
        cliff_threshold=float(cliff_threshold),
    )
    drift_cliff = next((row for row in rows if row["exceeds_cliff_threshold"]), None)
    output_row = next((row for row in rows if row["name"] == "output"), None)
    return {
        "schema": PUBLIC_ARCHIVE_DECODER_TRACE_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "source_pr": 95,
        "submission": "hnerv_muon",
        "evidence_grade": "[macOS-MLX research-signal]",
        "mlx_device": mlx_device,
        "conv2d_accumulation_mode": conv_mode,
        "conv2d_accumulation_overrides": conv_overrides,
        "sample_indices": indices,
        "sample_count": len(indices),
        "elapsed_seconds": elapsed,
        "cliff_threshold": float(cliff_threshold),
        "trace_count": len(rows),
        "drift_cliff": drift_cliff,
        "output_delta": output_row,
        "rows": rows,
        "archive_packet": packet.custody_manifest(),
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "local_mlx_decoder_trace_probe_is_not_contest_auth_eval",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        "authority_status": (
            "PR95 PyTorch-vs-MLX decoder traces are local diagnostic "
            "implementation evidence only; full-frame inflate parity and exact "
            "contest CPU/CUDA auth eval remain required for score claims and "
            "promotion."
        ),
        **FALSE_AUTHORITY,
    }


def _trace_torch_pr95_decoder(torch_model: Any, z_np: np.ndarray) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    z = torch.from_numpy(np.asarray(z_np, dtype=np.float32, order="C"))
    batch = int(z.shape[0])
    trace: dict[str, np.ndarray] = {}
    x = torch_model.stem(z).view(
        batch,
        torch_model.channels[0],
        torch_model.base_h,
        torch_model.base_w,
    )
    trace["stem.view"] = _torch_tensor_to_numpy(x)
    x = torch.sin(x)
    trace["stem.sin"] = _torch_tensor_to_numpy(x)
    for index, (block, skip) in enumerate(
        zip(torch_model.blocks, torch_model.skips, strict=True)
    ):
        identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        trace[f"blocks.{index}.identity_resize"] = _torch_tensor_to_numpy(identity)
        identity = skip(identity)
        trace[f"blocks.{index}.identity_skip"] = _torch_tensor_to_numpy(identity)
        decoded = torch_model.ps(block(x))
        trace[f"blocks.{index}.pixel_shuffle"] = _torch_tensor_to_numpy(decoded)
        x = torch.sin(decoded + identity)
        trace[f"blocks.{index}.output"] = _torch_tensor_to_numpy(x)
    refined = torch_model.refine(x)
    trace["refine.residual"] = _torch_tensor_to_numpy(refined)
    x = x + 0.1 * torch.sin(refined)
    trace["features"] = _torch_tensor_to_numpy(x)
    f0 = torch.sigmoid(torch_model.rgb_0(x)) * 255.0
    f1 = torch.sigmoid(torch_model.rgb_1(x)) * 255.0
    trace["rgb_0"] = _torch_tensor_to_numpy(f0)
    trace["rgb_1"] = _torch_tensor_to_numpy(f1)
    trace["output"] = _torch_tensor_to_numpy(torch.stack([f0, f1], dim=1))
    return trace


def _trace_mlx_pr95_decoder(model: HNeRVDecoderMLX, z_np: np.ndarray) -> dict[str, np.ndarray]:
    require_mlx()
    z = mx.array(np.asarray(z_np, dtype=np.float32, order="C"))  # type: ignore[union-attr]
    batch = int(z.shape[0])
    trace: dict[str, np.ndarray] = {}
    x = model.stem(z)
    x = mx.reshape(  # type: ignore[union-attr]
        x,
        (batch, model.channels[0], model.base_h, model.base_w),
    )
    trace["stem.view"] = _mlx_nchw_to_numpy(x)
    x = mx.transpose(x, (0, 2, 3, 1))  # type: ignore[union-attr]
    x = mx.sin(x)  # type: ignore[union-attr]
    trace["stem.sin"] = _mlx_nhwc_to_nchw_numpy(x)
    for index, block in enumerate(model.blocks):
        identity = bilinear_resize2x_align_corners_false_nhwc(x)
        trace[f"blocks.{index}.identity_resize"] = _mlx_nhwc_to_nchw_numpy(identity)
        if block.skip_conv is not None:
            identity = block.skip_conv(identity)
        trace[f"blocks.{index}.identity_skip"] = _mlx_nhwc_to_nchw_numpy(identity)
        decoded = pixel_shuffle_2x_nhwc(block.conv(x))
        trace[f"blocks.{index}.pixel_shuffle"] = _mlx_nhwc_to_nchw_numpy(decoded)
        x = mx.sin(decoded + identity)  # type: ignore[union-attr]
        trace[f"blocks.{index}.output"] = _mlx_nhwc_to_nchw_numpy(x)
    refined0 = model.refine0(x)
    refined = model.refine1(refined0)
    trace["refine.residual"] = _mlx_nhwc_to_nchw_numpy(refined)
    x = x + 0.1 * mx.sin(refined)  # type: ignore[union-attr]
    trace["features"] = _mlx_nhwc_to_nchw_numpy(x)
    f0 = mx.sigmoid(model.rgb_0(x)) * 255.0  # type: ignore[union-attr]
    f1 = mx.sigmoid(model.rgb_1(x)) * 255.0  # type: ignore[union-attr]
    trace["rgb_0"] = _mlx_nhwc_to_nchw_numpy(f0)
    trace["rgb_1"] = _mlx_nhwc_to_nchw_numpy(f1)
    pair = mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]
    trace["output"] = _mlx_n2hwc_to_n2chw_numpy(pair)
    return trace


def _torch_tensor_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=True)


def _mlx_nchw_to_numpy(value: Any) -> np.ndarray:
    mx.eval(value)  # type: ignore[union-attr]
    return np.asarray(value, dtype=np.float32).copy()


def _mlx_nhwc_to_nchw_numpy(value: Any) -> np.ndarray:
    mx.eval(value)  # type: ignore[union-attr]
    return np.transpose(np.asarray(value, dtype=np.float32), (0, 3, 1, 2)).copy()


def _mlx_n2hwc_to_n2chw_numpy(value: Any) -> np.ndarray:
    mx.eval(value)  # type: ignore[union-attr]
    return np.transpose(np.asarray(value, dtype=np.float32), (0, 1, 4, 2, 3)).copy()


def _decoder_trace_rows(
    *,
    torch_trace: dict[str, np.ndarray],
    mlx_trace: dict[str, np.ndarray],
    cliff_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(torch_trace):
        lhs = torch_trace[name]
        rhs = mlx_trace.get(name)
        if rhs is None:
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "present_in_torch": True,
                    "present_in_mlx": False,
                    "shape_match": False,
                    "exceeds_cliff_threshold": True,
                    "blockers": ["missing_mlx_trace"],
                }
            )
            continue
        summary = _trace_array_delta_summary(lhs, rhs)
        rows.append(
            {
                "index": index,
                "name": name,
                **summary,
                "exceeds_cliff_threshold": (
                    summary.get("max_abs_delta") is None
                    or float(summary["max_abs_delta"]) > cliff_threshold
                ),
            }
        )
    for name in mlx_trace:
        if name not in torch_trace:
            rows.append(
                {
                    "index": len(rows),
                    "name": name,
                    "present_in_torch": False,
                    "present_in_mlx": True,
                    "shape_match": False,
                    "exceeds_cliff_threshold": True,
                    "blockers": ["missing_torch_trace"],
                }
            )
    return rows


def _trace_array_delta_summary(lhs_value: np.ndarray, rhs_value: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(lhs_value, dtype=np.float32)
    rhs = np.asarray(rhs_value, dtype=np.float32)
    if lhs.shape != rhs.shape:
        return {
            "present_in_torch": True,
            "present_in_mlx": True,
            "shape_match": False,
            "torch_shape": [int(dim) for dim in lhs.shape],
            "mlx_shape": [int(dim) for dim in rhs.shape],
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "rms_delta": None,
            "p99_abs_delta": None,
            "blockers": ["shape_mismatch"],
        }
    diff = np.abs(lhs - rhs).astype(np.float64, copy=False)
    return {
        "present_in_torch": True,
        "present_in_mlx": True,
        "shape_match": True,
        "torch_shape": [int(dim) for dim in lhs.shape],
        "mlx_shape": [int(dim) for dim in rhs.shape],
        "max_abs_delta": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_delta": float(np.mean(diff)) if diff.size else 0.0,
        "rms_delta": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "p99_abs_delta": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        "blockers": [],
    }


def write_pr95_public_archive_pytorch_export_forward_parity(
    packet: Pr95PublicArchivePacket,
    torch_decoder_cls: Any,
    *,
    output_pt_path: Path,
    run_id: str,
    sample_indices: Sequence[int] | None = None,
    mlx_device: str = "cpu",
    atol_max: float = 2e-3,
    atol_mean: float = 1e-4,
    conv2d_accumulation_mode: str = PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    conv2d_accumulation_overrides: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export a decoded PR95 MLX archive checkpoint to ``.pt`` and prove parity.

    This is the queue-owned bridge from native MLX training artifacts to the
    public PR95 PyTorch runtime.  It proves that the exported checkpoint can be
    loaded by the source PyTorch decoder and that PyTorch and MLX agree on the
    selected latent rows.  It is still local implementation evidence only:
    full-frame inflate parity and exact CPU/CUDA auth eval remain separate gates.
    """

    from tac.local_acceleration.mlx_to_pytorch_export import (
        export_mlx_state_dict_to_torch_pt,
    )

    output_pt_path = Path(output_pt_path)
    export_manifest = export_mlx_state_dict_to_torch_pt(
        {
            name: np.asarray(value, dtype=np.float32)
            for name, value in packet.state_dict.items()
        },
        output_pt_path,
        substrate_id="pr95_hnerv_mlx",
        run_id=run_id,
        overwrite=overwrite,
    )
    forward_parity = compare_pr95_public_archive_forward_with_pytorch(
        packet,
        torch_decoder_cls,
        sample_indices=sample_indices,
        mlx_device=mlx_device,
        atol_max=atol_max,
        atol_mean=atol_mean,
        conv2d_accumulation_mode=conv2d_accumulation_mode,
        conv2d_accumulation_overrides=conv2d_accumulation_overrides,
    )
    parity = forward_parity.get("parity", {})
    passed = isinstance(parity, Mapping) and parity.get("passed") is True
    blockers = [
        "local_mlx_pytorch_export_parity_probe_is_not_contest_auth_eval",
        PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    ]
    if not passed:
        blockers.extend(
            [
                PR95_EXPORT_FORWARD_PARITY_BLOCKER,
                "requires_pytorch_export_forward_parity_on_source_checkpoint",
            ]
        )
    return {
        "schema": PR95_MLX_PYTORCH_EXPORT_FORWARD_PARITY_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "source_pr": 95,
        "submission": "hnerv_muon",
        "evidence_grade": "[macOS-MLX research-signal]",
        "run_id": run_id,
        "archive_packet": packet.custody_manifest(),
        "state_dict_pt_export": export_manifest,
        "pt_path": str(export_manifest["output_pt_path"]),
        "pt_sha256": str(export_manifest["file_sha256"]),
        "pt_bytes": int(export_manifest["file_size_bytes"]),
        "sample_indices": list(forward_parity["sample_indices"]),
        "sample_count": int(forward_parity["sample_count"]),
        "mlx_device": mlx_device,
        "conv2d_accumulation_mode": forward_parity["conv2d_accumulation_mode"],
        "conv2d_accumulation_overrides": forward_parity[
            "conv2d_accumulation_overrides"
        ],
        "pytorch_export_forward_parity_established": bool(passed),
        "forward_parity": forward_parity,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }


def partition_pr95_mlx_parameter_names(params: Any) -> dict[str, list[str]]:
    """Return the source-faithful PR95 stage-8 Muon/AdamW parameter split."""

    require_mlx()
    muon: list[str] = []
    adamw: list[str] = []
    for name, value in tree_flatten(params):  # type: ignore[misc]
        low = name.lower()
        ndim = len(getattr(value, "shape", ()))
        if (
            ndim >= 2
            and name.endswith("weight")
            and "stem" not in low
            and "rgb_" not in low
            and "latents" not in low
        ):
            muon.append(name)
        else:
            adamw.append(name)
    return {"muon": sorted(muon), "adamw": sorted(adamw)}


def pr95_mlx_parameter_shape_records(params: Any) -> list[dict[str, Any]]:
    """Return framework-neutral shape records for MLX parameter grouping."""

    require_mlx()
    return [
        {
            "name": str(name),
            "shape": [int(dim) for dim in getattr(value, "shape", ())],
        }
        for name, value in tree_flatten(params)  # type: ignore[misc]
    ]


def zeropower_via_newtonschulz5_mlx(
    gradient: Any,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    cast_float32_to_bfloat16: bool = True,
) -> Any:
    """PR95/Keller-Jordan Newton-Schulz orthogonalization in MLX."""

    require_mlx()
    if len(gradient.shape) != 2:
        raise ValueError(f"Newton-Schulz expects 2D input, got {gradient.shape}")
    original_dtype = gradient.dtype
    x = gradient.astype(mx.bfloat16) if cast_float32_to_bfloat16 else gradient  # type: ignore[union-attr]
    transposed = int(x.shape[-2]) > int(x.shape[-1])
    if transposed:
        x = x.T
    x = x / (mx.linalg.norm(x, keepdims=True) + eps)  # type: ignore[union-attr]
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.astype(original_dtype)


def _safe_descriptor_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def pr95_default_optimizer_descriptor_id(stage_index: int) -> str:
    """Return the default executable synthetic-timing descriptor for a PR95 stage."""

    try:
        return PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[int(stage_index)]
    except KeyError as exc:
        supported = sorted(PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS)
        raise ValueError(
            f"supported PR95 MLX timing stages are {supported}"
        ) from exc


def pr95_mlx_optimizer_descriptor_row(descriptor_id: str) -> dict[str, Any]:
    """Return a planning descriptor row from the canonical optimizer registry."""

    try:
        return default_optimizer_scheduler_registry().get(descriptor_id).to_planner_candidate()
    except OptimizerSchedulerRegistryError as exc:
        raise Pr95HNeRVMlxError(str(exc)) from exc


def _descriptor_stage_indices(descriptor: Mapping[str, Any]) -> list[int]:
    training_config = descriptor.get("training_config")
    if not isinstance(training_config, Mapping):
        return []
    indices = training_config.get("pr95_stage_indices")
    if not isinstance(indices, Sequence) or isinstance(indices, str | bytes):
        return []
    out: list[int] = []
    for index in indices:
        try:
            out.append(int(index))
        except (TypeError, ValueError):
            continue
    return out


def _as_bool(value: Any, *, default: bool = False) -> bool:
    return default if value is None else bool(value)


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _as_float(value, default=0.0)


def _as_betas(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        return (_as_float(value[0], default=0.9), _as_float(value[1], default=0.999))
    return (0.9, 0.999)


@dataclass(frozen=True)
class Pr95MlxOptimizerConfig:
    use_muon: bool
    adamw_lr: float = 3e-5
    latent_lr_mult: float = 10.0
    muon_lr: float = 2e-4
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    grad_clip_muon: float | None = 1.0
    cast_muon_float32_to_bfloat16: bool = True


def pr95_mlx_optimizer_config_from_descriptor(
    descriptor_id: str,
    *,
    stage_index: int,
) -> Pr95MlxOptimizerConfig:
    """Lower a PR95 MLX timing descriptor into the optimizer config."""

    descriptor = pr95_mlx_optimizer_descriptor_row(descriptor_id)
    training_config = descriptor.get("training_config")
    if not isinstance(training_config, Mapping):
        raise Pr95HNeRVMlxError(f"{descriptor_id}: descriptor missing training_config")
    backend_status = str(training_config.get("backend_status") or "")
    if backend_status != PR95_MLX_BACKEND_STATUS_LOCAL_TIMING_PROXY:
        raise Pr95HNeRVMlxError(
            f"{descriptor_id}: optimizer descriptor is not executable on MLX "
            f"(backend_status={backend_status or 'missing'})"
        )
    allowed_stages = _descriptor_stage_indices(descriptor)
    if int(stage_index) not in allowed_stages:
        raise Pr95HNeRVMlxError(
            f"{descriptor_id}: descriptor does not support PR95 stage {stage_index}"
        )
    optimizer_config = descriptor.get("optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        raise Pr95HNeRVMlxError(f"{descriptor_id}: descriptor missing optimizer_config")
    return Pr95MlxOptimizerConfig(
        use_muon=_as_bool(optimizer_config.get("use_muon")),
        adamw_lr=_as_float(optimizer_config.get("adamw_lr"), default=3e-5),
        latent_lr_mult=_as_float(
            optimizer_config.get("latent_lr_mult"),
            default=10.0,
        ),
        muon_lr=_as_float(optimizer_config.get("muon_lr"), default=2e-4),
        muon_momentum=_as_float(
            optimizer_config.get("muon_momentum"),
            default=0.95,
        ),
        muon_nesterov=_as_bool(optimizer_config.get("muon_nesterov"), default=True),
        muon_ns_steps=int(optimizer_config.get("muon_ns_steps") or 5),
        muon_weight_decay=_as_float(
            optimizer_config.get("muon_weight_decay"),
            default=0.0,
        ),
        adamw_betas=_as_betas(optimizer_config.get("adamw_betas")),
        adamw_eps=_as_float(optimizer_config.get("adamw_eps"), default=1e-8),
        adamw_weight_decay=_as_float(
            optimizer_config.get("adamw_weight_decay"),
            default=0.0,
        ),
        grad_clip=_as_optional_float(optimizer_config.get("grad_clip")),
        grad_clip_muon=_as_optional_float(optimizer_config.get("grad_clip_muon")),
        cast_muon_float32_to_bfloat16=_as_bool(
            optimizer_config.get("cast_muon_float32_to_bfloat16"),
            default=True,
        ),
    )


@dataclass
class Pr95MlxOptimizerState:
    step: int = 0
    muon_buffers: dict[str, Any] = field(default_factory=dict)
    adamw_m: dict[str, Any] = field(default_factory=dict)
    adamw_v: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pr95MlxStageSmokeConfig:
    stage_index: int
    stage_module: str
    optimizer: Pr95MlxOptimizerConfig
    optimizer_descriptor_id: str
    optimizer_config_sha256: str
    parameter_group_lr_policy: Mapping[str, Any]
    parameter_group_lr_policy_id: str
    parameter_group_lr_policy_sha256: str
    optimizer_backend_status: str
    source_stage_loss_contract: Mapping[str, Any]
    synthetic_loss: str = "normalized_rgb_pair_mse"


def stage_smoke_config(
    stage_index: int,
    *,
    optimizer_descriptor_id: str | None = None,
) -> Pr95MlxStageSmokeConfig:
    """Return the PR95-shaped optimizer switch for synthetic MLX timing stages."""

    if stage_index not in PR95_STAGE_MODULES:
        supported = sorted(PR95_STAGE_MODULES)
        raise ValueError(
            f"supported PR95 MLX timing stages are {supported}"
        )
    descriptor_id = optimizer_descriptor_id or pr95_default_optimizer_descriptor_id(
        stage_index
    )
    descriptor = pr95_mlx_optimizer_descriptor_row(descriptor_id)
    training_config = descriptor.get("training_config", {})
    optimizer = pr95_mlx_optimizer_config_from_descriptor(
        descriptor_id,
        stage_index=stage_index,
    )
    policy = descriptor["parameter_group_lr_policy"]
    return Pr95MlxStageSmokeConfig(
        stage_index=stage_index,
        stage_module=PR95_STAGE_MODULES[stage_index],
        optimizer=optimizer,
        optimizer_descriptor_id=descriptor_id,
        optimizer_config_sha256=str(descriptor["config_sha256"]),
        parameter_group_lr_policy=policy,
        parameter_group_lr_policy_id=str(descriptor["parameter_group_lr_policy_id"]),
        parameter_group_lr_policy_sha256=str(
            descriptor["parameter_group_lr_policy_sha256"]
        ),
        optimizer_backend_status=str(
            training_config.get("backend_status")
        ),
        source_stage_loss_contract=pr95_mlx_stage_loss_contract_from_training_config(
            training_config if isinstance(training_config, Mapping) else {},
            stage_index=stage_index,
        ),
    )


def _exact_readiness_blockers_for_timing_smoke(
    *,
    source_video_training: bool,
    source_faithfulness_blockers: Sequence[str],
    training_loss_surface: str,
) -> list[str]:
    blockers = [
        blocker
        for blocker in EXACT_READINESS_REFUSAL_BLOCKERS
        if blocker
        not in {
            "pr95_source_video_loader_not_ported_to_mlx",
            *PR95_MLX_SOURCE_FAITHFUL_BLOCKERS,
        }
    ]
    blockers.extend(str(blocker) for blocker in source_faithfulness_blockers)
    if source_video_training:
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "synthetic_targets_do_not_establish_contest_quality",
                "pr95_hnerv_mlx_training_is_synthetic_timing_only_not_source_faithful",
            }
        ]
        if training_loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE:
            blockers.append(
                "source_video_rgb_yuv6_preprocess_loss_is_not_score_authority"
            )
        else:
            blockers.append(PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER)
            blockers.append("source_video_rgb_timing_smoke_is_not_score_authority")
    return list(dict.fromkeys(blockers))


def _clip_flat_gradients(
    gradients: dict[str, Any],
    names: list[str],
    *,
    max_norm: float | None,
) -> None:
    require_mlx()
    if max_norm is None or max_norm <= 0 or not names:
        return
    norm_sq = None
    for name in names:
        grad = gradients.get(name)
        if grad is None:
            continue
        term = mx.sum(grad * grad)  # type: ignore[union-attr]
        norm_sq = term if norm_sq is None else norm_sq + term
    if norm_sq is None:
        return
    norm = mx.sqrt(norm_sq)  # type: ignore[union-attr]
    scale = mx.minimum(mx.array(1.0), mx.array(float(max_norm)) / (norm + 1e-6))  # type: ignore[union-attr]
    for name in names:
        if name in gradients and gradients[name] is not None:
            gradients[name] = gradients[name] * scale


def apply_pr95_mlx_optimizer_step(
    module: Any,
    gradients: Any,
    state: Pr95MlxOptimizerState,
    config: Pr95MlxOptimizerConfig,
    parameter_group_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one PR95-shaped optimizer step to an MLX module."""

    require_mlx()
    params_flat = dict(tree_flatten(module.parameters()))  # type: ignore[misc]
    grads_flat = dict(tree_flatten(gradients))  # type: ignore[misc]
    if parameter_group_fingerprint is None:
        parameter_group_fingerprint = build_parameter_group_lr_policy_fingerprint(
            pr95_mlx_parameter_shape_records(module.parameters())
        )
    parameter_classes = {
        str(record.get("name")): str(record.get("parameter_class"))
        for record in parameter_group_fingerprint.get("classification_records", [])
        if isinstance(record, Mapping)
    }
    split = partition_pr95_mlx_parameter_names(module.parameters())
    muon_names = split["muon"] if config.use_muon else []
    adamw_names = list(split["adamw"] + ([] if config.use_muon else split["muon"]))
    _clip_flat_gradients(grads_flat, adamw_names, max_norm=config.grad_clip)
    _clip_flat_gradients(grads_flat, muon_names, max_norm=config.grad_clip_muon)

    state.step += 1
    beta1, beta2 = config.adamw_betas
    updated: dict[str, Any] = {}
    for name, param in params_flat.items():
        grad = grads_flat.get(name)
        if grad is None:
            updated[name] = param
            continue
        if name in muon_names:
            base = (
                param * (1.0 - config.muon_lr * config.muon_weight_decay)
                if config.muon_weight_decay
                else param
            )
            buf = state.muon_buffers.get(name)
            if buf is None:
                buf = mx.zeros_like(grad)  # type: ignore[union-attr]
            buf = buf * config.muon_momentum + grad
            state.muon_buffers[name] = buf
            update = (
                grad + buf * config.muon_momentum
                if config.muon_nesterov
                else buf
            )
            original_shape = update.shape
            if len(update.shape) == 4:
                rows = int(update.shape[0])
                cols = math.prod(int(dim) for dim in update.shape[1:])
                update_2d = mx.reshape(update, (rows, cols))  # type: ignore[union-attr]
                update_2d = zeropower_via_newtonschulz5_mlx(
                    update_2d,
                    steps=config.muon_ns_steps,
                    cast_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
                )
                scale = max(1.0, math.sqrt(rows / cols))
                update = mx.reshape(update_2d * scale, original_shape)  # type: ignore[union-attr]
            elif len(update.shape) == 2:
                rows = int(update.shape[0])
                cols = int(update.shape[1])
                update = zeropower_via_newtonschulz5_mlx(
                    update,
                    steps=config.muon_ns_steps,
                    cast_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
                )
                update = update * max(1.0, math.sqrt(rows / cols))
            updated[name] = base - config.muon_lr * update
            continue

        lr = config.adamw_lr * (
            config.latent_lr_mult
            if parameter_classes.get(name) == "embedding_like"
            else 1.0
        )
        base = (
            param * (1.0 - lr * config.adamw_weight_decay)
            if config.adamw_weight_decay
            else param
        )
        m = state.adamw_m.get(name)
        v = state.adamw_v.get(name)
        if m is None:
            m = mx.zeros_like(grad)  # type: ignore[union-attr]
        if v is None:
            v = mx.zeros_like(grad)  # type: ignore[union-attr]
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        state.adamw_m[name] = m
        state.adamw_v[name] = v
        bias_corrected_lr = lr * math.sqrt(1.0 - beta2**state.step) / (
            1.0 - beta1**state.step
        )
        updated[name] = base - bias_corrected_lr * m / (mx.sqrt(v) + config.adamw_eps)  # type: ignore[union-attr]

    module.update(tree_unflatten(list(updated.items())))  # type: ignore[misc]
    return {
        "schema": "pr95_hnerv_mlx_optimizer_step_summary_v1",
        "step": state.step,
        "use_muon": config.use_muon,
        "muon_tensor_count": len(muon_names),
        "adamw_tensor_count": len(adamw_names),
        "muon_parameter_names": muon_names,
        "adamw_parameter_names": sorted(adamw_names),
        "parameter_group_fingerprint_sha256": parameter_group_fingerprint.get(
            "fingerprint_sha256"
        ),
    }


def run_pr95_mlx_synthetic_timing_smoke(
    *,
    stage_index: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    seed: int,
    base_channels: int = 36,
    latent_dim: int = 28,
    optimizer_descriptor_id: str | None = None,
    pr95_public_archive_export_path: Path | None = None,
    target_pairs_n2chw: Any | None = None,
    target_source: Mapping[str, Any] | None = None,
    training_loss_surface: str = PR95_MLX_LOSS_SURFACE_RGB_MSE,
) -> dict[str, Any]:
    """Run a local MLX timing smoke against synthetic or supplied RGB targets."""

    require_mlx()
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    loss_surface = str(training_loss_surface).strip()
    if loss_surface not in PR95_MLX_LOSS_SURFACES:
        raise ValueError(
            "training_loss_surface must be one of "
            f"{', '.join(PR95_MLX_LOSS_SURFACES)}; got {training_loss_surface!r}"
        )

    source_video_training = target_pairs_n2chw is not None
    target_source_payload = dict(target_source or {})
    target_yuv6 = None
    target_yuv6_shape: list[int] | None = None
    yuv6_preprocess_kind: str | None = None
    rgb_to_yuv6_loss_fn = None
    loss_surface_weights: dict[str, float] = {"rgb_mse": 1.0}
    if source_video_training:
        target_np = np.asarray(target_pairs_n2chw, dtype=np.float32)
        if target_np.ndim != 5 or target_np.shape[1:] != (2, 3, 384, 512):
            raise ValueError(
                "target_pairs_n2chw must have shape (n_pairs, 2, 3, 384, 512); "
                f"got {target_np.shape}"
            )
        training_pair_count = int(target_np.shape[0])
        if training_pair_count < batch_size:
            raise ValueError("source-video target pair count must be >= batch_size")
        target_kind = str(
            target_source_payload.get("kind") or "pr95_source_video_rgb_pairs"
        )
        target = mx.array(target_np)  # type: ignore[union-attr]
        if loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE:
            from tac.local_acceleration.pr95_hnerv_mlx_training import (
                rgb_to_yuv6_mlx as rgb_to_yuv6_loss_fn,
            )

            training_fidelity = (
                PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY
            )
            source_faithfulness_blockers = list(
                PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS
            )
            target_nhwc = mx.transpose(target, (0, 1, 3, 4, 2))  # type: ignore[union-attr]
            target_yuv6 = rgb_to_yuv6_loss_fn(target_nhwc)
            target_yuv6_shape = [int(dim) for dim in target_yuv6.shape]
            yuv6_preprocess_kind = "pr95_mlx_rgb_to_yuv6_scorer_preprocess"
            loss_surface_weights = {"rgb_mse": 0.5, "yuv6_mse": 0.5}
        else:
            training_fidelity = PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_TIMING_ONLY
            source_faithfulness_blockers = list(PR95_MLX_SOURCE_VIDEO_RGB_BLOCKERS)
    else:
        if loss_surface != PR95_MLX_LOSS_SURFACE_RGB_MSE:
            raise ValueError(
                "non-source-video timing smokes only support rgb_mse loss surface"
            )
        if synthetic_pairs < batch_size:
            raise ValueError("synthetic_pairs must be >= batch_size")
        training_pair_count = synthetic_pairs
        target_kind = "synthetic_rgb_pairs"
        training_fidelity = PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY
        source_faithfulness_blockers = list(PR95_MLX_SOURCE_FAITHFUL_BLOCKERS)
        target_key = mx.random.key(seed + 1)  # type: ignore[union-attr]
        target = mx.random.uniform(  # type: ignore[union-attr]
            0,
            255,
            shape=(training_pair_count, 2, 3, 384, 512),
            key=target_key,
        )

    stage = stage_smoke_config(
        stage_index,
        optimizer_descriptor_id=optimizer_descriptor_id,
    )
    mx.random.seed(seed)  # type: ignore[union-attr]
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=training_pair_count,
        latent_dim=latent_dim,
        base_channels=base_channels,
        seed=seed,
        output_layout="n2chw",
    )
    optimizer_state = Pr95MlxOptimizerState()
    parameter_group_fingerprint = build_parameter_group_lr_policy_fingerprint(
        pr95_mlx_parameter_shape_records(bundle.parameters()),
        policy=stage.parameter_group_lr_policy,
    )

    def loss_fn(model: Any, indices: Any) -> Any:
        pred = model(indices)
        selected = mx.take(target, indices, axis=0)  # type: ignore[union-attr]
        residual = (pred - selected) / 255.0
        rgb_loss = mx.mean(residual * residual)  # type: ignore[union-attr]
        if target_yuv6 is None:
            return rgb_loss
        pred_nhwc = mx.transpose(pred, (0, 1, 3, 4, 2))  # type: ignore[union-attr]
        pred_yuv6 = rgb_to_yuv6_loss_fn(pred_nhwc)
        selected_yuv6 = mx.take(target_yuv6, indices, axis=0)  # type: ignore[union-attr]
        yuv6_residual = (pred_yuv6 - selected_yuv6) / 255.0
        yuv6_loss = mx.mean(yuv6_residual * yuv6_residual)  # type: ignore[union-attr]
        return 0.5 * rgb_loss + 0.5 * yuv6_loss

    loss_and_grad = nn.value_and_grad(bundle, loss_fn)  # type: ignore[union-attr]
    last_loss = None
    step_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(steps):
        start = (step * batch_size) % training_pair_count
        raw_indices = [
            (start + offset) % training_pair_count for offset in range(batch_size)
        ]
        indices = mx.array(raw_indices, dtype=mx.uint32)  # type: ignore[union-attr]
        loss, grads = loss_and_grad(bundle, indices)
        step_summary = apply_pr95_mlx_optimizer_step(
            bundle,
            grads,
            optimizer_state,
            stage.optimizer,
            parameter_group_fingerprint=parameter_group_fingerprint,
        )
        mx.eval(loss, bundle.parameters())  # type: ignore[union-attr]
        last_loss = float(loss)
        step_summaries.append(step_summary)
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / steps

    split = partition_pr95_mlx_parameter_names(bundle.parameters())
    descriptor_fragment = _safe_descriptor_fragment(stage.optimizer_descriptor_id)
    profile_id = (
        f"pr95_hnerv_mlx_stage{stage_index}_{descriptor_fragment}"
        f"_seed{seed}_steps{steps}_c{base_channels}"
    )
    if source_video_training and loss_surface == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE:
        profile_id += "_source_video_rgb_yuv6"
    elif source_video_training:
        profile_id += "_source_video_rgb"
    source_video_target_loss_training = bool(source_video_training)
    source_faithful_training = False
    source_faithful_training_scope = (
        "source_video_target_loss_only"
        if source_video_training
        else "synthetic_timing_only"
    )
    runtime_profile = {
        "schema": "trainer_runtime_profile_observation.v1",
        "profile_id": profile_id,
        "candidate_id": profile_id,
        "lane_id": LANE_ID,
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_backend": "mlx",
        "training_fidelity": training_fidelity,
        "source_faithful_training": source_faithful_training,
        "source_faithful_training_scope": source_faithful_training_scope,
        "full_pr95_source_faithful_training": False,
        "source_video_training": source_video_training,
        "source_video_target_loss_training": source_video_target_loss_training,
        "source_faithfulness_blockers": source_faithfulness_blockers,
        "training_loss_surface": loss_surface,
        "loss_surface_weights": loss_surface_weights,
        "target_yuv6_shape": target_yuv6_shape,
        "yuv6_preprocess_kind": yuv6_preprocess_kind,
        "target_source_kind": target_kind,
        "target_source": target_source_payload,
        "source_stage_loss_contract": stage.source_stage_loss_contract,
        "device": "mlx",
        "hardware_substrate": f"{platform.system()}_{platform.machine()}_mlx",
        "seed": seed,
        "stage_id": stage.stage_module,
        "stage_index": stage.stage_index,
        "optimizer_descriptor_id": stage.optimizer_descriptor_id,
        "optimizer_config_sha256": stage.optimizer_config_sha256,
        "parameter_group_lr_policy_id": stage.parameter_group_lr_policy_id,
        "parameter_group_lr_policy_sha256": stage.parameter_group_lr_policy_sha256,
        "parameter_group_fingerprint_sha256": parameter_group_fingerprint[
            "fingerprint_sha256"
        ],
        "seconds_per_step": seconds_per_step,
        "examples_per_second": batch_size / seconds_per_step if seconds_per_step else None,
        "state_bytes": _param_count_from_tree(bundle.parameters()) * 4,
        "kernel_fusion_strategy_id": "native_mlx_pr95_hnerv_decoder_muon_adamw_v1",
        "operator_mix": {
            "conv2d": 15,
            "pixel_shuffle_2x": 6,
            "bilinear_resize2x": 6,
            "linear": 1,
            "newton_schulz5": len(split["muon"]) if stage.optimizer.use_muon else 0,
        },
        "packet_compiler_bridge": {
            "packet_compiler_target_declared": True,
            "archive_export_schema": SMOKE_ARCHIVE_SCHEMA,
            "archive_export_tool": "tools/run_pr95_mlx_timing_smoke.py",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_present": False,
            "blockers": [
                "runtime_consumption_proof_missing",
                "byte_closed_contest_archive_export_missing",
            ],
        },
        "local_cloud_substitution": {
            "intended_to_replace_cloud_gpu_training": True,
            "cloud_gpu_reference": "PR95 HNeRV Muon stage timing and reproduction lane",
        },
        **FALSE_AUTHORITY,
    }
    manifest = {
        "schema": SMOKE_MANIFEST_SCHEMA,
        "candidate_id": profile_id,
        "lane_id": LANE_ID,
        "generated_utc": datetime.now(UTC).isoformat(),
        "stage_index": stage.stage_index,
        "stage_module": stage.stage_module,
        "steps": steps,
        "batch_size": batch_size,
        "synthetic_pairs": synthetic_pairs if not source_video_training else None,
        "training_pair_count": training_pair_count,
        "seed": seed,
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_backend": "mlx",
        "training_fidelity": training_fidelity,
        "source_faithful_training": source_faithful_training,
        "source_faithful_training_scope": source_faithful_training_scope,
        "full_pr95_source_faithful_training": False,
        "source_video_training": source_video_training,
        "source_video_target_loss_training": source_video_target_loss_training,
        "source_faithfulness_blockers": source_faithfulness_blockers,
        "training_loss_surface": loss_surface,
        "loss_surface_weights": loss_surface_weights,
        "target_yuv6_shape": target_yuv6_shape,
        "yuv6_preprocess_kind": yuv6_preprocess_kind,
        "target_source_kind": target_kind,
        "target_source": target_source_payload,
        "source_stage_loss_contract": stage.source_stage_loss_contract,
        "evidence_grade": "[macOS-MLX research-signal]",
        "timing": {
            "elapsed_seconds": elapsed,
            "seconds_per_step": seconds_per_step,
            "examples_per_second": runtime_profile["examples_per_second"],
        },
        "last_loss": last_loss,
        "architecture": bundle.decoder.architecture_manifest(),
        "optimizer_recipe": {
            "schema": "pr95_hnerv_mlx_optimizer_recipe_v1",
            "id": stage.optimizer_descriptor_id,
            "optimizer_descriptor_id": stage.optimizer_descriptor_id,
            "optimizer_config_sha256": stage.optimizer_config_sha256,
            "optimizer_backend_status": stage.optimizer_backend_status,
            "training_fidelity": training_fidelity,
            "source_faithful_training": source_faithful_training,
            "source_faithful_training_scope": source_faithful_training_scope,
            "full_pr95_source_faithful_training": False,
            "source_video_training": source_video_training,
            "source_video_target_loss_training": source_video_target_loss_training,
            "source_faithfulness_blockers": source_faithfulness_blockers,
            "training_loss_surface": loss_surface,
            "loss_surface_weights": loss_surface_weights,
            "target_yuv6_shape": target_yuv6_shape,
            "yuv6_preprocess_kind": yuv6_preprocess_kind,
            "source_stage_loss_contract": stage.source_stage_loss_contract,
            "parameter_group_lr_policy_id": stage.parameter_group_lr_policy_id,
            "parameter_group_lr_policy_sha256": stage.parameter_group_lr_policy_sha256,
            "parameter_group_fingerprint_sha256": parameter_group_fingerprint[
                "fingerprint_sha256"
            ],
            "parameter_group_fingerprint": parameter_group_fingerprint,
            "stage_uses_muon": stage.optimizer.use_muon,
            "muon_lr": stage.optimizer.muon_lr if stage.optimizer.use_muon else None,
            "adamw_lr": stage.optimizer.adamw_lr,
            "latent_lr_mult": stage.optimizer.latent_lr_mult,
            "muon_weight_decay": (
                stage.optimizer.muon_weight_decay if stage.optimizer.use_muon else None
            ),
            "muon_partition": split["muon"],
            "adamw_partition": split["adamw"],
            "source_stage8_partition_contract": (
                "Muon for hidden 2D+ non-stem/non-rgb weights; AdamW for latents, "
                "stem, RGB heads, biases, and 1D/scalar parameters."
            ),
        },
        "runtime_profile": runtime_profile,
        "step_summaries": step_summaries[-3:],
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": _exact_readiness_blockers_for_timing_smoke(
                source_video_training=source_video_training,
                source_faithfulness_blockers=source_faithfulness_blockers,
                training_loss_surface=loss_surface,
            ),
        },
        "pytorch_export_parity": {
            "schema": "pr95_hnerv_pytorch_export_parity_status_v1",
            "state_dict_names_match_public_pr95": True,
            "conv_layout_conversion": "MLX O,H,W,I <-> PyTorch O,I,H,W",
            "forward_parity_on_source_checkpoint": False,
            "blocker": "requires source checkpoint load and paired PyTorch/MLX forward smoke",
        },
        **FALSE_AUTHORITY,
    }
    if pr95_public_archive_export_path is not None:
        public_archive_export = write_pr95_public_archive_zip(
            pytorch_state_dict_from_mlx(bundle.decoder),
            np.asarray(bundle.latents).astype(np.float32, copy=False),
            meta={
                "n_pairs": training_pair_count,
                "latent_dim": latent_dim,
                "base_channels": base_channels,
                "eval_size": [384, 512],
                "training_fidelity": training_fidelity,
                "training_loss_surface": loss_surface,
                "target_source_kind": target_kind,
            },
            output_zip_path=Path(pr95_public_archive_export_path),
        )
        runtime_profile["packet_compiler_bridge"].update(
            {
                "archive_export_schema": PR95_ARCHIVE_EXPORT_SCHEMA,
                "byte_closed_contest_archive_export_present": True,
                "byte_closed_contest_archive_export_path": public_archive_export[
                    "archive_zip_path"
                ],
                "byte_closed_contest_archive_export_sha256": public_archive_export[
                    "archive_zip_sha256"
                ],
            }
        )
        runtime_profile["packet_compiler_bridge"]["blockers"] = [
            blocker
            for blocker in runtime_profile["packet_compiler_bridge"].get("blockers", [])
            if blocker != "byte_closed_contest_archive_export_missing"
        ]
        manifest["pr95_public_archive_export"] = public_archive_export
    return manifest


def write_pr95_mlx_byte_closed_smoke_archive(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a deterministic byte-closed smoke archive for queue plumbing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SMOKE_ARCHIVE_SCHEMA,
        "candidate_id": manifest["candidate_id"],
        "lane_id": manifest["lane_id"],
        "generated_utc": manifest["generated_utc"],
        "stage_index": manifest["stage_index"],
        "stage_module": manifest["stage_module"],
        "runtime_profile": manifest["runtime_profile"],
        "architecture": manifest["architecture"],
        "exact_readiness_refusal": manifest["exact_readiness_refusal"],
        **FALSE_AUTHORITY,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload_path = output_dir / "0.bin"
    payload_path.write_bytes(payload_bytes)
    archive_path = output_dir / "archive.zip"
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.extra = b""
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.comment = b""
        zf.writestr(info, payload_bytes)
    return {
        "schema": "byte_closed_archive_smoke_summary.v1",
        "path": archive_path.as_posix(),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member": "0.bin",
        "member_bytes": len(payload_bytes),
        "member_sha256": _sha256_bytes(payload_bytes),
        "compression_method": "stored",
        "runtime_consumption_proof_present": False,
        "receiver_proof_present": False,
        **FALSE_AUTHORITY,
    }


__all__ = [
    "EXACT_READINESS_REFUSAL_BLOCKERS",
    "FALSE_AUTHORITY",
    "LANE_ID",
    "PR95_ARCHIVE_EXPORT_SCHEMA",
    "PR95_ARCHIVE_N_QUANT",
    "PR95_MLX_BACKEND_STATUS_LOCAL_TIMING_PROXY",
    "PR95_MLX_CONV2D_ACCUMULATION_MODES",
    "PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS",
    "PR95_MLX_LOSS_SURFACES",
    "PR95_MLX_LOSS_SURFACE_RGB_MSE",
    "PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE",
    "PR95_MLX_PYTORCH_EXPORT_FORWARD_PARITY_SCHEMA",
    "PR95_MLX_SOURCE_FAITHFUL_BLOCKERS",
    "PR95_MLX_SOURCE_VIDEO_RGB_BLOCKERS",
    "PR95_MLX_SOURCE_VIDEO_RGB_YUV6_BLOCKERS",
    "PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_TIMING_ONLY",
    "PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY",
    "PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY",
    "PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS",
    "PR95_STAGE_MODULES",
    "PUBLIC_ARCHIVE_DECODER_TRACE_SCHEMA",
    "PUBLIC_ARCHIVE_FORWARD_PARITY_SCHEMA",
    "PUBLIC_ARCHIVE_PACKET_SCHEMA",
    "SMOKE_ARCHIVE_SCHEMA",
    "SMOKE_MANIFEST_SCHEMA",
    "HNeRVDecoderMLX",
    "HNeRVSyntheticTrainingBundleMLX",
    "Pr95HNeRVMlxError",
    "Pr95MlxOptimizerConfig",
    "Pr95MlxOptimizerState",
    "Pr95PublicArchivePacket",
    "apply_pr95_mlx_optimizer_step",
    "bilinear_resize2x_align_corners_false_nhwc",
    "bilinear_resize_nhwc",
    "build_pr95_public_archive_member",
    "compare_pr95_public_archive_forward_with_pytorch",
    "load_pytorch_state_dict_into_mlx",
    "parse_pr95_public_archive_member",
    "parse_pr95_public_archive_zip",
    "partition_pr95_mlx_parameter_names",
    "pixel_shuffle_2x_nhwc",
    "pr95_default_optimizer_descriptor_id",
    "pr95_mlx_conv2d_accumulation_overrides_from_items",
    "pr95_mlx_conv2d_accumulation_overrides_from_preset",
    "pr95_mlx_optimizer_config_from_descriptor",
    "pr95_mlx_optimizer_descriptor_row",
    "pr95_mlx_parameter_shape_records",
    "pytorch_state_dict_from_mlx",
    "require_mlx",
    "run_pr95_mlx_synthetic_timing_smoke",
    "stage_smoke_config",
    "trace_pr95_public_archive_decoder_with_pytorch",
    "validate_pr95_mlx_conv2d_accumulation_mode",
    "validate_pr95_mlx_conv2d_accumulation_overrides",
    "write_pr95_mlx_byte_closed_smoke_archive",
    "write_pr95_public_archive_pytorch_export_forward_parity",
    "write_pr95_public_archive_zip",
    "zeropower_via_newtonschulz5_mlx",
]
