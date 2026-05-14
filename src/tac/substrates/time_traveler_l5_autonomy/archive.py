# SPDX-License-Identifier: MIT
"""Time-Traveler L5 Autonomy archive grammar — TT5L monolithic 0.bin.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (in single-zip-member ``x`` slot when
  packaged in archive.zip)
* L4 inflate.py ≤ 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, fp16 state_dict, fixed brotli quality)

The 4-stage grammar (from the design memo):

::

    MAGIC(4)                  b"TT5L"
    VERSION(1)                u8 (== 1)
    NUM_PAIRS(2)              u16
    HIDDEN_DIM(2)             u16
    NUM_HIDDEN_LAYERS(1)      u8
    OUTPUT_HEIGHT(2)          u16
    OUTPUT_WIDTH(2)           u16
    FOVEATION_GRID_H(1)       u8
    FOVEATION_GRID_W(1)       u8
    POSE_DIM(1)               u8
    PER_PAIR_BYTES(1)         u8
    WORLD_MODEL_LEN(4)        u32   brotli(state_dict, fp16)
    PER_PAIR_SIDE_INFO_LEN(4) u32   brotli(int8-quantized side info)
    AC_STATE_LEN(4)           u32   brotli(arithmetic-coder probability tables)
    META_LEN(4)               u32   utf-8 JSON of float meta + per-pair scales
    WORLD_MODEL_BLOB          ...
    PER_PAIR_SIDE_INFO_BLOB   ...
    AC_STATE_BLOB             ...
    META_BLOB                 ...

Sections:

* **WORLD_MODEL_BLOB** (Stage 1): brotli-compressed pickle of the trained
  renderer + foveation grid + dynamics matrices + per-pair pose codes (FP16
  state_dict).
* **PER_PAIR_SIDE_INFO_BLOB** (Stage 2): brotli-compressed int8 sequence;
  ``num_pairs * per_pair_bytes`` raw int8s before brotli. Layout per pair:
  ``[se3_lie(12 B) | seg_boundary(18 B) | hf_residual(6 B) | predict_residual(9 B)]``.
* **AC_STATE_BLOB** (Stage 3): brotli-compressed arithmetic-coder state for
  the per-pair side info. Empty in v1 (placeholder).
* **META_BLOB** (Stage 4): JSON ``{"int8_scale", "first_omega", "hidden_omega",
  "coord_feature_freqs", ...}`` — the floats that don't fit in u8/u16 header
  fields.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, brotli quality fixed)
- No /tmp paths
- No scorer load
- No score claim
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

TT5L_MAGIC: bytes = b"TT5L"
"""Time-Traveler L5 archive magic."""

TT5L_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + HIDDEN_DIM(2)
#              + NUM_HIDDEN_LAYERS(1) + OUTPUT_H(2) + OUTPUT_W(2)
#              + FOV_H(1) + FOV_W(1) + POSE_DIM(1) + PER_PAIR_BYTES(1)
#              + WORLD_LEN(4) + SIDEINFO_LEN(4) + AC_LEN(4) + META_LEN(4)
# = 4+1+2+2+1+2+2+1+1+1+1+4+4+4+4 = 34 bytes
TT5L_HEADER_FMT: str = "<4sBHHBHHBBBBIIII"
TT5L_HEADER_SIZE: int = struct.calcsize(TT5L_HEADER_FMT)
assert TT5L_HEADER_SIZE == 34, (
    f"TT5L header size invariant: expected 34, got {TT5L_HEADER_SIZE}"
)

# Brotli quality (deterministic at 9 — matches SIREN).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class TimeTravelerArchive:
    """Parsed TT5L archive — the inflate-time data contract."""

    world_model_state_dict: dict[str, torch.Tensor]
    """Trained world-model parameters (renderer + foveation + dynamics + pose codes)."""

    per_pair_side_info: np.ndarray
    """Int8 array shape ``(num_pairs, per_pair_bytes)`` — Stage 2 side info."""

    ac_state: bytes
    """Raw arithmetic-coder state (placeholder; empty in v1)."""

    meta: dict[str, object]
    """Sidecar JSON meta with hparams (omega, freqs, int8 scales, ...)."""

    schema_version: int
    num_pairs: int
    hidden_dim: int
    num_hidden_layers: int
    output_height: int
    output_width: int
    foveation_grid_h: int
    foveation_grid_w: int
    pose_dim: int
    per_pair_bytes: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu, contiguous)."""
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _serialize_per_pair_side_info(side_info: np.ndarray) -> bytes:
    """Brotli-compress a contiguous int8 array."""
    if side_info.dtype != np.int8:
        raise ValueError(
            f"per_pair_side_info dtype must be int8; got {side_info.dtype}"
        )
    arr = np.ascontiguousarray(side_info)
    return bytes(brotli.compress(arr.tobytes(), quality=_BROTLI_QUALITY))


def _deserialize_per_pair_side_info(
    blob: bytes, *, num_pairs: int, per_pair_bytes: int
) -> np.ndarray:
    raw = brotli.decompress(blob)
    expected = num_pairs * per_pair_bytes
    if len(raw) != expected:
        raise ValueError(
            f"per_pair_side_info byte count mismatch: got {len(raw)} "
            f"expected {expected} (num_pairs={num_pairs}, "
            f"per_pair_bytes={per_pair_bytes})"
        )
    return np.frombuffer(raw, dtype=np.int8).reshape(num_pairs, per_pair_bytes).copy()


def pack_archive(
    *,
    world_model_state_dict: dict[str, torch.Tensor],
    per_pair_side_info: np.ndarray,
    meta: dict[str, object],
    num_pairs: int,
    hidden_dim: int,
    num_hidden_layers: int,
    output_height: int,
    output_width: int,
    foveation_grid_h: int,
    foveation_grid_w: int,
    pose_dim: int,
    per_pair_bytes: int,
    ac_state: bytes = b"",
    schema_version: int = TT5L_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained world model + per-pair side info into TT5L 0.bin bytes.

    Returns the monolithic single-file archive bytes (suitable as
    archive.zip member ``0.bin`` or single-zip-member ``x``).
    """
    if schema_version != TT5L_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("hidden_dim", hidden_dim, 0xFFFF),
        ("num_hidden_layers", num_hidden_layers, 0xFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
        ("foveation_grid_h", foveation_grid_h, 0xFF),
        ("foveation_grid_w", foveation_grid_w, 0xFF),
        ("pose_dim", pose_dim, 0xFF),
        ("per_pair_bytes", per_pair_bytes, 0xFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of range [1, {max_v}]")

    if per_pair_side_info.shape != (num_pairs, per_pair_bytes):
        raise ValueError(
            f"per_pair_side_info shape {per_pair_side_info.shape} != "
            f"(num_pairs={num_pairs}, per_pair_bytes={per_pair_bytes})"
        )

    world_blob = _serialize_state_dict(world_model_state_dict)
    side_blob = _serialize_per_pair_side_info(per_pair_side_info)
    ac_blob = brotli.compress(ac_state, quality=_BROTLI_QUALITY) if ac_state else b""
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        TT5L_HEADER_FMT,
        TT5L_MAGIC,
        schema_version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        foveation_grid_h,
        foveation_grid_w,
        pose_dim,
        per_pair_bytes,
        len(world_blob),
        len(side_blob),
        len(ac_blob),
        len(meta_bytes),
    )
    return header + world_blob + side_blob + ac_blob + meta_bytes


def parse_archive(blob: bytes) -> TimeTravelerArchive:
    """Parse TT5L 0.bin bytes back into a typed ``TimeTravelerArchive``."""
    if len(blob) < TT5L_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {TT5L_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        foveation_grid_h,
        foveation_grid_w,
        pose_dim,
        per_pair_bytes,
        world_len,
        side_len,
        ac_len,
        meta_len,
    ) = struct.unpack(TT5L_HEADER_FMT, blob[:TT5L_HEADER_SIZE])

    if magic != TT5L_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {TT5L_MAGIC!r})")
    if version != TT5L_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    pos = TT5L_HEADER_SIZE
    world_blob = blob[pos : pos + world_len]
    pos += world_len
    side_blob = blob[pos : pos + side_len]
    pos += side_len
    ac_blob = blob[pos : pos + ac_len]
    pos += ac_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    sd = _deserialize_state_dict(world_blob)
    side_info = _deserialize_per_pair_side_info(
        side_blob, num_pairs=int(num_pairs), per_pair_bytes=int(per_pair_bytes)
    )
    ac_state = brotli.decompress(ac_blob) if ac_blob else b""
    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}

    return TimeTravelerArchive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        ac_state=ac_state,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_hidden_layers),
        output_height=int(output_height),
        output_width=int(output_width),
        foveation_grid_h=int(foveation_grid_h),
        foveation_grid_w=int(foveation_grid_w),
        pose_dim=int(pose_dim),
        per_pair_bytes=int(per_pair_bytes),
    )


def quantize_per_pair_residual_int8(
    residual: torch.Tensor, *, scale: float
) -> np.ndarray:
    """Quantize a per-pair residual tensor to int8.

    Args:
        residual: float tensor shape ``(num_pairs, per_pair_bytes)``.
        scale: ``int8 = round(real * scale).clamp(-128, 127)``.

    Returns:
        ``(num_pairs, per_pair_bytes)`` int8 numpy array.
    """
    if residual.dim() != 2:
        raise ValueError(
            f"residual must be 2D (num_pairs, per_pair_bytes); got "
            f"{tuple(residual.shape)}"
        )
    q = (
        (residual.detach().cpu().float() * float(scale))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    return q.numpy().astype(np.int8, copy=False)


def dequantize_per_pair_residual(
    side_info: np.ndarray, *, scale: float
) -> torch.Tensor:
    """Inverse of ``quantize_per_pair_residual_int8``. Returns float32 tensor."""
    arr = side_info.astype(np.float32) / float(scale)
    return torch.from_numpy(arr.copy())


__all__ = [
    "TT5L_HEADER_FMT",
    "TT5L_HEADER_SIZE",
    "TT5L_MAGIC",
    "TT5L_SCHEMA_VERSION",
    "TimeTravelerArchive",
    "dequantize_per_pair_residual",
    "pack_archive",
    "parse_archive",
    "quantize_per_pair_residual_int8",
]
