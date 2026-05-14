"""SAR coherent pose-pair substrate archive grammar — SARC monolithic 0.bin.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (in single-zip-member ``0.bin`` slot
  when packaged in archive.zip)
* L4 inflate.py ≤ 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, fp16 state_dict, fixed brotli quality)

The 4-stage grammar:

::

    MAGIC(4)                  b"SARC"
    VERSION(1)                u8 (== 1)
    NUM_PAIRS(2)              u16
    HIDDEN_DIM(2)             u16
    NUM_HIDDEN_LAYERS(1)      u8
    OUTPUT_HEIGHT(2)          u16
    OUTPUT_WIDTH(2)           u16
    POSE_DIM(1)               u8
    POSE_CODE_DIM(1)          u8
    PER_PAIR_RESIDUAL_BYTES(1) u8
    SAR_TOPK(2)               u16
    RENDERER_LEN(4)           u32   brotli(state_dict, fp16)
    POSE_CODEC_LEN(4)         u32   brotli(int16 sparse rFFT bytes)
    PER_PAIR_RESIDUAL_LEN(4)  u32   brotli(int8 RGB residual bytes)
    META_LEN(4)               u32   utf-8 JSON of float meta + scales
    RENDERER_BLOB             ...
    POSE_CODEC_BLOB           ...
    PER_PAIR_RESIDUAL_BLOB    ...
    META_BLOB                 ...

Sections:

* **RENDERER_BLOB** (Stage 1): brotli-compressed pickle of the trained
  ``SARCoherentRenderer`` (FP16 state_dict).
* **POSE_CODEC_BLOB** (Stage 2): brotli-compressed sparse rFFT representation
  of the per-pair pose deltas. Each retained coefficient is encoded as
  ``[u16 bin_index][i16 real][i16 imag]`` per pose dim. Total: K * pose_dim
  positions × 6 B/position.
* **PER_PAIR_RESIDUAL_BLOB** (Stage 3): brotli-compressed int8 sequence;
  ``num_pairs * per_pair_residual_bytes`` raw int8s before brotli.
* **META_BLOB** (Stage 4): JSON ``{"int8_scale", "sar_int16_scale",
  "first_omega", "hidden_omega", "coord_feature_freqs", ...}``.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, brotli quality fixed)
- No /tmp paths
- No scorer load
- No score claim
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

SARC_MAGIC: bytes = b"SARC"
"""SAR Coherent archive magic."""

SARC_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout (per docstring):
# MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + HIDDEN_DIM(2) + NUM_HIDDEN_LAYERS(1)
# + OUTPUT_H(2) + OUTPUT_W(2) + POSE_DIM(1) + POSE_CODE_DIM(1)
# + PER_PAIR_RESIDUAL_BYTES(1) + SAR_TOPK(2)
# + RENDERER_LEN(4) + POSE_CODEC_LEN(4) + RESIDUAL_LEN(4) + META_LEN(4)
# = 4+1+2+2+1+2+2+1+1+1+2+4+4+4+4 = 35 bytes
SARC_HEADER_FMT: str = "<4sBHHBHHBBBHIIII"
SARC_HEADER_SIZE: int = struct.calcsize(SARC_HEADER_FMT)
assert SARC_HEADER_SIZE == 35, (
    f"SARC header size invariant: expected 35, got {SARC_HEADER_SIZE}"
)

# Brotli quality (deterministic at 9 — matches sister substrates).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class SARCoherentArchive:
    """Parsed SARC archive — the inflate-time data contract."""

    renderer_state_dict: dict[str, torch.Tensor]
    """Trained renderer parameters (FP16 state_dict)."""

    pose_codec_bytes: bytes
    """Raw int16-quantized sparse rFFT bytes (K * pose_dim * 6 B)."""

    per_pair_residual: np.ndarray
    """Int8 array shape ``(num_pairs, per_pair_residual_bytes)``."""

    meta: dict[str, object]
    """Sidecar JSON meta with hparams + scales."""

    schema_version: int
    num_pairs: int
    hidden_dim: int
    num_hidden_layers: int
    output_height: int
    output_width: int
    pose_dim: int
    pose_code_dim: int
    per_pair_residual_bytes: int
    sar_topk: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict to deterministic fp16 tensor bytes.

    Python pickle includes storage memoization details that can drift between
    otherwise identical tensor dictionaries. Archive bytes are contest evidence,
    so the wire format is explicit: sorted key order, fp16 CPU tensors, shape
    metadata, and raw C-order tensor bytes under a small ``SDT1`` header.
    """
    out = bytearray(b"SDT1")
    out.extend(struct.pack("<I", len(sd)))
    for key in sorted(sd):
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"state_dict key too long: {key!r}")
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        shape = tuple(int(v) for v in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"state_dict tensor rank too high for {key!r}: {len(shape)}")
        data = tensor.numpy().tobytes(order="C")
        out.extend(struct.pack("<H", len(key_bytes)))
        out.extend(key_bytes)
        out.extend(struct.pack("<B", len(shape)))
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"state_dict tensor dim out of range for {key!r}: {dim}")
            out.extend(struct.pack("<I", dim))
        out.extend(struct.pack("<I", len(data)))
        out.extend(data)
    return bytes(brotli.compress(bytes(out), quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    if len(raw) < 8 or raw[:4] != b"SDT1":
        raise ValueError("state_dict blob missing SDT1 deterministic header")
    pos = 4
    (count,) = struct.unpack("<I", raw[pos : pos + 4])
    pos += 4
    sd: dict[str, torch.Tensor] = {}
    for _ in range(count):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated before key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        key_end = pos + key_len
        if key_end > len(raw):
            raise ValueError("state_dict blob truncated in key")
        key = raw[pos:key_end].decode("utf-8")
        pos = key_end
        if key in sd:
            raise ValueError(f"duplicate state_dict key in archive: {key!r}")
        if pos + 1 > len(raw):
            raise ValueError("state_dict blob truncated before rank")
        rank = raw[pos]
        pos += 1
        shape: list[int] = []
        for _dim in range(rank):
            if pos + 4 > len(raw):
                raise ValueError("state_dict blob truncated in shape")
            (dim,) = struct.unpack("<I", raw[pos : pos + 4])
            pos += 4
            shape.append(int(dim))
        if pos + 4 > len(raw):
            raise ValueError("state_dict blob truncated before tensor bytes")
        (data_len,) = struct.unpack("<I", raw[pos : pos + 4])
        pos += 4
        data_end = pos + data_len
        if data_end > len(raw):
            raise ValueError("state_dict blob truncated in tensor bytes")
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float16).itemsize
        if data_len != expected:
            raise ValueError(
                f"state_dict tensor byte count mismatch for {key!r}: "
                f"got {data_len}, expected {expected}"
            )
        arr = np.frombuffer(raw[pos:data_end], dtype=np.float16).reshape(tuple(shape)).copy()
        sd[key] = torch.from_numpy(arr)
        pos = data_end
    if pos != len(raw):
        raise ValueError(f"state_dict blob has {len(raw) - pos} trailing bytes")
    return sd


def _serialize_per_pair_residual(side_info: np.ndarray) -> bytes:
    """Brotli-compress a contiguous int8 array."""
    if side_info.dtype != np.int8:
        raise ValueError(
            f"per_pair_residual dtype must be int8; got {side_info.dtype}"
        )
    arr = np.ascontiguousarray(side_info)
    return bytes(brotli.compress(arr.tobytes(), quality=_BROTLI_QUALITY))


def _deserialize_per_pair_residual(
    blob: bytes, *, num_pairs: int, per_pair_residual_bytes: int
) -> np.ndarray:
    raw = brotli.decompress(blob)
    expected = num_pairs * per_pair_residual_bytes
    if len(raw) != expected:
        raise ValueError(
            f"per_pair_residual byte count mismatch: got {len(raw)} "
            f"expected {expected} (num_pairs={num_pairs}, "
            f"per_pair_residual_bytes={per_pair_residual_bytes})"
        )
    return (
        np.frombuffer(raw, dtype=np.int8)
        .reshape(num_pairs, per_pair_residual_bytes)
        .copy()
    )


def encode_pose_codec_bytes(
    sparse_coeffs: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    int16_scale: float,
) -> bytes:
    """Encode the sparse rFFT pose-codec into deterministic int16 bytes.

    Layout per pose dim: K positions × ``[u16 bin_index][i16 real][i16 imag]``
    = 6 B/position.

    Args:
        sparse_coeffs: complex tensor (n_rfft_bins, pose_dim) with zeros
            outside the retained top-K positions.
        topk_indices: int64 tensor (K, pose_dim) of retained bin indices.
        int16_scale: ``int16 = round(real * scale).clamp(-32768, 32767)``.

    Returns:
        Raw bytes (uncompressed); brotli applied by ``pack_archive``.
    """
    K, pose_dim = topk_indices.shape
    sparse_cpu = sparse_coeffs.detach().cpu()
    indices_cpu = topk_indices.detach().cpu()
    out = bytearray()
    for d in range(pose_dim):
        for k_idx in range(K):
            bin_idx = int(indices_cpu[k_idx, d].item())
            coeff = sparse_cpu[bin_idx, d]
            real_q = int(round(float(coeff.real) * int16_scale))
            imag_q = int(round(float(coeff.imag) * int16_scale))
            real_q = max(-32768, min(32767, real_q))
            imag_q = max(-32768, min(32767, imag_q))
            if bin_idx < 0 or bin_idx > 0xFFFF:
                raise ValueError(
                    f"bin_idx {bin_idx} out of u16 range; archive corrupt"
                )
            out.extend(struct.pack("<Hhh", bin_idx, real_q, imag_q))
    return bytes(out)


def decode_pose_codec_bytes(
    blob: bytes,
    *,
    n_rfft_bins: int,
    pose_dim: int,
    int16_scale: float,
) -> torch.Tensor:
    """Inverse of ``encode_pose_codec_bytes``.

    Returns:
        complex64 tensor (n_rfft_bins, pose_dim) with retained coefficients
        scattered into a sparse zero tensor.
    """
    if len(blob) % 6 != 0:
        raise ValueError(
            f"pose codec blob length {len(blob)} not divisible by 6 (per-position size)"
        )
    K = len(blob) // 6 // pose_dim
    if K * pose_dim * 6 != len(blob):
        raise ValueError(
            f"pose codec blob length {len(blob)} != K({K}) * pose_dim({pose_dim}) * 6"
        )
    out = torch.zeros((n_rfft_bins, pose_dim), dtype=torch.complex64)
    pos = 0
    for d in range(pose_dim):
        for _k in range(K):
            bin_idx, real_q, imag_q = struct.unpack("<Hhh", blob[pos : pos + 6])
            pos += 6
            if bin_idx >= n_rfft_bins:
                raise ValueError(
                    f"bin_idx {bin_idx} >= n_rfft_bins {n_rfft_bins}; archive corrupt"
                )
            real = float(real_q) / int16_scale
            imag = float(imag_q) / int16_scale
            out[bin_idx, d] = complex(real, imag)
    return out


def pack_archive(
    *,
    renderer_state_dict: dict[str, torch.Tensor],
    pose_codec_bytes: bytes,
    per_pair_residual: np.ndarray,
    meta: dict[str, object],
    num_pairs: int,
    hidden_dim: int,
    num_hidden_layers: int,
    output_height: int,
    output_width: int,
    pose_dim: int,
    pose_code_dim: int,
    per_pair_residual_bytes: int,
    sar_topk: int,
    schema_version: int = SARC_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained renderer + pose codec + per-pair residual into SARC bytes."""
    if schema_version != SARC_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("hidden_dim", hidden_dim, 0xFFFF),
        ("num_hidden_layers", num_hidden_layers, 0xFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
        ("pose_dim", pose_dim, 0xFF),
        ("pose_code_dim", pose_code_dim, 0xFF),
        ("per_pair_residual_bytes", per_pair_residual_bytes, 0xFF),
        ("sar_topk", sar_topk, 0xFFFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of range [1, {max_v}]")

    if per_pair_residual.shape != (num_pairs, per_pair_residual_bytes):
        raise ValueError(
            f"per_pair_residual shape {per_pair_residual.shape} != "
            f"(num_pairs={num_pairs}, per_pair_residual_bytes={per_pair_residual_bytes})"
        )

    renderer_blob = _serialize_state_dict(renderer_state_dict)
    pose_blob = brotli.compress(pose_codec_bytes, quality=_BROTLI_QUALITY) if pose_codec_bytes else b""
    residual_blob = _serialize_per_pair_residual(per_pair_residual)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        SARC_HEADER_FMT,
        SARC_MAGIC,
        schema_version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        pose_dim,
        pose_code_dim,
        per_pair_residual_bytes,
        sar_topk,
        len(renderer_blob),
        len(pose_blob),
        len(residual_blob),
        len(meta_bytes),
    )
    return header + renderer_blob + pose_blob + residual_blob + meta_bytes


def parse_archive(blob: bytes) -> SARCoherentArchive:
    """Parse SARC 0.bin bytes back into a typed ``SARCoherentArchive``."""
    if len(blob) < SARC_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {SARC_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        pose_dim,
        pose_code_dim,
        per_pair_residual_bytes,
        sar_topk,
        renderer_len,
        pose_len,
        residual_len,
        meta_len,
    ) = struct.unpack(SARC_HEADER_FMT, blob[:SARC_HEADER_SIZE])

    if magic != SARC_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {SARC_MAGIC!r})")
    if version != SARC_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    pos = SARC_HEADER_SIZE
    renderer_blob = blob[pos : pos + renderer_len]
    pos += renderer_len
    pose_blob = blob[pos : pos + pose_len]
    pos += pose_len
    residual_blob = blob[pos : pos + residual_len]
    pos += residual_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    sd = _deserialize_state_dict(renderer_blob)
    pose_codec_bytes = brotli.decompress(pose_blob) if pose_blob else b""
    residual = _deserialize_per_pair_residual(
        residual_blob,
        num_pairs=int(num_pairs),
        per_pair_residual_bytes=int(per_pair_residual_bytes),
    )
    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}

    return SARCoherentArchive(
        renderer_state_dict=sd,
        pose_codec_bytes=pose_codec_bytes,
        per_pair_residual=residual,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_hidden_layers),
        output_height=int(output_height),
        output_width=int(output_width),
        pose_dim=int(pose_dim),
        pose_code_dim=int(pose_code_dim),
        per_pair_residual_bytes=int(per_pair_residual_bytes),
        sar_topk=int(sar_topk),
    )


def quantize_per_pair_residual_int8(
    residual: torch.Tensor, *, scale: float
) -> np.ndarray:
    """Quantize a per-pair residual tensor to int8."""
    if residual.dim() != 2:
        raise ValueError(
            f"residual must be 2D (num_pairs, per_pair_residual_bytes); got "
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
    "SARC_HEADER_FMT",
    "SARC_HEADER_SIZE",
    "SARC_MAGIC",
    "SARC_SCHEMA_VERSION",
    "SARCoherentArchive",
    "decode_pose_codec_bytes",
    "dequantize_per_pair_residual",
    "encode_pose_codec_bytes",
    "pack_archive",
    "parse_archive",
    "quantize_per_pair_residual_int8",
]
