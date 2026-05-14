"""C1 archive grammar -- C1WMFV1 monolithic 0.bin (substrate-engineering scope).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (in single-zip-member ``0.bin`` slot
  when packaged in archive.zip)
* L4 inflate.py <= 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, fp16 packed weights, fixed brotli quality)

The 6-section grammar (single monolithic 0.bin):

::

    MAGIC(4)              b"WMF\\x01"
    VERSION(1)            u8 (== 1)
    NUM_PAIRS(2)          u16
    RECURRENCE_MODE(1)    u8 (0=GRU, 1=LSTM, 2=TRANSFORMER)
    FOVEATION_STRATEGY(1) u8 (0=UNIFORM, 1=EGO_MOTION_RADIAL, 2=LEARNED_PER_PIXEL)
    LATENT_DIM(2)         u16
    OUTPUT_H(2)           u16
    OUTPUT_W(2)           u16
    WM_LEN(4)             u32 brotli(world_model state_dict fp16)
    DECODER_LEN(4)        u32 brotli(decoder state_dict fp16)
    ZINIT_LEN(4)          u32 brotli(z_init fp16)
    FOV_META_LEN(4)       u32 brotli(foveation meta fp16 OR JSON)
    RESIDUAL_LEN(4)       u32 brotli(per-frame residual int8)
    META_LEN(4)           u32 sorted-keys JSON
    WM_BLOB(WM_LEN)
    DECODER_BLOB(DECODER_LEN)
    ZINIT_BLOB(ZINIT_LEN)
    FOV_META_BLOB(FOV_META_LEN)
    RESIDUAL_BLOB(RESIDUAL_LEN)
    META_BLOB(META_LEN)

Sections:

* **WM_BLOB**: brotli-compressed world-model state_dict as fp16 byte stream
  (GRU/LSTM/Transformer weights ~10-25 KB raw, ~5-15 KB after brotli).

* **DECODER_BLOB**: brotli-compressed decoder state_dict as fp16 byte stream
  (linear+convs ~25-60 KB raw, ~15-40 KB after brotli + FP4 future).

* **ZINIT_BLOB**: brotli-compressed initial latent z_init (latent_dim floats,
  ~256 B at latent_dim=64 fp16).

* **FOV_META_BLOB**: foveation meta. For UNIFORM/EGO_MOTION_RADIAL this is
  JSON (~100 B with center coords + sigma). For LEARNED_PER_PIXEL this is
  the learned_head state_dict as fp16 (~200 B).

* **RESIDUAL_BLOB**: per-frame residual surprise blob (int8 quantized,
  brotli-compressed). ~50-100 B/frame * 1200 = 60-120 KB.

* **META_BLOB**: JSON ``{"trained_at_utc", "git_head", "lane_id",
  "recurrence_mode_label", "foveation_strategy_label",
  "design_notes", ...}``.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 weight blob, brotli quality fixed)
- No /tmp paths
- No scorer load
- No score claim
- Single-zip-member archive
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import torch

C1WMFV1_MAGIC: bytes = b"WMF\x01"
"""C1 archive magic (4 bytes; distinct from WZF/D1P/TT5L/etc.)."""

C1WMFV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
# MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + RECURRENCE_MODE(1) +
# FOVEATION_STRATEGY(1) + LATENT_DIM(2) + OUTPUT_H(2) + OUTPUT_W(2) +
# WM_LEN(4) + DECODER_LEN(4) + ZINIT_LEN(4) + FOV_META_LEN(4) +
# RESIDUAL_LEN(4) + META_LEN(4) = 4+1+2+1+1+2+2+2+4+4+4+4+4+4 = 39 bytes
C1WMFV1_HEADER_FMT: str = "<4sBHBBHHHIIIIII"
C1WMFV1_HEADER_SIZE: int = struct.calcsize(C1WMFV1_HEADER_FMT)
assert C1WMFV1_HEADER_SIZE == 39, (
    f"C1WMFV1 header size invariant: expected 39, got {C1WMFV1_HEADER_SIZE}"
)

# Deterministic brotli quality (matches WZF / D1 / TT5L siblings).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class WorldModelFoveationArchive:
    """Parsed C1WMFV1 archive sections (immutable forensic record)."""

    version: int
    num_pairs: int
    recurrence_mode: int  # 0=GRU 1=LSTM 2=TRANSFORMER
    foveation_strategy: int  # 0=UNIFORM 1=EGO_MOTION_RADIAL 2=LEARNED_PER_PIXEL
    latent_dim: int
    output_h: int
    output_w: int
    world_model_blob: bytes  # brotli-compressed
    decoder_blob: bytes  # brotli-compressed
    z_init_blob: bytes  # brotli-compressed
    foveation_meta_blob: bytes  # brotli-compressed (JSON or fp16 state_dict)
    residual_blob: bytes  # brotli-compressed int8 residual
    meta: dict[str, Any]  # parsed META JSON


def _tensor_dict_to_fp16_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict to deterministic fp16 byte stream.

    Format: for each key (sorted), pack:
        keylen(u16) + key(bytes) + ndim(u8) + dims(u32*ndim) + data(fp16)

    Deterministic across runs (sorted keys; little-endian byte order).
    """
    parts: list[bytes] = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().to(torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"state_dict key too long for u16 len: {key!r}")
        ndim = tensor.dim()
        if ndim > 0xFF:
            raise ValueError(f"state_dict tensor ndim too large: {ndim}")
        header = struct.pack(
            f"<H{len(key_bytes)}sB" + "I" * ndim,
            len(key_bytes),
            key_bytes,
            ndim,
            *tensor.shape,
        )
        parts.append(header + tensor.numpy().tobytes())
    return b"".join(parts)


def _fp16_bytes_to_tensor_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Deserialize a fp16 byte stream back to state_dict.

    Inverse of ``_tensor_dict_to_fp16_bytes``. Raises ``ValueError`` on
    truncated or malformed input.
    """
    import numpy as np

    out: dict[str, torch.Tensor] = {}
    pos = 0
    while pos < len(blob):
        if pos + 2 > len(blob):
            raise ValueError("truncated state_dict: missing keylen")
        (keylen,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        if pos + keylen > len(blob):
            raise ValueError("truncated state_dict: missing key")
        key = blob[pos : pos + keylen].decode("utf-8")
        pos += keylen
        if pos + 1 > len(blob):
            raise ValueError("truncated state_dict: missing ndim")
        (ndim,) = struct.unpack_from("<B", blob, pos)
        pos += 1
        if pos + 4 * ndim > len(blob):
            raise ValueError("truncated state_dict: missing dims")
        dims = struct.unpack_from("<" + "I" * ndim, blob, pos)
        pos += 4 * ndim
        n_elem = 1
        for d in dims:
            n_elem *= d
        n_bytes = n_elem * 2  # fp16 = 2 bytes/elem
        if pos + n_bytes > len(blob):
            raise ValueError("truncated state_dict: missing tensor data")
        arr = np.frombuffer(blob[pos : pos + n_bytes], dtype=np.float16).copy()
        tensor = torch.from_numpy(arr.reshape(dims)).to(torch.float32)
        out[key] = tensor
        pos += n_bytes
    return out


def pack_archive(
    *,
    num_pairs: int,
    recurrence_mode: int,
    foveation_strategy: int,
    latent_dim: int,
    output_h: int,
    output_w: int,
    world_model_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    z_init: torch.Tensor,
    foveation_meta: dict[str, Any] | dict[str, torch.Tensor],
    residual_blob: bytes,
    meta: dict[str, Any],
) -> bytes:
    """Pack the C1WMFV1 archive bytes.

    Args:
        num_pairs: contest pair count.
        recurrence_mode: 0=GRU 1=LSTM 2=TRANSFORMER.
        foveation_strategy: 0=UNIFORM 1=EGO_MOTION_RADIAL 2=LEARNED_PER_PIXEL.
        latent_dim: world-model latent dim.
        output_h, output_w: scorer resolution height + width.
        world_model_state_dict: brotli'd as the WM_BLOB.
        decoder_state_dict: brotli'd as the DECODER_BLOB.
        z_init: ``(latent_dim,)`` tensor; brotli'd as the ZINIT_BLOB.
        foveation_meta: dict to brotli as FOV_META_BLOB. If values are
            tensors (LEARNED_PER_PIXEL state_dict), encoded via the fp16
            state_dict format. Otherwise encoded as sorted-keys JSON.
        residual_blob: pre-encoded brotli'd per-frame residual blob.
            Caller is responsible for the int8 quantization + brotli
            compression contract.
        meta: sorted-keys JSON metadata.

    Returns:
        Packed archive bytes ready for the contest 0.bin slot.
    """
    if not (0 <= recurrence_mode <= 2):
        raise ValueError(f"recurrence_mode must be 0/1/2; got {recurrence_mode}")
    if not (0 <= foveation_strategy <= 2):
        raise ValueError(
            f"foveation_strategy must be 0/1/2; got {foveation_strategy}"
        )
    if not (1 <= num_pairs <= 0xFFFF):
        raise ValueError(f"num_pairs out of u16 range: {num_pairs}")
    if not (1 <= latent_dim <= 0xFFFF):
        raise ValueError(f"latent_dim out of u16 range: {latent_dim}")
    if not (1 <= output_h <= 0xFFFF):
        raise ValueError(f"output_h out of u16 range: {output_h}")
    if not (1 <= output_w <= 0xFFFF):
        raise ValueError(f"output_w out of u16 range: {output_w}")

    wm_bytes = _tensor_dict_to_fp16_bytes(world_model_state_dict)
    decoder_bytes = _tensor_dict_to_fp16_bytes(decoder_state_dict)
    z_init_bytes = _tensor_dict_to_fp16_bytes({"z_init": z_init})

    # Foveation meta: tensor dict vs JSON dispatch.
    if foveation_meta and all(
        isinstance(v, torch.Tensor) for v in foveation_meta.values()
    ):
        fov_meta_bytes = _tensor_dict_to_fp16_bytes(foveation_meta)  # type: ignore[arg-type]
    else:
        fov_meta_bytes = json.dumps(
            foveation_meta, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    wm_brotli = brotli.compress(wm_bytes, quality=_BROTLI_QUALITY)
    decoder_brotli = brotli.compress(decoder_bytes, quality=_BROTLI_QUALITY)
    z_init_brotli = brotli.compress(z_init_bytes, quality=_BROTLI_QUALITY)
    fov_meta_brotli = brotli.compress(fov_meta_bytes, quality=_BROTLI_QUALITY)
    # residual_blob is already brotli-compressed by caller
    meta_brotli = brotli.compress(meta_bytes, quality=_BROTLI_QUALITY)

    header = struct.pack(
        C1WMFV1_HEADER_FMT,
        C1WMFV1_MAGIC,
        C1WMFV1_SCHEMA_VERSION,
        num_pairs,
        recurrence_mode,
        foveation_strategy,
        latent_dim,
        output_h,
        output_w,
        len(wm_brotli),
        len(decoder_brotli),
        len(z_init_brotli),
        len(fov_meta_brotli),
        len(residual_blob),
        len(meta_brotli),
    )
    return (
        header
        + wm_brotli
        + decoder_brotli
        + z_init_brotli
        + fov_meta_brotli
        + residual_blob
        + meta_brotli
    )


def parse_archive(archive_bytes: bytes) -> WorldModelFoveationArchive:
    """Parse C1WMFV1 archive bytes into a WorldModelFoveationArchive dataclass.

    Args:
        archive_bytes: raw 0.bin contents.

    Returns:
        WorldModelFoveationArchive with all sections decompressed back to
        raw bytes (caller must call _fp16_bytes_to_tensor_dict if a
        state_dict is needed).

    Raises:
        ValueError: malformed magic, version mismatch, truncated body.
    """
    if len(archive_bytes) < C1WMFV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short for C1WMFV1 header: {len(archive_bytes)} bytes"
        )
    (
        magic,
        version,
        num_pairs,
        recurrence_mode,
        foveation_strategy,
        latent_dim,
        output_h,
        output_w,
        wm_len,
        decoder_len,
        z_init_len,
        fov_meta_len,
        residual_len,
        meta_len,
    ) = struct.unpack_from(C1WMFV1_HEADER_FMT, archive_bytes, 0)

    if magic != C1WMFV1_MAGIC:
        raise ValueError(
            f"C1WMFV1 magic mismatch: expected {C1WMFV1_MAGIC!r}, got {magic!r}"
        )
    if version != C1WMFV1_SCHEMA_VERSION:
        raise ValueError(
            f"C1WMFV1 version mismatch: expected {C1WMFV1_SCHEMA_VERSION}, "
            f"got {version}"
        )

    expected_total = (
        C1WMFV1_HEADER_SIZE
        + wm_len
        + decoder_len
        + z_init_len
        + fov_meta_len
        + residual_len
        + meta_len
    )
    if len(archive_bytes) < expected_total:
        raise ValueError(
            f"C1WMFV1 archive truncated: expected {expected_total} bytes, "
            f"got {len(archive_bytes)}"
        )

    pos = C1WMFV1_HEADER_SIZE
    wm_brotli = archive_bytes[pos : pos + wm_len]
    pos += wm_len
    decoder_brotli = archive_bytes[pos : pos + decoder_len]
    pos += decoder_len
    z_init_brotli = archive_bytes[pos : pos + z_init_len]
    pos += z_init_len
    fov_meta_brotli = archive_bytes[pos : pos + fov_meta_len]
    pos += fov_meta_len
    residual_blob = archive_bytes[pos : pos + residual_len]
    pos += residual_len
    meta_brotli = archive_bytes[pos : pos + meta_len]
    pos += meta_len

    wm_bytes = brotli.decompress(wm_brotli) if wm_len > 0 else b""
    decoder_bytes = brotli.decompress(decoder_brotli) if decoder_len > 0 else b""
    z_init_bytes = brotli.decompress(z_init_brotli) if z_init_len > 0 else b""
    fov_meta_bytes = (
        brotli.decompress(fov_meta_brotli) if fov_meta_len > 0 else b""
    )
    meta_bytes = brotli.decompress(meta_brotli) if meta_len > 0 else b"{}"

    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"C1WMFV1 META blob not valid JSON: {exc}") from exc

    return WorldModelFoveationArchive(
        version=version,
        num_pairs=num_pairs,
        recurrence_mode=recurrence_mode,
        foveation_strategy=foveation_strategy,
        latent_dim=latent_dim,
        output_h=output_h,
        output_w=output_w,
        world_model_blob=wm_bytes,
        decoder_blob=decoder_bytes,
        z_init_blob=z_init_bytes,
        foveation_meta_blob=fov_meta_bytes,
        residual_blob=residual_blob,
        meta=meta,
    )


def deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Public wrapper for ``_fp16_bytes_to_tensor_dict`` used by inflate.py."""
    return _fp16_bytes_to_tensor_dict(blob)


__all__ = [
    "C1WMFV1_HEADER_FMT",
    "C1WMFV1_HEADER_SIZE",
    "C1WMFV1_MAGIC",
    "C1WMFV1_SCHEMA_VERSION",
    "WorldModelFoveationArchive",
    "deserialize_state_dict",
    "pack_archive",
    "parse_archive",
]
