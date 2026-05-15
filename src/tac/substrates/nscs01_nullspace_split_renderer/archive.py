# SPDX-License-Identifier: MIT
"""NSCS01 NSP1 archive grammar — monolithic 0.bin (substrate-engineering).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin``
* L4 inflate.py ≤ 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, brotli quality fixed, fp16 packing)

The 4-section grammar (single monolithic 0.bin):

::

    MAGIC(4)               b"NSP\\x01"
    VERSION(1)             u8 (== 1)
    NUM_PAIRS(2)           u16
    LATENT_DIM(2)          u16
    HEAD0_BITS(1)          u8 (4 / 6 / 8)
    HEAD1_BITS(1)          u8 (6 / 8)
    LATENT_BITS(1)         u8 (8 / 12)
    HEAD0_BASE_CH(2)       u16
    HEAD1_BASE_CH(2)       u16
    HEAD0_LEN(4)           u32  brotli(frame_0_head weights packed at HEAD0_BITS)
    HEAD1_LEN(4)           u32  brotli(frame_1_head weights packed at HEAD1_BITS)
    LATENT_LEN(4)          u32  brotli(per-pair latents packed at LATENT_BITS)
    META_LEN(4)            u32  sorted-keys JSON
    HEAD0_BLOB(HEAD0_LEN)
    HEAD1_BLOB(HEAD1_LEN)
    LATENT_BLOB(LATENT_LEN)
    META_BLOB(META_LEN)

Sections:

* **HEAD0_BLOB**: brotli-compressed frame_0_head state-dict packed at
  ``HEAD0_BITS`` bits per weight (the "small" head; targeted ~30K params at
  4-bit ≈ 15 KB raw → ~5-8 KB after brotli).
* **HEAD1_BLOB**: brotli-compressed frame_1_head state-dict packed at
  ``HEAD1_BITS`` bits per weight (the "large" head; targeted ~150K params at
  8-bit ≈ 150 KB raw → ~50-80 KB after brotli).
* **LATENT_BLOB**: brotli-compressed per-pair latents packed at
  ``LATENT_BITS`` bits per scalar (600 * latent_dim entries).
* **META_BLOB**: sorted-keys JSON ``{"head0_arch", "head1_arch",
  "design_notes", ...}``.

NSCS01 design point: ``HEAD0_BITS`` is INDEPENDENTLY CONFIGURABLE from
``HEAD1_BITS`` so the small head can be more aggressively compressed
(SegNet doesn't see frame_0).

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 quantization sentinel, brotli quality fixed)
- No /tmp paths
- No scorer load
- No score claim
- No mutation of base substrate bytes
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

NSP1_MAGIC: bytes = b"NSP\x01"
"""NSCS01 archive magic (4 bytes; distinct from WZF/Z3G/HNRV/etc.)."""

NSP1_SCHEMA_VERSION: int = 1

# Header layout:
# MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + LATENT_DIM(2)
# + HEAD0_BITS(1) + HEAD1_BITS(1) + LATENT_BITS(1)
# + HEAD0_BASE_CH(2) + HEAD1_BASE_CH(2)
# + HEAD0_LEN(4) + HEAD1_LEN(4) + LATENT_LEN(4) + META_LEN(4)
# = 4 + 1 + 2 + 2 + 1 + 1 + 1 + 2 + 2 + 4 + 4 + 4 + 4 = 32 bytes
NSP1_HEADER_FMT: str = "<4sBHHBBBHHIIII"
NSP1_HEADER_SIZE: int = struct.calcsize(NSP1_HEADER_FMT)
assert NSP1_HEADER_SIZE == 32, (
    f"NSP1 header invariant: expected 32, got {NSP1_HEADER_SIZE}"
)

_BROTLI_QUALITY: int = 9


NSP1_SECTION_ROLES: dict[str, str] = {
    "MAGIC": "format_identifier",
    "HEAD0_BLOB": "score_affecting_quantized_weights",
    "HEAD1_BLOB": "score_affecting_quantized_weights",
    "LATENT_BLOB": "score_affecting_quantized_per_pair_latents",
    "META_BLOB": "training_provenance_only",
}


def _quantize_int(weights_flat: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """Per-tensor symmetric int quantization.

    Returns (q, scale, zero_point) where ``q.dtype = int8`` (we always store
    int8 even at 4/6 bit; the reduction comes from the byte-packed layout).
    """
    if bits not in (4, 6, 8, 12):
        raise ValueError(f"unsupported bits: {bits}")
    levels = (1 << bits)
    half = levels // 2
    lo = float(weights_flat.min())
    hi = float(weights_flat.max())
    if hi - lo < 1e-12:
        # Degenerate range — quantize to -half (Catalog #161).
        return np.full(weights_flat.shape, -half, dtype=np.int16), 1.0, lo
    scale = (hi - lo) / (levels - 1)
    q = np.round((weights_flat - lo) / scale).astype(np.int32) - half
    q = np.clip(q, -half, half - 1)
    return q.astype(np.int16), scale, lo


def _dequantize_int(q: np.ndarray, scale: float, lo: float, bits: int) -> np.ndarray:
    levels = (1 << bits)
    half = levels // 2
    return (q.astype(np.float64) + half) * scale + lo


def _pack_bits(q_int16: np.ndarray, bits: int) -> bytes:
    """Pack signed integers (already in [-half, half-1]) at ``bits`` width.

    For 8/12 bit we use straightforward packing; for 4/6 we use bit-packing
    via numpy.packbits-style logic.
    """
    if bits == 8:
        return q_int16.astype(np.int8).tobytes()
    if bits == 12:
        # 12-bit: pack 2 values per 3 bytes. Shift to unsigned [0, 4095].
        u = (q_int16.astype(np.int32) + (1 << (bits - 1))).astype(np.uint16)
        if u.size % 2 != 0:
            u = np.concatenate([u, np.zeros(1, dtype=np.uint16)])
        out = np.empty(u.size // 2 * 3, dtype=np.uint8)
        a = u[0::2]
        b = u[1::2]
        out[0::3] = (a & 0xFF).astype(np.uint8)
        out[1::3] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).astype(np.uint8)
        out[2::3] = ((b >> 4) & 0xFF).astype(np.uint8)
        return out.tobytes()
    if bits == 4:
        u = (q_int16.astype(np.int16) + (1 << (bits - 1))).astype(np.uint8) & 0x0F
        if u.size % 2 != 0:
            u = np.concatenate([u, np.zeros(1, dtype=np.uint8)])
        return ((u[0::2] | (u[1::2] << 4)).astype(np.uint8)).tobytes()
    if bits == 6:
        # 6-bit: pack 4 values per 3 bytes.
        u = (q_int16.astype(np.int16) + (1 << (bits - 1))).astype(np.uint8) & 0x3F
        pad = (-u.size) % 4
        if pad:
            u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
        out = np.empty(u.size // 4 * 3, dtype=np.uint8)
        a = u[0::4]
        b = u[1::4]
        c = u[2::4]
        d = u[3::4]
        out[0::3] = (a | ((b & 0x03) << 6)).astype(np.uint8)
        out[1::3] = (((b >> 2) & 0x0F) | ((c & 0x0F) << 4)).astype(np.uint8)
        out[2::3] = (((c >> 4) & 0x03) | (d << 2)).astype(np.uint8)
        return out.tobytes()
    raise ValueError(f"unsupported bits for packing: {bits}")


def _unpack_bits(packed: bytes, bits: int, n_values: int) -> np.ndarray:
    """Inverse of _pack_bits; returns int16 in [-half, half-1]."""
    half = 1 << (bits - 1)
    if bits == 8:
        return np.frombuffer(packed, dtype=np.int8).astype(np.int16)[:n_values]
    if bits == 12:
        n_pairs = (n_values + 1) // 2
        raw = np.frombuffer(packed, dtype=np.uint8)
        a = raw[0::3].astype(np.uint16) | ((raw[1::3].astype(np.uint16) & 0x0F) << 8)
        b = ((raw[1::3].astype(np.uint16) >> 4) & 0x0F) | (raw[2::3].astype(np.uint16) << 4)
        u = np.empty(n_pairs * 2, dtype=np.uint16)
        u[0::2] = a
        u[1::2] = b
        u = u[:n_values]
        return (u.astype(np.int32) - half).astype(np.int16)
    if bits == 4:
        raw = np.frombuffer(packed, dtype=np.uint8)
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        u = np.empty(raw.size * 2, dtype=np.uint8)
        u[0::2] = lo
        u[1::2] = hi
        u = u[:n_values]
        return (u.astype(np.int16) - half)
    if bits == 6:
        raw = np.frombuffer(packed, dtype=np.uint8)
        n_quads = raw.size // 3
        u = np.empty(n_quads * 4, dtype=np.uint8)
        u[0::4] = raw[0::3] & 0x3F
        u[1::4] = ((raw[0::3] >> 6) & 0x03) | ((raw[1::3] & 0x0F) << 2)
        u[2::4] = ((raw[1::3] >> 4) & 0x0F) | ((raw[2::3] & 0x03) << 4)
        u[3::4] = (raw[2::3] >> 2) & 0x3F
        u = u[:n_values]
        return (u.astype(np.int16) - half)
    raise ValueError(f"unsupported bits for unpacking: {bits}")


def _serialize_state_dict(
    state_dict: dict[str, torch.Tensor], bits: int
) -> tuple[bytes, list[dict]]:
    """Serialize a state_dict as a single packed-bits blob with shape manifest.

    Returns (blob_bytes, shape_manifest) where shape_manifest is a JSON-able
    list of per-tensor entries (name, shape, dtype, scale, lo, n_values).
    The blob concatenates packed bits per tensor in deterministic name order.
    """
    manifest: list[dict] = []
    blob_parts: list[bytes] = []
    # sort by key for deterministic byte layout
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().to(torch.float32).numpy()
        shape = list(tensor.shape)
        flat = tensor.flatten()
        q, scale, lo = _quantize_int(flat, bits)
        packed = _pack_bits(q, bits)
        manifest.append({
            "name": name,
            "shape": shape,
            "dtype": "float32",
            "scale": float(scale),
            "lo": float(lo),
            "n_values": int(flat.size),
            "packed_bytes": len(packed),
        })
        blob_parts.append(packed)
    return b"".join(blob_parts), manifest


def _deserialize_state_dict(
    blob: bytes, bits: int, manifest: list[dict]
) -> dict[str, torch.Tensor]:
    """Inverse of _serialize_state_dict; returns float32 tensors."""
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for entry in manifest:
        n_values = int(entry["n_values"])
        packed_bytes = int(entry["packed_bytes"])
        chunk = blob[offset : offset + packed_bytes]
        offset += packed_bytes
        q = _unpack_bits(chunk, bits, n_values)
        deq = _dequantize_int(q, float(entry["scale"]), float(entry["lo"]), bits)
        tensor = torch.from_numpy(deq.astype(np.float32)).reshape(entry["shape"])
        out[str(entry["name"])] = tensor
    if offset != len(blob):
        raise ValueError(
            f"blob has trailing bytes: consumed {offset}, total {len(blob)}"
        )
    return out


@dataclass(frozen=True)
class NullspaceSplitArchive:
    """Parsed NSP1 archive — the inflate-time data contract."""

    version: int
    num_pairs: int
    latent_dim: int
    head0_bits: int
    head1_bits: int
    latent_bits: int
    head0_base_channels: int
    head1_base_channels: int
    head0_blob: bytes
    head1_blob: bytes
    latent_blob: bytes
    meta: dict[str, object]


def pack_archive(
    *,
    head0_state_dict: dict[str, torch.Tensor],
    head1_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    head0_bits: int,
    head1_bits: int,
    latent_bits: int,
    head0_base_channels: int,
    head1_base_channels: int,
    extra_meta: dict[str, object] | None = None,
) -> bytes:
    """Pack NSCS01 archive bytes (NSP1 grammar).

    Args:
        head0_state_dict / head1_state_dict: render-head weights.
        latents: ``(num_pairs, latent_dim)`` per-pair latent tensor.
        head0_bits / head1_bits / latent_bits: per-section bit-widths.
        head0_base_channels / head1_base_channels: arch reconstruction params.
        extra_meta: optional sidecar dict (merged into META_BLOB).

    Returns:
        bytes of the monolithic 0.bin payload.
    """
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got shape {tuple(latents.shape)}")
    num_pairs, latent_dim = latents.shape

    head0_packed, head0_manifest = _serialize_state_dict(head0_state_dict, head0_bits)
    head1_packed, head1_manifest = _serialize_state_dict(head1_state_dict, head1_bits)

    # Latents: per-tensor quantize the entire (num_pairs * latent_dim) array.
    lat = latents.detach().cpu().to(torch.float32).numpy().flatten()
    lat_q, lat_scale, lat_lo = _quantize_int(lat, latent_bits)
    lat_packed = _pack_bits(lat_q, latent_bits)
    latent_manifest = {
        "shape": [num_pairs, latent_dim],
        "scale": float(lat_scale),
        "lo": float(lat_lo),
        "n_values": int(lat.size),
        "packed_bytes": len(lat_packed),
    }

    head0_blob = brotli.compress(head0_packed, quality=_BROTLI_QUALITY)
    head1_blob = brotli.compress(head1_packed, quality=_BROTLI_QUALITY)
    latent_blob = brotli.compress(lat_packed, quality=_BROTLI_QUALITY)

    meta_dict: dict[str, object] = {
        "head0_manifest": head0_manifest,
        "head1_manifest": head1_manifest,
        "latent_manifest": latent_manifest,
        "head0_arch": "small_pixelshuffle_renderer_v1",
        "head1_arch": "large_pixelshuffle_renderer_v1",
        "design_notes": (
            "NSCS01 nullspace split renderer; frame_0 head is PoseNet-only "
            "(SegNet sees only frame_1 per upstream/modules.py:108)"
        ),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta_dict[k] = v
    meta_blob = json.dumps(
        meta_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    header = struct.pack(
        NSP1_HEADER_FMT,
        NSP1_MAGIC,
        NSP1_SCHEMA_VERSION,
        num_pairs,
        latent_dim,
        head0_bits,
        head1_bits,
        latent_bits,
        head0_base_channels,
        head1_base_channels,
        len(head0_blob),
        len(head1_blob),
        len(latent_blob),
        len(meta_blob),
    )
    return header + head0_blob + head1_blob + latent_blob + meta_blob


def parse_archive(archive_bytes: bytes) -> NullspaceSplitArchive:
    """Parse an NSP1 archive blob into typed sections."""
    if len(archive_bytes) < NSP1_HEADER_SIZE:
        raise ValueError(
            f"archive too small: {len(archive_bytes)} < header {NSP1_HEADER_SIZE}"
        )
    header = struct.unpack(
        NSP1_HEADER_FMT, archive_bytes[:NSP1_HEADER_SIZE]
    )
    (
        magic,
        version,
        num_pairs,
        latent_dim,
        head0_bits,
        head1_bits,
        latent_bits,
        head0_base_channels,
        head1_base_channels,
        head0_len,
        head1_len,
        latent_len,
        meta_len,
    ) = header
    if magic != NSP1_MAGIC:
        raise ValueError(f"NSP1 magic mismatch: expected {NSP1_MAGIC!r}, got {magic!r}")
    if version != NSP1_SCHEMA_VERSION:
        raise ValueError(f"NSP1 version mismatch: expected {NSP1_SCHEMA_VERSION}, got {version}")
    cursor = NSP1_HEADER_SIZE
    head0_blob_compressed = archive_bytes[cursor : cursor + head0_len]
    cursor += head0_len
    head1_blob_compressed = archive_bytes[cursor : cursor + head1_len]
    cursor += head1_len
    latent_blob_compressed = archive_bytes[cursor : cursor + latent_len]
    cursor += latent_len
    meta_blob_bytes = archive_bytes[cursor : cursor + meta_len]
    cursor += meta_len
    if cursor != len(archive_bytes):
        raise ValueError(
            f"archive has trailing bytes: consumed {cursor}, total {len(archive_bytes)}"
        )
    head0_blob = brotli.decompress(head0_blob_compressed)
    head1_blob = brotli.decompress(head1_blob_compressed)
    latent_blob = brotli.decompress(latent_blob_compressed)
    meta = json.loads(meta_blob_bytes.decode("utf-8"))
    return NullspaceSplitArchive(
        version=int(version),
        num_pairs=int(num_pairs),
        latent_dim=int(latent_dim),
        head0_bits=int(head0_bits),
        head1_bits=int(head1_bits),
        latent_bits=int(latent_bits),
        head0_base_channels=int(head0_base_channels),
        head1_base_channels=int(head1_base_channels),
        head0_blob=head0_blob,
        head1_blob=head1_blob,
        latent_blob=latent_blob,
        meta=meta,
    )


def deserialize_head_state_dicts(
    arc: NullspaceSplitArchive,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Deserialize the per-head state-dicts from the parsed archive."""
    head0_manifest = arc.meta.get("head0_manifest")
    head1_manifest = arc.meta.get("head1_manifest")
    if not isinstance(head0_manifest, list) or not isinstance(head1_manifest, list):
        raise ValueError("archive meta missing head0/head1 manifest")
    head0_sd = _deserialize_state_dict(arc.head0_blob, arc.head0_bits, head0_manifest)
    head1_sd = _deserialize_state_dict(arc.head1_blob, arc.head1_bits, head1_manifest)
    return head0_sd, head1_sd


def deserialize_latents(arc: NullspaceSplitArchive) -> torch.Tensor:
    """Deserialize the per-pair latents from the parsed archive."""
    lat_manifest = arc.meta.get("latent_manifest")
    if not isinstance(lat_manifest, dict):
        raise ValueError("archive meta missing latent_manifest")
    n_values = int(lat_manifest["n_values"])
    q = _unpack_bits(arc.latent_blob, arc.latent_bits, n_values)
    deq = _dequantize_int(
        q, float(lat_manifest["scale"]), float(lat_manifest["lo"]), arc.latent_bits
    )
    shape = lat_manifest["shape"]
    return torch.from_numpy(deq.astype(np.float32)).reshape(shape)


__all__ = [
    "NSP1_HEADER_FMT",
    "NSP1_HEADER_SIZE",
    "NSP1_MAGIC",
    "NSP1_SCHEMA_VERSION",
    "NSP1_SECTION_ROLES",
    "NullspaceSplitArchive",
    "deserialize_head_state_dicts",
    "deserialize_latents",
    "pack_archive",
    "parse_archive",
]
