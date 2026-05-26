# SPDX-License-Identifier: MIT
"""DreamerV3 RSSM archive grammar — RSSMC1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

L0 SCAFFOLD scope: the grammar defined here is the canonical contract; the
PyTorch port for inflate-time consumption is queued per Path 3 cascade
(symposium op-routable #2 + sister #1251 export bridge + #1257 inflate parity
closure). The archive grammar is byte-deterministic from MLX-trained weights
via the canonical write/parse round-trip implemented below.

Grammar:

::

    MAGIC(4)            b"RSSC"  # RSSM Categorical v1
    VERSION(1)          u8       schema version (currently 1)
    NUM_GROUPS_G(2)     u16      cfg.num_groups (e.g. 24)
    NUM_CATEGORIES_K(2) u16      cfg.num_categories (e.g. 256)
    NUM_PAIRS(2)        u16      cfg.num_pairs (e.g. 600)
    DECODER_LATENT(2)   u16      cfg.decoder_latent_dim (e.g. 28)
    BASE_CHANNELS(2)    u16      cfg.base_channels (e.g. 24)
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder + cat_proj weights len
    INDICES_BLOB_LEN(4) u32      raw int8/u8 per-pair category indices len
    META_BLOB_LEN(4)    u32      sorted-keys JSON utf-8 bytes len
    DECODER_BLOB        ...      brotli(quality=9) of state_dict bytes (fp16)
    INDICES_BLOB        ...      per-pair category indices (num_pairs * G bytes
                                 for K<=256; num_pairs * G * 2 for K>256)
    META_BLOB           ...      sorted-keys JSON: {"gumbel_temperature",
                                  "use_straight_through", "categorical_bits", ...}

The decoder + cat_to_continuous projection state_dict is the inflate-time
consumer. The per-pair learned logits (training only) are REDUCED to argmax
indices for archive serialization (canonical Hafner 2024 + vdOord VQ-VAE
discrete-latent contract). Inflate reconstructs one-hot from indices.

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load at inflate
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  fp16 state_dict cast on CPU; raw bytes for category indices)

Per Catalog #139 + #105 + #272 distinguishing-feature contract:
- decoder_blob bytes ARE frame-affecting at inflate (decoder weights consumed)
- indices_blob bytes ARE frame-affecting at inflate (categorical samples
  consumed; THIS IS THE DISTINGUISHING FEATURE vs C6 IBPS continuous latent)
- header + meta bytes are parse/config gates (control_or_metadata role)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np


RSSMC1_MAGIC: bytes = b"RSSC"
"""DreamerV3 RSSM Categorical variant 1 archive magic."""

RSSMC1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring:
# MAGIC(4) + VERSION(1) + G(2) + K(2) + N(2) + L(2) + BC(2) + DEC_LEN(4) + IDX_LEN(4) + META_LEN(4)
# = 4+1+2+2+2+2+2+4+4+4 = 27 bytes
RSSMC1_HEADER_FMT: str = "<4sBHHHHHIII"
RSSMC1_HEADER_SIZE: int = struct.calcsize(RSSMC1_HEADER_FMT)
assert RSSMC1_HEADER_SIZE == 27, (
    f"RSSMC1 header size invariant: expected 27, got {RSSMC1_HEADER_SIZE}"
)

# Deterministic brotli quality (matches C6 IBPS / sane_hnerv / WZF01 siblings).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class DreamerV3RSSMArchive:
    """Parsed RSSMC1 archive — the inflate-time data contract."""

    decoder_state_dict: dict[str, Any]
    """Decoder + cat_to_continuous state_dict (the inflate-time consumer)."""

    category_indices: np.ndarray
    """``(num_pairs, num_groups)`` int32 (decoded from u8/u16 archive bytes)."""

    meta: dict[str, Any]
    """Sidecar JSON meta: gumbel_temperature, categorical_bits, ..."""

    schema_version: int
    num_groups: int
    num_categories: int
    num_pairs: int
    decoder_latent_dim: int
    base_channels: int


def _serialize_state_dict_fp16(sd: dict[str, Any]) -> bytes:
    """Serialize a state_dict deterministically as length-prefixed records.

    Format per entry (sorted by key for determinism):

        u16 key_len + key_bytes (utf-8)
        u8 ndim
        u32 shape[0..ndim-1] (little-endian)
        fp16 tensor bytes (row-major contiguous, numpy)

    Returns brotli-compressed concatenation. Mirrors the canonical C6 IBPS +
    sane_hnerv pattern; pickle avoided per Catalog #14 weights_only discipline.

    Tensor input may be any object exposing ``.shape`` + numpy-castable values
    (numpy arrays, MLX arrays via ``np.asarray``, torch tensors via
    ``.detach().cpu().numpy()``).
    """
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key]
        # Normalize to numpy float32 first (MLX arrays, torch tensors, numpy)
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        # Cast to fp16 for archive (saves 50% vs fp32)  # DOCSTRING_PERCENT_CLAIM_OK:fp16_is_2_bytes_fp32_is_4_bytes_50_percent_savings_is_definitional_arithmetic_not_an_empirical_claim
        arr16 = arr.astype(np.float16)
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} too long for u16 length")
        shape = tuple(int(s) for s in arr16.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} has too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(arr16.tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_state_dict_fp16(blob: bytes) -> dict[str, np.ndarray]:
    """Inverse of ``_serialize_state_dict_fp16``. Returns numpy fp16 arrays.

    Caller can cast to torch.Tensor or mlx.array as needed.
    """
    raw = brotli.decompress(blob)
    sd: dict[str, np.ndarray] = {}
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2  # fp16
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(raw[pos : pos + tensor_bytes], dtype=np.float16).reshape(
            shape
        ).astype(np.float16, copy=True)
        sd[key] = arr
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(
            f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})"
        )
    return sd


def _pack_indices(indices: np.ndarray, num_categories: int) -> bytes:
    """Pack ``(num_pairs, num_groups)`` int32 indices to compact bytes.

    K<=256: 1 byte/index (uint8); K<=65536: 2 bytes/index (uint16).
    """
    if num_categories <= 0:
        raise ValueError(f"num_categories must be positive; got {num_categories}")
    if indices.dtype.kind not in ("i", "u"):
        raise ValueError(f"indices must be integer dtype; got {indices.dtype}")
    if indices.min() < 0 or indices.max() >= num_categories:
        raise ValueError(
            f"indices out of [0, {num_categories}) range: "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )
    if num_categories <= 256:
        return indices.astype(np.uint8).tobytes(order="C")
    if num_categories <= 65536:
        return indices.astype(np.uint16).tobytes(order="C")
    raise ValueError(f"num_categories={num_categories} exceeds u16 packing limit")


def _unpack_indices(
    raw: bytes,
    *,
    num_pairs: int,
    num_groups: int,
    num_categories: int,
) -> np.ndarray:
    """Inverse of ``_pack_indices``."""
    if num_categories <= 256:
        dtype = np.uint8
        bytes_per_idx = 1
    elif num_categories <= 65536:
        dtype = np.uint16
        bytes_per_idx = 2
    else:
        raise ValueError(f"num_categories={num_categories} exceeds u16 packing limit")
    expected_bytes = num_pairs * num_groups * bytes_per_idx
    if len(raw) != expected_bytes:
        raise ValueError(
            f"indices_blob: got {len(raw)} bytes, expected {expected_bytes} "
            f"(num_pairs={num_pairs}, num_groups={num_groups}, "
            f"bytes_per_idx={bytes_per_idx})"
        )
    arr = np.frombuffer(raw, dtype=dtype).reshape(num_pairs, num_groups).copy()
    return arr.astype(np.int32, copy=False)


def pack_archive(
    decoder_state_dict: dict[str, Any],
    category_indices: np.ndarray,
    meta: dict[str, Any],
    *,
    num_groups: int,
    num_categories: int,
    num_pairs: int,
    decoder_latent_dim: int,
    base_channels: int,
    schema_version: int = RSSMC1_SCHEMA_VERSION,
) -> bytes:
    """Serialize substrate weights + per-pair category indices + meta to RSSMC1 bytes."""
    if schema_version != RSSMC1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if category_indices.ndim != 2:
        raise ValueError(
            f"category_indices must be 2-D (num_pairs, num_groups); "
            f"got shape {category_indices.shape}"
        )
    if category_indices.shape != (num_pairs, num_groups):
        raise ValueError(
            f"category_indices shape {category_indices.shape} != expected "
            f"({num_pairs}, {num_groups})"
        )
    for name, value in [
        ("num_groups", num_groups),
        ("num_categories", num_categories),
        ("num_pairs", num_pairs),
        ("decoder_latent_dim", decoder_latent_dim),
        ("base_channels", base_channels),
    ]:
        if value <= 0 or value > 0xFFFF:
            raise ValueError(f"{name} {value} out of u16 range [1, 65535]")

    decoder_blob = _serialize_state_dict_fp16(decoder_state_dict)
    indices_bytes = _pack_indices(category_indices, num_categories)

    meta_bytes = json.dumps(
        meta, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        RSSMC1_HEADER_FMT,
        RSSMC1_MAGIC,
        schema_version,
        num_groups,
        num_categories,
        num_pairs,
        decoder_latent_dim,
        base_channels,
        len(decoder_blob),
        len(indices_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + indices_bytes + meta_bytes


def parse_rssmc1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for RSSMC1 grammar.

    Canonical section-offset parser. The returned mapping is the data contract
    consumed by ``tac.analysis.scorer_conditional_mdl`` (Tier A density
    estimation) + ``tools.mdl_scorer_conditional_ablation`` (CLI three-tier
    ablation; per-substrate auto-detection by ``b"RSSC"`` magic prefix).

    Returned sections (Tier A / Tier B targets):

    - ``rssmc1_header`` — 27-byte header (control_or_metadata; fixed layout)
    - ``decoder_blob`` — brotli q=9 compressed decoder + cat_proj weights
      (decoder_weight_stream — the actual inflate-time consumer)
    - ``indices_blob`` — packed per-pair category indices (latent_stream;
      DISTINGUISHING FEATURE per Catalog #272)
    - ``meta_blob`` — sorted-keys JSON sidecar (control_or_metadata)

    Raises ``ValueError`` on short header / bad magic / wrong version /
    declared end_meta != len(archive_bytes).
    """
    if len(archive_bytes) < RSSMC1_HEADER_SIZE:
        raise ValueError(
            f"rssmc1 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {RSSMC1_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        num_groups,
        num_categories,
        num_pairs,
        decoder_latent_dim,
        base_channels,
        decoder_len,
        indices_len,
        meta_len,
    ) = struct.unpack(RSSMC1_HEADER_FMT, archive_bytes[:RSSMC1_HEADER_SIZE])
    if magic != RSSMC1_MAGIC:
        raise ValueError(
            f"rssmc1 archive: bad magic {magic!r} (expected {RSSMC1_MAGIC!r})"
        )
    if version != RSSMC1_SCHEMA_VERSION:
        raise ValueError(
            f"rssmc1 archive: unsupported schema version {version} "
            f"(expected {RSSMC1_SCHEMA_VERSION})"
        )
    # Validate indices length matches declared dims
    expected_idx_bytes = int(num_pairs) * int(num_groups) * (
        1 if int(num_categories) <= 256 else 2
    )
    if int(indices_len) != expected_idx_bytes:
        raise ValueError(
            f"rssmc1 archive: indices_len {int(indices_len)} != expected "
            f"{expected_idx_bytes} (num_pairs={int(num_pairs)}, "
            f"num_groups={int(num_groups)}, num_categories={int(num_categories)})"
        )
    end_header = RSSMC1_HEADER_SIZE
    end_decoder = end_header + int(decoder_len)
    end_indices = end_decoder + int(indices_len)
    end_meta = end_indices + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"rssmc1 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    return {
        "rssmc1_header": (0, RSSMC1_HEADER_SIZE),
        "decoder_blob": (end_header, int(decoder_len)),
        "indices_blob": (end_decoder, int(indices_len)),
        "meta_blob": (end_indices, int(meta_len)),
    }


# Canonical optimization-role mapping for RSSMC1 sections (mirrors C6 IBPS1
# pattern at IBPS1_SECTION_ROLES).
RSSMC1_SECTION_ROLES: dict[str, str] = {
    "rssmc1_header": "control_or_metadata",
    "decoder_blob": "decoder_weight_stream",
    "indices_blob": "latent_stream",
    "meta_blob": "control_or_metadata",
}


def parse_archive(blob: bytes) -> DreamerV3RSSMArchive:
    """Parse RSSMC1 0.bin bytes back into decoder + indices + meta."""
    sections = parse_rssmc1_archive_bytes(blob)
    # Re-extract header fields (parse_rssmc1_archive_bytes already validated)
    (
        _magic,
        version,
        num_groups,
        num_categories,
        num_pairs,
        decoder_latent_dim,
        base_channels,
        _decoder_len,
        _indices_len,
        _meta_len,
    ) = struct.unpack(RSSMC1_HEADER_FMT, blob[:RSSMC1_HEADER_SIZE])

    dec_start, dec_len = sections["decoder_blob"]
    idx_start, idx_len = sections["indices_blob"]
    meta_start, meta_len = sections["meta_blob"]

    decoder_sd_fp16 = _deserialize_state_dict_fp16(blob[dec_start : dec_start + dec_len])
    category_indices = _unpack_indices(
        blob[idx_start : idx_start + idx_len],
        num_pairs=int(num_pairs),
        num_groups=int(num_groups),
        num_categories=int(num_categories),
    )
    meta = json.loads(blob[meta_start : meta_start + meta_len].decode("utf-8"))

    return DreamerV3RSSMArchive(
        decoder_state_dict={k: v for k, v in decoder_sd_fp16.items()},
        category_indices=category_indices,
        meta=meta,
        schema_version=int(version),
        num_groups=int(num_groups),
        num_categories=int(num_categories),
        num_pairs=int(num_pairs),
        decoder_latent_dim=int(decoder_latent_dim),
        base_channels=int(base_channels),
    )


__all__ = [
    "DreamerV3RSSMArchive",
    "RSSMC1_HEADER_FMT",
    "RSSMC1_HEADER_SIZE",
    "RSSMC1_MAGIC",
    "RSSMC1_SCHEMA_VERSION",
    "RSSMC1_SECTION_ROLES",
    "pack_archive",
    "parse_archive",
    "parse_rssmc1_archive_bytes",
]
