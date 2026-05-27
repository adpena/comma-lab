# SPDX-License-Identifier: MIT
"""ATW codec V1 archive grammar — ATW1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

ATW1 is BYTEWISE-DISTINCT from sister Z3HP1 / Z4CR1 / IBPS1 grammars:

* Different magic ``b"ATW1"`` (4 bytes) — so ``parse_atw1_archive_bytes``
  refuses non-ATW1 archives at byte 0.
* Adds two new sections beyond Z4CR1's encoder + decoder + latent + meta:
  ``wz_side_info_head_blob`` (the tiny WZ MLP weights) and
  ``scorer_class_prior_table_blob`` (per-pair scorer class prior fp16 table).
* The ``latent_blob`` ships ``z_residual`` rather than ``z`` (this is the
  Wyner-Ziv compression mechanism — ``z`` is reconstructed at inflate time
  via ``z = z_residual + wz_side_info_head(class_prior_table[pair])``).

Grammar:

::

    MAGIC(4)                    b"ATW1"
    VERSION(1)                  u8       schema version (currently 1)
    LATENT_DIM(2)               u16      cfg.latent_dim (e.g. 24)
    NUM_PAIRS(2)                u16      cfg.num_pairs (e.g. 600)
    SCORER_CLASS_PRIOR_DIM(2)   u16      cfg.scorer_class_prior_dim (e.g. 16)
    ENCODER_BLOB_LEN(4)         u32      brotli-compressed encoder state_dict bytes
    DECODER_BLOB_LEN(4)         u32      brotli-compressed decoder state_dict bytes
    WZ_HEAD_BLOB_LEN(4)         u32      brotli-compressed wz_side_info_head bytes
    LATENT_RESIDUAL_BLOB_LEN(4) u32      int8 z_residual bytes (= num_pairs * latent_dim)
    CLASS_PRIOR_TABLE_BLOB_LEN(4) u32   fp16 scorer_class_prior_table bytes
                                          (= num_pairs * scorer_class_prior_dim * 2)
    META_BLOB_LEN(4)            u32      sorted-keys JSON utf-8 bytes
    ENCODER_BLOB               ...
    DECODER_BLOB               ...
    WZ_HEAD_BLOB               ...
    LATENT_RESIDUAL_BLOB       ...
    CLASS_PRIOR_TABLE_BLOB     ...
    META_BLOB                  ...

Round-trip contract:

    bytes → parse_archive → (encoder_sd, decoder_sd, wz_head_sd, latents,
                            class_prior_table, meta)
    (encoder_sd, decoder_sd, wz_head_sd, latents, class_prior_table, meta) →
        pack_archive → bytes

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load at inflate time (scorer class prior is PRECOMPUTED at compress)
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  fp16 state_dict cast on CPU; deterministic int8 latent quantization)
- Per Catalog #161 degenerate-range fix mirrored from Z4 + sane_hnerv
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

ATW1_MAGIC: bytes = b"ATW1"
"""ATW codec V1 archive magic."""

ATW1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#                + SCORER_CLASS_PRIOR_DIM(2)
#                + ENCODER_LEN(4) + DECODER_LEN(4) + WZ_HEAD_LEN(4)
#                + LATENT_RESIDUAL_LEN(4) + CLASS_PRIOR_TABLE_LEN(4) + META_LEN(4)
# = 4+1+2+2+2+4+4+4+4+4+4 = 35 bytes
ATW1_HEADER_FMT: str = "<4sBHHHIIIIII"
ATW1_HEADER_SIZE: int = struct.calcsize(ATW1_HEADER_FMT)
assert ATW1_HEADER_SIZE == 35, f"header size invariant: expected 35, got {ATW1_HEADER_SIZE}"

# Deterministic brotli quality (matches sister substrates).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class ATWCodecArchive:
    """Parsed ATW1 archive — the inflate-time data contract."""

    encoder_state_dict: dict[str, torch.Tensor]
    """Encoder state_dict (provenance; not strictly required at inflate)."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (the inflate-time consumer)."""

    wz_side_info_head_state_dict: dict[str, torch.Tensor]
    """WZ side-info head state_dict (inflate-time consumer for z reconstruction)."""

    latent_residual: torch.Tensor
    """``(num_pairs, latent_dim)`` float32 dequantized z_residual."""

    scorer_class_prior_table: torch.Tensor
    """``(num_pairs, scorer_class_prior_dim)`` fp32 scorer class prior table."""

    meta: dict[str, object]
    """Sidecar JSON meta with atw_codec_meta provenance fields."""

    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict deterministically as length-prefixed records.

    Format mirrors Z4's deterministic serializer — no pickle (memo IDs vary
    across instantiations); fp16 cast on CPU; sorted by key.
    """
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} is too long for u16 length")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} has too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(tensor.numpy().tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_numpy_state_dict(blob: bytes) -> dict[str, "np.ndarray"]:
    """Torch-free deserialize of the ATW1 length-prefixed fp16 state_dict blob.

    Reads the EXACT same byte format ``_serialize_state_dict`` produces, but
    reconstructs each tensor as an ``np.float32`` ndarray with NO torch import.
    The shipped numpy-portable inflate runtime calls this (8th MLX-first
    standing directive: ``torch`` is FORBIDDEN at decode time). Returns ``{}``
    for an empty blob (matching the torch-side helper).
    """
    if not blob:
        return {}
    raw = brotli.decompress(blob)
    sd: dict[str, "np.ndarray"] = {}
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
        arr = np.frombuffer(raw[pos : pos + tensor_bytes], dtype=np.float16).copy()
        sd[key] = arr.reshape(shape).astype(np.float32)
        pos += tensor_bytes
    return sd


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    if not blob:
        return {}
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
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
        arr = np.frombuffer(raw[pos : pos + tensor_bytes], dtype=np.float16).copy()
        sd[key] = torch.from_numpy(arr).reshape(shape).to(torch.float32)
        pos += tensor_bytes
    return sd


def _quantize_latents_to_int8(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize latents to int8 with the canonical degenerate-range fix.

    Per Catalog #161: if hi <= lo (all-equal), q is filled with -127 so
    dequant(-127) = 0 * scale + lo = lo. Mirrors Z4 and sane_hnerv exactly.
    """
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -127, dtype=torch.int8),
            1.0,
            lo,
        )
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return (q, scale, lo)


def _dequantize_latents(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 127.0
    return q_unsigned * float(scale) + float(zero_point)


def _make_atw_codec_meta_block(
    base_meta: dict[str, object],
    *,
    atw_kappa_ib: float,
    atw_lambda_wz: float,
    atw_lambda_pixel: float,
    atw_atick_redlich_form: bool,
    wz_head_enabled: bool,
) -> dict[str, object]:
    """Inject the ATW codec provenance tag into the meta dict.

    The tag is structured so audit tools can distinguish an ATW archive
    from Z3/Z4 archives at meta-blob hash level — ATW1 carries explicit
    knob values (κ_IB / λ_WZ / λ_pixel) so the four corner regimes
    (Atick-only / ATW canonical / Tishby IB / Z3 baseline) are
    recoverable from archive bytes alone.
    """
    meta = dict(base_meta)
    meta["atw_codec_meta"] = {
        "kappa_ib": float(atw_kappa_ib),
        "lambda_wz": float(atw_lambda_wz),
        "lambda_pixel": float(atw_lambda_pixel),
        "atick_redlich_form": bool(atw_atick_redlich_form),
        "wz_head_enabled": bool(wz_head_enabled),
        "literature_anchor": [
            "Atick-Redlich1990",
            "Tishby-Pereira-Bialek1999",
            "Wyner-Ziv1976",
        ],
        "predicted_band_lo": 0.18,
        "predicted_band_hi": 0.21,
        "composite_id": "atw_codec_v1",
    }
    return meta


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    wz_side_info_head_state_dict: dict[str, torch.Tensor],
    latent_residual: torch.Tensor,
    scorer_class_prior_table: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = ATW1_SCHEMA_VERSION,
    atw_kappa_ib: float = 0.0,
    atw_lambda_wz: float = 1.0,
    atw_lambda_pixel: float = 0.0,
    atw_atick_redlich_form: bool = True,
    wz_head_enabled: bool = True,
) -> bytes:
    """Serialize trained ATW substrate weights + WZ residual + class prior table → ATW1 0.bin bytes."""
    if schema_version != ATW1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latent_residual.dim() != 2:
        raise ValueError(
            f"latent_residual must be 2-D (num_pairs, latent_dim); "
            f"got {tuple(latent_residual.shape)}"
        )
    if scorer_class_prior_table.dim() != 2:
        raise ValueError(
            f"scorer_class_prior_table must be 2-D (num_pairs, prior_dim); "
            f"got {tuple(scorer_class_prior_table.shape)}"
        )
    num_pairs, latent_dim = int(latent_residual.shape[0]), int(latent_residual.shape[1])
    table_pairs, prior_dim = (
        int(scorer_class_prior_table.shape[0]),
        int(scorer_class_prior_table.shape[1]),
    )
    if num_pairs != table_pairs:
        raise ValueError(
            f"latent_residual and scorer_class_prior_table num_pairs mismatch: "
            f"{num_pairs} vs {table_pairs}"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if prior_dim <= 0 or prior_dim > 0xFFFF:
        raise ValueError(f"scorer_class_prior_dim {prior_dim} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int8(latent_residual)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)
    wz_head_blob = _serialize_state_dict(wz_side_info_head_state_dict)

    table_fp16 = (
        scorer_class_prior_table.detach().to("cpu", dtype=torch.float16).contiguous()
    )
    class_prior_table_bytes = table_fp16.numpy().tobytes(order="C")

    meta_with_tag = _make_atw_codec_meta_block(
        meta,
        atw_kappa_ib=atw_kappa_ib,
        atw_lambda_wz=atw_lambda_wz,
        atw_lambda_pixel=atw_lambda_pixel,
        atw_atick_redlich_form=atw_atick_redlich_form,
        wz_head_enabled=wz_head_enabled,
    )
    meta_with_tag["_lat_scale"] = float(scale)
    meta_with_tag["_lat_zero_point"] = float(zero_point)
    meta_with_tag["_scorer_class_prior_dim"] = int(prior_dim)
    meta_bytes = json.dumps(
        meta_with_tag, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        ATW1_HEADER_FMT,
        ATW1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        prior_dim,
        len(encoder_blob),
        len(decoder_blob),
        len(wz_head_blob),
        len(latent_bytes),
        len(class_prior_table_bytes),
        len(meta_bytes),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + wz_head_blob
        + latent_bytes
        + class_prior_table_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> ATWCodecArchive:
    """Parse ATW1 0.bin bytes back into the full ATW substrate state."""
    if len(blob) < ATW1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {ATW1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        prior_dim,
        encoder_len,
        decoder_len,
        wz_head_len,
        latent_len,
        class_prior_table_len,
        meta_len,
    ) = struct.unpack(ATW1_HEADER_FMT, blob[:ATW1_HEADER_SIZE])
    if magic != ATW1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {ATW1_MAGIC!r})")
    if version != ATW1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim  # int8 = 1 byte
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_residual blob: latent_len {latent_len} != "
            f"num_pairs*latent_dim = {expected_latent_bytes}"
        )
    expected_class_prior_table_bytes = num_pairs * prior_dim * 2  # fp16 = 2 bytes
    if class_prior_table_len != expected_class_prior_table_bytes:
        raise ValueError(
            f"class_prior_table blob: class_prior_table_len "
            f"{class_prior_table_len} != num_pairs*prior_dim*2 = "
            f"{expected_class_prior_table_bytes}"
        )

    end_header = ATW1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_wz_head = end_decoder + int(wz_head_len)
    end_latents = end_wz_head + int(latent_len)
    end_class_prior_table = end_latents + int(class_prior_table_len)
    end_meta = end_class_prior_table + int(meta_len)
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_blob = blob[end_header:end_encoder]
    decoder_blob = blob[end_encoder:end_decoder]
    wz_head_blob = blob[end_decoder:end_wz_head]
    latent_blob = blob[end_wz_head:end_latents]
    class_prior_table_blob = blob[end_latents:end_class_prior_table]
    meta_blob = blob[end_class_prior_table:end_meta]

    encoder_sd = _deserialize_state_dict(encoder_blob)
    decoder_sd = _deserialize_state_dict(decoder_blob)
    wz_head_sd = _deserialize_state_dict(wz_head_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np

    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int8).copy()
    ).view(num_pairs, latent_dim)
    scale = float(meta.pop("_lat_scale"))
    zp = float(meta.pop("_lat_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    class_prior_table = torch.from_numpy(
        np.frombuffer(class_prior_table_blob, dtype=np.float16).copy()
    ).view(num_pairs, prior_dim).to(torch.float32)

    return ATWCodecArchive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        wz_side_info_head_state_dict=wz_head_sd,
        latent_residual=latents,
        scorer_class_prior_table=class_prior_table,
        meta=meta,
        schema_version=int(version),
    )


@dataclass(frozen=True)
class ATWCodecArchiveNumpy:
    """Torch-free parsed ATW1 archive — the numpy-portable inflate-time contract.

    Sister of :class:`ATWCodecArchive` but with ``np.ndarray`` weights / latents
    so the shipped inflate runtime needs ONLY numpy + brotli (no torch) per the
    8th MLX-first standing directive. The ``decoder_state_dict`` +
    ``wz_side_info_head_state_dict`` are the inflate-time consumers; the encoder
    is provenance-only (kept for parity with the torch-side parse).
    """

    encoder_state_dict: dict[str, "np.ndarray"]
    decoder_state_dict: dict[str, "np.ndarray"]
    wz_side_info_head_state_dict: dict[str, "np.ndarray"]
    latent_residual: "np.ndarray"
    """``(num_pairs, latent_dim)`` fp32 dequantized z_residual."""
    scorer_class_prior_table: "np.ndarray"
    """``(num_pairs, scorer_class_prior_dim)`` fp32 scorer class prior table."""
    meta: dict[str, object]
    schema_version: int


def parse_archive_numpy(blob: bytes) -> ATWCodecArchiveNumpy:
    """Torch-free parse of an ATW1 archive for the numpy-portable inflate.

    Identical section walk to :func:`parse_archive` but reconstructs every
    weight / latent / class-prior as a numpy array with NO torch import. Per the
    8th MLX-first directive's bridge contract the shipped inflate runtime reads
    weights via this path so the runtime tree carries only numpy + brotli.
    """
    if len(blob) < ATW1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {ATW1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        prior_dim,
        encoder_len,
        decoder_len,
        wz_head_len,
        latent_len,
        class_prior_table_len,
        meta_len,
    ) = struct.unpack(ATW1_HEADER_FMT, blob[:ATW1_HEADER_SIZE])
    if magic != ATW1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {ATW1_MAGIC!r})")
    if version != ATW1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim  # int8 = 1 byte
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_residual blob: latent_len {latent_len} != "
            f"num_pairs*latent_dim = {expected_latent_bytes}"
        )
    expected_class_prior_table_bytes = num_pairs * prior_dim * 2  # fp16 = 2 bytes
    if class_prior_table_len != expected_class_prior_table_bytes:
        raise ValueError(
            f"class_prior_table blob: class_prior_table_len "
            f"{class_prior_table_len} != num_pairs*prior_dim*2 = "
            f"{expected_class_prior_table_bytes}"
        )

    end_header = ATW1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_wz_head = end_decoder + int(wz_head_len)
    end_latents = end_wz_head + int(latent_len)
    end_class_prior_table = end_latents + int(class_prior_table_len)
    end_meta = end_class_prior_table + int(meta_len)
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_sd = _deserialize_numpy_state_dict(blob[end_header:end_encoder])
    decoder_sd = _deserialize_numpy_state_dict(blob[end_encoder:end_decoder])
    wz_head_sd = _deserialize_numpy_state_dict(blob[end_decoder:end_wz_head])
    meta = json.loads(blob[end_class_prior_table:end_meta].decode("utf-8"))

    q = np.frombuffer(blob[end_wz_head:end_latents], dtype=np.int8).reshape(
        num_pairs, latent_dim
    )
    scale = float(meta.pop("_lat_scale"))
    zp = float(meta.pop("_lat_zero_point"))
    # dequant mirrors _dequantize_latents: (q + 127) * scale + zero_point
    latents = (q.astype(np.float32) + 127.0) * scale + zp

    class_prior_table = (
        np.frombuffer(
            blob[end_latents:end_class_prior_table], dtype=np.float16
        )
        .reshape(num_pairs, prior_dim)
        .astype(np.float32)
    )

    return ATWCodecArchiveNumpy(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        wz_side_info_head_state_dict=wz_head_sd,
        latent_residual=latents,
        scorer_class_prior_table=class_prior_table,
        meta=meta,
        schema_version=int(version),
    )


def parse_atw1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for ATW1 grammar.

    Canonical section-offset parser for ATW1 inner-blob bytes. The returned
    mapping is the data contract consumed by:

    - :mod:`tac.analysis.scorer_conditional_mdl` (Tier A density estimation)
    - :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
      dispatch — ATW1 auto-detection by ``b"ATW1"`` magic prefix)

    Returned sections (Tier A / Tier B targets):

    - ``atw1_header`` — 35-byte header (control_or_metadata; fixed layout)
    - ``encoder_blob`` — brotli q=9 compressed encoder state_dict
      (training_provenance_only — parsed/loaded for provenance, but not
      score-affecting at inflate because ``frames_for_encoder=None``)
    - ``decoder_blob`` — brotli q=9 compressed decoder state_dict
      (decoder_weight_stream — the actual inflate-time consumer)
    - ``wz_head_blob`` — brotli q=9 compressed wz_side_info_head state_dict
      (decoder_weight_stream — operational consumer for z reconstruction)
    - ``latent_residual_blob`` — int8 quantized z_residual (latent_stream)
    - ``class_prior_table_blob`` — fp16 scorer_class_prior_table
      (decoder_side_information — fixed table consumed by WZ head at inflate)
    - ``meta_blob`` — sorted-keys JSON with atw_codec_meta provenance tag

    Cheaper than ``parse_archive`` (no torch state_dict deserialization);
    safe against brotli-tampered blobs.

    Raises ``ValueError`` on:

    - short header (< 35 bytes)
    - bad magic (!= ``b"ATW1"``)
    - unsupported schema version (!= 1)
    - latent_len != num_pairs * latent_dim
    - class_prior_table_len != num_pairs * prior_dim * 2
    - archive size mismatch (declared end_meta != len(archive_bytes))
    """
    if len(archive_bytes) < ATW1_HEADER_SIZE:
        raise ValueError(
            f"atw1 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {ATW1_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        prior_dim,
        encoder_len,
        decoder_len,
        wz_head_len,
        latent_len,
        class_prior_table_len,
        meta_len,
    ) = struct.unpack(ATW1_HEADER_FMT, archive_bytes[:ATW1_HEADER_SIZE])
    if magic != ATW1_MAGIC:
        raise ValueError(
            f"atw1 archive: bad magic {magic!r} (expected {ATW1_MAGIC!r})"
        )
    if version != ATW1_SCHEMA_VERSION:
        raise ValueError(
            f"atw1 archive: unsupported schema version {version} "
            f"(expected {ATW1_SCHEMA_VERSION})"
        )
    expected_latent_bytes = int(num_pairs) * int(latent_dim)
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"atw1 archive: latent_len {latent_len} != num_pairs*latent_dim "
            f"= {expected_latent_bytes}"
        )
    expected_class_prior_table_bytes = int(num_pairs) * int(prior_dim) * 2
    if class_prior_table_len != expected_class_prior_table_bytes:
        raise ValueError(
            f"atw1 archive: class_prior_table_len {class_prior_table_len} != "
            f"num_pairs*prior_dim*2 = {expected_class_prior_table_bytes}"
        )
    end_header = ATW1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_wz_head = end_decoder + int(wz_head_len)
    end_latents = end_wz_head + int(latent_len)
    end_class_prior_table = end_latents + int(class_prior_table_len)
    end_meta = end_class_prior_table + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"atw1 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    return {
        "atw1_header": (0, ATW1_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "wz_head_blob": (end_decoder, int(wz_head_len)),
        "latent_residual_blob": (end_wz_head, int(latent_len)),
        "class_prior_table_blob": (end_latents, int(class_prior_table_len)),
        "meta_blob": (end_class_prior_table, int(meta_len)),
    }


# Canonical optimization-role mapping for ATW1 sections (consumed by
# tac.analysis.scorer_conditional_mdl and tac.analysis.hnerv_packet_sections).
ATW1_SECTION_ROLES: dict[str, str] = {
    "atw1_header": "control_or_metadata",
    "encoder_blob": "training_provenance_only",
    "decoder_blob": "decoder_weight_stream",
    "wz_head_blob": "decoder_weight_stream",
    "latent_residual_blob": "latent_stream",
    "class_prior_table_blob": "decoder_side_information",
    "meta_blob": "control_or_metadata",
}


__all__ = [
    "ATW1_HEADER_FMT",
    "ATW1_HEADER_SIZE",
    "ATW1_MAGIC",
    "ATW1_SCHEMA_VERSION",
    "ATW1_SECTION_ROLES",
    "ATWCodecArchive",
    "ATWCodecArchiveNumpy",
    "pack_archive",
    "parse_archive",
    "parse_archive_numpy",
    "parse_atw1_archive_bytes",
]
