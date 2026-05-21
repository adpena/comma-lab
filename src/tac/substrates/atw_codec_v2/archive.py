# SPDX-License-Identifier: MIT
"""ATW codec V2 archive grammar — ATW2 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (<=200 LOC inflate substrate-engineering waiver) + L8 (deterministic).
Per V2 design memo §10.

ATW2 is BYTEWISE-DISTINCT from V1 ATW1 + sister Z3HP1 / Z4CR1 / IBPS1 grammars:

* Different magic ``b"ATW2"`` (4 bytes) — so ``parse_atw2_archive_bytes``
  refuses non-ATW2 archives at byte 0.
* Adds TWO new sections beyond V1 ATW1's 7 sections:
  ``distill_head_blob`` (the G1 5-way scorer-class distill head fp16) and
  ``cdf_table_blob`` (the B3 scorer-conditional CDF table fp16).
* Adds a 1-byte ``variant`` field at offset 5 (0 = Variant A three-knob;
  1 = Variant B WZ-only). The variant value is also carried in meta_blob
  ``atw_v2_codec_meta`` so audit tools can recover the regime from byte 5
  OR from JSON.
* The ``latent_residual_blob`` ships ``z_residual`` rather than ``z``
  (Wyner-Ziv compression mechanism inherited from V1).

Grammar:

::

    MAGIC(4)                    b"ATW2"
    VERSION(1)                  u8       schema version (currently 1)
    VARIANT(1)                  u8       0 = Variant A (three-knob), 1 = Variant B (WZ-only)
    LATENT_DIM(2)               u16      cfg.latent_dim (e.g. 24)
    NUM_PAIRS(2)                u16      cfg.num_pairs (e.g. 600)
    SCORER_CLASS_PRIOR_DIM(2)   u16      cfg.scorer_class_prior_dim (e.g. 16)
    CDF_NUM_CLASSES(2)          u16      5 (upstream SegNet classes)
    CDF_NUM_SYMBOLS(2)          u16      256 (int8 latent symbols)
    ENCODER_BLOB_LEN(4)         u32      brotli(fp16 encoder state)
    DECODER_BLOB_LEN(4)         u32      brotli(fp16 decoder state)
    WZ_HEAD_BLOB_LEN(4)         u32      brotli(fp16 wz_side_info_head state)
    DISTILL_HEAD_BLOB_LEN(4)    u32      brotli(fp16 g1_distill_head state)   <-- NEW V2
    LATENT_RESIDUAL_BLOB_LEN(4) u32      int8 z_residual = num_pairs * latent_dim
    CLASS_PRIOR_TABLE_BLOB_LEN(4) u32   fp16 = num_pairs * scorer_class_prior_dim * 2
    CDF_TABLE_BLOB_LEN(4)       u32      fp16 = cdf_classes * cdf_symbols * 2  <-- NEW V2
    META_BLOB_LEN(4)            u32      sorted-keys JSON utf-8

Round-trip contract:

    bytes -> parse_archive -> (encoder_sd, decoder_sd, wz_head_sd, distill_head_sd,
                              latents, class_prior_table, cdf_table, meta)
    (encoder_sd, decoder_sd, wz_head_sd, distill_head_sd, latents,
     class_prior_table, cdf_table, meta) -> pack_archive -> bytes

CLAUDE.md compliance:

* No silent defaults (caller passes config + variant explicitly)
* No /tmp paths
* No scorer load at inflate time (scorer class prior + CDF table PRECOMPUTED
  at compress time per Catalog #6 strict-scorer-rule)
* Deterministic: same input -> same bytes (sorted-keys JSON; fixed brotli
  quality; fp16 state_dict cast on CPU; deterministic int8 latent quantization
  per Catalog #161 degenerate-range fix)
* Inherits the Catalog #161 degenerate-range fix from V1 + Z4 + sane_hnerv
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

ATW2_MAGIC: bytes = b"ATW2"
"""ATW codec V2 archive magic."""

ATW2_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
# MAGIC(4) + VERSION(1) + VARIANT(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#   + SCORER_CLASS_PRIOR_DIM(2) + CDF_NUM_CLASSES(2) + CDF_NUM_SYMBOLS(2)
#   + ENCODER_LEN(4) + DECODER_LEN(4) + WZ_HEAD_LEN(4)
#   + DISTILL_HEAD_LEN(4) + LATENT_RESIDUAL_LEN(4)
#   + CLASS_PRIOR_TABLE_LEN(4) + CDF_TABLE_LEN(4) + META_LEN(4)
# = 4+1+1+2+2+2+2+2+4+4+4+4+4+4+4+4 = 48 bytes
ATW2_HEADER_FMT: str = "<4sBBHHHHHIIIIIIII"
ATW2_HEADER_SIZE: int = struct.calcsize(ATW2_HEADER_FMT)
assert ATW2_HEADER_SIZE == 48, f"header size invariant: expected 48, got {ATW2_HEADER_SIZE}"

ATW2_CDF_DEAD_SECTION_SENTINEL: bytes = b"ACD0\x01\x00\x00\x00"
"""Typed compact sentinel for the current-runtime dead ``cdf_table_blob``."""

# Deterministic brotli quality (matches sister substrates).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class ATWv2CodecArchive:
    """Parsed ATW2 archive — the inflate-time data contract."""

    encoder_state_dict: dict[str, torch.Tensor]
    """Encoder state_dict (provenance; not strictly required at inflate)."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (the inflate-time consumer)."""

    wz_side_info_head_state_dict: dict[str, torch.Tensor]
    """WZ side-info head state_dict (inflate-time consumer for z reconstruction)."""

    distill_head_state_dict: dict[str, torch.Tensor]
    """G1 scorer-class distill head state_dict (inflate consumer for B3 gating)."""

    latent_residual: torch.Tensor
    """``(num_pairs, latent_dim)`` float32 dequantized z_residual."""

    scorer_class_prior_table: torch.Tensor
    """``(num_pairs, scorer_class_prior_dim)`` fp32 scorer class prior table."""

    cdf_table: torch.Tensor
    """``(num_classes, num_symbols)`` fp32 scorer-conditional CDF table (B3)."""

    meta: dict[str, object]
    """Sidecar JSON meta with atw_v2_codec_meta provenance fields."""

    schema_version: int
    """Archive schema version byte."""

    variant: int
    """Variant byte: 0=Variant A (three-knob), 1=Variant B (WZ-only)."""


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict deterministically as length-prefixed records.

    Format mirrors V1 ATW1 + sister Z4 deterministic serializer — no pickle
    (memo IDs vary across instantiations); fp16 cast on CPU; sorted by key.
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


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    if not blob:
        return {}
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    import numpy as np

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


def _is_compact_cdf_dead_section_blob(blob: bytes) -> bool:
    return blob == ATW2_CDF_DEAD_SECTION_SENTINEL


def _uniform_cdf_table(cdf_classes: int, cdf_symbols: int) -> torch.Tensor:
    if cdf_classes <= 0 or cdf_symbols <= 0:
        raise ValueError(
            f"cdf_classes and cdf_symbols must be positive; got "
            f"{cdf_classes}, {cdf_symbols}"
        )
    return torch.full(
        (int(cdf_classes), int(cdf_symbols)),
        1.0 / float(cdf_symbols),
        dtype=torch.float32,
    )


def _quantize_latents_to_int8(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize latents to int8 with the canonical degenerate-range fix.

    Per Catalog #161: if hi <= lo (all-equal), q is filled with -127 so
    dequant(-127) = 0 * scale + lo = lo. Mirrors V1 ATW1 + Z4 + sane_hnerv exactly.
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


def _make_atw_v2_codec_meta_block(
    base_meta: dict[str, object],
    *,
    variant: int,
    atw_kappa_ib: float,
    atw_lambda_wz: float,
    atw_lambda_pixel: float,
    wz_head_enabled: bool,
    g1_distill_enabled: bool,
    b3_cdf_enabled: bool,
) -> dict[str, object]:
    """Inject the ATW V2 codec provenance tag into the meta dict."""
    meta = dict(base_meta)
    meta["atw_v2_codec_meta"] = {
        "variant": int(variant),
        "kappa_ib": float(atw_kappa_ib),
        "lambda_wz": float(atw_lambda_wz),
        "lambda_pixel": float(atw_lambda_pixel),
        "wz_head_enabled": bool(wz_head_enabled),
        "g1_distill_enabled": bool(g1_distill_enabled),
        "b3_cdf_enabled": bool(b3_cdf_enabled),
        "literature_anchor": [
            "Atick-Redlich1990",
            "Tishby-Pereira-Bialek1999",
            "Wyner-Ziv1976",
        ],
        # Per Catalog #287 + #294 forbidden empirical-claim-without-evidence-tag
        # and predicted-band Dykstra-feasibility check: predicted band is NULL
        # pending D4 probe verdict + Dykstra-feasibility check per design memo §18.
        "predicted_band_status": "NULL_pending_d4_probe_verdict_plus_dykstra_feasibility",
        "composite_id": "atw_codec_v2",
    }
    return meta


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    wz_side_info_head_state_dict: dict[str, torch.Tensor],
    distill_head_state_dict: dict[str, torch.Tensor],
    latent_residual: torch.Tensor,
    scorer_class_prior_table: torch.Tensor,
    cdf_table: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = ATW2_SCHEMA_VERSION,
    variant: int = 1,
    atw_kappa_ib: float = 0.0,
    atw_lambda_wz: float = 1.0,
    atw_lambda_pixel: float = 0.0,
    wz_head_enabled: bool = True,
    g1_distill_enabled: bool = True,
    b3_cdf_enabled: bool = True,
) -> bytes:
    """Serialize trained ATW V2 weights + WZ residual + G1 head + B3 CDF -> ATW2 0.bin bytes."""
    if schema_version != ATW2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if variant not in (0, 1):
        raise ValueError(f"variant must be 0 (A) or 1 (B); got {variant}")
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
    if cdf_table.dim() != 2:
        raise ValueError(
            f"cdf_table must be 2-D (num_classes, num_symbols); "
            f"got {tuple(cdf_table.shape)}"
        )
    num_pairs, latent_dim = int(latent_residual.shape[0]), int(latent_residual.shape[1])
    table_pairs, prior_dim = (
        int(scorer_class_prior_table.shape[0]),
        int(scorer_class_prior_table.shape[1]),
    )
    cdf_classes, cdf_symbols = int(cdf_table.shape[0]), int(cdf_table.shape[1])
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
    if cdf_classes <= 0 or cdf_classes > 0xFFFF:
        raise ValueError(f"cdf_classes {cdf_classes} out of u16 range")
    if cdf_symbols <= 0 or cdf_symbols > 0xFFFF:
        raise ValueError(f"cdf_symbols {cdf_symbols} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int8(latent_residual)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)
    wz_head_blob = _serialize_state_dict(wz_side_info_head_state_dict)
    distill_head_blob = _serialize_state_dict(distill_head_state_dict)

    table_fp16 = (
        scorer_class_prior_table.detach().to("cpu", dtype=torch.float16).contiguous()
    )
    class_prior_table_bytes = table_fp16.numpy().tobytes(order="C")

    cdf_table_fp16 = cdf_table.detach().to("cpu", dtype=torch.float16).contiguous()
    cdf_table_bytes = cdf_table_fp16.numpy().tobytes(order="C")

    meta_with_tag = _make_atw_v2_codec_meta_block(
        meta,
        variant=variant,
        atw_kappa_ib=atw_kappa_ib,
        atw_lambda_wz=atw_lambda_wz,
        atw_lambda_pixel=atw_lambda_pixel,
        wz_head_enabled=wz_head_enabled,
        g1_distill_enabled=g1_distill_enabled,
        b3_cdf_enabled=b3_cdf_enabled,
    )
    meta_with_tag["_lat_scale"] = float(scale)
    meta_with_tag["_lat_zero_point"] = float(zero_point)
    meta_with_tag["_scorer_class_prior_dim"] = int(prior_dim)
    meta_with_tag["_cdf_num_classes"] = int(cdf_classes)
    meta_with_tag["_cdf_num_symbols"] = int(cdf_symbols)
    meta_bytes = json.dumps(
        meta_with_tag, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        ATW2_HEADER_FMT,
        ATW2_MAGIC,
        schema_version,
        variant,
        latent_dim,
        num_pairs,
        prior_dim,
        cdf_classes,
        cdf_symbols,
        len(encoder_blob),
        len(decoder_blob),
        len(wz_head_blob),
        len(distill_head_blob),
        len(latent_bytes),
        len(class_prior_table_bytes),
        len(cdf_table_bytes),
        len(meta_bytes),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + wz_head_blob
        + distill_head_blob
        + latent_bytes
        + class_prior_table_bytes
        + cdf_table_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> ATWv2CodecArchive:
    """Parse ATW2 0.bin bytes back into the full ATW V2 substrate state."""
    if len(blob) < ATW2_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {ATW2_HEADER_SIZE})"
        )
    (
        magic,
        version,
        variant,
        latent_dim,
        num_pairs,
        prior_dim,
        cdf_classes,
        cdf_symbols,
        encoder_len,
        decoder_len,
        wz_head_len,
        distill_head_len,
        latent_len,
        class_prior_table_len,
        cdf_table_len,
        meta_len,
    ) = struct.unpack(ATW2_HEADER_FMT, blob[:ATW2_HEADER_SIZE])
    if magic != ATW2_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {ATW2_MAGIC!r})")
    if version != ATW2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    if variant not in (0, 1):
        raise ValueError(f"bad variant byte: {variant} (expected 0 or 1)")

    expected_latent_bytes = num_pairs * latent_dim  # int8 = 1 byte
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_residual blob: latent_len {latent_len} != "
            f"num_pairs*latent_dim = {expected_latent_bytes}"
        )
    expected_class_prior_table_bytes = num_pairs * prior_dim * 2  # fp16
    if class_prior_table_len != expected_class_prior_table_bytes:
        raise ValueError(
            f"class_prior_table blob: class_prior_table_len "
            f"{class_prior_table_len} != num_pairs*prior_dim*2 = "
            f"{expected_class_prior_table_bytes}"
        )
    expected_cdf_table_bytes = cdf_classes * cdf_symbols * 2  # fp16
    compact_cdf_bytes = len(ATW2_CDF_DEAD_SECTION_SENTINEL)
    if cdf_table_len not in (expected_cdf_table_bytes, compact_cdf_bytes):
        raise ValueError(
            f"cdf_table blob: cdf_table_len {cdf_table_len} != "
            f"cdf_classes*cdf_symbols*2 = {expected_cdf_table_bytes} "
            f"or compact sentinel bytes = {compact_cdf_bytes}"
        )

    end_header = ATW2_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_wz_head = end_decoder + int(wz_head_len)
    end_distill_head = end_wz_head + int(distill_head_len)
    end_latents = end_distill_head + int(latent_len)
    end_class_prior_table = end_latents + int(class_prior_table_len)
    end_cdf_table = end_class_prior_table + int(cdf_table_len)
    end_meta = end_cdf_table + int(meta_len)
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_blob = blob[end_header:end_encoder]
    decoder_blob = blob[end_encoder:end_decoder]
    wz_head_blob = blob[end_decoder:end_wz_head]
    distill_head_blob = blob[end_wz_head:end_distill_head]
    latent_blob = blob[end_distill_head:end_latents]
    class_prior_table_blob = blob[end_latents:end_class_prior_table]
    cdf_table_blob = blob[end_class_prior_table:end_cdf_table]
    meta_blob = blob[end_cdf_table:end_meta]

    encoder_sd = _deserialize_state_dict(encoder_blob)
    decoder_sd = _deserialize_state_dict(decoder_blob)
    wz_head_sd = _deserialize_state_dict(wz_head_blob)
    distill_head_sd = _deserialize_state_dict(distill_head_blob)
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

    if cdf_table_len == expected_cdf_table_bytes:
        cdf_table = torch.from_numpy(
            np.frombuffer(cdf_table_blob, dtype=np.float16).copy()
        ).view(cdf_classes, cdf_symbols).to(torch.float32)
    elif _is_compact_cdf_dead_section_blob(cdf_table_blob):
        cdf_table = _uniform_cdf_table(cdf_classes, cdf_symbols)
    else:
        raise ValueError("cdf_table blob has compact length but bad sentinel")

    return ATWv2CodecArchive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        wz_side_info_head_state_dict=wz_head_sd,
        distill_head_state_dict=distill_head_sd,
        latent_residual=latents,
        scorer_class_prior_table=class_prior_table,
        cdf_table=cdf_table,
        meta=meta,
        schema_version=int(version),
        variant=int(variant),
    )


def parse_atw2_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for ATW2 grammar.

    Canonical section-offset parser for ATW2 inner-blob bytes. Cheaper than
    ``parse_archive`` (no torch state_dict deserialization); safe against
    brotli-tampered blobs.

    Returned sections (Tier A / Tier B targets):

    * ``atw2_header`` — 48-byte header (control_or_metadata)
    * ``encoder_blob`` — brotli q=9 encoder state_dict (training_provenance_only)
    * ``decoder_blob`` — brotli q=9 decoder state_dict (decoder_weight_stream)
    * ``wz_head_blob`` — brotli q=9 wz_side_info_head (decoder_weight_stream)
    * ``distill_head_blob`` — brotli q=9 G1 head (decoder_side_information)
    * ``latent_residual_blob`` — int8 z_residual (latent_stream)
    * ``class_prior_table_blob`` — fp16 scorer_class_prior_table
      (decoder_side_information)
    * ``cdf_table_blob`` — fp16 B3 scorer-conditional CDF
      (decoder_side_information)
    * ``meta_blob`` — sorted-keys JSON with atw_v2_codec_meta provenance tag

    Raises ``ValueError`` on bad magic / version / variant / size mismatch.
    """
    if len(archive_bytes) < ATW2_HEADER_SIZE:
        raise ValueError(
            f"atw2 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {ATW2_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        variant,
        latent_dim,
        num_pairs,
        prior_dim,
        cdf_classes,
        cdf_symbols,
        encoder_len,
        decoder_len,
        wz_head_len,
        distill_head_len,
        latent_len,
        class_prior_table_len,
        cdf_table_len,
        meta_len,
    ) = struct.unpack(ATW2_HEADER_FMT, archive_bytes[:ATW2_HEADER_SIZE])
    if magic != ATW2_MAGIC:
        raise ValueError(
            f"atw2 archive: bad magic {magic!r} (expected {ATW2_MAGIC!r})"
        )
    if version != ATW2_SCHEMA_VERSION:
        raise ValueError(
            f"atw2 archive: unsupported schema version {version} "
            f"(expected {ATW2_SCHEMA_VERSION})"
        )
    if variant not in (0, 1):
        raise ValueError(
            f"atw2 archive: bad variant byte {variant} (expected 0 or 1)"
        )
    expected_latent_bytes = int(num_pairs) * int(latent_dim)
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"atw2 archive: latent_len {latent_len} != num_pairs*latent_dim "
            f"= {expected_latent_bytes}"
        )
    expected_class_prior_table_bytes = int(num_pairs) * int(prior_dim) * 2
    if class_prior_table_len != expected_class_prior_table_bytes:
        raise ValueError(
            f"atw2 archive: class_prior_table_len {class_prior_table_len} != "
            f"num_pairs*prior_dim*2 = {expected_class_prior_table_bytes}"
        )
    expected_cdf_table_bytes = int(cdf_classes) * int(cdf_symbols) * 2
    compact_cdf_bytes = len(ATW2_CDF_DEAD_SECTION_SENTINEL)
    if cdf_table_len not in (expected_cdf_table_bytes, compact_cdf_bytes):
        raise ValueError(
            f"atw2 archive: cdf_table_len {cdf_table_len} != "
            f"cdf_classes*cdf_symbols*2 = {expected_cdf_table_bytes} "
            f"or compact sentinel bytes = {compact_cdf_bytes}"
        )
    end_header = ATW2_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_wz_head = end_decoder + int(wz_head_len)
    end_distill_head = end_wz_head + int(distill_head_len)
    end_latents = end_distill_head + int(latent_len)
    end_class_prior_table = end_latents + int(class_prior_table_len)
    end_cdf_table = end_class_prior_table + int(cdf_table_len)
    end_meta = end_cdf_table + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"atw2 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    if cdf_table_len == compact_cdf_bytes and cdf_table_len != expected_cdf_table_bytes:
        cdf_blob = archive_bytes[end_class_prior_table:end_cdf_table]
        if not _is_compact_cdf_dead_section_blob(cdf_blob):
            raise ValueError("atw2 archive: compact cdf_table_blob has bad sentinel")
    return {
        "atw2_header": (0, ATW2_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "wz_head_blob": (end_decoder, int(wz_head_len)),
        "distill_head_blob": (end_wz_head, int(distill_head_len)),
        "latent_residual_blob": (end_distill_head, int(latent_len)),
        "class_prior_table_blob": (end_latents, int(class_prior_table_len)),
        "cdf_table_blob": (end_class_prior_table, int(cdf_table_len)),
        "meta_blob": (end_cdf_table, int(meta_len)),
    }


# Canonical optimization-role mapping for ATW2 sections (consumed by
# tac.analysis.scorer_conditional_mdl + tac.analysis.hnerv_packet_sections).
ATW2_SECTION_ROLES: dict[str, str] = {
    "atw2_header": "control_or_metadata",
    "encoder_blob": "training_provenance_only",
    "decoder_blob": "decoder_weight_stream",
    "wz_head_blob": "decoder_weight_stream",
    "distill_head_blob": "decoder_weight_stream",
    "latent_residual_blob": "latent_stream",
    "class_prior_table_blob": "decoder_side_information",
    "cdf_table_blob": "decoder_side_information",
    "meta_blob": "control_or_metadata",
}


__all__ = [
    "ATW2_HEADER_FMT",
    "ATW2_HEADER_SIZE",
    "ATW2_CDF_DEAD_SECTION_SENTINEL",
    "ATW2_MAGIC",
    "ATW2_SCHEMA_VERSION",
    "ATW2_SECTION_ROLES",
    "ATWv2CodecArchive",
    "pack_archive",
    "parse_archive",
    "parse_atw2_archive_bytes",
]
