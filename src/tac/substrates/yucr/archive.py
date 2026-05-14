"""YUCR archive grammar — YUCR1 monolithic 0.bin sidecar.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (single-zip-member when packaged).
* L4 inflate.py <= 200 LOC substrate-engineering waiver.
* L8 deterministic (sorted-keys JSON, fixed brotli quality, no timestamps).

The 4-section grammar:

::

    MAGIC(4)                     b"YUCR"
    VERSION(1)                   u8 (== 1)
    HEIGHT(2)                    u16   cost map height
    WIDTH(2)                     u16   cost map width
    COST_MAP_SCALE(4)            f32   recovered_scale for int8 dequant
    BASE_SUBSTRATE_LEN(1)        u8    e.g. 2 ('a1')
    BASE_SHA256_TRUNC_LEN(1)     u8    16 (sha256 truncated to 8 bytes hex)
    COST_MAP_LEN(4)              u32   brotli-compressed int8 cost map
    STC_PAYLOAD_LEN(4)           u32   pre-encoded by encode_stc_payload
    META_LEN(4)                  u32   utf-8 sorted-keys JSON
    BASE_SUBSTRATE_BLOB          ...   utf-8 base substrate id
    BASE_SHA256_BLOB             ...   utf-8 truncated sha256 hex
    COST_MAP_BLOB                ...   brotli(int8 cost map)
    STC_PAYLOAD_BLOB             ...   pre-encoded
    META_BLOB                    ...   utf-8 JSON

NO scorer load. NO score claim. NO /tmp paths. Deterministic across runs.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Mapping

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.substrates.yucr.cost_map import (
    dequantize_cost_map_int8,
    quantize_cost_map_int8,
)

YUCR1_MAGIC: bytes = b"YUCR"
"""YUCR archive magic (4 bytes; distinct from WZ1/TT5L/SBO1/CMLR/etc.)."""

YUCR1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header: MAGIC(4) + VERSION(1) + H(2) + W(2) + COST_SCALE(4)
#       + BASE_LEN(1) + SHA_LEN(1) + COST_LEN(4) + STC_LEN(4) + META_LEN(4)
# = 4+1+2+2+4+1+1+4+4+4 = 27 bytes
YUCR1_HEADER_FMT: str = "<4sBHHfBBIII"
YUCR1_HEADER_SIZE: int = struct.calcsize(YUCR1_HEADER_FMT)
assert YUCR1_HEADER_SIZE == 27, (
    f"YUCR1 header size invariant: expected 27, got {YUCR1_HEADER_SIZE}"
)

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class YUCRArchive:
    """Parsed YUCR1 archive — the inflate-time data contract."""

    cost_map_int8: np.ndarray
    """Int8 cost map shape ``(H, W)``. Dequant via ``cost_map_scale``."""

    cost_map_scale: float
    """Recovered scale for int8 dequantization (encoder-side max / 127)."""

    stc_payload: bytes
    """Pre-encoded STC payload (decode via :func:`tac.substrates.yucr.stc_encoder.decode_stc_payload`)."""

    base_substrate_id: str
    """Base substrate identifier (e.g. ``"a1"``)."""

    base_archive_sha256_truncated: str
    """Truncated sha256 hex (16 chars = 8 bytes) of the base archive."""

    meta: dict[str, object]
    """Sorted-keys JSON metadata dict."""

    schema_version: int
    height: int
    width: int

    def cost_map_float(self) -> np.ndarray:
        """Dequantize the int8 cost map back to float32."""
        return dequantize_cost_map_int8(
            torch.from_numpy(self.cost_map_int8),
            recovered_scale=self.cost_map_scale,
        ).numpy()


def pack_archive(
    *,
    cost_map: torch.Tensor,
    stc_payload: bytes,
    base_substrate_id: str,
    base_archive_sha256: str,
    base_archive_bytes: int,
    config,  # YUCRConfig (avoid circular import)
    extra_meta: Mapping[str, object],
    schema_version: int = YUCR1_SCHEMA_VERSION,
) -> bytes:
    """Pack a YUCR sidecar 0.bin from cost map + STC payload + metadata.

    Args:
        cost_map: ``(H, W)`` float tensor (will be int8-quantized inside).
        stc_payload: Bytes from :func:`tac.substrates.yucr.stc_encoder.encode_stc_payload`.
        base_substrate_id: Composable base identifier (must be in
            :data:`tac.substrates.yucr.architecture.YUCR_BASE_SUBSTRATE_IDS`).
        base_archive_sha256: Full 64-char hex sha256 of the base archive bytes.
        base_archive_bytes: Size of the base archive (recorded in meta).
        config: :class:`YUCRConfig` (passed for cost-map-int8-scale + recipe trace).
        extra_meta: Sidecar dict (anchor sha, predicted ΔS, trainer hash, ...).
        schema_version: Schema version (default :data:`YUCR1_SCHEMA_VERSION`).

    Returns:
        Packed YUCR1 0.bin bytes.

    Raises:
        ValueError: when inputs violate the grammar invariants.
    """
    if schema_version != YUCR1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if cost_map.dim() != 2:
        raise ValueError(
            f"pack_archive expects 2D cost_map (H, W); got {tuple(cost_map.shape)}"
        )
    h, w = cost_map.shape
    if h <= 0 or h > 0xFFFF:
        raise ValueError(f"cost_map height={h} out of range [1, 0xFFFF]")
    if w <= 0 or w > 0xFFFF:
        raise ValueError(f"cost_map width={w} out of range [1, 0xFFFF]")

    base_id_bytes = base_substrate_id.encode("utf-8")
    if len(base_id_bytes) == 0 or len(base_id_bytes) > 0xFF:
        raise ValueError(
            f"base_substrate_id length out of range [1, 0xFF]; "
            f"got {len(base_id_bytes)}"
        )
    if len(base_archive_sha256) != 64:
        raise ValueError(
            f"base_archive_sha256 must be 64-char hex; got len={len(base_archive_sha256)}"
        )
    base_sha_truncated = base_archive_sha256[:16]  # 8 bytes hex
    base_sha_bytes = base_sha_truncated.encode("utf-8")
    if len(base_sha_bytes) != 16:
        raise ValueError("truncated base sha must be exactly 16 hex chars")

    int8_cm, scale = quantize_cost_map_int8(
        cost_map, scale=float(config.cost_map_int8_scale)
    )
    cost_map_blob = brotli.compress(
        int8_cm.cpu().numpy().tobytes(order="C"),
        quality=_BROTLI_QUALITY,
    )

    # Meta JSON — sorted keys, deterministic separators.
    meta_payload: dict[str, object] = {
        "yucr_schema_version": schema_version,
        "base_substrate_id": base_substrate_id,
        "base_archive_sha256_full": base_archive_sha256,
        "base_archive_bytes": int(base_archive_bytes),
        "cost_map_resolution": [int(h), int(w)],
        "cost_map_int8_scale": float(config.cost_map_int8_scale),
        "stc_payload_bits": int(config.stc_payload_bits),
        "l_inf_noise_cap": float(config.l_inf_noise_cap),
        "pose_sqrt_weight": float(config.pose_sqrt_weight),
        "seg_weight": float(config.seg_weight),
        "cost_map_mode": config.cost_map_mode,
        "score_claim": False,
        "evidence_grade": "proxy",
        "ready_for_exact_eval_dispatch": False,
    }
    for k, v in extra_meta.items():
        if k in meta_payload:
            raise ValueError(
                f"extra_meta key {k!r} collides with reserved YUCR meta key"
            )
        meta_payload[k] = v
    meta_bytes = json.dumps(
        meta_payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    if len(stc_payload) > 0xFFFFFFFF:
        raise ValueError(f"stc_payload size {len(stc_payload)} exceeds u32")
    if len(cost_map_blob) > 0xFFFFFFFF:
        raise ValueError(f"cost_map_blob size {len(cost_map_blob)} exceeds u32")
    if len(meta_bytes) > 0xFFFFFFFF:
        raise ValueError(f"meta_bytes size {len(meta_bytes)} exceeds u32")

    header = struct.pack(
        YUCR1_HEADER_FMT,
        YUCR1_MAGIC,
        schema_version,
        h,
        w,
        scale,
        len(base_id_bytes),
        len(base_sha_bytes),
        len(cost_map_blob),
        len(stc_payload),
        len(meta_bytes),
    )

    return (
        header
        + base_id_bytes
        + base_sha_bytes
        + cost_map_blob
        + stc_payload
        + meta_bytes
    )


def parse_archive(blob: bytes) -> YUCRArchive:
    """Parse YUCR1 0.bin bytes back to a typed :class:`YUCRArchive`."""
    if len(blob) < YUCR1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {YUCR1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        h,
        w,
        scale,
        base_id_len,
        sha_len,
        cost_map_len,
        stc_len,
        meta_len,
    ) = struct.unpack(YUCR1_HEADER_FMT, blob[:YUCR1_HEADER_SIZE])

    if magic != YUCR1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {YUCR1_MAGIC!r})")
    if version != YUCR1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    if sha_len != 16:
        raise ValueError(f"sha truncated len={sha_len}; YUCR1 expects 16")
    if scale <= 0:
        raise ValueError(
            f"cost_map_scale={scale} must be > 0 (encoder-side bug; refuse archive)"
        )

    pos = YUCR1_HEADER_SIZE
    base_id_blob = blob[pos : pos + base_id_len]
    pos += base_id_len
    base_sha_blob = blob[pos : pos + sha_len]
    pos += sha_len
    cost_map_blob = blob[pos : pos + cost_map_len]
    pos += cost_map_len
    stc_payload = blob[pos : pos + stc_len]
    pos += stc_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    base_substrate_id = base_id_blob.decode("utf-8")
    base_archive_sha256_truncated = base_sha_blob.decode("utf-8")

    raw_int8 = brotli.decompress(cost_map_blob)
    expected = h * w
    if len(raw_int8) != expected:
        raise ValueError(
            f"cost_map int8 size mismatch: got {len(raw_int8)}, expected H*W={expected}"
        )
    cost_map_int8 = np.frombuffer(raw_int8, dtype=np.int8).reshape(h, w).copy()

    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}
    if not isinstance(meta, dict):
        raise ValueError(
            f"meta JSON must be an object; got {type(meta).__name__}"
        )

    return YUCRArchive(
        cost_map_int8=cost_map_int8,
        cost_map_scale=float(scale),
        stc_payload=stc_payload,
        base_substrate_id=base_substrate_id,
        base_archive_sha256_truncated=base_archive_sha256_truncated,
        meta=meta,
        schema_version=int(version),
        height=int(h),
        width=int(w),
    )


def build_readiness_manifest(
    *,
    base_substrate_id: str,
    base_archive_bytes: int,
    yucr_overhead_bytes: int,
    config,
    predicted_score_band: tuple[float, float] = (0.153, 0.173),
) -> dict[str, object]:
    """Build a non-promotable readiness manifest for autopilot consumption.

    Per CLAUDE.md "Apples-to-apples evidence discipline": every score
    prediction is tagged ``[time-traveler-prediction]`` and carries
    ``score_claim=False``, ``ready_for_exact_eval_dispatch=False``,
    ``promotion_eligible=False`` until both ``[contest-CUDA]`` and
    ``[contest-CPU]`` paired auth eval lands on 1:1 contest-CI hardware.
    """
    return {
        "yucr_schema_version": YUCR1_SCHEMA_VERSION,
        "base_substrate_id": base_substrate_id,
        "base_archive_bytes": int(base_archive_bytes),
        "yucr_overhead_bytes": int(yucr_overhead_bytes),
        "total_archive_bytes": int(base_archive_bytes + yucr_overhead_bytes),
        "predicted_score_band_low": float(predicted_score_band[0]),
        "predicted_score_band_high": float(predicted_score_band[1]),
        "predicted_score_evidence_grade": "time-traveler-prediction",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "stc_payload_bits": int(config.stc_payload_bits),
        "cost_map_resolution": list(config.cost_map_resolution),
        "cost_map_mode": str(config.cost_map_mode),
        "l_inf_noise_cap": float(config.l_inf_noise_cap),
    }


__all__ = [
    "YUCR1_HEADER_FMT",
    "YUCR1_HEADER_SIZE",
    "YUCR1_MAGIC",
    "YUCR1_SCHEMA_VERSION",
    "YUCRArchive",
    "build_readiness_manifest",
    "pack_archive",
    "parse_archive",
]
