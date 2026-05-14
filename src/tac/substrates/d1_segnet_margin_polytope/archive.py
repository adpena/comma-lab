"""D1 archive grammar — D1POLY1 monolithic 0.bin sidecar.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (single-zip-member when packaged).
* L4 inflate.py <= 200 LOC substrate-engineering waiver.
* L8 deterministic (sorted-keys JSON, fixed brotli quality, no timestamps).

The 5-section grammar:

::

    MAGIC(4)                     b"D1PY"
    VERSION(1)                   u8 (== 1)
    HEIGHT(2)                    u16   margin map height
    WIDTH(2)                     u16   margin map width
    MARGIN_MAP_SCALE(4)          f32   recovered_scale for int8 dequant
    JACOBIAN_LIPSCHITZ(4)        f32   ``L`` (SegNet Jacobian operator-norm bound)
    BASE_SUBSTRATE_LEN(1)        u8    e.g. 2 ('a1')
    BASE_SHA256_TRUNC_LEN(1)     u8    16 (sha256 truncated to 8 bytes hex)
    MARGIN_MAP_LEN(4)            u32   brotli-compressed int8 margin map
    POLYTOPE_PAYLOAD_LEN(4)      u32   pre-encoded by encode_polytope_payload
    META_LEN(4)                  u32   utf-8 sorted-keys JSON
    BASE_SUBSTRATE_BLOB          ...   utf-8 base substrate id
    BASE_SHA256_BLOB             ...   utf-8 truncated sha256 hex
    MARGIN_MAP_BLOB              ...   brotli(int8 margin map)
    POLYTOPE_PAYLOAD_BLOB        ...   pre-encoded
    META_BLOB                    ...   utf-8 JSON

Header total: 4+1+2+2+4+4+1+1+4+4+4 = 31 bytes. Distinct from YUCR
(``YUCR``, 27 bytes) so the magic-byte router can dispatch the right
parser at inflate time.

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

from tac.substrates.d1_segnet_margin_polytope.margin_map import (
    dequantize_margin_map_int8,
    quantize_margin_map_int8,
)

D1POLY1_MAGIC: bytes = b"D1PY"
"""D1 polytope archive magic (4 bytes; distinct from YUCR/WZ1/TT5L/SBO1/...)."""

D1POLY1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header: MAGIC(4) + VERSION(1) + H(2) + W(2) + MARGIN_SCALE(4)
#       + JACOBIAN_LIPSCHITZ(4) + BASE_LEN(1) + SHA_LEN(1)
#       + MARGIN_LEN(4) + POLYTOPE_LEN(4) + META_LEN(4)
# = 4+1+2+2+4+4+1+1+4+4+4 = 31 bytes
D1POLY1_HEADER_FMT: str = "<4sBHHffBBIII"
D1POLY1_HEADER_SIZE: int = struct.calcsize(D1POLY1_HEADER_FMT)
assert D1POLY1_HEADER_SIZE == 31, (
    f"D1POLY1 header size invariant: expected 31, got {D1POLY1_HEADER_SIZE}"
)

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class D1PolytopeArchive:
    """Parsed D1POLY1 archive — the inflate-time data contract."""

    margin_map_int8: np.ndarray
    """Int8 margin map shape ``(H, W)``. Dequant via ``margin_map_scale``."""

    margin_map_scale: float
    """Recovered scale for int8 dequantization (encoder-side max / 127)."""

    jacobian_lipschitz: float
    """SegNet Jacobian operator-norm bound ``L`` used at encode time.
    Receiver inverts ``B_safe = margin / L`` deterministically.
    """

    polytope_payload: bytes
    """Pre-encoded polytope payload (decode via
    :func:`tac.substrates.d1_segnet_margin_polytope.polytope_encoder.decode_polytope_payload`).
    """

    base_substrate_id: str
    """Base substrate identifier (e.g. ``"a1"``)."""

    base_archive_sha256_truncated: str
    """Truncated sha256 hex (16 chars = 8 bytes) of the base archive."""

    meta: dict[str, object]
    """Sorted-keys JSON metadata dict."""

    schema_version: int
    height: int
    width: int

    def margin_map_float(self) -> np.ndarray:
        """Dequantize the int8 margin map back to float32."""
        return dequantize_margin_map_int8(
            torch.from_numpy(self.margin_map_int8),
            recovered_scale=self.margin_map_scale,
        ).numpy()


def pack_archive(
    *,
    margin_map: torch.Tensor,
    polytope_payload: bytes,
    jacobian_lipschitz: float,
    base_substrate_id: str,
    base_archive_sha256: str,
    base_archive_bytes: int,
    config,  # D1PolytopeConfig (avoid circular import)
    extra_meta: Mapping[str, object],
    schema_version: int = D1POLY1_SCHEMA_VERSION,
) -> bytes:
    """Pack a D1 sidecar 0.bin from margin map + polytope payload + metadata.

    Args:
        margin_map: ``(H, W)`` float tensor (will be int8-quantized inside).
        polytope_payload: Bytes from
            :func:`tac.substrates.d1_segnet_margin_polytope.polytope_encoder.encode_polytope_payload`.
        jacobian_lipschitz: SegNet Jacobian operator-norm upper bound ``L``.
            Recorded both in the header AND inside the polytope payload so
            the receiver can verify consistency.
        base_substrate_id: Composable base identifier (must be in
            :data:`tac.substrates.d1_segnet_margin_polytope.architecture.D1POLY_BASE_SUBSTRATE_IDS`).
        base_archive_sha256: Full 64-char hex sha256 of the base archive
            bytes.
        base_archive_bytes: Size of the base archive (recorded in meta).
        config: :class:`D1PolytopeConfig`.
        extra_meta: Sidecar dict (anchor sha, predicted ΔS, trainer hash...).
        schema_version: Schema version (default
            :data:`D1POLY1_SCHEMA_VERSION`).

    Returns:
        Packed D1POLY1 0.bin bytes.

    Raises:
        ValueError: when inputs violate the grammar invariants.
    """
    if schema_version != D1POLY1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if margin_map.dim() != 2:
        raise ValueError(
            f"pack_archive expects 2D margin_map (H, W); got "
            f"{tuple(margin_map.shape)}"
        )
    h, w = margin_map.shape
    if h <= 0 or h > 0xFFFF:
        raise ValueError(f"margin_map height={h} out of range [1, 0xFFFF]")
    if w <= 0 or w > 0xFFFF:
        raise ValueError(f"margin_map width={w} out of range [1, 0xFFFF]")
    if jacobian_lipschitz <= 0:
        raise ValueError(
            f"jacobian_lipschitz={jacobian_lipschitz} must be > 0"
        )

    base_id_bytes = base_substrate_id.encode("utf-8")
    if len(base_id_bytes) == 0 or len(base_id_bytes) > 0xFF:
        raise ValueError(
            "base_substrate_id length out of range [1, 0xFF]; got "
            f"{len(base_id_bytes)}"
        )
    if len(base_archive_sha256) != 64:
        raise ValueError(
            f"base_archive_sha256 must be 64-char hex; got len="
            f"{len(base_archive_sha256)}"
        )
    base_sha_truncated = base_archive_sha256[:16]  # 8 bytes hex
    base_sha_bytes = base_sha_truncated.encode("utf-8")
    if len(base_sha_bytes) != 16:
        raise ValueError("truncated base sha must be exactly 16 hex chars")

    int8_mm, scale = quantize_margin_map_int8(
        margin_map, scale=float(config.margin_map_int8_scale)
    )
    margin_map_blob = brotli.compress(
        int8_mm.cpu().numpy().tobytes(order="C"),
        quality=_BROTLI_QUALITY,
    )

    # Meta JSON — sorted keys, deterministic separators.
    meta_payload: dict[str, object] = {
        "d1poly_schema_version": schema_version,
        "base_substrate_id": base_substrate_id,
        "base_archive_sha256_full": base_archive_sha256,
        "base_archive_bytes": int(base_archive_bytes),
        "margin_map_resolution": [int(h), int(w)],
        "margin_map_int8_scale": float(config.margin_map_int8_scale),
        "polytope_payload_bits": int(config.polytope_payload_bits),
        "jacobian_lipschitz": float(jacobian_lipschitz),
        "margin_threshold": float(config.margin_threshold),
        "pose_sqrt_weight": float(config.pose_sqrt_weight),
        "seg_weight": float(config.seg_weight),
        "margin_map_mode": config.margin_map_mode,
        "score_claim": False,
        "evidence_grade": "proxy",
        "ready_for_exact_eval_dispatch": False,
    }
    for k, v in extra_meta.items():
        if k in meta_payload:
            raise ValueError(
                f"extra_meta key {k!r} collides with reserved D1POLY meta key"
            )
        meta_payload[k] = v
    meta_bytes = json.dumps(
        meta_payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    if len(polytope_payload) > 0xFFFFFFFF:
        raise ValueError(
            f"polytope_payload size {len(polytope_payload)} exceeds u32"
        )
    if len(margin_map_blob) > 0xFFFFFFFF:
        raise ValueError(
            f"margin_map_blob size {len(margin_map_blob)} exceeds u32"
        )
    if len(meta_bytes) > 0xFFFFFFFF:
        raise ValueError(f"meta_bytes size {len(meta_bytes)} exceeds u32")

    header = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        schema_version,
        h,
        w,
        scale,
        float(jacobian_lipschitz),
        len(base_id_bytes),
        len(base_sha_bytes),
        len(margin_map_blob),
        len(polytope_payload),
        len(meta_bytes),
    )

    return (
        header
        + base_id_bytes
        + base_sha_bytes
        + margin_map_blob
        + polytope_payload
        + meta_bytes
    )


def parse_archive(blob: bytes) -> D1PolytopeArchive:
    """Parse D1POLY1 0.bin bytes back to a typed :class:`D1PolytopeArchive`."""
    if len(blob) < D1POLY1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= "
            f"{D1POLY1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        h,
        w,
        scale,
        jacobian_lipschitz,
        base_id_len,
        sha_len,
        margin_map_len,
        polytope_len,
        meta_len,
    ) = struct.unpack(D1POLY1_HEADER_FMT, blob[:D1POLY1_HEADER_SIZE])

    if magic != D1POLY1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {D1POLY1_MAGIC!r})")
    if version != D1POLY1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    if sha_len != 16:
        raise ValueError(f"sha truncated len={sha_len}; D1POLY1 expects 16")
    if scale <= 0:
        raise ValueError(
            f"margin_map_scale={scale} must be > 0 (encoder-side bug; "
            "refuse archive)"
        )
    if jacobian_lipschitz <= 0:
        raise ValueError(
            f"jacobian_lipschitz={jacobian_lipschitz} must be > 0 "
            "(encoder-side bug; refuse archive)"
        )

    pos = D1POLY1_HEADER_SIZE
    base_id_blob = blob[pos : pos + base_id_len]
    pos += base_id_len
    base_sha_blob = blob[pos : pos + sha_len]
    pos += sha_len
    margin_map_blob = blob[pos : pos + margin_map_len]
    pos += margin_map_len
    polytope_payload = blob[pos : pos + polytope_len]
    pos += polytope_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    base_substrate_id = base_id_blob.decode("utf-8")
    base_archive_sha256_truncated = base_sha_blob.decode("utf-8")

    raw_int8 = brotli.decompress(margin_map_blob)
    expected = h * w
    if len(raw_int8) != expected:
        raise ValueError(
            f"margin_map int8 size mismatch: got {len(raw_int8)}, "
            f"expected H*W={expected}"
        )
    margin_map_int8 = np.frombuffer(raw_int8, dtype=np.int8).reshape(h, w).copy()

    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}
    if not isinstance(meta, dict):
        raise ValueError(
            f"meta JSON must be an object; got {type(meta).__name__}"
        )

    return D1PolytopeArchive(
        margin_map_int8=margin_map_int8,
        margin_map_scale=float(scale),
        jacobian_lipschitz=float(jacobian_lipschitz),
        polytope_payload=polytope_payload,
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
    d1_overhead_bytes: int,
    config,
    predicted_score_band: tuple[float, float] = (0.181, 0.188),
    runtime_overlay_consumed: bool = True,
) -> dict[str, object]:
    """Build a non-promotable readiness manifest for autopilot consumption.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192
    every score prediction is tagged ``[first-principles-bound]`` and
    carries ``score_claim=False``, ``promotion_eligible=False`` until both
    ``[contest-CUDA]`` and ``[contest-CPU]`` paired auth eval lands on 1:1
    contest-CI hardware. ``ready_for_exact_eval_dispatch`` flips to True
    once L2 overlay is operational (default 2026-05-14).

    Projected band ``[0.181, 0.188]`` is the L2 OPERATIONAL design
    estimate. Catalog #220 STRICT preflight refuses byte addition >1 KB
    without operational score-improvement mechanism; ``runtime_overlay_consumed``
    defaults to True post-L2-integration because the inflate runtime applies
    the polytope overlay (see
    :func:`tac.substrates.d1_segnet_margin_polytope.overlay.apply_l2_overlay_for_video_list`).
    """
    overlay_ready = bool(runtime_overlay_consumed)
    return {
        "d1poly_schema_version": D1POLY1_SCHEMA_VERSION,
        "base_substrate_id": base_substrate_id,
        "base_archive_bytes": int(base_archive_bytes),
        "d1_overhead_bytes": int(d1_overhead_bytes),
        "total_archive_bytes": int(base_archive_bytes + d1_overhead_bytes),
        "runtime_overlay_consumed": overlay_ready,
        "current_runtime_effect": (
            "d1_overlay_active" if overlay_ready else "base_renderer_plus_rate_only"
        ),
        "predicted_score_band_low": float(predicted_score_band[0]) if overlay_ready else None,
        "predicted_score_band_high": float(predicted_score_band[1]) if overlay_ready else None,
        "l2_projected_score_band_low": float(predicted_score_band[0]),
        "l2_projected_score_band_high": float(predicted_score_band[1]),
        "predicted_score_evidence_grade": (
            "first-principles-bound" if overlay_ready else "blocked_l1_noop_overlay"
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": overlay_ready,
        "promotion_eligible": False,
        "dispatch_blockers": []
        if overlay_ready
        else [
            "d1_runtime_overlay_not_consumed",
            "current_l1_packet_is_base_renderer_plus_rate_overhead",
            "exact_eval_would_measure_noop_sidecar_rate_penalty_not_d1_score_lowering",
        ],
        "polytope_payload_bits": int(config.polytope_payload_bits),
        "margin_map_resolution": list(config.margin_map_resolution),
        "margin_map_mode": str(config.margin_map_mode),
        "margin_threshold": float(config.margin_threshold),
    }


__all__ = [
    "D1POLY1_HEADER_FMT",
    "D1POLY1_HEADER_SIZE",
    "D1POLY1_MAGIC",
    "D1POLY1_SCHEMA_VERSION",
    "D1PolytopeArchive",
    "build_readiness_manifest",
    "pack_archive",
    "parse_archive",
]
