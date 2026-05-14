# SPDX-License-Identifier: MIT
"""D4 archive grammar — WZF01 monolithic 0.bin (substrate-engineering scope).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (in single-zip-member ``0.bin`` slot
  when packaged in archive.zip)
* L4 inflate.py ≤ 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, fp16 packed motion, fixed brotli quality)

The 5-section grammar (single monolithic 0.bin; substrate-engineering composition justifies embedding the base substrate bytes):

::

    MAGIC(4)               b"WZF\\x01"
    VERSION(1)             u8 (== 1)
    NUM_PAIRS(2)           u16
    MOTION_MODE(1)         u8 (0=SE3, 1=OPTICAL_FLOW)
    FLOW_GRID_H(2)         u16  (0 if SE3)
    FLOW_GRID_W(2)         u16  (0 if SE3)
    RESIDUAL_COARSE_H(2)   u16
    RESIDUAL_COARSE_W(2)   u16
    BASE_SHA_LEN(1)        u8 (== 64 for hex sha256)
    BASE_BYTES_LEN(4)      u32 embedded base substrate 0.bin bytes (length;
                                may be 0 if the operator promises the base
                                archive is co-located via a sister zip
                                member; v1 default is to embed for monolithic
                                custody).
    MOTION_LEN(4)          u32 brotli(motion params)
    RESIDUAL_LEN(4)        u32 brotli(int8-quantized residual)
    META_LEN(4)            u32 sorted-keys JSON
    BASE_SHA(BASE_SHA_LEN) base substrate's archive sha256 (hex ASCII)
    BASE_BYTES(BASE_BYTES_LEN)  raw base 0.bin bytes (NOT brotli-recompressed;
                                base substrate already applies its own
                                compression contract)
    MOTION_BLOB(MOTION_LEN)
    RESIDUAL_BLOB(RESIDUAL_LEN)
    META_BLOB(META_LEN)

Sections:

* **MOTION_BLOB**: brotli-compressed motion parameters. For SE(3) mode the
  raw payload is ``num_pairs * 6 * float16`` = 7.2 KB (closes to ~4-5 KB
  after brotli). For OPTICAL_FLOW mode the raw payload is
  ``num_pairs * 2 * grid_h * grid_w * float16`` = 460 KB at 12×16 (closes
  to ~30-50 KB after brotli).

* **RESIDUAL_BLOB**: brotli-compressed int8 residual. See
  ``residual_codec.encode_residual_blob`` for the inner layout.

* **META_BLOB**: JSON ``{"motion_clamp", "residual_dtype", "base_substrate_id",
  "design_notes", ...}`` — sidecar parameters that don't fit in header fields.

* **BASE_SHA**: hex-ASCII sha256 of the base substrate's archive bytes. The
  D4 inflate runtime resolves the base archive via the file-list and verifies
  this matches at decode time (or refuses to inflate on mismatch — a
  custody-cardinal violation).

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 motion blob, brotli quality fixed)
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

WZF01_MAGIC: bytes = b"WZF\x01"
"""D4 archive magic (4 bytes, distinct from WZ1/TT5L/SBO1/etc.)."""

WZF01_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
# MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + MOTION_MODE(1)
# + FLOW_GRID_H(2) + FLOW_GRID_W(2)
# + RESIDUAL_COARSE_H(2) + RESIDUAL_COARSE_W(2)
# + BASE_SHA_LEN(1)
# + BASE_BYTES_LEN(4) + MOTION_LEN(4) + RESIDUAL_LEN(4) + META_LEN(4)
# = 4+1+2+1+2+2+2+2+1+4+4+4+4 = 33 bytes
WZF01_HEADER_FMT: str = "<4sBHBHHHHBIIII"
WZF01_HEADER_SIZE: int = struct.calcsize(WZF01_HEADER_FMT)
assert WZF01_HEADER_SIZE == 33, (
    f"WZF01 header size invariant: expected 33, got {WZF01_HEADER_SIZE}"
)

# Deterministic brotli quality (matches WZ1 / TT5L / SIREN siblings).
_BROTLI_QUALITY: int = 9

BASE_SHA_HEX_LEN: int = 64
"""Length of hex-ASCII sha256 (64 chars = 32 raw bytes)."""

_MOTION_MODE_SE3: int = 0
_MOTION_MODE_OPTICAL_FLOW: int = 1


@dataclass(frozen=True)
class WynerZivFrame0Archive:
    """Parsed WZF01 archive — the inflate-time data contract."""

    motion_blob_raw: bytes
    """Decompressed motion-parameter bytes (downstream parser resolves shape
    from header ``motion_mode`` + ``flow_grid_h`` + ``flow_grid_w``)."""

    residual_blob: bytes
    """The original (brotli-compressed) residual blob; downstream uses
    ``residual_codec.decode_residual_blob`` to expand to ``(N, 3, h, w)``."""

    meta: dict[str, object]
    """Sidecar JSON meta with non-header hparams."""

    base_substrate_archive_sha256_hex: str
    """The hex sha256 of the base substrate's archive bytes (custody anchor)."""

    base_substrate_bytes: bytes
    """The embedded base substrate ``0.bin`` bytes (may be ``b""`` when the
    base archive is co-located via a sister zip member at inflate time)."""

    schema_version: int
    num_pairs: int
    motion_mode: int  # 0=SE3, 1=OPTICAL_FLOW
    flow_grid_h: int
    flow_grid_w: int
    residual_coarse_h: int
    residual_coarse_w: int

    @property
    def motion_mode_label(self) -> str:
        if self.motion_mode == _MOTION_MODE_SE3:
            return "se3_parametric"
        elif self.motion_mode == _MOTION_MODE_OPTICAL_FLOW:
            return "optical_flow"
        else:
            return f"unknown({self.motion_mode})"


def _serialize_motion(
    *,
    motion_mode: int,
    se3_flat: torch.Tensor | None,
    flow_uv: torch.Tensor | None,
    num_pairs: int,
    flow_grid_h: int,
    flow_grid_w: int,
) -> bytes:
    """Serialize motion params to deterministic brotli-packed fp16 bytes."""
    if motion_mode == _MOTION_MODE_SE3:
        if se3_flat is None:
            raise ValueError("se3_flat is required for SE3 mode")
        if se3_flat.shape != (num_pairs, 6):
            raise ValueError(
                f"se3_flat shape {tuple(se3_flat.shape)} != ({num_pairs}, 6)"
            )
        arr = (
            se3_flat.detach().to("cpu", dtype=torch.float16).contiguous().numpy()
        )
        raw = arr.tobytes(order="C")
    elif motion_mode == _MOTION_MODE_OPTICAL_FLOW:
        if flow_uv is None:
            raise ValueError("flow_uv is required for OPTICAL_FLOW mode")
        expected = (num_pairs, 2, flow_grid_h, flow_grid_w)
        if flow_uv.shape != expected:
            raise ValueError(
                f"flow_uv shape {tuple(flow_uv.shape)} != {expected}"
            )
        arr = (
            flow_uv.detach().to("cpu", dtype=torch.float16).contiguous().numpy()
        )
        raw = arr.tobytes(order="C")
    else:
        raise ValueError(f"unknown motion mode: {motion_mode}")
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_motion_raw(blob: bytes) -> bytes:
    """Decompress the motion blob; downstream parses shape from header."""
    return brotli.decompress(blob)


def deserialize_motion_to_tensor(
    raw: bytes,
    *,
    motion_mode: int,
    num_pairs: int,
    flow_grid_h: int,
    flow_grid_w: int,
) -> torch.Tensor:
    """Convert the decompressed motion bytes into a torch tensor.

    For SE3 mode: returns ``(num_pairs, 6)`` float32.
    For OPTICAL_FLOW mode: returns ``(num_pairs, 2, grid_h, grid_w)`` float32.
    """
    if motion_mode == _MOTION_MODE_SE3:
        expected_floats = num_pairs * 6
        expected_bytes = expected_floats * 2  # fp16
        if len(raw) != expected_bytes:
            raise ValueError(
                f"motion raw byte count mismatch (SE3): got {len(raw)}, "
                f"expected {expected_bytes}"
            )
        arr = np.frombuffer(raw, dtype=np.float16).reshape(num_pairs, 6).astype(np.float32)
        return torch.from_numpy(arr.copy())
    elif motion_mode == _MOTION_MODE_OPTICAL_FLOW:
        expected_floats = num_pairs * 2 * flow_grid_h * flow_grid_w
        expected_bytes = expected_floats * 2
        if len(raw) != expected_bytes:
            raise ValueError(
                f"motion raw byte count mismatch (FLOW): got {len(raw)}, "
                f"expected {expected_bytes}"
            )
        arr = np.frombuffer(raw, dtype=np.float16).reshape(
            num_pairs, 2, flow_grid_h, flow_grid_w
        ).astype(np.float32)
        return torch.from_numpy(arr.copy())
    else:
        raise ValueError(f"unknown motion mode: {motion_mode}")


def pack_archive(
    *,
    motion_mode: int,
    se3_flat: torch.Tensor | None,
    flow_uv: torch.Tensor | None,
    residual_blob: bytes,
    meta: dict[str, object],
    base_substrate_archive_sha256_hex: str,
    base_substrate_bytes: bytes,
    num_pairs: int,
    flow_grid_h: int,
    flow_grid_w: int,
    residual_coarse_h: int,
    residual_coarse_w: int,
    schema_version: int = WZF01_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained substrate's motion + residual into WZF01 0.bin bytes.

    Args:
        motion_mode: 0 for SE(3), 1 for OPTICAL_FLOW.
        se3_flat: required for SE3 mode; shape ``(num_pairs, 6)``.
        flow_uv: required for OPTICAL_FLOW mode; shape
            ``(num_pairs, 2, flow_grid_h, flow_grid_w)``.
        residual_blob: brotli-compressed residual bytes from
            ``residual_codec.encode_residual_blob``.
        meta: extra metadata (sorted-keys JSON).
        base_substrate_archive_sha256_hex: hex sha256 of the base substrate
            archive bytes. Must be exactly ``BASE_SHA_HEX_LEN`` characters
            (64 = sha256 hex).
        base_substrate_bytes: the embedded base substrate ``0.bin`` bytes.
            Pass ``b""`` only if you commit to a sister-zip-member layout
            (NOT the v1 default — monolithic embedding is the canonical
            substrate-engineering composition).
        num_pairs: contest pair count.
        flow_grid_h, flow_grid_w: required for OPTICAL_FLOW mode (set to 0
            for SE3 mode).
        residual_coarse_h, residual_coarse_w: residual codec spatial size.
        schema_version: must equal ``WZF01_SCHEMA_VERSION``.

    Returns:
        The full 0.bin bytes ready to write to archive.zip member ``0.bin``.

    Raises:
        ValueError: on shape mismatch, unknown mode, or invalid base sha.
    """
    if schema_version != WZF01_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if motion_mode not in (_MOTION_MODE_SE3, _MOTION_MODE_OPTICAL_FLOW):
        raise ValueError(f"unknown motion_mode: {motion_mode}")

    if (
        not isinstance(base_substrate_archive_sha256_hex, str)
        or len(base_substrate_archive_sha256_hex) != BASE_SHA_HEX_LEN
    ):
        raise ValueError(
            f"base_substrate_archive_sha256_hex must be a {BASE_SHA_HEX_LEN}-char "
            f"hex string; got {base_substrate_archive_sha256_hex!r}"
        )
    try:
        int(base_substrate_archive_sha256_hex, 16)
    except ValueError as exc:
        raise ValueError(
            f"base_substrate_archive_sha256_hex must be hex; got "
            f"{base_substrate_archive_sha256_hex!r}"
        ) from exc

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("flow_grid_h", flow_grid_h, 0xFFFF),
        ("flow_grid_w", flow_grid_w, 0xFFFF),
        ("residual_coarse_h", residual_coarse_h, 0xFFFF),
        ("residual_coarse_w", residual_coarse_w, 0xFFFF),
    ):
        if v < 0 or v > max_v:
            raise ValueError(f"{name}={v} out of range [0, {max_v}]")
    if num_pairs == 0:
        raise ValueError("num_pairs must be > 0")
    if residual_coarse_h == 0 or residual_coarse_w == 0:
        raise ValueError("residual_coarse_h/w must be > 0")
    if motion_mode == _MOTION_MODE_OPTICAL_FLOW and (
        flow_grid_h == 0 or flow_grid_w == 0
    ):
        raise ValueError("flow_grid_h/w must be > 0 for OPTICAL_FLOW mode")

    motion_blob = _serialize_motion(
        motion_mode=motion_mode,
        se3_flat=se3_flat,
        flow_uv=flow_uv,
        num_pairs=num_pairs,
        flow_grid_h=flow_grid_h,
        flow_grid_w=flow_grid_w,
    )

    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC,
        schema_version,
        num_pairs,
        motion_mode,
        flow_grid_h,
        flow_grid_w,
        residual_coarse_h,
        residual_coarse_w,
        BASE_SHA_HEX_LEN,
        len(base_substrate_bytes),
        len(motion_blob),
        len(residual_blob),
        len(meta_bytes),
    )
    base_sha_bytes = base_substrate_archive_sha256_hex.encode("ascii")
    return (
        header
        + base_sha_bytes
        + base_substrate_bytes
        + motion_blob
        + residual_blob
        + meta_bytes
    )


def parse_wzf01_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for WZF01 (D4 Wyner-Ziv frame-0) grammar.

    Canonical section-offset parser for WZF01 inner-blob bytes. The returned
    mapping is the data contract consumed by:

    - :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
      section-aware Tier A density estimation)
    - :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
      dispatch — WZF01 auto-detection by ``b"WZF\\x01"`` magic prefix)

    Returned sections (Tier A / Tier B targets):

    - ``wzf01_header`` — 33-byte header (control_or_metadata; fixed layout)
    - ``base_substrate_archive_sha256`` — 64-byte hex-ASCII sha256 of the base
      substrate archive (control_or_metadata; custody anchor)
    - ``base_substrate_bytes`` — embedded base substrate ``0.bin`` bytes
      (decoder_weight_stream — the base renderer reconstructs frame 1 which
      is the side-information for Wyner-Ziv derivation of frame 0)
    - ``motion_blob`` — brotli-compressed motion parameters (SE(3) or
      optical-flow) — sidecar_or_correction_stream (per-pair motion side-info)
    - ``residual_blob`` — brotli-compressed int8 photometric residual
      (latent_stream — the rate-limited frame_0 - warp(frame_1) signal)
    - ``meta_blob`` — sorted-keys utf-8 JSON (control_or_metadata)

    The byte ranges returned here MUST agree with the writer in
    :func:`pack_archive` and with the canonical full-decode parser
    :func:`parse_archive`. The single-source-of-truth for WZF01 byte layout
    is :data:`WZF01_HEADER_FMT` + :data:`WZF01_HEADER_SIZE`.

    Differs from :func:`parse_archive` in that this function returns
    section-offset tuples only (no brotli decompression). It is cheaper and
    is safe to call with brotli-tampered blobs.

    Raises ``ValueError`` on:

    - short header (< 33 bytes)
    - bad magic (!= ``b"WZF\\x01"``)
    - unsupported schema version (!= 1)
    - base_sha_len != 64
    - unknown motion_mode (must be 0 or 1)
    - archive size mismatch (declared end != len(archive_bytes)),
      covering BOTH truncated archives and trailing-byte schema drift.
    """
    if len(archive_bytes) < WZF01_HEADER_SIZE:
        raise ValueError(
            f"wzf01 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {WZF01_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        _num_pairs,
        motion_mode,
        _flow_grid_h,
        _flow_grid_w,
        _residual_coarse_h,
        _residual_coarse_w,
        base_sha_len,
        base_bytes_len,
        motion_len,
        residual_len,
        meta_len,
    ) = struct.unpack(WZF01_HEADER_FMT, archive_bytes[:WZF01_HEADER_SIZE])
    if magic != WZF01_MAGIC:
        raise ValueError(
            f"wzf01 archive: bad magic {magic!r} (expected {WZF01_MAGIC!r})"
        )
    if version != WZF01_SCHEMA_VERSION:
        raise ValueError(
            f"wzf01 archive: unsupported schema version {version} "
            f"(expected {WZF01_SCHEMA_VERSION})"
        )
    if base_sha_len != BASE_SHA_HEX_LEN:
        raise ValueError(
            f"wzf01 archive: base_sha_len {base_sha_len} != expected "
            f"{BASE_SHA_HEX_LEN}"
        )
    if motion_mode not in (_MOTION_MODE_SE3, _MOTION_MODE_OPTICAL_FLOW):
        raise ValueError(
            f"wzf01 archive: unknown motion_mode {motion_mode} "
            f"(expected 0=SE3 or 1=OPTICAL_FLOW)"
        )
    end_header = WZF01_HEADER_SIZE
    end_base_sha = end_header + int(base_sha_len)
    end_base_bytes = end_base_sha + int(base_bytes_len)
    end_motion = end_base_bytes + int(motion_len)
    end_residual = end_motion + int(residual_len)
    end_meta = end_residual + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"wzf01 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    return {
        "wzf01_header": (0, WZF01_HEADER_SIZE),
        "base_substrate_archive_sha256": (end_header, int(base_sha_len)),
        "base_substrate_bytes": (end_base_sha, int(base_bytes_len)),
        "motion_blob": (end_base_bytes, int(motion_len)),
        "residual_blob": (end_motion, int(residual_len)),
        "meta_blob": (end_residual, int(meta_len)),
    }


# Canonical optimization-role mapping for WZF01 sections (consumed by
# tac.analysis.scorer_conditional_mdl and tac.analysis.hnerv_packet_sections).
# Mirrors the role taxonomy in ``ROLE_WEIGHTS`` / ``IBPS1_SECTION_ROLES``.
#
# Section semantics rationale:
# - base_substrate_bytes: the embedded base substrate 0.bin reconstructs
#   frame 1 which IS the cooperative-receiver side-information for Wyner-Ziv
#   derivation of frame 0. The base renderer's decoder weights are the
#   dominant score-affecting bytes (frame_1 reconstruction quality drives
#   both PoseNet pose error AND the inverse warp accuracy used by D4).
#   Therefore: decoder_weight_stream.
# - motion_blob: per-pair SE(3) or optical-flow parameters. These are
#   side-information for the warp that derives frame_0; they are correction
#   signal that adjusts the warp prediction. sidecar_or_correction_stream.
# - residual_blob: the brotli-compressed int8 photometric residual is the
#   rate-limited frame_0 - warp(frame_1) signal — the latent stream that
#   carries the unpredictable residual energy. latent_stream.
# - All header/identity/meta sections are control_or_metadata.
WZF01_SECTION_ROLES: dict[str, str] = {
    "wzf01_header": "control_or_metadata",
    "base_substrate_archive_sha256": "control_or_metadata",
    "base_substrate_bytes": "decoder_weight_stream",
    "motion_blob": "sidecar_or_correction_stream",
    "residual_blob": "latent_stream",
    "meta_blob": "control_or_metadata",
}


def parse_archive(blob: bytes) -> WynerZivFrame0Archive:
    """Parse WZF01 0.bin bytes back into a typed ``WynerZivFrame0Archive``."""
    if len(blob) < WZF01_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {WZF01_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        motion_mode,
        flow_grid_h,
        flow_grid_w,
        residual_coarse_h,
        residual_coarse_w,
        base_sha_len,
        base_bytes_len,
        motion_len,
        residual_len,
        meta_len,
    ) = struct.unpack(WZF01_HEADER_FMT, blob[:WZF01_HEADER_SIZE])

    if magic != WZF01_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {WZF01_MAGIC!r})")
    if version != WZF01_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    if base_sha_len != BASE_SHA_HEX_LEN:
        raise ValueError(
            f"unexpected base_sha_len {base_sha_len} (expected {BASE_SHA_HEX_LEN})"
        )
    if motion_mode not in (_MOTION_MODE_SE3, _MOTION_MODE_OPTICAL_FLOW):
        raise ValueError(f"unknown motion_mode: {motion_mode}")

    pos = WZF01_HEADER_SIZE
    base_sha_end = pos + base_sha_len
    if base_sha_end > len(blob):
        raise ValueError("archive truncated in base_sha section")
    base_sha = blob[pos:base_sha_end].decode("ascii")
    pos = base_sha_end

    base_bytes_end = pos + base_bytes_len
    if base_bytes_end > len(blob):
        raise ValueError("archive truncated in base_substrate_bytes section")
    base_substrate_bytes = blob[pos:base_bytes_end]
    pos = base_bytes_end

    motion_blob = blob[pos : pos + motion_len]
    if len(motion_blob) != motion_len:
        raise ValueError("archive truncated in motion section")
    pos += motion_len

    residual_blob = blob[pos : pos + residual_len]
    if len(residual_blob) != residual_len:
        raise ValueError("archive truncated in residual section")
    pos += residual_len

    meta_blob = blob[pos : pos + meta_len]
    if len(meta_blob) != meta_len:
        raise ValueError("archive truncated in meta section")
    pos += meta_len

    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    motion_raw = _deserialize_motion_raw(motion_blob)
    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}

    return WynerZivFrame0Archive(
        motion_blob_raw=motion_raw,
        residual_blob=residual_blob,
        meta=meta,
        base_substrate_archive_sha256_hex=base_sha,
        base_substrate_bytes=base_substrate_bytes,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        motion_mode=int(motion_mode),
        flow_grid_h=int(flow_grid_h),
        flow_grid_w=int(flow_grid_w),
        residual_coarse_h=int(residual_coarse_h),
        residual_coarse_w=int(residual_coarse_w),
    )


__all__ = [
    "BASE_SHA_HEX_LEN",
    "WZF01_HEADER_FMT",
    "WZF01_HEADER_SIZE",
    "WZF01_MAGIC",
    "WZF01_SCHEMA_VERSION",
    "WZF01_SECTION_ROLES",
    "WynerZivFrame0Archive",
    "deserialize_motion_to_tensor",
    "pack_archive",
    "parse_archive",
    "parse_wzf01_archive_bytes",
]
