# SPDX-License-Identifier: MIT
"""WLV2 archive grammar — monolithic single-file ``0.bin`` (v8 Path B successor to v7's CH06).

Per design memo Section 9: WLV2 = "WaveLet Variant 2" (distinct from canonical
wavelet substrate's WLV1 magic to avoid collision). Magic ``b"WLV2"``.

Schema v1 (the first wavelet+Wyner-Ziv grammar):

    MAGIC(4)             b"WLV2"     wavelet-residual variant 2
    VERSION(1)           u8 == 1     schema version
    NUM_PAIRS(2)         u16         number of (frame_0, frame_1) pairs
    EVAL_HEIGHT(2)       u16         contest eval height (384)
    EVAL_WIDTH(2)        u16         contest eval width  (512)
    OUTPUT_HEIGHT(2)     u16         contest output (camera) height (874)
    OUTPUT_WIDTH(2)      u16         contest output (camera) width  (1164)
    DWT_LEVEL(1)         u8 == 2     wavelet decomposition depth
    NUM_SUBBANDS(1)      u8 == 7     subbands per channel per frame
    LAPLACIAN_LEN(4)     u32         NUM_SUBBANDS*NUM_SEGNET_CLASSES*4 bytes
    PER_PAIR_OFFSETS_LEN(4) u32      offsets table length
    GRAY_F0_LEN(4)       u32         gray frame_0 arith stream length
    GRAY_F1RES_LEN(4)    u32         gray frame_1 residual arith stream length
    CB_F0_LEN(4)         u32
    CB_F1RES_LEN(4)      u32
    CR_F0_LEN(4)         u32
    CR_F1RES_LEN(4)      u32
    CLS_LEN(4)           u32         arith-coded class-label subbands stream
    META_LEN(4)          u32         utf-8 json meta length
    QUANT_STEPS_LEN(4)   u32         NUM_SUBBANDS*4 bytes (uint32 per-subband step)

    PAYLOAD BLOBS (in order):
      LAPLACIAN_BLOB           NUM_SUBBANDS*NUM_SEGNET_CLASSES*4 bytes float32
      PER_PAIR_OFFSETS_BLOB    NUM_PAIRS*7-streams*uint32 byte offsets
      GRAY_F0_BLOB             arith bytes
      GRAY_F1RES_BLOB          arith bytes
      CB_F0_BLOB               arith bytes
      CB_F1RES_BLOB            arith bytes
      CR_F0_BLOB               arith bytes
      CR_F1RES_BLOB            arith bytes
      CLS_BLOB                 arith bytes
      META_BLOB                utf-8 json
      QUANT_STEPS_BLOB         NUM_SUBBANDS uint32 step sizes

Header layout: ``<4sBHHHHHBBIIIIIIIIIII`` -> 4+1+2+2+2+2+2+1+1+ 11*4 = 61 bytes.

Per Catalog #124: this is the export-first grammar; the L1 SCAFFOLD declares
every byte-level field BEFORE any inflate code is written. Per Catalog #220:
the operational-mechanism contract is "every wavelet-stream byte is consumed
by inflate via decode_subband_arith + idwt2_db4_depth2 + Wyner-Ziv add".

NO scorer, NO torch at inflate. Reviewable in 30s per HNeRV parity L12.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON meta, fixed-precision arith state)
- No /tmp paths
- No scorer load
- Apples-to-apples: archive bytes are the ground truth; provenance.json
  records per-blob lengths for byte-addressable observability
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
    NUM_SEGNET_CLASSES,
)

from .wavelet_codec import (
    NUM_SUBBANDS,
    PerSubbandLaplacianPriors,
)

WLV2_MAGIC: bytes = b"WLV2"
WLV2_SCHEMA_VERSION: int = 1

# Header struct: 4s (magic) + B (version) + 5 u16 (n_pairs, eval_h, eval_w, out_h, out_w)
# + 2 u8 (dwt_level, num_subbands) + 11 u32 (lengths).
WLV2_HEADER_FMT: str = "<4sBHHHHHBBIIIIIIIIIII"
WLV2_HEADER_SIZE: int = struct.calcsize(WLV2_HEADER_FMT)
assert WLV2_HEADER_SIZE == 61, (
    f"WLV2 v1 header size invariant violated: {WLV2_HEADER_SIZE} != 61"
)


@dataclass(frozen=True)
class WaveletResidualArchive:
    """Parsed WLV2 archive — the inflate-time data contract.

    Per design memo Section 10 inflate runtime contract. All blobs are bytes;
    the caller decodes via :func:`tac.substrates.nscs06_v8_path_b_wavelet.wavelet_codec.decode_subband_arith`
    + :func:`idwt2_db4_depth2` + :func:`reconstruct_frame1_from_frame0_and_residual`.
    """

    schema_version: int
    num_pairs: int
    eval_height: int
    eval_width: int
    output_height: int
    output_width: int
    dwt_level: int
    num_subbands: int
    priors: PerSubbandLaplacianPriors
    per_pair_offsets: np.ndarray  # (num_pairs, 7-streams) uint32 byte offsets within each blob
    gray_f0_bytes: bytes
    gray_f1res_bytes: bytes
    cb_f0_bytes: bytes
    cb_f1res_bytes: bytes
    cr_f0_bytes: bytes
    cr_f1res_bytes: bytes
    cls_bytes: bytes
    meta: dict[str, object]
    quant_steps: tuple[int, ...]  # length NUM_SUBBANDS

    @property
    def grayscale_downsample(self) -> int:
        return int(self.meta.get("grayscale_downsample", 1))


def pack_archive(
    *,
    priors: PerSubbandLaplacianPriors,
    per_pair_offsets: np.ndarray,
    gray_f0_bytes: bytes,
    gray_f1res_bytes: bytes,
    cb_f0_bytes: bytes,
    cb_f1res_bytes: bytes,
    cr_f0_bytes: bytes,
    cr_f1res_bytes: bytes,
    cls_bytes: bytes,
    meta: dict[str, object],
    quant_steps: tuple[int, ...] | list[int],
    num_pairs: int,
    eval_height: int,
    eval_width: int,
    output_height: int,
    output_width: int,
) -> bytes:
    """Serialize a WLV2 archive deterministically.

    Args + invariants match :class:`WaveletResidualArchive`. ``quant_steps``
    MUST have NUM_SUBBANDS entries. ``per_pair_offsets`` MUST have shape
    (num_pairs, 7).
    """
    quant_steps = tuple(int(x) for x in quant_steps)
    if len(quant_steps) != NUM_SUBBANDS:
        raise ValueError(
            f"quant_steps must have {NUM_SUBBANDS} entries; got {len(quant_steps)}"
        )
    if any(s <= 0 for s in quant_steps):
        raise ValueError(f"quant_steps must all be > 0; got {quant_steps}")
    if per_pair_offsets.shape != (num_pairs, 7):
        raise ValueError(
            f"per_pair_offsets shape must be ({num_pairs}, 7); got {per_pair_offsets.shape}"
        )
    if per_pair_offsets.dtype != np.uint32:
        raise ValueError(
            f"per_pair_offsets must be uint32; got {per_pair_offsets.dtype}"
        )
    if priors.scales.shape != (NUM_SUBBANDS, NUM_SEGNET_CLASSES):
        raise ValueError(
            f"priors.scales shape must be ({NUM_SUBBANDS}, {NUM_SEGNET_CLASSES}); "
            f"got {priors.scales.shape}"
        )

    laplacian_blob = priors.scales.astype(np.float32).tobytes()
    offsets_blob = per_pair_offsets.astype(np.uint32).tobytes()
    quant_steps_blob = np.asarray(quant_steps, dtype=np.uint32).tobytes()
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")

    header = struct.pack(
        WLV2_HEADER_FMT,
        WLV2_MAGIC,
        WLV2_SCHEMA_VERSION,
        int(num_pairs),
        int(eval_height),
        int(eval_width),
        int(output_height),
        int(output_width),
        2,  # DWT_LEVEL
        NUM_SUBBANDS,
        len(laplacian_blob),
        len(offsets_blob),
        len(gray_f0_bytes),
        len(gray_f1res_bytes),
        len(cb_f0_bytes),
        len(cb_f1res_bytes),
        len(cr_f0_bytes),
        len(cr_f1res_bytes),
        len(cls_bytes),
        len(meta_bytes),
        len(quant_steps_blob),
    )
    parts = [
        header,
        laplacian_blob,
        offsets_blob,
        gray_f0_bytes,
        gray_f1res_bytes,
        cb_f0_bytes,
        cb_f1res_bytes,
        cr_f0_bytes,
        cr_f1res_bytes,
        cls_bytes,
        meta_bytes,
        quant_steps_blob,
    ]
    return b"".join(parts)


def parse_archive(data: bytes) -> WaveletResidualArchive:
    """Parse a WLV2 archive byte stream."""
    if len(data) < WLV2_HEADER_SIZE:
        raise ValueError(
            f"archive too short for WLV2 header: {len(data)} < {WLV2_HEADER_SIZE}"
        )
    fields = struct.unpack(WLV2_HEADER_FMT, data[:WLV2_HEADER_SIZE])
    (
        magic,
        version,
        num_pairs,
        eval_h,
        eval_w,
        out_h,
        out_w,
        dwt_level,
        num_subbands,
        laplacian_len,
        offsets_len,
        gray_f0_len,
        gray_f1res_len,
        cb_f0_len,
        cb_f1res_len,
        cr_f0_len,
        cr_f1res_len,
        cls_len,
        meta_len,
        quant_steps_len,
    ) = fields
    if magic != WLV2_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {WLV2_MAGIC!r})")
    if version != WLV2_SCHEMA_VERSION:
        raise ValueError(f"unsupported WLV2 version: {version}")
    if dwt_level != 2:
        raise ValueError(f"only DWT_LEVEL=2 is supported; got {dwt_level}")
    if num_subbands != NUM_SUBBANDS:
        raise ValueError(
            f"NUM_SUBBANDS mismatch: header={num_subbands} expected {NUM_SUBBANDS}"
        )
    expected_laplacian = NUM_SUBBANDS * NUM_SEGNET_CLASSES * 4
    if laplacian_len != expected_laplacian:
        raise ValueError(
            f"LAPLACIAN_LEN={laplacian_len}; expected {expected_laplacian}"
        )
    expected_offsets = num_pairs * 7 * 4
    if offsets_len != expected_offsets:
        raise ValueError(
            f"PER_PAIR_OFFSETS_LEN={offsets_len}; expected {expected_offsets}"
        )
    expected_quant_steps = NUM_SUBBANDS * 4
    if quant_steps_len != expected_quant_steps:
        raise ValueError(
            f"QUANT_STEPS_LEN={quant_steps_len}; expected {expected_quant_steps}"
        )

    cursor = WLV2_HEADER_SIZE
    laplacian_blob = data[cursor : cursor + laplacian_len]
    cursor += laplacian_len
    offsets_blob = data[cursor : cursor + offsets_len]
    cursor += offsets_len
    gray_f0_bytes = data[cursor : cursor + gray_f0_len]
    cursor += gray_f0_len
    gray_f1res_bytes = data[cursor : cursor + gray_f1res_len]
    cursor += gray_f1res_len
    cb_f0_bytes = data[cursor : cursor + cb_f0_len]
    cursor += cb_f0_len
    cb_f1res_bytes = data[cursor : cursor + cb_f1res_len]
    cursor += cb_f1res_len
    cr_f0_bytes = data[cursor : cursor + cr_f0_len]
    cursor += cr_f0_len
    cr_f1res_bytes = data[cursor : cursor + cr_f1res_len]
    cursor += cr_f1res_len
    cls_bytes = data[cursor : cursor + cls_len]
    cursor += cls_len
    meta_blob = data[cursor : cursor + meta_len]
    cursor += meta_len
    quant_steps_blob = data[cursor : cursor + quant_steps_len]
    cursor += quant_steps_len
    if cursor != len(data):
        raise ValueError(
            f"WLV2 trailing bytes: cursor={cursor} != len(data)={len(data)}"
        )

    priors_scales = np.frombuffer(laplacian_blob, dtype=np.float32).reshape(
        NUM_SUBBANDS, NUM_SEGNET_CLASSES
    ).copy()
    priors = PerSubbandLaplacianPriors(scales=priors_scales)
    per_pair_offsets = np.frombuffer(offsets_blob, dtype=np.uint32).reshape(
        num_pairs, 7
    ).copy()
    quant_steps = tuple(int(x) for x in np.frombuffer(quant_steps_blob, dtype=np.uint32))
    meta = json.loads(meta_blob.decode("utf-8")) if meta_len > 0 else {}

    return WaveletResidualArchive(
        schema_version=int(version),
        num_pairs=int(num_pairs),
        eval_height=int(eval_h),
        eval_width=int(eval_w),
        output_height=int(out_h),
        output_width=int(out_w),
        dwt_level=int(dwt_level),
        num_subbands=int(num_subbands),
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=gray_f0_bytes,
        gray_f1res_bytes=gray_f1res_bytes,
        cb_f0_bytes=cb_f0_bytes,
        cb_f1res_bytes=cb_f1res_bytes,
        cr_f0_bytes=cr_f0_bytes,
        cr_f1res_bytes=cr_f1res_bytes,
        cls_bytes=cls_bytes,
        meta=meta,
        quant_steps=quant_steps,
    )
