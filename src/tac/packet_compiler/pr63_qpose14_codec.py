# SPDX-License-Identifier: MIT
"""PR63 ``qpose14`` pose codec primitives — reusable byte-grammar pieces.

This module extracts the REUSABLE byte-grammar primitives from the PR63
public submission (``submissions/qpose14/inflate.py``) into typed,
golden-vector-backed transducers.

Two primitives land here:

1. **qpose14 uint16-view int16 codec**
   (:func:`encode_qpose14_uint16_view_int16` /
    :func:`decode_qpose14_uint16_view_int16`)

   PR63 stores a `(n, 6)` pose tensor as a flat ``uint16[n*6]`` little-endian
   buffer where column 0 is genuinely unsigned (encodes longitudinal speed
   via the affine ``q / 512.0 + 20.0``) and columns 1-5 are signed values
   reinterpreted (`np.view(np.int16)` in numpy parlance) and recovered as
   ``int16 / 2048.0``. The "view-cast" trick saves the per-column sign
   indicator byte because the storage column is `uint16` and the consumer
   knows which columns to re-view as `int16` from the convention.

   Source (PR63, lines 300-303): ::

       q = np.frombuffer(brotli.decompress(pose_q_br_data), dtype=np.uint16).reshape(-1, 6)
       pose_np[:, 0] = q[:, 0].astype(np.float32) / 512.0 + 20.0
       pose_np[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0

2. **qpose14 packed-payload single-zip-member grammar**
   (:func:`pack_qpose14_packed_payload` /
    :func:`unpack_qpose14_packed_payload`)

   PR63 packs three brotli-compressed streams (mask, model, pose_q) into a
   single ZIP member ``p`` with HARDCODED fixed-length sections. From PR63
   lines 269-276: ::

       packed_payload = data_dir / "p"
       payload = packed_payload.read_bytes()
       mask_br_data = payload[:219472]
       model_br_data = payload[219472:219472 + 66841]
       pose_q_br_data = payload[219472 + 66841:]

   The fixed-offset variant is byte-faithful to PR63 but BRITTLE — any
   change in member sizes breaks the offsets. The reusable primitive we
   land below is the **LENGTH-PREFIXED variant** (per HNeRV parity
   discipline lesson 3 "Fixed offsets ... in source"): the grammar stores
   ``[u32 n_members][u32 size_0][bytes_0]...`` so the decoder reads the
   sizes from the payload itself.

   This intentionally diverges from PR63's hardcoded-offset layout. The
   PR63 layout is byte-identical to ``encode_length_prefixed_sections`` in
   ``pr97_h3_grammar`` when the consumer knows the section count in
   advance; we provide a thin pose-codec-specific wrapper here that pins
   the section count at construction time (PR63 always uses exactly 3:
   mask, model, pose_q) so callers cannot mis-frame the payload.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr63_intake_20260505_auto/source/submissions/qpose14/inflate.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + struct + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr63_qpose14_uint16_int16_v1`` and ``pr63_qpose14_single_zip_member_v1``
  golden vectors.
* OSS-friendly: public surface is the 4 names re-exported from
  ``tac.packet_compiler``.

[empirical:src/tac/packet_compiler/golden_vectors/pr63_qpose14_uint16_int16_v1.json
 + src/tac/packet_compiler/golden_vectors/pr63_qpose14_single_zip_member_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ── Constants ───────────────────────────────────────────────────────────────

#: PR63 uses 6 pose dimensions (1 unsigned vel column + 5 view-as-int16 cols).
PR63_QPOSE14_N_DIMS: int = 6
#: PR63 hardcodes ``1/512.0`` for column-0 velocity scale.
PR63_VEL_SCALE: float = 1.0 / 512.0
#: PR63 hardcodes ``20.0`` for column-0 velocity bias.
PR63_VEL_BIAS: float = 20.0
#: PR63 hardcodes ``1/2048.0`` for columns 1-5 (signed) recovery scale.
PR63_SIGNED_SCALE: float = 1.0 / 2048.0


# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QPose14Uint16Int16Stream:
    """Encoded qpose14 pose payload (PR63 uint16-view-int16 grammar).

    Attributes
    ----------
    payload:
        Flat ``uint16[n_frames * 6]`` little-endian bytes; column 0 is
        unsigned, columns 1-5 are int16 reinterpreted as uint16 storage.
    n_frames:
        Frame count (PR63 uses 600 for a standard 1200-frame chunk).
    """

    payload: bytes
    n_frames: int


@dataclass(frozen=True)
class QPose14PackedPayload:
    """Result of decoding the PR63 single-zip-member packed payload.

    Attributes
    ----------
    sections:
        Tuple of bytes for the (mask, model, pose_q) brotli-compressed
        streams (or whatever the consumer packs — the primitive is
        agnostic).
    n_members:
        Section count (pinned to 3 by PR63 convention).
    total_bytes:
        Total payload length, including the header.
    """

    sections: tuple[bytes, ...]
    n_members: int
    total_bytes: int


# ── qpose14 uint16-view int16 codec ─────────────────────────────────────────


def encode_qpose14_uint16_view_int16(poses: np.ndarray) -> QPose14Uint16Int16Stream:
    """Encode a ``(n, 6)`` pose tensor under PR63's view-cast grammar.

    Parameters
    ----------
    poses:
        Float pose tensor of shape ``(n_frames, 6)``. Column 0 (velocity)
        is quantised as ``round((p - bias) / vel_scale)`` and must fit in
        uint16 ``[0, 65535]``. Columns 1-5 are quantised as
        ``round(p / signed_scale)`` and must fit in int16 ``[-32768, 32767]``.

    Returns
    -------
    QPose14Uint16Int16Stream
        Container with the encoded payload and frame count.

    Raises
    ------
    ValueError
        On wrong shape, empty input, or out-of-range quantisation in
        either the uint16 (col 0) or int16 (cols 1-5) range.
    """
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"poses must be 2D; got shape {arr.shape}")
    n_frames, n_dims = arr.shape
    if n_dims != PR63_QPOSE14_N_DIMS:
        raise ValueError(
            f"poses must have {PR63_QPOSE14_N_DIMS} dims; got {n_dims}"
        )
    if n_frames == 0:
        raise ValueError("poses must be non-empty")

    q_col0 = np.round((arr[:, 0] - PR63_VEL_BIAS) / PR63_VEL_SCALE).astype(np.int64)
    if q_col0.min() < 0 or q_col0.max() > 0xFFFF:
        raise ValueError(
            f"col-0 velocity out of uint16 range [0, 65535]; "
            f"got [{q_col0.min()}, {q_col0.max()}]"
        )

    q_signed = np.round(arr[:, 1:] / PR63_SIGNED_SCALE).astype(np.int64)
    if q_signed.min() < -32768 or q_signed.max() > 32767:
        raise ValueError(
            f"signed-col quantised values out of int16 range; "
            f"got [{q_signed.min()}, {q_signed.max()}]"
        )

    out = np.empty((n_frames, PR63_QPOSE14_N_DIMS), dtype=np.uint16)
    out[:, 0] = q_col0.astype(np.uint16)
    # View int16 storage as uint16 bit-pattern (numpy preserves bytes).
    out[:, 1:] = q_signed.astype(np.int16).view(np.uint16)

    return QPose14Uint16Int16Stream(
        payload=out.tobytes(),
        n_frames=int(n_frames),
    )


def decode_qpose14_uint16_view_int16(
    payload: bytes,
    *,
    n_frames: int,
) -> np.ndarray:
    """Decode a PR63 qpose14 payload back to a ``(n_frames, 6)`` float pose tensor.

    Parameters
    ----------
    payload:
        Flat ``uint16[n_frames * 6]`` little-endian bytes from the encoder.
    n_frames:
        Expected frame count.

    Returns
    -------
    np.ndarray
        Float32 pose tensor of shape ``(n_frames, 6)``.

    Raises
    ------
    ValueError
        On truncated/oversized payload, non-positive n_frames, or trailing
        bytes.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    payload = bytes(payload)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0; got {n_frames}")
    expected_len = n_frames * PR63_QPOSE14_N_DIMS * 2
    if len(payload) != expected_len:
        raise ValueError(
            f"payload length {len(payload)} != expected {expected_len} "
            f"for n_frames={n_frames}"
        )
    q = np.frombuffer(payload, dtype=np.uint16).reshape(
        n_frames, PR63_QPOSE14_N_DIMS
    )
    out = np.empty((n_frames, PR63_QPOSE14_N_DIMS), dtype=np.float32)
    out[:, 0] = q[:, 0].astype(np.float32) * np.float32(PR63_VEL_SCALE) + np.float32(
        PR63_VEL_BIAS
    )
    out[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) * np.float32(
        PR63_SIGNED_SCALE
    )
    return out


# ── qpose14 packed-payload single-zip-member grammar ────────────────────────


def pack_qpose14_packed_payload(
    sections: list[bytes] | tuple[bytes, ...],
    *,
    expected_n_members: int = 3,
) -> QPose14PackedPayload:
    """Pack N brotli-compressed sections into a single-zip-member payload.

    PR63 hardcodes 3 sections (mask / model / pose_q) at fixed offsets in
    its ``p`` ZIP member. The PR63 grammar saves per-section ZIP header +
    central-directory overhead by packing everything into one ZIP member.
    Per HNeRV parity discipline lesson 3 we land the LENGTH-PREFIXED
    variant (sizes encoded in payload, not hardcoded in source).

    Wire format:
        ``[u32 n_members][u32 size_0][bytes_0][u32 size_1][bytes_1]...``

    Parameters
    ----------
    sections:
        Iterable of byte sections to pack. Length is checked against
        ``expected_n_members``.
    expected_n_members:
        Expected section count. PR63 uses 3 (mask/model/pose_q).

    Returns
    -------
    QPose14PackedPayload
        Container with the framed bytes + section count.

    Raises
    ------
    ValueError
        On mismatched ``expected_n_members``, non-bytes section, or
        oversized section (> 4 GiB).
    """
    if expected_n_members <= 0:
        raise ValueError(
            f"expected_n_members must be > 0; got {expected_n_members}"
        )
    sections = tuple(sections)
    if len(sections) != expected_n_members:
        raise ValueError(
            f"section count {len(sections)} != expected_n_members {expected_n_members}"
        )
    out = bytearray()
    out += struct.pack("<I", len(sections))
    for i, s in enumerate(sections):
        if not isinstance(s, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"section {i} must be bytes-like; got {type(s)!r}"
            )
        s = bytes(s)
        if len(s) > 0xFFFFFFFF:
            raise ValueError(
                f"section {i} too large ({len(s)} bytes); max {0xFFFFFFFF}"
            )
        out += struct.pack("<I", len(s))
        out += s
    framed = bytes(out)
    return QPose14PackedPayload(
        sections=tuple(bytes(s) for s in sections),
        n_members=int(expected_n_members),
        total_bytes=len(framed),
    )


def unpack_qpose14_packed_payload(
    payload: bytes,
    *,
    expected_n_members: int = 3,
) -> QPose14PackedPayload:
    """Unpack the PR63 single-zip-member packed payload.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`pack_qpose14_packed_payload`.
    expected_n_members:
        Expected section count from the wire format (defaults to 3).
        ``ValueError`` is raised on mismatch.

    Returns
    -------
    QPose14PackedPayload
        Container with the parsed sections + section count.

    Raises
    ------
    ValueError
        On truncated header, section-count mismatch, truncated body,
        or trailing bytes.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    payload = bytes(payload)
    if len(payload) < 4:
        raise ValueError(
            f"truncated header: need 4 bytes have {len(payload)}"
        )
    (n_members,) = struct.unpack_from("<I", payload, 0)
    if n_members != expected_n_members:
        raise ValueError(
            f"n_members {n_members} != expected_n_members {expected_n_members}"
        )
    o = 4
    sections: list[bytes] = []
    for i in range(n_members):
        if o + 4 > len(payload):
            raise ValueError(
                f"truncated section-{i} length prefix at offset {o}"
            )
        (sz,) = struct.unpack_from("<I", payload, o)
        o += 4
        if o + sz > len(payload):
            raise ValueError(
                f"truncated section-{i} body at offset {o}: "
                f"need {sz} bytes have {len(payload) - o}"
            )
        sections.append(bytes(payload[o : o + sz]))
        o += sz
    if o != len(payload):
        raise ValueError(
            f"trailing bytes: {o} consumed vs {len(payload)} total"
        )
    return QPose14PackedPayload(
        sections=tuple(sections),
        n_members=int(n_members),
        total_bytes=len(payload),
    )


__all__ = [
    "PR63_QPOSE14_N_DIMS",
    "PR63_SIGNED_SCALE",
    "PR63_VEL_BIAS",
    "PR63_VEL_SCALE",
    "QPose14PackedPayload",
    "QPose14Uint16Int16Stream",
    "decode_qpose14_uint16_view_int16",
    "encode_qpose14_uint16_view_int16",
    "pack_qpose14_packed_payload",
    "unpack_qpose14_packed_payload",
]
