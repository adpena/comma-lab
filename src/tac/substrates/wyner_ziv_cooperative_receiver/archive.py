# SPDX-License-Identifier: MIT
"""Wyner-Ziv cooperative-receiver archive grammar — WZ1 monolithic 0.bin.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (in single-zip-member ``0.bin`` slot
  when packaged in archive.zip)
* L4 inflate.py ≤ 200 LOC substrate-engineering waiver
* L8 deterministic (sorted-keys JSON, fp16 state_dict, fixed brotli quality)

The 4-section grammar:

::

    MAGIC(4)                     b"WZ1\\0"
    VERSION(1)                   u8 (== 1)
    NUM_PAIRS(2)                 u16
    HIDDEN_DIM(2)                u16
    NUM_HIDDEN_LAYERS(1)         u8
    SIDE_INFO_HIDDEN_DIM(2)      u16
    SIDE_INFO_NUM_LAYERS(1)      u8
    OUTPUT_HEIGHT(2)             u16
    OUTPUT_WIDTH(2)              u16
    POSE_DIM(1)                  u8
    COSET_INDEX_BITS(1)          u8
    RENDERER_LEN(4)              u32   brotli(renderer state_dict, fp16)
    SIDE_INFO_PRED_LEN(4)        u32   brotli(side-info predictor state_dict + pose_codes, fp16)
    COSET_INDICES_LEN(4)         u32   brotli(per-pair coset indices, packed by bits)
    META_LEN(4)                  u32   utf-8 JSON of float meta
    RENDERER_BLOB                ...
    SIDE_INFO_PRED_BLOB          ...
    COSET_INDICES_BLOB           ...
    META_BLOB                    ...

Sections:

* **RENDERER_BLOB**: brotli-compressed pickle of the trained renderer
  (FP16 state_dict, ~30 KB target).
* **SIDE_INFO_PRED_BLOB**: brotli-compressed pickle of the side-info
  predictor + per-pair pose codes (FP16 state_dict, ~18 KB target).
* **COSET_INDICES_BLOB**: brotli-compressed bit-packed per-pair coset
  indices (1 B per pair before brotli at ``coset_index_bits=8``; closes
  to ~ 600 B uncompressed, < 600 B after brotli).
* **META_BLOB**: JSON ``{"first_omega", "hidden_omega", "coord_feature_freqs",
  "wyner_ziv_dither_std", "search_grid_size", ...}`` — floats that don't
  fit in u8/u16 header fields.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, brotli quality fixed)
- No /tmp paths
- No scorer load
- No score claim
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

WZ1_MAGIC: bytes = b"WZ1\x00"
"""Wyner-Ziv archive magic (4 bytes, distinct from TT5L/SBO1/CMLR/etc.)."""

WZ1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + HIDDEN_DIM(2)
#              + NUM_HIDDEN_LAYERS(1) + SIDE_INFO_HIDDEN_DIM(2)
#              + SIDE_INFO_NUM_LAYERS(1) + OUTPUT_H(2) + OUTPUT_W(2)
#              + POSE_DIM(1) + COSET_INDEX_BITS(1)
#              + RENDERER_LEN(4) + SIDE_INFO_PRED_LEN(4)
#              + COSET_INDICES_LEN(4) + META_LEN(4)
# = 4+1+2+2+1+2+1+2+2+1+1+4+4+4+4 = 35 bytes
WZ1_HEADER_FMT: str = "<4sBHHBHBHHBBIIII"
WZ1_HEADER_SIZE: int = struct.calcsize(WZ1_HEADER_FMT)
assert WZ1_HEADER_SIZE == 35, (
    f"WZ1 header size invariant: expected 35, got {WZ1_HEADER_SIZE}"
)

# Brotli quality (deterministic at 9 — matches TT5L / SIREN / sister substrates).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class WynerZivArchive:
    """Parsed WZ1 archive — the inflate-time data contract."""

    renderer_state_dict: dict[str, torch.Tensor]
    """Trained renderer parameters (FP16)."""

    side_info_predictor_state_dict: dict[str, torch.Tensor]
    """Trained side-info-predictor + per-pair pose codes (FP16)."""

    coset_indices: np.ndarray
    """Uint16 array shape ``(num_pairs,)`` (truncated to ``coset_index_bits``)."""

    meta: dict[str, object]
    """Sidecar JSON meta with hparams (omega, freqs, search grid, ...)."""

    schema_version: int
    num_pairs: int
    hidden_dim: int
    num_hidden_layers: int
    side_info_hidden_dim: int
    side_info_num_layers: int
    output_height: int
    output_width: int
    pose_dim: int
    coset_index_bits: int

    @property
    def num_cosets(self) -> int:
        return 1 << self.coset_index_bits


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict to deterministic fp16 tensor bytes.

    Pickle is too loose for contest archive custody: equivalent tensor values
    can produce byte-different blobs. The WZ1 wire format uses sorted keys,
    explicit shape metadata, and raw fp16 C-order bytes under an ``SDT1``
    header before the deterministic brotli pass.
    """
    out = bytearray(b"SDT1")
    out.extend(struct.pack("<I", len(sd)))
    for key in sorted(sd):
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"state_dict key too long: {key!r}")
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        shape = tuple(int(v) for v in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"state_dict tensor rank too high for {key!r}: {len(shape)}")
        data = tensor.numpy().tobytes(order="C")
        out.extend(struct.pack("<H", len(key_bytes)))
        out.extend(key_bytes)
        out.extend(struct.pack("<B", len(shape)))
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"state_dict tensor dim out of range for {key!r}: {dim}")
            out.extend(struct.pack("<I", dim))
        out.extend(struct.pack("<I", len(data)))
        out.extend(data)
    return bytes(brotli.compress(bytes(out), quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    if len(raw) < 8 or raw[:4] != b"SDT1":
        raise ValueError("state_dict blob missing SDT1 deterministic header")
    pos = 4
    (count,) = struct.unpack("<I", raw[pos : pos + 4])
    pos += 4
    sd: dict[str, torch.Tensor] = {}
    for _ in range(count):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated before key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        key_end = pos + key_len
        if key_end > len(raw):
            raise ValueError("state_dict blob truncated in key")
        key = raw[pos:key_end].decode("utf-8")
        pos = key_end
        if key in sd:
            raise ValueError(f"duplicate state_dict key in archive: {key!r}")
        if pos + 1 > len(raw):
            raise ValueError("state_dict blob truncated before rank")
        rank = raw[pos]
        pos += 1
        shape: list[int] = []
        for _dim in range(rank):
            if pos + 4 > len(raw):
                raise ValueError("state_dict blob truncated in shape")
            (dim,) = struct.unpack("<I", raw[pos : pos + 4])
            pos += 4
            shape.append(int(dim))
        if pos + 4 > len(raw):
            raise ValueError("state_dict blob truncated before tensor bytes")
        (data_len,) = struct.unpack("<I", raw[pos : pos + 4])
        pos += 4
        data_end = pos + data_len
        if data_end > len(raw):
            raise ValueError("state_dict blob truncated in tensor bytes")
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float16).itemsize
        if data_len != expected:
            raise ValueError(
                f"state_dict tensor byte count mismatch for {key!r}: "
                f"got {data_len}, expected {expected}"
            )
        arr = np.frombuffer(raw[pos:data_end], dtype=np.float16).reshape(tuple(shape)).copy()
        sd[key] = torch.from_numpy(arr)
        pos = data_end
    if pos != len(raw):
        raise ValueError(f"state_dict blob has {len(raw) - pos} trailing bytes")
    return sd


def _serialize_coset_indices(
    coset_indices: np.ndarray, *, num_pairs: int, coset_index_bits: int
) -> bytes:
    """Brotli-compress the per-pair coset index stream.

    For ``coset_index_bits <= 8`` we pack as uint8; for larger bit budgets
    we use uint16 (full 2 bytes per pair). Brotli closes any redundancy.
    """
    if coset_indices.shape != (num_pairs,):
        raise ValueError(
            f"coset_indices shape {coset_indices.shape} != ({num_pairs},)"
        )
    max_val = (1 << coset_index_bits) - 1
    if coset_indices.min() < 0 or coset_indices.max() > max_val:
        raise ValueError(
            f"coset_indices out of range [0, {max_val}]; "
            f"got [{coset_indices.min()}, {coset_indices.max()}]"
        )
    if coset_index_bits <= 8:
        arr = coset_indices.astype(np.uint8)
    else:
        arr = coset_indices.astype(np.uint16)
    return bytes(brotli.compress(np.ascontiguousarray(arr).tobytes(), quality=_BROTLI_QUALITY))


def _deserialize_coset_indices(
    blob: bytes, *, num_pairs: int, coset_index_bits: int
) -> np.ndarray:
    raw = brotli.decompress(blob)
    if coset_index_bits <= 8:
        arr = np.frombuffer(raw, dtype=np.uint8)
    else:
        arr = np.frombuffer(raw, dtype=np.uint16)
    if arr.shape != (num_pairs,):
        raise ValueError(
            f"coset_indices length mismatch: got {arr.shape} expected ({num_pairs},)"
        )
    return arr.astype(np.int64).copy()


def pack_archive(
    *,
    renderer_state_dict: dict[str, torch.Tensor],
    side_info_predictor_state_dict: dict[str, torch.Tensor],
    coset_indices: np.ndarray,
    meta: dict[str, object],
    num_pairs: int,
    hidden_dim: int,
    num_hidden_layers: int,
    side_info_hidden_dim: int,
    side_info_num_layers: int,
    output_height: int,
    output_width: int,
    pose_dim: int,
    coset_index_bits: int,
    schema_version: int = WZ1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained substrate + coset indices into WZ1 0.bin bytes."""
    if schema_version != WZ1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("hidden_dim", hidden_dim, 0xFFFF),
        ("num_hidden_layers", num_hidden_layers, 0xFF),
        ("side_info_hidden_dim", side_info_hidden_dim, 0xFFFF),
        ("side_info_num_layers", side_info_num_layers, 0xFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
        ("pose_dim", pose_dim, 0xFF),
        ("coset_index_bits", coset_index_bits, 0xFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of range [1, {max_v}]")

    renderer_blob = _serialize_state_dict(renderer_state_dict)
    side_info_blob = _serialize_state_dict(side_info_predictor_state_dict)
    coset_blob = _serialize_coset_indices(
        coset_indices, num_pairs=num_pairs, coset_index_bits=coset_index_bits
    )
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        WZ1_HEADER_FMT,
        WZ1_MAGIC,
        schema_version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        side_info_hidden_dim,
        side_info_num_layers,
        output_height,
        output_width,
        pose_dim,
        coset_index_bits,
        len(renderer_blob),
        len(side_info_blob),
        len(coset_blob),
        len(meta_bytes),
    )
    return header + renderer_blob + side_info_blob + coset_blob + meta_bytes


def parse_archive(blob: bytes) -> WynerZivArchive:
    """Parse WZ1 0.bin bytes back into a typed ``WynerZivArchive``."""
    if len(blob) < WZ1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {WZ1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        side_info_hidden_dim,
        side_info_num_layers,
        output_height,
        output_width,
        pose_dim,
        coset_index_bits,
        renderer_len,
        side_info_len,
        coset_len,
        meta_len,
    ) = struct.unpack(WZ1_HEADER_FMT, blob[:WZ1_HEADER_SIZE])

    if magic != WZ1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {WZ1_MAGIC!r})")
    if version != WZ1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    pos = WZ1_HEADER_SIZE
    renderer_blob = blob[pos : pos + renderer_len]
    pos += renderer_len
    side_info_blob = blob[pos : pos + side_info_len]
    pos += side_info_len
    coset_blob = blob[pos : pos + coset_len]
    pos += coset_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {pos} from header"
        )

    renderer_sd = _deserialize_state_dict(renderer_blob)
    side_info_sd = _deserialize_state_dict(side_info_blob)
    coset_indices = _deserialize_coset_indices(
        coset_blob,
        num_pairs=int(num_pairs),
        coset_index_bits=int(coset_index_bits),
    )
    meta = json.loads(meta_blob.decode("utf-8")) if meta_blob else {}

    return WynerZivArchive(
        renderer_state_dict=renderer_sd,
        side_info_predictor_state_dict=side_info_sd,
        coset_indices=coset_indices,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_hidden_layers),
        side_info_hidden_dim=int(side_info_hidden_dim),
        side_info_num_layers=int(side_info_num_layers),
        output_height=int(output_height),
        output_width=int(output_width),
        pose_dim=int(pose_dim),
        coset_index_bits=int(coset_index_bits),
    )


def parse_wz1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for WZ1 (Wyner-Ziv cooperative-receiver) grammar.

    Canonical section-offset parser for WZ1 inner-blob bytes. The returned
    mapping is the data contract consumed by:

    - :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
      section-aware Tier A density estimation)
    - :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
      dispatch — WZ1 auto-detection by ``b"WZ1\\x00"`` magic prefix)

    Returned sections (Tier A / Tier B targets):

    - ``wz1_header`` — 35-byte header (control_or_metadata; fixed layout)
    - ``renderer_blob`` — brotli q=9 compressed renderer state_dict
      (decoder_weight_stream — the inflate-time renderer for Wyner-Ziv
      cooperative-receiver class)
    - ``side_info_predictor_blob`` — brotli q=9 compressed side-info predictor
      + per-pair pose codes
      (sidecar_or_correction_stream — DISCUS-style side-information at receiver)
    - ``coset_indices_blob`` — brotli-compressed bit-packed per-pair coset
      indices (entropy_model_or_range_stream — coset coding is the
      Wyner-Ziv rate-distortion-with-side-info entropy primitive)
    - ``meta_blob`` — sorted-keys JSON utf-8 (control_or_metadata)

    The byte ranges returned here MUST agree with the writer in
    :func:`pack_archive` and with the canonical full-decode parser
    :func:`parse_archive`. The single-source-of-truth for WZ1 byte layout
    is :data:`WZ1_HEADER_FMT` + :data:`WZ1_HEADER_SIZE`.

    Differs from :func:`parse_archive` in that this function returns
    section-offset tuples only (no torch state_dict deserialization). It is
    cheaper and is safe to call with brotli-tampered blobs (it never invokes
    ``brotli.decompress``).

    Raises ``ValueError`` on:

    - short header (< 35 bytes)
    - bad magic (!= ``b"WZ1\\x00"``)
    - unsupported schema version (!= 1)
    - archive size mismatch (declared end_meta != len(archive_bytes)),
      covering BOTH truncated archives and trailing-byte schema drift. The
      exact-equality contract matches :func:`parse_archive`.
    """
    if len(archive_bytes) < WZ1_HEADER_SIZE:
        raise ValueError(
            f"wz1 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {WZ1_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        _num_pairs,
        _hidden_dim,
        _num_hidden_layers,
        _side_info_hidden_dim,
        _side_info_num_layers,
        _output_h,
        _output_w,
        _pose_dim,
        _coset_index_bits,
        renderer_len,
        side_info_len,
        coset_len,
        meta_len,
    ) = struct.unpack(WZ1_HEADER_FMT, archive_bytes[:WZ1_HEADER_SIZE])
    if magic != WZ1_MAGIC:
        raise ValueError(
            f"wz1 archive: bad magic {magic!r} (expected {WZ1_MAGIC!r})"
        )
    if version != WZ1_SCHEMA_VERSION:
        raise ValueError(
            f"wz1 archive: unsupported schema version {version} "
            f"(expected {WZ1_SCHEMA_VERSION})"
        )
    end_header = WZ1_HEADER_SIZE
    end_renderer = end_header + int(renderer_len)
    end_side_info = end_renderer + int(side_info_len)
    end_coset = end_side_info + int(coset_len)
    end_meta = end_coset + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"wz1 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    return {
        "wz1_header": (0, WZ1_HEADER_SIZE),
        "renderer_blob": (end_header, int(renderer_len)),
        "side_info_predictor_blob": (end_renderer, int(side_info_len)),
        "coset_indices_blob": (end_side_info, int(coset_len)),
        "meta_blob": (end_coset, int(meta_len)),
    }


# Canonical optimization-role mapping for WZ1 sections (consumed by
# tac.analysis.scorer_conditional_mdl and tac.analysis.hnerv_packet_sections).
# Mirrors the role taxonomy in ``ROLE_WEIGHTS`` / sister-substrate maps.
#
# Note: Wyner-Ziv class-shift design uses coset coding as the entropy model
# (rate-distortion-with-side-info primitive); the side-info predictor is
# a correction stream from the receiver's side-information channel.
WZ1_SECTION_ROLES: dict[str, str] = {
    "wz1_header": "control_or_metadata",
    "renderer_blob": "decoder_weight_stream",
    "side_info_predictor_blob": "sidecar_or_correction_stream",
    "coset_indices_blob": "entropy_model_or_range_stream",
    "meta_blob": "control_or_metadata",
}


__all__ = [
    "WZ1_HEADER_FMT",
    "WZ1_HEADER_SIZE",
    "WZ1_MAGIC",
    "WZ1_SCHEMA_VERSION",
    "WZ1_SECTION_ROLES",
    "WynerZivArchive",
    "pack_archive",
    "parse_archive",
    "parse_wz1_archive_bytes",
]
