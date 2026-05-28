# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec archive grammar (L0 SCAFFOLD).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L3
(monolithic single-file ``0.bin``) + L4 (≤200 LOC inflate runtime + ≤2 deps).
At L0 the archive grammar is DECLARED via :data:`WZPSC01_MAGIC` +
:data:`HEADER_FORMAT` + section-offset layout; the L1 trainer emits real
archive bytes per this grammar.

Archive grammar (WZPSC01 = Wyner-Ziv Pipeline-Stage Codec v01)
==============================================================

The monolithic single-file ``0.bin`` payload layout:

::

    +--------+------+--------------+---------------+-----------+------+
    | MAGIC  | VER  | OFFSET_TBL   | META_JSON_LEN | META_JSON | DATA |
    +--------+------+--------------+---------------+-----------+------+
       4 B    1 B     4 * 4 B        4 B                Z          *

* ``MAGIC`` = ``b"WZPS"`` (4 ASCII bytes; sister of D1 ``WZF0`` + DP1 ``DP1\\x00``)
* ``VER`` = 1 (1 byte; this is WZPSC01 v01)
* ``OFFSET_TBL`` = (main_offset, main_len, side_offset, side_len) as 4 big-endian
  uint32 (4 * 4 = 16 B); offsets relative to start of DATA section
* ``META_JSON_LEN`` = uint32 BE byte count of META_JSON section
* ``META_JSON`` = sorted-keys JSON dict per CLAUDE.md "Beauty, simplicity, and
  developer experience" + Catalog #128 fcntl-locked JSONL discipline mirror;
  fields: ``intercept_location`` / ``side_info_source`` / ``main_codec`` /
  ``compression_codec_for_side`` / ``schema_version`` / ``commit_sha`` /
  ``substrate_id`` / ``lane_id``
* ``DATA`` = concatenation of main_compressed bytes + side_compressed_baked
  bytes at the offsets declared in OFFSET_TBL

Y derivation at inflate time
----------------------------

The side_info_y is NOT in the archive; it is re-derived at inflate time per
the canonical primitive's ``side_info_source`` taxonomy. For ``"Comma2k19"``
side_info_source the inflate runtime routes through
``tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache``
per Catalog #213 + #245 canonical 4-layer pattern. For ``"torch_defaults"``
or ``"math_constants"`` Y is derived from canonical constants baked into
inflate.py per HNeRV parity L4 ≤200 LOC budget.

Per CLAUDE.md "Strict scorer rule": ``side_info_source='scorer_compressed'``
is FORBIDDEN unless explicit operator attestation per the primitive's
``WynerZivLayerConfig.operator_attested_scorer_side_info`` + the sister
strict gate Catalog #320.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from typing import Any


__all__ = (
    "WZPSC01_MAGIC",
    "WZPSC01_VERSION",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "ArchiveLayout",
    "encode_archive_bytes_scaffold",
    "decode_archive_bytes_scaffold",
    "L0_SCAFFOLD_ARCHIVE_NOT_IMPLEMENTED_MESSAGE",
)


WZPSC01_MAGIC: bytes = b"WZPS"
WZPSC01_VERSION: int = 1

# Header: MAGIC(4) + VER(B) + OFFSET_TBL(4 * I = 16) + META_JSON_LEN(I) = 25 B fixed
HEADER_FORMAT: str = ">4sB IIII I"
HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)


L0_SCAFFOLD_ARCHIVE_NOT_IMPLEMENTED_MESSAGE = (
    "Wyner-Ziv pipeline-stage codec L0 SCAFFOLD: archive encode/decode is "
    "DECLARED via WZPSC01_MAGIC + HEADER_FORMAT + ArchiveLayout but the L1 "
    "trainer's real archive emission is council-gated pending L1 build per "
    "the sister design memo. Use trainer.py --smoke for the L0 roundtrip "
    "demonstration; the L1 archive emission lands paired CUDA+CPU auth-eval "
    "per Catalog #246."
)


@dataclasses.dataclass(frozen=True)
class ArchiveLayout:
    """Frozen layout of a WZPSC01 archive payload.

    Per Catalog #305 ``decomposable_per_signal`` + ``cite_able`` facets: the
    layout exposes per-section byte counts + the canonical
    (intercept_location, side_info_source, main_codec, compression_codec_for_side)
    tuple so a reviewer can decompose the archive into Provenance-eligible
    sub-payloads.
    """

    main_offset: int
    main_len: int
    side_offset: int
    side_len: int
    meta_json_len: int
    intercept_location: str
    side_info_source: str
    main_codec: str
    compression_codec_for_side: str
    schema_version: str = "wzpsc01_v1"

    def to_meta_json_bytes(self) -> bytes:
        """Serialize the meta JSON section per the canonical sorted-keys schema."""
        meta = {
            "intercept_location": self.intercept_location,
            "side_info_source": self.side_info_source,
            "main_codec": self.main_codec,
            "compression_codec_for_side": self.compression_codec_for_side,
            "schema_version": self.schema_version,
        }
        return json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_archive_bytes_scaffold(
    *,
    main_compressed: bytes,
    side_compressed_baked: bytes,
    intercept_location: str,
    side_info_source: str,
    main_codec: str,
    compression_codec_for_side: str,
) -> bytes:
    """Encode WZPSC01 archive payload bytes per the canonical grammar.

    Per HNeRV parity L3 monolithic single-file: the entire archive is a single
    contiguous byte string ready to be written to ``0.bin`` inside the contest
    archive.zip. The archive.zip wrapper (with ZipFile.writestr deterministic
    timestamp) is the responsibility of the L1 trainer + the canonical
    archive-builder helper per Catalog #146.

    Args:
        main_compressed: post-WZ-split + post-entropy-coded main stream.
        side_compressed_baked: post-WZ-split + post-entropy-coded side stream
            (baked into inflate.py per HNeRV parity L4 ≤200 LOC budget — this
            archive payload retains the side stream for canonical roundtrip
            verification + sister substrate composition).
        intercept_location: where in the wrapped substrate's pipeline this
            WZ stage was inserted (per :class:`InterceptLocation`).
        side_info_source: canonical Y-derivation source.
        main_codec: codec for the main stream (post-WZ-split → archive).
        compression_codec_for_side: codec for the side stream
            (baked into inflate.py).

    Returns:
        The encoded WZPSC01 archive payload bytes ready to ship in ``0.bin``.
    """
    main_offset = 0
    main_len = len(main_compressed)
    side_offset = main_len
    side_len = len(side_compressed_baked)

    layout = ArchiveLayout(
        main_offset=main_offset,
        main_len=main_len,
        side_offset=side_offset,
        side_len=side_len,
        meta_json_len=0,  # filled below
        intercept_location=intercept_location,
        side_info_source=side_info_source,
        main_codec=main_codec,
        compression_codec_for_side=compression_codec_for_side,
    )
    meta_json = layout.to_meta_json_bytes()
    meta_json_len = len(meta_json)

    header = struct.pack(
        HEADER_FORMAT,
        WZPSC01_MAGIC,
        WZPSC01_VERSION,
        main_offset,
        main_len,
        side_offset,
        side_len,
        meta_json_len,
    )
    data = main_compressed + side_compressed_baked
    return header + meta_json + data


def decode_archive_bytes_scaffold(archive_bytes: bytes) -> dict[str, Any]:
    """Decode WZPSC01 archive payload bytes into (layout, main, side) parts.

    Args:
        archive_bytes: the WZPSC01 archive payload (start of ``0.bin``).

    Returns:
        Dict with ``layout`` (:class:`ArchiveLayout`) + ``main_compressed``
        bytes + ``side_compressed_baked`` bytes + ``meta`` (decoded JSON).

    Raises:
        ValueError: malformed magic bytes / version / offsets.
    """
    if len(archive_bytes) < HEADER_SIZE:
        raise ValueError(
            f"archive_bytes too short for WZPSC01 header: {len(archive_bytes)} B "
            f"< {HEADER_SIZE} B"
        )

    magic, ver, main_offset, main_len, side_offset, side_len, meta_json_len = struct.unpack(
        HEADER_FORMAT, archive_bytes[:HEADER_SIZE]
    )
    if magic != WZPSC01_MAGIC:
        raise ValueError(
            f"archive magic mismatch: expected {WZPSC01_MAGIC!r}; got {magic!r}"
        )
    if ver != WZPSC01_VERSION:
        raise ValueError(
            f"archive version mismatch: expected {WZPSC01_VERSION}; got {ver}"
        )

    meta_start = HEADER_SIZE
    meta_end = meta_start + meta_json_len
    if len(archive_bytes) < meta_end:
        raise ValueError(
            f"archive too short for meta JSON: need {meta_end} B; got {len(archive_bytes)} B"
        )
    meta_json_bytes = archive_bytes[meta_start:meta_end]
    meta = json.loads(meta_json_bytes.decode("utf-8"))

    data_start = meta_end
    main_start = data_start + main_offset
    main_end = main_start + main_len
    side_start = data_start + side_offset
    side_end = side_start + side_len

    if len(archive_bytes) < max(main_end, side_end):
        raise ValueError(
            f"archive too short for data section: need {max(main_end, side_end)} B; "
            f"got {len(archive_bytes)} B"
        )

    main_compressed = archive_bytes[main_start:main_end]
    side_compressed_baked = archive_bytes[side_start:side_end]

    layout = ArchiveLayout(
        main_offset=main_offset,
        main_len=main_len,
        side_offset=side_offset,
        side_len=side_len,
        meta_json_len=meta_json_len,
        intercept_location=meta["intercept_location"],
        side_info_source=meta["side_info_source"],
        main_codec=meta["main_codec"],
        compression_codec_for_side=meta["compression_codec_for_side"],
        schema_version=meta.get("schema_version", "wzpsc01_v1"),
    )
    return {
        "layout": layout,
        "main_compressed": main_compressed,
        "side_compressed_baked": side_compressed_baked,
        "meta": meta,
    }
