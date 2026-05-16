# SPDX-License-Identifier: MIT
"""STC v2 archive grammar — monolithic single-file ``0.bin`` (Catalog #146).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 3
(archive grammar must be a monolithic single-file ``0.bin`` with fixed
offsets declared in source) the stc_v2 substrate ships its bytes inside a
single-file payload with a small header that the contest inflate.py parses.

Wire format (little-endian):

    magic                  : 4 bytes  = b"STC2"
    version                : 2 bytes  uint16 = 1
    output_height          : 2 bytes  uint16 = 874  (contest camera resolution)
    output_width           : 2 bytes  uint16 = 1164 (contest camera resolution)
    num_pairs              : 4 bytes  uint32
    stcb_len               : 8 bytes  uint64
    renderer_bin_len       : 8 bytes  uint64
    poses_pt_len           : 8 bytes  uint64
    --- payload (concatenated) ---
    stcb_blob              : stcb_len bytes (STCB v1)
    renderer_bin_blob      : renderer_bin_len bytes (Lane A renderer.bin)
    poses_pt_blob          : poses_pt_len bytes (Lane A optimized_poses.pt)

Per the design memo Section 2.2.7 STC bytes replace the AV1 monochrome masks
slot of an existing archive. The simplest possible archive: bundle the STC
masks + the Lane A renderer + the Lane A poses inside one ``0.bin`` so the
inflate runtime is dependency-closure of (numpy + torch + pyav) only.

Strict-scorer-rule compliance: NO scorer / SegNet / PoseNet bytes ship in the
archive (the renderer.bin is the LEARNED-RENDERER, not the scorer; per
Yousfi's PR #35 rule the renderer is contest-permitted while scorer weights
are not).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

STC_V2_MAGIC: bytes = b"STC2"
STC_V2_VERSION: int = 1

# Fixed offsets per Catalog #146 / HNeRV parity L3.
_HEADER_LEN: int = (
    len(STC_V2_MAGIC)  # 4
    + 2  # version
    + 2  # output_height
    + 2  # output_width
    + 4  # num_pairs
    + 8  # stcb_len
    + 8  # renderer_bin_len
    + 8  # poses_pt_len
)  # = 38 bytes

CONTEST_OUTPUT_HW: tuple[int, int] = (874, 1164)


@dataclass(frozen=True)
class StcV2Archive:
    """Parsed STC v2 archive (the inflate-side view of ``0.bin``)."""

    version: int
    output_height: int
    output_width: int
    num_pairs: int
    stcb_blob: bytes
    renderer_bin_blob: bytes
    poses_pt_blob: bytes


def build_stc_v2_archive_bytes(
    *,
    stcb_blob: bytes,
    renderer_bin_blob: bytes,
    poses_pt_blob: bytes,
    num_pairs: int,
    output_height: int = CONTEST_OUTPUT_HW[0],
    output_width: int = CONTEST_OUTPUT_HW[1],
) -> bytes:
    """Pack the STC v2 archive payload into a single ``0.bin`` byte string.

    Args:
        stcb_blob: STCB v1 bytes from ``encode_stc_v2_masks``.
        renderer_bin_blob: Lane A ``renderer.bin`` bytes (pulled from anchor).
        poses_pt_blob: Lane A ``optimized_poses.pt`` bytes (pulled from anchor).
        num_pairs: number of frame-pairs the masks decode to (typically 600).
        output_height/output_width: contest camera resolution (default 874x1164).

    Returns:
        Single ``0.bin`` byte string containing the STC v2 header + payload.
    """
    if output_height > 65535 or output_width > 65535:
        raise ValueError(
            f"output dims must fit in uint16; got ({output_height}, {output_width})"
        )
    if num_pairs > 0xFFFFFFFF:
        raise ValueError(f"num_pairs must fit in uint32; got {num_pairs}")

    header = (
        STC_V2_MAGIC
        + struct.pack("<H", STC_V2_VERSION)
        + struct.pack("<H", output_height)
        + struct.pack("<H", output_width)
        + struct.pack("<I", num_pairs)
        + struct.pack("<Q", len(stcb_blob))
        + struct.pack("<Q", len(renderer_bin_blob))
        + struct.pack("<Q", len(poses_pt_blob))
    )
    assert len(header) == _HEADER_LEN, f"header LOC drift: {len(header)} != {_HEADER_LEN}"
    return header + stcb_blob + renderer_bin_blob + poses_pt_blob


def parse_stc_v2_archive(bin_bytes: bytes) -> StcV2Archive:
    """Parse an STC v2 ``0.bin`` payload back into its component blobs.

    Raises:
        ValueError: magic / version / length mismatch.
    """
    if len(bin_bytes) < _HEADER_LEN:
        raise ValueError(
            f"STC v2 archive too short: {len(bin_bytes)} < {_HEADER_LEN}"
        )
    if bin_bytes[: len(STC_V2_MAGIC)] != STC_V2_MAGIC:
        raise ValueError(
            f"STC v2 magic mismatch: {bin_bytes[: len(STC_V2_MAGIC)]!r} "
            f"!= {STC_V2_MAGIC!r}"
        )

    offset = len(STC_V2_MAGIC)
    (version,) = struct.unpack("<H", bin_bytes[offset : offset + 2])
    offset += 2
    if version != STC_V2_VERSION:
        raise ValueError(
            f"STC v2 version mismatch: got {version} expected {STC_V2_VERSION}"
        )

    (output_height,) = struct.unpack("<H", bin_bytes[offset : offset + 2])
    offset += 2
    (output_width,) = struct.unpack("<H", bin_bytes[offset : offset + 2])
    offset += 2
    (num_pairs,) = struct.unpack("<I", bin_bytes[offset : offset + 4])
    offset += 4
    (stcb_len,) = struct.unpack("<Q", bin_bytes[offset : offset + 8])
    offset += 8
    (renderer_bin_len,) = struct.unpack("<Q", bin_bytes[offset : offset + 8])
    offset += 8
    (poses_pt_len,) = struct.unpack("<Q", bin_bytes[offset : offset + 8])
    offset += 8

    expected_total = offset + stcb_len + renderer_bin_len + poses_pt_len
    if len(bin_bytes) != expected_total:
        raise ValueError(
            f"STC v2 archive length mismatch: got {len(bin_bytes)} "
            f"expected {expected_total} (header={offset} + stcb={stcb_len} "
            f"+ renderer={renderer_bin_len} + poses={poses_pt_len})"
        )

    stcb_blob = bin_bytes[offset : offset + stcb_len]
    offset += stcb_len
    renderer_bin_blob = bin_bytes[offset : offset + renderer_bin_len]
    offset += renderer_bin_len
    poses_pt_blob = bin_bytes[offset : offset + poses_pt_len]
    offset += poses_pt_len

    return StcV2Archive(
        version=version,
        output_height=output_height,
        output_width=output_width,
        num_pairs=num_pairs,
        stcb_blob=stcb_blob,
        renderer_bin_blob=renderer_bin_blob,
        poses_pt_blob=poses_pt_blob,
    )


__all__ = [
    "CONTEST_OUTPUT_HW",
    "STC_V2_MAGIC",
    "STC_V2_VERSION",
    "StcV2Archive",
    "build_stc_v2_archive_bytes",
    "parse_stc_v2_archive",
]
