"""Inflate-time runtime for the stack-of-stacks composition format.

This module is the contest-faithful sister of :mod:`compose`. It parses
the SOS1 magic-byte trailer at the end of the archive ``x`` blob, locates
each arm's offset, runs per-pair-best-of-K selection, and delegates to
the inner-substrate decoder for actual RGB rendering.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
lesson 4 (``inflate.py ≤ 100 LOC`` with 200 LOC substrate-engineering
exemption), this module stays ≤ 200 LOC of decoder logic. The actual
per-substrate decoders are imported lazily from the registered
substrate package (e.g. ``tac.substrates.a1_plus_lapose.inflate``).

Per Catalog #6 (no scorer load at inflate): this module NEVER loads
PoseNet or SegNet — the per-pair-best-of-K selector is precomputed at
TRAIN time and stored in the SOS1 selector; inflate just reads the
selector byte.
"""

from __future__ import annotations

import struct
from typing import Any

import brotli

SOS_SIDECAR_MAGIC = b"SOS1"
SOS_SIDECAR_VERSION = 1
SOS_HEADER_STRUCT = struct.Struct("<4sBBHBH2s")


def parse_sos_trailer(archive_bytes: bytes) -> dict[str, Any]:
    """Parse the SOS1 trailer from the END of archive_bytes.

    Returns a dict with:
        ``arm_concat_bytes``: bytes preceding the trailer (the concatenated
            per-arm composed payloads).
        ``selector``: bytes of length n_pairs (per-pair arm index).
        ``arm_meta``: deserialized JSON dict describing each arm's grammar.
        ``k``: number of arms.
        ``n_pairs``: number of pairs.
        ``layer_mask``: which composition layers are active.

    Raises:
        ValueError: archive_bytes does not end with a valid SOS1 trailer.
    """
    if len(archive_bytes) < SOS_HEADER_STRUCT.size:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} B) for SOS1 trailer"
        )
    idx = archive_bytes.rfind(SOS_SIDECAR_MAGIC)
    if idx < 0:
        raise ValueError("no SOS1 magic found")
    magic, version, layer_mask, n_pairs, k, meta_len, _reserved = (
        SOS_HEADER_STRUCT.unpack_from(archive_bytes, idx)
    )
    if magic != SOS_SIDECAR_MAGIC:
        raise ValueError(f"unexpected magic: {magic!r}")
    if version != SOS_SIDECAR_VERSION:
        raise ValueError(f"unsupported SOS1 version: {version}")

    selector_start = idx + SOS_HEADER_STRUCT.size
    selector_end = selector_start + n_pairs
    meta_end = selector_end + meta_len
    if meta_end != len(archive_bytes):
        raise ValueError(
            f"SOS1 length mismatch: trailer_end={meta_end} "
            f"archive_len={len(archive_bytes)}"
        )
    selector = bytes(archive_bytes[selector_start:selector_end])
    meta_brotli = bytes(archive_bytes[selector_end:meta_end])
    meta_json = brotli.decompress(meta_brotli)
    import json as _json  # local import keeps this module's API surface tight

    arm_meta = _json.loads(meta_json.decode("utf-8"))
    return {
        "arm_concat_bytes": bytes(archive_bytes[:idx]),
        "selector": selector,
        "arm_meta": arm_meta,
        "k": int(k),
        "n_pairs": int(n_pairs),
        "layer_mask": int(layer_mask),
    }


def slice_arm_bytes(parsed: dict[str, Any], arm_index: int) -> bytes:
    """Return the bytes belonging to arm ``arm_index`` from a parsed trailer.

    Uses the arm_offset / arm_length fields written by the composer.
    """
    arm_meta = parsed["arm_meta"]
    arms = arm_meta.get("arms")
    if not isinstance(arms, list) or arm_index < 0 or arm_index >= len(arms):
        raise ValueError(
            f"arm_index {arm_index} out of range; arm_meta has {len(arms or [])} arms"
        )
    arm = arms[arm_index]
    offset = int(arm.get("arm_offset", 0))
    length = int(arm.get("arm_length", 0))
    arm_concat = parsed["arm_concat_bytes"]
    end = offset + length
    if end > len(arm_concat):
        raise ValueError(
            f"arm {arm_index} bytes range [{offset}, {end}) exceeds arm_concat length "
            f"({len(arm_concat)})"
        )
    return bytes(arm_concat[offset:end])


def selector_for_pair(parsed: dict[str, Any], pair_index: int) -> int:
    """Return the arm index for ``pair_index`` from the selector.

    Args:
        parsed: dict returned by :func:`parse_sos_trailer`.
        pair_index: 0-based pair index in [0, n_pairs).

    Returns:
        Arm index ∈ ``[0, k)``.
    """
    selector = parsed["selector"]
    n_pairs = parsed["n_pairs"]
    k = parsed["k"]
    if pair_index < 0 or pair_index >= n_pairs:
        raise ValueError(
            f"pair_index {pair_index} out of range [0, {n_pairs})"
        )
    arm = int(selector[pair_index])
    if arm < 0 or arm >= k:
        raise ValueError(
            f"selector byte {arm} for pair {pair_index} out of range [0, {k})"
        )
    return arm


__all__ = [
    "SOS_HEADER_STRUCT",
    "SOS_SIDECAR_MAGIC",
    "SOS_SIDECAR_VERSION",
    "parse_sos_trailer",
    "selector_for_pair",
    "slice_arm_bytes",
]
