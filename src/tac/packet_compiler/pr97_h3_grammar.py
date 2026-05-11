"""PR97 H3 wire-format grammar — reusable byte-grammar primitives.

This module extracts the REUSABLE WIRE-FORMAT pieces from the PR97 public
submission (``vibe_coder_final_boss/inflate.py``) into typed primitives that
can be applied to any multi-section payload — not just the H3 mask/pose/model
layout PR97 itself uses.

Two primitives land here:

1. **Length-prefixed multi-section payload grammar**
   (:func:`encode_length_prefixed_sections` /
    :func:`decode_length_prefixed_sections`)

   PR97's ``split_payload`` packs N independent byte sections into a single
   blob via a sequence of ``[u32 len_i][bytes_i]`` pairs. Optional trailing
   sections (used for forward-compatibility — PR97's "sidecar slot") are
   tolerated when the consumer expects them. We surface this as a pure
   typed grammar: encoder takes a list of bytes, decoder takes a fixed
   minimum-section-count and yields all sections including any trailing
   optional ones.

2. **Tile-band multi-stream wire format**
   (:func:`encode_tile_band_streams` / :func:`decode_tile_band_streams`)

   PR97's ``decode_mask_range`` packs a 2D image into N bands × per-band
   W-splits independent compressed streams via the wire format
   ``[u32 n_chunks][u32 size_0][bytes_0]...[u32 size_{n-1}][bytes_{n-1}]``.
   PR97 uses this for 4 horizontal bands × per-band W-splits = 22 chunks.
   We surface it as a typed grammar that does NOT bake in the per-band
   split count: callers declare the chunk count, and the encoder/decoder
   round-trip a list of byte streams. (The 2D tile-reassembly logic is
   PR97-specific and stays in PR97; the wire format itself is generic.)

The actual range-coded mask compression in PR97 is performed by an external
C++ binary (``range_mask_codec.cpp``) which is NOT extracted here — only the
*wire format that wraps the per-stream bytes* is reusable as a packet-compiler
primitive. Callers can plug any per-stream codec into this grammar.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr97_intake_20260505_auto/source/submissions/vibe_coder_final_boss/inflate.py``
(SHA pinned in ``check_public_pr_intake_clones_pristine``-protected intake;
see also handoff
``~/Downloads/pact_score_lowering_handoff_2026-05-11.md`` P3 and the public
mechanism map ``.omx/research/public_pr95_plus_mechanism_map_20260511_codex.md``).

CLAUDE.md compliance:

* No scorer load — pure stdlib.
* No MPS / torch import.
* No ``/tmp`` paths anywhere.
* Frozen dataclass for the typed result; ``encode→decode`` is covered by
  focused Python conformance tests. Native ports must add golden vectors
  before promotion.
* OSS-friendly: public surface is the 5 names re-exported from
  ``tac.packet_compiler``; everything else is module-private (``_``-prefixed).

[empirical:src/tac/packet_compiler/golden_vectors/pr97_h3_length_prefixed_sections_v1.json
 + src/tac/packet_compiler/golden_vectors/pr97_h3_tile_band_streams_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
(byte-faithful port of public PR97 wire format; downstream archive-producing
consumers must run their own contest-CUDA + contest-CPU adjudication on the
exact archive bytes that ship).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class LengthPrefixedSectionPayload:
    """Result of decoding a length-prefixed multi-section payload."""

    sections: tuple[bytes, ...]
    total_bytes: int


@dataclass(frozen=True)
class TileBandStreamPayload:
    """Result of decoding a tile-band multi-stream wire-format payload."""

    streams: tuple[bytes, ...]
    n_chunks: int
    total_bytes: int


def encode_length_prefixed_sections(sections: list[bytes] | tuple[bytes, ...]) -> bytes:
    """Pack N byte sections into a single blob via ``[u32 len_i][bytes_i]`` pairs.

    The returned blob has NO global section-count prefix; the consumer must
    know how many sections to read (PR97 reads at least 3 — mask/pose/model
    — and tolerates an optional trailing 4th sidecar section).
    """
    out = bytearray()
    for s in sections:
        if not isinstance(s, (bytes, bytearray, memoryview)):
            raise TypeError("each section must be bytes-like")
        out += struct.pack("<I", len(s))
        out += bytes(s)
    return bytes(out)


def decode_length_prefixed_sections(
    blob: bytes,
    *,
    min_sections: int,
    max_sections: int | None = None,
) -> LengthPrefixedSectionPayload:
    """Decode a length-prefixed multi-section payload.

    The consumer declares ``min_sections`` (required) and optional
    ``max_sections`` (capped). Any sections beyond ``min_sections`` and up
    to ``max_sections`` (or until blob exhausted) are returned. Trailing
    bytes after the last consumed section trigger ``ValueError`` so silent
    corruption is impossible.
    """
    if max_sections is not None and max_sections < min_sections:
        raise ValueError("max_sections cannot be less than min_sections")
    sections: list[bytes] = []
    o = 0
    while o < len(blob):
        if max_sections is not None and len(sections) >= max_sections:
            break
        if o + 4 > len(blob):
            raise ValueError(
                f"truncated length prefix at offset {o}: need 4 bytes have {len(blob) - o}"
            )
        (n,) = struct.unpack_from("<I", blob, o)
        o += 4
        if o + n > len(blob):
            raise ValueError(
                f"truncated section {len(sections)} at offset {o}: need {n} bytes have {len(blob) - o}"
            )
        sections.append(bytes(blob[o : o + n]))
        o += n
    if len(sections) < min_sections:
        raise ValueError(
            f"only decoded {len(sections)} sections, required min {min_sections}"
        )
    if o != len(blob):
        raise ValueError(f"payload trailing bytes: {o} consumed vs {len(blob)} total")
    return LengthPrefixedSectionPayload(
        sections=tuple(sections),
        total_bytes=len(blob),
    )


def encode_tile_band_streams(streams: list[bytes] | tuple[bytes, ...]) -> bytes:
    """Pack N per-tile streams into ``[u32 n_chunks][u32 size_i][bytes_i]...``.

    PR97 uses this for 4 horizontal bands × per-band W-splits = 22 streams,
    one per (band, W-split) tile, where each stream is range-coded mask data
    for that tile. The tile reassembly is PR97-specific and stays out of
    this primitive.
    """
    out = bytearray()
    out += struct.pack("<I", len(streams))
    for s in streams:
        if not isinstance(s, (bytes, bytearray, memoryview)):
            raise TypeError("each stream must be bytes-like")
        out += struct.pack("<I", len(s))
        out += bytes(s)
    return bytes(out)


def decode_tile_band_streams(
    blob: bytes,
    *,
    expected_n_chunks: int | None = None,
) -> TileBandStreamPayload:
    """Decode a tile-band multi-stream wire-format payload.

    If ``expected_n_chunks`` is given, the decoded chunk count must match
    or ``ValueError`` is raised. Trailing bytes also raise.
    """
    if len(blob) < 4:
        raise ValueError(f"truncated header: need 4 bytes have {len(blob)}")
    (n_chunks,) = struct.unpack_from("<I", blob, 0)
    if expected_n_chunks is not None and n_chunks != expected_n_chunks:
        raise ValueError(
            f"chunk count mismatch: {n_chunks} != expected {expected_n_chunks}"
        )
    o = 4
    streams: list[bytes] = []
    for i in range(n_chunks):
        if o + 4 > len(blob):
            raise ValueError(
                f"truncated stream-{i} length prefix at offset {o}"
            )
        (sz,) = struct.unpack_from("<I", blob, o)
        o += 4
        if o + sz > len(blob):
            raise ValueError(
                f"truncated stream-{i} body at offset {o}: need {sz} bytes have {len(blob) - o}"
            )
        streams.append(bytes(blob[o : o + sz]))
        o += sz
    if o != len(blob):
        raise ValueError(f"payload trailing bytes: {o} consumed vs {len(blob)} total")
    return TileBandStreamPayload(
        streams=tuple(streams),
        n_chunks=n_chunks,
        total_bytes=len(blob),
    )
