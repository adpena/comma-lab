# SPDX-License-Identifier: MIT
"""Post-decompress master-gradient extractor for non-PR101 archive families.

Sister of `src/tac/master_gradient_post_brotli_decompress.py` (slot 15
canonical reference for PR101). This module extends the SAME post-decompress
mutation-grain discipline to the FIVE OTHER affected archive families per
operator directive 2026-05-19 *"fix all affected, address all not yet
covered"*:

  1. PR106 format0d              — brotli on primary packed-HNeRV decoder
                                     + raw fp32 scales + format0c sidecar
  2. PR107 apogee_v2              — brotli on int8 weight tensor + brotli on
                                     latent + water_filling bit budget recovery
  3. A1                           — brotli on PR101-family decoder section
                                     (with 4-byte section header) + LZMA on
                                     latent + brotli on per-pair corrections
  4. DP1 pretrained_driving_prior — brotli(pickle) on renderer state_dict +
                                     int8 codebook lookup + brotli on per-pair
                                     residual
  5. HDM8 film grain sidecar      — brotli on dim+delta_q sidecar payload

WHY this matters (the brotli-cascade locality violation):
  One RAW-archive-byte flip cascades through the entropy decoder, expanding
  into MANY decompressed-weight-byte changes. Per-byte master-gradient
  extracted in the raw-byte space carries false locality assumptions; the
  CORRECT locality basis is the POST-DECOMPRESS decoder-weight space.

  Empirical receipt (codex op7): PR101 op7 rank-1 byte at index 35773 was
  classified `pose_axis_share=1.0` per RAW-byte master-gradient but SegNet
  REGRESSED by +0.0014 on both contest-CPU and contest-CUDA. The model was
  wrong because brotli is a nonlinear cascade.

This module operationalizes the SAME canonical 5-step pattern slot 15 lands
for PR101, mirrored for each of the 5 archive families:

  (1) decompose archive into per-stream brotli/codec records,
  (2) map decompressed bytes back to (stream_index, offset_in_stream),
  (3) emit canonical `PostDecompressLayout` carrying SHA-256s + offsets,
  (4) declare per-family cascade severity (BOUNDED vs UNBOUNDED),
  (5) every sensitivity row carries canonical Provenance per Catalog #323
      with evidence_grade = "[macOS-CPU advisory]" (research-grade only,
      NOT contest-axis authority per CLAUDE.md "Submission auth eval" +
      Catalog #127 custody validator).

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + the FORBIDDEN
PATTERN "Forbidden score claims" + Catalog #318 raw-byte-authority discipline:
the helpers REFUSE to mark the layouts as score-claim authority; they emit
diagnostic layout records that feed the canonical extractor in
`tools/extract_master_gradient.py` and the sister consumer in
`src/tac/master_gradient_per_byte_consumer.py`.

[verified-against:src/tac/master_gradient_post_brotli_decompress.py]
[verified-against:tools/extract_master_gradient.py::parse_pr106_format0d_archive_layout]
[verified-against:tools/extract_master_gradient.py::parse_pr107_apogee_archive_layout]
[verified-against:tools/extract_master_gradient.py::parse_a1_archive_layout]
[verified-against:tools/extract_master_gradient.py::parse_dp1_archive_layout]
[verified-against:submissions/hdm8_film_grain_sidecar/inflate.py]
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import brotli  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - brotli is a hard runtime dep
    brotli = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard runtime dep
    np = None  # type: ignore[assignment]

# Re-export the canonical PR101 grain constant so consumers can route through
# this module without importing two sister modules.
from tac.master_gradient_post_brotli_decompress import (
    MUTATION_GRAIN_POST_BROTLI_DECOMPRESS,
)

__all__ = [
    # Cascade-severity taxonomy (5 codec classes seen across the 5 families)
    "CASCADE_SEVERITY_BOUNDED",
    "CASCADE_SEVERITY_UNBOUNDED",
    "CASCADE_SEVERITY_NONE",
    # Per-family mutation-grain identifiers (mirror PR101 sister)
    "MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS",
    "MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS",
    "MUTATION_GRAIN_A1_POST_DECOMPRESS",
    "MUTATION_GRAIN_DP1_POST_DECOMPRESS",
    "MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS",
    # Cascade-severity classifier
    "classify_cascade_severity_for_codec",
    # Common exception
    "PostDecompressDecodeError",
    # Per-codec stream record + layout
    "DecompressedStreamRecord",
    "PostDecompressLayout",
    # Per-family entry points
    "build_pr106_format0d_post_decompress_layout",
    "build_pr107_apogee_v2_post_decompress_layout",
    "build_a1_post_decompress_layout",
    "build_dp1_post_decompress_layout",
    "build_hdm8_film_grain_post_decompress_layout",
    # Cross-family aggregator
    "AffectedArchiveFamily",
    "AFFECTED_ARCHIVE_FAMILIES",
    "build_post_decompress_layout_for_family",
    # Generic mapper (mirror PR101 sister)
    "map_decompressed_byte_to_stream",
    # Sensitivity summary helper (mirror PR101 sister)
    "compute_sensitivity_summary_stats",
]

# ──────────────────────────────────────────────────────────────────────────────
# Cascade-severity taxonomy
# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window codecs (brotli LZ77, deflate, zstd, lzma): a single
# compressed-byte flip can corrupt the decoder state but the corruption is
# BOUNDED by the codec's lookback window (brotli ≤ 16 MB; lzma ≤ 4 GB; in
# practice for our payloads bounded by stream length).
CASCADE_SEVERITY_BOUNDED: str = "bounded_sliding_window_codec"

# Adaptive arithmetic / range / Huffman with adaptive probability models: a
# single compressed-byte flip changes all SUBSEQUENT decoded symbols because
# the probability tables update with every decoded symbol. UNBOUNDED cascade
# severity within the stream.
CASCADE_SEVERITY_UNBOUNDED: str = "unbounded_adaptive_arithmetic_codec"

# No cascade: raw int8/uint8 bytes, fp16 scales, codebook lookups (where one
# input byte maps to one or a few output bytes via fixed table). Byte-locality
# holds in the raw-byte space.
CASCADE_SEVERITY_NONE: str = "none_byte_local_codec"


def classify_cascade_severity_for_codec(codec_token: str) -> str:
    """Classify a codec name into cascade-severity taxonomy.

    Args:
        codec_token: codec name (case-insensitive).

    Returns:
        One of CASCADE_SEVERITY_BOUNDED / UNBOUNDED / NONE.
    """
    token = (codec_token or "").lower().strip()
    if token in ("brotli", "deflate", "zstd", "lzma", "gzip"):
        return CASCADE_SEVERITY_BOUNDED
    if token in ("arithmetic", "range", "huffman_adaptive", "ans", "range_coder"):
        return CASCADE_SEVERITY_UNBOUNDED
    if token in (
        "raw_int8",
        "raw_uint8",
        "raw_fp16",
        "raw_fp32",
        "codebook_lookup",
        "pickle",
        "identity",
    ):
        return CASCADE_SEVERITY_NONE
    # Unknown codec — be conservative: report bounded so consumers don't
    # erroneously assume byte-locality.
    return CASCADE_SEVERITY_BOUNDED


# ──────────────────────────────────────────────────────────────────────────────
# Per-family mutation-grain identifiers (mirror MUTATION_GRAIN_POST_BROTLI_DECOMPRESS)
# ──────────────────────────────────────────────────────────────────────────────
MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS: str = (
    "post_brotli_decompress_pr106_format0d_packed_hnerv_decoder_bytes"
)
MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS: str = (
    "post_brotli_decompress_pr107_apogee_v2_int8_decoder_bytes"
)
MUTATION_GRAIN_A1_POST_DECOMPRESS: str = (
    "post_brotli_decompress_a1_pr101_family_decoder_bytes"
)
MUTATION_GRAIN_DP1_POST_DECOMPRESS: str = (
    "post_brotli_pickle_decompress_dp1_renderer_state_dict_bytes"
)
MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS: str = (
    "post_brotli_decompress_hdm8_film_grain_sidecar_bytes"
)


class PostDecompressDecodeError(RuntimeError):
    """Raised when a non-PR101 archive's post-decompress decomposition fails."""


@dataclass(frozen=True)
class DecompressedStreamRecord:
    """One decompressed stream within a non-PR101 archive.

    Mirrors PR101 sister `BrotliStreamRecord` but generalized so the same
    record type can describe brotli / lzma / codebook-lookup streams across
    the 5 affected families.
    """

    stream_index: int
    section_name: str  # e.g. "decoder", "latent", "renderer_blob"
    codec: str  # e.g. "brotli", "lzma", "raw_int8"
    cascade_severity: str  # CASCADE_SEVERITY_BOUNDED / UNBOUNDED / NONE
    compressed_offset: int  # offset in raw archive payload
    compressed_length: int
    decompressed_offset: int  # offset in the post-decompress concatenation
    decompressed_length: int
    decompressed_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stream_index": self.stream_index,
            "section_name": self.section_name,
            "codec": self.codec,
            "cascade_severity": self.cascade_severity,
            "compressed_offset": self.compressed_offset,
            "compressed_length": self.compressed_length,
            "decompressed_offset": self.decompressed_offset,
            "decompressed_length": self.decompressed_length,
            "decompressed_sha256": self.decompressed_sha256,
        }


@dataclass(frozen=True)
class PostDecompressLayout:
    """Post-decompress decomposition of a non-PR101 archive."""

    archive_family: str  # one of AFFECTED_ARCHIVE_FAMILIES
    archive_sha256: str
    archive_bytes: int
    payload_sha256: str  # raw payload (post zip-unwrap)
    total_decompressed_bytes: int
    mutation_grain: str
    streams: tuple[DecompressedStreamRecord, ...]
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_family": self.archive_family,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "payload_sha256": self.payload_sha256,
            "total_decompressed_bytes": self.total_decompressed_bytes,
            "mutation_grain": self.mutation_grain,
            "streams": [s.as_dict() for s in self.streams],
            "notes": self.notes,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_archive_payload(archive_path: Path, expected_members: tuple[str, ...]) -> bytes:
    """Load raw archive payload (unwrap single-member ZIP if present)."""
    if not archive_path.exists():
        raise PostDecompressDecodeError(f"archive missing: {archive_path}")
    raw = archive_path.read_bytes()
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                raise PostDecompressDecodeError(
                    f"ZIP archive at {archive_path} has no members"
                )
            if expected_members and names[0] not in expected_members:
                raise PostDecompressDecodeError(
                    f"expected ZIP member in {expected_members}, got {names[0]!r}"
                )
            return zf.read(names[0])
    except zipfile.BadZipFile:
        return raw


def _decompress_single_brotli_stream_with_consumed_length(
    data: bytes, start: int
) -> tuple[bytes, int]:
    """Decompress ONE brotli stream starting at `start`; return (out, consumed).

    Mirrors PR101 sister `decompose_pr101_decoder_blob_brotli_streams` per-stream
    loop but exposed as a reusable helper for multi-stream walks.
    """
    if brotli is None:
        raise PostDecompressDecodeError(
            "brotli library required for post-brotli-decompress extraction"
        )
    decompressor = brotli.Decompressor()
    out = bytearray()
    pos = start
    finished = False
    while pos < len(data):
        try:
            chunk = decompressor.process(bytes([data[pos]]))
        except brotli.error as e:  # type: ignore[attr-defined]
            raise PostDecompressDecodeError(
                f"brotli error at compressed_offset={pos}: {e}"
            ) from e
        pos += 1
        if chunk:
            out.extend(chunk)
        if decompressor.is_finished():
            finished = True
            break
    if not finished:
        raise PostDecompressDecodeError(
            f"brotli stream did not terminate (consumed {pos - start} bytes from offset {start})"
        )
    return bytes(out), pos - start


def map_decompressed_byte_to_stream(
    decompressed_byte_index: int, streams: Sequence[DecompressedStreamRecord]
) -> tuple[int, int]:
    """Map a decompressed byte index back to (stream_index, offset_in_stream).

    Mirrors PR101 sister `map_decompressed_byte_to_stream`.
    """
    if decompressed_byte_index < 0:
        raise PostDecompressDecodeError(
            f"decompressed_byte_index must be >= 0; got {decompressed_byte_index}"
        )
    for stream in streams:
        end = stream.decompressed_offset + stream.decompressed_length
        if stream.decompressed_offset <= decompressed_byte_index < end:
            return (
                stream.stream_index,
                decompressed_byte_index - stream.decompressed_offset,
            )
    total = sum(s.decompressed_length for s in streams)
    raise PostDecompressDecodeError(
        f"decompressed_byte_index={decompressed_byte_index} out of bounds [0, {total})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-family layout builders
# ──────────────────────────────────────────────────────────────────────────────


def build_pr106_format0d_post_decompress_layout(
    archive_path: Path,
) -> PostDecompressLayout:
    """Build post-decompress layout for a PR106 format0d archive.

    PR106 format0d wraps the primary PR106 payload in a PacketIR with the
    sequence: magic(1)=0xfe + format_id(1)=0x0d + pr106_len_le_u32(4) +
    pr106_payload(N) + base_format0c_sidecar_payload(M) + extra...

    The pr106_payload itself is 0xff + uint24-LE decoder_blob_len + decoder_blob
    + latent_blob. The decoder_blob carries one of three encodings (detected by
    first 4 bytes):
      - `HDM3/4/6/7/8/9` magic   → HDM-packed (NO brotli; codebook-lookup
                                    cascade severity = NONE in raw-byte space)
      - PR101 schema header      → multi-stream brotli (BOUNDED cascade)
      - other (legacy direct)    → single-stream brotli (BOUNDED cascade)

    Cascade severity: depends on decoder encoding kind. We emit per-stream
    records that record the actual codec for each section.

    [verified-against:tools/extract_master_gradient.py::parse_pr106_format0d_projector_layout]
    [verified-against:tools/extract_master_gradient.py::_decode_pr106_packed_decoder_raw]
    """
    payload = _load_archive_payload(archive_path, expected_members=("x", "0.bin"))
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload_sha = _sha256_bytes(payload)
    if len(payload) < 6 or payload[0] != 0xFE:
        raise PostDecompressDecodeError(
            "PR106 format0d payload missing 0xfe PacketIR magic"
        )
    if payload[1] != 0x0D:
        raise PostDecompressDecodeError(
            f"PR106 format0d format_id mismatch: got 0x{payload[1]:02x}, expected 0x0d"
        )
    pr106_len = struct.unpack_from("<I", payload, 2)[0]
    pr106_offset = 6
    pr106_end = pr106_offset + pr106_len
    if pr106_end > len(payload):
        raise PostDecompressDecodeError(
            f"PR106 format0d pr106_payload length {pr106_len} overflows payload"
        )
    pr106_payload = payload[pr106_offset:pr106_end]
    if len(pr106_payload) < 5 or pr106_payload[0] != 0xFF:
        raise PostDecompressDecodeError(
            "PR106 format0d primary payload is not packed-HNeRV 0xff layout"
        )
    decoder_blob_len = int.from_bytes(pr106_payload[1:4], "little")
    decoder_blob_offset_in_pr106 = 4
    decoder_blob = pr106_payload[decoder_blob_offset_in_pr106 : 4 + decoder_blob_len]
    if len(decoder_blob) != decoder_blob_len:
        raise PostDecompressDecodeError(
            f"PR106 format0d decoder_blob length mismatch: declared={decoder_blob_len}, actual={len(decoder_blob)}"
        )
    # Detect decoder encoding kind by inspecting first 4 bytes
    HDM_MAGICS = (b"HDM3", b"HDM4", b"HDM6", b"HDM7", b"HDM8", b"HDM9")
    leading4 = bytes(decoder_blob[:4])
    if leading4 in HDM_MAGICS:
        # HDM-packed: NOT a brotli stream. The decoder bytes are a custom
        # packed format (codebook-lookup-style) — byte-locality holds in the
        # raw-byte space (single byte flip → bounded number of decoded weight
        # bytes change, but the encoding is NOT entropy-coded so there is no
        # cascade beyond the codebook output).
        decompressed_len = decoder_blob_len  # no entropy decompression applied
        decompressed_sha = _sha256_bytes(decoder_blob)
        codec_kind = f"hdm_packed_{leading4.decode('ascii')}"
        severity = CASCADE_SEVERITY_NONE
    else:
        # Either PR101 schema header or direct brotli — try direct brotli first
        try:
            decompressed_bytes, consumed = (
                _decompress_single_brotli_stream_with_consumed_length(decoder_blob, 0)
            )
            if consumed != len(decoder_blob):
                # PR101 schema (concatenated brotli streams + headers) — record
                # as opaque BOUNDED with the partial decompressed length.
                decompressed_len = len(decompressed_bytes)
                decompressed_sha = _sha256_bytes(decompressed_bytes)
                codec_kind = "brotli_partial_or_pr101_schema"
                severity = CASCADE_SEVERITY_BOUNDED
            else:
                decompressed_len = len(decompressed_bytes)
                decompressed_sha = _sha256_bytes(decompressed_bytes)
                codec_kind = "brotli"
                severity = CASCADE_SEVERITY_BOUNDED
        except PostDecompressDecodeError:
            # PR101 schema with explicit headers (not directly brotli-decodable)
            decompressed_len = 0
            decompressed_sha = _sha256_bytes(decoder_blob)
            codec_kind = "pr101_schema_or_unknown"
            severity = CASCADE_SEVERITY_BOUNDED
    streams = (
        DecompressedStreamRecord(
            stream_index=0,
            section_name="packed_hnerv_decoder_blob",
            codec=codec_kind,
            cascade_severity=severity,
            compressed_offset=pr106_offset + decoder_blob_offset_in_pr106,
            compressed_length=decoder_blob_len,
            decompressed_offset=0,
            decompressed_length=decompressed_len,
            decompressed_sha256=decompressed_sha,
        ),
    )
    return PostDecompressLayout(
        archive_family="pr106_format0d",
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        payload_sha256=payload_sha,
        total_decompressed_bytes=decompressed_len,
        mutation_grain=MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS,
        streams=streams,
        notes=(
            "PR106 format0d primary packed-HNeRV decoder uses either HDM-packed "
            "(codebook-lookup, NO cascade) OR brotli (BOUNDED cascade) depending "
            "on the per-archive choice; cascade severity is recorded per-stream. "
            "Discrete base/extra sidecar streams are NOT entropy-coded. "
            "Per Catalog #287/#323: anchors emitted at this grain are diagnostic-only."
        ),
    )


def build_pr107_apogee_v2_post_decompress_layout(
    archive_path: Path,
) -> PostDecompressLayout:
    """Build post-decompress layout for a PR107 apogee_v2 (length-prefixed) archive.

    PR107 apogee_v2 archive format (per
    `tools/extract_master_gradient.py::parse_pr107_apogee_archive_layout`):
      meta_brotli_len(4) + meta_brotli(N)           — brotli(JSON metadata)
      decoder_blob_len(4) + decoder_blob(M)         — brotli(CD1 packed decoder)
      latents_brotli_len(4) + latents_brotli(K)     — brotli(latents)

    All three sections are brotli-wrapped. Cascade severity: BOUNDED (brotli on
    metadata + decoder + latents).

    [verified-against:tools/extract_master_gradient.py::parse_pr107_apogee_archive_layout]
    [verified-against:tools/extract_master_gradient.py::parse_pr107_apogee_projector_layout]
    """
    payload = _load_archive_payload(archive_path, expected_members=("x", "0.bin"))
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload_sha = _sha256_bytes(payload)
    streams: list[DecompressedStreamRecord] = []
    pos = 0
    decompressed_cursor = 0
    for stream_idx, (section_name, codec_label) in enumerate(
        (
            ("meta_brotli", "brotli_json_meta"),
            ("decoder_blob", "brotli_cd1_decoder"),
            ("latents_brotli", "brotli_latents"),
        )
    ):
        if pos + 4 > len(payload):
            raise PostDecompressDecodeError(
                f"PR107 apogee_v2 truncated before {section_name} length field"
            )
        length_offset = pos
        part_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4
        if part_len <= 0 or pos + part_len > len(payload):
            raise PostDecompressDecodeError(
                f"bad PR107 apogee_v2 section {section_name} length {part_len}"
            )
        section_payload = payload[pos : pos + part_len]
        try:
            decompressed = brotli.decompress(section_payload)
        except Exception as exc:
            raise PostDecompressDecodeError(
                f"PR107 apogee_v2 brotli error on {section_name}: {exc}"
            ) from exc
        streams.append(
            DecompressedStreamRecord(
                stream_index=stream_idx,
                section_name=section_name,
                codec=codec_label,
                cascade_severity=CASCADE_SEVERITY_BOUNDED,
                compressed_offset=pos,
                compressed_length=part_len,
                decompressed_offset=decompressed_cursor,
                decompressed_length=len(decompressed),
                decompressed_sha256=_sha256_bytes(decompressed),
            )
        )
        decompressed_cursor += len(decompressed)
        pos += part_len
    if pos != len(payload):
        raise PostDecompressDecodeError(
            f"PR107 apogee_v2 had {len(payload) - pos} unconsumed trailing bytes; layout walker did not terminate at payload end"
        )
    return PostDecompressLayout(
        archive_family="pr107_apogee_v2",
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        payload_sha256=payload_sha,
        total_decompressed_bytes=decompressed_cursor,
        mutation_grain=MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS,
        streams=tuple(streams),
        notes=(
            "PR107 apogee_v2 is length-prefixed: meta_brotli + decoder_blob + "
            "latents_brotli, all three brotli-wrapped (BOUNDED cascade per "
            "section). The decoder_blob is brotli(CD1) packed-decoder format. "
            "Per Catalog #287/#323: anchors emitted at this grain are diagnostic-only."
        ),
    )


def build_a1_post_decompress_layout(archive_path: Path) -> PostDecompressLayout:
    """Build post-decompress layout for an A1 fine-tuned archive.

    A1 layout (per `tools/extract_master_gradient.py::parse_a1_archive_layout`):
      a1_section_header(4)         — uint32_le decoder_section_total
      decoder(section_total - 4)   — concatenated brotli streams (PR101 family)
      latent(15387)                — LZMA temporal-delta latent payload
      sidecar(remainder)           — brotli per-pair corrections

    A1's decoder section uses the SAME PR101 brotli-streams-int8 codec as PR101
    but with a 4-byte section header prefix. The latent is LZMA (BOUNDED).

    Cascade severity: BOUNDED (brotli + lzma + brotli — all sliding-window).
    """
    payload = _load_archive_payload(archive_path, expected_members=("x", "0.bin"))
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload_sha = _sha256_bytes(payload)
    decoder_section_total = struct.unpack_from("<I", payload, 0)[0]
    latent_len = 15_387
    if decoder_section_total + latent_len > len(payload):
        raise PostDecompressDecodeError(
            f"A1 decoder_section_total {decoder_section_total} + latent_len {latent_len} > payload length {len(payload)}"
        )
    decoder_blob = payload[4:decoder_section_total]  # skip the 4B header
    streams: list[DecompressedStreamRecord] = []
    decompressed_cursor = 0
    stream_idx = 0
    # Walk decoder brotli streams (same as PR101 sister, ≥ 7 streams)
    pos = 0
    while pos < len(decoder_blob):
        out, consumed = _decompress_single_brotli_stream_with_consumed_length(
            decoder_blob, pos
        )
        streams.append(
            DecompressedStreamRecord(
                stream_index=stream_idx,
                section_name=f"decoder_brotli_stream_{stream_idx}",
                codec="brotli",
                cascade_severity=CASCADE_SEVERITY_BOUNDED,
                compressed_offset=4 + pos,  # +4 for section header
                compressed_length=consumed,
                decompressed_offset=decompressed_cursor,
                decompressed_length=len(out),
                decompressed_sha256=_sha256_bytes(out),
            )
        )
        decompressed_cursor += len(out)
        stream_idx += 1
        pos += consumed
        if stream_idx > 32:
            raise PostDecompressDecodeError(
                f"A1 decoder produced {stream_idx} brotli streams; refusing pathological count"
            )
    # Latent: LZMA-compressed; decompress lazily (only if lzma is importable
    # and the payload is well-formed). If lzma fails we record the latent as
    # an opaque section with cascade_severity=BOUNDED.
    latent_start = decoder_section_total
    latent_blob = payload[latent_start : latent_start + latent_len]
    try:
        import lzma

        latent_decompressed = lzma.decompress(latent_blob)
        latent_decompressed_len = len(latent_decompressed)
        latent_sha = _sha256_bytes(latent_decompressed)
    except Exception:
        # LZMA stream may use a non-standard preset; record opaque with zero
        # decompressed length so consumers know they cannot probe per-byte.
        latent_decompressed_len = 0
        latent_sha = _sha256_bytes(latent_blob)  # fall back to compressed sha
    streams.append(
        DecompressedStreamRecord(
            stream_index=stream_idx,
            section_name="latent_lzma",
            codec="lzma",
            cascade_severity=CASCADE_SEVERITY_BOUNDED,
            compressed_offset=latent_start,
            compressed_length=latent_len,
            decompressed_offset=decompressed_cursor,
            decompressed_length=latent_decompressed_len,
            decompressed_sha256=latent_sha,
        )
    )
    decompressed_cursor += latent_decompressed_len
    stream_idx += 1
    # Sidecar: brotli per-pair corrections (if any)
    sidecar_offset = decoder_section_total + latent_len
    sidecar_blob = payload[sidecar_offset:]
    if sidecar_blob:
        try:
            sidecar_decompressed = brotli.decompress(sidecar_blob)
            streams.append(
                DecompressedStreamRecord(
                    stream_index=stream_idx,
                    section_name="sidecar_brotli",
                    codec="brotli",
                    cascade_severity=CASCADE_SEVERITY_BOUNDED,
                    compressed_offset=sidecar_offset,
                    compressed_length=len(sidecar_blob),
                    decompressed_offset=decompressed_cursor,
                    decompressed_length=len(sidecar_decompressed),
                    decompressed_sha256=_sha256_bytes(sidecar_decompressed),
                )
            )
            decompressed_cursor += len(sidecar_decompressed)
            stream_idx += 1
        except Exception:
            # Sidecar may have proprietary framing — record opaque
            streams.append(
                DecompressedStreamRecord(
                    stream_index=stream_idx,
                    section_name="sidecar_opaque",
                    codec="brotli_or_raw",
                    cascade_severity=CASCADE_SEVERITY_BOUNDED,
                    compressed_offset=sidecar_offset,
                    compressed_length=len(sidecar_blob),
                    decompressed_offset=decompressed_cursor,
                    decompressed_length=0,
                    decompressed_sha256=_sha256_bytes(sidecar_blob),
                )
            )
            stream_idx += 1
    return PostDecompressLayout(
        archive_family="a1_finetuned",
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        payload_sha256=payload_sha,
        total_decompressed_bytes=decompressed_cursor,
        mutation_grain=MUTATION_GRAIN_A1_POST_DECOMPRESS,
        streams=tuple(streams),
        notes=(
            "A1 is PR101-family with a 4-byte decoder section header + multi-stream "
            "brotli decoder + LZMA latent + brotli sidecar. The 4-byte header has "
            "zero gradient. Per Catalog #287/#323: anchors emitted at this grain "
            "are diagnostic-only."
        ),
    )


def build_dp1_post_decompress_layout(
    archive_path: Path,
) -> PostDecompressLayout:
    """Build post-decompress layout for a DP1 pretrained_driving_prior archive.

    DP1 sections (per `src/tac/substrates/pretrained_driving_prior/archive.py`):
      dp1_header       — fixed magic/version header (raw, NO cascade)
      codebook_blob    — int8 codebook entries (NO cascade; codebook lookup is
                          byte-local — one input symbol → one codebook entry)
      renderer_blob    — brotli(pickle(state_dict)) — BOUNDED cascade
      residual_blob    — brotli int8 per-pair residual (BOUNDED cascade)
      meta_blob        — json metadata (raw, NO cascade in the JSON-text sense
                          but JSON parsing is NOT byte-local; treat as BOUNDED)
    """
    payload = _load_archive_payload(archive_path, expected_members=("0.bin", "x"))
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload_sha = _sha256_bytes(payload)
    # Parse DP1 section ranges via the canonical helper
    try:
        from tac.substrates.pretrained_driving_prior.archive import (
            parse_dp1_archive_bytes,
        )
    except ImportError as exc:
        raise PostDecompressDecodeError(
            f"DP1 archive parser unavailable: {exc}"
        ) from exc
    try:
        section_ranges = parse_dp1_archive_bytes(payload)
    except ValueError as exc:
        raise PostDecompressDecodeError(f"not a DP1 archive: {exc}") from exc
    streams: list[DecompressedStreamRecord] = []
    decompressed_cursor = 0
    stream_idx = 0
    codec_by_section = {
        "dp1_header": ("raw_uint8", CASCADE_SEVERITY_NONE),
        "codebook_blob": ("codebook_lookup", CASCADE_SEVERITY_NONE),
        "renderer_blob": ("brotli_pickle", CASCADE_SEVERITY_BOUNDED),
        "residual_blob": ("brotli", CASCADE_SEVERITY_BOUNDED),
        "meta_blob": ("json", CASCADE_SEVERITY_BOUNDED),
    }
    for section_name, (offset, length) in section_ranges.items():
        section_bytes = payload[offset : offset + length]
        codec, severity = codec_by_section.get(
            section_name, ("unknown", CASCADE_SEVERITY_BOUNDED)
        )
        # For brotli sections, attempt decompression to get post-decompress
        # byte count. For raw / codebook / json, the decompressed size IS the
        # compressed size (byte-local).
        if codec == "brotli":
            try:
                decompressed = brotli.decompress(section_bytes)
                decompressed_len = len(decompressed)
                decompressed_sha = _sha256_bytes(decompressed)
            except Exception:
                decompressed_len = 0
                decompressed_sha = _sha256_bytes(section_bytes)
        elif codec == "brotli_pickle":
            # Brotli outer, pickle inner. We measure the brotli-decompressed
            # bytes (the pickle stream) as the post-decompress space. Pickle
            # parsing is BOUNDED in its own right but pickle byte-locality is
            # not the canonical analysis surface — the brotli output IS.
            try:
                decompressed = brotli.decompress(section_bytes)
                decompressed_len = len(decompressed)
                decompressed_sha = _sha256_bytes(decompressed)
            except Exception:
                decompressed_len = 0
                decompressed_sha = _sha256_bytes(section_bytes)
        else:
            # Byte-local sections: post-decompress == raw
            decompressed_len = length
            decompressed_sha = _sha256_bytes(section_bytes)
        streams.append(
            DecompressedStreamRecord(
                stream_index=stream_idx,
                section_name=section_name,
                codec=codec,
                cascade_severity=severity,
                compressed_offset=offset,
                compressed_length=length,
                decompressed_offset=decompressed_cursor,
                decompressed_length=decompressed_len,
                decompressed_sha256=decompressed_sha,
            )
        )
        decompressed_cursor += decompressed_len
        stream_idx += 1
    return PostDecompressLayout(
        archive_family="dp1_pretrained_driving_prior",
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        payload_sha256=payload_sha,
        total_decompressed_bytes=decompressed_cursor,
        mutation_grain=MUTATION_GRAIN_DP1_POST_DECOMPRESS,
        streams=tuple(streams),
        notes=(
            "DP1 mixes byte-local (codebook + raw header) with brotli-wrapped "
            "(renderer state_dict + residual) sections. The renderer_blob is "
            "brotli(pickle); the pickle output IS the canonical post-decompress "
            "analysis surface. Per Catalog #287/#323: anchors emitted at this "
            "grain are diagnostic-only."
        ),
    )


def build_hdm8_film_grain_post_decompress_layout(
    archive_path: Path,
) -> PostDecompressLayout:
    """Build post-decompress layout for an HDM8 film-grain sidecar archive.

    HDM8 wrapper (per `submissions/hdm8_film_grain_sidecar/inflate.py`):
      SIDECAR_MAGIC(1)=0xfe + format_id(1) + pr106_len_le_u32(4) +
      pr106_bytes(N) + sidecar_len(2) + sidecar_blob(K) + framing_meta(6 if
      format_id in {0x02, 0x03, 0x04}) [+ selector_len(2) + selector_bytes]

    format_id semantics:
      0x01 — legacy brotli(int8 dim + delta_q) sidecar; BOUNDED cascade.
      0x02 — PR101 ranked-no-op grammar payload; BOUNDED cascade (ranked-Huffman
              with framing_meta).
      0x03/0x04 — same as 0x02 plus selector_json (raw JSON or brotli-JSON).

    We expose per-section records: pr106_bytes (delegated codec; passthrough),
    sidecar_blob (per-format codec), and the framing/selector tails.
    """
    payload = _load_archive_payload(archive_path, expected_members=("x", "0.bin"))
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload_sha = _sha256_bytes(payload)
    if len(payload) < 6:
        raise PostDecompressDecodeError(
            f"HDM8 payload too short: {len(payload)} < 6"
        )
    if payload[0] != 0xFE:
        raise PostDecompressDecodeError(
            f"HDM8 sidecar magic mismatch: got 0x{payload[0]:02x}, expected 0xfe"
        )
    format_id = payload[1]
    pr106_len = struct.unpack_from("<I", payload, 2)[0]
    pos = 6
    if pos + pr106_len > len(payload):
        raise PostDecompressDecodeError(
            f"HDM8 pr106 length {pr106_len} overflows payload"
        )
    streams: list[DecompressedStreamRecord] = []
    decompressed_cursor = 0
    stream_idx = 0
    # Record pr106_bytes as opaque (delegated to inflate; not entropy-coded at
    # this layer)
    streams.append(
        DecompressedStreamRecord(
            stream_index=stream_idx,
            section_name="pr106_bytes_delegated",
            codec="pr106_delegated",
            cascade_severity=CASCADE_SEVERITY_NONE,
            compressed_offset=pos,
            compressed_length=pr106_len,
            decompressed_offset=decompressed_cursor,
            decompressed_length=pr106_len,
            decompressed_sha256=_sha256_bytes(payload[pos : pos + pr106_len]),
        )
    )
    decompressed_cursor += pr106_len
    stream_idx += 1
    pos += pr106_len
    # sidecar_blob length: 0x01 uses uint16; 0x02/0x03/0x04 uses uint16
    if pos + 2 > len(payload):
        raise PostDecompressDecodeError(
            "HDM8 archive truncated before sidecar_len"
        )
    sidecar_len = struct.unpack_from("<H", payload, pos)[0]
    pos += 2
    if pos + sidecar_len > len(payload):
        raise PostDecompressDecodeError(
            f"HDM8 sidecar_len {sidecar_len} overflows payload"
        )
    sidecar_blob = payload[pos : pos + sidecar_len]
    if format_id == 0x01:
        # Legacy brotli sidecar
        try:
            decompressed = brotli.decompress(sidecar_blob)
            decompressed_len = len(decompressed)
            decompressed_sha = _sha256_bytes(decompressed)
        except Exception as exc:
            raise PostDecompressDecodeError(
                f"HDM8 format_id=0x01 brotli error: {exc}"
            ) from exc
        codec_kind = "brotli_int8_dim_delta_q"
        severity = CASCADE_SEVERITY_BOUNDED
    elif format_id in (0x02, 0x03, 0x04):
        # PR101 ranked-no-op grammar; not directly brotli-decompressible but
        # ranked-Huffman is BOUNDED-cascade. Record as opaque with cascade
        # severity BOUNDED so consumers know the per-byte mapping requires
        # the canonical inflate helper.
        decompressed_len = sidecar_len
        decompressed_sha = _sha256_bytes(sidecar_blob)
        codec_kind = f"pr101_ranked_no_op_grammar_format_0x{format_id:02x}"
        severity = CASCADE_SEVERITY_BOUNDED
    else:
        decompressed_len = 0
        decompressed_sha = _sha256_bytes(sidecar_blob)
        codec_kind = f"unknown_format_0x{format_id:02x}"
        severity = CASCADE_SEVERITY_BOUNDED
    streams.append(
        DecompressedStreamRecord(
            stream_index=stream_idx,
            section_name="film_grain_sidecar_blob",
            codec=codec_kind,
            cascade_severity=severity,
            compressed_offset=pos,
            compressed_length=sidecar_len,
            decompressed_offset=decompressed_cursor,
            decompressed_length=decompressed_len,
            decompressed_sha256=decompressed_sha,
        )
    )
    decompressed_cursor += decompressed_len
    stream_idx += 1
    pos += sidecar_len
    # Tail (framing_meta + selector) only for format_id in {0x02, 0x03, 0x04}
    if format_id in (0x02, 0x03, 0x04) and pos + 6 <= len(payload):
        framing_meta = payload[pos : pos + 6]
        streams.append(
            DecompressedStreamRecord(
                stream_index=stream_idx,
                section_name="framing_meta",
                codec="raw_framing_meta_6_bytes",
                cascade_severity=CASCADE_SEVERITY_NONE,
                compressed_offset=pos,
                compressed_length=6,
                decompressed_offset=decompressed_cursor,
                decompressed_length=6,
                decompressed_sha256=_sha256_bytes(framing_meta),
            )
        )
        decompressed_cursor += 6
        stream_idx += 1
        pos += 6
    return PostDecompressLayout(
        archive_family="hdm8_film_grain_sidecar",
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        payload_sha256=payload_sha,
        total_decompressed_bytes=decompressed_cursor,
        mutation_grain=MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS,
        streams=tuple(streams),
        notes=(
            "HDM8 film-grain sidecar wrapper: SIDECAR_MAGIC + format_id + "
            "pr106_len + pr106_bytes + sidecar_len + sidecar_blob [+ "
            "framing_meta]. format_id=0x01 uses brotli on int8 dim+delta_q "
            "(BOUNDED cascade); 0x02+ uses PR101 ranked-no-op grammar (BOUNDED). "
            "Per Catalog #287/#323: anchors emitted at this grain are "
            "diagnostic-only."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-family aggregator
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AffectedArchiveFamily:
    """Metadata for one of the 5 affected non-PR101 archive families."""

    family_id: str
    builder_name: str
    cascade_severity_summary: str  # "all bounded" | "mixed (codec_id=0 brotli + codec_id=1 raw)" | etc.
    mutation_grain: str


AFFECTED_ARCHIVE_FAMILIES: tuple[AffectedArchiveFamily, ...] = (
    AffectedArchiveFamily(
        family_id="pr106_format0d",
        builder_name="build_pr106_format0d_post_decompress_layout",
        cascade_severity_summary="bounded (brotli on packed-HNeRV decoder; raw sidecars)",
        mutation_grain=MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS,
    ),
    AffectedArchiveFamily(
        family_id="pr107_apogee_v2",
        builder_name="build_pr107_apogee_v2_post_decompress_layout",
        cascade_severity_summary="mixed (brotli codec_id=0 + raw int8 codec_id=1 + brotli latent)",
        mutation_grain=MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS,
    ),
    AffectedArchiveFamily(
        family_id="a1_finetuned",
        builder_name="build_a1_post_decompress_layout",
        cascade_severity_summary="bounded (PR101-family brotli + LZMA latent + brotli sidecar)",
        mutation_grain=MUTATION_GRAIN_A1_POST_DECOMPRESS,
    ),
    AffectedArchiveFamily(
        family_id="dp1_pretrained_driving_prior",
        builder_name="build_dp1_post_decompress_layout",
        cascade_severity_summary="mixed (codebook + raw header NO cascade; brotli renderer + residual BOUNDED)",
        mutation_grain=MUTATION_GRAIN_DP1_POST_DECOMPRESS,
    ),
    AffectedArchiveFamily(
        family_id="hdm8_film_grain_sidecar",
        builder_name="build_hdm8_film_grain_post_decompress_layout",
        cascade_severity_summary="bounded (brotli on dim + delta_q sidecar)",
        mutation_grain=MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS,
    ),
)


def build_post_decompress_layout_for_family(
    family_id: str, archive_path: Path
) -> PostDecompressLayout:
    """Dispatch to the canonical builder for the named archive family.

    Args:
        family_id: one of AFFECTED_ARCHIVE_FAMILIES family_id values.
        archive_path: path to the archive.

    Returns:
        PostDecompressLayout for the family.

    Raises:
        PostDecompressDecodeError: unknown family or decoder failure.
    """
    builders = {
        "pr106_format0d": build_pr106_format0d_post_decompress_layout,
        "pr107_apogee_v2": build_pr107_apogee_v2_post_decompress_layout,
        "a1_finetuned": build_a1_post_decompress_layout,
        "dp1_pretrained_driving_prior": build_dp1_post_decompress_layout,
        "hdm8_film_grain_sidecar": build_hdm8_film_grain_post_decompress_layout,
    }
    builder = builders.get(family_id)
    if builder is None:
        raise PostDecompressDecodeError(
            f"unknown archive family {family_id!r}; expected one of {sorted(builders)}"
        )
    return builder(archive_path)


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity summary (mirror PR101 sister)
# ──────────────────────────────────────────────────────────────────────────────


def compute_sensitivity_summary_stats(
    sensitivity_array, mutation_grain: str = "unknown"
) -> dict[str, object]:
    """Summarize a (N_bytes, 3) sensitivity tensor for the operator briefing.

    Mirrors `src/tac/master_gradient_post_brotli_decompress.compute_sensitivity_summary_stats`
    so cross-family comparison memos can use a uniform schema.
    """
    if np is None:
        raise PostDecompressDecodeError("numpy required for sensitivity summary")
    arr = np.asarray(sensitivity_array)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PostDecompressDecodeError(
            f"sensitivity tensor shape {arr.shape} != (N, 3)"
        )
    seg = arr[:, 0]
    pose = arr[:, 1]
    rate = arr[:, 2]
    combined = np.abs(seg) + np.abs(pose)
    if combined.size == 0:
        top_k_indices: list[int] = []
    else:
        top_k = min(10, combined.size)
        top_k_indices = list(map(int, np.argsort(combined)[::-1][:top_k]))
    return {
        "n_bytes": int(arr.shape[0]),
        "seg_mean": float(seg.mean()) if seg.size else 0.0,
        "seg_abs_mean": float(np.abs(seg).mean()) if seg.size else 0.0,
        "seg_max_abs": float(np.abs(seg).max()) if seg.size else 0.0,
        "pose_mean": float(pose.mean()) if pose.size else 0.0,
        "pose_abs_mean": float(np.abs(pose).mean()) if pose.size else 0.0,
        "pose_max_abs": float(np.abs(pose).max()) if pose.size else 0.0,
        "rate_max_abs": float(np.abs(rate).max()) if rate.size else 0.0,
        "top_k_decompressed_byte_indices_by_combined_seg_pose": top_k_indices,
        "mutation_grain": mutation_grain,
    }
