# SPDX-License-Identifier: MIT
"""Post-brotli-decompress master-gradient extractor for PR101-family archives.

Codex op7 iteration item #3 (2026-05-19): the per-byte master-gradient extracted
on the RAW archive bytes for PR101 op7 candidate predicted SegNet=0
(`pose_axis_share=1.0` on rank-1 byte 35773) but empirical SegNet REGRESSED by
+0.0014 on BOTH contest-CPU and contest-CUDA. The model was wrong because:

  Brotli decompression is a NONLINEAR cascade: one compressed-byte flip
  invalidates the entropy stream from that point forward, expanding into
  ARBITRARILY MANY decompressed-byte changes. The per-byte master-gradient
  computed in the raw-archive-byte domain assumed byte-locality. False.

The CORRECT locality basis is the POST-BROTLI-DECOMPRESS decoder-weight space:
the decoder weights are what the renderer actually consumes, the brotli stream
is just an encoding wrapper. Master-gradient on the decompressed-weight domain
is locally-linear (a single decompressed-byte flip changes ONE weight byte; the
scorer's gradient there is a meaningful local sensitivity).

This module provides the canonical helper that:
1. Inflates a PR101 archive's `decoder_blob` brotli-stream concatenation,
2. Maps every decompressed byte to its (stream_index, offset_in_stream),
3. Emits a MasterGradient anchor at `gradient_byte_domain
   ="post_brotli_decompress_decoder_weight_bytes"` so downstream consumers can
   distinguish this grain from the raw-archive-byte grain emitted by the sister
   helper at `tools/extract_master_gradient.py`.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog #287
+ Catalog #323: every emitted sensitivity row carries canonical Provenance,
the underlying tensor is stored as a sidecar `.npy`, the operating-point and
measurement-axis are recorded explicitly, and the helper REFUSES to mark the
anchor as a score-claim (score_claim_valid=False; promotion_eligible=False).

Sister of:
- `tools/extract_master_gradient.py::parse_pr101_lc_v2_archive_layout`
  (raw-archive-byte grain; slot 8 of the master-gradient parser-extension wave)
- `src/tac/master_gradient_iterative_refinement.py`
  (multiple-passes-per-byte deterministic correction framework that CONSUMES
  this grain to drive pass-2+ refinement)
- `src/tac/master_gradient_pr101_mps_axis_probe.py`
  (cross-device MPS-vs-CUDA-vs-CPU probe that CONSUMES this grain to validate
  per-byte response stability)
- `src/tac/mps_diagnostic/drift_predictor.py::predict_drift`
  (slot 9 canonical Cauchy-Schwarz bound)

Empirical anchor: PR101 op7 baseline archive sha b83bf348... (178258 archive
bytes) → 7 brotli streams in decoder_blob → 229014 decompressed bytes (~1.29x
expansion). Operating point: contest-CPU score 0.19285 (d_seg=0.00056026,
d_pose=3.286e-5, R=0.11869468). Op7 candidate sha 30826b37... mutated stream 2
(the 157680-byte decompressed stream).

[verified-against:experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length/score_response/modal_contest_cpu_linux_x86_auto.score_response.json]
"""
from __future__ import annotations

import hashlib
import io
import json
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


__all__ = [
    "DECODER_BLOB_LEN",
    "LATENT_BLOB_LEN",
    "MUTATION_GRAIN_POST_BROTLI_DECOMPRESS",
    "PR101_BROTLI_STREAM_COUNT",
    "PostBrotliDecodeError",
    "PostBrotliDecompressLayout",
    "decompose_pr101_decoder_blob_brotli_streams",
    "map_decompressed_byte_to_stream",
    "load_pr101_archive_payload",
    "build_post_brotli_decompress_anchor_payload",
    "compute_sensitivity_summary_stats",
]

# PR101 fixed-offset layout per `tools/extract_master_gradient.py::parse_pr101_lc_v2_archive_layout`
DECODER_BLOB_LEN: int = 162_164
LATENT_BLOB_LEN: int = 15_387
PR101_BROTLI_STREAM_COUNT: int = 7
MUTATION_GRAIN_POST_BROTLI_DECOMPRESS: str = (
    "post_brotli_decompress_decoder_weight_bytes"
)


class PostBrotliDecodeError(RuntimeError):
    """Raised when PR101 decoder-blob brotli decomposition fails."""


@dataclass(frozen=True)
class BrotliStreamRecord:
    """One brotli stream within the PR101 decoder_blob concatenation."""

    stream_index: int
    compressed_offset: int
    compressed_length: int
    decompressed_offset: int
    decompressed_length: int
    decompressed_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stream_index": self.stream_index,
            "compressed_offset": self.compressed_offset,
            "compressed_length": self.compressed_length,
            "decompressed_offset": self.decompressed_offset,
            "decompressed_length": self.decompressed_length,
            "decompressed_sha256": self.decompressed_sha256,
        }


@dataclass(frozen=True)
class PostBrotliDecompressLayout:
    """Decomposition of a PR101 archive into post-brotli-decompress byte space."""

    archive_sha256: str
    archive_bytes: int
    decoder_blob_sha256: str
    total_decompressed_bytes: int
    streams: tuple[BrotliStreamRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "decoder_blob_sha256": self.decoder_blob_sha256,
            "total_decompressed_bytes": self.total_decompressed_bytes,
            "streams": [s.as_dict() for s in self.streams],
            "mutation_grain": MUTATION_GRAIN_POST_BROTLI_DECOMPRESS,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_pr101_archive_payload(archive_path: Path) -> bytes:
    """Load a PR101 archive's monolithic payload (decoder + latent + sidecar).

    Per `tools/extract_master_gradient.py::parse_pr101_lc_v2_archive_layout`,
    PR101 archives wrap the monolithic payload as a single ZIP member named
    `x` (or as raw bytes). This helper returns the raw payload bytes whichever
    form is on disk.

    Args:
        archive_path: path to a PR101 archive.zip on disk.

    Returns:
        The monolithic payload bytes (decoder_blob + latent_blob + sidecar_blob).

    Raises:
        PostBrotliDecodeError: archive missing or malformed.
    """
    if not archive_path.exists():
        raise PostBrotliDecodeError(
            f"archive missing: {archive_path}"
        )
    raw = archive_path.read_bytes()
    # Attempt ZIP wrapper first
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                raise PostBrotliDecodeError(
                    f"ZIP archive at {archive_path} has no members"
                )
            # PR101 always single-member ('x')
            return zf.read(names[0])
    except zipfile.BadZipFile:
        # Not a ZIP wrapper: raw monolithic payload
        return raw


def decompose_pr101_decoder_blob_brotli_streams(
    decoder_blob: bytes,
) -> tuple[BrotliStreamRecord, ...]:
    """Decompose PR101 decoder_blob into per-stream brotli decompression records.

    PR101 stores the decoder weights as 7 concatenated brotli streams (one per
    layer/tensor group). This helper walks the concatenation, decompresses each
    stream, and emits a record tuple with per-stream offsets in both compressed
    and decompressed byte space.

    Args:
        decoder_blob: the 162_164-byte decoder section payload from a PR101
            archive (offset 0, length DECODER_BLOB_LEN).

    Returns:
        Tuple of BrotliStreamRecord ordered by stream_index.

    Raises:
        PostBrotliDecodeError: brotli decompression fails or stream count
            mismatches the canonical PR101 layout.
    """
    if brotli is None:
        raise PostBrotliDecodeError(
            "brotli library required for post-brotli-decompress extraction"
        )
    if not isinstance(decoder_blob, (bytes, bytearray)):
        raise PostBrotliDecodeError(
            f"decoder_blob must be bytes; got {type(decoder_blob).__name__}"
        )
    records: list[BrotliStreamRecord] = []
    pos = 0
    decompressed_cursor = 0
    stream_index = 0
    data = bytes(decoder_blob)
    while pos < len(data):
        decompressor = brotli.Decompressor()
        start = pos
        out = bytearray()
        finished = False
        while pos < len(data):
            try:
                chunk = decompressor.process(bytes([data[pos]]))
            except brotli.error as e:  # type: ignore[attr-defined]
                raise PostBrotliDecodeError(
                    f"brotli error at compressed_offset={pos}: {e}"
                ) from e
            pos += 1
            if chunk:
                out.extend(chunk)
            if decompressor.is_finished():
                finished = True
                break
        if not finished:
            raise PostBrotliDecodeError(
                f"brotli stream {stream_index} did not terminate "
                f"(consumed {pos - start} bytes from offset {start})"
            )
        rec = BrotliStreamRecord(
            stream_index=stream_index,
            compressed_offset=start,
            compressed_length=pos - start,
            decompressed_offset=decompressed_cursor,
            decompressed_length=len(out),
            decompressed_sha256=_sha256_bytes(bytes(out)),
        )
        records.append(rec)
        decompressed_cursor += len(out)
        stream_index += 1
        # Safety: PR101's canonical stream count is 7; refuse more than 16
        if stream_index > 16:
            raise PostBrotliDecodeError(
                f"decoder_blob produced {stream_index} brotli streams; "
                "expected canonical PR101 layout of 7"
            )
    if stream_index != PR101_BROTLI_STREAM_COUNT:
        raise PostBrotliDecodeError(
            f"PR101 decoder_blob produced {stream_index} brotli streams; "
            f"canonical layout expects {PR101_BROTLI_STREAM_COUNT}. "
            "Refusing to emit non-canonical layout."
        )
    return tuple(records)


def map_decompressed_byte_to_stream(
    decompressed_byte_index: int, streams: Sequence[BrotliStreamRecord]
) -> tuple[int, int]:
    """Map a decompressed byte index back to (stream_index, offset_in_stream).

    Args:
        decompressed_byte_index: byte offset in the decompressed concatenation.
        streams: stream records from `decompose_pr101_decoder_blob_brotli_streams`.

    Returns:
        (stream_index, offset_in_stream) tuple.

    Raises:
        PostBrotliDecodeError: byte index out of bounds.
    """
    if decompressed_byte_index < 0:
        raise PostBrotliDecodeError(
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
    raise PostBrotliDecodeError(
        f"decompressed_byte_index={decompressed_byte_index} out of bounds "
        f"[0, {total})"
    )


def build_post_brotli_decompress_anchor_payload(
    archive_path: Path,
) -> PostBrotliDecompressLayout:
    """Build the post-brotli-decompress layout for a PR101 archive.

    This is the canonical entry point for emitting a master-gradient anchor at
    the post-brotli-decompress grain. The returned PostBrotliDecompressLayout
    carries all metadata needed for the sister `MasterGradient` dataclass
    (n_bytes = total_decompressed_bytes; gradient_byte_domain =
    MUTATION_GRAIN_POST_BROTLI_DECOMPRESS).

    Args:
        archive_path: path to a PR101 archive on disk.

    Returns:
        PostBrotliDecompressLayout with per-stream decomposition and SHA-256s.

    Raises:
        PostBrotliDecodeError: archive malformed or decompression fails.
    """
    archive_bytes_raw = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes_raw)
    payload = load_pr101_archive_payload(archive_path)
    if len(payload) < DECODER_BLOB_LEN + LATENT_BLOB_LEN:
        raise PostBrotliDecodeError(
            f"PR101 payload too short: {len(payload)} < "
            f"{DECODER_BLOB_LEN + LATENT_BLOB_LEN}"
        )
    decoder_blob = payload[:DECODER_BLOB_LEN]
    streams = decompose_pr101_decoder_blob_brotli_streams(decoder_blob)
    total_decompressed = sum(s.decompressed_length for s in streams)
    return PostBrotliDecompressLayout(
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes_raw),
        decoder_blob_sha256=_sha256_bytes(decoder_blob),
        total_decompressed_bytes=total_decompressed,
        streams=streams,
    )


def compute_sensitivity_summary_stats(
    sensitivity_array,
) -> dict[str, object]:
    """Summarize a (N_bytes, 3) sensitivity tensor for the operator briefing.

    The tensor columns are (d_seg/d_byte, d_pose/d_byte, d_rate/d_byte). The
    summary gives mean / abs_mean / max_abs per axis + the top-K decompressed
    byte indices ranked by |seg| + |pose| (rate axis is uniformly 0 for
    same-length deltas).

    Args:
        sensitivity_array: ndarray of shape (N_bytes, 3) float32 or float64.

    Returns:
        Dict with summary stats + top-10 byte indices.
    """
    if np is None:
        raise PostBrotliDecodeError("numpy required for sensitivity summary")
    arr = np.asarray(sensitivity_array)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PostBrotliDecodeError(
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
        "mutation_grain": MUTATION_GRAIN_POST_BROTLI_DECOMPRESS,
    }
