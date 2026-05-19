# SPDX-License-Identifier: MIT
"""Tests for tac.master_gradient_post_brotli_decompress.

Sister of `tools/extract_master_gradient.py` (slot 8 raw-archive-byte grain).
This module covers the post-brotli-decompress grain (codex op7 iteration #3).

[verified-against:experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip]
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tac.master_gradient_post_brotli_decompress import (
    DECODER_BLOB_LEN,
    LATENT_BLOB_LEN,
    MUTATION_GRAIN_POST_BROTLI_DECOMPRESS,
    PR101_BROTLI_STREAM_COUNT,
    BrotliStreamRecord,
    PostBrotliDecodeError,
    PostBrotliDecompressLayout,
    build_post_brotli_decompress_anchor_payload,
    compute_sensitivity_summary_stats,
    decompose_pr101_decoder_blob_brotli_streams,
    load_pr101_archive_payload,
    map_decompressed_byte_to_stream,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PR101_BASELINE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
PR101_OP7_CANDIDATE = (
    REPO_ROOT
    / "experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/archive.zip"
)


def test_canonical_constants_match_pr101_layout() -> None:
    assert DECODER_BLOB_LEN == 162_164
    assert LATENT_BLOB_LEN == 15_387
    assert PR101_BROTLI_STREAM_COUNT == 7
    assert (
        MUTATION_GRAIN_POST_BROTLI_DECOMPRESS
        == "post_brotli_decompress_decoder_weight_bytes"
    )


def test_brotli_stream_record_as_dict_roundtrip() -> None:
    rec = BrotliStreamRecord(
        stream_index=2,
        compressed_offset=1970,
        compressed_length=113227,
        decompressed_offset=2434,
        decompressed_length=157680,
        decompressed_sha256="a" * 64,
    )
    d = rec.as_dict()
    assert d["stream_index"] == 2
    assert d["compressed_offset"] == 1970
    assert d["decompressed_length"] == 157680
    assert d["decompressed_sha256"] == "a" * 64


def test_load_pr101_archive_payload_zip_wrapper() -> None:
    if not PR101_BASELINE.exists():
        pytest.skip("PR101 baseline archive not on disk")
    payload = load_pr101_archive_payload(PR101_BASELINE)
    # PR101 payload = decoder_blob(162164) + latent_blob(15387) + sidecar
    assert len(payload) >= DECODER_BLOB_LEN + LATENT_BLOB_LEN
    assert payload[:4] == b"\x1b\xcd\x03\xf8"  # first brotli stream prefix


def test_load_pr101_archive_payload_missing_raises() -> None:
    with pytest.raises(PostBrotliDecodeError):
        load_pr101_archive_payload(Path("/nonexistent/archive.zip"))


def test_decompose_pr101_decoder_blob_brotli_streams_canonical() -> None:
    if not PR101_BASELINE.exists():
        pytest.skip("PR101 baseline archive not on disk")
    payload = load_pr101_archive_payload(PR101_BASELINE)
    decoder_blob = payload[:DECODER_BLOB_LEN]
    streams = decompose_pr101_decoder_blob_brotli_streams(decoder_blob)
    # Per the canonical PR101 layout: 7 streams
    assert len(streams) == PR101_BROTLI_STREAM_COUNT
    # Stream 0 starts at compressed_offset=0
    assert streams[0].compressed_offset == 0
    # Streams are contiguous in both spaces
    for i in range(1, len(streams)):
        prev = streams[i - 1]
        cur = streams[i]
        assert cur.compressed_offset == prev.compressed_offset + prev.compressed_length
        assert cur.decompressed_offset == prev.decompressed_offset + prev.decompressed_length
    # Total compressed bytes equal the decoder_blob length
    total_comp = sum(s.compressed_length for s in streams)
    assert total_comp == DECODER_BLOB_LEN
    # Total decompressed bytes is much larger (brotli expansion); we expect
    # ~229014 bytes per the empirical anchor in the module docstring
    total_decomp = sum(s.decompressed_length for s in streams)
    assert total_decomp == 229_014


def test_decompose_refuses_non_bytes_input() -> None:
    with pytest.raises(PostBrotliDecodeError):
        decompose_pr101_decoder_blob_brotli_streams("not bytes")  # type: ignore[arg-type]


def test_decompose_refuses_corrupted_brotli_stream() -> None:
    # Random bytes are extremely unlikely to be valid brotli
    corrupt = b"\x00" * 100
    with pytest.raises(PostBrotliDecodeError):
        decompose_pr101_decoder_blob_brotli_streams(corrupt)


def test_map_decompressed_byte_to_stream_correct_dispatch() -> None:
    streams = (
        BrotliStreamRecord(
            stream_index=0,
            compressed_offset=0,
            compressed_length=782,
            decompressed_offset=0,
            decompressed_length=974,
            decompressed_sha256="a" * 64,
        ),
        BrotliStreamRecord(
            stream_index=1,
            compressed_offset=782,
            compressed_length=1188,
            decompressed_offset=974,
            decompressed_length=1460,
            decompressed_sha256="b" * 64,
        ),
        BrotliStreamRecord(
            stream_index=2,
            compressed_offset=1970,
            compressed_length=113227,
            decompressed_offset=2434,
            decompressed_length=157680,
            decompressed_sha256="c" * 64,
        ),
    )
    # First byte of stream 0
    assert map_decompressed_byte_to_stream(0, streams) == (0, 0)
    # Last byte of stream 0
    assert map_decompressed_byte_to_stream(973, streams) == (0, 973)
    # First byte of stream 1
    assert map_decompressed_byte_to_stream(974, streams) == (1, 0)
    # Middle of stream 2 (35773 in the op7 anchor context — but here mapped
    # into the 3-stream test fixture's address space)
    assert map_decompressed_byte_to_stream(2434, streams) == (2, 0)
    # Last byte in fixture
    assert map_decompressed_byte_to_stream(2434 + 157679, streams) == (2, 157679)


def test_map_decompressed_byte_negative_raises() -> None:
    streams = (
        BrotliStreamRecord(
            stream_index=0,
            compressed_offset=0,
            compressed_length=10,
            decompressed_offset=0,
            decompressed_length=20,
            decompressed_sha256="a" * 64,
        ),
    )
    with pytest.raises(PostBrotliDecodeError):
        map_decompressed_byte_to_stream(-1, streams)


def test_map_decompressed_byte_out_of_bounds_raises() -> None:
    streams = (
        BrotliStreamRecord(
            stream_index=0,
            compressed_offset=0,
            compressed_length=10,
            decompressed_offset=0,
            decompressed_length=20,
            decompressed_sha256="a" * 64,
        ),
    )
    with pytest.raises(PostBrotliDecodeError):
        map_decompressed_byte_to_stream(20, streams)


def test_build_post_brotli_decompress_anchor_payload_baseline() -> None:
    if not PR101_BASELINE.exists():
        pytest.skip("PR101 baseline archive not on disk")
    layout = build_post_brotli_decompress_anchor_payload(PR101_BASELINE)
    assert isinstance(layout, PostBrotliDecompressLayout)
    # Empirical anchor: baseline sha b83bf348...
    assert layout.archive_sha256.startswith("b83bf3488625dbd7")
    assert layout.archive_bytes == 178258
    assert layout.total_decompressed_bytes == 229_014
    assert len(layout.streams) == PR101_BROTLI_STREAM_COUNT


def test_build_post_brotli_decompress_anchor_payload_op7_candidate() -> None:
    if not PR101_OP7_CANDIDATE.exists():
        pytest.skip("op7 candidate archive not on disk")
    layout = build_post_brotli_decompress_anchor_payload(PR101_OP7_CANDIDATE)
    # Empirical anchor: candidate sha 30826b37...
    assert layout.archive_sha256.startswith("30826b37093ee3af")
    assert layout.archive_bytes == 178258
    assert layout.total_decompressed_bytes == 229_014
    assert len(layout.streams) == PR101_BROTLI_STREAM_COUNT


def test_layout_as_dict_includes_mutation_grain() -> None:
    layout = PostBrotliDecompressLayout(
        archive_sha256="b" * 64,
        archive_bytes=100,
        decoder_blob_sha256="c" * 64,
        total_decompressed_bytes=200,
        streams=(),
    )
    d = layout.as_dict()
    assert d["mutation_grain"] == MUTATION_GRAIN_POST_BROTLI_DECOMPRESS
    assert d["archive_bytes"] == 100
    assert d["total_decompressed_bytes"] == 200


def test_compute_sensitivity_summary_stats_synthetic() -> None:
    import numpy as np

    arr = np.array(
        [
            [0.1, 0.05, 0.0],
            [0.5, 0.2, 0.0],
            [0.01, 0.001, 0.0],
            [-0.3, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    stats = compute_sensitivity_summary_stats(arr)
    assert stats["n_bytes"] == 4
    # Top byte by combined |seg| + |pose| should be index 1 (0.5 + 0.2 = 0.7)
    assert stats["top_k_decompressed_byte_indices_by_combined_seg_pose"][0] == 1
    # Stream stats sanity
    assert stats["seg_max_abs"] == pytest.approx(0.5, rel=1e-6)
    assert stats["pose_max_abs"] == pytest.approx(0.2, rel=1e-6)
    assert stats["mutation_grain"] == MUTATION_GRAIN_POST_BROTLI_DECOMPRESS


def test_compute_sensitivity_refuses_wrong_shape() -> None:
    import numpy as np

    arr = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(PostBrotliDecodeError):
        compute_sensitivity_summary_stats(arr)


def test_compute_sensitivity_empty_array_handles_gracefully() -> None:
    import numpy as np

    arr = np.zeros((0, 3), dtype=np.float32)
    stats = compute_sensitivity_summary_stats(arr)
    assert stats["n_bytes"] == 0
    assert stats["top_k_decompressed_byte_indices_by_combined_seg_pose"] == []


def test_module_exports_canonical_api() -> None:
    """Sister of Catalog #176 (CLAUDE.md row regression) — verify __all__."""
    import tac.master_gradient_post_brotli_decompress as mod

    expected = {
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
    }
    assert set(mod.__all__) == expected
