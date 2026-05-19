# SPDX-License-Identifier: MIT
"""Tests for post-decompress master-gradient extension to 5 archive families.

Sister of `test_master_gradient_post_brotli_decompress.py` (PR101 canonical
reference) covering the 5 NEW post-decompress parsers:
  - PR106 format0d
  - PR107 apogee_v2 (length-prefixed)
  - A1
  - DP1
  - HDM8 film grain sidecar

Plus the cross-family aggregator + cascade-severity classifier helpers.

Per Catalog #229 PV: all real-archive integration tests are marked @pytest.mark.slow
and require local fixtures. Unit tests use synthetic mini-archives.

[verified-against:src/tac/master_gradient_post_decompress_multi_archive.py]
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

import brotli  # type: ignore[import-untyped]

from tac.master_gradient_post_decompress_multi_archive import (
    AFFECTED_ARCHIVE_FAMILIES,
    AffectedArchiveFamily,
    CASCADE_SEVERITY_BOUNDED,
    CASCADE_SEVERITY_NONE,
    CASCADE_SEVERITY_UNBOUNDED,
    DecompressedStreamRecord,
    MUTATION_GRAIN_A1_POST_DECOMPRESS,
    MUTATION_GRAIN_DP1_POST_DECOMPRESS,
    MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS,
    MUTATION_GRAIN_POST_BROTLI_DECOMPRESS,
    MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS,
    MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS,
    PostDecompressDecodeError,
    PostDecompressLayout,
    build_a1_post_decompress_layout,
    build_dp1_post_decompress_layout,
    build_hdm8_film_grain_post_decompress_layout,
    build_post_decompress_layout_for_family,
    build_pr106_format0d_post_decompress_layout,
    build_pr107_apogee_v2_post_decompress_layout,
    classify_cascade_severity_for_codec,
    compute_sensitivity_summary_stats,
    map_decompressed_byte_to_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _wrap_as_single_member_zip(payload: bytes, member_name: str = "x") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(member_name, payload)
    return buf.getvalue()


def _make_synthetic_pr107_apogee_v2(tmp_path: Path) -> Path:
    """Build a synthetic PR107 apogee_v2 length-prefixed archive."""
    meta = json.dumps(
        {"n_pairs": 2, "latent_dim": 4, "base_channels": 8, "eval_size": [16, 16]}
    ).encode("utf-8")
    meta_brotli = brotli.compress(meta, quality=11)
    decoder_raw = b"\x00" * 64
    decoder_brotli = brotli.compress(decoder_raw, quality=11)
    latents_raw = b"\x01" * 32
    latents_brotli = brotli.compress(latents_raw, quality=11)
    payload = (
        struct.pack("<I", len(meta_brotli))
        + meta_brotli
        + struct.pack("<I", len(decoder_brotli))
        + decoder_brotli
        + struct.pack("<I", len(latents_brotli))
        + latents_brotli
    )
    archive_bytes = _wrap_as_single_member_zip(payload, "x")
    path = tmp_path / "pr107_apogee_v2_synth.zip"
    path.write_bytes(archive_bytes)
    return path


def _make_synthetic_pr106_format0d_hdm9(tmp_path: Path) -> Path:
    """Build a synthetic PR106 format0d with HDM9-magic decoder_blob."""
    hdm9_decoder = b"HDM9" + b"\x00" * 100  # 104-byte HDM9-packed decoder
    pr106_payload_inner = (
        b"\xff" + len(hdm9_decoder).to_bytes(3, "little") + hdm9_decoder + b"\x42" * 16
    )
    pr106_len = len(pr106_payload_inner)
    payload = (
        bytes([0xFE, 0x0D])
        + struct.pack("<I", pr106_len)
        + pr106_payload_inner
        + b"\x99" * 4  # trailing extra sections (sidecar)
    )
    archive_bytes = _wrap_as_single_member_zip(payload, "x")
    path = tmp_path / "pr106_format0d_hdm9_synth.zip"
    path.write_bytes(archive_bytes)
    return path


def _make_synthetic_pr106_format0d_brotli(tmp_path: Path) -> Path:
    """Build a synthetic PR106 format0d with brotli-compressed decoder_blob."""
    raw_decoder_weights = b"\x10" * 200
    decoder_brotli = brotli.compress(raw_decoder_weights, quality=11)
    pr106_payload_inner = (
        b"\xff" + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + b"\x42" * 16
    )
    pr106_len = len(pr106_payload_inner)
    payload = (
        bytes([0xFE, 0x0D])
        + struct.pack("<I", pr106_len)
        + pr106_payload_inner
        + b"\x99" * 4
    )
    archive_bytes = _wrap_as_single_member_zip(payload, "x")
    path = tmp_path / "pr106_format0d_brotli_synth.zip"
    path.write_bytes(archive_bytes)
    return path


def _make_synthetic_hdm8_format01(tmp_path: Path) -> Path:
    """Build a synthetic HDM8 with format_id=0x01 (legacy brotli sidecar)."""
    pr106_bytes = b"\xAA" * 50  # opaque pr106 bytes
    raw_sidecar = struct.pack("<H", 3) + bytes([1, 2, 3, 4, 5, 6])  # n=3 + (dim, delta_q) tuples
    sidecar_brotli = brotli.compress(raw_sidecar, quality=11)
    payload = (
        bytes([0xFE, 0x01])  # magic + format_id=0x01
        + struct.pack("<I", len(pr106_bytes))
        + pr106_bytes
        + struct.pack("<H", len(sidecar_brotli))
        + sidecar_brotli
    )
    archive_bytes = _wrap_as_single_member_zip(payload, "x")
    path = tmp_path / "hdm8_format01_synth.zip"
    path.write_bytes(archive_bytes)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Cascade-severity classifier (5 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_classify_cascade_severity_brotli_returns_bounded() -> None:
    assert classify_cascade_severity_for_codec("brotli") == CASCADE_SEVERITY_BOUNDED


def test_classify_cascade_severity_lzma_returns_bounded() -> None:
    assert classify_cascade_severity_for_codec("LZMA") == CASCADE_SEVERITY_BOUNDED


def test_classify_cascade_severity_arithmetic_returns_unbounded() -> None:
    assert (
        classify_cascade_severity_for_codec("arithmetic")
        == CASCADE_SEVERITY_UNBOUNDED
    )


def test_classify_cascade_severity_raw_int8_returns_none() -> None:
    assert classify_cascade_severity_for_codec("raw_int8") == CASCADE_SEVERITY_NONE


def test_classify_cascade_severity_unknown_conservatively_bounded() -> None:
    """Unknown codec returns bounded (conservative; don't claim NO cascade)."""
    assert (
        classify_cascade_severity_for_codec("xyzzy_unknown_codec")
        == CASCADE_SEVERITY_BOUNDED
    )


# ──────────────────────────────────────────────────────────────────────────────
# AffectedArchiveFamily registry (3 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_affected_archive_families_count_is_five() -> None:
    assert len(AFFECTED_ARCHIVE_FAMILIES) == 5


def test_affected_archive_families_ids_are_canonical() -> None:
    ids = {f.family_id for f in AFFECTED_ARCHIVE_FAMILIES}
    assert ids == {
        "pr106_format0d",
        "pr107_apogee_v2",
        "a1_finetuned",
        "dp1_pretrained_driving_prior",
        "hdm8_film_grain_sidecar",
    }


def test_affected_archive_families_each_has_unique_mutation_grain() -> None:
    grains = {f.mutation_grain for f in AFFECTED_ARCHIVE_FAMILIES}
    assert len(grains) == 5


# ──────────────────────────────────────────────────────────────────────────────
# PR106 format0d parser (4 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_pr106_format0d_synthetic_hdm9_packed_recognized(tmp_path: Path) -> None:
    path = _make_synthetic_pr106_format0d_hdm9(tmp_path)
    layout = build_pr106_format0d_post_decompress_layout(path)
    assert layout.archive_family == "pr106_format0d"
    assert layout.mutation_grain == MUTATION_GRAIN_PR106_FORMAT0D_POST_DECOMPRESS
    assert len(layout.streams) == 1
    s = layout.streams[0]
    assert s.codec == "hdm_packed_HDM9"
    assert s.cascade_severity == CASCADE_SEVERITY_NONE


def test_pr106_format0d_synthetic_brotli_recognized(tmp_path: Path) -> None:
    path = _make_synthetic_pr106_format0d_brotli(tmp_path)
    layout = build_pr106_format0d_post_decompress_layout(path)
    assert layout.archive_family == "pr106_format0d"
    assert len(layout.streams) == 1
    s = layout.streams[0]
    assert s.codec == "brotli"
    assert s.cascade_severity == CASCADE_SEVERITY_BOUNDED
    assert s.decompressed_length == 200


def test_pr106_format0d_wrong_magic_raises(tmp_path: Path) -> None:
    payload = b"\xAA\x0D" + b"\x00" * 100  # wrong magic
    path = tmp_path / "bad_magic.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="missing 0xfe PacketIR magic"):
        build_pr106_format0d_post_decompress_layout(path)


def test_pr106_format0d_wrong_format_id_raises(tmp_path: Path) -> None:
    payload = bytes([0xFE, 0x05]) + struct.pack("<I", 10) + b"\x00" * 10
    path = tmp_path / "bad_format.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="format_id mismatch"):
        build_pr106_format0d_post_decompress_layout(path)


# ──────────────────────────────────────────────────────────────────────────────
# PR107 apogee_v2 length-prefixed parser (4 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_pr107_apogee_v2_synthetic_archive_parses_three_sections(
    tmp_path: Path,
) -> None:
    path = _make_synthetic_pr107_apogee_v2(tmp_path)
    layout = build_pr107_apogee_v2_post_decompress_layout(path)
    assert layout.archive_family == "pr107_apogee_v2"
    assert layout.mutation_grain == MUTATION_GRAIN_PR107_APOGEE_V2_POST_DECOMPRESS
    assert len(layout.streams) == 3
    section_names = {s.section_name for s in layout.streams}
    assert section_names == {"meta_brotli", "decoder_blob", "latents_brotli"}


def test_pr107_apogee_v2_all_three_sections_bounded_cascade(tmp_path: Path) -> None:
    path = _make_synthetic_pr107_apogee_v2(tmp_path)
    layout = build_pr107_apogee_v2_post_decompress_layout(path)
    for s in layout.streams:
        assert s.cascade_severity == CASCADE_SEVERITY_BOUNDED


def test_pr107_apogee_v2_corrupt_meta_brotli_raises(tmp_path: Path) -> None:
    bad_meta = b"\x99" * 30  # not brotli
    payload = (
        struct.pack("<I", len(bad_meta))
        + bad_meta
        + struct.pack("<I", 10)
        + b"\x00" * 10
        + struct.pack("<I", 5)
        + b"\x00" * 5
    )
    path = tmp_path / "pr107_corrupt.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="brotli error"):
        build_pr107_apogee_v2_post_decompress_layout(path)


def test_pr107_apogee_v2_truncated_length_field_raises(tmp_path: Path) -> None:
    payload = b"\x01\x00\x00"  # only 3 bytes before length field
    path = tmp_path / "pr107_trunc.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="truncated"):
        build_pr107_apogee_v2_post_decompress_layout(path)


# ──────────────────────────────────────────────────────────────────────────────
# A1 parser (3 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_a1_synthetic_truncated_payload_raises(tmp_path: Path) -> None:
    payload = struct.pack("<I", 10) + b"\x00" * 5  # too short for latent
    path = tmp_path / "a1_trunc.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError):
        build_a1_post_decompress_layout(path)


def test_a1_grain_constant_pinned() -> None:
    """The A1 mutation grain identifier is stable across releases."""
    assert MUTATION_GRAIN_A1_POST_DECOMPRESS == (
        "post_brotli_decompress_a1_pr101_family_decoder_bytes"
    )


def test_a1_real_archive_parses(tmp_path: Path) -> None:
    """Real A1 archive at submissions/a1/ parses cleanly."""
    a1_path = REPO_ROOT / "submissions" / "a1" / "archive.zip"
    if not a1_path.exists():
        pytest.skip("A1 reference archive not available")
    layout = build_a1_post_decompress_layout(a1_path)
    assert layout.archive_family == "a1_finetuned"
    assert layout.archive_bytes > 100_000  # A1 frontier archive is ~178KB
    # At least 1 decoder brotli stream + latent + sidecar
    assert len(layout.streams) >= 3
    # First stream must be decoder brotli
    assert layout.streams[0].section_name.startswith("decoder_brotli_stream")
    assert layout.streams[0].codec == "brotli"


# ──────────────────────────────────────────────────────────────────────────────
# DP1 parser (3 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_dp1_grain_constant_pinned() -> None:
    assert MUTATION_GRAIN_DP1_POST_DECOMPRESS == (
        "post_brotli_pickle_decompress_dp1_renderer_state_dict_bytes"
    )


def test_dp1_invalid_payload_raises(tmp_path: Path) -> None:
    payload = b"NOT_A_DP1_ARCHIVE_BYTES"
    path = tmp_path / "bad_dp1.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "0.bin"))
    with pytest.raises(PostDecompressDecodeError, match="not a DP1 archive"):
        build_dp1_post_decompress_layout(path)


def test_dp1_real_archive_classifies_sections(tmp_path: Path) -> None:
    """Real DP1 archive yields the 5 canonical sections with correct cascade."""
    candidates = [
        REPO_ROOT / "experiments/results/dp1_smoke_v2_hardening/archive.zip",
        REPO_ROOT / "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/archive.zip",
    ]
    archive_path = next((p for p in candidates if p.exists()), None)
    if archive_path is None:
        pytest.skip("DP1 reference archive not available")
    layout = build_dp1_post_decompress_layout(archive_path)
    assert layout.archive_family == "dp1_pretrained_driving_prior"
    # DP1 has 5 canonical sections
    assert len(layout.streams) == 5
    section_names = [s.section_name for s in layout.streams]
    assert section_names == [
        "dp1_header",
        "codebook_blob",
        "renderer_blob",
        "residual_blob",
        "meta_blob",
    ]
    # dp1_header + codebook_blob are byte-local
    header_stream = next(s for s in layout.streams if s.section_name == "dp1_header")
    assert header_stream.cascade_severity == CASCADE_SEVERITY_NONE
    codebook_stream = next(
        s for s in layout.streams if s.section_name == "codebook_blob"
    )
    assert codebook_stream.cascade_severity == CASCADE_SEVERITY_NONE
    # renderer_blob and residual_blob are bounded brotli cascade
    renderer_stream = next(
        s for s in layout.streams if s.section_name == "renderer_blob"
    )
    assert renderer_stream.cascade_severity == CASCADE_SEVERITY_BOUNDED


# ──────────────────────────────────────────────────────────────────────────────
# HDM8 parser (4 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_hdm8_synthetic_format01_parses(tmp_path: Path) -> None:
    path = _make_synthetic_hdm8_format01(tmp_path)
    layout = build_hdm8_film_grain_post_decompress_layout(path)
    assert layout.archive_family == "hdm8_film_grain_sidecar"
    assert (
        layout.mutation_grain == MUTATION_GRAIN_HDM8_FILM_GRAIN_POST_DECOMPRESS
    )
    # At least 2 streams: pr106_bytes_delegated + film_grain_sidecar_blob
    assert len(layout.streams) >= 2
    sidecar_stream = next(
        s for s in layout.streams if s.section_name == "film_grain_sidecar_blob"
    )
    assert sidecar_stream.codec == "brotli_int8_dim_delta_q"
    assert sidecar_stream.cascade_severity == CASCADE_SEVERITY_BOUNDED


def test_hdm8_wrong_magic_raises(tmp_path: Path) -> None:
    payload = bytes([0xAA, 0x01]) + struct.pack("<I", 10) + b"\x00" * 10
    path = tmp_path / "hdm8_bad_magic.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="sidecar magic mismatch"):
        build_hdm8_film_grain_post_decompress_layout(path)


def test_hdm8_too_short_raises(tmp_path: Path) -> None:
    payload = b"\xFE"  # 1-byte payload
    path = tmp_path / "hdm8_short.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    with pytest.raises(PostDecompressDecodeError, match="too short"):
        build_hdm8_film_grain_post_decompress_layout(path)


def test_hdm8_format02_pr101_grammar_bounded(tmp_path: Path) -> None:
    pr106_bytes = b"\xAA" * 30
    pr101_sidecar = b"\x33" * 40  # PR101 ranked-no-op opaque
    framing_meta = b"\x00\x00\x00\x00\x00\x00"
    payload = (
        bytes([0xFE, 0x02])
        + struct.pack("<I", len(pr106_bytes))
        + pr106_bytes
        + struct.pack("<H", len(pr101_sidecar))
        + pr101_sidecar
        + framing_meta
    )
    path = tmp_path / "hdm8_format02.zip"
    path.write_bytes(_wrap_as_single_member_zip(payload, "x"))
    layout = build_hdm8_film_grain_post_decompress_layout(path)
    sidecar_stream = next(
        s for s in layout.streams if s.section_name == "film_grain_sidecar_blob"
    )
    assert sidecar_stream.codec.startswith("pr101_ranked_no_op_grammar")
    assert sidecar_stream.cascade_severity == CASCADE_SEVERITY_BOUNDED
    framing_stream = next(
        s for s in layout.streams if s.section_name == "framing_meta"
    )
    assert framing_stream.cascade_severity == CASCADE_SEVERITY_NONE


# ──────────────────────────────────────────────────────────────────────────────
# Cross-family aggregator (4 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_build_for_family_dispatch_pr106(tmp_path: Path) -> None:
    path = _make_synthetic_pr106_format0d_brotli(tmp_path)
    layout = build_post_decompress_layout_for_family("pr106_format0d", path)
    assert layout.archive_family == "pr106_format0d"


def test_build_for_family_dispatch_pr107(tmp_path: Path) -> None:
    path = _make_synthetic_pr107_apogee_v2(tmp_path)
    layout = build_post_decompress_layout_for_family("pr107_apogee_v2", path)
    assert layout.archive_family == "pr107_apogee_v2"


def test_build_for_family_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(PostDecompressDecodeError, match="unknown archive family"):
        build_post_decompress_layout_for_family("not_a_family", tmp_path / "x.zip")


def test_build_for_family_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(PostDecompressDecodeError, match="archive missing"):
        build_post_decompress_layout_for_family(
            "pr106_format0d", tmp_path / "does_not_exist.zip"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Generic map_decompressed_byte_to_stream (5 tests)
# ──────────────────────────────────────────────────────────────────────────────


def _make_synthetic_stream(idx: int, dec_off: int, dec_len: int) -> DecompressedStreamRecord:
    return DecompressedStreamRecord(
        stream_index=idx,
        section_name=f"section_{idx}",
        codec="brotli",
        cascade_severity=CASCADE_SEVERITY_BOUNDED,
        compressed_offset=0,
        compressed_length=10,
        decompressed_offset=dec_off,
        decompressed_length=dec_len,
        decompressed_sha256=_sha256(b"x"),
    )


def test_map_decompressed_byte_index_zero_to_first_stream() -> None:
    streams = (_make_synthetic_stream(0, 0, 100), _make_synthetic_stream(1, 100, 50))
    assert map_decompressed_byte_to_stream(0, streams) == (0, 0)


def test_map_decompressed_byte_index_boundary_to_second_stream() -> None:
    streams = (_make_synthetic_stream(0, 0, 100), _make_synthetic_stream(1, 100, 50))
    assert map_decompressed_byte_to_stream(100, streams) == (1, 0)


def test_map_decompressed_byte_index_within_second_stream() -> None:
    streams = (_make_synthetic_stream(0, 0, 100), _make_synthetic_stream(1, 100, 50))
    assert map_decompressed_byte_to_stream(125, streams) == (1, 25)


def test_map_decompressed_byte_negative_raises() -> None:
    streams = (_make_synthetic_stream(0, 0, 100),)
    with pytest.raises(PostDecompressDecodeError, match=">= 0"):
        map_decompressed_byte_to_stream(-1, streams)


def test_map_decompressed_byte_out_of_bounds_raises() -> None:
    streams = (_make_synthetic_stream(0, 0, 100),)
    with pytest.raises(PostDecompressDecodeError, match="out of bounds"):
        map_decompressed_byte_to_stream(500, streams)


# ──────────────────────────────────────────────────────────────────────────────
# compute_sensitivity_summary_stats (4 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_sensitivity_summary_basic_shape() -> None:
    import numpy as np

    arr = np.array([[1.0, 2.0, 0.0], [-3.0, 4.0, 0.0], [0.5, -0.5, 0.0]])
    summary = compute_sensitivity_summary_stats(arr, mutation_grain="test_grain")
    assert summary["n_bytes"] == 3
    assert summary["mutation_grain"] == "test_grain"
    assert summary["seg_max_abs"] == 3.0
    assert summary["pose_max_abs"] == 4.0


def test_sensitivity_summary_top_k_ranked_by_combined_seg_pose() -> None:
    import numpy as np

    # Combined |seg|+|pose|: row 1 = 7, row 0 = 3, row 2 = 1 → top-3 = [1, 0, 2]
    arr = np.array([[1.0, 2.0, 0.0], [-3.0, 4.0, 0.0], [0.5, -0.5, 0.0]])
    summary = compute_sensitivity_summary_stats(arr)
    assert summary["top_k_decompressed_byte_indices_by_combined_seg_pose"] == [1, 0, 2]


def test_sensitivity_summary_empty_tensor_handles_gracefully() -> None:
    import numpy as np

    arr = np.zeros((0, 3))
    summary = compute_sensitivity_summary_stats(arr)
    assert summary["n_bytes"] == 0
    assert summary["top_k_decompressed_byte_indices_by_combined_seg_pose"] == []


def test_sensitivity_summary_wrong_shape_raises() -> None:
    import numpy as np

    arr = np.zeros((10, 2))  # only 2 columns
    with pytest.raises(PostDecompressDecodeError, match="!= \\(N, 3\\)"):
        compute_sensitivity_summary_stats(arr)


# ──────────────────────────────────────────────────────────────────────────────
# PostDecompressLayout / DecompressedStreamRecord (3 tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_decompressed_stream_record_as_dict_round_trips() -> None:
    rec = DecompressedStreamRecord(
        stream_index=0,
        section_name="test_section",
        codec="brotli",
        cascade_severity=CASCADE_SEVERITY_BOUNDED,
        compressed_offset=100,
        compressed_length=200,
        decompressed_offset=0,
        decompressed_length=400,
        decompressed_sha256="abc" * 21 + "a",
    )
    d = rec.as_dict()
    assert d["stream_index"] == 0
    assert d["section_name"] == "test_section"
    assert d["codec"] == "brotli"
    assert d["cascade_severity"] == CASCADE_SEVERITY_BOUNDED


def test_post_decompress_layout_as_dict_carries_mutation_grain() -> None:
    layout = PostDecompressLayout(
        archive_family="test_family",
        archive_sha256="x" * 64,
        archive_bytes=1000,
        payload_sha256="y" * 64,
        total_decompressed_bytes=2000,
        mutation_grain="test_grain_v1",
        streams=(),
        notes="test notes",
    )
    d = layout.as_dict()
    assert d["archive_family"] == "test_family"
    assert d["mutation_grain"] == "test_grain_v1"
    assert d["notes"] == "test notes"
    assert d["streams"] == []


def test_pr101_canonical_grain_re_export_matches() -> None:
    """The PR101 grain constant is re-exported from the sister module."""
    from tac.master_gradient_post_brotli_decompress import (
        MUTATION_GRAIN_POST_BROTLI_DECOMPRESS as canonical,
    )
    assert MUTATION_GRAIN_POST_BROTLI_DECOMPRESS == canonical


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests (3 tests, slow)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_integration_pr107_apogee_v2_real_archive() -> None:
    """Real PR107 apogee archive yields 3 brotli-wrapped sections."""
    candidates = [
        REPO_ROOT
        / "experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/archive.zip",
    ]
    archive_path = next((p for p in candidates if p.exists()), None)
    if archive_path is None:
        pytest.skip("PR107 apogee reference archive not available")
    layout = build_pr107_apogee_v2_post_decompress_layout(archive_path)
    assert layout.archive_family == "pr107_apogee_v2"
    assert len(layout.streams) == 3
    assert all(s.cascade_severity == CASCADE_SEVERITY_BOUNDED for s in layout.streams)
    # Decoder blob must be the largest section by far
    decoder_stream = next(
        s for s in layout.streams if s.section_name == "decoder_blob"
    )
    assert decoder_stream.decompressed_length > 100_000


@pytest.mark.slow
def test_integration_pr106_format0d_real_archive() -> None:
    """Real PR106 format0d archive recognized + HDM-packed cascade severity."""
    candidates = [
        REPO_ROOT
        / "experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip",
    ]
    archive_path = next((p for p in candidates if p.exists()), None)
    if archive_path is None:
        pytest.skip("PR106 format0d reference archive not available")
    layout = build_pr106_format0d_post_decompress_layout(archive_path)
    assert layout.archive_family == "pr106_format0d"
    assert len(layout.streams) == 1
    # Real PR106 frontier archive uses HDM9 packed (cascade severity NONE)
    assert layout.streams[0].codec.startswith("hdm_packed_")
    assert layout.streams[0].cascade_severity == CASCADE_SEVERITY_NONE


@pytest.mark.slow
def test_integration_hdm8_real_archive() -> None:
    """Real HDM8 archive parses into pr106_bytes + sidecar + framing tail."""
    candidates = [
        REPO_ROOT
        / "experiments/results/hdm8_fixed_even_rgb_bias_m1_p05_p05_positive_control_20260515_codex/archive.zip",
    ]
    archive_path = next((p for p in candidates if p.exists()), None)
    if archive_path is None:
        pytest.skip("HDM8 reference archive not available")
    layout = build_hdm8_film_grain_post_decompress_layout(archive_path)
    assert layout.archive_family == "hdm8_film_grain_sidecar"
    assert len(layout.streams) >= 2
