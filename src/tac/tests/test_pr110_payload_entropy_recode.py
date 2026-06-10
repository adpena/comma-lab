# SPDX-License-Identifier: MIT
"""Behavioral tests for tac.packet_compiler.pr110_payload_entropy_recode.

These tests verify ACTUAL recode behavior (NO FAKE per CLAUDE.md): they round-trip
the entropy coder, prove byte-identical reconstruction of the raw decoder streams /
latent payload / sidecar, prove the recoded member is smaller, and prove the
selector + DQS1 tail are preserved. The real R3 frontier archive is used as the
fixture when present (Catalog #213 real-input discipline); a constructed minimal
FP11 fixture covers the container + wrapper logic when constriction or the archive
is unavailable.
"""
from __future__ import annotations

import lzma
import zipfile
from pathlib import Path

import pytest

from tac.packet_compiler import ctx_range_coder
from tac.packet_compiler import pr110_payload_entropy_recode as rec
from tac.packet_compiler.feca_selector_reparameterize import (
    FECA_MAGIC,
    split_fp11_member,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
R3_SUBMISSION = (
    REPO_ROOT
    / "experiments/results/pr110pp_r3_onhost_mode_table_20260610/submission_dir"
)
R3_ARCHIVE = R3_SUBMISSION / "archive.zip"

constriction_required = pytest.mark.skipif(
    ctx_range_coder.constriction is None, reason="constriction not installed"
)
r3_required = pytest.mark.skipif(
    not R3_ARCHIVE.exists(), reason="R3 frontier archive fixture not present"
)


# --- CTXR container (no external deps) -------------------------------------
def test_ctxr_container_roundtrip():
    dec = b"\x01\x02\x03\x04"
    lat = b"\xaa" * 17
    sidecar = b"S" * rec.SIDECAR_LEN
    container = rec.pack_ctxr_container(dec, lat, sidecar)
    out_dec, out_lat, out_side = rec.unpack_ctxr_container(container)
    assert out_dec == dec
    assert out_lat == lat
    assert out_side == sidecar


def test_ctxr_header_length_constant():
    container = rec.pack_ctxr_container(b"", b"", b"")
    assert len(container) == rec.CTXR_HEADER_LEN
    assert container[:4] == rec.CTXR_MAGIC
    assert container[4] == rec.CTXR_VERSION


def test_ctxr_rejects_bad_magic():
    container = bytearray(rec.pack_ctxr_container(b"x", b"y", b"z"))
    container[0] = ord("Z")
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.unpack_ctxr_container(bytes(container))


def test_ctxr_rejects_bad_version():
    container = bytearray(rec.pack_ctxr_container(b"x", b"y", b"z"))
    container[4] = 99
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.unpack_ctxr_container(bytes(container))


def test_ctxr_rejects_trailing_bytes():
    container = rec.pack_ctxr_container(b"x", b"y", b"z") + b"TRAILING"
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.unpack_ctxr_container(container)


def test_ctxr_section_length_exceeds_u24():
    too_big = b"\x00" * (0x1000000)
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.pack_ctxr_container(too_big, b"", b"")


def test_infer_decoder_blob_len_from_codec():
    if not (R3_SUBMISSION / "src" / "codec.py").exists():
        pytest.skip("R3 submission_dir not present")
    assert rec.infer_decoder_blob_len(R3_SUBMISSION) == 162127


def test_infer_decoder_blob_len_missing_codec(tmp_path):
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.infer_decoder_blob_len(tmp_path)


# --- real-archive recode (constriction + R3 fixture required) --------------
@constriction_required
@r3_required
def test_real_archive_recode_is_lossless_and_smaller():
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    # ACTUAL behavior: member is smaller and lossless proven
    assert result.recoded_member_bytes < result.orig_member_bytes
    assert result.member_delta_bytes < 0
    assert result.lossless_proof["decoder_raw_byte_identical"] is True
    assert result.lossless_proof["latent_raw_byte_identical"] is True
    assert result.lossless_proof["sidecar_byte_identical"] is True


@constriction_required
@r3_required
def test_real_archive_recode_known_deltas():
    """The recode deltas are the measured, reproducible byte savings (the
    canonical anchor); if the coder regresses this catches it."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    assert result.metrics["decoder_delta_bytes"] == -1023
    assert result.metrics["latent_delta_bytes"] == -317
    assert result.member_delta_bytes == -1326
    assert result.metrics["dec_sec_bytes"] == 161104
    assert result.metrics["lat_sec_bytes"] == 15070


@constriction_required
@r3_required
def test_real_archive_recode_preserves_selector_and_dqs1():
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    rt_parts = split_fp11_member(
        result.recoded_member, allowed_selector_magics=(FECA_MAGIC,)
    )
    assert rt_parts["selector_payload"] == parts["selector_payload"]
    assert rt_parts["dqs1_tail"] == parts["dqs1_tail"]
    assert result.selector_payload == parts["selector_payload"]
    assert result.dqs1_tail == parts["dqs1_tail"]


@constriction_required
@r3_required
def test_reconstructed_raw_decoder_streams_byte_identical():
    """The recoded source decodes to the exact raw decoder streams the PR #101
    reshape consumes (the lossless-at-the-pixel-source guarantee)."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    source = parts["source_payload"]
    decoder_blob = source[:dbl]
    orig_joined = b"".join(
        rec.decompress_brotli_streams_to_list(decoder_blob, rec.N_DECODER_STREAMS)
    )
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    rt_parts = split_fp11_member(
        result.recoded_member, allowed_selector_magics=(FECA_MAGIC,)
    )
    rt_streams, _rt_latent, _rt_sidecar = rec.reconstruct_raw_sections(
        rt_parts["source_payload"]
    )
    assert b"".join(rt_streams) == orig_joined


@constriction_required
@r3_required
def test_reconstructed_latent_raw_byte_identical():
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    source = parts["source_payload"]
    latent_blob = source[dbl : dbl + rec.LATENT_BLOB_LEN]
    orig_latent_raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=rec.LATENT_LZMA_FILTERS
    )
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    rt_parts = split_fp11_member(
        result.recoded_member, allowed_selector_magics=(FECA_MAGIC,)
    )
    _rt_streams, rt_latent, _rt_sidecar = rec.reconstruct_raw_sections(
        rt_parts["source_payload"]
    )
    assert rt_latent == orig_latent_raw


@constriction_required
@r3_required
def test_recoded_archive_member_count_and_stored():
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    out = REPO_ROOT / "experiments/results/_pytest_recode_tmp_archive.zip"
    try:
        rec.write_stored_archive(out, member_name="x", payload=result.recoded_member)
        with zipfile.ZipFile(out) as zf:
            infos = zf.infolist()
            assert len(infos) == 1
            assert infos[0].compress_type == zipfile.ZIP_STORED
            assert zf.read("x") == result.recoded_member
    finally:
        out.unlink(missing_ok=True)


@constriction_required
@r3_required
def test_verify_lossless_raises_on_corrupted_source():
    """If the source payload's length contract is violated, fail closed."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    # truncate the source payload -> length contract violated
    bad_source = parts["source_payload"][:-1]
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.verify_lossless(bad_source, decoder_blob_len=dbl)


@constriction_required
@r3_required
def test_recode_then_decode_is_deterministic():
    """Two independent recodes of the same member produce byte-identical output
    (IEEE-exact float64 tables + deterministic search)."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    a = rec.recode_fp11_member(member, decoder_blob_len=dbl).recoded_member
    b = rec.recode_fp11_member(member, decoder_blob_len=dbl).recoded_member
    assert a == b


@constriction_required
def test_encode_source_payload_length_contract():
    """encode_source_payload rejects a source whose length != dec+lat+sidecar."""
    with pytest.raises(rec.Pr110PayloadEntropyRecodeError):
        rec.encode_source_payload(b"\x00" * 100, decoder_blob_len=10)


@constriction_required
@r3_required
def test_decoder_section_roundtrips_real_streams():
    """ctx_range_coder.decode_decoder_section reproduces the exact raw streams."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    decoder_blob = parts["source_payload"][:dbl]
    raw_streams = rec.decompress_brotli_streams_to_list(
        decoder_blob, rec.N_DECODER_STREAMS
    )
    dec_sec = ctx_range_coder.encode_decoder_section(raw_streams)
    rt = ctx_range_coder.decode_decoder_section(dec_sec)
    assert len(rt) == len(raw_streams)
    assert all(a == b for a, b in zip(rt, raw_streams, strict=True))
    # the recoded decoder section MUST be smaller than the raw streams it codes
    assert len(dec_sec) < sum(len(s) for s in raw_streams)


@constriction_required
@r3_required
def test_latent_section_roundtrips_real_payload():
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    parts = split_fp11_member(member, allowed_selector_magics=(FECA_MAGIC,))
    latent_blob = parts["source_payload"][dbl : dbl + rec.LATENT_BLOB_LEN]
    latent_raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=rec.LATENT_LZMA_FILTERS
    )
    lat_sec = ctx_range_coder.encode_latent_section(latent_raw)
    rt = ctx_range_coder.decode_latent_section(lat_sec)
    assert rt == latent_raw


@constriction_required
@r3_required
def test_member_smaller_means_lower_rate_term():
    """The recode strictly reduces archive bytes => strictly reduces the rate
    contribution (25*bytes/37545489), with d_seg/d_pose unchanged by losslessness."""
    member_name, member = rec.read_single_member(R3_ARCHIVE)
    dbl = rec.infer_decoder_blob_len(R3_SUBMISSION)
    result = rec.recode_fp11_member(member, decoder_blob_len=dbl)
    orig_archive_member = len(member)
    recoded_archive_member = len(result.recoded_member)
    ORIG = 37_545_489
    rate_delta = 25.0 * (recoded_archive_member - orig_archive_member) / ORIG
    assert rate_delta < 0.0


def test_constants_match_pr101_schema():
    assert rec.LATENT_BLOB_LEN == 15_387
    assert rec.SIDECAR_LEN == 607
    assert rec.N_DECODER_STREAMS == 7
