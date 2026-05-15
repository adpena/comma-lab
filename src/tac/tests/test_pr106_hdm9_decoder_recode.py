# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES,
    PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES,
    PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    build_pr106_sidecar_recode_candidate_packet,
    decode_hdm9_decoder_to_hdm8_payload,
    decode_hlm3_latents_to_hlm2_payload,
    decode_pr106_sidecar_packet_dim_delta,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    encode_hlm3_latents_from_hlm2_payload,
    lossless_pr106_sidecar_recode_candidates,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
    pr106_sidecar_manifest,
    read_single_stored_member_archive,
    recode_pr106_hdm8_hlm2_packet_to_hdm9,
    recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3,
    recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3_magicless,
)
from tac.packetir_exact_closure import _runtime_score_affecting_sections_match

REPO = Path(__file__).resolve().parents[3]
HDM8_FORMAT07_ARCHIVE = REPO / "src/tac/tests/fixtures/pr106_hdm8_format07.archive.zip"
RUNTIME_CODEC = REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py"
RUNTIME_INFLATE = REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py"


def test_hdm9_recode_emits_byte_smaller_packet_and_runtime_decodes() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(source_packet)
    source_decoder = _decoder_section(source_packet.pr106_bytes)

    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    emitted = emit_pr106_sidecar_packet(hdm9_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)
    hdm9_decoder = _decoder_section(reparsed.pr106_bytes)
    codec = _load_module(RUNTIME_CODEC, "pr106_hdm9_runtime_codec")

    assert hdm9_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    )
    assert emitted.startswith(b"HDM9")
    assert len(hdm9_decoder) == PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
    assert len(emitted) == len(member.payload) - 28
    assert decode_hdm9_decoder_to_hdm8_payload(hdm9_decoder) == source_decoder
    assert codec.decode_hdm9_decoder_raw(hdm9_decoder) == codec.decode_hdm8_decoder_raw(
        source_decoder
    )
    assert (reparsed_dims == source_dims).all()
    assert (reparsed_deltas == source_deltas).all()
    assert proof["score_affecting_section_names"] == [
        "pr106_hdm9_hlm2_payload_without_inner_header",
        "sidecar_payload",
    ]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"]["hdm9_hlm2_inner_headerless_packet"] is True
    assert manifest["derived_fixed_meta"]["hdm9_hlm2_fixed_lengths"] == {
        "decoder_payload_bytes": 169950,
        "latent_payload_bytes": 15776,
        "decoder_magic": "HDM9",
        "latent_magic": "HLM2",
        "elided_inner_header_bytes": 4,
        "scale_low3_bytes": 84,
        "scale_high_mask_bytes": 4,
        "scale_high_base": 59,
    }


def test_hdm9_recode_archive_delta_is_member_payload_delta() -> None:
    source_archive = HDM8_FORMAT07_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(source_archive)
    source_packet = parse_pr106_sidecar_packet(member.payload)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    candidate_payload = emit_pr106_sidecar_packet(hdm9_packet)
    candidate_member = dataclasses.replace(member, payload=candidate_payload)
    candidate_archive = emit_single_stored_member_archive(candidate_member)

    assert len(candidate_payload) == len(member.payload) - 28
    assert len(candidate_archive) == len(source_archive) - 28


def test_hdm9_hlm3_recode_saves_three_more_bytes_and_runtime_decodes() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(source_packet)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    hdm10_source = dataclasses.replace(
        hdm9_packet,
        sidecar_payload=hdm9_packet.sidecar_payload[:-1],
    )

    hdm10_packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3(hdm10_source)
    emitted = emit_pr106_sidecar_packet(hdm10_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)
    codec = _load_module(RUNTIME_CODEC, "pr106_hdm10_runtime_codec")
    inflate = _load_module(RUNTIME_INFLATE, "pr106_hdm10_runtime_inflate")
    hdm9_inner = hdm9_packet.pr106_bytes[4:]
    hdm10_inner = reparsed.pr106_bytes[4:]
    hdm9_latents = hdm9_inner[PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:]
    hdm10_latents = hdm10_inner[PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:]

    assert hdm10_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    assert emitted.startswith(b"HDM9")
    assert len(hdm9_latents) == PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    assert len(hdm10_latents) == PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
    assert len(emitted) == len(emit_pr106_sidecar_packet(hdm9_packet)) - 3
    assert decode_hlm3_latents_to_hlm2_payload(hdm10_latents) == hdm9_latents
    assert encode_hlm3_latents_from_hlm2_payload(hdm9_latents) == hdm10_latents
    assert codec.decode_hlm3_fixed_latents_raw(hdm10_latents) == (
        codec.decode_hlm2_fixed_latents_raw(hdm9_latents)
    )
    parsed = inflate.parse_sidecar_archive(emitted)
    assert parsed[0] == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    assert parsed[1] == hdm10_packet.pr106_bytes
    assert (reparsed_dims == source_dims).all()
    assert (reparsed_deltas == source_deltas).all()
    assert proof["score_affecting_section_names"] == [
        "pr106_hdm9_hlm3_payload_without_inner_header",
        "sidecar_payload",
    ]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"]["noop_rank_elided"] is True
    assert manifest["derived_fixed_meta"]["hdm9_hlm3_fixed_lengths"] == {
        "decoder_payload_bytes": 169950,
        "latent_payload_bytes": 15774,
        "decoder_magic": "HDM9",
        "latent_magic": "HLM3",
        "elided_inner_header_bytes": 4,
        "elided_section_magic_bytes": 0,
        "scale_low3_bytes": 84,
        "scale_high_mask_bytes": 4,
        "scale_high_base": 59,
        "elided_hlm2_lo_brotli_len_bytes": 2,
        "elided_noop_rank_bytes": 1,
    }


def test_hdm9_hlm3_magicless_recode_saves_fixed_magic_bytes_and_runtime_decodes() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(source_packet)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    hdm10_source = dataclasses.replace(
        hdm9_packet,
        sidecar_payload=hdm9_packet.sidecar_payload[:-1],
    )
    hdm10_packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3(hdm10_source)

    hdm11_packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3_magicless(
        hdm10_source
    )
    emitted = emit_pr106_sidecar_packet(hdm11_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)
    inflate = _load_module(RUNTIME_INFLATE, "pr106_hdm11_runtime_inflate")

    assert hdm11_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    assert not emitted.startswith(b"HDM9")
    assert len(emitted) == len(emit_pr106_sidecar_packet(hdm10_packet)) - 8
    parsed = inflate.parse_sidecar_archive(emitted)
    assert parsed[0] == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    assert parsed[1] == hdm11_packet.pr106_bytes
    assert (reparsed_dims == source_dims).all()
    assert (reparsed_deltas == source_deltas).all()
    assert proof["score_affecting_section_names"] == [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"]["hdm9_hlm3_magicless_packet"] is True
    assert manifest["derived_fixed_meta"]["hdm9_hlm3_fixed_lengths"][
        "elided_section_magic_bytes"
    ] == 8


def test_hdm9_hlm3_magicless_exact_radix_recode_runtime_decodes() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(source_packet)
    candidates = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(
            source_dims,
            source_deltas,
        )
    }
    format0b = candidates[
        "pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b"
    ]
    format0c = candidates[
        "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_"
        "noop_rank_elided_sidecar_format_0x0c"
    ]

    packet0b = build_pr106_sidecar_recode_candidate_packet(source_packet, format0b)
    packet0c = build_pr106_sidecar_recode_candidate_packet(source_packet, format0c)
    emitted0c = emit_pr106_sidecar_packet(packet0c)
    reparsed = parse_pr106_sidecar_packet(emitted0c)
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    inflate = _load_module(RUNTIME_INFLATE, "pr106_hdm12_runtime_inflate")
    parsed = inflate.parse_sidecar_archive(emitted0c)
    runtime_dims, runtime_deltas = (
        inflate.decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar(
            parsed[2]
        )
    )

    assert packet0c.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    assert parsed[0] == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    assert parsed[1] == packet0c.pr106_bytes
    assert len(emitted0c) == len(emit_pr106_sidecar_packet(packet0b)) - 14
    assert (reparsed_dims == source_dims).all()
    assert (reparsed_deltas == source_deltas).all()
    assert (runtime_dims == source_dims).all()
    assert (runtime_deltas == source_deltas).all()


def test_hdm9_hlm3_magicless_parser_rejects_same_length_garbage() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    hdm10_source = dataclasses.replace(
        hdm9_packet,
        sidecar_payload=hdm9_packet.sidecar_payload[:-1],
    )
    hdm11_packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3_magicless(
        hdm10_source
    )
    valid = emit_pr106_sidecar_packet(hdm11_packet)
    sidecar_len = len(hdm11_packet.sidecar_payload)
    same_length_garbage = b"\x00" * (len(valid) - sidecar_len) + valid[-sidecar_len:]
    first_byte_ff_garbage = b"\xff\x00\x00\x00" + b"\x00" * 600
    off_by_one = valid[:-1]
    inflate = _load_module(RUNTIME_INFLATE, "pr106_hdm11_runtime_inflate_negative")

    with pytest.raises(ValueError, match=r"HDM9 q-Brotli chunk|Brotli"):
        parse_pr106_sidecar_packet(same_length_garbage)
    with pytest.raises(ValueError, match=r"HDM9 q stream chunk|Brotli"):
        inflate.parse_sidecar_archive(same_length_garbage)
    with pytest.raises(ValueError, match="decoder length must be positive"):
        parse_pr106_sidecar_packet(first_byte_ff_garbage)
    with pytest.raises(ValueError, match="decoder length must be positive"):
        inflate.parse_sidecar_archive(first_byte_ff_garbage)
    with pytest.raises(ValueError):
        parse_pr106_sidecar_packet(off_by_one)
    with pytest.raises(ValueError):
        inflate.parse_sidecar_archive(off_by_one)


def test_hdm9_scale_high_mask_padding_fails_closed() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    hdm9_decoder = bytearray(_decoder_section(hdm9_packet.pr106_bytes))
    hdm9_decoder[-1] |= 0xF0
    codec = _load_module(RUNTIME_CODEC, "pr106_hdm9_runtime_codec_bad_mask")

    with pytest.raises(ValueError, match="HDM9 scale high-byte mask padding must be zero"):
        decode_hdm9_decoder_to_hdm8_payload(bytes(hdm9_decoder))
    with pytest.raises(ValueError, match="HDM9 scale high-byte mask padding must be zero"):
        codec.decode_hdm9_decoder_raw(bytes(hdm9_decoder))


def test_inflate_parser_accepts_hdm9_headerless_packet() -> None:
    member = read_single_stored_member_archive(HDM8_FORMAT07_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(member.payload)
    hdm9_packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(source_packet)
    emitted = emit_pr106_sidecar_packet(hdm9_packet)
    inflate = _load_module(RUNTIME_INFLATE, "pr106_hdm9_runtime_inflate")

    format_id, pr106_bytes, sidecar_blob, framing_meta = inflate.parse_sidecar_archive(emitted)

    assert format_id == (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    )
    assert pr106_bytes == hdm9_packet.pr106_bytes
    assert sidecar_blob == hdm9_packet.sidecar_payload
    assert framing_meta is None


def test_packetir_closure_accepts_hdm9_runtime_section_alias() -> None:
    result = _runtime_score_affecting_sections_match(
        expected_sections={"pr106_hdm9_hlm2_payload_without_inner_header", "sidecar_payload"},
        actual_sections={"pr106_payload", "sidecar_payload"},
        proof={"format_id": "0x09", "inner_pr106_payload_sha256_unchanged": True},
    )

    assert result["matched"] is True
    assert result["mode"] == "format_0x09_hdm9_hlm2_reconstructed_pr106_payload_alias"


def test_packetir_closure_accepts_hdm9_hlm3_runtime_section_alias() -> None:
    result = _runtime_score_affecting_sections_match(
        expected_sections={"pr106_hdm9_hlm3_payload_without_inner_header", "sidecar_payload"},
        actual_sections={"pr106_payload", "sidecar_payload"},
        proof={"format_id": "0x0A", "inner_pr106_payload_sha256_unchanged": True},
    )

    assert result["matched"] is True
    assert result["mode"] == "format_0x0a_hdm9_hlm3_reconstructed_pr106_payload_alias"


def test_packetir_closure_accepts_hdm9_hlm3_magicless_runtime_section_alias() -> None:
    result = _runtime_score_affecting_sections_match(
        expected_sections={
            "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
            "sidecar_payload",
        },
        actual_sections={"pr106_payload", "sidecar_payload"},
        proof={"format_id": "0x0B", "inner_pr106_payload_sha256_unchanged": True},
    )

    assert result["matched"] is True
    assert result["mode"] == (
        "format_0x0b_hdm9_hlm3_magicless_reconstructed_pr106_payload_alias"
    )


def _decoder_section(pr106_bytes: bytes) -> bytes:
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    return pr106_bytes[4 : 4 + decoder_len]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_codec = sys.modules.pop("codec", None)
    old_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        if old_codec is not None:
            sys.modules["codec"] = old_codec
    return module
