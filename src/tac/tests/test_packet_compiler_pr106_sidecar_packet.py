# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from tac.packet_compiler import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    build_pr106_sidecar_recode_candidate_packet,
    decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta,
    decode_pr106_sidecar_packet_dim_delta,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    encode_pr101_ranked_sidecar_payload,
    lossless_pr106_sidecar_recode_candidates,
    mutate_pr106_sidecar_semantic_correction,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
    pr106_sidecar_manifest,
    pr106_sidecar_mutation_manifest,
    prove_pr106_sidecar_packet_ir_identity,
    read_single_stored_member_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_R2_ARCHIVE = REPO_ROOT / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_PR101_ARCHIVE = (
    REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _single_member_zip(
    name: str,
    payload: bytes,
    *,
    archive_comment: bytes = b"",
) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
        zf.comment = archive_comment
    return out.getvalue()


@pytest.mark.parametrize(
    ("archive_path", "format_id", "sidecar_kind", "expected_archive_sha"),
    [
        (
            PR106_R2_ARCHIVE,
            PR106_SIDECAR_FORMAT_BROTLI,
            "brotli_dim_delta",
            "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f",
        ),
        (
            PR106_R2_PR101_ARCHIVE,
            PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
            "pr101_ranked_no_op",
            "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383",
        ),
    ],
)
def test_pr106_sidecar_packet_ir_identity_on_release_archives(
    archive_path: Path,
    format_id: int,
    sidecar_kind: str,
    expected_archive_sha: str,
) -> None:
    archive_bytes = archive_path.read_bytes()
    assert _sha(archive_bytes) == expected_archive_sha

    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    emitted_payload = emit_pr106_sidecar_packet(packet)
    emitted_archive = emit_single_stored_member_archive(member)
    manifest = pr106_sidecar_manifest(packet, archive_sha256=_sha(archive_bytes))

    assert packet.format_id == format_id
    assert packet.sidecar_kind == sidecar_kind
    assert emitted_payload == member.payload
    assert emitted_archive == archive_bytes
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_sha256"] == expected_archive_sha
    assert manifest["emitted_payload_sha256"] == _sha(member.payload)


@pytest.mark.parametrize(
    ("archive_path", "expected_format_id", "expected_archive_sha"),
    [
        (
            PR106_R2_ARCHIVE,
            "0x01",
            "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f",
        ),
        (
            PR106_R2_PR101_ARCHIVE,
            "0x02",
            "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383",
        ),
    ],
)
def test_pr106_sidecar_packet_ir_identity_proof_is_operator_facing_and_nonpromotable(
    archive_path: Path,
    expected_format_id: str,
    expected_archive_sha: str,
) -> None:
    proof = prove_pr106_sidecar_packet_ir_identity(
        archive_path=archive_path,
        expected_archive_sha256=expected_archive_sha,
    )

    assert proof["schema"] == "pr106_sidecar_packet_ir_identity_proof_v1"
    assert proof["packet_ir_identity_passed"] is True
    assert proof["blockers"] == []
    assert proof["runtime_consumption_claim"] is False
    assert proof["full_frame_inflate_output_parity_claim"] is False
    assert proof["contest_axis_claim"] is False
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert proof["archive"]["sha256"] == expected_archive_sha
    assert proof["archive"]["expected_sha256_matches"] is True
    assert proof["member"]["name"] == "0.bin"
    assert proof["packet"]["format_id"] == expected_format_id
    assert proof["emitted_payload"]["byte_identical_to_source_member"] is True
    assert proof["emitted_archive"]["byte_identical_to_source_archive"] is True
    assert proof["proof_not_score"] is True
    assert proof["evidence_axis"] == "packet-ir-parser-local-no-score"
    assert proof["byte_exact_identity"] == {
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": expected_archive_sha,
        "source_member_name": "0.bin",
        "source_member_payload_bytes": proof["member"]["payload_bytes"],
        "source_member_payload_sha256": proof["member"]["payload_sha256"],
        "emitted_payload_bytes": proof["emitted_payload"]["bytes"],
        "emitted_payload_sha256": proof["emitted_payload"]["sha256"],
        "emitted_archive_bytes": proof["emitted_archive"]["bytes"],
        "emitted_archive_sha256": proof["emitted_archive"]["sha256"],
        "payload_byte_identical": True,
        "archive_byte_identical": True,
        "expected_archive_sha256": expected_archive_sha,
        "expected_archive_sha256_matches": True,
        "expected_member_name": None,
        "expected_member_name_matches": None,
    }
    consumed = proof["packet"]["packet_ir_consumed_byte_proof"]
    assert consumed["all_payload_bytes_accounted"] is True
    assert consumed["runtime_consumption_claim"] is False


def test_pr106_sidecar_packet_ir_identity_proof_fails_closed_on_sha_mismatch() -> None:
    proof = prove_pr106_sidecar_packet_ir_identity(
        archive_path=PR106_R2_ARCHIVE,
        expected_archive_sha256="0" * 64,
    )

    assert proof["packet_ir_identity_passed"] is False
    assert proof["archive"]["expected_sha256_matches"] is False
    assert proof["blockers"] == ["expected_archive_sha256_mismatch"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_pr106_sidecar_packet_ir_identity_proof_fails_closed_on_malformed_sha() -> None:
    proof = prove_pr106_sidecar_packet_ir_identity(
        archive_path=PR106_R2_ARCHIVE,
        expected_archive_sha256="not-a-sha",
    )

    assert proof["packet_ir_identity_passed"] is False
    assert proof["archive"]["expected_sha256_well_formed"] is False
    assert proof["blockers"] == ["expected_archive_sha256_malformed"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_single_member_archive_autodetects_x_member_and_preserves_name() -> None:
    payload = b"\xfe\x01\x00\x00\x00\x00\x00\x00"
    archive_bytes = _single_member_zip("x", payload)

    member = read_single_stored_member_archive(archive_bytes)
    emitted = emit_single_stored_member_archive(member)

    assert member.name == "x"
    assert member.payload == payload
    assert emitted == archive_bytes


def test_single_member_archive_preserves_archive_comment() -> None:
    payload = b"\xfe\x01\x00\x00\x00\x00\x00\x00"
    archive_comment = b"packetir-comment: exact central directory metadata"
    archive_bytes = _single_member_zip(
        "0.bin",
        payload,
        archive_comment=archive_comment,
    )

    member = read_single_stored_member_archive(archive_bytes)
    emitted = emit_single_stored_member_archive(member)

    assert member.archive_comment == archive_comment
    assert emitted == archive_bytes


def test_pr106_sidecar_packet_ir_identity_proof_preserves_archive_comment(
    tmp_path: Path,
) -> None:
    payload = b"\xfe\x01\x00\x00\x00\x00\x00\x00"
    archive_bytes = _single_member_zip(
        "0.bin",
        payload,
        archive_comment=b"packetir-comment: identity contract",
    )
    archive_path = tmp_path / "commented_archive.zip"
    archive_path.write_bytes(archive_bytes)
    expected_sha = _sha(archive_bytes)

    proof = prove_pr106_sidecar_packet_ir_identity(
        archive_path=archive_path,
        expected_archive_sha256=expected_sha,
    )

    assert proof["packet_ir_identity_passed"] is True
    assert proof["archive"]["zip_comment_bytes"] > 0
    assert proof["archive"]["expected_sha256_matches"] is True
    assert proof["emitted_archive"]["byte_identical_to_source_archive"] is True
    assert proof["score_claim"] is False


def test_single_member_archive_explicit_expected_name_still_fails_closed() -> None:
    archive_bytes = _single_member_zip("x", b"payload")

    with pytest.raises(ValueError, match="expected ZIP member"):
        read_single_stored_member_archive(
            archive_bytes,
            expected_member_name="0.bin",
        )


def test_single_member_archive_rejects_unsupported_expected_name() -> None:
    archive_bytes = _single_member_zip("0.bin", b"payload")

    with pytest.raises(ValueError, match="unsupported expected PR106 ZIP member"):
        read_single_stored_member_archive(
            archive_bytes,
            expected_member_name="../0.bin",
        )


def test_pr106_sidecar_packet_rejects_unknown_format_id() -> None:
    archive_bytes = PR106_R2_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    mutated = bytearray(member.payload)
    mutated[1] = 0x99

    with pytest.raises(ValueError, match="unsupported PR106 sidecar format_id"):
        parse_pr106_sidecar_packet(bytes(mutated))


@pytest.mark.parametrize(
    "archive_path",
    [PR106_R2_ARCHIVE, PR106_R2_PR101_ARCHIVE],
)
def test_pr106_sidecar_packet_rejects_trailing_payload_bytes(
    archive_path: Path,
) -> None:
    """Parser refuses unaccounted trailing bytes after the declared sidecar."""
    archive_bytes = archive_path.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)

    with pytest.raises(ValueError, match="trailing bytes"):
        parse_pr106_sidecar_packet(member.payload + b"\x00")


@pytest.mark.parametrize(
    ("archive_path", "expected_section_names", "expected_score_sections"),
    [
        (
            PR106_R2_ARCHIVE,
            [
                "magic",
                "format_id",
                "pr106_len_le_u32",
                "pr106_payload",
                "sidecar_len_le_u16",
                "sidecar_payload",
            ],
            ["pr106_payload", "sidecar_payload"],
        ),
        (
            PR106_R2_PR101_ARCHIVE,
            [
                "magic",
                "format_id",
                "pr106_len_le_u32",
                "pr106_payload",
                "sidecar_len_le_u16",
                "sidecar_payload",
                "framing_meta",
            ],
            ["pr106_payload", "sidecar_payload", "framing_meta"],
        ),
    ],
)
def test_pr106_sidecar_manifest_carries_contiguous_consumed_byte_proof(
    archive_path: Path,
    expected_section_names: list[str],
    expected_score_sections: list[str],
) -> None:
    """Manifest accounts for every emitted payload byte without runtime overclaim."""
    archive_bytes = archive_path.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    emitted = emit_pr106_sidecar_packet(packet)
    manifest = pr106_sidecar_manifest(packet, archive_sha256=_sha(archive_bytes))
    proof = manifest["packet_ir_consumed_byte_proof"]
    assert isinstance(proof, dict)

    assert proof["proof_scope"] == (
        "packet_ir_parser_accounting_not_runtime_inflate_consumption"
    )
    assert proof["runtime_consumption_claim"] is False
    assert proof["all_payload_bytes_accounted"] is True
    assert proof["unconsumed_trailing_bytes"] == 0
    assert proof["section_gaps"] == []
    assert proof["emitted_payload_bytes"] == len(emitted)
    assert proof["emitted_payload_sha256"] == _sha(emitted)
    assert proof["score_affecting_section_names"] == expected_score_sections

    sections = proof["sections"]
    assert isinstance(sections, list)
    assert [row["name"] for row in sections] == expected_section_names
    cursor = 0
    for row in sections:
        assert row["offset"] == cursor
        end = cursor + int(row["bytes"])
        assert row["end_offset"] == end
        assert row["sha256"] == _sha(emitted[cursor:end])
        cursor = end
    assert cursor == len(emitted)
    assert proof["accounted_payload_bytes"] == len(emitted)


def test_pr106_sidecar_consumed_byte_proof_sha_tracks_sidecar_mutation() -> None:
    """A sidecar byte change updates the parser-level consumed-byte proof."""
    archive_bytes = PR106_R2_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    baseline_proof = pr106_sidecar_consumed_byte_proof(packet)

    mutated_sidecar = bytearray(packet.sidecar_payload)
    mutated_sidecar[0] ^= 0x01
    mutated_packet = type(packet)(
        format_id=packet.format_id,
        pr106_bytes=packet.pr106_bytes,
        sidecar_payload=bytes(mutated_sidecar),
        framing_meta=packet.framing_meta,
    )
    mutated_proof = pr106_sidecar_consumed_byte_proof(mutated_packet)

    assert mutated_proof["emitted_payload_sha256"] != baseline_proof[
        "emitted_payload_sha256"
    ]
    baseline_sidecar = {
        row["name"]: row for row in baseline_proof["sections"]
    }["sidecar_payload"]
    mutated_sidecar_row = {
        row["name"]: row for row in mutated_proof["sections"]
    }["sidecar_payload"]
    assert mutated_sidecar_row["sha256"] != baseline_sidecar["sha256"]
    assert mutated_proof["all_payload_bytes_accounted"] is True


@pytest.mark.parametrize(
    "archive_path",
    [PR106_R2_ARCHIVE, PR106_R2_PR101_ARCHIVE],
)
def test_pr106_sidecar_semantic_mutation_is_valid_and_nonpromotable(
    archive_path: Path,
) -> None:
    archive_bytes = archive_path.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    source_packet = parse_pr106_sidecar_packet(member.payload)
    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(source_packet)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    reparsed = parse_pr106_sidecar_packet(mutated_payload)
    manifest = pr106_sidecar_mutation_manifest(
        source_packet,
        reparsed,
        mutation,
        source_archive_sha256=_sha(archive_bytes),
        mutated_archive_sha256=_sha(
            emit_single_stored_member_archive(
                type(member)(
                    name=member.name,
                    payload=mutated_payload,
                    date_time=member.date_time,
                    external_attr=member.external_attr,
                    create_system=member.create_system,
                    flag_bits=member.flag_bits,
                    comment=member.comment,
                    extra=member.extra,
                )
            )
        ),
    )

    assert mutation.section_name == "sidecar_payload"
    assert mutation.old_delta_q != mutation.new_delta_q
    assert reparsed.pr106_bytes == source_packet.pr106_bytes
    assert reparsed.sidecar_payload != source_packet.sidecar_payload
    assert manifest["payload_sha256_changed"] is True
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["sidecar_payload_sha256_changed"] is True
    assert manifest["parser_consumed_byte_accounting_passed"] is True
    assert manifest["source_packet_ir_consumed_byte_proof"][
        "all_payload_bytes_accounted"
    ] is True
    assert manifest["mutated_packet_ir_consumed_byte_proof"][
        "all_payload_bytes_accounted"
    ] is True
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["full_frame_inflate_output_parity_claim"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["proof_not_score"] is True
    assert manifest["evidence_axis"] == "packet-ir-mutation-local-no-score"
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_pr106_sidecar_packet_rejects_pr101_missing_framing_meta() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)

    with pytest.raises(ValueError, match="requires framing_meta"):
        emit_pr106_sidecar_packet(
            type(packet)(
                format_id=packet.format_id,
                pr106_bytes=packet.pr106_bytes,
                sidecar_payload=packet.sidecar_payload,
                framing_meta=None,
            )
        )


def test_pr106_sidecar_rank_elided_rejects_surplus_payload_bytes() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    payload, framing_meta = encode_pr101_ranked_sidecar_payload(dims, deltas)
    noop_count = int.from_bytes(framing_meta[0:2], "little")
    dim_bytes = int.from_bytes(framing_meta[2:4], "little")
    rank_bytes = framing_meta[4]
    noop_rank_bytes = framing_meta[5]
    assert rank_bytes == 1
    rank_elided_payload = payload[:dim_bytes] + payload[dim_bytes + rank_bytes :]
    rank_elided_meta = (
        noop_count.to_bytes(2, "little")
        + dim_bytes.to_bytes(2, "little")
        + bytes([noop_rank_bytes])
    )
    rank_elided_packet = type(packet)(
        format_id=PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
        pr106_bytes=packet.pr106_bytes,
        sidecar_payload=rank_elided_payload,
        framing_meta=rank_elided_meta,
    )
    emitted = emit_pr106_sidecar_packet(rank_elided_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    assert reparsed.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED
    assert decode_pr106_sidecar_packet_dim_delta(reparsed)[0].shape == dims.shape

    with pytest.raises(ValueError, match="payload length mismatch"):
        parse_pr106_sidecar_packet(emitted[:-5] + b"x" + emitted[-5:])


def test_pr106_sidecar_fixed_meta_rank_elided_candidate_roundtrips() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    fixed = {
        candidate.name: candidate
        for candidate in candidates
    }["pr101_fixed_meta_rank_elided_sidecar_format_0x05"]

    fixed_packet = build_pr106_sidecar_recode_candidate_packet(packet, fixed)
    emitted = emit_pr106_sidecar_packet(fixed_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    fixed_dims, fixed_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    direct_dims, direct_deltas = (
        decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
            fixed.encoded_bytes
        )
    )
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)

    assert fixed.sidecar_format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED
    assert fixed.runtime_decoder_implemented is True
    assert fixed_packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED
    assert fixed_packet.framing_meta is None
    assert len(emitted) == len(member.payload) - 9
    assert [row["name"] for row in proof["sections"]] == [
        "magic",
        "format_id",
        "pr106_len_le_u32",
        "pr106_payload",
        "sidecar_payload",
    ]
    assert proof["score_affecting_section_names"] == ["pr106_payload", "sidecar_payload"]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"] is not None
    assert (fixed_dims == dims).all()
    assert (fixed_deltas == deltas).all()
    assert (direct_dims == dims).all()
    assert (direct_deltas == deltas).all()


def test_pr106_sidecar_implicit_len_fixed_meta_candidate_roundtrips() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    implicit = {
        candidate.name: candidate
        for candidate in candidates
    }["pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06"]

    implicit_packet = build_pr106_sidecar_recode_candidate_packet(packet, implicit)
    emitted = emit_pr106_sidecar_packet(implicit_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    implicit_dims, implicit_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)

    assert implicit.sidecar_format_id == (
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert implicit.runtime_decoder_implemented is True
    assert implicit_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert implicit_packet.framing_meta is None
    assert len(emitted) == len(member.payload) - 13
    assert [row["name"] for row in proof["sections"]] == [
        "magic",
        "format_id",
        "pr106_payload",
        "sidecar_payload",
    ]
    assert proof["score_affecting_section_names"] == ["pr106_payload", "sidecar_payload"]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"]["implicit_pr106_len"] is True
    assert (implicit_dims == dims).all()
    assert (implicit_deltas == deltas).all()


def test_pr106_sidecar_headerless_implicit_len_fixed_meta_candidate_roundtrips() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    headerless = {
        candidate.name: candidate
        for candidate in candidates
    }["pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07"]

    headerless_packet = build_pr106_sidecar_recode_candidate_packet(packet, headerless)
    emitted = emit_pr106_sidecar_packet(headerless_packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    headerless_dims, headerless_deltas = decode_pr106_sidecar_packet_dim_delta(reparsed)
    proof = pr106_sidecar_consumed_byte_proof(reparsed)
    manifest = pr106_sidecar_manifest(reparsed)

    assert headerless.sidecar_format_id == (
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert headerless.runtime_decoder_implemented is True
    assert headerless_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert headerless_packet.framing_meta is None
    assert emitted[0] != 0xFE
    assert len(emitted) == len(member.payload) - 15
    assert [row["name"] for row in proof["sections"]] == [
        "pr106_payload",
        "sidecar_payload",
    ]
    assert proof["score_affecting_section_names"] == ["pr106_payload", "sidecar_payload"]
    assert proof["all_payload_bytes_accounted"] is True
    assert manifest["derived_fixed_meta"]["implicit_pr106_len"] is True
    assert manifest["derived_fixed_meta"]["headerless_packet"] is True
    assert (headerless_dims == dims).all()
    assert (headerless_deltas == deltas).all()


def test_pr106_sidecar_fixed_meta_rank_elided_mutation_is_valid() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    fixed = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_fixed_meta_rank_elided_sidecar_format_0x05"]
    fixed_packet = build_pr106_sidecar_recode_candidate_packet(packet, fixed)

    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(
        fixed_packet,
        pair_index=0,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(mutated_packet))

    assert mutated_packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED
    assert mutated_packet.framing_meta is None
    assert mutation.section_name == "sidecar_payload"
    assert mutation.old_delta_q != mutation.new_delta_q
    assert reparsed.pr106_bytes == fixed_packet.pr106_bytes
    assert reparsed.sidecar_payload != fixed_packet.sidecar_payload


def test_pr106_sidecar_implicit_len_fixed_meta_mutation_is_valid() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    implicit = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06"]
    implicit_packet = build_pr106_sidecar_recode_candidate_packet(packet, implicit)

    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(
        implicit_packet,
        pair_index=0,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(mutated_packet))

    assert mutated_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert mutated_packet.framing_meta is None
    assert mutation.section_name == "sidecar_payload"
    assert mutation.old_delta_q != mutation.new_delta_q
    assert reparsed.pr106_bytes == implicit_packet.pr106_bytes
    assert reparsed.sidecar_payload != implicit_packet.sidecar_payload


def test_pr106_sidecar_headerless_implicit_len_fixed_meta_mutation_is_valid() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    headerless = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07"]
    headerless_packet = build_pr106_sidecar_recode_candidate_packet(packet, headerless)

    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(
        headerless_packet,
        pair_index=0,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(mutated_packet))

    assert mutated_packet.format_id == (
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    assert mutated_packet.framing_meta is None
    assert mutation.section_name == "sidecar_payload"
    assert mutation.old_delta_q != mutation.new_delta_q
    assert reparsed.pr106_bytes == headerless_packet.pr106_bytes
    assert reparsed.sidecar_payload != headerless_packet.sidecar_payload


def test_pr106_sidecar_fixed_meta_rank_elided_rejects_bad_payload_length() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    fixed = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_fixed_meta_rank_elided_sidecar_format_0x05"]
    fixed_packet = build_pr106_sidecar_recode_candidate_packet(packet, fixed)
    emitted = emit_pr106_sidecar_packet(fixed_packet)

    with pytest.raises(ValueError, match="payload length mismatch"):
        parse_pr106_sidecar_packet(emitted + b"x")
