# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import struct
import zipfile
from collections.abc import Callable

import pytest

from tac.witness_dsl import taskspace_outer_archive_codec as codec


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zipfile_variant(
    *,
    member_extra: bytes = b"",
    member_comment: bytes = b"",
    archive_comment: bytes = b"",
    second_member: bool = False,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        info = zipfile.ZipInfo(codec.MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.create_version = 20
        info.extract_version = 10
        info.external_attr = codec.CANONICAL_EXTERNAL_ATTR
        info.extra = member_extra
        info.comment = member_comment
        archive.writestr(info, b"payload")
        if second_member:
            archive.writestr("1.bin", b"second")
        archive.comment = archive_comment
    return output.getvalue()


def _unsafe_path(archive: bytes) -> bytes:
    mutated = bytearray(archive)
    local_name_offset = codec._LOCAL_FILE_HEADER.size
    central_offset = archive.rfind(b"PK\x01\x02")
    central_name_offset = central_offset + codec._CENTRAL_DIRECTORY_HEADER.size
    mutated[local_name_offset : local_name_offset + len(codec.MEMBER_NAME_BYTES)] = b"../x/"
    mutated[central_name_offset : central_name_offset + len(codec.MEMBER_NAME_BYTES)] = b"../x/"
    return bytes(mutated)


def _bad_timestamp(archive: bytes) -> bytes:
    mutated = bytearray(archive)
    central_offset = archive.rfind(b"PK\x01\x02")
    struct.pack_into("<H", mutated, 10, 1)
    struct.pack_into("<H", mutated, central_offset + 12, 1)
    return bytes(mutated)


def _bad_mode(archive: bytes) -> bytes:
    mutated = bytearray(archive)
    central_offset = archive.rfind(b"PK\x01\x02")
    struct.pack_into("<I", mutated, central_offset + 38, 0)
    return bytes(mutated)


def _append_inside_deflate_stream(archive: bytes) -> bytes:
    original_central_offset = archive.rfind(b"PK\x01\x02")
    mutated = bytearray(archive[:original_central_offset] + b"\0" + archive[original_central_offset:])
    central_offset = original_central_offset + 1
    end_offset = len(mutated) - codec._END_OF_CENTRAL_DIRECTORY.size
    local_compressed_nbytes = struct.unpack_from("<I", mutated, 18)[0]
    central_compressed_nbytes = struct.unpack_from("<I", mutated, central_offset + 20)[0]
    struct.pack_into("<I", mutated, 18, local_compressed_nbytes + 1)
    struct.pack_into("<I", mutated, central_offset + 20, central_compressed_nbytes + 1)
    struct.pack_into("<I", mutated, end_offset + 16, central_offset)
    return bytes(mutated)


def test_highly_compressible_member_selects_exact_deflated_archive() -> None:
    member = b"task-space-level-set\0" * 4096
    result = codec.build_taskspace_outer_archive(member)

    assert result.deflated_price_bytes == len(result.deflated.archive_bytes)
    assert result.stored_price_bytes == len(result.stored.archive_bytes)
    assert result.deflated_price_bytes < result.stored_price_bytes
    assert result.deflated_price_bytes < len(member)
    assert result.selected is result.deflated
    assert result.selected.encoding is codec.OuterArchiveEncoding.DEFLATED
    assert result.selection_reason == "lower_archive_nbytes"
    assert result.stored.member_bytes == result.deflated.member_bytes == member
    assert result.stored.member_sha256 == result.deflated.member_sha256 == _sha256(member)
    assert result.stored.archive_sha256 == _sha256(result.stored.archive_bytes)
    assert result.deflated.archive_sha256 == _sha256(result.deflated.archive_bytes)
    assert result.truth.authority == "structural"
    assert result.truth.research_only is True
    assert result.truth.inner_payload_semantics_verified is False
    assert result.truth.candidate_claim is False
    assert result.truth.score_claim is False
    assert result.truth.originality_claim is False
    assert result.truth.promotion_eligible is False


def test_incompressible_member_selects_stored_by_measured_archive_bytes() -> None:
    member = hashlib.shake_256(b"g3-outer-archive-incompressible").digest(8192)
    result = codec.build_taskspace_outer_archive(member)

    assert result.stored_price_bytes < result.deflated_price_bytes
    assert result.selected is result.stored
    assert result.selected.encoding is codec.OuterArchiveEncoding.STORED


def test_exact_size_tie_prefers_stored_deterministically() -> None:
    result = codec.build_taskspace_outer_archive(b"\0" * 5)

    assert result.stored_price_bytes == result.deflated_price_bytes
    assert result.selected is result.stored
    assert result.selection_reason == "equal_archive_nbytes_prefer_zip_stored"
    assert result.tie_break_rule == "prefer_zip_stored"


def test_repeat_build_is_byte_identical_for_both_encodings() -> None:
    member = hashlib.shake_256(b"g3-repeat").digest(4096)

    first = codec.build_taskspace_outer_archive(member)
    second = codec.build_taskspace_outer_archive(member)

    assert first == second
    assert first.stored.archive_bytes == second.stored.archive_bytes
    assert first.deflated.archive_bytes == second.deflated.archive_bytes
    assert first.determinism_scope == (
        "exact bytes for recorded zlib compile/runtime/profile; receiver parse never recompresses"
    )


def test_empty_member_and_permissive_truth_construction_fail_closed() -> None:
    with pytest.raises(codec.TaskspaceOuterArchiveError, match="must not be empty"):
        codec.build_taskspace_outer_archive(b"")
    with pytest.raises(codec.TaskspaceOuterArchiveError, match="must not be empty"):
        codec.parse_taskspace_outer_archive(codec._build_exact_archive(b"", codec.OuterArchiveEncoding.STORED))
    with pytest.raises(TypeError):
        vars(codec)["OuterArchiveTruthLabels"](score_claim=True)


def test_parser_rejects_member_tamper_via_crc_custody() -> None:
    result = codec.build_taskspace_outer_archive(b"member-custody")
    tampered = bytearray(result.stored.archive_bytes)
    data_offset = codec._LOCAL_FILE_HEADER.size + len(codec.MEMBER_NAME_BYTES)
    tampered[data_offset] ^= 0x01

    with pytest.raises(codec.TaskspaceOuterArchiveError, match="CRC-32"):
        codec.parse_taskspace_outer_archive(bytes(tampered))


def test_parser_rejects_trailing_bytes_and_archive_comment() -> None:
    exact_archive = codec.build_taskspace_outer_archive(b"member").stored.archive_bytes

    with pytest.raises(codec.TaskspaceOuterArchiveError, match="exact archive EOF"):
        codec.parse_taskspace_outer_archive(exact_archive + b"trailing")
    with pytest.raises(codec.TaskspaceOuterArchiveError):
        codec.parse_taskspace_outer_archive(_zipfile_variant(archive_comment=b"forbidden"))


def test_parser_requires_exact_deflate_stream_eof() -> None:
    exact_archive = codec.build_taskspace_outer_archive(b"deflate EOF custody" * 128).deflated.archive_bytes

    with pytest.raises(codec.TaskspaceOuterArchiveError, match="after stream EOF"):
        codec.parse_taskspace_outer_archive(_append_inside_deflate_stream(exact_archive))


@pytest.mark.parametrize(
    "mutation",
    [
        _unsafe_path,
        _bad_timestamp,
        _bad_mode,
    ],
    ids=["unsafe-path", "timestamp", "mode"],
)
def test_parser_rejects_noncanonical_raw_metadata(mutation: Callable[[bytes], bytes]) -> None:
    exact_archive = codec.build_taskspace_outer_archive(b"metadata").stored.archive_bytes

    with pytest.raises(codec.TaskspaceOuterArchiveError):
        codec.parse_taskspace_outer_archive(mutation(exact_archive))


@pytest.mark.parametrize(
    "archive",
    [
        _zipfile_variant(member_extra=b"\x01\x00\x00\x00"),
        _zipfile_variant(member_comment=b"forbidden"),
        _zipfile_variant(second_member=True),
    ],
    ids=["extra-field", "member-comment", "second-member"],
)
def test_parser_rejects_extra_metadata_or_members(archive: bytes) -> None:
    with pytest.raises(codec.TaskspaceOuterArchiveError):
        codec.parse_taskspace_outer_archive(archive)


def test_parser_binds_expected_exact_hashes() -> None:
    result = codec.build_taskspace_outer_archive(b"hash-custody")

    parsed = codec.parse_taskspace_outer_archive(
        result.deflated.archive_bytes,
        expected_encoding="zip_deflated",
        expected_archive_sha256=result.deflated.archive_sha256,
        expected_member_sha256=result.deflated.member_sha256,
    )
    assert parsed == result.deflated

    with pytest.raises(codec.TaskspaceOuterArchiveError, match="archive SHA-256"):
        codec.parse_taskspace_outer_archive(
            result.deflated.archive_bytes,
            expected_archive_sha256="0" * 64,
        )
    with pytest.raises(codec.TaskspaceOuterArchiveError, match="member SHA-256"):
        codec.parse_taskspace_outer_archive(
            result.deflated.archive_bytes,
            expected_member_sha256="0" * 64,
        )


def test_deflated_parse_does_not_recompress_received_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    result = codec.build_taskspace_outer_archive(b"no receiver-side re-deflate" * 128)

    def forbidden_compressobj(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"receiver attempted re-DEFLATE: {args!r}, {kwargs!r}")

    monkeypatch.setattr(codec.zlib, "compressobj", forbidden_compressobj)
    parsed = codec.parse_taskspace_outer_archive(
        result.deflated.archive_bytes,
        expected_archive_sha256=result.deflated.archive_sha256,
        expected_member_sha256=result.deflated.member_sha256,
    )

    assert parsed.member_bytes == result.deflated.member_bytes
    assert parsed.archive_bytes == result.deflated.archive_bytes
