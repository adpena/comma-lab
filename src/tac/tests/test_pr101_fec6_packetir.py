# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tac.packet_compiler.pr101_fec6_packetir import (
    FEC6_FIXED_K16_CODE_BITS,
    PR101FEC6PacketIRError,
    emit_pr101_fec6_packetir_member,
    parse_pr101_fec6_packetir_member,
    prove_pr101_fec6_packetir_identity,
    read_single_stored_fec6_member_archive,
)
from tac.repo_io import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "prove_pr101_fec6_packetir_identity.py"
FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip"
)
FEC6_ARCHIVE_SHA256 = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("prove_pr101_fec6_packetir_identity", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fec6_selector(codes: list[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    pad = (-len(bits)) % 8
    padded = bits + ("0" * pad)
    payload = bytes(int(padded[index : index + 8], 2) for index in range(0, len(padded), 8))
    return b"FEC6" + len(codes).to_bytes(2, "little") + payload


def _fp11_member(source: bytes = b"source-pr101", codes: list[int] | None = None) -> bytes:
    selector = _fec6_selector([0, 2, 7, 13] if codes is None else codes)
    return b"FP11" + len(source).to_bytes(4, "little") + source + len(selector).to_bytes(2, "little") + selector


def _single_member_archive(tmp_path: Path, payload: bytes, *, name: str = "x") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    return archive


def test_synthetic_fp11_fec6_parse_emit_identity_sections() -> None:
    member_payload = _fp11_member()

    packet = parse_pr101_fec6_packetir_member(member_payload)
    emitted = emit_pr101_fec6_packetir_member(packet)
    sections = {section.name: section.to_manifest() for section in packet.sections}

    assert emitted == member_payload
    assert packet.source_len == len(b"source-pr101")
    assert packet.selector_len == len(_fec6_selector([0, 2, 7, 13]))
    assert packet.selector_codes == (0, 2, 7, 13)
    assert sections["fp11_magic"]["offset"] == 0
    assert sections["source_len_u32le"]["offset"] == 4
    assert sections["source_pr101_payload"]["offset"] == 8
    assert sections["selector_len_u16le"]["offset"] == 20
    assert sections["selector_fec6_payload"]["offset"] == 22
    assert sections["selector_fec6_fixed_huffman_bitstream"]["length"] == 2
    assert sections["packet_member_payload"]["sha256"] == sha256_bytes(member_payload)


def test_synthetic_archive_identity_proof_is_nonpromotable(tmp_path: Path) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())

    proof = prove_pr101_fec6_packetir_identity(archive_path=archive)

    assert proof["schema"] == "pr101_fec6_packetir_identity_proof_v1"
    assert proof["packet_ir_identity_passed"] is True
    assert proof["reemit_identity"] is True
    assert proof["member_reemit_identity"] is True
    assert proof["archive_reemit_identity"] is True
    assert proof["blockers"] == []
    assert proof["member_name"] == "x"
    assert proof["packet"]["all_member_bytes_accounted"] is True
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_parser_rejects_bad_fp11_magic() -> None:
    with pytest.raises(PR101FEC6PacketIRError, match="FP11 magic mismatch"):
        parse_pr101_fec6_packetir_member(b"NOPE" + _fp11_member()[4:])


def test_parser_rejects_truncated_source_payload() -> None:
    with pytest.raises(PR101FEC6PacketIRError, match="truncated in source_pr101_payload"):
        parse_pr101_fec6_packetir_member(b"FP11" + (99).to_bytes(4, "little") + b"short")


def test_parser_rejects_trailing_bytes_after_selector() -> None:
    with pytest.raises(PR101FEC6PacketIRError, match="trailing bytes"):
        parse_pr101_fec6_packetir_member(_fp11_member() + b"\x00")


def test_parser_rejects_bad_fec6_magic() -> None:
    member = _fp11_member()
    source_len = int.from_bytes(member[4:8], "little")
    selector_start = 8 + source_len + 2
    broken = member[:selector_start] + b"FECX" + member[selector_start + 4 :]

    with pytest.raises(PR101FEC6PacketIRError, match="FEC6 selector magic mismatch"):
        parse_pr101_fec6_packetir_member(broken)


def test_parser_rejects_nonzero_fec6_padding_bits() -> None:
    source = b"source"
    selector = b"FEC6" + (1).to_bytes(2, "little") + b"\x01"
    member = b"FP11" + len(source).to_bytes(4, "little") + source + len(selector).to_bytes(2, "little") + selector

    with pytest.raises(PR101FEC6PacketIRError, match="non-zero padding bits"):
        parse_pr101_fec6_packetir_member(member)


def test_expected_archive_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())

    proof = prove_pr101_fec6_packetir_identity(
        archive_path=archive,
        expected_archive_sha256="0" * 64,
    )

    assert proof["packet_ir_identity_passed"] is False
    assert proof["reemit_identity"] is True
    assert proof["blockers"] == ["expected_archive_sha256_mismatch"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_cli_writes_json_and_markdown_summary(tmp_path: Path) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())
    out_json = tmp_path / "proof.json"
    out_md = tmp_path / "proof.md"
    tool = _load_tool()

    rc = tool.main(
        [
            "--archive",
            str(archive),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    proof = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")
    assert rc == 0
    assert proof["packet_ir_identity_passed"] is True
    assert proof["reemit_identity"] is True
    assert proof["score_claim"] is False
    assert "# PR101/FEC6 PacketIR Identity Proof" in md
    assert "| selector_fec6_payload |" in md


def test_real_pr101_fec6_archive_identity_proof_if_present() -> None:
    if not FEC6_ARCHIVE.exists():
        pytest.skip(f"missing local PR101/FEC6 archive: {FEC6_ARCHIVE.relative_to(REPO_ROOT)}")

    proof = prove_pr101_fec6_packetir_identity(
        archive_path=FEC6_ARCHIVE,
        expected_archive_sha256=FEC6_ARCHIVE_SHA256,
    )

    assert proof["packet_ir_identity_passed"] is True
    assert proof["archive_sha256"] == FEC6_ARCHIVE_SHA256
    assert proof["member_name"] == "x"
    assert proof["member_bytes"] == 178_417
    assert proof["member_sha256"] == "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
    assert proof["packet"]["source_len"] == 178_158
    assert proof["packet"]["selector_len"] == 249
    assert proof["packet"]["selector"]["n_pairs"] == 600
    assert proof["packet"]["selector"]["selector_index_bytes"] == 243
    assert proof["packet"]["selector"]["selector_code_bits_total"] == 1_944
    assert proof["packet"]["selector"]["zero_padding_bits"] == 0
    assert proof["reemit_identity"] is True


def test_real_archive_member_reader_preserves_zip_identity_if_present() -> None:
    if not FEC6_ARCHIVE.exists():
        pytest.skip(f"missing local PR101/FEC6 archive: {FEC6_ARCHIVE.relative_to(REPO_ROOT)}")
    archive_bytes = FEC6_ARCHIVE.read_bytes()
    member = read_single_stored_fec6_member_archive(archive_bytes)
    assert member.name == "x"
    assert sha256_bytes(member.payload) == "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
