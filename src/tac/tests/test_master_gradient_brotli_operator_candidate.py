# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

import brotli
import pytest

from tac.frontier_archive_layout import inspect_frontier_archive_layout
from tac.master_gradient_brotli_operator_candidate import (
    MasterGradientBrotliOperatorError,
    build_master_gradient_brotli_operator_candidate,
)
from tac.monolithic_packet_candidate import sha256_bytes


def _write_zip(path: Path, *, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _pr106_payload(decoder: bytes, tail: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail


def _member_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None
        infos = zf.infolist()
        assert len(infos) == 1
        return zf.read(infos[0].filename)


def test_brotli_operator_builds_byte_saving_pr106_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    raw_decoder = (b"decoder-stream-" * 3000) + bytes(range(64))
    raw_tail = b"latent-tail" * 1000
    source_decoder = brotli.compress(raw_decoder, quality=0)
    source_tail = brotli.compress(raw_tail, quality=11)
    _write_zip(source, name="0.bin", payload=_pr106_payload(source_decoder, source_tail))

    manifest = build_master_gradient_brotli_operator_candidate(
        source_archive=source,
        output_dir=out_dir,
        target_section="decoder_packed_brotli",
        candidate_id="unit-brotli-op",
        qualities=(0, 5, 10, 11),
        lgwin_values=(10, 16, 22),
    )

    archive = Path(manifest["candidate_archive"]["path"])
    replacement = Path(manifest["replacement_payload"]["path"])
    assert archive.is_file()
    assert replacement.is_file()
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["packet_proofs"]["repacked_archive"] is True
    assert manifest["packet_proofs"]["updated_zip_crc"] is True
    assert manifest["packet_proofs"]["parser_reparse_success"] is True
    assert manifest["packet_proofs"]["structural_non_noop_section_changed"] is True
    assert manifest["packet_proofs"]["inflate_success_proof"] is False
    assert manifest["packet_proofs"]["runtime_byte_consumption_noop_detector"] is False
    assert manifest["replacement_payload"]["section_byte_delta"] < 0
    assert manifest["candidate_archive"]["archive_byte_delta"] < 0
    assert "runtime_consumption_proof_missing" in manifest["dispatch_blockers"]
    assert "contest_cuda_auth_eval_missing" in manifest["promotion_blockers"]

    candidate_member = _member_payload(archive)
    candidate_layout = inspect_frontier_archive_layout(archive)
    sections = {
        section["name"]: section
        for section in candidate_layout["logical_layout"]["sections"]
    }
    decoder_section = sections["decoder_packed_brotli"]
    candidate_decoder = candidate_member[
        decoder_section["offset"]: decoder_section["offset"] + decoder_section["len"]
    ]
    assert sha256_bytes(candidate_decoder) == manifest["replacement_payload"]["sha256"]
    assert brotli.decompress(candidate_decoder) == raw_decoder
    assert candidate_member[:4] == bytes([0xFF]) + len(candidate_decoder).to_bytes(3, "little")


def test_brotli_operator_rejects_unsupported_layout(tmp_path: Path) -> None:
    source = tmp_path / "pr101.zip"
    out_dir = tmp_path / "out"
    _write_zip(source, name="x", payload=b"d" * 162_164 + b"l" * 15_387 + b"s")

    with pytest.raises(MasterGradientBrotliOperatorError, match="pr106_ff_packed_hnerv"):
        build_master_gradient_brotli_operator_candidate(
            source_archive=source,
            output_dir=out_dir,
            target_section="decoder_packed_brotli",
            candidate_id="bad-layout",
        )


def test_brotli_operator_rejects_non_improving_default(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    raw_decoder = b"short"
    decoder = brotli.compress(raw_decoder, quality=11)
    tail = brotli.compress(b"tail", quality=11)
    _write_zip(source, name="0.bin", payload=_pr106_payload(decoder, tail))

    with pytest.raises(MasterGradientBrotliOperatorError, match="no byte-saving"):
        build_master_gradient_brotli_operator_candidate(
            source_archive=source,
            output_dir=tmp_path / "out",
            target_section="decoder_packed_brotli",
            candidate_id="no-win",
            qualities=(11,),
            lgwin_values=(22,),
        )
