# SPDX-License-Identifier: MIT
"""Conformance tests for ``tac.residual_basis.pr106_materializer_helpers``.

Covers:

* PR106 0.bin extraction from a single-member zip;
* refusal of empty or multi-member zips;
* end-to-end materialize_family_archive emission for all 5 families;
* manifest promotion-status invariants pinned False;
* materialization_manifest.json fields complete;
* deterministic archive zip bytes (byte-for-byte) given identical inputs;
* no-op detector byte-mutation smoke positive + negative paths;
* every emitted archive parses cleanly via parse_archive().
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tac.residual_basis.pr106_materializer_helpers import (
    PR106_BIN_MEMBER_NAME,
    MaterializerError,
    emit_archive_zip,
    extract_pr106_bytes,
    materialize_family_archive,
    run_no_op_detector_byte_mutation,
    sha256_bytes,
)
from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    parse_archive,
)


def _write_single_member_zip(path: Path, contents: bytes) -> None:
    info = zipfile.ZipInfo(filename=PR106_BIN_MEMBER_NAME, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, mode="w") as zf:
        zf.writestr(info, contents)


@pytest.fixture
def fake_pr106_archive(tmp_path: Path) -> Path:
    payload = b"\x42" * 1024  # 1 KB synthetic
    arc = tmp_path / "fake_pr106.zip"
    _write_single_member_zip(arc, payload)
    return arc


@pytest.fixture
def residual_blob() -> bytes:
    return b"\x01\x02\x03\x04" * 64  # 256 B


# ── extract_pr106_bytes ─────────────────────────────────────────────────────


def test_extract_pr106_bytes_round_trip(fake_pr106_archive: Path) -> None:
    pr106_bytes, sha = extract_pr106_bytes(fake_pr106_archive)
    assert pr106_bytes == b"\x42" * 1024
    assert sha == hashlib.sha256(fake_pr106_archive.read_bytes()).hexdigest()


def test_extract_refuses_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(MaterializerError, match="not found"):
        extract_pr106_bytes(tmp_path / "does_not_exist.zip")


def test_extract_refuses_zip_without_0_bin(tmp_path: Path) -> None:
    arc = tmp_path / "wrong.zip"
    info = zipfile.ZipInfo(filename="not_0.bin", date_time=(2025, 1, 1, 0, 0, 0))
    with zipfile.ZipFile(arc, mode="w") as zf:
        zf.writestr(info, b"x")
    with pytest.raises(MaterializerError, match="0.bin"):
        extract_pr106_bytes(arc)


def test_extract_refuses_multi_member_zip(tmp_path: Path) -> None:
    arc = tmp_path / "multi.zip"
    with zipfile.ZipFile(arc, mode="w") as zf:
        zf.writestr(zipfile.ZipInfo(filename="0.bin", date_time=(2025, 1, 1, 0, 0, 0)), b"x")
        zf.writestr(zipfile.ZipInfo(filename="extra.bin", date_time=(2025, 1, 1, 0, 0, 0)), b"y")
    with pytest.raises(MaterializerError, match="only '0.bin'"):
        extract_pr106_bytes(arc)


def test_extract_refuses_empty_0_bin(tmp_path: Path) -> None:
    arc = tmp_path / "empty.zip"
    _write_single_member_zip(arc, b"")
    with pytest.raises(MaterializerError, match="empty"):
        extract_pr106_bytes(arc)


# ── emit_archive_zip ────────────────────────────────────────────────────────


def test_emit_archive_zip_deterministic(tmp_path: Path) -> None:
    """Same input bytes produce byte-identical zip outputs."""
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    emit_archive_zip(b"hello world", a)
    emit_archive_zip(b"hello world", b)
    assert a.read_bytes() == b.read_bytes()


def test_emit_archive_zip_can_be_read_back(tmp_path: Path) -> None:
    out = tmp_path / "ok.zip"
    emit_archive_zip(b"\x42\x43\x44", out)
    with zipfile.ZipFile(out, mode="r") as zf:
        assert zf.namelist() == [PR106_BIN_MEMBER_NAME]
        assert zf.read(PR106_BIN_MEMBER_NAME) == b"\x42\x43\x44"


# ── materialize_family_archive end-to-end ──────────────────────────────────


@pytest.mark.parametrize("family", sorted(PR106_RESIDUAL_FORMAT_IDS))
def test_materialize_each_family_emits_archive_and_manifest(
    family: str, fake_pr106_archive: Path, residual_blob: bytes, tmp_path: Path
) -> None:
    output_dir = tmp_path / f"out_{family}"
    archive_zip, manifest_path, manifest, build_result = materialize_family_archive(
        family=family,
        pr106_archive=fake_pr106_archive,
        residual_bytes=residual_blob,
        output_dir=output_dir,
    )
    assert archive_zip.is_file()
    assert manifest_path.is_file()
    # Manifest is valid JSON with the family + format_id reported correctly.
    data = json.loads(manifest_path.read_text())
    assert data["family"] == family
    assert data["format_id"] == PR106_RESIDUAL_FORMAT_IDS[family]
    # Promotion-status invariants pinned False.
    assert data["score_claim"] is False
    assert data["promotion_eligible"] is False
    assert data["ready_for_exact_eval_dispatch"] is False
    assert data["evidence_grade"] == "research_signal"
    # The archive zip contains 0.bin with the expected sha.
    with zipfile.ZipFile(archive_zip, mode="r") as zf:
        members = zf.namelist()
        assert members == [PR106_BIN_MEMBER_NAME]
        archive_bytes_from_zip = zf.read(PR106_BIN_MEMBER_NAME)
    assert archive_bytes_from_zip == build_result.archive_bytes
    assert manifest.archive_sha256 == sha256_bytes(build_result.archive_bytes)
    # Round-trip parse passes.
    parsed = parse_archive(build_result.archive_bytes)
    assert parsed.family == family
    assert parsed.residual_bytes == residual_blob


def test_materialize_refuses_unknown_family(fake_pr106_archive: Path, tmp_path: Path) -> None:
    with pytest.raises(MaterializerError, match="unknown family"):
        materialize_family_archive(
            family="nerv",
            pr106_archive=fake_pr106_archive,
            residual_bytes=b"\x00",
            output_dir=tmp_path / "out",
        )


def test_materialize_deterministic_bytes(
    fake_pr106_archive: Path, residual_blob: bytes, tmp_path: Path
) -> None:
    """Two runs with identical inputs produce byte-identical archive zips."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    archive_a, _, _, _ = materialize_family_archive(
        family="wavelet",
        pr106_archive=fake_pr106_archive,
        residual_bytes=residual_blob,
        output_dir=a_dir,
    )
    archive_b, _, _, _ = materialize_family_archive(
        family="wavelet",
        pr106_archive=fake_pr106_archive,
        residual_bytes=residual_blob,
        output_dir=b_dir,
    )
    assert archive_a.read_bytes() == archive_b.read_bytes()


def test_manifest_extra_field_round_trip(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    """The 'extra' field carries family-specific metadata into the manifest."""
    extra = {"residual_encoder": "haar_db1_single_level", "byte_budget_kib": 4}
    _, manifest_path, _, _ = materialize_family_archive(
        family="wavelet",
        pr106_archive=fake_pr106_archive,
        residual_bytes=b"\x00\x01\x02\x03",
        output_dir=tmp_path / "out",
        extra=extra,
    )
    data = json.loads(manifest_path.read_text())
    assert data["extra"] == extra


# ── no-op detector smoke ─────────────────────────────────────────────────────


def test_no_op_detector_byte_mutation_detects_change(
    fake_pr106_archive: Path, residual_blob: bytes, tmp_path: Path
) -> None:
    _, _, _, build = materialize_family_archive(
        family="wavelet",
        pr106_archive=fake_pr106_archive,
        residual_bytes=residual_blob,
        output_dir=tmp_path / "out",
    )
    smoke = run_no_op_detector_byte_mutation(
        archive_bytes=build.archive_bytes,
        expected_format_id=PR106_RESIDUAL_FORMAT_IDS["wavelet"],
    )
    assert smoke["result"] == "passed"
    assert smoke["offset_mutated"] > 0


def test_no_op_detector_skips_empty_residual(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    _, _, _, build = materialize_family_archive(
        family="c3",
        pr106_archive=fake_pr106_archive,
        residual_bytes=b"",
        output_dir=tmp_path / "out",
    )
    smoke = run_no_op_detector_byte_mutation(
        archive_bytes=build.archive_bytes,
        expected_format_id=PR106_RESIDUAL_FORMAT_IDS["c3"],
    )
    assert smoke["result"] == "skipped_empty_residual"


def test_no_op_detector_refuses_wrong_format_id(
    fake_pr106_archive: Path, residual_blob: bytes, tmp_path: Path
) -> None:
    _, _, _, build = materialize_family_archive(
        family="siren",
        pr106_archive=fake_pr106_archive,
        residual_bytes=residual_blob,
        output_dir=tmp_path / "out",
    )
    # Pass the WRONG expected_format_id to verify refusal.
    with pytest.raises(MaterializerError, match="format_id mismatch"):
        run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS["wavelet"],
        )


def test_emit_archive_zip_overwrite_safe(tmp_path: Path) -> None:
    """Emitting twice to the same path overwrites cleanly."""
    out = tmp_path / "x.zip"
    emit_archive_zip(b"\x01", out)
    emit_archive_zip(b"\x02", out)
    with zipfile.ZipFile(out, mode="r") as zf:
        assert zf.read(PR106_BIN_MEMBER_NAME) == b"\x02"
