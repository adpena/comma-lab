"""Tests for tools/pr106_archive_substitution_surgery.py."""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "pr106_archive_substitution_surgery.py"


def _load_surgery_module():
    spec = importlib.util.spec_from_file_location(
        "pr106_archive_substitution_surgery", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr106_archive_substitution_surgery"] = module
    spec.loader.exec_module(module)
    return module


def _build_synthetic_pr106_archive(
    path: Path,
    decoder_payload: bytes,
    tail_payload: bytes,
) -> None:
    """Write a synthetic PR106 archive with controllable section bytes."""
    decoder_brotli = brotli.compress(decoder_payload)
    tail_brotli = brotli.compress(tail_payload)
    header = bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little")
    inner = header + decoder_brotli + tail_brotli
    info = zipfile.ZipInfo(filename="0.bin")
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, inner)


def test_verify_byte_layout_synthetic(tmp_path: Path) -> None:
    surgery = _load_surgery_module()
    archive = tmp_path / "synthetic.zip"
    _build_synthetic_pr106_archive(archive, b"decoder_payload" * 100, b"tail" * 50)
    layout = surgery.verify_byte_layout(archive)
    assert layout["header"]["magic_byte"] == "0xff"
    assert layout["decoder_packed_brotli"]["len"] > 0
    assert layout["latents_and_sidecar_brotli"]["len"] > 0


def test_substitute_decoder_round_trip(tmp_path: Path) -> None:
    """Replace decoder; verify tail is byte-faithful + report flags set."""
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    out_archive = tmp_path / "out.zip"
    original_decoder = b"original_decoder_payload" * 200
    tail = b"original_tail_payload" * 100
    _build_synthetic_pr106_archive(in_archive, original_decoder, tail)

    new_decoder_payload = b"new_decoder" * 150
    new_decoder_brotli = brotli.compress(new_decoder_payload)
    report = surgery.substitute_decoder(
        input_archive=in_archive,
        replacement_decoder=new_decoder_brotli,
        output_archive=out_archive,
    )

    # The replacement decoder lives at the new offset; original tail bytes
    # must be preserved exactly.
    assert report.score_affecting_payload_changed is True
    assert report.charged_bits_changed is True
    assert report.target_modes == ["contest_exact_eval"]
    assert report.deployment_target == "t4_contest_runtime"
    assert report.decoder_replacement_len == len(new_decoder_brotli)
    # Tail SHA must not change
    assert report.sha256_input_latents_sidecar == report.sha256_output_latents_sidecar

    # Output archive must reparse cleanly with brotli on both sections
    out_layout = surgery.verify_byte_layout(out_archive)
    out_decoder_offset = out_layout["decoder_packed_brotli"]["offset"]
    out_decoder_len = out_layout["decoder_packed_brotli"]["len"]
    inner = surgery._read_inner_blob(out_archive)
    assert (
        inner[out_decoder_offset : out_decoder_offset + out_decoder_len]
        == new_decoder_brotli
    )


def test_substitute_decoder_rejects_non_brotli(tmp_path: Path) -> None:
    """Raw garbage bytes should fail at the brotli sanity gate."""
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    out_archive = tmp_path / "out.zip"
    _build_synthetic_pr106_archive(in_archive, b"orig" * 100, b"tail" * 50)

    with pytest.raises(ValueError, match="not brotli-decompressible"):
        surgery.substitute_decoder(
            input_archive=in_archive,
            replacement_decoder=b"\x00\x01\x02\x03not_brotli_bytes",
            output_archive=out_archive,
        )
    # No archive should have been written on failure
    assert not out_archive.exists()


def test_substitute_decoder_rejects_empty(tmp_path: Path) -> None:
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    _build_synthetic_pr106_archive(in_archive, b"orig" * 100, b"tail" * 50)
    with pytest.raises(ValueError, match="empty"):
        surgery.substitute_decoder(
            input_archive=in_archive,
            replacement_decoder=b"",
            output_archive=in_archive.parent / "out.zip",
        )


def test_substitute_latents_and_sidecar_preserves_decoder(tmp_path: Path) -> None:
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    out_archive = tmp_path / "out.zip"
    decoder = b"decoder_payload" * 200
    _build_synthetic_pr106_archive(in_archive, decoder, b"orig_tail" * 100)

    new_tail_payload = b"new_tail" * 80
    new_tail_brotli = brotli.compress(new_tail_payload)
    report = surgery.substitute_latents_and_sidecar(
        input_archive=in_archive,
        replacement_tail=new_tail_brotli,
        output_archive=out_archive,
    )

    assert report.score_affecting_payload_changed is True
    assert report.sha256_input_decoder == report.sha256_output_decoder, (
        "decoder bytes changed during latents+sidecar substitution"
    )
    assert report.latents_sidecar_output_len == len(new_tail_brotli)


def test_target_modes_pass_through(tmp_path: Path) -> None:
    """Custom target_modes appear verbatim in the SubstitutionReport."""
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    out_archive = tmp_path / "out.zip"
    _build_synthetic_pr106_archive(in_archive, b"orig" * 100, b"tail" * 50)
    new_decoder = brotli.compress(b"new" * 50)
    report = surgery.substitute_decoder(
        input_archive=in_archive,
        replacement_decoder=new_decoder,
        output_archive=out_archive,
        target_modes=["openpilot_edge", "production_generalized"],
        deployment_target="comma_ai_production",
    )
    assert report.target_modes == ["openpilot_edge", "production_generalized"]
    assert report.deployment_target == "comma_ai_production"


def test_substitute_decoder_oversize_rejected(tmp_path: Path) -> None:
    """Replacement >24-bit must fail (header limit)."""
    surgery = _load_surgery_module()
    in_archive = tmp_path / "in.zip"
    _build_synthetic_pr106_archive(in_archive, b"orig" * 100, b"tail" * 50)
    # 24-bit limit is 16,777,215 — we don't actually allocate that, just
    # sniff the guard
    with pytest.raises(ValueError, match="24-bit"):
        surgery.substitute_decoder(
            input_archive=in_archive,
            replacement_decoder=b"\x00" * 0x1000000,  # 16,777,216 = limit + 1
            output_archive=in_archive.parent / "out.zip",
        )
