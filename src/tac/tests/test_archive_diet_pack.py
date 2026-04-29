"""Tests for src/tac/archive_diet_pack.py — CPU-only archive diet packer.

Designed by codex gpt-5.5 xhigh under Stage 3 of the orchestrated workflow.
Score-savings deltas computed by this module are [advisory only] because
they reflect archive-byte changes, not a fresh contest-CUDA authoritative eval.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from tac.archive_diet_pack import diet_pack
from tac.submission_archive import ORIGINAL_VIDEO_BYTES


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a minimal ZIP containing the given (name -> bytes) members."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _renderer_archive(tmp_path: Path) -> Path:
    """A renderer-style archive with renderer.bin + masks.mkv + poses."""
    return _write_zip(
        tmp_path / "renderer_archive.zip",
        {
            "renderer.bin": b"R" * 4096,
            "masks.mkv": b"M" * 8192,
            "optimized_poses.bin": b"P" * 1024,
        },
    )


def test_diet_pack_renderer_archive_smoke(tmp_path: Path) -> None:
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    result = diet_pack(src, dst, verify=True)

    assert result["bit_exact"] is True
    assert result["input_bytes"] == src.stat().st_size
    assert result["output_bytes"] == dst.stat().st_size
    assert result["savings_bytes"] == result["input_bytes"] - result["output_bytes"]
    # Score savings are advisory only (rate-only delta).
    expected_savings_pts = 25.0 * result["savings_bytes"] / ORIGINAL_VIDEO_BYTES
    assert result["savings_score_pts"] == pytest.approx(expected_savings_pts, abs=1e-12)


def test_diet_pack_output_uses_br_suffix(tmp_path: Path) -> None:
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    diet_pack(src, dst, verify=True)
    with zipfile.ZipFile(dst, "r") as zf:
        names = sorted(info.filename for info in zf.infolist())
    assert all(name.endswith(".br") for name in names)
    assert "renderer.bin.br" in names
    assert "masks.mkv.br" in names
    assert "optimized_poses.bin.br" in names


def test_diet_pack_decoded_members_byte_exact(tmp_path: Path) -> None:
    """Decoding `.br` members must be byte-identical to inputs."""
    import brotli

    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    diet_pack(src, dst, verify=True)

    with zipfile.ZipFile(src, "r") as src_zip, zipfile.ZipFile(dst, "r") as dst_zip:
        src_members = {info.filename: src_zip.read(info) for info in src_zip.infolist()}
        for info in dst_zip.infolist():
            assert info.filename.endswith(".br")
            logical = info.filename[:-3]
            decoded = brotli.decompress(dst_zip.read(info))
            assert decoded == src_members[logical], f"Brotli round-trip mismatch on {logical}"


def test_diet_pack_deterministic_output(tmp_path: Path) -> None:
    """Two runs on the same input produce byte-identical archives."""
    src = _renderer_archive(tmp_path)
    dst1 = tmp_path / "out1.zip"
    dst2 = tmp_path / "out2.zip"
    diet_pack(src, dst1, verify=False)
    diet_pack(src, dst2, verify=False)
    assert dst1.read_bytes() == dst2.read_bytes()


def test_diet_pack_deterministic_zip_timestamps(tmp_path: Path) -> None:
    """All ZipInfo timestamps must be the canonical 1980-01-01 sentinel."""
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    diet_pack(src, dst, verify=False)
    with zipfile.ZipFile(dst, "r") as zf:
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0), info.filename


def test_diet_pack_rejects_invalid_brotli_quality(tmp_path: Path) -> None:
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="brotli_quality"):
        diet_pack(src, dst, brotli_quality=12)
    with pytest.raises(ValueError, match="brotli_quality"):
        diet_pack(src, dst, brotli_quality=-1)


def test_diet_pack_rejects_non_zip_input(tmp_path: Path) -> None:
    src = tmp_path / "fake.zip"
    src.write_bytes(b"not a zip file at all")
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="not a valid zip archive"):
        diet_pack(src, dst)


def test_diet_pack_rejects_missing_input(tmp_path: Path) -> None:
    src = tmp_path / "missing.zip"
    dst = tmp_path / "out.zip"
    with pytest.raises(FileNotFoundError):
        diet_pack(src, dst)


def test_diet_pack_rejects_empty_archive(tmp_path: Path) -> None:
    src = tmp_path / "empty.zip"
    with zipfile.ZipFile(src, "w") as _:
        pass  # empty archive
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="empty archive"):
        diet_pack(src, dst)


def test_diet_pack_rejects_unsupported_layout(tmp_path: Path) -> None:
    """Archives without renderer.bin OR segmap_weights.tar.xz must be rejected."""
    src = _write_zip(
        tmp_path / "wrong_layout.zip",
        {"random_payload.dat": b"not a renderer or segmap"},
    )
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="unsupported archive layout"):
        diet_pack(src, dst)


def test_diet_pack_rejects_already_brotli_repacked(tmp_path: Path) -> None:
    """Refuse to double-Brotli a previously diet-packed archive."""
    src = _renderer_archive(tmp_path)
    intermediate = tmp_path / "intermediate.zip"
    diet_pack(src, intermediate, verify=False)
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="already contains .br members"):
        diet_pack(intermediate, dst)


def test_diet_pack_rejects_unsafe_member_paths(tmp_path: Path) -> None:
    src = _write_zip(
        tmp_path / "unsafe.zip",
        {
            "renderer.bin": b"R" * 16,
            "../escape.bin": b"X",
        },
    )
    dst = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="unsafe archive member path"):
        diet_pack(src, dst)


def test_diet_pack_components_breakdown(tmp_path: Path) -> None:
    """`components` field reports per-member input/output byte sizes."""
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    result = diet_pack(src, dst, verify=False)
    assert "renderer.bin" in result["components"]
    rb = result["components"]["renderer.bin"]
    assert rb["in"] == 4096
    assert 0 < rb["out"] < rb["in"], "Brotli should compress repetitive data"
    # Total output sum must be ≤ archive bytes (header overhead is small).
    components_total = sum(c["out"] for c in result["components"].values())
    assert components_total <= result["output_bytes"]


def test_diet_pack_advisory_only_savings_pts_is_rate_only(tmp_path: Path) -> None:
    """savings_score_pts is purely rate (no distortion claim)."""
    src = _renderer_archive(tmp_path)
    dst = tmp_path / "out.zip"
    result = diet_pack(src, dst, verify=False)
    # Formula: 25 * Δbytes / 37,545,489 (RAW_VIDEO_BYTES). Tagged [advisory only]
    # because byte-savings ≠ contest score; only contest-CUDA evaluate.py counts.
    assert result["savings_score_pts"] == pytest.approx(
        25.0 * result["savings_bytes"] / ORIGINAL_VIDEO_BYTES, abs=1e-12
    )
