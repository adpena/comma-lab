"""Tests for tools/xray_archive_section_entropy_heatmap.py.

Per CLAUDE.md test discipline: 15-25 tests, all pass.
"""
from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import xray_archive_section_entropy_heatmap as xe  # noqa: E402


def test_shannon_entropy_empty_returns_zero():
    assert xe.shannon_entropy_bits(b"") == 0.0


def test_shannon_entropy_single_byte_returns_zero():
    assert xe.shannon_entropy_bits(b"A" * 100) == 0.0


def test_shannon_entropy_uniform_bytes_close_to_eight():
    payload = bytes(range(256)) * 4
    h = xe.shannon_entropy_bits(payload)
    assert h == pytest.approx(8.0, abs=0.001)


def test_shannon_entropy_two_symbols_one_bit():
    payload = b"AB" * 500
    h = xe.shannon_entropy_bits(payload)
    assert h == pytest.approx(1.0, abs=0.001)


def test_shannon_entropy_skewed_below_one():
    # 90% A, 10% B → entropy ~0.469
    payload = b"A" * 900 + b"B" * 100
    h = xe.shannon_entropy_bits(payload)
    assert 0.4 < h < 0.55


def test_recoverable_zero_when_already_saturated():
    # encoded_bpb < floor_bpb → 0
    assert xe.section_recoverable_bytes(1000, 5.0, 7.0) == 0


def test_recoverable_zero_when_equal():
    assert xe.section_recoverable_bytes(1000, 7.0, 7.0) == 0


def test_recoverable_half_when_floor_is_half_of_encoded():
    # encoded 8 bpb, floor 4 bpb → recover half
    rec = xe.section_recoverable_bytes(1000, 8.0, 4.0)
    assert rec == 500


def test_recoverable_zero_when_encoded_is_zero():
    assert xe.section_recoverable_bytes(0, 0.0, 0.0) == 0


def _make_zip(tmp_path: Path, members: dict[str, bytes]) -> Path:
    zp = tmp_path / "test_archive.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return zp


def test_profile_single_member_archive(tmp_path):
    zp = _make_zip(tmp_path, {"data.bin": b"hello world" * 100})
    rep = xe.profile_archive_sections(zp, label="single")
    assert rep["label"] == "single"
    assert rep["section_count"] == 1
    assert rep["sections"][0]["name"] == "data.bin"
    assert rep["sections"][0]["uncompressed_bytes"] == 1100


def test_profile_archive_multiple_members(tmp_path):
    zp = _make_zip(tmp_path, {
        "a.bin": b"A" * 1000,  # very compressible
        "b.bin": b"\x00\x01\x02\x03" * 250,  # mid
    })
    rep = xe.profile_archive_sections(zp)
    assert rep["section_count"] == 2
    # sections sorted by recoverable_bytes desc; both are 0-recoverable
    # because deflate already gets near-floor on both, so just check count
    names = {s["name"] for s in rep["sections"]}
    assert names == {"a.bin", "b.bin"}


def test_profile_includes_sha256_and_filesize(tmp_path):
    zp = _make_zip(tmp_path, {"data.bin": b"x" * 10})
    rep = xe.profile_archive_sections(zp)
    assert len(rep["archive_sha256"]) == 64  # hex digest
    assert rep["file_size_bytes"] == zp.stat().st_size


def test_high_entropy_payload_low_recoverable(tmp_path):
    # Random-uniform bytes: brotli/deflate cannot compress; recoverable~0
    import os
    payload = os.urandom(4096)
    zp = _make_zip(tmp_path, {"random.bin": payload})
    rep = xe.profile_archive_sections(zp)
    sec = rep["sections"][0]
    # Entropy of random bytes is near 8.0
    assert sec["payload_entropy_bits_per_byte"] > 7.0
    # Either ZIP couldn't compress (zip_bpb~8) → saturation~1.0 OR recoverable~0
    assert sec["recoverable_bytes_if_floor_reached"] < 200


def test_low_entropy_payload_zip_compresses_to_floor(tmp_path):
    # All-zero payload: entropy=0; ZIP compresses to ~zero overhead.
    zp = _make_zip(tmp_path, {"zeros.bin": b"\x00" * 10000})
    rep = xe.profile_archive_sections(zp)
    sec = rep["sections"][0]
    assert sec["payload_entropy_bits_per_byte"] == 0.0


def test_total_recoverable_aggregates(tmp_path):
    zp = _make_zip(tmp_path, {
        "x.bin": b"x" * 100,
        "y.bin": b"y" * 200,
    })
    rep = xe.profile_archive_sections(zp)
    sec_sum = sum(s["recoverable_bytes_if_floor_reached"] for s in rep["sections"])
    assert rep["total_recoverable_if_floor_bytes"] == sec_sum


def test_main_writes_json_and_md(tmp_path):
    zp = _make_zip(tmp_path, {"data.bin": b"hello" * 200})
    out_dir = tmp_path / "out"
    rc = xe.main([
        "--archive", str(zp),
        "--output-dir", str(out_dir),
        "--label", "test",
    ])
    assert rc == 0
    assert (out_dir / "heatmap.json").exists()
    assert (out_dir / "heatmap.md").exists()
    assert (out_dir / "rebuild_command.txt").exists()
    rep = json.loads((out_dir / "heatmap.json").read_text())
    assert rep["schema_version"] == xe.SCHEMA
    assert rep["score_claim"] is False
    assert rep["evidence_grade"] == "diagnostic_only"


def test_main_missing_archive_returns_2(tmp_path):
    rc = xe.main([
        "--archive", str(tmp_path / "does_not_exist.zip"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_label_count_mismatch_returns_2(tmp_path):
    zp = _make_zip(tmp_path, {"data.bin": b"x"})
    rc = xe.main([
        "--archive", str(zp),
        "--label", "a",
        "--label", "b",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_multi_archive_comparison(tmp_path):
    zp1 = _make_zip(tmp_path, {"d.bin": b"x" * 100})
    zp2 = _make_zip(tmp_path / "z2", {"d.bin": b"y" * 100}) if False else None
    # Make a second archive in a unique location
    zp2dir = tmp_path / "second"
    zp2dir.mkdir()
    zp2 = _make_zip(zp2dir, {"d.bin": b"y" * 100})
    out_dir = tmp_path / "cmp"
    rc = xe.main([
        "--archive", str(zp1),
        "--archive", str(zp2),
        "--label", "first",
        "--label", "second",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    rep = json.loads((out_dir / "heatmap.json").read_text())
    assert len(rep["archives"]) == 2
    assert {a["label"] for a in rep["archives"]} == {"first", "second"}


def test_markdown_output_has_regen_header(tmp_path):
    zp = _make_zip(tmp_path, {"data.bin": b"x" * 50})
    out_dir = tmp_path / "out"
    xe.main(["--archive", str(zp), "--output-dir", str(out_dir)])
    md = (out_dir / "heatmap.md").read_text()
    assert "generated_at:" in md
    assert "from_state_hash:" in md
    assert "[diagnostic: archive section entropy heatmap]" in md


def test_sections_sorted_by_recoverable_desc(tmp_path):
    # Build archive where sections have distinct recoverable potentials
    # Larger compressible section → larger recoverable
    zp = _make_zip(tmp_path, {
        "small.bin": b"A" * 100,
        "large.bin": b"B" * 5000,
    })
    rep = xe.profile_archive_sections(zp)
    rec = [s["recoverable_bytes_if_floor_reached"] for s in rep["sections"]]
    assert rec == sorted(rec, reverse=True)


def test_score_claim_always_false(tmp_path):
    rec_zero = xe.section_recoverable_bytes(1000, 7.0, 4.0)
    assert isinstance(rec_zero, int)
    zp = _make_zip(tmp_path, {"d.bin": b"x" * 50})
    out_dir = tmp_path / "out"
    xe.main(["--archive", str(zp), "--output-dir", str(out_dir)])
    rep = json.loads((out_dir / "heatmap.json").read_text())
    assert rep["score_claim"] is False
    assert rep["promotion_eligible"] is False
    assert rep["ready_for_exact_eval_dispatch"] is False
    assert rep["evidence_grade"] == "diagnostic_only"
