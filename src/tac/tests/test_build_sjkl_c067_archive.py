"""Tests for experiments/build_sjkl_c067_archive.py — recovery + roundtrip verification."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = REPO_ROOT / "experiments" / "build_sjkl_c067_archive.py"
spec = importlib.util.spec_from_file_location("build_sjkl_c067_archive", _BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
sys.modules["build_sjkl_c067_archive"] = builder
spec.loader.exec_module(builder)


def _make_synthetic_c067_archive(path: Path, payload_bytes: bytes = b"\x5b\x98\x68\x43" + b"\x00" * 100) -> bytes:
    """Synthesize a C067-style single-member 'p' archive for testing."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        zi = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, payload_bytes)
    return payload_bytes


def _make_synthetic_sjkl_bin(path: Path) -> bytes:
    # Use realistic sjkl.bin shape (SJKL magic + TOC stub + dummy basis + dummy alpha)
    data = b"SJKL" + b"\x10\x00\x00\x00" + b"\x20\x00\x00\x00" + b"\xaa" * 16 + b"SJK2" + b"\xbb" * 28
    path.write_bytes(data)
    return data


def test_top_level_sibling_preserves_source_bytes(tmp_path):
    """Critical correctness invariant: source `p` payload bytes are byte-identical."""
    src_path = tmp_path / "source.zip"
    sjkl_path = tmp_path / "sjkl.bin"
    p_payload = b"\x5b\x98\x68\x43" + b"\x42" * 1000
    _make_synthetic_c067_archive(src_path, payload_bytes=p_payload)
    _make_synthetic_sjkl_bin(sjkl_path)

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="sjkl.bin",
    )
    manifest = builder.build_sjkl_c067_archive(cfg)

    out_archive = tmp_path / "out" / "archive.zip"
    assert out_archive.is_file()

    with zipfile.ZipFile(out_archive, "r") as z:
        names = z.namelist()
        assert "p" in names
        assert "sjkl.bin" in names
        # source p payload bytes unchanged
        assert z.read("p") == p_payload
        # sjkl.bin matches what we wrote
        assert z.read("sjkl.bin") == sjkl_path.read_bytes()


def test_top_level_sibling_emits_manifest(tmp_path):
    src_path = tmp_path / "source.zip"
    sjkl_path = tmp_path / "sjkl.bin"
    _make_synthetic_c067_archive(src_path)
    _make_synthetic_sjkl_bin(sjkl_path)

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="sjkl.bin",
    )
    builder.build_sjkl_c067_archive(cfg)
    manifest_path = tmp_path / "out" / "sjkl_c067_archive_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["archive_layout"] == "top_level_sibling"
    assert manifest["score_claim"] is False
    assert manifest["sjkl_bin_bytes"] == sjkl_path.stat().st_size
    assert "p" in manifest["source_member_names"]
    assert manifest["sjkl_member_name"] == "sjkl.bin"


def test_packed_rpk1_layout_fails_loud(tmp_path):
    """SAFE STUB: packed_rpk1 must raise NotImplementedError, not silently mis-pack."""
    src_path = tmp_path / "source.zip"
    sjkl_path = tmp_path / "sjkl.bin"
    _make_synthetic_c067_archive(src_path)
    _make_synthetic_sjkl_bin(sjkl_path)

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="packed_rpk1",
        sjkl_member_name="sjkl.bin",
    )
    with pytest.raises(NotImplementedError, match="recovery stub"):
        builder.build_sjkl_c067_archive(cfg)


def test_refuses_overwriting_existing_member(tmp_path):
    """If source archive already contains a member with sjkl_member_name, refuse."""
    src_path = tmp_path / "source.zip"
    with zipfile.ZipFile(src_path, "w") as z:
        z.writestr("p", b"\x00" * 100)
        z.writestr("sjkl.bin", b"PRE-EXISTING")  # collision
    sjkl_path = tmp_path / "sjkl.bin"
    _make_synthetic_sjkl_bin(sjkl_path)

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="sjkl.bin",
    )
    with pytest.raises(SystemExit, match="already contains member"):
        builder.build_sjkl_c067_archive(cfg)


def test_refuses_missing_source_archive(tmp_path):
    cfg = builder.ArchiveBuildConfig(
        source_archive=tmp_path / "does_not_exist.zip",
        sjkl_bin=tmp_path / "sjkl.bin",
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="sjkl.bin",
    )
    with pytest.raises(SystemExit, match="source archive not found"):
        builder.build_sjkl_c067_archive(cfg)


def test_refuses_empty_sjkl_bin(tmp_path):
    src_path = tmp_path / "source.zip"
    _make_synthetic_c067_archive(src_path)
    sjkl_path = tmp_path / "sjkl.bin"
    sjkl_path.write_bytes(b"")

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="sjkl.bin",
    )
    with pytest.raises(SystemExit, match="sjkl.bin is empty"):
        builder.build_sjkl_c067_archive(cfg)


def test_deterministic_output_byte_identical_across_runs(tmp_path):
    """Determinism: two builds with same inputs produce byte-identical output ZIPs."""
    src_path = tmp_path / "source.zip"
    sjkl_path = tmp_path / "sjkl.bin"
    _make_synthetic_c067_archive(src_path)
    _make_synthetic_sjkl_bin(sjkl_path)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        cfg = builder.ArchiveBuildConfig(
            source_archive=src_path, sjkl_bin=sjkl_path,
            output_dir=out, archive_layout="top_level_sibling",
            sjkl_member_name="sjkl.bin",
        )
        builder.build_sjkl_c067_archive(cfg)
    assert (out1 / "archive.zip").read_bytes() == (out2 / "archive.zip").read_bytes()


def test_custom_sjkl_member_name(tmp_path):
    """Operator can override the ZIP member name (e.g., for sibling-layout variants)."""
    src_path = tmp_path / "source.zip"
    sjkl_path = tmp_path / "sjkl.bin"
    _make_synthetic_c067_archive(src_path)
    _make_synthetic_sjkl_bin(sjkl_path)

    cfg = builder.ArchiveBuildConfig(
        source_archive=src_path, sjkl_bin=sjkl_path,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
        sjkl_member_name="charged_sjkl.bin",  # custom name
    )
    builder.build_sjkl_c067_archive(cfg)
    with zipfile.ZipFile(tmp_path / "out" / "archive.zip") as z:
        assert "charged_sjkl.bin" in z.namelist()
        assert "sjkl.bin" not in z.namelist()


def test_cli_argv_parses():
    parser = builder.build_parser()
    args = parser.parse_args([
        "--source-archive", "/tmp/src.zip",
        "--sjkl-bin", "/tmp/sjkl.bin",
        "--output-dir", "/tmp/out",
        "--archive-layout", "top_level_sibling",
        "--sjkl-member-name", "alt.bin",
    ])
    assert args.archive_layout == "top_level_sibling"
    assert args.sjkl_member_name == "alt.bin"
