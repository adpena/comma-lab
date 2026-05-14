# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "build_ibps1_byte_patch_archive.py"


def _load_module():
    name = "_ibps_patch_builder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ibps_inner() -> bytes:
    enc = b"E" * 3
    dec = bytearray(b"D" * 8)
    lat = b"L" * 4
    meta = b"{}"
    header = struct.pack(
        "<4sBHHIIII",
        b"IBPS",
        1,
        2,
        2,
        len(enc),
        len(dec),
        len(lat),
        len(meta),
    )
    return header + enc + bytes(dec) + lat + meta


def test_build_patch_archive_flips_section_relative_byte(tmp_path: Path):
    mod = _load_module()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", _ibps_inner())
    output = tmp_path / "out" / "archive.zip"
    manifest = mod.build_patch_archive(
        source_archive=source,
        output_archive=output,
        patch_specs=["decoder_blob:2"],
    )
    assert manifest["score_claim"] is False
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        patched = zf.read("0.bin")
    sections = mod.parse_ibps1_archive_bytes(patched)
    start, _length = sections["decoder_blob"]
    assert patched[start + 2] == (ord("D") ^ 0xFF)
    assert manifest["patches"][0]["relative_offset"] == 2


def test_main_writes_manifest(tmp_path: Path):
    mod = _load_module()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", _ibps_inner())
    output = tmp_path / "archive.zip"
    manifest = tmp_path / "manifest.json"
    rc = mod.main([
        "--source-archive", str(source),
        "--output-archive", str(output),
        "--manifest", str(manifest),
        "--patch", "decoder_blob:0",
    ])
    assert rc == 0
    assert output.exists()
    assert manifest.exists()
