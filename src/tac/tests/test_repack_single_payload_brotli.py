# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "repack_single_payload_brotli.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("repack_single_payload_brotli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def test_repack_archive_preserves_raw_payload_and_records_custody(tmp_path: Path) -> None:
    repack = _load_script()
    raw = (b"comma-video-payload" * 2048) + bytes(range(251))
    source_payload = brotli.compress(raw, quality=1, mode=brotli.MODE_GENERIC, lgwin=18)
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": source_payload})

    out = tmp_path / "out" / "archive.zip"
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = repack.repack_archive(
        source_archive=source,
        output_archive=out,
        manifest_json=manifest_path,
        quality=11,
        mode=brotli.MODE_FONT,
        lgwin=18,
        lgblock=0,
    )

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["p"]
        assert brotli.decompress(zf.read("p")) == raw
        info = zf.getinfo("p")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["lossless_payload_roundtrip"] is True
    assert manifest["archive_delta_bytes"] < 0
    assert manifest_path.exists()


def test_repack_rejects_multi_member_archives(tmp_path: Path) -> None:
    repack = _load_script()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": brotli.compress(b"payload"), "debug.txt": b"sidecar"})

    try:
        repack.repack_archive(
            source_archive=source,
            output_archive=tmp_path / "out.zip",
            manifest_json=tmp_path / "manifest.json",
            require_improvement=False,
        )
    except repack.RepackError as exc:
        assert "expected single member" in str(exc)
    else:
        raise AssertionError("multi-member source archive was accepted")


def test_repack_rejects_non_brotli_single_payload_with_domain_error(tmp_path: Path) -> None:
    repack = _load_script()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"raw-not-brotli"})

    try:
        repack.repack_archive(
            source_archive=source,
            output_archive=tmp_path / "out.zip",
            manifest_json=tmp_path / "manifest.json",
            require_improvement=False,
        )
    except repack.RepackError as exc:
        assert "not a Brotli-compressed payload" in str(exc)
    else:
        raise AssertionError("non-Brotli payload was accepted")
