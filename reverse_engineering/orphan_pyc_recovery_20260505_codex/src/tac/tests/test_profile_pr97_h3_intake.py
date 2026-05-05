# pyc-recovery pass2: rehydrated from git blob aa0ee02119690be8270efd364f8e80528e5a9384 via `git fsck --lost-found`
# original path: src/tac/tests/test_profile_pr97_h3_intake.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
from __future__ import annotations

import importlib.util
import io
import lzma
import struct
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "profile_pr97_h3_intake.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr97_h3_intake", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_runtime(root: Path) -> Path:
    root.mkdir()
    (root / "schema_h3.py").write_text(
        "SCHEMA = [\n"
        "  ('conv.weight', 'fp4_w', (2, 1, 3, 3)),\n"
        "  ('conv.bias', 'fp16_b', (2,)),\n"
        "  ('linear.weight', 'fp16_w', (2, 2)),\n"
        "]\n",
        encoding="utf-8",
    )
    (root / "inflate.py").write_text("import brotli, lzma\nfrom pathlib import Path\n", encoding="utf-8")
    (root / "inflate.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    (root / "sidecar.py").write_text("import lzma\n", encoding="utf-8")
    return root


def _model_raw() -> bytes:
    fp4_blocks = 1
    conv = b"\x12" * 16 + b"\x00\x3c" * fp4_blocks
    bias = b"\x00\x00" * 2
    linear = b"\x00\x00" * 4
    return conv + bias + linear


def _pose_blob() -> bytes:
    raw = io.BytesIO()
    raw.write(struct.pack("<II", 3, 2))
    raw.write(bytes([3, 2]))
    raw.write(struct.pack("<ff", 1.0, 0.25))
    raw.write(struct.pack("<ff", -1.0, 0.5))
    raw.write(b"\xaa\xbb")
    return brotli.compress(raw.getvalue(), quality=5)


def _sidecar_blob() -> bytes:
    raw = io.BytesIO()
    raw.write(b"BPGD")
    raw.write(struct.pack("<H", 2))
    raw.write(struct.pack("<H", 1))
    raw.write(bytes([1]))
    raw.write(bytes([1]))
    raw.write(b"\x01\x02\x03")
    raw.write(bytes([2]))
    raw.write(bytes([16]))
    raw.write(struct.pack("<bb", 2, -1))
    return lzma.compress(raw.getvalue(), format=lzma.FORMAT_XZ, preset=6)


def _payload() -> bytes:
    mask = struct.pack("<I", 2) + struct.pack("<I", 64) + b"A" * 64 + struct.pack("<I", 32) + b"B" * 32
    parts = [
        mask,
        _pose_blob(),
        brotli.compress(_model_raw(), quality=5),
        _sidecar_blob(),
    ]
    return b"".join(struct.pack("<I", len(part)) + part for part in parts)


def _write_archive(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p", (2026, 5, 3, 23, 13, 44))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_profile_pr97_h3_parses_subformats_and_candidates(tmp_path: Path) -> None:
    module = load_module()
    runtime = _write_runtime(tmp_path / "runtime")
    archive = tmp_path / "archive.zip"
    payload = _payload()
    _write_archive(archive, payload)

    profile = module.build_profile(archive, runtime, tmp_path / "out")

    assert profile["schema"] == "pr97_h3_static_intake_profile_v1"
    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["archive"]["members"][0]["name"] == "p"
    assert profile["payload"]["parts"]["mask"]["bytes"] == 108
    assert profile["mask"]["chunk_count"] == 2
    assert profile["pose"]["bits_per_dim"] == [3, 2]
    assert profile["pose"]["needed_bitstream_bytes"] == 2
    assert profile["model"]["schema_entries"] == 3
    assert profile["model"]["fp4_params"] == 18
    assert profile["sidecar"]["pair_record_count"] == 2
    assert profile["sidecar"]["counts"]["x2_pairs"] == 1
    assert profile["sidecar"]["counts"]["warp_pairs"] == 1

    candidates = {row["label"]: row for row in profile["byte_opportunities"]["safe_repack_candidates"]}
    assert "pr97_deflated_p" in candidates
    assert "pr97_pose_model_br10_deflated_p" in candidates
    with zipfile.ZipFile(candidates["pr97_deflated_p"]["archive"], "r") as zf:
        assert zf.read("p") == payload
        assert zf.getinfo("p").compress_type == zipfile.ZIP_DEFLATED


def test_profile_pr97_h3_rejects_bad_model_schema_length(tmp_path: Path) -> None:
    module = load_module()
    runtime = _write_runtime(tmp_path / "runtime")
    archive = tmp_path / "archive.zip"
    (runtime / "schema_h3.py").write_text(
        "SCHEMA = [\n"
        "  ('conv.weight', 'fp4_w', (2, 1, 3, 3)),\n"
        "  ('conv.bias', 'fp16_b', (2,)),\n"
        "  ('linear.weight', 'fp16_w', (2, 2)),\n"
        "  ('unexpected.extra', 'fp16_w', (8,)),\n"
        "]\n",
        encoding="utf-8",
    )
    _write_archive(archive, _payload())

    with pytest.raises(module.PR97ProfileError, match="model schema consumed"):
        module.build_profile(archive, runtime, None)
