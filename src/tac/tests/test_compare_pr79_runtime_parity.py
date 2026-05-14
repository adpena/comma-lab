# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "compare_pr79_runtime_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_pr79_runtime_parity_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qp1(words: list[int]) -> bytes:
    script = _load_script()
    out = bytearray(b"QP1" + int(words[0]).to_bytes(2, "little"))
    prev = int(words[0])
    for word in words[1:]:
        delta = int(word) - prev
        prev = int(word)
        zz = (delta << 1) ^ (delta >> 31)
        out += script.uvarint(zz)
    return bytes(out)


def _write_archive(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", payload)


def _p3_payload(actions_wire: bytes) -> bytes:
    import brotli

    mask_br = brotli.compress(b"\x12\x00\x0a\x0amask-obu")
    model_br = brotli.compress(b"QZS3model")
    pose_br = brotli.compress(_qp1([10_240, 10_241, 10_239]))
    return (
        b"P3"
        + len(mask_br).to_bytes(4, "little")
        + len(model_br).to_bytes(2, "little")
        + len(actions_wire).to_bytes(2, "little")
        + mask_br
        + model_br
        + actions_wire
        + pose_br
    )


def _s2_action_wire() -> bytes:
    import brotli

    script = _load_script()
    meta = b"S1" + script.uvarint(1) + script.uvarint(2) + script.uvarint(1) + script.uvarint(3)
    meta_br = brotli.compress(meta)
    return b"S2" + script.uvarint(1) + script.uvarint(len(meta_br)) + script.uvarint(8) + meta_br + b"\x00"


def test_public_and_robust_profiles_match_for_public_p3_payload(tmp_path: Path) -> None:
    import brotli

    script = _load_script()
    actions_raw = (3).to_bytes(2, "little") + bytes([2, 7])
    payload = _p3_payload(brotli.compress(actions_raw))
    archive = tmp_path / "archive.zip"
    _write_archive(archive, payload)

    _inventory, members = script.archive_inventory(archive)
    _container, raw_payload = script.read_payload_from_members(members)
    public = script.public_pr79_profile(raw_payload)

    extract_dir = tmp_path / "extract"
    script.extract_archive(archive, extract_dir)
    robust = script.robust_current_profile(extract_dir)
    comparison = script.compare_profiles(public, robust)

    assert public.ok is True
    assert robust.ok is True
    assert comparison["decoded_member_hashes_all_equal"] is True
    assert comparison["seg_tile_actions"]["canonical_records_exact_equal"] is True
    assert comparison["pose_float32_sha256"]["exact_equal"] is True


def test_s2_action_payload_flags_public_runtime_gap_but_robust_decodes(tmp_path: Path) -> None:
    script = _load_script()
    payload = _p3_payload(_s2_action_wire())
    archive = tmp_path / "archive.zip"
    _write_archive(archive, payload)

    _inventory, members = script.archive_inventory(archive)
    _container, raw_payload = script.read_payload_from_members(members)
    public = script.public_pr79_profile(raw_payload)

    extract_dir = tmp_path / "extract"
    script.extract_archive(archive, extract_dir)
    robust = script.robust_current_profile(extract_dir)
    comparison = script.compare_profiles(public, robust)

    assert public.ok is False
    assert public.diagnostics["failure_class"] == "error"
    assert "brotli" in (public.error or "").lower()
    assert robust.ok is True
    assert robust.diagnostics["seg_tile_action_stats"]["record_count"] == 1
    assert "public_pr79_parse_or_decode_failed" in comparison["parity_gap_classes"]


def test_archive_inventory_rejects_duplicate_or_unsafe_members(tmp_path: Path) -> None:
    script = _load_script()
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"one")
        zf.writestr("p", b"two")
    try:
        script.archive_inventory(dup)
    except script.ParityError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate member was accepted")

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("../p", b"bad")
    try:
        script.archive_inventory(unsafe)
    except script.ParityError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("unsafe member was accepted")


def test_runtime_source_manifest_hashes_files(tmp_path: Path) -> None:
    script = _load_script()
    path = tmp_path / "inflate.py"
    path.write_text("print('x')\n", encoding="utf-8")

    manifest = script.source_file_manifest({"inflate.py": path, "missing.py": tmp_path / "missing.py"})

    assert manifest[0]["label"] == "inflate.py"
    assert manifest[0]["bytes"] == len("print('x')\n")
    assert manifest[0]["sha256"] == script.sha256_path(path)
    assert manifest[1]["missing"] is True
