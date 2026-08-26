from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "ddm_sr3_ap_certify_compress_reclaim.py"
SPEC = importlib.util.spec_from_file_location("ddm_sr3_certifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sr3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sr3
SPEC.loader.exec_module(sr3)


def test_manifest_archive_extract_and_hash_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "zeros.raw").write_bytes(b"\0" * 200_000)
    (source / "payload.bin").write_bytes(bytes(range(256)) * 100)
    (source / "link").symlink_to("payload.bin")

    observed = sr3.scan_tree(source)
    hashed = sr3.hash_entries(source, observed, workers=2)
    manifest = tmp_path / "manifest.jsonl"
    tree_hash = sr3.write_manifest(manifest, hashed)
    reread, reread_hash = sr3.read_manifest(manifest)
    assert reread == hashed
    assert reread_hash == tree_hash

    archive = tmp_path / "tree.tar.zst"
    info = sr3.build_archive(source, hashed, archive)
    assert info["archive_sha256"] == sr3.sha256_file(archive)
    repeat = tmp_path / "tree.repeat.tar.zst"
    repeat_info = sr3.build_archive(source, hashed, repeat)
    assert repeat_info["archive_sha256"] == info["archive_sha256"]

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    sr3.extract_archive(archive, extracted)
    extracted_meta = sr3.scan_tree(extracted)
    sr3.assert_entry_metadata_equal(hashed, extracted_meta, require_mtime=False)
    extracted_hashed = sr3.hash_entries(extracted, extracted_meta, workers=2)
    sr3.assert_hashes_equal(hashed, extracted_hashed, require_metadata=False)


def test_exact_protected_and_custody_roots_refuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "pact"
    fake_root.mkdir()
    live = fake_root / "live"
    live.mkdir()
    cold = fake_root / "cold_store_existing"
    cold.mkdir()
    monkeypatch.setattr(sr3, "AP_ROOT", fake_root)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", {live})

    with pytest.raises(sr3.CertifyError, match="protected"):
        sr3.validate_tree(str(live))
    nested = fake_root / "nested"
    nested.mkdir()
    with pytest.raises(sr3.CertifyError, match="protected"):
        sr3.validate_tree(str(nested / ".." / "live"))
    with pytest.raises(sr3.CertifyError, match="custody namespace"):
        sr3.validate_tree(str(cold))


def test_remove_only_manifest_named_top_level_paths(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    original = tree / "original"
    original.mkdir()
    (original / "payload").write_bytes(b"kept in archive")
    custody = tree / sr3.ARCHIVE_NAME
    custody.write_bytes(b"archive")

    entries = sr3.hash_entries(tree, [
        sr3.Entry("original", "dir", 0o755, original.lstat().st_mtime_ns),
        sr3.Entry(
            "original/payload",
            "file",
            0o644,
            (original / "payload").lstat().st_mtime_ns,
            bytes=(original / "payload").stat().st_size,
        ),
    ])
    removed = sr3.remove_original_top_level(tree, entries)
    assert removed == [str(original)]
    assert not original.exists()
    assert custody.read_bytes() == b"archive"

    # A crash after the exact original target vanished is resumable and leaves
    # custody untouched; the ordinary pre-certificate path still refuses it.
    with pytest.raises(sr3.CertifyError, match="vanished"):
        sr3.remove_original_top_level(tree, entries)
    assert sr3.remove_original_top_level(tree, entries, allow_already_absent=True) == []
    assert custody.read_bytes() == b"archive"


def test_partial_reclaim_resume_accepts_absence_but_refuses_same_size_tamper(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    first = tree / "first"
    first.mkdir()
    (first / "a.bin").write_bytes(b"a" * 4096)
    second = tree / "second"
    second.mkdir()
    payload = second / "b.bin"
    payload.write_bytes(b"b" * 4096)

    expected = sr3.hash_entries(tree, sr3.scan_tree(tree), workers=2)
    sr3.remove_original_top_level(
        tree,
        [row for row in expected if row.path == "first" or row.path.startswith("first/")],
    )
    remaining = sr3.scan_tree(tree)
    sr3.assert_source_subset_equal(tree, expected, remaining)

    expected_payload = next(row for row in expected if row.path == "second/b.bin")
    payload.write_bytes(b"c" * 4096)
    # Defeat the cheap size+mtime gate deliberately; the full SHA gate must
    # still catch the mutation before resumed removal.
    payload.touch()
    payload_stat = payload.stat()
    payload.chmod(expected_payload.mode)
    import os

    os.utime(
        payload,
        ns=(payload_stat.st_atime_ns, expected_payload.mtime_ns),
    )
    with pytest.raises(sr3.CertifyError, match="content differs"):
        sr3.assert_source_subset_equal(tree, expected, sr3.scan_tree(tree))


def test_detached_progress_pipes_are_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPipe:
        def write(self, _value: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

    monkeypatch.setattr(sr3.sys, "stderr", BrokenPipe())
    sr3.progress("detached")
    monkeypatch.setattr(sr3.sys, "stdout", BrokenPipe())
    sr3.emit_json({"still": "detached"})
