from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tac.research_evidence import EvidenceSealError, restore_bundle, seal_research_evidence, verify_bundle


def _write_tree(root: Path) -> Path:
    source = root / "research" / "run"
    (source / "nested").mkdir(parents=True)
    (source / "run.json").write_text('{"run":1}\n', encoding="utf-8")
    (source / "nested" / "receipt.txt").write_bytes(b"receipt-bytes\x00\xff")
    return source


def test_seal_has_deterministic_bundle_bytes_and_content_address(tmp_path: Path) -> None:
    source = _write_tree(tmp_path)
    first = seal_research_evidence(source, output_dir=tmp_path / "sealed-a", repo_root=tmp_path)
    second = seal_research_evidence(source, output_dir=tmp_path / "sealed-b", repo_root=tmp_path)

    assert first.manifest == second.manifest
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.bundle_path.read_bytes() == second.bundle_path.read_bytes()
    assert verify_bundle(first.bundle_path, repo_root=tmp_path) == first.manifest


def test_mutating_source_changes_manifest_and_bundle_hash(tmp_path: Path) -> None:
    source = _write_tree(tmp_path)
    first = seal_research_evidence(source, output_dir=tmp_path / "sealed-a", repo_root=tmp_path)
    (source / "nested" / "receipt.txt").write_bytes(b"mutated")
    second = seal_research_evidence(source, output_dir=tmp_path / "sealed-b", repo_root=tmp_path)

    assert first.manifest.tree_sha256 != second.manifest.tree_sha256
    assert first.bundle_sha256 != second.bundle_sha256


def test_refuses_unsafe_output_and_traversal_bundle_member(tmp_path: Path) -> None:
    source = _write_tree(tmp_path)
    with pytest.raises(EvidenceSealError, match="inside evidence source"):
        seal_research_evidence(source, output_dir=source / "bundle-output", repo_root=tmp_path)

    bundle = tmp_path / "unsafe.evidence.bundle"
    import gzip

    with (
        bundle.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(EvidenceSealError, match="unsafe evidence path"):
        verify_bundle(bundle, repo_root=tmp_path)


def test_restore_recreates_full_tree_and_refuses_overwrite(tmp_path: Path) -> None:
    source = _write_tree(tmp_path)
    sealed = seal_research_evidence(source, output_dir=tmp_path / "sealed", repo_root=tmp_path)
    destination = tmp_path / "fresh-checkout" / "restored-run"

    restored_manifest = restore_bundle(sealed.bundle_path, destination, repo_root=tmp_path)

    assert restored_manifest == sealed.manifest
    assert sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file()) == sorted(
        path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
    )
    for original in source.rglob("*"):
        if original.is_file():
            assert original.read_bytes() == (destination / original.relative_to(source)).read_bytes()
    with pytest.raises(EvidenceSealError, match="refusing to overwrite"):
        restore_bundle(sealed.bundle_path, destination, repo_root=tmp_path)


def test_external_seal_path_is_rejected_without_creating_directories(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = _write_tree(repo_root)
    external_parent = tmp_path / "external-seal"
    destination = external_parent / "nested" / "sealed"

    with pytest.raises(EvidenceSealError, match="inside repository root"):
        seal_research_evidence(source, output_dir=destination, repo_root=repo_root)

    assert not external_parent.exists()


def test_external_restore_path_is_rejected_without_creating_directories(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = _write_tree(repo_root)
    sealed = seal_research_evidence(source, output_dir=repo_root / "sealed", repo_root=repo_root)
    external_parent = tmp_path / "external-restore"
    destination = external_parent / "nested" / "restored"

    with pytest.raises(EvidenceSealError, match="inside repository root"):
        restore_bundle(sealed.bundle_path, destination, repo_root=repo_root)

    assert not external_parent.exists()


def test_symlink_components_are_rejected_before_seal_or_restore_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = _write_tree(repo_root)
    sealed = seal_research_evidence(source, output_dir=repo_root / "sealed", repo_root=repo_root)
    external = tmp_path / "external"
    external.mkdir()
    escape = repo_root / "escape"
    escape.symlink_to(external, target_is_directory=True)

    seal_destination = escape / "seal-parent" / "sealed"
    with pytest.raises(EvidenceSealError, match="symbolic-link component"):
        seal_research_evidence(source, output_dir=seal_destination, repo_root=repo_root)
    assert not (external / "seal-parent").exists()

    restore_destination = escape / "restore-parent" / "restored"
    with pytest.raises(EvidenceSealError, match="symbolic-link component"):
        restore_bundle(sealed.bundle_path, restore_destination, repo_root=repo_root)
    assert not (external / "restore-parent").exists()
