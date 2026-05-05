from __future__ import annotations

import json
from pathlib import Path

from tools.materialize_pr_archive_kaggle_mirror import (
    build_dataset_metadata,
    materialize_kaggle_mirror,
    should_skip_release_view_file,
)


def test_build_dataset_metadata_has_kaggle_required_fields() -> None:
    metadata = build_dataset_metadata("adpena/comma-video-compression-pr-archive")

    assert metadata["id"] == "adpena/comma-video-compression-pr-archive"
    assert metadata["title"] == "Comma Video Compression PR Archive"
    assert metadata["licenses"] == [{"name": "other"}]
    assert metadata["expectedUpdateFrequency"] == "never"
    assert any(r["path"] == "FETCH_SUMMARY.json" for r in metadata["resources"])
    assert "Hugging Face" in metadata["description"]


def test_should_skip_release_view_file_blocks_local_state() -> None:
    assert should_skip_release_view_file(Path(".cache/huggingface/x.metadata"))
    assert should_skip_release_view_file(Path("source/.git/config"))
    assert should_skip_release_view_file(Path("source/__pycache__/x.pyc"))
    assert should_skip_release_view_file(Path("dataset-metadata.json"))
    assert not should_skip_release_view_file(Path("public_pr101/archive.zip"))


def test_materialize_kaggle_mirror_adds_metadata_and_preserves_release_files(tmp_path: Path) -> None:
    source = tmp_path / "release"
    output = tmp_path / "kaggle"
    (source / "public_pr101").mkdir(parents=True)
    (source / "FETCH_SUMMARY.json").write_text('{"n_with_archive": 1}\n', encoding="utf-8")
    (source / "OMITTED_SHARED_ASSETS.json").write_text('{"omitted_file_count": 0}\n', encoding="utf-8")
    (source / "README.md").write_text("# card\n", encoding="utf-8")
    (source / "public_pr101" / "archive.zip").write_bytes(b"zip-bytes")
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / ".cache" / "huggingface" / "x.metadata").write_text("cache\n", encoding="utf-8")

    manifest = materialize_kaggle_mirror(
        source,
        output,
        dataset_id="adpena/comma-video-compression-pr-archive",
        force=True,
    )

    assert manifest["included_file_count"] == 4
    assert manifest["skipped_file_count"] == 1
    assert (output / "public_pr101" / "archive.zip").read_bytes() == b"zip-bytes"
    assert not (output / ".cache").exists()
    metadata = json.loads((output / "dataset-metadata.json").read_text(encoding="utf-8"))
    mirror_manifest = json.loads((output / "KAGGLE_MIRROR_MANIFEST.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "adpena/comma-video-compression-pr-archive"
    assert mirror_manifest["schema"] == "comma_pr_archive_kaggle_mirror_v1"


def test_materialize_kaggle_mirror_requires_release_manifests(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()

    try:
        materialize_kaggle_mirror(source, tmp_path / "out", dataset_id="adpena/demo", force=True)
    except FileNotFoundError as exc:
        assert "FETCH_SUMMARY.json" in str(exc)
    else:
        raise AssertionError("expected missing release summary to fail closed")
