# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tac.repo_io import (
    ArtifactWriteError,
    artifact_dir_transaction,
    json_line,
    json_text,
    read_json,
    repo_relative,
    sha256_bytes,
    sha256_file,
    tree_sha256,
    write_json,
    write_json_artifact,
    write_text_artifact,
)


def test_json_text_is_stable_and_rejects_nan() -> None:
    assert json_text({"b": 1, "a": [2]}) == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'


def test_json_line_is_stable_single_line_jsonl() -> None:
    assert json_line({"b": 1, "a": [2]}) == '{"a":[2],"b":1}\n'


def test_write_read_json_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json(path, {"z": 1, "a": 2})

    assert read_json(path) == {"a": 2, "z": 1}
    assert path.read_text(encoding="utf-8") == json.dumps({"a": 2, "z": 1}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert sha256_file(path) == "4b8884e8891f3aaedfad4ff8f8b08c6159a253c2f93a8b8e1112bd22e18a4162"
    assert sha256_bytes(path.read_bytes()) == sha256_file(path)


def test_write_text_artifact_is_no_clobber_by_default(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "payload.txt"

    result = write_text_artifact(path, "first")

    assert path.read_text(encoding="utf-8") == "first"
    assert result.bytes_written == len(b"first")
    assert result.sha256 == sha256_bytes(b"first")
    assert result.allow_overwrite is False
    with pytest.raises(ArtifactWriteError, match="refusing to overwrite"):
        write_text_artifact(path, "second")
    assert path.read_text(encoding="utf-8") == "first"


def test_write_text_artifact_overwrite_requires_matching_expected_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("first", encoding="utf-8")

    with pytest.raises(ArtifactWriteError, match="expected_existing_sha256"):
        write_text_artifact(path, "second", allow_overwrite=True)

    result = write_text_artifact(
        path,
        "second",
        allow_overwrite=True,
        expected_existing_sha256=sha256_file(path),
    )

    assert path.read_text(encoding="utf-8") == "second"
    assert result.sha256 == sha256_bytes(b"second")
    path.write_text("third", encoding="utf-8")
    with pytest.raises(ArtifactWriteError, match="sha256 mismatch"):
        write_text_artifact(
            path,
            "fourth",
            allow_overwrite=True,
            expected_existing_sha256=sha256_bytes(b"second"),
        )
    assert path.read_text(encoding="utf-8") == "third"


def test_write_json_artifact_rejects_low_free_space(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    free_bytes = shutil.disk_usage(tmp_path).free

    with pytest.raises(ArtifactWriteError, match="insufficient free space"):
        write_json_artifact(path, {"a": 1}, min_free_bytes=free_bytes + 1)

    assert not path.exists()


def test_artifact_dir_transaction_preserves_existing_dir_on_failed_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dir_artifact"
    target.mkdir()
    (target / "sentinel.txt").write_text("old", encoding="utf-8")

    with (
        pytest.raises(ArtifactWriteError, match="expected_existing_tree_sha256"),
        artifact_dir_transaction(target, allow_overwrite=True) as txn,
    ):
        (txn.staging / "new.txt").write_text("new", encoding="utf-8")

    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


def test_artifact_dir_transaction_replaces_only_matching_tree(tmp_path: Path) -> None:
    target = tmp_path / "dir_artifact"
    target.mkdir()
    (target / "sentinel.txt").write_text("old", encoding="utf-8")
    expected = tree_sha256(target)

    with artifact_dir_transaction(
        target,
        allow_overwrite=True,
        expected_existing_tree_sha256=expected,
    ) as txn:
        (txn.staging / "new.txt").write_text("new", encoding="utf-8")

    assert not (target / "sentinel.txt").exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "new"


def test_repo_relative_uses_posix_path(tmp_path: Path) -> None:
    child = tmp_path / "a" / "b.txt"
    child.parent.mkdir()
    child.write_text("x", encoding="utf-8")

    assert repo_relative(child, tmp_path) == "a/b.txt"
