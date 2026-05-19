# SPDX-License-Identifier: MIT
"""Tests for shared substrate inflate runtime helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.substrates._shared.inflate_runtime import raw_output_path
from tac.substrates._shared.inflate_runtime_extensions import (
    inflate_loop_per_video,
    iter_file_list_entries,
    load_per_substrate_state_dict,
    require_sha256,
    sha256_file,
)


def test_raw_output_path_preserves_safe_relative_subdirs(tmp_path: Path) -> None:
    assert raw_output_path(tmp_path, "0.mkv") == tmp_path / "0.raw"
    assert raw_output_path(tmp_path, "nested/0.mkv") == tmp_path / "nested" / "0.raw"


@pytest.mark.parametrize(
    "name",
    ["", "../0.mkv", "nested/../../0.mkv", "/tmp/0.mkv", "nested//0.mkv"],
)
def test_raw_output_path_rejects_escape_or_empty_entries(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(ValueError, match=r"unsafe|escapes"):
        raw_output_path(tmp_path, name)


def test_iter_file_list_entries_strips_blanks_and_preserves_order(
    tmp_path: Path,
) -> None:
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("\n0.mkv\n\n nested/1.mkv \n", encoding="utf-8")

    assert iter_file_list_entries(file_list) == ["0.mkv", "nested/1.mkv"]


def test_inflate_loop_per_video_uses_safe_raw_paths(tmp_path: Path) -> None:
    file_list = tmp_path / "file_list.txt"
    archive_dir = tmp_path / "archive"
    output_dir = tmp_path / "out"
    archive_dir.mkdir()
    file_list.write_text("0.mkv\nnested/1.mkv\n", encoding="utf-8")
    calls: list[tuple[Path, Path, str]] = []

    def render_fn(archive_root: Path, raw_path: Path, video_name: str) -> int:
        calls.append((archive_root, raw_path, video_name))
        raw_path.write_bytes(f"rendered:{video_name}".encode())
        return raw_path.stat().st_size

    records = inflate_loop_per_video(
        file_list=file_list,
        archive_dir=archive_dir,
        output_dir=output_dir,
        render_fn=render_fn,
    )

    assert [record.video_name for record in records] == ["0.mkv", "nested/1.mkv"]
    assert [record.raw_output_path for record in records] == [
        output_dir / "0.raw",
        output_dir / "nested" / "1.raw",
    ]
    assert [record.render_result for record in records] == [
        len("rendered:0.mkv"),
        len("rendered:nested/1.mkv"),
    ]
    assert calls == [
        (archive_dir, output_dir / "0.raw", "0.mkv"),
        (archive_dir, output_dir / "nested" / "1.raw", "nested/1.mkv"),
    ]
    assert (output_dir / "nested" / "1.raw").read_bytes() == b"rendered:nested/1.mkv"


def test_inflate_loop_per_video_reuses_raw_output_path_rejections(
    tmp_path: Path,
) -> None:
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("../escape.mkv\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsafe|escapes"):
        inflate_loop_per_video(
            file_list=file_list,
            archive_dir=tmp_path / "archive",
            output_dir=tmp_path / "out",
            render_fn=lambda *_args: None,
        )


def test_load_per_substrate_state_dict_checks_hash_and_archive_boundary(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    state_path = archive_dir / "state.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, state_path)
    expected_sha = sha256_file(state_path)

    loaded = load_per_substrate_state_dict(
        archive_dir,
        "state.pt",
        expected_sha256=expected_sha,
    )

    assert torch.equal(loaded["weight"], torch.tensor([1.0, 2.0]))
    assert require_sha256(state_path, expected_sha) == expected_sha
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_per_substrate_state_dict(
            archive_dir,
            "state.pt",
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="inside archive_dir"):
        load_per_substrate_state_dict(archive_dir, "../state.pt")
