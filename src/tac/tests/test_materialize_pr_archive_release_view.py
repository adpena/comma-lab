# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools.materialize_pr_archive_release_view import (
    materialize,
    omission_reason,
    sanitize_public_text,
)


def test_omission_reason_skips_fixed_video_and_model_assets() -> None:
    assert (
        omission_reason(
            Path("public_pr95/source/videos/0.mkv"),
            100,
            25_000_000,
        )
        == "fixed_contest_video_reconstructable_from_upstream"
    )
    assert (
        omission_reason(
            Path("public_pr95/source/models/posenet.safetensors"),
            100,
            25_000_000,
        )
        == "fixed_contest_posenet_weight_reconstructable_from_upstream"
    )


def test_sanitize_public_text_replaces_home_paths() -> None:
    private_url = "https://github.com/adpena/" + "comma-lab"
    text = (
        "uncompressed_dir: /home/runner/work/challenge/videos\n"
        "local: /Users/example/Projects/pact/report.txt\n"
        f"repo: {private_url}\n"
    )

    sanitized, count = sanitize_public_text(text)

    assert count == 3
    assert "/home/runner" not in sanitized
    assert "/Users/example" not in sanitized
    assert private_url not in sanitized
    assert sanitized.count("${LOCAL_PATH}") == 2
    assert sanitized.count("${PUBLIC_COMMA_LAB_REPO_URL}") == 1


def test_materialize_sanitizes_public_text_without_touching_archives(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "release"
    pr_dir = source / "public_pr100_intake_20260505_auto"
    (pr_dir / "source").mkdir(parents=True)
    (pr_dir / "archive.zip").write_bytes(b"archive")
    (pr_dir / "pr_body.md").write_text(
        "video_names_file: /home/runner/work/challenge/public_test_video_names.txt\n",
        encoding="utf-8",
    )
    (pr_dir / "source" / "README.md").write_text(
        "uncompressed_dir: /home/batman/comma_video_compression_challenge/videos\n",
        encoding="utf-8",
    )

    manifest = materialize(source, output, force=True, source_size_limit=25_000_000)

    assert manifest["included_file_count"] == 3
    assert manifest["sanitized_file_count"] == 2
    assert manifest["sanitized_replacement_count"] == 2
    assert (output / "public_pr100_intake_20260505_auto" / "archive.zip").read_bytes() == b"archive"
    assert "/home/" not in (output / "public_pr100_intake_20260505_auto" / "pr_body.md").read_text(
        encoding="utf-8"
    )
    omitted = json.loads((output / "OMITTED_SHARED_ASSETS.json").read_text(encoding="utf-8"))
    assert omitted["sanitized_file_count"] == 2
    assert omitted["public_link_violation_count"] == 0
