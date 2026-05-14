# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.reproducibility import (
    collect_source_transparency,
    normalize_git_remote_url,
    transparency_report_markdown,
)


def test_normalize_git_remote_url_handles_github_forms() -> None:
    assert (
        normalize_git_remote_url("git@github.com:commaai/pact.git")
        == "https://github.com/commaai/pact"
    )
    assert (
        normalize_git_remote_url("ssh://git@github.com/commaai/pact.git")
        == "https://github.com/commaai/pact"
    )
    assert (
        normalize_git_remote_url("https://github.com/commaai/pact.git")
        == "https://github.com/commaai/pact"
    )


def test_collect_source_transparency_records_artifacts_and_commands(tmp_path: Path) -> None:
    source = tmp_path / "tool.py"
    artifact = tmp_path / "archive.zip"
    source.write_text("print('build')\n", encoding="utf-8")
    artifact.write_bytes(b"packet")

    payload = collect_source_transparency(
        repo_root=tmp_path,
        source_paths=[source],
        artifact_paths=[artifact],
        commands=[["python", "tool.py"], "modal run eval"],
        generated_at_utc="2026-05-14T12:00:00Z",
    )

    assert payload["schema"] == "tac_source_transparency_v1"
    assert payload["generated_at_utc"] == "2026-05-14T12:00:00Z"
    assert payload["source_paths"][0]["sha256"]
    assert payload["artifact_paths"][0]["bytes"] == 6
    assert payload["reproduction_commands"] == [["python", "tool.py"], ["modal run eval"]]
    assert payload["release_contract"]["include_in_submission_packets"] is True
    assert payload["release_contract"]["score_claim_from_metadata"] is False


def test_transparency_report_markdown_includes_public_release_fields(tmp_path: Path) -> None:
    payload = collect_source_transparency(
        repo_root=tmp_path,
        commands=[["python", "build.py"]],
        generated_at_utc="2026-05-14T12:00:00Z",
    )

    text = transparency_report_markdown(payload)

    assert "## Source Transparency" in text
    assert "working_tree_fingerprint_sha256" in text
    assert "python build.py" in text
