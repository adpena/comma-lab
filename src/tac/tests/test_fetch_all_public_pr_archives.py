# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_fetcher():
    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "fetch_all_public_pr_archives_module",
        repo_root / "tools" / "fetch_all_public_pr_archives.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("fetch_all_public_pr_archives.py not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pr102_comment_attachment_archive_url_is_discoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = _load_fetcher()
    archive_url = "https://github.com/user-attachments/files/27369164/archive.zip"

    def fake_gh_api(path: str, raise_on_fail: bool = False):
        assert path == "repos/commaai/comma_video_compression_challenge/issues/102/comments?per_page=100"
        return [{"body": f"[archive.zip]({archive_url})\r\n"}]

    monkeypatch.setattr(fetcher, "_gh_api", fake_gh_api)

    assert fetcher._walk_pr_comments_for_archive(102) == [archive_url]


def test_pr102_body_archive_path_hint_wins_even_for_small_tree_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _load_fetcher()
    body = "Included in this PR under `submissions/hnerv_lc_v2_scale095_rplus1/archive.zip`."

    def fake_gh_api(path: str, raise_on_fail: bool = False):
        assert path == "repos/EthanYangTW/comma_video_compression_challenge/git/trees/deadbeef?recursive=1"
        return {
            "tree": [
                {
                    "path": "submissions/hnerv_lc_v2_scale095_rplus1/archive.zip",
                    "size": 132,
                }
            ]
        }

    monkeypatch.setattr(fetcher, "_gh_api", fake_gh_api)

    assert fetcher._walk_pr_commits_for_lfs(
        "EthanYangTW/comma_video_compression_challenge",
        "deadbeef",
        body,
    ) == [
        "https://raw.githubusercontent.com/EthanYangTW/comma_video_compression_challenge/"
        "deadbeef/submissions/hnerv_lc_v2_scale095_rplus1/archive.zip"
    ]


def test_release_archive_candidates_are_ranked_by_pr_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _load_fetcher()
    body = "submission name: hnerv_lc_v2_scale095_rplus1"

    def fake_gh_api(path: str, raise_on_fail: bool = False):
        assert path == "repos/EthanYangTW/comma_video_compression_challenge/releases?per_page=20"
        return [
            {
                "assets": [
                    {
                        "name": "archive.zip",
                        "browser_download_url": (
                            "https://github.com/EthanYangTW/comma_video_compression_challenge/"
                            "releases/download/qpose14-r55-segactions-minp/archive.zip"
                        ),
                    }
                ]
            },
            {
                "assets": [
                    {
                        "name": "archive.zip",
                        "browser_download_url": (
                            "https://github.com/EthanYangTW/comma_video_compression_challenge/"
                            "releases/download/v2-hnerv-lc-scale095/archive.zip"
                        ),
                    }
                ]
            },
        ]

    monkeypatch.setattr(fetcher, "_gh_api", fake_gh_api)

    urls = fetcher._walk_releases_for_archive(
        "EthanYangTW/comma_video_compression_challenge",
        102,
        body,
        "hnerv_lc_v2_scale095_rplus1",
    )

    assert urls[0].endswith("/v2-hnerv-lc-scale095/archive.zip")
    assert urls[1].endswith("/qpose14-r55-segactions-minp/archive.zip")


def test_repo_relative_display_accepts_relative_repo_paths() -> None:
    fetcher = _load_fetcher()

    assert fetcher._repo_relative_display(
        Path("experiments/results/public_pr102_intake_20260507_auto")
    ) == "experiments/results/public_pr102_intake_20260507_auto"
