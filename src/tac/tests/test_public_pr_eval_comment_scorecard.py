# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pr108_policy_guidance_comment() -> dict:
    return {
        "author": {"login": "YassineYousfi"},
        "createdAt": "2026-05-11T19:19:57Z",
        "url": (
            "https://github.com/commaai/comma_video_compression_challenge/"
            "pull/108#issuecomment-4424101686"
        ),
        "body": (
            "closing this pr per the new submission guidelines, the tricks used "
            "are already established in several past submissions\r\n\r\n"
            '"""\r\n'
            " is this submission competitive or innovative? explain why\r\n"
            "competitive: better than top # 1 submission\r\n"
            "innovative: it has a novel idea that is not on the leaderboard yet, "
            "might not be competitive, but has potential\r\n"
            '"""'
        ),
    }


def test_parse_eval_comment_recomputes_public_pr102_cpu_score() -> None:
    mod = _load_tool("public_pr_eval_comment_scorecard")
    body = """## Eval Results: `hnerv_lc_v2_scale095_rplus1`

```
=== Evaluation config ===
  device: cpu

=== Results ===
  Average PoseNet Distortion: 0.00003460
  Average SegNet Distortion: 0.00057599
  Submission file size: 178,981 bytes

  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.20
```
"""

    row = mod.parse_eval_comment(body)

    assert row["device"] == "cpu"
    assert row["archive_bytes"] == 178981
    assert abs(row["recomputed_score_from_rounded_comment_components"] - 0.19537617652649764) < 1e-15
    assert row["printed_score"] == 0.20


def test_parse_pr_json_keeps_cuda_and_cpu_rows_separate() -> None:
    mod = _load_tool("public_pr_eval_comment_scorecard")
    pr_json = {
        "number": 102,
        "title": "fixture",
        "url": "https://example.test/pr/102",
        "author": {"login": "tester"},
        "headRefOid": "abc",
        "comments": [
            {
                "author": {"login": "github-actions"},
                "createdAt": "2026-05-04T00:00:00Z",
                "body": (
                    "## Eval Results: `x`\n```\n  device: cuda\n"
                    "Average PoseNet Distortion: 0.00017347\n"
                    "Average SegNet Distortion: 0.00067565\n"
                    "Submission file size: 178,981 bytes\n"
                    "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.23\n```"
                ),
            },
            {
                "author": {"login": "github-actions"},
                "createdAt": "2026-05-05T00:00:00Z",
                "body": (
                    "## Eval Results: `x`\n```\n  device: cpu\n"
                    "Average PoseNet Distortion: 0.00003460\n"
                    "Average SegNet Distortion: 0.00057599\n"
                    "Submission file size: 178,981 bytes\n"
                    "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.20\n```"
                ),
            },
        ],
    }

    scorecard = mod.parse_pr_json(pr_json)

    assert [row["device"] for row in scorecard["eval_comments"]] == ["cuda", "cpu"]
    assert scorecard["eval_comments"][0]["recomputed_score_from_rounded_comment_components"] > 0.22
    assert scorecard["eval_comments"][1]["recomputed_score_from_rounded_comment_components"] < 0.20
    assert scorecard["post_deadline_policy_guidance_present"] is False
    assert scorecard["post_deadline_policy_guidance_comments"] == []


def test_parse_pr_json_preserves_pr108_policy_guidance_separate_from_evals() -> None:
    mod = _load_tool("public_pr_eval_comment_scorecard")
    pr_json = {
        "number": 108,
        "title": "andimin01",
        "url": "https://github.com/commaai/comma_video_compression_challenge/pull/108",
        "author": {"login": "andrei-minca"},
        "headRefOid": "59c1bbd544bb2aa166656d24d7de117ad3e3e62e",
        "comments": [
            {
                "author": {"login": "github-actions"},
                "createdAt": "2026-05-05T16:33:07Z",
                "url": (
                    "https://github.com/commaai/comma_video_compression_challenge/"
                    "pull/108#issuecomment-4381165882"
                ),
                "body": (
                    "Thanks for the submission @andrei-minca!\n\n"
                    "A maintainer will review your PR shortly."
                ),
            },
            _pr108_policy_guidance_comment(),
        ],
    }

    scorecard = mod.parse_pr_json(pr_json)

    assert scorecard["eval_comments"] == []
    assert scorecard["post_deadline_policy_guidance_present"] is True
    assert len(scorecard["post_deadline_policy_guidance_comments"]) == 1
    guidance = scorecard["post_deadline_policy_guidance_comments"][0]
    assert guidance["author"] == "YassineYousfi"
    assert guidance["createdAt"] == "2026-05-11T19:19:57Z"
    assert guidance["url"].endswith("#issuecomment-4424101686")
    assert "competitive or innovative" in guidance["excerpt"]
    assert "competitive: better than top # 1 submission" in guidance["body"]


def test_build_scorecard_sets_aggregate_policy_guidance_flag(monkeypatch) -> None:
    mod = _load_tool("public_pr_eval_comment_scorecard")

    def fake_fetch_pr_json(repo: str, pr: int) -> dict:
        return {
            "number": pr,
            "title": "fixture",
            "url": f"https://example.test/pr/{pr}",
            "author": {"login": "tester"},
            "headRefOid": "abc",
            "comments": [_pr108_policy_guidance_comment()] if pr == 108 else [],
        }

    monkeypatch.setattr(mod, "fetch_pr_json", fake_fetch_pr_json)

    scorecard = mod.build_scorecard("example/repo", [107, 108])

    assert scorecard["post_deadline_policy_guidance_present"] is True
    rows_by_pr = {row["pr"]: row for row in scorecard["rows"]}
    assert rows_by_pr[107]["post_deadline_policy_guidance_present"] is False
    assert rows_by_pr[108]["post_deadline_policy_guidance_present"] is True
