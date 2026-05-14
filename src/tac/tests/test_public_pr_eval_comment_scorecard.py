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
