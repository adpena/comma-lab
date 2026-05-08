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


def test_analyze_pair_decomposes_pr102_cpu_cuda_gap() -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    pr_row = {
        "pr": 102,
        "title": "fixture",
        "url": "https://example.test/pr/102",
        "eval_comments": [
            {
                "device": "cuda",
                "created_at": "2026-05-04T16:58:03Z",
                "pose": 0.00017347,
                "seg": 0.00067565,
                "archive_bytes": 178_981,
            },
            {
                "device": "cpu",
                "created_at": "2026-05-05T17:05:18Z",
                "pose": 0.00003460,
                "seg": 0.00057599,
                "archive_bytes": 178_981,
            },
        ],
    }

    pair = mod.analyze_pair(pr_row)

    assert pair is not None
    assert pair["same_archive_bytes"] is True
    assert pair["cuda"]["score"] > pair["cpu"]["score"]
    assert abs(pair["ratios"]["pose_distortion_cuda_over_cpu"] - 5.013583815) < 1e-6
    assert abs(pair["ratios"]["seg_distortion_cuda_over_cpu"] - 1.173023837) < 1e-6
    assert 0.68 < pair["gaps_cuda_minus_cpu"]["pose_gap_share"] < 0.72


def test_build_analysis_preserves_non_promotable_mechanism_boundary() -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    scorecard = {
        "schema": "public_pr_eval_comment_scorecard.v1",
        "rows": [
            {
                "pr": 100,
                "title": "fixture",
                "url": "https://example.test/pr/100",
                "eval_comments": [
                    {
                        "device": "cuda",
                        "created_at": "2026-05-04T00:00:00Z",
                        "pose": 0.00017198,
                        "seg": 0.00067623,
                        "archive_bytes": 178_981,
                    },
                    {
                        "device": "cpu",
                        "created_at": "2026-05-05T00:00:00Z",
                        "pose": 0.00003443,
                        "seg": 0.00057654,
                        "archive_bytes": 178_981,
                    },
                ],
            }
        ],
    }

    analysis = mod.build_analysis(scorecard)

    assert analysis["paired_pr_count"] == 1
    assert analysis["input_schema"] == "public_pr_eval_comment_scorecard.v1"
    assert analysis["score_claim"] is False
    assert analysis["promotion_eligible"] is False
    assert analysis["rank_or_kill_eligible"] is False
    assert analysis["mechanism_claim_proven"] is False
    assert abs(analysis["summary"]["pose_tau_from_public_cpu_comments"] - 0.00586770824) < 1e-12
