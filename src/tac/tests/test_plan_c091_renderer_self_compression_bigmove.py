from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_c091_renderer_self_compression_bigmove.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("c091_renderer_bigmove_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_byte_gap_matches_c091_sub314_target() -> None:
    planner = _load_planner()

    assert planner.byte_gap_to_target(frontier_score=0.31516575028285976) == 1751


def test_score_if_components_unchanged_uses_archive_delta() -> None:
    planner = _load_planner()

    score = planner.score_if_components_unchanged(
        frontier_score=0.31516575028285976,
        frontier_bytes=276_481,
        candidate_bytes=274_681,
    )

    assert score < 0.314
    assert round(score, 12) == round(0.31516575028285976 - 1800 * planner.RATE_SCORE_PER_BYTE, 12)


def test_dispatch_recommendation_requires_pose_safety_and_byte_target() -> None:
    planner = _load_planner()
    rows = [
        {
            "candidate_id": "unsafe_big",
            "archive_bytes": 274_000,
            "byte_sufficient_for_sub314_if_components_unchanged": True,
            "plausible_bigmove_byte_savings": True,
            "pose_safety": {"safe_for_exact_eval_dispatch": False},
        },
        {
            "candidate_id": "safe_small",
            "archive_bytes": 276_000,
            "byte_sufficient_for_sub314_if_components_unchanged": False,
            "plausible_bigmove_byte_savings": False,
            "pose_safety": {"safe_for_exact_eval_dispatch": True},
        },
    ]

    recommendation = planner.dispatch_recommendation(rows)

    assert recommendation["recommendation"] == "do_not_dispatch_yet_safe_but_too_small"
    assert recommendation["candidate"]["candidate_id"] == "safe_small"


def test_validate_frontier_custody_fails_closed_on_sha_mismatch(tmp_path: Path) -> None:
    planner = _load_planner()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"not-c091")
    eval_json = tmp_path / "contest_auth_eval.adjudicated.json"
    eval_json.write_text(
        json.dumps(
            {
                "archive_size_bytes": len(b"not-c091"),
                "canonical_score": planner.FRONTIER_SCORE,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": "bad",
                    "device": "cuda",
                },
            }
        ),
        encoding="utf-8",
    )

    custody = planner.validate_frontier_custody(
        archive=archive,
        eval_json=eval_json,
        expected_bytes=len(b"not-c091"),
        expected_sha256="0" * 64,
    )

    assert custody["ok"] is False
    assert "archive_sha256_does_not_match_expected_c091" in custody["failures"]
    assert "eval_archive_sha256_mismatch" in custody["failures"]
