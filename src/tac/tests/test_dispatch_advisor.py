"""Tests for tools/dispatch_advisor.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_advisor.py"


def _load_advisor():
    spec = importlib.util.spec_from_file_location("dispatch_advisor", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_advisor"] = module
    spec.loader.exec_module(module)
    return module


def test_advise_candidate_pr106_frontier_classifies_pose_dominated() -> None:
    """PR106 frontier sits at pose=3.4e-5 < flip_threshold; pose dominates."""
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="pr106_frontier",
        d_seg=6.7e-4,
        d_pose=3.4e-5,
        archive_bytes=178258,
    )
    assert advice.operating_regime_summary["pose_dominates"] is True
    assert advice.operating_regime_summary["seg_dominates"] is False
    # Top priority must be POSE
    assert advice.axis_priorities[0]["name"] == "pose"


def test_advise_candidate_legacy_regime_classifies_seg_dominated() -> None:
    """Legacy 1.x regime (pose ~ 0.18) is seg-dominated; top priority = seg."""
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="legacy_1x",
        d_seg=0.001,
        d_pose=0.18,
        archive_bytes=300_000,
    )
    assert advice.operating_regime_summary["seg_dominates"] is True
    assert advice.operating_regime_summary["pose_dominates"] is False
    assert advice.axis_priorities[0]["name"] == "seg"


def test_advise_candidate_target_score_curves_when_feasible() -> None:
    """Target_score=0.190 from PR106 frontier produces both pose+bytes feasible curves."""
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="pr106",
        d_seg=6.7e-4,
        d_pose=3.4e-5,
        archive_bytes=178258,
        target_score=0.190,
    )
    tsc = advice.target_score_curves
    assert tsc["target_score"] == 0.190
    pose_curve = tsc["to_reach_target_holding_seg_and_bytes"]
    bytes_curve = tsc["to_reach_target_holding_seg_and_pose"]
    assert pose_curve["feasible"] is True
    assert bytes_curve["feasible"] is True
    assert pose_curve["required_d_pose"] < 3.4e-5  # must improve
    assert bytes_curve["required_archive_bytes"] < 178258  # must shrink


def test_advise_candidate_target_score_infeasible_when_too_aggressive() -> None:
    """Target=0.05 from PR106 frontier: seg+rate already > 0.05 → infeasible."""
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="pr106",
        d_seg=6.7e-4,
        d_pose=3.4e-5,
        archive_bytes=178258,
        target_score=0.05,
    )
    tsc = advice.target_score_curves
    assert tsc["to_reach_target_holding_seg_and_bytes"]["feasible"] is False


def test_score_decomposition_sums() -> None:
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="x",
        d_seg=0.001,
        d_pose=1e-4,
        archive_bytes=200_000,
    )
    decomp = advice.score_decomposition
    total = decomp["seg_term"] + decomp["pose_term"] + decomp["rate_term"]
    assert abs(total - decomp["total"]) < 1e-9
    assert abs(total - advice.score) < 1e-9


def test_advise_pareto_json_round_trip(tmp_path: Path) -> None:
    """Build a synthetic 3-axis Pareto JSON, run the advisor over it."""
    advisor = _load_advisor()
    pareto_json = tmp_path / "pareto.json"
    pareto_payload = {
        "candidates": [
            {"label": "c1", "d_seg": 6.7e-4, "d_pose": 3.4e-5, "archive_bytes": 178258},
            {"label": "c2", "d_seg": 1e-3, "d_pose": 1e-4, "archive_bytes": 200_000},
            {"label": "legacy", "d_seg": 0.001, "d_pose": 0.18, "archive_bytes": 300_000},
        ],
    }
    pareto_json.write_text(json.dumps(pareto_payload), encoding="utf-8")
    rows = advisor.advise_pareto_json(pareto_json_path=pareto_json, target_score=0.20)
    assert len(rows) == 3
    # First candidate (PR106 frontier) is pose-dominated
    assert rows[0].operating_regime_summary["pose_dominates"] is True
    # Last candidate (legacy) is seg-dominated
    assert rows[2].operating_regime_summary["seg_dominates"] is True


def test_advise_pareto_json_rejects_missing_fields(tmp_path: Path) -> None:
    """Required fields missing → ValueError."""
    advisor = _load_advisor()
    pareto_json = tmp_path / "broken.json"
    pareto_json.write_text(
        json.dumps({"candidates": [{"label": "x", "d_seg": 0.001}]}),  # missing d_pose
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required fields"):
        advisor.advise_pareto_json(pareto_json_path=pareto_json)


def test_render_advice_summary_includes_axis_priorities() -> None:
    """The text summary lists axis priorities in order."""
    advisor = _load_advisor()
    advice = advisor.advise_candidate(
        label="test",
        d_seg=6.7e-4,
        d_pose=3.4e-5,
        archive_bytes=178258,
        target_score=0.190,
    )
    text = advisor.render_advice_summary(advice)
    assert "Dispatch Advice: test" in text
    assert "Axis priorities" in text
    assert "POSE" in text  # pose is top priority at this operating point
    # Target curves must appear since we supplied target_score
    assert "Pose-only path" in text
    assert "Bytes-only path" in text


def test_axis_effort_class_thresholds() -> None:
    """Effort thresholds match the documented anchors."""
    advisor = _load_advisor()
    # seg < 5e-4 is tight
    advice = advisor.advise_candidate(label="t", d_seg=1e-4, d_pose=1e-4, archive_bytes=200000)
    seg_priority = next(p for p in advice.axis_priorities if p["name"] == "seg")
    assert seg_priority["estimated_effort_class"] == "tight"

    # pose > 1e-3 is wide_open
    advice2 = advisor.advise_candidate(label="t2", d_seg=1e-3, d_pose=0.05, archive_bytes=200000)
    pose_priority = next(p for p in advice2.axis_priorities if p["name"] == "pose")
    assert pose_priority["estimated_effort_class"] == "wide_open"

    # bytes < 150KB is tight
    advice3 = advisor.advise_candidate(label="t3", d_seg=1e-3, d_pose=1e-4, archive_bytes=100000)
    bytes_priority = next(p for p in advice3.axis_priorities if p["name"] == "bytes")
    assert bytes_priority["estimated_effort_class"] == "tight"
