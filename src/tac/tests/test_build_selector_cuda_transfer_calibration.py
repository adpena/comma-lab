# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "build_selector_cuda_transfer_calibration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_selector_cuda_transfer_calibration_under_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_build_calibration_blocks_broad_waterfill_when_cuda_controls_regress(tmp_path: Path) -> None:
    review = _write_json(
        tmp_path / "selector_review.json",
        {
            "canonical_score": 0.207,
            "baseline_score": 0.206,
            "custody": {"archive_sha256": "a" * 64},
            "exact_cuda_evidence": True,
            "failure_class": "legitimate_score_regression_or_component_collapse",
            "measured_config_status": "measured_config_retired",
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "score_axis": "contest_cuda",
            "score_recomputation": {
                "archive_bytes": 123,
                "avg_posenet_dist": 0.2,
                "avg_segnet_dist": 0.1,
            },
            "technique": "fixture_selector",
        },
    )
    paired = _write_json(
        tmp_path / "paired_axis.json",
        {
            "classification": "cpu_positive_cuda_miss_due_to_component_drift",
            "components": {
                "contest_cpu": {"score": 0.192},
                "contest_cuda": {"score": 0.226},
                "delta_cuda_minus_cpu": {"score_delta_cuda_minus_cpu": 0.034},
                "dominant_score_delta_component": "pose",
                "score_delta_byte_equivalent": 51_300.2,
            },
            "raw_output_comparison": {"aggregate_sha256_match": False},
            "target_gaps": {},
        },
    )

    payload = mod.build_calibration([review], paired)

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    decision = payload["decision"]
    assert decision["calibration_status"] == "blocked"
    assert decision["ready_for_broad_waterfill_dispatch"] is False
    assert decision["selector_regression_rows"] == 1
    assert decision["selector_positive_or_neutral_rows"] == 0
    assert "measured_selector_controls_transfer_negative_on_cuda" in decision["blockers"]
    assert "cpu_cuda_gap_component_dominated_not_rate_limited" in decision["blockers"]


def test_render_markdown_keeps_axis_and_no_dispatch_flags(tmp_path: Path) -> None:
    review = _write_json(
        tmp_path / "selector_review.json",
        {
            "canonical_score": 0.206,
            "baseline_score": 0.206,
            "custody": {"archive_sha256": "b" * 64},
            "exact_cuda_evidence": True,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "score_axis": "contest_cuda",
            "score_recomputation": {},
            "technique": "neutral_fixture",
        },
    )
    paired = _write_json(
        tmp_path / "paired_axis.json",
        {
            "classification": "matched",
            "components": {
                "delta_cuda_minus_cpu": {"score_delta_cuda_minus_cpu": 0.0},
                "dominant_score_delta_component": "rate",
                "score_delta_byte_equivalent": 0.0,
            },
            "raw_output_comparison": {"aggregate_sha256_match": True},
            "target_gaps": {},
        },
    )

    markdown = mod.render_markdown(mod.build_calibration([review], paired))

    assert "score_claim: `false`" in markdown
    assert "ready_for_exact_eval_dispatch: `false`" in markdown
    assert "`neutral_fixture`" in markdown
