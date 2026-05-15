# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "xray_hardpair_hitlist.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("xray_hardpair_hitlist", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _pair_xray_payload() -> dict[str, Any]:
    return {
        "schema": "pair_component_error_xray_v1",
        "label": "fixture_pair_xray",
        "device": "cpu",
        "evidence_grade": "diagnostic_pair_component_xray_cpu",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rows": [
            {
                "pair_idx": 0,
                "pose_score_contribution": 0.03,
                "seg_score_contribution": 0.02,
                "component_score_no_rate": 0.05,
                "frame0_l1": 1.0,
                "frame1_l1": 2.0,
            },
            {
                "pair_idx": 1,
                "pose_score_contribution": 0.01,
                "seg_score_contribution": 0.08,
                "component_score_no_rate": 0.09,
                "frame0_l1": 3.0,
                "frame1_l1": 4.0,
            },
            {
                "pair_idx": 2,
                "pose_score_contribution": 0.06,
                "seg_score_contribution": 0.01,
                "component_score_no_rate": 0.07,
                "frame0_l1": 5.0,
                "frame1_l1": 6.0,
            },
        ],
    }


def _paired_axis_payload() -> dict[str, Any]:
    return {
        "schema_version": "xray_paired_cpu_cuda_axis_delta_v1",
        "classification": "cpu_positive_cuda_miss_due_to_component_drift",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "components": {
            "dominant_score_delta_component": "pose",
            "score_delta_byte_equivalent": 51300.2,
            "delta_cuda_minus_cpu": {
                "score_delta_cuda_minus_cpu": 0.034,
                "seg_score_contribution_delta": 0.010,
                "pose_score_contribution_delta": 0.024,
                "rate_score_contribution_delta": 0.0,
            },
        },
        "target_gaps": {
            "contest_cpu": {"byte_gap_if_components_unchanged": 78},
            "contest_cuda": {"byte_gap_if_components_unchanged": 51378},
        },
        "raw_output_comparison": {
            "aggregate_sha256_match": False,
        },
    }


def test_build_hitlist_keeps_false_authority_and_prioritizes_axis_dominant_pose(tmp_path: Path) -> None:
    module = _load_module()
    pair_path = _write_json(tmp_path / "pair_component_xray.json", _pair_xray_payload())
    axis_path = _write_json(tmp_path / "paired_axis_delta.json", _paired_axis_payload())

    report = module.build_hitlist(
        pair_observations=module.load_pair_xray(pair_path),
        axis_contexts=[module.load_axis_context(axis_path)],
        label="fixture",
        top_k=2,
    )

    assert report["schema"] == "xray_hardpair_hitlist_v1"
    assert report["authority"]["research_only"] is True
    assert report["authority"]["score_claim"] is False
    assert report["authority"]["promotion_eligible"] is False
    assert report["authority"]["rank_or_kill_eligible"] is False
    assert report["authority"]["ready_for_exact_eval_dispatch"] is False
    assert report["primary_axis_context"]["dominant_component"] == "pose"
    assert report["hitlist"][0]["pair_idx"] == 2
    assert report["hitlist"][0]["priority"] == pytest.approx(0.13)
    assert report["hitlist"][0]["dominant_component"] == "pose"
    assert report["hitlist"][0]["axis_byte_equivalent_gap"] == 51378
    assert "cuda_pose_repair" in report["hitlist"][0]["suggested_lane_tags"]
    assert "discard_byte_only" in report["hitlist"][0]["suggested_lane_tags"]
    assert "cpu_cuda_raw_output_mismatch" in report["hitlist"][0]["suggested_lane_tags"]


def test_markdown_axis_context_can_shift_priority_to_seg_repair(tmp_path: Path) -> None:
    module = _load_module()
    pair_path = _write_json(tmp_path / "pair_component_xray.json", _pair_xray_payload())
    md_path = tmp_path / "paired_axis_delta.md"
    md_path.write_text(
        "\n".join(
            [
                "# Paired CPU/CUDA axis delta xray",
                "- verdict: `mixed_axis_gap`",
                "- dominant score-delta component: `seg`",
                "- score-delta byte equivalent: `1200.0` bytes",
                "| axis | target | score_gap | byte_gap_if_components_unchanged |",
                "|---|---:|---:|---:|",
                "| contest-CPU | 0.192000 | 0.000010 | 16 |",
                "| contest-CUDA | 0.192000 | 0.000800 | 1202 |",
                "- aggregate match: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.build_hitlist(
        pair_observations=module.load_pair_xray(pair_path),
        axis_contexts=[module.load_axis_context(md_path)],
        label="seg_fixture",
        top_k=0,
    )

    assert report["primary_axis_context"]["dominant_component"] == "seg"
    assert report["hitlist"][0]["pair_idx"] == 1
    assert report["hitlist"][0]["priority"] == pytest.approx(0.17)
    assert report["hitlist"][0]["dominant_component"] == "seg"
    assert "cuda_seg_repair" in report["hitlist"][0]["suggested_lane_tags"]
    assert "cpu_leaderboard_seg_repair" in report["hitlist"][0]["suggested_lane_tags"]


def test_markdown_axis_context_reads_target_gap_not_axis_score_table(tmp_path: Path) -> None:
    module = _load_module()
    md_path = tmp_path / "paired_axis_delta.md"
    md_path.write_text(
        "\n".join(
            [
                "# Paired CPU/CUDA axis delta xray",
                "- verdict: `cpu_positive_cuda_miss_due_to_component_drift`",
                "- dominant score-delta component: `pose`",
                "- score-delta byte equivalent: `51300.2` bytes",
                "## Axis scores",
                "| axis | score | seg contrib | pose contrib | rate contrib |",
                "|---|---:|---:|---:|---:|",
                "| contest-CUDA | 0.226210 | 0.066299 | 0.041043878959 | 0.118867 |",
                "## Target gaps",
                "| axis | target | score_gap | byte_gap_if_components_unchanged |",
                "|---|---:|---:|---:|",
                "| contest-CUDA | 0.192000 | 0.034210 | 51378 |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context = module.load_axis_context(md_path)

    assert context.contest_cuda_byte_gap == 51378


def test_cli_writes_json_markdown_and_rebuild_command(tmp_path: Path) -> None:
    module = _load_module()
    pair_path = _write_json(tmp_path / "pair_component_xray.json", _pair_xray_payload())
    axis_path = _write_json(tmp_path / "paired_axis_delta.json", _paired_axis_payload())
    out_dir = tmp_path / "out"

    assert module.main(
        [
            "--pair-xray-json",
            str(pair_path),
            "--paired-axis-artifact",
            str(axis_path),
            "--label",
            "cli_fixture",
            "--top-k",
            "1",
            "--output-dir",
            str(out_dir),
        ]
    ) == 0

    report = json.loads((out_dir / "hardpair_hitlist.json").read_text(encoding="utf-8"))
    assert report["label"] == "cli_fixture"
    assert len(report["hitlist"]) == 1
    assert (out_dir / "hardpair_hitlist.md").exists()
    assert "score_claim: `false`" in (out_dir / "hardpair_hitlist.md").read_text(encoding="utf-8")
    assert "xray_hardpair_hitlist.py" in (out_dir / "rebuild_command.txt").read_text(encoding="utf-8")
    assert "--paired-axis-artifact" in module.build_parser().format_help()
