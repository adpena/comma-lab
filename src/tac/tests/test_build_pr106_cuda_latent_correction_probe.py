# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "build_pr106_cuda_latent_correction_probe.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_pr106_cuda_latent_correction_probe",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hardpair_payload() -> dict[str, Any]:
    return {
        "schema": "xray_hardpair_hitlist_v1",
        "label": "fixture_hardpairs",
        "authority": {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
        "hitlist": [
            {
                "pair_idx": 5,
                "priority": 0.09,
                "dominant_component": "pose",
                "axis_dominant_component": "pose",
                "pose_score_contribution": 0.04,
                "seg_score_contribution": 0.01,
                "component_score_no_rate": 0.05,
                "suggested_lane_tags": ["cuda_pose_repair", "hardpair_tail"],
            },
            {
                "pair_idx": 2,
                "priority": 0.08,
                "dominant_component": "seg",
                "axis_dominant_component": "pose",
                "pose_score_contribution": 0.02,
                "seg_score_contribution": 0.05,
                "component_score_no_rate": 0.07,
                "suggested_lane_tags": ["cuda_seg_repair"],
            },
            {
                "pair_idx": 1,
                "priority": 0.08,
                "dominant_component": "seg",
                "axis_dominant_component": "pose",
                "pose_score_contribution": 0.02,
                "seg_score_contribution": 0.04,
                "component_score_no_rate": 0.06,
                "suggested_lane_tags": ["discard_byte_only"],
            },
        ],
    }


def _pair_xray_payload() -> dict[str, Any]:
    return {
        "schema": "pair_component_error_xray_v1",
        "label": "fixture_pair_xray",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rows": [
            {
                "pair_idx": 9,
                "pose_score_contribution": 0.01,
                "seg_score_contribution": 0.02,
                "component_score_no_rate": 0.03,
            }
        ],
    }


def _axis_payload() -> dict[str, Any]:
    return {
        "schema_version": "xray_paired_cpu_cuda_axis_delta_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "classification": "cpu_positive_cuda_miss_due_to_component_drift",
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
    }


def _format0c_ledger(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "# PR106 format0C paired CPU/CUDA auth eval",
                "- Archive bytes: `186327`",
                "| Axis | Modal call | Output dir | Score | SegNet | PoseNet |",
                "| --- | --- | --- | ---: | ---: | ---: |",
                "| `[contest-CUDA]` | `fc-cuda` | `cuda_dir` | `0.2063163866158099` | `0.0006426` | `0.00003236` |",
                "| `[contest-CPU]` | `fc-cpu` | `cpu_dir` | `0.22776488386973992` | `0.00063198` | `0.00016402` |",
                "Do not infer either axis from the other.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_build_plan_false_authority_and_byte_budget_selection(tmp_path: Path) -> None:
    module = _load_module()
    hardpair = _write_json(tmp_path / "hardpair_hitlist.json", _hardpair_payload())
    pair_xray = _write_json(tmp_path / "pair_component_xray.json", _pair_xray_payload())
    axis = _write_json(tmp_path / "paired_axis.json", _axis_payload())
    ledger = _format0c_ledger(tmp_path / "pr106_format0c.md")
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")

    pair_records = module.load_pair_records(
        pair_hitlist_paths=[hardpair],
        pair_xray_paths=[pair_xray],
    )
    plan = module.build_plan(
        pair_records=pair_records,
        axis_contexts=[module.load_axis_context(axis)],
        source_archive=source_archive,
        byte_budget=6,
        max_pairs=10,
        label="fixture_probe",
        pair_hitlist_paths=[hardpair],
        pair_xray_paths=[pair_xray],
        paired_axis_artifacts=[axis],
        pr106_format0c_ledgers=[ledger],
    )

    assert plan["schema"] == "pr106_cuda_latent_correction_probe_plan_v1"
    assert plan["authority"]["score_claim"] is False
    assert plan["authority"]["promotion_eligible"] is False
    assert plan["authority"]["frontier_language_allowed"] is False
    assert plan["authority"]["paired_cpu_cuda_exact_eval_required"] is True
    assert [row["pair_idx"] for row in plan["candidate_pairs"]] == [5, 2]
    assert plan["byte_accounting"]["selected_pair_payload_bytes"] == 6
    assert plan["byte_accounting"]["byte_budget_remaining"] == 0
    assert plan["selection_policy"]["probe_modes_per_pair"] == 112
    assert plan["source_lessons"][0]["source"].startswith("PR100")
    assert plan["source_lessons"][1]["source"].startswith("PR101")
    assert plan["source_lessons"][2]["source"].startswith("PR103")
    assert plan["source_lessons"][3]["source"].startswith("PR106")
    assert plan["materialization"]["supported"] is False
    assert plan["materialization"]["placeholder_fails_closed"] is True
    assert "paired exact [contest-CUDA] and [contest-CPU]" in plan["required_next_proofs"][3]


def test_cli_writes_json_markdown_and_rebuild_command(tmp_path: Path) -> None:
    module = _load_module()
    hardpair = _write_json(tmp_path / "hardpair_hitlist.json", _hardpair_payload())
    axis = _write_json(tmp_path / "paired_axis.json", _axis_payload())
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    output_dir = tmp_path / "out"

    assert module.main(
        [
            "--pair-hitlist",
            str(hardpair),
            "--paired-axis-artifact",
            str(axis),
            "--source-archive",
            str(source_archive),
            "--byte-budget",
            "4",
            "--max-pairs",
            "5",
            "--label",
            "cli_fixture",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    plan = json.loads(
        (output_dir / "pr106_cuda_latent_correction_probe_plan.json").read_text(
            encoding="utf-8",
        )
    )
    markdown = (output_dir / "pr106_cuda_latent_correction_probe_plan.md").read_text(
        encoding="utf-8",
    )
    rebuild = (output_dir / "rebuild_command.txt").read_text(encoding="utf-8")

    assert plan["label"] == "cli_fixture"
    assert [row["pair_idx"] for row in plan["candidate_pairs"]] == [5]
    assert "score_claim: `false`" in markdown
    assert "paired CPU/CUDA exact eval required before frontier language: `true`" in markdown
    assert "build_pr106_cuda_latent_correction_probe.py" in rebuild
    assert "--materialize" in module.build_parser().format_help()


def test_materialize_placeholder_fails_closed_without_outputs(tmp_path: Path) -> None:
    module = _load_module()
    hardpair = _write_json(tmp_path / "hardpair_hitlist.json", _hardpair_payload())
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    output_dir = tmp_path / "out"

    assert module.main(
        [
            "--pair-hitlist",
            str(hardpair),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(output_dir),
            "--materialize",
        ]
    ) == 2
    assert not output_dir.exists()
