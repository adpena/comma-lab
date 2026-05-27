# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.optimization.decoder_q_pairset_acquisition import (
    build_decoder_q_pairset_acquisition_plan,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, object]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _selector_pareto(pair_count: int = 8) -> dict[str, object]:
    pairs = list(range(pair_count))
    return {
        "schema": "decoder_q_selective_selector_pareto.v1",
        **_false_authority(),
        "summary": {"recommended_selector_id": "selector_top"},
        "candidates": [
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "selector_top",
                "selector_kind": "top_rank_prefix",
                "selector_rank": 1,
                "rank_order_pair_indices": pairs,
                "selected_pair_indices": pairs,
                "selected_pair_count": len(pairs),
                "payload_bytes": 64,
                "predicted_score_mean": 0.19203,
            }
        ],
    }


def _component_xray(pair_count: int = 8) -> dict[str, object]:
    return {
        "schema": "pair_component_error_xray_v1",
        **_false_authority(),
        "rows": [
            {
                "pair_idx": pair,
                "pose_score_contribution": float(pair_count - pair) * 0.25,
                "seg_score_contribution": float(pair_count - pair) * 0.75,
                "component_score_no_rate": float(pair_count - pair),
            }
            for pair in range(pair_count)
        ],
    }


def _write_5d_canvas(path: Path) -> Path:
    cell = PairFrameScorerGeometryCell(
        pair_idx=0,
        frame_idx=0,
        scorer_axis=ScorerAxis.SEGNET_5CLASS,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
        predicted_delta_score=-0.01,
        predicted_byte_cost=0,
        receiver_feasibility=True,
    )
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256="c" * 64,
        cells={cell.coordinate: cell},
    )
    return _write_json(
        path,
        {
            "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
            "archive_sha256": canvas.archive_sha256,
            "cells": [item.as_dict() for item in canvas._cells.values()],
        },
    )


def _action_summary(repo: Path) -> Path:
    summary_dir = repo / "experiments" / "results" / "portfolio"
    portfolio = _write_json(
        summary_dir / "portfolio.json",
        {
            "operator_action_rows": [
                {
                    **_false_authority(),
                    "candidate_id": "pairset_drop_one_rank001_pair0007",
                    "operator_next_action": (
                        "materialize_pairset_archive_and_run_local_controls"
                    ),
                    "source_metadata": {
                        "selected_pair_count": 3,
                        "selected_pair_indices": [0, 1, 7],
                    },
                }
            ],
        },
    )
    return _write_json(
        summary_dir / "action_summary.json",
        {
            **_false_authority(),
            "schema": "cross_family_candidate_portfolio_action_summary.v1",
            "json_out": str(portfolio),
            "top_operator_actions": [
                {
                    **_false_authority(),
                    "candidate_id": "pairset_drop_one_rank001_pair0007",
                    "operator_action_rank": 1,
                    "operator_next_action": (
                        "materialize_pairset_archive_and_run_local_controls"
                    ),
                }
            ],
        },
    )


def test_refresh_cli_builds_pair_frame_lattice_and_wires_queue(tmp_path: Path) -> None:
    pairset_path = _write_json(
        tmp_path / "pairset_acquisition.json",
        build_decoder_q_pairset_acquisition_plan(
            _selector_pareto(),
            prefix_ks=[8],
            diversity_ks=[8],
            max_drop_two=0,
            max_swap_in=0,
            include_drop_one=False,
        ),
    )
    xray_path = _write_json(tmp_path / "pair_component_xray.json", _component_xray())
    frontier_artifact_root = tmp_path / "frontier_artifacts"
    frontier_artifact_root.mkdir()
    output_dir = tmp_path / "refresh"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_rate_attack_feedback_refresh.py",
            "--action-summary",
            str(_action_summary(tmp_path)),
            "--frontier-artifact-root",
            str(frontier_artifact_root),
            "--pair-frame-pairset-acquisition",
            str(pairset_path),
            "--pair-component-xray",
            str(xray_path),
            "--pair-frame-drop-counts",
            "3,4",
            "--pair-frame-max-requests",
            "8",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_pair_frame_generated_unit",
            "--candidate-limit",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pair_frame_geometry_queue_request_count"] == 2
    generated = payload["generated_pair_frame_geometry_lattice"]
    assert generated["queue_executable_request_count"] == 2
    assert generated["score_claim"] is False

    report = json.loads(
        (output_dir / "feedback_refresh_report.json").read_text(encoding="utf-8")
    )
    assert report["artifacts"]["generated_pair_frame_geometry_lattice"].endswith(
        "pair_frame_scorer_geometry_lattice.json"
    )
    assert report["pair_frame_geometry_discovery"][
        "queue_executable_request_count"
    ] == 2
    lattice = json.loads(
        (output_dir / "pair_frame_scorer_geometry_lattice.json").read_text(
            encoding="utf-8"
        )
    )
    assert lattice["coverage"]["geometry_coverage"] == 1.0
    queue = json.loads((output_dir / "dqs1_followup_queue.json").read_text())
    assert queue["experiments"][0]["metadata"]["source_metadata"][
        "queue_source_kind"
    ] == "pair_frame_scorer_geometry_lattice"
    assert queue["experiments"][0]["metadata"]["score_claim"] is False


def test_refresh_cli_wires_pair_frame_5d_followup_search_roots_and_commands(
    tmp_path: Path,
) -> None:
    canvas_path = _write_5d_canvas(tmp_path / "populated_5d_canvas.json")
    frontier_artifact_root = tmp_path / "frontier_artifacts"
    extra_followup_root = tmp_path / "component_cache_root"
    frontier_artifact_root.mkdir()
    extra_followup_root.mkdir()
    output_dir = tmp_path / "refresh"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_rate_attack_feedback_refresh.py",
            "--action-summary",
            str(_action_summary(tmp_path)),
            "--frontier-artifact-root",
            str(frontier_artifact_root),
            "--pair-frame-5d-canvas",
            str(canvas_path),
            "--pair-frame-5d-followup-search-root",
            str(extra_followup_root),
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_pair_frame_5d_unit",
            "--candidate-limit",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    summary = payload["pair_frame_5d_coverage_acquisition_queue_summary"]
    assert summary["followup_readiness_refresh_planned_by_queue"] is True
    assert any(root.endswith("component_cache_root") for root in summary["followup_search_roots"])
    assert summary["followup_readiness_report_path"].endswith(
        "followup_readiness_report.json"
    )

    report = json.loads(
        (output_dir / "feedback_refresh_report.json").read_text(encoding="utf-8")
    )
    assert report["operator_commands"][
        "inspect_pair_frame_5d_followup_input_binding_report_after_acquisition"
    ][-1].endswith("followup_input_binding_report.json")
    assert report["operator_commands"][
        "inspect_pair_frame_5d_followup_readiness_report_after_acquisition"
    ][-1].endswith("followup_readiness_report.json")
    run_cmd = report["operator_commands"][
        "run_pair_frame_5d_followup_execution_queue_bounded_local_after_acquisition"
    ]
    assert run_cmd[-2:] == [
        "--output",
        summary["followup_execution_worker_result_path"],
    ]
