# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.mlx_dynamic_learned_sweep_local_autopilot import (
    MLXDynamicLearnedSweepLocalAutopilotError,
    run_local_mlx_sweep_autopilot,
)
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.tests.test_mlx_dynamic_learned_sweep_local_actuator import (
    INCUMBENT_SCORE,
    _candidate_payload,
    _fake_response_builder,
    _plan,
    _selection,
)


def test_local_autopilot_runs_bounded_cycles_and_replans(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    candidate_payload = _candidate_payload(selection)
    plan = _plan(selection)
    observation_jsonl = tmp_path / "observations.jsonl"

    summary = run_local_mlx_sweep_autopilot(
        initial_plan=plan,
        selection=selection,
        candidate_payloads=[candidate_payload],
        incumbent_score=INCUMBENT_SCORE,
        output_dir=tmp_path / "autopilot",
        observation_jsonl=observation_jsonl,
        max_iterations=2,
        max_new_observations=2,
        rows_per_replan=1,
        response_builder=_fake_response_builder,
    )

    assert summary["schema"] == "mlx_dynamic_learned_sweep_local_autopilot.v1"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["cycle_count"] == 1
    assert summary["executed_row_count"] == 1
    assert summary["new_observation_row_count"] == 1
    assert summary["stopping_reason"] == "ready_rows_exhausted"
    assert summary["cycles"][0]["actuation_summary"]["executed_filter_match"] is True
    assert summary["final_plan_summary"]["suppressed_observed_row_count"] == 1
    assert summary["final_plan_summary"]["local_ready_row_count"] == 1
    rows = load_observation_rows(observation_jsonl)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "mlx_scorer_response:window:10:11"
    cycle = summary["cycles"][0]
    assert Path(cycle["replan_json_out"]).is_file()
    assert Path(cycle["replan_md_out"]).is_file()
    assert cycle["actuation_summary"]["executed_queue_candidate_ids"] == [
        "mlx_scorer_response:window:10:11::mlx_local_response::micro"
    ]


def test_local_autopilot_exact_queue_candidate_filter_is_single_cycle(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    queue_candidate_id = "mlx_scorer_response:window:10:11::mlx_local_response::micro"

    summary = run_local_mlx_sweep_autopilot(
        initial_plan=_plan(selection),
        selection=selection,
        candidate_payloads=[_candidate_payload(selection)],
        incumbent_score=INCUMBENT_SCORE,
        output_dir=tmp_path / "autopilot",
        observation_jsonl=tmp_path / "observations.jsonl",
        max_iterations=1,
        max_new_observations=1,
        rows_per_replan=1,
        queue_candidate_ids=[queue_candidate_id],
        response_builder=_fake_response_builder,
    )

    assert summary["queue_candidate_id_filters"] == [queue_candidate_id]
    assert summary["cycles"][0]["actuation_summary"]["executed_filter_match"] is True
    assert summary["cycles"][0]["actuation_summary"][
        "executed_unique_queue_candidate_id"
    ] == queue_candidate_id


def test_local_autopilot_rejects_chained_exact_queue_candidate_filter(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    with pytest.raises(
        MLXDynamicLearnedSweepLocalAutopilotError,
        match="one bounded exact-row cycle",
    ):
        run_local_mlx_sweep_autopilot(
            initial_plan=_plan(selection),
            selection=selection,
            candidate_payloads=[_candidate_payload(selection)],
            incumbent_score=INCUMBENT_SCORE,
            output_dir=tmp_path / "autopilot",
            observation_jsonl=tmp_path / "observations.jsonl",
            max_iterations=2,
            max_new_observations=1,
            rows_per_replan=1,
            queue_candidate_ids=[
                "mlx_scorer_response:window:10:11::mlx_local_response::micro"
            ],
            response_builder=_fake_response_builder,
        )


def test_local_autopilot_honors_new_observation_cap(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)

    summary = run_local_mlx_sweep_autopilot(
        initial_plan=plan,
        selection=selection,
        candidate_payloads=[_candidate_payload(selection)],
        incumbent_score=INCUMBENT_SCORE,
        output_dir=tmp_path / "autopilot",
        observation_jsonl=tmp_path / "observations.jsonl",
        max_iterations=5,
        max_new_observations=1,
        rows_per_replan=4,
        response_builder=_fake_response_builder,
    )

    assert summary["stopping_reason"] == "max_new_observations_reached"
    assert summary["cycle_count"] == 1
    assert summary["executed_row_count"] == 1
    assert summary["new_observation_row_count"] == 1


def test_local_autopilot_records_local_mlx_device_without_gpu_authority(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)

    summary = run_local_mlx_sweep_autopilot(
        initial_plan=_plan(selection),
        selection=selection,
        candidate_payloads=[_candidate_payload(selection)],
        incumbent_score=INCUMBENT_SCORE,
        output_dir=tmp_path / "autopilot",
        observation_jsonl=tmp_path / "observations.jsonl",
        max_iterations=1,
        max_new_observations=1,
        rows_per_replan=1,
        device_type="gpu",
        allow_gpu_research_signal=True,
        response_builder=_fake_response_builder,
    )

    assert summary["gpu_launched"] is False
    assert summary["local_mlx_device_used"] is True
    assert summary["cycles"][0]["actuation_summary"]["gpu_launched"] is False
    assert summary["cycles"][0]["actuation_summary"]["local_mlx_device_used"] is True


def test_local_autopilot_refuses_unsupported_config(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    with pytest.raises(
        MLXDynamicLearnedSweepLocalAutopilotError,
        match="unsupported sweep_config_id",
    ):
        run_local_mlx_sweep_autopilot(
            initial_plan=_plan(selection),
            selection=selection,
            candidate_payloads=[_candidate_payload(selection)],
            incumbent_score=INCUMBENT_SCORE,
            output_dir=tmp_path / "autopilot",
            observation_jsonl=tmp_path / "observations.jsonl",
            sweep_config_id="macos_cpu_advisory",
            response_builder=_fake_response_builder,
        )


def test_local_autopilot_rejects_truthy_authority(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    plan["ready_for_exact_eval_dispatch"] = True
    with pytest.raises(
        MLXDynamicLearnedSweepLocalAutopilotError,
        match="ready_for_exact_eval_dispatch",
    ):
        run_local_mlx_sweep_autopilot(
            initial_plan=plan,
            selection=selection,
            candidate_payloads=[_candidate_payload(selection)],
            incumbent_score=INCUMBENT_SCORE,
            output_dir=tmp_path / "autopilot",
            observation_jsonl=tmp_path / "observations.jsonl",
            response_builder=_fake_response_builder,
        )


def test_local_autopilot_refuses_nonpositive_max_seconds(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    with pytest.raises(
        MLXDynamicLearnedSweepLocalAutopilotError,
        match="max_seconds must be positive",
    ):
        run_local_mlx_sweep_autopilot(
            initial_plan=_plan(selection),
            selection=selection,
            candidate_payloads=[_candidate_payload(selection)],
            incumbent_score=INCUMBENT_SCORE,
            output_dir=tmp_path / "autopilot",
            observation_jsonl=tmp_path / "observations.jsonl",
            max_seconds=0.0,
            response_builder=_fake_response_builder,
        )
