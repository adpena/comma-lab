# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_snerv_scorer_guard_feedback_smoke import (
    main as run_snerv_scorer_guard_feedback_smoke_main,
)
from tools.run_snerv_scorer_tether_smoke import (
    main as run_snerv_scorer_tether_smoke_main,
)
from tools.run_snerv_scorer_tether_smoke import run_snerv_scorer_tether_smoke

mlx_available = True
try:
    import mlx.core as _mx  # noqa: F401
except Exception:
    mlx_available = False

requires_mlx = pytest.mark.skipif(
    not mlx_available,
    reason="MLX unavailable; SNeRV scorer tether smoke requires MLX",
)


@requires_mlx
def test_snerv_scorer_tether_smoke_passes_current_pr95_adapter_path() -> None:
    report = run_snerv_scorer_tether_smoke(steps=2)

    assert report["schema"] == "snerv_scorer_tether_smoke.v1"
    assert report["passed"] is True
    assert report["blockers"] == []
    final = report["metric_summary"]["final"]
    assert (
        final["dual_ascent_missing_metric__snerv_segnet_last_frame_distill"]
        == 0.0
    )
    assert (
        final["dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill"]
        == 0.0
    )
    assert final["dual_ascent_lambda__snerv_segnet_last_frame_distill"] > 0.0
    assert final["dual_ascent_lambda__snerv_posenet_yuv6_pair_distill"] > 0.0
    assert (
        final["dual_ascent_missing_metric__snerv_scorer_input_distribution_guard"]
        == 0.0
    )
    assert (
        final["dual_ascent_lambda__snerv_scorer_input_distribution_guard"]
        > 0.0
    )
    assert final["loss_part_scorer_input_distribution_guard"] is not None
    telemetry_contract = report["score_aware_long_training_telemetry_contract"]
    assert telemetry_contract["passed"] is True
    assert telemetry_contract["expected_scorer_input_guard_metric"] is True
    assert telemetry_contract["scorer_input_guard_metric_observed"] is True
    assert telemetry_contract["scorer_input_guard_dual_metric_observed"] is True
    control_contract = report[
        "score_aware_long_training_required_control_contract"
    ]
    control = control_contract["controls"]["scorer_input_distribution_guard"]
    assert control["required"] is True
    assert control["bound"] is True
    assert control["telemetry_observed"] is True
    assert control["passed"] is True
    assert final["loss_part_distill"] == pytest.approx(
        final["loss_part_pr95_stage_seg_surrogate"]
    )
    assert final["loss_part_pose_distill"] == pytest.approx(
        final["loss_part_pr95_stage_pose_surrogate"]
    )
    assert final["loss_part_pose_score_term"] == pytest.approx(
        final["loss_part_pr95_stage_pose_surrogate"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


@requires_mlx
def test_snerv_scorer_tether_smoke_cli_writes_false_authority_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "snerv_scorer_tether_smoke.json"

    assert run_snerv_scorer_tether_smoke_main(["--output-json", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["blockers"] == []
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False


@requires_mlx
def test_snerv_scorer_guard_feedback_smoke_writes_queue_feedback(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "snerv_guard_feedback"

    assert (
        run_snerv_scorer_guard_feedback_smoke_main(
            ["--output-dir", str(output_dir), "--steps", "2"]
        )
        == 0
    )

    row = json.loads(
        (output_dir / "nerv_candidate_byte_feedback_row.json").read_text(
            encoding="utf-8"
        )
    )
    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["family"] == "snerv"
    assert row["snerv_scorer_domain_tether_passed"] is True
    assert row["snerv_scorer_input_distribution_guard_proof_passed"] is True
    proof = row["snerv_scorer_input_distribution_guard_proof"]
    assert proof["required"] is True
    assert proof["bound"] is True
    assert proof["metric_observed"] is True
    assert proof["dual_metric_observed"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
