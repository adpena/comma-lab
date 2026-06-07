# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tac.analysis.action_effect import ActionEffect
from tac.analysis.hinerv_hard_region_miner import (
    HARD_REGION_PLAN_SCHEMA,
    OUTCOME_BIRTH_ACCEPTED,
    OUTCOME_BIRTH_REJECTED,
    REPRESENTATIVE_COVERAGE_SCHEMA,
    HardRegionMinerError,
    build_hard_region_mining_plan,
    build_representative_coverage_row,
    mine_hard_regions,
    mine_hard_regions_from_margin_map,
    size_class_for_pixels,
)
from tac.analysis.nerv_long_run_launch_gate import (
    ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
    BIRTH_HYSTERESIS_SCHEMA,
    BIRTH_RECEIPT_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    SOURCE_QUALIFIED_METRICS_SCHEMA,
    evaluate_nerv_long_run_launch_gate,
)
from tac.optimization.proxy_candidate_contract import (
    CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_AUTHORITY_OR_READINESS_KEYS = {
    *CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    "readiness",
    "readiness_verdict",
    "dispatch_readiness",
    "score_readiness",
    "exact_readiness",
}


def _logits_like(labels: np.ndarray, *, class_count: int = 5) -> np.ndarray:
    logits = np.zeros((*labels.shape, class_count), dtype=np.float64)
    logits[..., 0] = 1.0
    return logits


def _forbidden_key_paths(payload: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_AUTHORITY_OR_READINESS_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(value, prefix=path))
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            paths.extend(_forbidden_key_paths(value, prefix=f"{prefix}[{index}]"))
    return paths


def _representative_labels() -> np.ndarray:
    labels = np.zeros((1, 96, 96), dtype=np.int64)
    labels[0, 0:4, 0:4] = 1
    labels[0, 0:3, 8:11] = 1
    labels[0, 0:10, 20:30] = 2
    labels[0, 20:85, 20:85] = 3
    return labels


def test_mine_hard_regions_is_deterministic_and_round_robins_across_classes() -> None:
    labels = _representative_labels()
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels)

    first = mine_hard_regions(labels, candidate, logits, top_k=4)
    second = mine_hard_regions(labels, candidate, logits, top_k=4)

    assert [region.as_dict() for region in first] == [region.as_dict() for region in second]
    assert [region.rank for region in first] == [0, 1, 2, 3]
    assert [region.class_index for region in first] == [1, 2, 3, 1]
    assert [region.debt.region_pixel_count for region in first] == [16, 100, 4225, 9]
    assert [region.size_class for region in first] == ["small", "medium", "large", "small"]

    plan = build_hard_region_mining_plan(first, source="unit-test", top_k=4)
    assert plan["distinct_classes"] == [1, 2, 3]
    assert plan["distinct_class_size_buckets"] == [[1, "small"], [2, "medium"], [3, "large"]]
    assert plan["size_class_histogram"] == {"small": 2, "medium": 1, "large": 1}
    assert plan["authority"] == "planning_control_false_authority"
    require_no_truthy_authority_fields(plan, context="hard_region_plan")
    assert _forbidden_key_paths(plan) == []


def test_compact_target_margin_miner_matches_full_logits_ranking() -> None:
    labels = _representative_labels()
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels)
    target_logits = np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    masked_logits = logits.copy()
    np.put_along_axis(masked_logits, labels[..., None], -np.inf, axis=-1)
    target_margin = np.max(masked_logits, axis=-1) - target_logits

    full = mine_hard_regions(labels, candidate, logits, top_k=4)
    compact = mine_hard_regions_from_margin_map(
        labels,
        candidate,
        target_margin,
        top_k=4,
    )

    assert [row.as_dict() for row in compact] == [row.as_dict() for row in full]


def test_size_class_boundaries_are_closed_open() -> None:
    assert size_class_for_pixels(1) == "small"
    assert size_class_for_pixels(63) == "small"
    assert size_class_for_pixels(64) == "medium"
    assert size_class_for_pixels(4095) == "medium"
    assert size_class_for_pixels(4096) == "large"
    with pytest.raises(HardRegionMinerError, match="region_pixel_count must be >= 1"):
        size_class_for_pixels(0)


def test_pose_coupling_is_annotation_and_tiebreak_within_class() -> None:
    labels = np.zeros((1, 8, 8), dtype=np.int64)
    labels[0, 1:3, 1:3] = 1
    labels[0, 5:7, 5:7] = 1
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels, class_count=3)
    pose = np.zeros(labels.shape, dtype=np.float64)
    pose[0, 5:7, 5:7] = 9.0

    regions = mine_hard_regions(labels, candidate, logits, top_k=2, pose_coupling=pose)

    assert [region.region_label for region in regions] == [2, 1]
    assert [region.pose_coupling_risk_mean for region in regions] == [pytest.approx(9.0), pytest.approx(0.0)]
    assert all("pose_coupling_risk_mean" in region.as_dict() for region in regions)


def test_invalid_inputs_fail_closed_before_planning_rows_are_emitted() -> None:
    labels = np.zeros((1, 4, 4), dtype=np.int64)
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels, class_count=2)

    with pytest.raises(HardRegionMinerError, match="top_k must be >= 1"):
        mine_hard_regions(labels, candidate, logits, top_k=0)
    with pytest.raises(HardRegionMinerError, match="logits BHW must match labels BHW"):
        mine_hard_regions(labels, candidate, np.zeros((1, 4, 5, 2)))
    with pytest.raises(HardRegionMinerError, match="pose_coupling BHW must match labels BHW"):
        mine_hard_regions(labels, candidate, logits, pose_coupling=np.zeros((1, 5, 4)))
    with pytest.raises(ValueError, match="target_labels and candidate_argmax shapes must match"):
        mine_hard_regions(labels, np.zeros((1, 4, 5), dtype=np.int64), logits)


def test_coverage_row_accepts_named_rejections_and_counts_only_accepted_buckets() -> None:
    row = build_representative_coverage_row(
        [
            {
                "region": {"class_index": 1, "region_pixel_count": 9},
                "outcome": OUTCOME_BIRTH_ACCEPTED,
            },
            {
                "region": {"class_index": 1, "region_pixel_count": 100},
                "outcome": OUTCOME_BIRTH_ACCEPTED,
            },
            {
                "region": {"class_index": 2, "region_pixel_count": 4096},
                "outcome": OUTCOME_BIRTH_ACCEPTED,
            },
            {
                "region": {"class_index": 3, "region_pixel_count": 16},
                "outcome": OUTCOME_BIRTH_REJECTED,
                "first_failing_surface": "parseback_uint8_birth_survival",
            },
        ],
        min_distinct_classes=2,
        min_distinct_class_size_buckets=3,
    )

    assert row["schema"] == REPRESENTATIVE_COVERAGE_SCHEMA
    assert row["passed"] is True
    assert row["region_classes_covered"] == 3
    assert row["distinct_classes_accepted"] == 2
    assert row["accepted_class_size_buckets"] == [[1, "medium"], [1, "small"], [2, "large"]]
    assert row["rejected_class_size_buckets"] == [[3, "small"]]
    assert row["first_failing_surfaces"] == ["parseback_uint8_birth_survival"]
    assert row["authority"] == "planning_control_false_authority"
    assert row["human_visual_fidelity_objective"] is False
    require_no_truthy_authority_fields(row, context="coverage_row")
    assert _forbidden_key_paths(row) == []


def test_coverage_row_rejects_unnamed_or_contradictory_first_failing_surfaces() -> None:
    with pytest.raises(HardRegionMinerError, match="first_failing_surface is empty"):
        build_representative_coverage_row(
            [
                {
                    "region": {"class_index": 1, "region_pixel_count": 9},
                    "outcome": OUTCOME_BIRTH_REJECTED,
                }
            ]
        )

    with pytest.raises(HardRegionMinerError, match="birth_accepted but names first_failing_surface"):
        build_representative_coverage_row(
            [
                {
                    "region": {"class_index": 1, "region_pixel_count": 9},
                    "outcome": OUTCOME_BIRTH_ACCEPTED,
                    "first_failing_surface": "fakequant_mlx",
                }
            ]
        )

    with pytest.raises(HardRegionMinerError, match="outcome\\[0\\]\\.outcome must be one of"):
        build_representative_coverage_row(
            [
                {
                    "region": {"class_index": 1, "region_pixel_count": 9},
                    "outcome": "birth_maybe",
                }
            ]
        )


# --------------------------------------------------------------------------- #
# THE integration test: coverage row consumed by the launch gate (L4 -> L5).
# This is the miner's whole reason to exist; it proves the schema the gate
# DEFINES is satisfiable by a real coverage row and flips the gate verdict.
# --------------------------------------------------------------------------- #
def _support_stats() -> dict[str, float]:
    return {
        "receiver_surface_target_hard_won_count": 5.0,
        "receiver_surface_net_target_support_delta": 3.0,
    }


def _action_effect_row(
    *,
    action_id: str,
    arm: str | None = None,
    action_kind: str = "target_region_birth",
) -> dict[str, Any]:
    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind=action_kind,
        authority="archive_parseback_planning_false_authority",
        producer="hinerv_hard_region_miner_test",
        consumer="nerv_long_run_launch_gate",
        pair_ids=[0],
        region_ids=["b0/c1/r0"],
        payload_sections=["head_rgb_1.weight"],
        old_d_seg=0.0010,
        new_d_seg=0.0007,
        old_d_pose=1.0e-4,
        new_d_pose=8.0e-5,
        old_bytes=178_258,
        new_bytes=178_258,
        exact_score_decision="accept",
        raw_cap_decision="satisfied",
        catastrophic_guard_decision="satisfied",
        would_accept_exact_score_if_raw_cap_disabled=True,
        would_accept_without_catastrophic_guard=True,
        rejected_by_raw_cap=False,
        rejected_by_exact_score=False,
        rejected_by_catastrophic_guard=False,
        fakequant_survived=True,
        parseback_survived=True,
        inflate_survived=True,
        hard_won_count=5,
        wrong_to_target=5,
        target_to_wrong=0,
        wrong_to_wrong=0,
        net_target_support_delta=3,
        uint8_changed_count_region=16,
        seg_input_delta_linf_region=1.0 / 255.0,
        posenet_input_delta_linf_pair=1.0 / 255.0,
        arm=arm,
    ).as_dict()


def _four_arm_action_effect_rows(*, action_id: str) -> list[dict[str, Any]]:
    return [
        _action_effect_row(action_id=action_id, arm="A", action_kind="birth_only"),
        _action_effect_row(action_id=action_id, arm="B", action_kind="frame0_pose_target_only"),
        _action_effect_row(action_id=action_id, arm="C", action_kind="independent_birth_plus_frame0_pose"),
        _action_effect_row(action_id=action_id, arm="D", action_kind="joint_line_search_composite"),
    ]


def _write_full_birth_ladder(run_root: Path, *, action_id: str, with_coverage: bool) -> None:
    """Write the complete L2->L5 fixture ladder the launch gate requires.

    Birth receipt (live) + pose-trusted exact_nonrate + survival on all three
    surfaces carrying the SAME action_id + hysteresis + (optionally) a coverage
    row built by the miner under test.
    """

    support = _support_stats()
    birth = {
        "schema": BIRTH_RECEIPT_SCHEMA,
        "surface": "live_mlx",
        "accepted_step_count": 3,
        "action_id": action_id,
        "runtime_sidecar_bytes": 0,
        "pose_guard": {
            "available": True,
            "pose_input_contest_resolution": True,
            "max_accepted_pose_output_delta_l2": 0.025,
            "max_pose_output_delta_l2": 0.05,
        },
        "exact_nonrate": {"pose_term_available": True, "delta_score_nonrate": -0.01},
        **support,
    }
    (run_root / "birth.json").write_text(json.dumps(birth), encoding="utf-8")
    for surface, name in (
        ("fakequant_mlx", "surv_fakequant.json"),
        ("parseback_mlx", "surv_parseback.json"),
        ("inflated_torch_cpu", "surv_inflate.json"),
    ):
        row = {
            "schema": BIRTH_SURVIVAL_SCHEMA,
            "surface": surface,
            "action_id": action_id,
            "survived": True,
            **support,
        }
        (run_root / name).write_text(json.dumps(row), encoding="utf-8")
    (run_root / "hysteresis.json").write_text(
        json.dumps({"schema": BIRTH_HYSTERESIS_SCHEMA, "action_id": action_id, "passed": True}),
        encoding="utf-8",
    )
    (run_root / "action_effect.json").write_text(
        json.dumps(
            {
                "rows": [
                    _action_effect_row(action_id=action_id),
                    *_four_arm_action_effect_rows(action_id=action_id),
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_root / "parseback_contract.json").write_text(
        json.dumps(
            {
                "schema": ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
                "parseback_selection_required": True,
                "archive_parseback_axis_required": True,
                "live_only_improvement_is_false_authority": True,
                "fail_closed_on_axis_divergence": True,
                "selection_authority_order": ["archive_parseback", "live_mlx_advisory"],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "source_metrics.json").write_text(
        json.dumps(
            {
                "schema": SOURCE_QUALIFIED_METRICS_SCHEMA,
                "family": "hinerv",
                "source_qualified": True,
                "metric_source": "upstream_evaluate_geometry",
                "seg_metric_source": "segnet_last_frame_argmax",
                "pose_metric_source": "posenet_yuv6_pair",
            }
        ),
        encoding="utf-8",
    )
    if with_coverage:
        coverage = build_representative_coverage_row(
            [
                {"region": {"class_index": 1, "region_pixel_count": 9}, "outcome": OUTCOME_BIRTH_ACCEPTED},
                {"region": {"class_index": 3, "region_pixel_count": 100}, "outcome": OUTCOME_BIRTH_ACCEPTED},
                {"region": {"class_index": 2, "region_pixel_count": 5000}, "outcome": OUTCOME_BIRTH_ACCEPTED},
            ]
        )
        (run_root / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")


def _fresh_frontier_pointer(run_root: Path) -> Path:
    pointer = run_root / "frontier.json"
    pointer.write_text(json.dumps({"last_refreshed_utc": datetime.now(UTC).isoformat()}), encoding="utf-8")
    return pointer


def test_coverage_row_flips_launch_gate_to_l5(tmp_path: Path) -> None:
    action_id = "swarmc_action_xyz"
    _write_full_birth_ladder(tmp_path, action_id=action_id, with_coverage=True)
    pointer = _fresh_frontier_pointer(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(family="hinerv", run_root=tmp_path, frontier_pointer=pointer)
    assert "representative_region_coverage_missing" not in verdict["blocking_evidence"]
    assert verdict["highest_level"] == "L5"
    assert verdict["approved"] is True


def test_missing_coverage_keeps_gate_at_l4(tmp_path: Path) -> None:
    # Negative control: proves the L5 assertion above is not a tautology — the
    # gate emits the coverage blocker and stalls at L4 without the miner's row.
    _write_full_birth_ladder(tmp_path, action_id="swarmc_action_xyz", with_coverage=False)
    pointer = _fresh_frontier_pointer(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(family="hinerv", run_root=tmp_path, frontier_pointer=pointer)
    assert "representative_region_coverage_missing" in verdict["blocking_evidence"]
    assert verdict["highest_level"] == "L4"
    assert verdict["approved"] is False


# --------------------------------------------------------------------------- #
# CLI subprocess smoke
# --------------------------------------------------------------------------- #
def test_cli_emits_durable_plan_on_npz(tmp_path: Path) -> None:
    # NPZ inputs live in the pytest tmp dir (reading is fine); the plan output
    # must land on a DURABLE path because the CLI refuses /tmp outputs per the
    # CLAUDE.md custody non-negotiable.  Use a unique experiments/results subdir
    # and remove it afterwards so the test leaves no orphan bytes.
    import shutil
    import uuid

    labels = _representative_labels()
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels)
    npz = tmp_path / "inputs.npz"
    np.savez(npz, labels=labels, argmax=candidate, logits=logits)

    out_dir = REPO_ROOT / "experiments" / "results" / f"_test_hinerv_hard_region_cli_{uuid.uuid4().hex}"
    out = out_dir / "plan.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "mine_hinerv_hard_regions.py"),
        "--target-labels",
        f"{npz}::labels",
        "--candidate-argmax",
        f"{npz}::argmax",
        "--logits",
        f"{npz}::logits",
        "--top-k",
        "4",
        "--output",
        str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
        assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
        assert out.is_file()
        plan = json.loads(out.read_text(encoding="utf-8"))
        assert plan["schema"] == HARD_REGION_PLAN_SCHEMA
        assert plan["distinct_classes"] == [1, 2, 3]
        require_no_truthy_authority_fields(plan, context="cli_plan")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_cli_refuses_tmp_output_path(tmp_path: Path) -> None:
    # The CLI must fail closed when an operator points --output at a /tmp tier.
    labels = _representative_labels()
    candidate = np.zeros_like(labels)
    logits = _logits_like(labels)
    npz = tmp_path / "inputs.npz"
    np.savez(npz, labels=labels, argmax=candidate, logits=logits)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "mine_hinerv_hard_regions.py"),
        "--target-labels",
        f"{npz}::labels",
        "--candidate-argmax",
        f"{npz}::argmax",
        "--logits",
        f"{npz}::logits",
        "--output",
        "/tmp/swarmc_should_be_refused.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
    assert result.returncode != 0
    assert "tmp" in (result.stderr + result.stdout).lower()
