# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tac.analysis.action_effect import ACTION_EFFECT_V1_SCHEMA
from tac.analysis.nerv_long_run_launch_gate import (
    ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
    BIRTH_RECEIPT_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    SOURCE_QUALIFIED_METRICS_SCHEMA,
    evaluate_nerv_long_run_launch_gate,
)

NOW = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
ACTION = "action-effect-gate-contract"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pointer(root: Path) -> Path:
    path = root / "frontier.json"
    _write(path, {"last_refreshed_utc": NOW.isoformat()})
    return path


def _birth() -> dict:
    return {
        "schema": BIRTH_RECEIPT_SCHEMA,
        "surface": "live_mlx",
        "action_id": ACTION,
        "accepted_step_count": 1,
        "argmax_transitions": {
            "target_hard_won_count": 1,
            "net_target_support_delta": 1,
        },
        "pose_guard": {
            "available": True,
            "pose_input_contest_resolution": True,
            "max_accepted_pose_output_delta_l2": 0.1,
            "max_pose_output_delta_l2": 0.2,
        },
        "exact_nonrate": {
            "pose_term_available": True,
            "delta_score_nonrate": -0.1,
        },
    }


def _parseback_contract() -> dict:
    return {
        "schema": ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
        "parseback_selection_required": True,
        "archive_parseback_axis_required": True,
        "live_only_improvement_is_false_authority": True,
        "fail_closed_on_axis_divergence": True,
        "selection_authority_order": ["archive_parseback", "live_mlx_advisory"],
    }


def _source_metrics() -> dict:
    return {
        "schema": SOURCE_QUALIFIED_METRICS_SCHEMA,
        "family": "hinerv",
        "source_qualified": True,
        "metric_source": "fixture",
    }


def test_gate_rejects_malformed_action_effect_with_exact_blocker_names(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _birth())
    _write(root / "parseback_contract.json", _parseback_contract())
    _write(root / "source_metrics.json", _source_metrics())
    _write(
        root / "bad_action_effect.json",
        {
            "schema": ACTION_EFFECT_V1_SCHEMA,
            "action_id": ACTION,
            "family": "hinerv",
            "producer": "fixture",
            "normalization_scope": "batch_local",
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    blocking = verdict["blocking_evidence"]
    assert "action_effect_untyped_authority" in blocking
    assert f"action_effect_invalid:{ACTION}:action_effect_untyped_authority" in blocking
    assert "action_effect_missing" in blocking


def test_gate_emits_generic_survival_action_id_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _birth())
    _write(root / "parseback_contract.json", _parseback_contract())
    _write(root / "source_metrics.json", _source_metrics())
    _write(
        root / "fakequant_survival.json",
        {
            "schema": BIRTH_SURVIVAL_SCHEMA,
            "surface": "fakequant_mlx",
            "action_id": "different",
            "survived": True,
            "argmax_transitions": {
                "target_hard_won_count": 1,
                "net_target_support_delta": 1,
            },
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "action_id_survival_mismatch" in verdict["blocking_evidence"]
