# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.canonical_equations.ddm_rr9_mem_probe_fire_protocol_20260807 import (
    EQUATION_ID,
    build_ddm_rr9_mem_probe_fire_protocol_v1,
    metal_fire_protocol_status,
)


def test_required_missing_mem_probe_refuses_metal_training_fire() -> None:
    status = metal_fire_protocol_status(
        mem_probe_receipt_required=True,
        mem_probe_receipt_exists=False,
        mem_probe_status=None,
        has_load_stage_samples=False,
        launch_kind="mlx-train",
    )
    assert status == {"allowed": False, "reason": "missing_required_mem_probe_receipt"}


def test_passed_mem_probe_allows_metal_training_fire() -> None:
    status = metal_fire_protocol_status(
        mem_probe_receipt_required=True,
        mem_probe_receipt_exists=True,
        mem_probe_status="passed",
        has_load_stage_samples=True,
        launch_kind="mlx-train",
    )
    assert status["allowed"] is True


def test_cpu_or_non_training_fire_is_out_of_scope_allowed() -> None:
    status = metal_fire_protocol_status(
        mem_probe_receipt_required=True,
        mem_probe_receipt_exists=False,
        mem_probe_status=None,
        has_load_stage_samples=False,
        launch_kind="byte-pricing",
    )
    assert status["allowed"] is True
    assert status["reason"] == "not_metal_training_fire"


def test_equation_builds_guard_anchor() -> None:
    eq = build_ddm_rr9_mem_probe_fire_protocol_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors[0].predicted_output["allowed"] is False
    assert eq.domain_of_validity["score_claim"] is False
