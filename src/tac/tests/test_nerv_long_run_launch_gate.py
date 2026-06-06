# SPDX-License-Identifier: MIT
"""Fail-closed behavior of the NeRV long-run launch gate.

All evidence files here are synthetic fixtures (labelled, tmp-dir only) used
to verify the gate's refusal logic; they are not empirical anchors and grant
no score authority.  The gate must approve ONLY on a complete, consistent
ladder, and every missing/mismatched row must be NAMED in the verdict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac.analysis.action_effect import ActionEffect
from tac.analysis.nerv_long_run_launch_gate import (
    ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
    BIRTH_HYSTERESIS_SCHEMA,
    BIRTH_RECEIPT_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    REPRESENTATIVE_COVERAGE_SCHEMA,
    SNERV_SOURCE_FORWARD_SCHEMA,
    SOURCE_QUALIFIED_METRICS_SCHEMA,
    NervLongRunLaunchGateError,
    evaluate_nerv_long_run_launch_gate,
)
from tac.analysis.snerv_source_forward_proof import (
    build_snerv_payload_bitflip_falsification,
    build_snerv_source_forward_proof_action_effect,
)

NOW = datetime(2026, 6, 6, 21, 0, 0, tzinfo=UTC)
ACTION = "a" * 64


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snerv_tensor_surfaces(*, delta: float = 0.0) -> dict:
    base = {
        "coord_time_embedding": [[0.0, 1.0]],
        "mfu_in": [[[[1.0, 2.0], [3.0, 4.0]]]],
        "mfu_out": [[[[2.0, 3.0], [4.0, 5.0]]]],
        "hfr_in": [[[[2.0, 3.0], [4.0, 5.0]]]],
        "hfr_out": [[[[0.1, 0.2], [0.3, 0.4]]]],
        "tub_in": [[[[0.5, 0.6], [0.7, 0.8]]]],
        "tub_out": [[[[0.6, 0.7], [0.8, 0.9]]]],
        "output_2": [[[[0.01, 0.02], [0.03, 0.04]]]],
        "rgb_pair_float": [[[[[1.0, 2.0], [3.0, 4.0]]]]],
        "rgb_pair_uint8": [[[[[1, 2], [3, 4]]]]],
        "segnet_input": [[[[0.1, 0.2], [0.3, 0.4]]]],
        "posenet_input": [[[[0.1, 0.2], [0.3, 0.4]]]],
        "segnet_logits": [[[[0.0, 2.0], [1.0, 3.0]]]],
        "segnet_argmax": [[1, 1]],
        "posenet_output": [[0.25, 0.5, 0.75]],
    }
    surfaces = {}
    for surface in ("official_torch", "pact_mlx", "archive_parseback", "numpy_receiver"):
        surfaces[surface] = dict(base)
    if delta:
        surfaces["numpy_receiver"]["output_2"] = [[[[delta, 0.02], [0.03, 0.04]]]]
    return surfaces


def _snerv_source_forward_action_row(
    *,
    bitflip_passes_proof: bool = False,
    tensor_delta: float = 0.0,
) -> dict:
    bitflip = build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=bitflip_passes_proof,
        first_failed_tensor=None if bitflip_passes_proof else "output_2",
        first_failed_surface=None if bitflip_passes_proof else "archive_parseback",
        bit_offset=17,
        bit_mask=1,
    )
    return build_snerv_source_forward_proof_action_effect(
        action_id=ACTION,
        archive_sha256="1" * 64,
        archive_bytes=12345,
        payload_section_hashes={
            "lf_payload": "a" * 64,
            "decoder_payload": "2" * 64,
            "output_2": "5" * 64,
        },
        pair_ids=[0],
        tensors_by_surface=_snerv_tensor_surfaces(delta=tensor_delta),
        scorer_deltas={
            "d_seg": 0.0,
            "d_pose": 0.0,
            "delta_score_nonrate": 0.0,
        },
        destructive_payload_bit_flip=bitflip,
    )


def _pointer(tmp_path: Path, *, age_hours: float = 1.0) -> Path:
    path = tmp_path / "canonical_frontier_pointer.json"
    refreshed = NOW - timedelta(hours=age_hours)
    _write(path, {"last_refreshed_utc": refreshed.isoformat()})
    return path


def _live_birth_receipt(
    *,
    action_id: str = ACTION,
    pose_trusted: bool = True,
    hard_won: int = 7932,
    net_support: int = 7932,
) -> dict:
    return {
        "schema": BIRTH_RECEIPT_SCHEMA,
        "fixture_not_real": True,
        "surface": "live_mlx",
        "action_id": action_id,
        "accepted_step_count": 1,
        "runtime_sidecar_bytes": 0,
        "argmax_transitions": {
            "target_hard_won_count": hard_won,
            "target_hard_lost_count": max(0, hard_won - net_support),
            "net_target_support_delta": net_support,
        },
        "pose_guard": {
            "available": pose_trusted,
            "pose_input_contest_resolution": pose_trusted,
            "max_accepted_pose_output_delta_l2": 0.025 if pose_trusted else None,
            "max_pose_output_delta_l2": 0.05,
        },
        "exact_nonrate": {
            "pose_term_available": pose_trusted,
            "delta_score_nonrate": -0.012 if pose_trusted else None,
        },
    }


def _survival(
    surface: str,
    *,
    action_id: str = ACTION,
    survived: bool = True,
    include_support: bool = True,
    hard_won: int = 2048,
    net_support: int = 2048,
) -> dict:
    argmax_transitions = (
        {
            "target_hard_won_count": hard_won,
            "target_hard_lost_count": max(0, hard_won - net_support),
            "net_target_support_delta": net_support,
        }
        if include_support
        else None
    )
    return {
        "schema": BIRTH_SURVIVAL_SCHEMA,
        "fixture_not_real": True,
        "surface": surface,
        "action_id": action_id,
        "survived": survived,
        "argmax_transitions": argmax_transitions,
    }


def _hi_nerv_action_effect() -> dict:
    return ActionEffect.build(
        action_id=ACTION,
        family="hinerv",
        authority="archive_parseback_planning_false_authority",
        producer="hinerv_v6_four_arm_composite_ablation",
        consumer="nerv_long_run_launch_gate",
        pair_ids=[0],
        region_ids=["b0/c2/r1"],
        payload_sections=["head_rgb_1.weight"],
        old_d_seg=0.0010,
        new_d_seg=0.0008,
        old_d_pose=1.0e-4,
        new_d_pose=9.0e-5,
        old_bytes=178_258,
        new_bytes=178_258,
        parseback_survived=True,
        inflate_survived=True,
        fakequant_survived=True,
        hard_won_count=2048,
        wrong_to_target=2048,
        target_to_wrong=0,
        wrong_to_wrong=0,
        net_target_support_delta=2048,
        uint8_changed_count_region=4096,
        seg_input_delta_linf_region=1.0 / 255.0,
        posenet_input_delta_linf_pair=1.0 / 255.0,
    ).as_dict()


def _parseback_selection_contract() -> dict:
    return {
        "schema": ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
        "fixture_not_real": True,
        "parseback_selection_required": True,
        "archive_parseback_axis_required": True,
        "live_only_improvement_is_false_authority": True,
        "fail_closed_on_axis_divergence": True,
        "selection_authority_order": ["archive_parseback", "live_mlx_advisory"],
    }


def _source_qualified_metrics() -> dict:
    return {
        "schema": SOURCE_QUALIFIED_METRICS_SCHEMA,
        "fixture_not_real": True,
        "family": "hinerv",
        "source_qualified": True,
        "metric_source": "upstream_evaluate_geometry",
        "seg_metric_source": "segnet_last_frame_argmax",
        "pose_metric_source": "posenet_yuv6_pair",
    }


def _full_hi_nerv_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(root / "fakequant.json", _survival("fakequant_mlx"))
    _write(root / "parseback.json", _survival("parseback_mlx"))
    _write(root / "inflate.json", _survival("inflated_torch_cpu"))
    _write(root / "action_effect.json", _hi_nerv_action_effect())
    _write(root / "parseback_contract.json", _parseback_selection_contract())
    _write(root / "source_metrics.json", _source_qualified_metrics())
    _write(
        root / "hysteresis.json",
        {
            "schema": BIRTH_HYSTERESIS_SCHEMA,
            "fixture_not_real": True,
            "action_id": ACTION,
            "passed": True,
        },
    )
    _write(
        root / "coverage.json",
        {
            "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
            "fixture_not_real": True,
            "passed": True,
            "region_classes_covered": 3,
            "distinct_classes_accepted": 2,
            "accepted_count": 3,
            "min_distinct_classes": 2,
            "min_distinct_class_size_buckets": 3,
        },
    )
    return root


def test_unknown_family_and_missing_root_fail_loud(tmp_path: Path) -> None:
    with pytest.raises(NervLongRunLaunchGateError, match="family"):
        evaluate_nerv_long_run_launch_gate(family="nope", run_root=tmp_path, now_utc=NOW)
    with pytest.raises(NervLongRunLaunchGateError, match="run_root"):
        evaluate_nerv_long_run_launch_gate(family="hi_nerv", run_root=tmp_path / "missing", now_utc=NOW)


def test_empty_root_blocks_everything(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert verdict["highest_level"] == "none"
    assert "real_video_birth_receipt_missing" in verdict["blocking_evidence"]
    # The gate itself is planning-only and never a score authority.
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False


def test_live_birth_without_pose_trust_is_l2(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt(pose_trusted=False))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["highest_level"] == "L2"
    assert "pose_trusted_birth_receipt_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_live_birth_without_pose_cap_telemetry_is_not_pose_trusted(tmp_path: Path) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt(pose_trusted=True)
    row["pose_guard"].pop("max_accepted_pose_output_delta_l2")
    _write(root / "birth.json", row)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["highest_level"] == "L2"
    assert "pose_trusted_birth_receipt_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_zero_net_support_is_not_a_birth(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt(hard_won=1, net_support=0))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["highest_level"] == "none"
    assert "real_video_birth_receipt_missing" in verdict["blocking_evidence"]
    assert "live_birth_target_support_not_positive" in verdict["blocking_evidence"]


def test_survival_action_id_mismatch_is_named(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(
        root / "fakequant.json",
        _survival("fakequant_mlx", action_id="b" * 64),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "l4_survival_action_id_mismatch:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["highest_level"] == "L3"


def test_not_survived_row_blocks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(root / "parseback.json", _survival("parseback_mlx", survived=False))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert "birth_not_survived:parseback_mlx" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_pose_compensation_must_survive_even_when_target_support_survives(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    row = _survival("fakequant_mlx")
    row["pose_compensation_required"] = True
    row["pose_compensation_survived"] = False
    _write(root / "fakequant.json", row)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "birth_survival_pose_compensation_not_survived:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["highest_level"] == "L3"
    assert verdict["approved"] is False


def test_survived_row_without_target_support_blocks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(
        root / "fakequant.json",
        _survival("fakequant_mlx", include_support=False),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "birth_survival_target_support_missing:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["approved"] is False


def test_full_ladder_with_fresh_pointer_approves(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["blocking_evidence"] == []
    assert verdict["highest_level"] == "L5"
    assert verdict["approved"] is True


def test_failed_representative_coverage_blocks_l5(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "coverage.json",
        {
            "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
            "fixture_not_real": True,
            "passed": False,
            "region_classes_covered": 3,
            "distinct_classes_accepted": 2,
            "accepted_count": 3,
            "min_distinct_classes": 2,
            "min_distinct_class_size_buckets": 3,
        },
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert verdict["highest_level"] == "L4"
    assert "representative_region_coverage_not_passed" in verdict["blocking_evidence"]


def test_contradictory_representative_coverage_blocks_l5(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "coverage.json",
        {
            "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
            "fixture_not_real": True,
            "passed": True,
            "region_classes_covered": 1,
            "distinct_classes_accepted": 1,
            "accepted_count": 1,
            "min_distinct_classes": 2,
            "min_distinct_class_size_buckets": 3,
        },
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert verdict["highest_level"] == "L4"
    blocking = verdict["blocking_evidence"]
    assert "representative_region_coverage_region_classes_below_threshold" in blocking
    assert "representative_region_coverage_distinct_classes_below_threshold" in blocking


def test_hinerv_family_alias_is_canonicalized(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["family"] == "hinerv"
    assert verdict["approved"] is True


def test_stale_pointer_blocks_even_complete_ladder(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path, age_hours=48.0),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert "frontier_pointer_stale" in verdict["blocking_evidence"]


def test_missing_pointer_blocks(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=tmp_path / "nope.json",
        now_utc=NOW,
    )
    assert "frontier_pointer_missing" in verdict["blocking_evidence"]


def test_truthy_authority_evidence_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    receipt = _live_birth_receipt()
    receipt["score_claim"] = True  # forged authority on an evidence row
    _write(root / "birth.json", receipt)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert any(item.startswith("evidence_truthy_authority:") for item in verdict["blocking_evidence"])
    assert verdict["highest_level"] == "none"


def test_snerv_requires_proof_and_bitflip(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "snerv_full_source_forward_parity_missing" in blocking
    assert "snerv_payload_bitflip_falsification_missing" in blocking

    # The pre-action-effect metadata shape must stay blocked even when it
    # claims parity and a named tensor failure.
    _write(
        root / "proof.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "full_tub_source_forward_parity_proven": True,
        },
    )
    _write(
        root / "bitflip.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "bitflip_section": "TUB",
            "proof_passed": False,
            "first_failed_tensor": "TUB_out",
        },
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert "snerv_full_source_forward_parity_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False

    _write(root / "action_effect.json", _snerv_source_forward_action_row())
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["blocking_evidence"] == []
    assert verdict["highest_level"] == "L4"
    assert verdict["approved"] is True


def test_snerv_bitflip_that_passes_proof_is_not_falsification(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(bitflip_passes_proof=True),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert "snerv_payload_bitflip_falsification_missing" in verdict["blocking_evidence"]
    assert "snerv_full_source_forward_parity_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_snerv_source_forward_tensor_delta_blocks_launch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "proof.json", _snerv_source_forward_action_row(tensor_delta=9.0))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert any(
        item.endswith("source_forward_tensor_delta_exceeds_tolerance:numpy_receiver:output_2")
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False
