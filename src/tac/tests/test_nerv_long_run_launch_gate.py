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
from tac.analysis.evaluator_action_lowering_race import (
    LOWERING_TARGETS,
    build_lowering_race_report,
)
from tac.analysis.hinerv_hard_region_miner import (
    OUTCOME_BIRTH_ACCEPTED,
    build_representative_coverage_row,
)
from tac.analysis.inverse_scorer_actions import (
    SIDECAR_FALLBACK_ACCEPTED,
    TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
    WALL_NORMAL_BRANCH_RECEIPT_SCHEMA,
)
from tac.analysis.nerv_long_run_launch_gate import (
    ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
    BIRTH_HYSTERESIS_SCHEMA,
    BIRTH_RECEIPT_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    EVALUATOR_ACTION_LOWERING_RACE_SCHEMA,
    HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
    HI_NERV_TARGET_REGION_ACTION_LOWERING_RACE_SCHEMA,
    REPRESENTATIVE_COVERAGE_SCHEMA,
    SNERV_SOURCE_FORWARD_SCHEMA,
    SOURCE_QUALIFIED_METRICS_SCHEMA,
    NervLongRunLaunchGateError,
    evaluate_nerv_long_run_launch_gate,
)
from tac.analysis.snerv_source_forward_proof import (
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
    SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
    SOURCE_IDENTICAL,
    build_snerv_payload_bitflip_falsification,
    build_snerv_payload_bitflip_falsification_matrix,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
)

NOW = datetime(2026, 6, 6, 21, 0, 0, tzinfo=UTC)
ACTION = "a" * 64


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _charged_target_region_actions(
    *,
    payload_codec: str = "raw_v1",
    support_encoding: str = "explicit_yx_u16_coordinates",
    support_encoded_bytes: int = 128,
) -> dict[str, object]:
    return {
        "schema": "hi_nerv_target_region_archive_actions.v1",
        "charged_as_hiv1_meta_blob": True,
        "receiver_consumed": True,
        "payload_codec": payload_codec,
        "encoded_program_sha256": "d" * 64,
        "decoded_support_sha256": "e" * 64,
        "decoded_action_sha256": "f" * 64,
        "support_encoding": support_encoding,
        "support_encoded_bytes": support_encoded_bytes,
        "payload_bytes": 128,
        "base64_text_bytes": 172,
        "charged_meta_json_bytes": 512,
    }


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
    include_scorer_by_surface: bool = True,
    parseback_d_pose: float = 0.0,
    include_surface_provenance: bool = True,
    provenance_authority: str = "real_surface_forward_capture",
    output2_verdict: str = SOURCE_IDENTICAL,
) -> dict:
    bitflip = build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=bitflip_passes_proof,
        first_failed_tensor=None if bitflip_passes_proof else "output_2",
        first_failed_surface=None if bitflip_passes_proof else "archive_parseback",
        receiver_replay_failed=not bitflip_passes_proof,
        bit_offset=17,
        bit_mask=1,
    )
    bitflip_matrix = _snerv_bitflip_matrix(
        proof_passed_after_bitflip=bitflip_passes_proof
    )
    scorer_deltas = {
        "d_seg": 0.0,
        "d_pose": 0.0,
        "delta_score_nonrate": 0.0,
    }
    if include_scorer_by_surface:
        scorer_deltas["by_surface"] = {
            surface: {"d_seg": 0.0, "d_pose": 0.0}
            for surface in (
                "official_torch",
                "pact_mlx",
                "archive_parseback",
                "numpy_receiver",
            )
        }
        scorer_deltas["by_surface"]["archive_parseback"]["d_pose"] = parseback_d_pose
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
        scorer_deltas=scorer_deltas,
        destructive_payload_bit_flip=bitflip,
        destructive_payload_bit_flip_matrix=bitflip_matrix,
        output2_boundary_verdict=_snerv_output2_boundary_verdict(
            verdict=output2_verdict
        ),
        surface_provenance=(
            _snerv_surface_provenance(provenance_authority=provenance_authority)
            if include_surface_provenance
            else None
        ),
    )


def _snerv_bitflip_matrix(*, proof_passed_after_bitflip: bool = False) -> dict:
    first_tensors = {
        "metadata_payload": "coord_time_embedding",
        "lf_payload": "tub_in",
        "decoder_payload.mfu": "mfu_out",
        "decoder_payload.hfr": "hfr_out",
        "decoder_payload.tub": "tub_in",
        "decoder_payload.output_2": "output_2",
        "step_map_packet": "rgb_pair_uint8",
    }
    return build_snerv_payload_bitflip_falsification_matrix(
        {
            section: build_snerv_payload_bitflip_falsification(
                bitflip_section=section,
                baseline_section_sha256=f"{idx + 1:x}" * 64,
                mutated_section_sha256=f"{idx + 5:x}" * 64,
                proof_passed_after_bitflip=proof_passed_after_bitflip,
                first_failed_tensor=(
                    None if proof_passed_after_bitflip else first_tensor
                ),
                first_failed_surface=(
                    None if proof_passed_after_bitflip else "archive_parseback"
                ),
                receiver_replay_failed=(
                    not proof_passed_after_bitflip and first_tensor != "rgb_pair_uint8"
                ),
                rgb_pair_uint8_changed=(
                    not proof_passed_after_bitflip and first_tensor == "rgb_pair_uint8"
                ),
                bit_offset=idx,
                bit_mask=1,
            )
            for idx, (section, first_tensor) in enumerate(first_tensors.items())
        }
    )


def _snerv_output2_boundary_verdict(*, verdict: str = SOURCE_IDENTICAL) -> dict:
    passed = verdict == SOURCE_IDENTICAL
    return {
        "schema": SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
        "verdict": verdict,
        "passed": passed,
        "has_output2_by_surface": {
            "official_torch": True,
            "pact_mlx": True,
            "archive_parseback": True,
            "numpy_receiver": True,
        },
        "output2_shapes_by_surface": {
            "official_torch": [1, 1, 2, 2],
            "pact_mlx": [1, 1, 2, 2],
            "archive_parseback": [1, 1, 2, 2],
            "numpy_receiver": [1, 1, 2, 2],
        },
        "archive_tub_output2_storage": {
            "section": "decoder_payload.output_2",
            "sha256": "5" * 64,
            "bytes": 64,
            "stored": True,
            "source_payload_present": True,
            "receiver_executes_output2_fusion_from_payload": passed,
            "receiver_frame_decode_consumes_output2": passed,
            "receiver_output2_frame_shape_match": passed,
            "shape_adapter_forbidden": True,
            "shape_adapter_applied": False,
        },
        "minimal_causal_basis_recommendation": (
            ["keep_output2_source_forward_bound"]
            if passed
            else [
                "lf_carrier",
                "hf_carrier",
                "mfu_state",
                "hfr_state",
                "tub_temporal_state",
                "pair_adapter",
                "derive_output_2",
            ]
        ),
        "blockers": [] if passed else ["snerv_output2_not_in_selected_source_forward_basis"],
        "required_next_step": (
            "output2_boundary_closed"
            if passed
            else "store_lf_hf_mfu_hfr_tub_pair_adapter_and_derive_output2"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_surface_provenance(
    *,
    provenance_authority: str = "real_surface_forward_capture",
) -> dict[str, dict[str, object]]:
    surfaces = (
        "official_torch",
        "pact_mlx",
        "archive_parseback",
        "numpy_receiver",
    )
    tensor_authority = dict.fromkeys(
        surfaces,
        provenance_authority,
    )
    if provenance_authority == "real_surface_forward_capture":
        tensor_authority["official_torch"] = "upstream_snerv_t_forward_source_graph"
    return build_snerv_source_forward_surface_provenance(
        pair_ids=[0],
        archive_sha256="1" * 64,
        producer_by_surface={
            surface: f"{surface}_producer" for surface in surfaces
        },
        tensor_capture_authority_by_surface=tensor_authority,
        scorer_capture_authority_by_surface=dict.fromkeys(
            surfaces,
            provenance_authority,
        ),
        extra_by_surface={
            "official_torch": {
                "trained_checkpoint_lineage": "official_trained_checkpoint_state_dict",
                "checkpoint_sha256": "6" * 64,
                "state_dict_sha256": "7" * 64,
                "model_source_sha256": "8" * 64,
                "source_config_lineage": "official_trained_run_config",
                "source_config_sha256": "9" * 64,
                "source_config_kind": "official_snerv_t_train_config",
                "source_config_source": "unit_test_exact_trained_config",
                "source_config_is_fixture": False,
                "source_scope": "official_trained_checkpoint",
                "capture_origin": "official_upstream_trained_checkpoint",
            }
        },
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
    support_sha256: str = "9" * 64,
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
        "support_sha256": support_sha256,
        "survived": survived,
        "argmax_transitions": argmax_transitions,
    }


def _hi_nerv_action_effect(
    *,
    arm: str | None = None,
    action_kind: str = "target_region_birth",
    exact_score_decision: str = "accept",
    new_d_seg: float = 0.0008,
    new_d_pose: float = 9.0e-5,
    raw_cap_decision: str = "satisfied",
    catastrophic_guard_decision: str = "satisfied",
    rejected_by_raw_cap: bool = False,
    rejected_by_exact_score: bool = False,
    rejected_by_catastrophic_guard: bool = False,
    target_to_wrong: int = 0,
    wrong_to_wrong: int = 0,
    support_sha256: str = "9" * 64,
) -> dict:
    return ActionEffect.build(
        action_id=ACTION,
        family="hinerv",
        action_kind=action_kind,
        authority="archive_parseback_planning_false_authority",
        producer="hinerv_v6_four_arm_composite_ablation",
        consumer="nerv_long_run_launch_gate",
        pair_ids=[0],
        region_ids=["b0/c2/r1"],
        payload_sections=["head_rgb_1.weight"],
        old_d_seg=0.0010,
        new_d_seg=new_d_seg,
        old_d_pose=1.0e-4,
        new_d_pose=new_d_pose,
        old_bytes=178_258,
        new_bytes=178_258,
        exact_score_decision=exact_score_decision,
        raw_cap_decision=raw_cap_decision,
        catastrophic_guard_decision=catastrophic_guard_decision,
        would_accept_exact_score_if_raw_cap_disabled=exact_score_decision == "accept",
        would_accept_without_catastrophic_guard=exact_score_decision == "accept",
        rejected_by_raw_cap=rejected_by_raw_cap,
        rejected_by_exact_score=rejected_by_exact_score,
        rejected_by_catastrophic_guard=rejected_by_catastrophic_guard,
        parseback_survived=True,
        inflate_survived=True,
        fakequant_survived=True,
        hard_won_count=2048,
        wrong_to_target=2048,
        target_to_wrong=target_to_wrong,
        wrong_to_wrong=wrong_to_wrong,
        net_target_support_delta=2048 - target_to_wrong,
        uint8_changed_count_region=4096,
        seg_input_delta_linf_region=1.0 / 255.0,
        posenet_input_delta_linf_pair=1.0 / 255.0,
        support_sha256=support_sha256,
        support_source="archive_executable_target_region_action_support",
        support_cardinality=2048,
        support_encoding="target_region_action_coordinates_v1",
        support_encoded_bytes=8192,
        support_research_only=False,
        arm=arm,
    ).as_dict()


def _wall_normal_lift(
    *,
    action_id: str = ACTION,
    selected_next_operator: str = "backend_fit_live",
    direct_crossed: bool = True,
    direct_exact_score_decision: str | None = None,
    direct_exact_delta_score_nonrate: float | None = None,
    direct_teacher_is_true_wall_normal: bool = True,
    backend_realized: bool = True,
    backend_wrong_to_target: int = 2048,
    exact_score_decision: str = "accept",
    backend_action_effect_action_id: str | None = None,
    blockers: list[str] | None = None,
) -> dict:
    return {
        "schema": TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
        "fixture_not_real": True,
        "operator": "TargetRegionWallNormalLift",
        "action_id": action_id,
        "authority": "batch_local_live_mlx",
        "pair_id": 0,
        "target_class": 2,
        "region_id": "b0/c2/r1",
        "direct_teacher": {
            "available": True,
            "crossed_target_wall": direct_crossed,
            "teacher_is_true_wall_normal": direct_teacher_is_true_wall_normal,
            "inverse_source": (
                "segnet_margin_vjp"
                if direct_teacher_is_true_wall_normal
                else "masked_residual"
            ),
            "wrong_to_target_count": 4096 if direct_crossed else 0,
            "target_to_wrong_count": 0,
            "exact_delta_score_nonrate": (
                direct_exact_delta_score_nonrate
                if direct_exact_delta_score_nonrate is not None
                else (-1.0 if direct_crossed else 0.0)
            ),
            "exact_score_decision": (
                direct_exact_score_decision
                if direct_exact_score_decision is not None
                else ("accept" if direct_crossed else "reject")
            ),
        },
        "backend_fit": {
            "attempted": True,
            "realized_target_wall": backend_realized,
            "wrong_to_target_count": backend_wrong_to_target,
            "target_to_wrong_count": 0,
            "exact_score_decision": exact_score_decision,
            "action_effect": (
                None
                if backend_action_effect_action_id is None
                else {"action_id": backend_action_effect_action_id}
            ),
        },
        "sidecar_fallback": {"available": False, "payload_bytes": 0},
        "selected_next_operator": selected_next_operator,
        "next_required_surface": (
            "fakequant_archive_parseback_survival"
            if selected_next_operator == "backend_fit_live"
            else "archive_materialize_parseback_inflate"
        ),
        "promotion_eligible": False,
        "score_claim": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [] if blockers is None else list(blockers),
    }


def _wall_normal_branch_receipt(
    *,
    action_id: str = ACTION,
    first_failing_surface: str = SIDECAR_FALLBACK_ACCEPTED,
    same_action_id: bool = True,
    same_support_sha256: bool = True,
    support_required_count: int = 1,
    support_executable_count: int = 1,
    blockers: list[str] | None = None,
) -> dict:
    return {
        "schema": WALL_NORMAL_BRANCH_RECEIPT_SCHEMA,
        "fixture_not_real": True,
        "branch_count": 1,
        "action_ids": [action_id] if same_action_id else [action_id, f"{action_id}:other"],
        "same_action_id": same_action_id,
        "support_sha256s": ["9" * 64],
        "same_support_sha256": same_support_sha256,
        "support_required_count": support_required_count,
        "support_executable_count": support_executable_count,
        "first_failing_surface": first_failing_surface,
        "first_failing_action_id": action_id,
        "first_failing_action_kind": "sidecar_grammar",
        "branches": [],
        "blockers": [] if blockers is None else list(blockers),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hi_nerv_four_arm_action_effects(*, omit_arm: str | None = None) -> list[dict]:
    rows = [
        _hi_nerv_action_effect(arm="A", action_kind="birth_only"),
        _hi_nerv_action_effect(
            arm="B",
            action_kind="frame0_pose_target_only",
            new_d_seg=0.0010,
            new_d_pose=8.0e-5,
        ),
        _hi_nerv_action_effect(
            arm="C",
            action_kind="independent_birth_plus_frame0_pose",
            new_d_seg=0.00075,
            new_d_pose=8.0e-5,
        ),
        _hi_nerv_action_effect(
            arm="D",
            action_kind="joint_line_search_composite",
            new_d_seg=0.0007,
            new_d_pose=8.0e-5,
            raw_cap_decision="violated_counterfactual_only",
        ),
        _hi_nerv_action_effect(
            arm="E",
            action_kind="frame0_pose_then_birth_composite",
            new_d_seg=0.00072,
            new_d_pose=8.0e-5,
            raw_cap_decision="violated_counterfactual_only",
        ),
    ]
    if omit_arm is not None:
        rows = [row for row in rows if row.get("arm") != omit_arm]
    return rows


_LOWERING_TARGET_ACTION_KINDS = {
    "backend_realization": "birth_only_backend_realization",
    "pair_local_latent_action": "pair_local_latent_adapter",
    "frame0_pose_compensation": "frame0_pose_compensation",
    "frame1_seg_wall_crossing": "frame1_seg_wall_crossing",
    "byte_priced_sidecar": "byte_priced_sidecar_support_codec",
    "pose_compensated_composite": "joint_line_search_composite",
    "snerv_source_state_action": "snerv_lf_hf_mfu_hfr_tub_source_state",
    "semantic_pose_primitive": "semantic_pose_primitive",
    "byte_entropy_rewrite": "byte_entropy_rewrite_ans",
}


def _hi_nerv_lowering_effect(
    *,
    lowering_target: str,
    viable: bool = False,
) -> ActionEffect:
    return ActionEffect.build(
        action_id=ACTION,
        family="hinerv",
        action_kind=_LOWERING_TARGET_ACTION_KINDS[lowering_target],
        inverse_source="joint_seg_pose_projection",
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status="measured",
        authority="inflate_raw",
        normalization_scope="batch_local",
        producer="unit_test_lowering_race_fixture",
        consumer="nerv_long_run_launch_gate",
        pair_ids=[0],
        region_ids=["b0/c2/r1"],
        payload_sections=(
            f"lowering_target={lowering_target}",
            "support_codec=target_region_action_coordinates_v1",
            "action_payload_bytes=0",
            "metadata_bytes=0",
        ),
        old_d_seg=0.0010,
        new_d_seg=0.0007 if viable else 0.0010,
        old_d_pose=1.0e-4,
        new_d_pose=8.0e-5 if viable else 1.0e-4,
        old_bytes=178_258,
        new_bytes=178_258,
        receiver_surface={
            "uint8_changed_pixels": 4096 if viable else 0,
            "seg_argmax_changed_pixels": 2048 if viable else 0,
            "seg_wrong_to_target_count": 2048 if viable else 0,
            "seg_target_hard_lost_count": 0,
        },
        exact_score_decision="accept" if viable else "reject",
        parseback_survived=True,
        inflate_survived=True,
        fakequant_survived=True,
        hard_won_count=2048 if viable else 0,
        wrong_to_target=2048 if viable else 0,
        target_to_wrong=0,
        wrong_to_wrong=0,
        net_target_support_delta=2048 if viable else 0,
        uint8_changed_count_region=4096 if viable else 0,
        seg_input_delta_linf_region=1.0 / 255.0 if viable else 0.0,
        posenet_input_delta_linf_pair=1.0 / 255.0 if viable else 0.0,
        support_source="archive_executable_target_region_action_support",
        support_cardinality=2048,
        support_sha256="9" * 64,
        support_encoding="target_region_action_coordinates_v1",
        support_encoded_bytes=8192,
        support_research_only=False,
    )


def _hi_nerv_evaluator_action_lowering_race(
    *,
    omit_target: str | None = None,
    viable_target: str = "backend_realization",
) -> dict:
    return build_lowering_race_report(
        action_id=ACTION,
        action_effects=[
            _hi_nerv_lowering_effect(
                lowering_target=target,
                viable=target == viable_target,
            )
            for target in LOWERING_TARGETS
            if target != omit_target
        ],
        expected_support_sha256="9" * 64,
    )


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


def _source_qualified_readiness_report(*, mock: bool = False) -> dict:
    return {
        "schema": HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
        "fixture_not_real": True,
        "teacher_gate": {
            "real_segnet_teacher_requested": True,
            "direct_live_segnet_requested": True,
            "real_posenet_teacher_requested": False,
            "direct_live_posenet_requested": True,
            "mock_scorer_teacher_allowed": mock,
            "unscored_research_smoke_enabled": False,
        },
        "direct_live_segnet_gate": {
            "metrics": {"loss_part_segnet_direct_live_argmax_disagreement": 0.01}
        },
        "direct_live_posenet_gate": {
            "metrics": {"loss_part_pose_direct_live_score_term": 0.14}
        },
        "posenet_distill_gate": {"metrics": {}},
        "receiver_surface_identity_gate": {
            "archive_identity_present": True,
            "direct_receiver_parseback_present": True,
            "archive_sha256_mismatch": False,
            "candidate_cache_manifest_bound": True,
        },
    }


def _hi_nerv_lowering_race(
    *,
    action_id: str = ACTION,
    best_lowering: str = "pose_compensated_composite",
    first_failing_surface: str = "none",
    same_support: bool = True,
    delta_score_total: float | None = -0.01,
    authority: str = "inflate_raw",
    include_target_accounting: bool = True,
) -> dict:
    targets = [
        "backend_realization",
        "pair_local_latent_action",
        "frame0_pose_compensation",
        "frame1_seg_wall_crossing",
        "byte_priced_sidecar",
        "pose_compensated_composite",
        "snerv_source_state_action",
        "semantic_pose_primitive",
        "byte_entropy_rewrite",
    ]
    improved = (
        first_failing_surface == "none"
        and delta_score_total is not None
        and delta_score_total < 0.0
    )
    backend_complete = best_lowering == "backend_realization" and improved
    sidecar_complete = best_lowering == "byte_priced_sidecar" and improved
    row = {
        "schema": HI_NERV_TARGET_REGION_ACTION_LOWERING_RACE_SCHEMA,
        "fixture_not_real": True,
        "action_id": action_id,
        "support_sha256": "9" * 64,
        "direct_teacher_support_sha256": "9" * 64 if same_support else "8" * 64,
        "same_support_as_direct_teacher": same_support,
        "best_lowering": best_lowering,
        "first_failing_surface": first_failing_surface,
        "backend_realization_complete": backend_complete,
        "sidecar_lowering_complete": sidecar_complete,
        "verdict": {
            "schema": "tac.evaluator_action_lowering_verdict.v1",
            "action_id": action_id,
            "best_lowering": best_lowering,
            "first_failing_surface": first_failing_surface,
            "backend_realization_complete": backend_complete,
            "sidecar_lowering_complete": sidecar_complete,
            "authority": authority,
            "delta_score_nonrate": -0.02,
            "delta_score_total": delta_score_total,
            "delta_bytes": 128,
            "value_per_byte": 7.8125e-5,
        },
        "current_sidecar_candidate_id": "current_hiv1_target_region_action_brotli",
        "candidate_count": len(targets),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if include_target_accounting:
        row["target_accounting"] = {
            "schema": "tac.evaluator_action_lowering_race.target_accounting.v1",
            "expected_targets": targets,
            "present_targets": list(targets),
            "missing_targets": [],
            "all_targets_accounted": True,
        }
    return row


def _native_hi_nerv_lowering_race(
    *,
    action_id: str = ACTION,
    include_support_identity: bool = True,
    all_candidates_same_support: bool = True,
    support_failure: str | None = None,
    lowering_candidate_count: int = 1,
    best_lowering: str = "byte_priced_sidecar",
    first_failing_surface: str = "none",
    delta_score_total: float | None = -0.01,
    authority: str = "inflate_raw",
) -> dict:
    targets = [
        "backend_realization",
        "pair_local_latent_action",
        "frame0_pose_compensation",
        "frame1_seg_wall_crossing",
        "byte_priced_sidecar",
        "pose_compensated_composite",
        "snerv_source_state_action",
        "semantic_pose_primitive",
        "byte_entropy_rewrite",
    ]
    if lowering_candidate_count <= 0:
        candidate_targets: list[str] = []
    elif lowering_candidate_count >= len(targets):
        candidate_targets = [
            best_lowering,
            *[target for target in targets if target != best_lowering],
        ][:lowering_candidate_count]
    else:
        candidate_targets = [best_lowering] * lowering_candidate_count
    missing_targets = [target for target in targets if target not in set(candidate_targets)]
    improved = (
        first_failing_surface == "none"
        and delta_score_total is not None
        and delta_score_total < 0.0
    )
    backend_complete = best_lowering == "backend_realization" and improved
    sidecar_complete = best_lowering == "byte_priced_sidecar" and improved
    row = {
        "schema": EVALUATOR_ACTION_LOWERING_RACE_SCHEMA,
        "fixture_not_real": True,
        "action_id": action_id,
        "backend_realization_complete": backend_complete,
        "sidecar_lowering_complete": sidecar_complete,
        "lowering_candidates": [
            {
                "schema": "tac.evaluator_action_lowering_candidate.v1",
                "action_id": action_id,
                "lowering_target": target,
                "authority": authority,
                "support_sha256": "9" * 64,
                "support_encoding": "target_region_action_coordinates_v1",
                "support_encoded_bytes": 8192,
                "action_payload_bytes": 0,
                "metadata_bytes": 0,
                "viable": target == best_lowering,
                "first_failing_surface": (
                    "none" if target == best_lowering else "fixture_not_selected"
                ),
            }
            for target in candidate_targets
        ],
        "target_accounting": {
            "schema": "tac.evaluator_action_lowering_race.target_accounting.v1",
            "expected_targets": targets,
            "present_targets": candidate_targets,
            "missing_targets": missing_targets,
            "all_targets_accounted": not missing_targets,
        },
        "verdict": {
            "schema": "tac.evaluator_action_lowering_verdict.v1",
            "action_id": action_id,
            "best_lowering": best_lowering,
            "first_failing_surface": first_failing_surface,
            "backend_realization_complete": backend_complete,
            "sidecar_lowering_complete": sidecar_complete,
            "authority": authority,
            "delta_score_nonrate": -0.02,
            "delta_score_total": delta_score_total,
            "delta_bytes": 128,
            "value_per_byte": 7.8125e-5,
        },
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if include_support_identity:
        row["support_identity"] = {
            "schema": "tac.evaluator_action_lowering_race.support_identity.v1",
            "expected_support_sha256": "9" * 64,
            "support_sha256s": ["9" * 64],
            "missing_support_sha256_count": 0,
            "all_candidates_same_support": all_candidates_same_support,
            "failure": support_failure,
            "blockers": [] if support_failure is None else [support_failure],
        }
    return row


def _full_hi_nerv_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(root / "fakequant.json", _survival("fakequant_mlx"))
    _write(root / "parseback.json", _survival("parseback_mlx"))
    _write(root / "inflate.json", _survival("inflated_torch_cpu"))
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(),
                *_hi_nerv_four_arm_action_effects(),
            ],
        },
    )
    _write(root / "wall_normal_lift.json", _wall_normal_lift())
    _write(root / "wall_normal_branch_receipt.json", _wall_normal_branch_receipt())
    _write(root / "parseback_contract.json", _parseback_selection_contract())
    _write(root / "source_metrics.json", _source_qualified_metrics())
    _write(
        root / "lowering_race.json",
        _hi_nerv_lowering_race(best_lowering="backend_realization"),
    )
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
        build_representative_coverage_row(
            [
                {"region": {"class_index": 1, "region_pixel_count": 9}, "outcome": OUTCOME_BIRTH_ACCEPTED},
                {"region": {"class_index": 3, "region_pixel_count": 256}, "outcome": OUTCOME_BIRTH_ACCEPTED},
                {"region": {"class_index": 2, "region_pixel_count": 5000}, "outcome": OUTCOME_BIRTH_ACCEPTED},
            ]
        ),
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


def test_positive_masked_oracle_is_classified_as_archive_unclosed_birth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_masked_residual_archive_grammar_missing",
                "hinerv_target_region_masked_residual_parseback_missing",
                "hinerv_target_region_masked_residual_value_per_byte_missing",
            ],
        }
    }
    _write(root / "birth.json", row)

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["highest_level"] == "none"
    assert "real_video_birth_receipt_missing" not in verdict["blocking_evidence"]
    assert "real_video_birth_receipt_archive_unclosed" in verdict["blocking_evidence"]
    assert (
        "hinerv_target_region_masked_residual_archive_grammar_missing"
        in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_masked_residual_value_per_byte_missing"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_charged_target_region_archive_evidence_retires_materialization_blockers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "target_region_action_section_telemetry": {"support_sha256": "a" * 64},
            "blockers": [
                "hinerv_target_region_action_archive_meta_not_materialized",
                "hinerv_target_region_action_parseback_survival_missing",
                "hinerv_target_region_action_inflate_survival_missing",
                "hinerv_target_region_action_archive_zip_byte_delta_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "archive_telemetry.json",
        {
            "hi_nerv_archive_codec_custody": {
                "archive_section_telemetry": {
                    "target_region_actions": _charged_target_region_actions()
                }
            },
            "receiver_replay_archive_selection": {
                "selected_archive_path": "/ssd/archive.zip",
                "selected_archive_bytes": 439003,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "real_video_birth_receipt_archive_unclosed" in verdict["blocking_evidence"]
    assert (
        "hinerv_target_region_action_archive_meta_not_materialized"
        not in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_action_archive_zip_byte_delta_missing"
        not in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_action_parseback_survival_missing"
        in verdict["blocking_evidence"]
    )
    assert "hinerv_target_region_action_inflate_survival_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_target_region_archive_evidence_rejects_stale_tile_support_encoding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_action_archive_meta_not_materialized",
                "hinerv_target_region_action_archive_zip_byte_delta_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "archive_telemetry.json",
        {
            "hi_nerv_archive_codec_custody": {
                "archive_section_telemetry": {
                    "target_region_actions": _charged_target_region_actions(
                        payload_codec="tile_brotli_v1",
                        support_encoding="explicit_yx_u16_coordinates",
                        support_encoded_bytes=139_908,
                    )
                }
            },
            "receiver_replay_archive_selection": {
                "selected_archive_path": "/ssd/archive.zip",
                "selected_archive_bytes": 439003,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert (
        "hinerv_target_region_action_archive_meta_not_materialized"
        in verdict["blocking_evidence"]
    )
    assert "target_region_action_support_encoding_mismatch" in verdict[
        "blocking_evidence"
    ]
    assert (
        "hinerv_target_region_action_archive_zip_byte_delta_missing"
        not in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_target_region_archive_evidence_uses_valid_receiver_tile_over_stale_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_action_archive_meta_not_materialized",
                "hinerv_target_region_action_archive_zip_byte_delta_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "stale_archive_telemetry.json",
        {
            "hi_nerv_archive_codec_custody": {
                "archive_section_telemetry": {
                    "target_region_actions": _charged_target_region_actions(
                        payload_codec="brotli_wrapped_v1",
                        support_encoding="explicit_yx_u16_coordinates",
                        support_encoded_bytes=139_908,
                    )
                }
            },
            "receiver_replay_archive_selection": {
                "selected_archive_path": "/ssd/stale.zip",
                "selected_archive_bytes": 439003,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write(
        root / "tile_receiver_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "parseback_mlx",
            "action_id": ACTION,
            "archive_path": "/ssd/tile.zip",
            "archive_bytes": 319575,
            "survived": True,
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": True,
            "target_region_actions": _charged_target_region_actions(
                payload_codec="tile_brotli_v1",
                support_encoding="brotli_tile_bitmap_little_endian",
                support_encoded_bytes=3413,
            )
            | {
                "charged_meta_json_bytes": None,
                "payload_bytes": 94633,
                "base64_text_bytes": 126180,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "real_video_birth_receipt_archive_unclosed" in verdict["blocking_evidence"]
    assert (
        "hinerv_target_region_action_archive_meta_not_materialized"
        not in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_action_archive_zip_byte_delta_missing"
        not in verdict["blocking_evidence"]
    )
    assert "target_region_action_support_encoding_mismatch" not in verdict[
        "blocking_evidence"
    ]
    assert "target_region_action_encoded_program_sha256_missing" not in verdict[
        "blocking_evidence"
    ]
    assert "target_region_action_decoded_support_sha256_missing" not in verdict[
        "blocking_evidence"
    ]
    assert "target_region_action_decoded_action_sha256_missing" not in verdict[
        "blocking_evidence"
    ]
    assert verdict["approved"] is False


def test_target_region_action_parseback_survival_retires_parseback_blocker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_action_archive_meta_not_materialized",
                "hinerv_target_region_action_parseback_survival_missing",
                "hinerv_target_region_action_inflate_survival_missing",
                "hinerv_target_region_action_archive_zip_byte_delta_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "archive_telemetry.json",
        {
            "hi_nerv_archive_codec_custody": {
                "archive_section_telemetry": {
                    "target_region_actions": _charged_target_region_actions()
                }
            },
            "receiver_replay_archive_selection": {
                "selected_archive_path": "/ssd/archive.zip",
                "selected_archive_bytes": 439003,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write(
        root / "target_region_action_parseback_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "parseback_mlx",
            "action_id": ACTION,
            "survived": True,
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": False,
            "total_action_pixels": 32,
            "exact_uint8_action_pixels_applied": 32,
            "receiver_changed_action_pixels": 32,
            "target_region_actions": {"support_sha256": "9" * 64},
            "blockers": ["target_region_action_inflate_survival_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert (
        "hinerv_target_region_action_archive_meta_not_materialized"
        not in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_action_archive_zip_byte_delta_missing"
        not in verdict["blocking_evidence"]
    )
    assert (
        "hinerv_target_region_action_parseback_survival_missing"
        not in verdict["blocking_evidence"]
    )
    assert "hinerv_target_region_action_inflate_survival_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_target_region_action_parseback_and_inflate_must_be_same_row(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_action_parseback_survival_missing",
                "hinerv_target_region_action_inflate_survival_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "target_region_action_parseback_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "parseback_mlx",
            "action_id": ACTION,
            "survived": True,
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": False,
            "total_action_pixels": 32,
            "exact_uint8_action_pixels_applied": 32,
            "receiver_changed_action_pixels": 32,
            "target_region_actions": {"support_sha256": "9" * 64},
            "blockers": ["target_region_action_inflate_survival_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write(
        root / "target_region_action_inflate_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "inflated_torch_cpu",
            "action_id": ACTION,
            "survived": False,
            "fakequant_survived": False,
            "parseback_survived": False,
            "inflate_survived": True,
            "target_region_actions": {"support_sha256": "9" * 64},
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "hinerv_target_region_action_parseback_survival_missing" not in (
        verdict["blocking_evidence"]
    )
    assert "hinerv_target_region_action_inflate_survival_missing" not in (
        verdict["blocking_evidence"]
    )
    assert "target_region_action_parseback_inflate_same_row_survival_missing" in (
        verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_target_region_action_parseback_survival_requires_same_action_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "blockers": [
                "hinerv_target_region_action_parseback_survival_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "target_region_action_parseback_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "parseback_mlx",
            "action_id": "b" * 64,
            "survived": True,
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": False,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "hinerv_target_region_action_parseback_survival_missing" in (
        verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_target_region_action_parseback_survival_requires_same_support(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt()
    row["accepted_step_count"] = 0
    row["candidate_frontier_telemetry"] = {
        "masked_residual_oracle": {
            "schema": "hi_nerv_target_region_masked_residual_oracle.v1",
            "authority": "receiver_surface_oracle_false_authority",
            "archive_closed": False,
            "promotion_blocked": True,
            "exact_accepted_before_archive_closure": True,
            "target_support_moved": True,
            "target_region_action_section_telemetry": {"support_sha256": "a" * 64},
            "blockers": [
                "hinerv_target_region_action_parseback_survival_missing",
            ],
        }
    }
    _write(root / "birth.json", row)
    _write(
        root / "target_region_action_parseback_survival.json",
        {
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "surface": "parseback_mlx",
            "action_id": ACTION,
            "survived": True,
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": False,
            "target_region_actions": {"support_sha256": "b" * 64},
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "hinerv_target_region_action_parseback_survival_missing" in (
        verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


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


def test_pair_local_composite_action_effect_can_supply_pose_trust(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(root / "birth.json", _live_birth_receipt(pose_trusted=False))

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "pose_trusted_birth_receipt_missing" not in verdict["blocking_evidence"]
    assert verdict["highest_level"] == "L5"
    assert verdict["approved"] is True


def test_hinerv_gate_requires_evaluator_action_lowering_race(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    (root / "lowering_race.json").unlink()

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_missing" in verdict["blocking_evidence"]
    assert f"evaluator_action_lowering_race_missing:{ACTION}" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_failed_evaluator_action_lowering_race(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "lowering_race.json",
        _hi_nerv_lowering_race(
            same_support=False,
            best_lowering="none",
            first_failing_surface="support_identity_mismatch",
            delta_score_total=None,
            authority="none",
        ),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert (
        f"evaluator_action_lowering_race_support_mismatch:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_native_lowering_race_without_support_identity(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "lowering_race.json",
        _native_hi_nerv_lowering_race(include_support_identity=False),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert (
        f"evaluator_action_lowering_race_support_identity_missing:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_native_lowering_race_without_candidate_rows(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "lowering_race.json",
        _native_hi_nerv_lowering_race(lowering_candidate_count=0),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert (
        f"evaluator_action_lowering_race_candidate_count_not_positive:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_native_lowering_race_without_all_targets(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "lowering_race.json",
        _native_hi_nerv_lowering_race(lowering_candidate_count=1),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert any(
        blocker.startswith(f"evaluator_action_lowering_race_targets_missing:{ACTION}:")
        for blocker in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_stale_lowering_race_target_accounting(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    row = _native_hi_nerv_lowering_race(lowering_candidate_count=9)
    row["target_accounting"]["expected_targets"] = [
        "backend_realization",
        "byte_priced_sidecar",
        "pose_compensated_composite",
        "semantic_pose_primitive",
    ]
    row["target_accounting"]["missing_targets"] = []
    row["target_accounting"]["all_targets_accounted"] = True
    _write(root / "lowering_race.json", row)

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert any(
        "pair_local_latent_action" in blocker
        and blocker.startswith(f"evaluator_action_lowering_race_targets_missing:{ACTION}:")
        for blocker in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_accepts_real_evaluator_action_lowering_race(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    report = _hi_nerv_evaluator_action_lowering_race()
    _write(root / "lowering_race.json", report)

    assert report["schema"] == EVALUATOR_ACTION_LOWERING_RACE_SCHEMA
    assert report["support_identity"]["all_candidates_same_support"] is True
    assert report["support_identity"]["support_sha256s"] == ["9" * 64]
    assert {row["action_id"] for row in report["lowering_candidates"]} == {ACTION}
    assert report["target_accounting"]["all_targets_accounted"] is True
    assert set(report["target_accounting"]["present_targets"]) == set(LOWERING_TARGETS)
    assert sum(row["viable"] is True for row in report["lowering_candidates"]) == 1
    assert report["backend_realization_complete"] is True
    assert report["sidecar_lowering_complete"] is False
    assert report["verdict"]["delta_score_total"] < 0.0

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is True
    assert "evaluator_action_lowering_race_not_accepted" not in verdict[
        "blocking_evidence"
    ]
    assert not any(
        blocker.startswith("evaluator_action_lowering_race")
        for blocker in verdict["blocking_evidence"]
    )


def test_hinerv_gate_accepts_sidecar_lowering_without_backend_realization(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    report = _hi_nerv_evaluator_action_lowering_race(
        viable_target="byte_priced_sidecar"
    )
    _write(root / "lowering_race.json", report)

    assert report["sidecar_lowering_complete"] is True
    assert report["backend_realization_complete"] is False

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" not in verdict[
        "blocking_evidence"
    ]
    assert (
        f"evaluator_action_lowering_race_backend_realization_incomplete:{ACTION}"
        not in verdict["blocking_evidence"]
    )
    assert not any(
        blocker.startswith("evaluator_action_lowering_race")
        for blocker in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is True


def test_hinerv_gate_requires_sidecar_completion_for_sidecar_lowering(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    report = _hi_nerv_evaluator_action_lowering_race(
        viable_target="byte_priced_sidecar"
    )
    report["sidecar_lowering_complete"] = False
    report["verdict"]["sidecar_lowering_complete"] = False
    _write(root / "lowering_race.json", report)

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict[
        "blocking_evidence"
    ]
    assert (
        f"evaluator_action_lowering_race_sidecar_lowering_incomplete:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert (
        f"evaluator_action_lowering_race_backend_realization_incomplete:{ACTION}"
        not in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_blocks_real_lowering_race_missing_one_target(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    missing_target = "byte_entropy_rewrite"
    report = _hi_nerv_evaluator_action_lowering_race(omit_target=missing_target)
    _write(root / "lowering_race.json", report)

    assert report["target_accounting"]["missing_targets"] == [missing_target]

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict[
        "blocking_evidence"
    ]
    assert (
        f"evaluator_action_lowering_race_targets_missing:{ACTION}:{missing_target}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_names_lowering_race_parseback_blocker(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    row = _native_hi_nerv_lowering_race(
        lowering_candidate_count=10,
        best_lowering="discard",
        first_failing_surface="PARSEBACK_FAILED",
        delta_score_total=None,
        authority="none",
    )
    row["lowering_candidates"][0]["viable"] = False
    row["lowering_candidates"][0]["first_failing_surface"] = "PARSEBACK_FAILED"
    _write(root / "lowering_race.json", row)

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert (
        f"evaluator_action_lowering_race_parseback_missing:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_wrapper_lowering_race_without_target_accounting(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "lowering_race.json",
        _hi_nerv_lowering_race(include_target_accounting=False),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "evaluator_action_lowering_race_not_accepted" in verdict["blocking_evidence"]
    assert (
        f"evaluator_action_lowering_race_target_accounting_missing:{ACTION}"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_hinerv_gate_rejects_same_action_survival_with_different_support(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(root / "fakequant.json", _survival("fakequant_mlx", support_sha256="8" * 64))
    _write(root / "parseback.json", _survival("parseback_mlx", support_sha256="8" * 64))
    _write(root / "inflate.json", _survival("inflated_torch_cpu", support_sha256="8" * 64))

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    blocking = verdict["blocking_evidence"]
    assert verdict["approved"] is False
    assert "birth_survival_support_sha256_mismatch:fakequant_mlx" in blocking
    assert "birth_survival_support_sha256_mismatch:parseback_mlx" in blocking
    assert "birth_survival_support_sha256_mismatch:inflated_torch_cpu" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:parseback_mlx" in blocking
    assert "birth_survival_receipt_missing:inflated_torch_cpu" in blocking


def test_pair_local_composite_without_pose_improvement_does_not_supply_pose_trust(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(root / "birth.json", _live_birth_receipt(pose_trusted=False))
    four_arm_rows = [
        (
            _hi_nerv_action_effect(
                arm="D",
                action_kind="joint_line_search_composite",
                new_d_pose=1.1e-4,
            )
            if row.get("arm") == "D"
            else row
        )
        for row in _hi_nerv_four_arm_action_effects()
    ]
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(),
                *four_arm_rows,
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "pose_trusted_birth_receipt_missing" in verdict["blocking_evidence"]
    assert "pose_trust_pair_local_servo_pose_not_improved" in verdict["blocking_evidence"]
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
    assert "real_video_birth_receipt_not_accepted" in verdict["blocking_evidence"]
    assert "real_video_birth_receipt_missing" not in verdict["blocking_evidence"]
    assert "live_birth_target_support_not_positive" in verdict["blocking_evidence"]


def test_rejected_live_birth_receipt_carries_first_failed_surface(tmp_path: Path) -> None:
    root = tmp_path / "run"
    row = _live_birth_receipt(hard_won=0, net_support=0)
    row["accepted_step_count"] = 0
    row["blockers"] = ["hinerv_target_region_birth_no_accepted_step"]
    row["birth_rejection_breakdown"] = {
        "schema": "hi_nerv_target_region_birth_rejection_breakdown.v1",
        "state": "rejected",
        "first_failed_surface": "segnet_argmax_margin_crossing",
        "causes": {
            "receiver_pixels_moved_without_argmax_birth": True,
            "pose_trust_failed": False,
        },
    }
    _write(root / "birth.json", row)

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    blockers = verdict["blocking_evidence"]
    assert "real_video_birth_receipt_not_accepted" in blockers
    assert "real_video_birth_receipt_missing" not in blockers
    assert "live_birth_rejection_state:rejected" in blockers
    assert (
        "live_birth_rejection_first_failed_surface:segnet_argmax_margin_crossing"
        in blockers
    )
    assert (
        "live_birth_rejection_cause:receiver_pixels_moved_without_argmax_birth"
        in blockers
    )


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


def test_hinerv_gate_requires_wall_normal_lift_receipt(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    (root / "wall_normal_lift.json").unlink()

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_missing" in verdict["blocking_evidence"]
    assert f"target_region_wall_normal_lift_missing:{ACTION}" in verdict["blocking_evidence"]


def test_hinerv_gate_blocks_nonrealized_wall_normal_lift(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_lift.json",
        _wall_normal_lift(
            selected_next_operator="byte_priced_action_fallback",
            backend_realized=False,
            backend_wrong_to_target=0,
            exact_score_decision="reject",
            blockers=[
                "target_region_wall_normal_backend_not_realized",
                "target_region_wall_normal_sidecar_archive_unclosed",
            ],
        ),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    blocking = verdict["blocking_evidence"]
    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_not_backend_realized" in blocking
    assert (
        f"target_region_wall_normal_lift_selected_next_operator:{ACTION}:byte_priced_action_fallback"
        in blocking
    )
    assert f"target_region_wall_normal_lift_backend_not_realized:{ACTION}" in blocking
    assert (
        f"target_region_wall_normal_lift_wrong_to_target_missing_or_nonpositive:{ACTION}"
        in blocking
    )
    assert (
        "target_region_wall_normal_lift_blocker:target_region_wall_normal_backend_not_realized"
        in blocking
    )


def test_hinerv_gate_requires_wall_normal_branch_receipt(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    (root / "wall_normal_branch_receipt.json").unlink()

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_branch_receipt_missing" in verdict["blocking_evidence"]
    assert (
        f"target_region_wall_normal_branch_receipt_missing:{ACTION}"
        in verdict["blocking_evidence"]
    )


def test_hinerv_gate_blocks_failed_wall_normal_branch_receipt(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_branch_receipt.json",
        _wall_normal_branch_receipt(
            first_failing_surface="BACKEND_REALIZATION_FAILED",
            same_action_id=False,
            same_support_sha256=False,
            blockers=["wall_normal_branch_action_id_mismatch"],
        ),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    blocking = verdict["blocking_evidence"]
    assert verdict["approved"] is False
    assert "target_region_wall_normal_branch_receipt_not_accepted" in blocking
    assert (
        f"target_region_wall_normal_branch_receipt_action_id_mismatch:{ACTION}"
        in blocking
    )
    assert (
        f"target_region_wall_normal_branch_receipt_support_mismatch:{ACTION}"
        in blocking
    )
    assert (
        f"target_region_wall_normal_branch_receipt_failed_surface:{ACTION}:BACKEND_REALIZATION_FAILED"
        in blocking
    )
    assert (
        "target_region_wall_normal_branch_receipt_blocker:wall_normal_branch_action_id_mismatch"
        in blocking
    )


def test_hinerv_gate_rejects_wall_normal_without_direct_exact_score(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_lift.json",
        _wall_normal_lift(direct_exact_score_decision="reject"),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_not_backend_realized" in (
        verdict["blocking_evidence"]
    )
    assert (
        f"target_region_wall_normal_lift_direct_teacher_exact_score_not_accepted:{ACTION}"
        in verdict["blocking_evidence"]
    )


def test_hinerv_gate_rejects_wall_normal_direct_accept_without_score_improvement(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_lift.json",
        _wall_normal_lift(direct_exact_delta_score_nonrate=0.0),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_not_backend_realized" in (
        verdict["blocking_evidence"]
    )
    assert (
        f"target_region_wall_normal_lift_direct_teacher_nonnegative_delta:{ACTION}"
        in verdict["blocking_evidence"]
    )


def test_hinerv_gate_rejects_wall_normal_backend_action_id_mismatch(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_lift.json",
        _wall_normal_lift(backend_action_effect_action_id=f"{ACTION}:other"),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_not_backend_realized" in (
        verdict["blocking_evidence"]
    )
    assert (
        f"target_region_wall_normal_lift_backend_action_id_mismatch:{ACTION}:{ACTION}:other"
        in verdict["blocking_evidence"]
    )


def test_hinerv_gate_rejects_masked_residual_as_wall_normal_teacher(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "wall_normal_lift.json",
        _wall_normal_lift(direct_teacher_is_true_wall_normal=False),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "target_region_wall_normal_lift_not_backend_realized" in (
        verdict["blocking_evidence"]
    )
    assert (
        f"target_region_wall_normal_lift_direct_teacher_not_true_wall_normal:{ACTION}"
        in verdict["blocking_evidence"]
    )


def test_legacy_readiness_report_can_supply_source_qualified_metrics(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    (root / "source_metrics.json").unlink()
    _write(root / "readiness.json", _source_qualified_readiness_report())

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "source_qualified_metrics_missing" not in verdict["blocking_evidence"]
    assert verdict["approved"] is True


def test_legacy_readiness_report_with_mock_teacher_does_not_supply_source_metrics(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    (root / "source_metrics.json").unlink()
    _write(root / "readiness.json", _source_qualified_readiness_report(mock=True))

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "source_qualified_metrics_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_wrong_to_wrong_churn_is_not_spill_when_exact_score_improves(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(wrong_to_wrong=1406),
                *[
                    dict(row, wrong_to_wrong=1406)
                    for row in _hi_nerv_four_arm_action_effects()
                ],
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "action_effect_v1_spill_positive:wrong_to_wrong" not in verdict["blocking_evidence"]
    assert verdict["approved"] is True


def test_target_to_wrong_still_blocks_hinerv_action_effect(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(target_to_wrong=1),
                *_hi_nerv_four_arm_action_effects(),
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "action_effect_v1_spill_positive:target_to_wrong" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_hinerv_action_effect_survival_must_be_same_support(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(support_sha256="9" * 64),
                _hi_nerv_action_effect(support_sha256="8" * 64),
                *_hi_nerv_four_arm_action_effects(),
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert (
        "action_effect_same_action_same_support_sha256_mismatch"
        in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_missing_hinerv_four_arm_action_effect_blocks_ladder(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(),
                *_hi_nerv_four_arm_action_effects(omit_arm="D"),
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is False
    assert "action_effect_four_arm_missing:D" in verdict["blocking_evidence"]


def test_hinerv_reverse_order_action_effect_is_optional_for_four_arm_gate(
    tmp_path: Path,
) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "action_effect.json",
        {
            "rows": [
                _hi_nerv_action_effect(),
                *_hi_nerv_four_arm_action_effects(omit_arm="E"),
            ],
        },
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert verdict["approved"] is True
    assert "action_effect_four_arm_missing:E" not in verdict["blocking_evidence"]


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


def test_scalar_only_representative_coverage_blocks_l5(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    _write(
        root / "coverage.json",
        {
            "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
            "fixture_not_real": True,
            "passed": True,
            "region_classes_covered": 3,
            "distinct_classes_accepted": 2,
            "distinct_size_classes_accepted": 3,
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
    blocking = verdict["blocking_evidence"]
    assert "representative_region_coverage_outcomes_missing" in blocking
    assert "representative_region_coverage_accepted_buckets_missing" in blocking
    assert "representative_region_coverage_all_buckets_missing" in blocking


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


def test_snerv_source_forward_rejects_nonidentical_output2_basis(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(
            output2_verdict=DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        ),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "snerv_full_source_forward_parity_missing" in verdict["blocking_evidence"]
    assert (
        "snerv_native_export_output2_source_identical_missing"
        in verdict["blocking_evidence"]
    )
    assert (
        "snerv_native_export_output2_boundary_not_source_identical:"
        "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS"
        in verdict["blocking_evidence"]
    )
    assert (
        "snerv_native_export_output2_fusion_not_payload_bound"
        in verdict["blocking_evidence"]
    )
    assert (
        "snerv_native_export_output2_not_consumed_by_receiver"
        in verdict["blocking_evidence"]
    )
    assert (
        "snerv_native_export_output2_frame_shape_mismatch"
        in verdict["blocking_evidence"]
    )
    assert any(
        item.endswith(
            "snerv_output2_boundary_not_source_identical:"
            "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS"
        )
        for item in verdict["blocking_evidence"]
    )
    assert any(
        item.endswith("snerv_source_forward_launch_gate_clearable_false")
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_snerv_native_export_requires_output2_boundary_row(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    row = _snerv_source_forward_action_row()
    row.pop("output2_boundary_verdict")
    row["launch_gate_clearable"] = False
    row["passed"] = False
    row["source_forward_replay_authority"] = False
    _write(root / "proof.json", row)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert "snerv_full_source_forward_parity_missing" in verdict["blocking_evidence"]
    assert (
        "snerv_native_export_output2_boundary_missing"
        in verdict["blocking_evidence"]
    )
    assert any(
        item.endswith("snerv_output2_boundary_verdict_missing")
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_snerv_source_forward_requires_per_surface_scorer_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(include_scorer_by_surface=False),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert any(
        item.endswith("snerv_source_forward_scorer_by_surface_missing")
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_snerv_source_forward_scorer_surface_delta_blocks_launch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(parseback_d_pose=0.25),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert any(
        item.endswith(
            "snerv_source_forward_scorer_surface_delta_exceeds_tolerance:archive_parseback:d_pose"
        )
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_snerv_source_forward_requires_surface_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(include_surface_provenance=False),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert any(
        item.endswith(
            "snerv_source_forward_surface_provenance_surface_missing:archive_parseback"
        )
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False


def test_snerv_source_forward_rejects_synthetic_surface_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        _snerv_source_forward_action_row(
            provenance_authority="synthetic_fixture_capture"
        ),
    )

    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )

    assert any(
        item.endswith(
            "snerv_source_forward_surface_provenance_authority_not_real:official_torch:tensor_capture_authority"
        )
        for item in verdict["blocking_evidence"]
    )
    assert verdict["approved"] is False
