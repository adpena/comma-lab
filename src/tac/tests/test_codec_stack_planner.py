# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.codec_stack_planner import (
    ABSOLUTE_MAX_PASSES,
    BALLE_HYPERPRIOR_REPAIR_CRITERIA,
    CANONICAL_QAT_PASSES,
    DEFAULT_STATIC_PMF_DELTA_BYTES,
    HSTACK_VSTACK_AXIS_SEMANTICS,
    MODEL_WEIGHT_BLOCK_FP_STREAMS,
    NESTED_OPTIMIZATION_LEVELS,
    PREDICTED_NESTED_SCORE_BAND,
    QUALITY_MANDATE_11,
    RENDERER_WEIGHT_BLOCK_FP_STREAMS,
    STACK_PLANNER_SCHEMA,
    ByteEvidenceSemantics,
    CodecComponent,
    CodecStackPlan,
    FailClosedPromotionPolicy,
    build_hstack_vstack_multipass_plan,
    canonical_json_sha256,
    plan_from_manifest,
    summarize_plan,
)


def _component(plan: CodecStackPlan, component_id: str) -> CodecComponent:
    for component in plan.components:
        if component.component_id == component_id:
            return component
    raise AssertionError(f"missing component {component_id}")


def test_default_plan_is_deterministic_and_scoreless() -> None:
    plan = build_hstack_vstack_multipass_plan(anchor_id="pr101_frontier")
    repeat = build_hstack_vstack_multipass_plan(anchor_id="pr101_frontier")

    manifest = plan.to_manifest()
    repeat_manifest = repeat.to_manifest()

    assert manifest == repeat_manifest
    assert manifest["schema"] == STACK_PLANNER_SCHEMA
    assert manifest["metadata"]["score_claim"] is False
    assert manifest["metadata"]["dispatch_attempted"] is False
    assert manifest["metadata"]["axis_semantics"] == HSTACK_VSTACK_AXIS_SEMANTICS
    assert "archive-component" not in manifest["metadata"]["axis_semantics"]["hstack_parallel"]
    assert "parser-proven logical stream" in manifest["metadata"]["axis_semantics"]["hstack_parallel"]
    assert manifest["metadata"]["nested_optimization"]["levels"] == list(NESTED_OPTIMIZATION_LEVELS)
    assert manifest["metadata"]["nested_optimization"]["score_band_prediction"] == PREDICTED_NESTED_SCORE_BAND
    assert manifest["metadata"]["nested_optimization"]["score_band_prediction"]["score_claim"] is False
    score_band = manifest["metadata"]["nested_optimization"]["score_band_prediction"]
    assert score_band["claim_count"] == 8
    assert len(score_band["claims"]) == 8
    for row in score_band["claims"]:
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["requires_exact_1to1_anchor"] is True
    assert manifest["metadata"]["canonical_qat_pipeline"]["passes"] == list(CANONICAL_QAT_PASSES)
    assert manifest["metadata"]["quality_mandate"] == list(QUALITY_MANDATE_11)
    assert (
        manifest["metadata"]["archive_layout"]["shape"]
        == "single_member_monolithic_packet_with_internal_parser_proven_logical_sections"
    )
    assert manifest["metadata"]["archive_layout"]["member_level_component_budgets_valid"] is False
    assert manifest["metadata"]["archive_layout"]["logical_stream_budget_requires_internal_parser_proof"] is True
    assert "no_member_level_mask_pose_budget_claim" in manifest["metadata"]["archive_layout"]["dispatch_requires"]
    assert manifest["promotion_status"]["score_claim"] is False
    assert manifest["promotion_status"]["dispatchable"] is False
    assert manifest["promotion_status"]["promotion_eligible"] is False
    assert "planning_artifact_only" in manifest["promotion_status"]["blockers"]
    assert manifest["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def test_balle_hyperprior_family_records_required_blockers_and_repairs() -> None:
    plan = build_hstack_vstack_multipass_plan(learned_model_overhead_bytes=2048)
    hyperprior = _component(plan, "balle_full_learned_hyperprior")

    assert hyperprior.family == "balle_full_learned_hyperprior"
    assert hyperprior.role == "hyperprior"
    assert hyperprior.stack_axis == "vstack_serial"
    assert hyperprior.exact_roundtrip_required is True
    assert hyperprior.runtime_packet_required is True
    assert hyperprior.byte_semantics.score_claim is False
    assert hyperprior.byte_semantics.model_overhead_bytes == 2048
    assert hyperprior.byte_semantics.byte_delta_vs_baseline == DEFAULT_STATIC_PMF_DELTA_BYTES
    assert "learned_model_overhead_not_amortized" in hyperprior.all_blockers
    assert "exact_reconstruction_roundtrip_missing" in hyperprior.all_blockers
    assert "runtime_packet_consumption_missing" in hyperprior.all_blockers
    assert f"static_shared_pmf_k12_negative_plus_{DEFAULT_STATIC_PMF_DELTA_BYTES}_bytes_requires_repair" in (
        hyperprior.all_blockers
    )
    for criterion in BALLE_HYPERPRIOR_REPAIR_CRITERIA:
        assert criterion in hyperprior.all_repair_criteria


def test_block_fp_weight_streams_always_pair_qint_with_exponents() -> None:
    plan = build_hstack_vstack_multipass_plan()
    required_pairs = {
        "renderer_weights_qint": "renderer_weight_exponents",
        "model_weight_qint": "model_weight_exponents",
    }

    assert RENDERER_WEIGHT_BLOCK_FP_STREAMS == ("renderer_weights_qint", "renderer_weight_exponents")
    assert MODEL_WEIGHT_BLOCK_FP_STREAMS == ("model_weight_qint", "model_weight_exponents")
    for component in plan.components:
        streams = set(component.streams)
        for qint_stream, exp_stream in required_pairs.items():
            if qint_stream in streams:
                assert exp_stream in streams, component.component_id


def test_plan_models_hstack_vstack_and_multipass_surfaces() -> None:
    plan = build_hstack_vstack_multipass_plan(max_passes=4)
    manifest = plan.to_manifest()

    assert [item["transform_id"] for item in manifest["serial_transforms"]] == [
        "vstack_deconstruct_to_qint_streams",
        "vstack_learned_hyperprior_repair",
        "vstack_arithmetic_terminal_encode",
        "vstack_packet_materialize",
    ]
    assert manifest["parallel_groups"] == [
        {
            "group_id": "hstack_parallel_stream_components",
            "merge_policy": "independent_streams_then_deterministic_packet_merge",
            "component_ids": [
                "mask_stream_component",
                "pose_stream_component",
                "residual_stream_component",
                "arithmetic_terminal_coder",
            ],
            "byte_semantics": manifest["parallel_groups"][0]["byte_semantics"],
            "blockers": [
                "single_member_internal_section_map_missing",
                "parallel_stream_byte_ledger_missing",
                "cross_stream_budget_reconciliation_missing",
            ],
            "repair_criteria": ["record_per_stream_internal_offsets_lengths_sha256_and_merge_order_sha256"],
        }
    ]
    assert [item["pass_id"] for item in manifest["passes"]] == [
        "pass_00_anchor_deconstruction",
        "pass_01_vstack_hyperprior_repair",
        "pass_02_hstack_packet_closure",
        "pass_03_multipass_refinement",
    ]
    assert manifest["passes"][1]["vstack_transform_ids"] == [
        "vstack_learned_hyperprior_repair",
        "vstack_arithmetic_terminal_encode",
    ]
    assert manifest["passes"][2]["hstack_group_ids"] == ["hstack_parallel_stream_components"]


def test_stack_axis_semantics_match_project_definition() -> None:
    plan = build_hstack_vstack_multipass_plan(max_passes=3)
    manifest = plan.to_manifest()

    serial_axes = {
        item["component_id"]: item["stack_axis"]
        for item in manifest["components"]
        if item["component_id"]
        in {
            "quantized_symbol_streams",
            "static_shared_pmf_k12_negative_control",
            "balle_full_learned_hyperprior",
            "arithmetic_terminal_coder",
            "runtime_packet_materializer",
        }
    }
    parallel_axes = {
        item["component_id"]: item["stack_axis"]
        for item in manifest["components"]
        if item["component_id"]
        in {"mask_stream_component", "pose_stream_component", "residual_stream_component"}
    }

    assert set(serial_axes.values()) == {"vstack_serial"}
    assert set(parallel_axes.values()) == {"hstack_parallel"}
    for component in manifest["components"]:
        if component["component_id"] in parallel_axes:
            assert f"{component['streams'][0]}_internal_parser_section_missing" in component["blockers"]
    assert all(item["transform_id"].startswith("vstack_") for item in manifest["serial_transforms"])
    assert all(item["group_id"].startswith("hstack_") for item in manifest["parallel_groups"])


def test_legacy_v1_manifest_aliases_roundtrip_to_canonical_axes() -> None:
    plan = build_hstack_vstack_multipass_plan(max_passes=2)
    manifest = json.loads(json.dumps(plan.to_manifest(), sort_keys=True))
    manifest["schema"] = "tac_hstack_vstack_multipass_plan_v1"
    for item in manifest["passes"]:
        item["hstack_transform_ids"] = item.pop("vstack_transform_ids")
        item["vstack_group_ids"] = item.pop("hstack_group_ids")

    loaded = plan_from_manifest(manifest)

    assert summarize_plan(loaded)["passes"] == 2
    loaded_manifest = loaded.to_manifest()
    assert "vstack_transform_ids" in loaded_manifest["passes"][0]
    assert "hstack_group_ids" in loaded_manifest["passes"][0]


def test_manifest_roundtrip_preserves_summary() -> None:
    plan = build_hstack_vstack_multipass_plan(
        anchor_id="candidate_pr106x",
        static_pmf_delta_bytes=123,
        max_passes=2,
    )
    manifest = json.loads(json.dumps(plan.to_manifest(), sort_keys=True))
    loaded = plan_from_manifest(manifest)

    assert summarize_plan(loaded) == summarize_plan(plan)
    assert loaded.to_manifest() == plan.to_manifest()


def test_fail_closed_policy_rejects_invalid_score_claims() -> None:
    with pytest.raises(ValueError, match="score_claim=True requires exact CUDA-grade evidence"):
        ByteEvidenceSemantics(
            evidence_grade="prediction",
            evidence_semantics="invalid_prediction_score_claim",
            score_claim=True,
        )

    policy = FailClosedPromotionPolicy()
    assert "full_sample_exact_cuda_auth_eval_json" in policy.score_promotion_requires
    assert "exact_reconstruction_roundtrip_passed" in policy.exact_eval_dispatch_requires


def test_validation_rejects_unknown_references_and_too_many_passes() -> None:
    with pytest.raises(ValueError, match="max_passes"):
        build_hstack_vstack_multipass_plan(max_passes=ABSOLUTE_MAX_PASSES + 1)

    plan = build_hstack_vstack_multipass_plan()
    bad_component = dataclasses_replace(
        _component(plan, "arithmetic_terminal_coder"),
        depends_on=("missing_component",),
    )
    components = tuple(
        bad_component if component.component_id == "arithmetic_terminal_coder" else component
        for component in plan.components
    )

    with pytest.raises(ValueError, match="depends on unknown components"):
        CodecStackPlan(
            plan_id="bad",
            anchor_id="anchor",
            static_pmf_delta_bytes=0,
            components=components,
            serial_transforms=plan.serial_transforms,
            parallel_groups=plan.parallel_groups,
            passes=plan.passes,
        )


def dataclasses_replace(component: CodecComponent, **kwargs: object) -> CodecComponent:
    payload = component.to_manifest()
    payload.update(kwargs)
    if "depends_on" in payload:
        payload["depends_on"] = tuple(payload["depends_on"])
    return CodecComponent(
        component_id=payload["component_id"],
        family=payload["family"],
        role=payload["role"],
        stack_axis=payload["stack_axis"],
        streams=tuple(payload["streams"]),
        byte_semantics=component.byte_semantics,
        depends_on=tuple(payload["depends_on"]),
        exact_roundtrip_required=bool(payload["exact_roundtrip_required"]),
        runtime_packet_required=bool(payload["runtime_packet_required"]),
        deterministic=bool(payload["deterministic"]),
        blockers=tuple(payload["blockers"]),
        repair_criteria=tuple(payload["repair_criteria"]),
        notes=tuple(payload["notes"]),
    )
