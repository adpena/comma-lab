# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

import tac.optimization as optimization
from tac.optimization.parameter_group_lr_policy import (
    DEFAULT_PARAMETER_GROUP_LR_POLICY,
    EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    EMBEDDING_THETA1_POLICY_ID,
    FALSE_AUTHORITY_FIELDS,
    PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA,
    PARAMETER_GROUP_LR_POLICY_SCHEMA,
    ParameterGroupLRPolicyError,
    ParameterShapeRecord,
    build_parameter_group_lr_policy_fingerprint,
    classify_parameter_record,
    classify_parameter_records,
    parameter_group_lr_policy_fingerprint_sha256,
    parameter_group_lr_policy_sha256,
    validate_parameter_group_lr_policy,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def _records() -> list[tuple[str, tuple[int, ...] | None]]:
    return [
        ("renderer.frame_embedding.weight", (1024, 64)),
        ("decoder.latent_codebook", (256, 8)),
        ("blocks.0.mlp.fc1.weight", (256, 64)),
        ("blocks.0.norm.weight", (64,)),
        ("decoder.lm_head.weight", (5, 64)),
        ("posenet.encoder.weight", (32, 16)),
        ("runtime.opaque_token", None),
    ]


def test_embedding_theta1_policy_classifies_generic_name_shape_records() -> None:
    rows = classify_parameter_records(_records())
    by_name = {row.name: row for row in rows}

    assert by_name["renderer.frame_embedding.weight"].parameter_class == "embedding_like"
    assert by_name["renderer.frame_embedding.weight"].optimizer == "AdamW"
    assert by_name["renderer.frame_embedding.weight"].lr_scaling_policy == (
        "theta_1_not_inverse_width"
    )
    assert by_name["decoder.latent_codebook"].parameter_class == "embedding_like"
    assert by_name["blocks.0.mlp.fc1.weight"].parameter_class == "hidden_matrix"
    assert by_name["blocks.0.mlp.fc1.weight"].optimizer == "Muon"
    assert by_name["blocks.0.norm.weight"].parameter_class == "head_scalar_norm"
    assert by_name["decoder.lm_head.weight"].parameter_class == "head_scalar_norm"
    assert by_name["posenet.encoder.weight"].parameter_class == "hidden_matrix"
    assert by_name["runtime.opaque_token"].parameter_class == "unclassified"
    assert by_name["runtime.opaque_token"].optimizer == "manual_review"


def test_single_record_inputs_accept_tuple_mapping_and_dataclass_forms() -> None:
    tuple_row = classify_parameter_record(("decoder.position_embedding.weight", (32, 4)))
    mapping_row = classify_parameter_record({"name": "decoder.position_embedding.weight", "shape": [32, 4]})
    dataclass_row = classify_parameter_record(
        ParameterShapeRecord(name="decoder.position_embedding.weight", shape=(32, 4))
    )

    assert tuple_row == mapping_row == dataclass_row
    assert tuple_row.parameter_class == "embedding_like"
    assert tuple_row.policy_id == EMBEDDING_THETA1_POLICY_ID


def test_fingerprint_is_json_safe_order_stable_proxy_only_and_shape_sensitive() -> None:
    first = build_parameter_group_lr_policy_fingerprint(_records())
    reversed_input = build_parameter_group_lr_policy_fingerprint(list(reversed(_records())))
    changed = build_parameter_group_lr_policy_fingerprint(
        [
            *(_records()[:-1]),
            ("runtime.opaque_token", (1,)),
        ]
    )

    assert first["schema"] == PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA
    assert first["policy"]["schema"] == PARAMETER_GROUP_LR_POLICY_SCHEMA
    assert first["policy_id"] == EMBEDDING_THETA1_POLICY_ID
    assert len(first["policy_sha256"]) == 64
    assert len(first["fingerprint_sha256"]) == 64
    assert first["fingerprint_sha256"] == reversed_input["fingerprint_sha256"]
    assert first["fingerprint_sha256"] != changed["fingerprint_sha256"]
    assert first["fingerprint_sha256"] == parameter_group_lr_policy_fingerprint_sha256(_records())
    assert first["class_counts"] == {
        "embedding_like": 2,
        "hidden_matrix": 2,
        "head_scalar_norm": 2,
        "unclassified": 1,
    }
    assert first["unknown_shape_count"] == 1
    assert first["planning_only"] is True
    assert first["rank_score_field"] == "parameter_group_lr_policy_planning_not_score"
    assert validate_proxy_candidate(first) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert first[key] is expected
        assert first["false_authority"][key] is expected
    json.dumps(first, sort_keys=True, allow_nan=False)


def test_pytorch_tuple_and_mlx_mapping_records_produce_same_group_fingerprint() -> None:
    pytorch_style = [
        ("model.embed_tokens.weight", (512, 32)),
        ("model.blocks.0.attn.qkv.weight", (96, 32)),
        ("model.blocks.0.norm.bias", (32,)),
    ]
    mlx_style = [
        {"name": "model.blocks.0.norm.bias", "shape": [32]},
        {"name": "model.blocks.0.attn.qkv.weight", "shape": [96, 32]},
        {"name": "model.embed_tokens.weight", "shape": [512, 32]},
    ]

    assert classify_parameter_records(pytorch_style) == classify_parameter_records(mlx_style)
    assert (
        parameter_group_lr_policy_fingerprint_sha256(pytorch_style)
        == parameter_group_lr_policy_fingerprint_sha256(mlx_style)
    )


def test_single_group_baseline_policy_routes_every_parameter_to_primary_optimizer() -> None:
    rows = classify_parameter_records(_records(), policy=DEFAULT_PARAMETER_GROUP_LR_POLICY)

    assert {row.optimizer for row in rows} == {"primary_optimizer"}
    assert {row.lr_scaling_policy for row in rows} == {"same_as_base_lr"}


def test_parameter_group_policy_helpers_are_public_lazy_optimization_exports() -> None:
    assert optimization.EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY["policy_id"] == (
        EMBEDDING_THETA1_POLICY_ID
    )
    assert optimization.PARAMETER_GROUP_LR_POLICY_SCHEMA == PARAMETER_GROUP_LR_POLICY_SCHEMA
    assert optimization.classify_parameter_record(("embed.weight", (8, 4))).parameter_class == (
        "embedding_like"
    )


def test_policy_and_shape_inputs_fail_closed() -> None:
    validate_parameter_group_lr_policy(EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY)
    assert len(parameter_group_lr_policy_sha256(EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY)) == 64

    with pytest.raises(ParameterGroupLRPolicyError, match="score_claim"):
        validate_parameter_group_lr_policy(
            {
                **EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
                "score_claim": True,
            }
        )
    with pytest.raises(ParameterGroupLRPolicyError, match="score_claim"):
        validate_parameter_group_lr_policy(
            {
                **EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
                "score_claim": "false",
            }
        )
    with pytest.raises(ParameterGroupLRPolicyError, match="ready_for_exact_eval_dispatch"):
        validate_parameter_group_lr_policy(
            {
                **EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
                "metadata": {"ready_for_exact_eval_dispatch": None},
            }
        )
    with pytest.raises(ParameterGroupLRPolicyError, match="duplicate parameter names"):
        classify_parameter_records(
            [
                ("dup.weight", (2, 2)),
                ("dup.weight", (2, 2)),
            ]
        )
    with pytest.raises(ParameterGroupLRPolicyError, match="non-negative"):
        classify_parameter_record(("bad.weight", (-1, 4)))
    with pytest.raises(ParameterGroupLRPolicyError, match="integers"):
        classify_parameter_record(("bad.weight", (1.5, 4)))


def test_hidden_param_patterns_gate_hidden_matrix_classification() -> None:
    policy = {
        **EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
        "hidden_param_patterns": [],
    }

    row = classify_parameter_record(("blocks.0.mlp.fc1.weight", (256, 64)), policy=policy)

    assert row.parameter_class == "unclassified"
    assert row.optimizer == "manual_review"
    assert row.reason == "no_hidden_param_pattern_matched"
