from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers
from tac.witness_dsl.curriculum_dsl import BASELINE, YhatNativeGenerator
from tac.witness_dsl.lever_registry import lever_factories, name_composable_levers, resolve_composable_lever
from tac.witness_dsl.yhat_native_generator_policy import YhatNativeGeneratorPolicy, YhatNativeGeneratorPolicyError


def test_sealed_policy_contract_is_json_safe_and_owns_both_planes() -> None:
    contract = YhatNativeGeneratorPolicy().compile()
    json.dumps(contract)
    assert contract["camera_hw"] == (874, 1164)
    assert contract["scorer_hw"] == (384, 512)
    assert contract["posenet_plane_ownership"] == "both_re_realized_rgb_frames"
    assert contract["segnet_plane_ownership"] == "shared_resized_re_realized_frame1_rgb"
    assert "upstream/modules.py" in dict(contract["value_provenance"])["geometry"]
    assert contract["activation_state"] == "BUILT_NOT_ACTIVATED_RECEIVER_ARCHIVE_GATES_OWED"
    assert contract["completed_gates"] == ("n24_exact_rational_plane_native_f32_ulp_receipt_closed_20260719",)
    assert "n>=24_full_hard_oracle_equivalence_receipt" not in contract["owed_gates"]
    assert "compact_description_receiver_closure" in contract["owed_gates"]


def test_authority_escalation_fails_closed() -> None:
    contract = YhatNativeGeneratorPolicy().compile()
    authority_fields = (
        "trainer_activation",
        "live_v10_integration",
        "launch",
        "paid_dispatch",
        "score_claim",
        "promotion",
        "promotion_eligible",
        "pointer_movement",
        "pointer_moved",
    )
    assert all(contract[field] is False for field in authority_fields)
    with pytest.raises(YhatNativeGeneratorPolicyError, match="cannot authorize"):
        YhatNativeGeneratorPolicy().compile(launch=True)
    for field in authority_fields:
        with pytest.raises(YhatNativeGeneratorPolicyError, match="sealed"):
            replace(YhatNativeGeneratorPolicy(), **{field: True}).compile()


def test_non_authority_policy_semantics_are_sealed() -> None:
    changed = replace(
        YhatNativeGeneratorPolicy(),
        frozen_scorer_order=("wrong_order",),
    )
    with pytest.raises(YhatNativeGeneratorPolicyError, match="frozen_scorer_order"):
        changed.compile()


def test_policy_lever_is_argv_inert_and_preserves_baseline_compile() -> None:
    lever = YhatNativeGenerator(policy=YhatNativeGeneratorPolicy())
    assert lever.name == "YhatNativeGenerator"
    assert lever.overrides == {} and lever.epochs_delta == 0
    assert lever.lawrefs == {} and lever.constant_manifest == {}
    assert BASELINE.with_lever(lever).compile_trainer_argv() == BASELINE.compile_trainer_argv()


def test_ast_knows_non_nilary_factory_but_launcher_refuses_bare_construction(tmp_path: Path) -> None:
    assert "YhatNativeGenerator" in lever_factories()
    assert "YhatNativeGenerator" in known_levers()
    assert "YhatNativeGenerator" in duty_to_measure(path=tmp_path / "empty-ledger.jsonl")
    assert "YhatNativeGenerator" not in name_composable_levers()
    with pytest.raises(ValueError, match="requires explicit args"):
        resolve_composable_lever("YhatNativeGenerator")
    with pytest.raises(TypeError, match="required keyword-only"):
        YhatNativeGenerator()  # type: ignore[call-arg]
