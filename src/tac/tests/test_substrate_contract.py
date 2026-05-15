# SPDX-License-Identifier: MIT
"""Pydantic-style validators on SubstrateContract.

Per `.omx/research/substrate_meta_layer_design_20260515.md` § 5.3, the
contract refuses internally-inconsistent declarations at construction time.
"""

from __future__ import annotations

from typing import Any

import pytest

from tac.substrate_registry.contract import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    SubstrateContractError,
)


def _baseline_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal valid contract kwargs; tests override the field they exercise."""
    base: dict[str, Any] = dict(
        id="testsubstrate",
        lane_id="lane_testsubstrate_20260515",
        target_modes=("research_substrate",),
        deployment_target="desktop_research",
        council_verdict_provenance=None,
        archive_grammar="test_grammar",
        parser_section_manifest={"header": "magic"},
        inflate_runtime_loc_budget=80,
        runtime_dep_closure=("torch",),
        export_format="fp16_brotli",
        score_aware_loss="scorer_loss_terms_btchw",
        bolt_on_loc_budget=200,
        no_op_detector_planned=True,
        archive_bytes_added=None,
        score_improvement_mechanism_status="RESEARCH_ONLY",
        runtime_overlay_consumed=False,
        recipe_smoke_only=True,
        recipe_research_only=True,
        recipe_min_smoke_gpu="T4",
        recipe_min_vram_gb=16,
        recipe_pyav_decode_strategy="cpu_thread_async_upload",
        recipe_canary_status="independent_substrate",
        recipe_video_input_strategy="per_dispatch_local_copy",
        recipe_canary_dependency=None,
        cost_band_epochs=10,
        cost_band_gpu_key="T4",
        cost_band_platform_key="modal",
        cost_band_p50_usd=0.10,
        hook_sensitivity_contribution=NOT_APPLICABLE_WITH_RATIONALE,
        hook_pareto_constraint=NOT_APPLICABLE_WITH_RATIONALE,
        hook_bit_allocator_class=NOT_APPLICABLE_WITH_RATIONALE,
        hook_autopilot_ranker_class_shift_token=None,
        hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,
        hook_probe_disambiguator=None,
        catalog_compliance_declarations=("catalog_205_select_inflate_device_used",),
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": "test",
            "hook_pareto_constraint": "test",
            "hook_bit_allocator_class": "test",
            "hook_continual_learning_anchor_kind": "test",
            "hook_probe_disambiguator": "test",
        },
    )
    base.update(overrides)
    return base


def test_baseline_contract_validates() -> None:
    SubstrateContract(**_baseline_kwargs())


def test_id_must_match_pattern() -> None:
    with pytest.raises(SubstrateContractError, match="id="):
        SubstrateContract(**_baseline_kwargs(id="Bad-Id"))
    with pytest.raises(SubstrateContractError, match="id="):
        SubstrateContract(**_baseline_kwargs(id=""))


def test_lane_id_must_match_pattern() -> None:
    with pytest.raises(SubstrateContractError, match="lane_id"):
        SubstrateContract(**_baseline_kwargs(lane_id="not_a_lane"))


def test_target_modes_must_be_legal_and_nonempty() -> None:
    with pytest.raises(SubstrateContractError, match="target_modes"):
        SubstrateContract(**_baseline_kwargs(target_modes=()))
    with pytest.raises(SubstrateContractError, match="illegal"):
        SubstrateContract(**_baseline_kwargs(target_modes=("not_a_mode",)))


def test_deployment_target_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="deployment_target"):
        SubstrateContract(**_baseline_kwargs(deployment_target="bad"))


def test_export_format_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="export_format"):
        SubstrateContract(**_baseline_kwargs(export_format="bogus"))


def test_score_aware_loss_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="score_aware_loss"):
        SubstrateContract(**_baseline_kwargs(score_aware_loss="bogus"))


def test_inflate_loc_budget_must_be_positive() -> None:
    with pytest.raises(SubstrateContractError, match="inflate_runtime_loc_budget"):
        SubstrateContract(**_baseline_kwargs(inflate_runtime_loc_budget=0))


def test_runtime_dep_closure_must_be_nonempty() -> None:
    with pytest.raises(SubstrateContractError, match="runtime_dep_closure"):
        SubstrateContract(**_baseline_kwargs(runtime_dep_closure=()))


def test_operational_status_invariant() -> None:
    # OPERATIONAL requires runtime_overlay_consumed=True.
    with pytest.raises(SubstrateContractError, match="runtime_overlay_consumed=True"):
        SubstrateContract(
            **_baseline_kwargs(
                score_improvement_mechanism_status="OPERATIONAL",
                runtime_overlay_consumed=False,
            )
        )
    # runtime_overlay_consumed=True requires OPERATIONAL.
    with pytest.raises(SubstrateContractError, match="OPERATIONAL"):
        SubstrateContract(
            **_baseline_kwargs(
                score_improvement_mechanism_status="RESEARCH_ONLY",
                runtime_overlay_consumed=True,
            )
        )
    # OPERATIONAL + runtime_overlay_consumed=True → OK.
    SubstrateContract(
        **_baseline_kwargs(
            score_improvement_mechanism_status="OPERATIONAL",
            runtime_overlay_consumed=True,
        )
    )


def test_catalog_220_byte_addition_invariant() -> None:
    """SCAFFOLD with >1 KB byte addition is the research-substrate trap."""
    with pytest.raises(SubstrateContractError, match="Catalog #220"):
        SubstrateContract(
            **_baseline_kwargs(
                score_improvement_mechanism_status="SCAFFOLD_DEFERRED_INTEGRATION",
                archive_bytes_added="~43 KB sidecar",
            )
        )
    # Same byte addition with OPERATIONAL is OK.
    SubstrateContract(
        **_baseline_kwargs(
            score_improvement_mechanism_status="OPERATIONAL",
            runtime_overlay_consumed=True,
            archive_bytes_added="~43 KB sidecar",
        )
    )
    # SCAFFOLD with <1 KB byte addition is OK.
    SubstrateContract(
        **_baseline_kwargs(
            score_improvement_mechanism_status="SCAFFOLD_DEFERRED_INTEGRATION",
            archive_bytes_added="~0 KB tracking only",
        )
    )


def test_smoke_only_inconsistent_with_high_epochs() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_smoke_only=True"):
        SubstrateContract(
            **_baseline_kwargs(
                recipe_smoke_only=True,
                cost_band_epochs=2000,
            )
        )


def test_canary_dependency_required_when_post_canary() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_canary_dependency"):
        SubstrateContract(
            **_baseline_kwargs(
                recipe_canary_status="post_canary_dependent",
                recipe_canary_dependency=None,
            )
        )


def test_canary_dependency_forbidden_when_independent() -> None:
    with pytest.raises(SubstrateContractError, match="post_canary_dependent"):
        SubstrateContract(
            **_baseline_kwargs(
                recipe_canary_status="independent_substrate",
                recipe_canary_dependency="some_other_substrate",
            )
        )


def test_canary_dependency_ok_when_post_canary() -> None:
    SubstrateContract(
        **_baseline_kwargs(
            recipe_canary_status="post_canary_dependent",
            recipe_canary_dependency="parent_canary_id",
        )
    )


def test_min_vram_gb_floor() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_min_vram_gb"):
        SubstrateContract(**_baseline_kwargs(recipe_min_vram_gb=0))


def test_min_smoke_gpu_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_min_smoke_gpu"):
        SubstrateContract(**_baseline_kwargs(recipe_min_smoke_gpu="P100"))


def test_pyav_decode_strategy_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_pyav_decode_strategy"):
        SubstrateContract(**_baseline_kwargs(recipe_pyav_decode_strategy="bogus"))


def test_video_input_strategy_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="recipe_video_input_strategy"):
        SubstrateContract(**_baseline_kwargs(recipe_video_input_strategy="bogus"))


def test_cost_band_epochs_floor() -> None:
    with pytest.raises(SubstrateContractError, match="cost_band_epochs"):
        SubstrateContract(**_baseline_kwargs(cost_band_epochs=0))


def test_cost_band_gpu_key_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="cost_band_gpu_key"):
        SubstrateContract(**_baseline_kwargs(cost_band_gpu_key="P100"))


def test_cost_band_platform_key_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="cost_band_platform_key"):
        SubstrateContract(**_baseline_kwargs(cost_band_platform_key="aws"))


def test_cost_band_p50_usd_floor() -> None:
    with pytest.raises(SubstrateContractError, match="cost_band_p50_usd"):
        SubstrateContract(**_baseline_kwargs(cost_band_p50_usd=-0.5))


def test_hook_sensitivity_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="hook_sensitivity_contribution"):
        SubstrateContract(**_baseline_kwargs(hook_sensitivity_contribution="bogus"))


def test_hook_pareto_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="hook_pareto_constraint"):
        SubstrateContract(**_baseline_kwargs(hook_pareto_constraint="bogus"))


def test_hook_bit_allocator_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="hook_bit_allocator_class"):
        SubstrateContract(**_baseline_kwargs(hook_bit_allocator_class="bogus"))


def test_hook_continual_learning_must_be_legal() -> None:
    with pytest.raises(SubstrateContractError, match="hook_continual_learning_anchor_kind"):
        SubstrateContract(**_baseline_kwargs(hook_continual_learning_anchor_kind="bogus"))


def test_hook_not_applicable_requires_rationale() -> None:
    """Bare not_applicable_with_rationale without entry → reject."""
    kwargs = _baseline_kwargs(
        hook_sensitivity_contribution=NOT_APPLICABLE_WITH_RATIONALE,
    )
    kwargs["hook_not_applicable_rationale"] = {
        "hook_pareto_constraint": "test",
        "hook_bit_allocator_class": "test",
        "hook_continual_learning_anchor_kind": "test",
        "hook_probe_disambiguator": "test",
    }
    with pytest.raises(SubstrateContractError, match="hook_sensitivity_contribution"):
        SubstrateContract(**kwargs)


def test_hook_probe_disambiguator_none_requires_rationale() -> None:
    kwargs = _baseline_kwargs(hook_probe_disambiguator=None)
    kwargs["hook_not_applicable_rationale"] = {
        "hook_sensitivity_contribution": "test",
        "hook_pareto_constraint": "test",
        "hook_bit_allocator_class": "test",
        "hook_continual_learning_anchor_kind": "test",
    }
    with pytest.raises(SubstrateContractError, match="hook_probe_disambiguator"):
        SubstrateContract(**kwargs)


def test_hook_probe_disambiguator_string_must_be_nonempty() -> None:
    with pytest.raises(SubstrateContractError, match="hook_probe_disambiguator"):
        SubstrateContract(**_baseline_kwargs(hook_probe_disambiguator="   "))


def test_hook_probe_disambiguator_string_ok() -> None:
    SubstrateContract(
        **_baseline_kwargs(hook_probe_disambiguator="tools/probe_test_disambiguator.py")
    )


def test_hook_autopilot_token_can_be_none_or_str() -> None:
    SubstrateContract(**_baseline_kwargs(hook_autopilot_ranker_class_shift_token=None))
    SubstrateContract(
        **_baseline_kwargs(hook_autopilot_ranker_class_shift_token="MDL-IBPS")
    )
    with pytest.raises(SubstrateContractError, match="hook_autopilot_ranker_class_shift_token"):
        SubstrateContract(**_baseline_kwargs(hook_autopilot_ranker_class_shift_token=42))


def test_hook_rationale_keys_must_be_legal() -> None:
    kwargs = _baseline_kwargs()
    kwargs["hook_not_applicable_rationale"] = dict(kwargs["hook_not_applicable_rationale"])
    kwargs["hook_not_applicable_rationale"]["bogus_hook_name"] = "x"
    with pytest.raises(SubstrateContractError, match="illegal key"):
        SubstrateContract(**kwargs)


def test_to_dict_is_serializable() -> None:
    import json

    c = SubstrateContract(**_baseline_kwargs())
    d = c.to_dict()
    j = json.dumps(d, sort_keys=True)
    assert "testsubstrate" in j
    # All 36 fields present
    assert len(d) == 36


def test_contract_is_frozen() -> None:
    c = SubstrateContract(**_baseline_kwargs())
    with pytest.raises(Exception):  # FrozenInstanceError
        c.id = "different"  # type: ignore[misc]
