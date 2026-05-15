"""@register_substrate decorator + registry behavior."""

from __future__ import annotations

from typing import Any

import pytest

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    SubstrateContractError,
    _clear_registry_for_tests,
    _REGISTERED_SUBSTRATES,
    get_registered_substrates,
    query_substrates_by_compliance_token,
    query_substrates_for_autopilot_ranker,
    query_substrates_for_bit_allocator_hook,
    query_substrates_for_continual_learning_anchor_kind,
    query_substrates_for_pareto_hook,
    query_substrates_for_probe_disambiguators,
    query_substrates_for_sensitivity_hook,
    register_substrate,
    validate_all_registered,
)


def _baseline_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        id="reg_test",
        lane_id="lane_reg_test_20260515",
        target_modes=("research_substrate",),
        deployment_target="desktop_research",
        council_verdict_provenance=None,
        archive_grammar="g",
        parser_section_manifest={"h": "magic"},
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


@pytest.fixture(autouse=True)
def _isolated_registry() -> None:
    """Clear registry before/after each test for isolation.

    The example_template module is imported by other test modules; we save
    and restore the snapshot so this fixture doesn't affect them.
    """
    snapshot = dict(_REGISTERED_SUBSTRATES)
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()
    _REGISTERED_SUBSTRATES.update(snapshot)


def test_decorator_registers_substrate() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="alpha"))

    @register_substrate(c)
    def main(argv=None):
        return 0

    assert "alpha" in get_registered_substrates()
    assert get_registered_substrates()["alpha"] is c
    assert main.__substrate_contract__ is c


def test_decorator_returns_unmodified_callable() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="passthrough"))

    @register_substrate(c)
    def worker(x: int) -> int:
        return x * 2

    assert worker(21) == 42


def test_decorator_rejects_non_contract() -> None:
    with pytest.raises(SubstrateContractError, match="SubstrateContract"):
        register_substrate({"id": "x"})  # type: ignore[arg-type]


def test_duplicate_id_with_different_contract_refused() -> None:
    c1 = SubstrateContract(**_baseline_kwargs(id="dup", recipe_min_vram_gb=16))
    c2 = SubstrateContract(**_baseline_kwargs(id="dup", recipe_min_vram_gb=24))

    register_substrate(c1)(lambda: 0)
    with pytest.raises(SubstrateContractError, match="Duplicate"):
        register_substrate(c2)(lambda: 0)


def test_re_registration_idempotent_on_same_contract_identity() -> None:
    """Re-importing a module re-runs the decorator with the SAME contract instance."""
    c = SubstrateContract(**_baseline_kwargs(id="reentrant"))
    register_substrate(c)(lambda: 0)
    # Same contract instance — must not raise.
    register_substrate(c)(lambda: 0)
    assert get_registered_substrates()["reentrant"] is c


def test_get_registered_returns_copy() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="copytest"))
    register_substrate(c)(lambda: 0)
    snap = get_registered_substrates()
    snap["evil_injection"] = c  # type: ignore[assignment]
    assert "evil_injection" not in get_registered_substrates()


def test_validate_all_registered_passes_for_clean_registry() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="clean1")))(lambda: 0)
    register_substrate(SubstrateContract(**_baseline_kwargs(id="clean2")))(lambda: 0)
    assert validate_all_registered() == []


def test_query_sensitivity_excludes_n_a() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="s_na")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="s_active",
                hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
            )
        )
    )(lambda: 0)
    ids = {c.id for c in query_substrates_for_sensitivity_hook()}
    assert ids == {"s_active"}


def test_query_pareto_excludes_n_a() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="p_na")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="p_active",
                hook_pareto_constraint="rate_distortion_v1",
            )
        )
    )(lambda: 0)
    ids = {c.id for c in query_substrates_for_pareto_hook()}
    assert ids == {"p_active"}


def test_query_bit_allocator_excludes_n_a() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="b_na")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="b_active",
                hook_bit_allocator_class="per_channel_lsq",
            )
        )
    )(lambda: 0)
    ids = {c.id for c in query_substrates_for_bit_allocator_hook()}
    assert ids == {"b_active"}


def test_query_autopilot_token_map() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="within_class")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="cross_class",
                hook_autopilot_ranker_class_shift_token="MDL-IBPS",
            )
        )
    )(lambda: 0)
    m = query_substrates_for_autopilot_ranker()
    assert m == {"cross_class": "MDL-IBPS"}


def test_query_continual_learning_excludes_n_a() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="cl_na")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="cl_paired",
                hook_continual_learning_anchor_kind="paired_axis",
            )
        )
    )(lambda: 0)
    m = query_substrates_for_continual_learning_anchor_kind()
    assert m == {"cl_paired": "paired_axis"}


def test_query_probe_disambiguators_excludes_none() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="probe_none")))(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="probe_present",
                hook_probe_disambiguator="tools/probe_x_disambiguator.py",
            )
        )
    )(lambda: 0)
    m = query_substrates_for_probe_disambiguators()
    assert m == {"probe_present": "tools/probe_x_disambiguator.py"}


def test_query_compliance_token() -> None:
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="compl_a",
                catalog_compliance_declarations=("catalog_205_select_inflate_device_used",),
            )
        )
    )(lambda: 0)
    register_substrate(
        SubstrateContract(
            **_baseline_kwargs(
                id="compl_b",
                catalog_compliance_declarations=("catalog_226_gate_auth_eval_call_used",),
            )
        )
    )(lambda: 0)
    cs = query_substrates_by_compliance_token("catalog_205_select_inflate_device_used")
    assert {c.id for c in cs} == {"compl_a"}


def test_example_template_imports_clean() -> None:
    """Reference implementation must validate against its own contract."""
    from tac.substrate_registry import example_template

    assert example_template.EXAMPLE_TEMPLATE_CONTRACT.id == "example_template"


def test_clear_registry_for_tests_idempotent() -> None:
    register_substrate(SubstrateContract(**_baseline_kwargs(id="zzz")))(lambda: 0)
    _clear_registry_for_tests()
    assert get_registered_substrates() == {}
    _clear_registry_for_tests()  # idempotent
    assert get_registered_substrates() == {}
