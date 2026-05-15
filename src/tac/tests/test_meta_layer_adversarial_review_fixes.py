# SPDX-License-Identifier: MIT
"""Regression tests for META-layer adversarial-review-round-1+2 fixes (2026-05-15).

Per `feedback_meta_layer_adversarial_review_round_1_2_landed_20260515.md`,
five HIGH/MEDIUM findings landed structural fixes:

  * **K1** (Carmack MEDIUM): ``register_substrate(contract)(non_callable)``
    silently registered a "ghost" substrate with no actual training entry
    point. Decorator now refuses non-callables and rolls back the
    registration to keep the registry clean.
  * **Q2** (Quantizr HIGH): out-of-band registry mutation propagated to
    auto_wire query helpers as ``AttributeError``. ``validate_all_registered``
    now exposes ``prune_corrupt=True``; ``_all()`` defensively skips
    non-SubstrateContract rows.
  * **F1** (Fridrich MEDIUM): ``archive_grammar="monolithic_single_file_*"``
    paired with multi-file ``parser_section_manifest`` was internally
    inconsistent; now refused at ``__post_init__``.
  * **M1** (MacKay MEDIUM): ``cost_band_gpu_key`` cheaper than
    ``recipe_min_smoke_gpu`` is impossible (cost-band budgets for a class
    smoke literally cannot run on); now refused.
  * **Q1** (Quantizr LOW): ``catalog_compliance_declarations`` accepts free
    strings (intentional — substrates can declare new catalog hooks as they
    land), but non-string / empty entries are now refused.

These fixes are LANDED in ``src/tac/substrate_registry/{decorator,contract,auto_wire}.py``
and this test pins each fix's contract.
"""

from __future__ import annotations

import pytest

from tac.substrate_registry import (
    _REGISTERED_SUBSTRATES,
    SubstrateContract,
    SubstrateContractError,
    _clear_registry_for_tests,
    register_substrate,
    validate_all_registered,
)
from tac.substrate_registry.auto_wire import (
    query_substrates_for_sensitivity_hook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_HOOK_RATIONALE = {
    "hook_sensitivity_contribution": "test",
    "hook_pareto_constraint": "test",
    "hook_bit_allocator_class": "test",
    "hook_continual_learning_anchor_kind": "test",
    "hook_probe_disambiguator": "test",
}


def _baseline_kwargs(**overrides):
    base = {
        "id": "test_substrate",
        "lane_id": "lane_test_substrate_a_20260515",
        "target_modes": ("research_substrate",),
        "deployment_target": "desktop_research",
        "council_verdict_provenance": None,
        "archive_grammar": "multi_file_archive",
        "parser_section_manifest": {"k": "v"},
        "inflate_runtime_loc_budget": 80,
        "runtime_dep_closure": ("torch",),
        "export_format": "fp16_brotli",
        "score_aware_loss": "scorer_loss_terms_btchw",
        "bolt_on_loc_budget": 200,
        "no_op_detector_planned": True,
        "archive_bytes_added": None,
        "score_improvement_mechanism_status": "RESEARCH_ONLY",
        "runtime_overlay_consumed": False,
        "recipe_smoke_only": True,
        "recipe_research_only": True,
        "recipe_min_smoke_gpu": "T4",
        "recipe_min_vram_gb": 16,
        "recipe_pyav_decode_strategy": "cpu_thread_async_upload",
        "recipe_canary_status": "independent_substrate",
        "recipe_video_input_strategy": "per_dispatch_local_copy",
        "recipe_canary_dependency": None,
        "cost_band_epochs": 10,
        "cost_band_gpu_key": "T4",
        "cost_band_platform_key": "modal",
        "cost_band_p50_usd": 0.10,
        "hook_sensitivity_contribution": "not_applicable_with_rationale",
        "hook_pareto_constraint": "not_applicable_with_rationale",
        "hook_bit_allocator_class": "not_applicable_with_rationale",
        "hook_autopilot_ranker_class_shift_token": None,
        "hook_continual_learning_anchor_kind": "not_applicable_with_rationale",
        "hook_probe_disambiguator": None,
        "catalog_compliance_declarations": (),
        "hook_not_applicable_rationale": dict(_REQUIRED_HOOK_RATIONALE),
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Finding K1 — ghost registration via non-callable decoration (MEDIUM)
# ---------------------------------------------------------------------------


def test_k1_decorator_refuses_integer_target():
    contract = SubstrateContract(**_baseline_kwargs(id="k1_int"))
    with pytest.raises(SubstrateContractError, match="must decorate a callable"):
        register_substrate(contract)(42)
    # Registry must be CLEAN — no partial registration left behind.
    assert "k1_int" not in _REGISTERED_SUBSTRATES


def test_k1_decorator_refuses_string_target():
    contract = SubstrateContract(**_baseline_kwargs(id="k1_str"))
    with pytest.raises(SubstrateContractError, match="must decorate a callable"):
        register_substrate(contract)("not a function")
    assert "k1_str" not in _REGISTERED_SUBSTRATES


def test_k1_decorator_refuses_none_target():
    contract = SubstrateContract(**_baseline_kwargs(id="k1_none"))
    with pytest.raises(SubstrateContractError, match="must decorate a callable"):
        register_substrate(contract)(None)
    assert "k1_none" not in _REGISTERED_SUBSTRATES


def test_k1_decorator_accepts_function_target():
    contract = SubstrateContract(**_baseline_kwargs(id="k1_fn"))

    @register_substrate(contract)
    def trainer():
        return 0

    assert "k1_fn" in _REGISTERED_SUBSTRATES
    assert trainer.__substrate_contract__ is contract  # type: ignore[attr-defined]


def test_k1_decorator_accepts_class_target():
    contract = SubstrateContract(**_baseline_kwargs(id="k1_cls"))

    @register_substrate(contract)
    class Trainer:
        pass

    assert "k1_cls" in _REGISTERED_SUBSTRATES
    assert Trainer().__class__.__name__ == "Trainer"


def test_k1_decorator_rollback_does_not_unregister_unrelated_id():
    """Rollback only pops THIS contract's id, not a sibling id."""
    sibling = SubstrateContract(**_baseline_kwargs(id="k1_sibling"))

    @register_substrate(sibling)
    def trainer():
        return 0

    bad = SubstrateContract(**_baseline_kwargs(id="k1_bad"))
    with pytest.raises(SubstrateContractError):
        register_substrate(bad)(7)

    # Sibling survives the rollback.
    assert "k1_sibling" in _REGISTERED_SUBSTRATES
    # Bad id removed.
    assert "k1_bad" not in _REGISTERED_SUBSTRATES


# ---------------------------------------------------------------------------
# Finding Q2 — out-of-band registry mutation propagates to consumers (HIGH)
# ---------------------------------------------------------------------------


def test_q2_validate_all_registered_warns_on_corrupt_row():
    class _Fake:
        id = "corrupt"

    _REGISTERED_SUBSTRATES["corrupt"] = _Fake()  # type: ignore[assignment]
    errors = validate_all_registered()
    assert any("corrupt" in e for e in errors)
    # Default (prune_corrupt=False): the bad row STAYS in the registry.
    assert "corrupt" in _REGISTERED_SUBSTRATES


def test_q2_validate_all_registered_prunes_when_requested():
    class _Fake:
        id = "corrupt"

    _REGISTERED_SUBSTRATES["corrupt"] = _Fake()  # type: ignore[assignment]
    errors = validate_all_registered(prune_corrupt=True)
    assert errors  # at least one error reported
    # With prune_corrupt=True, the bad row is GONE.
    assert "corrupt" not in _REGISTERED_SUBSTRATES


def test_q2_auto_wire_helper_skips_corrupt_row():
    """Auto-wire query helpers must NOT AttributeError on a corrupt row."""

    class _Fake:
        id = "corrupt"

    contract = SubstrateContract(**_baseline_kwargs(
        id="q2_real",
        hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
        hook_not_applicable_rationale={
            "hook_pareto_constraint": "test",
            "hook_bit_allocator_class": "test",
            "hook_continual_learning_anchor_kind": "test",
            "hook_probe_disambiguator": "test",
        },
    ))

    @register_substrate(contract)
    def trainer():
        return 0

    _REGISTERED_SUBSTRATES["corrupt"] = _Fake()  # type: ignore[assignment]

    # Must not raise; must return only the real contract.
    result = query_substrates_for_sensitivity_hook()
    assert len(result) == 1
    assert result[0].id == "q2_real"


def test_q2_validate_prunes_invalid_real_contract():
    """A SubstrateContract that fails re-validation is also pruned."""
    contract = SubstrateContract(**_baseline_kwargs(id="q2_valid"))
    _REGISTERED_SUBSTRATES["q2_valid"] = contract

    # Construct a syntactically-valid contract object then corrupt one field
    # via object.__setattr__ to bypass frozen=True.
    object.__setattr__(contract, "deployment_target", "BOGUS_NOT_A_LEGAL_VALUE")
    errors = validate_all_registered(prune_corrupt=True)
    assert any("q2_valid" in e for e in errors)
    assert "q2_valid" not in _REGISTERED_SUBSTRATES


# ---------------------------------------------------------------------------
# Finding F1 — monolithic-grammar + multi-file manifest inconsistency (MEDIUM)
# ---------------------------------------------------------------------------


def test_f1_monolithic_grammar_with_single_file_section_accepted():
    """Single section is fine even with a monolithic grammar."""
    c = SubstrateContract(
        **_baseline_kwargs(
            archive_grammar="monolithic_single_file_0_bin",
            parser_section_manifest={"weights": "decoder_weight_stream"},
        )
    )
    assert c.archive_grammar == "monolithic_single_file_0_bin"


def test_f1_monolithic_grammar_with_logical_sections_accepted():
    """Logical (non-file-extension) section labels under a monolithic grammar are fine."""
    c = SubstrateContract(
        **_baseline_kwargs(
            archive_grammar="monolithic_single_file_0_bin",
            parser_section_manifest={
                "header": "magic",
                "weights": "fp16_block",
                "latents": "int8_stream",
            },
        )
    )
    assert len(c.parser_section_manifest) == 3


def test_f1_monolithic_grammar_with_multiple_bin_files_refused():
    """Multiple .bin file-like sections + monolithic grammar = REFUSED."""
    with pytest.raises(SubstrateContractError, match="monolithic"):
        SubstrateContract(
            **_baseline_kwargs(
                archive_grammar="monolithic_single_file_0_bin",
                parser_section_manifest={"a.bin": "1", "b.bin": "2", "c.bin": "3"},
            )
        )


def test_f1_multi_file_grammar_with_multiple_files_accepted():
    """Multi-file grammar with multiple file sections is consistent."""
    c = SubstrateContract(
        **_baseline_kwargs(
            archive_grammar="multi_file_zip_with_sidecar",
            parser_section_manifest={"a.bin": "1", "b.bin": "2"},
        )
    )
    assert c.archive_grammar == "multi_file_zip_with_sidecar"


# ---------------------------------------------------------------------------
# Finding M1 — cost_band_gpu < recipe_min_smoke_gpu (MEDIUM)
# ---------------------------------------------------------------------------


def test_m1_cost_band_t4_smoke_a100_refused():
    with pytest.raises(SubstrateContractError, match="cheaper than"):
        SubstrateContract(
            **_baseline_kwargs(
                cost_band_gpu_key="T4",
                recipe_min_smoke_gpu="A100",
            )
        )


def test_m1_cost_band_a100_smoke_t4_accepted():
    """Reverse direction (cost-band promotes; smoke can run on cheaper) is fine."""
    c = SubstrateContract(
        **_baseline_kwargs(cost_band_gpu_key="A100", recipe_min_smoke_gpu="T4")
    )
    assert c.cost_band_gpu_key == "A100"


def test_m1_equal_classes_accepted():
    c = SubstrateContract(
        **_baseline_kwargs(cost_band_gpu_key="A100", recipe_min_smoke_gpu="A100")
    )
    assert c.recipe_min_smoke_gpu == c.cost_band_gpu_key


def test_m1_l40s_above_a10g_accepted():
    """L40S (rank 3) > A10G (rank 2): cost-band L40S, smoke A10G is consistent."""
    c = SubstrateContract(
        **_baseline_kwargs(cost_band_gpu_key="L40S", recipe_min_smoke_gpu="A10G")
    )
    assert c.cost_band_gpu_key == "L40S"


# ---------------------------------------------------------------------------
# Finding Q1 — catalog_compliance_declarations entry hygiene (LOW)
# ---------------------------------------------------------------------------


def test_q1_unknown_compliance_token_accepted_advisory():
    """Unknown tokens are advisory; substrates can declare new catalog hooks."""
    c = SubstrateContract(
        **_baseline_kwargs(
            catalog_compliance_declarations=(
                "catalog_999_future_hook_not_yet_in_known_set",
                "catalog_220_operational_mechanism_declared",
            ),
        )
    )
    assert "catalog_999_future_hook_not_yet_in_known_set" in c.catalog_compliance_declarations


def test_q1_non_string_compliance_token_refused():
    with pytest.raises(SubstrateContractError, match="non-string"):
        SubstrateContract(
            **_baseline_kwargs(
                catalog_compliance_declarations=("catalog_220_operational_mechanism_declared", 999),
            )
        )


def test_q1_empty_string_compliance_token_refused():
    with pytest.raises(SubstrateContractError, match="non-string or empty"):
        SubstrateContract(
            **_baseline_kwargs(
                catalog_compliance_declarations=("catalog_220_operational_mechanism_declared", ""),
            )
        )


def test_q1_whitespace_only_compliance_token_refused():
    with pytest.raises(SubstrateContractError, match="non-string or empty"):
        SubstrateContract(
            **_baseline_kwargs(
                catalog_compliance_declarations=("catalog_220_operational_mechanism_declared", "   "),
            )
        )
