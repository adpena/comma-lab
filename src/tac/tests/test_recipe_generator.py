"""Recipe-generator output validates against the canonical YAML schema.

Cross-validates the generated recipe against the C6 / sane_hnerv canonical
templates: every required key in the canonical templates must appear in the
generated output (and we test the inverse: no key was missed).
"""

from __future__ import annotations

from typing import Any

from tac.substrate_registry.contract import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
)
from tac.substrate_registry.recipe_generator import (
    default_recipe_relpath,
    generate_recipe_yaml,
)


def _baseline_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "rgen_test",
        "lane_id": "lane_rgen_test_20260515",
        "target_modes": ("research_substrate",),
        "deployment_target": "desktop_research",
        "council_verdict_provenance": None,
        "archive_grammar": "g",
        "parser_section_manifest": {"h": "magic"},
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
        "hook_sensitivity_contribution": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_pareto_constraint": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_bit_allocator_class": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_autopilot_ranker_class_shift_token": None,
        "hook_continual_learning_anchor_kind": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_probe_disambiguator": None,
        "catalog_compliance_declarations": ("catalog_205_select_inflate_device_used",),
        "hook_not_applicable_rationale": {
            "hook_sensitivity_contribution": "test",
            "hook_pareto_constraint": "test",
            "hook_bit_allocator_class": "test",
            "hook_continual_learning_anchor_kind": "test",
            "hook_probe_disambiguator": "test",
        },
    }
    base.update(overrides)
    return base


def test_default_recipe_relpath_canonical() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="alpha"))
    assert default_recipe_relpath(c) == (
        ".omx/operator_authorize_recipes/substrate_alpha_modal_t4_dispatch.yaml"
    )


def test_generate_recipe_yaml_is_deterministic() -> None:
    c = SubstrateContract(**_baseline_kwargs())
    a = generate_recipe_yaml(c)
    b = generate_recipe_yaml(c)
    assert a == b


def test_generate_recipe_yaml_has_canonical_keys() -> None:
    """Every key the canonical recipes use must appear in generator output."""
    c = SubstrateContract(**_baseline_kwargs(id="canonical_check"))
    y = generate_recipe_yaml(c)
    required_top_level_keys = [
        "schema_version:",
        "name:",
        "lane_id:",
        "summary:",
        "platform:",
        "gpu:",
        "min_vram_gb:",
        "min_smoke_gpu:",
        "video_input_strategy:",
        "pyav_decode_strategy:",
        "target_modes:",
        "canary_status:",
        "smoke_only:",
        "research_only:",
        "cost_band:",
        "remote_driver:",
        "timeout_hours:",
        "required_input_files:",
        "modal:",
        "required_input_files_trainer:",
        "sentinel_files:",
        "catalog_compliance_declarations:",
        "hook_sensitivity_contribution:",
        "hook_pareto_constraint:",
        "hook_bit_allocator_class:",
        "hook_autopilot_ranker_class_shift_token:",
        "hook_continual_learning_anchor_kind:",
        "hook_probe_disambiguator:",
        "notes:",
    ]
    for key in required_top_level_keys:
        assert key in y, f"missing key {key!r} in generated recipe"


def test_generate_recipe_yaml_canary_dependency_only_when_post_canary() -> None:
    c1 = SubstrateContract(**_baseline_kwargs(recipe_canary_status="independent_substrate"))
    y1 = generate_recipe_yaml(c1)
    assert "canary_dependency:" not in y1

    c2 = SubstrateContract(
        **_baseline_kwargs(
            recipe_canary_status="post_canary_dependent",
            recipe_canary_dependency="parent_canary_id",
        )
    )
    y2 = generate_recipe_yaml(c2)
    assert "canary_dependency: parent_canary_id" in y2


def test_generate_recipe_yaml_smoke_only_truthful() -> None:
    c1 = SubstrateContract(**_baseline_kwargs(recipe_smoke_only=True))
    y1 = generate_recipe_yaml(c1)
    assert "smoke_only: true" in y1
    c2 = SubstrateContract(
        **_baseline_kwargs(recipe_smoke_only=False, cost_band_epochs=2000)
    )
    y2 = generate_recipe_yaml(c2)
    assert "smoke_only: false" in y2


def test_generate_recipe_yaml_target_modes_list_form() -> None:
    c = SubstrateContract(
        **_baseline_kwargs(target_modes=("contest_one_video_replay", "research_substrate"))
    )
    y = generate_recipe_yaml(c)
    assert "  - contest_one_video_replay" in y
    assert "  - research_substrate" in y


def test_generate_recipe_yaml_min_smoke_gpu_quoted() -> None:
    # Note: per adversarial-review finding M1 (2026-05-15) the contract
    # refuses recipe_min_smoke_gpu > cost_band_gpu_key (cost-band would
    # budget for a class less capable than smoke can run on); promote
    # cost_band_gpu_key to A100 so this YAML-quoting test still exercises
    # the smoke-gpu='A100' code path.
    c = SubstrateContract(
        **_baseline_kwargs(recipe_min_smoke_gpu="A100", cost_band_gpu_key="A100")
    )
    y = generate_recipe_yaml(c)
    assert 'min_smoke_gpu: "A100"' in y


def test_generate_recipe_yaml_includes_hook_token() -> None:
    c = SubstrateContract(
        **_baseline_kwargs(
            hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
            hook_autopilot_ranker_class_shift_token="MDL-IBPS",
        )
    )
    y = generate_recipe_yaml(c)
    assert "hook_sensitivity_contribution: scorer_conditional_entropy_map_v1" in y
    assert "MDL-IBPS" in y


def test_generate_recipe_yaml_compliance_declarations_emitted_as_list() -> None:
    c = SubstrateContract(
        **_baseline_kwargs(
            catalog_compliance_declarations=(
                "catalog_146_3arg_archive_grammar_honored",
                "catalog_205_select_inflate_device_used",
                "catalog_226_gate_auth_eval_call_used",
            )
        )
    )
    y = generate_recipe_yaml(c)
    for token in (
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_205_select_inflate_device_used",
        "catalog_226_gate_auth_eval_call_used",
    ):
        assert f"  - {token}" in y


def test_generate_recipe_yaml_no_runtime_clock_drift() -> None:
    """Generated YAML must not embed clock/host/random values."""
    import datetime
    import re

    c = SubstrateContract(**_baseline_kwargs())
    y = generate_recipe_yaml(c)
    # No ISO-8601 timestamp.
    iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    assert not iso_pattern.search(y), "generated YAML embeds an ISO timestamp"
    # No 8-digit YYYYMMDD that isn't part of the lane_id we declared.
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    extras = [m for m in re.findall(r"\b\d{8}\b", y) if m != "20260515"]
    assert today not in extras
