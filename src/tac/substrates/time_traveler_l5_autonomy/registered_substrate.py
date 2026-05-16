# SPDX-License-Identifier: MIT
"""Substrate registry contract for Time-Traveler L5 Autonomy.

Keep this contract in the importable substrate package so Cathedral/autopilot
can discover TT5L without importing the trainer script.
"""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_SIZE,
    TT5L_MAGIC,
    TT5L_SECTION_ROLES,
)

TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="time_traveler_l5_autonomy",
    lane_id="lane_time_traveler_l5_autonomy_substrate_20260513",
    target_modes=("contest_one_video_replay", "research_substrate"),
    deployment_target="comma_ai_production",
    council_verdict_provenance=(
        ".omx/research/grand_council_omnibus_design_decisions_20260514.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        f"{TT5L_MAGIC.decode('ascii')} monolithic single-file 0.bin: "
        f"{TT5L_HEADER_SIZE}-byte header plus length-prefixed logical sections "
        f"{tuple(TT5L_SECTION_ROLES.keys())}"
    ),
    parser_section_manifest=dict(TT5L_SECTION_ROLES),
    inflate_runtime_loc_budget=150,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=1290,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added="27 KB per-pair side-info stream before brotli",
    score_improvement_mechanism_status="OPERATIONAL",
    runtime_overlay_consumed=True,
    # 2.4 Recipe schema (8) mirrors substrate recipe YAML
    recipe_smoke_only=False,
    recipe_research_only=False,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=40,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=3000,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=8.0,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution=NOT_APPLICABLE_WITH_RATIONALE,
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class=NOT_APPLICABLE_WITH_RATIONALE,
    hook_autopilot_ranker_class_shift_token="time_traveler_l5_autonomy",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_197_full_cpu_coupled_flags_required",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "TT5L is a full predictive-receiver renderer; sensitivity is "
            "tracked through temporal side-info and exact eval anchors."
        ),
        "hook_bit_allocator_class": (
            "TT5L v1 uses a custom world-model packet plus side-info stream; "
            "per-tensor allocator hooks are not the controlling mechanism."
        ),
        "hook_probe_disambiguator": (
            "No callable C1/Z5/TT5L disambiguator has landed yet; the hook "
            "stays explicitly unsatisfied rather than naming a planned file."
        ),
    },
)


@register_substrate(TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT)
def registered_substrate_entrypoint(argv: list[str] | None = None) -> int:
    """Registry-only entrypoint; the trainer script owns executable training."""
    raise NotImplementedError(
        "Run experiments/train_substrate_time_traveler_l5_autonomy.py for TT5L training."
    )


__all__ = [
    "TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT",
    "registered_substrate_entrypoint",
]
