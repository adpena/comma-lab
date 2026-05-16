# SPDX-License-Identifier: MIT
"""NSCS02 downsampled-renderer SubstrateContract registration."""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)

NSCS02_DOWNSAMPLED_RENDERER_CONTRACT = SubstrateContract(
    id="nscs02_downsampled_renderer",
    lane_id="lane_nscs02_downsampled_renderer_inflate_upsample_20260515",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=".omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json",
    archive_grammar="ns02_monolithic_lowres_renderer_inflate_upsample",
    parser_section_manifest={
        "header": "NS02 fixed header",
        "decoder_blob": "brotli(fp16 decoder state)",
        "latent_blob": "low-resolution per-pair latents",
        "meta_blob": "sorted json resize/provenance manifest",
    },
    inflate_runtime_loc_budget=160,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=320,
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
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=5.00,
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_tensor_uniform",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,
    hook_probe_disambiguator=None,
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_220_operational_mechanism_declared",
    ),
    hook_not_applicable_rationale={
        "hook_continual_learning_anchor_kind": (
            "No empirical paired-axis anchor exists; posterior update waits for "
            "the resizing-chain ablation and paired exact eval."
        ),
        "hook_probe_disambiguator": (
            "The required resizing-chain ablation is queued in the assumptions "
            "matrix; no dispatch-ranking probe artifact exists yet."
        ),
    },
)


@register_substrate(NSCS02_DOWNSAMPLED_RENDERER_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Package-level entry: delegates to the canonical trainer's main().

    Per the UNIQUE-AND-COMPLETE-PER-METHOD operating mode (2026-05-15), the
    trainer's ``_full_main`` is now implemented. Promotion remains gated by
    the resizing-chain ablation + paired CPU+CUDA exact-eval custody per
    the recipe's ``research_only=true`` flag; the trainer-side full path is
    available for operator-approved dispatches.
    """
    # Import the trainer lazily so package-side construction does not pull
    # the heavy torch + scorer + pyav dependency graph into every importer.
    from pathlib import Path
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[4]
    trainer_path = repo_root / "experiments" / "train_substrate_nscs02_downsampled_renderer.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"NSCS02 trainer missing: {trainer_path}")
    spec = importlib.util.spec_from_file_location(
        "_nscs02_trainer", trainer_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load NSCS02 trainer spec from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_nscs02_trainer"] = module
    spec.loader.exec_module(module)
    return int(module.main(argv))


__all__ = ["NSCS02_DOWNSAMPLED_RENDERER_CONTRACT", "main"]
