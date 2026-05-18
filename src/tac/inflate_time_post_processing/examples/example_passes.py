# SPDX-License-Identifier: MIT
"""Canonical example inflate-time post-processing passes — one per builder.

Six minimal passes that exercise the decorator + composition API:

  1. ``raw_inflate_example``                       : seed/passthrough stage
  2. ``bilateral_denoise_per_frame_example``       : BilateralFilterPostProcessor sample
  3. ``nlm_denoise_per_pair_example``              : NLMDenoisingPostProcessor sample
  4. ``learned_unet_4block_per_frame_example``     : LearnedPostFilterApplier sample
  5. ``lanczos_upscale_384_to_874_example``        : SuperResolutionUpscaler sample
  6. ``multi_pass_inflate_7_variants_example``     : MultiPassInflateRefinement sample

The example bodies are TOY (zero frames_processed / identity transform) so
the passes are testable without GPU. Real consumers replace the body with
substrate-specific logic (e.g. cv2.bilateralFilter / cv2.fastNlMeansDenoising /
torch model forward).

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in the
docstrings is backed by an executable body.
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.inflate_time_post_processing.bilateral_filter import (
    BilateralFilterPostProcessor,
    BilateralFilterSpec,
)
from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
)
from tac.inflate_time_post_processing.decorator import (
    inflate_time_post_filter,
)
from tac.inflate_time_post_processing.learned_post_filter import (
    LearnedPostFilterApplier,
    LearnedPostFilterSpec,
)
from tac.inflate_time_post_processing.multi_pass_refinement import (
    MultiPassInflateRefinement,
    MultiPassInflateRefinementSpec,
)
from tac.inflate_time_post_processing.nlm_denoising import (
    NLMDenoisingPostProcessor,
    NLMDenoisingSpec,
)
from tac.inflate_time_post_processing.super_resolution_upscaler import (
    SuperResolutionUpscaler,
    SuperResolutionUpscalerSpec,
)

# Lane id shared across the example passes for provenance.
_EXAMPLE_LANE_ID = (
    "lane_tac_inflate_time_post_processing_namespace_decorator_api_20260517"
)
_EXAMPLE_DESIGN_MEMO = (
    ".omx/research/"
    "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
)


# ---------------------------------------------------------------------------
# 1. Seed / passthrough stage (raw inflate placeholder)
# ---------------------------------------------------------------------------


@inflate_time_post_filter(
    InflateTimePostProcessingContract(
        id="raw_inflate_example",
        parent_pass_id=None,
        stage_phase="inflate",
        description=(
            "Seed/passthrough stage: emits frames_v0 from the seed state's "
            "decoded_frames key. Used as the root of example pipelines."
        ),
        consumes=frozenset({"decoded_frames"}),
        emits=frozenset({"frames_v0"}),
        correction_kind="transform",
        correction_resolution="global",
        applies_to_frames="all",
        deterministic=True,
        scorer_free=True,
        max_wallclock_seconds=1.0,
        archive_bytes_added=0,
        score_axis_affected=(),
        requires_scorer_surrogate=False,
        requires_cpu_only=True,
        seed=42,
        merge_policy="last_writer_wins",
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="inflate_wallclock_envelope_v1",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind=(
            "inflate_time_post_processing_pass_outcomes_v1"
        ),
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "Passthrough emits the input unchanged; sensitivity weighting "
                "is meaningless at the seed."
            ),
            "hook_bit_allocator_class": (
                "Passthrough does not allocate archive bytes; bit allocation "
                "is structurally undefined."
            ),
            "hook_probe_disambiguator": (
                "Passthrough has a single canonical interpretation: emit "
                "what was consumed. No defensible alternative."
            ),
        },
        lane_id=_EXAMPLE_LANE_ID,
        design_memo=_EXAMPLE_DESIGN_MEMO,
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL — this is the canonical seed stage."
        ),
    )
)
def raw_inflate_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Passthrough seed stage."""
    return {"frames_v0": state.get("decoded_frames"), "frames_processed": 0}


# ---------------------------------------------------------------------------
# 2. BilateralFilterPostProcessor example
# ---------------------------------------------------------------------------

_BILATERAL_SPEC = BilateralFilterSpec(
    pass_id="bilateral_denoise_per_frame_example",
    sigma_spatial=2.0,
    sigma_intensity=0.1,
    kernel_diameter=5,
    applies_to_frames="all",
    score_axis_affected=("seg",),
    max_wallclock_seconds=60.0,
    seed=42,
    description=(
        "Toy bilateral denoise example: returns input unchanged so the test "
        "harness can assert the decorator + pipeline machinery work end-to-end."
    ),
    lane_id=_EXAMPLE_LANE_ID,
)


@inflate_time_post_filter(
    BilateralFilterPostProcessor(spec=_BILATERAL_SPEC).build_contract()
)
def bilateral_denoise_per_frame_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy bilateral pass — emits frames_v1 == frames_v0 (identity)."""
    return {"frames_v1": state.get("frames_v0"), "frames_processed": 0}


# ---------------------------------------------------------------------------
# 3. NLMDenoisingPostProcessor example
# ---------------------------------------------------------------------------

_NLM_SPEC = NLMDenoisingSpec(
    pass_id="nlm_denoise_per_pair_example",
    patch_size=7,
    search_window=21,
    h=0.05,
    applies_to_frames="pairs_only",
    score_axis_affected=("pose",),
    max_wallclock_seconds=180.0,
    seed=42,
    description=(
        "Toy NLM denoise example: returns input unchanged. Real impl wraps "
        "cv2.fastNlMeansDenoising."
    ),
    lane_id=_EXAMPLE_LANE_ID,
)


@inflate_time_post_filter(
    NLMDenoisingPostProcessor(spec=_NLM_SPEC).build_contract()
)
def nlm_denoise_per_pair_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy NLM pass — emits frames_v1 == frames_v0 (identity)."""
    return {"frames_v1": state.get("frames_v0"), "frames_processed": 0}


# ---------------------------------------------------------------------------
# 4. LearnedPostFilterApplier example
# ---------------------------------------------------------------------------

_LEARNED_SPEC = LearnedPostFilterSpec(
    pass_id="learned_unet_4block_per_frame_example",
    model_identifier="example_distilled_unet_v0_test_only",
    expected_input_shape=(1, 3, 384, 512),
    expected_output_shape=(1, 3, 384, 512),
    applies_to_frames="all",
    score_axis_affected=("seg", "pose"),
    max_wallclock_seconds=240.0,
    seed=42,
    description=(
        "Toy learned post-filter example: identity model. Real impl loads a "
        "distilled U-Net from archive bytes."
    ),
    lane_id=_EXAMPLE_LANE_ID,
)


@inflate_time_post_filter(
    LearnedPostFilterApplier(spec=_LEARNED_SPEC).build_contract()
)
def learned_unet_4block_per_frame_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy learned-filter pass — emits frames_v1 == frames_v0 (identity)."""
    return {"frames_v1": state.get("frames_v0"), "frames_processed": 0}


# ---------------------------------------------------------------------------
# 5. SuperResolutionUpscaler example
# ---------------------------------------------------------------------------

_UPSCALE_SPEC = SuperResolutionUpscalerSpec(
    pass_id="lanczos_upscale_384_to_874_example",
    upscaler_kind="lanczos",
    input_shape=(384, 512),
    output_shape=(874, 1164),
    learned_model_identifier=None,
    applies_to_frames="all",
    score_axis_affected=("seg", "pose"),
    max_wallclock_seconds=120.0,
    seed=42,
    description=(
        "Toy lanczos upscale example: returns shape-tag only (real impl "
        "wraps cv2.resize / PIL.Image.Resampling.LANCZOS)."
    ),
    lane_id=_EXAMPLE_LANE_ID,
)


@inflate_time_post_filter(
    SuperResolutionUpscaler(spec=_UPSCALE_SPEC).build_contract()
)
def lanczos_upscale_384_to_874_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy upscaler pass — emits frames_v1_upscaled tag (identity payload)."""
    return {
        "frames_v1_upscaled": state.get("frames_v0"),
        "frames_processed": 0,
    }


# ---------------------------------------------------------------------------
# 6. MultiPassInflateRefinement example
# ---------------------------------------------------------------------------

_MULTI_SPEC = MultiPassInflateRefinementSpec(
    pass_id="multi_pass_inflate_7_variants_example",
    num_variants=7,
    surrogate_identifier="example_distilled_surrogate_v0_test_only",
    ranking_criterion="combined_score",
    applies_to_frames="pairs_only",
    score_axis_affected=("seg", "pose"),
    max_wallclock_seconds=600.0,
    seed=42,
    description=(
        "Toy multi-variant refinement example: returns variant 0 unchanged. "
        "Real impl invokes the surrogate to rank N variants."
    ),
    lane_id=_EXAMPLE_LANE_ID,
)


@inflate_time_post_filter(
    MultiPassInflateRefinement(spec=_MULTI_SPEC).build_contract()
)
def multi_pass_inflate_7_variants_example(
    state: Mapping[str, Any],
    *,
    scorer_surrogate: Any = None,
    policy: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Toy multi-variant pass — emits frames_v1_best_variant (variant 0)."""
    return {
        "frames_v1_best_variant": state.get("frames_v0"),
        "variant_selections": [0] * 0,
        "frames_processed": 0,
    }
