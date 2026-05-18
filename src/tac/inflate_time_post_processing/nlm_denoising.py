# SPDX-License-Identifier: MIT
"""NLMDenoisingPostProcessor — non-local-means denoising at inflate time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G inflate-time row 1 (deterministic denoiser / sharpener):

Non-local means (Buades-Coll-Morel 2005) denoises each pixel by averaging
over similar patches across the frame (not just neighboring pixels).
Unlike bilateral filter (which uses spatial proximity), NLM uses
*self-similarity* — useful for highly-textured driving scenes where
PoseNet's feature-consistency benefits from suppressing patch-level
noise without destroying repeating road / lane / curb textures.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the NLM ALGORITHM
is canonical (parametric in patch_size + search_window + h-filter
strength); the substrate-specific choice of these parameters lives in
the decorated function's per-frame invocation. This builder produces the
CONTRACT.

Per spec §G + CLAUDE.md "Strict scorer rule": scorer_free=True; the
denoiser is image-domain only. Default targets PoseNet (h=0.05 is a
mild filter that preserves feature points; aggressive denoising would
wash out the corner descriptors PoseNet uses).
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
)

__all__ = [
    "NLMDenoisingPostProcessor",
    "NLMDenoisingSpec",
]


@dataclass(frozen=True)
class NLMDenoisingSpec:
    """Specification for a single NLM denoising post-processing pass.

    Frozen so spec composition is structurally immutable. NLM's 3
    parameters (patch_size, search_window, h-filter-strength) are
    captured at decoration time; the implementation is deterministic by
    construction.
    """

    pass_id: str
    patch_size: int = 7  # odd; typical 5 / 7 / 9
    search_window: int = 21  # odd; > patch_size; typical 11 / 21 / 35
    h: float = 0.05  # filter strength; lower preserves features
    applies_to_frames: str = "all"
    score_axis_affected: tuple[str, ...] = ("pose",)
    max_wallclock_seconds: float = 180.0  # NLM is slower than bilateral
    seed: int = 42
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError(
                f"patch_size={self.patch_size} must be odd and >= 3"
            )
        if self.search_window < 3 or self.search_window % 2 == 0:
            raise ValueError(
                f"search_window={self.search_window} must be odd and >= 3"
            )
        if self.search_window <= self.patch_size:
            raise ValueError(
                f"search_window={self.search_window} must be > patch_size="
                f"{self.patch_size}"
            )
        if self.h <= 0:
            raise ValueError(f"h={self.h} must be > 0")
        if self.max_wallclock_seconds <= 0:
            raise ValueError(
                f"max_wallclock_seconds={self.max_wallclock_seconds} must be > 0"
            )
        if self.max_wallclock_seconds > MAX_INFLATE_COMPUTE_BUDGET_SECONDS:
            raise ValueError(
                f"max_wallclock_seconds={self.max_wallclock_seconds} exceeds "
                f"the 30-min T4 ceiling ({MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s)"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class NLMDenoisingPostProcessor:
    """Builder for a per-frame NLM denoising contract.

    Usage::

        from tac.inflate_time_post_processing import (
            NLMDenoisingPostProcessor, NLMDenoisingSpec,
            inflate_time_post_filter,
        )

        spec = NLMDenoisingSpec(
            pass_id="nlm_denoise_per_pair_h005",
            patch_size=7,
            search_window=21,
            h=0.05,
            applies_to_frames="pairs_only",
            score_axis_affected=("pose",),
            max_wallclock_seconds=180.0,
            seed=42,
            description=(
                "Per-pair NLM denoising; h=0.05 preserves PoseNet's "
                "feature-point descriptors."
            ),
            lane_id="lane_my_substrate_20260601",
        )
        contract = NLMDenoisingPostProcessor(spec=spec).build_contract()

        @inflate_time_post_filter(contract)
        def nlm_denoise_per_pair_h005(state, *, policy, seed=42):
            # Substrate-specific NLM loop using cv2.fastNlMeansDenoising
            # or sklearn equivalent. The decorated function provides the
            # loop; the contract documents the 3 parameters.
            ...
            return {"frames_v1": ..., "frames_processed": N}
    """

    def __init__(self, *, spec: NLMDenoisingSpec) -> None:
        if not isinstance(spec, NLMDenoisingSpec):
            raise TypeError(
                f"spec must be NLMDenoisingSpec; got {type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> InflateTimePostProcessingContract:
        """Build the InflateTimePostProcessingContract for this NLM pass.

        Emits the canonical pattern:
          - stage_phase="inflate"
          - correction_kind="denoise"
          - correction_resolution="per_pixel" (NLM operates per-pixel
            with patch-level similarity)
          - deterministic=True
          - scorer_free=True (image-domain only)
          - archive_bytes_added=0
          - requires_cpu_only=True (canonical CPU OpenCV impl; GPU
            variants exist but are not required)
        """
        consumes: frozenset[str] = frozenset({"frames_v0"})
        emits: frozenset[str] = frozenset({"frames_v1"})
        return InflateTimePostProcessingContract(
            id=self.spec.pass_id,
            parent_pass_id=None,
            stage_phase="inflate",
            description=(
                self.spec.description
                or (
                    f"NLM denoising; patch_size={self.spec.patch_size} / "
                    f"search_window={self.spec.search_window} / h="
                    f"{self.spec.h} (Buades-Coll-Morel 2005)."
                )
            ),
            consumes=consumes,
            emits=emits,
            correction_kind="denoise",
            correction_resolution="per_pixel",
            applies_to_frames=self.spec.applies_to_frames,
            deterministic=True,
            scorer_free=True,
            max_wallclock_seconds=self.spec.max_wallclock_seconds,
            inflate_compute_budget_seconds=MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
            archive_bytes_added=0,
            score_axis_affected=tuple(self.spec.score_axis_affected),
            requires_scorer_surrogate=False,
            requires_cpu_only=True,
            seed=self.spec.seed,
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
                    "NLM uses self-similarity in the image domain; per-byte "
                    "sensitivity weighting is structurally meaningless at "
                    "inflate time (archive bytes are already frozen)."
                ),
                "hook_bit_allocator_class": (
                    "Inflate-time post-processing does NOT allocate archive "
                    "bytes (archive_bytes_added=0 invariant)."
                ),
                "hook_probe_disambiguator": (
                    "NLM has a single canonical algorithm (Buades-Coll-Morel "
                    "2005); the 3 parameters (patch / window / h) are the "
                    "disambiguation knobs at the spec level."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the NLM algorithm; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-frame loop body "
                "(substrate-specific; provided by the decorated function)."
            ),
        )
