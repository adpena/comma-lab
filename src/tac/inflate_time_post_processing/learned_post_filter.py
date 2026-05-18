# SPDX-License-Identifier: MIT
"""LearnedPostFilterApplier — apply a CPU-only learned post-filter model
at inflate time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G inflate-time row 1 (per-frame post-filter network baked into inflate.py
per Catalog #146):

  | per-frame post-filter network baked into inflate.py per Catalog #146;
  ~$0 wiring |

The learned post-filter is a tiny CPU-trained nn.Module (typically
distilled OFFLINE from a larger teacher; e.g. Hinton-style distillation
of an EfficientNet-B0 denoiser into a 4-block convolutional student).
The student's weights ship as PART of the archive bytes via the
COMPRESS-time archive grammar — NOT loaded ad-hoc at inflate (which
would violate CLAUDE.md "Strict scorer rule").

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the canonical
contract is the per-frame invocation pattern + the (B, 3, H, W) →
(B, 3, H, W) shape preservation. The decorated function supplies the
model instance + the specific distillation lineage.

The contract sets requires_scorer_surrogate=False because a learned
post-filter is NOT a scorer surrogate — it's an image-to-image filter.
Scorer surrogates (Hinton-distilled SegNet/PoseNet for variant ranking)
are the domain of MultiPassInflateRefinement.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
)

__all__ = [
    "LearnedPostFilterApplier",
    "LearnedPostFilterSpec",
]


@dataclass(frozen=True)
class LearnedPostFilterSpec:
    """Specification for applying a learned post-filter model.

    Frozen so spec composition is structurally immutable. The model
    weights themselves are NOT captured here — they ship via the
    archive's grammar. The model_identifier field names which model the
    inflate.py will load from the archive's slot.
    """

    pass_id: str
    model_identifier: str  # e.g. "distilled_unet_4block_v2_2025q4"
    expected_input_shape: tuple[int, int, int, int] = (1, 3, 384, 512)
    expected_output_shape: tuple[int, int, int, int] = (1, 3, 384, 512)
    applies_to_frames: str = "all"
    score_axis_affected: tuple[str, ...] = ("seg", "pose")
    max_wallclock_seconds: float = 240.0  # CPU inference budget per pass
    seed: int = 42
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_identifier, str)
            or not self.model_identifier.strip()
        ):
            raise ValueError(
                f"model_identifier must be a non-empty string; got "
                f"{self.model_identifier!r}"
            )
        for fname, shape in (
            ("expected_input_shape", self.expected_input_shape),
            ("expected_output_shape", self.expected_output_shape),
        ):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 4
                or any((not isinstance(d, int) or d < 1) for d in shape)
            ):
                raise ValueError(
                    f"{fname}={shape!r} must be a 4-tuple of positive ints "
                    f"(B, C, H, W)"
                )
        if self.expected_input_shape != self.expected_output_shape:
            # Shape preservation is the canonical contract for image-to-
            # image post-filters; non-preserving models belong in the
            # SuperResolutionUpscaler builder which has its own contract.
            raise ValueError(
                f"expected_input_shape={self.expected_input_shape} must "
                f"equal expected_output_shape={self.expected_output_shape}. "
                f"Use SuperResolutionUpscaler for shape-changing models."
            )
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


class LearnedPostFilterApplier:
    """Builder for a learned-post-filter inflate-time pass contract.

    Usage::

        from tac.inflate_time_post_processing import (
            LearnedPostFilterApplier, LearnedPostFilterSpec,
            inflate_time_post_filter,
        )

        spec = LearnedPostFilterSpec(
            pass_id="learned_unet_4block_per_frame",
            model_identifier="distilled_unet_4block_v2_2025q4",
            expected_input_shape=(1, 3, 384, 512),
            expected_output_shape=(1, 3, 384, 512),
            applies_to_frames="all",
            score_axis_affected=("seg", "pose"),
            max_wallclock_seconds=240.0,
            seed=42,
            description=(
                "4-block U-Net distilled offline from EfficientNet-B0 "
                "denoiser teacher; CPU inference at 384x512."
            ),
            lane_id="lane_my_substrate_20260601",
        )
        contract = LearnedPostFilterApplier(spec=spec).build_contract()

        @inflate_time_post_filter(contract)
        def learned_unet_4block_per_frame(state, *, policy, seed=42):
            # Substrate-specific model.eval() + per-frame forward pass.
            # The decorated function loads the model from the archive
            # bytes (Catalog #146 compliant) and applies it.
            ...
            return {"frames_v1": ..., "frames_processed": N}
    """

    def __init__(self, *, spec: LearnedPostFilterSpec) -> None:
        if not isinstance(spec, LearnedPostFilterSpec):
            raise TypeError(
                f"spec must be LearnedPostFilterSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> InflateTimePostProcessingContract:
        """Build the InflateTimePostProcessingContract for this pass.

        Emits the canonical pattern:
          - stage_phase="inflate"
          - correction_kind="refine"  (learned filter ≠ deterministic kernel)
          - correction_resolution="per_frame"
          - deterministic=True (model.eval() + no rng IS deterministic)
          - scorer_free=True (learned filter is NOT a scorer)
          - archive_bytes_added=0 (model weights are part of archive bytes
            via the COMPRESS-time grammar, not added at inflate)
          - requires_scorer_surrogate=False (this is an image filter, not
            a variant ranker)
          - requires_cpu_only=True (inflate runtime is CPU; Catalog #146
            inflate.py budget ≤ 100 LOC + ≤ 2 deps)
        """
        consumes: frozenset[str] = frozenset(
            {"frames_v0", "learned_post_filter_weights"}
        )
        emits: frozenset[str] = frozenset({"frames_v1"})
        return InflateTimePostProcessingContract(
            id=self.spec.pass_id,
            parent_pass_id=None,
            stage_phase="inflate",
            description=(
                self.spec.description
                or (
                    f"Apply learned post-filter {self.spec.model_identifier!r} "
                    f"at shape {self.spec.expected_input_shape} (CPU; "
                    f"max_wallclock={self.spec.max_wallclock_seconds}s)."
                )
            ),
            consumes=consumes,
            emits=emits,
            correction_kind="refine",
            correction_resolution="per_frame",
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
                    "Learned post-filter applies a fixed model; per-byte "
                    "sensitivity weighting is meaningless at inflate time. "
                    "(The model itself MAY have been trained with master-"
                    "gradient supervision OFFLINE; that is captured in the "
                    "compress-time training curriculum, not here.)"
                ),
                "hook_bit_allocator_class": (
                    "Inflate-time post-processing does NOT allocate archive "
                    "bytes (archive_bytes_added=0 invariant). The model's "
                    "weight budget was paid at compress time via the archive "
                    "grammar."
                ),
                "hook_probe_disambiguator": (
                    "Model identity is the disambiguation; multiple "
                    "candidate models live in the operator's offline "
                    "distillation pipeline, not here."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the per-frame application loop; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the model itself "
                "(substrate-specific; distilled offline from a teacher; the "
                "decorated function loads it from the archive bytes per "
                "Catalog #146)."
            ),
        )
