# SPDX-License-Identifier: MIT
"""BilateralFilterPostProcessor — edge-preserving smoothing applied to
decoded frames.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G inflate-time row 1 (Per-frame post-processing):

  | deterministic denoiser / sharpener / wavelet post-filter | per-frame
  post-filter network baked into inflate.py per Catalog #146 |

The bilateral filter is the canonical image-domain prior at inflate time —
it smooths noise WITHOUT crossing edges (parametrized by sigma_spatial +
sigma_intensity). SegNet's stride-2 EfficientNet-B2 stem loses
half-resolution boundary information; per-frame bilateral filtering at
inflate time can stabilize the argmax decision at class boundaries
without modifying any archive bytes.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the bilateral
KERNEL is canonical (parametric in 2 sigmas); the substrate-specific
choice of sigma values lives in the decorated function's loss-callable
analog (or in the operator's smoke-sweep). This builder produces the
CONTRACT; the substrate-specific function provides the per-frame filter
invocation.

Per spec §G + CLAUDE.md "Strict scorer rule": all parameters are static
(no scorer access; image-domain prior only). The default sigma_spatial=2.0
+ sigma_intensity=0.1 are conservative values aligned with typical 384x512
contest-resolution dashcam noise levels.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
)

__all__ = [
    "BilateralFilterPostProcessor",
    "BilateralFilterSpec",
]


@dataclass(frozen=True)
class BilateralFilterSpec:
    """Specification for a single bilateral-filter post-processing pass.

    Frozen so spec composition is structurally immutable. The 2 sigmas +
    kernel diameter are captured at decoration time; the per-frame filter
    is deterministic by construction (no randomness; identical input →
    identical output).
    """

    pass_id: str
    sigma_spatial: float = 2.0
    sigma_intensity: float = 0.1
    kernel_diameter: int = 5  # odd, >= 3
    applies_to_frames: str = "all"
    score_axis_affected: tuple[str, ...] = ("seg",)
    max_wallclock_seconds: float = 60.0
    seed: int = 42
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.sigma_spatial <= 0:
            raise ValueError(
                f"sigma_spatial={self.sigma_spatial} must be > 0"
            )
        if self.sigma_intensity <= 0:
            raise ValueError(
                f"sigma_intensity={self.sigma_intensity} must be > 0"
            )
        if self.kernel_diameter < 3 or self.kernel_diameter % 2 == 0:
            raise ValueError(
                f"kernel_diameter={self.kernel_diameter} must be odd and >= 3"
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


class BilateralFilterPostProcessor:
    """Builder for a per-frame bilateral-filter post-processing contract.

    Usage::

        from tac.inflate_time_post_processing import (
            BilateralFilterPostProcessor, BilateralFilterSpec,
            inflate_time_post_filter,
        )

        spec = BilateralFilterSpec(
            pass_id="bilateral_denoise_per_frame_sigma2",
            sigma_spatial=2.0,
            sigma_intensity=0.1,
            kernel_diameter=5,
            applies_to_frames="all",
            score_axis_affected=("seg",),
            max_wallclock_seconds=60.0,
            seed=42,
            description=(
                "Per-frame bilateral filter; sigma_spatial=2 / "
                "sigma_intensity=0.1 / kernel=5x5 for SegNet boundary "
                "stability."
            ),
            lane_id="lane_my_substrate_20260601",
        )
        contract = BilateralFilterPostProcessor(spec=spec).build_contract()

        @inflate_time_post_filter(contract)
        def bilateral_denoise_per_frame_sigma2(state, *, policy, seed=42):
            # Substrate-specific per-frame loop using cv2.bilateralFilter
            # or torch.nn.functional equivalent. The decorated function
            # provides the loop; the contract documents the parameters.
            ...
            return {"frames_v1": ..., "frames_processed": N}

    The builder does NOT execute the filter — it produces the CONTRACT.
    This separation is the canonical "infrastructure vs engineering" split
    per CLAUDE.md HNeRV parity discipline L7.
    """

    def __init__(self, *, spec: BilateralFilterSpec) -> None:
        if not isinstance(spec, BilateralFilterSpec):
            raise TypeError(
                f"spec must be BilateralFilterSpec; got {type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> InflateTimePostProcessingContract:
        """Build the InflateTimePostProcessingContract for this filter.

        Emits the canonical pattern:
          - stage_phase="inflate"
          - correction_kind="denoise"
          - correction_resolution="per_frame"
          - deterministic=True (per-frame filter is deterministic by
            construction; seed pinned for any rng-using extensions)
          - scorer_free=True (image-domain prior only; no scorer access)
          - archive_bytes_added=0 (no archive mutation at inflate)
          - requires_cpu_only=True (typical CPU implementation; the GPU
            implementation is available but not required)
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
                    f"Per-frame bilateral filter; sigma_spatial="
                    f"{self.spec.sigma_spatial} / sigma_intensity="
                    f"{self.spec.sigma_intensity} / kernel="
                    f"{self.spec.kernel_diameter}x{self.spec.kernel_diameter}."
                )
            ),
            consumes=consumes,
            emits=emits,
            correction_kind="denoise",
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
                    "Bilateral filter uses a uniform image-domain prior "
                    "(2 sigmas); per-byte sensitivity weighting is "
                    "structurally meaningless at this stage (the archive "
                    "bytes are already frozen)."
                ),
                "hook_bit_allocator_class": (
                    "Inflate-time post-processing does NOT allocate archive "
                    "bytes (archive_bytes_added=0 invariant). Bit allocation "
                    "lives at compress time per tac.compress_time_optimization."
                ),
                "hook_probe_disambiguator": (
                    "Bilateral filter has a single canonical interpretation "
                    "(2-sigma edge-preserving smoother); per-pass disambiguation "
                    "lives in the operator's sigma-sweep at the spec level."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the 2-sigma bilateral kernel; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-frame loop body "
                "(substrate-specific; provided by the decorated function)."
            ),
        )
