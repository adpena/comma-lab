# SPDX-License-Identifier: MIT
"""SuperResolutionUpscaler — upscale decoded frames from training resolution
to camera resolution at inflate time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G inflate-time + the eval_roundtrip discipline:

The contest pipeline runs ``rgb_to_yuv6`` then resize to (512, 384) for
PoseNet + (384, 512) for SegNet. Renderers typically produce frames at
384x512 (training resolution); the contest scorer evaluates at 874x1164
(camera resolution). The CANONICAL resize is bicubic — but bicubic loses
high-frequency information that PoseNet's FastViT feature extractor uses
for fine-grained pose disambiguation.

SuperResolutionUpscaler swaps bicubic for a learned or higher-fidelity
upscaler at inflate time:

  - "bicubic": canonical baseline (matches eval_roundtrip)
  - "lanczos": sharper than bicubic; no model weights; deterministic
  - "learned": tiny ESPCN-style upscaler distilled offline (Catalog #146
    compliant; weights ship in archive bytes)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the upscaler is
unique to inflate-time post-processing (compress-time has no analog
because compress-time operates on archive bytes, not decoded frames).
The 3 algorithm choices are explicit; deterministic for all 3.

Per spec §G: avoids the eval_roundtrip resize step's signal loss without
modifying archive bytes (archive_bytes_added=0 for the bicubic / lanczos
variants; for "learned" the weights are part of archive bytes via the
COMPRESS-time grammar, NOT added at inflate).
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
)

__all__ = [
    "LEGAL_UPSCALER_KIND",
    "SuperResolutionUpscaler",
    "SuperResolutionUpscalerSpec",
]


LEGAL_UPSCALER_KIND: frozenset[str] = frozenset(
    {
        "bicubic",   # canonical baseline (matches eval_roundtrip)
        "lanczos",   # sharper; no model weights; deterministic
        "learned",   # ESPCN-style learned upscaler (weights in archive)
    }
)


@dataclass(frozen=True)
class SuperResolutionUpscalerSpec:
    """Specification for the resolution upscaler at inflate time.

    Frozen so spec composition is structurally immutable. The 3 upscaler
    kinds are mutually exclusive; the choice is captured at decoration
    time so the pipeline composer can detect ambiguous double-upscale
    pipelines.
    """

    pass_id: str
    upscaler_kind: str = "lanczos"
    input_shape: tuple[int, int] = (384, 512)  # (H, W) at training res
    output_shape: tuple[int, int] = (874, 1164)  # (H, W) at camera res
    learned_model_identifier: str | None = None  # required iff kind == "learned"
    applies_to_frames: str = "all"
    score_axis_affected: tuple[str, ...] = ("seg", "pose")
    max_wallclock_seconds: float = 120.0
    seed: int = 42
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.upscaler_kind not in LEGAL_UPSCALER_KIND:
            raise ValueError(
                f"upscaler_kind={self.upscaler_kind!r} not in "
                f"{sorted(LEGAL_UPSCALER_KIND)}"
            )
        for fname, shape in (
            ("input_shape", self.input_shape),
            ("output_shape", self.output_shape),
        ):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any((not isinstance(d, int) or d < 1) for d in shape)
            ):
                raise ValueError(
                    f"{fname}={shape!r} must be a 2-tuple of positive ints "
                    f"(H, W)"
                )
        if (
            self.output_shape[0] <= self.input_shape[0]
            or self.output_shape[1] <= self.input_shape[1]
        ):
            raise ValueError(
                f"output_shape={self.output_shape} must be strictly larger "
                f"than input_shape={self.input_shape} along both axes (this "
                f"is an UPSCALER; for downscale use SegNet's existing "
                f"x[:, -1, ...] slice)."
            )
        if self.upscaler_kind == "learned":
            if (
                not isinstance(self.learned_model_identifier, str)
                or not self.learned_model_identifier.strip()
            ):
                raise ValueError(
                    f"learned_model_identifier must be a non-empty string "
                    f"when upscaler_kind='learned'; got "
                    f"{self.learned_model_identifier!r}"
                )
        else:
            if self.learned_model_identifier is not None:
                raise ValueError(
                    f"learned_model_identifier={self.learned_model_identifier!r} "
                    f"must be None when upscaler_kind="
                    f"{self.upscaler_kind!r} (only 'learned' kind ships a model)"
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


class SuperResolutionUpscaler:
    """Builder for a resolution-upscaler inflate-time pass contract.

    Usage::

        from tac.inflate_time_post_processing import (
            SuperResolutionUpscaler, SuperResolutionUpscalerSpec,
            inflate_time_post_filter,
        )

        spec = SuperResolutionUpscalerSpec(
            pass_id="lanczos_upscale_384_to_874",
            upscaler_kind="lanczos",
            input_shape=(384, 512),
            output_shape=(874, 1164),
            applies_to_frames="all",
            score_axis_affected=("seg", "pose"),
            max_wallclock_seconds=120.0,
            seed=42,
            description=(
                "Lanczos upscale 384x512 → 874x1164; sharper than the "
                "canonical bicubic eval_roundtrip resize without learned "
                "model weights."
            ),
            lane_id="lane_my_substrate_20260601",
        )
        contract = SuperResolutionUpscaler(spec=spec).build_contract()

        @inflate_time_post_filter(contract)
        def lanczos_upscale_384_to_874(state, *, policy, seed=42):
            # Substrate-specific cv2.resize(interpolation=cv2.INTER_LANCZOS4)
            # or PIL.Image.Resampling.LANCZOS loop.
            ...
            return {"frames_v1": ..., "frames_processed": N}
    """

    def __init__(self, *, spec: SuperResolutionUpscalerSpec) -> None:
        if not isinstance(spec, SuperResolutionUpscalerSpec):
            raise TypeError(
                f"spec must be SuperResolutionUpscalerSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> InflateTimePostProcessingContract:
        """Build the InflateTimePostProcessingContract for this upscaler.

        Emits the canonical pattern:
          - stage_phase="inflate"
          - correction_kind="upscale"
          - correction_resolution="per_frame"
          - deterministic=True (all 3 upscaler kinds are deterministic)
          - scorer_free=True
          - archive_bytes_added=0
          - requires_scorer_surrogate=False
          - requires_cpu_only=True
        """
        consumes_set: set[str] = {"frames_v0"}
        if self.spec.upscaler_kind == "learned":
            consumes_set.add("learned_upscaler_weights")
        consumes: frozenset[str] = frozenset(consumes_set)
        emits: frozenset[str] = frozenset({"frames_v1_upscaled"})

        return InflateTimePostProcessingContract(
            id=self.spec.pass_id,
            parent_pass_id=None,
            stage_phase="inflate",
            description=(
                self.spec.description
                or (
                    f"{self.spec.upscaler_kind.title()} upscale "
                    f"{self.spec.input_shape} → {self.spec.output_shape} "
                    f"(max_wallclock={self.spec.max_wallclock_seconds}s)."
                )
            ),
            consumes=consumes,
            emits=emits,
            correction_kind="upscale",
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
                    "Upscaler is a uniform deterministic kernel (bicubic / "
                    "lanczos / learned); per-byte sensitivity weighting is "
                    "meaningless at inflate time."
                ),
                "hook_bit_allocator_class": (
                    "Inflate-time post-processing does NOT allocate archive "
                    "bytes (archive_bytes_added=0 invariant). For the "
                    "'learned' upscaler kind the model weights ship via the "
                    "COMPRESS-time grammar."
                ),
                "hook_probe_disambiguator": (
                    "3 canonical upscaler kinds (bicubic / lanczos / learned) "
                    "are explicit at the spec level; the operator chooses one "
                    "per dispatch."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the upscaler infrastructure (3 kinds: "
                "bicubic / lanczos / learned); FORK_BECAUSE_PRINCIPLED_MISMATCH "
                "for the per-frame loop body (substrate-specific; the "
                "decorated function provides the cv2 / PIL / torch invocation "
                "appropriate to the inflate runtime)."
            ),
        )
