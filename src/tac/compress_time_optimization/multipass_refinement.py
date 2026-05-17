# SPDX-License-Identifier: MIT
"""MultipassRefinement — canonical loop primitive for iterative refinement.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G compress-time row 2:

  | Multipass refinement | single-pass build | `Lane 8 multi-pass`
  (iterative: quantize → measure → re-quantize residual → re-measure) |
  `Lane 8` Level 3 on different substrate; not yet applied to fec6 |

The canonical pattern is N-pass: at each pass, quantize the parameter set
(or substrate-specific state), measure the residual / distortion / proxy
loss, then re-quantize the residual with refined precision. Stages N+1
consume the prior stage's residual; the cascade terminates when residual
norm drops below a threshold OR N passes are exhausted.

This builder is the substrate-agnostic loop primitive. The sister
``tac.multipass_compressor`` (PV-3) is a SUBSTRATE-SPECIFIC compressor with
its own loop logic; future op-routable: deduplicate that into this primitive
via a wave-3 cleanup once both have empirical anchors.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the loop's STRUCTURE
is canonical; the per-pass ``refine_step_fn`` is substrate-specific (the
decorated function provides it).
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)

__all__ = [
    "MultipassRefinement",
    "MultipassRefinementSpec",
]


@dataclass(frozen=True)
class MultipassRefinementSpec:
    """Specification for a multipass refinement pass.

    Frozen so spec composition is structurally immutable. The depth N is
    captured at decoration time; the loop count is fixed for byte-stable
    archive reproducibility (per Catalog #158).
    """

    pass_id: str
    depth: int  # number of refinement passes (>= 1)
    residual_termination_threshold: float | None = None  # None = always run N
    seed: int = 42
    max_wallclock_seconds: int | None = None
    sensitivity_weighted: bool = False
    correction_resolution: str = "per_pair"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth={self.depth} must be >= 1")
        if self.residual_termination_threshold is not None:
            if self.residual_termination_threshold < 0:
                raise ValueError(
                    f"residual_termination_threshold="
                    f"{self.residual_termination_threshold} must be >= 0 or None"
                )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class MultipassRefinement:
    """Builder for an N-pass multipass refinement contract.

    The empirical anchor (Lane 8 multi-pass) is::

        Pass 1: quantize → measure residual_v1
        Pass 2: quantize residual_v1 with refined precision → residual_v2
        ...
        Pass N: quantize residual_v{N-1} → residual_vN

    The terminal archive concatenates the per-pass residual encodings.
    Each pass's emit key is versioned (``_v<N>``) so the pipeline's
    ambiguous-emit detector accepts the multipass cascade as structurally
    unambiguous (each version consumed by exactly one downstream pass).

    Usage::

        from tac.compress_time_optimization import (
            MultipassRefinement, MultipassRefinementSpec, compress_time_pass,
        )

        spec = MultipassRefinementSpec(
            pass_id="multipass_quant_depth_3",
            depth=3,
            residual_termination_threshold=1e-4,
            seed=42,
            correction_resolution="per_block",
            description=(
                "Depth-3 iterative quant refinement; each pass re-quantizes "
                "prior pass's residual with refined precision."
            ),
            lane_id="lane_my_substrate_20260601",
        )
        contract = MultipassRefinement(spec=spec).build_contract()

        @compress_time_pass(contract)
        def multipass_quant_depth_3(state, *, policy, seed):
            # Substrate-specific N-pass quant loop. The decorated function
            # implements the loop body; the contract documents the
            # termination / depth / seed.
            ...
            return {
                "archive_bytes_v1": ...,
                "bytes_added": delta_bytes,
                "residual_norm_final": ...,
            }
    """

    def __init__(self, *, spec: MultipassRefinementSpec) -> None:
        if not isinstance(spec, MultipassRefinementSpec):
            raise TypeError(
                f"spec must be MultipassRefinementSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> CompressTimePassContract:
        """Build the CompressTimePassContract for this multipass pass.

        Emits the canonical pattern:
          - stage_phase="compress"
          - correction_kind="refinement"
          - deterministic=True (seed pinned on contract; depth pinned)
          - scorer_free=True
        """
        consumes: frozenset[str] = frozenset({"archive_bytes_v0"})
        if self.spec.sensitivity_weighted:
            consumes = consumes | frozenset({"master_gradient"})
        return CompressTimePassContract(
            id=self.spec.pass_id,
            parent_pass_id=None,
            stage_phase="compress",
            description=(
                self.spec.description
                or (
                    f"Depth-{self.spec.depth} multipass refinement; "
                    f"each pass re-quantizes prior pass's residual "
                    f"(termination_threshold="
                    f"{self.spec.residual_termination_threshold!r})."
                )
            ),
            consumes=consumes,
            emits=frozenset(
                {
                    "archive_bytes_v1",
                    "residual_norm_final",
                }
            ),
            correction_kind="refinement",
            correction_resolution=self.spec.correction_resolution,
            deterministic=True,
            scorer_free=True,
            sensitivity_weighted=self.spec.sensitivity_weighted,
            max_wallclock_seconds=self.spec.max_wallclock_seconds,
            seed=self.spec.seed,
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution=(
                "master_gradient_v1"
                if self.spec.sensitivity_weighted
                else "not_applicable_with_rationale"
            ),
            hook_pareto_constraint="rate_distortion_v1",
            hook_bit_allocator_class="iterated_bisection",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "compress_time_optimization_pass_outcomes_v1"
            ),
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_probe_disambiguator": (
                    "Multipass refinement has a canonical iterative interpretation "
                    "(quantize → measure residual → re-quantize). The depth + "
                    "termination_threshold are the disambiguation knobs at the "
                    "spec level."
                ),
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Pass does not consume master_gradient; operator "
                            "opted for uniform per-residual precision."
                        )
                    }
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the N-pass refinement loop; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-pass "
                "quantization function (substrate-specific; provided by the "
                "decorated function)."
            ),
        )
