# SPDX-License-Identifier: MIT
"""IteratedBisectionRateKnee — per-tensor / per-block iterated bisection
on the rate-distortion knee at compress time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G compress-time row 6:

  | Iterated bisection on quantization knee | not used | per-tensor
  bisection on int8 scale + per-block-within-tensor scale | trivial CPU
  fan-out |

The pattern: for each (tensor, block) pair, bisect over a 1-D scale
parameter (e.g. int8 quantization scale) to find the point where rate /
distortion crosses a threshold. Iterated = multiple bisection passes over
the parameter set, each refining the previous estimate.

This builder canonicalizes the bisection PATTERN as a first-class
compress-time pass contract. The per-tensor scoring + scale-space
definition is substrate-specific (provided by the decorated function).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the bisection LOOP
(low / high / mid / test / refine) is canonical; the rate-distortion
evaluator at each bisection point is substrate-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)

__all__ = [
    "IteratedBisectionRateKnee",
    "IteratedBisectionRateKneeSpec",
    "LEGAL_BISECTION_GRANULARITY",
]

LEGAL_BISECTION_GRANULARITY: frozenset[str] = frozenset(
    {
        "per_tensor",  # one bisection per tensor (e.g. int8 scale)
        "per_block",   # per-block-within-tensor (block-FP analog)
        "per_channel", # per-channel within tensor (per Catalog #170 LSQ pattern)
    }
)


@dataclass(frozen=True)
class IteratedBisectionRateKneeSpec:
    """Specification for an iterated-bisection-on-R-D-knee pass.

    Frozen so spec composition is structurally immutable. The bisection
    parameters (max_iterations + convergence_tolerance) are pinned at
    decoration time for byte-stable reproducibility per Catalog #158.
    """

    pass_id: str
    granularity: str  # one of LEGAL_BISECTION_GRANULARITY
    num_outer_iterations: int = 4  # number of full sweep passes
    max_inner_iterations: int = 20  # max bisection iters per element
    convergence_tolerance: float = 1e-4
    scale_range_log10: tuple[float, float] = (-3.0, 1.0)  # log10 scale bounds
    seed: int = 42
    max_wallclock_seconds: int | None = None
    sensitivity_weighted: bool = False
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.granularity not in LEGAL_BISECTION_GRANULARITY:
            raise ValueError(
                f"granularity={self.granularity!r} not in "
                f"{sorted(LEGAL_BISECTION_GRANULARITY)}"
            )
        if self.num_outer_iterations < 1:
            raise ValueError(
                f"num_outer_iterations={self.num_outer_iterations} must be "
                f">= 1"
            )
        if self.max_inner_iterations < 1:
            raise ValueError(
                f"max_inner_iterations={self.max_inner_iterations} must be "
                f">= 1"
            )
        if self.convergence_tolerance <= 0:
            raise ValueError(
                f"convergence_tolerance={self.convergence_tolerance} must be "
                f"> 0"
            )
        if len(self.scale_range_log10) != 2:
            raise ValueError(
                f"scale_range_log10={self.scale_range_log10!r} must be "
                f"(low, high)"
            )
        low, high = self.scale_range_log10
        if low >= high:
            raise ValueError(
                f"scale_range_log10={self.scale_range_log10!r} must satisfy "
                f"low < high"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class IteratedBisectionRateKnee:
    """Builder for an iterated-bisection-on-R-D-knee pass contract.

    The canonical iterated-bisection algorithm:
      For outer_iter in range(num_outer_iterations):
        For each element (tensor / block / channel) in parameter set:
          low, high = scale_range_log10
          for inner_iter in range(max_inner_iterations):
            mid = (low + high) / 2
            rate, distortion = evaluate(element, scale=10**mid)
            if at_knee(rate, distortion, tolerance):
              break
            elif over_rate_budget: high = mid
            else: low = mid

    The outer iteration allows the bisection to re-converge after other
    elements' scales have settled (Lagrangian-style coupling across the
    parameter set). The substrate-specific ``evaluate`` function provides
    the rate / distortion measurement; the bisection LOOP is canonical.

    Usage::

        from tac.compress_time_optimization import (
            IteratedBisectionRateKnee, IteratedBisectionRateKneeSpec,
            compress_time_pass,
        )

        spec = IteratedBisectionRateKneeSpec(
            pass_id="bisect_int8_scale_per_block",
            granularity="per_block",
            num_outer_iterations=4,
            max_inner_iterations=20,
            convergence_tolerance=1e-4,
            scale_range_log10=(-3.0, 1.0),
            seed=42,
            description=(
                "Per-block int8 scale bisection; 4 outer iterations across "
                "the parameter set with up to 20 inner bisections per block."
            ),
            lane_id="lane_my_substrate_bisect_20260601",
        )
        contract = IteratedBisectionRateKnee(spec=spec).build_contract()

        @compress_time_pass(contract)
        def bisect_int8_scale_per_block(state, *, policy, seed):
            # Substrate-specific bisection body (provides per-block
            # rate + distortion evaluator).
            ...
            return {"archive_bytes_v1": ..., "bytes_added": delta_bytes}
    """

    def __init__(self, *, spec: IteratedBisectionRateKneeSpec) -> None:
        if not isinstance(spec, IteratedBisectionRateKneeSpec):
            raise TypeError(
                f"spec must be IteratedBisectionRateKneeSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> CompressTimePassContract:
        """Build the CompressTimePassContract for this bisection pass.

        Emits the canonical pattern:
          - stage_phase="compress"
          - correction_kind="bisection"
          - deterministic=True (seed pinned; bisection is deterministic
            given the evaluator is deterministic)
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
                    f"Iterated bisection on R-D knee; "
                    f"granularity={self.spec.granularity!r}, "
                    f"outer_iters={self.spec.num_outer_iterations}, "
                    f"max_inner_iters={self.spec.max_inner_iterations}, "
                    f"convergence_tol={self.spec.convergence_tolerance}, "
                    f"scale_range_log10={self.spec.scale_range_log10!r}."
                )
            ),
            consumes=consumes,
            emits=frozenset({"archive_bytes_v1"}),
            correction_kind="bisection",
            correction_resolution=(
                "per_tensor"
                if self.spec.granularity == "per_tensor"
                else "per_block"
            ),
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
                    "Iterated bisection has a canonical 1-D-bisection "
                    "interpretation; the granularity (per_tensor / per_block "
                    "/ per_channel) is the design surface."
                ),
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Bisection evaluator uses substrate-specific "
                            "rate + distortion proxies (not master_gradient)."
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
                "ADOPT_CANONICAL for the bisection loop structure "
                "(low/high/mid/test); FORK_BECAUSE_PRINCIPLED_MISMATCH for "
                "the per-element rate-distortion evaluator (substrate-"
                "specific; the decorated function provides it)."
            ),
        )
