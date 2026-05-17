# SPDX-License-Identifier: MIT
"""SimulatedAnnealingOnDiscreteCodes — SA over discrete codes (selector
indices, palette entries, etc.) at compress time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G compress-time row 4:

  | Simulated annealing on discrete codes | not used | SA on selector
  indices using master gradient as energy landscape | requires
  ``tac.optimization`` SA primitive; not built |

Note (PV-3): ``tac.contrib.cross_disciplinary_optimizers`` has a generic SA
implementation registered in the optimizer factory. The substrate-specific
SA passes built via this contract MAY delegate to that primitive; this
builder canonicalizes the SA-on-discrete-codes PATTERN as a first-class
contract so the cathedral autopilot can rank such passes uniformly.

The discrete-codes target is unique to this namespace's compress-time
context — SA over continuous parameters is covered by GenericTTOHarness.
SA over discrete codes is uniquely suited to selector indices (fec6
K-mode), palette entries (K=16 palette), and quant-bin assignments.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the loop structure is
canonical (SA loop with temperature schedule); the energy_callable is
substrate-specific (the decorated function provides it; if master_gradient
is available it can be used as the energy landscape per §G row 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)

__all__ = [
    "SimulatedAnnealingOnDiscreteCodes",
    "SimulatedAnnealingSpec",
    "LEGAL_SA_TEMP_SCHEDULE",
    "LEGAL_SA_DISCRETE_TARGET",
]

LEGAL_SA_TEMP_SCHEDULE: frozenset[str] = frozenset(
    {
        "exp",        # exponential cooling: T_n = T_0 * alpha^n
        "linear",     # linear cooling: T_n = T_0 - delta * n
        "logarithmic", # log cooling: T_n = T_0 / log(n+2) — Cauchy SA
        "adaptive",   # adaptive cooling: T adjusts based on acceptance rate
    }
)

LEGAL_SA_DISCRETE_TARGET: frozenset[str] = frozenset(
    {
        "selector_indices",   # fec6 K-mode selector
        "palette",            # K=16 palette entries
        "quant_bin_indices",  # quant-bin assignments
        "stream_assignment",  # per-byte stream assignment
    }
)


@dataclass(frozen=True)
class SimulatedAnnealingSpec:
    """Specification for an SA-on-discrete-codes pass.

    Frozen so spec composition is structurally immutable. The temperature
    schedule + step count + seed are pinned at decoration time for
    byte-stable reproducibility per Catalog #158.
    """

    pass_id: str
    discrete_target: str  # one of LEGAL_SA_DISCRETE_TARGET
    temp_schedule: str = "exp"
    initial_temperature: float = 1.0
    cooling_alpha: float = 0.995  # used when temp_schedule="exp"
    num_steps: int = 10000
    seed: int = 42
    max_wallclock_seconds: int | None = None
    sensitivity_weighted: bool = False  # if True, master_gradient is energy
    correction_resolution: str = "per_pair"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.discrete_target not in LEGAL_SA_DISCRETE_TARGET:
            raise ValueError(
                f"discrete_target={self.discrete_target!r} not in "
                f"{sorted(LEGAL_SA_DISCRETE_TARGET)}"
            )
        if self.temp_schedule not in LEGAL_SA_TEMP_SCHEDULE:
            raise ValueError(
                f"temp_schedule={self.temp_schedule!r} not in "
                f"{sorted(LEGAL_SA_TEMP_SCHEDULE)}"
            )
        if self.initial_temperature <= 0:
            raise ValueError(
                f"initial_temperature={self.initial_temperature} must be > 0"
            )
        if not (0 < self.cooling_alpha <= 1.0):
            raise ValueError(
                f"cooling_alpha={self.cooling_alpha} must be in (0, 1.0]"
            )
        if self.num_steps < 1:
            raise ValueError(f"num_steps={self.num_steps} must be >= 1")
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class SimulatedAnnealingOnDiscreteCodes:
    """Builder for a simulated-annealing-on-discrete-codes pass contract.

    The canonical SA loop:
      1. Initialize discrete code vector + temperature T = initial_temperature
      2. For step in range(num_steps):
         a. Propose a neighbor (e.g. flip one selector index)
         b. Compute energy delta ΔE = E(neighbor) - E(current)
         c. Accept with probability min(1, exp(-ΔE / T))
         d. Cool: T_{n+1} per temp_schedule

    If ``sensitivity_weighted=True`` the energy landscape SHOULD be the
    master_gradient at the current code vector (per §G row 4). The
    decorated function decides how to consume master_gradient — the
    contract level only declares the dependency.

    Usage::

        from tac.compress_time_optimization import (
            SimulatedAnnealingOnDiscreteCodes, SimulatedAnnealingSpec,
            compress_time_pass,
        )

        spec = SimulatedAnnealingSpec(
            pass_id="sa_fec6_selector_indices",
            discrete_target="selector_indices",
            temp_schedule="exp",
            initial_temperature=1.0,
            cooling_alpha=0.995,
            num_steps=10000,
            seed=42,
            sensitivity_weighted=True,
            description="SA on fec6 K=16 selector indices using master_gradient.",
            lane_id="lane_fec6_sa_refinement_20260601",
        )
        contract = SimulatedAnnealingOnDiscreteCodes(spec=spec).build_contract()

        @compress_time_pass(contract)
        def sa_fec6_selector_indices(state, *, master_gradient, policy, seed):
            # Substrate-specific SA loop body.
            ...
            return {"archive_bytes_v1": ..., "bytes_added": delta_bytes}
    """

    def __init__(self, *, spec: SimulatedAnnealingSpec) -> None:
        if not isinstance(spec, SimulatedAnnealingSpec):
            raise TypeError(
                f"spec must be SimulatedAnnealingSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> CompressTimePassContract:
        """Build the CompressTimePassContract for this SA pass.

        Emits the canonical pattern:
          - stage_phase="compress"
          - correction_kind="search"
          - deterministic=True (seed pinned; SA randomness derived from seed)
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
                    f"SA on discrete codes; target="
                    f"{self.spec.discrete_target!r}, "
                    f"schedule={self.spec.temp_schedule!r}, "
                    f"T_0={self.spec.initial_temperature}, "
                    f"num_steps={self.spec.num_steps}, seed={self.spec.seed}."
                )
            ),
            consumes=consumes,
            emits=frozenset({"archive_bytes_v1"}),
            correction_kind="search",
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
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "compress_time_optimization_pass_outcomes_v1"
            ),
            hook_probe_disambiguator=(
                "tools/probe_sa_temperature_schedule_disambiguator.py"
            ),
            hook_not_applicable_rationale={
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Pass uses substrate-specific energy callable "
                            "(not master_gradient); operator chose proxy "
                            "distortion energy."
                        )
                    }
                ),
                "hook_bit_allocator_class": (
                    "SA on discrete codes does not directly allocate bits; "
                    "the codes themselves index into a fixed-size table."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the SA loop structure + temp schedule; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the neighbor proposal "
                "and energy callable (substrate-specific; the decorated "
                "function provides them). The temp_schedule disambiguator "
                "(exp vs linear vs log vs adaptive) is a 4-defensible-"
                "interpretation surface, so a probe disambiguator path is "
                "named per CLAUDE.md 'Forbidden premature KILL' discipline."
            ),
        )
