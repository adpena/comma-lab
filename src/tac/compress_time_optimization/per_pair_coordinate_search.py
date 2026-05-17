# SPDX-License-Identifier: MIT
"""PerPairCoordinateSearch — exhaustive per-pair coordinate search across
mode × palette-entry × pose-delta product space at compress time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G compress-time row 3:

  | Per-pair coordinate search | partial (task #466 + #470) | exhaustive
  across mode × palette-entry × pose-delta product | M5 Max fan-out 12-core
  parallel; CPU-bound |

The pattern: for each of the 600 contest pairs, enumerate the Cartesian
product of (K modes) × (M palette entries) × (P pose deltas) candidates
and select the best per a substrate-specific scoring function. CPU-bound;
M5 Max fan-out via ``concurrent.futures.ProcessPoolExecutor``.

This builder canonicalizes the PATTERN; the per-pair scoring function is
substrate-specific (provided by the decorated function). The builder
declares the search-space cardinality at decoration time so the pipeline
can reason about wallclock budget.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: the fan-out
infrastructure (process pool, future collection) is canonical; the
per-pair scoring function is substrate-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)

__all__ = [
    "PerPairCoordinateSearch",
    "PerPairCoordinateSearchSpec",
]


@dataclass(frozen=True)
class PerPairCoordinateSearchSpec:
    """Specification for a per-pair coordinate search pass.

    Frozen so spec composition is structurally immutable. The product-space
    cardinality is captured at decoration time so the pipeline can pre-
    compute the wallclock budget.
    """

    pass_id: str
    num_pairs: int = 600  # contest standard
    num_modes: int = 16   # fec6 K=16
    num_palette_entries: int = 1  # set > 1 to enable palette axis
    num_pose_deltas: int = 1      # set > 1 to enable pose-delta axis
    cpu_parallelism: int = 12     # M5 Max fan-out default
    seed: int = 42
    max_wallclock_seconds: int | None = None
    sensitivity_weighted: bool = False
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.num_pairs < 1:
            raise ValueError(f"num_pairs={self.num_pairs} must be >= 1")
        if self.num_modes < 1:
            raise ValueError(f"num_modes={self.num_modes} must be >= 1")
        if self.num_palette_entries < 1:
            raise ValueError(
                f"num_palette_entries={self.num_palette_entries} must be >= 1"
            )
        if self.num_pose_deltas < 1:
            raise ValueError(
                f"num_pose_deltas={self.num_pose_deltas} must be >= 1"
            )
        if self.cpu_parallelism < 1:
            raise ValueError(
                f"cpu_parallelism={self.cpu_parallelism} must be >= 1"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")

    @property
    def product_space_size_per_pair(self) -> int:
        """Cartesian product cardinality per pair."""
        return (
            self.num_modes
            * self.num_palette_entries
            * self.num_pose_deltas
        )

    @property
    def total_candidate_evaluations(self) -> int:
        """Total candidate evaluations across all pairs."""
        return self.num_pairs * self.product_space_size_per_pair


class PerPairCoordinateSearch:
    """Builder for a per-pair coordinate search pass contract.

    The canonical pattern (per-pair, CPU-parallel):
      For pair_idx in 0..num_pairs:
        candidates = product(modes, palette_entries, pose_deltas)
        best = argmin(scoring_fn(candidate, pair_idx)) over candidates
        emit best.encoding to archive

    The fan-out uses ``concurrent.futures.ProcessPoolExecutor`` with
    ``max_workers=cpu_parallelism`` (default 12 for M5 Max). Per-pair work
    is independent so process-pool parallelism is the canonical pattern;
    the substrate-specific scoring_fn is pickleable per the
    multiprocessing contract.

    Usage::

        from tac.compress_time_optimization import (
            PerPairCoordinateSearch, PerPairCoordinateSearchSpec,
            compress_time_pass,
        )

        spec = PerPairCoordinateSearchSpec(
            pass_id="coord_search_fec6_K16_palette8_pose1",
            num_pairs=600,
            num_modes=16,
            num_palette_entries=8,
            num_pose_deltas=1,
            cpu_parallelism=12,
            seed=42,
            description=(
                "Exhaustive per-pair coordinate search over K=16 × P=8 = 128 "
                "candidates per pair (76800 total evaluations)."
            ),
            lane_id="lane_fec6_coord_search_20260601",
        )
        contract = PerPairCoordinateSearch(spec=spec).build_contract()

        @compress_time_pass(contract)
        def coord_search_fec6_K16_palette8_pose1(state, *, policy, seed):
            # Substrate-specific search body.
            ...
            return {"archive_bytes_v1": ..., "bytes_added": delta_bytes}
    """

    def __init__(self, *, spec: PerPairCoordinateSearchSpec) -> None:
        if not isinstance(spec, PerPairCoordinateSearchSpec):
            raise TypeError(
                f"spec must be PerPairCoordinateSearchSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> CompressTimePassContract:
        """Build the CompressTimePassContract for this coord-search pass.

        Emits the canonical pattern:
          - stage_phase="compress"
          - correction_kind="search"
          - correction_resolution="per_pair"
          - deterministic=True (seed pinned; search is deterministic given
            the scoring function is deterministic)
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
                    f"Per-pair coordinate search; "
                    f"num_pairs={self.spec.num_pairs}, "
                    f"K={self.spec.num_modes} × "
                    f"P={self.spec.num_palette_entries} × "
                    f"Δpose={self.spec.num_pose_deltas} = "
                    f"{self.spec.product_space_size_per_pair} candidates per "
                    f"pair ({self.spec.total_candidate_evaluations} total "
                    f"evaluations); cpu_parallelism={self.spec.cpu_parallelism}."
                )
            ),
            consumes=consumes,
            emits=frozenset({"archive_bytes_v1"}),
            correction_kind="search",
            correction_resolution="per_pair",
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
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_probe_disambiguator": (
                    "Per-pair coordinate search has a canonical exhaustive "
                    "interpretation. The product-space cardinality is the "
                    "design surface."
                ),
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Pass scoring function does not consume "
                            "master_gradient; operator opted for uniform "
                            "per-candidate distortion proxy."
                        )
                    }
                ),
                "hook_bit_allocator_class": (
                    "Coord search emits the index into the existing K × P × "
                    "Δpose codebook; no bit allocation."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the CPU-parallel fan-out infrastructure "
                "(ProcessPoolExecutor + future collection); "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-pair scoring "
                "function (substrate-specific; the decorated function "
                "provides it)."
            ),
        )
