# SPDX-License-Identifier: MIT
"""Canonical example compress-time passes — one per builder.

Six minimal passes that exercise the decorator + composition API:

  1. ``raw_quant_example``                         : seed/passthrough stage
  2. ``tto_pose_per_pair_example``                 : GenericTTOHarness sample
  3. ``multipass_quant_depth_3_example``           : MultipassRefinement sample
  4. ``sa_fec6_selector_indices_example``          : SimulatedAnnealing sample
  5. ``coord_search_fec6_k16_palette8_example``    : PerPairCoordinateSearch sample
  6. ``bisect_int8_scale_per_block_example``       : IteratedBisectionRateKnee sample

The example bodies are TOY (zero residual / fixed bytes_added) so the
passes are testable without GPU. Real consumers replace the body with
substrate-specific logic.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in the
docstrings is backed by an executable body.
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)
from tac.compress_time_optimization.decorator import compress_time_pass
from tac.compress_time_optimization.generic_tto_harness import (
    GenericTTOHarness,
    GenericTTOHarnessSpec,
)
from tac.compress_time_optimization.iterated_bisection import (
    IteratedBisectionRateKnee,
    IteratedBisectionRateKneeSpec,
)
from tac.compress_time_optimization.multipass_refinement import (
    MultipassRefinement,
    MultipassRefinementSpec,
)
from tac.compress_time_optimization.per_pair_coordinate_search import (
    PerPairCoordinateSearch,
    PerPairCoordinateSearchSpec,
)
from tac.compress_time_optimization.simulated_annealing import (
    SimulatedAnnealingOnDiscreteCodes,
    SimulatedAnnealingSpec,
)


# Lane id shared across the example passes for provenance.
_EXAMPLE_LANE_ID = (
    "lane_tac_compress_time_optimization_namespace_decorator_api_20260517"
)
_EXAMPLE_DESIGN_MEMO = (
    ".omx/research/"
    "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
)


# ---------------------------------------------------------------------------
# 1. Seed / passthrough stage (raw quant placeholder)
# ---------------------------------------------------------------------------


@compress_time_pass(
    CompressTimePassContract(
        id="raw_quant_example",
        parent_pass_id=None,
        stage_phase="compress",
        description=(
            "Seed/passthrough stage: emits archive_bytes_v0 from the seed "
            "state's seed_archive key. Used as the root of example pipelines."
        ),
        consumes=frozenset({"seed_archive"}),
        emits=frozenset({"archive_bytes_v0"}),
        correction_kind="transform",
        correction_resolution="global",
        deterministic=True,
        scorer_free=True,
        sensitivity_weighted=False,
        max_wallclock_seconds=1,
        seed=42,
        merge_policy="last_writer_wins",
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="rate_distortion_v1",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind=(
            "compress_time_optimization_pass_outcomes_v1"
        ),
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "Passthrough emits zero residual; sensitivity weighting "
                "is meaningless at the seed."
            ),
            "hook_bit_allocator_class": (
                "Passthrough emits zero bytes; bit allocation is undefined."
            ),
            "hook_probe_disambiguator": (
                "Seed-passthrough has a single canonical interpretation."
            ),
        },
        lane_id=_EXAMPLE_LANE_ID,
        design_memo=_EXAMPLE_DESIGN_MEMO,
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL: seed stage is identical across substrates "
            "(emits raw archive bytes; zero residual)."
        ),
    )
)
def raw_quant_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Seed stage: emits archive_bytes_v0 unchanged."""
    seed_archive = state.get("seed_archive")
    if seed_archive is None:
        raise ValueError(
            "raw_quant_example requires seed_archive in state; got "
            f"{sorted(state.keys())}"
        )
    return {
        "archive_bytes_v0": seed_archive,
        "bytes_added": 0,
    }


# ---------------------------------------------------------------------------
# 2. GenericTTOHarness sample
# ---------------------------------------------------------------------------

_tto_pose_per_pair_example_contract = GenericTTOHarness(
    spec=GenericTTOHarnessSpec(
        pass_id="tto_pose_per_pair_example",
        target_kind="parameter_tensor",
        optimizer="adamw",
        learning_rate=1e-3,
        num_steps=10,  # toy: 10 steps so test runs in <1ms
        seed=42,
        sensitivity_weighted=False,
        correction_resolution="per_pair",
        description="Toy per-pair pose TTO example.",
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@compress_time_pass(_tto_pose_per_pair_example_contract)
def tto_pose_per_pair_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy per-pair pose TTO. Real consumers run AdamW for N steps on the
    pose parameter tensor; this example just returns the seed bytes
    unchanged + a fixed bytes_added=64.
    """
    archive_bytes_v0 = state["archive_bytes_v0"]
    return {
        "archive_bytes_v1": archive_bytes_v0,
        "bytes_added": 64,
    }


# ---------------------------------------------------------------------------
# 3. MultipassRefinement sample
# ---------------------------------------------------------------------------

_multipass_quant_depth_3_example_contract = MultipassRefinement(
    spec=MultipassRefinementSpec(
        pass_id="multipass_quant_depth_3_example",
        depth=3,
        residual_termination_threshold=1e-4,
        seed=42,
        correction_resolution="per_block",
        description="Toy depth-3 multipass quant example.",
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@compress_time_pass(_multipass_quant_depth_3_example_contract)
def multipass_quant_depth_3_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy depth-3 multipass quant refinement. Real consumers re-quantize
    per-block residuals; this example returns fixed values.
    """
    archive_bytes_v0 = state["archive_bytes_v0"]
    return {
        "archive_bytes_v1": archive_bytes_v0,
        "residual_norm_final": 0.0,
        "bytes_added": 128,
    }


# ---------------------------------------------------------------------------
# 4. SimulatedAnnealing sample
# ---------------------------------------------------------------------------

_sa_fec6_selector_indices_example_contract = SimulatedAnnealingOnDiscreteCodes(
    spec=SimulatedAnnealingSpec(
        pass_id="sa_fec6_selector_indices_example",
        discrete_target="selector_indices",
        temp_schedule="exp",
        initial_temperature=1.0,
        cooling_alpha=0.995,
        num_steps=100,  # toy: 100 steps so test runs fast
        seed=42,
        sensitivity_weighted=False,
        description="Toy SA on fec6 K=16 selector indices example.",
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@compress_time_pass(_sa_fec6_selector_indices_example_contract)
def sa_fec6_selector_indices_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy SA on fec6 K=16 selector indices. Real consumers run the full
    SA loop with per-pair distortion energy.
    """
    archive_bytes_v0 = state["archive_bytes_v0"]
    return {
        "archive_bytes_v1": archive_bytes_v0,
        "bytes_added": 256,
    }


# ---------------------------------------------------------------------------
# 5. PerPairCoordinateSearch sample
# ---------------------------------------------------------------------------

_coord_search_fec6_k16_palette8_example_contract = PerPairCoordinateSearch(
    spec=PerPairCoordinateSearchSpec(
        pass_id="coord_search_fec6_k16_palette8_example",
        num_pairs=600,
        num_modes=16,
        num_palette_entries=8,
        num_pose_deltas=1,
        cpu_parallelism=12,
        seed=42,
        description=(
            "Toy per-pair coord search; 600 pairs × 128 candidates = 76800 "
            "total candidate evaluations (in a real run; this example just "
            "returns the seed)."
        ),
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@compress_time_pass(_coord_search_fec6_k16_palette8_example_contract)
def coord_search_fec6_k16_palette8_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy per-pair coord search. Real consumers fan out over a process
    pool and evaluate per-pair candidates.
    """
    archive_bytes_v0 = state["archive_bytes_v0"]
    return {
        "archive_bytes_v1": archive_bytes_v0,
        "bytes_added": 600 * 7 // 8,  # 7-bit selector index per pair
    }


# ---------------------------------------------------------------------------
# 6. IteratedBisectionRateKnee sample
# ---------------------------------------------------------------------------

_bisect_int8_scale_per_block_example_contract = IteratedBisectionRateKnee(
    spec=IteratedBisectionRateKneeSpec(
        pass_id="bisect_int8_scale_per_block_example",
        granularity="per_block",
        num_outer_iterations=4,
        max_inner_iterations=20,
        convergence_tolerance=1e-4,
        scale_range_log10=(-3.0, 1.0),
        seed=42,
        description=(
            "Toy per-block int8 scale bisection example; 4 outer iterations "
            "with up to 20 inner bisections per block."
        ),
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@compress_time_pass(_bisect_int8_scale_per_block_example_contract)
def bisect_int8_scale_per_block_example(
    state: Mapping[str, Any], *, policy: Mapping[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Toy per-block int8 scale bisection. Real consumers bisect on the
    R-D evaluator until convergence_tolerance is reached.
    """
    archive_bytes_v0 = state["archive_bytes_v0"]
    return {
        "archive_bytes_v1": archive_bytes_v0,
        "bytes_added": 32,
    }
