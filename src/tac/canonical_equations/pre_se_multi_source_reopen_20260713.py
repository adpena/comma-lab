# SPDX-License-Identifier: MIT
"""Canonical #484 retained-mass and cheap-global tile-cost laws."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pre_se_multi_source_cheap_global_reopen_v1"
DAG_FEED = ".omx/research/pre_se_multi_source_reopen_DAG_FEED_20260713.md"
RECEIPT = "experiments/results/pre_se_multi_source_reopen_20260713/receipt.json"
RECEIPT_SHA256 = "a092dd5cf791ab060a4300ac3b9c1d49a196ddd83b158121b70fae6a130dc643"
MEASUREMENT_UTC = "2026-07-13T23:00:38.863427Z"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]"


def retained_mass_fraction(*, retained_mass: float, total_mass: float) -> float:
    """Return exact selected costate mass divided by total costate mass."""

    values = (retained_mass, total_mass)
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise ValueError("mass values must be finite")
    if retained_mass < 0.0 or total_mass <= 0.0 or retained_mass > total_mass:
        raise ValueError("mass values violate 0 <= retained <= total")
    return float(retained_mass) / float(total_mass)


def cheap_global_tile_flops(
    *, local_conv_forward_macs_sum: int, global_forward_plus_vjp_flops: int, tile_count: int
) -> float:
    """Amortize once-only global work over independently executed local tiles."""

    for name, value in (
        ("local_conv_forward_macs_sum", local_conv_forward_macs_sum),
        ("global_forward_plus_vjp_flops", global_forward_plus_vjp_flops),
        ("tile_count", tile_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return (4 * local_conv_forward_macs_sum + global_forward_plus_vjp_flops) / tile_count


def joint_reopen_admitted(
    *, retained_mass: float, retained_mass_bar: float, tileable_modulo_cheap_globals: bool
) -> bool:
    """Require both coupled #484 bars; neither bar can substitute for the other."""

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
        for value in (retained_mass, retained_mass_bar)
    ):
        raise ValueError("retained-mass values must be finite fractions")
    if not isinstance(tileable_modulo_cheap_globals, bool):
        raise ValueError("tileability must be boolean")
    return retained_mass >= retained_mass_bar and tileable_modulo_cheap_globals


def pre_se_multi_source_reopen_laws(
    *,
    retained_mass: float,
    total_mass: float,
    retained_mass_bar: float,
    tileable_modulo_cheap_globals: bool,
    local_conv_forward_macs_sum: int,
    global_forward_plus_vjp_flops: int,
    tile_count: int,
) -> dict[str, float | bool]:
    """Evaluate the retained-mass, tile-cost, and coupled admission laws together."""

    fraction = retained_mass_fraction(retained_mass=retained_mass, total_mass=total_mass)
    return {
        "retained_mass_fraction": fraction,
        "true_per_tile_forward_plus_vjp_flops": cheap_global_tile_flops(
            local_conv_forward_macs_sum=local_conv_forward_macs_sum,
            global_forward_plus_vjp_flops=global_forward_plus_vjp_flops,
            tile_count=tile_count,
        ),
        "joint_reopen_admitted": joint_reopen_admitted(
            retained_mass=fraction,
            retained_mass_bar=retained_mass_bar,
            tileable_modulo_cheap_globals=tileable_modulo_cheap_globals,
        ),
    }


def build_pre_se_multi_source_reopen_v1() -> CanonicalEquation:
    """Build the measured family-kill anchor with exact verdict scope."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "a feature provider outside the frozen shallow+block2+block3 PRE-SE family supplies "
            "new target-ordering information and is preregistered against the same n600/4.70-percent gate; "
            "otherwise route to the whole-teacher distilled student #455"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp64_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="pre_se_multi_source_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_states": 600,
            "train_states": 480,
            "heldout_states": 120,
            "costate_hash_matches_prior": 120,
            "requested_area_fraction": 0.047,
            "realized_area_fraction": 0.047017415364583336,
            "selected_prefix_cells": 2311,
            "prefix_cells": 49152,
            "multi_source_feature_count": 476,
            "ordered_class_pair_count": 20,
            "nonlinear_seeds": [455, 456, 457],
            "tile_grid": [2, 2],
            "aligned_input_halo": 56,
            "unique_SE_reductions": 7,
            "broadcast_gate_scalars": 864,
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output={
            "retained_mass_minimum": 0.47,
            "tileable_modulo_cheap_globals": True,
            "same_area_oracle_ceiling": 0.5278150212253758,
        },
        empirical_output={
            "verdict": "RETAINED-MASS-FAMILY-KILL",
            "verdict_scope": (
                "FAMILY x CHEAP-PRE-SE-LOCALIZATION x SINGLE-AND-MULTI-SOURCE x "
                "CONVEX-AND-NONLINEAR-RUNGS x FIXED-n600-REPLAY x 4.70%-AREA"
            ),
            "convex_multi_source_retained_mass": 0.11225888402810756,
            "nonlinear_multi_source_retained_mass": 0.31562159104967574,
            "nonlinear_seed_retained_mass": [
                0.2995934738746486,
                0.289790392967775,
                0.2897676787268581,
            ],
            "same_area_oracle_retained_mass": 0.5278150212253758,
            "tileable_modulo_cheap_globals": True,
            "receptive_field_input_pixels": 111,
            "output_stride_input_pixels": 8,
            "aligned_halo_input_pixels": 56,
            "global_forward_plus_vjp_flops_once": 16864000,
            "true_overlap_per_tile_forward_plus_vjp_flops": 1049488384.0,
            "ideal_equal_area_per_tile_forward_plus_vjp_flops": 668210368.0,
            "overlap_tiling_cost_ratio": 1.5705957798008905,
            "physical_crop_max_abs": 4.57763671875e-5,
            "full_frame_zero_embedded_core_bitwise_equal": True,
            "campaign_honest_teacher_starts": 600,
            "teacher_retries": 0,
            "route": "#455 whole-teacher DISTILLED student",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.15437840895032423,
        source_artifact=RECEIPT,
        measurement_method=(
            "preregistered fixed n600 replay; 480 immutable exact compact targets plus 120 exact heldout "
            "replays whose costate hashes match prior custody; shared base-42 plus block2 PRE-SE 144 plus "
            "block3 PRE-SE 288 plus sensitivity-2; twenty exact pair-block RankRLS Moore-Penrose heads; "
            "three deterministic width-32 pair-gated MLP seeds; exact 2x2 donated-SE-gate tile proof"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="PRE-SE multi-source retained mass and cheap-global tile cost",
        one_line_summary=(
            "Donated SE globals make the composed prefix tileable, but joint PRE-SE features still "
            "retain only 31.56 percent of exact costate mass at 4.70 percent area."
        ),
        latex_form=(
            r"\rho_k(s)=\frac{\sum_{i\in\operatorname{TopK}(s,k)}\|\lambda_i\|_2^2}"
            r"{\sum_i\|\lambda_i\|_2^2},\quad "
            r"C_{tile}=\frac{4\sum_{t=1}^{T}\operatorname{MAC}_{local,t}+G_{SE}}{T},\quad "
            r"A_{reopen}\iff \max(\rho_{MP},\rho_{MLP})\ge0.47\land T_{modG}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pre_se_multi_source_reopen_20260713:"
            "pre_se_multi_source_reopen_laws"
        ),
        domain_of_validity={
            "scope_level": "family x frozen PRE-SE feature charts x fixed replay",
            "included": (
                "same seed455 n600 replay, same 2311/49152 area, same exact costate target",
                "single-source block2/block3 and joint shallow+block2+block3 PRE-SE charts",
                "pair-block convex MP and three-seed width-32 nonlinear ensemble",
                "tileability conditional on seven exact full-frame SE gates and stage barriers",
            ),
            "excluded": (
                "new learned feature providers, dense whole-teacher distilled students, or larger attention models",
                "wall-clock speedup, scorer score, archive bytes, contest-CPU, CUDA, MPS, or promotion authority",
            ),
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "mass": "squared SegNet input-costate units",
            "local_conv_forward_macs": "multiply-accumulates",
            "global_cost": "forward-plus-input-VJP FLOPs",
        },
        units_out={
            "retained_mass_fraction": "dimensionless",
            "true_per_tile_forward_plus_vjp_flops": "FLOPs per tile",
            "joint_reopen_admitted": "boolean",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "best_retained_mass_shortfall_to_0.47": 0.15437840895032423
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.pre_se_multi_source_reopen_policy_20260713",
            "tac.probe_outcomes_ledger",
        ),
        canonical_producers=("tools.probe_pre_se_multi_source_reopen_20260713",),
        provenance=provenance,
    )


def populate_pre_se_multi_source_reopen_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicit main-review registration surface; never called at import time."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_pre_se_multi_source_reopen_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-484; retained-mass-family-kill; cheap-globals tileability pass; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_UTC",
    "RECEIPT",
    "RECEIPT_SHA256",
    "build_pre_se_multi_source_reopen_v1",
    "cheap_global_tile_flops",
    "joint_reopen_admitted",
    "populate_pre_se_multi_source_reopen_v1",
    "pre_se_multi_source_reopen_laws",
    "retained_mass_fraction",
]
