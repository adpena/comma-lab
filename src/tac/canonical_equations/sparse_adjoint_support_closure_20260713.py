# SPDX-License-Identifier: MIT
"""Masked-adjoint error, global-SE support, and low-rank closure law.

The identities in this module distinguish three quantities that are easy to
conflate: numerical concentration, exact graph support, and dense-kernel wall
time.  Concentration can justify a bounded approximation; it cannot by itself
make a frozen dense network's VJP exact or cheaper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "sparse_adjoint_mask_error_and_se_support_closure_v1"
MEMO = ".omx/research/p0_sparse_adjoint_costate_vjp_20260713.md"
RECEIPT = (
    "experiments/results/p0_sparse_adjoint_costate_vjp_20260713/"
    "measurement_receipt.json"
)
RECEIPT_SHA256 = "52a22f4b60367fc27ca0fca7293b0741da4b809724479cd3ef7e92291c250cef"
MEASUREMENT_UTC = "2026-07-13T22:36:44.543192Z"
AXIS = "[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]"

# MEASURED on all 600 source-bound states unless noted otherwise.
BOUNDARY_AREA_FRACTION = 0.047365976969
INPUT_TOP_AREA_L1_MASS = 0.2628567254219988
INPUT_TOP_AREA_L2_ENERGY = 0.6101338801879865
OUTPUT_TOP_AREA_L1_MASS = 0.6430417835037928
OUTPUT_TOP_AREA_L2_ENERGY = 0.8641338301404815
INPUT_EXACT_ZERO_FRACTION = 0.0
OUTPUT_EXACT_ZERO_FRACTION = 0.0

# MEASURED on the 120-state exact full-grid heldout subset.
ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR = 0.3635363542899355
ORACLE_MASK_GLOBAL_COSINE = 0.9545542892524271
SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR = 0.7934344363907141
SOURCE_MARGIN_MASK_GLOBAL_COSINE = 0.7754554442293313
RAW_RANK_FOR_95PCT_ENERGY = 68
RAW_RANK_FOR_99PCT_ENERGY = 100
RAW_RANK64_RELATIVE_FROBENIUS_ERROR = 0.23819231942546068
STATE_RANK_CEILING = 120
FULL_GRID_ELEMENT_COUNT = 70_778_880

# DERIVED from recorded per-convolution dense FLOPs and numerical support.
DENSE_BACKWARD_CONV_FLOPS = 2_378_240_148_480.0
ORACLE_MASK_IDEAL_SPATIAL_FLOPS = 1_076_819_892_484.5938
ORACLE_MASK_IDEAL_SPATIAL_SPEEDUP = 2.208577465069467
EXACT_SPARSE_BACKWARD_SPEEDUP = 1.0
GLOBAL_SQUEEZE_EXCITE_COUNT = 23
EXACT_HALO_PIXELS = 685


def masked_adjoint_error_bound(
    *, jacobian_operator_norm: float, omitted_output_gradient_norm: float
) -> float:
    """Return ``||J||_2 ||(I-M)g||_2``, an upper bound on VJP error."""

    values = (jacobian_operator_norm, omitted_output_gradient_norm)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("norms must be finite numbers")
    if any(value < 0.0 for value in values):
        raise ValueError("norms must be non-negative")
    return float(jacobian_operator_norm * omitted_output_gradient_norm)


def ideal_backward_speedup(*, dense_flops: float, active_flops: float) -> float:
    """Return an arithmetic ceiling, not dense-framework realized wall time."""

    values = (dense_flops, active_flops)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("FLOPs must be finite numbers")
    if dense_flops <= 0.0 or active_flops <= 0.0 or active_flops > dense_flops:
        raise ValueError("require 0 < active_flops <= dense_flops")
    return float(dense_flops / active_flops)


def eckart_young_relative_error(
    *, singular_values: Sequence[float], rank: int
) -> float:
    """Return the optimal rank-r relative Frobenius error."""

    values = tuple(float(value) for value in singular_values)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("singular_values must be a non-empty finite non-negative sequence")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= len(values):
        raise ValueError("rank must be an integer in [0, len(singular_values)]")
    energy = sum(value * value for value in values)
    if energy == 0.0:
        return 0.0
    tail = sum(value * value for value in values[rank:])
    return math.sqrt(max(0.0, tail / energy))


def sparse_adjoint_support_laws(
    *,
    jacobian_operator_norm: float,
    omitted_output_gradient_norm: float,
    dense_flops: float,
    active_flops: float,
    singular_values: Sequence[float],
    rank: int,
) -> dict[str, float]:
    """Evaluate the three machine-readable law legs together."""

    return {
        "masked_adjoint_error_upper_bound": masked_adjoint_error_bound(
            jacobian_operator_norm=jacobian_operator_norm,
            omitted_output_gradient_norm=omitted_output_gradient_norm,
        ),
        "ideal_backward_speedup": ideal_backward_speedup(
            dense_flops=dense_flops, active_flops=active_flops
        ),
        "optimal_rank_r_relative_frobenius_error": eckart_young_relative_error(
            singular_values=singular_values, rank=rank
        ),
    }


def build_sparse_adjoint_mask_error_and_se_support_closure_v1() -> CanonicalEquation:
    """Build the analytic law plus its source-bound n600 empirical anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "a practical current-witness mask plus custom sparse kernels must pass n600 "
            "input-costate error, renderer-gradient regret, wall-time, and full-facet gates"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 CPU Torch 2.12.1; NumPy-fp32 spectrum",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="task455_n600_sparse_adjoint_costate_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "pair_states": 600,
            "heldout_full_grid_states": 120,
            "heldout_full_grid_elements": FULL_GRID_ELEMENT_COUNT,
            "mask_area_fraction": BOUNDARY_AREA_FRACTION,
            "mask_schemes": ["top_output_gradient_l2_oracle", "source_margin_low"],
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output={
            "exact_mask_condition": "J_F(x)^T (I-M) g = 0",
            "low_rank_condition": "rank r is much smaller than state cohort rank",
            "speedup_semantics": "dense_flops / numerically-active custom-kernel flops",
        },
        empirical_output={
            "task455_hash_matches": 600,
            "input_top_area_l1_mass": INPUT_TOP_AREA_L1_MASS,
            "input_top_area_l2_energy": INPUT_TOP_AREA_L2_ENERGY,
            "output_top_area_l1_mass": OUTPUT_TOP_AREA_L1_MASS,
            "output_top_area_l2_energy": OUTPUT_TOP_AREA_L2_ENERGY,
            "input_exact_zero_fraction": INPUT_EXACT_ZERO_FRACTION,
            "output_exact_zero_fraction": OUTPUT_EXACT_ZERO_FRACTION,
            "oracle_mask_global_relative_l2_error": ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR,
            "source_margin_mask_global_relative_l2_error": (
                SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR
            ),
            "raw_rank_for_95pct_energy": RAW_RANK_FOR_95PCT_ENERGY,
            "raw_rank_for_99pct_energy": RAW_RANK_FOR_99PCT_ENERGY,
            "ideal_spatial_backward_speedup_upper_bound": (
                ORACLE_MASK_IDEAL_SPATIAL_SPEEDUP
            ),
            "dense_kernel_realized_speedup": 1.0,
            "exact_sparse_backward_speedup": EXACT_SPARSE_BACKWARD_SPEEDUP,
            "verdict": "NO_GO_DENSE_FULLRANK",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR,
        source_artifact=RECEIPT,
        measurement_method=(
            "hash-validated exact batch-1 frozen-SegNet CE costate regeneration on all 600 "
            "task455 states; full masked VJPs and per-convolution cotangent-support FLOP "
            "propagation on 120 heldout states; full NumPy-fp32 120x120 Gram eigenspectrum over "
            "120x589824 input-costate elements"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Masked-adjoint error, global-SE support, and low-rank closure",
        one_line_summary=(
            "Masking is exact iff the omitted logit adjoint lies in the VJP nullspace; "
            "global SE generically destroys spatial support, and SVD tail energy fixes rank error."
        ),
        latex_form=(
            r"\lambda=J_F(x)^Tg,\ \lambda_M=J_F(x)^TMg,\ "
            r"\|\lambda-\lambda_M\|_2\le\|J_F(x)\|_2\|(I-M)g\|_2;\quad "
            r"\lambda_x(p)=s\odot\lambda_y(p)+P^{-1}J_s(\bar x)^T"
            r"\sum_q x(q)\odot\lambda_y(q);\quad "
            r"\min_{\operatorname{rank}(G_r)\le r}\frac{\|G-G_r\|_F}{\|G\|_F}="
            r"\sqrt{1-\frac{\sum_{i\le r}\sigma_i^2}{\sum_i\sigma_i^2}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.sparse_adjoint_support_closure_20260713:"
            "sparse_adjoint_support_laws"
        ),
        domain_of_validity={
            "research_only": True,
            "included": (
                "frozen task455 EfficientNet-B2 U-Net SegNet; finite fp32 CE/margin-like "
                "output adjoint; binary spatial output mask; cohort-SVD of exact input costates"
            ),
            "exactness": "masking is exact iff J_F(x)^T(I-M)g=0, not merely when g is small",
            "global_support": (
                "each non-degenerate squeeze-excite mean VJP contributes a spatially dense term"
            ),
            "speedup_boundary": (
                "ideal support FLOPs require custom sparse kernels; ordinary dense kernels "
                "realize 1x even when numerical values are zero"
            ),
            "excluded": (
                "exact argmax loss; local scorer with SE removed; stale-SE approximation; "
                "learned current-witness mask; renderer-gradient/optimizer regret; score or pointer claims"
            ),
            "verdict_scope": (
                "4.7366pct output-masked adjoint and high-fidelity cross-state low-rank basis "
                "on the source-bound task455 n600 replay"
            ),
            "fallback": "full dense teacher VJP",
        },
        units_in={
            "J_F": "logit_units_per_input_pixel_unit",
            "g": "surrogate_loss_per_logit_unit",
            "M": "dimensionless_binary_mask",
            "dense_flops": "multiply_add_operations",
            "active_flops": "multiply_add_operations",
            "sigma": "input_costate_frobenius_units",
        },
        units_out={
            "lambda": "surrogate_loss_per_input_pixel_unit",
            "error_bound": "surrogate_loss_per_input_pixel_unit",
            "speedup": "dimensionless_ratio",
            "rank_error": "dimensionless_relative_frobenius_error",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "oracle_4p7366pct_input_costate_relative_l2": (
                ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR
            ),
            "source_margin_4p7366pct_input_costate_relative_l2": (
                SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR
            ),
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.scorer_gradient_policy",
            "tac.bit_allocator",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=("tools.probe_sparse_adjoint_costate_vjp",),
        provenance=provenance,
    )


def populate_sparse_adjoint_mask_error_and_se_support_closure_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicit append-only registration surface; never called at import time."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_sparse_adjoint_mask_error_and_se_support_closure_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-p0-sparse-adjoint; research_only; NO_GO_DENSE_FULLRANK",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR",
    "ORACLE_MASK_IDEAL_SPATIAL_SPEEDUP",
    "RAW_RANK_FOR_95PCT_ENERGY",
    "RECEIPT_SHA256",
    "SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR",
    "build_sparse_adjoint_mask_error_and_se_support_closure_v1",
    "eckart_young_relative_error",
    "ideal_backward_speedup",
    "masked_adjoint_error_bound",
    "populate_sparse_adjoint_mask_error_and_se_support_closure_v1",
    "sparse_adjoint_support_laws",
]
