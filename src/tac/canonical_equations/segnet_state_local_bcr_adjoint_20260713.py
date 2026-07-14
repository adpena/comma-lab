# SPDX-License-Identifier: MIT
"""Conditional state-local BCR adjoint compression law for frozen SegNet.

This is a DESIGN-only equation definition. It deliberately does not append the
shared canonical-equation registry: that registry was already modified by a
live sibling when this file was created. ``populate_*`` is the explicit
main-review surface after collision review and empirical block-rank evidence.

The law does not assert that the EfficientNet-B2 U-Net Jacobian is a
pseudodifferential operator. It states the consequences *if* its wavelet
off-diagonal blocks have uniformly bounded numerical rank on a content-bound
state ball, and it exposes the state-drift and amortization debts that a real
provider must discharge.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from tac.canonical_equations.equation import RECALIBRATE_NEVER_AUTO, CanonicalEquation
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "segnet_state_local_bcr_adjoint_v1"
_MEMO = ".omx/research/invprob_operator_fold_20260713.md"
_UTC = "2026-07-13T22:04:36Z"
_AXIS = "[DERIVED/design-only; NumPy-fp32 bound arithmetic; no empirical or score authority]"


def state_local_bcr_error_bound_numpy(
    *,
    operator_approximation_error: float,
    jacobian_lipschitz_bound: float,
    state_radius: float,
    output_costate_norm: float,
    renderer_jacobian_norm_bound: float,
) -> dict[str, float]:
    """Return the conditional costate and renderer-gradient error bounds.

    With ``A0`` approximating ``J_F(x0)^T`` to operator error ``eps`` and
    ``||J_F(x)-J_F(x0)|| <= L_J ||x-x0||``, the current costate error obeys

    ``||J_F(x)^T q - A0 q|| <= (eps + L_J rho) ||q||``.

    Left multiplication by the renderer Jacobian gives the second bound. All
    arithmetic is performed in NumPy float32 as the portable algebraic
    reference; this is not an empirical certificate for any scorer state.
    """

    values = np.asarray(
        [
            operator_approximation_error,
            jacobian_lipschitz_bound,
            state_radius,
            output_costate_norm,
            renderer_jacobian_norm_bound,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all() or np.any(values < np.float32(0.0)):
        raise ValueError("all bounds and norms must be finite and non-negative")

    eps, lip, radius, q_norm, renderer_norm = values
    total_operator_error = np.float32(eps + np.float32(lip * radius))
    costate_error = np.float32(total_operator_error * q_norm)
    gradient_error = np.float32(renderer_norm * costate_error)
    if not np.isfinite(np.asarray([total_operator_error, costate_error, gradient_error], dtype=np.float32)).all():
        raise ValueError("float32 bound arithmetic overflowed")
    return {
        "total_operator_error_upper_bound": float(total_operator_error),
        "input_costate_error_upper_bound": float(costate_error),
        "renderer_gradient_error_upper_bound": float(gradient_error),
    }


def bcr_nonstandard_work_units(
    *,
    pixel_count: int,
    input_channels: int,
    output_channels: int,
    coefficients_by_scale: Sequence[int],
    near_width_by_scale: Sequence[int],
    far_rank_by_scale: Sequence[int],
    wavelet_work_per_value: float = 1.0,
) -> float:
    """Return the derived multiscale apply-work proxy.

    The proxy is

    ``c_W(C_in+C_out)N + sum_l N_l(s_l r_l + r_l^2)``.

    It is linear in ``N`` only when the dyadic coefficient count sums to
    ``O(N)`` and the admitted near widths and far ranks stay bounded. Work
    units are deliberately not seconds; a real wall-time benchmark is owed.
    """

    integer_values = (pixel_count, input_channels, output_channels)
    if any(not isinstance(value, int) or value < 1 for value in integer_values):
        raise ValueError("pixel_count and channel counts must be positive integers")
    if not (
        len(coefficients_by_scale) == len(near_width_by_scale) == len(far_rank_by_scale)
        and len(coefficients_by_scale) > 0
    ):
        raise ValueError("all per-scale sequences must have the same nonzero length")
    if not math.isfinite(float(wavelet_work_per_value)) or wavelet_work_per_value < 0.0:
        raise ValueError("wavelet_work_per_value must be finite and non-negative")

    work = float(wavelet_work_per_value) * float(input_channels + output_channels) * float(pixel_count)
    for count, width, rank in zip(
        coefficients_by_scale,
        near_width_by_scale,
        far_rank_by_scale,
        strict=True,
    ):
        if not all(isinstance(value, int) and value >= 0 for value in (count, width, rank)):
            raise ValueError("per-scale counts, widths, and ranks must be non-negative integers")
        work += float(count) * float(width * rank + rank * rank)
    if not math.isfinite(work):
        raise ValueError("work proxy overflowed")
    return work


def amortized_bcr_backward_seconds(*, apply_seconds: float, build_seconds: float, reuse_horizon: int) -> float:
    """Return ``T_apply + T_build/K`` for one state-local BCR provider cycle."""

    if not isinstance(reuse_horizon, int) or reuse_horizon < 1:
        raise ValueError("reuse_horizon must be an integer >= 1")
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in (apply_seconds, build_seconds)):
        raise ValueError("times must be finite and non-negative")
    return float(apply_seconds) + float(build_seconds) / float(reuse_horizon)


def minimum_strict_crossover_horizon(
    *, exact_backward_seconds: float, apply_seconds: float, build_seconds: float
) -> int | None:
    """Return the smallest integer ``K`` with ``apply + build/K < exact``.

    ``None`` means the approximate apply alone is not faster than the exact
    backward, so no amortization horizon can create a strict timing win.
    """

    values = (exact_backward_seconds, apply_seconds, build_seconds)
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        raise ValueError("times must be finite and non-negative")
    exact = float(exact_backward_seconds)
    apply = float(apply_seconds)
    build = float(build_seconds)
    gap = exact - apply
    if gap <= 0.0:
        return None
    if build == 0.0:
        return 1
    return math.floor(build / gap) + 1


def build_segnet_state_local_bcr_adjoint_v1() -> CanonicalEquation:
    """Build the conditional rank, error, and amortization law."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="State-local wavelet-BCR frozen-SegNet input adjoint",
        one_line_summary=(
            "Uniformly bounded wavelet off-diagonal ranks imply a linear-work approximate "
            "input-costate apply, but state drift, build cost, and exact-descent validation remain charged."
        ),
        latex_form=(
            r"\operatorname{rank}_{\varepsilon}(P_IW_iJ_F(x_0)^TW_o^TP_J)\le r_*;\quad "
            r"\widetilde\lambda=W_i^TA_{\rm BCR}(x_0)W_oq;\quad "
            r"C_{\rm apply}=c_W(C_i+C_o)N+\sum_\ell N_\ell(c_ss_\ell r_\ell+c_rr_\ell^2)=\Theta(N);\quad "
            r"\|J_F(x)^Tq-\widetilde\lambda\|\le(\varepsilon+L_J\rho)\|q\|;\quad "
            r"\|g_\theta-\widetilde g_\theta\|\le\|J_{x(\theta)}\|(\varepsilon+L_J\rho)\|q\|;\quad "
            r"\overline T_{\rm bwd}(K)=T_{\rm apply}+T_{\rm build}/K"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.segnet_state_local_bcr_adjoint_20260713:state_local_bcr_error_bound_numpy"
        ),
        domain_of_validity={
            "research_only": True,
            "source_theory": (
                "Lexing Ying, Solving Inverse Problems with Deep Learning, ICM 2022: "
                "non-standard wavelet form for PDOs and butterfly factorization for FIOs"
            ),
            "included": (
                "the real frozen five-logit SegNet relaxation at one content-bound scorer input x0; "
                "classwise two-dimensional wavelet frames; explicit near-field interactions; "
                "held-out operator probes showing bounded epsilon-rank for every admitted far block; "
                "a content-bound state ball with a valid Jacobian Lipschitz upper bound; "
                "NumPy-fp32 reference arithmetic plus framework parity"
            ),
            "static_graph_support": (
                "source introspection derives a local/pointwise U-Net graph plus 23 EfficientNet-B2 "
                "squeeze-excitation global-pooling modules whose reduction bottlenecks sum to 626; "
                "this bounds one source of nonlocal coupling but does not prove global BCR rank"
            ),
            "excluded": (
                "asserting that an arbitrary CNN Jacobian is a PDO; identifying singular support with "
                "ordinary spatial support; using one costate image's matrix rank as an operator-block-rank proof; "
                "exact argmax differentiation; uncharged randomized probing or factor fitting; "
                "cross-state reuse without drift custody; speedup from big-O alone; score or pointer claims"
            ),
            "asymptotic_caveat": (
                "the exact fixed-architecture CNN VJP is already linear in pixel count up to its network "
                "coefficient; the BCR arm is a constant-factor and activation-graph replacement claim, not "
                "an O(N^2)-to-O(N) claim about the autograd implementation"
            ),
            "fallback": "fresh exact frozen-SegNet VJP on any rank, drift, parity, descent, custody, or timing failure",
            "verdict_scope": "conditional theorem and design-only provider law; empirical admission deferred",
            "score_claim": False,
            "promotion_eligible": False,
            "numpy_fp32_authority": True,
        },
        units_in={
            "N": "SegNet_input_pixels",
            "q": "surrogate_loss_per_SegNet_logit",
            "epsilon": "input_costate_per_output_costate_operator_norm",
            "L_J": "Jacobian_operator_norm_per_SegNet_input_norm",
            "rho": "SegNet_input_norm",
            "T": "seconds_per_backward_slice",
        },
        units_out={
            "lambda_tilde": "surrogate_loss_per_SegNet_input_value",
            "gradient_error_bound": "surrogate_loss_per_renderer_parameter_L2_norm",
            "C_apply": "dimensionless_apply_work_units",
            "T_bar_bwd": "seconds_per_amortized_backward_slice",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.scorer_gradient_policy",
            "tools.probe_sparse_adjoint_costate_vjp",
        ),
        canonical_producers=("tools.probe_sparse_adjoint_costate_vjp",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "append real early, boundary, and late hierarchical block spectra; held-out global and annulus "
                "costate error; renderer-gradient direction; strict exact-teacher descent; NumPy/Torch/MLX "
                "parity; charged build/apply/fallback wall-time; and in-loop n600 evidence"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="symbolic_derivation_plus_numpy_fp32_bound_reference",
            captured_at_utc=_UTC,
        ),
    )


def populate_segnet_state_local_bcr_adjoint_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Main-review append surface; do not call while the shared registry is held."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_segnet_state_local_bcr_adjoint_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-p0-backward-wave; conditional BCR rank/cost/error law; design only",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "amortized_bcr_backward_seconds",
    "bcr_nonstandard_work_units",
    "build_segnet_state_local_bcr_adjoint_v1",
    "minimum_strict_crossover_horizon",
    "populate_segnet_state_local_bcr_adjoint_v1",
    "state_local_bcr_error_bound_numpy",
]
