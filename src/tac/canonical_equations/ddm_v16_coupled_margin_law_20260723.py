# SPDX-License-Identifier: MIT
"""Canonical law anchor for the DDM v16 coupled margin solve.

Let ``u`` concatenate counted shared-template bytes and measured sparse
camera-support compensation bytes.  For signed scorer margins ``m(u)`` and a
receiver-through-R Jacobian ``M = dm/du``, one frozen-pattern SQP subproblem is

    minimize     1/2 delta.T (G + M.T W M + lambda I) delta
    subject to   M delta >= epsilon - m(u)
                 -rho <= delta <= rho.

The closed KKT system is exact for that conditional affine QP.  It is not a
claim that the nonlinear scorer is globally affine.  Pair-local 2x2 phase is a
separate bounded activation-pattern enumeration.  The continuous solution is
projected to the uint8 lattice by Hessian-metric Babai nearest plane and must be
remeasured through the actual receiver before admission.
"""
from __future__ import annotations

from typing import Final

LAW_ID: Final = "DDM_V16_COUPLED_MARGIN_KKT_UINT8_RECEIVER_RELINEARIZE_20260723"
EQUATION: Final = "min 1/2*d^T(G+M^TWM+lambda*I)d s.t. M*d>=epsilon-m(u), |d|<=rho"
KKT_SYSTEM: Final = "[H,-A^T;A,0][d,mu]=[0,b]"
OUTER_LOOP: Final = "VJP_M -> conditional_KKT -> Babai_uint8 -> real_receiver -> diff -> changed-pattern_relinearize"


def describe() -> dict[str, object]:
    """Return the authority boundary and reusable equation identifiers."""

    return {
        "schema": "ddm_v16_coupled_margin_law.v1",
        "law_id": LAW_ID,
        "equation": EQUATION,
        "kkt_system": KKT_SYSTEM,
        "outer_loop": OUTER_LOOP,
        "variables": {
            "u": "counted shared 2x2 template bytes plus sparse measured-support RGB compensation",
            "M": "backprop/VJP derivative of signed SegNet and PoseNet trust margins through receiver and exact R",
            "G": "positive description-length metric",
            "W": "nonnegative constraint-row weights",
            "rho": "per-DOF local trust box",
        },
        "authority": {
            "receiver_R": "EXACT",
            "conditional_affine_QP_and_closed_KKT": "EXACT_WITHIN_FIXED_LOCAL_MODEL",
            "M": "MEASURED_LOCAL_DERIVATIVE_FD_VALIDATED",
            "Gauss_Newton": "MODELED_LOCAL_CURVATURE",
            "Babai": "BOUNDED_INTEGER_PROJECTION",
            "phase": "BOUNDED_COMBINATORIAL_SEARCH",
            "admission": "REAL_RECEIVER_AND_FROZEN_SCORER_ONLY",
            "global_nonlinear_optimum": "NOT_CLAIMED",
        },
        "verdict_scope": "FORMULATION or INSTANCE, never family-wide, until full receiver ladder is measured",
        "score_claim": False,
    }


__all__ = ["EQUATION", "KKT_SYSTEM", "LAW_ID", "OUTER_LOOP", "describe"]
