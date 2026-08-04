# SPDX-License-Identifier: MIT
"""ddm_lr2 — the frame_1 yuv6 pose-null subspace is AC-ONLY: it contains no DC.

THE LAW.  Per 2x2 scorer-lattice cell, the frame_1 pose-null subspace is
``null(A)`` with ``A`` the 6x12 constraint matrix (four per-pixel ``dY = 0``
rows with the luma weights ``K_Y = (0.299, 0.587, 0.114)``, plus the two
block-mean rows ``mean dR = 0`` and ``mean dB = 0`` that kill ``dV``/``dU`` —
derived from ``upstream/frame_utils.py`` yuv6 packing by ``ddm_sq1`` §2.7).
Every CELL-CONSTANT RGB delta ``c ⊗ 1_4`` lies ENTIRELY in ``rowspace(A)``:

    c = alpha * K_Y + (beta, 0, gamma)   is solvable for every c in R^3
    (alpha = dG/0.587, then beta, gamma free),

so the orthogonal projector ``P = I - pinv(A) A`` onto the null space
annihilates every constant EXACTLY: ``P (c ⊗ 1_4) = 0``.

CONSEQUENCES (why this is a scorer-structure law, not a trivium):

  1. **DC color shifts — the cheapest frozen-SegNet argmax movers measured
     (per-region prototype/mean shifts, the m95 solve family) — are 100%
     pose-VISIBLE.**  No projection, masking, or clever placement makes a
     per-region constant paint pose-neutral; the pose price of DC paint is
     its FULL price.  This is the mechanism behind the measured eta<->d_pose
     coupling (ddm_et1 §7: seg gain bought with pose, monotone in budget).
  2. **Pose-neutral paint must carry within-cell TEXTURE (AC) structure.**
     A pose-neutral actuator is parameterised through ``P`` with a
     non-constant per-cell pattern; ddm_lr2's FO-3/AC arm realises this with
     a receiver-derivable zero-mean luma atom (measured d_pose ratio 1.0031
     on its first realized pair — near-neutral as constructed).
  3. Sister caveat (m85 / ddm_sq1 §2.7): even INSIDE the null space, exact
     nullity does not survive the integer uint8 actuator — "pose-neutral to
     ~4%" (paint form) / ~0.3% (additive support-write form) is the honest
     realized statement.  This law is about the FLOAT geometry: DC is not
     even in the subspace to begin with.

EMPIRICAL CONFIRMATION.  ddm_lr2 ran the projected per-block-constant
actuator on 8 stratified pairs x 4 block-budgets (32 cells): eta exactly
0.000000, flips fixed 0, flips introduced 0, d_pose ratio exactly 1.0 in all
32 — the m50 vacuity signature, here PREDICTED by the algebra before it was
measured.  Numeric check: ``max_c |P (c ⊗ 1_4)|_inf <= 1.03e-6`` (fp64
pinv noise) over random and basis constants.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.  Pointer UNMOVED.
"""

from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pose_null_subspace_is_ac_only_v1"

#: Canonical luma weights (upstream frame_utils yuv6 packing).
K_Y = (0.299, 0.587, 0.114)


def _constraint_matrix() -> np.ndarray:
    a = np.zeros((6, 12), dtype=np.float64)
    for p in range(4):
        a[p, 3 * p: 3 * p + 3] = K_Y
        a[4, 3 * p + 0] = 0.25   # block-mean dR = 0  (kills dV)
        a[5, 3 * p + 2] = 0.25   # block-mean dB = 0  (kills dU)
    return a


def dc_projection_residual(n_random: int = 16, seed: int = 0) -> float:
    """The law's pure callable: max_c ||P (c x 1_4)||_inf over basis + random constants.

    Predicted output: 0.0 exactly (<= ~1e-6 floating-point pinv noise).  Any materially
    nonzero return falsifies the law (or flags a drifted projector implementation).
    """
    a = _constraint_matrix()
    proj = np.eye(12) - np.linalg.pinv(a) @ a
    worst = 0.0
    consts = [np.eye(3)[i] * 255.0 for i in range(3)]
    rng = np.random.default_rng(seed)
    consts += [rng.uniform(-255.0, 255.0, 3) for _ in range(max(0, int(n_random)))]
    for c in consts:
        worst = max(worst, float(np.abs(proj @ np.tile(c, 4)).max()))
    return worst


def ac_energy_fraction(delta_cell_12: np.ndarray) -> float:
    """Fraction of a per-cell 12-vector's L2 energy that survives the pose-null projector.

    1.0 = fully pose-null (pure AC in the null space); 0.0 = fully pose-visible (e.g. any
    constant).  Consumers use this to price the pose-visibility of a candidate paint pattern
    BEFORE solving with it.
    """
    d = np.asarray(delta_cell_12, dtype=np.float64).reshape(12)
    n = float(np.dot(d, d))
    if n == 0.0:
        return 0.0
    a = _constraint_matrix()
    proj = np.eye(12) - np.linalg.pinv(a) @ a
    pd = proj @ d
    return float(np.dot(pd, pd) / n)


def build_pose_null_subspace_is_ac_only_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        ".omx/research/ddm_lr2_legal_realization_ladder_20260804.md",
        reactivation_criteria=(
            "exact algebraic statement about the fixed yuv6 packing; recalibration is only "
            "meaningful if upstream frame_utils yuv6 packing or the scorer preprocess changes"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_m5_max_cpu",
        captured_at_utc="2026-08-04T10:10:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="lr2_dc_paint_vacuity_32_of_32_20260804",
            measurement_utc="2026-08-04T09:30:00Z",
            inputs={
                "actuator": "per-block-constant int8 RGB shifts through the rank-6 projector",
                "pairs": [0, 20, 48, 115, 154, 170, 179, 180],
                "block_budgets": [8, 16, 32, 64],
                "vehicle": "pu2 live-best decode (archive sha c72ef357)",
            },
            predicted_output={
                "eta": 0.0,
                "flips_fixed": 0,
                "flips_introduced": 0,
                "d_pose_ratio": 1.0,
                "dc_projection_residual_inf": 0.0,
            },
            empirical_output={
                "eta_all_32_cells": 0.0,
                "flips_fixed_pooled": 0,
                "flips_introduced_pooled": 0,
                "d_pose_ratio_all_cells": 1.0,
                "numeric_projection_residual_max": 1.03e-06,
            },
            residual=0.0,
            source_artifact="/Volumes/VertigoDataTier/pact/ddm_lr2_20260804/lr2_solve0_n8.json",
            measurement_method=(
                "realized C0-null arms (frozen CPU-torch SegNet/PoseNet, real camera "
                "round-trip) + fp64 pinv numeric check of P(c x 1_4) on basis/random constants"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="The frame_1 yuv6 pose-null subspace is AC-only (contains no per-cell DC)",
        one_line_summary=(
            "constants lie entirely in rowspace(A) => P(c x 1_4)=0 exactly: DC color paint is "
            "100% pose-visible; pose-neutral paint must be within-cell AC through P"
        ),
        latex_form=(
            r"\forall c\in\mathbb{R}^3:\ P\,(c\otimes\mathbf{1}_4)=0,\quad "
            r"P = I - A^{+}A,\ A\in\mathbb{R}^{6\times 12}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_null_subspace_is_ac_only_20260804:"
            "dc_projection_residual"
        ),
        domain_of_validity={
            "operator": "upstream frame_utils yuv6 packing (4 luma + 2 block-mean chroma) at "
                        "the 384x512 scorer lattice, frame_1 leg",
            "exactness": "float geometry; integer uint8 realization adds the m85 residual "
                         "(measured ~0.3% additive-write / ~4% flat-paint)",
            "verdict_scope": "DERIVED+MEASURED — exact algebra + 32/32 vacuity confirmation",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"n_random": "count", "seed": "rng_seed"},
        units_out={"dc_projection_residual": "uint8_lsb_infnorm"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"projection_residual_inf": 1.03e-06},
        last_calibration_utc="2026-08-04T10:10:00Z",
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "experiments/ddm_lr2_realization_ladder.py (run_fo1 AC arm: pose-neutral paint "
            "parameterised through P with a receiver-derivable AC atom)",
            "m85 integer-actuator caveat (ddm_sq1 §2.7: exact nullity unreachable by the "
            "integer actuator; this law adds that DC never enters the subspace at all)",
            "burn-spec / pose-price arithmetic (the pose price of per-region DC paint is its "
            "FULL price at the current dS/dd_pose; no projection discount exists)",
        ),
        canonical_producers=(
            ".omx/research/ddm_lr2_legal_realization_ladder_20260804.md §4",
            "experiments/ddm_sq1_pose_null_constrained_paint.py (pose_null_projector — the "
            "canonical P this law is stated against)",
        ),
        provenance=provenance,
    )


def populate_pose_null_subspace_is_ac_only_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (triality EQUATIONS leg of ddm_lr2 §4)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_pose_null_subspace_is_ac_only_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "K_Y",
    "ac_energy_fraction",
    "build_pose_null_subspace_is_ac_only_v1",
    "dc_projection_residual",
    "populate_pose_null_subspace_is_ac_only_equation",
]
