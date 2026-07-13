# SPDX-License-Identifier: MIT
"""VR-GHAL theorem-body law and scoped Pact solver-admission gate.

This source-inspected equation supersedes only the generic paper-recursion
reconstruction in the task-462 equation memo.  It preserves the existing
moving-operator and teacher-query debt laws.
"""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "vrghal_high_probability_fixed_operator_law_v2"
UTC = "2026-07-13T22:41:36Z"
MEMO = ".omx/research/vrghal_theorem_deepen_20260713.md"
DAG_FEED = ".omx/research/vrghal_theorem_deepen_DAG_FEED_20260713.md"
SUPERSEDES_MEMO = ".omx/research/vrghal_95kill_fixedpoint_equations_20260713.md"
PAPER_URL = "https://arxiv.org/pdf/2607.09097"
AXIS = "[MEANS; source-inspected theorem; no score or pointer authority]"


def vrghal_epoch_residual_upper_bound(
    *,
    epoch: int,
    beta: float,
    a0: float,
    a1: float,
    a2: float,
    a_five_halves: float,
) -> float:
    """Return the exact Theorem-1 induction envelope at one epoch."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    values = (beta, a0, a1, a2, a_five_halves)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("all scalar inputs must be finite")
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must lie strictly between zero and one")
    if any(float(value) < 0.0 for value in (a0, a1, a2, a_five_halves)):
        raise ValueError("A_p constants must be nonnegative")
    k = float(epoch)
    return float(beta) ** epoch * (
        float(a0)
        + float(a1) * k
        + float(a2) * (k + 2.0) ** 2
        + float(a_five_halves) * (k + 2.0) ** 2.5
    )


def vrghal_theorem_admission(
    *,
    fixed_operator: bool,
    lipschitz_upper_bound: float,
    unbiased_oracle: bool,
    bounded_native_second_moment: bool,
    quadratically_smoothable_space: bool,
) -> bool:
    """Fail closed unless the paper's load-bearing base premises are present."""

    if not all(
        isinstance(value, bool)
        for value in (
            fixed_operator,
            unbiased_oracle,
            bounded_native_second_moment,
            quadratically_smoothable_space,
        )
    ):
        raise ValueError("theorem-premise flags must be boolean")
    gamma = float(lipschitz_upper_bound)
    if not math.isfinite(gamma):
        raise ValueError("lipschitz_upper_bound must be finite")
    return bool(
        fixed_operator
        and 0.0 < gamma <= 1.0
        and unbiased_oracle
        and bounded_native_second_moment
        and quadratically_smoothable_space
    )


def build_vrghal_high_probability_fixed_operator_law_v2() -> CanonicalEquation:
    """Build the paper-custodied recurrence plus its no-stray domain gate."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "for the current pre-SE convex rung: show that direct sufficient statistics, "
            "factorization, and deterministic exact-enough alternatives violate a recorded "
            "resource budget while an unbiased stochastic oracle remains affordable; for "
            "witness SGD: freeze the update map and prove native-norm gamma<=1, unbiasedness, "
            "bounded second moment, and quadratically smoothable geometry"
        ),
        measurement_axis=AXIS,
        hardware_substrate="substrate_independent_paper_theorem_and_cached_source_audit",
        captured_at_utc=UTC,
    )
    paper_anchor = EmpiricalAnchor(
        anchor_id="arxiv_2607_09097v1_theorem_body_20260713",
        measurement_utc=UTC,
        inputs={
            "paper": PAPER_URL,
            "version": "v1",
            "fixed_operator": True,
            "gamma_range": "(0,1]",
            "oracle": "iid unbiased with bounded native-norm second central moment",
            "space": "quadratically smoothable real separable Banach",
            "supersedes_memo": SUPERSEDES_MEMO,
        },
        predicted_output={
            "clipping_radius": "bar_gamma*norm(x-y)",
            "anytime_residual_envelope": (
                "beta^k[A0+A1*k+A2*(k+2)^2+A5/2*(k+2)^(5/2)]"
            ),
        },
        empirical_output={
            "theorem_body_status": "MEASURED_FROM_PAPER",
            "clipping_radius_has_free_c": False,
            "theorem_epoch_constant_fully_numeric": False,
            "epoch_constant_limit": "contains O_beta(1)",
            "corollary_leading_constants_exposed": False,
            "current_pre_se_vrghal_verdict": "NO-GO_DOMINATED_BY_EXACT_SOLVE",
            "current_theorem_admitted_pact_locus": "NONE",
            "only_forced_iterative_candidate": (
                "frozen-stage/frozen-replay/fixed-loss witness-SGD solve window"
            ),
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=MEMO,
        measurement_method=(
            "full-paper source inspection of assumptions, Algorithm 3, Lemma 5, Assumption 4, "
            "Theorem 1, Lemmas 7-8, and Corollaries 1-3; read-only re-derivation of the n600 "
            "pre-SE normal-equation solve from source and protected receipt"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="VR-GHAL high-probability fixed-operator law and Pact admission gate",
        one_line_summary=(
            "VR-GHAL has an anytime beta^k-polynomial residual envelope only for one fixed "
            "nonexpansive/contractive operator with a qualified unbiased stochastic oracle."
        ),
        latex_form=(
            r"\operatorname{Cl}_{\bar\gamma}\Delta(x,y)="
            r"\min\!\left\{1,\frac{\bar\gamma\lVert x-y\rVert}{\lVert\Delta(x,y)\rVert}\right\}\Delta(x,y),\quad "
            r"\widetilde R_k\le\beta^k\!\left[A_0+A_1k+A_2(k+2)^2+A_{5/2}(k+2)^{5/2}\right]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.vrghal_theorem_body_20260713:"
            "vrghal_theorem_admission"
        ),
        domain_of_validity={
            "included": [
                "one fixed gamma-Lipschitz operator with gamma in (0,1]",
                "iid unbiased oracle independent of algorithmic past",
                "bounded native-norm second central moment",
                "quadratically smoothable real separable Banach space",
                "paper-v1 theorem recurrence and source constants",
            ],
            "excluded": [
                "moving witness-induced distributions or changing teacher semantics",
                "current pre-SE convex rung with available certified direct MP solve",
                "costate refresh/reuse safety decisions that are not fixed-point solves",
                "unproved nonconvex witness-SGD update maps",
                "fully numeric oracle/epoch constants hidden by O_beta(1) or Otilde",
                "score, contest CPU/CUDA, archive, or pointer authority",
            ],
            "verdict_scope": (
                "FORMULATION x CURRENT-FIXED-N600-PRE_SE-CONVEX-RUNG x SOLVER-SELECTION"
            ),
            "current_pre_se_verdict": "NO-GO_DOMINATED_BY_CERTIFIED_DIRECT_SOLVE",
            "current_non_dominated_theorem_admitted_locus": "NONE",
            "conditional_candidate": (
                "frozen-stage/frozen-replay/fixed-loss witness-SGD solve window"
            ),
            "supersedes": SUPERSEDES_MEMO,
            "preserves": [
                "EQ-VRGHAL-455-MOVING-OPERATOR-DEBT-v1",
                "EQ-VRGHAL-455-QUERY-TO-TEACHER-v1",
            ],
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "epoch": "count",
            "beta": "dimensionless",
            "A_p": "native-norm residual units",
            "gamma": "native-norm Lipschitz ratio",
        },
        units_out={
            "residual_upper_bound": "native-norm residual units",
            "theorem_admission": "boolean",
        },
        empirical_anchors=(paper_anchor,),
        predicted_vs_empirical_residual={"paper_transcription": 0.0},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(DAG_FEED, "tac.canonical_equations.registry"),
        canonical_producers=(MEMO, PAPER_URL),
        provenance=provenance,
    )


def populate_vrghal_high_probability_fixed_operator_law_v2(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked helper; never mutate registry bytes directly."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_vrghal_high_probability_fixed_operator_law_v2()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "task462 theorem deepen; source-inspected; pre-SE direct-solve dominance; "
            "research-only; pointer-unmoved"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEMO",
    "PAPER_URL",
    "SUPERSEDES_MEMO",
    "build_vrghal_high_probability_fixed_operator_law_v2",
    "populate_vrghal_high_probability_fixed_operator_law_v2",
    "vrghal_epoch_residual_upper_bound",
    "vrghal_theorem_admission",
]

