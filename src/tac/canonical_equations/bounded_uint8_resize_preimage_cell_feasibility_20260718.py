# SPDX-License-Identifier: MIT
"""Canonical factor-2 law (Task #540): exact uint8 realization of a disjoint
rational resize target is a bounded four-variable Diophantine feasibility
problem, followed by independent hard decoded-uint8 winner-cell acceptance.

This RATIFIES the v10 uint8-lattice arm's staged candidate
(`bounded_uint8_resize_preimage_cell_feasibility_v1`,
`.omx/research/canonical_equation_candidates_uint8_lattice_20260718.jsonl`) into
the live registry — the equations leg of the completeness-matrix factor 2 (the
load-bearing MISSING term: a real-valued minimum-norm solve + clip(round) breaks
argmax feasibility Δ≈63, whereas the per-disjoint-2×2-resize-block bounded
Diophantine predicate is an IFF feasibility test that realizes d_seg=0.0 exactly
on the uint8 lattice).

HONESTY / SCOPE (per the byte-close gate on #540):
- The FEASIBILITY PREDICATE is deterministic exact math: the resize supports are
  VERIFIED_VIA_SOURCE_INSPECTION from upstream `modules.py` (align_corners=false
  half-pixel bilinear, disjoint two-tap supports), and `c^T z = T, z in [0,255]^4`
  is an exact gcd-pruned bounded search — this law is registerable now.
- The RATE / d_seg SCORE claims (full-n600 realization + receiver-closure + byte
  accounting) remain byte-close-gated and are the recalibration trigger; this
  module ships NO empirical anchor and makes NO score/promotion claim.
"""

from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.optimization.uint8_lattice_feasibility import (
    BlockSolveStatus,
    solve_bounded_integer_block,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "bounded_uint8_resize_preimage_cell_feasibility_v1"
SOURCE_MEMO = ".omx/research/codex_findings_v10_uint8_lattice_20260718_codex.md"
CANDIDATE_LEDGER = (
    ".omx/research/canonical_equation_candidates_uint8_lattice_20260718.jsonl"
)
PENDING_UTC = "2026-07-18T00:00:00Z"


def bounded_cell_feasibility_certificate(
    coefficients,
    common_denominator: int,
    target: float,
    *,
    target_integer: int | None = None,
    max_nodes: int = 4096,
) -> dict[str, Any]:
    """The IFF feasibility predicate as a clean typed certificate.

    Surfaces `solve_bounded_integer_block` (the exact gcd-pruned bounded
    Diophantine search) as the law's evaluator: FEASIBLE_EXACT gives an exact
    uint8 witness `c^T z = T`; INFEASIBLE_EXHAUSTIVE is a proof of exact-lattice
    infeasibility for the cell; NOT_FOUND_BUDGET is honest non-knowledge
    (`max_nodes` interrupted the search), never an infeasibility claim.
    """

    # The EXACT feasibility predicate needs the integer numerator `target_integer`
    # (c^T z = target_integer); `target` alone is only the real-valued projection
    # target (a minimum-norm heuristic). Derive it from the exact rational
    # target*denominator when the caller did not supply one, so the certificate is
    # the true Diophantine IFF test rather than a heuristic corner.
    if target_integer is None:
        scaled = target * common_denominator
        rounded = round(scaled)
        if abs(scaled - rounded) <= 1e-9 * max(1.0, abs(scaled)):
            target_integer = int(rounded)
    result = solve_bounded_integer_block(
        coefficients,
        common_denominator,
        target,
        target_integer=target_integer,
        max_nodes=max_nodes,
    )
    status = result.status
    feasible_exact = status == BlockSolveStatus.FEASIBLE_EXACT
    return {
        "status": str(status),
        "feasible_exact": feasible_exact,
        "proven_lattice_infeasible": status == BlockSolveStatus.INFEASIBLE_EXHAUSTIVE,
        "budget_exhausted_unknown": status == BlockSolveStatus.NOT_FOUND_BUDGET,
        "exact_uint8_witness": list(result.values) if feasible_exact else None,
        "exact_target_rational": bool(result.exact_target_rational),
        "projection_residual": float(result.projection_residual),
        "nodes_visited": int(result.nodes_visited),
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_bounded_uint8_resize_preimage_cell_feasibility_v1() -> CanonicalEquation:
    """Build the structural factor-2 feasibility law (no empirical anchor)."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_MEMO,
        reactivation_criteria=(
            "land the full-n600 exact-lattice realization + receiver-closed byte "
            "accounting (the score claims); a new measured receipt binds the first "
            "empirical anchor and triggers recalibration. Contest promotion "
            "additionally requires exact archive evaluation."
        ),
        measurement_axis="[research-signal]",
        hardware_substrate="macos_arm64",
        captured_at_utc=PENDING_UTC,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Bounded uint8 resize-preimage cell feasibility (Diophantine IFF)",
        one_line_summary=(
            "Exact uint8 realization of a disjoint rational resize target is a "
            "bounded four-variable Diophantine feasibility problem followed by "
            "independent hard decoded-uint8 winner-cell acceptance."
        ),
        latex_form=(
            r"\mathrm{FEASIBLE\_EXACT}(T_{jk};c_j)\;\Longleftrightarrow\;"
            r"\exists z_{jk}\in\{0,\ldots,255\}^{4}:c_j^{\mathsf T}z_{jk}=T_{jk};"
            r"\quad \mathrm{HARD\_ACCEPT}(z,\hat c)\;\Longleftrightarrow\;"
            r"\hat c_{jk}=\arg\max_c N_{\mathrm{seg}}(A(Q_{\mathrm{uint8}}z))_{jkc}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bounded_uint8_resize_preimage_cell_feasibility_20260718:"
            "bounded_cell_feasibility_certificate"
        ),
        domain_of_validity={
            "camera_hw": [874, 1164],
            "scorer_hw": [384, 512],
            "resize": (
                "canonical align_corners=false half-pixel bilinear with disjoint "
                "two-tap supports"
            ),
            "integer_domain": "uint8 [0,255]",
            "exact_target_custody_required": True,
            "feasibility_predicate_authority": (
                "VERIFIED_VIA_SOURCE_INSPECTION: the two-tap disjoint resize supports "
                "are exact from upstream modules.py; c^T z = T over [0,255]^4 is an "
                "exact gcd-pruned bounded search (an iff predicate)"
            ),
            "score_authority": (
                "BYTE_CLOSE_GATED: full-n600 exact-lattice realization + receiver-closed "
                "byte accounting are OWED; this law makes no d_seg/rate/score claim"
            ),
            "candidate_ledger": CANDIDATE_LEDGER,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": (
                "structural factor-2 feasibility law (deterministic math); the n6 "
                "receipt is advisory; never a family negative or frontier claim"
            ),
        },
        units_in={
            "T": "exact integer resize numerator",
            "c": "integer tap-product coefficients",
            "target_class": "SegNet class index",
        },
        units_out={
            "certificate": "typed per-cell feasibility proof status plus exact uint8 witness",
            "hard_acceptance": "decoded uint8 through A and frozen CPU SegNet",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=PENDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "future V10 factor-2 completeness manifest after MAIN adoption",
            "future receiver-closed uint8 preimage compiler",
        ),
        canonical_producers=(
            "tac.optimization.uint8_lattice_feasibility.DisjointResizeOperator.solve_uint8",
            "tac.optimization.uint8_lattice_feasibility.solve_bounded_integer_block",
        ),
        provenance=provenance,
    )


def populate_bounded_uint8_resize_preimage_cell_feasibility_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Register the structural factor-2 feasibility law into the registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_bounded_uint8_resize_preimage_cell_feasibility_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "bounded_cell_feasibility_certificate",
    "build_bounded_uint8_resize_preimage_cell_feasibility_v1",
    "populate_bounded_uint8_resize_preimage_cell_feasibility_equation",
]

# Anchor to the ratified candidate so the anti-duplicate-SoT gate (#533) sees the
# cross-reference: this module IS the live ratification of the arm's candidate
# in .omx/research/canonical_equation_candidates_uint8_lattice_20260718.jsonl.
