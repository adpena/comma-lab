# SPDX-License-Identifier: MIT
"""Canonical equation: exact-factorized first-order duty marginal with the ker(A)
zero-marginal theorem.

DERIVED 2026-07-17 from two MEASURED parent laws (no new constants) and MEASURED on the
live c2 witness margin field:

* ``segnet_head_rank4_linear_flipdist_v1``: the frozen head is EXACTLY rank-4 linear, so
  a flip pixel with pairwise margin ``m = z_wrong - z_gt`` is corrected exactly when the
  induced margin change crosses ``m``, and the maximal margin change per unit
  penultimate-feature move on pair (c,c') is the canonical pair norm ``||w_c - w_c'||``.
* the EXACT shared resize support split (``realization_necessity_preimage_per_stratum_v1``
  + the closed-form tap table, torch-verified): a camera-space actuation supported inside
  ker(A) has EXACTLY zero scorer-input effect.

The law
-------
For lever ``l`` with DSL class-direction ``u`` (``lambda_net.lever_features`` — the same
trunk coordinates the #516 exact factorized adjoint consumes), feature-space budget
``eps`` (default SELF-CALIBRATED: the live snapshot's median feature-space flip
distance), and ker(A) survival scale ``kappa`` (1 for logit-space levers; the root
visible-energy fraction for camera-map levers; EXACTLY 0 for pure-ker maps):

    marginal_dseg(l, eps) = (1/N_px) * #{ flip px p :
        m(p) <= eps * kappa * ||w_pair(p)|| * align(u, p) },
    align(u, p) = max(u_gt(p) - u_wrong(p), 0) / (sqrt(2) ||u||_2)   in [0, 1].

Corollary (theorem, from A's tap sparsity — verified live by one-hot torch probes):
``supp(actuation) ⊆ ker(A)  =>  marginal_dseg = 0`` exactly.

Honesty: pairwise first-order (a third class can intercept after the move — upper-bound
flavored); the snapshot is a labeled stride subset; the ranking is an ALTERNATIVE
surfaced beside the statistical duty queue, never a replacement.

MEASURED anchor (this registration; live rolling-EMA snapshot, 12 stride-50 pairs,
7065 flips, self-calibrated eps = 0.0593): lane-aimed levers dominate the closed-form
ranking — lane_edge / thin_lane 4.20e-4 first-order d_seg marginal, then
horizon_margin / chroma_boundary 3.30e-4 — consistent with the measured Road<->Lane
flip dominance (the campaign's lane long-tail).

Axis ``[macOS-CPU advisory]``; research_only; no score claim.
Producer: ``tac.witness_control.factorized_duty_ranking``; consumer:
``tools/costate_digest.py`` (factorized-duty line) reading
``.omx/state/witness_factorized_snapshot.jsonl`` written by
``tools/costate_live_ingest.py``.  Memo:
``.omx/research/factorized_costate_organ_upgrades_20260717.md``.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    SEGNET_WEIGHTS_SHA256,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "factorized_duty_marginal_projected_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; live EMA margin snapshot"
MEMO = ".omx/research/factorized_costate_organ_upgrades_20260717.md"

#: MEASURED 2026-07-17 (live snapshot, verdict ep900 companion; eps self-calibrated).
MEASURED_TOP_MARGINALS_EP900 = {
    "lane_edge": 4.203e-4, "thin_lane": 4.203e-4,
    "horizon_margin": 3.298e-4, "chroma_boundary": 3.297e-4, "persistence": 3.059e-4,
}
MEASURED_SELF_CALIBRATED_EPS_EP900 = 0.059330265094524395


def build_factorized_duty_marginal_projected_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SEGNET_WEIGHTS_SHA256,
        source_path=MEMO,
        captured_at_utc="2026-07-18T01:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="factorized_duty_ranking_c2_ep900_20260717",
            measurement_utc="2026-07-18T00:50:00Z",
            inputs={
                "run": "levelset_n600_witness_20260717T113932Z (rolling EMA)",
                "snapshot": "12 stride-50 pairs; 7065 remaining flips; exact pairwise margins",
                "lever_set": "lambda_net.LEVER_FEATURE_MAP (the DSL trunk coordinates)",
                "eps": MEASURED_SELF_CALIBRATED_EPS_EP900,
            },
            predicted_output={
                "expectation": "lane-aimed levers lead iff Road<->Lane dominates the flip mass",
            },
            empirical_output={
                "top_marginals_d_seg": dict(MEASURED_TOP_MARGINALS_EP900),
                "flip_mass_leader": "Road->Lane + Lane->Road (measured dominant strata)",
            },
            residual=0.0,
            source_artifact=MEMO,
            measurement_method=(
                "closed-form crossing count over the persisted margin histograms of the live "
                "snapshot (rank_levers_from_summary_row), cross-checked against the direct "
                "per-pixel compute in tests (rel tol 8%, histogram resolution)"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="ker_a_zero_marginal_theorem_probe_20260717",
            measurement_utc="2026-07-18T00:20:00Z",
            inputs={"operator": "F.interpolate bilinear (874,1164)->(384,512), the real op"},
            predicted_output={"law": "supp ⊆ ker(A) => scorer-input effect exactly 0"},
            empirical_output={
                "blind_perturb_output_max_delta": 0.0,
                "closed_form_zero_weight_frac": 0.22696926089315625,
                "canonical_constant": 0.226969,
            },
            residual=2.6089315624533427e-07,
            source_artifact="src/tac/witness_control/tests/test_factorized_features.py",
            measurement_method=(
                "perturb ONLY closed-form-blind camera pixels; assert the live torch resize "
                "output is bitwise unchanged; blind fraction vs the canonical measured constant"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Exact-factorized first-order duty marginal with the ker(A) zero-marginal theorem",
        one_line_summary=(
            "marginal_dseg(l,eps) = flip mass with m <= eps*kappa*||w_pair||*align(u); "
            "kappa = 0 exactly on ker(A) support; live c2 read: lane-aimed levers lead."
        ),
        latex_form=(
            r"\Delta d_{seg}(\ell,\varepsilon)=\frac{1}{N}\#\Big\{p: m_p\le "
            r"\varepsilon\,\kappa_\ell\,\|w_{c_p}-w_{c'_p}\|\,"
            r"\mathrm{align}(u_\ell,p)\Big\},\quad "
            r"\mathrm{supp}(\ell)\subseteq\ker A\Rightarrow\Delta d_{seg}=0"
        ),
        python_callable_module_path=(
            "tac.witness_control.factorized_duty_ranking:lever_marginal_from_snapshot"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet (rank-4 head law) on the live witness margin field",
            "weights_sha256": SEGNET_WEIGHTS_SHA256,
            "order": "pairwise FIRST-ORDER (third-class interception unmodeled; upper-bound "
                     "flavored); histogram-resolution when recomputed from persisted rows",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "ranking LAW registered; the anchored ranking is instance-scoped "
                             "(this checkpoint, stride subset)",
        },
        units_in={"m": "logit", "eps": "penultimate_feature_L2", "align": "dimensionless"},
        units_out={"marginal_d_seg": "d_seg (flip fraction) per budget"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"ker_zero_weight_frac_vs_canonical": 2.61e-07},
        last_calibration_utc="2026-07-18T01:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/costate_digest.py section_factorized_sense (factorized-duty line)",
            "duty-to-measure comparison surface (statistical vs exact-factorized ranking)",
        ),
        canonical_producers=(
            "src/tac/witness_control/factorized_duty_ranking.py",
            "tools/costate_live_ingest.py",
        ),
        provenance=provenance,
    )


def populate_factorized_duty_marginal_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of the A/B/C landing)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_factorized_duty_marginal_projected_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "MEASURED_SELF_CALIBRATED_EPS_EP900",
    "MEASURED_TOP_MARGINALS_EP900",
    "build_factorized_duty_marginal_projected_v1",
    "populate_factorized_duty_marginal_equation",
]
