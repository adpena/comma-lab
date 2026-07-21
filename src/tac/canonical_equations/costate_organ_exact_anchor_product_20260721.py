# SPDX-License-Identifier: MIT
"""Canonical composition law for exact-anchor costate ORGAN v2.

The law was kept FORMALIZATION_PENDING during implementation and is registered
only after its read-only #205+C2 retrospective backtest anchored the four-factor
composition.  The anchor is development evidence, never score/promotion authority.
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "costate_organ_exact_anchor_product_v2"
AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"
SOURCE = "src/tac/witness_control/costate_organ_v2.py"
BACKTEST = ".omx/research/costate_organ_v2_exact_anchor_backtest_20260721.json"
SOURCE_SHA256 = "a0af91a42b6ea079b178ea3ea302309da748b07b4f283872de339ad3955a4fb3"
BACKTEST_SHA256 = "f733187fbc8e69e03d4854a8f45baa0951af4a2807a01bfc1841cffca8d59410"


def build_costate_organ_exact_anchor_product_v2() -> CanonicalEquation:
    source_provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SOURCE_SHA256,
        source_path=SOURCE,
        captured_at_utc="2026-07-21T02:08:57Z",
    )
    backtest_provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=BACKTEST_SHA256,
        source_path=BACKTEST,
        captured_at_utc="2026-07-21T02:08:57Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="costate_organ_v2_n205_c2_rank_backtest_20260721",
        measurement_utc="2026-07-21T02:08:57Z",
        inputs={
            "corpora": "#205 as-of temporal decisions + C2 palette/trained-witness smokes",
            "n_rows": 24,
            "factor_order": ["exact_gap", "visibility", "realizability", "byte_price"],
            "apparatus_gate": "bench-contaminated rows excluded",
            "break_even_law": "realization_breakeven_bytes_v1 latest event domain_refined",
            "pool_kkt_law": "witness_measured_reverse_waterfill_v1",
            "fisher_bank_sha256": (
                "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"),
        },
        predicted_output={
            "acceptance": "Spearman(exact-anchor v2, realized) > Spearman(old DECIDE, realized)",
        },
        empirical_output={
            "old_decide_spearman": 0.5697387260063718,
            "exact_anchor_v2_spearman": 0.6978255654102463,
            "improvement": 0.1280868394038745,
            "ablate_exact_gap_spearman": 0.6569697460914578,
            "ablate_visibility_spearman": 0.6689900347043337,
            "ablate_realizability_spearman": 0.5698089945389212,
            "ablate_byte_price_spearman": 0.6978255654102463,
            "source_bytes_unchanged": True,
            "acceptance": "PASS_RETROSPECTIVE_DEVELOPMENT",
        },
        residual=0.1280868394038745,
        source_artifact=BACKTEST,
        measurement_method=(
            "deterministic average-rank Spearman over 24 read-only historical intervention "
            "rows; exact source hashes checked before/after; per-factor leave-one-out ablation"),
        provenance=backtest_provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Exact-gap times visibility times realizability times byte-price costate",
        one_line_summary=(
            "lambda(pair,site)=exact_gap*visibility*realizability*byte_price, with "
            "apparatus-validity, exclusive-pool KKT, and explicit maturity readback."),
        latex_form=(
            r"\lambda_{p,s}=G^{\mathrm{exact}}_{p,s}\,V_{p,s}\,"
            r"R_{p,s}\,B_{p,s},\qquad 0\le V,R,B\le1"),
        python_callable_module_path="tac.witness_control.costate_organ_v2:compose_lambda",
        domain_of_validity={
            "formalization_status": "ANCHORED_RETROSPECTIVE_DEVELOPMENT",
            "exact_gap": "realized-through-R debt to #547 n600 anchor; canonical fp32 support fill",
            "visibility": "full real-linear resize geometry; task/frame/channel scoped",
            "realizability": "uint8/resize/parse-back route; formulation scoped",
            "byte_price": "realized recovery only; domain-refined break-even law",
            "pool_semantics": (
                "same-pool candidates compete under witness_measured_reverse_waterfill_v1; "
                "dedicated opportunity-pool law registration remains pending"),
            "maturity": "_dev by default; only explicit _prod can become pointer-eligible",
            "axis": AXIS,
            "advisory_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": (
                "rank improvement is retrospective on #205+C2 development rows; the C2 "
                "trained-witness flat-amplitude negative is formulation-scoped; no live, "
                "cross-run, family, or contest-axis generalization"),
        },
        units_in={
            "exact_gap": "S units",
            "visibility": "dimensionless [0,1]",
            "realizability": "dimensionless [0,1]",
            "byte_price": "dimensionless net-rent fraction [0,1]",
        },
        units_out={"lambda": "advisory S opportunity units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "rank_correlation_improvement": 0.1280868394038745,
            "byte_price_ablation_discrimination": 0.0,
        },
        last_calibration_utc="2026-07-21T02:08:57Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.shadow_controller.build_shadow_report",
            "tools/costate_digest.py section_shadow",
            "tools/costate_shadow_report.py",
        ),
        canonical_producers=(SOURCE, "tools/costate_organ_v2_backtest.py"),
        provenance=source_provenance,
    )


def populate_costate_organ_exact_anchor_product_v2(
    *, path: str | Path | None = None, lock_path: str | Path | None = None,
    agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_costate_organ_exact_anchor_product_v2()
    register_canonical_equation(
        equation, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id)
    return equation


__all__ = [
    "AXIS", "BACKTEST", "BACKTEST_SHA256", "EQUATION_ID", "SOURCE", "SOURCE_SHA256",
    "build_costate_organ_exact_anchor_product_v2",
    "populate_costate_organ_exact_anchor_product_v2",
]
