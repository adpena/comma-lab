# SPDX-License-Identifier: MIT
"""PR110-lineage sidecar-fold integer-grid boundary law (overhead-stack, 2026-07-11).

THE MEASURED FACT (one anchor, three load-bearing clauses)
----------------------------------------------------------
The PR101 607-byte enum-rank latent sidecar on the PR110-lineage frontier payload is a
NEARLY BREAK-EVEN-EFFICIENT section at the integer-grid boundary: deleting it saves
DS_rate = 25*607/37,545,489 = 4.04e-4, while its sub-grid (delta/100) corrections carry
4.16e-6 of d_seg (= +4.16e-4 of S) that NO integer-grid fold reproduces — measured at
per-pair EXACT optimum over 4 integer options per affected pair (597 pairs; 2,388
option-renders; choices 439 nearest / 20 zero / 121 other-rounding / 17 far-side;
selection exact by pair-locality since the per-pair metric means decompose).
Net margin: fold loses by **+2.75e-5** `[macOS-CPU advisory]` NON-PROMOTABLE.

Corollary clause (the actionable composition): fold-without-re-search is RED
(verdict_scope: FORMULATION, this vehicle), but fold + a subsequent exact-gated click
search ON THE FOLDED TABLE is the winning composition — any search recovery > 2.75e-5
banks the -607 B for free (round-1 click search already recovered 7.9e-5 on 48 pairs;
552 pairs unpolished). PR128's fold paid precisely because it ran inside such a search.

Byte clause: the PR112 AR latent-coder length is INVARIANT to every fold variant tried
(all candidates = 176,562 B = base - 607 exactly): moved integer codes are rate-free;
the entire saving is the deleted section.

NO-FAKE #7: borrowed fold idea (PR128, MIT, [external unverified]) on borrowed-lineage
substrate — DEFENSIVE-BANK context, never an originality claim. Pointer 0.19108282
[contest-CPU] UNMOVED. Sibling equation:
``pr110_lineage_click_polish_byte_neutral_slack_v1`` (the click half of the same stack).
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pr110_lineage_sidecar_fold_integer_grid_boundary_v1"

_UTC = "2026-07-12T01:03:23Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/frontier_overhead_stack_20260711T232000Z.md"
_FOLDOPT_SHA = "3c43cd26345d0d9f45fe74ba383d028c28d0406d54802489ae3d0211892f0199"


def build_pr110_lineage_sidecar_fold_integer_grid_boundary_v1() -> CanonicalEquation:
    """Build the sidecar-fold break-even law with its measured optimal-form anchor."""

    anchor = EmpiricalAnchor(
        anchor_id="overhead_stack_fold_opt4_perpair_exact_n600_20260711",
        measurement_utc=_UTC,
        inputs={
            "mechanism": (
                "fold the 607-B PR101 enum-rank latent sidecar (597/600 pairs, (dim, delta/100) "
                "corrections, 0.35-5.6 grid steps, median 1.26) into the base 8-bit latent codes; "
                "per-pair EXACT selection over 4 integer options {nearest, zero, other-rounding, "
                "far-side} using per-pair d_seg/d_pose rows (pair-locality => selection exact)"
            ),
            "substrate": "frontier archive b4689726… (177,169 B), pr110_payload_entropy_recode lineage",
            "harness": "tac.click_polish FrozenPacket/Renderer/Scorer, same GT cache as #399",
            "protocol": "chunked resumable FOREGROUND n600 passes (<=235 s/invocation, npz state)",
        },
        predicted_output={
            "hypothesis": "PR128-style sidecar fold yields -607 B ~ -4.04e-4 S at recoverable distortion",
        },
        empirical_output={
            "base_advisory_S_n600": 0.19110945,
            "fold_naive_S": 0.19144690,
            "fold_opt3_S": 0.19114860,
            "fold_opt4_S": 0.19113692,
            "fold_opt4_dS_vs_base": +2.75e-05,
            "unrecoverable_d_seg": 4.16e-06,
            "rate_saving_S": -4.04e-04,
            "choice_counts": {"nearest": 439, "zero": 20, "other": 121, "far": 17},
            "bytes_all_fold_variants": "176,562 = base - 607 EXACTLY (AR coder length invariant)",
            "foldopt_candidate_sha256": _FOLDOPT_SHA,
            "composed_naive_with_round1_clicks_S": 0.19138227,
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "full_n600_advisory_eval_per_option_rows_then_exact_perpair_argmin_selection_byte_closed"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "a click-search round run ON THE FOLDED table recovering > 2.75e-5 makes the fold "
                "locally rate-beneficial (then the -607 B is free); or an exact contest-CPU row "
                "on a folded candidate supersedes the advisory margin"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "PR110-lineage sidecar-fold integer-grid boundary: the 607-B PR101 sidecar is "
            "break-even-efficient (sub-grid d_seg value 4.16e-4 S vs rate 4.04e-4 S); "
            "deterministic fold RED by +2.75e-5 at per-pair exact optimal form; fold pays only "
            "composed with a post-fold click search (> 2.75e-5 recovery threshold)"
        ),
        one_line_summary=(
            "Fold of the 607-B latent sidecar loses +2.75e-5 at exact per-pair integer optimum; "
            "the sidecar's sub-grid precision ~ its rate cost; fold+re-search is the winning combo."
        ),
        latex_form=(
            r"\min_{k \in \mathbb{Z}^{597}} S_{600}(Q_0{+}k) - S_{600}(Q_0,\mathrm{sidecar})"
            r" = +2.75\times10^{-5} \;>\; 0,\qquad 100\,\Delta d_{seg}^{min} = 4.16\times10^{-4}"
            r" \approx \frac{25\cdot 607}{37{,}545{,}489} = 4.04\times10^{-4}"
        ),
        python_callable_module_path="tac.click_polish",
        domain_of_validity={
            "vehicle": ["pr110_lineage_frontier_payload"],
            "verdict_scope": (
                "FORMULATION -- deterministic fold (any per-pair integer choice on the sidecar-"
                "designated cells) on THIS vehicle. NOT a family verdict on sidecar folding: PR128's "
                "fold succeeded inside a full exact-gated search. Sister negatives cited in the memo: "
                "L2 deletion RED (#153 P-SUFF), L3 re-quant RED (frontier_exact_bitalloc_solve "
                "rd_curve, best 0.313), L4 recode DONE (entropy ceiling)."
            ),
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": "defensive bank per NO-FAKE #7; pointer moves only through an exact row.",
        },
        units_in={"fold_steps": "integer_latent_grid_steps_on_sidecar_cells", "options_per_pair": "4"},
        units_out={"delta_S": "advisory_score_units", "delta_bytes": "archive_bytes (-607 by construction)"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"fold_opt4_measured": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _MEMO,
            "click_polish_rounds_2plus_on_folded_table",  # the composition path this law prescribes
            "pr110_lineage_click_polish_byte_neutral_slack_v1",  # sibling equation (click half)
        ),
        canonical_producers=(
            "tac.click_polish",
            "experiments/results/frontier_overhead_stack_run1",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "folded-table click rounds or an exact contest-CPU row supersede this "
                "formulation-scope margin"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_pr110_lineage_sidecar_fold_integer_grid_boundary_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins). EQUATIONS leg of the
    overhead-stack FEED; DSL leg = N/A-with-reason (frozen-archive packet transform,
    not a trainer/curriculum lever)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_pr110_lineage_sidecar_fold_integer_grid_boundary_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes=(
            "sidecar_fold_integer_grid_boundary_20260711 (overhead-stack measured at optimal form; "
            "fold RED +2.75e-5; fold+post-fold-click-search prescribed; advisory NON-PROMOTABLE)"
        ),
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_pr110_lineage_sidecar_fold_integer_grid_boundary_v1",
    "populate_pr110_lineage_sidecar_fold_integer_grid_boundary_equation",
]
