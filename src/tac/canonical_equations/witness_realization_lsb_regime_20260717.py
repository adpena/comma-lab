# SPDX-License-Identifier: MIT
"""Canonical equation: witness realization-vs-gradient regime from the sub-LSB fraction
of the REMAINING flip mass.

MEASURED 2026-07-17 on the LIVE c2 witness run (``levelset_n600_witness_20260717T113932Z``,
rolling EMA shadow ``__epoch`` 900) through the real chain: canonical torch level-set
decode (realized through R) -> frozen CPU-torch SegNet (weights sha256 pinned) -> exact
pairwise margins at witness-vs-GT flip pixels (bit-exact ``gt_n600.npz`` lstars) ->
full-chain per-pixel VJP to the camera-res tensor through the EXACT shared bilinear
resize (bitwise-parity-asserted against ``segnet.preprocess_input``).

The law
-------
Let ``F`` be the witness's remaining flip set and, per flip pixel, ``a_max = m *
max|g| / ||g||^2`` the largest per-coordinate amplitude (0-255 units) of the min-norm
camera displacement that crosses the correcting pairwise margin ``m = z_wrong - z_gt``
(gradient ``g = dm/dx_camera``).  With the necessity-solver convention ``sub-LSB iff
a_max < 0.5`` (``realization_necessity_preimage_per_stratum_v1``), define

    frac_subLSB = flip-mass-weighted fraction of F with a_max < 0.5

    regime = realization_limited  iff frac_subLSB >= 0.5   (terminal SOLVE #341/#342
                                                            admissible: the MAJORITY of
                                                            what remains cannot be
                                                            realized by amplitude
                                                            training through uint8-R)
             gradient_limited     iff frac_subLSB <= 0.25  (keep training)
             mixed                otherwise

The 0.5/0.25 regime cuts are a DERIVED classification convention (majority /
supermajority-open), stated as such; the continuous fraction is the primary output.
Sub-LSB of the MIN-NORM move is a necessary-side indicator: wider-support dithered
moves may still realize a sub-min-norm-LSB margin change (identical caveat to the
parent necessity law).

MEASURED anchor (this registration): frac_subLSB = 0.362 mass-weighted (0.368
unweighted; 125 VJP pixels stratified over 18,094 flips on 24 stride-25 pairs;
sample d_seg 0.003835) -> regime MIXED, terminal-SOLVE not yet majority-admissible.
Per-stratum structure: erased lanes (Road->Lane, the largest stratum) are mostly
AMPLITUDE-OPEN (sub-LSB 0.18, a_max med 1.60 LSB) while spurious lanes (Lane->Road,
sub-LSB 0.52) and Undrivable->Road (sub-LSB 1.00, a_max med 0.155) are dominated by
unrealizable min-norm moves.

Axis ``[macOS-CPU advisory]``; research_only; no score claim; subset-labeled
(stride-25 pairs, n=125 VJP sample — the parent law's own subset conventions).

Producer: ``tac.witness_control.realization_regime`` (module CLI; state rows in
``.omx/state/witness_realization_regime.jsonl``).  Memo:
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

EQUATION_ID = "witness_realization_lsb_regime_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; real live EMA checkpoint"
MEMO = ".omx/research/factorized_costate_organ_upgrades_20260717.md"

#: MEASURED 2026-07-17 (live run, rolling EMA ep900; 24 stride-25 pairs; 125 VJP px).
MEASURED_SUB_LSB_FRAC_MASS_WEIGHTED_EP900 = 0.3617450991512582
MEASURED_SUB_LSB_FRAC_UNWEIGHTED_EP900 = 0.368
MEASURED_PER_PAIR_SUB_LSB_EP900 = {
    "Road->Lane": 0.184, "Lane->Road": 0.517, "Road->Undrivable": 0.600,
    "Movable->Road": 0.143, "Undrivable->Road": 1.000,
}


def build_witness_realization_lsb_regime_v1() -> CanonicalEquation:
    from tac.witness_control.realization_regime import (
        GRADIENT_LIMITED_MAX_FRAC,
        REALIZATION_LIMITED_MIN_FRAC,
        SUB_LSB_MAX_COORD,
    )

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SEGNET_WEIGHTS_SHA256,
        source_path=MEMO,
        captured_at_utc="2026-07-18T01:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="witness_realization_regime_c2_ep900_20260717",
            measurement_utc="2026-07-18T01:10:00Z",
            inputs={
                "run": "levelset_n600_witness_20260717T113932Z",
                "checkpoint": "rolling EMA shadow levelset_witness_ema_mlx.npz __epoch=900",
                "segnet_weights_sha256": SEGNET_WEIGHTS_SHA256,
                "gt": "gt_n600.npz lstars (bit-exact frozen SegNet cache)",
                "sample": "24 stride-25 pairs; 125 stratified VJP pixels over 18094 flips",
            },
            predicted_output={
                "question": "is the c2 plateau realization- or gradient-limited?",
                "convention": "sub-LSB iff min-norm crossing max-coordinate < 0.5 (parent law)",
            },
            empirical_output={
                "sub_lsb_frac_mass_weighted": MEASURED_SUB_LSB_FRAC_MASS_WEIGHTED_EP900,
                "sub_lsb_frac_unweighted": MEASURED_SUB_LSB_FRAC_UNWEIGHTED_EP900,
                "regime": "mixed",
                "terminal_solve_admissible": False,
                "d_seg_sample": 0.003834618462456597,
                "per_pair_sub_lsb": dict(MEASURED_PER_PAIR_SUB_LSB_EP900),
                "road_lane_a_max_med_lsb": 1.599,
            },
            residual=0.0,
            source_artifact=MEMO,
            measurement_method=(
                "canonical torch level-set decode of the live EMA npz -> frozen CPU SegNet "
                "(preprocess_input; bitwise parity asserted for the differentiable mirror) -> "
                "per-flip pairwise margin backward to the camera tensor; a_max = m*max|g|/||g||^2; "
                "mass-weighted aggregation over oriented-pair strata"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Witness realization-vs-gradient regime from the sub-LSB remaining-flip fraction",
        one_line_summary=(
            "regime = realization_limited iff the flip-mass-weighted fraction of remaining "
            "flips with min-norm crossing max-coordinate < 0.5 LSB reaches 0.5; the live c2 "
            "read at ep900 is MIXED (0.362)."
        ),
        latex_form=(
            r"a_{\max}(p)=\frac{m_p\,\|g_p\|_\infty}{\|g_p\|_2^2},\quad "
            r"\phi=\Pr_{\mathrm{mass}}\!\big[a_{\max}<\tfrac12\big],\quad "
            r"\mathrm{regime}=\begin{cases}\text{realization-limited}&\phi\ge0.5\\"
            r"\text{gradient-limited}&\phi\le0.25\\\text{mixed}&\text{else}\end{cases}"
        ),
        python_callable_module_path=(
            "tac.witness_control.realization_regime:classify_fraction"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet through the exact shared bilinear resize",
            "weights_sha256": SEGNET_WEIGHTS_SHA256,
            "object": "the WITNESS's remaining flips vs the cached GT argmax (not GT fragility "
                      "— that is the parent necessity law)",
            "convention": f"sub-LSB max-coord < {0.5}; thresholds "
                          f"{REALIZATION_LIMITED_MIN_FRAC}/{GRADIENT_LIMITED_MAX_FRAC} are a "
                          "stated classification convention; the fraction is primary; "
                          f"uint8 LSB scale, SUB_LSB_MAX_COORD={SUB_LSB_MAX_COORD}",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "instance (this checkpoint, stride-25/125-px subset); the "
                             "classifier LAW is the registered object",
        },
        units_in={"sub_lsb_frac": "flip-mass fraction in [0,1]"},
        units_out={"regime": "one of realization_limited/mixed/gradient_limited"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"convention_matches_parent_law": 0.0},
        last_calibration_utc="2026-07-18T01:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/costate_digest.py section_factorized_sense (realization-regime line)",
            "terminal SOLVE #341/#342 admissibility (operator run-length decision input)",
        ),
        canonical_producers=(
            "src/tac/witness_control/realization_regime.py",
        ),
        provenance=provenance,
    )


def populate_witness_realization_lsb_regime_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of the A/B/C landing)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_witness_realization_lsb_regime_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "MEASURED_PER_PAIR_SUB_LSB_EP900",
    "MEASURED_SUB_LSB_FRAC_MASS_WEIGHTED_EP900",
    "MEASURED_SUB_LSB_FRAC_UNWEIGHTED_EP900",
    "build_witness_realization_lsb_regime_v1",
    "populate_witness_realization_lsb_regime_equation",
]
