# SPDX-License-Identifier: MIT
"""Canonical law: the correction-stream POSITION-COST band (ddm_pp1 R3).

Registers ``ddm_pp1_correction_stream_position_band_v1`` — the DERIVED + MEASURED
law that a correction stream over a base with error rate rho = k/N pays a POSITION
(support) cost per error bounded above by the uniform combinatorial rate

    b_pos(rho) = log2(N/k)/8  =  log2(1/rho)/8   [bytes per error],

which crosses the registered 1.2731 B/flip water level at rho_u = 2^(-8*1.2731) =
8.59e-4. Context/coherence LOWERS the achievable cost below this bound; the MEASURED
coherent position cost (margin-thresholded boundary-clustered support, n600 GT) crosses
the water level at rho_c = 5.02e-4 (context shifts the band edge DOWN ~1.7x). Random
(incoherent) subsampled support + generic LZMA overshoots the bound and crosses higher
(~1.5e-3). Consequence (the law's teeth): a correction stream is rational ONLY in the
band rho ~ [5e-4, ~1e-2] — below rho_c the position floor alone exceeds the water level
so CONCEDING dominates; above ~1e-2 the total support bytes explode (sp1's 421 KB support
wall at rho=0.864%). Design spec: a carrier must be natively <= ~5e-4 (ideally <= 3e-4,
PR130's rail) to ship NO correction stream; sub-rho_c correction machinery is permanently
pointless.

Empirical anchor (ddm_pp1 R2, 2026-07-28, [macOS-CPU advisory], n600): the uniform bound
log2(1/rho)/8 is an UPPER bound on the measured coherent position cost at all 9 sweep
densities (rho 2.2e-4..7.0e-2; e.g. 1.469<1.515, 0.359<0.811 B/err); the measured coherent
water crossing is rho_c=5.02e-4 vs the derived uniform crossing rho_u=8.59e-4; cross-check:
at fc1's rho=0.864% the curve interpolates ~0.44 B/err vs fc1's measured 0.413 B/err (LZMA
support). Receipt: .omx/research/ddm_pp1_band_lemma_receipt_20260728.json.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_pp1_correction_stream_position_band_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/ddm_pp1_band_lemma_receipt_20260728.json"

WATER_B_PER_FLIP = 1.2731  # registered region_merge water level (bytes per conceded argmax flip)
UNIFORM_CROSSING_DENSITY = 2.0 ** (-8.0 * WATER_B_PER_FLIP)  # 8.59e-4 (derived)
MEASURED_COHERENT_CROSSING_DENSITY = 5.02e-4  # ddm_pp1 R2 measured (context shifts the edge down)
BAND_UPPER_DENSITY = 1.0e-2  # above this total support bytes explode (sp1 421KB wall @ 0.864%)


def position_cost_band(density: float) -> dict[str, float | bool | str]:
    """Uniform position-cost upper bound + rational-band regime for a base error rate.

    ``density`` = k/N (base error rate = fraction of sites the correction stream must fix).
    Returns the uniform per-error position bound (bytes/error), whether it sits below the
    1.2731 water level, and the regime tag. This is the DERIVED bound; the measured coherent
    cost sits BELOW it (see the equation's anchor for the context shift).
    """
    d = float(density)
    if not (0.0 < d < 1.0):
        raise ValueError("density (base error rate) must be a fraction in (0, 1)")
    b_uniform = math.log2(1.0 / d) / 8.0
    if d < MEASURED_COHERENT_CROSSING_DENSITY:
        regime = "concede"  # position floor exceeds water -> conceding at 1.2731 dominates
    elif d <= BAND_UPPER_DENSITY:
        regime = "correct"  # correction stream rational (affordable per-error AND small total)
    else:
        regime = "explode"  # per-error cheap but total support bytes blow the budget
    return {
        "density": d,
        "uniform_bound_b_per_err": b_uniform,
        "below_water_uniform": bool(b_uniform <= WATER_B_PER_FLIP),
        "rational_correction": bool(regime == "correct"),
        "regime": regime,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, float | bool | str]:
    if set(inputs) != {"density"}:
        raise ValueError("position-band inputs differ from the canonical callable contract "
                         "(expected exactly {'density'})")
    return position_cost_band(inputs["density"])


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_pp1_correction_stream_position_band_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the correction-stream position-cost band law with its measured anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor if a position coder beats the measured coherent curve at any density "
            "(would move rho_c), if the 1.2731 water level is re-derived, or if a base with "
            "measured error < rho_c ships a correction stream that helps (would falsify the "
            "concede-dominates leg). Advisory axis; contest-CPU/CUDA has no bearing (this is a "
            "coder-rate law, not a score)."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_cpu_python_lzma_rangecoder",
        captured_at_utc="2026-07-28T00:00:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_pp1_band_lemma_curve_n600_20260728",
        measurement_utc="2026-07-28T00:00:00Z",
        inputs={
            "n_sweep_densities": 9,
            "density_range": "2.2e-4 .. 7.0e-2 (margin-thresholded coherent) + 1e-4..1e-2 (random)",
            "water_B_per_flip": WATER_B_PER_FLIP,
            "fc1_cross_check_density": 0.00864212883843316,
            "fc1_cross_check_b_per_err_measured": 421366 / 1019467,
        },
        predicted_output={
            "uniform_crossing_density": UNIFORM_CROSSING_DENSITY,
            "uniform_bound_is_upper_bound_on_coherent": True,
        },
        empirical_output={
            "measured_coherent_crossing_density": MEASURED_COHERENT_CROSSING_DENSITY,
            "coherent_below_uniform_at_all_9_points": True,
            "incoherent_random_overshoots_uniform": True,
            "fc1_curve_interp_b_per_err_at_0864pct": 0.44,
        },
        # residual = the uniform bound holds as an upper bound at every measured coherent point
        # (the derived-bound claim is confirmed with 0 violations); the context shift is a
        # documented output, not a prediction error.
        residual=0.0,
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            "real position-coding bytes (packbits->LZMA1-x9e incumbent + #307 contour chain "
            "coder, round-trip-guaranteed) over n600 margin-thresholded coherent + random "
            "incoherent correction-support fields; crossing located by log-density interpolation"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Correction-stream position-cost band (ddm_pp1 R3)",
        one_line_summary=(
            "Correction-stream per-error position cost <= log2(1/rho)/8; crosses the 1.2731 "
            "water at rho_c=5.0e-4 measured (8.6e-4 derived) -> corrections rational only for "
            "base error ~5e-4..1e-2."
        ),
        latex_form=(
            r"b_{pos}(\rho)=\tfrac{1}{8}\log_2\!\tfrac{1}{\rho}\ \ge\ b_{pos}^{\text{coherent}}(\rho),"
            r"\quad b_{pos}(\rho_u)=1.2731\Rightarrow\rho_u=2^{-8\cdot1.2731}=8.6\times10^{-4},"
            r"\quad \rho_c^{\text{meas}}=5.0\times10^{-4};\ \ \text{rational iff }\rho\in[\rho_c,\,10^{-2}]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_pp1_correction_stream_position_band_20260728:"
            "position_cost_band"
        ),
        domain_of_validity={
            "object": "correction/support stream position coding over the n600 GT scorer geometry",
            "coder_families": "packbits->LZMA1-x9e + #307 contour chain coder (position streams)",
            "evidence_axis": "[macOS-CPU advisory] — coder-rate law, NOT a contest score",
            "band": "rational correction only for base error rho in ~[5e-4, 1e-2]; below rho_c "
                    "concede at 1.2731 dominates; above 1e-2 total support bytes explode",
            "context_shift": "uniform crossing 8.6e-4 -> measured coherent 5.0e-4 (boundary "
                             "clustering lowers position entropy below the uniform combinatorial rate)",
            "design_spec": "a carrier must be natively <= ~5e-4 (ideally <= 3e-4, PR130 rail) to "
                           "ship NO correction stream; sub-rho_c correction machinery is pointless",
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "FORMULATION_CORRECTION_STREAM_POSITION_COST",
            "excluded": [
                "contest score / frontier movement (advisory axis)",
                "the LABEL (which-class) cost of a correction stream (this law is POSITION only)",
                "any promotion or submission use",
            ],
        },
        units_in={"density": "base error rate rho = k/N (fraction of sites the stream must fix)"},
        units_out={
            "uniform_bound_b_per_err": "bytes per error (upper bound on the coherent position cost)",
            "below_water_uniform": "bool (uniform bound <= 1.2731 water)",
            "rational_correction": "bool (regime == correct)",
            "regime": "concede | correct | explode",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"uniform_bound_violations_on_coherent": 0.0},
        last_calibration_utc="2026-07-28T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.costate_digest",
            "tac.optimization.direct_description_minimizer",
        ),
        canonical_producers=(
            "experiments.ddm_pp1_band_lemma_curve",
        ),
        provenance=provenance,
    )


def populate_ddm_pp1_correction_stream_position_band_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the correction-stream position-band law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_pp1_correction_stream_position_band_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "ddm_pp1 R3 band lemma — validated by the $0 synthetic-density falsifier before "
            "registration (measured coherent crossing 5.0e-4 vs derived uniform 8.6e-4); advisory "
            "axis; score_claim=false; sharpens the carrier native-error spec (<= ~5e-4)"
        ),
    )
    return equation


__all__ = [
    "BAND_UPPER_DENSITY",
    "EQUATION_ID",
    "MEASURED_COHERENT_CROSSING_DENSITY",
    "UNIFORM_CROSSING_DENSITY",
    "WATER_B_PER_FLIP",
    "build_ddm_pp1_correction_stream_position_band_v1",
    "populate_ddm_pp1_correction_stream_position_band_v1",
    "position_cost_band",
]
