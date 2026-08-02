# SPDX-License-Identifier: MIT
"""Canonical law: the correction-stream LABEL cost, and the band edge it moves (ddm_dc1).

Sister of ``ddm_pp1_correction_stream_position_band_v1``, which prices a correction
stream's POSITION cost and whose ``domain_of_validity["excluded"]`` states verbatim:
*"the LABEL (which-class) cost of a correction stream (this law is POSITION only)."*
That excluded half is measured here, on the SAME supports, the SAME coders and the
SAME densities as the pp1 R2 receipt, so the position curve reproduces exactly (the
canary) and only the label term is new.

MEASURED (ddm_dc1, 2026-08-01, [macOS-CPU advisory], n600 GT partition):

* The label stream costs **0.082 - 0.255 B/flip** (best of the two coders) over
  rho 2.2e-4 .. 2.2e-2, i.e. **0.28 - 0.88x** the blind i.i.d. 5-ary bound
  ``log2(5)/8 = 0.2902410``.  The NEIGHBOUR-CONDITIONED coder alone buys only
  **1.14x** at the band edge (rho 5.6e-4) rising to 2.96x at rho 2.2e-2 -- NOT the
  "near-free after adaptation" the #307 coder's own docstring claims, and at the band
  edge the claim is essentially unearned.  At our live base (rho ~ 3.9e-3 .. 4.3e-3)
  the label term is **0.181 B/flip = 62% of blind** best-of-two, or 0.213 = 73% if the
  fully round-trip-verified all-contour pairing is used (the conservative reading).
* A **GENERIC** control coder (labels at support, raster order, LZMA1-x9e) BEATS the
  neighbour-conditioned contour ``cls`` stream at every rho >= 5.6e-4 -- i.e. at 6 of the 7
  measured densities, the coherent coder winning only at the sparsest (rho 2.2e-4).  Derived is not
  a synonym for correct (the ddm_rh1 lesson, reproduced on a second surface).
* Consequence -- **the band's lower edge moves**: the water crossing of
  position+label is **rho_c(pos+label) = 1.285e-3**, versus the registered
  position-only ``rho_c = 5.02e-4``.  The rational-correction band is
  **~[1.3e-3, 1e-2]**, not [5e-4, 1e-2]; its lower edge is **2.56x higher** than the
  position-only law implies, and the carrier design spec tightens with it (a carrier
  must be natively <= ~1.3e-3, not ~5e-4, to make a correction stream pointless).

Controls (both PASS at every density; P4 "no meter without a canary"):
  * POSITIVE -- the position streams are bit-identical under zeroed / real / random
    class maps, and reproduce the pp1 R2 ``b_per_err_best`` to 4 dp at tau in
    {0.008, 0.02, 0.05, 0.1, 0.2}; the position-only crossing recomputes to 5.0146e-4
    against the registered 5.02e-4.
  * NEGATIVE -- i.i.d. uniform random labels on the same support code at >= 0.90x the
    blind bound, so the label coder is not leaking position information.
  * Bit-exact decode of BOTH flip maps and class maps verified on a leading subset.

HONEST BOUND: the class map coded is the GT class alone.  A real receiver also knows
the base partition's (wrong) class at each corrected site and can exclude it, so every
label price here is a conservative UPPER bound on the deployable label cost.

Receipt: /Volumes/VertigoDataTier/pact/ddm_dc1_20260801/label_price_n600.json
(mirrored to .omx/research/ddm_dc1_label_price_n600_20260801.json).
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

EQUATION_ID = "ddm_dc1_correction_stream_label_cost_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/ddm_dc1_label_price_n600_20260801.json"

WATER_B_PER_FLIP = 1.2731  # registered region_merge water level (shared with the pp1 law)
BLIND_LABEL_BOUND = math.log2(5.0) / 8.0  # 0.2902410 B/flip, i.i.d. uniform 5-ary
#: measured (density, label B/flip best-of-two-coders) -- ddm_dc1 n600, ascending density
LABEL_CURVE: tuple[tuple[float, float], ...] = (
    (2.247450e-4, 0.2551),
    (5.632019e-4, 0.2483),
    (1.4131334e-3, 0.2295),
    (2.8235372e-3, 0.2042),
    (5.6219482e-3, 0.1657),
    (1.1123649e-2, 0.1236),
    (2.1704746e-2, 0.0817),
)
#: measured POSITION+LABEL water crossing (the band's true lower edge)
CROSSING_DENSITY_WITH_LABEL = 1.2853e-3
#: the registered position-only crossing this supersedes as the DESIGN edge
CROSSING_DENSITY_POSITION_ONLY = 5.02e-4


def label_cost(density: float) -> dict[str, float | bool | str]:
    """Label (which-class) cost per corrected error at a base error rate.

    ``density`` = k/N.  Returns the measured label B/flip (log-density interpolation of
    the 7-point n600 curve, clamped at the ends), the blind i.i.d. 5-ary bound, the
    neighbour-coherence gain, and whether ``density`` sits inside the band once the
    label term is paid.  This is an UPPER bound on the deployable label cost (the
    receiver also knows the base class and can exclude it).
    """
    d = float(density)
    if not (0.0 < d < 1.0):
        raise ValueError("density (base error rate) must be a fraction in (0, 1)")
    xs = [math.log10(p[0]) for p in LABEL_CURVE]
    ys = [p[1] for p in LABEL_CURVE]
    x = math.log10(d)
    if x <= xs[0]:
        b_label = ys[0]
    elif x >= xs[-1]:
        b_label = ys[-1]
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
        t = (x - xs[i]) / (xs[i + 1] - xs[i])
        b_label = ys[i] + t * (ys[i + 1] - ys[i])
    return {
        "density": d,
        "label_b_per_err": b_label,
        "blind_label_bound_b_per_err": BLIND_LABEL_BOUND,
        "neighbour_coherence_gain": BLIND_LABEL_BOUND / b_label,
        "label_is_upper_bound": True,
        "in_band_with_label": bool(CROSSING_DENSITY_WITH_LABEL <= d <= 1.0e-2),
        "band_lower_edge_with_label": CROSSING_DENSITY_WITH_LABEL,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, float | bool | str]:
    if set(inputs) != {"density"}:
        raise ValueError(
            "label-cost inputs differ from the canonical callable contract "
            "(expected exactly {'density'})"
        )
    return label_cost(inputs["density"])


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_dc1_correction_stream_label_cost_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the correction-stream label-cost law with its measured anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor if a label coder beats the measured curve (would lower the band edge), "
            "if a base-class-conditioned label stream is measured (this curve is a blind-to-base "
            "UPPER bound and such a coder would strictly beat it), if the 1.2731 water level is "
            "re-derived, or if a correction stream is measured on a REAL base-vs-GT flip field "
            "rather than the margin-thresholded supports shared with pp1. Advisory axis; "
            "contest-CPU/CUDA has no bearing (coder-rate law, not a score)."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_cpu_python_lzma_rangecoder",
        captured_at_utc="2026-08-01T00:00:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_dc1_label_stream_price_n600_20260801",
        measurement_utc="2026-08-01T00:00:00Z",
        inputs={
            "n_sweep_densities": 7,
            "density_range": "2.2e-4 .. 2.2e-2 (margin-thresholded coherent, pp1's supports)",
            "water_B_per_flip": WATER_B_PER_FLIP,
            "blind_label_bound": BLIND_LABEL_BOUND,
            "label_coders": "#307 adaptive cls stream (neighbour ctx) vs LZMA1-x9e raster (generic control)",
        },
        predicted_output={
            # ba31 sec B.3's hypothesis, pre-registered and tested here
            "ba31_hypothesis_label_largely_predictable_from_neighbours": True,
            "position_curve_reproduces_pp1": True,
        },
        empirical_output={
            "label_b_per_err_range": "0.0817 .. 0.2551",
            "label_over_blind_bound_range": "0.52 .. 0.88",
            "ba31_hypothesis_verdict": "REFUTED_AS_STATED (gain 1.14-1.93x, not near-free)",
            "generic_lzma_beats_neighbour_coder_above_density": 5.632e-4,
            "generic_beats_coherent_at_n_of_7_densities": 6,
            "crossing_density_with_label": CROSSING_DENSITY_WITH_LABEL,
            "crossing_density_position_only_recomputed": 5.0146e-4,
            "band_lower_edge_shift_factor": 2.56,
            "positive_control_position_invariant": True,
            "negative_control_random_near_blind": True,
            "roundtrip_flips_and_labels_exact": True,
        },
        # the position-only crossing recomputes to 5.0146e-4 against the registered 5.02e-4
        residual=abs(5.0146e-4 - CROSSING_DENSITY_POSITION_ONLY) / CROSSING_DENSITY_POSITION_ONLY,
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            "real round-trip-verified coder bytes over the cached n600 GT partition; the #307 "
            "contour encoder's four streams split into POSITION (counts+anchor+chain, exactly "
            "pp1's contour_pos_bytes) and LABEL (cls), with a generic LZMA raster label coder as "
            "the control; positive control = position invariance under zeroed/real/random class "
            "maps; negative control = i.i.d. uniform labels vs the blind 5-ary bound"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Correction-stream label cost + the band edge it moves (ddm_dc1)",
        one_line_summary=(
            "A correction stream's LABEL cost is 0.08-0.26 B/flip (0.52-0.88x the blind 5-ary "
            "bound); paying it moves the band's lower edge from rho_c=5.0e-4 (position-only) to "
            "1.29e-3 -- 2.56x higher."
        ),
        latex_form=(
            r"b_{tot}(\rho)=b_{pos}(\rho)+b_{lab}(\rho),\quad "
            r"b_{lab}(\rho)\in[0.52,0.88]\cdot\tfrac{\log_2 5}{8},\quad "
            r"b_{tot}(\rho_c^{+lab})=1.2731\Rightarrow \rho_c^{+lab}=1.29\times10^{-3}"
            r"\ \ (2.56\times\ \rho_c^{pos})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_dc1_correction_stream_label_cost_20260801:label_cost"
        ),
        domain_of_validity={
            "object": "correction/support stream LABEL (which-class) coding over the n600 GT "
                      "scorer geometry, on the same margin-thresholded supports as pp1 R2",
            "coder_families": "#307 adaptive range-coded cls stream (neighbour ctx) + LZMA1-x9e "
                              "raster labels (generic control)",
            "evidence_axis": "[macOS-CPU advisory] — coder-rate law, NOT a contest score",
            "band_with_label": "rational correction only for rho in ~[1.3e-3, 1e-2]",
            "design_spec": "a carrier must be natively <= ~1.3e-3 (not ~5e-4) to make a "
                           "correction stream pointless; PR130's 3e-4 rail still clears it",
            "upper_bound_note": "blind-to-base: a receiver that excludes the known-wrong base "
                                "class codes labels strictly cheaper than this curve",
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "FORMULATION_CORRECTION_STREAM_LABEL_COST",
            "excluded": [
                "contest score / frontier movement (advisory axis)",
                "the SOLVER cost of finding which sites to correct (this law is coding only)",
                "in-place token edits re-encoded through a carrier (a different object: their "
                "byte delta is entropy inflation of an existing stream, not a coded correction "
                "stream — e.g. QA03's 1.4518 B/flip is NOT comparable to this or the pp1 law)",
                "any promotion or submission use",
            ],
        },
        units_in={"density": "base error rate rho = k/N"},
        units_out={
            "label_b_per_err": "bytes per corrected error (label/which-class stream, upper bound)",
            "blind_label_bound_b_per_err": "bytes per error under i.i.d. uniform 5-ary labels",
            "neighbour_coherence_gain": "ratio blind/measured (1.0 = no gain from neighbours)",
            "in_band_with_label": "bool (density inside the label-inclusive rational band)",
            "band_lower_edge_with_label": "base error rate at which position+label crosses water",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "position_only_crossing_rel_error_vs_registered": abs(5.0146e-4 - 5.02e-4) / 5.02e-4,
        },
        last_calibration_utc="2026-08-01T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.costate_digest",
            "tac.optimization.direct_description_minimizer",
        ),
        canonical_producers=(
            "experiments.ddm_dc1_label_stream_price",
        ),
        provenance=provenance,
    )


def populate_ddm_dc1_correction_stream_label_cost_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the correction-stream label-cost law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_dc1_correction_stream_label_cost_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "ddm_dc1 — the half the pp1 band law explicitly excluded, measured on pp1's own "
            "supports with pp1's own coders (position reproduces to 4 dp; position-only crossing "
            "recomputes 5.0146e-4 vs registered 5.02e-4). Moves the band's lower edge 2.56x. "
            "advisory axis; score_claim=false"
        ),
    )
    return equation


__all__ = [
    "BLIND_LABEL_BOUND",
    "CROSSING_DENSITY_POSITION_ONLY",
    "CROSSING_DENSITY_WITH_LABEL",
    "EQUATION_ID",
    "LABEL_CURVE",
    "WATER_B_PER_FLIP",
    "build_ddm_dc1_correction_stream_label_cost_v1",
    "label_cost",
    "populate_ddm_dc1_correction_stream_label_cost_v1",
]
