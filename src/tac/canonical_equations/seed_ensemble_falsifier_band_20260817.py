# SPDX-License-Identifier: MIT
"""Canonical equation: SEED-ENSEMBLE FALSIFIER-BAND LAW (the cw1 seed-2 re-grade, 2026-08-17).

A control run is itself a random draw. A falsifier band on a stochastic training endpoint that is
calibrated against ONE control run silently assumes zero seed variance — the exact assumption the
band exists to guard against. The law:

    sigma_est(e1, e2) = |e1 - e2| / sqrt(2)          (two matched-config seeds; n=2, wide)
    band INVALID if the control's own seed spread straddles the band boundary
    (equivalently: would the control PASS its own falsifier under a different seed?)

MEASURED ANCHOR (EF3000 semantic-renderer window, [macOS-MPS training-signal], n600 advisory
``quantized_exact_seg``): seed 20260715 endpoint -2,286 flips vs init; seed 20260817 endpoint
-1,365 flips (identical argv otherwise, verified by set-diff at fire). |delta| = 921 flips ->
sigma_est ~= 651.2. The previously QUOTED band (sigma ~= 605, inferred) is CONFIRMED in magnitude
(ratio 1.08). CONSEQUENCE APPLIED: the cw1 rung-2 falsifier boundary (-1,430) was STRADDLED by the
control's own seed spread (-2,286 ... -1,365) -> the "REFUTED at 1.5 sigma" verdict was re-graded
WEAKENED-DIRECTIONAL (0.9 sigma vs the two-seed mean -1,825.5; sign stable, LR6E5 worse than BOTH
controls; routing unchanged: rung 3 cold, lr sweep direction DOWN). Three sister verdicts survived
re-check unchanged (FRD077 0.18 sigma seg-neutral; q3q4 0.61-band within-noise; F1 structural zero).

Sister genus: prefix-of-a-skewed-population ([[m88]]/[[m96]]) — one draw from a spread is not the
population, whether the draw is a data prefix or a control seed. NO score claim; the pointer is
untouched. DAG leg: FEED-2026-08-17c addendum 2. Memo leg: ddm_cw1_corrected_window_20260817.md
section 7.2 (this equation's verdict consumer).
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "seed_ensemble_falsifier_band_v1"

_UTC = "2026-08-17T00:00:00Z"
_AXIS = "[macOS-MPS training-signal]"
_MEMO = ".omx/research/ddm_cw1_corrected_window_20260817.md"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# The exact measured anchors (flips vs init on the shared EF3000 config; DEN = 117,964,800 px).
_SEED1_ENDPOINT_FLIPS = -2286      # seed 20260715 (the original control)
_SEED2_ENDPOINT_FLIPS = -1365      # seed 20260817 (the ensemble control, fired 08-17)
_SIGMA_MEASURED = 651.2            # |delta|/sqrt(2), n=2
_SIGMA_QUOTED_PRIOR = 605.0        # the inferred band four 08-17 verdicts quoted
_RUNG2_BOUNDARY_FLIPS = -1430      # the cw1 pre-registered REFUTED boundary that was straddled


def seed_sigma_est(endpoint_a: float, endpoint_b: float) -> float:
    """sigma estimate from two matched-config seed endpoints (n=2 — itself wide, ~76% rel. error)."""
    return abs(endpoint_a - endpoint_b) / math.sqrt(2.0)


def band_boundary_straddled(boundary: float, endpoint_a: float, endpoint_b: float) -> bool:
    """True when the control's own seed spread straddles the falsifier boundary — the band is then
    INVALID as calibrated (it was drawn against one seed as if it were the population mean)."""
    lo, hi = min(endpoint_a, endpoint_b), max(endpoint_a, endpoint_b)
    return lo <= boundary <= hi


def build_seed_ensemble_falsifier_band_v1() -> CanonicalEquation:
    """Build the seed-ensemble falsifier-band law with its measured EF3000 seed-pair anchor."""
    anchor_seed_pair = EmpiricalAnchor(
        anchor_id="ef3000_seed_pair_sigma_651_band_straddle_20260817",
        measurement_utc=_UTC,
        inputs={
            "config": "tac.pr130_lift.train_semantic_quantized_resumable EF3000 window "
                      "(3000 steps, lr 2e-5, ce/softplus 0.0, q3q4 ON, argv identical except --seed)",
            "seed_1": 20260715, "seed_2": 20260817,
            "payloads": "/Volumes/APDataStore/pact/ddm_cw1/{EF3000,EF3000_SEED2}/result.json "
                        "(retained per the payload law; init/end/parity read from artifact)",
            "axis": _AXIS, "score_claim": False,
        },
        predicted_output={
            "sigma_quoted_prior": _SIGMA_QUOTED_PRIOR,
            "rung2_boundary": _RUNG2_BOUNDARY_FLIPS,
            "prediction": "quoted band holds; rung-2 REFUTED at 1.5 sigma stands",
        },
        empirical_output={
            "seed1_endpoint_flips": _SEED1_ENDPOINT_FLIPS,
            "seed2_endpoint_flips": _SEED2_ENDPOINT_FLIPS,
            "sigma_measured": _SIGMA_MEASURED,
            "boundary_straddled": True,
            "rung2_regrade": "REFUTED -> WEAKENED-DIRECTIONAL (0.9 sigma vs two-seed mean -1825.5; "
                             "sign stable; routing unchanged)",
            "sister_verdicts_unchanged": ["FRD077 0.18sigma", "q3q4 0.61-band", "F1 structural zero"],
        },
        residual=abs(_SIGMA_MEASURED - _SIGMA_QUOTED_PRIOR) / _SIGMA_QUOTED_PRIOR,  # 0.0764
        source_artifact=_MEMO,
        measurement_method="matched_config_seed_pair_endpoint_comparison_n2",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="a third matched seed tightens sigma (n=3); any future falsifier "
                                  "band on this trainer family must cite ensemble mean +/- sigma, "
                                  "never a single control endpoint",
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_mps",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Seed-ensemble falsifier-band law: sigma_est = |e1-e2|/sqrt(2) from matched-config "
              "seeds; a falsifier band whose boundary the control's own seed spread straddles is "
              "invalid as calibrated (one draw is not the population — the seed costume of the "
              "prefix-bias genus)"),
        one_line_summary=(
            "EF3000 seed pair -2286/-1365 -> sigma 651 (quoted 605 CONFIRMED); boundary -1430 "
            "STRADDLED -> rung-2 re-graded WEAKENED-DIRECTIONAL; bands need seed ensembles."
        ),
        latex_form=(
            r"\hat\sigma = \frac{|e_1 - e_2|}{\sqrt{2}},\qquad "
            r"\mathrm{band\ invalid} \iff \min(e_1,e_2) \le b \le \max(e_1,e_2)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.seed_ensemble_falsifier_band_20260817:seed_sigma_est"
        ),
        domain_of_validity={
            "vehicle": ["pr130_lift semantic renderer (blocks.{0..3}, 66,339 params)"],
            "regime": "stochastic training-window endpoints on the advisory quantized_exact_seg "
                      "instrument; generalizes as METHOD to any stochastic-endpoint falsifier band",
            "measurement_axis": [_AXIS],
            "note": "n=2 sigma is itself wide (~76% rel. error); the LAW (ensemble-calibrate, "
                    "check straddle) is exact, the sigma VALUE is provisional until n>=3",
        },
        units_in={"endpoint_a": "flips_vs_init", "endpoint_b": "flips_vs_init"},
        units_out={"sigma_est": "flips"},
        empirical_anchors=(anchor_seed_pair,),
        predicted_vs_empirical_residual={"sigma_quoted_vs_measured": 0.0764},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _MEMO,   # section 7.2 — the verdict re-grade this law licensed
            _DAG,    # FEED-2026-08-17c addendum 2
        ),
        canonical_producers=(
            "tac.pr130_lift.train_semantic_quantized_resumable",  # the seed-pair endpoint producer
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="third seed lands OR any new falsifier band is pre-registered on "
                                  "a stochastic endpoint (the law binds at band-calibration time)",
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_mps",
        ),
    )


def populate_seed_ensemble_falsifier_band_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins). EQUATIONS leg of the cw1 seed-2
    adjudication (commit 0272719002); memo leg = ddm_cw1 section 7.2; DAG leg = FEED-08-17c add.2."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_seed_ensemble_falsifier_band_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="seed_ensemble_falsifier_band_20260817 (equations leg of the EF3000 seed-pair "
              "measurement; sister of the prefix-bias genus m88/m96 on the seed axis)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "band_boundary_straddled",
    "build_seed_ensemble_falsifier_band_v1",
    "populate_seed_ensemble_falsifier_band_equation",
    "seed_sigma_est",
]
