# SPDX-License-Identifier: MIT
"""Canonical equation: an END-TO-END wall-clock rate is ``r + F/n``, never ``r`` — so a
rate measured on a short smoke and extrapolated to a long run overstates by the
fixed-cost fraction.

MEASURED 2026-08-17 (MAIN, $0, a by-product of firing ``ddm_jr1``'s ``A2_repeat``). Axis
``[wall-clock]`` — timing only; this equation NEVER touches a score, a distortion, or a
byte count. Own-vehicle frontier UNMOVED.

The defect this extincts
------------------------
``.omx/research/ddm_b2e_sealed_launch_ticket_20260816.md:3`` records, verbatim:

    166.30 s / 50 steps end-to-end ... Derived window: <=3.33 s/step -> 3,000 steps ~ 2.8 h

That 3.33 s/step became the campaign's cost model. It propagated into ``ddm_jr1``'s
"~33 min per 600-step arm" and from there into MAIN's "Leg C = 2.20 h vs a 2.75 h window".
Nobody mis-multiplied: the number was measured honestly and then carried across a scope
boundary where its DENOMINATOR changed.

``A2_repeat`` ran 600 real steps (6 checkpoints, ``verdict = PASS``) in 408 s = 0.680
s/step end-to-end. The sealed budget was **4.90x too expensive**.

The law
-------
An end-to-end rate includes a fixed cost ``F`` (process start, ``gt_cache`` load, model
init, first eval) paid once regardless of ``n``::

    t(n)          = F + n * r
    rate_e2e(n)   = t(n) / n = r + F / n

Two points separate the terms. From (50, 166.30) and (600, 408.0):

    r = 0.4395 s/step        F = 144.3 s

INTERNAL CHECK: the fit reproduces b2e's own quoted 3.33 s/step at n=50 to three digits
(3.326). At n=50, **87% of the measured wall-clock was fixed cost** — which is exactly why
one point cannot separate the terms, and one point is what every budget on this trainer
had.

OUT-OF-SAMPLE VALIDATION (the anchor that matters, pre-registered before the run landed):
``L3000_off`` at n=3,000 was predicted at 1,463 s and measured **1,552 s, +6.1%**. The two
competing models: b2e's 3.33 s/step predicts 9,972 s (**6.4x off**); a naive
re-extrapolation of the 600-step end-to-end rate predicts 2,040 s (**+31%**).

The +89 s residual is CONSISTENT WITH the 6 extra checkpoint saves the 3,000-step run
performs (13 vs 7) at ~14.8 s each — the caveat declared in advance. It is recorded as an
ATTRIBUTION, not a measurement: two equations, three unknowns, and it does NOT close
against the b2e 50-step point (a 3-save model predicts ~107 s there vs 166.30 s measured).
A third term, or a differing save count in the smoke, is unresolved.

Genus
-----
This is the PREFIX-BIAS law (``m88``/``m96`` — a prefix of a skewed population is a
different population) on the TIME axis. Sister of the standing law that an instrument's
UNITS x LEVEL x AGGREGATION are part of its claim. The structural cure, and the reason
this is registered rather than memo'd: **a smoke-derived wall-clock budget must quote
(F, r) from >=2 points, or declare itself an upper bound.**

DOMAIN BOUNDARY, measured the same day
--------------------------------------
The 3,000-step run that validated this law also returned a FORMULATION-scoped negative on
a different axis: ``best_step = 0``, ``improved_over_init = False``, ending +4,887 flips
ABOVE its own initialization with the repayment rate decayed 32x (CE -3,378/100 at step
900 -> -104/100 at step 3,000). That is an ASYMPTOTE ABOVE INIT, not a truncation, and it
REFUTED MAIN's own linear-tail extrapolation (~1,131 steps to parity; reality closed 3,767
of 8,654 flips in 2,400 steps with the rate collapsing 7.4x). Recorded here only as the
domain note that this timing law says NOTHING about whether a longer window is WORTH
buying — it prices windows, it does not value them. Verdict memo:
``.omx/research/ddm_l3000_no_descent_verdict_20260817.md``.

Memo: ``.omx/research/ddm_wallclock_prefix_bias_law_20260817.md``.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "wallclock_fixed_cost_prefix_bias_v1"
AXIS = "[wall-clock] timing only; NEVER a score, distortion, or byte count"
MEMO = ".omx/research/ddm_wallclock_prefix_bias_law_20260817.md"
VERDICT_MEMO = ".omx/research/ddm_l3000_no_descent_verdict_20260817.md"
B2E_TICKET = ".omx/research/ddm_b2e_sealed_launch_ticket_20260816.md"

#: The shared provenance pin: BOTH runs initialise from this exact checkpoint.
INIT_CKPT_SHA256 = "3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647"
GT_CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"

#: MEASURED. n_steps -> (total_seconds, n_checkpoint_saves). The 50-step row is b2e's
#: own smoke receipt; the 600- and 3,000-step rows are this measurement.
MEASURED_RUNS = {
    50: (166.30, None),      # b2e smoke; save count not recorded in the ticket
    600: (408.0, 7),         # ddm_jr1/A2_repeat
    3000: (1552.0, 13),      # ddm_jr1/L3000_off
}

#: DERIVED by two-point separation over the (50, 600) rows.
MARGINAL_SECONDS_PER_STEP = 0.4395
FIXED_COST_SECONDS = 144.3

#: The out-of-sample scoring of the three competing models at n=3,000.
PREDICTION_SCORECARD = {
    "two_point_fit": (1463.0, 0.061),        # (predicted_s, signed_relative_error)
    "b2e_single_point": (9972.0, 5.424),
    "naive_from_600_e2e": (2040.0, 0.314),
}

#: ATTRIBUTION, not a measurement (2 equations, 3 unknowns; does not close against n=50).
RESIDUAL_SECONDS = 89.0
EXTRA_SAVES_IN_3000_RUN = 6
SECONDS_PER_SAVE_ATTRIBUTED = 14.8


def end_to_end_rate(n_steps: int) -> float:
    """Seconds per step INCLUDING amortised fixed cost, for a run of ``n_steps``."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return MARGINAL_SECONDS_PER_STEP + FIXED_COST_SECONDS / n_steps


def predicted_total_seconds(n_steps: int) -> float:
    """Total wall-clock for a run of ``n_steps`` on this trainer."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return FIXED_COST_SECONDS + n_steps * MARGINAL_SECONDS_PER_STEP


def build_wallclock_fixed_cost_prefix_bias_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=INIT_CKPT_SHA256,
        source_path=MEMO,
        captured_at_utc="2026-08-17T05:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="wallclock_two_point_separation_20260817",
            measurement_utc="2026-08-17T04:23:00Z",
            inputs={
                "points": "(n=50, 166.30 s) from b2e smoke; (n=600, 408.0 s) from "
                "ddm_jr1/A2_repeat",
                "init_ckpt_sha256": INIT_CKPT_SHA256,
                "gt_cache_sha256": GT_CACHE_SHA256,
                "trainer": "tac.pr130_lift.train_semantic_quantized_resumable",
                "device": "mps",
            },
            predicted_output=3.33,
            empirical_output=3.326,
            residual=0.0012,  # |3.326 - 3.33| / 3.326, normalized magnitude
            source_artifact="/Volumes/APDataStore/pact/ddm_jr1/A2_repeat/result.json",
            measurement_method=(
                "Two-point separation of F and r; internal check = does the fit reproduce "
                "b2e's own quoted 3.33 s/step at n=50? It does, to three digits."
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
        EmpiricalAnchor(
            anchor_id="wallclock_out_of_sample_n3000_20260817",
            measurement_utc="2026-08-17T04:52:00Z",
            inputs={
                "n_steps": 3000,
                "run": "ddm_jr1/L3000_off (lr 2e-5, band OFF, rc=0)",
                "pre_registered": "prediction written to the memo BEFORE the run landed",
                "competing_models": "b2e 9972 s; naive-from-600 2040 s",
            },
            predicted_output=1463.0,
            empirical_output=1552.0,
            # |1552 - 1463| / 1552, normalized magnitude. Note the two honest
            # normalisations differ and the memo quotes the other one: +89 s is 5.7% of the
            # MEASURED total and 6.1% of the PREDICTION. Both are correct; the registry
            # field is defined as a magnitude relative to the empirical value.
            residual=0.05735,
            source_artifact=(
                "/Volumes/APDataStore/pact/ddm_jr1/L3000_off/launch_manifest.json"
            ),
            measurement_method=(
                "Out-of-sample: the fit was built on n=50 and n=600 only, and the n=3,000 "
                "prediction was pre-registered in the memo before the run finished. "
                "Residual +6.1% vs b2e's 6.4x and naive-from-600's +31%. The +89 s is "
                "CONSISTENT WITH 6 extra checkpoint saves at ~14.8 s each (13 vs 7 saves) "
                "but that is an attribution, not an isolated measurement: 2 equations, 3 "
                "unknowns, and it does not close against the n=50 point."
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="End-to-end wall-clock rate is r + F/n (wall-clock prefix bias)",
        one_line_summary=(
            "A rate measured end-to-end on a short smoke overstates a long run by the "
            "fixed-cost fraction; separate (F, r) from >=2 points or declare an upper bound"
        ),
        latex_form=r"t(n) = F + n r \quad\Rightarrow\quad \mathrm{rate}_{e2e}(n) = r + F/n",
        python_callable_module_path=(
            "tac.canonical_equations.wallclock_fixed_cost_prefix_bias_20260817:"
            "predicted_total_seconds"
        ),
        domain_of_validity={
            "trainer": "tac.pr130_lift.train_semantic_quantized_resumable",
            "device": "mps",
            "init_ckpt_sha256": INIT_CKPT_SHA256,
            "gt_cache_sha256": GT_CACHE_SHA256,
            "n_steps_range": "[50, 3000] (measured at 50, 600, 3000)",
            "checkpoint_cadence": "every 100-250 steps",
            "what_it_prices": (
                "the WALL-CLOCK of a window. It says NOTHING about whether the window is "
                "worth buying — it prices windows, it does not value them. See "
                f"{VERDICT_MEMO} for the same-day FORMULATION negative on that question."
            ),
            "what_transfers": (
                "the r + F/n FORM is general to any end-to-end-timed run. The (F, r) "
                "CONSTANTS are this trainer's and must be re-separated from >=2 points "
                "on any other trainer (constants-are-poison)."
            ),
        },
        units_in={"n_steps": "training steps (integer count)"},
        units_out={"total_wall_clock": "seconds"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            # Normalized magnitudes vs the EMPIRICAL value at each measured n.
            # n=3000 is the OUT-OF-SAMPLE row and the only one that tests the law.
            "n50_internal_check": 0.0012,
            "n3000_out_of_sample": 0.05735,
        },
        last_calibration_utc="2026-08-17T04:52:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_producers=(
            "experiments/... (none: measured from launch manifests + run receipts)",
            MEMO,
        ),
        canonical_consumers=(
            "tools/launch_detached_process.py --walltime-cap-s (budget derivation)",
            "any charter quoting a per-arm or per-window wall-clock budget",
            VERDICT_MEMO,
        ),
        provenance=provenance,
    )
