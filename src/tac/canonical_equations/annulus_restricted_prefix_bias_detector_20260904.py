# SPDX-License-Identifier: MIT
"""Canonical equation: a GLOBAL sanity check has no power against a RESTRICTED prefix bias.

THE GAP THIS CODIFIES.  [[m88]] already says "a prefix of a skewed population is a
different population", and two sister laws carry two of its costumes: the TIME axis
(`wallclock_fixed_cost_prefix_bias_v1` -- an end-to-end rate is ``r + F/n``) and the SEED
axis (`seed_ensemble_falsifier_band_v1` -- one draw is not the population).  Neither
covers the costume ddm_dr1 measured on 2026-09-03: a statistic computed over a
RESTRICTED SUB-POPULATION of pixels (the boundary annulus) inherits a large prefix bias
that the SAME field's global-pixel statistics do not show at all.  A campaign that
sanity-checks a prefix by comparing global means -- the natural, cheap check -- is
checking the wrong number and will pass.

THE MEASURED INSTANCE (ddm_dr1, n600 vs the n96 contiguous prefix of the SAME cohort,
same tool, same field |m1 - m0| = the uint8-R margin perturbation):

    annulus-restricted p95 (delta_R)  0.019590163230895963 -> 0.021881818771362305  +11.698%
    all-pixel p95                     0.038173675537109375 -> 0.0383458137512207     + 0.451%
    all-pixel mean                    0.01356075331568718  -> 0.013560148887336254   - 0.004%

    amplification A = 11.698% / 0.451% = 25.94x

THE POSITIVE CONTROL that makes this a cohort result and not an instrument result:
recomputing delta_R over frames [0:96] of the n600 run's OWN retained payload reproduces
the independent 2026-07-12 n96 artifact at relative difference **0.000e+00** -- bit
identical across two months, two runs and two thread counts.  100% of the +11.698% is
attributable to WHICH FRAMES were measured.  There is no code, environment, or
instrument contribution left to explain it away.

THE MECHANISM (why global is blind).  The restriction selects a sub-population whose
membership is itself content-dependent: the annulus is where the model is uncertain, and
the prefix's annulus is a different SET of pixels, not merely a different sample of the
same one.  dr1 measured that too -- the annulus AREA FRACTION moved +4.17% while the
global statistics did not.  Diluting a restricted-set shift over the full pixel field
divides it by the restriction's area fraction (~2.7% here), which is precisely why a
global check cannot see it.

THE DETECTOR (the reusable part; this is what the equation exports).  Given a prefix and
a full population, a global-statistic agreement is NOT evidence that a restricted
statistic agrees.  The only valid check is to re-measure the RESTRICTED statistic on the
full population, or on a seeded random draw -- never on a contiguous prefix.  Any
constant defined on a boundary-, class-, or margin-restricted set and measured on a
contiguous prefix is SUSPECT by this mechanism until re-measured.

BAND-ROBUSTNESS (so the reader does not mistake the band choice for the confound).
Narrowing the annulus 4x (|margin| < 1.00 -> 0.25) moves delta_R only -8.28%, and even
the narrowest n600 band sits +2.45% ABOVE the n96 band-1.0 value.  The falsifier verdict
survives every annulus definition measured.

CONSEQUENCE FOR THE CAMPAIGN.  `m_safe = headroom * delta_R` was resolving to a value
11.7% too low, and `m_safe` is a satisficing TARGET -- too low stops the gradient early,
so pixels declared R-safe were still flippable.  The n96 constant was ANTI-conservative,
not merely stale.  That repoint landed separately (see the dr1 addendum); this equation
is the DETECTOR, so the next such constant is caught before it ships.

VERDICT.  An APPARATUS / measurement-discipline law.  Axis `[macOS-CPU advisory]`;
NON-PROMOTABLE; it moves no pointer and is not a d_seg / d_pose / rate lever.

Producer: `tools/measure_delta_R_noise_floor.py` (re-run, never rebuilt) via
`.omx/research/ddm_dr1_delta_R_noise_floor_n600_20260904.md`.
Consumers: every charter that measures a restricted-set constant, and the sister
`margin_band_satisficing_threshold_v1` whose delta_R input this bias moved.

Memories: [[m88]] (prefix of a skewed population), [[m96]] (sign by axis).
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "annulus_restricted_prefix_bias_detector_v1"

_UTC = "2026-09-04T00:00:00Z"
_AXIS = "[macOS-CPU advisory]"
_LEDGER = ".omx/research/ddm_dr1_delta_R_noise_floor_n600_20260904.md"
_RECEIPTS = "reports/delta_R_noise_floor_n600.json"
_BH1_LEDGER = ".omx/research/ddm_bh1_fresh_eyes_bug_hunt_20260904.md"

# --- MEASURED (ddm_dr1, 2026-09-03) -------------------------------------------------
PREFIX_N = 96
POPULATION_N = 600

DELTA_R_N96 = 0.019590163230895963
DELTA_R_N600 = 0.021881818771362305
ALL_PIXEL_P95_N96 = 0.038173675537109375
ALL_PIXEL_P95_N600 = 0.0383458137512207
ALL_PIXEL_MEAN_N96 = 0.01356075331568718
ALL_PIXEL_MEAN_N600 = 0.013560148887336254
ANNULUS_AREA_FRAC_N96 = 0.025631957583957247
ANNULUS_AREA_FRAC_N600 = 0.02670194837782118

# The two biases and their ratio -- the whole content of the detector.
RESTRICTED_BIAS = DELTA_R_N600 / DELTA_R_N96 - 1.0  # +0.11697991045077849
GLOBAL_BIAS = ALL_PIXEL_P95_N600 / ALL_PIXEL_P95_N96 - 1.0  # +0.004509343459578208
AMPLIFICATION = RESTRICTED_BIAS / GLOBAL_BIAS  # 25.941672329772032

# Positive control: the prefix of THIS run vs the independent n96 artifact.
POSITIVE_CONTROL_REL_DIFF = 0.0

# Band robustness (nested subsets of the same annulus, same pass).
DELTA_R_BY_BAND: Mapping[float, float] = {
    1.00: 0.021881818771362305,
    0.50: 0.020604742,
    0.25: 0.020070553,
}

# The falsifier band the charter pre-registered, and what actually happened.
PREREGISTERED_TOLERANCE = 0.10
FALSIFIER_FIRED = True

# Default screen threshold: a restricted bias this many times the global bias is a
# detection.  DERIVED as a round order-of-magnitude floor well under the measured 25.94x,
# not fitted -- one instance cannot fit a threshold.
DEFAULT_AMPLIFICATION_THRESHOLD = 3.0


def relative_bias(prefix_value: float, population_value: float) -> float:
    """Signed relative bias of a prefix statistic against its own full population."""
    if prefix_value == 0.0:
        raise ValueError("prefix_value must be non-zero to form a relative bias")
    return float(population_value) / float(prefix_value) - 1.0


def bias_amplification(
    *,
    restricted_prefix: float,
    restricted_population: float,
    global_prefix: float,
    global_population: float,
) -> float:
    """How many times larger the RESTRICTED prefix bias is than the GLOBAL one.

    Infinite when the global statistic shows no bias at all -- which is the pure form of
    the defect, not an error.
    """
    restricted = abs(relative_bias(restricted_prefix, restricted_population))
    glob = abs(relative_bias(global_prefix, global_population))
    if glob == 0.0:
        return float("inf")
    return restricted / glob


def global_check_is_blind(
    *,
    restricted_prefix: float,
    restricted_population: float,
    global_prefix: float,
    global_population: float,
    amplification_threshold: float = DEFAULT_AMPLIFICATION_THRESHOLD,
) -> bool:
    """THE DETECTOR: True when a global-statistic sanity check cannot see the restricted bias."""
    return (
        bias_amplification(
            restricted_prefix=restricted_prefix,
            restricted_population=restricted_population,
            global_prefix=global_prefix,
            global_population=global_population,
        )
        >= amplification_threshold
    )


def prefix_constant_is_suspect(
    *, statistic_is_restricted: bool, cohort_is_contiguous_prefix: bool
) -> bool:
    """The screening rule, in the only two facts it needs.

    A constant is suspect when it is defined on a restricted sub-population (boundary /
    class / margin / annulus) AND measured on a contiguous prefix.  A global-statistic
    agreement does NOT clear it; only re-measurement of the restricted statistic on the
    full population, or on a seeded random draw, does.
    """
    return bool(statistic_is_restricted and cohort_is_contiguous_prefix)


def producer_default_reinfects_cured_constant(
    *, consumers_cured: bool, producer_default_cohort_is_prefix: bool
) -> bool:
    """THE RE-INFECTION SCREEN (ddm_bh1, 2026-09-04): curing the consumers is only half a cure.

    A retired prefix constant is RE-INFECTABLE whenever the downstream consumers were moved to
    the population value while the PRODUCER that measured it still DEFAULTS to the prefix
    cohort.  The remediation then looks complete by census -- every live consumer carries the
    right number -- yet one flagless re-run of the producer regenerates the retired value and
    re-seeds them.  A constant census answers "what do the consumers hold?"; it cannot answer
    "what will the producer emit next time?", so the two screens are independent and the cure
    is not finished until BOTH return False.
    """
    return bool(consumers_cured and producer_default_cohort_is_prefix)


def build_annulus_restricted_prefix_bias_detector_v1() -> CanonicalEquation:
    """Build the restricted-statistic prefix-bias detector equation (ddm_dr1, 2026-09-04)."""
    dr1_anchor = EmpiricalAnchor(
        anchor_id="dr1_delta_r_n600_vs_n96_prefix_annulus_vs_global_20260904",
        measurement_utc="2026-09-03T22:11:45Z",
        inputs={
            "field": "|margin_with_uint8 - margin_without_uint8| under the exact R chain",
            "restriction": "boundary annulus, |GT margin| < 1.0",
            "prefix_cohort": f"frames [0:{PREFIX_N}] (contiguous)",
            "population_cohort": f"frames [0:{POPULATION_N}] (all pairs)",
            "annulus_pixels_n600": 3_149_890,
            "annulus_area_frac_n96": ANNULUS_AREA_FRAC_N96,
            "annulus_area_frac_n600": ANNULUS_AREA_FRAC_N600,
            "gt_npz_sha256": (
                "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
            ),
            "scorer": "frozen CPU-torch SegNet; PyAV frame lineage",
            "tool": "tools/measure_delta_R_noise_floor.py --n 600 --band 1.0 --threads 4",
        },
        predicted_output={
            "preregistered_band": [
                DELTA_R_N96 * (1.0 - PREREGISTERED_TOLERANCE),
                DELTA_R_N96 * (1.0 + PREREGISTERED_TOLERANCE),
            ],
            "prior_law": (
                "the charter predicted the n600 value would land within +/-10% of the n96 "
                "value -- i.e. that the prefix was representative for this statistic"
            ),
            "falsifier": "landing outside +/-10% falsifies prefix representativeness",
        },
        empirical_output={
            "delta_r_n96": DELTA_R_N96,
            "delta_r_n600": DELTA_R_N600,
            "restricted_bias": RESTRICTED_BIAS,
            "global_p95_bias": GLOBAL_BIAS,
            "global_mean_bias": ALL_PIXEL_MEAN_N600 / ALL_PIXEL_MEAN_N96 - 1.0,
            "amplification": AMPLIFICATION,
            "falsifier_fired": FALSIFIER_FIRED,
            "positive_control_rel_diff": POSITIVE_CONTROL_REL_DIFF,
            "positive_control_reading": (
                "the prefix of THIS run reproduces the independent n96 artifact bit-identically, "
                "so 100% of the deviation is cohort and 0% is instrument"
            ),
            "delta_r_by_annulus_band": {str(k): v for k, v in DELTA_R_BY_BAND.items()},
            "band_narrowing_4x_moves_delta_r": -0.0828,
            "per_frame_p95_spread": 6.86,
            "direction": (
                "the prefix UNDERSTATED the floor, so every m_safe derived from it was "
                "ANTI-conservative, not merely stale"
            ),
        },
        residual=abs(RESTRICTED_BIAS) - PREREGISTERED_TOLERANCE,
        source_artifact=_LEDGER,
        measurement_method=(
            "same tool, same field, two cohorts; the restricted (annulus) and global "
            "(all-pixel) statistics computed from the SAME pass, plus a prefix positive "
            "control recomputed from the run's own retained payload"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIPTS,
            reactivation_criteria=(
                "a second measured instance on a DIFFERENT restriction (per-class, per-margin-"
                "band, per-region) would turn the annulus result into a family law; until then "
                "the generic form is a mechanism claim, not a measured one"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    bh1_anchor = EmpiricalAnchor(
        anchor_id="bh1_producer_default_still_the_prefix_after_consumer_cure_20260904",
        measurement_utc="2026-09-04T00:00:00Z",
        inputs={
            "producer": "tools/measure_delta_R_noise_floor.py (the tool that MEASURED both values)",
            "consumer_census": (
                "ddm_ql2/ql3 census by VALUE over src/, tools/, experiments/, scripts/ -- every "
                "live consumer moved off the retired constant"
            ),
            "screen": "argparse defaults of the producer, read at source",
        },
        predicted_output={
            "producers_still_defaulting_to_the_prefix": 0,
            "prior_law": (
                "the dr1/ql2/ql3 remediation was recorded as complete, so the prediction was "
                "that the producer of the retired constant also defaults to the population"
            ),
            "falsifier": "any producer default still naming the prefix cohort",
        },
        empirical_output={
            "producers_still_defaulting_to_the_prefix": 1,
            "producer_default_gt_npz": f"gt_n{PREFIX_N}.npz",
            "producer_default_n": PREFIX_N,
            "reinfection_open": True,
            "reading": (
                "a flagless re-run of the producer regenerated the retired "
                f"{DELTA_R_N96!r} that the consumers had just been cured of; the constant "
                "census had no power over the producer's defaults"
            ),
            "cure_landed": (
                f"defaults moved to gt_n{POPULATION_N}.npz / --n {POPULATION_N}; the prefix is "
                "now reachable only by explicit flags, with three regression tests"
            ),
        },
        residual=1.0,
        source_artifact=_BH1_LEDGER,
        measurement_method=(
            "source read of the producer's argparse defaults, plus a caller census showing no "
            "programmatic caller supplies the flags -- so the default IS the operating value"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BH1_LEDGER,
            reactivation_criteria=(
                "a second measured instance in which a cured constant's producer also kept a "
                "prefix default would make the re-infection screen a family law; today it is "
                "one measured instance of a mechanism the dilution argument already predicts"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Restricted-statistic prefix bias -- a global-pixel sanity check has no power "
            "against a boundary-restricted contiguous-prefix bias"
        ),
        one_line_summary=(
            "dr1 n96->n600: annulus p95 +11.698%, all-pixel p95 +0.451% (25.94x amplification), "
            "prefix control bit-identical -- so global agreement never clears a restricted constant"
        ),
        latex_form=(
            r"A=\frac{\left|\hat\theta^{R}_{N}/\hat\theta^{R}_{n}-1\right|}"
            r"{\left|\hat\theta^{G}_{N}/\hat\theta^{G}_{n}-1\right|};\quad "
            r"A\gg 1\Rightarrow \text{a global check on the prefix is blind};\quad "
            r"\text{MEASURED } A=25.94\ (+11.698\%\ \text{vs}\ +0.451\%)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904"
            ":global_check_is_blind"
        ),
        domain_of_validity={
            "included": [
                "statistics restricted to a content-dependent sub-population of pixels "
                "(boundary annulus; by extension class-, margin- or region-restricted sets)",
                "contiguous-prefix cohorts of the 600-pair population",
                "deterministic instruments, so the prefix positive control is available",
            ],
            "excluded": [
                "use as a d_seg / d_pose / rate lever, a score, or a promotion claim",
                "treating the 25.94x amplification as a transferable CONSTANT -- it is one "
                "measured instance of a mechanism, and its size depends on the restriction's "
                "area fraction and on how the prefix's content differs",
                "seeded RANDOM draws, which are the cure, not the disease "
                "(see seed_ensemble_falsifier_band_v1 for what random draws still owe)",
            ],
            "measurement_axis": [_AXIS],
            "result_type": (
                "APPARATUS / measurement-discipline law; NON-PROMOTABLE; moves no pointer"
            ),
            "sister_laws": [
                "wallclock_fixed_cost_prefix_bias_v1 -- the TIME costume of [[m88]] (r + F/n)",
                "seed_ensemble_falsifier_band_v1 -- the SEED costume (one draw is not a population)",
                "margin_band_satisficing_threshold_v1 -- the consumer whose delta_R input moved",
            ],
            "known_boundary": (
                "n=1 restriction measured (the annulus). The generic statement over other "
                "restrictions is DERIVED from the dilution mechanism, not MEASURED."
            ),
            "cure": (
                "re-measure the RESTRICTED statistic on the full population, or on a seeded "
                "random draw; never clear a restricted constant with a global-statistic check"
            ),
        },
        units_in={
            "restricted_prefix": "same_units_as_the_restricted_statistic",
            "restricted_population": "same_units_as_the_restricted_statistic",
            "global_prefix": "same_units_as_the_global_statistic",
            "global_population": "same_units_as_the_global_statistic",
            "amplification_threshold": "dimensionless_ratio",
        },
        units_out={
            "relative_bias": "dimensionless_fraction",
            "bias_amplification": "dimensionless_ratio",
            "global_check_is_blind": "bool",
            "prefix_constant_is_suspect": "bool",
            "producer_default_reinfects_cured_constant": "bool",
        },
        empirical_anchors=(dr1_anchor, bh1_anchor),
        predicted_vs_empirical_residual={
            "dr1_delta_r_n600_vs_n96_prefix_annulus_vs_global_20260904": (
                abs(RESTRICTED_BIAS) - PREREGISTERED_TOLERANCE
            ),
            "bh1_producer_default_still_the_prefix_after_consumer_cure_20260904": 1.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _LEDGER,
            ".omx/research/ddm_eq1_equations_leg_backfill_20260904.md",
        ),
        canonical_producers=("tools/measure_delta_R_noise_floor.py", _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="annulus_restricted_prefix_bias_detector.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
