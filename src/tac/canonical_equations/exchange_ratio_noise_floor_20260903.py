# SPDX-License-Identifier: MIT
"""Canonical equation: the noise floor of the byte<->distortion EXCHANGE RATIO (ddm_xr1, #1248).

THE GAP THIS CODIFIES.  The campaign closes and re-opens rows on statements of the
form "row X is 1.14x / 0.94x / 1.66x its bar".  Until 2026-09-03 the ratio in those
statements was a POINT with no measured dispersion: `ddm_rn1_n600_reopen_sweep`
screened 300 such near-win rows, and `ddm_ww1_walls_that_werent` sec 3.5 recorded
plainly that "nobody has measured it".  A margin of 1.04x is a win or a loss
depending on a number the campaign had never produced.

THE ESTIMAND (MAIN's binding definition, ddm_xr1 charter 2026-09-03).

* Object: the exact RC64 token-stream byte count B(F, M, G) of a field F under the
  shipped model M and causal schedule G, and the realized distortion
  D(F) = (100*d_seg, sqrt(10*d_pose)) through the frozen CPU scorers.  The exchange
  ratio of an edit set E is r(E) = dS_rate(E) / dS_dist(E), with
  dS_rate = 25*dB/37_545_489.
* PHYSICAL repeat unit: one complete re-encode of the SAME field under the SAME
  model.  sigma_B is the spread of B across physically repeated encodes.
* STATISTICAL repeat unit: the PAIR-level bootstrap.  B and D are sums over 600
  pairs; resample pairs with replacement (seeded) to get the sampling interval of
  dB and dD.  Site-level resampling is FORBIDDEN -- `ddm_fs3` measured AVERAGE !=
  MARGINAL by 2.24x, so sites inside one pair are not exchangeable.
* Denominator: dS_dist in S units, never d_seg alone.
* ACCEPTANCE: a near win is ADMISSIBLE only if its dS is negative at the upper edge
  of the 95% pair-bootstrap interval.

THE LAW.  For a resample b of the 600 pair indices,

    dB_b   = sum_{i in b} delta_i + c,      c = dB_exact - sum_{i=1..600} delta_i
    dS_b   = 100*(dseg_cand_b - dseg_base_b)
             + sqrt(10*dpose_cand_b) - sqrt(10*dpose_base_b)
             + 25*dB_b / 37_545_489
    ADMISSIBLE(E)  iff  quantile_0.975({dS_b}) < 0

The fixed term c carries the sub-byte RC64 rounding residual and any fixed
container bytes.  It is held CONSTANT across resamples so the identity draw
reproduces the retained PHYSICAL integer-byte delta exactly: the interval is
centred on a number we actually measured, not on an ideal-codelength surrogate.
The same resample index vector must drive bytes AND both distortion axes, or the
pairing between rate and distortion is destroyed and the ratio is not a ratio.

MEASURED VALUES (2026-09-03, seed 20260903, 200 resamples, full n600 population):

* sigma_B = 0.0 B over 3 physical null re-encodes; all three streams and archives
  byte-identical.  The coder contributes NO byte noise; the entire exchange-ratio
  noise floor is STATISTICAL (which 600 pairs), not physical.
* JBP1 row A (5,506 XOV1 B/H/W edits across 567 pairs, exact dB = -2,950 B):
  95% interval [-3159.27, -2758.32] B, half-width 200.48 B.
* FCD3 published tau_1e-6 (exact dB = -2,940 B, realized n600 scorers): point
  dS = +0.00194332; 95% interval [+0.00171513, +0.00219814], half-width
  0.00024151 -- EXCLUDES ZERO, so the win-win cone stays REFUSED.
  r = -0.5018, 95% interval [-0.5311, -0.4733]: the rate credit pays back only
  about half the distortion it buys.

VERDICT (honest, NO-FAKE): an APPARATUS / measurement-discipline law.  It moves no
pointer and is not a d_seg / d_pose / rate lever.  Axis
`[macOS-CPU advisory / scorer-free exact byte replay plus retained-score bootstrap]`;
NON-PROMOTABLE; no scorer was run to produce it (retained per-pair receipts only).

TRANSFER BOUNDARY (binding): an interval belongs to ITS physical object and pair
population.  FCD3's interval may NOT be transferred to a different edit set.  A row
without matched same-object n600 per-pair byte and distortion receipts is
UNGRADABLE, and saying so is the correct answer.

Producer: `experiments/ddm_xr1_exchange_ratio_noise_floor.py`.
Consumers: the near-win acceptance rule in `.omx/research/ddm_rn1_n600_reopen_sweep_20260903.md`
and `.omx/research/ddm_xr1_exchange_ratio_noise_floor_20260903.md`.
"""

from __future__ import annotations

import numpy as np

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

EQUATION_ID = "exchange_ratio_noise_floor_v1"

_UTC = "2026-09-03T00:00:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free exact byte replay plus retained-score bootstrap]"
_LEDGER = ".omx/research/ddm_xr1_exchange_ratio_noise_floor_20260903.md"

# Contest scoring constants (upstream/evaluate.py).
PAIR_COUNT = 600
RATE_NUMERATOR = 25.0
RATE_DENOMINATOR_BYTES = 37_545_489

# Bootstrap spec, mirrored from the producer; the test cross-checks for drift.
BOOTSTRAP_SEED = 20_260_903
BOOTSTRAP_RESAMPLES = 200
INTERVAL_LOW_Q = 0.025
INTERVAL_HIGH_Q = 0.975

# MEASURED 2026-09-03 (ddm_xr1).  Re-derive at every pointer move.
SIGMA_B_BYTES = 0.0
PHYSICAL_REPEATS_MEASURED = 3
JBP1_ROW_A_EXACT_DELTA_BYTES = -2_950
JBP1_ROW_A_INTERVAL_BYTES = (-3159.2735487612390, -2758.3183078847120)
FCD3_EXACT_DELTA_BYTES = -2_940
FCD3_POINT_DELTA_S = 0.0019433243907622244
FCD3_DELTA_S_INTERVAL = (0.0017151288265233415, 0.0021981402807044960)
FCD3_EXCHANGE_RATIO_POINT = -0.5018330063791281
FCD3_EXCHANGE_RATIO_INTERVAL = (-0.5310976728579913, -0.4732671579713654)


def draw_pair_indices(
    *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES
) -> np.ndarray:
    """Seeded pair-resample index matrix: `resamples` rows of 600 pair ids, with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, PAIR_COUNT, size=(resamples, PAIR_COUNT), dtype=np.uint16)


def calibration_constant_bytes(pair_delta_bytes: np.ndarray, exact_delta_bytes: float) -> float:
    """`c` -- the fixed sub-byte rounding plus container residual held constant per resample."""
    return float(exact_delta_bytes) - float(np.asarray(pair_delta_bytes, dtype=np.float64).sum())


def bootstrap_delta_bytes(
    pair_delta_bytes: np.ndarray, draws: np.ndarray, *, exact_delta_bytes: float
) -> np.ndarray:
    """dB_b for every resample, exact-total calibrated so the identity draw gives dB_exact."""
    delta = np.asarray(pair_delta_bytes, dtype=np.float64)
    if delta.shape != (PAIR_COUNT,):
        raise ValueError("pair_delta_bytes must be exactly n600")
    return delta[np.asarray(draws)].sum(axis=1) + calibration_constant_bytes(
        delta, exact_delta_bytes
    )


def bootstrap_mean(values: np.ndarray, draws: np.ndarray, *, exact_mean: float) -> np.ndarray:
    """Resampled per-pair mean, calibrated so the identity draw reproduces the retained aggregate."""
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (PAIR_COUNT,):
        raise ValueError("per-pair vector must be exactly n600")
    return data[np.asarray(draws)].mean(axis=1) + (exact_mean - float(data.mean()))


def delta_s_from_components(
    *,
    base_d_seg: np.ndarray | float,
    candidate_d_seg: np.ndarray | float,
    base_d_pose: np.ndarray | float,
    candidate_d_pose: np.ndarray | float,
    delta_bytes: np.ndarray | float,
) -> np.ndarray:
    """dS = 100*d(d_seg) + [sqrt(10*d_pose_cand) - sqrt(10*d_pose_base)] + 25*dB/37,545,489."""
    seg = 100.0 * (np.asarray(candidate_d_seg, dtype=np.float64) - np.asarray(base_d_seg, dtype=np.float64))
    pose = np.sqrt(10.0 * np.asarray(candidate_d_pose, dtype=np.float64)) - np.sqrt(
        10.0 * np.asarray(base_d_pose, dtype=np.float64)
    )
    rate = RATE_NUMERATOR * np.asarray(delta_bytes, dtype=np.float64) / RATE_DENOMINATOR_BYTES
    return seg + pose + rate


def percentile_interval_95(values: np.ndarray) -> tuple[float, float]:
    """The symmetric 95% percentile band of a resample distribution."""
    low, high = np.quantile(
        np.asarray(values, dtype=np.float64), [INTERVAL_LOW_Q, INTERVAL_HIGH_Q], method="linear"
    )
    return float(low), float(high)


def near_win_is_admissible(delta_s_samples: np.ndarray) -> bool:
    """THE ACCEPTANCE RULE: admissible iff dS is negative at the 95% interval's UPPER edge.

    A point estimate below zero is NOT sufficient; a near win whose interval reaches
    zero is a coin flip dressed as a result.
    """
    return bool(percentile_interval_95(delta_s_samples)[1] < 0.0)


def exchange_ratio_is_defined(delta_s_distortion_samples: np.ndarray) -> bool:
    """r has a finite interval only when its denominator keeps one sign across resamples."""
    data = np.asarray(delta_s_distortion_samples, dtype=np.float64)
    return bool(np.all(data > 0.0) or np.all(data < 0.0))


def build_exchange_ratio_noise_floor_v1() -> CanonicalEquation:
    """Build the exchange-ratio noise-floor canonical equation (ddm_xr1, 2026-09-03)."""
    physical_anchor = EmpiricalAnchor(
        anchor_id="physical_null_reencode_x3_sigma_b_20260903",
        measurement_utc=_UTC,
        inputs={
            "object": "afr1 shipped field, shipped model, shipped causal schedule; null overlay",
            "repeat_unit": "one complete physical n600 RC64 re-encode via the RXC1 exact coder",
            "physical_repeats": PHYSICAL_REPEATS_MEASURED,
            "edit_tokens_changed": 0,
            "scorer_runs": 0,
        },
        predicted_output={
            "sigma_b_bytes": 0.0,
            "all_repeats_byte_identical": True,
            "prior_law": "the RC64 coder is deterministic, so physical byte noise is exactly zero",
        },
        empirical_output={
            "sigma_b_bytes": SIGMA_B_BYTES,
            "all_streams_byte_identical": True,
            "all_archives_byte_identical": True,
            "consequence": (
                "the exchange-ratio noise floor is ENTIRELY statistical (which 600 pairs), "
                "not physical; a re-encode adds no dispersion to a byte claim"
            ),
        },
        residual=0.0,
        source_artifact=_LEDGER,
        measurement_method="three_complete_physical_rc64_null_reencodes_byte_compared",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "re-measure sigma_B whenever the coder, the RC64 build, the shipped model, or the "
                "causal schedule changes; a nonzero sigma_B refutes the determinism premise every "
                "byte claim in the campaign rests on"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    jbp1_anchor = EmpiricalAnchor(
        anchor_id="jbp1_row_a_pair_bootstrap_delta_bytes_20260903",
        measurement_utc=_UTC,
        inputs={
            "object": "JBP1 row A -- 5,506 XOV1 B/H/W edits across 567 of 600 pairs",
            "exact_delta_bytes": JBP1_ROW_A_EXACT_DELTA_BYTES,
            "per_pair_ledger": "retained/exact/{null,xov1_bhw5506}/bits_per_frame_exact.npy",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "calibration_constant_bytes": 0.4534463945501557,
        },
        predicted_output={
            "interval_half_width_bytes_below": 600.0,
            "charter_prior_law": "narrower than +/-600 B, so the rate credit is real",
        },
        empirical_output={
            "interval_95_bytes": list(JBP1_ROW_A_INTERVAL_BYTES),
            "half_width_bytes": 200.47762043826356,
            "resample_sd_bytes": 105.225891,
            "monte_carlo_residual_bytes": 1.657060,
            "prediction_held": True,
            "interval_excludes_zero": True,
        },
        residual=1.657060,
        source_artifact=_LEDGER,
        measurement_method="seeded_pair_level_bootstrap_of_retained_per_frame_byte_ledger",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "recompute when JBP1's retained per-frame ledger or its exact -2,950 B receipt "
                "changes; raise the resample count if the Monte-Carlo residual matters at the "
                "precision a decision needs"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    fcd3_anchor = EmpiricalAnchor(
        anchor_id="fcd3_pair_bootstrap_delta_s_and_exchange_ratio_20260903",
        measurement_utc=_UTC,
        inputs={
            "object": "FCD3 published tau_1e-6 -- 4,194-edit field plus 448-pair pose carrier",
            "exact_delta_bytes": FCD3_EXACT_DELTA_BYTES,
            "body_exact_delta_bytes": -2_965,
            "fixed_pose_carrier_bytes": 25,
            "base_d_seg": 0.0003474002587608993,
            "candidate_d_seg": 0.0003874630492646247,
            "base_d_pose": 0.0001470109127694741,
            "candidate_d_pose": 0.00014620431466028094,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "scorer_runs": 0,
        },
        predicted_output={
            "delta_s_point": FCD3_POINT_DELTA_S,
            "charter_prior_law": "the interval EXCLUDES zero, so the win-win cone stays refused",
            "falsifier": "an interval that includes zero reopens the cone at n600",
        },
        empirical_output={
            "delta_s_interval_95": list(FCD3_DELTA_S_INTERVAL),
            "delta_s_half_width": 0.00024150572709057724,
            "delta_s_resample_sd": 0.000125289,
            "monte_carlo_residual_delta_s": 6.621e-06,
            "interval_excludes_zero": True,
            "admissible": False,
            "falsifier_fired": False,
            "exchange_ratio_point": FCD3_EXCHANGE_RATIO_POINT,
            "exchange_ratio_interval_95": list(FCD3_EXCHANGE_RATIO_INTERVAL),
            "exchange_ratio_denominator_sign_stable": True,
            "reading": (
                "r = -0.50 +/- 0.029: the rate credit pays back only about half the "
                "distortion it buys, and the interval never reaches the break-even r = -1"
            ),
        },
        residual=6.621e-06,
        source_artifact=_LEDGER,
        measurement_method=(
            "seeded_pair_level_bootstrap_coupling_the_same_resample_across_bytes_and_both_"
            "retained_n600_distortion_axes"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "recompute against a fresh n600 scorer order if FCD3's retained per-pair receipts "
                "are superseded; the interval is object-specific and must never be transferred to "
                "another edit set"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Exchange-ratio noise floor -- the pair-bootstrap interval a byte<->distortion "
            "near-win margin must clear before it is admissible"
        ),
        one_line_summary=(
            "sigma_B = 0 (deterministic coder), so the floor is the PAIR bootstrap: +/-200 B on a "
            "~3,000 B rate credit, +/-0.00024 S on a realized dS; admit iff the 95% upper dS < 0"
        ),
        latex_form=(
            r"\Delta B_b=\sum_{i\in b}\delta_i+c,\ "
            r"c=\Delta B_{\mathrm{exact}}-\sum_i \delta_i;\ "
            r"\Delta S_b=100\Delta d_{\mathrm{seg},b}"
            r"+\sqrt{10 d^{\mathrm{cand}}_{\mathrm{pose},b}}-\sqrt{10 d^{\mathrm{base}}_{\mathrm{pose},b}}"
            r"+\frac{25\Delta B_b}{37545489};\ "
            r"\mathrm{ADMIT}\iff Q_{0.975}(\Delta S_b)<0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.exchange_ratio_noise_floor_20260903:near_win_is_admissible"
        ),
        domain_of_validity={
            "population": ["full n600 pair population; seeded random resampling only"],
            "forbidden_resampling": [
                "site-level resampling (ddm_fs3 measured AVERAGE != MARGINAL by 2.24x)",
                "any contiguous prefix subset (pose prefixes measure 2.54-4.21x harder)",
            ],
            "measurement_axis": [_AXIS],
            "result_type": (
                "APPARATUS / measurement-discipline law; NOT a d_seg / d_pose / rate lever; "
                "moves no pointer"
            ),
            "transfer_rule": (
                "an interval is valid ONLY for its own physical object and pair population; a row "
                "without matched same-object n600 per-pair byte AND distortion receipts is "
                "UNGRADABLE"
            ),
            "constants_mirror": (
                "experiments/ddm_xr1_exchange_ratio_noise_floor.py (drift-guarded by the test)"
            ),
            "known_boundary": (
                "the 25 B fixed pose-carrier term is held constant across resamples because no "
                "per-pair breakdown of it was retained; its own dispersion is not modelled"
            ),
        },
        units_in={
            "pair_delta_bytes": "bytes_per_pair",
            "exact_delta_bytes": "bytes",
            "base_d_seg": "argmax_disagreement_fraction",
            "base_d_pose": "mean_squared_error",
        },
        units_out={
            "near_win_is_admissible": "bool",
            "delta_s_interval": "score_units",
            "delta_bytes_interval": "bytes",
        },
        empirical_anchors=(physical_anchor, jbp1_anchor, fcd3_anchor),
        predicted_vs_empirical_residual={
            "physical_null_reencode_x3_sigma_b_20260903": 0.0,
            "jbp1_row_a_pair_bootstrap_delta_bytes_20260903": 1.657060,
            "fcd3_pair_bootstrap_delta_s_and_exchange_ratio_20260903": 6.621e-06,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            ".omx/research/ddm_xr1_exchange_ratio_noise_floor_20260903.md",
            ".omx/research/ddm_rn1_n600_reopen_sweep_20260903.md",
        ),
        canonical_producers=("experiments/ddm_xr1_exchange_ratio_noise_floor.py",),
        provenance=build_provenance_for_predicted(
            model_id="exchange_ratio_noise_floor.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EQUATION_ID",
    "PAIR_COUNT",
    "RATE_DENOMINATOR_BYTES",
    "RATE_NUMERATOR",
    "SIGMA_B_BYTES",
    "bootstrap_delta_bytes",
    "bootstrap_mean",
    "build_exchange_ratio_noise_floor_v1",
    "calibration_constant_bytes",
    "delta_s_from_components",
    "draw_pair_indices",
    "exchange_ratio_is_defined",
    "near_win_is_admissible",
    "percentile_interval_95",
]
