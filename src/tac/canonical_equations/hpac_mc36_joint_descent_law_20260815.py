# SPDX-License-Identifier: MIT
"""Canonical equation: the HPAC-on-MC36-labels two-phase joint-byte DESCENT LAW
(exp-floor per phase; fitted, not latched — the equations-leg instrument that
sized the rx2 GPU-race N=480 extension; wc2 memo §5a/§5e).

THE LAW (per training phase, joint byte estimate vs epoch)
----------------------------------------------------------
Within one phase (continuous / discrete-QAT) of the identity-pinned HPAC
trainer on the FIXED MC36 token labels, the joint byte estimate
(tokens + model) follows an exponential approach to a floor::

    y(t) = y_inf + A * exp(-(t - t0) / tau)

selected over the power-law alternative by SSE (equal parameter count ->
SSE ordering == AIC ordering) on the trainer's own eval rows.  The stopping
epoch for a byte bar ``b`` follows in closed form::

    N*(b) = t0 + tau * ln(A / b)

with bars expressed in canonical score bands (1 band = 3.5e-6 S on the T4
axis = 3.5e-6 * 37_545_489 / 25 ~= 5.256 B on the rate term).

MEASURED anchors (60-epoch MPS race, run counter 8, rc=0; fit receipt
``descent_law_fit_full60.json`` sha 475fd58ea0099730..., source run.log sha
9a40f08d73f20070...; ``[macOS-MLX research-signal]`` NON-PROMOTABLE — MPS is
the training substrate, the CPU IHS1 pack is serialization authority, T4
exact eval is the only score authority):

  * continuous phase: y_inf = 135,248.42 B, A = 9,676.78 B, tau = 9.90 ep
    (rms 265.9 B over 16 rows) -> N*(1 band) = 74.4 relative epochs.
  * discrete_qat phase: y_inf = 132,798.41 B, A = 2,796.33 B, tau = 39.67 ep
    (rms 140.4 B over 15 rows) -> N*(1 band) = 249.0 relative epochs; the
    ep60 endpoint (134,323 B) sits 1,525 B ABOVE its own QAT asymptote.

TRANSFER FIREWALL (cross-regime constant-transfer law, memory m21/m22): the
CONTINUOUS-phase law projects within-regime (same archive-derived init, same
seed).  The QAT-phase law is ENTRY_STATE_CONDITIONAL — a longer run enters
QAT from a deeper continuous endpoint, so its constants are re-fit per run,
never transferred.  The N=480 extension (e480b) is the first live test of
that label.

CONSUMPTION (the law's first decision): N=480 at qat_fraction 0.5 gives 240
continuous epochs (saturated: remaining gain ~0 B) + 240 QAT epochs (~6 tau:
remaining gain ~7 B < 2 bands) = full squeeze; N=240 would strand ~137 B
(~9.1e-5 S on the rate term).  Sealed as wrapper mode ``full-mps-e480``.

means != ends: research-signal fit rows; the pointer moves only through a
byte-closed T4 exact row on the raced archive.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "hpac_mc36_joint_descent_law_v1"

_UTC_FIT = "2026-08-15T00:00:00Z"
_RATE_DENOM = 37_545_489
_BAND_BYTES = 3.5e-6 * _RATE_DENOM / 25.0  # ~5.256 B per canonical band
_RECEIPT = (
    "/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full/"
    "descent_law_fit_full60.json"
)
_RECEIPT_SHA16 = "475fd58ea0099730"
_RUNLOG_SHA16 = "9a40f08d73f20070"
_MEMO = ".omx/research/ddm_wc2_hpac_mps_port_20260814.md"

# Fitted constants (full60 receipt; run-scoped, NOT transferable across regimes)
CONTINUOUS_FIT = {"y_inf": 135_248.4237, "amp": 9_676.7752, "tau": 9.9004}
QAT_FIT = {"y_inf": 132_798.4125, "amp": 2_796.3283, "tau": 39.6722}


def remaining_gain_bytes(amp: float, tau: float, t_rel: float) -> float:
    """Fitted remaining descent (bytes) from relative epoch ``t_rel`` to the floor."""
    return float(amp) * math.exp(-float(t_rel) / float(tau))


def n_star_relative_epochs(amp: float, tau: float, bar_bytes: float = _BAND_BYTES) -> float:
    """Closed-form stopping epoch (relative to phase entry) for byte bar ``bar_bytes``."""
    if amp <= bar_bytes:
        return 0.0
    return float(tau) * math.log(float(amp) / float(bar_bytes))


def build_hpac_mc36_joint_descent_law_v1() -> CanonicalEquation:
    """Build the two-phase exp-floor descent law with the full60 fit anchors."""

    anchor_full60_fit = EmpiricalAnchor(
        anchor_id="hpac_mc36_full60_two_phase_exp_floor_fit_20260815",
        measurement_utc=_UTC_FIT,
        inputs={
            "vehicle": "rx2_mc36_label_hpac (PR130-lineage HPAC on FIXED MC36 tokens)",
            "trainer": "tools/train_ddm_cl1_hpac_capacity.py (via wc2 sealed MPS wrapper)",
            "fitter": "tools/fit_hpac_descent_law.py",
            "run": "gpu_race/full (60 ep, counter 8, rc=0, 49.6 s/ep, qat_fraction 0.5)",
            "metric": "estimated_joint_bytes (tokens + model)",
            "runlog_sha256_16": _RUNLOG_SHA16,
        },
        predicted_output={"law_form": "exp_floor beats power by SSE in BOTH phases"},
        empirical_output={
            "continuous": {**CONTINUOUS_FIT, "rms_bytes": 265.91, "n_rows": 16,
                           "n_star_1band_rel": 74.43, "n_star_10band_rel": 51.63},
            "discrete_qat": {**QAT_FIT, "rms_bytes": 140.39, "n_rows": 15,
                             "n_star_1band_rel": 249.01, "n_star_10band_rel": 157.66,
                             "transfer_label": "ENTRY_STATE_CONDITIONAL"},
            "endpoint_ep60_joint_bytes": 134_323,
            "endpoint_gap_above_qat_floor_bytes": 1_525,
            "selection": "exp_floor won both phases by SSE",
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method="scipy_curve_fit_sse_selection_over_trainer_eval_rows",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "re-fit at every run endpoint (constants are run-scoped); the e480b "
                "extension endpoint is the QAT ENTRY_STATE_CONDITIONAL label's first test"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_n480_sizing = EmpiricalAnchor(
        anchor_id="hpac_mc36_n480_extension_sized_from_law_20260815",
        measurement_utc=_UTC_FIT,
        inputs={
            "decision": "extension epoch budget for the wc2 MPS race line",
            "candidates_projected": [60, 120, 240, 480, 960],
            "band_bytes": _BAND_BYTES,
        },
        predicted_output={
            "n_480": "240 cont (saturated) + 240 QAT (~6 tau) leaves ~7 B < 2 bands",
            "n_240": "strands ~137 B (~9.1e-5 S rate)",
        },
        empirical_output={
            "selected": 480,
            "sealed_as": "tools/train_ddm_cl1_hpac_capacity_mps.py PORT_MODES['full-mps-e480']",
            "fired": "gpu_race/full_e480b (pid 13787, counter 10, watchers armed at launch)",
            "note": "CONSUMPTION event — the law's projection, not a measurement of it; "
                    "the e480b endpoint refit closes the loop",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="closed_form_n_star_from_fitted_law",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="compare the e480b measured endpoint vs the projected "
                                  "~132,805 B; a miss re-opens the law form",
            measurement_axis="[research-signal]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="HPAC/MC36 two-phase exp-floor joint-byte descent law (fitted per run, never latched)",
        one_line_summary=(
            "Per phase y(t)=y_inf+A*exp(-t/tau): cont floor 135,248 B (tau 9.9), QAT floor "
            "132,798 B (tau 39.7, ENTRY_STATE_CONDITIONAL); N*(b)=tau*ln(A/b) sized N=480."
        ),
        latex_form=(
            r"y(t)=y_\infty+A e^{-(t-t_0)/\tau};\quad N^*(b)=t_0+\tau\ln(A/b);\ "
            r"y_\infty^{cont}{=}135248,\ \tau{=}9.9;\ y_\infty^{qat}{=}132798,\ \tau{=}39.7"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.hpac_mc36_joint_descent_law_20260815:"
            "n_star_relative_epochs"
        ),
        domain_of_validity={
            "vehicle": ["rx2_mc36_label_hpac (archive-derived init, seed 20260716)"],
            "phase_scoped": True,
            "constants_run_scoped": True,
            "qat_transfer": "ENTRY_STATE_CONDITIONAL (re-fit per run entry state)",
            "measurement_axis": ["macOS-MLX research-signal"],
            "note": "byte ESTIMATES from the trainer's eval rows; the shipped archive byte "
                    "count comes only from the CPU IHS1 pack + identity race",
        },
        units_in={"t": "epochs (relative to phase entry)", "bar_bytes": "bytes"},
        units_out={"y": "estimated joint bytes", "n_star": "epochs"},
        empirical_anchors=(anchor_full60_fit, anchor_n480_sizing),
        predicted_vs_empirical_residual={
            "scipy_curve_fit_sse_selection_over_trainer_eval_rows": 0.0,
        },
        last_calibration_utc=_UTC_FIT,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/train_ddm_cl1_hpac_capacity_mps.py",
            "experiments/ddm_rx2_mc36_identity_race.py",
        ),
        canonical_producers=(
            "tools/fit_hpac_descent_law.py",
            "tools/train_ddm_cl1_hpac_capacity.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="hpac_mc36_joint_descent_law.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_hpac_mc36_joint_descent_law(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """APPEND-ONLY registration of the descent law into the canonical registry
    (latest-row-wins query semantics).  Equations leg of the wc2 §5a instrument;
    DAG leg = wc2 memo §5e; run leg = the e480b extension it sized."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_hpac_mc36_joint_descent_law_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="hpac_mc36_joint_descent_law_20260815 (wc2 §5a equations-leg debt paid: "
              "fitted two-phase exp-floor law + the N=480 sizing consumption anchor; "
              "receipt sha " + _RECEIPT_SHA16 + ")",
    )
    return eq
