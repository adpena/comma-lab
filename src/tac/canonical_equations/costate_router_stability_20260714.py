# SPDX-License-Identifier: MIT
"""Canonical law for costate-organ router margin, replay, and IS correction.

The equation is standalone and has no registry side effect.  This avoids the
shared-registry collision while preserving an explicit population function for a
main-owned merge or a temporary test registry.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "costate_router_stability_v1"
MEASUREMENT_UTC = "2026-07-14T11:55:00Z"
DAG_FEED = ".omx/research/costate_organ_router_stability_DAG_FEED_20260714.md"
RECEIPT = ".omx/research/codex_findings_costate_organ_router_stability_20260714_codex.md"
AXIS = "[macOS-CPU advisory; deterministic NumPy-fp32 router; no score authority]"


def signed_selection_margin(recent_slope: float, running_median: float) -> float:
    """Return the fp32 signed plateau/transient boundary margin."""
    recent = float(recent_slope)
    median = float(running_median)
    if not (math.isfinite(recent) and math.isfinite(median)):
        raise ValueError("router margin inputs must be finite")
    import numpy as np

    return float(np.float32(np.float32(recent) - np.float32(median)))


def self_normalized_ratio_weights(
    ratios: Sequence[float],
    mask: Sequence[bool],
    clip_low: float,
    clip_high: float,
) -> tuple[float, ...]:
    """Reference form of ``n_M M_i clip(r_i,l,u) / sum_j ...``.

    Distribution hashes and support are validated by the production
    ``router_stability`` module before ratios reach this pure equation helper.
    """
    if len(ratios) != len(mask) or not ratios:
        raise ValueError("ratios and mask must be equally sized and non-empty")
    if not (0.0 < clip_low <= clip_high):
        raise ValueError("clip bounds must satisfy 0 < low <= high")
    retained = [min(max(float(r), clip_low), clip_high) if bool(m) else 0.0
                for r, m in zip(ratios, mask, strict=True)]
    if any(not math.isfinite(v) or v < 0.0 for v in retained):
        raise ValueError("ratios must be finite and nonnegative")
    n_retained = sum(bool(m) for m in mask)
    total = sum(retained)
    if n_retained == 0 or total <= 0.0:
        raise ValueError("mask leaves no positive supported ratio")
    return tuple(n_retained * value / total for value in retained)


def sequential_beta_route_match_posterior(
    outcomes: Sequence[bool],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return ``(alpha_t, beta_t, E[q_t])`` after every route-match label.

    The Beta(1,1) default is an explicit ASSUMED data-neutral controller prior,
    not a promotion threshold.  Production calibration additionally requires the
    confidence-bin and distribution-custody guards in ``router_stability``.
    """
    alpha = float(prior_alpha)
    beta = float(prior_beta)
    if not (math.isfinite(alpha) and math.isfinite(beta) and alpha > 0.0 and beta > 0.0):
        raise ValueError("Beta prior parameters must be finite and > 0")
    path = []
    for matched in outcomes:
        alpha += float(bool(matched))
        beta += float(not bool(matched))
        path.append((alpha, beta, alpha / (alpha + beta)))
    return tuple(path)


def build_costate_router_stability_v1() -> CanonicalEquation:
    """Build the research-only router-stability law and real #205 anchor."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "accrue a separately hashed visited-live regime-density manifest, derive clip "
            "bounds on a past-only training surface, then rerun the real walk-forward IS "
            "gate; require >=2 independent trajectories before a transfer claim"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp32",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="costate_router_stability_real205_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "run_ref": "levelset_v752_baseline_20260710T185913Z",
            "daemon_log_sha256": (
                "7fdc44d19946121fb18e35060f5146bf1f48dea81c08891f8f4477d42b0bed82"),
            "n_verdicts": 10,
            "n_intervals": 9,
            "walk_forward_folds": 7,
            "gate_dtype": "numpy.float32",
            "policy": "transient->T_gp; plateau/uncertain->persistence",
        },
        predicted_output={
            "route_is_repeatable": True,
            "every_route_has_positive_margin": True,
            "is_weighting_is_identified": True,
        },
        empirical_output={
            "legacy_dispatcher_wf_mae": 0.0015959393896760557,
            "legacy_global_single_best_wf_mae": 0.00185206618604584,
            "legacy_persistence_wf_mae": 0.002791931483929152,
            "exact_zero_margin_folds": [75.0, 125.0],
            "zero_margin_fold_count": 2,
            "gate_unstable_fold_fraction": 2.0 / 7.0,
            "forecast_calibration": {
                "status": "MIS_CALIBRATED_INSTANCE",
                "float32_stable_oracle_matches": "3/5",
                "within_roundoff_oracle_matches": "2/2",
                "high_minus_low_match_rate": -0.4,
                "terminal_route_match_posterior": "Beta(6,3)",
                "terminal_posterior_match_probability": 2.0 / 3.0,
                "prior": "ASSUMED Beta(1,1); no promotion threshold",
                "compute_allocation": (
                    "K2 shadow selected-tool+A_ridge_solve for all folds; "
                    "current route unchanged; actuation NONE"),
            },
            "decision_replay_contract": "REPLAY_MATCH plus durable mismatch alarm",
            "router_learning_frozen": False,
            "is_status": "BLOCKED_DISTRIBUTION_CUSTODY",
            "density_custody_evidence": {
                "independent_real_trajectories": 1,
                "production_causal_manifest_files": 0,
                "executed_decision_rows": 0,
                "coverage_receipt_rows": 0,
            },
            "verdict": (
                "gate-drift sensitivity PRESENT at two exact ties; decide/apply mismatch "
                "was structurally possible but is now guarded; real IS robustness remains blocked"
            ),
            "verdict_scope": (
                "INSTANCE: one real #205 trajectory, advisory forecast routing; no live "
                "schedule actuation and no cross-trajectory distribution claim"
            ),
        },
        residual=2.0 / 7.0,
        source_artifact=RECEIPT,
        measurement_method=(
            "read-only real-#205 walk-forward re-derivation plus deterministic fp32 gate "
            "certificate and replay-ledger contract tests"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Costate-organ deterministic router margin, replay, and IS law",
        one_line_summary=(
            "Certify and replay fp32 routes, calibrate sequential reliability, allocate "
            "shadow compute from uncertainty, and IS-reweight only with density custody."
        ),
        latex_form=(
            r"m_t=\operatorname{f32}(s_t)-\operatorname{f32}"
            r"(\operatorname{median}_{\operatorname{f32}}s_{\le t});\;"
            r"g_t=\begin{cases}\mathrm{plateau}&m_t<0\\"
            r"\mathrm{transient}&m_t\ge0\end{cases};\;"
            r"a_{\mathrm{apply}}(d_t):=a_{\mathrm{decide}}[d_t];\;"
            r"q\mid y_{1:t}\sim\mathrm{Beta}(1+\sum_{i\le t}y_i,"
            r"1+t-\sum_{i\le t}y_i);\;"
            r"w_i=\frac{n_M M_i\,\mathrm{clip}_{[\ell,u]}"
            r"(p_{\mathrm{live}}(g_i)/p_{\mathrm{bt}}(g_i))}"
            r"{\sum_j M_j\,\mathrm{clip}_{[\ell,u]}"
            r"(p_{\mathrm{live}}(g_j)/p_{\mathrm{bt}}(g_j))};\;"
            r"\widehat L_a=\frac{\sum_i w_i\ell_i(a)}{\sum_iw_i}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.costate_router_stability_20260714:"
            "self_normalized_ratio_weights"
        ),
        domain_of_validity={
            "included": [
                "advisory costate-organ regime dispatch",
                "deterministic NumPy-fp32 scalar gates",
                "content-addressed decide/apply replay",
                "explicitly custodied regime-density ratios",
                "sequential Beta-Bernoulli route-match calibration",
                "advisory K>1 same-checkpoint shadow allocation",
            ],
            "excluded": [
                "score authority",
                "live/heavy actuation without operator GO",
                "uniform substitution for missing density ratios",
                "LLM token-level thresholds transferred as costate constants",
                "cross-trajectory generalization from one run",
                "interpreting a route-match posterior as a full physical-trajectory posterior",
            ],
            "tie_law": "m_t == 0 selects transient; surprise ratio == threshold does not defer",
            "research_only": True,
        },
        units_in={
            "s": "class-weighted d_seg slope magnitude per epoch",
            "density": "probability mass per named regime",
            "loss": "walk-forward absolute d_seg error",
            "y": "backtest-only indicator that the selected route matched the fold oracle",
        },
        units_out={
            "m": "class-weighted d_seg slope magnitude per epoch",
            "w": "dimensionless self-normalized importance weight",
            "L": "walk-forward absolute d_seg error",
            "q": "posterior route-match probability on the measured path",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "all_folds_positive_margin_prediction_error_fraction": 2.0 / 7.0,
            "is_identification_missing_fraction": 1.0,
            "margin_calibration_inversion_magnitude": 0.4,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.router_stability",
            "tac.witness_control.regime_dispatch",
            "tac.witness_dsl.costate_agent_dsl",
        ),
        canonical_producers=("tools.lambda_net_backtest",),
        provenance=provenance,
    )


def populate_costate_router_stability_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicitly register at the caller-supplied path; never mutate on import."""
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_costate_router_stability_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="costate-router-stability; research-only; live-density custody blocked",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_UTC",
    "RECEIPT",
    "build_costate_router_stability_v1",
    "populate_costate_router_stability_v1",
    "self_normalized_ratio_weights",
    "sequential_beta_route_match_posterior",
    "signed_selection_margin",
]
