# SPDX-License-Identifier: MIT
"""Canonical equation: quadratic head chart CONFIRMED + the subset-solve transfer law
(#341 basin-finisher Stage-0, DAG FEED-08d — the equations leg of the MEASURED verdict).

Two facts from ONE probe (tools/quadratic_basin_finisher_probe.py, n600, exact tau-stage loss
tau=0.3 softplus + 0.001*length verified against the live launch argv; MLX-CPU fp32 authority):

1. MORSE-LEMMA PREMISE CONFIRMED at tau-best (ep650): the exactly-affine head subset
   {out_sdf.weight/bias, out_tex.weight/bias, palette} (~791 params; FiLM gains excluded — not
   affine) is genuinely NEAR-QUADRATIC: damped Newton-CG (Levenberg lambda 0.1->0.0333->0.0111,
   16 CG iters, HVP = vjp-of-grad) accepted both LM rounds with gain ratios rho = 0.847 / 0.868,
   and the subset proxy's -3.3% predicted the deployed in-subset verdict's -3.4% essentially 1:1
   THROUGH int8 deploy + argmax. The quadratic-by-cell representation (council draft SS16) is a
   measured property of the head chart, not a hope.

2. THE SUBSET-SOLVE TRANSFER LAW (the tool-killer): a K-pair subset solve transfers to the full
   P-pair verdict as the K/P-weighted decomposition

       net_delta ~= (K/P) * delta_in_subset + (1 - K/P) * delta_held_out

   Measured at K=8, P=600: (8/600)(-3.4%) + (592/600)(+5.2%) = +5.11% ~= the MEASURED net +5.1%
   (0.0036878 vs probe baseline 0.0035103; 546/600 pairs worse). Mechanism = SUBSET OVERFIT
   (attribution decisive from both arms' per-pair jsonls: NOT proxy mismatch, NOT quantization,
   NOT basin radius). Consequences (all measured-reasoned): Stage-1 full-mask at K=8 REFUSED
   (overfit worsens with parameter count at fixed K); TerminalSolve as a post-run CPU SUBSET tool
   NO-GO; SS14 priming-via-subset-solve REFUSED; the only admissible solve is FULL-P, whose cost
   law is HVP_cost * P per CG iteration (measured 19 s/pair CPU => ~3.2 h/iter CPU, ~11 min/iter
   on the 17x GPU path) => a terminal solve, if the council wants one, is an IN-TRAINER GPU
   stage, never a post-hoc CPU tool.

Baseline provenance note: the probe baseline 0.0035103 differs from the checkpoint-logged
0.0033662 by the +4.3% self-orient reconstruction gap (itself measured; co-evolved state
reconstructs to ~0.7% in 1-2 fixed-point iterations).

means != ends: all rows [macOS-CPU advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

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

EQUATION_ID = "quadratic_head_chart_subset_solve_gap_v1"

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/basin_finisher_head_solve_probe_measured_20260707.md"
_ROW = "reports/basin_finisher_probe_20260707.json"

# --- MEASURED constants (#341 Stage-0, n600) ------------------------------------------------------
LM_GAIN_RATIO_ROUND1 = 0.847        # Levenberg-Marquardt rho (quadratic-model agreement), round 1
LM_GAIN_RATIO_ROUND2 = 0.868        # round 2
SUBSET_PROXY_DELTA = -0.033         # subset proxy loss change (fraction)
IN_SUBSET_VERDICT_DELTA = -0.034    # deployed in-subset verdict change (proxy->verdict ~1:1)
HELD_OUT_VERDICT_DELTA = +0.052     # held-out pairs verdict change (the overfit damage)
NET_N600_VERDICT_DELTA = +0.051     # measured full-n600 net (0.0036878 vs 0.0035103)
PROBE_K = 8                         # solve subset size (pairs)
PROBE_P = 600                       # full verdict size (pairs)
HVP_SECONDS_PER_PAIR_CPU = 19.0     # measured HVP cost => full-P CG iter ~3.2h CPU / ~11min GPU@17x


def subset_solve_net_transfer(delta_in: float, delta_out: float, k: int, p: int) -> float:
    """The K/P-weighted transfer law: expected full-set verdict delta from a K-pair subset solve.

    net = (k/p)*delta_in + (1 - k/p)*delta_out. At the measured point (K=8, P=600,
    in=-3.4%, out=+5.2%) this reproduces the measured net +5.1% to within rounding —
    the consistency check that pinned SUBSET OVERFIT as the mechanism. Fail-closed on
    invalid sizes."""
    if k <= 0 or p <= 0 or k > p:
        raise ValueError(f"require 0 < k <= p, got k={k!r} p={p!r}")
    w = float(k) / float(p)
    return w * float(delta_in) + (1.0 - w) * float(delta_out)


def build_quadratic_head_chart_subset_solve_gap_v1() -> CanonicalEquation:
    """Build the quadratic-head-chart + subset-solve-gap equation (#341 Stage-0 MEASURED)."""

    anchor_quadratic_confirmed = EmpiricalAnchor(
        anchor_id="head_chart_near_quadratic_lm_rho_20260707",
        measurement_utc=_UTC,
        inputs={
            "checkpoint": "mod32cap tau-best ep650 (levelset_witness_ema_BEST.npz)",
            "head_subset": "out_sdf.weight/bias + out_tex.weight/bias + palette (~791 params; "
                           "FiLM gains excluded — not affine)",
            "solver": "damped Newton-CG, Levenberg lambda 0.1->0.0333->0.0111, 16 CG iters, "
                      "HVP=vjp-of-grad, MLX-CPU fp32",
            "loss": "exact tau-stage loss (tau=0.3 softplus + 0.001*length), verified vs launch argv",
        },
        predicted_output={"morse_lemma": "near-quadratic head chart => LM gain ratio rho ~ 1"},
        empirical_output={
            "lm_rho_round1": LM_GAIN_RATIO_ROUND1,
            "lm_rho_round2": LM_GAIN_RATIO_ROUND2,
            "proxy_delta": SUBSET_PROXY_DELTA,
            "in_subset_verdict_delta": IN_SUBSET_VERDICT_DELTA,
            "verdict": "CONFIRMED — quadratic model agreement high AND proxy->deployed-verdict "
                       "transfer ~1:1 through int8 + argmax (the SS16 premise is measured)",
        },
        residual=abs(SUBSET_PROXY_DELTA - IN_SUBSET_VERDICT_DELTA),
        source_artifact=_MEMO,
        measurement_method="LM-damped Newton-CG on the exact training loss; chunked n600 verdict "
                           "through the exact R + frozen CPU-torch SegNet",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_ROW,
            reactivation_criteria="re-anchor on any future head-solve at a different stage/checkpoint",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_subset_gap = EmpiricalAnchor(
        anchor_id="subset_solve_k8_overfit_transfer_20260707",
        measurement_utc=_UTC,
        inputs={
            "k_pairs": PROBE_K, "p_pairs": PROBE_P,
            "baseline_dseg_probe": 0.0035103,
            "baseline_note": "probe baseline vs checkpoint-logged 0.0033662 = the measured +4.3% "
                             "self-orient reconstruction gap (separate finding, held)",
        },
        predicted_output={
            "transfer_law": "net = (K/P)*in + (1-K/P)*out",
            "law_prediction_at_measured_point": subset_solve_net_transfer(
                IN_SUBSET_VERDICT_DELTA, HELD_OUT_VERDICT_DELTA, PROBE_K, PROBE_P),
        },
        empirical_output={
            "solved_dseg_n600": 0.0036878,
            "net_delta": NET_N600_VERDICT_DELTA,
            "pairs_worse": "546/600",
            "mechanism": "SUBSET OVERFIT (decisive per-pair attribution; not proxy mismatch, "
                         "not quantization, not basin radius)",
            "consequences": "Stage-1@K=8 REFUSED; TerminalSolve post-run CPU subset tool NO-GO; "
                            "SS14 subset-priming REFUSED; admissible solve = FULL-P in-trainer "
                            "GPU stage (HVP 19 s/pair CPU => ~3.2 h/CG-iter CPU, ~11 min GPU@17x)",
        },
        residual=abs(
            subset_solve_net_transfer(
                IN_SUBSET_VERDICT_DELTA, HELD_OUT_VERDICT_DELTA, PROBE_K, PROBE_P)
            - NET_N600_VERDICT_DELTA),
        source_artifact=_MEMO,
        measurement_method="both arms' per-pair verdict jsonls (in-subset vs held-out split)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_ROW,
            reactivation_criteria="re-open ONLY at K -> P (full-P in-trainer GPU stage) or a "
                                  "measured-generalizing K; never re-run subset-K as-is",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Quadratic head chart CONFIRMED (LM rho ~0.85, proxy->verdict 1:1) + subset-solve "
              "transfer law net=(K/P)in+(1-K/P)out: K=8 solve overfits (+5.1% n600) => solve is "
              "admissible only FULL-P as an in-trainer GPU stage"),
        one_line_summary=(
            "Head chart is near-quadratic (Morse premise measured) but a K-pair terminal solve "
            "transfers as (K/P)in+(1-K/P)out — K=8 nets +5.1% WORSE; full-P in-trainer only."
        ),
        latex_form=(
            r"\Delta_{net}\approx \tfrac{K}{P}\Delta_{in}+\bigl(1-\tfrac{K}{P}\bigr)\Delta_{out};"
            r"\quad \rho_{LM}\in\{0.847,0.868\}\Rightarrow \text{quadratic chart}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.quadratic_head_chart_subset_solve_gap_20260707:"
            "subset_solve_net_transfer"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness (mod32cap lineage)"],
            "stage": "tau-best (ep650) head chart; other stages re-anchor before reuse",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("negative is IMPLEMENTATION-level for the subset TOOL-FORM, paradigm intact: "
                     "the quadratic premise is CONFIRMED; the reopening condition is K->P "
                     "(in-trainer GPU stage), recorded in the anchor's reactivation criteria"),
        },
        units_in={"delta_in": "fraction", "delta_out": "fraction", "k": "pairs", "p": "pairs"},
        units_out={"net_delta": "fraction"},
        empirical_anchors=(anchor_quadratic_confirmed, anchor_subset_gap),
        predicted_vs_empirical_residual={
            "transfer_law_vs_measured_net": abs(
                subset_solve_net_transfer(
                    IN_SUBSET_VERDICT_DELTA, HELD_OUT_VERDICT_DELTA, PROBE_K, PROBE_P)
                - NET_N600_VERDICT_DELTA),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # TerminalSolve validate/NO-GO context + SS14 priming
        ),
        canonical_producers=(
            "tools/quadratic_basin_finisher_probe.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="quadratic_head_chart_subset_solve_gap.v1",
            inputs_sha256="0" * 64,
        ),
    )


def populate_quadratic_head_chart_subset_solve_gap_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins query semantics). Equations leg of
    DAG FEED-08d (#341 Stage-0); DSL leg = ``TerminalSolve`` (whose post-run CPU subset form this
    law NO-GOes; the in-trainer full-P GPU stage is the reopening condition)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_quadratic_head_chart_subset_solve_gap_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="quadratic_head_chart_subset_solve_gap_20260707 (equations leg of FEED-08d / #341; "
              "quadratic premise CONFIRMED, subset tool-form NO-GO, full-P in-trainer reopening)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "HELD_OUT_VERDICT_DELTA",
    "HVP_SECONDS_PER_PAIR_CPU",
    "IN_SUBSET_VERDICT_DELTA",
    "LM_GAIN_RATIO_ROUND1",
    "LM_GAIN_RATIO_ROUND2",
    "NET_N600_VERDICT_DELTA",
    "PROBE_K",
    "PROBE_P",
    "SUBSET_PROXY_DELTA",
    "build_quadratic_head_chart_subset_solve_gap_v1",
    "populate_quadratic_head_chart_subset_solve_gap_equation",
    "subset_solve_net_transfer",
]
