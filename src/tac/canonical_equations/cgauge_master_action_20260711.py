# SPDX-License-Identifier: MIT
"""The V9·CGauge MASTER ACTION — the canonical system of equations (2026-07-11).

Memo: ``.omx/research/cgauge_master_action_and_parametrization_20260711.md`` (the
Einstein/Fable capstone pass). This module registers ONE top-level SYSTEM equation,
``cgauge_master_action_v1``, that does three things:

1. States the master action S and its IRREDUCIBLE AXIOMS (A1–A6) — everything else in
   the registry that touches the witness is a CORROLLARY, and the derivation tree here
   encodes each parent→child edge with its OPERATION (vary θ / boundary-integrate /
   Γ-limit / read-in-Fisher-metric / Legendre-transform / Morse-topology) and its honest
   LABEL (DERIVED_EXACT / DERIVED / MEASURED_INPUT / ASSUMED / FALSIFIED_MECHANISM).
   Child anchors are NOT duplicated — edges LINK to the children's equation_ids.

2. States the closed-loop OPTIMAL-CONTROL form (the crown): S is a Lagrangian; its
   Legendre transform gives the campaign Hamiltonian H; the COSTATE λ = ∂S/∂state is the
   marginal-ΔS shadow price (the already-registered ``costate_lambda_marginal_ds_v1``,
   #247) — Pontryagin selects the control. Forward dynamics = training (Euler–Lagrange
   of S); adjoint dynamics = the costate controller (sense→decide); the triality's three
   shadows (DAG=state, DSL=control, equations=law) close into a loop through λ.

3. Declares the VALUE-FORM vocabulary: a config value's natural form under S is not
   always a scalar — it may be an ODE/DE (flowing), a PDE/field, self-deriving
   (fixed-point/costate readout), linear, polynomial, or a polytope/KKT vertex.

The master action (schematic; full derivation in the memo)::

    S[θ, z, δξ] = 100·D_seg[argmax F_seg(R·W_θ(z))] + sqrt(10·D_pose[F_pose(R·W_θ)])
                  + (25/N)·L_MDL[counted statistic]
    subject to (A3): z_p = Hol_ξ(z̄) ⊕ gauge-phase_p ⊕ events_p

with distortion read in the frozen-scorer pullback Fisher metric (A2), the smooth
τ-relaxation family S_τ (A6) whose Γ-limit is S, and the measurement operator R fixing
the lattice gauge.

Pointer 0.19108282 UNMOVED — a system of equations is MEANS; only a byte-closed n600
exact row moves it.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-11T05:00:00Z"
_MEMO = ".omx/research/cgauge_master_action_and_parametrization_20260711.md"
_PROBE = ".omx/tmp/cgauge_probes/results.json"
_HARVEST = ".omx/research/n205_live_telemetry_harvest_for_v9cgauge_20260711.md"

MASTER_ACTION_EQUATION_ID = "cgauge_master_action_v1"

# ---- edge labels (NO-FAKE: every edge carries its honest epistemic status) -----------
DERIVED_EXACT = "DERIVED_EXACT"        # pure arithmetic/theorem from the axiom
DERIVED = "DERIVED"                    # derivation with stated technique-level assumptions
MEASURED_INPUT = "MEASURED_INPUT"      # an empirical input S consumes, NOT derivable from it
ASSUMED = "ASSUMED"                    # a postulate (audited where possible)
FALSIFIED_MECHANISM = "FALSIFIED_MECHANISM"  # derivation stands, its cure-mechanism measured-dead
EDGE_LABELS = (DERIVED_EXACT, DERIVED, MEASURED_INPUT, ASSUMED, FALSIFIED_MECHANISM)

# ---- value forms (the "values are equations" layer) ----------------------------------
FORM_SCALAR = "SCALAR"                 # S truly pins a number
FORM_ODE = "ODE"                       # value flows by a DE in epoch/time
FORM_PDE = "PDE_FIELD"                 # value is a spatial/temporal field solving a field eq
FORM_SELF_DERIVING = "SELF_DERIVING"   # value = solution of its own optimality condition (costate)
FORM_LINEAR = "LINEAR"                 # value = linear map of a measured driver
FORM_POLYNOMIAL = "POLYNOMIAL"         # value = polynomial schedule/function
FORM_POLYTOPE = "POLYTOPE_KKT"         # value = feasible-set vertex / KKT / exact solve
VALUE_FORMS = (
    FORM_SCALAR, FORM_ODE, FORM_PDE, FORM_SELF_DERIVING,
    FORM_LINEAR, FORM_POLYNOMIAL, FORM_POLYTOPE,
)


def axioms() -> tuple[dict, ...]:
    """The irreducible postulates of the CGauge system (everything else is an edge)."""
    return (
        {"id": "A1", "name": "score law (frozen apparatus)", "label": MEASURED_INPUT,
         "statement": "S = 100 d_seg + sqrt(10 d_pose) + 25 B/N on the exact grids/pairs "
                      "(evaluate.py:96, modules.py:113, 600 pairs, N=37545489)"},
        {"id": "A2", "name": "Fisher base metric", "label": ASSUMED,
         "statement": "distortion geometry = frozen-scorer pullback Fisher metric; margin "
                      "field is its measured surrogate (Pearson 0.978)",
         "audit": "calibrated on GT frames; validity AT the witness's operating point is "
                  "an open blind-spot (UU-1)"},
        {"id": "A3", "name": "general covariance", "label": ASSUMED,
         "statement": "pair-dependence factors through (xi-holonomy, R-lattice-phase) "
                      "plus scene events; anything else is wasted rate",
         "audit": "audited CONFIRMED-as-stated (bracket [0.42,0.93], residual 96.4% "
                  "allowed buckets, 0 texture) — witness_general_covariance_totality_v1"},
        {"id": "A4", "name": "scene constants (clip-specific)", "label": MEASURED_INPUT,
         "statement": "rank-8 separatrix manifold (SDF chart); spike rate 0.005318; dash "
                      "comb ~25 cyc/unit; class geometry/anisotropy — measured, "
                      "non-transferable"},
        {"id": "A5", "name": "MDL rate / rule-118 boundary", "label": MEASURED_INPUT,
         "statement": "rate = description length of the COUNTED statistic; the generic "
                      "generator is free; learned/video-derived payload is counted"},
        {"id": "A6", "name": "smooth relaxation family", "label": ASSUMED,
         "statement": "the tau-family S_tau (softmax temperature) with S = Gamma-lim S_tau; "
                      "tau = interface width eps = dequantization scale hbar (the "
                      "lambda-extension of this identification is measured-REJECTED)"},
    )


def derivation_edges() -> tuple[dict, ...]:
    """The derivation tree: every registered CGauge-relevant law as a child of the axioms.

    Each edge: child equation_id (or NEW: prefix for laws registered this pass), parents,
    the operation that yields it, and the honest label. Child anchors live on the child
    equations — this system LINKS, never duplicates.
    """
    return (
        # --- the crown: Hamiltonian / costate ------------------------------------------
        {"child": "costate_lambda_marginal_ds_v1", "parents": ("S",),
         "operation": "Legendre transform of S -> H; costate lambda = dS/d(state) "
                      "(Pontryagin adjoint; the #247 controller is its realization)",
         "label": DERIVED,
         "note": "honest limit: S's d_seg term is piecewise-constant in theta (argmax), so "
                 "the Legendre/adjoint structure is taken on the S_tau relaxation (A6) and "
                 "on the campaign state (config/lever space), not the raw argmax"},
        # --- exact structure of A1 ------------------------------------------------------
        {"child": "score_atomic_flip_byte_exchange_v1", "parents": ("A1",),
         "operation": "partial differentiation of A1 in its atomic units",
         "label": DERIVED_EXACT},
        {"child": "dseg_covariant_gauge_decomposition_v1", "parents": ("A1", "A3"),
         "operation": "change of variables to the xi-comoving chart",
         "label": DERIVED},
        {"child": "gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1",
         "parents": ("A1", "A3", "A4"),
         "operation": "variational restriction to label-smooth witnesses (minimize the "
                      "gauge term); constant 0.005318 is A4",
         "label": DERIVED},
        {"child": "witness_general_covariance_totality_v1", "parents": ("A3",),
         "operation": "n600 audit OF the axiom (regress code on (xi,phase); residual "
                      "bucket census)", "label": MEASURED_INPUT},
        # --- Gamma-limit branch (A6) ----------------------------------------------------
        {"child": "multiphase_modica_mortola_perimeter_gamma_limit_v1", "parents": ("A6",),
         "operation": "Gamma-limit tau->0 of the relaxed perimeter energy",
         "label": DERIVED},
        {"child": "junction_young_angle_sigma_fit_v1",
         "parents": ("multiphase_modica_mortola_perimeter_gamma_limit_v1",),
         "operation": "first variation at triple points (Young's law); sigma_cc' fit is "
                      "measured", "label": DERIVED},
        {"child": "mcf_minority_erasure_inevitability_v1",
         "parents": ("multiphase_modica_mortola_perimeter_gamma_limit_v1",),
         "operation": "gradient flow of the perimeter term (curve shortening)",
         "label": DERIVED},
        {"child": "island_topological_charge_conservation_v1", "parents": ("A6",),
         "operation": "Morse theory on the margin field (topology of level sets)",
         "label": DERIVED},
        {"child": "chan_vese_area_constraint_birth_balance_v1",
         "parents": ("island_topological_charge_conservation_v1",),
         "operation": "the conservation law FORCES an explicit source term for births",
         "label": DERIVED},
        {"child": "tau_eps_hbar_one_dequantization_two_scales_v1", "parents": ("A6",),
         "operation": "Maslov dequantization reading of the tau-family",
         "label": DERIVED,
         "note": "the lambda=tau=eps=hbar EXTENSION is measured-REJECTED (L87)"},
        {"child": "adaptive_eps_cfl_edge_tracking_v1", "parents": ("A6",),
         "operation": "Euler-Lagrange of the eikonal constraint + explicit-step CFL",
         "label": FALSIFIED_MECHANISM,
         "note": "the DE derivation stands as discretization stability analysis; the "
                 "CFL-edge CURE mechanism for the eikonal re-entry is FALSIFIED at n600 "
                 "(FEED-06g) — measurement over derivation"},
        # --- Fisher branch (A2) ---------------------------------------------------------
        {"child": "ce_softmax_mirror_descent_natural_gradient_v1", "parents": ("A2",),
         "operation": "CE descent read as mirror descent / natural gradient in the "
                      "categorical Fisher metric", "label": DERIVED},
        {"child": "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
         "parents": ("A2",),
         "operation": "computing the pullback Fisher trace on the categorical output "
                      "(margin correlation 0.978 is the calibration)",
         "label": MEASURED_INPUT},
        {"child": "posenet_luma_chroma_sensitivity_asymmetry_v1", "parents": ("A1", "A2"),
         "operation": "reading the pose half of the metric through the frame_utils YUV6 "
                      "box-average (chroma-HF null is exact from code; luma 11.1x and the "
                      "+2 partition-invisible DOF are measured)", "label": DERIVED},
        {"child": "store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
         "parents": ("A3", "A5"),
         "operation": "gauge-orbit directions have zero covariant cost => the storable "
                      "statistic is the zero-mode's integration constants (carriers)",
         "label": DERIVED},
        # --- rate branch (A5 + A4) ------------------------------------------------------
        {"child": "shearlet_nterm_upper_bounds_task_rate_v1", "parents": ("A4", "A5"),
         "operation": "N-term approximation of the C2 codim-1 separatrix (Candès–Donoho)",
         "label": DERIVED},
        {"child": "argmax_of_sdf_is_additively_weighted_power_diagram_v1",
         "parents": ("A6",),
         "operation": "structure of the level-set representation (argmax of SDF fields = "
                      "Laguerre/power diagram)", "label": DERIVED},
        # --- the #223 parametrization optima (NEW this pass) ----------------------------
        {"child": "cgauge_whitney_moddim_v1", "parents": ("A4",),
         "operation": "Whitney/prevalence embedding of the rank-8 manifold",
         "label": DERIVED},
        {"child": "cgauge_nyquist_bank_frequency_v1", "parents": ("A1",),
         "operation": "sampling limit of the measured R MTF on the scoring grid",
         "label": DERIVED},
        {"child": "cgauge_curvelet_parabolic_bank_v1",
         "parents": ("shearlet_nterm_upper_bounds_task_rate_v1", "A4"),
         "operation": "parabolic-scaling wedge geometry evaluated at the live bank; "
                      "locates the deficit as a C2-hypothesis failure on the dash comb",
         "label": DERIVED},
        {"child": "cgauge_beta2_window_v1", "parents": ("A4", "descent-of-S_tau"),
         "operation": "gradient-noise stationarity sandwich on the discretized descent",
         "label": DERIVED},
        # --- empirical corollaries the tree consumes but cannot derive ------------------
        {"child": "curvelet_directional_basis_dseg_reduction_v1", "parents": ("A4",),
         "operation": "measured -48% proxy (basis-match prior to capacity)",
         "label": MEASURED_INPUT},
        {"child": "dash_erasure_homogenization_v1",
         "parents": ("mcf_minority_erasure_inevitability_v1",
                     "cgauge_curvelet_parabolic_bank_v1"),
         "operation": "MCF erasure + along-tangent bandwidth deficit compose on the dash "
                      "tail", "label": DERIVED},
        {"child": "anisotropic_basis_along_tangent_frequency_deficit_v1",
         "parents": ("A4",), "operation": "measured 25-vs-8 cyc/unit deficit",
         "label": MEASURED_INPUT},
        {"child": "contest_r_operator_mtf_allpass_to_2px_v1", "parents": ("A1",),
         "operation": "measured MTF of the R chain", "label": MEASURED_INPUT},
        # --- optimizer empirics: honestly NOT derivable from S --------------------------
        {"child": "muon_finisher_schedule_warmstart_and_lr_anneal_v1", "parents": (),
         "operation": "none — optimizer choice empirics (-32% d_seg) are apparatus "
                      "physics OUTSIDE S; the action does not derive Muon",
         "label": MEASURED_INPUT},
    )


def closed_loop_roles() -> dict[str, str]:
    """The V9·CGauge system as ONE closed loop (the costate at the nexus)."""
    return {
        "forward": "training = Euler–Lagrange descent of S_tau (the witness evolves)",
        "adjoint": "costate lambda = marginal ΔS per DOF/lever (costate_lambda_marginal_"
                   "ds_v1, #247) — Pontryagin selects the control",
        "sense": "live-run telemetry harvest (binding-vs-inert) estimates measured-lambda; "
                 "gap (derived-lambda − measured-binding) = the innovation signal",
        "act": "the DSL lever set = the costate's action space; knob-verdict table = its "
               "prior; duty-to-measure queue = its exploration; OFF = a tracked queue",
        "law": "this registry (the equations leg) — updated by measured rows",
        "containment": "advisory-only autonomous; heavy/paid/stop/live-config actuation = "
                       "operator-GO (the derivation does NOT activate the loop)",
    }


def edge_label_of(child_equation_id: str) -> str | None:
    """Look up the epistemic label of a derivation edge by child id (None if absent)."""
    for e in derivation_edges():
        if e["child"] == child_equation_id:
            return e["label"]
    return None


def axiom_ids() -> tuple[str, ...]:
    return tuple(a["id"] for a in axioms())


def _prov():
    return build_provenance_for_research_sidecar(
        _MEMO,
        reactivation_criteria=(
            "re-issue the system when an axiom is refuted/extended (A2 operating-point "
            "audit UU-1; A3 exact-homography basis; A4 re-measure on a new clip)"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc=_UTC,
    )


def build_cgauge_master_action_v1() -> CanonicalEquation:
    prov = _prov()
    edges = derivation_edges()
    n_by_label: dict[str, int] = {}
    for e in edges:
        n_by_label[e["label"]] = n_by_label.get(e["label"], 0) + 1
    anchor = EmpiricalAnchor(
        anchor_id="n600_cgauge_system_consistency_and_chart_probe_20260711",
        measurement_utc=_UTC,
        inputs={
            "axioms": [a["id"] for a in axioms()],
            "n_edges": len(edges),
            "edges_by_label": n_by_label,
            "probe": _PROBE,
            "harvest": _HARVEST,
        },
        predicted_output={
            "consistency": "covariance law + Einstein-pass laws + pose mirror mutually "
                           "consistent (memo §1 audit); the term S DEMANDS that the "
                           "lever set under-supplies = along-tangent/event/phase "
                           "carriers (memo knob table)",
        },
        empirical_output={
            # the UU-2 chart probe (MEASURED 2026-07-11, $0 cached n600)
            "indicator_basis_cumvar_at_8": 0.5050,
            "indicator_basis_rank_for_sdf_0.9316": 293,
            "sdf_chart_is_linearizing": True,
            # UU-5 cross-pair predictability (rate headroom on per-pair channels)
            "code_ar1_bits_saved_top8_mean": 2.3236,
            "pose_ar1_bits_saved_mean": 0.8743,
            # UU-3 photometric gauge: measured NEGLIGIBLE at n600 aggregate
            "photometric_gauge_partial_r2_beyond_xi": 0.0094,
            # P299 GT-side census
            "lane_along_tangent_frac_above_nu8_median": 0.3014,
            # live-harvest joins (advisory)
            "live_eff_rank_at_ep125": 16.36,
            "live_xi_cca_mean_max": [0.55, 0.76],
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "$0 cached-n600 probe suite (.omx/tmp/cgauge_probes/probe.py: indicator-basis "
            "Gram spectrum, AR(1) code predictability, photometric-gauge partial R2, "
            "Lane along-tangent census) + the read-only live-run harvest join; no scorer "
            "forward; live run untouched"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=MASTER_ACTION_EQUATION_ID,
        name=(
            "V9·CGauge master action: one covariant Lagrangian (6 axioms) whose "
            "Legendre/costate closure + derivation tree yields every registered witness law"
        ),
        one_line_summary=(
            "S = 100 D_seg + sqrt(10 D_pose) + (25/N) L_MDL on the Fisher base, covariant "
            "under (xi,R); laws = tree edges; costate lambda = dS/dstate closes the loop."
        ),
        latex_form=(
            r"S[\theta,z,\delta\xi]=100\,D_{seg}+\sqrt{10\,D_{pose}}+\tfrac{25}{N}L_{MDL},\ "
            r"z_p=\mathrm{Hol}_\xi(\bar z)\oplus\varphi_p\oplus e_p;\ "
            r"H=\mathrm{Leg}(S_\tau),\ \lambda=\partial S/\partial x"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.cgauge_master_action_20260711:derivation_edges"
        ),
        domain_of_validity={
            "covariance_class": {
                "law": "COVARIANT_LAW",
                "A1_A4_A5_constants": "SCORER_FRAME_STRUCTURAL/GAUGE_MEASUREMENT",
            },
            "derivation_status": (
                "SYSTEM equation: links children, duplicates no anchors; every edge "
                "labeled; axioms A2/A3/A6 are ASSUMED (audited), A1/A4/A5 are "
                "MEASURED_INPUT; optimizer empirics honestly NOT derived from S"
            ),
            "honest_limits": [
                "d_seg is argmax-piecewise-constant: Legendre/adjoint taken on S_tau + "
                "campaign state, not the raw score",
                "A4 constants are clip-specific (never transfer)",
                "unknown-unknowns cannot be exhaustively enumerated (memo blind-spot map)",
            ],
            "value_forms": list(VALUE_FORMS),
            "edge_labels": list(EDGE_LABELS),
            "sisters": [
                "costate_lambda_marginal_ds_v1 (the adjoint/crown)",
                "witness_general_covariance_totality_v1 (the A3 audit)",
            ],
        },
        units_in={"child_equation_id": "equation_id"},
        units_out={"derivation_edges": "list[edge]"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"system_consistency_violations": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.shadow_controller",
            "tac.witness_autoconfig",
            "tac.witness_dsl.lawref",
        ),
        canonical_producers=(_MEMO,),
        provenance=prov,
    )


def populate_cgauge_master_action(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the master system equation."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_cgauge_master_action_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes=(
            "cgauge_master_action_20260711: the top-level CGauge system (Einstein/Fable "
            "capstone pass; memo "
            ".omx/research/cgauge_master_action_and_parametrization_20260711.md)"
        ),
    )
    return eq


__all__ = [
    "ASSUMED",
    "DERIVED",
    "DERIVED_EXACT",
    "EDGE_LABELS",
    "FALSIFIED_MECHANISM",
    "FORM_LINEAR",
    "FORM_ODE",
    "FORM_PDE",
    "FORM_POLYNOMIAL",
    "FORM_POLYTOPE",
    "FORM_SCALAR",
    "FORM_SELF_DERIVING",
    "MASTER_ACTION_EQUATION_ID",
    "MEASURED_INPUT",
    "VALUE_FORMS",
    "axiom_ids",
    "axioms",
    "build_cgauge_master_action_v1",
    "closed_loop_roles",
    "derivation_edges",
    "edge_label_of",
    "populate_cgauge_master_action",
]
