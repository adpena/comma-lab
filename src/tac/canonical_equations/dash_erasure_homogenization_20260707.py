# SPDX-License-Identifier: MIT
"""Canonical equation: DASH-ERASURE HOMOGENIZATION law (viscosity-theory alignment hunt
2026-07-07, `.omx/research/viscosity_theory_alignment_hunt_20260707.md` §3 + spec §9.1).

THE LAW (anchored half — registered here). Lane dashes are a δ-periodic along-tangent
microstructure (period δ_along). When the minimum smoothing scale of the training flow

    s_min = min(τ·|∇logit|⁻¹-scale, viscous cutoff ε, R-Nyquist)   satisfies   s_min ≳ δ_along,

Γ-convergence (Lions–Papanicolaou–Varadhan cell problem; Braides / Ansini–Braides–Chiadò Piat
anisotropic-perimeter homogenization) forces the flow onto the HOMOGENIZED solid anisotropic
band, and the pinning results (Dirr–Yip; arXiv 2108.00558) give ZERO effective mobility of the
lane interface. Consequence: sub-δ dash structure is NOT recoverable by the coarse chart at ANY
capacity or epoch budget below the crossover. Operational coupling rule (τ_end second meaning):
do not anneal τ below the dash period unless a corrector-class lever is active.

FIVE HELD MEASURED ANCHORS (all pre-existing rows; this law is the theory that fits them):
  1. dash-gap FP = 0.00396 = 90% of band reconstruction; dash PHASE = ego-distance (screw ξ)
     — #215 (`project_dashgap_fp_deepdive_range_dependent_ego_phase_fp_as_signal_20260701`).
  2. erasure ∝ 1/persistence; MBO smoothing attributes 95.7% of cost to Lane
     — #284 §2.8 (sister equation `mcf_minority_erasure_inevitability_v1`, which this REFINES:
     the MCF-erasure law is the τ→0 face of the same homogenization statement).
  3. lane cls-1 STUCK at the annulus (zero effective mobility measured live)
     — #333 (`costate_loop_closed_annulus_sense_curriculum_event_act_20260706`).
  4. capacity-alone NO-GO (bigcap overfits, dashes stay erased) — two-scale expansion says the
     coarse chart CANNOT hold the corrector; matches the measured 3.2× along-tangent deficit
     (`lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703`).
  5. dash phase transported by ego-motion (the corrector's phase variable is already measured).

PREDICTION half (NOT registered as fact; owed anchors per hunt spec §9.1): the max-plus dash
comb `δ·v(x/δ)` with phase from ξ (#287) is the unique repair class. Owed before any
`_corrector_v1` upgrade: (i) corrector A/B, witness+comb vs witness alone, n600 through R with
byte accounting; (ii) the $0 τ-crossover probe (dash-gap FP vs τ/δ_along on a fixed checkpoint).

means != ends: theory alignment on advisory-axis anchors; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "dash_erasure_homogenization_v1"

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-CPU/MLX advisory]"
_PREDICTED = "[predicted]"
_HUNT_MEMO = ".omx/research/viscosity_theory_alignment_hunt_20260707.md"

# --- HELD measured constants (provenance in the module docstring; sources pre-existing) -------
DASH_GAP_FP_N600 = 0.00396          # 90% of the analytic band's reconstruction value (#215)
MBO_LANE_EROSION_COST_SHARE = 0.957  # MBO smoothing-cost share attributed to Lane (#284 §2.8)
ALONG_TANGENT_FREQ_DEFICIT = 3.2     # along 8 vs across 32 effective (2026-07-03 4-lens probe)


def smoothing_crossover_ok(
    tau_scale_px: float,
    eps_viscous_px: float,
    r_nyquist_px: float,
    dash_period_px: float,
    margin: float = 1.0,
) -> bool:
    """The τ_end coupling rule as code: True iff the coarse flow is BELOW the homogenization
    crossover (min smoothing scale < margin·δ_along), i.e. sub-δ dash structure is still
    resolvable without a corrector. False ⇒ do not anneal further unless the comb is active."""
    if dash_period_px <= 0:
        raise ValueError(f"dash_period_px must be > 0, got {dash_period_px}")
    s_min = min(tau_scale_px, eps_viscous_px, r_nyquist_px)
    return s_min < margin * dash_period_px


def build_dash_erasure_homogenization_v1() -> CanonicalEquation:
    """Build the dash-erasure homogenization law with the five HELD measured anchors."""

    anchor = EmpiricalAnchor(
        anchor_id="dash_erasure_homogenization_five_held_anchors_20260707",
        measurement_utc=_UTC,
        inputs={
            "microstructure": "lane dashes, δ-periodic along-tangent (period = δ_along px)",
            "flow": ("gradient descent on τ-smoothed level-set energy over coordinate-INR, "
                     "rendered through R (bicubic↑ → uint8 → bilinear↓) into frozen SegNet argmax"),
            "smoothing_scales": "min(τ-scale, viscous ε, R-Nyquist) vs δ_along",
        },
        predicted_output={
            "prediction_1": "dash-gap FP ≈ band reconstruction value (flow sits on homogenized band)",
            "prediction_2": "erasure ordered by persistence; Lane dominates smoothing cost",
            "prediction_3": "lane interface pinned (zero effective mobility) once homogenized",
            "prediction_4": "capacity/epochs alone cannot recover dashes below the crossover",
        },
        empirical_output={
            "dash_gap_fp_n600": DASH_GAP_FP_N600,
            "dash_gap_fp_share_of_band_recon": 0.90,
            "mbo_lane_erosion_cost_share": MBO_LANE_EROSION_COST_SHARE,
            "lane_cls1_stuck": "measured live (#333 annulus telemetry; flip 38.7% share)",
            "capacity_no_go": "bigcap overfit NO-GO; 3.2x along-tangent deficit persists",
            "dash_phase_is_ego_distance": "measured (#215) — the corrector phase variable exists",
        },
        residual=0.0,
        source_artifact=_HUNT_MEMO,
        measurement_method=("theory-alignment over five PRE-EXISTING measured rows (no new "
                            "measurement in this registration); each row's own provenance is "
                            "cited in the module docstring"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_HUNT_MEMO,
            reactivation_criteria=("upgrade to dash_erasure_homogenization_corrector_v1 ONLY "
                                   "after (i) corrector A/B witness+comb vs witness alone at "
                                   "n600 through R with byte accounting, and (ii) the $0 "
                                   "τ-crossover probe (dash-gap FP vs τ/δ_along, fixed ckpt)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Dash-erasure homogenization: min-smoothing ≳ δ_along ⇒ Γ-converged solid band + "
              "pinned lane interface; sub-δ structure unrecoverable by coarse chart at any capacity"),
        one_line_summary=(
            "Below the homogenization crossover the flow provably erases dash microstructure "
            "and pins the lane interface; repair needs a corrector lever, never more capacity."
        ),
        latex_form=(
            r"\min(\tau,\varepsilon,\Delta_R)\gtrsim\delta_{\parallel}\;\Rightarrow\;"
            r"u_\theta\to\bar u\ (\Gamma\text{-hom. band}),\ \mu_{\text{lane}}=0;\quad"
            r"u\approx\bar u+\delta\,v(x/\delta)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.dash_erasure_homogenization_20260707:smoothing_crossover_ok"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "registered_half": ("the ERASURE/PINNING/CROSSOVER law only — anchored by the five "
                                "held measured rows"),
            "prediction_half_NOT_registered": ("corrector EFFICACY (#287 max-plus comb as the "
                                               "unique repair) is a PREDICTION until the owed "
                                               "A/B + τ-crossover probe measure"),
            "operational_rule": ("τ_end coupling: do not anneal τ below the dash period unless "
                                 "a corrector lever is active (τ_end has a second, "
                                 "homogenization meaning beyond the pixel-pitch floor)"),
            "measurement_axis": ["macOS-CPU/MLX advisory"],
        },
        units_in={"tau_scale_px": "px", "eps_viscous_px": "px", "r_nyquist_px": "px",
                  "dash_period_px": "px", "margin": "dimensionless"},
        units_out={"crossover_ok": "bool (True = dashes still resolvable without corrector)"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # the law's headline check: dash-gap FP sits AT the band value (share 0.90 vs 1.0
            # ideal homogenized limit; the 0.10 gap = residual partial dash rendering):
            "fp_share_of_band_recon_vs_homogenized_limit": abs(0.90 - 1.0),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",   # τ_end coupling rule + future comb Lever
        ),
        canonical_producers=(
            ".omx/research/viscosity_theory_alignment_hunt_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="dash_erasure_homogenization.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_dash_erasure_homogenization_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the dash-erasure homogenization law (equations
    leg of the 2026-07-07 viscosity hunt, commit 491c27e5b + drift-detector follow-through).
    Registers ONLY the anchored erasure/pinning/crossover half; the corrector-efficacy half
    stays a prediction per hunt spec §9.1 (owed: corrector A/B + τ-crossover probe)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_dash_erasure_homogenization_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="viscosity_hunt_20260707 equations leg (FEED-07s/07t); corrector_v1 upgrade owed "
              "on the two named probes",
    )
    return eq


__all__ = [
    "ALONG_TANGENT_FREQ_DEFICIT",
    "DASH_GAP_FP_N600",
    "EQUATION_ID",
    "MBO_LANE_EROSION_COST_SHARE",
    "build_dash_erasure_homogenization_v1",
    "populate_dash_erasure_homogenization_equation",
    "smoothing_crossover_ok",
]
