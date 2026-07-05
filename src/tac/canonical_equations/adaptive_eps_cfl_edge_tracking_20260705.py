# SPDX-License-Identifier: MIT
"""Canonical equation: ADAPTIVE-eps CFL-edge tracking (the DERIVED eikonal re-entry mechanism cure).

V6 #320 (DAG FEED-06c; DSL EikonalViscoStabGauge.ADAPTIVE_EPS; memo
adaptive_eps_mechanism_cure_20260705). Formalizes the equations leg so it AGREES with the DE
derivation (#318), the symposium (#317 §7.4), the DSL gauge chart, and the trainer build.

THE LAW (DE #318 §3-4). The eikonal penalty flow `(|grad m|-1)^2` on the decision margin m is a
provably ill-posed (anti-diffusive) PDE; the viscous (ViscoReg) regularization is discretely stable
iff the CFL group `pi_eik = eta*lambda_eik*|c_a|^2/(8*eps^2) <= 1`, i.e. eps above the LOWER edge
`eps_lower = |c_a|*sqrt(eta*lambda_eik/8)`, `c_a = (|grad m|-1)/|grad m|`. The v5 ep110 re-entry is a
TWO-SIDED squeeze: eps ANNEALS DOWN toward that edge while progressive sharpening GROWS |c_a(t)|
(raising the edge). The cure TRACKS the rising edge:

    eps(t) = clamp( |c_a(t)| * sqrt(eta * lambda_eik / 8) * (1 + margin), eps_floor, eps_upper )

means != ends: this is a 0-archive-byte train-time viscosity schedule; its NET n600 d_seg cure is
owed to main's bounded probe (the constant `8` and the exact |c_a| aggregation are MEASUREMENT-OWED /
FORMALIZATION_PENDING). Pointer contest-CPU 0.19110 UNMOVED. What IS measured + anchored here: the
law's BYTE-IDENTITY at OFF (the flag-absent path is unchanged) and the numpy<->MLX bit-parity of the
|c_a| proxy + the eps law (gt_n6 smoke + the 21-test suite).
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

ADAPTIVE_EPS_EQUATION_ID = "adaptive_eps_cfl_edge_tracking_v1"

_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"

_MEMO = ".omx/research/adaptive_eps_mechanism_cure_20260705.md"
_DE = ".omx/research/witness_config_differential_equations_derivation_20260705.md"
_REF = "src/tac/boundary_math/eikonal_sharpness_proxy_reference.py"

# MEASURED grounding (gt_n6 CPU-authority-class smoke + the 21-test suite; advisory / byte-identity).
# At ep0 both the anneal and adaptive paths use eps=0.3 => the ep1 loss_terms are BIT-IDENTICAL
# (total 363.537496, seg 319.979141, gnorm 5697.5688 in BOTH) => the |c_a| no-grad measurement adds
# ZERO perturbation. |c_a| measured from the real witness margin: 11.16 -> 10.08 over ep1-4.
BYTE_IDENTICAL_OFF = True
CA_MEASURED_GT_N6_EP1 = 11.159448          # real measured |c_a| (interior mean), gt_n6 ep1
NUMPY_MLX_PARITY_REL = 1e-4                 # |c_a| + eps-law numpy<->MLX agreement (test suite)


def build_adaptive_eps_cfl_edge_tracking_v1() -> CanonicalEquation:
    """Build the adaptive-eps CFL-edge-tracking canonical equation. Registered ONLY with the
    byte-identity + numpy-parity anchor (VERIFIED); the n600 cure-confirmation anchor is owed to
    main's bounded probe (ASSUMED_AWAITING_VERIFICATION; the constant is FORMALIZATION_PENDING)."""

    anchor_parity = EmpiricalAnchor(
        anchor_id="adaptive_eps_byte_identity_off_plus_numpy_mlx_parity_20260705",
        measurement_utc="2026-07-05T00:00:00Z",
        inputs={"proxy": "|c_a| = mean|(|grad m|-1)/|grad m|| on the witness decision margin interior",
                "eps_law": "clamp(|c_a|*sqrt(eta*lambda_eik/8)*(1+margin), floor, upper)",
                "smoke": "gt_n6 CPU-class 4ep; OFF-vs-adaptive; ep0 both eps=0.3"},
        predicted_output={"byte_identical_off": True,
                          "numpy_mlx_parity_rel": NUMPY_MLX_PARITY_REL},
        empirical_output={"ep1_loss_bit_identical_when_eps_equal": True,
                          "ep1_total_both": 363.537496, "ep1_gnorm_both": 5697.5688,
                          "c_a_measured_gt_n6_ep1": CA_MEASURED_GT_N6_EP1,
                          "note": "the |c_a| no-grad measurement adds ZERO perturbation (ep0 same-eps "
                                  "=> ep1 loss/grad bit-identical); |c_a| is a REAL measured field "
                                  "quantity, not a constant (NO-FAKE)"},
        residual=0.0,
        source_artifact=_REF,
        measurement_method="gt_n6_cpu_class_smoke_plus_numpy_mlx_parity_test_suite",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_REF,
            reactivation_criteria="re-run the gt_n6 OFF-vs-adaptive smoke + src/tac/tests/test_adaptive_visco_eps.py",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_n600 = EmpiricalAnchor(
        anchor_id="adaptive_eps_n600_cure_confirmation_owed_to_main_probe_20260705",
        measurement_utc="2026-07-05T00:00:00Z",
        inputs={"config": "v5 argv + --eikonal-viscosity-adaptive floor 0.3 upper 0.7 + "
                          "--eikonal-weight-end 0.05, resume bd_calib ep100 (CLEAN A/B vs v5)",
                "constant": "the `8` in sqrt(eta*lambda_eik/8) + the |c_a| aggregation are "
                            "MEASUREMENT-OWED (FORMALIZATION_PENDING)"},
        predicted_output={"verdict": "no eikonal re-entry through ep150; d_seg descends past ep110"},
        empirical_output={"net_n600_dseg": "owed to main's bounded probe (advisory until byte-closed)"},
        residual=0.0,
        source_artifact=_DE,
        measurement_method="bounded_n600_probe_to_ep150_promote_abort_gate",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="main runs the bounded n600 adaptive-eps probe; confirm no ep110 "
                                  "re-entry + d_seg descent past the ep110 point",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=ADAPTIVE_EPS_EQUATION_ID,
        name="Adaptive vanishing-viscosity eps: CFL-edge tracking (eikonal re-entry cure)",
        one_line_summary=(
            "eps(t)=clamp(|c_a(t)|*sqrt(eta*lambda_eik/8)*(1+margin), floor, upper) tracks the RISING "
            "CFL lower edge |c_a| grows under sharpening; a FIXED eps falls below it => the v5 ep110 re-entry"
        ),
        latex_form=(
            r"\varepsilon(t)=\mathrm{clamp}\!\Big(|c_a(t)|\sqrt{\tfrac{\eta\,\lambda_{eik}}{8}}\,(1+\mu),"
            r"\ \varepsilon_{\text{floor}},\ \varepsilon_{\text{upper}}\Big),\quad "
            r"c_a=\tfrac{|\nabla m|-1}{|\nabla m|},\quad \pi_{eik}=\tfrac{\eta\lambda_{eik}|c_a|^2}{8\varepsilon^2}\le 1"
        ),
        python_callable_module_path=(
            "tac.boundary_math.eikonal_sharpness_proxy_reference:adaptive_visco_eps"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "stage": "viscous eikonal regularizer (pre-tau eikonal re-entry window; ep~110 at n600)",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "byte-identity OFF + numpy<->MLX |c_a|/eps parity are MEASURED; the `8` constant + "
                    "the net n600 cure are FORMALIZATION_PENDING (owed to main's bounded probe)",
        },
        units_in={"c_a": "dimensionless", "eta": "lr", "lam_eik": "eikonal_weight",
                  "margin_factor": "dimensionless", "eps_floor": "viscosity", "eps_upper": "viscosity"},
        units_out={"eps": "viscosity"},
        empirical_anchors=(anchor_parity, anchor_n600),
        predicted_vs_empirical_residual={"adaptive_visco_eps": 0.0},
        last_calibration_utc="2026-07-05T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="adaptive_eps_cfl_edge_tracking.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_adaptive_eps_cfl_edge_tracking_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the adaptive-eps equation (via
    ``register_canonical_equation``; latest-row-wins query semantics)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_adaptive_eps_cfl_edge_tracking_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="v6_320_adaptive_eps_cfl_edge_tracking_20260705",
    )
    return eq


__all__ = [
    "ADAPTIVE_EPS_EQUATION_ID",
    "build_adaptive_eps_cfl_edge_tracking_v1",
    "populate_adaptive_eps_cfl_edge_tracking_equation",
]
