# SPDX-License-Identifier: MIT
"""Canonical equation: #121 d_seg-AWARE FOURIER-FEATURE AMPLITUDE TAPER (re-validate-at-convergence).

The level-set / coord-INR witness feeds a coordinate grid through a FIXED Fourier/curvelet-directional
basis to per-pixel features ``feats[px, k]`` that the ``in_proj`` layer consumes. Every column enters
with UNIT amplitude — a FLAT taper. This law REALLOCATES that per-column amplitude taper ``w[k]`` by the
d_seg SALIENCY field (= the GT top1-top2 SegNet argmax MARGIN: small ``|margin|`` ⇒ near the codim-1
boundary annulus ⇒ where d_seg is decided):

    S(px)   = mean_pair exp(-|margin_pair(px)| / scale)          (unit-mean saliency; boundary-high)
    e_k     = mean_px feats[:,k]^2                                (overall column energy)
    sw_k    = mean_px S(px) * feats[:,k]^2                        (boundary-annulus-weighted energy)
    r_k     = sw_k / e_k                                          (>1 ⇒ column concentrates near boundary)
    w_k     = 1 + strength * (r_k / mean_k r_k - 1)              (reallocate toward boundary-aligned cols)
    feats'[:,k] = feats[:,k] * max(w_k, floor)

This is the SPECTRAL analogue of the vendored ``configurable_taper_decoder.dseg_aware_taper`` capacity
reallocation (move representational precision to where d_seg lives), realised on the witness's OWN Fourier
basis instead of the HNeRV channel schedule.

TWO PROVABLE DEGENERACIES (VERIFIED_VIA_SOURCE_INSPECTION — the byte-identity contract): ``strength==0``
⇒ ``w==1`` exactly; a UNIFORM saliency ⇒ ``r_k`` constant ⇒ ``w==1`` exactly. And ``mean_k w_k == 1`` by
construction (a REALLOCATION, not a net amplitude change) ⇒ byte-NEUTRAL (adds ZERO trainable params;
archive unchanged) and rule-118 FREE (deterministic prior from GT geometry, recomputable at decode).

d_seg EFFECT = ASSUMED_AWAITING_VERIFICATION (the honest tier). Ledger #121: "+18% NO-GO RETRACTED
(under-converged ge300/3000); converged anchors flip sign to -8% ~0.03; RE-VALIDATE at convergence
(cheap disk A/B)". The sign/magnitude of the d_seg move is NOT settled until a CONVERGED byte-close A/B
measures it. means != ends: this is a train-time spectral prior (advisory, NON-PROMOTABLE); pointer
0.19110 UNMOVED.

DSL leg: ``tac.witness_dsl.curriculum_dsl.DsegAwareTaper`` (--dseg-aware-taper / -strength / -scale /
-floor). Mechanism: ``tac.boundary_math.dseg_aware_fourier_taper``.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "dseg_aware_fourier_taper_reweight_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- the ledger #121 relative-significance anchor numbers (load-bearing, quoted verbatim) ---------
LEDGER_UNDERCONVERGED_DELTA_PCT = +0.18   # +18% NO-GO measured at ge300/3000 (UNDER-CONVERGED, retracted)
LEDGER_CONVERGED_FLIP_DELTA_PCT = -0.08   # converged anchors flip sign to -8%
LEDGER_EST_DELTA_S = 0.03                 # est_delta_s at convergence (RE-VALIDATE)


def build_dseg_aware_fourier_taper_v1() -> CanonicalEquation:
    """Build the #121 d_seg-aware Fourier-feature amplitude taper law with its honest-tier anchors.

    Two anchors: (1) the byte-identity / byte-neutral DEGENERACIES, VERIFIED_VIA_SOURCE_INSPECTION
    (provable in code + pinned by the boundary_math tests); (2) the d_seg EFFECT,
    ASSUMED_AWAITING_VERIFICATION (ledger #121: re-validate at convergence)."""

    anchor_degeneracy = EmpiricalAnchor(
        anchor_id="dseg_aware_taper_off_degeneracy_byte_identical_20260709",
        measurement_utc=_UTC,
        inputs={
            "mechanism": "tac.boundary_math.dseg_aware_fourier_taper.compute_dseg_aware_fourier_taper",
            "tests": "src/tac/boundary_math/tests/test_dseg_aware_fourier_taper.py",
        },
        predicted_output={
            "claim": ("strength==0 ⇒ w==1 exactly; uniform saliency ⇒ w==1 exactly; mean_k w_k==1 "
                      "pre-floor (byte-neutral reallocation, ZERO new params)"),
        },
        empirical_output={
            "strength_zero_flat": "w all-ones (test_strength_zero_is_flat_taper_exactly)",
            "uniform_saliency_flat": "max|w-1|==0 (test_uniform_saliency_is_flat_taper_exactly)",
            "byte_neutral_mean_one": "|mean(w)-1|<1e-5 (test_taper_is_byte_neutral_mean_one_prefloor)",
            "off_byte_identical_on_real_curvelet_basis": (
                "np.array_equal(off, feats) on the ACTUAL curvelet basis "
                "(test_real_curvelet_basis_off_is_byte_identical_on_changes)"),
            "verdict": ("the default-OFF trainer path is byte-identical by construction (the compute "
                        "block is gated on --dseg-aware-taper, default False); the strength=0 / "
                        "uniform-S degeneracies are the runtime guarantee, not a promise"),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/dseg_aware_fourier_taper.py",
        measurement_method="source inspection + NO-FAKE unit tests (19 passing, $0 numpy)",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/dseg_aware_fourier_taper.py",
            reactivation_criteria=("re-verify the degeneracies if the reweight formula changes "
                                   "(strength==0 and uniform-S must stay exactly flat)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_effect_owed = EmpiricalAnchor(
        anchor_id="dseg_aware_taper_effect_revalidate_at_convergence_121_20260709",
        measurement_utc=_UTC,
        inputs={
            "ledger_row": ".omx/state/lever_relative_significance.jsonl (lever d_seg_aware_taper_121)",
            "source_anchor": "#121 / relative_significance_reaudit_20260708",
            "underconverged_window": "ge300/3000 (NOT converged)",
        },
        predicted_output={
            "hypothesis": ("moving spectral capacity toward the boundary annulus lowers d_seg at "
                           "convergence (the configurable_taper_decoder reallocation, on the witness "
                           "Fourier basis)"),
        },
        empirical_output={
            "underconverged_delta_pct": LEDGER_UNDERCONVERGED_DELTA_PCT,
            "converged_flip_delta_pct": LEDGER_CONVERGED_FLIP_DELTA_PCT,
            "est_delta_s": LEDGER_EST_DELTA_S,
            "verdict_scope": "instance-under-convergence",
            "status": ("OWED — +18% NO-GO was measured UNDER-CONVERGED and RETRACTED; converged HNeRV "
                       "anchors flip sign to -8% (~0.03 est ΔS). The witness-basis effect is UNMEASURED "
                       "at convergence. The owed anchor is a CONVERGED byte-close A/B (cheap disk A/B). "
                       "This landing BUILDS the fireable lever (default-OFF, byte-identical); it makes "
                       "NO score claim. means != ends: pointer 0.19110 UNMOVED."),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="ledger #121 relative-significance re-audit (2026-07-08); no witness A/B yet",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="dseg_aware_fourier_taper.effect_revalidate_at_convergence",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("#121 d_seg-aware Fourier-feature amplitude taper: reweight each fixed Fourier/curvelet "
              "column's amplitude by GT d_seg (margin) saliency — w_k = 1 + strength*(r_k/mean_r - 1), "
              "r_k = <S,feats_k^2>/<1,feats_k^2>; byte-neutral spectral reallocation"),
        one_line_summary=(
            "Reweight per-Fourier-column amplitude toward the boundary annulus (r_k>1); byte-neutral "
            "(mean w=1, 0 params); OFF/uniform-S ⇒ flat; d_seg effect re-validate at convergence (#121)."
        ),
        latex_form=(
            r"w_k=1+\sigma\left(\frac{r_k}{\bar r}-1\right),\quad "
            r"r_k=\frac{\langle S,\,f_k^2\rangle}{\langle 1,\,f_k^2\rangle},\quad "
            r"S(x)=\overline{\exp(-|m(x)|/s)},\quad f'_{\cdot k}=f_{\cdot k}\max(w_k,\phi)"
        ),
        python_callable_module_path=(
            "tac.boundary_math.dseg_aware_fourier_taper:compute_dseg_aware_fourier_taper"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.DsegAwareTaper "
                      "(--dseg-aware-taper/-strength/-scale/-floor)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("byte-neutral spectral prior on the FIXED curvelet/Fourier basis; the d_seg SIGN "
                     "and MAGNITUDE are ASSUMED_AWAITING_VERIFICATION until a CONVERGED byte-close A/B "
                     "measures them (ledger #121 re-validate-at-convergence)"),
        },
        units_in={"feats": "coord_fourier_features", "saliency": "unit_mean_margin_kernel",
                  "strength": "dimensionless", "floor": "dimensionless"},
        units_out={"taper": "per_column_amplitude_weight"},
        empirical_anchors=(anchor_degeneracy, anchor_effect_owed),
        predicted_vs_empirical_residual={
            # the DEGENERACIES are exact (residual 0, source-verified); the d_seg EFFECT is OWED to the
            # converged A/B (no witness measurement yet) -> 0 recorded with the ASSUMED anchor's status.
            "off_degeneracy_byte_identity": 0.0,
            "dseg_effect_owed_converged_ab": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",       # DSL leg: DsegAwareTaper factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the wire-in consumer
        ),
        canonical_producers=(
            "tac.boundary_math.dseg_aware_fourier_taper",
        ),
        provenance=build_provenance_for_predicted(
            model_id="dseg_aware_fourier_taper.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_dseg_aware_fourier_taper_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the #121 d_seg-aware Fourier taper law (mirrors
    ``populate_anisotropic_basis_two_regime_allocation_equation``; latest-row-wins). This is the
    EQUATIONS leg of the #121 lever build; the DSL leg is ``curriculum_dsl.DsegAwareTaper``, the
    mechanism is ``tac.boundary_math.dseg_aware_fourier_taper``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_dseg_aware_fourier_taper_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="dseg_aware_fourier_taper_20260709 (equations leg of the #121 lever build; DSL leg = "
              "DsegAwareTaper; re-validate-at-convergence)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "LEDGER_CONVERGED_FLIP_DELTA_PCT",
    "LEDGER_EST_DELTA_S",
    "LEDGER_UNDERCONVERGED_DELTA_PCT",
    "build_dseg_aware_fourier_taper_v1",
    "populate_dseg_aware_fourier_taper_equation",
]
