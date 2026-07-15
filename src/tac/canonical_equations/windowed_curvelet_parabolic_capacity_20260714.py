# SPDX-License-Identifier: MIT
"""Canonical equation: windowed-curvelet PARABOLIC-SCALING localization + boundary-annulus
N-term capacity law (task #502, DAG FEED-cvl).

The genuinely localized windowed-curvelet frame
(``tac.boundary_math.windowed_curvelet_frame``) places directional Gabor atoms whose
anisotropic Gaussian window obeys the CURVELET PARABOLIC-SCALING LAW: the support width
ACROSS the boundary (normal) scales as the length ALONG the boundary (tangent) SQUARED,

    sigma_n(j) = w0 * r**(-j)          sigma_t(j) = aniso * w0 * r**(-j/2)
    ==>  sigma_n = (sigma_t / aniso)**2 / w0     (i.e. width ~ length^2)

This is the Candes-Donoho (2004) law that makes curvelets the OPTIMAL sparse
(N-term) basis for C^2 curved-edge singularities -- exactly the codim-1 oriented
SegNet-argmax boundary annulus (memory L66, ~4.7% area; ~26.8% of n600 d_seg flip-mass).

Torralba-Weiss (arXiv 2607.07470) prove global isotropic Fourier is the optimal
contrastive first layer ONLY under global translation-invariant stationarity (Thm 2.1,
Peligrad-Wu). Our oriented boundary VIOLATES stationarity, so global plane waves are
provably sub-optimal, quantified two ways:

  (spectral, measured)  waterfill_boundary_spectrum_probe.py: boundary margin spectrum
      is 41x anisotropic (n600); reverse-waterfilling gives an oriented basis a 1.7-2.0x
      RATE capacity advantage (a linear UPPER bound).
  (spatial, measured)   curvelet_vs_fourier_capacity_probe.py: at MATCHED coefficient
      budget K (= matched counted bytes), OMP N-term reconstruction of real boundary
      patches with the localized curvelet frame needs ~1.09-1.23x FEWER coefficients to
      reach high fidelity than the current oriented-Fourier basis -- even with a SMALLER
      curvelet dictionary. The gain GROWS with fidelity (K), mirroring the waterfill trend.

The frame's genuine localization (vs the catalog-#351 fake of a Fourier basis carrying a
curvelet label) is CERTIFIED: paired envelope span 1.0 (Gaussian bump) vs the polar-Fourier
bank's 1.5e-7 (constant envelope); 98.7% energy in the top 10% of pixels; 1.99x tangent
elongation. A plain oriented-Fourier basis FAILS the same certificate (the swap-test).

means != ends: this is a representation/capacity derivation (advisory anchors,
NON-PROMOTABLE). The realized d_seg through-R on exact bytes is OWED (needs a run;
operator-GO / CONTAINMENT). Pointer UNMOVED (0.19108 submittable / 0.18804 bank).
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    INFERRED_FROM_DOMAIN_LITERATURE,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "windowed_curvelet_parabolic_capacity_v1"

_UTC = "2026-07-14T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- MEASURED constants (the anchors' load-bearing numbers) --------------------------------------
BOUNDARY_SPECTRUM_ANISOTROPY_N600 = 41.0        # max/min orientation energy (waterfill probe, n600)
SPECTRAL_CAPACITY_GAIN_LOW = 1.7                # reverse-waterfill B_iso/B_orient (D<=0.50/0.40)
SPECTRAL_CAPACITY_GAIN_HIGH = 2.0              # reverse-waterfill B_iso/B_orient (D<=0.70)
SPATIAL_NTERM_CAPACITY_GAIN_N600 = 1.09        # OMP K_fourier/K_curvelet at rel-err<=0.10 (n600)
SPATIAL_NTERM_CAPACITY_GAIN_N96 = 1.23         # OMP K_fourier/K_curvelet at rel-err<=0.10 (n96)
CURVELET_ENVELOPE_SPAN = 1.0                    # localized paired-envelope span (Gaussian bump)
FOURIER_ENVELOPE_SPAN = 1.514988134942996e-07  # polar-Fourier constant envelope (NOT localized)
CURVELET_ENERGY_CONCENTRATION = 0.987          # frac of atom energy in top-10% pixels
CURVELET_TANGENT_ELONGATION = 1.99             # median sigma_t_eff / sigma_n_eff (edge-shaped)


def parabolic_sigma_pair(
    j: int, *, w0: float = 0.5, width_ratio: float = 2.0, aniso: float = 1.0, min_sigma: float = 0.02
) -> tuple[float, float]:
    """The curvelet parabolic-scaling law: (sigma_n, sigma_t) at radial octave j.

    ``sigma_n = w0 * r**(-j)`` (across/normal width), ``sigma_t = aniso * w0 * r**(-j/2)``
    (along/tangent length). Then ``sigma_n = (sigma_t/aniso)**2 / w0`` -- width ~ length^2
    (Candes-Donoho parabolic scaling). Clamped by ``min_sigma`` so atoms never collapse
    below the sampling grid. Raises on non-positive / non-finite args (fail-closed)."""
    if isinstance(j, bool) or not isinstance(j, int) or j < 0:
        raise ValueError(f"j must be a non-negative int, got {j!r}")
    for nm, v in (("w0", w0), ("width_ratio", width_ratio), ("aniso", aniso), ("min_sigma", min_sigma)):
        if not math.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError(f"{nm} must be finite and > 0, got {v!r}")
    if float(width_ratio) <= 1.0:
        raise ValueError("width_ratio must exceed 1 (radial octave ratio)")
    if float(aniso) < 1.0:
        raise ValueError("aniso must be >= 1 (tangent elongation)")
    r = float(width_ratio)
    sigma_n = max(float(w0) * r ** (-j), float(min_sigma))
    sigma_t = max(float(aniso) * float(w0) * r ** (-0.5 * j), float(min_sigma))
    return sigma_n, sigma_t


def build_windowed_curvelet_parabolic_capacity_v1() -> CanonicalEquation:
    """Build the windowed-curvelet parabolic-scaling + capacity law with honest-tier anchors."""

    anchor_localization = EmpiricalAnchor(
        anchor_id="windowed_curvelet_localization_certificate_20260714",
        measurement_utc=_UTC,
        inputs={
            "primitive": "tac.boundary_math.windowed_curvelet_frame.windowed_curvelet_feats",
            "certificate": "localization_certificate (swap-test vs polar-Fourier bank)",
            "grid": "33x33 coord grid, default WindowedCurveletConfig",
        },
        predicted_output={"claim": "the frame is GENUINELY spatially localized (not Fourier)"},
        empirical_output={
            "curvelet_envelope_span": CURVELET_ENVELOPE_SPAN,
            "fourier_envelope_span": FOURIER_ENVELOPE_SPAN,
            "curvelet_energy_concentration_top10pct": CURVELET_ENERGY_CONCENTRATION,
            "curvelet_tangent_elongation": CURVELET_TANGENT_ELONGATION,
            "verdict": ("PASSES the catalog-#351 swap-test: paired-envelope span 1.0 (Gaussian "
                        "bump) vs polar-Fourier 1.5e-7 (constant); 98.7% energy in top-10% pixels; "
                        "1.99x tangent elongation. An oriented-Fourier basis FAILS the same gate."),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/windowed_curvelet_frame.py",
        measurement_method="localization_certificate() on the default frame vs the polar-Fourier bank",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/windowed_curvelet_frame.py",
            reactivation_criteria=("re-run localization_certificate if the atom construction changes; "
                                   "passes must stay True and Fourier must stay below the span gate"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_capacity = EmpiricalAnchor(
        anchor_id="curvelet_vs_fourier_nterm_capacity_20260714",
        measurement_utc=_UTC,
        inputs={
            "probe_spectral": "experiments/waterfill_boundary_spectrum_probe.py (reverse-waterfilling)",
            "probe_spatial": "experiments/curvelet_vs_fourier_capacity_probe.py (OMP N-term)",
            "data": "frozen-SegNet argmax boundary annulus, gt_n600.npz / gt_n96.npz",
            "control": "curvelet dictionary kept SMALLER than Fourier (192 vs 236 cols)",
        },
        predicted_output={
            "law": "localized oriented atoms give optimal N-term for C^2 curved edges",
            "spectral_upper_bound_capacity_gain": [SPECTRAL_CAPACITY_GAIN_LOW, SPECTRAL_CAPACITY_GAIN_HIGH],
        },
        empirical_output={
            "boundary_spectrum_anisotropy_n600": BOUNDARY_SPECTRUM_ANISOTROPY_N600,
            "spatial_nterm_capacity_gain_n600": SPATIAL_NTERM_CAPACITY_GAIN_N600,
            "spatial_nterm_capacity_gain_n96": SPATIAL_NTERM_CAPACITY_GAIN_N96,
            "verdict": ("CORROBORATED (same sign, comparable ballpark): spectral reverse-waterfill "
                        "gives 1.7-2.0x RATE gain (linear upper bound); spatial OMP N-term gives "
                        "~1.09x (n600) / ~1.23x (n96) FEWER coefficients at matched budget to reach "
                        "rel-err<=0.10, even with a SMALLER curvelet dictionary; gain GROWS with "
                        "fidelity (K). The spatial realized gain is below the spectral upper bound "
                        "(conservative greedy OMP + modest bank) -- honest, expected."),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method=("reverse-waterfilling (spectral) + OMP N-term reconstruction (spatial) on "
                            "the boundary annulus; matched coefficient budget = matched counted bytes"),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria=("the REALIZED d_seg gain is OWED: build train+inflate op-parity for "
                                   "the windowed-curvelet feats and measure d_seg through-R on exact "
                                   "n600 bytes (operator-GO / CONTAINMENT). Linear/greedy probes are "
                                   "an UPPER bound, not the score."),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_parabolic = EmpiricalAnchor(
        anchor_id="curvelet_parabolic_scaling_law_20260714",
        measurement_utc=_UTC,
        inputs={
            "literature": "Candes-Donoho 2004 (curvelets: parabolic scaling width ~ length^2 is the "
                          "optimal sparse basis for C^2-cartoon curved singularities)",
            "law": "sigma_n = (sigma_t/aniso)^2 / w0",
        },
        predicted_output={
            "sigma_pair_j0": list(parabolic_sigma_pair(0)),
            "sigma_pair_j2": list(parabolic_sigma_pair(2)),
            "note": "sigma_n shrinks faster than sigma_t in j -> anisotropy r^(j/2) grows with scale",
        },
        empirical_output={
            "status": ("the parabolic law is IMPLEMENTED in the primitive and CERTIFIED monotone "
                       "(parabolic_scaling_monotone=True); the OPTIMAL (w0, width_ratio, aniso) for "
                       "d_seg is OWED to the through-R A/B sweep"),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/windowed_curvelet_frame.py",
        measurement_method="derivation from Candes-Donoho parabolic scaling (law implemented, not yet swept)",
        empirical_verification_status=INFERRED_FROM_DOMAIN_LITERATURE,
        provenance=build_provenance_for_predicted(
            model_id="windowed_curvelet_parabolic_capacity.parabolic_law",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Windowed-curvelet parabolic-scaling localization + boundary-annulus N-term capacity: "
              "sigma_n ~ sigma_t^2 (Candes-Donoho); measured 41x boundary anisotropy -> 1.7-2.0x "
              "spectral / ~1.1-1.2x spatial N-term capacity gain over oriented Fourier"),
        one_line_summary=(
            "A genuinely localized curvelet frame (width ~ length^2) needs fewer coefficients than "
            "oriented Fourier on the measured boundary annulus; realized d_seg through-R is OWED."
        ),
        latex_form=(
            r"\sigma_n(j)=w_0 r^{-j},\ \sigma_t(j)=a\,w_0 r^{-j/2}\ \Rightarrow\ "
            r"\sigma_n=(\sigma_t/a)^2/w_0\quad(\text{width}\sim\text{length}^2)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.windowed_curvelet_parabolic_capacity_20260714:"
            "parabolic_sigma_pair"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "primitive": "tac.boundary_math.windowed_curvelet_frame.windowed_curvelet_feats",
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("linear/greedy capacity probes are an UPPER bound on the realized trained-witness "
                     "d_seg; the through-R d_seg on exact bytes is the score and is OWED (needs a run)"),
        },
        units_in={"j": "radial_octave_int", "w0": "coord_units", "width_ratio": "ratio", "aniso": "ratio"},
        units_out={"sigma_n": "coord_units", "sigma_t": "coord_units"},
        empirical_anchors=(anchor_localization, anchor_capacity, anchor_parabolic),
        predicted_vs_empirical_residual={
            # spectral (upper bound) vs spatial (realized-greedy) capacity gain gap, high-fidelity n96.
            "spectral_vs_spatial_gain_gap_n96": abs(
                SPECTRAL_CAPACITY_GAIN_LOW - SPATIAL_NTERM_CAPACITY_GAIN_N96),
            # Fourier localization leakage: how close the polar-Fourier bank comes to looking
            # localized = its paired-envelope span (~0 -> it decisively FAILS the curvelet claim).
            "fourier_localization_leakage": FOURIER_ENVELOPE_SPAN,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.windowed_curvelet_basis_lever_20260714",  # DSL leg: the (OWED-wire) lever
        ),
        canonical_producers=(
            "experiments/waterfill_boundary_spectrum_probe.py",
            "experiments/curvelet_vs_fourier_capacity_probe.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="windowed_curvelet_parabolic_capacity.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_windowed_curvelet_parabolic_capacity_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the windowed-curvelet parabolic-capacity law
    (EQUATIONS leg of DAG FEED-cvl / task #502; DSL leg = windowed_curvelet_basis_lever_20260714)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_windowed_curvelet_parabolic_capacity_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="windowed_curvelet_parabolic_capacity_20260714 (equations leg of task #502; DSL leg = "
              "windowed_curvelet_basis_lever_20260714; primitive = boundary_math.windowed_curvelet_frame)",
    )
    return eq


__all__ = [
    "BOUNDARY_SPECTRUM_ANISOTROPY_N600",
    "CURVELET_ENERGY_CONCENTRATION",
    "CURVELET_ENVELOPE_SPAN",
    "CURVELET_TANGENT_ELONGATION",
    "EQUATION_ID",
    "FOURIER_ENVELOPE_SPAN",
    "SPATIAL_NTERM_CAPACITY_GAIN_N600",
    "SPATIAL_NTERM_CAPACITY_GAIN_N96",
    "SPECTRAL_CAPACITY_GAIN_HIGH",
    "SPECTRAL_CAPACITY_GAIN_LOW",
    "build_windowed_curvelet_parabolic_capacity_v1",
    "parabolic_sigma_pair",
    "populate_windowed_curvelet_parabolic_capacity_equation",
]
