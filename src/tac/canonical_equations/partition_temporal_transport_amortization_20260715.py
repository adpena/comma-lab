# SPDX-License-Identifier: MIT
"""Canonical equation: partition temporal-transport rate amortization is JITTER-BOUND.

THE LAW (measured n600, lens-5 TEMPORAL/ADVECTION of the projection unification,
``projection_unification_and_eight_lenses_20260715.md``): for the SegNet argmax
partition trajectory ``L_0..L_{P-1}`` (frame_1 argmax per non-overlapping pair, 2-frame
step), the transport-conditional rate decomposes per Morse-Smale stratum as

    R(L_{p+1} | T(L_p, xi))  =  R_cell_residual (+~0)  +  R_jitter(edge, saddle) + R_flicker(cell)

and the TRANSPORTABLE stratum (cell interiors, 97.8% of pixels, 99.4% recovered by
transport) is exactly the content a generic per-frame coder already compresses to ~0
bits, while the irreducible residual (separatrix jitter ~54% of residual px + interior
flicker ~45% + saddle rebirth ~0.8%) has spatial entropy >= the whole partition's:

    mean residual bytes/frame (zlib-9 conditional proxy)  =  1,403 B (persist) / 1,411 B (screw)
    mean naive partition bytes/frame (zlib-9)             =  1,003 B
    =>  amortization ratio naive/trajectory = 0.715 (persist) / 0.711 (screw)  <  1

TEMPORAL TRANSPORT DOES NOT AMORTIZE THE PARTITION RATE in the raster conditional-coding
formulation; the per-frame necessary content IS the boundary jitter (the same object as
the flicker floor L85 + annulus boundary-jitter #333). The ego-screw xi's marginal value
over plain persistence at the 2-frame pair gap is ~0 for every stratum (total transport
d_seg 0.012465 screw vs 0.012456 persist; edge mean separatrix offset 4.60 vs 4.67 px;
saddle same-signature match 54.7% vs 54.4%). xi's REAL banked value stays where the
grok/screw probes put it: Road-bulk d_seg modulation + d_pose dual-use + the curve-domain
phase carrier (#424/#425) — NOT raster rate amortization.

verdict_scope: FORMULATION (raster label-grid transport + generic zlib-9 conditional
proxy + adjacent-pose xi proxy at the 2-frame non-overlapping-pair gap). NOT family-dead:
a boundary-context arithmetic coder, a 1-frame gap, or the CURVE-domain delta(s) carrier
with an explicit birth/escape model could still amortize — but the measured jitter prior
(d0 40.4% / <=1px 72.3% / <=2px 79.8% / >2px 20.2%) puts even the curve-domain per-site
offset entropy at ~2.2 bits/site ~= the naive frame rate again (DERIVED, flagged).

means != ends: a measured negative + a carrier prior; NO score claim; pointer UNMOVED.
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

EQUATION_ID = "partition_temporal_transport_amortization_jitter_bound_v1"

_UTC = "2026-07-15T00:00:00Z"
_AXIS = "[macOS advisory / research-signal]"
_MEMO = ".omx/research/temporal_advection_stratified_20260715.md"
_JSON = "experiments/results/temporal_advection_stratified_n600_20260715/results.json"

# --- measured n600 constants (source: _JSON) ---------------------------------------------------
NAIVE_BYTES_PER_FRAME = 601_931 / 600.0          # zlib-9 per-frame partition coding
L0_BYTES = 1_083                                 # exact zlib-9 bytes of lstars[0] (measured)
RESIDUAL_BYTES_PER_FRAME_PERSIST = (841_690 - L0_BYTES) / 599.0
RESIDUAL_BYTES_PER_FRAME_SCREW = (846_116 - L0_BYTES) / 599.0
CELL_TRANSPORT_RECOVERY_SCREW = 0.994279         # fraction of cell px supplied by transport
EDGE_TRANSPORT_RECOVERY_SCREW = 0.683784
SADDLE_TRANSPORT_RECOVERY_SCREW = 0.548130
EDGE_OFFSET_LE1PX_SCREW = 0.723408               # delta(s) jitter prior (curve carrier)
SADDLE_MATCH_R2_SCREW = 0.547328


def amortization_ratio(naive_bytes_total: float, trajectory_bytes_total: float) -> float:
    """naive / trajectory; > 1 means transport-conditional coding WINS (measured: 0.71 < 1)."""
    if naive_bytes_total <= 0 or trajectory_bytes_total <= 0:
        raise ValueError("byte totals must be positive")
    return float(naive_bytes_total) / float(trajectory_bytes_total)


def transport_pays(naive_frame_bytes: float, residual_frame_bytes: float,
                   xi_marginal_bytes_per_frame: float = 0.0) -> bool:
    """The decision rule the law encodes: temporal transport amortizes the partition rate
    iff the per-frame conditional residual (+ xi marginal) undercuts the per-frame coder.
    Measured n600: 1403 + 0 > 1003 => False (jitter-bound)."""
    return (float(residual_frame_bytes) + float(xi_marginal_bytes_per_frame)
            < float(naive_frame_bytes))


def build_partition_temporal_transport_amortization_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="temporal_advection_stratified_n600_20260715",
        measurement_utc=_UTC,
        inputs={
            "cache": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (frozen CPU-torch "
                     "SegNet argmax lstars + gt_poses, n600, 384x512)",
            "predictors": "persist | ground homography | single-twist screw stratified "
                          "(hood=identity, sky=rotonly, ground classes=plane homography); "
                          "3 global scalars fit on Road+Lane (first 100 transitions)",
            "tool": "tools/temporal_advection_stratified_measure.py",
        },
        predicted_output={
            "lens5_hope": "cells/edges advect cleanly by xi; trajectory rate = |one partition| "
                          "+ |xi| + small residual << naive per-frame coding",
        },
        empirical_output={
            "per_stratum_transport_recovery_screw": {
                "cell": CELL_TRANSPORT_RECOVERY_SCREW,
                "edge": EDGE_TRANSPORT_RECOVERY_SCREW,
                "saddle": SADDLE_TRANSPORT_RECOVERY_SCREW,
            },
            "xi_marginal_over_persist": {
                "total_dseg_transport": {"persist": 0.012456, "screw": 0.012465},
                "edge_mean_offset_px": {"persist": 4.670, "screw": 4.599},
                "saddle_match_r2": {"persist": 0.5442, "screw": 0.5473},
            },
            "rate_zlib9_proxy": {
                "naive_bytes_total_600_frames": 601_931,
                "trajectory_bytes_total": {"persist": 841_690, "screw": 846_116},
                "amortization_ratio": {"persist": 0.7151, "screw": 0.7114},
            },
            "residual_px_stratum_share_screw": {"cell": 0.449, "edge": 0.543, "saddle": 0.008},
            "edge_jitter_prior_screw": {"d0": 0.404, "le1": 0.723, "le2": 0.798, "gt2": 0.202},
            "verdict_scope": "FORMULATION (raster transport + zlib-9 conditional proxy + "
                             "adjacent-pose xi proxy at 2-frame gap); not family-dead",
        },
        residual=0.0,
        source_artifact=_JSON,
        measurement_method="deterministic numpy label-space transport vs frozen SegNet argmax "
                           "cache; zlib-9 conditional-coding proxy (PRE-R, PROXY, non-promotable)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-measure if (a) a boundary-context conditional coder is "
                                   "built, (b) the 1-frame gap becomes codable, or (c) the "
                                   "curve-domain delta(s) carrier (#424/#425) lands an explicit "
                                   "birth/escape model beating ~2.2 bits/site"),
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_cpu_numpy",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Partition temporal-transport amortization is jitter-bound: transport recovers the "
              "cell stratum (~99.4%) but the conditional residual (separatrix jitter + interior "
              "flicker) out-costs per-frame coding; xi marginal over persist ~0 at the pair gap"),
        one_line_summary=(
            "R(L_{p+1}|T(L_p,xi)) > R(L_{p+1}) under generic coding (ratio 0.71): the "
            "transportable content is the already-free cell bulk; the rate IS the boundary jitter."
        ),
        latex_form=(
            r"R\big(L_{p+1}\mid T(L_p,\xi)\big)\;\approx\;R_{\mathrm{jitter}}"
            r"(\partial L_{p+1})+R_{\mathrm{flicker}}\;\ge\;R(L_{p+1}),\qquad "
            r"\frac{\sum_p R(L_p)}{R(L_0)+\sum_p R(L_{p+1}\mid T)}=0.71<1"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.partition_temporal_transport_amortization_20260715:"
            "transport_pays"
        ),
        domain_of_validity={
            "vehicle": "frozen SegNet argmax partition trajectory (lstars), 384x512, 0.mkv",
            "transport": ["persist", "ground_homography", "single_twist_screw_stratified"],
            "gap": "2-frame (non-overlapping pairs); 1-frame gap UNMEASURED",
            "coder": "zlib-9 raster conditional proxy; boundary-context / curve-domain coders "
                     "OUTSIDE (reactivation criteria)",
            "measurement_axis": [_AXIS],
            "verdict_scope": "FORMULATION-level negative; xi value elsewhere intact "
                             "(Road bulk, d_pose dual-use, phase carrier #424/#425)",
        },
        units_in={"naive_frame_bytes": "bytes", "residual_frame_bytes": "bytes",
                  "xi_marginal_bytes_per_frame": "bytes"},
        units_out={"transport_pays": "bool", "amortization_ratio": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"n600_zlib9_amortization_ratio_screw": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "src/tac/boundary_math/phase_residual_carrier.py",   # curve-domain delta(s) carrier prior
            "src/tac/boundary_math/curve_relative_offset_coder.py",
            ".omx/research/projection_unification_and_eight_lenses_20260715.md",  # lens 5 verdict
        ),
        canonical_producers=(
            "tools/temporal_advection_stratified_measure.py",
            _MEMO,
        ),
        provenance=build_provenance_for_predicted(
            model_id="partition_temporal_transport_amortization.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_cpu_numpy",
        ),
    )


def populate_partition_temporal_transport_amortization_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins).

    EQUATIONS leg of the temporal-advection lens-5 unit; DAG leg = FEED-adv-strat row in
    the sub015 DAG; DSL leg = N/A (measurement-only standalone tool; the curve-domain
    routing already lives in the #424/#425 phase-carrier levers)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_partition_temporal_transport_amortization_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="temporal_advection_stratified_20260715 (lens 5 per-stratum xi-transport + "
              "trajectory-rate amortization, n600 measured; jitter-bound negative, "
              "FORMULATION-scoped)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "amortization_ratio",
    "build_partition_temporal_transport_amortization_v1",
    "populate_partition_temporal_transport_amortization_equation",
    "transport_pays",
]
