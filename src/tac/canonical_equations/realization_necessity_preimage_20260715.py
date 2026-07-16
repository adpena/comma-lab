# SPDX-License-Identifier: MIT
"""Canonical equation: realization necessity by inverse factorization, per Morse-Smale stratum.

MEASURED 2026-07-15 by ``tools/necessity_inverse_factorization_solver.py`` on the real
frozen weights + the bit-exact cached n600 GT (``gt_n600.npz``; margin parity vs the real
forward == 0.0 per stage_b1). Inverts the flattened chain
``d_seg = decision ∘ N_seg ∘ A`` (``frozen_scorer_exact_factorization_20260715.md``)
stratum by stratum: decision⁻¹ = rank-4 argmax polytope margins; N⁻¹ = full-chain VJP
min-norm displacement; A⁻¹ = exact closed-form resize-support pullback; ∩ uint8 =
sub-LSB amplitude of the min-norm flip displacement.

The law (per-stratum necessity split; supersedes any per-class-only phrasing)
------------------------------------------------------------------------------
NECESSARY realization content factorizes over the Morse-Smale strata of the argmax
partition, and the two necessity currencies SEPARATE:

  * BYTES concentrate on EDGES (1-cells). Camera-res strictly-necessary support is
    ~1.66% of pixels (edge 1.646% + saddle 0.016%, exact A-support pullback); cells are
    membership-only (75.6% loose) and 22.7% is certified free (zero-weight). Spatial-only
    rate floors, n600: H-ladder (cracks x H_turn + per-curve overhead) = 303,047 bytes;
    K-ladder (generator FREE per rule 118: polygon rasterizer + region fill; seed =
    DP-vertex chains, brotli -q11) = 143,552 bytes @ eps=1px — K/H = 0.47 MEASURED.
    Road-Lane alone = 61% of the H-floor (H_turn 1.516 bits/step, 21.4 components/frame
    = the dash long-tail).
  * PRECISION concentrates on SADDLES (0-cells). Saddles carry ~0 MARGINAL bytes given
    coded edges (they are crack-graph vertices of degree >= 3, implied by curve
    intersections) but dominate fragility: min-norm flip displacement p10 = 0.89 LSB
    (vs edge 4.06), and 29.2% of sampled saddle pixels are sub-LSB realization-limited
    (uint8 ∩ preimage TIGHT) vs 12.5% at edges and 0% in interiors.

Therefore the V9·CGauge carrier set = (generator, seed) pairs per stratum: cell -> region
fill + palette constant (~15 B/video); edge -> curve generator + per-pair seed (Road-Lane
-> openpilot-polynomial + dash-phase-ξ reduction; Road-MyCar static -> one-time seed);
saddle -> NO byte section — a variable-tolerance (tighter-eps-near-junction) precision
annotation on the edge curves (tie-locus #360).

Scope / honesty: stratification + margins + floors are n600 (ALL scored pairs) on the
bit-exact cached argmax/margin fields; A-support is a stride-25 subset with EXACT
closed-form matrices; VJP fragility is a stride-75 x sampled-pixel subset (n=120;
runner-up==neighbor-label validated 80/80). Axis ``[macOS-CPU advisory]``; floors are
geometric/codelength quantities, NOT d_seg claims; research_only; no score claim.

Artifacts: ``experiments/results/necessity_solver_20260715/{strat,kladder,asupport,vjp,floors}.json``
Memo:      ``.omx/research/necessity_solver_inverse_factorization_20260715.md``
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "realization_necessity_preimage_per_stratum_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached n600 GT"
SOLVER = "tools/necessity_inverse_factorization_solver.py"
ART_DIR = "experiments/results/necessity_solver_20260715"
FLOORS_ARTIFACT_SHA256 = "e5e17120cd4573a66c6cce71ccabb537758b91f14202c5967bd1d13ed7118340"

# MEASURED camera-res necessity attribution (exact A-support pullback, stride-25 subset).
CAMERA_SUPPORT_FRAC = {
    "saddle": 0.000158,
    "edge": 0.016465,
    "cell_loose_membership_only": 0.756408,
    "certified_free_zero_weight": 0.226969,
}
# MEASURED spatial-only rate floors, n600 (bytes).
H_LADDER_EDGE_BYTES_N600 = 303_046.53
K_LADDER_EDGE_BYTES_N600_EPS1 = 143_552.5
K_OVER_H = K_LADDER_EDGE_BYTES_N600_EPS1 / H_LADDER_EDGE_BYTES_N600
# MEASURED fragility (full-chain VJP subset; LSB units, camera-res min-norm displacement).
SUB_LSB_FRAC = {"saddle": 0.2917, "edge": 0.125, "interior": 0.0}
FLIPDIST_P10_LSB = {"saddle": 0.889, "edge": 4.056, "interior": 50.84}


def stratum_rate_floor_bytes(
    cracks: float, h_turn_bits: float, components: float, *, hw_bits: float = 17.585
) -> float:
    """H-ladder spatial floor for one edge stratum (bytes).

    ``bits = cracks * H_turn + components * (log2(H*W) + 2)``; ``hw_bits`` defaults to
    ``log2(384*512)``. Spatial-only — temporal/ξ-advection prediction is named headroom.
    """
    if cracks < 0 or components < 0 or h_turn_bits < 0:
        raise ValueError("cracks/components/h_turn_bits must be >= 0")
    return (cracks * h_turn_bits + components * (hw_bits + 2.0)) / 8.0


def necessity_split(camera_frac: dict[str, float] | None = None) -> dict[str, float]:
    """Strict-necessary vs loose vs free camera-res fractions (must sum to ~1)."""
    f = dict(CAMERA_SUPPORT_FRAC if camera_frac is None else camera_frac)
    strict = f["saddle"] + f["edge"]
    return {
        "strict_necessary": strict,
        "loose_membership_only": f["cell_loose_membership_only"],
        "free_certified": f["certified_free_zero_weight"],
    }


def build_realization_necessity_preimage_per_stratum_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=FLOORS_ARTIFACT_SHA256,
        source_path=f"{ART_DIR}/floors.json",
        captured_at_utc="2026-07-16T04:20:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="necessity_camera_support_split_20260715",
            measurement_utc="2026-07-16T04:10:00Z",
            inputs={"resize": "exact closed-form R_h(384,874) x R_w(512,1164)",
                    "frames": "stride-25 subset (24 of n600)"},
            predicted_output={"strict_necessary_lt": 0.05,
                              "reason": "annulus #333 ~4.7% area bounds the seen-necessary set"},
            empirical_output={"camera_support_frac": dict(CAMERA_SUPPORT_FRAC),
                              "strict_necessary": 0.016623},
            residual=0.0,
            source_artifact=f"{ART_DIR}/asupport.json",
            measurement_method=(
                "exact support pullback (R_h^T M R_w > 0) of the saddle/edge/cell strata masks; "
                "priority saddle > edge > cell; zero-weight set certified by the closed-form kernel"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="necessity_rate_floors_n600_20260715",
            measurement_utc="2026-07-16T04:10:00Z",
            inputs={"frames": "n600 (ALL scored pairs) cached lstars",
                    "models": "H: cracks x H_turn + comp x (log2(HW)+2); K: DP polygons + brotli q11"},
            predicted_output={"K_le_H": True, "reason": "Kolmogorov generator+seed <= entropy coding"},
            empirical_output={
                "H_ladder_edge_bytes_n600": H_LADDER_EDGE_BYTES_N600,
                "K_ladder_edge_bytes_n600_eps1": K_LADDER_EDGE_BYTES_N600_EPS1,
                "K_over_H": K_OVER_H,
                "road_lane_share_of_H": 0.61,
            },
            residual=0.0,
            source_artifact=f"{ART_DIR}/floors.json",
            measurement_method=(
                "n600 crack counts / degree-2 vertex turn statistics / connected components; "
                "DP-simplified per-class contours int16-delta packed + brotli -q11 (measured bytes)"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="necessity_saddle_fragility_sub_lsb_20260715",
            measurement_utc="2026-07-16T04:10:00Z",
            inputs={"frames": "stride-75 subset (8 of n600), 120 sampled pixels",
                    "chain": "camera-res -> exact resize -> frozen SegNet (full-chain VJP)"},
            predicted_output={"saddles_most_fragile": True,
                              "reason": "argmax least stable where >= 3 cells meet"},
            empirical_output={"sub_lsb_frac": dict(SUB_LSB_FRAC),
                              "flipdist_p10_lsb": dict(FLIPDIST_P10_LSB),
                              "runnerup_equals_neighbor": "80/80"},
            residual=0.0,
            source_artifact=f"{ART_DIR}/vjp.json",
            measurement_method=(
                "per-pixel margin backward to the camera-res tensor through the exact bilinear "
                "resize; min-norm displacement m/||g||; sub-LSB iff its max coordinate < 0.5"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Realization necessity by inverse factorization, per Morse-Smale stratum",
        one_line_summary=(
            "Necessity separates by stratum: BYTES on edges (strict camera support 1.66%; "
            "K/H=0.47, Road-Lane 61%), PRECISION on saddles (29.2% sub-LSB, ~0 marginal "
            "bytes given edges); cells = membership."
        ),
        latex_form=(
            r"\mathrm{necessary} = \bigsqcup_{\sigma \in \{cell,edge,saddle\}} "
            r"\mathrm{preimage}_\sigma,\quad "
            r"R_{floor} = \sum_\sigma |\mathrm{seed}_\sigma|,\quad "
            r"|\mathrm{seed}_{saddle}| \approx 0 \text{ given edges (precision, not bytes)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.realization_necessity_preimage_20260715:"
            "stratum_rate_floor_bytes"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet through the exact shared bilinear resize",
            "target": "cached n600 GT argmax partition (bit-exact vs the real forward)",
            "floors": "spatial-only geometric/codelength quantities at (384,512); "
                      "NOT d_seg claims; temporal/ξ-advection prediction is named headroom",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "n600 stratification + subset A-support/VJP (strides labeled)",
        },
        units_in={"cracks": "lattice arc length (crack count)",
                  "h_turn_bits": "bits/step", "components": "curves"},
        units_out={"rate_floor": "bytes"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"K_le_H": 0.0, "strict_necessary_lt_annulus": 0.0},
        last_calibration_utc="2026-07-16T04:20:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "V9·CGauge per-stratum carriers (SPEC_v8 #359/#380/#386: decoupled_field / "
            "road_undriv_bulk_field / curve_relative_offset_coder / saddle precision annotation)",
            "p0_UNIFICATION_projection_preimage_SUPREME_20260715 (crown mechanism)",
            "rate_law_ladder_v2_measured (sibling codelength surface)",
        ),
        canonical_producers=(SOLVER,),
        provenance=provenance,
    )


def populate_realization_necessity_preimage_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of FEED-necessity)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_realization_necessity_preimage_per_stratum_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "AXIS",
    "CAMERA_SUPPORT_FRAC",
    "EQUATION_ID",
    "FLIPDIST_P10_LSB",
    "H_LADDER_EDGE_BYTES_N600",
    "K_LADDER_EDGE_BYTES_N600_EPS1",
    "K_OVER_H",
    "SUB_LSB_FRAC",
    "build_realization_necessity_preimage_per_stratum_v1",
    "necessity_split",
    "populate_realization_necessity_preimage_equation",
    "stratum_rate_floor_bytes",
]
