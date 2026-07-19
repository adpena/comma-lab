# SPDX-License-Identifier: MIT
"""Canonical equation: dash-phase carrier rate law — object-domain δ(s) coding + measured
blink-back amortization + the site-prior → dash-centroid divergence (#425 STORE leg).

THE LAW (measured n600, `tools/measure_dash_phase_carrier_n600.py`, build wave 2026-07-17):
for the lane-dash trajectory coded as ξ-advected world-frame TRACKS with per-dash
curve-relative residuals (δs, δn) and explicit birth/death/rebirth events,

    R_dash = R_events(alive bits + anchors + rebirth idx) + R_δ(prior-code symbols + ESC)

with the MEASURED n600 accounting (frozen SegNet argmax lstars, advection-memo-calibrated
pose→ξ s_t=-0.00322 / s_r=0 / pitch=-0.01):

  * section 29,958 B excl-ξ (49.9 B/frame): 11.3× UNDER the lane-only per-frame naive
    (packbits+zlib9 338,523 B, measured) and 2.1× under the naive per-frame anchor stream
    (62,953 B) — the object-domain formulation AMORTIZES where the raster formulation
    measured 0.71 < 1 — but 16.6× OVER the 0.9–1.8 KB world-anchor budget band (which
    assumed a FREE deterministic visibility generator; the measured event stream is the gap).
  * **blink-back fraction 0.787 (FIRST MEASUREMENT** of the lane memo §4 open item):
    5,632 of 7,154 post-frame0 (re)appearances re-anchor a DORMANT world track (≤30-frame
    horizon) instead of paying a full anchor — the world-frame static-anchor-field reading
    is CONFIRMED at the track level (~79% of "births" are re-births of existing world paint).
  * **the SITE-level jitter prior does NOT transfer to the DASH-centroid level**: measured
    per-dash transport offset gt2 = 64.3% (vs the separatrix-site prior's 20.2%); the
    prior-derived Huffman code (pre-registered 4.53 bits/dash) realizes 9.58 bits/dash
    (ESC 26.5%), and LOSES to generic zlib9 on the realized δ stream (11,863 vs 9,498 B):
    an iid symbol code from the WRONG-level prior is dominated — the dash-centroid channel
    needs its OWN prior (+ real context), not the site prior.
  * recovery (LABEL-SPACE, honest scope — not through-R, not a score): phase-correct decode
    places matched dash centroids at mean 0.38 px (≤1 px 100%, lossless-to-q=1); lane-layer
    raster XOR (shape-persistence approximation, matched tracks) persist 1.129 /
    transport-only 1.173 / phase-correct 0.749 — the δ correction buys back ~34% of the
    lane-layer XOR error where transport ALONE is WORSE than persistence (consistent with
    the raster memo's ξ-marginal ≈ 0).

verdict_scope: FORMULATION (centroid-level δ, iid prior-derived symbol code, greedy
matching, interp gap-screw, dormant horizon 30). NOT family-dead: the measured levers are
(a) a dash-level MEASURED prior (this row's histogram) replacing the site prior,
(b) per-track context (AR / class-conditioned) beating iid-vs-zlib, (c) a deterministic
persistence-class visibility generator turning the event stream FREE (the 16.6× budget gap).

means ≠ ends: a measured carrier row + two measured divergences; NO score claim; the
pointer moves only through ``upstream/evaluate.py``. Axis: [macOS-CPU advisory].
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

EQUATION_ID = "dash_phase_carrier_rate_blinkback_prior_divergence_v1"

_UTC = "2026-07-17T00:00:00Z"
_AXIS = "[macOS-CPU advisory]"
_MEMO = ".omx/research/phase_carrier_425_build_20260717.md"
_JSON = "experiments/results/dash_phase_carrier_n600_20260717/results.json"

# --- measured n600 constants (source: _JSON) ---------------------------------------------------
SECTION_BYTES_EXCL_XI = 29_958
LANE_ONLY_NAIVE_BYTES = 338_523          # per-frame Lane packbits+zlib9, measured n600
NAIVE_ANCHOR_STREAM_BYTES = 62_953.0     # n_obs x ~5.5 B anchors
ANCHOR_BUDGET_BAND = (900, 1800)         # lane_channel refactor §5 (DERIVED, cited)
BLINK_BACK_FRACTION = 0.7870             # FIRST measurement of the §4 open item
EXPECTED_BITS_PER_DASH_PRIOR = 4.534     # pre-registered from the site prior
MEASURED_BITS_PER_MATCHED_DASH = 9.584   # realized (site prior does NOT transfer)
DASH_CENTROID_GT2 = 0.6429               # vs site-prior 0.202
PRIOR_CODE_DELTA_BYTES = 11_863
ZLIB9_DELTA_BYTES = 9_498
XOR_RATE_PERSIST = 1.1291
XOR_RATE_TRANSPORT_ONLY = 1.1733
XOR_RATE_PHASE_CORRECT = 0.7489


def dash_rate_amortizes(section_bytes: float, per_frame_naive_bytes: float) -> bool:
    """Object-domain amortization test: the carrier wins iff its total undercuts the
    per-frame naive total. Measured n600: 29,958 < 338,523 => True (11.3x)."""
    if section_bytes <= 0 or per_frame_naive_bytes <= 0:
        raise ValueError("byte totals must be positive")
    return float(section_bytes) < float(per_frame_naive_bytes)


def site_prior_transfers_to_dash(measured_bits: float, preregistered_bits: float,
                                 tolerance: float = 1.5) -> bool:
    """The prior-transfer test: the site-level jitter prior transfers iff the realized
    bits/dash stay within `tolerance`x the pre-registered expectation.
    Measured n600: 9.584 > 1.5 * 4.534 => False (the dash channel needs its own prior)."""
    return float(measured_bits) <= float(tolerance) * float(preregistered_bits)


def build_dash_phase_carrier_rate_law_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="dash_phase_carrier_n600_20260717",
        measurement_utc=_UTC,
        inputs={
            "cache": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (frozen CPU-torch "
                     "SegNet argmax lstars + gt_poses, n600, 384x512)",
            "codec": "src/tac/boundary_math/dash_phase_carrier.py (xi-advected world tracks, "
                     "prior-derived Huffman {0,±1,±2,ESC}, birth/death/rebirth events, "
                     "dormant horizon 30, q=1px, match radius 6px)",
            "calibration": "pose->xi s_t=-0.00322, s_r=0, pitch=-0.01 (advection-memo fit; "
                           "raw s_t=1 mis-advects: coverage 52% vs 87%)",
            "tool": "tools/measure_dash_phase_carrier_n600.py",
        },
        predicted_output={
            "preregistered_bits_per_dash_from_site_prior": EXPECTED_BITS_PER_DASH_PRIOR,
            "anchor_budget_band_bytes": list(ANCHOR_BUDGET_BAND),
        },
        empirical_output={
            "section_bytes_excl_xi": SECTION_BYTES_EXCL_XI,
            "lane_only_naive_bytes": LANE_ONLY_NAIVE_BYTES,
            "naive_anchor_stream_bytes": NAIVE_ANCHOR_STREAM_BYTES,
            "amortization_vs_lane_naive": LANE_ONLY_NAIVE_BYTES / SECTION_BYTES_EXCL_XI,
            "over_budget_factor_vs_1800B": SECTION_BYTES_EXCL_XI / ANCHOR_BUDGET_BAND[1],
            "blink_back_fraction": BLINK_BACK_FRACTION,
            "measured_bits_per_matched_dash": MEASURED_BITS_PER_MATCHED_DASH,
            "dash_centroid_offset_gt2": DASH_CENTROID_GT2,
            "site_prior_gt2_cited": 0.202,
            "prior_code_vs_zlib9_bytes": [PRIOR_CODE_DELTA_BYTES, ZLIB9_DELTA_BYTES],
            "label_space_xor_rates": {
                "persist": XOR_RATE_PERSIST,
                "transport_only": XOR_RATE_TRANSPORT_ONLY,
                "phase_correct": XOR_RATE_PHASE_CORRECT,
            },
            "verdict_scope": "FORMULATION (centroid-level delta, iid prior code, greedy match, "
                             "interp gap-screw); not family-dead",
        },
        residual=MEASURED_BITS_PER_MATCHED_DASH - EXPECTED_BITS_PER_DASH_PRIOR,
        source_artifact=_JSON,
        measurement_method="deterministic numpy codec on the frozen SegNet argmax cache; full "
                           "decode bit-identity self-check; label-space recovery (NOT through-R; "
                           "through-R d_seg A/B OWED)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-measure when (a) a dash-level measured prior replaces the "
                                   "site prior in the code table, (b) per-track context coding "
                                   "lands (iid lost to zlib9 by 20%), or (c) a deterministic "
                                   "persistence-class visibility generator makes events FREE "
                                   "(the 16.6x budget gap)"),
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_cpu_numpy",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Dash-phase carrier rate law: object-domain δ(s)+events amortizes where raster "
              "transport lost (11.3x under lane naive); blink-back 0.787 measured; the "
              "separatrix-site jitter prior does NOT transfer to dash centroids"),
        one_line_summary=(
            "R_dash=R_events+R_δ: 29,958 B n600 (11.3x under lane naive, 16.6x over anchor "
            "budget); blink-back 0.787; site prior 4.53 bits/dash realizes 9.58 (no transfer)."
        ),
        latex_form=(
            r"R_{\mathrm{dash}}=R_{\mathrm{events}}+\sum_{\mathrm{matched}}\ell(\delta s)+\ell(\delta n),"
            r"\quad \frac{R_{\mathrm{lane\,naive}}}{R_{\mathrm{dash}}}=11.3>1,\quad "
            r"p_{\mathrm{blink}}=0.787,\quad "
            r"\mathbb{E}_{\mathrm{site\,prior}}[\ell]=4.53 \ne 9.58=\hat{\ell}_{\mathrm{dash}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.dash_phase_carrier_rate_law_20260717:dash_rate_amortizes"
        ),
        domain_of_validity={
            "vehicle": "frozen SegNet argmax lane-dash trajectory (lstars), 384x512, 0.mkv",
            "codec": "xi-advected world tracks + iid prior-derived Huffman + explicit events "
                     "(dormant horizon 30, q=1px)",
            "recovery": "LABEL-SPACE only (shape-persistence raster substitution); through-R "
                        "d_seg on the c2 checkpoint OWED",
            "measurement_axis": [_AXIS],
            "verdict_scope": "FORMULATION-level; the three reactivation levers are named",
        },
        units_in={"section_bytes": "bytes", "per_frame_naive_bytes": "bytes",
                  "measured_bits": "bits/dash", "preregistered_bits": "bits/dash"},
        units_out={"dash_rate_amortizes": "bool", "site_prior_transfers_to_dash": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "bits_per_dash_prior_vs_measured": MEASURED_BITS_PER_MATCHED_DASH
            - EXPECTED_BITS_PER_DASH_PRIOR
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/levelset_byte_close_and_eval.py",  # --dash-phase-carrier section
            ".omx/research/phase_carrier_425_build_20260717.md",
        ),
        canonical_producers=(
            "src/tac/boundary_math/dash_phase_carrier.py",
            "tools/measure_dash_phase_carrier_n600.py",
            _MEMO,
        ),
        provenance=build_provenance_for_predicted(
            model_id="dash_phase_carrier_rate_law.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="apple_m5_max_cpu_numpy",
        ),
    )


def populate_dash_phase_carrier_rate_law_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins).

    EQUATIONS leg of the #425 STORE-side build (Arm G); DAG leg = the FEED row in the
    sub015 DAG; DSL leg = the byte-close `--dash-phase-carrier` selectable section
    (archive-shape carrier — a byte-close codec mode, not a trainer lever; the DSL's
    never-invent-flags gate forbids a trainer Lever with no trainer flag, mirroring the
    #359 precedent recorded in spec_c2_surgical `phase_residual_carrier_425`)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_dash_phase_carrier_rate_law_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="dash_phase_carrier n600 measured row (Arm G #425 STORE leg, 2026-07-17): "
              "object-domain amortization + blink-back 0.787 + site->dash prior divergence",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_dash_phase_carrier_rate_law_v1",
    "dash_rate_amortizes",
    "populate_dash_phase_carrier_rate_law_equation",
    "site_prior_transfers_to_dash",
]
