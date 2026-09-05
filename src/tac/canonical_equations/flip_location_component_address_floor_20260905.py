# SPDX-License-Identifier: MIT
"""Canonical equation: a CLUSTERING representation of a flip set can only pay when the set
actually clusters -- and on the shipped token stream it does not (ddm_hc2, 2026-09-05).

THE OBJECT.  hc1 split the shipped HPAC token stream losslessly into a binary INDICATOR ("is my
argmax right?") and a 4-way CONDITIONAL ("which class instead").  The indicator is 97.8 % of the
stream, and its "no" branch -- the per-site cost of saying WHERE the flips are -- had never been
decomposed.  ddm_hc2 asked whether any representation of the flip LOCATION SET (component code,
boundary offsets) beats per-site indicator coding by >= 5,000 B.

THE MEASURED INSTANCE (all 600 pairs, exact fs2 body; coder rows byte-identical to the shipped
stream 113,411 B sha 5601d6fd..., re-verified inside the shipped archive at member offset 66,512):

  indicator 110,909.07 B  =  yes-branch 34,642.82 B  +  no-branch 76,266.24 B
  conditional 2,501.80 B;  stream ideal 113,410.87 B (mc1 recorded 113,410.86 B)
  flips 227,555 of 117,964,800 positions; live (float32-unsaturated) 50,009,121; saturated flips 0
  8-connected components 172,193 (mean size 1.3215, median 1, p90 2, p99 5, MAX 21)
  4-connected components 192,719 (mean 1.1808, max 17)
  components with >= 16 sites: 9 (0.0052 % of components), 160 sites (0.070 %), 52.05 B
  flips whose token class sits at Chebyshev distance exactly 1 in the mixer argmax field: 99.03 %

THE ADDRESS-COUNT RATIO (the reusable quantity).  Any component representation pays one address
event per COMPONENT where the per-site incumbent pays one per SITE.  Its whole structural budget
is therefore bounded by the address-count ratio

    rho = components / sites = 172,193 / 227,555 = 0.7567  ->  24.33 % fewer address events,

and every non-seed site it absorbs must be bought back with a shape code.  When rho is close to 1
the representation is renaming the problem, not reducing it.  This is the fourth independent
instance of the address law ([[perfect-localization-is-worthless-the-address-is-the-tax]]): the
error is perfectly localized, so naming it costs what it holds.

THE BOUNDARY-OFFSET COROLLARY.  99.03 % of flips ARE one-pixel boundary moves and 99.45 % of them
lie in the D=1 band (2.61 % of the field), so the "flips are signed offsets of the mixer's
boundary" premise is TRUE.  It still cannot pay, because the mixer has already localised that
band: the incumbent spends only 5,338.2 B of its 110,909.07 B outside it.  A band restriction that
the model already performs is free information twice over -- the offset field's support IS the
band, and deciding which band position carries a nonzero offset IS the indicator.

VERDICT.  CEILING-REFUSED; the charter's falsifier fired.  hc1's wrong half is CLOSED at
FAMILY scope for location-set representations on this object.  Axis
``[macOS-CPU advisory / scorer-free EXACT byte measurement]`` for the control;
``[model-ledger code length on the coder's own rows; REFUSAL-ONLY]`` for the ceilings.
NON-PROMOTABLE; moves no pointer.

Producer: ``experiments/ddm_hc2_wrong_half_flip_location_decomposition.py`` (stages control /
features / ceiling) via ``.omx/research/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md``.
Consumers: every charter that proposes to re-address, cluster, contour, run-length, chain-code or
boundary-offset a sparse error/flip set on this body; the sister laws ``ddm_mi1`` (the conditioning
ledger this instrument extends), ``ddm_mc1`` (the same instrument on a temporal plane) and
``ddm_ra3`` (the per-flip correction carrier).
"""

from __future__ import annotations

from collections.abc import Mapping

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

EQUATION_ID = "flip_location_component_address_floor_v1"

_UTC = "2026-09-05T14:00:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
_CEILING_AXIS = "[model-ledger code length on the coder's own rows; REFUSAL-ONLY]"
_LEDGER = ".omx/research/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md"
_CHARTER = ".omx/research/charters/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md"
_PRODUCER = "experiments/ddm_hc2_wrong_half_flip_location_decomposition.py"

# --- MEASURED (ddm_hc2, 2026-09-05, all 600 pairs of the fs2 body) ------------------------
STREAM_BYTES = 113_411
STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
ROWS_SHA256 = "35ec67ca932112cfe11be31391ee784cb577bc6c7df1e0563f49f841fded67bf"
ARCHIVE_SHA256 = "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6"
STREAM_OFFSET_IN_ARCHIVE_MEMBER = 66_512

PAIRS = 600
POSITIONS = 117_964_800
LIVE_POSITIONS = 50_009_121
FLIPS = 227_555
SATURATED_FLIPS = 0

STREAM_BYTES_IDEAL = 113_410.87022613022
INDICATOR_BYTES = 110_909.06536781874
NO_BRANCH_BYTES = 76_266.24448721769
YES_BRANCH_BYTES = 34_642.82088060099
# The ceiling stage re-accumulates the indicator from a float32 copy of the per-position costs,
# so its comparison base is 0.031 B below the float64 decomposition above.  Every (a)/(b) delta
# below is against THIS number, and the two agree to 2.7e-7 relative.
CEILING_COMPARISON_INDICATOR_BYTES = 110_909.03490077237
CONDITIONAL_BYTES = 2_501.8048583114823
BITS_PER_FLIP_NO_BRANCH = 2.681241703754

COMPONENTS_8CONN = 172_193
COMPONENTS_4CONN = 192_719
COMPONENT_MEAN_SIZE_8CONN = 1.321511327405876
COMPONENT_MEDIAN_SIZE_8CONN = 1.0
COMPONENT_MAX_SIZE_8CONN = 21
COMPONENT_MAX_SIZE_4CONN = 17

# size bucket -> (components, sites, incumbent no-branch bytes on those sites), 8-connectivity
COMPONENT_SIZE_BUCKETS_8CONN: Mapping[str, tuple[int, int, float]] = {
    "1": (139_640, 139_640, 46_717.62),
    "2-3": (27_248, 61_655, 20_961.13),
    "4-7": (4_981, 23_152, 7_614.16),
    "8-15": (315, 2_948, 921.28),
    "16-31": (9, 160, 52.05),
}
GE16_COMPONENTS = 9
GE16_SITES = 160
GE16_BYTES = 52.05284
GE16_COMPONENT_SHARE = 5.227e-05

# Boundary-offset diagnostics: Chebyshev distance from a flip to the nearest argmax pixel of the
# class the flip codes.  Distance 1 == the flip IS a one-pixel boundary move.
FLIPS_AT_TOKEN_CLASS_DISTANCE_1 = 225_339
BAND_POSITIONS: Mapping[int, int] = {1: 3_077_386, 2: 5_562_417, 3: 7_860_580}
BAND_FLIPS: Mapping[int, int] = {1: 226_297, 2: 226_967, 3: 227_213}
BAND_OUT_OF_BAND_INDICATOR_BYTES: Mapping[int, float] = {
    1: 5_338.20,
    2: 3_193.22,
    3: 2_311.51,
}

REFUSE_BELOW_BYTES = 5_000.0  # the charter's pre-registered bar
PRIOR_LAW_PREDICTED_SAVING_BYTES = 11_000.0  # 15-30 % of the 76,600 B no-branch, lower edge

# --- Ceiling results (MEASURED; every number held-out, pair-level two-fold, seeds
# 20260905/777/31337, reported at the value MOST FAVOURABLE to the alternative) --------------
CEILING_A1_RASTER_GAP_BYTES = 231_091.54  # unconditioned raster-gap seed code + shapes
CEILING_A2_MIXER_CONDITIONED_BYTES = 138_873.45  # KT-per-q64 seed field + shapes
CEILING_A2B_BETA_PER_CELL_BYTES = 128_683.87  # beta-per-cell (q32 x d_other) seed field + shapes
CEILING_A_SEED_FIELD_BYTES = 88_853.80  # the a2b seed field alone
CEILING_A_SHAPE_BYTES = 39_830.07  # the cross-fitted shape dictionary alone
CEILING_A_DISTINCT_SHAPES = 1_553
CEILING_B_BEST_BYTES = 106_555.86  # D=3, geometry x q32, ACAUSAL band geometry
CEILING_B_BEST_LABEL = "D=3 geometry_x_q32 (band and distance read the FULL current-frame argmax)"
CEILING_B_GEOMETRY_ONLY_BEST_BYTES = 147_418.28  # D=3, no mixer probability
CLUSTERING_GAIN_BYTES = -23_754.00  # component cost - seed cost - shape cost, summed
GE16_SHARE_OF_CLUSTERING_GAIN = 0.0023103369084278335
CLUSTERING_GAIN_POSITIVE_COMPONENTS = 2_326  # of 172,193

# Family bound (mi1's beta-per-cell on the indicator), held-out, 3-seed minimum.
FAMILY_BOUND_BEST_CAUSAL_BYTES = 23.82  # pat4: the 4 raster-causal neighbours' flip bits
FAMILY_BOUND_CAUSAL_CELLS: Mapping[str, float] = {
    "none": 2.37,
    "q32": -32.64,
    "pat4": 23.82,
    "pat8": -3.39,
    "q32_x_pat4": -213.91,
    "q32_x_pat8": -403.70,
}
# ACAUSAL diagnostic: the current-frame argmax boundary geometry the receiver cannot compute in
# advance.  Reported because it is the only cell that clears the bar -- and it is not a candidate.
FAMILY_BOUND_ACAUSAL_CELLS: Mapping[str, float] = {"d_other": 4_147.72, "q32_x_dother": 5_823.45}
FAMILY_BOUND_INSTRUMENT_NOISE_FLOOR_BYTES = 2.37
# mi1's CAUSAL boundary feature (buckets of frame t-1), already in the shipped stack.
MI1_CAUSAL_BOUNDARY_D_HELD_OUT_BYTES = 5.27


def address_count_ratio(components: int = COMPONENTS_8CONN, sites: int = FLIPS) -> float:
    """rho = address events a component code must pay / address events the per-site code pays."""
    return float(components) / float(sites)


def clustering_headroom_fraction(components: int = COMPONENTS_8CONN, sites: int = FLIPS) -> float:
    """The fraction of address events a component representation can possibly remove (1 - rho)."""
    return 1.0 - address_count_ratio(components, sites)


def component_representation_headroom_bytes(
    *, components: int, sites: int, incumbent_bytes: float
) -> float:
    """The OPTIMISTIC upper bound on what any component representation can save: the incumbent's
    cost on the sites the representation absorbs (every site that is not its component's seed),
    priced at the incumbent's mean per-site cost, with the shape code taken as FREE.
    """
    absorbed = max(int(sites) - int(components), 0)
    return absorbed * (float(incumbent_bytes) / float(sites))


def component_representation_can_pay(
    *,
    components: int,
    sites: int,
    incumbent_bytes: float,
    required_saving_bytes: float = REFUSE_BELOW_BYTES,
) -> bool:
    """THE GATE, and it is only a NECESSARY condition -- read the boundary before quoting it.

    A component representation pays one address event per component where the incumbent pays one
    per site, so it can never save more than ``component_representation_headroom_bytes``.  When
    that bound is below the bar, the family is refused BEFORE anything is designed.

    On the hc2 body the bound is 18,554 B -- ABOVE the 5,000 B bar -- so the geometry alone did
    NOT refuse this arm, and the refusal came from the MEASURED shape code instead (the code that
    buys the absorbed sites back costs more than the 0.335 B/site it replaces).  Do not quote this
    gate as if the address-count ratio settled the question here; it only says a set with
    components == sites is hopeless a priori.
    """
    return (
        component_representation_headroom_bytes(
            components=components, sites=sites, incumbent_bytes=incumbent_bytes
        )
        >= float(required_saving_bytes)
    )


def boundary_band_is_already_localised(
    *, out_of_band_indicator_bytes: float, indicator_bytes: float, threshold: float = 0.10
) -> bool:
    """A boundary-offset representation buys the band restriction for free only if the incumbent
    is still spending a MATERIAL share of its indicator outside the band.  When that share is
    below ``threshold`` the model has already localised the boundary and the offset field's
    support carries no new information."""
    return float(out_of_band_indicator_bytes) / float(indicator_bytes) >= float(threshold)


def ceiling_refused(held_out_bytes_saved: float, refuse_below: float = REFUSE_BELOW_BYTES) -> bool:
    """The charter's pre-registered refusal: best ideal saving below the bar."""
    return float(held_out_bytes_saved) < float(refuse_below)


def _anchor_component_geometry() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="hc2_flip_set_does_not_cluster_address_count_ratio_0p757_20260905",
        measurement_utc="2026-09-05T13:10:00Z",
        inputs={
            "rows_sha256": ROWS_SHA256,
            "field_sha256": FIELD_SHA256,
            "pairs": PAIRS,
            "flip_definition": (
                "the coded token's probability is strictly below the coding row maximum "
                "(tie-safe; hc1's split is exact under it)"
            ),
            "labelling": "scipy.ndimage.label on the per-pair 384x512 flip mask, 4- and 8-connectivity",
            "producer": f"{_PRODUCER} --stage features",
        },
        predicted_output={
            "prior_law": (
                "flips are edge-concentrated, so they form connected components whose "
                "component-level description costs well below the per-site sum; the saving "
                "concentrates in the <= 20 % of components with >= 16 sites"
            ),
            "falsifier": "components are overwhelmingly singletons, so the address-count ratio is near 1",
        },
        empirical_output={
            "components_8conn": COMPONENTS_8CONN,
            "components_4conn": COMPONENTS_4CONN,
            "sites": FLIPS,
            "address_count_ratio": address_count_ratio(),
            "mean_size_8conn": COMPONENT_MEAN_SIZE_8CONN,
            "median_size_8conn": COMPONENT_MEDIAN_SIZE_8CONN,
            "max_size_8conn": COMPONENT_MAX_SIZE_8CONN,
            "size_buckets_8conn": {k: list(v) for k, v in COMPONENT_SIZE_BUCKETS_8CONN.items()},
            "ge16_components": GE16_COMPONENTS,
            "ge16_component_share": GE16_COMPONENT_SHARE,
            "ge16_sites": GE16_SITES,
            "ge16_bytes": GE16_BYTES,
            "reading": (
                "81.1 % of components are singletons; only 9 components in the whole body have "
                ">= 16 sites, holding 0.070 % of the flip sites and 52.05 B of 76,266.24 B -- the "
                "prior law's 20 %-of-components premise is wrong by ~3,800x"
            ),
        },
        # The prior law asked for >= 20 % of components at size >= 16; measured 0.0052 %.
        residual=0.20 - GE16_COMPONENT_SHARE,
        source_artifact=_LEDGER,
        measurement_method=(
            "exact connected-component labelling of the flip mask on all 600 pairs, with the "
            "incumbent's per-site indicator cost summed over each component from the coder's rows"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a body whose flip set has an address-count ratio materially below 1 "
                "(components << sites), i.e. genuinely clustered error"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_boundary_band() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="hc2_flips_are_one_pixel_boundary_moves_but_the_band_is_already_priced_20260905",
        measurement_utc="2026-09-05T13:15:00Z",
        inputs={
            "geometry": (
                "per pair, the Chebyshev distance transform of each class region of the mixer's "
                "argmax field; a flip's offset distance is the distance to the nearest argmax "
                "pixel of the class it codes"
            ),
            "band_D": sorted(BAND_POSITIONS),
            "producer": f"{_PRODUCER} --stage features",
        },
        predicted_output={
            "prior_law": (
                "if flips are signed offsets of the mixer's class boundary, restricting the "
                "indicator to the derivable boundary band is free side information and saves bytes"
            ),
            "falsifier": (
                "the incumbent already spends almost nothing outside the band, so the band "
                "restriction is information the model has already applied"
            ),
        },
        empirical_output={
            "flips_at_token_class_distance_1": FLIPS_AT_TOKEN_CLASS_DISTANCE_1,
            "share_one_pixel_boundary_moves": FLIPS_AT_TOKEN_CLASS_DISTANCE_1 / FLIPS,
            "band_positions": {str(k): v for k, v in BAND_POSITIONS.items()},
            "band_flips": {str(k): v for k, v in BAND_FLIPS.items()},
            "out_of_band_indicator_bytes": {
                str(k): v for k, v in BAND_OUT_OF_BAND_INDICATOR_BYTES.items()
            },
            "out_of_band_share_D1": BAND_OUT_OF_BAND_INDICATOR_BYTES[1] / INDICATOR_BYTES,
            "reading": (
                "the premise is TRUE (99.03 % of flips are one-pixel boundary moves, 99.45 % lie "
                "in the D=1 band, which is 2.61 % of the field) and it still cannot pay: the "
                "incumbent already spends only 5,338.2 B of 110,909.07 B (4.81 %) outside that band"
            ),
        },
        residual=BAND_OUT_OF_BAND_INDICATOR_BYTES[1] / INDICATOR_BYTES,
        source_artifact=_LEDGER,
        measurement_method=(
            "exact chessboard distance transforms of the argmax field per class, joined to the "
            "coder's per-position indicator cost"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a coder whose indicator still spends >= 10 % of its bytes outside the derivable "
                "boundary band -- i.e. one that has NOT already localised the boundary"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_ceiling() -> EmpiricalAnchor:
    # Best saving any representation achieved, against the comparator the ceiling stage used.
    best = max(
        CEILING_COMPARISON_INDICATOR_BYTES - CEILING_A2B_BETA_PER_CELL_BYTES,
        CEILING_COMPARISON_INDICATOR_BYTES - CEILING_B_BEST_BYTES,
        FAMILY_BOUND_BEST_CAUSAL_BYTES,
    )
    return EmpiricalAnchor(
        anchor_id="hc2_no_location_set_representation_clears_the_bar_20260905",
        measurement_utc="2026-09-05T13:40:00Z",
        inputs={
            "rows": (
                "every RC64 coding row of the shipped encoder on the fs2 tree; the retained "
                "control stream is byte-identical to the shipped token block"
            ),
            "stream_bytes": STREAM_BYTES,
            "stream_sha256": STREAM_SHA256,
            "archive_sha256": ARCHIVE_SHA256,
            "stream_offset_in_archive_member_p": STREAM_OFFSET_IN_ARCHIVE_MEMBER,
            "instrument": (
                "pair-level two-fold cross-fit, seeds 20260905/777/31337; component code = seed "
                "field + cross-fitted shape dictionary with escape; boundary-offset = derivable "
                "band + cross-fitted in-band model + incumbent out of band; family bound = mi1's "
                "q' = sigma(logit(1-pmax) + beta_cell) with the causal-neighbourhood flip pattern"
            ),
            "producer": f"{_PRODUCER} --stage ceiling",
        },
        predicted_output={
            "prior_law_saving_bytes": PRIOR_LAW_PREDICTED_SAVING_BYTES,
            "refuse_below_bytes": REFUSE_BELOW_BYTES,
            "falsifier": "no representation saves 5,000 B against the incumbent",
        },
        empirical_output={
            "incumbent_no_branch_bytes": NO_BRANCH_BYTES,
            "incumbent_full_indicator_bytes": INDICATOR_BYTES,
            "incumbent_ceiling_comparison_bytes": CEILING_COMPARISON_INDICATOR_BYTES,
            "best_saving_bytes": best,
            "a1_raster_gap_bytes": CEILING_A1_RASTER_GAP_BYTES,
            "a2_mixer_conditioned_bytes": CEILING_A2_MIXER_CONDITIONED_BYTES,
            "a2b_beta_per_cell_bytes": CEILING_A2B_BETA_PER_CELL_BYTES,
            "a_seed_field_bytes": CEILING_A_SEED_FIELD_BYTES,
            "a_shape_component_bytes": CEILING_A_SHAPE_BYTES,
            "a_distinct_shapes": CEILING_A_DISTINCT_SHAPES,
            "b_best_bytes": CEILING_B_BEST_BYTES,
            "b_best_label": CEILING_B_BEST_LABEL,
            "b_geometry_only_best_bytes": CEILING_B_GEOMETRY_ONLY_BEST_BYTES,
            "clustering_gain_bytes": CLUSTERING_GAIN_BYTES,
            "ge16_share_of_clustering_gain": GE16_SHARE_OF_CLUSTERING_GAIN,
            "clustering_gain_positive_components": CLUSTERING_GAIN_POSITIVE_COMPONENTS,
            "family_bound_causal_cells": dict(FAMILY_BOUND_CAUSAL_CELLS),
            "family_bound_acausal_cells_diagnostic": dict(FAMILY_BOUND_ACAUSAL_CELLS),
            "family_bound_best_causal_bytes": FAMILY_BOUND_BEST_CAUSAL_BYTES,
            "instrument_noise_floor_bytes": FAMILY_BOUND_INSTRUMENT_NOISE_FLOOR_BYTES,
            "mi1_causal_boundary_d_held_out_bytes": MI1_CAUSAL_BOUNDARY_D_HELD_OUT_BYTES,
            "typed_verdict": "CEILING-REFUSED",
            "reading": (
                "measured against the charter's comparator (the per-site sum on the flip sites, "
                "76,266.24 B) EVERY representation loses by >= 30,290 B; measured against the "
                "full indicator it must actually replace (110,909.03 B) the best -- an ACAUSAL "
                "boundary-offset band -- saves 4,353.17 B, still under the 5,000 B bar; the "
                "CAUSAL clustering family bound is +23.82 B, 210x under it"
            ),
        },
        residual=PRIOR_LAW_PREDICTED_SAVING_BYTES - best,
        source_artifact=_LEDGER,
        measurement_method=(
            "held-out ideal code length on the coder's own rows under models fitted on the other "
            "pair fold; the incumbent is the rows' own per-site indicator cost"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a location-set representation whose held-out ideal code length beats the "
                "incumbent indicator by >= 5,000 B on this body"
            ),
            measurement_axis=_CEILING_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_flip_location_component_address_floor_v1() -> CanonicalEquation:
    """Build the flip-location address-floor equation (ddm_hc2, 2026-09-05)."""
    geometry = _anchor_component_geometry()
    band = _anchor_boundary_band()
    ceiling = _anchor_ceiling()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Flip-location address floor -- a clustering or boundary-offset representation of a "
            "sparse flip set pays only when the set clusters, and this one does not"
        ),
        one_line_summary=(
            "hc2: 172,193 components hold 227,555 flip sites (rho 0.757, 81 % singletons); best "
            "location-set code 128,684 B vs a 76,266 B incumbent; causal clustering bound +23.82 B"
        ),
        latex_form=(
            r"\rho=\frac{|\mathcal{C}|}{|S|}=\frac{172{,}193}{227{,}555}=0.7567;\quad"
            r"\Delta B_{\max}\le (|S|-|\mathcal{C}|)\cdot \bar b_{\text{site}}"
            r"\ \text{with}\ \bar b_{\text{site}}=\tfrac{76{,}266.24}{227{,}555}\ \text{B};\quad"
            r"\text{admit}\iff \Delta B_{\max}\ge 5{,}000\ \text{B}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.flip_location_component_address_floor_20260905"
            ":component_representation_can_pay"
        ),
        domain_of_validity={
            "included": [
                "any re-addressing of a sparse error/flip set on the shipped token stream: "
                "connected components, run-length, chain code, contour, boundary offsets",
                "the shipped HPAC receptive field and its coding rows on the fs2 body",
                "sparse sets whose components can be labelled on the 384x512 per-pair field",
            ],
            "excluded": [
                "use as a d_seg / d_pose / rate lever, a score, or a promotion claim",
                "bodies whose error set genuinely clusters (rho materially below 1): the gate is "
                "the ratio, not this body's value",
                "moves that pay the address in MODEL bits rather than naming targets -- the "
                "escape named by the address law, untouched by this arm",
            ],
            "measurement_axis": [_AXIS, _CEILING_AXIS],
            "result_type": "LOCATION-SET REPRESENTATION closure; NON-PROMOTABLE; moves no pointer",
            "sister_laws": [
                "ddm_hc1 -- the lossless indicator/conditional split whose wrong half this decomposes",
                "ddm_mi1 -- the conditioning ledger and the beta-per-cell instrument reused here",
                "ddm_mc1 -- the same instrument on a temporal plane (CEILING-REFUSED at +159.60 B)",
                "ddm_ra3 -- the per-flip correction carrier closed by the same address arithmetic",
                "ddm_df1 -- the right half's address floor at 3.15x the prize",
            ],
            "known_boundary": (
                "one body, one coder; the ceilings are held-out ideal code lengths, generous to "
                "the alternative (raster decode order, acausal band geometry) so a refusal is "
                "stronger than a bare measurement, and an admission would have needed a receiver"
            ),
            "verdict_scope": (
                "family (location-set representations of the flip set) on the shipped object, "
                "with the address-count ratio as the transferable gate"
            ),
        },
        units_in={
            "components": "count",
            "sites": "count",
            "incumbent_bytes": "bytes",
            "required_saving_bytes": "bytes",
            "out_of_band_indicator_bytes": "bytes",
            "indicator_bytes": "bytes",
            "held_out_bytes_saved": "bytes",
        },
        units_out={
            "address_count_ratio": "dimensionless_fraction",
            "clustering_headroom_fraction": "dimensionless_fraction",
            "component_representation_can_pay": "bool",
            "boundary_band_is_already_localised": "bool",
            "ceiling_refused": "bool",
        },
        empirical_anchors=(geometry, band, ceiling),
        predicted_vs_empirical_residual={
            geometry.anchor_id: geometry.residual,
            band.anchor_id: band.residual,
            ceiling.anchor_id: ceiling.residual,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER, _CHARTER),
        canonical_producers=(_PRODUCER, _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="flip_location_component_address_floor.v1",
            inputs_sha256=ROWS_SHA256,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
