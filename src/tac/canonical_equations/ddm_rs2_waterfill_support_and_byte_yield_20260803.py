"""ddm_rs2 / ddm_br1 — three measured laws of the token-lattice drop surface.

These are the EQUATIONS leg owed by ``ddm_br1`` (which landed ``[no-triality]``) plus the
one law ``ddm_rs2`` added on top of it.  All three govern the same object: the
sensitivity-weighted reverse waterfill on the DR7/TR1 token lattice (task #766,
``experiments/ddm_wr1_reverse_waterfill.py``), which decides WHICH lattice cells to drop
and therefore sets the rate/seg exchange rate of every future drop rung.

L1  ``lattice_cell_drop_pricing_support_v1``   (rs2, NEW)
    A cell drop must be priced over the DECODER'S RECEPTIVE FIELD, not over the cell's own
    tile.  MEASURED on the live cx1 receiver: dropping lattice cell (13,17) perturbs
    SegNet's input plane over rows [174,257] x cols [239,320] = 84 x 82 = 6,888 box px
    (6,192 actually nonzero), against the 16x16 = 256 px tile that ``ddm_wr1:89`` attributes
    flips to.  Support ratio 24.19x (nonzero-px basis).  A tile-scoped safety key therefore
    integrates the risk over ~4% of the region the drop actually disturbs.

L2  ``token_lattice_byte_marginal_flat_uncorrelated_v1``   (br1)
    The per-unit byte marginal of a drop is ~FLAT and carries almost no ranking information:
    min -58 B, median 196 B, mean 211 B, max 472 B, and TWO live units have NEGATIVE
    marginals (dropping them makes the stream bigger).  Ranking a reverse waterfill by bytes
    is therefore ranking noise; the ordering must come from the damage side.  Group drops are
    mildly SUPERadditive (1.0206x on matched units), so greedy pricing slightly understates
    group yield.

L3  ``live_vs_dead_symbol_entropy_decomposition_v1``   (br1)
    Entropy is a property of the (stream, model) pair.  On the shipped lattice the coder
    appears to beat an order-0 model by 1.4776x, but 50.3% of the units are temporally DEAD
    and their zeros are LZ-copied for free.  On LIVE symbols only the advantage is 1.1093x --
    the live stream is 9.85% below its own order-0 bound.  Consequence, proved before it was
    built: an explicit live-unit support map costs ~70 B and saves ~0 B.

    CORRECTION carried forward: ``ddm_br1`` reports the all-symbol ratio as 1.4834x, but its
    own byte figures give 504,291/341,295 = 1.4776x.  The 1.1093x live ratio reproduces
    exactly.  The conclusion is unchanged; only the headline ratio is 0.4% lower.

Registration note (same discipline as ``ddm_b2b_rowband_flip_mass_20260731``): this module
DEFINES the equations + their measured anchors.  The locked-registry ``populate_*`` append and
the ``__init__`` export are the named landing follow-up, kept out of this arm because a sister
session holds uncommitted edits in ``src/tac/canonical_equations/__init__.py`` and folding them
into this commit would be the absorption-pattern bug class.  Every ``build_*`` below is
importable and self-validating now.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Advisory; ``score_claim=False``.  L1 is a
geometric FACT about the decoder, L2/L3 are byte FACTS about the encoder; none is a score claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

RS2_ARTIFACT = ".omx/research/ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md"
BR1_ARTIFACT = ".omx/research/ddm_br1_basis_race_and_drop_surface_20260803.md"

SUPPORT_EQUATION_ID = "lattice_cell_drop_pricing_support_v1"
BYTE_YIELD_EQUATION_ID = "token_lattice_byte_marginal_flat_uncorrelated_v1"
ENTROPY_SPLIT_EQUATION_ID = "live_vs_dead_symbol_entropy_decomposition_v1"

#: MEASURED (rs2 pilot, live cx1 receiver, cell (13,17), pair 136): the drop's footprint in
#: SegNet's own input plane, as (rows, cols) of the bounding box and the nonzero pixel count.
MEASURED_RF_BBOX_ROWS = (174, 257)
MEASURED_RF_BBOX_COLS = (239, 320)
MEASURED_RF_NONZERO_PX = 6192
#: The support ``ddm_wr1_reverse_waterfill.py:89`` actually integrates the risk over.
WR1_TILE_PX = 256

#: MEASURED (br1, exact individual marginals over the live range of the shipped lattice),
#: at the UNIT grain (cell x channel; 1,528 live of 3,072).
BYTE_MARGINAL_B = {"min": -58.0, "median": 196.0, "mean": 211.0, "max": 472.0}
BYTE_MARGINAL_N_NEGATIVE = 2
DROP_SUPERADDITIVITY = 1.0206

#: MEASURED (rs2, 384 exact re-encodes of the real lattice) at the CELL grain -- the grain the
#: waterfill actually operates on.  The unit-grain result does NOT transfer: at the cell grain
#: there are NO negative marginals and the spread is 4.98x.  Damage correlation stays weak, so
#: the ordering is still damage-dominated, but for a different reason than at the unit grain.
CELL_BYTE_MARGINAL_B = {"min": 248.0, "median": 861.0, "mean": 857.7083333333334, "max": 1234.0}
CELL_BYTE_MARGINAL_N_NEGATIVE = 0
#: Spearman of the REAL per-cell byte marginal against wr1's byte proxy ``residual_mass``.
CELL_MARGINAL_RHO_VS_WR1_RESIDUAL_MASS = 0.512801363127734

#: MEASURED (br1 ``br1_refine.py``, exact, on the shipped 1,843,200-symbol lattice).
ALL_SYMBOLS = 1_843_200
ALL_SHIPPED_B = 341_295
ALL_ORDER0_B = 504_291
LIVE_SYMBOLS = 916_800
LIVE_SHIPPED_B = 339_956
LIVE_ORDER0_B = 377_100


# --------------------------------------------------------------------------- L1


def drop_support_px(
    rf_bbox_rows: tuple[int, int] = MEASURED_RF_BBOX_ROWS,
    rf_bbox_cols: tuple[int, int] = MEASURED_RF_BBOX_COLS,
) -> int:
    """Bounding-box area, in scorer-plane pixels, of one lattice cell's drop footprint."""
    r0, r1 = rf_bbox_rows
    c0, c1 = rf_bbox_cols
    if r1 < r0 or c1 < c0:
        raise ValueError("receptive-field bbox must be non-empty and ordered")
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def support_mispricing(
    *,
    key_support_px: int = WR1_TILE_PX,
    measured_support_px: int = MEASURED_RF_NONZERO_PX,
) -> dict[str, float | bool]:
    """How badly a damage key priced on ``key_support_px`` covers the real drop footprint.

    Returns the ratio, the fraction of the real footprint the key sees, and a fail-closed
    ``key_support_is_sound`` flag.  A key whose support is smaller than the measured footprint
    UNDERSTATES the risk of every drop and cannot certify a cell "safe".
    """
    if key_support_px <= 0 or measured_support_px <= 0:
        raise ValueError("supports must be positive pixel counts")
    ratio = measured_support_px / key_support_px
    return {
        "support_ratio_measured_over_key": ratio,
        "fraction_of_real_footprint_seen_by_key": min(1.0, 1.0 / ratio),
        "key_support_is_sound": bool(key_support_px >= measured_support_px),
        "risk_understated": bool(key_support_px < measured_support_px),
    }


# --------------------------------------------------------------------------- L2


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    xs, ys = list(a), list(b)
    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("spearman needs two equal-length sequences of length >= 2")

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        for pos, i in enumerate(order):
            out[i] = float(pos)
        return out

    ra, rb = ranks(xs), ranks(ys)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def byte_side_is_rankable(
    byte_marginals: Sequence[float],
    damage_proxy: Sequence[float],
    *,
    min_spread: float = 3.0,
    min_abs_rho: float = 0.3,
) -> dict[str, float | bool | int]:
    """Decide whether the BYTE side of a drop surface carries usable ranking information.

    A reverse waterfill maximises ``bytes_saved / damage_caused``.  If the byte marginals are
    flat and uncorrelated with damage, the ratio's variation is entirely the damage term and
    ranking by bytes is ranking noise.  Fail-closed: negative marginals also break the greedy
    monotonicity assumption, so they are reported explicitly.
    """
    vals = [float(v) for v in byte_marginals]
    if not vals:
        raise ValueError("byte_marginals must be non-empty")
    positive = [v for v in vals if v > 0]
    spread = (max(positive) / min(positive)) if len(positive) >= 2 else float("inf")
    rho = _spearman(vals, damage_proxy)
    n_neg = sum(1 for v in vals if v < 0)
    return {
        "spread_max_over_min_positive": spread,
        "spearman_bytes_vs_damage": rho,
        "n_negative_marginals": n_neg,
        "greedy_monotonicity_holds": bool(n_neg == 0),
        "byte_side_is_rankable": bool(spread >= min_spread and abs(rho) >= min_abs_rho),
    }


# --------------------------------------------------------------------------- L3


def coder_advantage_split(
    *,
    all_shipped_b: int = ALL_SHIPPED_B,
    all_order0_b: int = ALL_ORDER0_B,
    live_shipped_b: int = LIVE_SHIPPED_B,
    live_order0_b: int = LIVE_ORDER0_B,
) -> dict[str, float]:
    """Split a coder's apparent advantage into the LIVE part and the free-riding DEAD part."""
    for v in (all_shipped_b, all_order0_b, live_shipped_b, live_order0_b):
        if v <= 0:
            raise ValueError("byte counts must be positive")
    apparent = all_order0_b / all_shipped_b
    live = live_order0_b / live_shipped_b
    return {
        "apparent_advantage_over_order0": apparent,
        "live_symbol_advantage_over_order0": live,
        "free_riding_inflation": apparent / live,
        "live_gap_to_own_order0_entropy": live - 1.0,
        "dead_payload_bytes": all_shipped_b - live_shipped_b,
    }


# --------------------------------------------------------------------------- builders


def _prov(sidecar: str, reactivation: str):
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=reactivation,
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_macos_cpu_numpy",
    )


def build_lattice_cell_drop_pricing_support_v1() -> CanonicalEquation:
    """L1 — a cell drop is priced over the decoder receptive field, not over its own tile."""
    prov = _prov(
        RS2_ARTIFACT,
        "re-measure the drop footprint whenever the decoder's upsample depth, kernel size or "
        "latent grid changes; the support ratio is a property of the renderer, not of the clip",
    )
    anchor = EmpiricalAnchor(
        anchor_id="rs2_cell_drop_receptive_field_cx1_20260803",
        measurement_utc="2026-08-03T00:00:00Z",
        inputs={
            "lattice": [600, 24, 32, 4],
            "cell": [13, 17],
            "pair": 136,
            "receiver": "inflate_runner_v4d.Decoder on v4d_composed_cx1_pj2ix2_archive.zip",
            "scorer_plane": [384, 512],
            "key_support_px": WR1_TILE_PX,
        },
        predicted_output={
            "footprint_bounded_by_decoder_receptive_field": True,
            "tile_support_is_sound": False,
        },
        empirical_output={
            "rf_bbox_rows": list(MEASURED_RF_BBOX_ROWS),
            "rf_bbox_cols": list(MEASURED_RF_BBOX_COLS),
            "rf_nonzero_px": MEASURED_RF_NONZERO_PX,
            "support_ratio": MEASURED_RF_NONZERO_PX / WR1_TILE_PX,
        },
        residual=0.0,
        source_artifact=RS2_ARTIFACT,
        measurement_method=(
            "drop one lattice cell to its temporal mode, re-render frame_1 through the real "
            "receiver, apply the frozen scorer downsample D (bilinear, align_corners=False, "
            "antialias=False), and take the bounding box of the nonzero difference"
        ),
        provenance=prov,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    return CanonicalEquation(
        equation_id=SUPPORT_EQUATION_ID,
        name="Lattice-cell drop pricing support",
        one_line_summary=(
            "One lattice cell's drop perturbs 6,192 scorer pixels, 24.2x the 16x16 tile a "
            "tile-scoped damage key integrates over, so such a key understates every drop's "
            "risk and cannot certify a cell safe."
        ),
        latex_form=(
            r"\mathrm{supp}(\partial f_1/\partial z_{r,c})=\mathrm{RF}(\mathrm{dec})"
            r"\supsetneq T_{16\times16},\quad |\mathrm{RF}|/|T|=24.2"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rs2_waterfill_support_and_byte_yield_20260803"
            ":support_mispricing"
        ),
        domain_of_validity={
            "included": [
                "task #766 / ddm_wr1 reverse-waterfill cell damage keys",
                "any per-cell safety or sensitivity key on the TR1 token lattice",
            ],
            "excluded": [
                "gradient keys computed by backprop through the decoder (ddm_gr1's cell_gsum), "
                "which carry the correct support BY CONSTRUCTION",
                "any claim about how many flips a drop actually causes (that is the queued "
                "scorer leg; this equation prices the SUPPORT, not the damage)",
                "score, promotion, or pointer movement",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"key_support_px": "scorer-plane pixels", "measured_support_px": "scorer-plane pixels"},
        units_out={"support_ratio_measured_over_key": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"support_ratio": 0.0},
        last_calibration_utc="2026-08-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("experiments.ddm_wr1_reverse_waterfill.cell_sensitivity",),
        canonical_producers=("scratchpad.rs2_drive_pilot", "scratchpad.rs2_drive_sweep"),
        provenance=prov,
    )


def build_token_lattice_byte_marginal_flat_uncorrelated_v1() -> CanonicalEquation:
    """L2 — the byte side of the drop surface carries almost no ranking information."""
    prov = _prov(
        BR1_ARTIFACT,
        "re-measure the individual byte marginals whenever the token coder, the layout or the "
        "mode factorisation changes; a coder with different match structure can restore spread",
    )
    anchor = EmpiricalAnchor(
        anchor_id="br1_unit_byte_marginal_flatness_cx1_20260803",
        measurement_utc="2026-08-03T00:00:00Z",
        inputs={
            "lattice": [600, 24, 32, 4],
            "units": 3072,
            "live_units": 1528,
            "encoder": "tac.optimization.ddm_ix2_archive_container.encode_token_frame",
        },
        predicted_output={"byte_side_is_rankable": False},
        empirical_output={
            "byte_marginal_B": dict(BYTE_MARGINAL_B),
            "n_negative_marginals": BYTE_MARGINAL_N_NEGATIVE,
            "group_superadditivity": DROP_SUPERADDITIVITY,
        },
        residual=0.0,
        source_artifact=BR1_ARTIFACT,
        measurement_method=(
            "exact individual byte marginals: re-encode the real lattice through the real "
            "encoder once per live unit and difference against the shipped token member"
        ),
        provenance=prov,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    cell_anchor = EmpiricalAnchor(
        anchor_id="rs2_cell_byte_marginal_cx1_20260803",
        measurement_utc="2026-08-03T00:00:00Z",
        inputs={
            "grain": "cell (4 channels dropped together)",
            "live_cells": 384,
            "encoder": "tac.optimization.ddm_ix2_archive_container.encode_token_frame",
            "wall_seconds": 388.8,
        },
        predicted_output={"unit_grain_result_transfers_to_cell_grain": False},
        empirical_output={
            "cell_byte_marginal_B": dict(CELL_BYTE_MARGINAL_B),
            "n_negative_marginals": CELL_BYTE_MARGINAL_N_NEGATIVE,
            "spread_max_over_min": CELL_BYTE_MARGINAL_B["max"] / CELL_BYTE_MARGINAL_B["min"],
            "rho_vs_wr1_residual_mass_proxy": CELL_MARGINAL_RHO_VS_WR1_RESIDUAL_MASS,
        },
        residual=0.0,
        source_artifact=RS2_ARTIFACT,
        measurement_method=(
            "384 exact re-encodes of the real shipped lattice, one per live cell, differenced "
            "against the shipped token member"
        ),
        provenance=prov,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    return CanonicalEquation(
        equation_id=BYTE_YIELD_EQUATION_ID,
        name="Token-lattice byte-marginal flatness",
        one_line_summary=(
            "Per-unit drop byte yield is ~flat (-58/196/211/472 B) and uncorrelated with "
            "activity, and 2 live units are NEGATIVE, so a reverse waterfill's ordering must "
            "come from the damage side, not the bytes."
        ),
        latex_form=r"\Delta b_i \approx \bar{b},\ \rho(\Delta b_i,\mathrm{act}_i)\approx 0",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rs2_waterfill_support_and_byte_yield_20260803"
            ":byte_side_is_rankable"
        ),
        domain_of_validity={
            "included": [
                "reverse-waterfill ordering on the TR1/DR7 token lattice under the ix2 coder",
                "greedy-vs-group pricing of unit drops (superadditive 1.0206x)",
            ],
            "excluded": [
                "other lattices or coders whose match structure differs",
                "the CELL grain, where the spread is 4.98x with NO negative marginals "
                "(second anchor); the flat/negative result is UNIT-grain only",
                "score, promotion, or pointer movement",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"byte_marginals": "bytes", "damage_proxy": "flips or a monotone proxy"},
        units_out={"spearman_bytes_vs_damage": "dimensionless"},
        empirical_anchors=(anchor, cell_anchor),
        predicted_vs_empirical_residual={"byte_side_rankability": 0.0},
        last_calibration_utc="2026-08-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("experiments.ddm_wr1_reverse_waterfill.cell_sensitivity",),
        canonical_producers=("scratchpad.br1_drop_surface", "scratchpad.br1_refine"),
        provenance=prov,
    )


def build_live_vs_dead_symbol_entropy_decomposition_v1() -> CanonicalEquation:
    """L3 — a coder's advantage over order-0 must be read on LIVE symbols only."""
    prov = _prov(
        BR1_ARTIFACT,
        "recompute both order-0 bounds whenever the dead-unit fraction moves (any new drop "
        "rung changes it); the live-only advantage is the only one that prices a new coder",
    )
    anchor = EmpiricalAnchor(
        anchor_id="br1_live_vs_dead_entropy_split_cx1_20260803",
        measurement_utc="2026-08-03T00:00:00Z",
        inputs={
            "all_symbols": ALL_SYMBOLS,
            "live_symbols": LIVE_SYMBOLS,
            "dead_unit_fraction": 1544 / 3072,
        },
        predicted_output={"apparent_advantage_is_inflated_by_dead_zeros": True},
        empirical_output={
            "all_shipped_b": ALL_SHIPPED_B,
            "all_order0_b": ALL_ORDER0_B,
            "live_shipped_b": LIVE_SHIPPED_B,
            "live_order0_b": LIVE_ORDER0_B,
            "apparent_advantage": ALL_ORDER0_B / ALL_SHIPPED_B,
            "live_advantage": LIVE_ORDER0_B / LIVE_SHIPPED_B,
        },
        residual=0.0,
        source_artifact=BR1_ARTIFACT,
        measurement_method=(
            "exact: order-0 bound from the empirical symbol histogram against the real coded "
            "size, computed once over all residual symbols and once over live units only"
        ),
        provenance=prov,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    return CanonicalEquation(
        equation_id=ENTROPY_SPLIT_EQUATION_ID,
        name="Live-vs-dead symbol entropy decomposition",
        one_line_summary=(
            "The coder's apparent 1.4776x advantage over order-0 is 1.1093x on LIVE symbols; "
            "the rest free-rides on LZ-copied dead zeros, so an explicit support map costs "
            "bytes and saves ~0."
        ),
        latex_form=(
            r"\frac{H_0(\mathrm{all})}{B(\mathrm{all})}=1.4776,\quad"
            r"\frac{H_0(\mathrm{live})}{B(\mathrm{live})}=1.1093"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rs2_waterfill_support_and_byte_yield_20260803"
            ":coder_advantage_split"
        ),
        domain_of_validity={
            "included": [
                "pricing any proposed replacement coder or explicit support map on this lattice",
                "reading a coder-vs-order-0 ratio on a stream with a large dead fraction",
            ],
            "excluded": [
                "streams with no dead/constant units, where the split is vacuous",
                "score, promotion, or pointer movement",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"all_shipped_b": "bytes", "live_order0_b": "bytes"},
        units_out={"free_riding_inflation": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"coder_advantage_split": 0.0},
        last_calibration_utc="2026-08-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("experiments.ddm_wr1_reverse_waterfill.cell_sensitivity",),
        canonical_producers=("scratchpad.br1_refine",),
        provenance=prov,
    )


__all__ = [
    "BYTE_YIELD_EQUATION_ID",
    "CELL_BYTE_MARGINAL_B",
    "CELL_MARGINAL_RHO_VS_WR1_RESIDUAL_MASS",
    "ENTROPY_SPLIT_EQUATION_ID",
    "SUPPORT_EQUATION_ID",
    "VERIFIED_VIA_SOURCE_INSPECTION",
    "build_lattice_cell_drop_pricing_support_v1",
    "build_live_vs_dead_symbol_entropy_decomposition_v1",
    "build_token_lattice_byte_marginal_flat_uncorrelated_v1",
    "byte_side_is_rankable",
    "coder_advantage_split",
    "drop_support_px",
    "support_mispricing",
]
