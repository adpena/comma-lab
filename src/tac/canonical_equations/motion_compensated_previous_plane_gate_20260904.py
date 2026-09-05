# SPDX-License-Identifier: MIT
"""Canonical equation: a temporal-context plane must beat the co-located plane on ALIGNMENT
before it can pay as a conditioning axis -- and on this field no decoder-derivable
motion-compensated plane does (ddm_mc1, 2026-09-04).

THE OBJECT.  The shipped integer HPAC mixer conditions on the previous pair's semantic field
CO-LOCATED (``prepare_frame_context`` -> ``conv_past``).  ddm_mc1 asked whether a
MOTION-COMPENSATED previous plane -- estimated from the two already-decoded fields by an
integer search and applied once more at constant velocity, so ZERO archive bytes -- carries
information the trained mixer could learn to read.  The charter's prior law (dc1's learned-
receptive-field mechanism) predicted >= 5,000 B of the 113,411 B stream at <= +1,500 B of model.

THE MEASURED INSTANCE (all 600 pairs, exact fs2 body, coder rows byte-identical to the
shipped stream 113,411 B sha 5601d6fd...):

  alignment, Lane IoU vs field_t   co-located 0.2495 | shift 0.2457 zoom 0.2423 planar 0.2429
                                   block 0.2295 block_gated 0.2460 block_median3 0.2180
  band agreement (edge band r=3)   co-located 0.8630 | 0.8580 0.8545 0.8544 0.8402 0.8554 0.8480
  ORACLE (reads field_t; diagnostic) Lane IoU: shift 0.2678 zoom 0.2727 planar 0.2742 block 0.3240
  block motion temporal consistency (derivable t-2->t-1 vs oracle t-1->t, road block-rows):
                                   exact match 0.395 / 0.083 / 0.031; corr dy +0.086 / +0.274 / -0.345
  carriage of the ORACLE block motion (0th-order entropy): 9,861 B = 16.5 B/pair
  coder ceiling (mi1 instrument, pair-level 2-fold, min over seeds), best decoder-derivable cell:
                                   shift +5.01 B, zoom +24.66 B, planar +32.78 B,
                                   block +159.60 B, block_gated +12.90 B, block_median3 +109.14 B
                                   (bar 5,000 B); ORACLE block plane +3,420.35 B (diagnostic)

THE MECHANISM (why the learned mixer cannot rescue it).  Two facts the diagnostics separate:
(a) the inter-pair change of this argmax field is mostly NOT rigid motion -- even oracle rigid /
ground-plane alignment buys <= +0.025 Lane IoU; (b) local block motion IS real (+0.074 Lane IoU
with oracle knowledge) but is not predictable from the previous transition (corr <= 0.27, one row
negative), so every decoder-derivable extrapolation lands WORSE than co-located.  A plane that
is worse-aligned than the plane the mixer already receives cannot add information through it,
and the coder's causal current-frame context already localises the moving edges.

THE GATE (the reusable part).  Before pricing any new temporal-context plane -- learned, warped,
carried or derived -- require that it beats the co-located plane on the alignment of the
bit-carrying classes AND on edge-band agreement.  Failing that gate predicts a coder ceiling at
the instrument's noise floor (~2.5 B here), which is what was measured.  A carried-motion plane
additionally owes its carriage bytes against the ORACLE plane's ceiling.

VERDICT.  CEILING-REFUSED; the prior-law falsifier fired.  The motion-compensated input is
CLOSED at formulation scope for the shipped receptive field on this field.  Axis
``[macOS-CPU advisory / scorer-free EXACT byte measurement]`` for the control;
``[model-ledger code length; REFUSAL-ONLY]`` for the ceiling.  NON-PROMOTABLE; moves no pointer.

Producer: ``experiments/ddm_mc1_motion_compensated_previous_plane.py`` (stages rows / motion /
oracle / ceiling / verdict) via ``.omx/research/ddm_mc1_motion_compensated_previous_plane_20260904.md``.
Consumers: every charter that proposes a temporal / warped / motion-aligned conditioning plane
for the token coder; the sister laws ``ddm_dc1`` (21-tap oracle floor) and ``ddm_mi1`` (the
conditioning ledger this instrument extends).
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

EQUATION_ID = "motion_compensated_previous_plane_alignment_gate_v1"

_UTC = "2026-09-05T00:00:00Z"
_AXIS = "[macOS-CPU advisory]"
_LEDGER = ".omx/research/ddm_mc1_motion_compensated_previous_plane_20260904.md"
_CHARTER = ".omx/research/charters/ddm_mc1_motion_compensated_previous_plane_20260903.md"
_PRODUCER = "experiments/ddm_mc1_motion_compensated_previous_plane.py"

# --- MEASURED (ddm_mc1, 2026-09-04/05) ---------------------------------------------------
STREAM_BYTES = 113_411
STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
PAIRS = 600
MC_PAIRS = 598  # pairs 0-1 are co-located by construction

REFUSE_BELOW_BYTES = 5_000.0  # the charter's pre-registered ceiling bar
PRIOR_LAW_PREDICTED_SAVING_BYTES = 5_000.0
INSTRUMENT_NOISE_FLOOR_BYTES = 2.51  # indicator `none` nesting control, pair split

IOU_COLOCATED: Mapping[str, float] = {
    "Lane": 0.2495,
    "Movable": 0.8431,
    "Road": 0.9522,
    "MyCar": 0.9931,
    "Undrivable": 0.9942,
}
BAND_AGREEMENT_COLOCATED = 0.8630

# Six decoder-derivable families (constant velocity from field[t-2] -> field[t-1]).
IOU_LANE_DERIVABLE: Mapping[str, float] = {
    "shift": 0.2457,
    "zoom": 0.2423,
    "planar": 0.2429,
    "block": 0.2295,
    "block_gated": 0.2460,
    "block_median3": 0.2180,
}
BAND_AGREEMENT_DERIVABLE: Mapping[str, float] = {
    "shift": 0.8580,
    "zoom": 0.8545,
    "planar": 0.8544,
    "block": 0.8402,
    "block_gated": 0.8554,
    "block_median3": 0.8480,
}
IOU_MOVABLE_DERIVABLE: Mapping[str, float] = {
    "shift": 0.8369,
    "zoom": 0.8382,
    "planar": 0.8378,
    "block": 0.8494,
    "block_gated": 0.8510,
    "block_median3": 0.8547,
}

# ORACLE (diagnostic; estimates on field_t itself -- never decoder-derivable).
IOU_LANE_ORACLE: Mapping[str, float] = {
    "shift": 0.2678,
    "zoom": 0.2727,
    "planar": 0.2742,
    "block": 0.3240,
}
BAND_AGREEMENT_ORACLE: Mapping[str, float] = {
    "shift": 0.8689,
    "zoom": 0.8732,
    "planar": 0.8736,
    "block": 0.9124,
}

# Block motion temporal consistency, road block-rows 2/3/4 (64 px rows), 598 pairs x 8 blocks.
BLOCK_ROW_MEAN_ABS_SHIFT_PX: Mapping[int, float] = {2: 2.04, 3: 5.97, 4: 4.46}
BLOCK_ROW_EXACT_MATCH: Mapping[int, float] = {2: 0.395, 3: 0.083, 4: 0.031}
BLOCK_ROW_CORR_DY: Mapping[int, float] = {2: 0.086, 3: 0.274, 4: -0.345}
BLOCK_ROW_CORR_DX: Mapping[int, float] = {2: 0.205, 3: 0.178, 4: -0.111}

# Carriage of the ORACLE block motion (DERIVED: zeroth-order entropy of the persisted oracle
# shifts, no context model), the price of the carried-motion road.
CARRIED_ORACLE_BLOCK_MOTION_BITS = 78_885.0
CARRIED_ORACLE_BLOCK_MOTION_BYTES = CARRIED_ORACLE_BLOCK_MOTION_BITS / 8.0  # 9,860.6 B
CARRIED_ORACLE_BLOCK_MOTION_BYTES_PER_PAIR = CARRIED_ORACLE_BLOCK_MOTION_BYTES / 599  # 16.46 B/pair

# Coder ceiling, mi1 instrument on the coder's own rows: best decoder-derivable cell, pair-level
# two-fold, min over the seeds run for that cell.  (family, cell) named for the record.
CEILING_BEST_HELD_OUT_BYTES: Mapping[str, float] = {
    "shift": 5.01,  # indicator `mc`
    "zoom": 24.66,  # tilt `mc_x_arg` (one seed); indicator `mc_x_arg` 3-seed min 13.77
    "planar": 32.78,  # tilt `mc_x_arg` (one seed); indicator `mc_x_arg` 3-seed min 16.66
    "block": 159.60,  # tilt `mc_x_coloc_x_arg` (one seed); indicator `mc_x_arg` 3-seed min 138.49
    "block_gated": 12.90,  # tilt `mc_x_arg`
    "block_median3": 109.14,  # tilt `mc_x_coloc_x_arg`; indicator `mc_x_arg` 3-seed min 80.39
}
CEILING_BEST_CELL: Mapping[str, str] = {
    "shift": "indicator:mc",
    "zoom": "tilt:mc_x_arg",
    "planar": "tilt:mc_x_arg",
    "block": "tilt:mc_x_coloc_x_arg",
    "block_gated": "tilt:mc_x_arg",
    "block_median3": "tilt:mc_x_coloc_x_arg",
}
BLOCK_INDICATOR_MC_X_ARG_3SEED_MIN_BYTES = 138.49
# The ORACLE block plane priced through the same instrument (diagnostic; reads field_t).
ORACLE_BLOCK_CEILING_HELD_OUT_BYTES = 3_420.35  # tilt `mc_x_coloc_x_arg_x_bd`
ORACLE_BLOCK_CEILING_INDICATOR_3SEED_MIN_BYTES = 3_109.82  # indicator `mc_x_coloc_x_arg`
# The charter's literal (b): a bare KT categorical of field_t, contexts {coloc} vs {coloc, mc},
# pairs 2..599.  Reported for completeness; its baseline is ~11x the shipped stream, so it is a
# different object and NEVER a rate claim.
BARE_CATEGORICAL_BASELINE_BYTES = 1_230_327.0
BARE_CATEGORICAL_SAVING_ADDING_MC_BYTES: Mapping[str, float] = {
    "shift": 59_620.0,
    "zoom": 74_346.0,
    "planar": 70_901.0,
    "block": 143_323.0,
    "block_gated": 56_755.0,
    "block_median3": 117_793.0,
}
BARE_CATEGORICAL_CTX_MC_ALONE_BYTES: Mapping[str, float] = {
    "shift": 1_268_748.0,
    "zoom": 1_287_311.0,
    "planar": 1_290_950.0,
    "block": 1_400_764.0,
    "block_gated": 1_298_257.0,
    "block_median3": 1_304_222.0,
    "oracle_block": 845_160.0,
}

DEFAULT_TEMPORAL_PREDICTABILITY_THRESHOLD = 0.5


def alignment_gain(iou_candidate: float, iou_colocated: float) -> float:
    """Signed IoU gain of a candidate previous-plane over the co-located plane."""
    return float(iou_candidate) - float(iou_colocated)


def plane_passes_alignment_gate(
    *,
    iou_candidate_lane: float,
    iou_colocated_lane: float,
    band_candidate: float,
    band_colocated: float,
) -> bool:
    """THE GATE: a temporal-context plane is admissible for a ceiling only if it beats the
    co-located plane on the bit-carrying class (Lane) AND on edge-band agreement."""
    return bool(
        alignment_gain(iou_candidate_lane, iou_colocated_lane) > 0.0
        and float(band_candidate) > float(band_colocated)
    )


def ceiling_refused(held_out_bytes_saved: float, refuse_below: float = REFUSE_BELOW_BYTES) -> bool:
    """The charter's pre-registered refusal: best ideal saving below the bar."""
    return float(held_out_bytes_saved) < float(refuse_below)


def temporal_predictability_supports_extrapolation(
    correlation: float, threshold: float = DEFAULT_TEMPORAL_PREDICTABILITY_THRESHOLD
) -> bool:
    """Constant-velocity extrapolation of a per-block motion is only worth applying when the
    transition-to-transition correlation clears ``threshold`` (0.5 = half the variance)."""
    return float(correlation) >= float(threshold)


def carried_motion_breakeven_open(
    *, carriage_bytes: float, oracle_plane_ceiling_bytes: float
) -> bool:
    """A CARRIED motion plane can only pay if the ORACLE-aligned plane's own conditioning
    ceiling exceeds the bytes needed to carry the motion."""
    return float(oracle_plane_ceiling_bytes) > float(carriage_bytes)


def _anchor_alignment() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="mc1_six_derivable_planes_all_worse_aligned_than_colocated_20260904",
        measurement_utc="2026-09-04T23:30:00Z",
        inputs={
            "field_sha256": FIELD_SHA256,
            "pairs_motion_compensated": MC_PAIRS,
            "estimation": "integer argmax of class agreement on the edge band (r=3) of field[t-1], identity-first tie-break",
            "extrapolation": "constant velocity: the t-2->t-1 transform re-applied to field[t-1]",
            "families": sorted(IOU_LANE_DERIVABLE),
            "producer": f"{_PRODUCER} --stage motion",
        },
        predicted_output={
            "prior_law": "the MC plane aligns better than co-located on the moving classes (Lane, Movable, Road edges)",
            "falsifier": "Lane IoU or band agreement below the co-located plane",
        },
        empirical_output={
            "iou_lane_colocated": IOU_COLOCATED["Lane"],
            "iou_lane_derivable": dict(IOU_LANE_DERIVABLE),
            "band_colocated": BAND_AGREEMENT_COLOCATED,
            "band_derivable": dict(BAND_AGREEMENT_DERIVABLE),
            "iou_movable_derivable": dict(IOU_MOVABLE_DERIVABLE),
            "reading": "every family loses Lane/Road/MyCar/Undrivable alignment; only block variants gain on Movable (1.24% of area)",
        },
        # Magnitude of the BEST family's shortfall against the predicted ">0" gain (block_gated, -0.0035).
        residual=abs(max(alignment_gain(v, IOU_COLOCATED["Lane"]) for v in IOU_LANE_DERIVABLE.values())),
        source_artifact=_LEDGER,
        measurement_method="per-pair per-class IoU and edge-band agreement of the predicted plane against field_t, all 598 MC pairs",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a decoder-derivable plane that beats co-located on Lane IoU AND band agreement on this field",
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_oracle_and_consistency() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="mc1_oracle_alignment_and_block_motion_unpredictability_20260904",
        measurement_utc="2026-09-04T23:40:00Z",
        inputs={
            "oracle": "same families estimated on field[t-1] -> field[t] (reads the target; diagnostic only)",
            "consistency": "derivable block shift at t (from t-2->t-1) vs oracle block shift at t (t-1->t), road block-rows",
            "carriage": "zeroth-order entropy of the oracle block shifts over 599 pairs",
        },
        predicted_output={
            "prior_law": "if extrapolation is the weak link, oracle alignment is large and transition motion is temporally consistent",
            "falsifier": "oracle rigid gain small, or block motion correlation << 1",
        },
        empirical_output={
            "iou_lane_oracle": dict(IOU_LANE_ORACLE),
            "band_oracle": dict(BAND_AGREEMENT_ORACLE),
            "block_row_exact_match": {str(k): v for k, v in BLOCK_ROW_EXACT_MATCH.items()},
            "block_row_corr_dy": {str(k): v for k, v in BLOCK_ROW_CORR_DY.items()},
            "block_row_corr_dx": {str(k): v for k, v in BLOCK_ROW_CORR_DX.items()},
            "carried_oracle_block_motion_bytes": CARRIED_ORACLE_BLOCK_MOTION_BYTES,
            "reading": "rigid/ground-plane oracle gains <= +0.025 Lane IoU (the change is not rigid motion); block oracle +0.074 but corr <= 0.27, so it is not extrapolable; carrying it costs 9,861 B",
        },
        residual=max(BLOCK_ROW_CORR_DY.values()),
        source_artifact=_LEDGER,
        measurement_method="oracle estimation on the true target field; per-block parameter comparison across consecutive transitions; empirical entropy of the oracle parameters",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a motion representation whose transition-to-transition correlation clears 0.5 on the road rows",
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_ceiling() -> EmpiricalAnchor:
    best = max(CEILING_BEST_HELD_OUT_BYTES.values())
    return EmpiricalAnchor(
        anchor_id="mc1_coder_ceiling_all_families_and_oracle_refused_20260905",
        measurement_utc="2026-09-05T00:15:00Z",
        inputs={
            "rows": "every RC64 coding row of the shipped encoder on the fs2 tree; control stream byte-identical to the shipped stream",
            "stream_bytes": STREAM_BYTES,
            "stream_sha256": STREAM_SHA256,
            "instrument": "ddm_mi1 family q' = sigma(logit(1-pmax) + beta_cell) on the argmax indicator, plus a 5-way log-linear tilt; 2-fold cross-fit, pair-level split (seeds 20260824/777/31337), saturated positions excluded",
            "producer": f"{_PRODUCER} --stage ceiling",
        },
        predicted_output={
            "prior_law_saving_bytes": PRIOR_LAW_PREDICTED_SAVING_BYTES,
            "refuse_below_bytes": REFUSE_BELOW_BYTES,
            "falsifier": "best held-out saving below 5,000 B",
        },
        empirical_output={
            "best_held_out_bytes_by_family": dict(CEILING_BEST_HELD_OUT_BYTES),
            "best_cell_by_family": dict(CEILING_BEST_CELL),
            "block_indicator_mc_x_arg_3seed_min_bytes": BLOCK_INDICATOR_MC_X_ARG_3SEED_MIN_BYTES,
            "oracle_block_ceiling_held_out_bytes": ORACLE_BLOCK_CEILING_HELD_OUT_BYTES,
            "oracle_block_ceiling_indicator_3seed_min_bytes": ORACLE_BLOCK_CEILING_INDICATOR_3SEED_MIN_BYTES,
            "carried_oracle_block_motion_bytes": CARRIED_ORACLE_BLOCK_MOTION_BYTES,
            "instrument_noise_floor_bytes": INSTRUMENT_NOISE_FLOOR_BYTES,
            "bare_categorical_baseline_bytes": BARE_CATEGORICAL_BASELINE_BYTES,
            "bare_categorical_saving_adding_mc_bytes": dict(BARE_CATEGORICAL_SAVING_ADDING_MC_BYTES),
            "bare_categorical_ctx_mc_alone_bytes": dict(BARE_CATEGORICAL_CTX_MC_ALONE_BYTES),
            "typed_verdict": "CEILING-REFUSED",
            "reading": (
                "best decoder-derivable cell +159.60 B held-out (block, tilt mc_x_coloc_x_arg; 3-seed "
                "indicator min +138.49 B) against a 5,000 B bar; the ORACLE plane reaches +3,420.35 B, "
                "below the bar and below its 9,861 B carriage; 625-cell rows go negative held-out on five "
                "of six derivable planes (mi1's oracle-mirage signature)"
            ),
        },
        residual=PRIOR_LAW_PREDICTED_SAVING_BYTES - best,
        source_artifact=_LEDGER,
        measurement_method="held-out code length on the coder's own rows under per-cell offsets fitted on the other pair fold; min over seeds",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a decoder-derivable plane that passes the alignment gate on this field; or a carried motion whose carriage is below the oracle plane's 3,420 B ceiling",
            measurement_axis="[model-ledger code length; REFUSAL-ONLY]",
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_motion_compensated_previous_plane_alignment_gate_v1() -> CanonicalEquation:
    """Build the alignment-gate equation for temporal-context planes (ddm_mc1, 2026-09-04)."""
    alignment = _anchor_alignment()
    oracle = _anchor_oracle_and_consistency()
    ceiling = _anchor_ceiling()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Temporal-context plane alignment gate -- a previous-plane that is worse-aligned than "
            "the co-located plane cannot pay as a conditioning axis, learned mixer or not"
        ),
        one_line_summary=(
            "mc1: six derivable MC planes all below co-located on Lane IoU (0.218-0.246 vs 0.2495); "
            "coder ceiling <= +159.60 B vs a 5,000 B bar; oracle plane +3,420 B < its 9,861 B carriage"
        ),
        latex_form=(
            r"\text{admit}(P)\iff \mathrm{IoU}_{\text{Lane}}(P,F_t) > \mathrm{IoU}_{\text{Lane}}(F_{t-1},F_t)"
            r"\ \wedge\ \mathrm{agree}_{\partial}(P,F_t) > \mathrm{agree}_{\partial}(F_{t-1},F_t);\quad"
            r"\text{else } \Delta B_{\text{held-out}} \le 159.60\ \text{B on } 113{,}411\ \text{B (MEASURED)};\quad"
            r"\text{carried: } \Delta B_{\text{oracle}}=3{,}420 < 9{,}861 = H(\text{motion})/8"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.motion_compensated_previous_plane_gate_20260904"
            ":plane_passes_alignment_gate"
        ),
        domain_of_validity={
            "included": [
                "any candidate previous-field plane offered to the token coder as a conditioning axis "
                "(motion-compensated, warped, carried, learned) on the 600-pair semantic field",
                "integer, decoder-derivable estimators with the identity in their candidate set",
                "the shipped HPAC receptive field (7x7 masked + dilated depthwise + conv_past + patch FiLM)",
            ],
            "excluded": [
                "use as a d_seg / d_pose / rate lever, a score, or a promotion claim",
                "fields whose inter-pair change IS rigid motion (a different video would need its own oracle row)",
                "the bare categorical numbers as rate claims: their baseline is ~11x the shipped stream",
            ],
            "measurement_axis": [_AXIS, "[model-ledger code length; REFUSAL-ONLY]"],
            "result_type": "CONDITIONING-AXIS closure; NON-PROMOTABLE; moves no pointer",
            "sister_laws": [
                "ddm_dc1 -- the 21-tap oracle floor and the learned-receptive-field mechanism this tested",
                "ddm_mi1 -- the conditioning ledger (2,162 B whole target; 211 B richest unconsumed axis) this instrument extends",
                "ddm_xi1 -- warp context on a count-based coder (+12,262 B by dilution); this arm fed the trained rows instead and still found <= 33 B",
            ],
            "known_boundary": (
                "one field, one body; the block-family and oracle-plane ceilings land in the ledger as "
                "separate rows; the gate's threshold (beat co-located) is the measured fact, not a fitted constant"
            ),
            "verdict_scope": "formulation (six integer families) with a measured diagnostic bound on the family (oracle rows)",
        },
        units_in={
            "iou_candidate_lane": "dimensionless_iou",
            "iou_colocated_lane": "dimensionless_iou",
            "band_candidate": "dimensionless_fraction",
            "band_colocated": "dimensionless_fraction",
            "held_out_bytes_saved": "bytes",
            "carriage_bytes": "bytes",
            "correlation": "dimensionless_pearson",
        },
        units_out={
            "alignment_gain": "dimensionless_iou",
            "plane_passes_alignment_gate": "bool",
            "ceiling_refused": "bool",
            "temporal_predictability_supports_extrapolation": "bool",
            "carried_motion_breakeven_open": "bool",
        },
        empirical_anchors=(alignment, oracle, ceiling),
        predicted_vs_empirical_residual={
            alignment.anchor_id: alignment.residual,
            oracle.anchor_id: oracle.residual,
            ceiling.anchor_id: ceiling.residual,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER, _CHARTER),
        canonical_producers=(_PRODUCER, _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="motion_compensated_previous_plane_alignment_gate.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
