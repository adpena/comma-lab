# SPDX-License-Identifier: MIT
"""Canonical equation: a trained field's exact error partitions by TRAJECTORY, and the
persistent class is a measured FLOOR on what any optimizer or schedule lever can leave behind.

THE GAP THIS CODIFIES.  Every prior read of a trained distortion term in this campaign is either
MACRO (a score at a milestone) or STATIC MICRO (one frozen field re-weighted -- sd1's
surrogate-vs-exact map, gm1's gradient mass).  Neither can answer the question every schedule
lever implicitly asks: **of the error that is still there at the end, how much did the optimizer
PUT there, and how much was there before it started and never moved?**  A campaign that cannot
separate those two spends optimizer budget on representation-bound error, or capacity budget on
error a warm start would have removed for free.

THE PARTITION (the equation's form).  Sample the trained weights at a checkpoint cadence
``t = 0..T-1`` and reconstruct the EXACT per-site correctness ``w[t, s] in {0, 1}`` through the
real render -> roundtrip -> frozen-scorer path.  Classify every site by a falling rule whose whole
point is that it PARTITIONS:

    CHURN            n_flips(s) > churn_flips            (first, because it is a COUNT rule and
                                                          would otherwise be absorbed)
    PERSISTENT       w[0, s] and mean_t w[t, s] >= f      (wrong at the start and essentially always)
    NEW_PERSISTENT   not w[0, s] and w[T-1, s]            (the run put it there and left it)
    TRANSIENT_BORN   not w[0, s] and not w[T-1, s]        (the run put it there and took it back)
    HEALED           w[0, s], mean_t w < f, few flips      (the RESIDUAL class: wrong at the start
                                                           and no longer persistently wrong.  Most
                                                           members end correct; the predicate does
                                                           NOT require it, and the class's terminal
                                                           contribution counts those that do not.)
    ALWAYS_CORRECT   no t with w[t, s]

The five error classes plus ALWAYS_CORRECT cover every site exactly once.  HEALED is the class a
partition REQUIRES and that a four-class reading omits; omitting it silently breaks the identity
below, which is why it is named here.

THE IDENTITY (why the decomposition is a gate, not an approximation).  A Horvitz-Thompson
distortion estimator with INTEGER pair weights and integer site counts,

    d_seg_hat(t) = ( sum_p w_p * n_wrong(p, t) ) / (N * H * W)

carries its whole content in the integer numerator ``W(t) = sum_p w_p * n_wrong(p, t)``.  The
class numerators therefore sum to the total EXACTLY -- no float summation order can move it.  The
calibration gate is ``max_t |sum_classes W_c(t) - W(t)| == 0`` in INTEGERS.  A float-tolerance
version of the same gate would pass on a broken partition; the integer version cannot.

THE READING (what the equation exports).  The PERSISTENT class's terminal contribution is a
measured FLOOR: it is what remains if every non-persistent terminal error were removed.  Its
complement is the optimizer-reachable share.  Neither is a prediction that a lever reaches the
floor -- the floor is an upper bound on the credit any schedule/optimizer lever can claim.

SCOPE AND WHAT IT IS NOT.  The classes are defined RELATIVE TO A CADENCE and to two thresholds
(``churn_flips``, ``persistent_fraction``).  A coarser cadence merges flips and moves mass from
CHURN into the endpoint classes; a finer one does the reverse.  The floor is therefore
cadence-conditional and must be reported with its cadence, exactly as a p95 is reported with its
sample.  This is an APPARATUS / measurement law: it moves no pointer, is not a d_seg / d_pose /
rate lever, and carries no score claim.

Producer: `experiments/ddm_md1_micro_to_macro.py` via
`.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md`.
Consumers: any arm that proposes a schedule, optimizer, or transition lever against a trained
distortion term and needs to know the credit ceiling before it spends.

Sisters: `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` (the per-site quantity whose sign
IS ``w``), `muon_finisher_schedule_warmstart_and_lr_anneal_v1` (the transition lever this bounds),
`persistence_topology_cldice_betti_island_recall_v1` (persistence in the topological sense; this
law is persistence in the TRAJECTORY sense and the two are different objects).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

_UTC = "2026-09-04T00:00:00Z"
_AXIS = (
    "[macOS-CPU advisory; exact argmax reconstructed from retained 16-step checkpoints; "
    "frozen CPU-torch SegNet; QBF1-born vehicle; n32 sealed selection; NON-PROMOTABLE]"
)
_LEDGER = ".omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md"
_ANALYSIS = (
    "/Volumes/APDataStore/pact/ddm_md1_micro_macro"
    "/ANALYSIS_cold_control_seed_20260902_dali.json"
)

TRAJECTORY_CLASSES = (
    "ALWAYS_CORRECT",
    "CHURN",
    "PERSISTENT",
    "NEW_PERSISTENT",
    "TRANSIENT_BORN",
    "HEALED",
)
ERROR_CLASSES = TRAJECTORY_CLASSES[1:]

DEFAULT_CHURN_FLIPS = 4
DEFAULT_PERSISTENT_FRACTION = 0.90

# ddm_md1 measured instance -- filled from ANALYSIS_cold_control_seed_20260902_dali.json,
# `forwards.shadow` (the shadow is the object that ships: the archive is re-encoded from
# `ema.shadow`, ddm_qbr1_born_fairform_burn_prep.py:629-632).
MD1_CADENCE_CHECKPOINTS = 71
MD1_TERMINAL_NUMERATOR = 331_080
MD1_PERSISTENT_NUMERATOR = 205_305
MD1_DENOMINATOR = 117_964_800.0
MD1_PERSISTENT_TERMINAL_SHARE = 0.6201069227981153
MD1_CALIBRATION_GATE_INTEGER_RESIDUAL = 0
MD1_PREREGISTERED_PREDICTION = 0.60
MD1_PREREGISTERED_FALSIFIER = 0.40

# Second measured instance: the ng1 warm-transition cell (same seed, same data order, one lever --
# r10's AdamW moments carried in), same 71-checkpoint cadence, same shadow forward, same authority.
MD1_WARM_TERMINAL_NUMERATOR = 343_320
MD1_WARM_PERSISTENT_NUMERATOR = 202_590
MD1_WARM_PERSISTENT_TERMINAL_SHARE = 0.5900897589898404

# Third measured instance (ddm_md2, 2026-09-05): the ng5 tau-band x carried-duals cell -- the burn
# default of record -- at the SAME 71-checkpoint cadence, same shadow forward, same DALI authority,
# same initial state.  This is the reactivation the md1 anchor's own criteria named.  Filled from
# /Volumes/APDataStore/pact/ddm_md2_persistent_under_burn_default
# /ANALYSIS_ng5_tau_band_seed_20260902_dali.json.
MD2_NG5_TERMINAL_NUMERATOR = 298_455
MD2_NG5_PERSISTENT_NUMERATOR = 187_890
MD2_NG5_PERSISTENT_TERMINAL_SHARE = 0.629542142031462
MD2_NG5_LIVE_PERSISTENT_TERMINAL_SHARE = 0.35438596491228064
# Site overlap of the two cells' PERSISTENT sets (shadow, DALI).  The null is two independent draws
# of the observed sizes from the STEP-0 WRONG POOL -- both sets are subsets of it by the class
# definition, and the pool is bit-identical across the cells because they share an initialisation.
# Reporting the all-sites null (492x) instead would overstate the reading.
MD2_NG5_VS_COLD_PERSISTENT_JACCARD = 0.8069079987353778
MD2_NG5_VS_COLD_PERSISTENT_JACCARD_WITHIN_POOL_CHANCE = 0.5263033946906265
MD2_NG5_VS_COLD_PERSISTENT_JACCARD_ATTAINABLE_MAX = 0.9305016044587063
MD2_STEP_ZERO_WRONG_POOL_SITES = 16_553

# md3 (2026-09-05): the DATA-ORDER control.  The retained wc3 seed_20260903 cell -- SAME init sha
# 991a1cc6..., schedule.seed 20260903 with a different schedule.sha256_u8, so the pairs arrive in a
# different order and nothing else moves.  It is the confound md2 named as the free prerequisite to
# the init question, and the fourth cell measured at md1's cadence on ONE initialisation.  Filled
# from /Volumes/APDataStore/pact/ddm_md3_different_initialisation
# /ANALYSIS_data_order_control_seed_20260903_dali.json.
MD3_DATA_ORDER_TERMINAL_NUMERATOR = 332_745
MD3_DATA_ORDER_PERSISTENT_NUMERATOR = 204_990
MD3_DATA_ORDER_PERSISTENT_TERMINAL_SHARE = 0.6160573412072308
MD3_DATA_ORDER_LIVE_PERSISTENT_TERMINAL_SHARE = 0.4056160088528219
MD3_DATA_ORDER_VS_COLD_PERSISTENT_JACCARD = 0.8535551718725475
MD3_DATA_ORDER_VS_COLD_PERSISTENT_INTERSECTION_SITES = 10_876
# md3 also measured that the step-0 pool is NOT a fixed property of the vehicle: across eight
# legitimately-produced starts (r6-r10 stage-03 ends and r10 periodics, every one copied verbatim
# from a retained governed run) the pool Jaccard against the incumbent falls 0.9909 -> 0.1175.  The
# unreachable sites inside it do not move with it -- over the four starts whose step-0 d_seg_hat is
# comparable to the incumbent's, 83.13% of the incumbent's PERSISTENT sites are wrong at ALL of
# them against 19.78% of the sites the optimizer DID reach (4.20x; 4.21x on PyAV).  Recorded here as
# anchor CONTEXT, not as a term of this law: it is a step-0 statement, not a trajectory partition.
MD3_START_LADDER_POOL_JACCARD_RANGE = (0.1175, 0.9909)
MD3_UNREACHABLE_WRONG_AT_ALL_COMPARABLE_STARTS = 0.8313
MD3_REACHED_WRONG_AT_ALL_COMPARABLE_STARTS = 0.1978


def partition_is_exact(
    class_numerators: Mapping[str, Sequence[int]], total_numerator: Sequence[int]
) -> bool:
    """THE GATE: the class integer numerators must sum to the total at EVERY checkpoint.

    Returns True only on an exact integer identity.  A float-tolerance version of this check
    would pass on a partition that double-counts or drops a class, which is precisely the defect
    the integer form exists to refuse.  An EMPTY class map is refused rather than passed: a gate
    with no classes in it is vacuous, and a vacuous pass is the failure mode that lets a broken
    decomposition ship with a green light.
    """

    if not class_numerators:
        raise ValueError("partition gate needs at least one class series; an empty map is vacuous")
    steps = len(total_numerator)
    for values in class_numerators.values():
        if len(values) != steps:
            raise ValueError("every class numerator series must cover every checkpoint")
    for index in range(steps):
        stacked = sum(int(values[index]) for values in class_numerators.values())
        if stacked != int(total_numerator[index]):
            return False
    return True


def reachability_floor(
    *, terminal_numerator: int, persistent_numerator: int, denominator: float
) -> dict[str, float]:
    """The credit ceiling for any optimizer/schedule lever, from the exact integer numerators.

    ``persistent_floor`` is the distortion that remains when every non-PERSISTENT terminal error
    is removed.  ``optimizer_reachable_share`` is the complement -- the largest share of the
    terminal distortion a schedule, transition, or objective lever could possibly claim.  It is a
    CEILING on credit, never a prediction that a lever collects it.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if terminal_numerator < 0 or persistent_numerator < 0:
        raise ValueError("numerators must be non-negative")
    if persistent_numerator > terminal_numerator:
        raise ValueError("the persistent numerator cannot exceed the terminal numerator")
    reachable = terminal_numerator - persistent_numerator
    return {
        "terminal_distortion": terminal_numerator / denominator,
        "persistent_floor": persistent_numerator / denominator,
        "optimizer_reachable_distortion": reachable / denominator,
        "optimizer_reachable_share": (
            reachable / terminal_numerator if terminal_numerator > 0 else 0.0
        ),
        "persistent_share": (
            persistent_numerator / terminal_numerator if terminal_numerator > 0 else 0.0
        ),
    }


def floor_clears_target(
    *, persistent_floor: float, target_distortion: float
) -> bool:
    """Can ANY optimizer/schedule lever alone reach the target on this object?

    False whenever the persistent floor already exceeds the target: no schedule lever can remove
    representation-bound error, so the answer is a change of representation, not of schedule.
    """

    if target_distortion <= 0:
        raise ValueError("target distortion must be positive")
    return float(persistent_floor) < float(target_distortion)


def build_checkpoint_trajectory_error_partition_v1() -> CanonicalEquation:
    """Build the trajectory-partition / reachability-floor equation (ddm_md1, 2026-09-04)."""

    md1_anchor = EmpiricalAnchor(
        anchor_id="md1_qbr1_cold_control_seed_20260902_shadow_trajectory_partition_20260904",
        measurement_utc=_UTC,
        inputs={
            "cell": "QBR1 born-fairform cold control, seed 20260902, 5,000 AdamW updates",
            "forward": "EMA shadow (the object the archive is re-encoded from)",
            "cadence": (
                "every 16 steps 0-512, every 64 to 2,048, every 256 to 5,000, plus the "
                "checkpointed milestone steps 2,000 and 4,000 and the terminal state"
            ),
            "checkpoints": MD1_CADENCE_CHECKPOINTS,
            "sites": 6_291_456,
            "selection": "n32 sealed no2 stratified Horvitz-Thompson, weights 15/30",
            "gt_authority": "DALI gt_cache_dali.pt",
            "churn_flips": DEFAULT_CHURN_FLIPS,
            "persistent_fraction": DEFAULT_PERSISTENT_FRACTION,
            "tool": "experiments/ddm_md1_micro_to_macro.py --mode sweep|analyze",
        },
        predicted_output={
            "prior_law": (
                "gc1's capacity closure -- Lane's mismatches fall only 1.16x while Road falls "
                "2.59x and Undrivable 3.03x as generator capacity rises 1.599x -- predicted a "
                "PERSISTENT share of at least 60% of the terminal error"
            ),
            "preregistered_prediction": MD1_PREREGISTERED_PREDICTION,
            "preregistered_falsifier": MD1_PREREGISTERED_FALSIFIER,
            "preregistration": ".omx/research/ddm_md1_prereg_20260904.md",
        },
        empirical_output={
            "terminal_weighted_wrong_site_numerator": MD1_TERMINAL_NUMERATOR,
            "persistent_weighted_wrong_site_numerator": MD1_PERSISTENT_NUMERATOR,
            "denominator": MD1_DENOMINATOR,
            "persistent_terminal_share": MD1_PERSISTENT_TERMINAL_SHARE,
            "calibration_gate_integer_residual": MD1_CALIBRATION_GATE_INTEGER_RESIDUAL,
            "terminal_d_seg_hat": MD1_TERMINAL_NUMERATOR / MD1_DENOMINATOR,
            "persistent_floor_d_seg_hat": MD1_PERSISTENT_NUMERATOR / MD1_DENOMINATOR,
            "sub_012_target_d_seg": 1.3646784205e-4,
            "persistent_floor_over_target": (
                MD1_PERSISTENT_NUMERATOR / MD1_DENOMINATOR / 1.3646784205e-4
            ),
            "start_numerator": 301_470,
            "repaired_over_the_run": 9_075 + 15_360,
            "created_over_the_run": 19_380 + 34_665,
            "created_over_repaired_ratio": (19_380 + 34_665) / (9_075 + 15_360),
            "persistent_numerator_removed_fraction": 9_075 / 214_380,
            "prediction_holds_on_the_shadow_forward": True,
            "falsifier_fired_on_the_shadow_forward": False,
            "live_forward_persistent_share": 0.35779,
            "live_forward_reading_note": (
                "the LIVE forward gives 35.779%, below the 40% falsifier, because the live weights "
                "are noisy checkpoint-to-checkpoint and CHURN absorbs 62.347% of the terminal "
                "error; the SHADOW is the object the archive is re-encoded from and is therefore "
                "the authoritative reading"
            ),
        },
        residual=abs(MD1_PERSISTENT_TERMINAL_SHARE - MD1_PREREGISTERED_PREDICTION),
        source_artifact=_LEDGER,
        measurement_method=(
            "exact per-site argmax reconstructed from every swept checkpoint through the "
            "trainer's own render/roundtrip/scorer path; the partition and the bridge are carried "
            "in integer Horvitz-Thompson numerators so the calibration gate is exact"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_ANALYSIS,
            reactivation_criteria=(
                "a second cell measured at the SAME cadence (the warm transition, or a second "
                "seed) turns the persistent share into a family reading; a different cadence "
                "re-opens the class boundaries and requires a fresh anchor"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    warm_anchor = EmpiricalAnchor(
        anchor_id="md1_ng1_warm_transition_seed_20260902_shadow_trajectory_partition_20260904",
        measurement_utc="2026-09-04T14:22:00Z",
        inputs={
            "cell": "ng1 warm transition, seed 20260902, 5,000 updates, r10 AdamW moments carried",
            "forward": "EMA shadow",
            "checkpoints": MD1_CADENCE_CHECKPOINTS,
            "sites": 6_291_456,
            "gt_authority": "DALI gt_cache_dali.pt",
            "churn_flips": DEFAULT_CHURN_FLIPS,
            "persistent_fraction": DEFAULT_PERSISTENT_FRACTION,
            "single_lever": (
                "optimizer_state_dict only; the step-0 milestone is bit-identical to the cold "
                "control's, so the two cells share a start and differ in one thing"
            ),
        },
        predicted_output={
            "prior_law": (
                "the cold instance measured 62.011%; a same-seed same-data-order twin differing "
                "only in the optimizer state should land near it if the persistent set is a "
                "property of the representation rather than of the trajectory"
            ),
            "preregistered_prediction": MD1_PREREGISTERED_PREDICTION,
            "preregistered_falsifier": MD1_PREREGISTERED_FALSIFIER,
            "preregistration": ".omx/research/ddm_md1_prereg_20260904.md",
        },
        empirical_output={
            "terminal_weighted_wrong_site_numerator": MD1_WARM_TERMINAL_NUMERATOR,
            "persistent_weighted_wrong_site_numerator": MD1_WARM_PERSISTENT_NUMERATOR,
            "denominator": MD1_DENOMINATOR,
            "persistent_terminal_share": MD1_WARM_PERSISTENT_TERMINAL_SHARE,
            "calibration_gate_integer_residual": MD1_CALIBRATION_GATE_INTEGER_RESIDUAL,
            "persistent_floor_d_seg_hat": MD1_WARM_PERSISTENT_NUMERATOR / MD1_DENOMINATOR,
            "persistent_floor_over_target": (
                MD1_WARM_PERSISTENT_NUMERATOR / MD1_DENOMINATOR / 1.3646784205e-4
            ),
            "cold_instance_persistent_share": MD1_PERSISTENT_TERMINAL_SHARE,
            "two_instance_spread_pp": (
                100.0 * (MD1_PERSISTENT_TERMINAL_SHARE - MD1_WARM_PERSISTENT_TERMINAL_SHARE)
            ),
            "warm_born_absent_from_cold_own_peak": 0.453982167145232,
            "warm_born_absent_from_cold_same_step_range": [0.3769, 0.6480],
            "reading": (
                "the persistent SHARE transfers across the optimizer lever to within 3.0 pp, but "
                "the SITE SETS the two cells break do not: 37.7-64.8% of the warm cell's born "
                "sites are absent from the cold cell's at every identical step"
            ),
        },
        residual=abs(MD1_WARM_PERSISTENT_TERMINAL_SHARE - MD1_PREREGISTERED_PREDICTION),
        source_artifact=_LEDGER,
        measurement_method=(
            "same instrument, cadence and authority as the cold anchor, one lever; born-set overlap "
            "measured both at each cell's own d_seg peak and at ten identical steps"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=(
                "/Volumes/APDataStore/pact/ddm_md1_micro_macro"
                "/ANALYSIS_warm_transition_seed_20260902_dali.json"
            ),
            reactivation_criteria=(
                "a DIFFERENT seed at the same cadence turns two same-seed instances into a "
                "seed-independent reading; until then the share is measured on one seed"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    ng5_anchor = EmpiricalAnchor(
        anchor_id=(
            "md2_ng5_tau_band_x_continuous_objective_seed_20260902_shadow"
            "_trajectory_partition_20260905"
        ),
        measurement_utc="2026-09-05T13:46:33Z",
        inputs={
            "cell": (
                "ng5 tau band x carried duals -- the burn default of record -- seed 20260902, "
                "5,000 updates"
            ),
            "forward": "EMA shadow",
            "checkpoints": MD1_CADENCE_CHECKPOINTS,
            "sites": 6_291_456,
            "gt_authority": "DALI gt_cache_dali.pt",
            "churn_flips": DEFAULT_CHURN_FLIPS,
            "persistent_fraction": DEFAULT_PERSISTENT_FRACTION,
            "levers_vs_the_cold_control": (
                "two: the ng3 tau band [2*delta_R, delta_R] and ng4's carried margin duals; the "
                "step-0 state is bit-identical to the cold control's (init sha 991a1cc6..., and "
                "the reconstruction reproduces d_seg_hat 0.0025555929 and numerator 301,470 on "
                "both cells)"
            ),
            "tool": (
                "experiments/ddm_md1_micro_to_macro.py UNCHANGED, plus "
                "experiments/ddm_md2_persistent_site_overlap.py for the site overlap"
            ),
        },
        predicted_output={
            "prior_law": (
                "the tau band cuts the excursion peak from +21.8% to +5.2% of S_hat and ends "
                "d_seg below its start, so the over-paint that BIRTHS the persistent set is "
                "suppressed; the md2 charter predicted the persistent share falls to 40-55%"
            ),
            "preregistered_prediction": 0.475,
            "preregistered_prediction_band": [0.40, 0.55],
            "preregistered_falsifier": MD1_PREREGISTERED_FALSIFIER,
            "preregistration": (
                ".omx/research/charters"
                "/ddm_md2_persistent_partition_under_the_burn_default_20260905.md"
            ),
        },
        empirical_output={
            "terminal_weighted_wrong_site_numerator": MD2_NG5_TERMINAL_NUMERATOR,
            "persistent_weighted_wrong_site_numerator": MD2_NG5_PERSISTENT_NUMERATOR,
            "denominator": MD1_DENOMINATOR,
            "persistent_terminal_share": MD2_NG5_PERSISTENT_TERMINAL_SHARE,
            "calibration_gate_integer_residual": MD1_CALIBRATION_GATE_INTEGER_RESIDUAL,
            "terminal_d_seg_hat": MD2_NG5_TERMINAL_NUMERATOR / MD1_DENOMINATOR,
            "persistent_floor_d_seg_hat": MD2_NG5_PERSISTENT_NUMERATOR / MD1_DENOMINATOR,
            "persistent_floor_over_target": (
                MD2_NG5_PERSISTENT_NUMERATOR / MD1_DENOMINATOR / 1.3646784205e-4
            ),
            "cold_instance_persistent_floor_over_target": (
                MD1_PERSISTENT_NUMERATOR / MD1_DENOMINATOR / 1.3646784205e-4
            ),
            "live_forward_persistent_share": MD2_NG5_LIVE_PERSISTENT_TERMINAL_SHARE,
            "persistent_numerator_removed_fraction": 9_900 / 197_790,
            "created_over_repaired_ratio": 28_815 / 31_830,
            "cold_created_over_repaired_ratio": (19_380 + 34_665) / (9_075 + 15_360),
            "persistent_site_overlap_with_cold_control_jaccard": (
                MD2_NG5_VS_COLD_PERSISTENT_JACCARD
            ),
            "persistent_site_overlap_within_pool_chance_jaccard": (
                MD2_NG5_VS_COLD_PERSISTENT_JACCARD_WITHIN_POOL_CHANCE
            ),
            "persistent_site_overlap_attainable_max_jaccard": (
                MD2_NG5_VS_COLD_PERSISTENT_JACCARD_ATTAINABLE_MAX
            ),
            "step_zero_wrong_pool_sites": MD2_STEP_ZERO_WRONG_POOL_SITES,
            "prediction_holds": False,
            "falsifier_fired": False,
            "reading": (
                "the burn default cuts the created/repaired ratio 2.2118x -> 0.9053x and ends "
                "1.00% BELOW its start, but the persistent SHARE ROSE 0.943 pp to 62.954% and the "
                "floor fell only 12.753x -> 11.671x the sub-0.12 corner; the two schedules fail on "
                "substantially the SAME sites (Jaccard 0.8069 against a within-pool chance of "
                "0.5263 and an attainable maximum of 0.9305), while the classes the run "
                "MANUFACTURES overlap at only 0.096-0.360"
            ),
        },
        residual=abs(MD2_NG5_PERSISTENT_TERMINAL_SHARE - 0.475),
        source_artifact=_LEDGER,
        measurement_method=(
            "md1's instrument unchanged at the same cadence and authority; the site overlap is "
            "measured on the persisted per-site class codes of both cells and reported against a "
            "null of independent draws from the shared step-0 wrong pool, never the all-sites null"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=(
                "/Volumes/APDataStore/pact/ddm_md2_persistent_under_burn_default"
                "/ANALYSIS_ng5_tau_band_seed_20260902_dali.json"
            ),
            reactivation_criteria=(
                "a cell with a DIFFERENT INITIALISATION.  All three retained wc3 seed controls "
                "declare the same initial_state sha 991a1cc6..., so varying the seed varies the "
                "data order and not the init, and cannot test whether the persistent POOL is a "
                "property of the initialisation"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    data_order_anchor = EmpiricalAnchor(
        anchor_id=(
            "md3_data_order_control_seed_20260903_shadow_trajectory_partition_20260905"
        ),
        measurement_utc="2026-09-05T14:53:21Z",
        inputs={
            "cell": (
                "the retained wc3 seed_20260903 control -- SAME initialisation (init sha "
                "991a1cc6..., bit-identical to the cold control and to ng5), schedule.seed 20260903 "
                "with a different schedule.sha256_u8, so the DATA ORDER differs and nothing else "
                "does; 5,000 updates"
            ),
            "forward": "EMA shadow",
            "checkpoints": MD1_CADENCE_CHECKPOINTS,
            "sites": 6_291_456,
            "gt_authority": "DALI gt_cache_dali.pt",
            "churn_flips": DEFAULT_CHURN_FLIPS,
            "persistent_fraction": DEFAULT_PERSISTENT_FRACTION,
            "levers_vs_the_cold_control": (
                "ZERO schedule levers; the only difference is the seeded pair order.  This is the "
                "confound md2 section 8 named as the free prerequisite to the initialisation "
                "question, and it isolates data order from every other variable"
            ),
            "tool": (
                "experiments/ddm_md1_micro_to_macro.py UNCHANGED, plus "
                "experiments/ddm_md2_persistent_site_overlap.py for the site overlap"
            ),
        },
        predicted_output={
            "prior_law": (
                "three cells on one initialisation agreed on the persistent SHARE within 4.0 pp on "
                "the shadow forward, but all three also shared one data order; the share's "
                "order-invariance was assumed, never measured"
            ),
            "preregistered_prediction": 0.62,
            "preregistered_prediction_band": [0.55, 0.65],
            "preregistered_falsifier": MD1_PREREGISTERED_FALSIFIER,
            "preregistration": (
                ".omx/research/charters"
                "/ddm_md3_different_initialisation_cell_20260905.md"
            ),
        },
        empirical_output={
            "terminal_weighted_wrong_site_numerator": MD3_DATA_ORDER_TERMINAL_NUMERATOR,
            "persistent_weighted_wrong_site_numerator": MD3_DATA_ORDER_PERSISTENT_NUMERATOR,
            "denominator": MD1_DENOMINATOR,
            "persistent_terminal_share": MD3_DATA_ORDER_PERSISTENT_TERMINAL_SHARE,
            "calibration_gate_integer_residual": MD1_CALIBRATION_GATE_INTEGER_RESIDUAL,
            "terminal_d_seg_hat": MD3_DATA_ORDER_TERMINAL_NUMERATOR / MD1_DENOMINATOR,
            "persistent_floor_d_seg_hat": (
                MD3_DATA_ORDER_PERSISTENT_NUMERATOR / MD1_DENOMINATOR
            ),
            "persistent_floor_over_target": (
                MD3_DATA_ORDER_PERSISTENT_NUMERATOR / MD1_DENOMINATOR / 1.3646784205e-4
            ),
            "live_forward_persistent_share": MD3_DATA_ORDER_LIVE_PERSISTENT_TERMINAL_SHARE,
            "persistent_site_overlap_with_cold_control_jaccard": (
                MD3_DATA_ORDER_VS_COLD_PERSISTENT_JACCARD
            ),
            "persistent_site_overlap_intersection_sites": (
                MD3_DATA_ORDER_VS_COLD_PERSISTENT_INTERSECTION_SITES
            ),
            "schedule_lever_jaccard_for_comparison": MD2_NG5_VS_COLD_PERSISTENT_JACCARD,
            "start_ladder_pool_jaccard_range": MD3_START_LADDER_POOL_JACCARD_RANGE,
            "unreachable_wrong_at_all_comparable_starts": (
                MD3_UNREACHABLE_WRONG_AT_ALL_COMPARABLE_STARTS
            ),
            "reached_wrong_at_all_comparable_starts": (
                MD3_REACHED_WRONG_AT_ALL_COMPARABLE_STARTS
            ),
            "prediction_holds": True,
            "falsifier_fired": False,
            "reading": (
                "data order alone leaves the persistent share at 61.606% -- a FOURTH instance, all "
                "four within 1.35 pp on the shadow -- and moves the persistent SITES LESS than the "
                "two-lever schedule does (Jaccard 0.8536 against ng5's 0.8069 on the same cold "
                "control).  The floor is 12.734x the sub-0.12 corner against the cold control's "
                "12.753x.  ONE reading does not carry over: the LIVE-forward share is 40.562% "
                "against the 35.3-35.8% md2 recorded across three cells, so the live forward is "
                "materially sensitive to data order where the shadow is not, and md2's "
                "live-forward cross-cell agreement was an agreement across cells that shared an "
                "order.  The shipped object is the shadow, so no verdict moves"
            ),
        },
        residual=abs(MD3_DATA_ORDER_PERSISTENT_TERMINAL_SHARE - 0.62),
        source_artifact=".omx/research/ddm_md3_different_initialisation_cell_20260905.md",
        measurement_method=(
            "md1's instrument unchanged at the same cadence and authority; the overlap is measured "
            "on the persisted per-site class codes of both cells.  Both cells share the step-0 "
            "wrong pool by construction -- md1's sweep reads step 0 from "
            "config['initial_state']['path'] -- so the within-pool null applies to this Jaccard "
            "exactly as it did to md2's"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=(
                "/Volumes/APDataStore/pact/ddm_md3_different_initialisation"
                "/ANALYSIS_data_order_control_seed_20260903_dali.json"
            ),
            reactivation_criteria=(
                "the sealed different-START cell "
                "seed_20260902_different_init_r10_live_control_native100 (re-rooted config sha "
                "8b726db1..., queue spec 6a147cc2...), whose initial_state points at r10's "
                "stage-03 LIVE weights instead of its EMA shadow.  md3 measured that a different "
                "random INITIALISATION is not purchasable: build_initial_state takes no seed, and "
                "the vehicle's random init is packet.initialize_params(20260827) behind the whole "
                "qbt1 r1..r10 chain, so pricing that chain's wall clock is the prerequisite to any "
                "init-search charter"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id="checkpoint_trajectory_error_partition_v1",
        name="Checkpoint-trajectory error partition and the optimizer-reachability floor",
        one_line_summary=(
            "Five trajectory classes whose integer HT numerators sum to the distortion EXACTLY; "
            "the persistent share is a cadence-conditional CEILING on optimizer/schedule credit."
        ),
        latex_form=(
            r"d(t)=\frac{1}{NHW}\sum_p w_p n_{\mathrm{wrong}}(p,t)"
            r"=\sum_{c\in\mathcal{C}} d_c(t),\quad "
            r"d_{\mathrm{floor}}=d_{\mathrm{PERSISTENT}}(T-1)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.checkpoint_trajectory_error_partition_20260904"
            ":reachability_floor"
        ),
        domain_of_validity={
            "included": [
                "a distortion term that is a mean over sites of an exact 0/1 correctness "
                "indicator, estimated with INTEGER-weighted Horvitz-Thompson pair weights",
                "a run that retained a weight state at a fixed checkpoint period, so the "
                "trajectory can be reconstructed exactly rather than logged",
                "reading the persistent share as a CEILING on optimizer/schedule credit",
            ],
            "excluded": [
                "transferring a measured persistent SHARE across cadences -- the classes are "
                "cadence-conditional by construction and a coarser cadence moves mass out of CHURN",
                "reading the floor as a prediction that any lever reaches it",
                "a float-tolerance calibration gate in place of the integer identity",
                "use as a score, a promotion claim, or a d_seg / d_pose / rate lever",
                "non-integral sample weights, for which the exact-integer identity does not hold",
            ],
            "measurement_axis": [_AXIS],
            "result_type": (
                "APPARATUS / measurement law; NON-PROMOTABLE; moves no pointer"
            ),
            "sister_laws": [
                "scalar_top1_top2_margin_is_exact_distance_to_flip_v1 -- the per-site scalar "
                "whose sign is the correctness indicator this law partitions",
                "muon_finisher_schedule_warmstart_and_lr_anneal_v1 -- the transition lever whose "
                "credit this law bounds",
                "persistence_topology_cldice_betti_island_recall_v1 -- persistence in the "
                "TOPOLOGICAL sense; this law is persistence in the TRAJECTORY sense",
            ],
            "known_boundary": (
                "one vehicle, ONE INITIALISATION, four cells (cold control, ng1 warm transition, "
                "ng5 tau-band x carried-duals burn default, md3 data-order control), two forwards "
                "each.  The generic statement is DERIVED from the integer identity.  The SIZE of "
                "the persistent share has FOUR measured instances agreeing within 1.35 pp on the "
                "shadow forward once the outlier is named (62.011% cold, 62.954% ng5, 61.606% "
                "md3 data-order; 59.009% warm is the low leg), and the persistent SITES overlap "
                "the cold control at Jaccard 0.8069 (schedule lever, md2) and 0.8536 (data order, "
                "md3) -- so both schedule-invariance and DATA-ORDER-invariance are now measured on "
                "the shadow forward.  The LIVE forward is NOT order-invariant: md3 measured "
                "40.562% against the 35.3-35.8% of the three cells that shared an order, so the "
                "earlier live-forward agreement was an agreement across one data order and must "
                "not be transferred.  INIT-invariance remains UNMEASURED, and md3 established that "
                "it is not cheaply purchasable: build_initial_state takes no seed (it copies r10's "
                "stage-03 EMA shadow verbatim) and the vehicle's random init is "
                "packet.initialize_params(20260827) behind the entire qbt1 r1..r10 chain.  What "
                "md3 could measure is the STEP-0 half: across eight legitimately-produced starts "
                "the step-0 wrong pool moves (Jaccard 0.9909 -> 0.1175 against the incumbent), so "
                "the pool is NOT fixed -- yet 83.13% of the incumbent's persistent sites are wrong "
                "at all four comparable-quality starts against 19.78% of the sites the optimizer "
                "did reach.  The trajectory half of that statement still needs a burned cell from "
                "a different start."
            ),
        },
        units_in={
            "terminal_numerator": "weighted_wrong_site_count_integer",
            "persistent_numerator": "weighted_wrong_site_count_integer",
            "denominator": "population_n_times_sites_per_pair",
        },
        units_out={
            "persistent_floor": "distortion_fraction",
            "optimizer_reachable_share": "dimensionless_fraction",
            "persistent_share": "dimensionless_fraction",
        },
        empirical_anchors=(md1_anchor, warm_anchor, ng5_anchor, data_order_anchor),
        predicted_vs_empirical_residual={
            "md1_qbr1_cold_control_seed_20260902_shadow_trajectory_partition_20260904": (
                abs(MD1_PERSISTENT_TERMINAL_SHARE - MD1_PREREGISTERED_PREDICTION)
            ),
            "md1_ng1_warm_transition_seed_20260902_shadow_trajectory_partition_20260904": (
                abs(MD1_WARM_PERSISTENT_TERMINAL_SHARE - MD1_PREREGISTERED_PREDICTION)
            ),
            (
                "md2_ng5_tau_band_x_continuous_objective_seed_20260902_shadow"
                "_trajectory_partition_20260905"
            ): abs(MD2_NG5_PERSISTENT_TERMINAL_SHARE - 0.475),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER,),
        canonical_producers=("experiments/ddm_md1_micro_to_macro.py", _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="checkpoint_trajectory_error_partition.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
