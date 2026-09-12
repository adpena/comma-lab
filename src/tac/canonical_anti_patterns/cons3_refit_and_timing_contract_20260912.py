# SPDX-License-Identifier: MIT
"""Three anti-patterns from the arms that closed late on 2026-09-11 (ddm_cons3).

All three are MEASURED, and each one blocked or misdirected real work on the day it was found.

1. ``refit_rungs_ranked_by_conditioned_mass_v1`` (ddm_tmx1 refuting ddm_hpr1's audit rule).
   hpr1's staleness audit ranked refit rungs by **the byte mass the stale state PRICES**, and on
   that rule the tail mixer was #1 on the board: *"60 B of state prices 118,511 B."*  tmx1 fired
   that rung and it LOSES.  The two sections condition the SAME 118,511 B stream; what separates
   them is the fitted capacity each holds: 12,262 B of HPAC prior repaid -887 B when refit, 40 B
   of mixer repays +20 B -- the wrong sign, on two bases, at six exact encodes.  The mixer's whole
   measured value when FRESH was 628 B over 117,964,800 symbols; its drift is below the resolution
   of any estimator that must be fit on the same field it codes.  The cure is tmx1's corrected
   ranking quantity -- fitted capacity against the noise in the objective -- and cons3 registers it
   as the anti-pattern's unwind path at n = 2, never as an established law.

2. ``noisy_local_proxy_projects_a_gate_on_a_measured_quantity_v1`` (ddm_pr19).
   The pre-fire decode-risk gate projected a candidate's T4 wall-clock from a LOCAL cold-decode
   ratio.  The local instrument spreads **2.83x** across cold n600 windows of ONE identity class
   (819.69 s to 2,317.38 s), including 1,957.60 s and 1,619.12 s on *identical* move-44 bytes, and
   the gate's ``max(0, ratio - 1)`` clamp is one-sided, so that noise can only ever INFLATE the
   projection.  It refused the move-48 candidate at a projected 1,367.77 s -- a candidate that
   measures 1,232.418725255 s on the real device.  This is the dwc1 genus at the level of an
   instrument: resolving an effect near zero with a proxy whose own spread is 183 %.

3. ``contract_code_landed_without_its_ledger_row_v1`` (ddm_pr19's landing, measured by ddm_hpr1).
   pr19 landed its implementation (``a9fd12704``, 17:00) and HELD the matching contract rows,
   because appending them invalidates every in-flight pre-fire intent under pr17's NO-GRACE rule.
   Both halves of that choice are defensible and together they closed the door completely: the
   frozen-contract check compares the LIVE file against the owner commit, so with the code landed
   and the row held, ``PREFIRE_CONTRACT_DRIFT_REFUSED`` fired and **no new intent could be emitted
   at all**.  A sister arm carrying a pointer-move candidate hit it with everything else built.
   The hold protects an intent already emitted; it does not let a new one be created.

The frontier is UNMOVED by this module: ``composition S 0.13638261682704697 @ 179,111 B
[contest-CUDA T4 n600] (move 48)``.  Every number is quoted from the arm that measured it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_DIAGNOSIS,
    PARADIGM_DISCIPLINE,
    PARADIGM_OBSERVABILITY,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_HIGH,
    SEVERITY_OBSERVED_MEDIUM,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_UTC = "2026-09-12T00:00:00Z"

_TMX1_MEMO = ".omx/research/ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911.md"
_HPR1_AUDIT = ".omx/research/ddm_hpr1_staleness_audit_20260911.md"
_HPR1_WALL = ".omx/research/ddm_hpr1_composition_and_inheritance_wall_20260911.md"
_PR19_MEMO = ".omx/research/ddm_pr19_risk_gate_identity_class_and_chain_inheritance_20260911.md"
_MOVE48_MEMO = (
    ".omx/research/ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911_"
    "pointer_move_48_20260911.md"
)

# MEASURED (class 1): the two sections, the one stream they both condition.
HPAC_FITTED_STATE_BYTES = 12_262
HPAC_REFIT_DELTA_B = -887
MIXER_FITTED_STATE_BYTES = 40
MIXER_REFIT_DELTA_B = 20
CONDITIONED_STREAM_BYTES = 118_511
MIXER_FRESH_VALUE_BYTES = 628
SYMBOLS = 117_964_800
ADMIT_DELTA_B = -30.04

# MEASURED (class 2): the local instrument, and the device it was projecting onto.
LOCAL_COLD_WINDOW_SECONDS = (819.69, 982.77, 1_619.12, 1_957.60, 2_317.38)
LOCAL_SPREAD_RATIO = 2.83
T4_CLASS_LEGS_SECONDS = (1_232.418725255, 1_140.8051148040001)
T4_CLASS_SPREAD_FRACTION = 0.0803
LEGACY_PROJECTION_SECONDS = 1_367.77
POLICY_LIMIT_SECONDS = 1_260.0
HARD_TIMEOUT_SECONDS = 1_800.0
STRESS_PROJECTION_SECONDS = 1_477.61
OBSERVED_LOCAL_FRACTION = 0.198951
RISK_MODE_ID = "measured_t4_identity_class_envelope"
RISK_MODE_DEFINITION = "tac.candidate_seal.validate_prefire_risk.identity_class_envelope.v1"
RECEIVER_BEHAVIOUR_DIGEST = (
    "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890"
)
BEHAVIOUR_DIGEST_ROWS = 49

# MEASURED (class 3): the drift the held row produced.
PR19_CODE_COMMIT = "a9fd12704"
AMENDMENT_IN_FORCE = "79a50df080a6b67bcc6d0bc63b63d237fd4ac6f8"
AMENDMENT_ROWS_TOTAL = 15
AMENDMENT_ROWS_DRIFTING = 2


def _provenance(sidecar: str, reactivation: str, axis: str) -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=reactivation,
        measurement_axis=axis,
        hardware_substrate="macOS CPU host + Tesla T4 (contest axis) + real coder/receiver loop",
        captured_at_utc=_UTC,
    )


def build_refit_rungs_ranked_by_conditioned_mass_v1() -> AntiPattern:
    """Ranking refit rungs by the byte mass they condition, refuted at the top of the board."""
    anti_pattern_id = "refit_rungs_ranked_by_conditioned_mass_v1"
    provenance = _provenance(
        _TMX1_MEMO,
        (
            "the REPLACEMENT ranking quantity -- fitted capacity against the noise in the "
            "objective that fits it -- is itself at n = 2 (hpac 12,262 B pays -887 B; mixer 40 B "
            "pays +20 B) and is NOT a law. A third section's refit is what would make it one; the "
            "`semantic` member (29,862 B) is the queued candidate and its staleness cannot yet be "
            "dated. Until then, rank however you like and PRICE by a real encode"
        ),
        "[macOS-CPU advisory; exact bytes, scorer-free]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="tmx1_mass_ranked_top_rung_returns_the_wrong_sign_20260911",
        measurement_method=(
            "fire the rung the mass-ranking rule put FIRST: four independent fits of the 40 "
            "counted int8 mixer weights on two bases with different HPAC priors, four frame-parity "
            "held-out folds, and six exact 600-frame encodes with the real coder over "
            f"{SYMBOLS:,} symbols, twin encodes agreeing, against two byte-identical live-loop "
            "controls"
        ),
        empirical_artifact_path=_TMX1_MEMO,
        empirical_output={
            "rule_under_test": (
                "rank refit rungs by the byte mass the stale state PRICES -- the audit's words: "
                f"'60 B of state prices {CONDITIONED_STREAM_BYTES:,} B'"
            ),
            "rank_this_rule_assigned": 1,
            "measured_exact_delta_b": {
                "best_of_four_fits": MIXER_REFIT_DELTA_B,
                "worst_of_four_fits": 203,
                "bar_it_had_to_beat": ADMIT_DELTA_B,
            },
            "the_section_the_rule_ranked_below_it": {
                "hpac_fitted_state_bytes": HPAC_FITTED_STATE_BYTES,
                "hpac_measured_delta_b": HPAC_REFIT_DELTA_B,
            },
            "what_is_identical_between_them": (
                f"the conditioned stream: {CONDITIONED_STREAM_BYTES:,} B in both cases"
            ),
            "what_differs": (
                f"fitted capacity: {HPAC_FITTED_STATE_BYTES:,} B of state against "
                f"{MIXER_FITTED_STATE_BYTES} B -- 306.6x (DERIVED from the two measured sizes)"
            ),
            "why_40_bytes_cannot_be_stale": (
                f"the mixer's entire measured value when FRESH was {MIXER_FRESH_VALUE_BYTES} B "
                f"over {SYMBOLS:,} symbols (549 B tc1 + 79 B tc3); its drift is below the "
                "resolution of any estimator fit on the same field it codes"
            ),
            "the_in_sample_mirage": (
                "the search DOES find -90.2 +- 25.4 B (move 47) and -98.2 +- 23.9 B (move 48) in "
                "sample at |t| > 3.5, and it reverses to about +55 B held out in all four fold "
                "directions; the real coder then prices it at +22 B and +20 B"
            ),
            "prior_measurement_that_pointed_here": (
                "tc3's variant B jointly refit all 35 + 5 on the move-40 field and returned 37 B "
                "against variant A's 79 B with the 35 FROZEN -- confounded, but pointing at tens "
                "of bytes rather than hundreds, and pointing the right way"
            ),
        },
        # DERIVED: the rule predicted the largest repayment on the board; the best exact row missed
        # the admit bar by (20 - (-30.04)) = 50.04 B, i.e. 1.67x the bar, with the sign reversed.
        falsification_residual=abs(MIXER_REFIT_DELTA_B - ADMIT_DELTA_B) / abs(ADMIT_DELTA_B),
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
        severity_observed=SEVERITY_OBSERVED_MEDIUM,
        operator_routable_unwind_path=(
            "rank by the FITTED CAPACITY a section holds, not by the mass it conditions, and "
            "price every rung by a real encode before it is claimed"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A queue of refit rungs is ordered by the byte mass each stale section CONDITIONS, so "
            "a tiny section sitting in front of a large stream ranks first. The quantity has no "
            "measured relationship to what a refit repays: 12,262 B of fitted HPAC prior repaid "
            "-887 B on a 118,511 B stream, while 40 B of fitted mixer state on the SAME stream "
            "repaid +20 B -- the wrong sign, on two independent bases, at six exact encodes. A "
            "section too small to hold a field change cannot have been made wrong by one."
        ),
        forbidden_pattern_predicate=(
            "refit_queue.rank_key == conditioned_stream_bytes AND "
            "rung.fitted_state_bytes << conditioned_stream_bytes AND "
            "rung.claimed_repayment is asserted from the rank rather than from a real encode"
        ),
        falsification_band={
            "hpac_fitted_state_bytes": float(HPAC_FITTED_STATE_BYTES),
            "hpac_measured_delta_b": float(HPAC_REFIT_DELTA_B),
            "mixer_fitted_state_bytes": float(MIXER_FITTED_STATE_BYTES),
            "mixer_measured_delta_b": float(MIXER_REFIT_DELTA_B),
            "conditioned_stream_bytes_common_to_both": float(CONDITIONED_STREAM_BYTES),
            "fitted_capacity_ratio_derived": float(
                HPAC_FITTED_STATE_BYTES / MIXER_FITTED_STATE_BYTES
            ),
            "support_n_sections": 2.0,
        },
        recurrence_conditions=(
            "a staleness audit produces a ranked board and the rank is read as a prediction",
            "a section's state is small enough that its fresh value is below the estimator's noise",
            "an offline surrogate is fit and evaluated on the same field the coder will code",
            "a rung is claimed from its rank before any real encode has priced it",
        ),
        canonical_source_anchor=f"{_TMX1_MEMO} section 7, and {_HPR1_AUDIT} (the refuted rule)",
        canonical_unwind_path=(
            "Rank refit rungs by the FITTED CAPACITY the section holds against the noise in the "
            "objective that fits it -- not by the mass it conditions. State that this replacement "
            "is at n = 2 (one positive, one negative) and therefore a PROPOSAL: it authorises "
            "nothing. Price every rung by a real encode on the live base, with held-out folds, "
            "and read the in-sample gain as an upper bound that transfer will take back (measured "
            "here: 112 B and 118 B of a ~94 B gain)."
        ),
        canonical_producers=(
            "experiments/ddm_tmx1_mixer_price.py",
            _TMX1_MEMO,
        ),
        canonical_consumers=(
            _HPR1_AUDIT,
            _TMX1_MEMO,
            ".omx/research/ddm_dpi1_hpac_depth_state_restored_warm_start_charter_20260912.md",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_MEDIUM,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_noisy_local_proxy_projects_a_gate_v1() -> AntiPattern:
    """A gate projected from a proxy whose own spread swamps the effect it is resolving."""
    anti_pattern_id = "noisy_local_proxy_projects_a_gate_on_a_measured_quantity_v1"
    provenance = _provenance(
        _PR19_MEMO,
        (
            "the identity-class envelope is NOT closed: the ceiling is the max over DECLARED legs, "
            "so a producer declaring only the fastest member of the class gets a lower ceiling "
            "than the class warrants (1,140.81 s instead of 1,232.42 s here). Closing it needs a "
            "registry of every retained t4_direct leg, which the contract does not have. "
            "Re-measure when such a registry exists, or when the class gains a third real leg"
        ),
        "[exact contract arithmetic; measured receipts, scorer-free]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="pr19_local_decode_ratio_spreads_2p83x_within_one_class_20260911",
        measurement_method=(
            "collect every cold n600 decode window of ONE identity class (same receiver behaviour "
            f"digest {RECEIVER_BEHAVIOUR_DIGEST[:16]}... over {BEHAVIOUR_DIGEST_ROWS} rows, same "
            "decoded_token_sha256 a92e7d90...) on the local host and on T4, and compare the "
            "spread of each instrument against the effect the gate is trying to resolve"
        ),
        empirical_artifact_path=_PR19_MEMO,
        empirical_output={
            "local_cold_n600_windows_seconds": list(LOCAL_COLD_WINDOW_SECONDS),
            "local_spread_ratio": LOCAL_SPREAD_RATIO,
            "two_of_them_are_identical_bytes": (
                "1,957.60 s and 1,619.12 s are the SAME move-44 archive -- 1.21x from the "
                "instrument alone"
            ),
            "clamp_is_one_sided": (
                "max(0, ratio - 1): noise can only ever INFLATE the projection, never deflate it"
            ),
            "t4_class_legs_seconds": list(T4_CLASS_LEGS_SECONDS),
            "t4_spread_fraction": T4_CLASS_SPREAD_FRACTION,
            "t4_spread_is_the_host_not_the_payload": (
                "move 46 consumed MORE coded bits (960,913 vs 958,049, +0.30 %) and took LESS "
                "time, and EVERY stage moved down together -- including the render stage, which "
                "is payload-independent once the token plane is identical"
            ),
            "what_the_legacy_projection_did": (
                f"projected {LEGACY_PROJECTION_SECONDS} s and refused a candidate that measures "
                f"{T4_CLASS_LEGS_SECONDS[0]} s on the real device"
            ),
            "the_delta_that_must_not_be_declared_absent": (
                "move 47 -> move-48 candidate, cold, same host, same day: archive_setup -0.3 %, "
                "frame0 -0.2 %, render +2.8 %, token_decode +28.4 %. Three stages flat and one up "
                "is not a host-factor signature; it is not attributable either, so 'fraction 0' "
                "is refused as firmly as the noisy projection is"
            ),
            "the_cure_measured": {
                "mode": RISK_MODE_ID,
                "definition": RISK_MODE_DEFINITION,
                "ceiling_seconds": T4_CLASS_LEGS_SECONDS[0],
                "policy_limit_seconds": POLICY_LIMIT_SECONDS,
                "spare_against_policy_seconds": 27.581274745,
                "stress_seconds": STRESS_PROJECTION_SECONDS,
                "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
                "spare_against_hard_timeout_seconds": 322.39,
                "stronger_in_five_places": (
                    "identical receiver required (the old mode admits a delta); token-plane "
                    "identity required; coded-bit and archive-byte domination required; "
                    "diagnostics bound to trees rather than to a bare stored digest; and an "
                    "1,800 s stress test that did not exist at all before"
                ),
            },
        },
        # DERIVED: how far the noisy projection sat from the real measurement it was standing in for.
        falsification_residual=(
            (LEGACY_PROJECTION_SECONDS - T4_CLASS_LEGS_SECONDS[0]) / T4_CLASS_LEGS_SECONDS[0]
        ),
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "project from the MAX of real measurements of the candidate's identity class, and "
            "re-aim the noisy proxy at the hard timeout instead of deleting it"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A gate projects a quantity it could measure, using a proxy on a different device "
            "whose own spread is larger than the effect. pr19 measured the local cold-decode "
            "instrument at a 2.83x spread within ONE identity class -- 1.21x of it between two "
            "runs of identical bytes -- against a gate trying to resolve a near-zero difference, "
            "with a one-sided clamp that can only inflate. The opposite cure is equally forbidden: "
            "declaring the observed delta ABSENT (fraction 0) when three stages are flat and one "
            "is +28.4 %, which is not a host signature and cannot be attributed either way."
        ),
        forbidden_pattern_predicate=(
            "gate.projection_source == local_proxy AND "
            "proxy.within_class_spread > effect_being_resolved AND "
            "(gate.clamp is one-sided OR gate.declares_delta_absent)"
        ),
        falsification_band={
            "local_within_class_spread_ratio": LOCAL_SPREAD_RATIO,
            "identical_bytes_spread_ratio": 1.209,
            "t4_within_class_spread_fraction": T4_CLASS_SPREAD_FRACTION,
            "legacy_projection_seconds": LEGACY_PROJECTION_SECONDS,
            "measured_ceiling_seconds": T4_CLASS_LEGS_SECONDS[0],
            "policy_limit_seconds": POLICY_LIMIT_SECONDS,
            "stress_projection_seconds": STRESS_PROJECTION_SECONDS,
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        },
        recurrence_conditions=(
            "a gate must bound a quantity measured on a device the producer cannot cheaply reach",
            "the proxy's spread has never been measured WITHIN one identity class",
            "a clamp is one-sided, so instrument noise has a preferred direction",
            "the convenient fix is to zero the term rather than to re-aim it at the real limit",
        ),
        canonical_source_anchor=f"{_PR19_MEMO} -- clause table rows 1, 3 and 7",
        canonical_unwind_path=(
            "Define the candidate's IDENTITY CLASS by things that pin decode work -- identical "
            "receiver behaviour digest, identical decoded token plane, and domination on both "
            "coded bits and archive bytes -- and take the ceiling as the MAX real measurement over "
            "the class's declared legs. Keep the noisy local ratio and re-aim it: it must now "
            "prove ceiling x (1 + observed fraction) <= the HARD timeout, where a real failure "
            "lives. Never claim the observed delta is absent; price it at face value. Implemented "
            f"as {RISK_MODE_ID} ({RISK_MODE_DEFINITION})."
        ),
        canonical_producers=("src/tac/candidate_seal.py", "src/tac/decode_wall_clock.py"),
        canonical_consumers=(_PR19_MEMO, _MOVE48_MEMO, _HPR1_WALL),
        paradigm_class=PARADIGM_OBSERVABILITY,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_contract_code_landed_without_its_ledger_row_v1() -> AntiPattern:
    """Landing contract CODE while holding its ledger ROW closes the door on every producer."""
    anti_pattern_id = "contract_code_landed_without_its_ledger_row_v1"
    provenance = _provenance(
        _HPR1_WALL,
        (
            "reopen when the frozen contract gains a grace window, a staged-amendment mode, or a "
            "documented emission freeze -- any of which would let code and row land at different "
            "times without closing emission. Until then the two must land together, and the "
            "sequencing of that landing is MAIN's, not a producer arm's"
        ),
        "[exact contract arithmetic; measured receipts, scorer-free]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="pr19_code_landed_row_held_blocked_move48_emission_20260911",
        measurement_method=(
            "attempt a real pre-fire intent emission on the live tree after the contract's "
            "implementation commit landed and its amendment rows were deliberately held; record "
            "the refusal, the amendment in force, and which of its implementation rows drift"
        ),
        empirical_artifact_path=_HPR1_WALL,
        empirical_output={
            "refusal": (
                "PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob: "
                "src/tac/candidate_seal.py"
            ),
            "amendment_in_force": AMENDMENT_IN_FORCE,
            "implementation_rows": AMENDMENT_ROWS_TOTAL,
            "rows_drifting": AMENDMENT_ROWS_DRIFTING,
            "only_commit_touching_either": PR19_CODE_COMMIT,
            "blob_pairs": {
                "src/tac/candidate_seal.py": ["707b520757...", "d47fed27d6... (LIVE)"],
                "src/tac/decode_wall_clock.py": ["ce1f8c7b48...", "e8040dfa2d... (LIVE)"],
            },
            "why_the_hold_did_not_help": (
                "the hold protects an intent already EMITTED; _pf_blob compares the LIVE file "
                "against the owner commit, so with the code landed and the append held, no NEW "
                "intent can be created at all"
            ),
            "why_the_row_was_held": (
                "appending invalidates every in-flight pre-fire intent under pr17's NO-GRACE "
                "rule -- both halves of the choice are defensible, which is why the cure is "
                "sequencing and not blame"
            ),
            "what_it_cost": (
                "a sister arm had its move-48 candidate built to the last step -- 18 receipts, "
                "smokes empty, receiver delta 0 rows, parse-back byte-identical -- and could not "
                "emit. The move landed later through the first-measurement route once MAIN "
                "sequenced the append; the pointer reached S 0.13638261682704697 @ 179,111 B"
            ),
            "a_second_defect_the_same_landing_exposed": (
                "the blocked arm's own risk receipt was in the mode the new code superseded "
                "(completed_t4_receiver_delta with fraction 0, from a matched pair whose "
                "candidate measured FASTER: 1,284.923 s against a 1,376.139 s base) -- exactly "
                "the shape the new rule rejects, so it would not have emitted anyway"
            ),
        },
        falsification_residual=None,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "land the contract code and its amendment row in ONE commit, sequenced by the owner "
            "of the frozen file while no intent is in flight"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "An arm lands the CODE of a frozen-contract amendment and holds the matching ledger "
            "ROW (usually to protect in-flight work from a no-grace invalidation). The "
            "frozen-file check compares the live file against the owner commit named by the "
            "amendment in force, so the held row does not delay the amendment -- it deletes the "
            "emission path. Measured: 2 of 15 implementation rows drifting, one commit "
            "responsible, and a pointer-move candidate refused at the last step with everything "
            "else built."
        ),
        forbidden_pattern_predicate=(
            "commit touches a file pinned by the frozen contract AND "
            "the matching amendment row is not appended in the same commit AND "
            "the contract validates producers by live-file-vs-owner-blob comparison"
        ),
        falsification_band={
            "implementation_rows_in_force": float(AMENDMENT_ROWS_TOTAL),
            "rows_drifting_after_the_split_landing": float(AMENDMENT_ROWS_DRIFTING),
            "commits_responsible": 1.0,
            "producers_able_to_emit_while_split": 0.0,
        },
        recurrence_conditions=(
            "a contract pins implementation files by blob hash and validates producers against it",
            "the amendment ledger has a no-grace rule that invalidates in-flight intents",
            "an arm owns the code change but not the sequencing of the ledger append",
            "a reviewer reads 'held, so nothing changed yet' as 'nothing is blocked yet'",
        ),
        canonical_source_anchor=f"{_HPR1_WALL} section 4 (Blocker 1), and {_PR19_MEMO}",
        canonical_unwind_path=(
            "Land contract code and its ledger row TOGETHER, in one commit, sequenced by the "
            "owner of the frozen file at a moment when no intent is in flight. If they must be "
            "split, the split must come with an explicit emission freeze that says so out loud, "
            "and the arms that would emit must be told -- a silent hold reads as 'safe' and is "
            "total."
        ),
        canonical_producers=(
            ".omx/research/ddm_pr19_20260911/PREFIRE_CONTRACT_AMENDMENT_ROW_PR19.json",
            ".omx/research/ddm_pr19_20260911/PREFIRE_IMPLEMENTATION_MANIFEST_PR19.json",
        ),
        canonical_consumers=(_HPR1_WALL, _PR19_MEMO, _MOVE48_MEMO),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_pr19_chain_inheritance_refusal_falsification() -> EmpiricalFalsification:
    """Appended to the EXISTING ``inherited_timing_leg_reinherited_v1``: adjudicated, not patched.

    cons2 registered that anti-pattern from ntb2's side, where the chain break was an obstacle.
    pr19 adjudicated it for the second family and REFUSED chains with no length bound -- and the
    decisive reason is not drift arithmetic but economics: **inheritance saves no dispatch.**
    """
    anti_pattern_id = "inherited_timing_leg_reinherited_v1"
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_PR19_MEMO,
        reactivation_criteria=(
            "a chain licence requires a per-step drift BOUND, and none exists: the single paired "
            "observation of the class (1,232.418725255 -> 1,140.8051148040001 s, 8.03 %) is "
            "CONFOUNDED with T4 host variance because every stage moved together. Reopen only "
            "with an unconfounded per-step distribution -- and even then, weigh it against the "
            "fact that a chain buys no dollars"
        ),
        measurement_axis="[exact contract arithmetic; measured receipts, scorer-free]",
        hardware_substrate="Tesla T4 (contest axis) + macOS CPU host",
        captured_at_utc=_UTC,
    )
    return EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="pr19_chain_inheritance_refused_on_the_real_trees_20260911",
        measurement_method=(
            "run both inheritance routes against the REAL move-47 and move-48 candidate trees and "
            "record the verbatim refusals, with a positive control that the same leg still "
            "inherits while it IS the pointer"
        ),
        empirical_artifact_path=_PR19_MEMO,
        empirical_output={
            "verdict": "REFUSED, with NO chain-length bound",
            "clauses": (
                "decode_wall_clock.py:574 (inheritance must point directly to a measured or "
                "t4_direct leg, never to an inherited one) and :578 (source measurement is not "
                "the pointer archive) together admit at most ONE inherited row per measured leg"
            ),
            "decisive_reason": (
                "a chain buys no dollars: every candidate is fired on T4 to produce its exact row "
                "either way, so a chain would only let a seal be minted BEFORE that fire, at the "
                "price of an unmeasured accumulating drift assumption"
            ),
            "no_per_step_bound_exists": (
                "n = 1 paired observation, 8.03 %, confounded with host variance; compounding it "
                "twice already exceeds the policy limit: 1,140.81 x 1.0803^2 = 1,331 s > 1,260 s"
            ),
            "the_system_self_heals": (
                "move 46 measured, move 47 inherited, move 48 measured again -- the alternating "
                "pattern costs nothing extra and keeps every timing statement within one archive "
                "step of a real measurement"
            ),
            "the_cheaper_legal_path": (
                "the first-measurement route (prefire intent -> authorization -> fire), which the "
                "identity-class envelope is what unblocks; it over-pays procedurally and "
                "under-pays in risk"
            ),
            "arms_that_hit_this_wall": ["ddm_ntb2 (move 46)", "ddm_hpr1 (move 48)"],
        },
        # DERIVED from pr19: two compounded steps of the only observed magnitude against the limit.
        falsification_residual=(1_140.8051148040001 * (1.0803**2)) / POLICY_LIMIT_SECONDS - 1.0,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_MEDIUM,
        operator_routable_unwind_path=(
            "take the first-measurement route and let the completion mint this move's own "
            "t4_direct leg; adjudicate the chain as a contract fact, never patch the clause"
        ),
    )
