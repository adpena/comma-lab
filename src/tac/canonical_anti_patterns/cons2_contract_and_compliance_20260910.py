# SPDX-License-Identifier: MIT
"""Three 2026-09-10/11 contract-and-compliance anti-patterns, each caught by a second family.

All three share a shape worth naming once: a rule that reads as satisfied to the arm that wrote it,
and fails when a DIFFERENT family reads the actual artifact.  None was found by review of a memo.

1. ``fitted_scalar_in_receiver_code_is_counted_content_v1`` -- pr8, 2026-09-10.  Move 41's receiver
   hard-coded ``LANE_CLASS = 1`` and the row band 128-319, both chosen from a census of THIS scored
   video.  The arm's own compliance line was "no per-frame fitted table", which is true and beside
   the point: **a fitted SCALAR is content**.  Rule 118 counts video-derived content wherever it
   lives; free code is the ALGORITHM, not constants read off the object.  The exact row stands
   (S 0.13758600733559048); the ARCHIVE does not qualify, so the pointer was retracted to move 40.

2. ``derived_listing_inside_identity_digest_v1`` -- pr14, 2026-09-10.  The frozen pre-fire contract
   refused its FIRST real producer.  rlc4's 180,178 B candidate was raw-identical to move 42 with
   **zero executable bytes changed**, and its normalized receiver digest still diverged -- because the
   digest included ``MANIFEST.sha256``, a DERIVED listing of RAW hashes, so the archive pins the
   normalizer had just stripped re-entered through the manifest.  The cure is SCOPE: exclude the
   listing from the BEHAVIOR digest only, validate it beside the digest, and never widen the
   exclusion into the CUSTODY digest.

3. ``inherited_timing_leg_reinherited_v1`` -- ntb2, 2026-09-11.  A decode wall-clock may be inherited
   from a measured leg once.  Move 45 inherited move 44's ``t4_direct``; move 46 then had no source
   at all -- move 45's leg is inherited (forbidden by contract text) and move 44's leg is no longer
   the pointer archive (refused).  The control that makes the refusal legible: the SAME leg with
   ``pointer_archive_sha256`` set to move 44 PASSES at 1232.418725255 s, so the refusal is about the
   POINTER, not the candidate's tree.  The cure is to heal the chain by MEASURING (the
   first-measurement path gives move 46 its own leg), never to patch the contract that blocked you.

The common unwind for all three: **the second family owns the rule, and it reads the artifact, not
the claim.**

Registered by ddm_cons2 on 2026-09-11.  ``research_only=true``; ``score_claim=false``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_PROVENANCE,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_OBSERVED_CRITICAL,
    SEVERITY_OBSERVED_HIGH,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_UTC = "2026-09-11T00:00:00Z"

_PR8_LEDGER = ".omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md"
_CPD1_LEDGER = ".omx/research/ddm_cpd1_pointer_disqualification_20260910.md"
_PR14_LEDGER = ".omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md"
_RLC4_LEDGER = ".omx/research/ddm_rlc4_resume_rebase_cure_onto_move42_prefire_intent_20260910.md"
_PR18_LEDGER = (
    ".omx/research/ddm_pr18_manifest_row_exclusion_from_receiver_behavior_digest_20260911.md"
)
_NTB2_LEDGER = ".omx/research/ddm_ntb2_non_tail_lossy_levers_with_resolve_20260911.md"

# MEASURED (pr8 / the move-41 retraction).
MOVE41_SCORE = 0.13758600733559048
MOVE41_FITTED_CONSTANTS = ("LANE_CLASS = 1", "row band 128-319")
MOVE41_INFLATE_SECONDS = 1336.7
MOVE41_SEAL_MARGIN_SECONDS = 1260
RECEIVER_CODE_DOOR_BYTES = 79  # the door stays real; it re-enters COUNTED
RLC1_COUNTED_RIDER_BYTES = 19  # the cure: the same constants, counted

# MEASURED (pr14 / rlc4 / pr18).
RLC4_CANDIDATE_BYTES = 180_178
RLC4_SAVING_BYTES = 60
RLC4_EXECUTABLE_BYTES_CHANGED = 0
CURED_DIGEST_PREFIX = "9f6e7168"
BEHAVIOR_DIGEST_SCHEMA = "measure_receiver_behavior_digest.v2"

# MEASURED (ntb2 / the timing chain).
T4_DIRECT_SECONDS = 1232.418725255
INHERITANCE_CHAIN = (
    ("44", "04758c0d", "t4_direct"),
    ("45", "145e02e2", "inherited from move 44's t4_direct"),
    ("46", "a0de607d", "none available"),
)
INHERITANCE_DEPTH_ALLOWED = 1


def _provenance(sidecar: str, reactivation: str, axis: str) -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=reactivation,
        measurement_axis=axis,
        hardware_substrate="macOS CPU (source and retained-byte inspection) + contest-CUDA T4 timing legs",
        captured_at_utc=_UTC,
    )


def build_fitted_scalar_in_receiver_code_is_counted_content_v1() -> AntiPattern:
    """A scalar chosen by looking at the scored video is CONTENT, wherever it is written."""
    anti_pattern_id = "fitted_scalar_in_receiver_code_is_counted_content_v1"
    provenance = _provenance(
        _PR8_LEDGER,
        (
            "the receiver-code door is NOT closed -- it re-enters COUNTED, fixed-point and timed "
            "under the margin (rlc1 did exactly that: the same constants in a 19-byte rider, 60 B "
            "net saving). This anti-pattern reopens only if a constant's provenance line shows it "
            "was derived WITHOUT the scored video"
        ),
        "[second-family source review of the runtime TREE; scorer-free]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="pr8_move41_lane_class_and_row_band_rule118_blocker_20260910",
        measurement_method=(
            "read the shipped runtime tree file-by-file and require a provenance line for every "
            "constant: what chose this value, and did the choice look at the scored video?"
        ),
        empirical_artifact_path=_PR8_LEDGER,
        empirical_output={
            "constants_classified_as_content": list(MOVE41_FITTED_CONSTANTS),
            "their_provenance": "rp1's census of THIS video's field",
            "the_arm_s_own_compliance_line": (
                "'no per-frame fitted table' -- true, and not the test. A fitted SCALAR is content"
            ),
            "exact_row_status": (
                f"the row STANDS at S {MOVE41_SCORE}; the ARCHIVE does not qualify, so it is not a "
                "pointer -- retracted in hot state, packet memo and ledger"
            ),
            "second_findings_in_the_same_review": {
                "inflate_seconds": MOVE41_INFLATE_SECONDS,
                "seal_margin_seconds": MOVE41_SEAL_MARGIN_SECONDS,
                "float64_geometry_branch": True,
                "contest_cpu_receipt": "absent -- the receiver refuses CPU by declaration",
            },
            "the_door_is_real": {
                "measured_saving_bytes": RECEIVER_CODE_DOOR_BYTES,
                "cure_rider_bytes": RLC1_COUNTED_RIDER_BYTES,
            },
        },
        falsification_residual=None,
        captured_at_utc="2026-09-10T00:00:00Z",
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_CRITICAL,
        operator_routable_unwind_path=(
            "COUNT the constant in the rider (a few bytes) or replace it with a rule declared "
            "without the video; then disqualify the non-qualifying row with "
            "tools/frontier_disqualify.py --reason-class rule118_content_in_code and refresh"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A receiver hard-codes a value chosen by looking at the scored video -- a class id, a "
            "row band, a bin edge, a threshold -- and the arm reports compliance because no TABLE "
            "was embedded. Rule 118 counts video-derived content wherever it lives: free code is "
            "the algorithm, not constants read off the object. Move 41 shipped LANE_CLASS = 1 and "
            "rows 128-319 from a census of this video; the exact row stands and the archive does "
            "not qualify."
        ),
        forbidden_pattern_predicate=(
            "constant_in_receiver_code AND constant.provenance_depends_on_scored_video AND "
            "constant.bytes_counted_in_archive == 0"
        ),
        falsification_band={
            "fitted_constants_shipped": float(len(MOVE41_FITTED_CONSTANTS)),
            "counted_bytes_required_to_cure": float(RLC1_COUNTED_RIDER_BYTES),
            "measured_door_saving_bytes": float(RECEIVER_CODE_DOOR_BYTES),
        },
        recurrence_conditions=(
            "an arm reports 'no fitted table' as evidence of rule-118 compliance",
            "a census of the scored video is used to pick a constant that then lives in free code",
            "a receiver change is fired on the producing arm's own compliance memo",
            "a class id or row band is treated as structure because it is small",
        ),
        canonical_source_anchor=f"{_PR8_LEDGER} (verdict) + {_CPD1_LEDGER} (the retraction mechanism)",
        canonical_unwind_path=(
            "(1) Give every receiver constant a provenance line, file:line, saying what chose it. "
            "(2) Have the OTHER model family review the runtime TREE before any receiver change "
            "fires. (3) Count the video-derived ones in the rider or replace them with a "
            "video-independent rule. (4) If a non-qualifying archive already became the pointer, "
            "retract it everywhere the pointer lives -- the exact row is still a real row."
        ),
        canonical_producers=("tools/frontier_disqualify.py", "tools/refresh_canonical_frontier.py"),
        canonical_consumers=(
            _PR8_LEDGER,
            _CPD1_LEDGER,
            ".omx/research/ddm_rlc1_rule118_cure_20260910.md",
        ),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_CRITICAL,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_derived_listing_inside_identity_digest_v1() -> AntiPattern:
    """A derived hash listing is never receiver content -- keep it out of a BEHAVIOR digest."""
    anti_pattern_id = "derived_listing_inside_identity_digest_v1"
    provenance = _provenance(
        _PR14_LEDGER,
        (
            "re-audit the digest's file list whenever a new identity comparison is introduced: of "
            "each file ask 'is this bytes-that-execute, or a listing ABOUT bytes?'. The exclusion "
            "is versioned and scoped to the BEHAVIOR digest; widening it into the CUSTODY digest "
            "is the pr14 dead-end and reopens nothing"
        ),
        "[second-family review; retained-byte re-derivation; scorer-free]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="pr14_rlc4_manifest_in_normalized_receiver_digest_20260910",
        measurement_method=(
            "re-derive both endpoints' normalized receiver digests over RETAINED bytes with the "
            "scoped function -- never re-timestamp, never re-run -- and require the reference and "
            "the candidate to agree from disk"
        ),
        empirical_artifact_path=_PR14_LEDGER,
        empirical_output={
            "producer_refused": "rlc4, the contract's FIRST real producer",
            "candidate_bytes": RLC4_CANDIDATE_BYTES,
            "saving_bytes": RLC4_SAVING_BYTES,
            "executable_bytes_changed": RLC4_EXECUTABLE_BYTES_CHANGED,
            "raw_output_identity": "byte-identical to move 42's complete cold public output",
            "mechanism": (
                "the normalizer strips archive pins from inflate.py, but MANIFEST.sha256 lists the "
                "RAW file hashes -- so a legitimate pin change altered the manifest and the pins "
                "re-entered the digest through a derived file"
            ),
            "cure_scope": (
                f"{BEHAVIOR_DIGEST_SCHEMA} excludes MANIFEST.sha256 from the BEHAVIOR digest and "
                "validates the listing independently; every custody/identity check keeps it"
            ),
            "both_endpoints_agree_after_cure_prefix": CURED_DIGEST_PREFIX,
            "why_it_had_not_fired_before": (
                "earlier moves passed the inherit path ONLY because their manifests were stale; "
                "honest regeneration closed it for everyone at once -- two candidates (pc3, ntb2) "
                "hit the same wall within hours"
            ),
        },
        falsification_residual=None,
        captured_at_utc="2026-09-10T00:00:00Z",
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "scope the exclusion to a NEW versioned behavior digest, validate the listing beside "
            "it (a stale listing must refuse), and add a test that a one-byte NON-PIN change in "
            "the executable still refuses"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "An identity digest meant to prove BEHAVIOR includes a derived listing of RAW hashes "
            "(a MANIFEST.sha256), so the very bytes the normalizer strips re-enter through the "
            "listing and two behaviourally identical trees refuse each other. rlc4's candidate was "
            "raw-identical to move 42 with zero executable bytes changed and was refused."
        ),
        forbidden_pattern_predicate=(
            "digest.purpose == 'behavior_identity' AND digest.files contains a derived listing of "
            "raw content hashes (MANIFEST.sha256 or equivalent)"
        ),
        falsification_band={
            "executable_bytes_changed_at_refusal": float(RLC4_EXECUTABLE_BYTES_CHANGED),
            "candidate_saving_bytes_blocked": float(RLC4_SAVING_BYTES),
            "producers_blocked_before_cure": 3.0,
        },
        recurrence_conditions=(
            "a normalizer strips a field that a derived listing in the same digest still carries",
            "two correct second-family requirements meet on a derived object",
            "a manifest is regenerated honestly and a previously passing inherit path starts refusing",
            "an exclusion is widened from the behavior digest into the custody digest to 'fix' it",
        ),
        canonical_source_anchor=f"{_PR14_LEDGER} + {_RLC4_LEDGER} (the refusal) + {_PR18_LEDGER} (the cure)",
        canonical_unwind_path=(
            "List every file in the digest and classify each as bytes-that-execute or a listing "
            "ABOUT bytes. Listings get a versioned, SCOPED exclusion from the behavior digest and "
            "an independent staleness validation beside it. Custody digests keep everything. "
            "Re-derive both endpoints over retained bytes; legacy-equal must still imply "
            "behavior-equal so no existing refusal is weakened."
        ),
        canonical_producers=("src/tac/candidate_seal.py",),
        canonical_consumers=(_PR14_LEDGER, _PR18_LEDGER, _RLC4_LEDGER),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_inherited_timing_leg_reinherited_v1() -> AntiPattern:
    """An inherited decode wall-clock cannot itself be inherited: heal the chain by measuring."""
    anti_pattern_id = "inherited_timing_leg_reinherited_v1"
    provenance = _provenance(
        _NTB2_LEDGER,
        (
            "this is a CONTRACT FACT, not an S-arithmetic law: it reopens only through a "
            "deliberate second-family amendment deciding how far one timing measurement may "
            "cover -- e.g. following an inheritance CHAIN to its terminating t4_direct when every "
            "link shares one receiver behavior digest. Nothing here licenses an arm blocked by it "
            "to patch it"
        ),
        "[contract text + retained seal legs; contest-CUDA T4 timing]",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="ntb2_move46_timing_inheritance_chain_break_20260911",
        measurement_method=(
            "run the seal's inheritance leg twice -- once with the live pointer archive and once "
            "with the move-44 archive -- so the refusal's CAUSE is measured rather than argued"
        ),
        empirical_artifact_path=_NTB2_LEDGER,
        empirical_output={
            "chain": [
                {"move": m, "archive_prefix": a, "leg_mode": mode} for m, a, mode in INHERITANCE_CHAIN
            ],
            "door_1_move45_leg_as_source": (
                "forbidden by contract text -- 'inheritance must point directly to a measured or "
                "t4_direct leg (never to an inherited one)'"
            ),
            "door_2_move44_leg_as_source": "REFUSED -- source measurement is not the pointer archive",
            "control_that_makes_it_legible": (
                f"the SAME leg with pointer_archive_sha256 set to move 44 PASSES at "
                f"{T4_DIRECT_SECONDS} s, so the refusal is about the POINTER, not the tree"
            ),
            "blast_radius": "every move-46 candidate, pc3's rebase included -- not just ntb2's",
            "resolution_taken": (
                "do NOT patch the contract and do NOT buy a timing-only Modal row (it buys no "
                "score): take the first-measurement path, so the completion gives move 46 its OWN "
                "t4_direct leg and move 47 inherits normally -- the chain heals itself"
            ),
            "inheritance_depth_allowed": INHERITANCE_DEPTH_ALLOWED,
        },
        falsification_residual=None,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "take the first-measurement path: carry the timing lineage in the intent, fire the "
            "real T4 decode, and let the completion mint this move's own t4_direct leg"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A seal tries to inherit a decode wall-clock from a leg that was itself inherited, or "
            "from a measured leg that is no longer the pointer archive. The contract tolerates "
            "inheriting timing measured on a different archive ONCE; the second hop has no valid "
            "source. The failure is silent in intent and total in effect -- it blocked every "
            "move-46 candidate. The cure is to MEASURE, not to widen the rule that stopped you."
        ),
        forbidden_pattern_predicate=(
            "seal.decode_wall_clock.mode == 'inherited' AND (source_leg.mode == 'inherited' OR "
            "source_leg.archive_sha256 != pointer_archive_sha256)"
        ),
        falsification_band={
            "inheritance_depth_allowed": float(INHERITANCE_DEPTH_ALLOWED),
            "t4_direct_seconds_at_the_terminating_leg": T4_DIRECT_SECONDS,
            "candidates_blocked": 2.0,
        },
        recurrence_conditions=(
            "a pointer moves on a row whose seal inherited its timing rather than measuring it",
            "an arm blocked by a contract proposes to amend the contract in the same unit",
            "a timing-only paid row is proposed to unblock a chain (it buys no score)",
            "the refusal is attributed to the candidate's tree without a pointer-swap control",
        ),
        canonical_source_anchor=(
            f"{_NTB2_LEDGER} 'the inheritance chain' + "
            ".omx/research/ddm_ntb2_20260911/TIMING_INHERITANCE_CHAIN_BREAK.json"
        ),
        canonical_unwind_path=(
            "Take the first-measurement path: the intent carries its timing lineage to the chain's "
            "terminating measurement, the real T4 decode is fired, and the completion mints this "
            "move's own t4_direct leg so the NEXT move inherits normally. Record the chain break "
            "as a contract fact for a second family to adjudicate later; patch nothing."
        ),
        canonical_producers=("tools/make_candidate_seal.py", "src/tac/candidate_seal.py"),
        canonical_consumers=(
            _NTB2_LEDGER,
            ".omx/research/ddm_ntb2_frame_even_hpac_prior_move45_first_measurement_20260911_pointer_move_46_20260911.md",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )
