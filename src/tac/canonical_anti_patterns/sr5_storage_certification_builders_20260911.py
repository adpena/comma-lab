# SPDX-License-Identifier: MIT
"""ddm_sr5 storage-custody anti-patterns: what a deletion certificate may and may not rest on.

Two classes, both MEASURED on 2026-09-11 while sr5 cleared 28.247 GiB (1,684 files, zero
refusals) off the SSD tier under exercised per-file certificates.  Both were caught by the
tool, not by review, which is the point: each cure is a REFUSAL the producer can execute.

1. ``unexercised_certificate_authorizes_deletion_v1`` -- a recorded hash is not a certificate.
   sr5 found a manifest carrying the literal ``sha256: "SKIPPED_LARGE_STREAMED"`` on two rows
   totalling 3,780,374,400 B (99.2 % of an already operator-authorized class).  That is not a
   hash: nothing binds those bytes to the manifest and no rebuild can be checked against
   anything, so the rule keeps them.  A sister class in the same adjudication: six of twelve
   checkpoint rows carry ``run_completed: false`` while claiming supersession "by the run's own
   RESULT.json" -- untrue for a run that did not finish -- so another 353,999,376 B stayed.  The
   deeper form of the rule is sr5's own sentence: **a certificate that has never been redeemed is
   a promise, not a property**, and the matching one for the gate: *a gate that has never refused
   anything is not known to be a gate*.

2. ``sampled_prefilter_decides_equality_v1`` -- a prefilter chooses what to hash, it never
   decides equality.  A 3 x 16 MiB-window prefilter over 14 ``0.raw`` files found 11 distinct
   signatures and TWO duplicate-candidate groups; the full sha256 CONFIRMED one and REFUTED the
   other.  The refuted group's three files matched on all three sampled windows and are not
   identical -- the differing bytes simply lie outside the windows.  The projection built on the
   unconfirmed prefilter was 10.233 GiB; the measured result was 3.411 GiB, a 3.0x over-claim,
   and the 3.411 GiB was released by HARDLINK with nothing deleted at all.

Both cures are in shipped tools (``tools/sr5_certify_delete_derived.py``,
``tools/sr5_certify_hardlink_dedup.py``), so these anti-patterns are enforced by a producer that
refuses, not by a memo that advises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_PROVENANCE,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_OBSERVED_HIGH,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_UTC = "2026-09-11T00:00:00Z"
_LEDGER = ".omx/research/ddm_sr5_certified_rebuildable_deletion_20260911.md"
_CERT_TOOL = "tools/sr5_certify_delete_derived.py"
_DEDUP_TOOL = "tools/sr5_certify_hardlink_dedup.py"
_DEDUP_RESULT = ".omx/research/ddm_sr5_20260911/PFS1_DEDUP_RESULT.json"

# MEASURED (class 1): the unexercised-certificate refusals.
PLACEHOLDER_HASH_LITERAL = "SKIPPED_LARGE_STREAMED"
PLACEHOLDER_ROWS = 2
PLACEHOLDER_BYTES_REFUSED = 3_780_374_400
PLACEHOLDER_SHARE_OF_CLASS = 0.992
INCOMPLETE_RUN_ROWS = 6
INCOMPLETE_RUN_ROWS_TOTAL = 12
INCOMPLETE_RUN_BYTES_REFUSED = 353_999_376
CERTIFIABLE_BYTES = 354_198_507
SR5_FILES_DELETED = 1_684
SR5_BYTES_DELETED = 30_330_292_880

# MEASURED (class 2): the prefilter that chose but did not decide.
PREFILTER_WINDOWS = 3
PREFILTER_WINDOW_BYTES = 16 * 1024 * 1024
RAW_FILES_SCANNED = 14
DISTINCT_SIGNATURES = 11
CANDIDATE_GROUPS = 2
CONFIRMED_GROUPS = 1
REFUTED_GROUP_FILES = 3
# sr5 states the projection in GiB only; do not manufacture a byte count for it.
PROJECTED_RELEASE_GIB = 10.233  # on the UNCONFIRMED prefilter
MEASURED_RELEASE_BYTES = 3_662_409_600  # 3.411 GiB, released by hardlink, nothing deleted


def _provenance(reactivation: str) -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=reactivation,
        measurement_axis="[macOS filesystem custody; scorer-free]",
        hardware_substrate="macOS + SSD tiers (VertigoDataTier, APDataStore)",
        captured_at_utc=_UTC,
    )


def build_unexercised_certificate_authorizes_deletion_v1() -> AntiPattern:
    """A recorded hash that was never redeemed is a promise, not a deletion certificate."""
    anti_pattern_id = "unexercised_certificate_authorizes_deletion_v1"
    provenance = _provenance(
        "re-measure when a store adds a rebuild path the four-binding gate cannot execute; the "
        "refused bytes are RETAINED, never deleted, so the class reopens by exercising the "
        "rebuild, never by loosening the gate"
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="sr5_ntb2_apdatastore_adjudication_20260911",
        measurement_method=(
            "read every manifest row of an operator-authorized 4.5 GB class and attempt the "
            "four-binding certificate on each: manifest names the file with bytes and sha256; "
            "the on-disk file hashes to that sha256; the retained rebuild source hashes to ITS "
            "recorded sha256; and the rebuild, executed there and then, serialises to the same "
            "sha256 as the file about to be removed"
        ),
        empirical_artifact_path=_LEDGER,
        empirical_output={
            "placeholder_hash_literal": PLACEHOLDER_HASH_LITERAL,
            "rows_carrying_it": PLACEHOLDER_ROWS,
            "bytes_refused_on_it": PLACEHOLDER_BYTES_REFUSED,
            "share_of_authorized_class": PLACEHOLDER_SHARE_OF_CLASS,
            "incomplete_run_rows": f"{INCOMPLETE_RUN_ROWS} of {INCOMPLETE_RUN_ROWS_TOTAL}",
            "incomplete_run_bytes_refused": INCOMPLETE_RUN_BYTES_REFUSED,
            "bytes_actually_certifiable_in_the_class": CERTIFIABLE_BYTES,
            "authorization_was_not_the_binding_constraint": (
                "MAIN had authorized the class; MEASUREMENT refused 99.2 % of it"
            ),
            "gate_proven_by_a_negative_control": (
                "a synthetic frame0_receiver holding frame 1 was REFUSED with the file intact -- "
                "a gate that has never refused anything is not known to be a gate"
            ),
            "arm_totals_under_exercised_certificates": {
                "files_deleted": SR5_FILES_DELETED,
                "bytes_deleted": SR5_BYTES_DELETED,
                "refusals_after_certification": 0,
            },
        },
        falsification_residual=None,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            f"execute the rebuild before the unlink ({_CERT_TOOL}); refuse on any missing "
            "binding; keep the bytes"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A deletion is authorized from a manifest row whose hash was never computed "
            "(a literal such as 'SKIPPED_LARGE_STREAMED') or whose rebuild was never executed. "
            "The row looks like a certificate and binds nothing: the bytes cannot be tied to the "
            "manifest and no rebuild can be checked against anything. sr5 refused 3,780,374,400 B "
            "(99.2 % of an operator-authorized class) on exactly this, plus 353,999,376 B whose "
            "supersession claim named a run that did not finish."
        ),
        forbidden_pattern_predicate=(
            "delete_authorized == true AND (manifest_row.sha256 is not a 64-hex digest "
            "OR on_disk_hash != manifest_row.sha256 OR rebuild_source_hash != recorded_source_hash "
            "OR rebuild_executed_now.sha256 != file_about_to_be_removed.sha256)"
        ),
        falsification_band={
            "bytes_refused_on_placeholder_hash": float(PLACEHOLDER_BYTES_REFUSED),
            "share_of_authorized_class_refused": PLACEHOLDER_SHARE_OF_CLASS,
            "bytes_refused_on_unfinished_run_supersession": float(INCOMPLETE_RUN_BYTES_REFUSED),
            "required_bindings": 4.0,
        },
        recurrence_conditions=(
            "a manifest writer streams a large file and records a token instead of a digest",
            "a supersession claim cites an artifact of a run whose completion flag is false",
            "a certificate is written at manifest time and never redeemed before the unlink",
            "operator authorization of a class is read as authorization of each row in it",
        ),
        canonical_source_anchor=f"{_LEDGER} sections 2, 5 and 8",
        canonical_unwind_path=(
            "Bind four things and refuse on any one missing: the manifest names the file with "
            "bytes and sha256; the on-disk file hashes to that sha256; the retained rebuild "
            "source hashes to its recorded sha256; and the rebuild, EXECUTED THERE AND THEN, "
            f"serialises to the same sha256 as the file about to be removed. Implemented in "
            f"{_CERT_TOOL}; exercise the restore path on a negative control so the gate is known "
            "to refuse."
        ),
        canonical_producers=(_CERT_TOOL, "tools/sr5_certify_delete_command_rebuildable.py"),
        canonical_consumers=(_LEDGER, "tools/sr5_certify_relocate_file.py"),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_sampled_prefilter_decides_equality_v1() -> AntiPattern:
    """A sampled prefilter chooses what to hash; it never decides equality."""
    anti_pattern_id = "sampled_prefilter_decides_equality_v1"
    provenance = _provenance(
        "the class does not reopen by sampling more windows -- any finite sample has bytes "
        "outside it. It reopens only for a comparison whose instrument reads EVERY byte"
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="sr5_pfs1_raw_dedup_prefilter_half_refuted_20260911",
        measurement_method=(
            f"{PREFILTER_WINDOWS} x 16 MiB-window signature over {RAW_FILES_SCANNED} inflated "
            "0.raw files to CHOOSE candidate groups, then a full sha256 of every candidate to "
            "DECIDE; hardlink only on full-hash equality, with both paths re-resolved after"
        ),
        empirical_artifact_path=_DEDUP_RESULT,
        empirical_output={
            "files_scanned": RAW_FILES_SCANNED,
            "distinct_prefilter_signatures": DISTINCT_SIGNATURES,
            "candidate_groups_from_prefilter": CANDIDATE_GROUPS,
            "groups_confirmed_by_full_sha256": CONFIRMED_GROUPS,
            "groups_refuted_by_full_sha256": CANDIDATE_GROUPS - CONFIRMED_GROUPS,
            "files_in_the_refuted_group": REFUTED_GROUP_FILES,
            "why_the_prefilter_was_wrong": (
                "the three refuted files matched on all three sampled windows; their differing "
                "bytes lie outside the windows"
            ),
            "projected_release_gib_on_the_unconfirmed_prefilter": PROJECTED_RELEASE_GIB,
            "measured_release_bytes": MEASURED_RELEASE_BYTES,
            "over_claim_factor": 3.0,
            "files_deleted_to_achieve_it": 0,
            "mechanism": "hardlink onto one inode; both paths still resolve",
        },
        falsification_residual=None,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            f"full sha256 before any grouping ({_DEDUP_TOOL}); prefer hardlink over deletion"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A sampled signature (N fixed windows of a large file) is treated as a decision about "
            "equality rather than as a chooser of what to hash in full. sr5's 3 x 16 MiB prefilter "
            "produced two duplicate-candidate groups; the full sha256 confirmed one and refuted "
            "the other, whose three files matched every sampled window and differ outside them. "
            "The projection built on the unconfirmed prefilter over-claimed 3.0x (10.233 GiB "
            "projected vs 3.411 GiB measured)."
        ),
        forbidden_pattern_predicate=(
            "files_grouped_as_identical == true AND comparison_instrument_reads_every_byte "
            "== false"
        ),
        falsification_band={
            "candidate_groups": float(CANDIDATE_GROUPS),
            "groups_surviving_full_hash": float(CONFIRMED_GROUPS),
            "prefilter_false_positive_rate_observed": 0.5,
            "projection_over_claim_factor": 3.0,
        },
        recurrence_conditions=(
            "a dedup or equality claim is made from sampled windows, head/tail reads, or a size match",
            "a release/saving projection is published before the full hash confirms its groups",
            "a large-file comparison is skipped for cost and the skip is recorded as a result",
        ),
        canonical_source_anchor=f"{_LEDGER} sections 6 and the PFS1 dedup table",
        canonical_unwind_path=(
            "Treat every sampled signature as a CHOOSER. Take a full sha256 of each candidate "
            "before grouping, publish no saving figure until the full hash confirms it, and "
            f"prefer a hardlink (nothing deleted, both paths resolve) over any deletion. "
            f"Implemented in {_DEDUP_TOOL}."
        ),
        canonical_producers=(_DEDUP_TOOL,),
        canonical_consumers=(_LEDGER, _DEDUP_RESULT),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


SR5_STORAGE_ANTI_PATTERNS = (
    build_unexercised_certificate_authorizes_deletion_v1,
    build_sampled_prefilter_decides_equality_v1,
)
