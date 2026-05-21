# SPDX-License-Identifier: MIT
"""Cathedral consumer for TT5L per-pair sideinfo consumption proof.

Per CODEX CROSS-POLLINATION audit `aafac7c84` §10.1 + operator blanket approval
2026-05-20 (WAVE-3-CATHEDRAL-CONSUMER-REGISTRATION-CODEX-AUDIT-CANDIDATES).

Source codex memo: `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.md`
documenting the local no-GPU L5-v2 gate proof for ``time_traveler_l5_autonomy``.
Verdict: ``PER_PAIR_SIDE_INFO_BLOB`` and ``AC_STATE_BLOB`` are parser-consumed
and change inflated raw output under byte mutation (per-pair sideinfo
consumption proof; bound by path and SHA). Classification:
``local_consumption_proof`` (no longer satisfies the full
``byte_closed_temporal_sideinfo_consumption`` gate by itself — full L5-v2 gate
requires contest-scale full-frame custody: 600 pairs / 1200 frames + file-list
SHA-256 + distinct source/candidate raw-output aggregate SHA-256s). Important
limitation per codex: ``AC_STATE_BLOB`` is consumed today as residual
calibration, NOT as a real range/ANS arithmetic decoder; next score-lowering
task is to replace calibration with a byte-closed entropy decode path.

This consumer annotates candidates that resemble TT5L sideinfo consumption
work (sideinfo / per_pair_side_info / ac_state_blob / time_traveler_l5 tokens
in candidate text) with the canonical proof-grade boundary surface per
Catalog #220 substrate L1+ operational mechanism declaration + CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.

Tier A (observability-only) per Catalog #341 canonical-routing-markers:

- ``predicted_delta_adjustment=0.0`` (NEVER mutates score signal)
- ``promotable=False`` (per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA")
- ``axis_tag="[predicted]"`` (per Catalog #287 canonical Provenance umbrella)

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch - ACTIVE (annotate candidates with
    proof-grade boundary reminders)
  * #5 continual-learning posterior - ACTIVE (NO-OP refresh path; sideinfo
    consumption proof anchors flow through ``tac.continual_learning.posterior_update_locked``)
  * #1 sensitivity-map - N/A (defensive annotation only)
  * #2 Pareto constraint - N/A
  * #3 bit-allocator - N/A
  * #6 probe-disambiguator - ACTIVE (the canonical proof-grade boundary IS
    the disambiguator between local_consumption_proof vs full
    byte_closed_temporal_sideinfo_consumption proof)

Sister of:
  * ``wr01_static_packet_custody_consumer`` (static packet custody validation)
  * ``xray_cuda_score_input_hardening_consumer`` (CUDA-axis input hardening)
  * ``venn_risk_composition_consumer`` (Venn rank-composition guard)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "tt5l_sideinfo_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical TT5L sideinfo consumption tokens lifted from the codex memo's
# "Verdict" + "Propagation" sections. A candidate whose text overlaps any of
# these tokens is treated as a TT5L sideinfo consumption candidate and
# annotated with the canonical proof-grade boundary reminder.
_TT5L_SIDEINFO_TOKENS: tuple[str, ...] = (
    "per_pair_side_info_blob",
    "ac_state_blob",
    "time_traveler_l5_autonomy",
    "tt5l",
    "sideinfo_consumed",
    "sideinfo_consumption",
    "byte_closed_temporal_sideinfo_consumption",
    "local_consumption_proof",
    "runtime_overlay_consumed",
)


# Canonical proof-grade boundary set per the codex memo. The two grade
# levels distinguish per-pair (path+SHA bound) proof from full
# byte_closed_temporal_sideinfo_consumption proof (which requires contest-
# scale custody).
_PROOF_GRADE_BOUNDARY_LOCAL: tuple[str, ...] = (
    "per_pair_2_frame_local_parser_inflate_consumption_proof",
    "bound_by_path_and_sha",
    "byte_mutation_changes_inflated_raw_output",
)


_PROOF_GRADE_BOUNDARY_FULL_REQUIREMENTS: tuple[str, ...] = (
    "contest_scale_600_pairs_1200_frames",
    "file_list_sha256_required",
    "distinct_source_candidate_raw_output_aggregate_sha256_required",
    "ac_state_blob_byte_closed_entropy_decode_path_required_to_replace_residual_calibration",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    NO-OP refresh path. Sideinfo consumption proof anchors flow through the
    canonical posterior surface (``tac.continual_learning.posterior_update_locked``
    per Catalog #128/#131 fcntl-locked discipline); this consumer does not
    maintain in-memory state because proof-grade boundary verdicts are per-
    candidate static-text matches against the canonical TT5L sideinfo token
    set.
    """
    _ = anchor  # explicit acknowledgment; no in-memory state to refresh


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Annotates candidates whose text overlaps the canonical TT5L sideinfo
    consumption token set with proof-grade boundary reminders. Returns
    ``predicted_delta_adjustment=0.0`` always (Tier A observability-only per
    Catalog #341 canonical-routing-markers).

    Hook #6 probe-disambiguator semantics: the canonical proof-grade
    boundary set IS the disambiguator between local_consumption_proof
    (per-pair 2-frame local parser/inflate proof) vs full
    byte_closed_temporal_sideinfo_consumption proof (contest-scale full-
    frame custody). The consumer surfaces both grade boundaries so
    operators can audit which proof level a candidate has cleared without
    re-reading the source.
    """
    candidate_text = " ".join(
        f"{k}={v}"
        for k, v in candidate.items()
        if isinstance(v, (str, int, float))
    ).lower()

    matched_tokens = [
        token for token in _TT5L_SIDEINFO_TOKENS if token in candidate_text
    ]

    if not matched_tokens:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no TT5L sideinfo consumption token match; consumer "
                "inactive for this candidate [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    matched_summary = ",".join(matched_tokens[:3])
    rationale = (
        f"TT5L sideinfo consumption candidate detected (matched: "
        f"{matched_summary}); proof-grade boundary surface: local "
        f"({','.join(_PROOF_GRADE_BOUNDARY_LOCAL)}) vs full "
        f"({','.join(_PROOF_GRADE_BOUNDARY_FULL_REQUIREMENTS)}) [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_tokens": tuple(matched_tokens),
        "proof_grade_boundary_local": _PROOF_GRADE_BOUNDARY_LOCAL,
        "proof_grade_boundary_full_requirements": (
            _PROOF_GRADE_BOUNDARY_FULL_REQUIREMENTS
        ),
    }
