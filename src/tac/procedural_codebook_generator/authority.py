# SPDX-License-Identifier: MIT
"""Authority classification for procedural-codebook seed carriers.

Procedural generation is not automatically a contest loophole or automatically
contest-safe. The carrier matters: archive members are charged by the rate
term; per-video literals in ``inflate.py`` are script-side payloads until the
operator has an explicit compliance ruling, while generic decoder constants
are ordinary code but still require self-contained scorer-free replay proof.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

SeedCarrier = Literal[
    "archive_member_seed",
    "archive_member_weight_derived",
    "inflate_py_literal_seed",
]
LiteralPayloadKind = Literal[
    "per_video_payload",
    "generic_decoder_constant",
    "compliance_ruled_payload",
]
AuthorityMode = Literal[
    "archive_seeded",
    "weight_derived",
    "runtime_constant",
]


_DEFAULT_AUTHORITY_MODES: tuple[AuthorityMode, ...] = (
    "archive_seeded",
    "weight_derived",
    "runtime_constant",
)


def classify_procedural_seed_authority(
    seed_carrier: SeedCarrier,
    *,
    score_affecting: bool = True,
    literal_payload_kind: LiteralPayloadKind = "per_video_payload",
    runtime_consumption_proof: bool = False,
    self_contained_archive_proof: bool = False,
    scorer_free_inflate_proof: bool = False,
    no_external_state_proof: bool = False,
    packet_compiler_target_declared: bool = False,
    exact_eval_validated: bool = False,
) -> dict[str, object]:
    """Return a fail-closed authority record for a procedural seed carrier.

    Use this before routing procedural generation into a candidate manifest.
    ``archive_member_seed`` is the canonical promotion path. The
    ``inflate_py_literal_seed`` path is deliberately kept as a separate
    probe-only variant when the literal is per-video payload, because upstream
    loophole PRs demonstrated script-side payload smuggling as an organizer-risk
    class.
    """

    carrier = str(seed_carrier)
    if carrier == "archive_member_seed":
        return _record(
            seed_carrier=carrier,
            compliance_class="canonical_archive_charged_seed",
            score_affecting=score_affecting,
            default_for_promotion=True,
            contest_compliance_risk="low_after_exact_eval",
            required_proofs=(
                "seed_member_inside_archive_zip",
                "archive_sha256_and_bytes_recorded",
                "seed_selection_scope_recorded",
                "no_untracked_inflate_time_side_input",
                "seed_mutation_changes_generated_bytes",
                "runtime_consumes_seed_member",
                "full_frame_inflate_output_mutation_proof",
                "exact_auth_eval",
            ),
            runtime_consumption_proof=runtime_consumption_proof,
            self_contained_archive_proof=self_contained_archive_proof,
            scorer_free_inflate_proof=scorer_free_inflate_proof,
            no_external_state_proof=no_external_state_proof,
            packet_compiler_target_declared=packet_compiler_target_declared,
            exact_eval_validated=exact_eval_validated,
        )
    if carrier == "archive_member_weight_derived":
        return _record(
            seed_carrier=carrier,
            compliance_class="canonical_weight_derived_from_charged_member",
            score_affecting=score_affecting,
            default_for_promotion=True,
            contest_compliance_risk="low_after_exact_eval",
            required_proofs=(
                "source_member_inside_archive_zip",
                "source_member_sha256_frozen",
                "derivation_scope_recorded",
                "no_new_members_or_bytes_proof",
                "no_untracked_inflate_time_side_input",
                "runtime_consumes_source_member",
                "full_frame_inflate_output_mutation_proof",
                "exact_auth_eval",
            ),
            runtime_consumption_proof=runtime_consumption_proof,
            self_contained_archive_proof=self_contained_archive_proof,
            scorer_free_inflate_proof=scorer_free_inflate_proof,
            no_external_state_proof=no_external_state_proof,
            packet_compiler_target_declared=packet_compiler_target_declared,
            exact_eval_validated=exact_eval_validated,
        )
    if carrier == "inflate_py_literal_seed":
        if literal_payload_kind not in {
            "per_video_payload",
            "generic_decoder_constant",
            "compliance_ruled_payload",
        }:
            raise ValueError(f"unknown literal payload kind: {literal_payload_kind!r}")
        if literal_payload_kind == "per_video_payload":
            return {
                "seed_carrier": carrier,
                "compliance_class": "organizer_risk_script_side_payload",
                "literal_payload_kind": literal_payload_kind,
                "score_affecting": bool(score_affecting),
                "default_for_promotion": False,
                "contest_compliance_risk": "high_without_explicit_ruling",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "research_only": True,
                "required_proofs": [
                    "explicit_operator_or_organizer_compliance_ruling",
                    "script_literal_size_and_semantics_recorded",
                    "runtime_consumption_proof",
                    "paired_archive_seed_control",
                    "exact_auth_eval_after_ruling",
                ],
                "authority_boundary": (
                    "score-affecting seeds in inflate.py are kept separate from "
                    "charged archive seeds; upstream loophole PRs treated "
                    "script-side payload smuggling as scoring-script gaming"
                ),
            }
        ready = bool(
            runtime_consumption_proof
            and self_contained_archive_proof
            and scorer_free_inflate_proof
            and no_external_state_proof
            and packet_compiler_target_declared
        )
        promoted = bool(ready and exact_eval_validated)
        return {
            "seed_carrier": carrier,
            "compliance_class": (
                "ordinary_decoder_code_constant"
                if literal_payload_kind == "generic_decoder_constant"
                else "compliance_ruled_script_payload"
            ),
            "literal_payload_kind": literal_payload_kind,
            "score_affecting": bool(score_affecting),
            "default_for_promotion": literal_payload_kind == "compliance_ruled_payload",
            "contest_compliance_risk": "low_if_not_data_payload",
            "score_claim": False,
            "promotion_eligible": promoted,
            "rank_or_kill_eligible": promoted,
            "ready_for_exact_eval_dispatch": ready,
            "research_only": not promoted,
            "required_proofs": [
                "constant_is_generic_decoder_code_not_per_video_payload",
                "runtime_consumption_proof",
                "self_contained_archive_or_fixed_contest_code_proof",
                "scorer_free_inflate_proof",
                "no_external_state_proof",
                "packet_compiler_target_declared",
                "exact_auth_eval",
            ],
            "authority_boundary": (
                "generic decoder constants may affect score through normal decode "
                "semantics, but they are code rather than hidden per-video payload; "
                "promotion still requires self-contained scorer-free replay and exact eval"
            ),
        }
    raise ValueError(f"unknown procedural seed carrier: {seed_carrier!r}")


def build_procedural_seed_authority_packet(
    candidate_id: str,
    *,
    modes: Iterable[AuthorityMode] = _DEFAULT_AUTHORITY_MODES,
    runtime_constant_kind: LiteralPayloadKind = "per_video_payload",
    score_affecting: bool = True,
    runtime_consumption_proof: bool = False,
    self_contained_archive_proof: bool = False,
    scorer_free_inflate_proof: bool = False,
    no_external_state_proof: bool = False,
    packet_compiler_target_declared: bool = False,
    exact_eval_validated: bool = False,
) -> dict[str, object]:
    """Build the authority packet for procedural seed/weight variants.

    Ambiguous procedural-generation designs should carry both the charged
    archive-seeded variant and any proposed ``inflate.py`` constant variant.
    The packet makes that arbitration explicit: archive-contained seed/weight
    carriers may become promotion-eligible after the proof stack and exact eval;
    a per-video runtime literal stays research-only unless it has an explicit
    compliance ruling. This is the code surface behind the repository docs'
    two-track ``archive_seeded`` / ``runtime_constant`` authority protocol.
    """

    normalized_modes: list[AuthorityMode] = []
    seen: set[str] = set()
    for mode in modes:
        if mode not in _DEFAULT_AUTHORITY_MODES:
            raise ValueError(f"unknown procedural authority mode: {mode!r}")
        if mode not in seen:
            normalized_modes.append(mode)
            seen.add(mode)
    if not normalized_modes:
        raise ValueError("at least one procedural authority mode is required")

    records: dict[str, dict[str, object]] = {}
    if "archive_seeded" in normalized_modes:
        records["archive_seeded"] = classify_procedural_seed_authority(
            "archive_member_seed",
            score_affecting=score_affecting,
            runtime_consumption_proof=runtime_consumption_proof,
            self_contained_archive_proof=self_contained_archive_proof,
            scorer_free_inflate_proof=scorer_free_inflate_proof,
            no_external_state_proof=no_external_state_proof,
            packet_compiler_target_declared=packet_compiler_target_declared,
            exact_eval_validated=exact_eval_validated,
        )
    if "weight_derived" in normalized_modes:
        records["weight_derived"] = classify_procedural_seed_authority(
            "archive_member_weight_derived",
            score_affecting=score_affecting,
            runtime_consumption_proof=runtime_consumption_proof,
            self_contained_archive_proof=self_contained_archive_proof,
            scorer_free_inflate_proof=scorer_free_inflate_proof,
            no_external_state_proof=no_external_state_proof,
            packet_compiler_target_declared=packet_compiler_target_declared,
            exact_eval_validated=exact_eval_validated,
        )
    if "runtime_constant" in normalized_modes:
        records["runtime_constant"] = classify_procedural_seed_authority(
            "inflate_py_literal_seed",
            score_affecting=score_affecting,
            literal_payload_kind=runtime_constant_kind,
            runtime_consumption_proof=runtime_consumption_proof,
            self_contained_archive_proof=self_contained_archive_proof,
            scorer_free_inflate_proof=scorer_free_inflate_proof,
            no_external_state_proof=no_external_state_proof,
            packet_compiler_target_declared=packet_compiler_target_declared,
            exact_eval_validated=exact_eval_validated,
        )

    preferred_mode = "archive_seeded" if "archive_seeded" in records else normalized_modes[0]
    preferred_record = records[preferred_mode]
    promotion_eligible_modes = [
        mode for mode, record in records.items() if bool(record.get("promotion_eligible"))
    ]
    ready_for_exact_eval_modes = [
        mode for mode, record in records.items() if bool(record.get("ready_for_exact_eval_dispatch"))
    ]
    return {
        "schema": "procedural_seed_authority_packet_v1",
        "candidate_id": str(candidate_id),
        "preferred_promotion_mode": preferred_mode,
        "mode_count": len(records),
        "modes": records,
        "promotion_eligible_modes": promotion_eligible_modes,
        "ready_for_exact_eval_modes": ready_for_exact_eval_modes,
        "score_claim": False,
        "promotion_eligible": bool(preferred_record.get("promotion_eligible")),
        "rank_or_kill_eligible": bool(preferred_record.get("rank_or_kill_eligible")),
        "ready_for_exact_eval_dispatch": bool(preferred_record.get("ready_for_exact_eval_dispatch")),
        "research_only": not bool(preferred_record.get("promotion_eligible")),
        "authority_protocol": (
            "compare archive_seeded and runtime_constant variants when both are "
            "defensible; promote only byte-closed variants whose score-bearing "
            "information is archive-charged or explicitly ruled decoder logic"
        ),
        "contest_compliance_authority": "docs/contest_compliance_authority.md",
    }


def _record(
    *,
    seed_carrier: str,
    compliance_class: str,
    score_affecting: bool,
    default_for_promotion: bool,
    contest_compliance_risk: str,
    required_proofs: tuple[str, ...],
    runtime_consumption_proof: bool,
    self_contained_archive_proof: bool,
    scorer_free_inflate_proof: bool,
    no_external_state_proof: bool,
    packet_compiler_target_declared: bool,
    exact_eval_validated: bool,
) -> dict[str, object]:
    ready = bool(
        runtime_consumption_proof
        and self_contained_archive_proof
        and scorer_free_inflate_proof
        and no_external_state_proof
        and packet_compiler_target_declared
    )
    promoted = bool(ready and exact_eval_validated)
    return {
        "seed_carrier": seed_carrier,
        "compliance_class": compliance_class,
        "score_affecting": bool(score_affecting),
        "default_for_promotion": bool(default_for_promotion),
        "contest_compliance_risk": contest_compliance_risk,
        "score_claim": False,
        "promotion_eligible": promoted,
        "rank_or_kill_eligible": promoted,
        "ready_for_exact_eval_dispatch": ready,
        "research_only": not promoted,
        "proof_status": {
            "runtime_consumption_proof": bool(runtime_consumption_proof),
            "self_contained_archive_proof": bool(self_contained_archive_proof),
            "scorer_free_inflate_proof": bool(scorer_free_inflate_proof),
            "no_external_state_proof": bool(no_external_state_proof),
            "packet_compiler_target_declared": bool(packet_compiler_target_declared),
            "exact_eval_validated": bool(exact_eval_validated),
        },
        "required_proofs": list(required_proofs),
        "authority_boundary": (
            "procedural generation derives bytes instead of shipping the full "
            "table, but promotion still requires charged-source custody, "
            "runtime consumption proof, full-frame output proof, and exact eval"
        ),
    }


__all__ = [
    "AuthorityMode",
    "LiteralPayloadKind",
    "SeedCarrier",
    "build_procedural_seed_authority_packet",
    "classify_procedural_seed_authority",
]
