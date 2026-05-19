# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.procedural_codebook_generator``.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.procedural_codebook_generator`` per wiring + integration audit
2026-05-19 (commit 3821cfb6b).

``tac.procedural_codebook_generator`` is the canonical helper for
procedurally-generated codebook expansion from archive seeds (per Catalog
#329 contest-compliance + per the inflate_py extreme compression symposium
2026-05-18 ProvenanceKind extension). Exposes ``classify_procedural_seed_authority``
/ ``derive_codebook_from_archive_bytes`` / ``expand_seed_to_codebook`` /
``verify_no_new_bytes_added`` / ``verify_generator_seed_mutation_smoke``.
Consumer surfaces availability + a clear ``[predicted]`` discipline marker
per Catalog #287; per-candidate codebook derivation must route through
explicit archive-seed inclusion checks.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "procedural_codebook_generator_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)
_EXPECTED_AUTHORITY_SCHEMA = "procedural_seed_authority_packet_v1"
_KNOWN_AUTHORITY_MODES = frozenset(
    {
        "archive_seeded",
        "weight_derived",
        "runtime_constant",
    }
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Procedural codebook derivation is
    a deterministic function of the input archive seed bytes; no
    anchor-driven posterior update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment authority annotation. Per-candidate procedural codebook
    contribution requires explicit archive-seed inclusion verification per
    Catalog #329 contest-compliance payload-kinds. This consumer reads that
    authority packet when present, but never turns it into score movement or
    promotion authority by itself.
    """
    authority = _find_authority_packet(candidate)
    summary = _authority_summary(authority)
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.procedural_codebook_generator canonical procedural codebook "
            "expansion helpers available (classify_procedural_seed_authority "
            "/ derive_codebook_from_archive_bytes / expand_seed_to_codebook / "
            "verify_no_new_bytes_added / verify_generator_seed_mutation_smoke); "
            f"{summary['rationale_suffix']} [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": summary["confidence"],
        "procedural_authority_detected": authority is not None,
        "procedural_authority": summary["procedural_authority"],
        "denied_uses": (
            "score_claim",
            "promotion",
            "rank_or_kill",
            "dispatch_readiness",
        ),
    }


def _find_authority_packet(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in (
        "procedural_seed_authority_packet",
        "procedural_codebook_authority_packet",
        "procedural_authority_packet",
        "procedural_seed_authority",
    ):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _authority_summary(authority: Mapping[str, Any] | None) -> dict[str, Any]:
    if authority is None:
        return {
            "confidence": 0.0,
            "rationale_suffix": "no per-candidate procedural authority packet present",
            "procedural_authority": {
                "packet_present": False,
                "preferred_promotion_mode": None,
                "ready_for_exact_eval_modes": (),
                "promotion_eligible_modes": (),
                "research_only": True,
                "score_claim": False,
                "authority_blockers": (
                    "procedural_seed_authority_packet_missing",
                ),
            },
        }

    global_blockers: list[str] = []
    schema = authority.get("schema")
    if schema != _EXPECTED_AUTHORITY_SCHEMA:
        global_blockers.append(f"packet_schema_untrusted:{schema!r}")
    if authority.get("score_claim") is True:
        global_blockers.append("packet_score_claim_not_allowed")

    modes = authority.get("modes")
    if not isinstance(modes, Mapping):
        global_blockers.append("modes_missing_or_malformed")
        mode_blockers_by_mode: dict[str, tuple[str, ...]] = {}
        derived_ready_modes: tuple[str, ...] = ()
        derived_promotion_modes: tuple[str, ...] = ()
    else:
        mode_blockers_by_mode = _mode_blockers_by_mode(modes)
        derived_ready_modes = _derive_modes(
            modes,
            mode_blockers_by_mode,
            field="ready_for_exact_eval_dispatch",
        )
        derived_promotion_modes = _derive_modes(
            modes,
            mode_blockers_by_mode,
            field="promotion_eligible",
            require_ready=True,
        )

    claimed_ready_modes = _declared_modes(
        authority.get("ready_for_exact_eval_modes"),
        field="ready_for_exact_eval_modes",
        global_blockers=global_blockers,
    )
    claimed_promotion_modes = _declared_modes(
        authority.get("promotion_eligible_modes"),
        field="promotion_eligible_modes",
        global_blockers=global_blockers,
    )
    if claimed_ready_modes is not None and claimed_ready_modes != derived_ready_modes:
        global_blockers.append("ready_for_exact_eval_modes_mismatch")
    if (
        claimed_promotion_modes is not None
        and claimed_promotion_modes != derived_promotion_modes
    ):
        global_blockers.append("promotion_eligible_modes_mismatch")

    mode_blockers = tuple(
        blocker
        for blockers in mode_blockers_by_mode.values()
        for blocker in blockers
    )
    if global_blockers:
        ready_modes = ()
        promotion_modes = ()
    else:
        ready_modes = derived_ready_modes
        promotion_modes = derived_promotion_modes
    preferred = authority.get("preferred_promotion_mode")
    confidence = 0.35 if ready_modes else 0.10
    if global_blockers or mode_blockers:
        confidence = min(confidence, 0.20)
    return {
        "confidence": confidence,
        "rationale_suffix": (
            "procedural authority packet present; "
            f"preferred_mode={preferred!r}; "
            f"ready_modes={list(ready_modes)!r}; "
            f"blocked_modes={list(mode_blockers)!r}"
        ),
        "procedural_authority": {
            "packet_present": True,
            "schema": authority.get("schema"),
            "preferred_promotion_mode": preferred,
            "ready_for_exact_eval_modes": ready_modes,
            "promotion_eligible_modes": promotion_modes,
            "research_only": not promotion_modes,
            "score_claim": False,
            "source_score_claim_blocked": authority.get("score_claim") is True,
            "authority_blockers": tuple(dict.fromkeys([*global_blockers, *mode_blockers])),
        },
    }


def _declared_modes(
    value: Any,
    *,
    field: str,
    global_blockers: list[str],
) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        global_blockers.append(f"{field}_missing_or_malformed")
        return None
    declared = tuple(str(mode) for mode in value)
    unknown = tuple(mode for mode in declared if mode not in _KNOWN_AUTHORITY_MODES)
    if unknown:
        global_blockers.append(f"{field}_unknown_modes:{','.join(unknown)}")
        return None
    return declared


def _derive_modes(
    modes: Mapping[str, Any],
    mode_blockers_by_mode: Mapping[str, tuple[str, ...]],
    *,
    field: str,
    require_ready: bool = False,
) -> tuple[str, ...]:
    derived: list[str] = []
    for mode, record in modes.items():
        normalized = str(mode)
        if normalized not in _KNOWN_AUTHORITY_MODES:
            continue
        if not isinstance(record, Mapping):
            continue
        if mode_blockers_by_mode.get(normalized):
            continue
        if record.get(field) is not True:
            continue
        if require_ready and record.get("ready_for_exact_eval_dispatch") is not True:
            continue
        derived.append(normalized)
    return tuple(derived)


def _mode_blockers_by_mode(modes: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    blockers_by_mode: dict[str, tuple[str, ...]] = {}
    for mode, record in modes.items():
        normalized = str(mode)
        blockers: list[str] = []
        if normalized not in _KNOWN_AUTHORITY_MODES:
            blockers.append(f"{normalized}:unknown_authority_mode")
            blockers_by_mode[normalized] = tuple(blockers)
            continue
        if not isinstance(record, Mapping):
            blockers.append(f"{normalized}:authority_record_malformed")
            blockers_by_mode[normalized] = tuple(blockers)
            continue
        if record.get("seed_carrier") == "inflate_py_literal_seed" and record.get(
            "literal_payload_kind"
        ) == "per_video_payload":
            blockers.append(f"{normalized}:script_side_per_video_payload_probe_only")
        if record.get("ready_for_exact_eval_dispatch") is not True:
            blockers.append(f"{normalized}:not_ready_for_exact_eval")
        if record.get("score_claim") is True:
            blockers.append(f"{normalized}:score_claim_not_allowed")
        blockers_by_mode[normalized] = tuple(dict.fromkeys(blockers))
    return blockers_by_mode
