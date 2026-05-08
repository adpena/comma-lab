"""Dispatch-closure guard for monolithic HNeRV packet candidates.

This module is intentionally JSON-artifact based: candidate builders, runtime
proof builders, and lane-claim exporters can all call the same pure function
before spending exact-eval GPU time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.hnerv_entropy_frontier_selector import (
    ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    build_rate_only_floor_proof,
    scorer_changing_stack_path_declaration,
)

SCHEMA = "tac_monolithic_packet_closure_gate_v1"
MONOLITHIC_MANIFEST_SCHEMA = "tac_monolithic_packet_candidate_v1"
RUNTIME_PROOF_SCHEMA = "tac_runtime_consumption_proof_v1"
ACTIVE_LANE_CLAIM_SCHEMA = "tac_active_lane_claim_json_v1"
CANONICAL_DISPATCH_CLAIMS_PATH = ".omx/state/active_lane_dispatch_claims.md"
DERIVED_LOGICAL_SECTION_NAMES = frozenset({"ff_header"})
FIXABLE_MANIFEST_DISPATCH_BLOCKERS = frozenset(
    {
        "runtime_consumption_proof_missing",
        "active_lane_claim_missing",
    }
)
TERMINAL_CLAIM_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)


def build_monolithic_packet_closure_gate(
    candidate_manifest: Mapping[str, Any],
    *,
    runtime_proof: Mapping[str, Any] | None = None,
    lane_claim: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    active_rate_only_floor_archive_bytes: int | None = ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    expected_lane_id: str | None = None,
    expected_instance_job_id: str | None = None,
) -> dict[str, Any]:
    """Return the fail-closed exact-dispatch gate for one monolithic candidate.

    The supplied ``lane_claim`` MUST match the candidate's intended lane and
    job. The expected values resolve in the following order:

      1. ``expected_lane_id`` / ``expected_instance_job_id`` keyword args
         (preferred — bound by the caller / dispatch tool).
      2. ``candidate_manifest["lane_claim"]["lane_id"]`` /
         ``candidate_manifest["lane_claim"]["instance_job_id"]`` (preferred
         when builder writes lane binding into the manifest).

    If neither source provides a binding, the gate refuses to authorize
    dispatch (codex HIGH finding #2 2026-05-08): a valid active claim for
    a *different* lane could otherwise satisfy the lane-claim section and
    misattribute GPU spend.

    Per ``CLAUDE.md`` "Cross-agent dispatch coordination" + Level-2
    dispatch-custody guard.
    """

    manifest_summary, manifest_blockers = _manifest_summary(candidate_manifest)
    mutation_summary, mutation_blockers = _logical_mutation_summary(candidate_manifest)
    runtime_summary, runtime_blockers = _runtime_consumption_summary(
        runtime_proof,
        candidate_manifest=candidate_manifest,
        changed_sections=mutation_summary["changed_sections"],
    )
    expected_lane = _resolve_expected_lane_binding(
        candidate_manifest=candidate_manifest,
        explicit_lane_id=expected_lane_id,
        explicit_instance_job_id=expected_instance_job_id,
    )
    lane_summary, lane_blockers = _lane_claim_summary(
        lane_claim,
        dry_run=dry_run,
        expected_lane_id=expected_lane["lane_id"],
        expected_instance_job_id=expected_lane["instance_job_id"],
        binding_blockers=expected_lane["blockers"],
    )
    floor_summary, floor_blockers = _rate_only_floor_summary(
        candidate_manifest,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
    )

    blockers = _dedupe(
        [
            *manifest_blockers,
            *mutation_blockers,
            *runtime_blockers,
            *lane_blockers,
            *floor_blockers,
        ]
    )
    closure_gate_passed = not blockers
    return {
        "schema": SCHEMA,
        "candidate_id": manifest_summary["candidate_id"],
        "dry_run": dry_run,
        "score_claim": False,
        "dispatch_attempted": False,
        "closure_gate_passed": closure_gate_passed,
        "ready_for_exact_eval_dispatch": closure_gate_passed and not dry_run,
        "dispatchable": closure_gate_passed and not dry_run,
        "dry_run_not_dispatch_authorization": dry_run,
        "candidate_manifest": manifest_summary,
        "logical_mutation": mutation_summary,
        "runtime_consumption": runtime_summary,
        "lane_claim": lane_summary,
        "rate_only_floor": floor_summary,
        "blockers": blockers,
        "manifest_blockers": manifest_blockers,
        "logical_mutation_blockers": mutation_blockers,
        "runtime_blockers": runtime_blockers,
        "lane_claim_blockers": lane_blockers,
        "rate_only_floor_blockers": floor_blockers,
    }


def _manifest_summary(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    schema = manifest.get("schema")
    candidate_id = manifest.get("candidate_id")
    ready = manifest.get("ready_for_exact_eval_dispatch")
    dispatch_blockers = _string_list(manifest.get("dispatch_blockers"))
    if schema != MONOLITHIC_MANIFEST_SCHEMA:
        blockers.append("candidate_manifest_schema_mismatch")
    if not isinstance(candidate_id, str) or not candidate_id:
        blockers.append("candidate_manifest_candidate_id_missing")
    if manifest.get("score_claim") is not False:
        blockers.append("candidate_manifest_score_claim_not_false")
    if manifest.get("promotion_eligible") is True:
        blockers.append("candidate_manifest_promotion_eligible_true")
    if manifest.get("rank_or_kill_eligible") is True:
        blockers.append("candidate_manifest_rank_or_kill_eligible_true")
    if not isinstance(ready, bool):
        blockers.append("candidate_manifest_ready_flag_not_bool")
    if dispatch_blockers is None:
        if manifest.get("dispatch_blockers") is not None:
            blockers.append("candidate_manifest_dispatch_blockers_not_string_list")
        dispatch_blockers = []
    for blocker in dispatch_blockers:
        if blocker not in FIXABLE_MANIFEST_DISPATCH_BLOCKERS:
            blockers.append(f"candidate_manifest_unresolved_dispatch_blocker:{blocker}")
    if ready is True and dispatch_blockers:
        blockers.append("candidate_manifest_ready_true_with_dispatch_blockers")
    archive = manifest.get("candidate_archive")
    archive_summary = {
        "path": None,
        "bytes": None,
        "sha256": None,
    }
    if not isinstance(archive, Mapping):
        blockers.append("candidate_manifest_candidate_archive_missing")
    else:
        path = archive.get("path")
        byte_count = archive.get("bytes")
        sha = archive.get("sha256")
        if not isinstance(path, str) or not path:
            blockers.append("candidate_manifest_candidate_archive_path_missing")
        if not isinstance(byte_count, int) or byte_count < 0:
            blockers.append("candidate_manifest_candidate_archive_bytes_invalid")
        if not _is_sha256(sha):
            blockers.append("candidate_manifest_candidate_archive_sha256_invalid")
        archive_summary = {
            "path": path if isinstance(path, str) else None,
            "bytes": byte_count if isinstance(byte_count, int) else None,
            "sha256": sha.lower() if _is_sha256(sha) else None,
        }
    return (
        {
            "schema": schema if isinstance(schema, str) else None,
            "candidate_id": candidate_id if isinstance(candidate_id, str) else None,
            "ready_for_exact_eval_dispatch": ready if isinstance(ready, bool) else None,
            "dispatch_blockers": dispatch_blockers,
            "candidate_archive": archive_summary,
        },
        blockers,
    )


def _logical_mutation_summary(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    layout = manifest.get("monolithic_layout")
    replacements = manifest.get("replacements")
    replacement_map = _replacement_map(replacements)
    changed_sections: list[dict[str, Any]] = []
    grammar: str | None = None
    member_sha: str | None = None
    parser_proven = False

    if not isinstance(layout, Mapping):
        blockers.append("candidate_manifest_monolithic_layout_missing")
        sections: list[Any] = []
    else:
        grammar = layout.get("grammar") if isinstance(layout.get("grammar"), str) else None
        member_sha = _sha_or_none(layout.get("new_member_sha256"))
        sections = layout.get("sections") if isinstance(layout.get("sections"), list) else []
        parser_proven = bool(grammar and sections)
        if not grammar:
            blockers.append("candidate_manifest_monolithic_layout_grammar_missing")
        if not _is_sha256(layout.get("new_member_sha256")):
            blockers.append("candidate_manifest_monolithic_layout_new_member_sha256_invalid")
        if not isinstance(layout.get("sections"), list):
            blockers.append("candidate_manifest_monolithic_layout_sections_missing")

    if not isinstance(replacements, list) or not replacements:
        blockers.append("candidate_manifest_replacements_missing")
    if not replacement_map:
        blockers.append("candidate_manifest_replacement_section_shas_missing")

    section_names: set[str] = set()
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            blockers.append(f"candidate_manifest_logical_section_not_object:{index}")
            continue
        name = section.get("name")
        if not isinstance(name, str) or not name:
            blockers.append(f"candidate_manifest_logical_section_name_missing:{index}")
            continue
        if name in section_names:
            blockers.append(f"candidate_manifest_duplicate_logical_section:{name}")
        section_names.add(name)
        if section.get("changed") is not True or name in DERIVED_LOGICAL_SECTION_NAMES:
            continue
        old_sha = _sha_or_none(section.get("old_sha256"))
        new_sha = _sha_or_none(section.get("new_sha256"))
        replacement_sha = replacement_map.get(name)
        if old_sha is None:
            blockers.append(f"candidate_manifest_changed_section_old_sha256_invalid:{name}")
            continue
        if new_sha is None:
            blockers.append(f"candidate_manifest_changed_section_new_sha256_invalid:{name}")
            continue
        if old_sha == new_sha:
            blockers.append(f"candidate_manifest_changed_section_sha256_unchanged:{name}")
            continue
        if replacement_sha is None:
            blockers.append(f"candidate_manifest_changed_section_without_replacement:{name}")
            continue
        if replacement_sha != new_sha:
            blockers.append(f"candidate_manifest_replacement_sha_mismatch:{name}")
            continue
        changed_sections.append(
            {
                "section_name": name,
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "old_bytes": section.get("old_len") if isinstance(section.get("old_len"), int) else None,
                "new_bytes": section.get("new_len") if isinstance(section.get("new_len"), int) else None,
            }
        )

    missing_layout_sections = sorted(set(replacement_map) - section_names)
    for section_name in missing_layout_sections:
        blockers.append(f"candidate_manifest_replacement_not_in_logical_sections:{section_name}")
    if parser_proven is not True:
        blockers.append("parser_proven_logical_layout_missing")
    if not changed_sections:
        blockers.append("parser_proven_logical_section_mutation_missing")

    return (
        {
            "parser_proven": parser_proven,
            "grammar": grammar,
            "new_member_sha256": member_sha,
            "changed_section_count": len(changed_sections),
            "changed_sections": changed_sections,
        },
        blockers,
    )


def _runtime_consumption_summary(
    proof: Mapping[str, Any] | None,
    *,
    candidate_manifest: Mapping[str, Any],
    changed_sections: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    candidate_sha = _nested_sha(candidate_manifest, "candidate_archive", "sha256")
    member_sha = _nested_sha(candidate_manifest, "monolithic_layout", "new_member_sha256")
    summary: dict[str, Any] = {
        "provided": isinstance(proof, Mapping),
        "schema": None,
        "ready_for_exact_eval_runtime": False,
        "candidate_archive_sha256_bound": False,
        "rebuilt_member_sha256_bound": False,
        "changed_sections_bound": [],
    }
    if not isinstance(proof, Mapping):
        return summary, ["runtime_consumption_proof_missing"]

    schema = proof.get("schema") if isinstance(proof.get("schema"), str) else None
    reported_archive_sha = proof.get("candidate_archive_sha256")
    reported_member_sha = proof.get("rebuilt_member_sha256", proof.get("new_member_sha256"))
    proof_sections = _changed_section_map(proof.get("changed_sections"))

    if schema != RUNTIME_PROOF_SCHEMA:
        blockers.append("runtime_proof_schema_mismatch")
    if proof.get("ready_for_exact_eval_runtime") is not True:
        blockers.append("runtime_proof_ready_for_exact_eval_runtime_not_true")
    if proof.get("blockers") not in (None, []):
        blockers.append("runtime_proof_reports_blockers")
    if proof.get("score_claim") is True:
        blockers.append("runtime_proof_score_claim_true")
    if not _sha_matches(reported_archive_sha, candidate_sha):
        blockers.append("runtime_proof_candidate_archive_sha256_mismatch")
    if not _sha_matches(reported_member_sha, member_sha):
        blockers.append("runtime_proof_rebuilt_member_sha256_mismatch")
    for field in ("command_sha256", "log_sha256"):
        if not _is_sha256(proof.get(field)):
            blockers.append(f"runtime_proof_{field}_invalid")

    bound_sections: list[str] = []
    for section in changed_sections:
        section_name = section["section_name"]
        expected_sha = section["new_sha256"]
        if _sha_matches(proof_sections.get(section_name), expected_sha):
            bound_sections.append(section_name)
        else:
            blockers.append(f"runtime_proof_changed_section_mismatch:{section_name}")

    summary.update(
        {
            "schema": schema,
            "ready_for_exact_eval_runtime": proof.get("ready_for_exact_eval_runtime") is True,
            "candidate_archive_sha256_bound": _sha_matches(reported_archive_sha, candidate_sha),
            "rebuilt_member_sha256_bound": _sha_matches(reported_member_sha, member_sha),
            "changed_sections_bound": sorted(bound_sections),
            "expected_changed_sections": sorted(section["section_name"] for section in changed_sections),
        }
    )
    return summary, blockers


def _lane_claim_summary(
    claim: Mapping[str, Any] | None,
    *,
    dry_run: bool,
    expected_lane_id: str | None = None,
    expected_instance_job_id: str | None = None,
    binding_blockers: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    # Binding blockers only apply when a claim is actually provided OR
    # when we are not in dry_run mode. dry_run + no claim is a closure
    # review path that cannot misattribute GPU spend (no dispatch will be
    # authorized regardless of binding).
    incoming_binding_blockers = list(binding_blockers or [])
    summary: dict[str, Any] = {
        "required": not dry_run,
        "provided": isinstance(claim, Mapping),
        "active": False,
        "lane_id": None,
        "instance_job_id": None,
        "claim_status": None,
        "claims_path": None,
        "missing_allowed_by_dry_run": dry_run and not isinstance(claim, Mapping),
        "expected_lane_id": expected_lane_id,
        "expected_instance_job_id": expected_instance_job_id,
        "lane_id_bound": False,
        "instance_job_id_bound": False,
    }
    if not isinstance(claim, Mapping):
        if dry_run:
            # No claim, no dispatch authorization: binding cannot misattribute.
            return summary, []
        return summary, [*incoming_binding_blockers, "active_lane_claim_missing"]
    # Claim is provided: surface binding-availability blockers so a wrong-lane
    # claim cannot satisfy the gate.
    blockers: list[str] = list(incoming_binding_blockers)

    status = claim.get("claim_status", claim.get("status"))
    lane_id = claim.get("lane_id")
    instance_job_id = claim.get("instance_job_id", claim.get("job_name"))
    claims_path = claim.get("claims_path")
    claimed_with = claim.get("claimed_with")
    row_hash = claim.get("claim_row_sha256")
    lane_id_bound = (
        isinstance(lane_id, str)
        and isinstance(expected_lane_id, str)
        and lane_id == expected_lane_id
    )
    instance_job_id_bound = (
        isinstance(instance_job_id, str)
        and isinstance(expected_instance_job_id, str)
        and instance_job_id == expected_instance_job_id
    )
    summary.update(
        {
            "active": claim.get("active") is True,
            "lane_id": lane_id if isinstance(lane_id, str) else None,
            "instance_job_id": instance_job_id if isinstance(instance_job_id, str) else None,
            "claim_status": status if isinstance(status, str) else None,
            "claims_path": claims_path if isinstance(claims_path, str) else None,
            "lane_id_bound": lane_id_bound,
            "instance_job_id_bound": instance_job_id_bound,
        }
    )
    if claim.get("schema") != ACTIVE_LANE_CLAIM_SCHEMA:
        blockers.append("lane_claim_schema_mismatch")
    if claim.get("active") is not True:
        blockers.append("lane_claim_not_active")
    if claim.get("blockers") not in (None, []):
        blockers.append("lane_claim_reports_blockers")
    if not isinstance(lane_id, str) or not lane_id:
        blockers.append("lane_claim_lane_id_missing")
    if not isinstance(instance_job_id, str) or not instance_job_id:
        blockers.append("lane_claim_instance_job_id_missing")
    if not isinstance(status, str) or not status:
        blockers.append("lane_claim_status_missing")
    elif _status_is_terminal(status):
        blockers.append("lane_claim_status_terminal")
    if claims_path != CANONICAL_DISPATCH_CLAIMS_PATH:
        blockers.append("lane_claim_claims_path_not_canonical")
    if not (
        isinstance(claimed_with, str)
        and "tools/claim_lane_dispatch.py" in claimed_with
        and " claim" in f" {claimed_with} "
    ):
        blockers.append("lane_claim_helper_not_represented")
    if not _is_sha256(row_hash):
        blockers.append("lane_claim_row_sha256_invalid")
    elif isinstance(claims_path, str) and not _claim_file_contains_row_hash(Path(claims_path), row_hash):
        blockers.append("lane_claim_row_not_found_in_claims_file")
    # Lane-binding enforcement: refuse readiness when supplied claim is
    # for a different lane / job than the candidate's intended dispatch.
    # Codex HIGH finding #2 2026-05-08.
    if (
        isinstance(expected_lane_id, str)
        and expected_lane_id
        and isinstance(lane_id, str)
        and lane_id
        and lane_id != expected_lane_id
    ):
        blockers.append("lane_claim_lane_id_mismatch")
    if (
        isinstance(expected_instance_job_id, str)
        and expected_instance_job_id
        and isinstance(instance_job_id, str)
        and instance_job_id
        and instance_job_id != expected_instance_job_id
    ):
        blockers.append("lane_claim_instance_job_id_mismatch")
    return summary, blockers


def _resolve_expected_lane_binding(
    *,
    candidate_manifest: Mapping[str, Any],
    explicit_lane_id: str | None,
    explicit_instance_job_id: str | None,
) -> dict[str, Any]:
    """Resolve the lane/job binding expected for this candidate.

    Precedence: explicit kwargs > ``candidate_manifest["lane_claim"]``.

    Returns blockers when no binding is available — the gate refuses to
    authorize exact-eval dispatch without an explicit lane / job binding
    (Level-2 dispatch-custody guard, codex HIGH finding #2).
    """

    blockers: list[str] = []
    lane_id: str | None = explicit_lane_id if (
        isinstance(explicit_lane_id, str) and explicit_lane_id
    ) else None
    instance_job_id: str | None = explicit_instance_job_id if (
        isinstance(explicit_instance_job_id, str) and explicit_instance_job_id
    ) else None
    manifest_binding = candidate_manifest.get("lane_claim")
    if isinstance(manifest_binding, Mapping):
        if lane_id is None:
            value = manifest_binding.get("lane_id")
            if isinstance(value, str) and value:
                lane_id = value
        if instance_job_id is None:
            value = manifest_binding.get("instance_job_id")
            if isinstance(value, str) and value:
                instance_job_id = value
    if lane_id is None:
        blockers.append("expected_lane_id_missing")
    if instance_job_id is None:
        blockers.append("expected_instance_job_id_missing")
    return {
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "blockers": blockers,
    }


def _rate_only_floor_summary(
    manifest: Mapping[str, Any],
    *,
    active_rate_only_floor_archive_bytes: int | None,
) -> tuple[dict[str, Any], list[str]]:
    candidate_archive = manifest.get("candidate_archive")
    candidate_bytes = (
        candidate_archive.get("bytes")
        if isinstance(candidate_archive, Mapping) and isinstance(candidate_archive.get("bytes"), int)
        else None
    )
    declaration = scorer_changing_stack_path_declaration(manifest)
    explicit_rate_only = manifest.get("rate_only_candidate", manifest.get("rate_only"))
    scorer_changing_meaningful = declaration["structured_proof"]["meaningful"] is True
    declared_rate_only = True
    blockers: list[str] = []
    if explicit_rate_only is True:
        declared_rate_only = True
    elif explicit_rate_only is False and scorer_changing_meaningful:
        declared_rate_only = False
    elif explicit_rate_only is False:
        declared_rate_only = True
        blockers.append("scorer_changing_stack_path_proof_required_to_escape_rate_only_floor")

    proof = build_rate_only_floor_proof(
        manifest,
        candidate_archive_bytes=candidate_bytes,
        declared_rate_only=declared_rate_only,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
    )
    blockers.extend(proof["blockers"])
    return (
        {
            "active_floor_archive_bytes": active_rate_only_floor_archive_bytes,
            "candidate_archive_bytes": candidate_bytes,
            "declared_rate_only": declared_rate_only,
            "explicit_rate_only_field": explicit_rate_only if isinstance(explicit_rate_only, bool) else None,
            "scorer_changing_stack_path_meaningful": scorer_changing_meaningful,
            "exact_eval_spend_allowed_by_policy": proof["exact_eval_spend_allowed_by_policy"],
            "proof": proof,
        },
        _dedupe(blockers),
    )


def _replacement_map(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    out: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("section_name")
        sha = item.get("new_sha256", item.get("expected_new_sha256"))
        if isinstance(name, str) and _is_sha256(sha):
            out[name] = sha.lower()
    return out


def _changed_section_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            str(name): str(sha).lower()
            for name, sha in value.items()
            if isinstance(name, str) and _is_sha256(sha)
        }
    if not isinstance(value, list):
        return {}
    out: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("section_name", item.get("name"))
        sha = item.get("new_sha256", item.get("sha256"))
        if isinstance(name, str) and _is_sha256(sha):
            out[name] = sha.lower()
    return out


def _nested_sha(mapping: Mapping[str, Any], first: str, second: str) -> str | None:
    value = mapping.get(first)
    if not isinstance(value, Mapping):
        return None
    return _sha_or_none(value.get(second))


def _sha_or_none(value: Any) -> str | None:
    return value.lower() if _is_sha256(value) else None


def _sha_matches(value: Any, expected: str | None) -> bool:
    return expected is not None and isinstance(value, str) and value.lower() == expected.lower()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _status_is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _claim_file_contains_row_hash(path: Path, row_hash: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(hashlib.sha256(line.encode("utf-8")).hexdigest() == row_hash for line in text.splitlines())


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
