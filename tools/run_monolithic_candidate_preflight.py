#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only preflight for an existing monolithic packet candidate manifest.

This wrapper intentionally consumes only JSON artifacts that already exist. It
does not rebuild candidates, mutate archives, read or write `.omx/state`, claim
score, or dispatch exact-eval work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_entropy_frontier_selector import ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES  # noqa: E402
from tac.monolithic_packet_closure_gate import build_monolithic_packet_closure_gate  # noqa: E402

SUMMARY_SCHEMA = "tac_monolithic_candidate_preflight_v1"
MONOLITHIC_MANIFEST_SCHEMA = "tac_monolithic_packet_candidate_v1"
RUNTIME_PROOF_SCHEMA = "tac_runtime_consumption_proof_v1"
ACTIVE_LANE_CLAIM_SCHEMA = "tac_active_lane_claim_json_v1"
FIXABLE_MANIFEST_DISPATCH_BLOCKERS = {
    "runtime_consumption_proof_missing",
    "active_lane_claim_missing",
}
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


class MonolithicCandidatePreflightError(ValueError):
    """Raised when the wrapper cannot safely load requested inputs."""


def build_preflight(
    candidate_manifest_path: Path,
    *,
    runtime_proof_json: Path | None = None,
    lane_claim_json: Path | None = None,
    dry_run: bool = False,
    active_rate_only_floor_archive_bytes: int | None = ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
) -> dict[str, Any]:
    """Return a deterministic, read-only preflight summary."""

    _reject_omx_state_path(candidate_manifest_path, label="candidate manifest")
    if runtime_proof_json is not None:
        _reject_omx_state_path(runtime_proof_json, label="runtime proof")
    if lane_claim_json is not None:
        _reject_omx_state_path(lane_claim_json, label="lane claim")

    manifest = _load_json_object(candidate_manifest_path, label="candidate manifest")
    manifest_sha = _sha256_file(candidate_manifest_path)
    manifest_summary, manifest_blockers = _validate_candidate_manifest(manifest)
    wrapper_blockers: list[str] = []

    runtime_proof_summary: dict[str, Any] = {"provided": False, "path": None}
    runtime_proof_payload: dict[str, Any] | None = None
    if runtime_proof_json is not None:
        runtime_proof_payload = _load_json_object(runtime_proof_json, label="runtime proof")
        runtime_proof_summary, runtime_blockers = _validate_runtime_proof(
            runtime_proof_payload,
            manifest=manifest,
            path=runtime_proof_json,
        )
        wrapper_blockers.extend(runtime_blockers)

    lane_claim_summary: dict[str, Any] = {"provided": False, "path": None}
    lane_claim_payload: dict[str, Any] | None = None
    if lane_claim_json is not None:
        lane_claim_payload = _load_json_object(lane_claim_json, label="lane claim")
        lane_claim_summary, lane_blockers = _validate_lane_claim(
            lane_claim_payload,
            manifest=manifest,
            path=lane_claim_json,
        )
        wrapper_blockers.extend(lane_blockers)

    closure_gate = build_monolithic_packet_closure_gate(
        manifest,
        runtime_proof=runtime_proof_payload,
        lane_claim=lane_claim_payload,
        dry_run=dry_run,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
    )
    blockers = _dedupe([*manifest_blockers, *wrapper_blockers, *closure_gate["blockers"]])
    ready = not blockers and closure_gate["ready_for_exact_eval_dispatch"] is True
    return {
        "schema": SUMMARY_SCHEMA,
        "candidate_manifest": {
            **manifest_summary,
            "path": str(candidate_manifest_path),
            "sha256": manifest_sha,
        },
        "runtime_proof": runtime_proof_summary,
        "lane_claim": lane_claim_summary,
        "closure_gate": closure_gate,
        "blockers": blockers,
        "manifest_blockers": manifest_blockers,
        "wrapper_blockers": wrapper_blockers,
        "ready_for_exact_eval_dispatch": ready,
        "score_claim": False,
        "dispatch_attempted": False,
        "archive_mutation_attempted": False,
        "omx_state_touched": False,
    }


def _validate_candidate_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    schema = manifest.get("schema")
    candidate_id = manifest.get("candidate_id")
    ready = manifest.get("ready_for_exact_eval_dispatch")
    dispatch_blockers = _string_list(manifest.get("dispatch_blockers"))
    candidate_archive = manifest.get("candidate_archive")
    monolithic_layout = manifest.get("monolithic_layout")
    replacements = manifest.get("replacements")

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

    if manifest.get("dispatch_blockers") is None:
        dispatch_blockers = []
    elif dispatch_blockers is None:
        blockers.append("candidate_manifest_dispatch_blockers_not_string_list")
        dispatch_blockers = []
    blockers.extend(
        f"candidate_manifest_unresolved_dispatch_blocker:{blocker}"
        for blocker in dispatch_blockers
        if blocker not in FIXABLE_MANIFEST_DISPATCH_BLOCKERS
    )
    if ready is True and dispatch_blockers:
        blockers.append("candidate_manifest_ready_true_with_dispatch_blockers")

    archive_summary, archive_blockers = _candidate_archive_summary(candidate_archive)
    layout_summary, layout_blockers = _monolithic_layout_summary(monolithic_layout)
    replacement_summary, replacement_blockers = _replacement_summary(replacements)
    blockers.extend(archive_blockers)
    blockers.extend(layout_blockers)
    blockers.extend(replacement_blockers)

    return (
        {
            "schema": schema if isinstance(schema, str) else None,
            "candidate_id": candidate_id if isinstance(candidate_id, str) else None,
            "ready_for_exact_eval_dispatch": ready if isinstance(ready, bool) else None,
            "dispatch_blockers": dispatch_blockers,
            "score_claim": manifest.get("score_claim"),
            "candidate_archive": archive_summary,
            "monolithic_layout": layout_summary,
            "replacement_count": replacement_summary["count"],
            "replacement_sections": replacement_summary["sections"],
        },
        _dedupe(blockers),
    )


def _candidate_archive_summary(value: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return {"path": None, "bytes": None, "sha256": None}, ["candidate_manifest_candidate_archive_missing"]
    path = value.get("path")
    byte_count = value.get("bytes")
    sha = value.get("sha256")
    if not isinstance(path, str) or not path:
        blockers.append("candidate_manifest_candidate_archive_path_missing")
    if not isinstance(byte_count, int) or byte_count < 0:
        blockers.append("candidate_manifest_candidate_archive_bytes_invalid")
    if not _is_sha256(sha):
        blockers.append("candidate_manifest_candidate_archive_sha256_invalid")
    return {
        "path": path if isinstance(path, str) else None,
        "bytes": byte_count if isinstance(byte_count, int) else None,
        "sha256": sha.lower() if _is_sha256(sha) else None,
    }, blockers


def _monolithic_layout_summary(value: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return {"grammar": None, "new_member_sha256": None}, ["candidate_manifest_monolithic_layout_missing"]
    grammar = value.get("grammar")
    new_member_sha = value.get("new_member_sha256")
    if not isinstance(grammar, str) or not grammar:
        blockers.append("candidate_manifest_monolithic_layout_grammar_missing")
    if not _is_sha256(new_member_sha):
        blockers.append("candidate_manifest_monolithic_layout_new_member_sha256_invalid")
    return {
        "grammar": grammar if isinstance(grammar, str) else None,
        "new_member_sha256": new_member_sha.lower() if _is_sha256(new_member_sha) else None,
    }, blockers


def _replacement_summary(value: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(value, list):
        return {"count": 0, "sections": []}, ["candidate_manifest_replacements_not_list"]
    if not value:
        blockers.append("candidate_manifest_replacements_empty")
    sections: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            blockers.append(f"candidate_manifest_replacement_not_object:{index}")
            continue
        section_name = item.get("section_name")
        section_sha = item.get("new_sha256")
        if not isinstance(section_name, str) or not section_name:
            blockers.append(f"candidate_manifest_replacement_section_name_missing:{index}")
            continue
        sections.append(section_name)
        if not _is_sha256(section_sha):
            blockers.append(f"candidate_manifest_replacement_new_sha256_invalid:{section_name}")
    return {"count": len(value), "sections": sections}, blockers


def _validate_runtime_proof(
    proof: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    path: Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    candidate_archive_sha = _nested_sha(manifest, "candidate_archive", "sha256")
    member_sha = _nested_sha(manifest, "monolithic_layout", "new_member_sha256")
    changed_sections = _changed_section_map(proof.get("changed_sections"))
    expected_sections = _expected_replacement_shas(manifest.get("replacements"))
    reported_archive_sha = proof.get("candidate_archive_sha256")
    reported_member_sha = proof.get("rebuilt_member_sha256", proof.get("new_member_sha256"))

    if proof.get("schema") != RUNTIME_PROOF_SCHEMA:
        blockers.append("runtime_proof_schema_mismatch")
    if proof.get("ready_for_exact_eval_runtime") is not True:
        blockers.append("runtime_proof_ready_for_exact_eval_runtime_not_true")
    if proof.get("blockers") not in (None, []):
        blockers.append("runtime_proof_reports_blockers")
    if proof.get("score_claim") is True:
        blockers.append("runtime_proof_score_claim_true")
    if not _sha_matches(reported_archive_sha, candidate_archive_sha):
        blockers.append("runtime_proof_candidate_archive_sha256_mismatch")
    if not _sha_matches(reported_member_sha, member_sha):
        blockers.append("runtime_proof_new_member_sha256_mismatch")
    for field in ("command_sha256", "log_sha256"):
        if not _is_sha256(proof.get(field)):
            blockers.append(f"runtime_proof_{field}_invalid")
    for section_name, expected_sha in expected_sections.items():
        if not _sha_matches(changed_sections.get(section_name), expected_sha):
            blockers.append(f"runtime_proof_changed_section_mismatch:{section_name}")

    return {
        "provided": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": proof.get("schema") if isinstance(proof.get("schema"), str) else None,
        "ready_for_exact_eval_runtime": proof.get("ready_for_exact_eval_runtime") is True,
        "candidate_archive_sha256_bound": _sha_matches(reported_archive_sha, candidate_archive_sha),
        "new_member_sha256_bound": _sha_matches(reported_member_sha, member_sha),
        "changed_sections_bound": sorted(
            section_name
            for section_name, expected_sha in expected_sections.items()
            if _sha_matches(changed_sections.get(section_name), expected_sha)
        ),
    }, blockers


def _validate_lane_claim(
    claim: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    path: Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    status = claim.get("claim_status", claim.get("status"))
    lane_id = claim.get("lane_id")
    instance_job_id = claim.get("instance_job_id", claim.get("job_name"))
    manifest_lane = manifest.get("lane_claim")
    expected_lane_id = manifest_lane.get("lane_id") if isinstance(manifest_lane, Mapping) else None
    expected_job_id = manifest_lane.get("instance_job_id") if isinstance(manifest_lane, Mapping) else None

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
    if not _is_sha256(claim.get("claim_row_sha256")):
        blockers.append("lane_claim_row_sha256_invalid")
    if isinstance(expected_lane_id, str) and expected_lane_id and lane_id != expected_lane_id:
        blockers.append("lane_claim_lane_id_mismatch")
    if isinstance(expected_job_id, str) and expected_job_id and instance_job_id != expected_job_id:
        blockers.append("lane_claim_instance_job_id_mismatch")

    return {
        "provided": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": claim.get("schema") if isinstance(claim.get("schema"), str) else None,
        "active": claim.get("active") is True,
        "lane_id": lane_id if isinstance(lane_id, str) else None,
        "instance_job_id": instance_job_id if isinstance(instance_job_id, str) else None,
        "claim_status": status if isinstance(status, str) else None,
    }, blockers


def _expected_replacement_shas(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    expected: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        section_name = item.get("section_name")
        section_sha = item.get("new_sha256")
        if isinstance(section_name, str) and _is_sha256(section_sha):
            expected[section_name] = section_sha.lower()
    return expected


def _changed_section_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            str(name): str(sha).lower()
            for name, sha in value.items()
            if isinstance(name, str) and _is_sha256(sha)
        }
    if not isinstance(value, list):
        return {}
    changed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("section_name", item.get("name"))
        sha = item.get("new_sha256", item.get("sha256"))
        if isinstance(name, str) and _is_sha256(sha):
            changed[name] = sha.lower()
    return changed


def _nested_sha(mapping: Mapping[str, Any], first: str, second: str) -> str | None:
    value = mapping.get(first)
    if not isinstance(value, Mapping):
        return None
    sha = value.get(second)
    return sha.lower() if _is_sha256(sha) else None


def _sha_matches(value: Any, expected: str | None) -> bool:
    return expected is not None and isinstance(value, str) and value.lower() == expected.lower()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _status_is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonolithicCandidatePreflightError(f"{label} at {path} must contain a JSON object")
    return payload


def _reject_omx_state_path(path: Path, *, label: str) -> None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == ".omx" and parts[index + 1] == "state":
            raise MonolithicCandidatePreflightError(
                f"refusing to touch .omx/state for {label}: {path}"
            )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return stable pretty JSON for stdout or `--json-out`."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--runtime-proof-json", type=Path)
    parser.add_argument("--lane-claim-json", type=Path)
    parser.add_argument(
        "--active-rate-only-floor-archive-bytes",
        type=int,
        default=ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Allow missing lane-claim proof for closure review; never authorizes dispatch.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.json_out is not None:
            _reject_omx_state_path(args.json_out, label="json output")
        payload = build_preflight(
            args.candidate_manifest,
            runtime_proof_json=args.runtime_proof_json,
            lane_claim_json=args.lane_claim_json,
            dry_run=args.dry_run,
            active_rate_only_floor_archive_bytes=args.active_rate_only_floor_archive_bytes,
        )
    except (MonolithicCandidatePreflightError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"monolithic candidate preflight failed: {exc}") from None

    text = dumps_json(payload)
    if args.json_out is None:
        print(text, end="")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if args.fail_if_not_ready and payload["ready_for_exact_eval_dispatch"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
