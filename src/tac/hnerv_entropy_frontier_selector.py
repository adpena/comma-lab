"""Deterministic HNeRV entropy-frontier candidate selection.

This module ranks local HNeRV entropy/archive candidates by archive bytes while
keeping exact-CUDA dispatch separate. It is a custody/readiness surface only:
it never claims a score, writes dispatch state, or authorizes GPU work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, repo_relative, sha256_file

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_entropy_frontier_selector"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
ACTIVE_RATE_ONLY_FLOOR_LABEL = "PR103-on-PR106 A++"
ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES = 185_578
ACTIVE_RATE_ONLY_FLOOR_SCORE = 0.2089810755823297
RATE_ONLY_FLOOR_BLOCKER_PREFIX = (
    "rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor"
)


class HnervEntropyFrontierSelectorError(ValueError):
    """Raised when candidate-selection input is malformed."""


def build_hnerv_entropy_frontier_selection(
    candidates: Sequence[tuple[str, str | Path]],
    *,
    active_candidates: Sequence[tuple[str, str | Path]] | None = None,
    active_rate_only_floor_archive_bytes: int | None = ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rank local HNeRV entropy candidates and select the next exact-eval packet.

    ``active_candidates`` are recorded for custody and excluded from the next
    selection. This is used when an exact eval is already queued/running and a
    worker needs the next artifact after that active row.
    """

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate_rows = [
        _candidate_record(label, Path(path), root, active=False) for label, path in candidates
    ]
    active_rows = [
        _candidate_record(label, Path(path), root, active=True)
        for label, path in (active_candidates or [])
    ]
    if not candidate_rows:
        raise HnervEntropyFrontierSelectorError("at least one candidate manifest is required")

    smallest_active = (
        min(active_rows, key=_selection_sort_key)
        if any(isinstance(row.get("candidate_archive_bytes"), int) for row in active_rows)
        else None
    )
    active_byte_floor = (
        int(smallest_active["candidate_archive_bytes"])
        if isinstance(smallest_active, Mapping)
        and isinstance(smallest_active.get("candidate_archive_bytes"), int)
        else None
    )
    rate_only_floor = _effective_rate_only_floor(
        active_byte_floor=active_byte_floor,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
    )
    candidate_rows = [
        _apply_rate_only_frontier_floor(row, active_rate_only_floor_archive_bytes=rate_only_floor)
        for row in candidate_rows
    ]
    exact_evaluable = [row for row in candidate_rows if row["exact_evaluable_after_lane_claim"] is True]
    selected = min(exact_evaluable, key=_selection_sort_key) if exact_evaluable else None
    selected_bytes = selected.get("candidate_archive_bytes") if isinstance(selected, Mapping) else None
    blocked_smaller = [
        row
        for row in candidate_rows
        if row["exact_evaluable_after_lane_claim"] is not True
        and isinstance(row.get("candidate_archive_bytes"), int)
        and isinstance(selected_bytes, int)
        and int(row["candidate_archive_bytes"]) < selected_bytes
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "selection_policy": {
            "purpose": "select_next_static_ready_hnerv_entropy_candidate_after_active_eval",
            "sort": "candidate_archive_bytes ascending, then label, then manifest path",
            "active_candidates_excluded": True,
            "active_candidate_bytes_are_rate_only_floor": active_byte_floor is not None,
            "active_rate_only_floor": {
                "label": ACTIVE_RATE_ONLY_FLOOR_LABEL,
                "archive_bytes": rate_only_floor,
                "score": ACTIVE_RATE_ONLY_FLOOR_SCORE,
            },
            "rate_only_exact_eval_spend_requires": [
                "candidate_archive_bytes_below_active_rate_only_floor",
                "or_explicit_scorer_changing_stack_path_declaration",
            ],
            "exact_evaluable_after_lane_claim_requires": [
                "candidate_archive_bytes_and_sha256",
                "byte_different_archive",
                "payload_changed_when_source_payload_sha256_is_available",
                "static_exact_eval_packet_ready",
                "score_claim_false",
                "dispatch_attempted_false",
                "no_remaining_blockers_except_lane_claim_and_exact_cuda",
            ],
        },
        "active_candidates": active_rows,
        "active_candidate": smallest_active,
        "active_candidate_byte_floor": active_byte_floor,
        "active_rate_only_floor_archive_bytes": rate_only_floor,
        "active_rate_only_floor_label": ACTIVE_RATE_ONLY_FLOOR_LABEL,
        "selected_next_candidate": selected,
        "selected_next_candidate_label": selected.get("label") if isinstance(selected, Mapping) else None,
        "ranked_candidates": sorted(candidate_rows, key=_selection_sort_key),
        "blocked_smaller_than_selected": sorted(blocked_smaller, key=_selection_sort_key),
        "dispatch_blockers": [
            "selection_manifest_is_not_dispatch_authorization",
            "requires_level2_lane_claim_before_gpu",
            "requires_exact_cuda_auth_eval_for_selected_archive_sha",
        ],
        "next_required_proofs": _next_required_proofs(selected),
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact operator-facing selection summary."""

    selected = manifest.get("selected_next_candidate")
    active = manifest.get("active_candidate")
    lines = [
        "# HNeRV Entropy Frontier Next-Candidate Selection",
        "",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool_text(manifest.get('dispatch_attempted') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
    ]
    if isinstance(active, Mapping):
        lines.extend(
            [
                "## Active Excluded Candidate",
                "",
                f"- label: `{active.get('label')}`",
                f"- archive_bytes: `{active.get('candidate_archive_bytes')}`",
                f"- archive_sha256: `{active.get('candidate_archive_sha256')}`",
                f"- active_candidate_byte_floor: `{manifest.get('active_candidate_byte_floor')}`",
                f"- active_rate_only_floor_archive_bytes: `{manifest.get('active_rate_only_floor_archive_bytes')}`",
                "",
            ]
        )
    if isinstance(selected, Mapping):
        lines.extend(
            [
                "## Selected Next Candidate",
                "",
                f"- label: `{selected.get('label')}`",
                f"- archive_path: `{selected.get('candidate_archive_path')}`",
                f"- archive_bytes: `{selected.get('candidate_archive_bytes')}`",
                f"- archive_sha256: `{selected.get('candidate_archive_sha256')}`",
                f"- payload_changed: `{_bool_text(selected.get('payload_changed') is True)}`",
                f"- rate_only_candidate: `{_bool_text(selected.get('rate_only_candidate') is True)}`",
                f"- scorer_changing_stack_path_declared: `{_bool_text(selected.get('scorer_changing_stack_path_declared') is True)}`",
                f"- byte_delta_vs_source_archive: `{selected.get('byte_delta_vs_source_archive')}`",
                f"- exact_evaluable_after_lane_claim: `{_bool_text(selected.get('exact_evaluable_after_lane_claim') is True)}`",
                f"- allowed_remaining_blockers: `{', '.join(selected.get('allowed_remaining_blockers') or [])}`",
                "",
            ]
        )
    else:
        lines.extend(["## Selected Next Candidate", "", "- none", ""])

    blocked = manifest.get("blocked_smaller_than_selected")
    if isinstance(blocked, list) and blocked:
        lines.extend(["## Smaller Blocked Candidates", ""])
        for row in blocked:
            if not isinstance(row, Mapping):
                continue
            blockers = ", ".join(str(item) for item in row.get("blocking_reasons") or [])
            lines.append(
                f"- `{row.get('label')}`: `{row.get('candidate_archive_bytes')}` bytes, "
                f"blocked by `{blockers}`"
            )
        lines.append("")
    ranked = manifest.get("ranked_candidates")
    if isinstance(ranked, list) and ranked:
        lines.extend(
            [
                "## Ranked Candidates",
                "",
                "| label | bytes | exact-evaluable after claim | blockers |",
                "|---|---:|---|---|",
            ]
        )
        for row in ranked:
            if not isinstance(row, Mapping):
                continue
            blockers = ", ".join(str(item) for item in row.get("blocking_reasons") or [])
            lines.append(
                "| {label} | {bytes_} | `{exact}` | {blockers} |".format(
                    label=row.get("label"),
                    bytes_=row.get("candidate_archive_bytes"),
                    exact=_bool_text(row.get("exact_evaluable_after_lane_claim") is True),
                    blockers=blockers or "none",
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Dispatch Boundary",
            "",
            "This manifest is local custody/readiness only. It does not claim a",
            "score and does not authorize or perform GPU dispatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_record(label: str, path: Path, root: Path, *, active: bool) -> dict[str, Any]:
    if not label:
        raise HnervEntropyFrontierSelectorError("candidate label must be nonempty")
    if not path.is_file():
        raise HnervEntropyFrontierSelectorError(f"candidate manifest does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise HnervEntropyFrontierSelectorError(f"candidate manifest is not a JSON object: {path}")

    candidate_archive = payload.get("candidate_archive")
    if not isinstance(candidate_archive, Mapping):
        candidate_archive = {}
    release_surface = payload.get("exact_eval_release_surface")
    if not isinstance(release_surface, Mapping):
        release_surface = {}
    candidate_diff = payload.get("candidate_diff")
    if not isinstance(candidate_diff, Mapping):
        candidate_diff = {}

    candidate_path = _first_str(
        payload.get("candidate_archive_path"),
        payload.get("output_archive_path"),
        candidate_archive.get("path"),
        release_surface.get("archive_path"),
        payload.get("archive_path"),
    )
    candidate_sha = _first_sha256(
        payload.get("candidate_archive_sha256"),
        payload.get("output_archive_sha256"),
        candidate_archive.get("archive_sha256"),
        payload.get("archive_sha256"),
    )
    candidate_bytes = _first_int(
        payload.get("candidate_archive_bytes"),
        payload.get("output_archive_bytes"),
        candidate_archive.get("archive_bytes"),
        payload.get("archive_bytes"),
    )
    candidate_payload_sha = _first_sha256(
        payload.get("candidate_payload_sha256"),
        candidate_archive.get("payload_sha256"),
        candidate_diff.get("candidate_payload_sha256"),
    )
    source_sha = _first_sha256(payload.get("source_archive_sha256"), candidate_diff.get("source_archive_sha256"))
    source_bytes = _first_int(payload.get("source_archive_bytes"), candidate_diff.get("source_archive_bytes"))
    source_payload_sha = _first_sha256(payload.get("source_payload_sha256"), candidate_diff.get("source_payload_sha256"))

    blockers = _collect_blockers(payload)
    allowed_remaining = [item for item in blockers if _is_allowed_remaining_blocker(item)]
    hard_blockers = [item for item in blockers if not _is_allowed_remaining_blocker(item)]
    archive_changed = None
    if candidate_sha and source_sha:
        archive_changed = candidate_sha != source_sha
    payload_changed = None
    if candidate_payload_sha and source_payload_sha:
        payload_changed = candidate_payload_sha != source_payload_sha
    identity_blockers = _identity_blockers(
        candidate_sha=candidate_sha,
        candidate_bytes=candidate_bytes,
        candidate_path=candidate_path,
        candidate_payload_sha=candidate_payload_sha,
        source_sha=source_sha,
        source_payload_sha=source_payload_sha,
        archive_changed=archive_changed,
        payload_changed=payload_changed,
    )
    static_packet_ready = _static_packet_ready(payload)
    score_claim = payload.get("score_claim") is True
    dispatch_attempted = payload.get("dispatch_attempted") is True
    archive_exists = _archive_exists(candidate_path, root, manifest_dir=path.parent)
    archive_file_sha = (
        _archive_file_sha(candidate_path, root, manifest_dir=path.parent)
        if archive_exists
        else ""
    )
    archive_file_bytes = (
        _archive_file_bytes(candidate_path, root, manifest_dir=path.parent)
        if archive_exists
        else None
    )
    archive_existence_blockers = [] if archive_exists else ["candidate_archive_missing"]
    exact_evaluable = (
        not active
        and static_packet_ready
        and not score_claim
        and not dispatch_attempted
        and archive_exists
        and not identity_blockers
        and not hard_blockers
    )
    blocking_reasons = _unique_ordered(
        [
            *identity_blockers,
            *archive_existence_blockers,
            *([] if static_packet_ready else ["static_exact_eval_packet_not_ready"]),
            *(["score_claim_true"] if score_claim else []),
            *(["dispatch_attempted_true"] if dispatch_attempted else []),
            *hard_blockers,
        ]
    )
    if archive_exists and candidate_sha and archive_file_sha != candidate_sha:
        blocking_reasons.append("candidate_archive_sha256_file_mismatch")
        exact_evaluable = False
    if archive_exists and candidate_bytes is not None and archive_file_bytes != candidate_bytes:
        blocking_reasons.append("candidate_archive_bytes_file_mismatch")
        exact_evaluable = False

    stack_declaration = scorer_changing_stack_path_declaration(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "active_excluded": active,
        "manifest_path": repo_relative(path, root),
        "manifest_bytes": path.stat().st_size,
        "manifest_sha256": sha256_file(path),
        "candidate_archive_path": candidate_path,
        "candidate_archive_exists": archive_exists,
        "candidate_archive_file_bytes": archive_file_bytes,
        "candidate_archive_file_sha256": archive_file_sha,
        "candidate_archive_sha256": candidate_sha,
        "candidate_archive_bytes": candidate_bytes,
        "candidate_payload_sha256": candidate_payload_sha,
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_bytes,
        "source_payload_sha256": source_payload_sha,
        "byte_delta_vs_source_archive": (
            candidate_bytes - source_bytes
            if isinstance(candidate_bytes, int) and isinstance(source_bytes, int)
            else None
        ),
        "rate_score_delta_if_components_equal": (
            round((candidate_bytes - source_bytes) * RATE_SCORE_PER_BYTE, 12)
            if isinstance(candidate_bytes, int) and isinstance(source_bytes, int)
            else None
        ),
        "archive_changed": archive_changed,
        "payload_changed": payload_changed,
        "rate_only_candidate": _declares_rate_only_candidate(payload),
        "scorer_changing_stack_path_declaration": stack_declaration,
        "scorer_changing_stack_path_declared": stack_declaration["declared"],
        "ready_for_archive_preflight": _ready_for_archive_preflight(payload),
        "static_packet_ready": static_packet_ready,
        "score_claim": score_claim,
        "dispatch_attempted": dispatch_attempted,
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch") is True,
        "exact_evaluable_after_lane_claim": exact_evaluable,
        "allowed_remaining_blockers": allowed_remaining,
        "hard_blockers": hard_blockers,
        "identity_blockers": identity_blockers,
        "blocking_reasons": _unique_ordered(blocking_reasons),
    }


def _effective_rate_only_floor(
    *,
    active_byte_floor: int | None,
    active_rate_only_floor_archive_bytes: int | None,
) -> int | None:
    floors = [
        floor
        for floor in (active_byte_floor, active_rate_only_floor_archive_bytes)
        if isinstance(floor, int) and not isinstance(floor, bool) and floor > 0
    ]
    return min(floors) if floors else None


def build_rate_only_floor_proof(
    payload: Mapping[str, Any],
    *,
    candidate_archive_bytes: int | None,
    declared_rate_only: bool,
    active_rate_only_floor_archive_bytes: int | None = ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
) -> dict[str, Any]:
    """Return the planning-only active-floor gate for rate-only exact-eval spend."""

    declaration = scorer_changing_stack_path_declaration(payload)
    blockers: list[str] = []
    beats_floor: bool | None = None
    if (
        declared_rate_only
        and active_rate_only_floor_archive_bytes is not None
        and not declaration["declared"]
    ):
        if candidate_archive_bytes is None:
            blockers.append("rate_only_candidate_archive_bytes_missing_for_active_floor_policy")
        else:
            beats_floor = candidate_archive_bytes < active_rate_only_floor_archive_bytes
            if not beats_floor:
                blockers.append(
                    f"{RATE_ONLY_FLOOR_BLOCKER_PREFIX}:{active_rate_only_floor_archive_bytes}"
                )
    return {
        "schema": "rate_only_frontier_floor_policy_v1",
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "active_floor_label": ACTIVE_RATE_ONLY_FLOOR_LABEL,
        "active_floor_archive_bytes": active_rate_only_floor_archive_bytes,
        "active_floor_score": ACTIVE_RATE_ONLY_FLOOR_SCORE,
        "candidate_archive_bytes": candidate_archive_bytes,
        "declared_rate_only": declared_rate_only,
        "beats_active_floor": beats_floor,
        "scorer_changing_stack_path_declared": declaration["declared"],
        "scorer_changing_stack_path_sources": declaration["sources"],
        "exact_eval_spend_allowed_by_policy": not blockers,
        "blockers": _unique_ordered(blockers),
    }


def scorer_changing_stack_path_declaration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Detect an explicit scorer-changing stack-path declaration in a manifest."""

    sources: list[str] = []
    for key in (
        "scorer_changing_stack_path",
        "scorer_changing_stack_paths",
        "score_changing_stack_path",
        "score_changing_stack_paths",
        "scorer_changing_stack_plan",
        "scorer_changing_stack_manifest",
    ):
        if _has_nonempty_declaration(payload.get(key)):
            sources.append(key)
    for key in ("stack_path", "stack_paths", "stack_plan", "stack_manifest"):
        text = _declaration_text(payload.get(key)).lower()
        if text and (
            "scorer_changing" in text
            or "scorer-changing" in text
            or "score_changing" in text
            or "score-changing" in text
        ):
            sources.append(key)
    return {
        "schema": "scorer_changing_stack_path_declaration_v1",
        "declared": bool(sources),
        "sources": _unique_ordered(sources),
    }


def _apply_rate_only_frontier_floor(
    row: dict[str, Any],
    *,
    active_rate_only_floor_archive_bytes: int | None,
) -> dict[str, Any]:
    """Block rate-only candidates that do not improve the active byte floor."""

    if active_rate_only_floor_archive_bytes is None or row.get("rate_only_candidate") is not True:
        return row
    archive_bytes = row.get("candidate_archive_bytes")
    if not isinstance(archive_bytes, int):
        return row
    proof_payload = {"scorer_changing_stack_path": True} if row.get(
        "scorer_changing_stack_path_declared"
    ) is True else {}
    proof = build_rate_only_floor_proof(
        proof_payload,
        candidate_archive_bytes=archive_bytes,
        declared_rate_only=True,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
    )
    out = dict(row)
    out["rate_only_floor_proof"] = proof
    if proof["exact_eval_spend_allowed_by_policy"] is True:
        return out
    out["exact_evaluable_after_lane_claim"] = False
    out["active_rate_only_floor_archive_bytes"] = active_rate_only_floor_archive_bytes
    out["byte_delta_vs_active_rate_only_floor"] = archive_bytes - active_rate_only_floor_archive_bytes
    out["rate_only_floor_proof"] = proof
    out["hard_blockers"] = _unique_ordered([*out.get("hard_blockers", []), *proof["blockers"]])
    out["blocking_reasons"] = _unique_ordered(
        [*out.get("blocking_reasons", []), *proof["blockers"]]
    )
    return out


def _next_required_proofs(selected: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(selected, Mapping):
        return [
            "static_exact_eval_packet_ready_candidate",
            "candidate_archive_bytes_and_sha256",
            "payload_change_proof",
        ]
    return [
        f"level2_lane_claim_for:{selected.get('label')}",
        f"exact_cuda_auth_eval_for_archive_sha:{selected.get('candidate_archive_sha256')}",
        "contest_auth_eval_json_with_recomputed_components",
        "terminal_dispatch_claim_linkage_after_eval",
    ]


def _identity_blockers(
    *,
    candidate_sha: str,
    candidate_bytes: int | None,
    candidate_path: str,
    candidate_payload_sha: str,
    source_sha: str,
    source_payload_sha: str,
    archive_changed: bool | None,
    payload_changed: bool | None,
) -> list[str]:
    blockers: list[str] = []
    if not candidate_path:
        blockers.append("candidate_archive_path_missing")
    if candidate_bytes is None:
        blockers.append("candidate_archive_bytes_missing")
    if not candidate_sha:
        blockers.append("candidate_archive_sha256_missing")
    if source_sha and archive_changed is False:
        blockers.append("candidate_archive_sha256_not_byte_different")
    if source_payload_sha and not candidate_payload_sha:
        blockers.append("candidate_payload_sha256_missing")
    if source_payload_sha and payload_changed is False:
        blockers.append("candidate_payload_sha256_not_changed")
    return blockers


def _collect_blockers(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "archive_build_blockers",
        "dispatch_blockers",
        "runtime_adapter_blockers",
        "exact_cuda_remaining_blockers",
        "readiness_blockers",
    ):
        values.extend(_string_items(payload.get(key)))
    for parent_key, child_keys in (
        ("exact_eval_packet_readiness", ("remaining_dispatch_blockers", "dispatch_blockers")),
        ("candidate_diff_audit", ("blockers", "dispatch_blockers")),
        ("fixed_runtime_preflight", ("blockers", "remaining_blockers")),
        ("runtime_adapter_proof", ("blockers",)),
        ("runtime_tree_inflate_output_parity", ("blockers", "remaining_blockers")),
    ):
        parent = payload.get(parent_key)
        if not isinstance(parent, Mapping):
            continue
        for child_key in child_keys:
            values.extend(_string_items(parent.get(child_key)))
    return _unique_ordered(values)


def _declares_rate_only_candidate(payload: Mapping[str, Any]) -> bool:
    if payload.get("rate_only") is True:
        return True
    for key in (
        "expected_total_score_delta_rate_only",
        "candidate_rate_score_delta_if_runtime_supported_and_components_equal",
        "rate_score_delta_if_components_equal",
    ):
        if payload.get(key) is not None:
            return True
    token_blob = " ".join(
        _declaration_text(payload.get(key))
        for key in (
            "tool",
            "family",
            "family_group",
            "pareto_scope",
            "role",
            "action_class",
            "evidence_grade",
            "interaction_assumptions",
            "packet_kind",
        )
    ).strip().lower()
    if not token_blob:
        return True
    return any(
        token in token_blob
        for token in (
            "rate_only",
            "rate-only",
            "raw_equivalent",
            "raw-equivalent",
            "byte_equivalent",
            "byte-equivalent",
            "components_equal",
        )
    )


def _has_nonempty_declaration(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_has_nonempty_declaration(item) for item in value)
    return False


def _declaration_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_declaration_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return " ".join(_declaration_text(item) for item in value)
    return str(value)


def _is_allowed_remaining_blocker(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    allowed_fragments = (
        "lane_dispatch_claim",
        "requires_lane_dispatch_claim",
        "level2_lane_claim",
        "claim_dispatch_lane",
        "exact_cuda_auth_eval",
        "requires_exact_cuda",
        "run_archive.zip_->_inflate.sh_->_upstream/evaluate.py_on_cuda",
        "terminal_dispatch_claim",
        "auth_eval_artifact_exists",
        "contest_auth_eval_json",
    )
    return any(fragment in normalized for fragment in allowed_fragments)


def _static_packet_ready(payload: Mapping[str, Any]) -> bool:
    if payload.get("static_packet_ready") is True:
        return True
    if payload.get("ready_for_exact_eval_packet") is True:
        return True
    if payload.get("passed") is True and payload.get("tool") == "tools.prove_pr103_pr106_final_runtime_packet":
        return True
    readiness = payload.get("exact_eval_packet_readiness")
    if isinstance(readiness, Mapping) and readiness.get("static_packet_ready") is True:
        return True
    release = payload.get("exact_eval_release_surface")
    if isinstance(release, Mapping) and release.get("contract"):
        return payload.get("ready_for_archive_preflight") is True
    return False


def _ready_for_archive_preflight(payload: Mapping[str, Any]) -> bool:
    if payload.get("ready_for_archive_preflight") is True:
        return True
    audit = payload.get("candidate_diff_audit")
    return isinstance(audit, Mapping) and audit.get("ready_for_archive_preflight") is True


def _archive_exists(value: str, root: Path, *, manifest_dir: Path | None = None) -> bool:
    if not value:
        return False
    return _resolve_candidate_path(value, root, manifest_dir=manifest_dir).is_file()


def _archive_file_sha(value: str, root: Path, *, manifest_dir: Path | None = None) -> str:
    return sha256_file(_resolve_candidate_path(value, root, manifest_dir=manifest_dir))


def _archive_file_bytes(value: str, root: Path, *, manifest_dir: Path | None = None) -> int:
    return _resolve_candidate_path(value, root, manifest_dir=manifest_dir).stat().st_size


def _resolve_repo_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_candidate_path(value: str, root: Path, *, manifest_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = root / path
    if repo_path.exists() or manifest_dir is None:
        return repo_path
    return manifest_dir / path


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    archive_bytes = row.get("candidate_archive_bytes")
    byte_key = archive_bytes if isinstance(archive_bytes, int) else 10**18
    return int(byte_key), str(row.get("label") or ""), str(row.get("manifest_path") or "")


def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _first_sha256(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and _is_sha256(value):
            return value
    return ""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _unique_ordered(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES",
    "ACTIVE_RATE_ONLY_FLOOR_LABEL",
    "ACTIVE_RATE_ONLY_FLOOR_SCORE",
    "HnervEntropyFrontierSelectorError",
    "build_hnerv_entropy_frontier_selection",
    "build_rate_only_floor_proof",
    "render_markdown",
    "scorer_changing_stack_path_declaration",
]
