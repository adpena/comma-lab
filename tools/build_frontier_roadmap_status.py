#!/usr/bin/env python3
"""Build a live-safe roadmap status from the cross-paradigm frontier queue.

This is an operator artifact. It does not build archives, dispatch jobs, or
claim scores. Its purpose is to make the next tranche safe on a shared ``main``
by joining the static frontier inventory with the current worktree state.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402
from tools.build_cross_paradigm_frontier_inventory import build_inventory  # noqa: E402
from tools.build_field_meta_dispatch_selection import build_selection_report  # noqa: E402

SCHEMA_VERSION = 4
DEFAULT_PACKET_MANIFEST_GLOBS = (
    "experiments/results/**/wr01_exact_eval_packet.json",
    "experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_*/hnerv_lowlevel_exact_eval_packet.json",
    "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_*/hnerv_lowlevel_exact_eval_packet.json",
)


def git_dirty_paths(repo_root: Path) -> list[str]:
    """Return sorted dirty paths from ``git status --porcelain=v1 -z``."""

    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=False,
    )
    raw = proc.stdout
    if not raw:
        return []
    entries = raw.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        text = entry.decode("utf-8", errors="replace")
        status = text[:2]
        path = text[3:]
        if (status[0] == "R" or status[0] == "C") and index < len(entries) and entries[index]:
            path = entries[index].decode("utf-8", errors="replace")
            index += 1
        if path:
            paths.append(path)
    return sorted(set(paths))


def _path_matches(candidate: str, dirty_path: str) -> bool:
    return (
        dirty_path == candidate
        or dirty_path.startswith(candidate.rstrip("/") + "/")
        or candidate.startswith(dirty_path.rstrip("/") + "/")
    )


def dirty_paths_for_row(row: dict[str, Any], dirty_paths: list[str]) -> list[str]:
    """Return dirty paths that intersect a frontier row's code/evidence paths."""

    watched = [*row.get("code_paths", []), *row.get("evidence_paths", [])]
    matches = []
    for dirty_path in dirty_paths:
        if any(_path_matches(str(path), dirty_path) for path in watched):
            matches.append(dirty_path)
    return sorted(set(matches))


def _readiness_stage(row: dict[str, Any], dirty_matches: list[str]) -> str:
    if dirty_matches:
        return "blocked_by_dirty_worktree"
    if row.get("score_snapshot"):
        return "exact_evidence_present_review_before_promotion"
    action = str(row.get("action_class") or "")
    if "exact_eval" in action or "promote" in action:
        return "needs_lane_claim_and_exact_cuda"
    if "build" in action or "prove" in action or "replace" in action:
        return "needs_byte_closed_candidate_or_fixture"
    return "needs_research_or_contract_hardening"


def _row_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["key"]): row for row in rows}


def _keys_present(rows_by_key: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if key in rows_by_key]


def build_next_comprehensive_tranche(
    rows: list[dict[str, Any]],
    *,
    field_meta_candidate_packet_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the next score-lowering tranche as a deterministic work plan.

    This intentionally remains an orchestration artifact. It names the next
    code/research tranche and its promotion gates, but never marks any candidate
    as dispatchable by itself.
    """

    rows_by_key = _row_by_key(rows)
    exact_eval_candidates = [
        row["key"]
        for row in rows
        if row["safe_to_touch_now"]
        and row["readiness_stage"]
        in {
            "needs_lane_claim_and_exact_cuda",
            "exact_evidence_present_review_before_promotion",
        }
    ]
    byte_closed_candidates = [
        row["key"]
        for row in rows
        if row["safe_to_touch_now"] and row["readiness_stage"] == "needs_byte_closed_candidate_or_fixture"
    ]
    research_hardening_candidates = [
        row["key"]
        for row in rows
        if row["safe_to_touch_now"] and row["readiness_stage"] == "needs_research_or_contract_hardening"
    ]

    workstreams = [
        {
            "id": "rate_frontier_closure",
            "objective": (
                "Turn HNeRV rate-only opportunities into byte-equivalent archives "
                "or retire them as measured rate-negative implementations."
            ),
            "keys": _keys_present(
                rows_by_key,
                ("hnerv_lowlevel_brotli_repack", "hnerv_per_tensor_context_entropy"),
            ),
            "acceptance_gates": [
                "candidate archive manifest records exact bytes and SHA-256",
                "decoder/runtime parity proves output equivalence or declares scorer-changing scope",
                "entropy-gap target ranking names concrete next artifact, not only a family",
                "ready_for_exact_eval_dispatch remains false until static preflight and matching active claim pass",
            ],
        },
        {
            "id": "scorer_changing_mask_payload",
            "objective": (
                "Build the first real byte-closed mask/categorical payload candidate "
                "instead of another fixture or unconsumed sidechannel."
            ),
            "keys": _keys_present(
                rows_by_key,
                (
                    "hnerv_wavelet_wr01_apply",
                    "categorical_qma9_clade_spade_openpilot",
                    "cmg3_predictive_mask_grammar",
                ),
            ),
            "acceptance_gates": [
                "charged archive member is consumed by inflate runtime",
                "full decode/re-encode or runtime-loader parity is proven",
                "component-collapse risks are recorded before any lane claim",
                "no uncharged openpilot/comma labels, weights, or sidecars are read at inflate",
            ],
        },
        {
            "id": "joint_stack_runtime_closure",
            "objective": (
                "Move JCSP and sensitivity-aware stack planning from typed manifests "
                "toward archive members with charged bytes and runtime consumers."
            ),
            "keys": _keys_present(
                rows_by_key,
                ("joint_admm_balle_arithmetic_stack", "sensitivity_omega_w_v3"),
            ),
            "acceptance_gates": [
                "per-stream bytes reconcile against charged archive members",
                "fixture-only streams stay blocked from dispatch",
                "stub sensitivity artifacts fail closed",
                "individual component wins are not treated as composable without stacked eval",
            ],
        },
        {
            "id": "geometry_pose_foveation_grounding",
            "objective": (
                "Convert LA-pose, RAFT/radial, and telescopic foveation from proposal "
                "signals into calibrated byte-bearing atom rows."
            ),
            "keys": _keys_present(
                rows_by_key,
                ("telescopic_foveation_field", "lapose_motion_atom_allocator", "raft_radial_openpilot_pose"),
            ),
            "acceptance_gates": [
                "geometry proposal records charged-artifact and runtime-consumer status",
                "pose/foveation confidence is calibrated against measured component evidence",
                "foveation remains ranking feedback until an archive consumer exists",
                "small pose-error cliffs are treated as dispatch blockers, not warnings",
            ],
        },
        {
            "id": "field_meta_selection",
            "objective": (
                "Let the meta-Lagrangian/field-equation planner choose the exact next "
                "candidate only after Pareto, KKT-proof, custody, and interaction gates pass."
            ),
            "keys": _keys_present(rows_by_key, ("meta_lagrangian_cross_paradigm_allocator",)),
            "acceptance_gates": [
                "candidate rows carry family/conflict and byte-closed manifest fields",
                "proxy, dominated, non-KKT-proof, and non-byte-closed rows carry diagnostic blockers",
                "expected-information-gain is recorded separately from predicted score",
                "selected exact-eval packet has a matching active lane claim plus candidate-specific static preflight",
            ],
        },
    ]
    workstreams = [stream for stream in workstreams if stream["keys"]]
    for stream in workstreams:
        keys = [str(key) for key in stream["keys"]]
        dirty_blocked_keys = [
            key
            for key in keys
            if rows_by_key[key].get("dirty_path_blockers")
        ]
        stream["unblocked_keys"] = [
            key
            for key in keys
            if rows_by_key[key].get("safe_to_touch_now") is True
        ]
        stream["dirty_blocked_keys"] = dirty_blocked_keys
        stream["all_keys_safe_to_touch_now"] = not dirty_blocked_keys
    return {
        "schema": "next_comprehensive_tranche_v1",
        "name": "byte-closed frontier closure and field-selected exact-eval tranche",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "objective": (
            "Prepare score-reduction attempts by converting the strongest current "
            "planning surfaces into byte-closed, runtime-consumed candidates, then "
            "select any exact CUDA packet only through Pareto/KKT-proof/meta-Lagrangian gates."
        ),
        "candidate_pools": {
            "exact_eval_or_review": exact_eval_candidates,
            "needs_byte_closed_candidate": byte_closed_candidates,
            "needs_research_or_contract_hardening": research_hardening_candidates,
        },
        "field_meta_candidate_packet_selection": field_meta_candidate_packet_selection
        or {
            "schema_version": 3,
            "tool": "tools/build_field_meta_dispatch_selection.py",
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
            "candidate_preflight_ready_for_exact_eval_dispatch": False,
            "candidate_local_preflight_ready": False,
            "candidate_static_preflight_ready": False,
            "candidate_count": 0,
            "candidate_local_preflight_ready_count": 0,
            "candidate_static_preflight_ready_count": 0,
            "ready_candidate_count": 0,
            "field_selection_ready_for_exact_eval_dispatch": False,
            "field_selection_ready_for_exact_eval_dispatch_count": 0,
            "dirty_blocked_candidate_count": 0,
            "kkt_ready_for_field_planning_count": 0,
            "pareto_summary": {
                "frontier_count": 0,
                "dominated_count": 0,
                "eligible_count": 0,
                "scope_counts": {},
            },
            "lexicographic_feasibility_order": [
                "field_selection_ready_for_exact_eval_dispatch desc",
                "candidate_static_preflight_ready desc",
                "archive_proof.byte_closed desc",
                "runtime_proof.runtime_closed desc",
                "clean_worktree_overlap desc",
                "pareto_frontier desc",
                "kkt_ready_for_field_planning desc",
                "expected_total_score_delta asc",
                "expected_information_gain_nats desc",
            ],
            "selected_candidate": None,
            "rows": [],
            "report_blockers": ["no_candidate_packet_manifests_supplied"],
        },
        "workstreams": workstreams,
        "global_acceptance_gates": [
            "no score claim without exact CUDA auth eval JSON",
            "no remote/GPU dispatch without an active lane claim",
            "archive.zip -> inflate.sh -> upstream/evaluate.py remains canonical",
            "all runtime inputs are charged archive members or fixed contest code",
            "all builders are deterministic and cross-platform by default",
            "negative, no-op, and blocked results are preserved as artifacts",
        ],
        "end_of_tranche_report_required": [
            "changed paths and owning workers",
            "focused tests plus all-lanes preflight result",
            "new candidate artifacts with SHA-256 and bytes when produced",
            "rows newly promoted, still blocked, or retired as scoped measured negatives",
            "the next comprehensive tranche generated from the updated roadmap status",
        ],
        "dispatch_blockers": [
            "tranche_plan_only",
            "requires_candidate_specific_archive_manifest",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def build_roadmap_status(
    *,
    repo_root: Path,
    dirty_paths: list[str] | None = None,
    packet_manifest_paths: list[Path] | None = None,
    packet_manifest_globs: list[str] | None = None,
    claims_path: Path | None = None,
    now_utc: str | None = None,
    operator_approved_exact_cuda: bool | None = None,
) -> dict[str, Any]:
    """Join static frontier inventory with live dirty-worktree blockers."""

    inventory = build_inventory(repo_root=repo_root)
    live_dirty_paths = git_dirty_paths(repo_root) if dirty_paths is None else sorted(set(dirty_paths))
    selection_manifest_paths = packet_manifest_paths or []
    selection_manifest_globs = (
        list(DEFAULT_PACKET_MANIFEST_GLOBS)
        if packet_manifest_paths is None and packet_manifest_globs is None
        else packet_manifest_globs or []
    )
    rows = []
    stage_counts: Counter[str] = Counter()
    dirty_blocked_count = 0
    for row in inventory["frontier_action_queue"]:
        source_row = next(item for item in inventory["rows"] if item["key"] == row["key"])
        dirty_matches = dirty_paths_for_row(source_row, live_dirty_paths)
        stage = _readiness_stage(source_row, dirty_matches)
        stage_counts[stage] += 1
        dirty_blocked_count += int(bool(dirty_matches))
        rows.append(
            {
                "key": row["key"],
                "title": source_row["title"],
                "priority_tier": row["priority_tier"],
                "action_class": row["action_class"],
                "role": row["role"],
                "status": row["status"],
                "evidence_grade": source_row["evidence_grade"],
                "stackability": source_row["stackability"],
                "replacement_potential": source_row["replacement_potential"],
                "readiness_stage": stage,
                "dirty_path_blockers": dirty_matches,
                "safe_to_touch_now": not dirty_matches,
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "next_patch": row["next_patch"],
                "blockers": row["blockers"],
                "missing_code_path_count": source_row["path_audit"]["code"]["missing_count"],
                "missing_evidence_path_count": source_row["path_audit"]["evidence"]["missing_count"],
            }
        )
    next_unblocked = [
        row["key"]
        for row in rows
        if row["safe_to_touch_now"]
    ][:5]
    field_meta_candidate_packet_selection = build_selection_report(
        repo_root=repo_root,
        manifest_paths=selection_manifest_paths,
        manifest_globs=selection_manifest_globs,
        claims_path=claims_path,
        now_utc=now_utc,
        dirty_paths=live_dirty_paths,
        operator_approved_exact_cuda=operator_approved_exact_cuda,
    )
    next_tranche = build_next_comprehensive_tranche(
        rows,
        field_meta_candidate_packet_selection=field_meta_candidate_packet_selection,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/build_frontier_roadmap_status.py",
        "inventory_tool": inventory["tool"],
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "row_count": len(rows),
        "dirty_path_count": len(live_dirty_paths),
        "dirty_paths": live_dirty_paths,
        "dirty_blocked_row_count": dirty_blocked_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "next_unblocked_keys": next_unblocked,
        "next_comprehensive_tranche": next_tranche,
        "rows": rows,
        "dispatch_blockers": [
            "roadmap_status_only",
            "requires_candidate_specific_archive_manifest",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    packet_selection = payload["next_comprehensive_tranche"]["field_meta_candidate_packet_selection"]
    selected_packet = packet_selection.get("selected_candidate") or {}
    selected_operator = selected_packet.get("operator_next_steps_summary") or {}
    selected_next_action = selected_operator.get("next_local_non_gpu_action") or {}
    lines = [
        "# Frontier Roadmap Status",
        "",
        "Live-safe operator roadmap. It does not claim scores or dispatch work.",
        "",
        f"- row_count: `{payload['row_count']}`",
        f"- dirty_path_count: `{payload['dirty_path_count']}`",
        f"- dirty_blocked_row_count: `{payload['dirty_blocked_row_count']}`",
        f"- next_unblocked_keys: `{', '.join(payload['next_unblocked_keys'])}`",
        "",
        "## Next Comprehensive Tranche",
        "",
        f"- name: `{_md(payload['next_comprehensive_tranche']['name'])}`",
        f"- objective: {_md(payload['next_comprehensive_tranche']['objective'])}",
        f"- candidate_packet_count: `{packet_selection['candidate_count']}`",
        f"- candidate_local_preflight_ready_count: `{packet_selection['candidate_local_preflight_ready_count']}`",
        f"- candidate_static_preflight_ready_count: `{packet_selection['candidate_static_preflight_ready_count']}`",
        f"- ready_candidate_packet_count: `{packet_selection['ready_candidate_count']}`",
        f"- field_selection_ready_candidate_packet_count: `{packet_selection.get('field_selection_ready_for_exact_eval_dispatch_count', 0)}`",
        f"- dirty_blocked_candidate_packet_count: `{packet_selection['dirty_blocked_candidate_count']}`",
        f"- pareto_frontier_candidate_packet_count: `{packet_selection['pareto_summary']['frontier_count']}`",
        f"- kkt_ready_candidate_packet_count: `{packet_selection['kkt_ready_for_field_planning_count']}`",
        f"- selected_candidate_packet: `{_md(selected_packet.get('candidate_id') or 'none')}`",
        f"- selected_candidate_decision: `{_md(selected_packet.get('selection_decision') or 'none')}`",
        f"- selected_candidate_frontier_reason: `{_md((selected_packet.get('non_dominated_frontier_reason') or {}).get('reason') or 'none')}`",
        f"- selected_candidate_exact_blocker_count: `{_md((selected_packet.get('exact_dispatch_blockers') or {}).get('blocker_count') or 0)}`",
        f"- selected_candidate_next_local_non_gpu_step: `{_md(selected_next_action.get('id') or 'none')}`",
        f"- selected_candidate_next_local_non_gpu_command: `{_md(selected_operator.get('next_local_non_gpu_command') or 'none')}`",
        f"- selected_candidate_claim_blockers: `{_md_list(selected_operator.get('claim_blockers') or [])}`",
        f"- selected_candidate_static_refresh_status: `{_md(selected_operator.get('static_refresh_status') or 'none')}`",
        f"- selected_candidate_refresh_blockers: `{_md_list(selected_operator.get('refresh_blockers') or [])}`",
        f"- selected_candidate_approval_blockers: `{_md_list(selected_operator.get('approval_blockers') or [])}`",
        f"- selected_candidate_operator_approval_source: `{_md((selected_operator.get('operator_approval_state') or {}).get('source') or 'none')}`",
        "",
        "| workstream | keys | dirty-blocked keys | acceptance gates |",
        "|---|---|---|---|",
    ]
    for stream in payload["next_comprehensive_tranche"]["workstreams"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(stream['id'])}`",
                    "<br>".join(f"`{_md(key)}`" for key in stream["keys"]),
                    (
                        "<br>".join(f"`{_md(key)}`" for key in stream["dirty_blocked_keys"])
                        if stream["dirty_blocked_keys"]
                        else "`none`"
                    ),
                    "<br>".join(_md(gate) for gate in stream["acceptance_gates"]),
                )
            )
            + " |"
        )
    if packet_selection.get("rows"):
        lines += [
            "",
            "## Candidate Packets",
            "",
            "| candidate | decision | local non-GPU step | claim blockers | refresh blockers | approval blockers | next local non-GPU command |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in packet_selection["rows"]:
            summary = row.get("operator_next_steps_summary") or {}
            action = summary.get("next_local_non_gpu_action") or {}
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{_md(row.get('candidate_id') or '')}`",
                        f"`{_md(row.get('selection_decision') or '')}`",
                        f"`{_md(action.get('id') or 'none')}`",
                        _md_list(summary.get("claim_blockers") or []),
                        _md_list(summary.get("refresh_blockers") or []),
                        _md_list(summary.get("approval_blockers") or []),
                        _md(summary.get("next_local_non_gpu_command") or "none"),
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Frontier Rows",
        "",
        "| key | tier | role | stage | safe | action | evidence | blockers | next patch |",
        "|---|---:|---|---|---|---|---|---:|---|",
    ]
    for row in payload["rows"]:
        blocker_count = (
            len(row["dirty_path_blockers"])
            + int(row["missing_code_path_count"])
            + int(row["missing_evidence_path_count"])
            + len(row["blockers"])
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(row['key'])}`",
                    str(row["priority_tier"]),
                    f"`{_md(row['role'])}`",
                    f"`{_md(row['readiness_stage'])}`",
                    "`yes`" if row["safe_to_touch_now"] else "`no`",
                    f"`{_md(row['action_class'])}`",
                    _md(row["evidence_grade"]),
                    str(blocker_count),
                    _md(row["next_patch"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _md_list(values: list[object]) -> str:
    if not values:
        return "none"
    return ", ".join(_md(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--packet-manifest", action="append", type=Path)
    parser.add_argument("--packet-manifest-glob", action="append")
    parser.add_argument("--claims-path", type=Path)
    parser.add_argument("--now-utc")
    parser.add_argument(
        "--operator-approved-exact-cuda",
        action="store_true",
        help=(
            "Record operator exact-CUDA approval for packet-selection context only; "
            "does not satisfy env, lane-claim, exact-CUDA, or review gates."
        ),
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_roadmap_status(
        repo_root=args.repo_root,
        packet_manifest_paths=args.packet_manifest,
        packet_manifest_globs=args.packet_manifest_glob,
        claims_path=args.claims_path,
        now_utc=args.now_utc,
        operator_approved_exact_cuda=args.operator_approved_exact_cuda or None,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        sys.stdout.write(json_text(payload) if args.format == "json" else render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
