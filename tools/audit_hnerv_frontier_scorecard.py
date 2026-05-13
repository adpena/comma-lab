#!/usr/bin/env python3
"""Audit the public HNeRV frontier scorecard for routing readiness.

The scorecard is a local planning artifact. This audit does not claim scores or
dispatch GPU work; it only fails closed when the scorecard is missing the exact
CUDA custody fields needed to route hidden-gem follow-ups.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tac.audit_contract import AuditReport, audit_exit_code
from tac.repo_io import json_text, read_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "public_hnerv_frontier_payload_profiles_20260504_codex"
    / "scorecard.json"
)
CONTEST_ORIGINAL_BYTES = 37_545_489
SCORE_TOLERANCE = 1e-6


def load_scorecard(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except FileNotFoundError as exc:
        raise ValueError(f"scorecard missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"scorecard is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("scorecard root must be a JSON object")
    return payload


def _row_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("label")): row for row in rows if isinstance(row, dict)}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _audit_score_row(row: dict[str, Any], repo_root: Path) -> list[str]:
    label = str(row.get("label") or "unknown")
    blockers: list[str] = []
    required_numbers = ("score", "archive_bytes", "avg_segnet_dist", "avg_posenet_dist")
    for key in required_numbers:
        if not isinstance(row.get(key), int | float):
            blockers.append(f"{label}_missing_{key}")
    for key in ("archive_sha256", "payload_sha256", "runtime_tree_sha256"):
        if not _valid_sha(row.get(key)):
            blockers.append(f"{label}_invalid_{key}")
    eval_artifact = row.get("eval_artifact")
    if not isinstance(eval_artifact, str) or not eval_artifact:
        blockers.append(f"{label}_missing_eval_artifact")
    elif not (repo_root / eval_artifact).is_file():
        blockers.append(f"{label}_missing_eval_artifact_file")
    if blockers:
        return blockers

    archive_bytes = int(row["archive_bytes"])
    seg = float(row["avg_segnet_dist"])
    pose = float(row["avg_posenet_dist"])
    recomputed = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES
    if abs(recomputed - float(row["score"])) > SCORE_TOLERANCE:
        blockers.append(f"{label}_score_formula_mismatch")
    expected_components = {
        "score_seg_contribution": 100.0 * seg,
        "score_pose_contribution": math.sqrt(10.0 * pose),
        "score_rate_contribution": 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES,
    }
    for key, expected in expected_components.items():
        if not isinstance(row.get(key), int | float):
            blockers.append(f"{label}_missing_{key}")
        elif abs(float(row[key]) - expected) > SCORE_TOLERANCE:
            blockers.append(f"{label}_{key}_mismatch")
    return blockers


_INTERNAL_SCORE_LOWERING_BLOCKER_MARKERS = (
    "score_claim_invalid",
    "regression_triggered",
    "lane_status_",
    "paper_claim_grade_",
    "evidence_grade_",
)


def _internal_score_lowering_blocked(row: dict[str, Any]) -> bool:
    blockers = row.get("canonicality_blockers") or []
    if not isinstance(blockers, list):
        return True
    return any(
        isinstance(blocker, str)
        and blocker.startswith(_INTERNAL_SCORE_LOWERING_BLOCKER_MARKERS)
        for blocker in blockers
    )


def _audit_internal_score_lowering_frontier(
    payload: dict[str, Any],
    labels: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    frontier = payload.get("score_lowering_frontier")
    target = payload.get("next_score_lowering_exact_evaluable_target")
    ranking = payload.get("score_lowering_hidden_gem_byte_mass_ranking")
    present = any(item is not None for item in (frontier, target, ranking))
    if not present:
        return blockers
    if not isinstance(frontier, dict):
        return ["score_lowering_frontier_not_object"]

    label = frontier.get("label")
    if not isinstance(label, str) or label not in labels:
        blockers.append("score_lowering_frontier_label_missing_from_rows")
        return blockers
    row = labels[label]
    if row.get("evidence_grade") != "A++":
        blockers.append(f"{label}_score_lowering_frontier_not_Aplusplus")
    if not isinstance(row.get("score"), int | float):
        blockers.append(f"{label}_score_lowering_frontier_missing_numeric_score")
    elif frontier.get("score") != row.get("score"):
        blockers.append(f"{label}_score_lowering_frontier_score_mismatch")
    if frontier.get("archive_sha256") != row.get("archive_sha256"):
        blockers.append(f"{label}_score_lowering_frontier_archive_sha_mismatch")
    if frontier.get("promotion_authority") is not False:
        blockers.append(f"{label}_score_lowering_frontier_promotion_authority_not_false")
    if frontier.get("frontier_scope") != "internal_exact_cuda_score_lowering":
        blockers.append(f"{label}_score_lowering_frontier_scope_invalid")
    if _internal_score_lowering_blocked(row):
        blockers.append(f"{label}_score_lowering_frontier_uses_severe_blocked_row")

    if ranking is not None and not isinstance(ranking, list):
        blockers.append("score_lowering_hidden_gem_byte_mass_ranking_not_list")
    if target is not None:
        if not isinstance(target, dict):
            blockers.append("next_score_lowering_exact_evaluable_target_not_object")
        elif target.get("frontier_label") != label:
            blockers.append("next_score_lowering_exact_evaluable_target_frontier_mismatch")
    return blockers


def audit_scorecard(payload: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    rows = payload.get("rows")
    groups = payload.get("payload_equivalence_groups")
    targets = payload.get("followup_targets")
    manifests = payload.get("payload_section_manifests")
    if payload.get("schema_version") != 1:
        blockers.append("schema_version_not_1")
    if payload.get("score_truth") != "exact_cuda_auth_eval_json":
        blockers.append("score_truth_not_exact_cuda_auth_eval_json")
    if not isinstance(rows, list) or not rows:
        blockers.append("missing_rows")
        rows = []
    if not isinstance(groups, list):
        blockers.append("missing_payload_equivalence_groups")
        groups = []
    if not isinstance(targets, list) or not targets:
        blockers.append("missing_followup_targets")
        targets = []
    if not isinstance(manifests, list) or not manifests:
        blockers.append("missing_payload_section_manifests")
        manifests = []

    labels = _row_by_label(rows)
    for row in rows:
        if isinstance(row, dict) and row.get("canonical_frontier_eligible") is True:
            blockers.extend(_audit_score_row(row, repo_root))
    pr106x = labels.get("PR106x")
    if pr106x is None:
        blockers.append("missing_PR106x_row")
    else:
        if pr106x.get("evidence_grade") != "A++":
            blockers.append("PR106x_not_Aplusplus")
        if pr106x.get("canonical_frontier_eligible") is not True:
            blockers.append("PR106x_not_canonical_frontier_eligible")
        if pr106x.get("profile_match_key") not in {"archive_sha256", "member_sha256"}:
            blockers.append("PR106x_missing_profile_match")
        if not isinstance(pr106x.get("archive_sha256"), str):
            blockers.append("PR106x_missing_archive_sha256")
        if not isinstance(pr106x.get("payload_sha256"), str):
            blockers.append("PR106x_missing_payload_sha256")

    has_pr106_control_group = False
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_labels = set(group.get("labels") or [])
        if {"PR106", "PR106x"}.issubset(group_labels) and {"PR106", "PR106x"}.issubset(labels):
            has_pr106_control_group = True
            if group.get("same_seg_contribution") is not True:
                blockers.append("PR106_control_group_seg_not_equal")
            if group.get("same_pose_contribution") is not True:
                blockers.append("PR106_control_group_pose_not_equal")
            if group.get("readiness") != "byte-identical payload pair; use as repack custody/control only":
                blockers.append("PR106_control_group_bad_readiness")
            break
    if not has_pr106_control_group:
        blockers.append("missing_PR106_PR106x_payload_control_group")

    actions = {str(target.get("suggested_action")) for target in targets if isinstance(target, dict)}
    if "decoder self-compression or weight-stream recoding fixture" not in actions:
        blockers.append("missing_decoder_self_compression_followup")
    if "latent/sidecar arithmetic-coding parity fixture" not in actions:
        blockers.append("missing_latent_sidecar_arithmetic_followup")

    manifest_labels: set[str] = set()
    manifest_roles: set[str] = set()
    for manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        label = manifest.get("label")
        if isinstance(label, str):
            manifest_labels.add(label)
        if manifest.get("score_claim") is not False:
            blockers.append(f"{label or 'unknown'}_section_manifest_score_claim_not_false")
        if manifest.get("dispatch_attempted") is not False:
            blockers.append(f"{label or 'unknown'}_section_manifest_dispatch_attempted_not_false")
        sections = manifest.get("sections")
        if not isinstance(sections, list) or not sections:
            blockers.append(f"{label or 'unknown'}_section_manifest_missing_sections")
            continue
        for section in sections:
            if not isinstance(section, dict):
                continue
            role = section.get("optimization_role")
            if isinstance(role, str):
                manifest_roles.add(role)
            if not isinstance(section.get("sha256"), str):
                blockers.append(f"{label or 'unknown'}_section_manifest_missing_section_sha256")
            if not isinstance(section.get("bytes"), int):
                blockers.append(f"{label or 'unknown'}_section_manifest_missing_section_bytes")
    if "PR106x" in labels and "PR106x" not in manifest_labels:
        blockers.append("missing_PR106x_payload_section_manifest")
    if "decoder_weight_stream" not in manifest_roles:
        blockers.append("missing_decoder_weight_stream_section_manifest")
    if "latent_stream" not in manifest_roles and "sidecar_or_correction_stream" not in manifest_roles:
        blockers.append("missing_latent_or_sidecar_section_manifest")

    blockers.extend(_audit_internal_score_lowering_frontier(payload, labels))

    score_lowering_frontier = payload.get("score_lowering_frontier")
    next_score_lowering_target = payload.get("next_score_lowering_exact_evaluable_target")
    summary = {
        "row_count": len(rows),
        "payload_equivalence_group_count": len(groups),
        "followup_target_count": len(targets),
        "payload_section_manifest_count": len(manifests),
        "score_lowering_frontier_label": (
            score_lowering_frontier.get("label") if isinstance(score_lowering_frontier, dict) else None
        ),
        "score_lowering_frontier_score": (
            score_lowering_frontier.get("score") if isinstance(score_lowering_frontier, dict) else None
        ),
        "next_score_lowering_target": (
            {
                "label": next_score_lowering_target.get("label"),
                "section": next_score_lowering_target.get("section"),
            }
            if isinstance(next_score_lowering_target, dict)
            else None
        ),
        "canonical_labels": [
            row.get("label")
            for row in rows
            if isinstance(row, dict) and row.get("canonical_frontier_eligible") is True
        ],
    }
    return blockers, summary


def build_report(scorecard: Path) -> AuditReport:
    try:
        payload = load_scorecard(scorecard)
        blockers, summary = audit_scorecard(payload)
    except ValueError as exc:
        blockers = [str(exc)]
        summary = {
            "canonical_labels": [],
            "followup_target_count": 0,
            "payload_equivalence_group_count": 0,
            "row_count": 0,
        }
    return AuditReport(
        audit="hnerv_frontier_scorecard",
        readiness_key="ready_for_hidden_gem_routing",
        ready=not blockers,
        blockers=tuple(blockers),
        summary=summary,
        metadata={"scorecard": str(scorecard)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=DEFAULT_SCORECARD,
        help="Scorecard JSON to audit.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.scorecard)
    if args.format == "json":
        print(json_text(report.to_dict()), end="")
    elif not report.ready:
        print(report.render_text())
    else:
        extra = ""
        if report.summary.get("score_lowering_frontier_label"):
            extra = (
                f", internal score-lowering={report.summary['score_lowering_frontier_label']} "
                f"({report.summary['score_lowering_frontier_score']})"
            )
        print(
            report.render_text(
                pass_detail=(
                    f"({report.summary['row_count']} rows, "
                    f"{report.summary['payload_equivalence_group_count']} payload groups, "
                    f"{report.summary['followup_target_count']} follow-up targets"
                    f"{extra})"
                )
            )
        )
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
