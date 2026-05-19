# SPDX-License-Identifier: MIT
"""Score-response matrix for PR101 pose-axis master-gradient candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.auth_eval_roundtrip_matrix import (
    AuthEvalRoundtripInput,
    build_auth_eval_roundtrip_matrix,
)
from tac.repo_io import repo_relative, sha256_file

SCHEMA = "pr101_pose_axis_score_response_matrix_v1"
OPERATOR_CANDIDATE_SCHEMA = "tac_pr101_pose_axis_decoder_recompression_candidate_v1"
SOURCE_OPERATOR_SCHEMA = "pose_byte_hoist_op7_manifest_v1"
PACKET_CANDIDATE_SCHEMA = "tac_monolithic_packet_candidate_v1"
RUNTIME_PROOF_SCHEMA = "tac_runtime_consumption_proof_v1"
CONTEST_TARGET_IDS = frozenset(
    {
        "modal_contest_cuda_t4_auto",
        "modal_contest_cpu_linux_x86_auto",
    }
)


class PR101ScoreResponseMatrixError(ValueError):
    """Raised when score-response matrix inputs are malformed."""


def build_pr101_pose_axis_score_response_matrix(
    *,
    source_archive: Path,
    source_submission_dir: Path,
    operator_candidate_manifest: Mapping[str, Any],
    packet_candidate_manifest: Mapping[str, Any],
    runtime_consumption_proof: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    repo_root: Path,
    label: str,
    lane_id: str,
    output_root: str,
    dispatch_claims_path: Path | None = None,
    include_diagnostics: bool = True,
    min_total_improvement: float = 0.001,
    min_scorer_term_improvement: float = 0.0005,
) -> dict[str, Any]:
    """Build paired auth-eval and score-response probe targets.

    This is a command-planning and authority-hardening artifact. It does not run
    the scorer, dispatch provider jobs, or claim score movement.
    """

    repo_root = Path(repo_root)
    source_archive = _resolve_existing_file(source_archive, repo_root, "source_archive")
    source_submission_dir = _resolve_existing_dir(
        source_submission_dir,
        repo_root,
        "source_submission_dir",
    )
    if not label:
        raise PR101ScoreResponseMatrixError("label is required")
    if not lane_id:
        raise PR101ScoreResponseMatrixError("lane_id is required")
    if not output_root:
        raise PR101ScoreResponseMatrixError("output_root is required")

    input_blockers = _input_authority_blockers(
        source_archive=source_archive,
        operator_candidate_manifest=operator_candidate_manifest,
        packet_candidate_manifest=packet_candidate_manifest,
        runtime_consumption_proof=runtime_consumption_proof,
        repo_root=repo_root,
    )
    candidate_archive_path = _candidate_archive_path(
        operator_candidate_manifest,
        packet_candidate_manifest,
        repo_root=repo_root,
    )
    archive_byte_delta = candidate_archive_path.stat().st_size - source_archive.stat().st_size
    max_ablation_archive_bytes_delta = abs(archive_byte_delta)

    baseline_matrix = build_auth_eval_roundtrip_matrix(
        candidate=AuthEvalRoundtripInput(
            archive=repo_relative(source_archive, repo_root),
            submission_dir=repo_relative(source_submission_dir, repo_root),
            label=f"{label}_baseline",
            output_root=output_root,
            lane_id=f"{lane_id}_baseline",
        ),
        runtime_manifest=dict(runtime_manifest),
        repo_root=repo_root,
        include_diagnostics=include_diagnostics,
    )
    candidate_matrix = build_auth_eval_roundtrip_matrix(
        candidate=AuthEvalRoundtripInput(
            archive=repo_relative(candidate_archive_path, repo_root),
            submission_dir=repo_relative(source_submission_dir, repo_root),
            label=f"{label}_candidate",
            output_root=output_root,
            lane_id=f"{lane_id}_candidate",
        ),
        runtime_manifest=dict(runtime_manifest),
        repo_root=repo_root,
        include_diagnostics=include_diagnostics,
    )
    target_pairs = _build_target_pairs(
        baseline_matrix=baseline_matrix,
        candidate_matrix=candidate_matrix,
        repo_root=repo_root,
        output_root=output_root,
        label=label,
        max_ablation_archive_bytes_delta=max_ablation_archive_bytes_delta,
        min_total_improvement=min_total_improvement,
        min_scorer_term_improvement=min_scorer_term_improvement,
    )
    contest_pairs = [pair for pair in target_pairs if pair["target_id"] in CONTEST_TARGET_IDS]
    missing_result_blockers = _missing_result_blockers(contest_pairs)
    missing_inflated_manifest_blockers = _missing_inflated_manifest_blockers(
        contest_pairs,
        repo_root=repo_root,
    )
    missing_result_review_blockers = _missing_result_review_blockers(contest_pairs)
    claim_lane_ids = _claimed_lane_ids(
        repo_root / ".omx/state/active_lane_dispatch_claims.md"
        if dispatch_claims_path is None
        else dispatch_claims_path
    )
    active_claim_blockers = [
        f"{pair['target_id']}_active_lane_claim_missing"
        for pair in contest_pairs
        if pair["dispatch_claim_required"] is True
        and (
            pair.get("baseline_lane_id") not in claim_lane_ids
            or pair.get("candidate_lane_id") not in claim_lane_ids
        )
    ]
    contest_lane_ids = {
        str(pair.get(key))
        for pair in contest_pairs
        for key in ("baseline_lane_id", "candidate_lane_id")
        if pair.get(key)
    }
    exact_eval_artifacts_present = bool(contest_pairs) and not missing_result_blockers
    dispatch_attempted = bool(contest_lane_ids & claim_lane_ids) or any(
        pair.get("baseline_result_json_exists") is True
        or pair.get("candidate_result_json_exists") is True
        for pair in contest_pairs
    )
    exact_eval_blockers = [
        f"{pair['target_id']}_exact_eval_missing"
        for pair in contest_pairs
        if pair["baseline_result_json_exists"] is not True
        or pair["candidate_result_json_exists"] is not True
    ]
    prerequisites_clear = not input_blockers
    ready_after_exact = prerequisites_clear and not active_claim_blockers
    ready_now = (
        ready_after_exact
        and not missing_result_blockers
        and not missing_inflated_manifest_blockers
        and not missing_result_review_blockers
    )

    return {
        "schema": SCHEMA,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": dispatch_attempted,
        "contest_exact_eval_artifacts_present": exact_eval_artifacts_present,
        "raw_archive_byte_coordinates_allowed": False,
        "candidate_specs_are_dispatchable": False,
        "ready_for_score_response_probe": ready_now,
        "ready_for_score_response_probe_after_exact_eval": prerequisites_clear,
        "ready_for_score_response_probe_after_exact_eval_and_lane_claim": ready_after_exact,
        "authority_blockers": input_blockers,
        "score_response_blockers": missing_result_blockers,
        "dispatch_blockers": [
            *active_claim_blockers,
            *exact_eval_blockers,
            *missing_inflated_manifest_blockers,
            *(
                ["paired_contest_cuda_cpu_exact_eval_missing"]
                if missing_result_blockers
                else []
            ),
            *missing_result_review_blockers,
        ],
        "source_archive": {
            "path": repo_relative(source_archive, repo_root),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "candidate_archive": {
            "path": repo_relative(candidate_archive_path, repo_root),
            "bytes": candidate_archive_path.stat().st_size,
            "sha256": sha256_file(candidate_archive_path),
            "archive_byte_delta": archive_byte_delta,
            "rate_delta_score_if_components_unchanged": (
                25.0 * archive_byte_delta / 37_545_489
            ),
        },
        "operator_candidate": _operator_summary(operator_candidate_manifest),
        "runtime_consumption_proof": _runtime_proof_summary(runtime_consumption_proof),
        "baseline_auth_eval_matrix": baseline_matrix,
        "candidate_auth_eval_matrix": candidate_matrix,
        "target_pairs": target_pairs,
        "contest_target_ids": sorted(CONTEST_TARGET_IDS),
        "notes": [
            "Matrix rows are commands and result locations, not score authority.",
            "Score-response probes become actionable only after matching baseline and candidate exact-eval JSONs exist.",
            "Paired contest-CUDA and contest-CPU result review is mandatory before promotion language.",
        ],
    }


def render_pr101_pose_axis_score_response_matrix_markdown(matrix: Mapping[str, Any]) -> str:
    """Render a compact operator-facing markdown view of the matrix."""

    candidate = matrix.get("candidate_archive") if isinstance(matrix.get("candidate_archive"), Mapping) else {}
    operator = matrix.get("operator_candidate") if isinstance(matrix.get("operator_candidate"), Mapping) else {}
    lines = [
        "# PR101 Pose-Axis Score-Response Matrix",
        "",
        "Authority:",
        "- score_claim: false",
        "- promotion_eligible: false",
        f"- ready_for_exact_eval_dispatch: {matrix.get('ready_for_exact_eval_dispatch')}",
        f"- dispatch_attempted: {matrix.get('dispatch_attempted')}",
        "- contest_exact_eval_artifacts_present: "
        f"{matrix.get('contest_exact_eval_artifacts_present')}",
        "",
        f"Candidate: `{candidate.get('path')}`",
        f"Archive bytes: `{candidate.get('bytes')}`",
        f"Archive SHA-256: `{candidate.get('sha256')}`",
        f"Mutation mode: `{operator.get('mutation_mode')}`",
        f"Component-moving candidate: `{operator.get('component_moving_candidate')}`",
        "",
        "## Readiness",
        "",
        f"- ready_for_score_response_probe: `{matrix.get('ready_for_score_response_probe')}`",
        "- ready_for_score_response_probe_after_exact_eval: "
        f"`{matrix.get('ready_for_score_response_probe_after_exact_eval')}`",
        "- ready_for_score_response_probe_after_exact_eval_and_lane_claim: "
        f"`{matrix.get('ready_for_score_response_probe_after_exact_eval_and_lane_claim')}`",
        "",
        "## Target Pairs",
        "",
        "| target | axis | contest | baseline json | candidate json | probe json |",
        "|---|---|---:|---|---|---|",
    ]
    for pair in matrix.get("target_pairs", []):
        if not isinstance(pair, Mapping):
            continue
        lines.append(
            "| "
            f"`{pair.get('target_id')}` | "
            f"`{pair.get('score_axis')}` | "
            f"{pair.get('contest_compliant')} | "
            f"`{pair.get('baseline_result_json')}` | "
            f"`{pair.get('candidate_result_json')}` | "
            f"`{pair.get('score_response_output_json')}` |"
        )
    lines.extend(
        [
            "",
            "## Result Presence",
            "",
            "| target | baseline exact | candidate exact | score-response probe |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in matrix.get("target_pairs", []):
        if not isinstance(pair, Mapping):
            continue
        lines.append(
            "| "
            f"`{pair.get('target_id')}` | "
            f"{pair.get('baseline_result_json_exists')} | "
            f"{pair.get('candidate_result_json_exists')} | "
            f"{pair.get('score_response_output_json_exists')} |"
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(matrix.get("authority_blockers") or []) + list(
        matrix.get("score_response_blockers") or []
    ) + list(matrix.get("dispatch_blockers") or [])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _input_authority_blockers(
    *,
    source_archive: Path,
    operator_candidate_manifest: Mapping[str, Any],
    packet_candidate_manifest: Mapping[str, Any],
    runtime_consumption_proof: Mapping[str, Any],
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    if operator_candidate_manifest.get("schema") != OPERATOR_CANDIDATE_SCHEMA:
        blockers.append("operator_candidate_manifest_schema_mismatch")
    if packet_candidate_manifest.get("schema") != PACKET_CANDIDATE_SCHEMA:
        blockers.append("packet_candidate_manifest_schema_mismatch")
    if runtime_consumption_proof.get("schema") != RUNTIME_PROOF_SCHEMA:
        blockers.append("runtime_consumption_proof_schema_mismatch")
    if operator_candidate_manifest.get("score_claim") is not False:
        blockers.append("operator_candidate_score_claim_not_false")
    if operator_candidate_manifest.get("promotion_eligible") is True:
        blockers.append("operator_candidate_promotion_eligible_true")
    if operator_candidate_manifest.get("rank_or_kill_eligible") is True:
        blockers.append("operator_candidate_rank_or_kill_eligible_true")
    if operator_candidate_manifest.get("raw_archive_byte_coordinates_allowed") is True:
        blockers.append("operator_candidate_raw_archive_byte_coordinates_allowed_true")
    if operator_candidate_manifest.get("candidate_specs_are_dispatchable") is True:
        blockers.append("operator_candidate_specs_are_dispatchable_true")
    if packet_candidate_manifest.get("score_claim") is not False:
        blockers.append("packet_candidate_score_claim_not_false")
    if packet_candidate_manifest.get("promotion_eligible") is True:
        blockers.append("packet_candidate_promotion_eligible_true")
    if packet_candidate_manifest.get("rank_or_kill_eligible") is True:
        blockers.append("packet_candidate_rank_or_kill_eligible_true")
    if runtime_consumption_proof.get("score_claim") is not False:
        blockers.append("runtime_consumption_proof_score_claim_not_false")
    if runtime_consumption_proof.get("promotion_eligible") is True:
        blockers.append("runtime_consumption_proof_promotion_eligible_true")
    if runtime_consumption_proof.get("rank_or_kill_eligible") is True:
        blockers.append("runtime_consumption_proof_rank_or_kill_eligible_true")
    if operator_candidate_manifest.get("component_moving_candidate") is not True:
        blockers.append("operator_candidate_not_component_moving")
    if runtime_consumption_proof.get("ready_for_exact_eval_runtime") is not True:
        blockers.append("runtime_consumption_proof_not_ready")
    if runtime_consumption_proof.get("blockers"):
        blockers.append("runtime_consumption_proof_blockers_present")
    blockers.extend(
        _source_operator_manifest_blockers(
            source_archive=source_archive,
            operator_candidate_manifest=operator_candidate_manifest,
            repo_root=repo_root,
        )
    )

    source = operator_candidate_manifest.get("source_archive")
    if isinstance(source, Mapping):
        expected_source_sha = source.get("sha256")
        expected_source_bytes = source.get("bytes")
        if isinstance(expected_source_sha, str) and sha256_file(source_archive) != expected_source_sha:
            blockers.append("source_archive_sha256_mismatch")
        if isinstance(expected_source_bytes, int) and source_archive.stat().st_size != expected_source_bytes:
            blockers.append("source_archive_bytes_mismatch")

    candidate_archive = _candidate_archive_path(
        operator_candidate_manifest,
        packet_candidate_manifest,
        repo_root=repo_root,
    )
    candidate_sha = sha256_file(candidate_archive)
    operator_archive = operator_candidate_manifest.get("candidate_archive")
    packet_archive = packet_candidate_manifest.get("candidate_archive")
    runtime_archive_sha = runtime_consumption_proof.get("candidate_archive_sha256")
    for label, payload in (
        ("operator_candidate", operator_archive),
        ("packet_candidate", packet_archive),
    ):
        if not isinstance(payload, Mapping):
            blockers.append(f"{label}_archive_missing")
            continue
        if payload.get("sha256") != candidate_sha:
            blockers.append(f"{label}_archive_sha256_mismatch")
        if payload.get("bytes") != candidate_archive.stat().st_size:
            blockers.append(f"{label}_archive_bytes_mismatch")
    if runtime_archive_sha != candidate_sha:
        blockers.append("runtime_proof_candidate_archive_sha256_mismatch")

    return _dedupe(blockers)


def _source_operator_manifest_blockers(
    *,
    source_archive: Path,
    operator_candidate_manifest: Mapping[str, Any],
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    source_operator = operator_candidate_manifest.get("source_operator_manifest")
    if not isinstance(source_operator, Mapping):
        return ["source_operator_manifest_missing"]

    path_value = source_operator.get("path")
    if not isinstance(path_value, str) or not path_value:
        return ["source_operator_manifest_path_missing"]
    manifest_path = Path(path_value)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        return ["source_operator_manifest_file_missing"]

    expected_sha = source_operator.get("sha256")
    actual_sha = sha256_file(manifest_path)
    if expected_sha != actual_sha:
        blockers.append("source_operator_manifest_sha256_mismatch")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append("source_operator_manifest_unreadable")
        return blockers
    if not isinstance(payload, Mapping):
        blockers.append("source_operator_manifest_not_object")
        return blockers

    if payload.get("schema") != SOURCE_OPERATOR_SCHEMA:
        blockers.append("source_operator_manifest_schema_mismatch")
    source_blockers = payload.get("blockers")
    if isinstance(source_blockers, list) and any(
        blocker == "anchor_score_axis_dominance_not_persisted"
        for blocker in source_blockers
    ):
        blockers.append("anchor_score_axis_dominance_not_persisted")

    source_anchor = payload.get("source_anchor")
    if not isinstance(source_anchor, Mapping):
        blockers.append("source_anchor_missing")
        return blockers
    if source_anchor.get("score_axis_dominance_available") is not True:
        blockers.append("source_anchor_score_axis_dominance_available_not_true")

    scored_sha = source_anchor.get("scored_archive_sha256")
    scored_bytes = source_anchor.get("scored_archive_bytes")
    if not isinstance(scored_sha, str) or not scored_sha:
        blockers.append("source_anchor_scored_archive_sha256_missing")
    elif scored_sha != sha256_file(source_archive):
        blockers.append("source_anchor_scored_archive_sha256_mismatch")
    if not isinstance(scored_bytes, int) or isinstance(scored_bytes, bool):
        blockers.append("source_anchor_scored_archive_bytes_missing")
    elif scored_bytes != source_archive.stat().st_size:
        blockers.append("source_anchor_scored_archive_bytes_mismatch")

    return blockers


def _candidate_archive_path(
    operator_candidate_manifest: Mapping[str, Any],
    packet_candidate_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    archive = operator_candidate_manifest.get("candidate_archive")
    packet_archive = packet_candidate_manifest.get("candidate_archive")
    path_value = None
    if isinstance(archive, Mapping):
        path_value = archive.get("path")
    if path_value is None and isinstance(packet_archive, Mapping):
        path_value = packet_archive.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PR101ScoreResponseMatrixError("candidate archive path missing")
    return _resolve_existing_file(Path(path_value), repo_root, "candidate_archive")


def _operator_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "candidate_id": manifest.get("candidate_id"),
        "mutation_mode": manifest.get("mutation_mode"),
        "mutation_operator": manifest.get("mutation_operator"),
        "component_moving_candidate": manifest.get("component_moving_candidate"),
        "semantic_equivalence_expected": manifest.get("semantic_equivalence_expected"),
        "ready_for_score_response_probe": manifest.get("ready_for_score_response_probe"),
        "selected_pose_axis_candidate": manifest.get("selected_pose_axis_candidate"),
        "replacement_stream": manifest.get("replacement_stream"),
        "source_operator_manifest": manifest.get("source_operator_manifest"),
    }


def _runtime_proof_summary(proof: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": proof.get("schema"),
        "candidate_id": proof.get("candidate_id"),
        "candidate_archive_sha256": proof.get("candidate_archive_sha256"),
        "runtime_grammar": proof.get("runtime_grammar"),
        "ready_for_exact_eval_runtime": proof.get("ready_for_exact_eval_runtime"),
        "ready_for_exact_eval_dispatch": proof.get("ready_for_exact_eval_dispatch"),
        "score_claim": proof.get("score_claim"),
        "blockers": list(proof.get("blockers") or []),
    }


def _build_target_pairs(
    *,
    baseline_matrix: Mapping[str, Any],
    candidate_matrix: Mapping[str, Any],
    repo_root: Path,
    output_root: str,
    label: str,
    max_ablation_archive_bytes_delta: int,
    min_total_improvement: float,
    min_scorer_term_improvement: float,
) -> list[dict[str, Any]]:
    baseline_targets = _targets_by_id(baseline_matrix)
    candidate_targets = _targets_by_id(candidate_matrix)
    pairs: list[dict[str, Any]] = []
    for target_id in sorted(set(baseline_targets) & set(candidate_targets)):
        baseline = baseline_targets[target_id]
        candidate = candidate_targets[target_id]
        baseline_json = _planned_result_json(baseline)
        candidate_json = _planned_result_json(candidate)
        output_json = (
            Path(output_root)
            / label
            / "score_response"
            / f"{target_id}.score_response.json"
        ).as_posix()
        output_md = (
            Path(output_root)
            / label
            / "score_response"
            / f"{target_id}.score_response.md"
        ).as_posix()
        command = [
            ".venv/bin/python",
            "tools/probe_substrate_score_response.py",
            "--baseline-json",
            baseline_json,
            "--candidate-json",
            candidate_json,
            "--output-json",
            output_json,
            "--output-md",
            output_md,
            "--title",
            f"PR101 Pose-Axis OP-7 {target_id} Score Response",
            "--mode",
            "ablation",
            "--min-total-improvement",
            str(min_total_improvement),
            "--min-scorer-term-improvement",
            str(min_scorer_term_improvement),
            "--max-ablation-archive-bytes-delta",
            str(max_ablation_archive_bytes_delta),
        ]
        score_axis = str(candidate.get("score_axis") or "")
        if score_axis in {"contest_cuda", "contest_cpu"}:
            command.extend(["--axis", score_axis])
        pairs.append(
            {
                "target_id": target_id,
                "score_axis": score_axis,
                "evidence_grade": candidate.get("evidence_grade"),
                "contest_compliant": candidate.get("contest_compliant") is True,
                "dispatch_claim_required": candidate.get("dispatch_claim_required") is True,
                "baseline_command": baseline.get("command"),
                "candidate_command": candidate.get("command"),
                "baseline_lane_id": _command_flag_value(baseline.get("command"), "--lane-id"),
                "candidate_lane_id": _command_flag_value(candidate.get("command"), "--lane-id"),
                "baseline_result_json": baseline_json,
                "candidate_result_json": candidate_json,
                "baseline_result_json_exists": (repo_root / baseline_json).is_file(),
                "candidate_result_json_exists": (repo_root / candidate_json).is_file(),
                "score_response_probe_command": command,
                "score_response_output_json": output_json,
                "score_response_output_md": output_md,
                "score_response_output_json_exists": (repo_root / output_json).is_file(),
            }
        )
    return pairs


def _targets_by_id(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    targets = matrix.get("targets")
    if not isinstance(targets, list):
        raise PR101ScoreResponseMatrixError("auth eval matrix targets missing")
    out: dict[str, Mapping[str, Any]] = {}
    for target in targets:
        if isinstance(target, Mapping) and isinstance(target.get("target_id"), str):
            out[str(target["target_id"])] = target
    return out


def _planned_result_json(target: Mapping[str, Any]) -> str:
    command = target.get("command")
    if not isinstance(command, list):
        raise PR101ScoreResponseMatrixError("target command missing")
    output_dir = _command_flag_value(command, "--output-dir")
    if output_dir:
        return (Path(output_dir) / "contest_auth_eval.json").as_posix()
    work_dir = _command_flag_value(command, "--work-dir")
    if work_dir and work_dir.endswith("/work"):
        return (Path(work_dir[: -len("/work")]) / "contest_auth_eval.json").as_posix()
    raise PR101ScoreResponseMatrixError("target result directory missing")


def _command_flag_value(command: list[Any], flag: str) -> str:
    if not isinstance(command, list):
        return ""
    parts = [str(part) for part in command]
    for index, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[index + 1]
    return ""


def _missing_result_blockers(contest_pairs: list[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for pair in contest_pairs:
        target_id = str(pair.get("target_id") or "unknown")
        if pair.get("baseline_result_json_exists") is not True:
            blockers.append(f"{target_id}_baseline_exact_eval_json_missing")
        if pair.get("candidate_result_json_exists") is not True:
            blockers.append(f"{target_id}_candidate_exact_eval_json_missing")
    return blockers


def _missing_inflated_manifest_blockers(
    contest_pairs: list[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    for pair in contest_pairs:
        target_id = str(pair.get("target_id") or "unknown")
        for side in ("baseline", "candidate"):
            result_json = pair.get(f"{side}_result_json")
            if not isinstance(result_json, str) or not result_json:
                continue
            result_path = Path(result_json)
            if not result_path.is_absolute():
                result_path = repo_root / result_path
            if not result_path.is_file():
                continue
            if not (result_path.parent / "inflated_outputs_manifest.json").is_file():
                blockers.append(f"{target_id}_{side}_inflated_outputs_manifest_missing")
    return blockers


def _missing_result_review_blockers(contest_pairs: list[Mapping[str, Any]]) -> list[str]:
    missing = [
        pair
        for pair in contest_pairs
        if pair.get("score_response_output_json_exists") is not True
    ]
    return ["paired_contest_cuda_and_cpu_result_review_missing"] if missing else []


def _claimed_lane_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    lane_ids: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) >= 3 and cells[2]:
            lane_ids.add(cells[2])
    return lane_ids


def _resolve_existing_file(path: Path, repo_root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    if not candidate.is_file():
        raise PR101ScoreResponseMatrixError(f"{label} not found: {path}")
    return candidate


def _resolve_existing_dir(path: Path, repo_root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    if not candidate.is_dir():
        raise PR101ScoreResponseMatrixError(f"{label} not found: {path}")
    return candidate


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "SCHEMA",
    "PR101ScoreResponseMatrixError",
    "build_pr101_pose_axis_score_response_matrix",
    "render_pr101_pose_axis_score_response_matrix_markdown",
]
