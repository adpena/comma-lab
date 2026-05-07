"""Promotion review for exact HNeRV lossless-control rows.

This module is intentionally narrow: it reviews an already evaluated
byte-different, lossless HNeRV Brotli control against local custody artifacts.
It does not build archives, dispatch GPU work, or create a new score claim.
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import repo_relative, sha256_bytes, sha256_file

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_lossless_control_promotion_review"
CONTEST_ORIGINAL_BYTES = 37_545_489
SCORE_TOLERANCE = 1e-9

PROMOTABLE_VERDICT = "promotable_existing_exact_control"
BLOCKED_VERDICT = "blocked_missing_or_inconsistent_evidence"


class HnervLosslessControlPromotionReviewError(ValueError):
    """Raised when promotion-review inputs are malformed."""


def inspect_single_member_archive(path: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Inspect a strict single-member ZIP archive for custody review."""

    archive = Path(path)
    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if len(infos) == 1:
            member = infos[0]
            payload = zf.read(member.filename)
            member_name = member.filename
            member_bytes = len(payload)
            member_sha256 = sha256_bytes(payload)
            crc32_hex = f"{member.CRC:08x}"
            compression_type = member.compress_type
        else:
            member_name = None
            member_bytes = None
            member_sha256 = None
            crc32_hex = None
            compression_type = None
        bad_member = zf.testzip()

    blockers: list[str] = []
    if len(infos) != 1:
        blockers.append(f"expected_exactly_one_zip_member_got_{len(infos)}")
    if duplicate_names:
        blockers.append("duplicate_zip_member_names")
    if bad_member is not None:
        blockers.append(f"zip_crc_failed:{bad_member}")
    if isinstance(member_name, str):
        if member_name.startswith("/") or ".." in Path(member_name).parts:
            blockers.append("zip_slip_member_name")
        if member_name.startswith(".") or "/." in member_name or member_name.startswith("__MACOSX/"):
            blockers.append("hidden_or_resource_fork_member")

    archive_path = repo_relative(archive, repo_root) if repo_root else archive.as_posix()
    return {
        "path": archive_path,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "member_name": member_name,
        "member_bytes": member_bytes,
        "member_sha256": member_sha256,
        "member_crc32_hex": crc32_hex,
        "member_compress_type": compression_type,
        "single_member_zip": len(infos) == 1,
        "duplicate_member_names": duplicate_names,
        "blockers": blockers,
    }


def build_lossless_control_promotion_review(
    *,
    target_label: str,
    scorecard: Mapping[str, Any],
    entropy_ranking: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    exact_eval: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    public_preflight: Mapping[str, Any] | None = None,
    candidate_archive: Mapping[str, Any] | None = None,
    exact_eval_archive: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Review whether an existing exact lossless-control row is promotable."""

    scorecard_rows = _scorecard_rows(scorecard)
    target_row = _row_by_label(scorecard_rows, target_label)
    source_label = _first_str(candidate_manifest.get("source_label"), target_row.get("candidate_source_label"))
    source_row = _row_by_label(scorecard_rows, source_label) if source_label else None
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    exact_provenance = _as_mapping(exact_eval.get("provenance"))
    runtime_manifest = _as_mapping(exact_provenance.get("inflate_runtime_manifest"))
    candidate_audit = _as_mapping(candidate_manifest.get("candidate_diff_audit"))
    candidate_preflight = _as_mapping(public_preflight) if public_preflight is not None else {}
    preflight_archive = _as_mapping(candidate_preflight.get("archive"))
    preflight_runtime = _as_mapping(candidate_preflight.get("runtime"))
    adjudication_gates = [
        gate
        for gate in adjudication.get("component_gates", [])
        if isinstance(gate, Mapping)
    ]

    candidate_sha = _first_str(candidate_manifest.get("candidate_archive_sha256"))
    candidate_bytes = _first_int(candidate_manifest.get("candidate_archive_bytes"))
    exact_sha = _first_str(exact_provenance.get("archive_sha256"), adjudication.get("contest_cuda_archive_sha256"))
    exact_bytes = _first_int(exact_eval.get("archive_size_bytes"), adjudication.get("contest_cuda_archive_bytes"))
    row_sha = _first_str(target_row.get("archive_sha256"))
    row_bytes = _first_int(target_row.get("archive_bytes"))
    baseline_bytes = _first_int(candidate_manifest.get("source_archive_bytes"), adjudication.get("baseline_archive_bytes"))
    byte_delta = _first_int(candidate_audit.get("total_byte_delta"))

    ranking_action = _as_mapping(entropy_ranking.get("next_rate_only_action"))
    ranking_controls = [
        row
        for row in entropy_ranking.get("exact_lossless_control_actions", [])
        if isinstance(row, Mapping) and row.get("target_label") == target_label
    ]
    ranking_control = ranking_controls[0] if ranking_controls else {}

    _check(
        checks,
        "entropy_ranker_selected_existing_exact_control_review",
        ranking_action.get("action_id") == "review_current_exact_lossless_brotli_control_before_promotion"
        and ranking_action.get("target_label") == target_label
        and ranking_action.get("score_claim") is False
        and ranking_action.get("dispatch_attempted") is False,
        requirement="ranker must route to review, not dispatch or score claim",
        evidence={
            "action_id": ranking_action.get("action_id"),
            "target_label": ranking_action.get("target_label"),
        },
    )
    _check(
        checks,
        "ranker_exact_control_ready_for_review",
        ranking_control.get("review_status") == "ready_for_promotion_review_existing_exact_custody"
        and ranking_control.get("raw_equivalence_closed") is True
        and not list(ranking_control.get("blockers") or []),
        requirement="ranker exact-control row must be ready for promotion review",
        evidence={
            "review_status": ranking_control.get("review_status"),
            "raw_equivalence_closed": ranking_control.get("raw_equivalence_closed"),
            "blockers": list(ranking_control.get("blockers") or []),
        },
    )
    _check(
        checks,
        "scorecard_target_row_exact_cuda_frontier_eligible",
        target_row.get("canonical_frontier_eligible") is True
        and target_row.get("evidence_grade") == "A++"
        and target_row.get("frontier_scope") == "exact_local_cuda_custody_lossless_repack_control"
        and not list(target_row.get("canonicality_blockers") or []),
        requirement="scorecard row must be A++ exact CUDA lossless-control custody",
        evidence={
            "label": target_row.get("label"),
            "evidence_grade": target_row.get("evidence_grade"),
            "frontier_scope": target_row.get("frontier_scope"),
            "canonicality_blockers": list(target_row.get("canonicality_blockers") or []),
        },
    )
    _check(
        checks,
        "candidate_and_exact_eval_archive_identity_match",
        _same_present(candidate_sha, exact_sha, row_sha)
        and _same_present(candidate_bytes, exact_bytes, row_bytes),
        requirement="candidate, scorecard, and exact eval must identify the same archive bytes",
        evidence={
            "candidate_archive_sha256": candidate_sha,
            "exact_eval_archive_sha256": exact_sha,
            "scorecard_archive_sha256": row_sha,
            "candidate_archive_bytes": candidate_bytes,
            "exact_eval_archive_bytes": exact_bytes,
            "scorecard_archive_bytes": row_bytes,
        },
    )
    _check(
        checks,
        "candidate_is_rate_positive_lossless_repack",
        isinstance(byte_delta, int)
        and byte_delta < 0
        and isinstance(baseline_bytes, int)
        and isinstance(candidate_bytes, int)
        and candidate_bytes < baseline_bytes
        and not list(candidate_audit.get("blockers") or []),
        requirement="candidate diff audit must be blocker-free and byte-positive",
        evidence={
            "source_archive_bytes": baseline_bytes,
            "candidate_archive_bytes": candidate_bytes,
            "total_byte_delta": byte_delta,
            "audit_blockers": list(candidate_audit.get("blockers") or []),
        },
    )
    raw_equivalence_rows = [
        row
        for row in candidate_manifest.get("brotli_raw_equivalence", [])
        if isinstance(row, Mapping)
    ]
    _check(
        checks,
        "brotli_raw_equivalence_closed",
        bool(raw_equivalence_rows) and all(row.get("raw_equal") is True for row in raw_equivalence_rows),
        requirement="all Brotli section recodes must decompress to the same raw bytes",
        evidence={
            "section_count": len(raw_equivalence_rows),
            "sections": [
                {
                    "section_name": row.get("section_name"),
                    "raw_equal": row.get("raw_equal"),
                    "raw_bytes": row.get("raw_bytes"),
                }
                for row in raw_equivalence_rows
            ],
        },
    )
    _check(
        checks,
        "public_replay_preflight_passed",
        public_preflight is not None
        and preflight_archive.get("status") == "passed"
        and not list(candidate_preflight.get("blockers") or [])
        and not list(preflight_archive.get("duplicate_member_names") or []),
        requirement="public replay preflight must pass on the candidate archive",
        evidence={
            "archive_status": preflight_archive.get("status"),
            "blockers": list(candidate_preflight.get("blockers") or []),
            "promotion_eligible_before_cuda": candidate_preflight.get("promotion_eligible"),
        },
    )
    _check(
        checks,
        "exact_eval_is_cuda_t4_full_sample",
        exact_provenance.get("device") == "cuda"
        and exact_provenance.get("cuda_available") is True
        and exact_provenance.get("gpu_t4_match") is True
        and exact_eval.get("n_samples") == 600,
        requirement="exact eval must be CUDA on contest-equivalent T4 with 600 samples",
        evidence={
            "device": exact_provenance.get("device"),
            "cuda_available": exact_provenance.get("cuda_available"),
            "gpu_model": exact_provenance.get("gpu_model"),
            "gpu_t4_match": exact_provenance.get("gpu_t4_match"),
            "n_samples": exact_eval.get("n_samples"),
        },
    )
    recomputed = _first_float(exact_eval.get("score_recomputed_from_components"))
    canonical = _first_float(exact_eval.get("canonical_score"), recomputed)
    structured_component_score = _structured_component_score(exact_eval)
    _check(
        checks,
        "exact_eval_score_recomputed_from_components",
        isinstance(recomputed, float)
        and isinstance(canonical, float)
        and isinstance(structured_component_score, float)
        and abs(recomputed - canonical) <= SCORE_TOLERANCE
        and abs(recomputed - structured_component_score) <= SCORE_TOLERANCE,
        requirement="review must use structured component recomputation, not rounded logs",
        evidence={
            "score_recomputed_from_components": recomputed,
            "canonical_score": canonical,
            "structured_component_score": structured_component_score,
        },
    )
    _check(
        checks,
        "exact_eval_runtime_tree_recorded",
        _valid_sha(runtime_manifest.get("runtime_tree_sha256"))
        and _valid_sha(target_row.get("runtime_tree_sha256"))
        and runtime_manifest.get("runtime_tree_sha256") == target_row.get("runtime_tree_sha256"),
        requirement="exact eval runtime tree hash must be recorded and match the scorecard",
        evidence={
            "exact_eval_runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "scorecard_runtime_tree_sha256": target_row.get("runtime_tree_sha256"),
        },
    )
    _check(
        checks,
        "adjudication_marks_existing_exact_control_promotable",
        adjudication.get("promotion_eligible") is True
        and adjudication.get("scientific_score_eligible") is True
        and adjudication.get("contest_equivalent_hardware") is True
        and not list(adjudication.get("component_gate_violations") or [])
        and adjudication.get("component_gate_triggered") is False,
        requirement="adjudication must explicitly permit promotion review use",
        evidence={
            "promotion_eligible": adjudication.get("promotion_eligible"),
            "scientific_score_eligible": adjudication.get("scientific_score_eligible"),
            "contest_equivalent_hardware": adjudication.get("contest_equivalent_hardware"),
            "component_gate_violations": list(adjudication.get("component_gate_violations") or []),
            "allowed_use": list(adjudication.get("allowed_use") or []),
        },
    )
    _check(
        checks,
        "adjudication_component_gates_passed",
        bool(adjudication_gates) and all(gate.get("passed") is True for gate in adjudication_gates),
        requirement="PoseNet and SegNet gates must pass against the source control",
        evidence={
            "component_gates": [
                {
                    "component": gate.get("component"),
                    "passed": gate.get("passed"),
                    "observed": gate.get("observed"),
                    "reference": gate.get("reference"),
                    "relative_to_reference": gate.get("relative_to_reference"),
                }
                for gate in adjudication_gates
            ],
        },
    )
    score_delta = _first_float(adjudication.get("score_delta_vs_baseline"))
    _check(
        checks,
        "adjudication_score_delta_is_rate_only_improvement",
        isinstance(score_delta, float)
        and score_delta < 0
        and adjudication.get("regression_triggered") is False,
        requirement="adjudication must show non-regressing score delta versus source control",
        evidence={
            "score_delta_vs_baseline": score_delta,
            "regression_triggered": adjudication.get("regression_triggered"),
            "archive_delta_bytes": adjudication.get("archive_delta_bytes"),
        },
    )
    if source_row is not None:
        _check(
            checks,
            "scorecard_source_and_candidate_components_match",
            _float_equal(target_row.get("avg_posenet_dist"), source_row.get("avg_posenet_dist"))
            and _float_equal(target_row.get("avg_segnet_dist"), source_row.get("avg_segnet_dist")),
            requirement="lossless control should preserve PoseNet and SegNet components in scorecard rows",
            evidence={
                "source_label": source_label,
                "source_posenet": source_row.get("avg_posenet_dist"),
                "candidate_posenet": target_row.get("avg_posenet_dist"),
                "source_segnet": source_row.get("avg_segnet_dist"),
                "candidate_segnet": target_row.get("avg_segnet_dist"),
            },
        )
    else:
        _check(
            checks,
            "scorecard_source_and_candidate_components_match",
            False,
            requirement="source scorecard row must exist for source/candidate component parity review",
            evidence={"source_label": source_label},
        )

    _archive_identity_check(
        checks,
        "candidate_archive_file_matches_manifest",
        candidate_archive,
        expected_sha=candidate_sha,
        expected_bytes=candidate_bytes,
        expected_member_sha=_first_str(candidate_manifest.get("candidate_payload_sha256"), target_row.get("payload_sha256")),
    )
    _archive_identity_check(
        checks,
        "exact_eval_archive_file_matches_eval_json",
        exact_eval_archive,
        expected_sha=exact_sha,
        expected_bytes=exact_bytes,
        expected_member_sha=_first_str(target_row.get("payload_sha256"), candidate_manifest.get("candidate_payload_sha256")),
    )

    current_frontier = scorecard.get("current_frontier")
    if not isinstance(current_frontier, Mapping) or current_frontier.get("label") != target_label:
        warnings.append(
            {
                "id": "scorecard_current_frontier_field_missing_or_stale",
                "severity": "warning",
                "message": "scorecard rows and entropy ranker select target, but current_frontier is absent or not target",
                "evidence": {
                    "current_frontier": current_frontier,
                    "target_label": target_label,
                },
            }
        )
    eval_pact_commit = exact_provenance.get("pact_commit")
    if isinstance(eval_pact_commit, str) and eval_pact_commit.startswith("<error:"):
        warnings.append(
            {
                "id": "exact_eval_pact_commit_unavailable",
                "severity": "warning",
                "message": "exact eval provenance could not record pact_commit; runtime tree and file hashes remain recorded",
                "evidence": {"pact_commit": eval_pact_commit},
            }
        )
    preflight_runtime_sha = _first_str(
        preflight_runtime.get("runtime_tree_sha256"),
        _as_mapping(preflight_runtime.get("runtime_manifest")).get("runtime_tree_sha256"),
    )
    exact_runtime_sha = _first_str(runtime_manifest.get("runtime_tree_sha256"))
    if preflight_runtime_sha and exact_runtime_sha and preflight_runtime_sha != exact_runtime_sha:
        warnings.append(
            {
                "id": "public_preflight_runtime_tree_differs_from_exact_eval",
                "severity": "warning",
                "message": "local public preflight runtime hash differs from exact eval runtime hash; compare future runs by runtime tree",
                "evidence": {
                    "public_preflight_runtime_tree_sha256": preflight_runtime_sha,
                    "exact_eval_runtime_tree_sha256": exact_runtime_sha,
                },
            }
        )

    blockers = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
    promotable = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "target_label": target_label,
        "review_verdict": PROMOTABLE_VERDICT if promotable else BLOCKED_VERDICT,
        "existing_exact_control_promotable": promotable,
        "promotion_review": True,
        "score_claim": False,
        "new_score_claim": False,
        "dispatch_attempted": False,
        "gpu_dispatch_performed": False,
        "lane_claim_required_for_this_review": False,
        "ready_for_new_exact_eval_dispatch": False,
        "review_basis": "existing exact CUDA artifact only",
        "archive": {
            "candidate_archive_sha256": candidate_sha,
            "candidate_archive_bytes": candidate_bytes,
            "source_archive_sha256": candidate_manifest.get("source_archive_sha256"),
            "source_archive_bytes": baseline_bytes,
            "byte_delta": byte_delta,
            "payload_sha256": _first_str(target_row.get("payload_sha256"), candidate_manifest.get("candidate_payload_sha256")),
        },
        "exact_eval": {
            "artifact": target_row.get("eval_artifact"),
            "archive_sha256": exact_sha,
            "archive_bytes": exact_bytes,
            "score_recomputed_from_components": recomputed,
            "avg_posenet_dist": exact_eval.get("avg_posenet_dist"),
            "avg_segnet_dist": exact_eval.get("avg_segnet_dist"),
            "n_samples": exact_eval.get("n_samples"),
            "device": exact_provenance.get("device"),
            "gpu_model": exact_provenance.get("gpu_model"),
            "runtime_tree_sha256": exact_runtime_sha,
        },
        "adjudication": {
            "promotion_eligible": adjudication.get("promotion_eligible"),
            "scientific_score_eligible": adjudication.get("scientific_score_eligible"),
            "contest_equivalent_hardware": adjudication.get("contest_equivalent_hardware"),
            "component_gate_triggered": adjudication.get("component_gate_triggered"),
            "score_delta_vs_baseline": score_delta,
            "allowed_use": list(adjudication.get("allowed_use") or []),
        },
        "blockers": blockers,
        "missing_evidence": [
            check["id"]
            for check in checks
            if check["blocking"] and not check["passed"] and str(check.get("failure_class")) == "missing_evidence"
        ],
        "warnings": warnings,
        "checks": checks,
        "input_paths": dict(input_paths or {}),
        "interpretation": (
            "This is a promotion review for an already exact-evaluated lossless "
            "Brotli control. It does not authorize new GPU dispatch and does not "
            "create a new score claim."
        ),
    }


def render_markdown(review: Mapping[str, Any]) -> str:
    """Render a compact human review note."""

    archive = _as_mapping(review.get("archive"))
    exact = _as_mapping(review.get("exact_eval"))
    adjudication = _as_mapping(review.get("adjudication"))
    lines = [
        "# PR106x Low-Level Brotli Promotion Review",
        "",
        f"- target_label: `{review.get('target_label')}`",
        f"- verdict: `{review.get('review_verdict')}`",
        f"- existing_exact_control_promotable: `{_bool_text(review.get('existing_exact_control_promotable') is True)}`",
        f"- score_claim: `{_bool_text(review.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool_text(review.get('dispatch_attempted') is True)}`",
        f"- review_basis: `{review.get('review_basis')}`",
        "",
        "## Exact Artifact",
        "",
        f"- archive_sha256: `{exact.get('archive_sha256')}`",
        f"- archive_bytes: `{exact.get('archive_bytes')}`",
        f"- payload_sha256: `{archive.get('payload_sha256')}`",
        f"- byte_delta_vs_source: `{archive.get('byte_delta')}`",
        f"- eval_artifact: `{exact.get('artifact')}`",
        f"- runtime_tree_sha256: `{exact.get('runtime_tree_sha256')}`",
        f"- device: `{exact.get('device')}` / `{exact.get('gpu_model')}`",
        f"- n_samples: `{exact.get('n_samples')}`",
        "",
        "## Adjudication",
        "",
        f"- promotion_eligible: `{_bool_text(adjudication.get('promotion_eligible') is True)}`",
        f"- scientific_score_eligible: `{_bool_text(adjudication.get('scientific_score_eligible') is True)}`",
        f"- contest_equivalent_hardware: `{_bool_text(adjudication.get('contest_equivalent_hardware') is True)}`",
        f"- component_gate_triggered: `{_bool_text(adjudication.get('component_gate_triggered') is True)}`",
        f"- score_delta_vs_baseline: `{adjudication.get('score_delta_vs_baseline')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = [str(item) for item in review.get("blockers") or []]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")

    warnings = [item for item in review.get("warnings") or [] if isinstance(item, Mapping)]
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- `{item.get('id')}`: {item.get('message')}")
    else:
        lines.append("- none")

    checks = [item for item in review.get("checks") or [] if isinstance(item, Mapping)]
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | status | requirement |",
            "|---|---:|---|",
        ]
    )
    for check in checks:
        status = "pass" if check.get("passed") is True else "fail"
        lines.append(f"| `{check.get('id')}` | `{status}` | {check.get('requirement')} |")

    lines.extend(
        [
            "",
            "Interpretation: this review is bounded to the existing exact CUDA",
            "artifact. It is not a new score claim, not a GPU dispatch",
            "authorization, and not evidence for future byte-different candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def _archive_identity_check(
    checks: list[dict[str, Any]],
    check_id: str,
    inspection: Mapping[str, Any] | None,
    *,
    expected_sha: str | None,
    expected_bytes: int | None,
    expected_member_sha: str | None,
) -> None:
    inspection_map = _as_mapping(inspection)
    _check(
        checks,
        check_id,
        bool(inspection)
        and not list(inspection_map.get("blockers") or [])
        and inspection_map.get("archive_sha256") == expected_sha
        and inspection_map.get("archive_bytes") == expected_bytes
        and inspection_map.get("member_sha256") == expected_member_sha,
        requirement="local archive copy must match reviewed SHA, bytes, and payload",
        evidence={
            "path": inspection_map.get("path"),
            "archive_sha256": inspection_map.get("archive_sha256"),
            "expected_archive_sha256": expected_sha,
            "archive_bytes": inspection_map.get("archive_bytes"),
            "expected_archive_bytes": expected_bytes,
            "member_sha256": inspection_map.get("member_sha256"),
            "expected_member_sha256": expected_member_sha,
            "blockers": list(inspection_map.get("blockers") or []),
        },
    )


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    requirement: str,
    evidence: Mapping[str, Any] | None = None,
    blocking: bool = True,
) -> None:
    failure_class = "ok"
    if not passed:
        failure_class = "missing_evidence" if _looks_missing(evidence) else "inconsistent_evidence"
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "blocking": blocking,
            "requirement": requirement,
            "failure_class": failure_class,
            "evidence": dict(evidence or {}),
        }
    )


def _looks_missing(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return True
    return all(value in (None, "", [], {}) for value in evidence.values())


def _scorecard_rows(scorecard: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = scorecard.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise HnervLosslessControlPromotionReviewError("scorecard rows must be a list")
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not out:
        raise HnervLosslessControlPromotionReviewError("scorecard rows must contain objects")
    return out


def _row_by_label(rows: Sequence[Mapping[str, Any]], label: str | None) -> dict[str, Any]:
    if not label:
        raise HnervLosslessControlPromotionReviewError("row label is required")
    for row in rows:
        if row.get("label") == label:
            return dict(row)
    raise HnervLosslessControlPromotionReviewError(f"scorecard row not found: {label}")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _same_present(*values: Any) -> bool:
    return all(value is not None for value in values) and len(set(values)) == 1


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _float_equal(left: Any, right: Any, *, tolerance: float = SCORE_TOLERANCE) -> bool:
    left_float = _first_float(left)
    right_float = _first_float(right)
    return left_float is not None and right_float is not None and abs(left_float - right_float) <= tolerance


def _structured_component_score(exact_eval: Mapping[str, Any]) -> float | None:
    component_values = [
        _first_float(exact_eval.get("score_seg_contribution")),
        _first_float(exact_eval.get("score_pose_contribution")),
        _first_float(exact_eval.get("score_rate_contribution")),
    ]
    if all(value is not None for value in component_values):
        return sum(value for value in component_values if value is not None)
    archive_bytes = _first_int(exact_eval.get("archive_size_bytes"))
    seg = _first_float(exact_eval.get("avg_segnet_dist"))
    pose = _first_float(exact_eval.get("avg_posenet_dist"))
    if archive_bytes is None or seg is None or pose is None:
        return None
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "BLOCKED_VERDICT",
    "PROMOTABLE_VERDICT",
    "HnervLosslessControlPromotionReviewError",
    "build_lossless_control_promotion_review",
    "inspect_single_member_archive",
    "render_markdown",
]
