"""Exact-eval closure review for PacketIR-derived archive candidates.

PacketIR/profile tools can prove byte accounting, but they cannot by
themselves decide score status. This module joins PacketIR custody, exact
CUDA/CPU auth-eval artifacts, and a current-best reference into one
fail-closed record that prevents duplicate dispatch and axis-mixing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.auth_eval_result import (
    CONTEST_UNCOMPRESSED_BYTES,
    parse_auth_eval_score_claim,
    parse_finite_auth_eval_score,
    recompute_contest_score_from_payload,
)
from tac.repo_io import repo_relative, sha256_file

SCHEMA = "packetir_exact_eval_closure_v1"
TOOL_NAME = "tac.packetir_exact_closure"
SCORE_TOLERANCE = 1e-9


class PacketIRExactClosureError(ValueError):
    """Raised when PacketIR exact-closure inputs are malformed."""


def build_packetir_exact_closure(
    *,
    lane_id: str,
    candidate_result: Mapping[str, Any],
    candidate_archive_path: str | Path,
    cuda_eval: Mapping[str, Any],
    cpu_eval: Mapping[str, Any] | None = None,
    source_cuda_eval: Mapping[str, Any] | None = None,
    current_best_cuda_eval: Mapping[str, Any] | None = None,
    recode_profile: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-score-claim exact closure manifest.

    The returned manifest is intentionally diagnostic: it may summarize an
    existing exact CUDA score, but it never marks the candidate promotion-ready
    or dispatch-ready. It is meant to close the loop after exact eval has
    already happened.
    """

    if not lane_id:
        raise PacketIRExactClosureError("lane_id must be non-empty")
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    candidate_sha = _first_str(candidate_result.get("candidate_archive_sha256"))
    candidate_bytes = _first_int(candidate_result.get("candidate_archive_bytes"))
    source_sha = _first_str(candidate_result.get("source_archive_sha256"))
    source_bytes = _first_int(candidate_result.get("source_archive_bytes"))
    candidate_archive = _archive_identity(candidate_archive_path, repo_root=repo_root)
    audit = _as_mapping(candidate_result.get("candidate_diff_audit"))
    consumed = _as_mapping(candidate_result.get("packet_ir_consumed_byte_proof"))
    byte_delta = _first_int(audit.get("total_byte_delta"))
    if byte_delta is None and isinstance(candidate_bytes, int) and isinstance(source_bytes, int):
        byte_delta = candidate_bytes - source_bytes

    _check(
        checks,
        "candidate_result_is_nonclaiming_static_artifact",
        candidate_result.get("score_claim") is False
        and candidate_result.get("dispatch_attempted") is False,
        requirement="PacketIR/static result must not claim score or dispatch by itself",
        evidence={
            "score_claim": candidate_result.get("score_claim"),
            "dispatch_attempted": candidate_result.get("dispatch_attempted"),
        },
    )
    _check(
        checks,
        "candidate_archive_file_matches_packetir_result",
        candidate_archive.get("exists") is True
        and candidate_archive.get("sha256") == candidate_sha
        and candidate_archive.get("bytes") == candidate_bytes,
        requirement="candidate archive on disk must match PacketIR result SHA and bytes",
        evidence={
            "path": candidate_archive.get("path"),
            "actual_sha256": candidate_archive.get("sha256"),
            "expected_sha256": candidate_sha,
            "actual_bytes": candidate_archive.get("bytes"),
            "expected_bytes": candidate_bytes,
        },
    )
    _check(
        checks,
        "candidate_diff_audit_closed_and_rate_positive",
        isinstance(byte_delta, int)
        and byte_delta < 0
        and not list(audit.get("blockers") or []),
        requirement="candidate diff audit must be blocker-free and byte-smaller than its source",
        evidence={
            "total_byte_delta": byte_delta,
            "audit_blockers": list(audit.get("blockers") or []),
            "source_archive_bytes": source_bytes,
            "candidate_archive_bytes": candidate_bytes,
        },
    )
    _check(
        checks,
        "packetir_consumed_byte_proof_complete",
        consumed.get("all_payload_bytes_accounted") is True
        and consumed.get("runtime_consumption_claim") is False,
        requirement="PacketIR proof must account for every payload byte without overstating runtime consumption",
        evidence={
            "all_payload_bytes_accounted": consumed.get("all_payload_bytes_accounted"),
            "runtime_consumption_claim": consumed.get("runtime_consumption_claim"),
            "unconsumed_trailing_bytes": consumed.get("unconsumed_trailing_bytes"),
        },
    )
    if isinstance(recode_profile, Mapping):
        _check(
            checks,
            "recode_profile_keeps_candidate_nonpromotable_before_exact_eval",
            _profile_row_is_nonpromotable(
                recode_profile,
                candidate_sha=candidate_sha,
                source_sha=source_sha,
            ),
            requirement=(
                "profile/emitted candidate or direct PacketIR source row must be "
                "non-claiming and exact-eval-blocked"
            ),
            evidence=_profile_evidence(
                recode_profile,
                candidate_sha=candidate_sha,
                source_sha=source_sha,
            ),
        )

    cuda_summary = _eval_summary(cuda_eval, required_axis="contest_cuda")
    _check(
        checks,
        "cuda_eval_is_valid_contest_cuda_score_claim",
        cuda_summary["claim_valid"] is True,
        requirement="CUDA artifact must pass score-claim parser and component recomputation",
        evidence=cuda_summary,
    )
    _check(
        checks,
        "cuda_eval_archive_matches_candidate",
        cuda_summary.get("archive_sha256") == candidate_sha
        and cuda_summary.get("archive_bytes") == candidate_bytes,
        requirement="CUDA auth eval must score the exact candidate archive",
        evidence={
            "cuda_archive_sha256": cuda_summary.get("archive_sha256"),
            "candidate_archive_sha256": candidate_sha,
            "cuda_archive_bytes": cuda_summary.get("archive_bytes"),
            "candidate_archive_bytes": candidate_bytes,
        },
    )

    cpu_summary: dict[str, Any] | None = None
    if cpu_eval is not None:
        cpu_summary = _eval_summary(cpu_eval, required_axis="contest_cpu")
        _check(
            checks,
            "cpu_eval_is_axis_labeled_diagnostic_not_cuda_claim",
            cpu_summary["finite_score"] is True
            and cpu_summary["score_axis"] == "contest_cpu"
            and cpu_summary["claim_valid"] is False,
            requirement="CPU artifact must stay on contest-CPU axis and not become CUDA score authority",
            evidence=cpu_summary,
        )
        _check(
            checks,
            "cpu_eval_archive_matches_candidate",
            cpu_summary.get("archive_sha256") == candidate_sha
            and cpu_summary.get("archive_bytes") == candidate_bytes,
            requirement="CPU auth eval must score the same candidate archive before comparing axes",
            evidence={
                "cpu_archive_sha256": cpu_summary.get("archive_sha256"),
                "candidate_archive_sha256": candidate_sha,
                "cpu_archive_bytes": cpu_summary.get("archive_bytes"),
                "candidate_archive_bytes": candidate_bytes,
            },
        )

    source_cuda_summary: dict[str, Any] | None = None
    if source_cuda_eval is not None:
        source_cuda_summary = _eval_summary(source_cuda_eval, required_axis="contest_cuda")
        _check(
            checks,
            "source_cuda_eval_matches_packetir_source",
            source_cuda_summary["claim_valid"] is True
            and source_cuda_summary.get("archive_sha256") == source_sha
            and source_cuda_summary.get("archive_bytes") == source_bytes,
            requirement="source CUDA baseline must be the PacketIR source archive, not an inferred neighbor",
            evidence={
                "source_eval": source_cuda_summary,
                "packetir_source_archive_sha256": source_sha,
                "packetir_source_archive_bytes": source_bytes,
            },
        )

    current_best_summary: dict[str, Any] | None = None
    if current_best_cuda_eval is not None:
        current_best_summary = _eval_summary(current_best_cuda_eval, required_axis="contest_cuda")
        _check(
            checks,
            "current_best_cuda_eval_is_valid_reference",
            current_best_summary["claim_valid"] is True,
            requirement="current-best reference must be a valid contest-CUDA score claim",
            evidence=current_best_summary,
        )

    blockers = [
        check["id"]
        for check in checks
        if check["blocking"] and check["passed"] is not True
    ]
    comparisons = _comparisons(
        candidate=cuda_summary,
        source=source_cuda_summary,
        current_best=current_best_summary,
        byte_delta=byte_delta,
    )
    classification = _classification(blockers, comparisons)
    axis_gap = None
    if cpu_summary is not None and _is_number(cpu_summary.get("score")) and _is_number(cuda_summary.get("score")):
        axis_gap = {
            "cpu_minus_cuda_score": float(cpu_summary["score"]) - float(cuda_summary["score"]),
            "cpu_score": cpu_summary["score"],
            "cuda_score": cuda_summary["score"],
            "interpretation": "axis divergence only; never convert CPU to CUDA or CUDA to CPU",
        }
    if current_best_summary and _is_number(current_best_summary.get("score")):
        best_score = float(current_best_summary["score"])
        cand_score = float(cuda_summary["score"]) if _is_number(cuda_summary.get("score")) else math.inf
        if cand_score > best_score + SCORE_TOLERANCE:
            warnings.append(
                {
                    "id": "candidate_exact_cuda_closed_but_not_current_frontier",
                    "severity": "info",
                    "message": "candidate improved its PacketIR source line but is worse than the current exact-CUDA reference",
                    "evidence": {
                        "candidate_score": cand_score,
                        "current_best_score": best_score,
                        "delta_vs_current_best": cand_score - best_score,
                    },
                }
            )

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "tool": TOOL_NAME,
        "lane_id": lane_id,
        "classification": classification,
        "score_claim": False,
        "new_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "duplicate_dispatch_blockers": _duplicate_dispatch_blockers(classification),
        "archive": {
            "source_archive_sha256": source_sha,
            "source_archive_bytes": source_bytes,
            "candidate_archive_sha256": candidate_sha,
            "candidate_archive_bytes": candidate_bytes,
            "candidate_archive_path": candidate_archive.get("path"),
            "byte_delta_vs_packetir_source": byte_delta,
            "official_rate_score_delta_if_components_equal": None
            if byte_delta is None
            else 25.0 * byte_delta / CONTEST_UNCOMPRESSED_BYTES,
        },
        "packetir": {
            "candidate_diff_audit_blockers": list(audit.get("blockers") or []),
            "consumed_byte_proof": {
                "all_payload_bytes_accounted": consumed.get("all_payload_bytes_accounted"),
                "runtime_consumption_claim": consumed.get("runtime_consumption_claim"),
                "unconsumed_trailing_bytes": consumed.get("unconsumed_trailing_bytes"),
                "proof_scope": consumed.get("proof_scope"),
            },
        },
        "axes": {
            "contest_cuda": cuda_summary,
            "contest_cpu": cpu_summary,
            "axis_gap": axis_gap,
        },
        "comparisons": comparisons,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "reactivation_criteria": [
            "do_not_dispatch_same_candidate_archive_sha256_again",
            "reactivate_only_if_a_new_runtime_consumed_recode_has_smaller_exact_cuda_score_than_current_best",
            "reactivate_if_component_distortion_improves_enough_to_beat_current_best_after_exact_cuda_eval",
            "reactivate_if_a PacketIR candidate clears runtime_decoder_implemented plus exact CUDA closure and is not already measured",
        ],
        "input_paths": dict(input_paths or {}),
        "interpretation": (
            "This closure joins existing exact-eval artifacts. It does not "
            "authorize promotion, rank/kill decisions outside the stated axes, "
            "or another dispatch of the same archive SHA."
        ),
    }


def render_packetir_exact_closure_markdown(closure: Mapping[str, Any]) -> str:
    """Render a compact operator-facing closure report."""

    archive = _as_mapping(closure.get("archive"))
    axes = _as_mapping(closure.get("axes"))
    cuda = _as_mapping(axes.get("contest_cuda"))
    cpu = _as_mapping(axes.get("contest_cpu"))
    comparisons = _as_mapping(closure.get("comparisons"))
    lines = [
        "# PacketIR Exact-Eval Closure",
        "",
        f"- lane_id: `{closure.get('lane_id')}`",
        f"- classification: `{closure.get('classification')}`",
        f"- score_claim: `{_bool_text(closure.get('score_claim') is True)}`",
        f"- promotion_eligible: `{_bool_text(closure.get('promotion_eligible') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(closure.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Archive",
        "",
        f"- source_sha256: `{archive.get('source_archive_sha256')}`",
        f"- candidate_sha256: `{archive.get('candidate_archive_sha256')}`",
        f"- candidate_bytes: `{archive.get('candidate_archive_bytes')}`",
        f"- byte_delta_vs_source: `{archive.get('byte_delta_vs_packetir_source')}`",
        f"- rate_delta_if_components_equal: `{archive.get('official_rate_score_delta_if_components_equal')}`",
        "",
        "## Axes",
        "",
        f"- [contest-CUDA] score: `{cuda.get('score')}` bytes: `{cuda.get('archive_bytes')}` sha: `{cuda.get('archive_sha256')}`",
        f"- [contest-CPU] score: `{cpu.get('score')}` bytes: `{cpu.get('archive_bytes')}` sha: `{cpu.get('archive_sha256')}`",
        "",
        "## Comparisons",
        "",
        f"- delta_vs_packetir_source_cuda: `{comparisons.get('delta_vs_source_cuda')}`",
        f"- delta_vs_current_best_cuda: `{comparisons.get('delta_vs_current_best_cuda')}`",
        f"- not_current_frontier: `{_bool_text(comparisons.get('not_current_frontier') is True)}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = [str(item) for item in closure.get("blockers") or []]
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Duplicate Dispatch Blockers", ""])
    dispatch_blockers = [str(item) for item in closure.get("duplicate_dispatch_blockers") or []]
    lines.extend(f"- `{item}`" for item in dispatch_blockers) if dispatch_blockers else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = [item for item in closure.get("warnings") or [] if isinstance(item, Mapping)]
    if warnings:
        for item in warnings:
            lines.append(f"- `{item.get('id')}`: {item.get('message')}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _eval_summary(payload: Mapping[str, Any], *, required_axis: str) -> dict[str, Any]:
    finite = parse_finite_auth_eval_score(payload, require_component_recompute=True)
    claim = parse_auth_eval_score_claim(
        payload,
        required_score_axis=required_axis,
        require_component_recompute=True,
    )
    provenance = _as_mapping(payload.get("provenance"))
    archive_sha = _first_str(
        provenance.get("archive_sha256"),
        payload.get("archive_sha256"),
        payload.get("contest_cuda_archive_sha256"),
    )
    archive_bytes = _first_int(
        payload.get("archive_size_bytes"),
        payload.get("archive_bytes"),
        provenance.get("archive_size_bytes"),
    )
    recomputed = recompute_contest_score_from_payload(payload)
    return {
        "score_axis": str(payload.get("score_axis") or ""),
        "lane_tag": str(payload.get("lane_tag") or ""),
        "evidence_grade": str(payload.get("evidence_grade") or ""),
        "finite_score": finite is not None,
        "claim_valid": claim is not None,
        "score": None if finite is None else finite.score,
        "score_source_key": None if finite is None else finite.source_key,
        "recomputed_score": recomputed,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "avg_segnet_dist": _first_float(payload.get("avg_segnet_dist")),
        "avg_posenet_dist": _first_float(payload.get("avg_posenet_dist")),
        "n_samples": _first_int(payload.get("n_samples")),
        "score_claim": payload.get("score_claim"),
        "score_claim_valid": payload.get("score_claim_valid"),
        "exact_cuda_eval_complete": payload.get("exact_cuda_eval_complete"),
        "promotion_eligible": payload.get("promotion_eligible"),
        "rank_or_kill_blockers": list(payload.get("rank_or_kill_blockers") or []),
        "promotion_blockers": list(payload.get("promotion_blockers") or []),
    }


def _comparisons(
    *,
    candidate: Mapping[str, Any],
    source: Mapping[str, Any] | None,
    current_best: Mapping[str, Any] | None,
    byte_delta: int | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "byte_delta_vs_packetir_source": byte_delta,
        "delta_vs_source_cuda": None,
        "delta_vs_current_best_cuda": None,
        "improves_packetir_source_cuda": None,
        "not_current_frontier": None,
    }
    cand_score = _first_float(candidate.get("score"))
    if source is not None:
        source_score = _first_float(source.get("score"))
        if cand_score is not None and source_score is not None:
            delta = cand_score - source_score
            out["delta_vs_source_cuda"] = delta
            out["improves_packetir_source_cuda"] = delta < -SCORE_TOLERANCE
    if current_best is not None:
        best_score = _first_float(current_best.get("score"))
        if cand_score is not None and best_score is not None:
            delta = cand_score - best_score
            out["delta_vs_current_best_cuda"] = delta
            out["not_current_frontier"] = delta > SCORE_TOLERANCE
    return out


def _classification(blockers: Sequence[str], comparisons: Mapping[str, Any]) -> str:
    if blockers:
        return "blocked_inconsistent_or_missing_evidence"
    if comparisons.get("not_current_frontier") is True:
        return "exact_measured_not_current_frontier"
    if comparisons.get("improves_packetir_source_cuda") is True:
        return "exact_measured_improves_packetir_source_cuda"
    return "exact_measured_closed_no_frontier_claim"


def _duplicate_dispatch_blockers(classification: str) -> list[str]:
    blockers = ["same_candidate_archive_already_exact_evaluated"]
    if classification == "exact_measured_not_current_frontier":
        blockers.append("candidate_not_current_frontier_on_contest_cuda")
    if classification.startswith("blocked_"):
        blockers.append("closure_evidence_inconsistent_fail_closed")
    return blockers


def _profile_row_is_nonpromotable(
    profile: Mapping[str, Any],
    *,
    candidate_sha: str | None,
    source_sha: str | None,
) -> bool:
    evidence = _profile_evidence(profile, candidate_sha=candidate_sha, source_sha=source_sha)
    return (
        evidence.get("found") is True
        and evidence.get("score_claim") is False
        and evidence.get("promotion_eligible") is False
        and evidence.get("ready_for_exact_eval_dispatch") is False
    )


def _profile_evidence(
    profile: Mapping[str, Any],
    *,
    candidate_sha: str | None,
    source_sha: str | None,
) -> dict[str, Any]:
    for row in profile.get("candidate_rows") or []:
        if not isinstance(row, Mapping):
            continue
        emitted_sha = row.get("emitted_candidate_archive_sha256")
        matches_exact_candidate = candidate_sha is not None and emitted_sha == candidate_sha
        matches_packetir_source = source_sha is not None and emitted_sha == source_sha
        if not (matches_exact_candidate or matches_packetir_source):
            continue
        return {
            "found": True,
            "name": row.get("name"),
            "emitted_candidate_archive_sha256": emitted_sha,
            "matches_exact_candidate_archive": matches_exact_candidate,
            "matches_packetir_source_archive": matches_packetir_source,
            "score_claim": row.get("score_claim"),
            "promotion_eligible": row.get("promotion_eligible"),
            "ready_for_exact_eval_dispatch": row.get("ready_for_exact_eval_dispatch"),
            "runtime_decoder_implemented": row.get("runtime_decoder_implemented"),
        }
    return {
        "found": False,
        "candidate_archive_sha256": candidate_sha,
        "source_archive_sha256": source_sha,
    }


def _archive_identity(path: str | Path, *, repo_root: str | Path | None) -> dict[str, Any]:
    archive = Path(path)
    exists = archive.is_file()
    return {
        "path": repo_relative(archive, repo_root) if repo_root is not None else archive.as_posix(),
        "exists": exists,
        "bytes": archive.stat().st_size if exists else None,
        "sha256": sha256_file(archive) if exists else None,
    }


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    requirement: str,
    evidence: Mapping[str, Any] | None = None,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "blocking": blocking,
            "requirement": requirement,
            "failure_class": "ok" if passed else _failure_class(evidence),
            "evidence": dict(evidence or {}),
        }
    )


def _failure_class(evidence: Mapping[str, Any] | None) -> str:
    if not evidence or all(value in (None, "", [], {}) for value in evidence.values()):
        return "missing_evidence"
    return "inconsistent_evidence"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out = float(value)
            if math.isfinite(out):
                return out
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "PacketIRExactClosureError",
    "build_packetir_exact_closure",
    "render_packetir_exact_closure_markdown",
]
