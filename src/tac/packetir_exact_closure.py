# SPDX-License-Identifier: MIT
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
    runtime_consumption_proof: Mapping[str, Any] | None = None,
    full_frame_parity_proof: Mapping[str, Any] | None = None,
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
    expected_score_sections = tuple(
        str(item)
        for item in consumed.get("score_affecting_section_names") or []
        if isinstance(item, str) and item
    )
    byte_delta = _first_int(audit.get("total_byte_delta"))
    if byte_delta is None and isinstance(candidate_bytes, int) and isinstance(source_bytes, int):
        byte_delta = candidate_bytes - source_bytes
    cuda_summary = _eval_summary(cuda_eval, required_axis="contest_cuda")

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
    runtime_consumption_summary = _runtime_consumption_summary(
        runtime_consumption_proof,
        candidate_sha=candidate_sha,
        candidate_bytes=candidate_bytes,
        expected_score_affecting_sections=expected_score_sections,
    )
    _check(
        checks,
        "runtime_consumption_proof_binds_candidate_and_score_affecting_sections",
        runtime_consumption_summary["valid"] is True,
        requirement=(
            "runtime proof must bind the candidate archive and show every "
            "score-affecting PacketIR section is decoded/applied by the runtime"
        ),
        evidence=runtime_consumption_summary,
    )
    full_frame_parity_summary = _full_frame_parity_summary(
        full_frame_parity_proof,
        candidate_sha=candidate_sha,
        candidate_bytes=candidate_bytes,
        source_sha=source_sha,
        source_bytes=source_bytes,
        expected_n_pairs=_first_int(cuda_summary.get("n_samples")),
        expected_total_bytes=_first_int(cuda_summary.get("inflated_output_total_bytes")),
    )
    _check(
        checks,
        "same_runtime_full_frame_parity_binds_candidate_source_and_runtime",
        full_frame_parity_summary["valid"] is True,
        requirement=(
            "same-runtime full-frame parity must bind source/candidate archives "
            "and prove all frame bytes match before treating a PacketIR recode "
            "as rate-only"
        ),
        evidence=full_frame_parity_summary,
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
        source_cuda_summary is not None
        and source_cuda_summary["claim_valid"] is True
        and source_cuda_summary.get("archive_sha256") == source_sha
        and source_cuda_summary.get("archive_bytes") == source_bytes,
        requirement=(
            "source CUDA baseline must be supplied and must be the PacketIR "
            "source archive, not an inferred neighbor"
        ),
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
        current_best_summary is not None and current_best_summary["claim_valid"] is True,
        requirement="current-best reference must be supplied as a valid contest-CUDA score claim",
        evidence=current_best_summary,
    )

    _check(
        checks,
        "runtime_identity_matches_cuda_eval_runtime",
        _runtime_identity_matches_eval(
            full_frame_parity_summary,
            runtime_consumption_summary,
            cuda_summary,
        ),
        requirement=(
            "local runtime proofs must identify the same inflate runtime content "
            "that exact CUDA auth eval scored"
        ),
        evidence={
            "full_frame_runtime_inflate_py_sha256": full_frame_parity_summary.get(
                "runtime_inflate_py_sha256"
            ),
            "runtime_consumption_runtime_inflate_py_sha256": runtime_consumption_summary.get(
                "runtime_inflate_py_sha256"
            ),
            "runtime_consumption_runtime_dir": runtime_consumption_summary.get("runtime_dir"),
            "runtime_consumption_runtime_source_tree_sha256": runtime_consumption_summary.get(
                "runtime_source_tree_sha256"
            ),
            "cuda_runtime_inflate_py_sha256": cuda_summary.get("runtime_inflate_py_sha256"),
            "cuda_runtime_content_tree_sha256": cuda_summary.get(
                "runtime_content_tree_sha256"
            ),
        },
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
        "schema_version": 2,
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
            "runtime_consumption_proof": runtime_consumption_summary,
            "same_runtime_full_frame_parity": full_frame_parity_summary,
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
        "exact_eval_duplicate_keys": _exact_eval_duplicate_keys(cuda_summary),
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
    packetir = _as_mapping(closure.get("packetir"))
    runtime_proof = _as_mapping(packetir.get("runtime_consumption_proof"))
    full_frame_parity = _as_mapping(packetir.get("same_runtime_full_frame_parity"))
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
        f"- [contest-CUDA] runtime_content_tree_sha256: `{cuda.get('runtime_content_tree_sha256')}`",
        f"- [contest-CUDA] inflated_output_aggregate_sha256: `{cuda.get('inflated_output_aggregate_sha256')}`",
        f"- [contest-CPU] score: `{cpu.get('score')}` bytes: `{cpu.get('archive_bytes')}` sha: `{cpu.get('archive_sha256')}`",
        "",
        "## Runtime Proofs",
        "",
        f"- runtime_consumption_valid: `{_bool_text(runtime_proof.get('valid') is True)}`",
        f"- runtime_all_score_affecting_sections_consumed: `{runtime_proof.get('runtime_all_score_affecting_sections_consumed')}`",
        f"- same_runtime_full_frame_parity_valid: `{_bool_text(full_frame_parity.get('valid') is True)}`",
        f"- full_frame_streaming_raw_sha256: `{full_frame_parity.get('candidate_streaming_raw_sha256')}`",
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
    lines.extend(["", "## Exact Eval Duplicate Keys", ""])
    duplicate_keys = [
        str(item.get("key"))
        for item in closure.get("exact_eval_duplicate_keys") or []
        if isinstance(item, Mapping)
    ]
    lines.extend(f"- `{item}`" for item in duplicate_keys) if duplicate_keys else lines.append("- none")
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
    runtime_manifest = _as_mapping(provenance.get("inflate_runtime_manifest"))
    inflated_manifest = _as_mapping(
        _as_mapping(provenance.get("inflated_output_manifest")).get("payload")
    )
    runtime_files = [
        file
        for file in runtime_manifest.get("files", [])
        if isinstance(file, Mapping)
    ]
    inflate_py_sha = None
    inflate_sh_sha = None
    runtime_file_sha256s: dict[str, str] = {}
    for file in runtime_files:
        rel = file.get("relative_path")
        sha = _first_str(file.get("sha256"))
        if isinstance(rel, str) and sha:
            runtime_file_sha256s[rel] = sha
        if rel == "inflate.py":
            inflate_py_sha = sha
        elif rel == "inflate.sh":
            inflate_sh_sha = sha
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
        "runtime_tree_sha256": _first_str(runtime_manifest.get("runtime_tree_sha256")),
        "runtime_content_tree_sha256": _first_str(
            runtime_manifest.get("runtime_content_tree_sha256")
        ),
        "runtime_inflate_py_sha256": inflate_py_sha,
        "runtime_inflate_sh_sha256": inflate_sh_sha,
        "runtime_file_sha256s": runtime_file_sha256s,
        "inflated_output_aggregate_sha256": _first_str(
            inflated_manifest.get("aggregate_sha256")
        ),
        "inflated_output_total_bytes": _first_int(inflated_manifest.get("total_bytes")),
    }


def _runtime_consumption_summary(
    payload: Mapping[str, Any] | None,
    *,
    candidate_sha: str | None,
    candidate_bytes: int | None,
    expected_score_affecting_sections: Sequence[str],
) -> dict[str, Any]:
    data = _as_mapping(payload)
    archive = _as_mapping(data.get("archive"))
    consumed_sections = _as_mapping(data.get("runtime_consumed_score_affecting_sections"))
    blockers = list(data.get("blockers") or []) if isinstance(data.get("blockers"), list) else []
    runtime_source_manifest = _as_mapping(data.get("runtime_source_manifest"))
    runtime_files = [
        file
        for file in runtime_source_manifest.get("files", [])
        if isinstance(file, Mapping)
    ]
    runtime_file_sha256s = {
        str(file.get("path")): str(file.get("sha256"))
        for file in runtime_files
        if file.get("path") and _first_str(file.get("sha256"))
    }
    expected_sections = set(expected_score_affecting_sections)
    actual_sections = {str(key) for key in consumed_sections}
    runtime_inflate_py_sha = _first_str(data.get("runtime_inflate_py_sha256"))
    valid = (
        bool(data)
        and data.get("schema") == "pr106_sidecar_runtime_decode_consumption_proof_v1"
        and data.get("proof_scope")
        == "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame"
        and archive.get("sha256") == candidate_sha
        and archive.get("bytes") == candidate_bytes
        and not blockers
        and data.get("parser_consumed_byte_accounting_passed") is True
        and data.get("runtime_all_score_affecting_sections_consumed") is True
        and bool(consumed_sections)
        and bool(expected_sections)
        and actual_sections == expected_sections
        and all(value is True for value in consumed_sections.values())
        and data.get("runtime_corrected_latents_digest_changed") is True
        and isinstance(runtime_inflate_py_sha, str)
        and bool(runtime_file_sha256s)
        and runtime_file_sha256s.get("inflate.py") == runtime_inflate_py_sha
        and data.get("score_claim") is False
        and data.get("contest_axis_claim") is False
        and data.get("ready_for_exact_eval_dispatch") is False
    )
    return {
        "valid": valid,
        "schema": data.get("schema"),
        "archive_sha256": archive.get("sha256"),
        "archive_bytes": archive.get("bytes"),
        "expected_archive_sha256": candidate_sha,
        "expected_archive_bytes": candidate_bytes,
        "blockers": blockers,
        "runtime_dir": data.get("runtime_dir"),
        "runtime_inflate_py_sha256": runtime_inflate_py_sha,
        "runtime_source_tree_sha256": runtime_source_manifest.get(
            "runtime_source_tree_sha256"
        ),
        "runtime_source_files": [
            {
                "path": file.get("path"),
                "sha256": file.get("sha256"),
                "bytes": file.get("bytes"),
            }
            for file in runtime_files
        ],
        "runtime_file_sha256s": runtime_file_sha256s,
        "parser_consumed_byte_accounting_passed": data.get(
            "parser_consumed_byte_accounting_passed"
        ),
        "runtime_all_score_affecting_sections_consumed": data.get(
            "runtime_all_score_affecting_sections_consumed"
        ),
        "runtime_consumed_score_affecting_sections": dict(consumed_sections),
        "expected_score_affecting_sections": sorted(expected_sections),
        "actual_score_affecting_sections": sorted(actual_sections),
        "score_affecting_section_set_matches_packetir": actual_sections == expected_sections,
        "runtime_corrected_latents_digest_changed": data.get(
            "runtime_corrected_latents_digest_changed"
        ),
        "proof_scope": data.get("proof_scope"),
        "score_claim": data.get("score_claim"),
        "contest_axis_claim": data.get("contest_axis_claim"),
        "ready_for_exact_eval_dispatch": data.get("ready_for_exact_eval_dispatch"),
    }


def _full_frame_parity_summary(
    payload: Mapping[str, Any] | None,
    *,
    candidate_sha: str | None,
    candidate_bytes: int | None,
    source_sha: str | None,
    source_bytes: int | None,
    expected_n_pairs: int | None,
    expected_total_bytes: int | None,
) -> dict[str, Any]:
    data = _as_mapping(payload)
    candidate = _as_mapping(data.get("candidate"))
    source = _as_mapping(data.get("source"))
    candidate_archive = _as_mapping(data.get("candidate_archive"))
    source_archive = _as_mapping(data.get("source_archive"))
    candidate_n_pairs_hashed = _first_int(candidate.get("n_pairs_hashed"))
    candidate_n_pairs_total = _first_int(candidate.get("n_pairs_total"))
    source_n_pairs_hashed = _first_int(source.get("n_pairs_hashed"))
    source_n_pairs_total = _first_int(source.get("n_pairs_total"))
    candidate_total_bytes = _first_int(candidate.get("total_bytes"))
    source_total_bytes = _first_int(source.get("total_bytes"))
    valid = (
        bool(data)
        and data.get("schema") == "pr106_same_runtime_streaming_frame_parity_v1"
        and data.get("proof_scope") == "same_runtime_streaming_full_frame_hash"
        and data.get("device_axis_label") == "local-cpu-streaming-runtime"
        and data.get("full_frame_inflate_output_parity_claim") is True
        and data.get("prefix_parity_claim") is False
        and data.get("streaming_output_sha256_equal") is True
        and data.get("streaming_output_total_bytes_equal") is True
        and data.get("score_claim") is False
        and data.get("contest_axis_claim") is False
        and data.get("ready_for_exact_eval_dispatch") is False
        and candidate_archive.get("sha256") == candidate_sha
        and candidate_archive.get("bytes") == candidate_bytes
        and source_archive.get("sha256") == source_sha
        and source_archive.get("bytes") == source_bytes
        and candidate_n_pairs_total is not None
        and source_n_pairs_total is not None
        and expected_n_pairs is not None
        and candidate_n_pairs_hashed == candidate_n_pairs_total
        and source_n_pairs_hashed == source_n_pairs_total
        and candidate_n_pairs_total == expected_n_pairs
        and source_n_pairs_total == expected_n_pairs
        and candidate_total_bytes is not None
        and candidate_total_bytes > 0
        and expected_total_bytes is not None
        and candidate.get("streaming_raw_sha256") == source.get("streaming_raw_sha256")
        and candidate_total_bytes == source_total_bytes
        and candidate_total_bytes == expected_total_bytes
        and candidate.get("full_frame_digest") is True
        and source.get("full_frame_digest") is True
    )
    return {
        "valid": valid,
        "schema": data.get("schema"),
        "runtime_dir": data.get("runtime_dir"),
        "runtime_inflate_py_sha256": data.get("runtime_inflate_py_sha256"),
        "proof_scope": data.get("proof_scope"),
        "device_axis_label": data.get("device_axis_label"),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_archive_bytes": source_archive.get("bytes"),
        "expected_candidate_archive_sha256": candidate_sha,
        "expected_candidate_archive_bytes": candidate_bytes,
        "expected_source_archive_sha256": source_sha,
        "expected_source_archive_bytes": source_bytes,
        "full_frame_inflate_output_parity_claim": data.get(
            "full_frame_inflate_output_parity_claim"
        ),
        "streaming_output_sha256_equal": data.get("streaming_output_sha256_equal"),
        "streaming_output_total_bytes_equal": data.get(
            "streaming_output_total_bytes_equal"
        ),
        "candidate_streaming_raw_sha256": candidate.get("streaming_raw_sha256"),
        "source_streaming_raw_sha256": source.get("streaming_raw_sha256"),
        "candidate_total_bytes": candidate.get("total_bytes"),
        "source_total_bytes": source.get("total_bytes"),
        "candidate_n_pairs_hashed": candidate.get("n_pairs_hashed"),
        "candidate_n_pairs_total": candidate.get("n_pairs_total"),
        "source_n_pairs_hashed": source.get("n_pairs_hashed"),
        "source_n_pairs_total": source.get("n_pairs_total"),
        "expected_n_pairs": expected_n_pairs,
        "expected_total_bytes": expected_total_bytes,
        "candidate_full_frame_digest": candidate.get("full_frame_digest"),
        "source_full_frame_digest": source.get("full_frame_digest"),
        "score_claim": data.get("score_claim"),
        "contest_axis_claim": data.get("contest_axis_claim"),
        "ready_for_exact_eval_dispatch": data.get("ready_for_exact_eval_dispatch"),
    }


def _runtime_identity_matches_eval(
    full_frame_parity_summary: Mapping[str, Any],
    runtime_consumption_summary: Mapping[str, Any],
    cuda_summary: Mapping[str, Any],
) -> bool:
    inflate_py = full_frame_parity_summary.get("runtime_inflate_py_sha256")
    consumption_inflate_py = runtime_consumption_summary.get("runtime_inflate_py_sha256")
    cuda_inflate_py = cuda_summary.get("runtime_inflate_py_sha256")
    cuda_runtime_files = _as_mapping(cuda_summary.get("runtime_file_sha256s"))
    consumption_runtime_files = _as_mapping(
        runtime_consumption_summary.get("runtime_file_sha256s")
    )
    consumption_files_match_cuda = bool(consumption_runtime_files) and all(
        cuda_runtime_files.get(path) == sha
        for path, sha in consumption_runtime_files.items()
    )
    return (
        full_frame_parity_summary.get("valid") is True
        and runtime_consumption_summary.get("valid") is True
        and isinstance(inflate_py, str)
        and inflate_py
        and inflate_py == consumption_inflate_py
        and inflate_py == cuda_inflate_py
        and consumption_files_match_cuda
        and isinstance(cuda_summary.get("runtime_content_tree_sha256"), str)
    )


def _exact_eval_duplicate_keys(cuda_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    archive_sha = cuda_summary.get("archive_sha256")
    runtime_sha = cuda_summary.get("runtime_content_tree_sha256")
    score_axis = cuda_summary.get("score_axis")
    if not (
        isinstance(archive_sha, str)
        and isinstance(runtime_sha, str)
        and isinstance(score_axis, str)
    ):
        return []
    return [
        {
            "archive_sha256": archive_sha,
            "runtime_content_tree_sha256": runtime_sha,
            "score_axis": score_axis,
            "key": f"{archive_sha}:{runtime_sha}:{score_axis}",
        }
    ]


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
