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
from itertools import pairwise
from pathlib import Path
from typing import Any

from tac.auth_eval_result import (
    CONTEST_UNCOMPRESSED_BYTES,
    parse_auth_eval_score_claim,
    parse_finite_auth_eval_score,
    recompute_contest_score_from_payload,
)
from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.packet_compiler.pr106_sidecar_packet import PR106_PACKET_IR_SECTION_HASH_DOMAIN
from tac.repo_io import repo_relative, sha256_file

SCHEMA = "packetir_exact_eval_closure_v1"
TOOL_NAME = "tac.packetir_exact_closure"
SCORE_TOLERANCE = 1e-9
FORMAT0D_SCORE_AFFECTING_SECTIONS = (
    "pr106_payload",
    "base_format0c_sidecar_payload",
    "extra_pr101_ranked_no_op_payload",
    "extra_framing_meta",
)
FORMAT0D_RUNTIME_APPLY_ORDER = (
    "base_format0c_corrections",
    "extra_pr101_ranked_no_op_corrections",
)


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
    consumed = _as_mapping(
        candidate_result.get("packet_ir_consumed_byte_proof")
        or candidate_result.get("candidate_packet_ir_consumed_byte_proof")
    )
    expected_score_sections = tuple(
        str(item)
        for item in consumed.get("score_affecting_section_names") or []
        if isinstance(item, str) and item
    )
    byte_delta = _first_int(audit.get("total_byte_delta"))
    if byte_delta is None:
        byte_delta = _first_int(
            candidate_result.get("candidate_archive_byte_delta_vs_source"),
            candidate_result.get("candidate_archive_byte_delta"),
        )
    cuda_summary = _eval_summary(cuda_eval, required_axis="contest_cuda")
    source_cuda_summary: dict[str, Any] | None = None
    if source_cuda_eval is not None:
        source_cuda_summary = _eval_summary(source_cuda_eval, required_axis="contest_cuda")
        if source_sha is None:
            source_sha = _first_str(source_cuda_summary.get("archive_sha256"))
        if source_bytes is None:
            source_bytes = _first_int(source_cuda_summary.get("archive_bytes"))
    current_best_summary: dict[str, Any] | None = None
    if current_best_cuda_eval is not None:
        current_best_summary = _eval_summary(current_best_cuda_eval, required_axis="contest_cuda")
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
    runtime_consumption_summary = _runtime_consumption_summary(
        runtime_consumption_proof,
        candidate_sha=candidate_sha,
        candidate_bytes=candidate_bytes,
        expected_score_affecting_sections=expected_score_sections,
        candidate_consumed_byte_proof=consumed,
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
        "cuda_eval_axis_semantics_are_contest_cuda",
        not cuda_summary["axis_semantics_blockers"],
        requirement=(
            "CUDA artifact must carry contest-CUDA device semantics and full "
            "600-sample coverage, not just a CUDA-looking score label"
        ),
        evidence={
            "score_axis": cuda_summary.get("score_axis"),
            "evidence_grade": cuda_summary.get("evidence_grade"),
            "n_samples": cuda_summary.get("n_samples"),
            "eval_device": cuda_summary.get("eval_device"),
            "hardware": cuda_summary.get("hardware"),
            "axis_semantics_blockers": cuda_summary["axis_semantics_blockers"],
        },
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
    cuda_authority_blockers = _axis_authority_blockers(cuda_summary)
    _check(
        checks,
        "cuda_eval_promotion_and_rank_authority_blockers_absent",
        not cuda_authority_blockers,
        requirement=(
            "CUDA auth-eval rows with promotion or rank/kill blockers must not "
            "be summarized by a blocker-free top-level PacketIR closure"
        ),
        evidence={
            "promotion_eligible": cuda_summary.get("promotion_eligible"),
            "promotion_blockers": list(cuda_summary.get("promotion_blockers") or []),
            "rank_or_kill_blockers": list(
                cuda_summary.get("rank_or_kill_blockers") or []
            ),
            "cuda_authority_blockers": cuda_authority_blockers,
        },
    )

    cpu_summary: dict[str, Any] | None = None
    if cpu_eval is not None:
        cpu_summary = _eval_summary(cpu_eval, required_axis="contest_cpu")
        _mark_non_cuda_axis_diagnostic(cpu_summary)
        _check(
            checks,
            "cpu_eval_is_axis_labeled_diagnostic_not_cuda_claim",
            cpu_summary["finite_score"] is True
            and cpu_summary["score_axis"] == "contest_cpu"
            and cpu_summary["claim_valid"] is False
            and not cpu_summary["axis_semantics_blockers"],
            requirement="CPU artifact must stay on contest-CPU axis and not become CUDA score authority",
            evidence={
                **cpu_summary,
                "axis_semantics_blockers": cpu_summary["axis_semantics_blockers"],
            },
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
            "full_frame_runtime_content_tree_sha256": full_frame_parity_summary.get(
                "runtime_content_tree_sha256"
            ),
            "runtime_consumption_runtime_content_tree_sha256": runtime_consumption_summary.get(
                "runtime_content_tree_sha256"
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
        "schema_version": 3,
        "tool": TOOL_NAME,
        "lane_id": lane_id,
        "classification": classification,
        "score_claim": False,
        "new_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "closure_authority_blockers": cuda_authority_blockers,
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
        f"- runtime_consumption_runtime_content_tree_sha256: `{runtime_proof.get('runtime_content_tree_sha256')}`",
        f"- same_runtime_full_frame_parity_valid: `{_bool_text(full_frame_parity.get('valid') is True)}`",
        f"- full_frame_runtime_content_tree_sha256: `{full_frame_parity.get('runtime_content_tree_sha256')}`",
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
    score_axis = str(payload.get("score_axis") or "")
    evidence_grade = str(payload.get("evidence_grade") or "")
    n_samples = _first_int(payload.get("n_samples"))
    eval_device = _first_str(
        payload.get("scorer_device"),
        payload.get("actual_device"),
        payload.get("provenance_device"),
        payload.get("device"),
        provenance.get("actual_device"),
        provenance.get("device"),
    )
    hardware = _first_str(
        payload.get("gpu_model"),
        payload.get("hardware"),
        provenance.get("gpu_model"),
        provenance.get("hardware"),
        provenance.get("device"),
    )
    axis_semantics_blockers = _eval_axis_semantics_blockers(
        required_axis=required_axis,
        score_axis=score_axis,
        evidence_grade=evidence_grade,
        n_samples=n_samples,
        eval_device=eval_device,
        hardware=hardware,
    )
    return {
        "score_axis": score_axis,
        "lane_tag": str(payload.get("lane_tag") or ""),
        "evidence_grade": evidence_grade,
        "finite_score": finite is not None,
        "claim_valid": claim is not None,
        "score": None if finite is None else finite.score,
        "score_source_key": None if finite is None else finite.source_key,
        "recomputed_score": recomputed,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "avg_segnet_dist": _first_float(payload.get("avg_segnet_dist")),
        "avg_posenet_dist": _first_float(payload.get("avg_posenet_dist")),
        "n_samples": n_samples,
        "eval_device": eval_device,
        "hardware": hardware,
        "axis_semantics_blockers": axis_semantics_blockers,
        "score_claim": payload.get("score_claim"),
        "score_claim_valid": payload.get("score_claim_valid"),
        "exact_cuda_eval_complete": payload.get("exact_cuda_eval_complete"),
        "promotion_eligible": payload.get("promotion_eligible"),
        "rank_or_kill_blockers": _str_list(payload.get("rank_or_kill_blockers")),
        "promotion_blockers": _str_list(payload.get("promotion_blockers")),
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


def _mark_non_cuda_axis_diagnostic(summary: dict[str, Any]) -> None:
    """Keep non-CUDA axis rows visibly non-authoritative for row consumers."""

    blockers = [
        "not_contest_cuda_axis",
        "cpu_axis_not_rank_or_kill_authority",
        "requires_cuda_cpu_policy_review",
    ]
    for key in ("promotion_blockers", "rank_or_kill_blockers"):
        existing = [str(item) for item in summary.get(key) or []]
        summary[key] = [*existing, *[item for item in blockers if item not in existing]]


def _eval_axis_semantics_blockers(
    *,
    required_axis: str,
    score_axis: str,
    evidence_grade: str,
    n_samples: int | None,
    eval_device: str | None,
    hardware: str | None,
) -> list[str]:
    """Return blockers for exact-eval axis/device/sample semantics."""

    blockers: list[str] = []
    if score_axis != required_axis:
        blockers.append("score_axis_mismatch")
    if n_samples != CONTEST_EXACT_SAMPLE_COUNT:
        blockers.append("n_samples_not_contest_exact")
    eval_device_text = str(eval_device or "").strip().lower()
    hardware_text = str(hardware or "").strip().lower()
    if required_axis == "contest_cuda":
        if "cuda" not in eval_device_text:
            blockers.append("eval_device_not_cuda")
        if not any(token in hardware_text for token in ("cuda", "gpu", "t4", "a10", "a100", "h100", "l4")):
            blockers.append("hardware_not_cuda")
        if "cuda" not in str(evidence_grade or "").strip().lower():
            blockers.append("evidence_grade_not_cuda")
    elif required_axis == "contest_cpu":
        if "cpu" not in eval_device_text:
            blockers.append("eval_device_not_cpu")
        if "cpu" not in str(evidence_grade or "").strip().lower():
            blockers.append("evidence_grade_not_cpu")
    return blockers


def _axis_authority_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return exact-axis blockers that must be visible at closure top level."""

    blockers: list[str] = []
    for key in ("promotion_blockers", "rank_or_kill_blockers"):
        blockers.extend(_str_list(summary.get(key)))
    return list(dict.fromkeys(blockers))


def _runtime_consumption_summary(
    payload: Mapping[str, Any] | None,
    *,
    candidate_sha: str | None,
    candidate_bytes: int | None,
    expected_score_affecting_sections: Sequence[str],
    candidate_consumed_byte_proof: Mapping[str, Any] | None = None,
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
    actual_sections = _runtime_consumed_section_names(consumed_sections)
    section_match = _runtime_score_affecting_sections_match(
        expected_sections=expected_sections,
        actual_sections=actual_sections,
        proof=data,
        candidate_consumed_byte_proof=candidate_consumed_byte_proof,
    )
    runtime_inflate_py_sha = _first_str(data.get("runtime_inflate_py_sha256"))
    runtime_content_tree_sha = _first_str(
        data.get("runtime_content_tree_sha256"),
        runtime_source_manifest.get("runtime_content_tree_sha256"),
    )
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
        and section_match["matched"] is True
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
        "runtime_content_tree_sha256": runtime_content_tree_sha,
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
        "score_affecting_section_set_matches_packetir": section_match["matched"],
        "score_affecting_section_match_mode": section_match["mode"],
        "score_affecting_section_match_evidence": section_match["evidence"],
        "runtime_corrected_latents_digest_changed": data.get(
            "runtime_corrected_latents_digest_changed"
        ),
        "proof_scope": data.get("proof_scope"),
        "score_claim": data.get("score_claim"),
        "contest_axis_claim": data.get("contest_axis_claim"),
        "ready_for_exact_eval_dispatch": data.get("ready_for_exact_eval_dispatch"),
    }


def _runtime_score_affecting_sections_match(
    *,
    expected_sections: set[str],
    actual_sections: set[str],
    proof: Mapping[str, Any],
    candidate_consumed_byte_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return whether runtime proof consumption covers PacketIR sections.

    Most PacketIR recodes preserve the same section names through runtime
    decode. Format 0x0D is multi-pass and must bind the base 0x0C stream, the
    extra PR101 stream, the extra framing metadata, and the runtime's base-then-
    extra apply order. Formats 0x08, 0x09, 0x0A, 0x0B, and 0x0C deliberately
    elide the PR106 inner header from fixed HDM8/HLM2, HDM9/HLM2, or HDM9/HLM3
    payloads; 0x0B and 0x0C also elide fixed HDM9/HLM3 section magics. The
    runtime reconstructs those constants and consumes the reconstructed generic
    ``pr106_payload``. Accept that alias only when the proof binds the fixed
    format and records unchanged inner PR106 semantics.
    """

    if _format_id_text(proof.get("format_id")) == "0x0D":
        identity = _format0d_closure_identity(
            proof,
            candidate_consumed_byte_proof=candidate_consumed_byte_proof,
        )
        if (
            expected_sections == set(FORMAT0D_SCORE_AFFECTING_SECTIONS)
            and actual_sections == expected_sections
            and identity["valid"] is True
        ):
            return {
                "matched": True,
                "mode": "format_0x0d_base_then_extra_runtime_closure",
                "evidence": {
                    "format0d_expected_sections": list(
                        FORMAT0D_SCORE_AFFECTING_SECTIONS
                    ),
                    "format0d_closure_identity": identity,
                },
            }
        return {
            "matched": False,
            "mode": "mismatch",
            "evidence": {
                "format0d_expected_sections": list(FORMAT0D_SCORE_AFFECTING_SECTIONS),
                "format0d_closure_identity": identity,
            },
        }

    if actual_sections == expected_sections:
        return {
            "matched": True,
            "mode": "exact_section_names",
            "evidence": {},
        }
    generic_actual = {"pr106_payload", "sidecar_payload"}
    headerless_aliases = {
        "0x08": (
            "pr106_hdm8_hlm2_payload_without_inner_header",
            "format_0x08_hdm8_hlm2_reconstructed_pr106_payload_alias",
        ),
        "0x09": (
            "pr106_hdm9_hlm2_payload_without_inner_header",
            "format_0x09_hdm9_hlm2_reconstructed_pr106_payload_alias",
        ),
        "0x0A": (
            "pr106_hdm9_hlm3_payload_without_inner_header",
            "format_0x0a_hdm9_hlm3_reconstructed_pr106_payload_alias",
        ),
        "0x0B": (
            "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
            "format_0x0b_hdm9_hlm3_magicless_reconstructed_pr106_payload_alias",
        ),
        "0x0C": (
            "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
            "format_0x0c_hdm9_hlm3_magicless_exact_radix_reconstructed_pr106_payload_alias",
        ),
    }
    alias = headerless_aliases.get(_format_id_text(proof.get("format_id")))
    if alias is not None:
        expected_headerless_section, mode = alias
        identity = _headerless_alias_identity(
            proof,
            expected_section=expected_headerless_section,
            candidate_consumed_byte_proof=candidate_consumed_byte_proof,
        )
        if (
            expected_sections == {expected_headerless_section, "sidecar_payload"}
            and actual_sections == generic_actual
            and proof.get("inner_pr106_payload_sha256_unchanged") is True
            and identity["valid"] is True
        ):
            return {
                "matched": True,
                "mode": mode,
                "evidence": {
                    "expected_headerless_section": expected_headerless_section,
                    "runtime_consumed_reconstructed_section": "pr106_payload",
                    "inner_pr106_payload_sha256_unchanged": proof.get(
                        "inner_pr106_payload_sha256_unchanged"
                    ),
                    "headerless_alias_identity": identity,
                },
            }
        return {
            "matched": False,
            "mode": "mismatch",
            "evidence": {
                "expected_headerless_section": expected_headerless_section,
                "runtime_consumed_reconstructed_section": "pr106_payload",
                "inner_pr106_payload_sha256_unchanged": proof.get(
                    "inner_pr106_payload_sha256_unchanged"
                ),
                "headerless_alias_identity": identity,
            },
        }
    return {
        "matched": False,
        "mode": "mismatch",
        "evidence": {},
    }


def _format0d_closure_identity(
    proof: Mapping[str, Any],
    *,
    candidate_consumed_byte_proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate Format0D runtime closure against PacketIR section identities."""

    consumed_sections = _as_mapping(proof.get("runtime_consumed_score_affecting_sections"))
    runtime_apply_order = _format0d_runtime_apply_order(proof)
    section_evidence: dict[str, dict[str, Any]] = {}
    for section_name in FORMAT0D_SCORE_AFFECTING_SECTIONS:
        candidate_section = _candidate_consumed_section_identity(
            candidate_consumed_byte_proof,
            expected_section=section_name,
        )
        runtime_section = _runtime_consumed_section_identity(
            proof,
            consumed_sections=consumed_sections,
            expected_section=section_name,
        )
        sha_match = (
            isinstance(candidate_section["sha256"], str)
            and candidate_section["sha256"] == runtime_section["sha256"]
        )
        length_match = (
            candidate_section["length"] is not None
            and candidate_section["length"] == runtime_section["length"]
        )
        offset_match = (
            candidate_section["offset"] is not None
            and candidate_section["offset"] == runtime_section["offset"]
        )
        candidate_hash_domain_valid = (
            candidate_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        )
        runtime_hash_domain_valid = (
            runtime_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        )
        section_evidence[section_name] = {
            "candidate_section_found": candidate_section["found"],
            "candidate_section_sha256": candidate_section["sha256"],
            "candidate_section_hash_domain": candidate_section["hash_domain"],
            "candidate_section_length": candidate_section["length"],
            "candidate_section_offset": candidate_section["offset"],
            "candidate_section_score_affecting": candidate_section["score_affecting"],
            "runtime_section_found": runtime_section["found"],
            "runtime_section_consumed": runtime_section["consumed"],
            "runtime_section_sha256": runtime_section["sha256"],
            "runtime_section_hash_domain": runtime_section["hash_domain"],
            "runtime_section_length": runtime_section["length"],
            "runtime_section_offset": runtime_section["offset"],
            "sha256_matches": sha_match,
            "hash_domain_matches": (
                candidate_hash_domain_valid
                and runtime_hash_domain_valid
                and candidate_section["hash_domain"] == runtime_section["hash_domain"]
            ),
            "candidate_hash_domain_valid": candidate_hash_domain_valid,
            "runtime_hash_domain_valid": runtime_hash_domain_valid,
            "length_matches": length_match,
            "offset_matches": offset_match,
            "identity_valid": (
                candidate_section["found"] is True
                and candidate_section["score_affecting"] is True
                and runtime_section["found"] is True
                and runtime_section["consumed"] is True
                and sha_match
                and candidate_hash_domain_valid
                and runtime_hash_domain_valid
                and length_match
                and offset_match
            ),
        }

    ordered_sections = (
        "base_format0c_sidecar_payload",
        "extra_pr101_ranked_no_op_payload",
        "extra_framing_meta",
    )
    ordered_offsets = [
        section_evidence[name]["candidate_section_offset"]
        for name in ordered_sections
    ]
    candidate_base_extra_ordered = all(
        isinstance(offset, int) for offset in ordered_offsets
    ) and all(
        int(left) < int(right)
        for left, right in pairwise(ordered_offsets)
    )
    runtime_apply_order_valid = runtime_apply_order == list(FORMAT0D_RUNTIME_APPLY_ORDER)
    valid = (
        _format_id_text(proof.get("format_id")) == "0x0D"
        and runtime_apply_order_valid
        and candidate_base_extra_ordered
        and all(row["identity_valid"] is True for row in section_evidence.values())
    )
    return {
        "valid": valid,
        "format_id": _format_id_text(proof.get("format_id")),
        "runtime_apply_order": runtime_apply_order,
        "runtime_apply_order_valid": runtime_apply_order_valid,
        "candidate_base_extra_section_ordered": candidate_base_extra_ordered,
        "sections": section_evidence,
    }


def _format0d_runtime_apply_order(proof: Mapping[str, Any]) -> list[str]:
    candidates = (
        proof.get("runtime_apply_order"),
        proof.get("format0d_runtime_apply_order"),
        _as_mapping(proof.get("format0d_layout")).get("runtime_apply_order"),
        _as_mapping(proof.get("runtime_format0d")).get("runtime_apply_order"),
        _as_mapping(_as_mapping(proof.get("candidate_manifest")).get("format0d_layout")).get(
            "runtime_apply_order"
        ),
        _as_mapping(
            _as_mapping(proof.get("source_packet_manifest")).get("format0d_layout")
        ).get("runtime_apply_order"),
        _as_mapping(
            _as_mapping(proof.get("candidate_packet_manifest")).get("format0d_layout")
        ).get("runtime_apply_order"),
        _as_mapping(proof.get("semantic_materialization")).get("runtime_apply_order"),
    )
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, str):
            return [str(item) for item in candidate]
    return []


def _runtime_consumed_section_names(consumed_sections: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in consumed_sections.items():
        if value is True or (
            isinstance(value, Mapping) and _runtime_section_consumed(value) is True
        ):
            names.add(str(key))
    return names


def _runtime_consumed_section_identity(
    proof: Mapping[str, Any],
    *,
    consumed_sections: Mapping[str, Any],
    expected_section: str,
) -> dict[str, Any]:
    direct = consumed_sections.get(expected_section)
    if isinstance(direct, Mapping):
        return _runtime_section_identity_from_row(
            expected_section,
            direct,
            default_consumed=_runtime_section_consumed(direct),
        )
    for row in _iter_runtime_section_identity_rows(proof):
        if row.get("name") != expected_section:
            continue
        return _runtime_section_identity_from_row(
            expected_section,
            row,
            default_consumed=direct is True,
        )
    return {
        "found": False,
        "consumed": direct is True,
        "sha256": None,
        "length": None,
        "offset": None,
        "hash_domain": None,
    }


def _iter_runtime_section_identity_rows(proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "runtime_consumed_score_affecting_section_identities",
        "runtime_consumed_score_affecting_sections_identity",
        "runtime_consumed_section_identities",
        "runtime_score_affecting_section_identities",
        "runtime_section_identities",
        "runtime_consumed_sections",
    ):
        container = proof.get(key)
        if isinstance(container, Mapping):
            for section_name, value in container.items():
                if isinstance(value, Mapping):
                    row = dict(value)
                    row.setdefault("name", section_name)
                    rows.append(row)
        elif isinstance(container, Sequence) and not isinstance(container, str):
            for value in container:
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    return rows


def _runtime_section_identity_from_row(
    expected_section: str,
    row: Mapping[str, Any],
    *,
    default_consumed: bool,
) -> dict[str, Any]:
    return {
        "found": True,
        "consumed": _runtime_section_consumed(row) or default_consumed,
        "sha256": _first_str(
            row.get("sha256"),
            row.get("payload_sha256"),
            row.get("section_sha256"),
            row.get("runtime_sha256"),
        ),
        "hash_domain": _section_hash_domain(row),
        "length": _first_int(
            row.get("bytes"),
            row.get("byte_count"),
            row.get("length"),
            row.get("runtime_bytes"),
        ),
        "offset": _first_int(
            row.get("offset"),
            row.get("offset_start"),
            row.get("start"),
            row.get("runtime_offset"),
        ),
        "name": row.get("name", expected_section),
    }


def _section_hash_domain(row: Mapping[str, Any]) -> str | None:
    return _first_str(
        row.get("hash_domain"),
        row.get("sha256_domain"),
        row.get("section_hash_domain"),
        row.get("payload_hash_domain"),
    )


def _runtime_section_consumed(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(key) is True
        for key in (
            "consumed",
            "runtime_consumed",
            "runtime_consumption_claim",
            "decode_consumed",
            "apply_consumed",
            "consumption_claim",
        )
    )


def _format_id_text(value: Any) -> str:
    if isinstance(value, int):
        return f"0x{value:02X}"
    text = str(value)
    if text.lower().startswith("0x"):
        try:
            return f"0x{int(text, 16):02X}"
        except ValueError:
            return text
    return text


def _headerless_alias_identity(
    proof: Mapping[str, Any],
    *,
    expected_section: str,
    candidate_consumed_byte_proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate SHA-bound identity for reconstructed PR106 payload aliases."""
    byte_exact_identity = _as_mapping(proof.get("byte_exact_identity"))
    source_sha = _first_str(
        proof.get("source_inner_pr106_payload_sha256"),
        proof.get("expected_inner_pr106_payload_sha256"),
        proof.get("source_payload_sha256"),
        byte_exact_identity.get("source_payload_sha256"),
    )
    runtime_sha = _first_str(
        proof.get("runtime_inner_pr106_payload_sha256"),
        proof.get("reconstructed_inner_pr106_payload_sha256"),
        proof.get("runtime_reconstructed_pr106_payload_sha256"),
        proof.get("source_payload_sha256"),
        byte_exact_identity.get("source_payload_sha256"),
    )
    candidate_section_sha = _first_str(
        proof.get("candidate_headerless_section_sha256"),
        proof.get("candidate_pr106_payload_without_inner_header_sha256"),
        proof.get("candidate_section_sha256"),
    )
    candidate_section_offset = _first_int(
        proof.get("candidate_headerless_section_offset"),
        proof.get("candidate_section_offset"),
        proof.get("candidate_headerless_section_start"),
    )
    candidate_section_length = _first_int(
        proof.get("candidate_headerless_section_length"),
        proof.get("candidate_section_length"),
        proof.get("candidate_headerless_section_bytes"),
    )
    consumed_section = _candidate_consumed_section_identity(
        candidate_consumed_byte_proof,
        expected_section=expected_section,
    )
    candidate_section_bound = (
        consumed_section["found"] is True
        and consumed_section["score_affecting"] is True
        and consumed_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        and isinstance(candidate_section_sha, str)
        and consumed_section["sha256"] == candidate_section_sha
        and candidate_section_offset is not None
        and consumed_section["offset"] == candidate_section_offset
        and candidate_section_length is not None
        and consumed_section["length"] == candidate_section_length
    )
    valid = (
        isinstance(source_sha, str)
        and isinstance(runtime_sha, str)
        and source_sha == runtime_sha
        and isinstance(candidate_section_sha, str)
        and candidate_section_offset is not None
        and candidate_section_offset >= 0
        and candidate_section_length is not None
        and candidate_section_length > 0
        and consumed_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        and candidate_section_bound is True
    )
    return {
        "valid": valid,
        "source_inner_pr106_payload_sha256": source_sha,
        "runtime_inner_pr106_payload_sha256": runtime_sha,
        "candidate_headerless_section_sha256": candidate_section_sha,
        "candidate_headerless_section_offset": candidate_section_offset,
        "candidate_headerless_section_length": candidate_section_length,
        "candidate_consumed_section_name": expected_section,
        "candidate_consumed_section_found": consumed_section["found"],
        "candidate_consumed_section_sha256": consumed_section["sha256"],
        "candidate_consumed_section_hash_domain": consumed_section["hash_domain"],
        "candidate_hash_domain_valid": (
            consumed_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        ),
        "candidate_consumed_section_offset": consumed_section["offset"],
        "candidate_consumed_section_length": consumed_section["length"],
        "candidate_consumed_section_score_affecting": consumed_section[
            "score_affecting"
        ],
        "candidate_section_bound_to_consumed_byte_proof": candidate_section_bound,
    }


def _candidate_consumed_section_identity(
    proof: Mapping[str, Any] | None,
    *,
    expected_section: str,
) -> dict[str, Any]:
    consumed = _as_mapping(proof)
    for row in consumed.get("sections") or []:
        if not isinstance(row, Mapping) or row.get("name") != expected_section:
            continue
        return {
            "found": True,
            "sha256": _first_str(
                row.get("sha256"),
                row.get("payload_sha256"),
                row.get("section_sha256"),
            ),
            "hash_domain": _section_hash_domain(row),
            "offset": _first_int(
                row.get("offset"),
                row.get("offset_start"),
                row.get("start"),
            ),
            "length": _first_int(
                row.get("bytes"),
                row.get("byte_count"),
                row.get("length"),
            ),
            "score_affecting": row.get("score_affecting"),
        }
    return {
        "found": False,
        "sha256": None,
        "hash_domain": None,
        "offset": None,
        "length": None,
        "score_affecting": None,
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
    runtime_inflate_py_sha = _first_str(data.get("runtime_inflate_py_sha256"))
    runtime_content_tree_sha = _first_str(
        data.get("runtime_content_tree_sha256"),
        runtime_source_manifest.get("runtime_content_tree_sha256"),
    )
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
        and isinstance(runtime_inflate_py_sha, str)
        and bool(runtime_file_sha256s)
        and runtime_file_sha256s.get("inflate.py") == runtime_inflate_py_sha
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
        "runtime_inflate_py_sha256": runtime_inflate_py_sha,
        "runtime_source_tree_sha256": runtime_source_manifest.get(
            "runtime_source_tree_sha256"
        ),
        "runtime_content_tree_sha256": runtime_content_tree_sha,
        "runtime_source_files": [
            {
                "path": file.get("path"),
                "sha256": file.get("sha256"),
                "bytes": file.get("bytes"),
            }
            for file in runtime_files
        ],
        "runtime_file_sha256s": runtime_file_sha256s,
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
    parity_runtime_files = _as_mapping(
        full_frame_parity_summary.get("runtime_file_sha256s")
    )
    consumption_runtime_files = _as_mapping(
        runtime_consumption_summary.get("runtime_file_sha256s")
    )
    cuda_runtime_content_tree = cuda_summary.get("runtime_content_tree_sha256")
    parity_runtime_content_tree = full_frame_parity_summary.get(
        "runtime_content_tree_sha256"
    )
    consumption_runtime_content_tree = runtime_consumption_summary.get(
        "runtime_content_tree_sha256"
    )
    parity_files_match_consumption = bool(parity_runtime_files) and all(
        parity_runtime_files.get(path) == sha
        for path, sha in consumption_runtime_files.items()
    )
    parity_files_match_cuda = bool(parity_runtime_files) and all(
        cuda_runtime_files.get(path) == sha
        for path, sha in parity_runtime_files.items()
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
        and parity_files_match_consumption
        and parity_files_match_cuda
        and consumption_files_match_cuda
        and isinstance(cuda_runtime_content_tree, str)
        and bool(cuda_runtime_content_tree)
        and parity_runtime_content_tree == cuda_runtime_content_tree
        and consumption_runtime_content_tree == cuda_runtime_content_tree
    )


def _exact_eval_duplicate_keys(cuda_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    archive_sha = cuda_summary.get("archive_sha256")
    runtime_content_sha = cuda_summary.get("runtime_content_tree_sha256")
    runtime_tree_sha = cuda_summary.get("runtime_tree_sha256")
    score_axis = cuda_summary.get("score_axis")
    if not (
        isinstance(archive_sha, str)
        and isinstance(runtime_content_sha, str)
        and isinstance(runtime_tree_sha, str)
        and isinstance(score_axis, str)
    ):
        return []
    return [
        {
            "archive_sha256": archive_sha,
            "runtime_content_tree_sha256": runtime_content_sha,
            "runtime_tree_sha256": runtime_tree_sha,
            "score_axis": score_axis,
            "key": f"{archive_sha}:{runtime_content_sha}:{runtime_tree_sha}:{score_axis}",
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


def _str_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if item]
    return []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "PacketIRExactClosureError",
    "build_packetir_exact_closure",
    "render_packetir_exact_closure_markdown",
]
