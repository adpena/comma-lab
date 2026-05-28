# SPDX-License-Identifier: MIT
"""Fail-closed exact-ready bridge for scorer-region cascade artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.scorer_region_operator_contract import (
    SCORER_REGION_OPERATOR_CONTRACT_SCHEMA,
    build_scorer_region_operator_contract,
)
from tac.optimization.scorer_region_waterfill import (
    FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA,
)
from tac.optimizer.exact_readiness import QUEUE_SCHEMA as EXACT_READY_QUEUE_SCHEMA
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.repo_io import read_json, sha256_file

SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA = (
    "scorer_region_exact_ready_bridge_report.v1"
)
SCORER_REGION_EXACT_READY_SOURCE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA = "scorer_region_selector_chain_report.v1"
SHELL_INFLATE_OUTPUT_CHANGE_PROOF_SCHEMA = "shell_inflate_output_change_proof_v1"


class ScorerRegionExactReadyBridgeError(ValueError):
    """Raised when scorer-region exact-ready bridge construction fails."""


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _filtered_patch_blockers(
    patch_blockers: Sequence[str],
    *,
    runtime_custody_complete: bool,
    output_change_proven: bool,
) -> list[str]:
    cleared: set[str] = set()
    if runtime_custody_complete:
        cleared.add("runtime_consumption_proof_required_before_exact_eval")
    if output_change_proven:
        cleared.add("inflated_output_change_proof_required_before_budget_spend_claim")
    return [blocker for blocker in patch_blockers if blocker not in cleared]


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _read_json_object(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionExactReadyBridgeError(f"JSON artifact missing: {path}")
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise ScorerRegionExactReadyBridgeError(f"JSON artifact must be an object: {path}")
    return payload


def _file_custody(
    *,
    path: str | Path | None,
    repo_root: str | Path,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    text = str(path or "").strip()
    if not text:
        blockers.append(f"{label}_path_missing")
        return {
            "schema": "scorer_region_exact_ready_file_custody.v1",
            "label": label,
            "path": None,
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    resolved = _resolve(text, repo_root)
    if not resolved.is_file():
        blockers.append(f"{label}_file_missing")
        return {
            "schema": "scorer_region_exact_ready_file_custody.v1",
            "label": label,
            "path": _repo_rel(resolved, repo_root),
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    actual_sha = sha256_file(resolved)
    actual_bytes = resolved.stat().st_size
    if expected_sha256 is not None and actual_sha != expected_sha256:
        blockers.append(f"{label}_sha256_mismatch")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        blockers.append(f"{label}_bytes_mismatch")
    return {
        "schema": "scorer_region_exact_ready_file_custody.v1",
        "label": label,
        "path": _repo_rel(resolved, repo_root),
        "present": True,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "custody_complete": not blockers,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }, blockers


def _submission_runtime_custody(
    *,
    submission_dir: str | Path | None,
    candidate_archive_sha256: str | None,
    candidate_archive_bytes: int | None,
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if submission_dir is None or not str(submission_dir).strip():
        blockers.append("receiver_patch_submission_dir_missing")
        return {
            "schema": "scorer_region_exact_ready_submission_runtime_custody.v1",
            "submission_dir": None,
            "runtime_manifest": None,
            "runtime_tree_sha256": None,
            "runtime_content_tree_sha256": None,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    resolved = _resolve(submission_dir, repo_root)
    archive = resolved / "archive.zip"
    inflate_py = resolved / "inflate.py"
    inflate_sh = resolved / "inflate.sh"
    if not resolved.is_dir():
        blockers.append("receiver_patch_submission_dir_not_directory")
    if not archive.is_file():
        blockers.append("receiver_patch_archive_zip_missing")
    else:
        actual_sha = sha256_file(archive)
        actual_bytes = archive.stat().st_size
        if candidate_archive_sha256 and actual_sha != candidate_archive_sha256:
            blockers.append("receiver_patch_archive_sha256_mismatch")
        if candidate_archive_bytes is not None and actual_bytes != candidate_archive_bytes:
            blockers.append("receiver_patch_archive_bytes_mismatch")
    if not inflate_py.is_file():
        blockers.append("receiver_patch_inflate_py_missing")
    if not inflate_sh.is_file():
        blockers.append("receiver_patch_inflate_sh_missing")
    runtime_manifest = None
    if not blockers:
        try:
            runtime_manifest = runtime_dependency_manifest(resolved, Path(repo_root))
        except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
            blockers.append(f"runtime_dependency_manifest_failed:{exc}")
    runtime_tree_sha = (
        runtime_manifest.get("runtime_tree_sha256")
        if isinstance(runtime_manifest, Mapping)
        else None
    )
    runtime_content_sha = (
        runtime_manifest.get("runtime_content_tree_sha256")
        if isinstance(runtime_manifest, Mapping)
        else None
    )
    if not isinstance(runtime_tree_sha, str) or len(runtime_tree_sha) != 64:
        blockers.append("runtime_tree_sha256_missing")
    if not isinstance(runtime_content_sha, str) or len(runtime_content_sha) != 64:
        blockers.append("runtime_content_tree_sha256_missing")
    return {
        "schema": "scorer_region_exact_ready_submission_runtime_custody.v1",
        "submission_dir": _repo_rel(resolved, repo_root),
        "archive_path": _repo_rel(archive, repo_root),
        "inflate_py_path": _repo_rel(inflate_py, repo_root),
        "inflate_sh_path": _repo_rel(inflate_sh, repo_root),
        "runtime_manifest": runtime_manifest,
        "runtime_tree_sha256": runtime_tree_sha,
        "runtime_content_tree_sha256": runtime_content_sha,
        "custody_complete": not blockers,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }, blockers


def _output_change_proof_custody(
    *,
    path: str | Path | None,
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    text = str(path or "").strip()
    if not text:
        blockers.append("shell_inflate_output_change_proof_missing")
        return {
            "schema": "scorer_region_exact_ready_output_change_proof_custody.v1",
            "path": None,
            "present": False,
            "proof": None,
            "full_frame_output_change_proven": False,
            "contest_full_sample_change_proven": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    resolved = _resolve(text, repo_root)
    if not resolved.is_file():
        blockers.append("shell_inflate_output_change_proof_file_missing")
        return {
            "schema": "scorer_region_exact_ready_output_change_proof_custody.v1",
            "path": _repo_rel(resolved, repo_root),
            "present": False,
            "proof": None,
            "full_frame_output_change_proven": False,
            "contest_full_sample_change_proven": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }, blockers
    proof = _read_json_object(resolved, repo_root=repo_root)
    if proof.get("schema") != SHELL_INFLATE_OUTPUT_CHANGE_PROOF_SCHEMA:
        blockers.append("shell_inflate_output_change_proof_schema_mismatch")
    require_no_truthy_authority_fields(
        proof,
        context="scorer_region_bridge_output_change_proof",
    )
    if proof.get("output_change_observed") is not True:
        blockers.append("shell_inflate_output_change_not_observed")
    if proof.get("raw_shape_preserving_output_change_observed") is not True:
        blockers.append("raw_shape_preserving_output_change_not_observed")
    if proof.get("full_frame_output_change_claim") is not True:
        blockers.append("full_frame_output_change_claim_missing")
    if proof.get("contest_full_sample_change_claim") is not True:
        blockers.append("contest_full_sample_change_claim_missing")
    proof_blockers = _string_list(proof.get("blockers"))
    if proof_blockers:
        blockers.extend(f"output_change_proof:{item}" for item in proof_blockers)
    return {
        "schema": "scorer_region_exact_ready_output_change_proof_custody.v1",
        "path": _repo_rel(resolved, repo_root),
        "present": True,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "proof": proof,
        "full_frame_output_change_proven": proof.get("full_frame_output_change_claim")
        is True,
        "contest_full_sample_change_proven": proof.get("contest_full_sample_change_claim")
        is True,
        "differing_byte_count": proof.get("differing_byte_count"),
        "differing_output_count": proof.get("differing_output_count"),
        "custody_complete": not blockers,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }, blockers


def build_scorer_region_exact_ready_bridge(
    *,
    chain_report_path: str | Path,
    receiver_patch_manifest_path: str | Path,
    shell_inflate_output_change_proof_path: str | Path | None = None,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build blocked exact-ready queue inputs from a scorer-region receiver patch."""

    chain_report = _read_json_object(chain_report_path, repo_root=repo_root)
    patch_manifest = _read_json_object(receiver_patch_manifest_path, repo_root=repo_root)
    if chain_report.get("schema") != SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA:
        raise ScorerRegionExactReadyBridgeError("chain report schema mismatch")
    if patch_manifest.get("schema") != FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA:
        raise ScorerRegionExactReadyBridgeError("receiver patch manifest schema mismatch")
    require_no_truthy_authority_fields(chain_report, context="scorer_region_bridge_chain_report")
    require_no_truthy_authority_fields(patch_manifest, context="scorer_region_bridge_patch_manifest")

    candidate_archive = _mapping(patch_manifest.get("candidate_archive"))
    source_archive = _mapping(patch_manifest.get("source_archive"))
    runtime_patch = _mapping(patch_manifest.get("runtime_patch"))
    patched_inflate = _mapping(patch_manifest.get("patched_inflate"))
    archive_custody, archive_blockers = _file_custody(
        path=candidate_archive.get("path"),
        repo_root=repo_root,
        label="candidate_archive",
        expected_sha256=str(candidate_archive.get("sha256") or "").strip() or None,
        expected_bytes=candidate_archive.get("bytes")
        if isinstance(candidate_archive.get("bytes"), int)
        and not isinstance(candidate_archive.get("bytes"), bool)
        else None,
    )
    patch_custody, patch_blockers = _file_custody(
        path=runtime_patch.get("path"),
        repo_root=repo_root,
        label="runtime_patch",
        expected_sha256=str(runtime_patch.get("sha256") or "").strip() or None,
    )
    inflate_custody, inflate_blockers = _file_custody(
        path=patched_inflate.get("path"),
        repo_root=repo_root,
        label="patched_inflate",
        expected_sha256=str(patched_inflate.get("sha256") or "").strip() or None,
    )
    runtime_custody, runtime_blockers = _submission_runtime_custody(
        submission_dir=patch_manifest.get("output_submission_dir"),
        candidate_archive_sha256=archive_custody.get("sha256")
        if isinstance(archive_custody.get("sha256"), str)
        else None,
        candidate_archive_bytes=archive_custody.get("bytes")
        if isinstance(archive_custody.get("bytes"), int)
        else None,
        repo_root=repo_root,
    )
    output_change_custody, output_change_blockers = _output_change_proof_custody(
        path=shell_inflate_output_change_proof_path,
        repo_root=repo_root,
    )
    patch_manifest_blockers = _filtered_patch_blockers(
        _string_list(patch_manifest.get("blockers")),
        runtime_custody_complete=runtime_custody.get("custody_complete") is True,
        output_change_proven=output_change_custody.get("custody_complete") is True,
    )
    chain_operator_contract = _mapping(chain_report.get("operator_contract"))
    operator_contract = build_scorer_region_operator_contract(
        chain_label=str(chain_report.get("chain_label") or "unknown"),
        receiver_patch_enabled=True,
    )
    operator_contract_source = "bridge_rebuilt_contract"
    if chain_operator_contract.get("schema") == SCORER_REGION_OPERATOR_CONTRACT_SCHEMA:
        operator_contract_source = "chain_report_contract_revalidated_with_receiver_patch"
    candidate_id = (
        "scorer_region_exact_ready__"
        f"{_slug(chain_report.get('chain_label'))}__"
        f"{_slug(patch_manifest.get('receiver_contract_target'))}"
    )
    blockers = ordered_unique(
        [
            *archive_blockers,
            *patch_blockers,
            *inflate_blockers,
            *runtime_blockers,
            *output_change_blockers,
            *_string_list(chain_report.get("readiness_blockers")),
            *patch_manifest_blockers,
            "contest_cpu_or_cuda_auth_axis_required_before_score_or_dispatch",
            "lane_dispatch_claim_required_before_exact_eval",
            "promote_optimizer_candidate_for_exact_eval_required_before_dispatch_ready",
        ]
    )
    bridge_row = {
        "candidate_id": candidate_id,
        "candidate_family": "scorer_region_frame1_waterfill_runtime_patch",
        "chain_label": chain_report.get("chain_label"),
        "lane_id": "scorer_region_frame1_waterfill_exact_ready_bridge",
        "operator_contract": operator_contract,
        "operator_contract_source": operator_contract_source,
        "target_modes": ["contest_exact_eval"],
        "target_score_axes_required": ["contest_cpu", "contest_cuda"],
        "archive_path": archive_custody.get("path"),
        "candidate_archive_path": archive_custody.get("path"),
        "candidate_archive_sha256": archive_custody.get("sha256"),
        "archive_sha256": archive_custody.get("sha256"),
        "candidate_archive_bytes": archive_custody.get("bytes"),
        "archive_bytes": archive_custody.get("bytes"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_archive_bytes": source_archive.get("bytes"),
        "score_affecting_runtime_changed": True,
        "runtime_patch_path": patch_custody.get("path"),
        "patched_inflate_path": inflate_custody.get("path"),
        "submission_dir": runtime_custody.get("submission_dir"),
        "runtime_tree_sha256": runtime_custody.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime_custody.get("runtime_content_tree_sha256"),
        "shell_inflate_output_change_proof": output_change_custody.get("path"),
        "full_frame_output_change_proven": output_change_custody.get(
            "full_frame_output_change_proven"
        )
        is True,
        "contest_full_sample_change_proven": output_change_custody.get(
            "contest_full_sample_change_proven"
        )
        is True,
        "receiver_contract_target": patch_manifest.get("receiver_contract_target"),
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": (
            "runtime_content_tree_custody_present"
            if runtime_custody.get("custody_complete") is True
            else "runtime_content_tree_custody_incomplete"
        ),
        "dispatch_blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(bridge_row, context="scorer_region_bridge_source_row")
    source_queue = {
        "schema": SCORER_REGION_EXACT_READY_SOURCE_QUEUE_SCHEMA,
        "tool": "comma_lab.scheduler.scorer_region_exact_ready_bridge",
        "top_k": [bridge_row],
        "dispatch_ready": [],
        "dispatch_ready_count": 0,
        "n_candidates": 1,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    blocked_exact_ready_queue = {
        "schema": EXACT_READY_QUEUE_SCHEMA,
        "tool": "comma_lab.scheduler.scorer_region_exact_ready_bridge",
        "n_candidates": 1,
        "top_k_count": 1,
        "dispatch_ready_count": 0,
        "dispatch_ready": [],
        "top_k": [bridge_row],
        "top_k_forensic": [bridge_row],
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    report = {
        "schema": SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA,
        "operator_contract": operator_contract,
        "operator_contract_source": operator_contract_source,
        "chain_report": _file_custody(
            path=chain_report_path,
            repo_root=repo_root,
            label="chain_report",
        )[0],
        "receiver_patch_manifest": _file_custody(
            path=receiver_patch_manifest_path,
            repo_root=repo_root,
            label="receiver_patch_manifest",
        )[0],
        "candidate_count": 1,
        "archive_custody_proven_count": int(archive_custody.get("custody_complete") is True),
        "runtime_patch_custody_proven_count": int(patch_custody.get("custody_complete") is True),
        "runtime_content_tree_custody_proven_count": int(
            runtime_custody.get("custody_complete") is True
        ),
        "output_change_proof_proven_count": int(
            output_change_custody.get("custody_complete") is True
        ),
        "source_optimizer_queue_schema": source_queue["schema"],
        "blocked_exact_ready_queue_schema": blocked_exact_ready_queue["schema"],
        "blocked_exact_ready_dispatch_ready_count": 0,
        "rows": [
            {
                "schema": "scorer_region_exact_ready_bridge_row.v1",
                "candidate_id": candidate_id,
                "archive_custody": archive_custody,
                "runtime_patch_custody": patch_custody,
                "patched_inflate_custody": inflate_custody,
                "submission_runtime_custody": runtime_custody,
                "output_change_proof_custody": output_change_custody,
                "bridge_source_queue_row": bridge_row,
                "blockers": blockers,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
        ],
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "scorer_region_exact_ready_bridge_fail_closed_inputs",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    for payload, context in (
        (source_queue, "scorer_region_source_optimizer_queue"),
        (blocked_exact_ready_queue, "scorer_region_blocked_exact_ready_queue"),
        (report, "scorer_region_exact_ready_bridge_report"),
    ):
        require_no_truthy_authority_fields(payload, context=context)
    return {
        "source_optimizer_queue": source_queue,
        "blocked_exact_ready_queue": blocked_exact_ready_queue,
        "bridge_report": report,
    }


__all__ = [
    "SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA",
    "SCORER_REGION_EXACT_READY_SOURCE_QUEUE_SCHEMA",
    "SCORER_REGION_OPERATOR_CONTRACT_SCHEMA",
    "SHELL_INFLATE_OUTPUT_CHANGE_PROOF_SCHEMA",
    "ScorerRegionExactReadyBridgeError",
    "build_scorer_region_exact_ready_bridge",
]
