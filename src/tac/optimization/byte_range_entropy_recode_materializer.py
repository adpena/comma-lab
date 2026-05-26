# SPDX-License-Identifier: MIT
"""Fail-closed byte-range entropy-recode materializer contract.

This module is the reusable bridge between broad byte-shaving plans and the
first concrete non-DQS1 implementation path: PR103-style arithmetic histogram
retuning. It may emit byte-different local candidate archives, but it never
clears promotion or exact-eval authority without an explicit runtime-consumption
proof for the rewritten byte ranges.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import build_candidate_diff_manifest
from tac.hnerv_lowlevel_packer import read_strict_single_member_zip
from tac.hnerv_pr103_lc_ac_schema import (
    AC_STREAM_SPECS,
    HI_SYMBOL_COUNT,
    PUBLIC_PR103_LAYOUT,
    Pr103LcAcLayout,
)
from tac.pr103_arithmetic_transform_plan import (
    CANDIDATE_SCHEMA as PR103_CANDIDATE_SCHEMA,
)
from tac.pr103_arithmetic_transform_plan import (
    Pr103ArithmeticTransformPlanError,
    materialize_pr103_arithmetic_histogram_candidate,
)
from tac.repo_io import read_json, sha256_bytes, sha256_file

PLAN_SCHEMA = "byte_range_entropy_recode_plan_v1"
CANDIDATE_SCHEMA = "byte_range_entropy_recode_candidate_v1"
VERIFIED_CANDIDATE_SCHEMA = "byte_range_entropy_recode_candidate_receiver_verified_v1"
RECEIVER_PROOF_SCHEMA = "byte_range_entropy_recode_receiver_proof_v1"
MATERIALIZER_ID = "byte_range_entropy_recode_adapter"
TARGET_KIND = "byte_range_entropy_recode_v1"
RECEIVER_CONTRACT_ID = "byte_range_entropy_recode_receiver.v1"
RECEIVER_CONTRACT_KIND = "archive_charged_byte_range_entropy_recode"
REQUIRED_CONTEXT_FIELDS = (
    "archive_member_name",
    "archive_byte_range",
    "runtime_consumption_proof",
)
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


class ByteRangeEntropyRecodeMaterializerError(ValueError):
    """Raised when byte-range entropy-recode contract inputs are malformed."""


def build_byte_range_entropy_recode_plan(
    *,
    archive_member_name: str | None,
    archive_byte_range: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a planning row for a byte-range entropy-recode operation."""

    ranges = _normalize_byte_ranges(archive_byte_range)
    proof = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_archive_member_name=archive_member_name,
    )
    blockers = [
        *(["archive_member_name_missing"] if not _clean_str(archive_member_name) else []),
        *(["archive_byte_range_missing"] if not ranges else []),
        *proof["blockers"],
    ]
    return {
        "schema": PLAN_SCHEMA,
        "materializer_id": MATERIALIZER_ID,
        "target_kind": TARGET_KIND,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "required_context_fields": list(REQUIRED_CONTEXT_FIELDS),
        "archive_member_name": _clean_str(archive_member_name),
        "archive_byte_ranges": ranges,
        "receiver_contract_satisfied": not blockers,
        "receiver_verification": proof,
        "readiness_blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def materialize_byte_range_entropy_recode_candidate(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    beam_probe_reports: Sequence[str | Path | Mapping[str, Any]],
    output_archive: str | Path,
    source_archive: str | Path | None = None,
    global_combo_report: str | Path | Mapping[str, Any] | None = None,
    member_name: str | None = None,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
    retune_brotli_sections: Sequence[str] = (),
) -> dict[str, Any]:
    """Materialize a PR103-backed entropy-recode candidate under this contract."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        pr103_report = materialize_pr103_arithmetic_histogram_candidate(
            schema_manifest=schema_manifest,
            beam_probe_reports=beam_probe_reports,
            output_archive=output_archive,
            source_archive=source_archive,
            global_combo_report=global_combo_report,
            member_name=member_name,
            repo_root=repo,
            layout=layout,
            stream_specs=stream_specs,
            hi_symbol_count=hi_symbol_count,
            retune_brotli_sections=retune_brotli_sections,
        )
    except Pr103ArithmeticTransformPlanError as exc:
        raise ByteRangeEntropyRecodeMaterializerError(str(exc)) from exc

    candidate_archive = _mapping(pr103_report.get("candidate_archive"))
    source_archive_record = _mapping(pr103_report.get("source_archive"))
    candidate_sha = _clean_str(candidate_archive.get("sha256"))
    candidate_member_sha = _clean_str(candidate_archive.get("member_sha256"))
    archive_member_name = _clean_str(candidate_archive.get("member_name"))
    archive_byte_ranges = _changed_archive_byte_ranges(
        pr103_report,
        archive_member_name=archive_member_name,
    )
    receiver_verification = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_archive_member_name=archive_member_name,
        required_candidate_archive_sha256=candidate_sha or None,
        required_candidate_member_sha256=candidate_member_sha or None,
    )
    source_archive_path = _resolve_repo_path(source_archive_record.get("path"), repo)
    candidate_archive_path = _resolve_repo_path(candidate_archive.get("path"), repo)
    diff_manifest: dict[str, Any] | None = None
    if source_archive_path is not None and candidate_archive_path is not None:
        diff_manifest = build_candidate_diff_manifest(
            source_archive=source_archive_path,
            candidate_archive=candidate_archive_path,
            source_label="source_pr103_archive",
            candidate_label="byte_range_entropy_recode_candidate",
        )

    pr103_blockers = [str(item) for item in pr103_report.get("readiness_blockers") or []]
    if receiver_verification["receiver_contract_satisfied"] is True:
        pr103_blockers = [
            blocker
            for blocker in pr103_blockers
            if blocker != "candidate_runtime_adapter_missing"
        ]
    readiness_blockers = _ordered_unique(
        [
            *pr103_blockers,
            *receiver_verification["blockers"],
            *(
                []
                if receiver_verification["receiver_contract_satisfied"] is True
                else ["byte_range_entropy_recode_receiver_contract_not_satisfied"]
            ),
        ]
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "source_materializer_schema": PR103_CANDIDATE_SCHEMA,
        "materializer_id": MATERIALIZER_ID,
        "target_kind": TARGET_KIND,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "required_context_fields": list(REQUIRED_CONTEXT_FIELDS),
        "byte_closed_candidate_emitted": bool(candidate_sha),
        "archive_member_name": archive_member_name,
        "archive_byte_ranges": archive_byte_ranges,
        "source_archive": source_archive_record,
        "candidate_archive": candidate_archive,
        "archive_diff_manifest": diff_manifest,
        "pr103_candidate": pr103_report,
        "receiver_verification": receiver_verification,
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "readiness_blockers": readiness_blockers,
        "ready_for_archive_preflight": False,
        **FALSE_AUTHORITY,
    }


def build_byte_range_entropy_recode_receiver_proof(
    *,
    runtime_adapter_manifest: str | Path | Mapping[str, Any],
    candidate_manifest: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a PR103 runtime-adapter manifest into the byte-range proof."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    adapter, adapter_record = _load_runtime_adapter_manifest(
        runtime_adapter_manifest,
        repo=repo,
    )
    candidate, candidate_record = _load_candidate_manifest_for_adapter(
        adapter,
        candidate_manifest=candidate_manifest,
        repo=repo,
        adapter_record=adapter_record,
    )
    adapter_candidate = _mapping(adapter.get("candidate_archive"))
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    archive_member_name = _clean_str(candidate_archive.get("member_name"))
    archive_byte_ranges = _changed_archive_byte_ranges(
        candidate,
        archive_member_name=archive_member_name,
    )
    runtime_probe = _mapping(adapter.get("runtime_consumption_probe"))
    decoder_parity = _mapping(adapter.get("decoder_state_parity_proof"))
    blockers = _ordered_unique(
        [
            *(
                []
                if adapter.get("score_claim") is False
                else ["runtime_adapter_manifest_must_not_claim_score"]
            ),
            *(
                []
                if adapter.get("dispatch_attempted") is False
                else ["runtime_adapter_manifest_must_not_dispatch"]
            ),
            *(
                []
                if runtime_probe.get("passed") is True
                else ["runtime_consumption_probe_not_passed"]
            ),
            *(
                []
                if decoder_parity.get("passed") is True
                else ["decoder_state_parity_not_passed"]
            ),
            *(
                []
                if _clean_str(adapter_candidate.get("sha256"))
                == _clean_str(candidate_archive.get("sha256"))
                else ["runtime_adapter_candidate_archive_sha_mismatch"]
            ),
            *(
                []
                if _clean_str(adapter_candidate.get("sha256"))
                else ["runtime_adapter_candidate_archive_sha_missing"]
            ),
            *(["archive_member_name_missing"] if not archive_member_name else []),
            *(["archive_byte_ranges_missing"] if not archive_byte_ranges else []),
        ]
    )
    return {
        "schema": RECEIVER_PROOF_SCHEMA,
        "materializer_id": MATERIALIZER_ID,
        "target_kind": TARGET_KIND,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "receiver_contract_satisfied": not blockers,
        "runtime_consumption_proof_passed": not blockers,
        "passed": not blockers,
        "ready_for_exact_eval_runtime": not blockers,
        "runtime_adapter_manifest": adapter_record,
        "candidate_manifest": candidate_record,
        "archive_member_name": archive_member_name,
        "archive_byte_ranges": archive_byte_ranges,
        "candidate_archive_sha256": _clean_str(candidate_archive.get("sha256")),
        "candidate_member_sha256": _clean_str(candidate_archive.get("member_sha256")),
        "runtime_tree_sha256": _clean_str(adapter.get("runtime_tree_sha256")),
        "runtime_file_records_sha256": _clean_str(
            adapter.get("runtime_file_records_sha256")
        ),
        "runtime_consumption_probe": runtime_probe,
        "decoder_state_parity_proof": decoder_parity,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def verify_byte_range_entropy_recode_candidate_manifest(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    runtime_consumption_proof: str | Path | Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an existing materialized candidate without rewriting its archive."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate, candidate_record = _load_required_mapping_with_record(
        candidate_manifest,
        repo=repo,
        label="byte-range candidate manifest",
    )
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ByteRangeEntropyRecodeMaterializerError(
            f"candidate manifest must have schema {CANDIDATE_SCHEMA}"
        )
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    archive_member_name = _clean_str(candidate.get("archive_member_name"))
    candidate_sha = _clean_str(candidate_archive.get("sha256"))
    candidate_member_sha = _clean_str(candidate_archive.get("member_sha256"))
    archive_byte_ranges = _normalize_byte_ranges(candidate.get("archive_byte_ranges"))
    receiver_verification = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=runtime_consumption_proof,
        required_archive_member_name=archive_member_name,
        required_candidate_archive_sha256=candidate_sha or None,
        required_candidate_member_sha256=candidate_member_sha or None,
    )
    custody = _candidate_archive_custody(
        candidate_archive,
        required_member_name=archive_member_name,
        repo=repo,
    )
    original_blockers = [
        str(item) for item in candidate.get("readiness_blockers") or [] if str(item)
    ]
    if (
        receiver_verification["receiver_contract_satisfied"] is True
        and not custody["blockers"]
    ):
        original_blockers = [
            blocker
            for blocker in original_blockers
            if blocker
            not in {
                "candidate_runtime_adapter_missing",
                "byte_range_entropy_recode_receiver_contract_not_satisfied",
                "runtime_consumption_proof_missing",
            }
        ]
    readiness_blockers = _ordered_unique(
        [
            *original_blockers,
            *receiver_verification["blockers"],
            *custody["blockers"],
            *(
                []
                if receiver_verification["receiver_contract_satisfied"] is True
                and not custody["blockers"]
                else ["byte_range_entropy_recode_receiver_contract_not_satisfied"]
            ),
        ]
    )
    return {
        "schema": VERIFIED_CANDIDATE_SCHEMA,
        "source_candidate_schema": candidate.get("schema"),
        "materializer_id": MATERIALIZER_ID,
        "target_kind": TARGET_KIND,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "source_candidate_manifest": candidate_record,
        "archive_member_name": archive_member_name,
        "archive_byte_ranges": archive_byte_ranges,
        "candidate_archive": candidate_archive,
        "candidate_archive_custody": custody,
        "receiver_verification": receiver_verification,
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
            and not custody["blockers"]
        ),
        "readiness_blockers": readiness_blockers,
        "ready_for_archive_preflight": False,
        **FALSE_AUTHORITY,
    }


def verify_byte_range_entropy_recode_receiver_contract(
    *,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None,
    required_archive_member_name: str | None = None,
    required_candidate_archive_sha256: str | None = None,
    required_candidate_member_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate that a receiver/runtime proof consumes the rewritten bytes."""

    blockers: list[str] = []
    proof = _load_optional_mapping(runtime_consumption_proof)
    if proof is None:
        blockers.append("runtime_consumption_proof_missing")
    else:
        if proof.get("schema") != RECEIVER_PROOF_SCHEMA:
            blockers.append("runtime_consumption_proof_schema_mismatch")
        if proof.get("score_claim") is not False:
            blockers.append("runtime_consumption_proof_must_not_claim_score")
        if proof.get("promotion_eligible") is not False:
            blockers.append("runtime_consumption_proof_must_not_promote")
        if proof.get("rank_or_kill_eligible") is not False:
            blockers.append("runtime_consumption_proof_must_not_rank_or_kill")
        if proof.get("ready_for_exact_eval_runtime") is not True:
            blockers.append("runtime_consumption_proof_not_ready")
        if required_archive_member_name and _clean_str(
            proof.get("archive_member_name")
        ) != _clean_str(required_archive_member_name):
            blockers.append("runtime_consumption_proof_member_mismatch")
        if required_candidate_archive_sha256 and _clean_str(
            proof.get("candidate_archive_sha256")
        ) != _clean_str(required_candidate_archive_sha256):
            blockers.append("runtime_consumption_proof_archive_sha_mismatch")
        if required_candidate_member_sha256 and _clean_str(
            proof.get("candidate_member_sha256")
        ) != _clean_str(required_candidate_member_sha256):
            blockers.append("runtime_consumption_proof_member_sha_mismatch")
        if not _normalize_byte_ranges(proof.get("archive_byte_ranges")):
            blockers.append("runtime_consumption_proof_byte_ranges_missing")

    return {
        "schema": "byte_range_entropy_recode_receiver_verification_v1",
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "receiver_contract_satisfied": not blockers,
        "proof_schema": proof.get("schema") if proof is not None else None,
        "proof_candidate_archive_sha256": (
            proof.get("candidate_archive_sha256") if proof is not None else ""
        ),
        "proof_candidate_member_sha256": (
            proof.get("candidate_member_sha256") if proof is not None else ""
        ),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _candidate_archive_custody(
    record: Mapping[str, Any],
    *,
    required_member_name: str,
    repo: Path,
) -> dict[str, Any]:
    path = _resolve_repo_path(record.get("path"), repo)
    blockers: list[str] = []
    if path is None:
        blockers.append("candidate_archive_path_missing_or_unreadable")
        return {
            "path": _clean_str(record.get("path")),
            "exists": False,
            "blockers": blockers,
        }
    expected_sha = _clean_str(record.get("sha256"))
    expected_bytes = record.get("bytes")
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    if expected_sha and actual_sha != expected_sha:
        blockers.append("candidate_archive_sha_mismatch")
    if expected_bytes is not None and int(expected_bytes) != actual_bytes:
        blockers.append("candidate_archive_bytes_mismatch")
    try:
        member = read_strict_single_member_zip(path)
    except Exception:  # pragma: no cover - exact zip error type is not stable
        member = None
        blockers.append("candidate_archive_single_member_read_failed")
    expected_member_sha = _clean_str(record.get("member_sha256"))
    if member is not None:
        actual_member_sha = sha256_bytes(member.payload)
        if required_member_name and member.member_name != required_member_name:
            blockers.append("candidate_archive_member_name_mismatch")
        if expected_member_sha and actual_member_sha != expected_member_sha:
            blockers.append("candidate_archive_member_sha_mismatch")
    return {
        "path": path.relative_to(repo).as_posix() if _path_is_relative_to(path, repo) else path.as_posix(),
        "exists": True,
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "member_name": member.member_name if member is not None else "",
        "member_sha256": sha256_bytes(member.payload) if member is not None else "",
        "blockers": _ordered_unique(blockers),
    }


def _changed_archive_byte_ranges(
    report: Mapping[str, Any],
    *,
    archive_member_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("section_diffs") or []:
        section = _mapping(row)
        if section.get("changed") is not True:
            continue
        rows.append(
            {
                "schema": "byte_range_entropy_recode_archive_range_v1",
                "archive_member_name": archive_member_name,
                "section_name": _clean_str(section.get("name")),
                "source_start": section.get("source_start"),
                "source_end": section.get("source_end"),
                "source_bytes": section.get("source_bytes"),
                "source_sha256": section.get("source_sha256"),
                "candidate_start": section.get("candidate_start"),
                "candidate_end": section.get("candidate_end"),
                "candidate_bytes": section.get("candidate_bytes"),
                "candidate_sha256": section.get("candidate_sha256"),
                "byte_delta": section.get("byte_delta"),
            }
        )
    return rows


def _normalize_byte_ranges(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Any,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        raw_rows: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        raw_rows = value
    else:
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        member_name = _clean_str(
            item.get("archive_member_name") or item.get("member_name")
        )
        section_name = _clean_str(item.get("section_name"))
        start = item.get("candidate_start", item.get("start"))
        end = item.get("candidate_end", item.get("end"))
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            continue
        if start_int < 0 or end_int <= start_int:
            continue
        rows.append(
            {
                "archive_member_name": member_name,
                "section_name": section_name,
                "candidate_start": start_int,
                "candidate_end": end_int,
                "candidate_bytes": end_int - start_int,
            }
        )
    return rows


def _load_optional_mapping(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ByteRangeEntropyRecodeMaterializerError(
            f"runtime consumption proof unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ByteRangeEntropyRecodeMaterializerError(
            f"runtime consumption proof is not a JSON object: {path}"
        )
    return payload


def _load_runtime_adapter_manifest(
    value: str | Path | Mapping[str, Any],
    *,
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _load_required_mapping_with_record(
        value,
        repo=repo,
        label="runtime adapter manifest",
    )
    if payload.get("schema") != "pr103_lc_ac_runtime_adapter_v1":
        raise ByteRangeEntropyRecodeMaterializerError(
            "runtime adapter manifest must have schema pr103_lc_ac_runtime_adapter_v1"
        )
    return payload, record


def _load_candidate_manifest_for_adapter(
    adapter: Mapping[str, Any],
    *,
    candidate_manifest: str | Path | Mapping[str, Any] | None,
    repo: Path,
    adapter_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_manifest is not None:
        payload, record = _load_required_mapping_with_record(
            candidate_manifest,
            repo=repo,
            label="candidate manifest",
        )
    else:
        manifest_ref = _mapping(adapter.get("candidate_manifest"))
        path_text = _clean_str(manifest_ref.get("path"))
        if not path_text:
            raise ByteRangeEntropyRecodeMaterializerError(
                "runtime adapter manifest missing candidate_manifest.path"
            )
        manifest_path = _candidate_manifest_path_from_adapter_record(
            path_text,
            adapter_record=adapter_record,
            repo=repo,
        )
        payload, record = _load_required_mapping_with_record(
            manifest_path,
            repo=repo,
            label="candidate manifest",
        )
        expected_sha = _clean_str(manifest_ref.get("sha256"))
        if expected_sha and record.get("sha256") != expected_sha:
            raise ByteRangeEntropyRecodeMaterializerError(
                "candidate manifest sha256 does not match runtime adapter record"
            )
    if payload.get("schema") != PR103_CANDIDATE_SCHEMA:
        raise ByteRangeEntropyRecodeMaterializerError(
            f"candidate manifest must have schema {PR103_CANDIDATE_SCHEMA}"
        )
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise ByteRangeEntropyRecodeMaterializerError(
            "candidate manifest must be a no-score local artifact"
        )
    return payload, record


def _candidate_manifest_path_from_adapter_record(
    path_text: str,
    *,
    adapter_record: Mapping[str, Any],
    repo: Path,
) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    repo_path = repo / path
    if repo_path.exists():
        return repo_path
    adapter_path_text = _clean_str(adapter_record.get("path"))
    if adapter_path_text:
        adapter_path = Path(adapter_path_text)
        if not adapter_path.is_absolute():
            adapter_path = repo / adapter_path
        sibling_path = adapter_path.parent / path
        if sibling_path.exists():
            return sibling_path
    return repo_path


def _load_required_mapping_with_record(
    value: str | Path | Mapping[str, Any],
    *,
    repo: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value), {"provided_inline": True, "path": "", "sha256": ""}
    path = _resolve_existing_repo_path(value, repo=repo)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ByteRangeEntropyRecodeMaterializerError(
            f"{label} unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ByteRangeEntropyRecodeMaterializerError(
            f"{label} is not a JSON object: {path}"
        )
    return payload, {
        "provided_inline": False,
        "path": path.relative_to(repo).as_posix() if _path_is_relative_to(path, repo) else path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _resolve_repo_path(value: Any, repo: Path) -> Path | None:
    text = _clean_str(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = repo / path
    return path if path.exists() else None


def _resolve_existing_repo_path(value: str | Path, *, repo: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise ByteRangeEntropyRecodeMaterializerError(f"path does not exist: {path}")
    return path


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "CANDIDATE_SCHEMA",
    "FALSE_AUTHORITY",
    "MATERIALIZER_ID",
    "PLAN_SCHEMA",
    "RECEIVER_CONTRACT_ID",
    "RECEIVER_CONTRACT_KIND",
    "RECEIVER_PROOF_SCHEMA",
    "REQUIRED_CONTEXT_FIELDS",
    "TARGET_KIND",
    "VERIFIED_CANDIDATE_SCHEMA",
    "ByteRangeEntropyRecodeMaterializerError",
    "build_byte_range_entropy_recode_plan",
    "build_byte_range_entropy_recode_receiver_proof",
    "materialize_byte_range_entropy_recode_candidate",
    "verify_byte_range_entropy_recode_candidate_manifest",
    "verify_byte_range_entropy_recode_receiver_contract",
]
