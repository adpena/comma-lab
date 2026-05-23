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
from tac.repo_io import read_json

PLAN_SCHEMA = "byte_range_entropy_recode_plan_v1"
CANDIDATE_SCHEMA = "byte_range_entropy_recode_candidate_v1"
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

    readiness_blockers = _ordered_unique(
        [
            *[str(item) for item in pr103_report.get("readiness_blockers") or []],
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


def _resolve_repo_path(value: Any, repo: Path) -> Path | None:
    text = _clean_str(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = repo / path
    return path if path.exists() else None


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
    "ByteRangeEntropyRecodeMaterializerError",
    "build_byte_range_entropy_recode_plan",
    "materialize_byte_range_entropy_recode_candidate",
    "verify_byte_range_entropy_recode_receiver_contract",
]
